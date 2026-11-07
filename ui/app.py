"""The application: window, frame loop, input, and everything drawn on top of the sim.

Simulation and rendering are decoupled. Each frame banks real elapsed time multiplied
by the speed setting, spends it on whole fixed ticks, then draws the latest state,
interpolated between the last two ticks. Rendering therefore cannot change a race's
outcome, and the race plays out identically at any speed or frame rate.

neat-python calls our fitness function and waits for it to return, so one generation
runs as a nested loop inside ``_evaluate``. Restart and quit unwind with exceptions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from ai.car import CarView
from ai.evolution import EvolutionManager, GenerationResult, summarize_generation
from ai.neural_network import NetworkController, layout_topology
from rendering.camera import Camera, CameraMode, CameraScene, Director
from rendering.car_renderer import LIVERIES, CarRenderer
from rendering.effects import EffectLayer, draw_sensor_rays
from rendering.particles import ParticleSystem
from rendering.track_renderer import TrackRenderer
from settings import CAR_ASSET_DIR, RECORDING_DIR, Settings
from simulation.environment import CarStatus, RaceEnvironment, build_track
from ui import menus
from ui.dashboard import Dashboard, HudData
from ui.display import Display
from ui.recorder import VideoRecorder
from ui.reels import ReelsOverlay
from ui.statistics import PlotWriter
from ui.theme import AMBER, DANGER, DATA, GOLD, INK, SIGNAL, Theme


class QuitRequested(Exception):
    pass


class RestartRequested(Exception):
    pass


class RecordingComplete(Exception):
    pass


@dataclass
class AppOptions:
    presentation: bool = False      # clean layout, automatic director, no prompts
    record_path: Path | None = None
    record_seconds: float | None = None
    show_every: int = 1             # recordings can skip showing some generations
    show_title: bool = True
    speed: float = 1.0
    camera: str = CameraMode.FULL
    fullscreen: bool = False
    window: tuple[int, int] | None = None
    frame_limit: int | None = None  # exit after this many frames (used by tests)
    screenshot_frames: tuple = ()   # save the canvas on these frame numbers (used by tests)
    screenshot_dir: Path | None = None
    uncapped: bool = False          # do not wait between frames (used by benchmarks)
    reels: bool = False             # vertical short-form video: fast cuts, stops at the first lap


@dataclass
class ReelPlan:
    """What a generation will do, known in advance from its dry run."""

    hero: int               # the car that gets furthest, which the camera follows
    hero_end: float         # simulated time its run ends
    hero_crashes: bool
    final: bool             # this generation completes a lap
    last_car: int | None    # the last car to crash, if worth showing after the hero
    last_end: float


class SimulatorApp:
    REELS_IMPACT_SECONDS = 0.45      # simulated time before a crash shown in slow motion
    REELS_BUILD_UP_SECONDS = 2.5     # real time, never faster, leading up to a crash
    REELS_CRASH_HOLD_SECONDS = 0.7   # how long the camera stays on a crash afterwards
    REELS_LAST_CAR_WINDOW = 8.0      # follow the last survivor only if it crashes this soon

    def __init__(self, settings: Settings, options: AppOptions):
        self.settings = settings
        self.options = options
        self.size = (settings.display.width, settings.display.height)
        self.fixed_step = options.record_path is not None

        pygame.init()
        self.display = Display(self.size, options.fullscreen, options.window)
        self.canvas = self.display.canvas
        self.clock = pygame.time.Clock()
        self.theme = Theme()
        self._present_loading(0.02, "Building circuit")

        self.track, self.field = build_track(settings)
        self.env = RaceEnvironment(settings, self.track, self.field)
        self.renderer = TrackRenderer(self.track, self.field, settings, self.theme, self._present_loading)
        self.cars = CarRenderer(settings.physics, fine_zoom=options.reels)
        self.camera = Camera(self.renderer, settings.display.follow_zoom, self.track, self.size)
        self.camera.continuous_zoom = self.fixed_step
        if options.reels:
            self.camera.ZOOM_SMOOTHING = 1.1  # gentle, so the zoom never feels like breathing
        self.director = Director(self.camera)
        self.effects = EffectLayer(self.track)
        self.particles = ParticleSystem(settings.display.particle_limit)
        self.dashboard = Dashboard(self.theme, settings)
        self.plots = PlotWriter()
        self.liveries = [livery[0] for livery in LIVERIES]
        self._write_car_preview()

        self.recorder: VideoRecorder | None = None
        if options.record_path is not None:
            cfg = settings.recording
            self.recorder = VideoRecorder(options.record_path, self.size, cfg.fps, cfg.crf, cfg.preset,
                                          silent_audio=options.reels)

        self.manager: EvolutionManager | None = None
        self.controller: NetworkController | None = None
        self.genomes: list = []
        self.config = None
        self.species_of: dict = {}
        self.hud = HudData()
        self.camera.set_mode(CameraMode.ACTION if options.reels else options.camera, instant=True)
        self.reels = ReelsOverlay(self.theme, self.track.length * 0.1) if options.reels else None
        self.story_over = False
        self.first_lap_seconds: float | None = None

        self.paused = False
        self.show_hud = True
        self.show_sensors = settings.display.show_sensors
        self.speed = options.speed
        self.speed_before_slow = options.speed
        self.slow_motion = False
        self.accumulator = 0.0
        self.alpha = 1.0
        self.frame_number = 0
        self.real_speed = options.speed
        self.leader = 0
        self.fitness_now = np.zeros(1)
        self.leader_timer = 0.0
        self.banners: list[menus.Banner] = []
        self.best_gate = -10 ** 9
        self.best_lap_shown = 0
        self.flag_shown = False
        self.status = "TRAINING"
        self.topology_for: tuple | None = None
        self.topology = None
        self._last_time = time.perf_counter()
        self.frame_dt = 1.0 / settings.display.target_fps
        self.start_pressed = False
        self.generation_started = time.perf_counter()
        self.evaluation_run = 0

    # ------------------------------------------------------------------ window

    def _present_loading(self, fraction: float, label: str) -> None:
        pygame.event.pump()
        menus.draw_loading(self.canvas, self.theme, fraction, label)
        self.display.present()

    def _write_car_preview(self) -> None:
        path = CAR_ASSET_DIR / "liveries.png"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self.cars.save_preview(path, 16)

    # ------------------------------------------------------------------ frame plumbing

    def _begin_frame(self) -> float:
        """Handle input and return the frame's dt. Fixed while recording, for smooth video."""
        self._handle_events()
        if self.fixed_step:
            self.frame_dt = 1.0 / self.settings.recording.fps
        else:
            now = time.perf_counter()
            self.frame_dt = min(now - self._last_time, 0.1)
            self._last_time = now
        return self.frame_dt

    def _end_frame(self) -> None:
        self.frame_number += 1
        options = self.options
        if self.recorder is not None:
            self.recorder.add_frame(self.canvas)
            if self.recorder.error:
                raise RuntimeError(f"Recording failed: {self.recorder.error}")
            if options.record_seconds and self.recorder.seconds >= options.record_seconds:
                raise RecordingComplete()
        if self.frame_number in options.screenshot_frames and options.screenshot_dir:
            options.screenshot_dir.mkdir(parents=True, exist_ok=True)
            pygame.image.save(self.canvas, str(options.screenshot_dir / f"frame_{self.frame_number:05d}.png"))
        self.display.present()
        if options.frame_limit and self.frame_number >= options.frame_limit:
            raise QuitRequested()
        if not self.fixed_step and not options.uncapped:
            self.clock.tick(self.settings.display.target_fps)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.WINDOWCLOSE):
                raise QuitRequested()
            if event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

    def _handle_key(self, key: int) -> None:
        steps = self.settings.display.speed_steps
        if key == pygame.K_ESCAPE:
            raise QuitRequested()
        if key == pygame.K_RETURN:
            self.start_pressed = True
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            raise RestartRequested()
        elif key == pygame.K_f:
            self.slow_motion = False
            current = min(range(len(steps)), key=lambda i: abs(steps[i] - self.speed))
            self.speed = steps[(current + 1) % len(steps)]
        elif key == pygame.K_s:
            self.slow_motion = not self.slow_motion
            if self.slow_motion:
                self.speed_before_slow = self.speed
                self.speed = self.settings.display.slow_motion
            else:
                self.speed = self.speed_before_slow
        elif key == pygame.K_c:
            self.camera.next_mode()
        elif key == pygame.K_v:
            self.show_sensors = not self.show_sensors
        elif key == pygame.K_h:
            self.show_hud = not self.show_hud
            self.camera.center_panels = self.show_hud
        elif key == pygame.K_F11:
            self.display.toggle_fullscreen()
        elif key == pygame.K_p:
            self._screenshot()

    def _screenshot(self) -> None:
        folder = RECORDING_DIR / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"shot_{time.strftime('%Y%m%d-%H%M%S')}_{self.frame_number}.png"
        pygame.image.save(self.canvas, str(path))
        self.banners.append(menus.Banner("SCREENSHOT SAVED", path.name, DATA, life=1.6))

    # ------------------------------------------------------------------ training

    def run_training(self, manager: EvolutionManager, generations: int | None,
                     new_manager=None) -> EvolutionManager:
        """Train until told to stop. ``new_manager`` builds a fresh run when R is pressed."""
        self.manager = manager
        run_until = None if generations is None else manager.generation + generations
        try:
            if self.options.show_title and not self.reels:
                self._title_sequence(wait_for_key=not self.options.presentation)
            while (run_until is None or self.manager.generation < run_until) and not self.story_over:
                try:
                    self.manager.run_generation(self._evaluate)
                    self._after_generation()
                except RestartRequested:
                    if new_manager is None:
                        continue
                    self.manager = new_manager()
                    if generations is not None:
                        run_until = self.manager.generation + generations
                    self.hud = HudData()
                    self.banners.append(menus.Banner("TRAINING RESTARTED", "Generation 0", AMBER))
            if self.reels:
                self._reels_end_card()
            elif self.options.presentation:
                self._end_card()
        except (QuitRequested, RecordingComplete):
            pass
        finally:
            self.close()
        return self.manager

    def _after_generation(self) -> None:
        manager = self.manager
        every = self.settings.training.stats_plot_every
        if every and len(manager.history) % every == 0:
            self.plots.request(manager.history, manager.result_dir / "training_stats.png")

    def close(self) -> None:
        # The video comes first: trouble saving statistics must never cost a finished recording.
        if self.recorder is not None:
            finished = self.recorder.close()
            where = self.recorder.path if self.recorder.image_dir is None else self.recorder.image_dir
            if finished:
                print(f"Recording saved: {where} ({self.recorder.seconds:.1f}s, {self.recorder.frames} frames)")
            else:
                print(f"Recording failed: {self.recorder.error}")
            self.recorder = None
        try:
            if self.manager is not None and self.manager.history:
                self.plots.finish()
                self.plots.request(self.manager.history, self.manager.result_dir / "training_stats.png")
                self.plots.finish()
                self.manager.write_summary()
        except OSError as error:
            print(f"Could not save the training summary: {error}")
        finally:
            pygame.quit()

    def _evaluate(self, genome_pairs: list, config) -> GenerationResult:
        started = time.perf_counter()
        self.generation_started = started
        manager = self.manager
        self.config = config
        self.genomes = [genome for _, genome in genome_pairs]
        self.species_of = manager.species_map()
        self.controller = NetworkController(self.genomes, config)
        self.env.reset(len(self.genomes), self.controller)
        self._reset_race_visuals()
        generation = manager.generation
        self.status = "TRAINING"

        # Skipped generations are still real training, they just aren't shown.
        show = generation % max(self.options.show_every, 1) == 0 or generation == 0
        if self.reels:
            show = False
            self._reels_race(generation)
        elif show:
            previous = manager.previous
            self._intro(generation, previous)
            self._race()
        else:
            self.env.run_to_end()
            self.env.drain_events()

        fitness = self.env.current_fitness()
        for genome, value in zip(self.genomes, fitness.tolist()):
            genome.fitness = float(value)
        # While rendering a video, wall-clock time only measures how slow the encoding is, so
        # "training time" counts the simulated race instead and the video reads consistently.
        wall = self.env.time if self.fixed_step else time.perf_counter() - started
        result = summarize_generation(self.env, self.genomes, fitness, generation, manager.species_count(),
                                      self.species_of, wall)
        if show:
            self._results(result, manager.previous)
        return result

    # ------------------------------------------------------------------ evaluation

    def run_evaluation(self, genome, config, runs: int | None = None) -> None:
        """Watch one saved genome drive, alone, from the centre grid slot."""
        self.config = config
        self.genomes = [genome]
        self.species_of = {}
        self.status = "EVALUATING"
        count = 0
        try:
            while runs is None or count < runs:
                count += 1
                self.evaluation_run = count
                self.controller = NetworkController(self.genomes, config)
                self.env.reset(1, self.controller, slots=[self.settings.training.grid_lanes // 2])
                self._reset_race_visuals()
                try:
                    self._intro(count, None, evaluating=True)
                    self._race()
                    fitness = self.env.current_fitness()
                    result = summarize_generation(self.env, self.genomes, fitness, count, 1, {}, 0.0)
                    line = f"Run {count}  ·  lasted {result.sim_seconds:.0f}s"
                    self._results(result, None, summary_line=line)
                except RestartRequested:
                    continue
        except (QuitRequested, RecordingComplete):
            pass
        finally:
            self.close()

    # ------------------------------------------------------------------ sequences

    def _reset_race_visuals(self) -> None:
        self.accumulator = 0.0
        self.alpha = 1.0
        self.leader = 0
        self.leader_timer = 0.0
        self.best_gate = -10 ** 9
        self.best_lap_shown = 0
        self.flag_shown = False
        self.effects.clear()
        self.particles.clear()
        self.fitness_now = np.zeros(self.env.count)
        self.topology_for = None

    def _title_sequence(self, wait_for_key: bool) -> None:
        duration = self.settings.recording.title_seconds
        elapsed = 0.0
        self._last_time = time.perf_counter()
        population = self.manager.config.pop_size if self.manager else self.settings.training.population
        tagline = f"{population} neural networks. One circuit. No driving rules."
        self.start_pressed = False
        while True:
            dt = self._begin_frame()
            elapsed += dt
            if self.start_pressed:
                self.paused = False
                return
            # A slow drift across the circuit behind the title.
            drift = math.sin(elapsed * 0.25) * 60.0
            self._draw_background(self.renderer.fit_center[0] + drift, self.renderer.fit_center[1], self.renderer.fit_zoom * 1.12)
            menus.draw_title(self.canvas, self.theme, elapsed, wait_for_key, tagline)
            self._end_frame()
            if not wait_for_key and elapsed >= duration:
                return

    def _draw_background(self, x: float, y: float, zoom: float) -> None:
        left = x - self.size[0] / 2 / zoom
        top = y - self.size[1] / 2 / zoom
        renderer = self.renderer
        left = max(renderer.view_x, min(left, renderer.view_x + renderer.view_w - self.size[0] / zoom))
        top = max(renderer.view_y, min(top, renderer.view_y + renderer.view_h - self.size[1] / zoom))
        view = renderer.draw(self.canvas, left, top, zoom)
        self.camera.snap_view(*view)

    def _transition_length(self, base: float) -> float:
        """Long ceremonies get in the way when fast-forwarding, so they shrink."""
        if self.options.presentation:
            return base
        if self.speed >= 8.0:
            return min(base, 0.35)
        if self.speed >= 4.0:
            return base * 0.5
        return base

    def _intro(self, generation: int, previous: GenerationResult | None, evaluating: bool = False) -> None:
        cfg = self.settings.recording
        duration = self._transition_length(cfg.intro_seconds if self.options.presentation else cfg.intro_seconds * 0.8)
        elapsed = 0.0
        self._refresh_leader()
        if self.options.presentation:
            self.director.begin_generation()
        self._update_hud(generation, 0.0)
        self._last_time = time.perf_counter()
        while elapsed < duration:
            dt = self._begin_frame()
            if not self.paused:
                elapsed += dt
            self.alpha = 1.0
            self._update_visuals(dt)
            self._draw_scene()
            menus.draw_generation_intro(self.canvas, self.theme, elapsed, duration, generation,
                                        previous.best_fitness if previous else None,
                                        previous.best_laps if previous else None, self.env.count, evaluating)
            self._draw_pause_if_needed()
            self._end_frame()

    def _results(self, result: GenerationResult, previous: GenerationResult | None,
                 summary_line: str | None = None) -> None:
        cfg = self.settings.recording
        duration = cfg.results_seconds if self.options.presentation else self._transition_length(cfg.live_results_seconds)
        elapsed = 0.0
        self.status = "RESULTS"
        self._refresh_leader()  # so the HUD agrees with the card's final scores
        self._update_hud(result.generation, 0.0)
        self._last_time = time.perf_counter()
        while elapsed < duration:
            dt = self._begin_frame()
            if not self.paused:
                elapsed += dt
            self.alpha = 1.0
            self._update_visuals(dt)
            self._draw_scene()
            menus.draw_results(self.canvas, self.theme, elapsed, result, previous, self.liveries, summary_line)
            self._draw_pause_if_needed()
            self._end_frame()
        self.status = "TRAINING" if self.manager is not None else "EVALUATING"

    def _end_card(self) -> None:
        manager = self.manager
        if not manager.history:
            return
        elapsed = 0.0
        duration = 6.0
        best = max(manager.history, key=lambda r: r.best_fitness)
        best_lap = min((r.best_lap_seconds for r in manager.history if r.best_lap_seconds > 0), default=0.0)
        while elapsed < duration:
            dt = self._begin_frame()
            elapsed += dt
            self._draw_background(*self.renderer.fit_center, self.renderer.fit_zoom)
            menus.draw_end_card(self.canvas, self.theme, elapsed, len(manager.history), best.best_fitness,
                                max(r.best_distance_m for r in manager.history), best_lap, manager.training_seconds)
            self._end_frame()

    # ------------------------------------------------------------------ vertical video

    def _reels_plan(self) -> ReelPlan:
        """Read the outcome of the dry run: who to follow, and when the crashes happen."""
        env = self.env
        dt = env.dt
        ends = np.where(env.end_tick >= 0, env.end_tick, env.tick_count) * dt
        final = bool(env.progress.laps.max() >= 1)
        if final:
            finishers = np.nonzero(env.progress.laps >= 1)[0]
            hero = int(finishers[np.argmin(ends[finishers])])
        else:
            hero = int(np.argmax(env.progress.distance))
        hero_crashes = int(env.status[hero]) == CarStatus.CRASHED

        # The last car to crash gets its moment too, as long as it doesn't take long to arrive.
        last_car, last_end = None, 0.0
        crashed = np.nonzero(env.status == CarStatus.CRASHED)[0]
        if not final and crashed.size:
            candidate = int(crashed[np.argmax(ends[crashed])])
            if ends[hero] < ends[candidate] <= ends[hero] + self.REELS_LAST_CAR_WINDOW:
                last_car, last_end = candidate, float(ends[candidate])
        return ReelPlan(hero, float(ends[hero]), hero_crashes, final, last_car, last_end)

    def _reels_speed(self, plan: ReelPlan, stage: str, focus_end: float, focus_crashes: bool,
                     record_m: float, finishing: bool) -> float:
        """Playback speed for this frame. It only changes how many fixed ticks are shown per
        video frame, so what the cars do is unaffected.

        Dull stretches run at up to 2x, but a crash that matters is never fast-forwarded:
        real time as it approaches, then slow motion for the impact.
        """
        env = self.env
        if finishing:
            return 0.5
        remaining = focus_end - env.time
        if stage == "hold":
            return 0.5 if focus_crashes else 1.0
        if plan.final and 0.0 < self.track.length - float(env.progress.high_water[plan.hero]) < 240.0:
            return 0.5
        if focus_crashes and remaining <= self.REELS_IMPACT_SECONDS:
            return 0.5
        if remaining <= self.REELS_BUILD_UP_SECONDS or env.time < 3.0:
            return 1.0
        if stage == "last":
            return 2.0
        hero_m = max(float(env.progress.distance[plan.hero]), 0.0) * 0.1
        if hero_m > record_m - 60.0 > 0.0:
            return 1.0
        return 1.5 if plan.final else 2.0

    def _reels_race(self, generation: int) -> None:
        env = self.env
        manager = self.manager
        record_m = max((r.best_distance_m for r in manager.history), default=0.0)
        limit = self.options.record_seconds

        # Races are deterministic, so a dry run shows how this attempt will play out. The
        # editing can then follow the right car and slow down before a crash, not after it.
        env.run_to_end()
        env.drain_events()
        plan = self._reels_plan()
        self.controller = NetworkController(self.genomes, self.config)
        env.reset(env.count, self.controller)
        self._reset_race_visuals()

        self.leader = plan.hero
        self._refresh_leader()
        self._update_hud(generation, 0.0)
        self.camera.cut_to(self._camera_scene())
        self.reels.begin_generation(generation, record_m, env.count, opening=len(manager.history) == 0)

        stage = "hero"
        next_stage_car = plan.last_car
        focus_end, focus_crashes = plan.hero_end, plan.hero_crashes
        hold = 0.0
        finish_age: float | None = None
        record_announced = False
        limit_reached = False

        while True:
            dt = self._begin_frame()
            self.speed = self._reels_speed(plan, stage, focus_end, focus_crashes, record_m, finish_age is not None)
            if not self.paused:
                self._advance_simulation(dt)

            if plan.final and finish_age is None and env.progress.laps[plan.hero] >= 1:
                finish_age = 0.0
                ticks = int(env.progress.best_lap_ticks[plan.hero])
                self.first_lap_seconds = ticks * env.dt if ticks > 0 else None
                self.reels.caption("LAP COMPLETE!", GOLD, life=3.0)
            elif finish_age is not None:
                finish_age += dt
                if finish_age >= 2.8:
                    break

            hero_m = max(float(env.progress.distance[plan.hero]), 0.0) * 0.1
            if not record_announced and record_m > 0.0 and hero_m > record_m + 5.0 and finish_age is None:
                record_announced = True
                self.reels.caption("NEW RECORD!", DATA, life=1.3)

            if stage in ("hero", "last") and finish_age is None and env.time >= focus_end:
                stage = "hold"
                hold = 0.0
                if next_stage_car is None and env.active_count == 0:
                    crashed = int(np.count_nonzero(env.status == CarStatus.CRASHED))
                    self.reels.caption("ALL CRASHED" if crashed >= env.count * 0.5 else "ALL OUT", DANGER, life=0.8)
            elif stage == "hold":
                hold += dt
                wait = self.REELS_CRASH_HOLD_SECONDS if focus_crashes else 0.2
                if hold >= wait:
                    if next_stage_car is None:
                        break
                    # Go to the survivor: everything now rides on this one car.
                    self.leader = next_stage_car
                    self.camera.cut_to(self._camera_scene())
                    focus_end, focus_crashes = plan.last_end, True
                    next_stage_car = None
                    stage = "last"
                    self.reels.caption("LAST CAR", AMBER, life=1.2)

            # The camera runs on simulation time, so it can't fall behind during fast-forward.
            self._update_visuals(dt * min(self.speed, 1.0), camera_dt=dt * self.speed)
            self._update_hud(generation, env.time)
            self._draw_scene()
            self._end_frame()
            if limit and self.recorder and self.recorder.seconds >= limit:
                limit_reached = True
                break

        self.story_over = finish_age is not None or limit_reached
        # Finish the generation off camera so its scores and the best genome are still saved.
        env.run_to_end()
        env.drain_events()

    def _reels_end_card(self) -> None:
        history = self.manager.history
        if not history:
            return
        succeeded = self.first_lap_seconds is not None
        generation = history[-1].generation
        elapsed = 0.0
        self.show_hud = False  # the card is the whole message now
        while elapsed < 4.0:
            dt = self._begin_frame()
            elapsed += dt
            self._update_visuals(dt * 0.5)
            self._draw_scene()
            self.reels.draw_end(self.canvas, elapsed, generation, self.first_lap_seconds, succeeded)
            self._end_frame()

    def _race(self) -> None:
        self._last_time = time.perf_counter()
        self.status = "TRAINING" if self.manager is not None else "EVALUATING"
        while not self.env.is_over:
            dt = self._begin_frame()
            if not self.paused:
                self._advance_simulation(dt)
            self._update_visuals(dt * (self.speed if self.speed < 1.0 else 1.0) if not self.paused else 0.0)
            generation = self.manager.generation if self.manager else self.evaluation_run
            self._update_hud(generation, self.env.time)
            self._draw_scene()
            self._draw_pause_if_needed()
            self._end_frame()
        # Show the final state of the race rather than cutting away mid-interpolation.
        self.alpha = 1.0

    # ------------------------------------------------------------------ simulation stepping

    def _advance_simulation(self, dt: float) -> None:
        cfg = self.settings
        env = self.env
        self.accumulator += dt * self.speed * cfg.physics.tick_rate
        ticks = 0
        started = time.perf_counter()
        budget = cfg.display.max_sim_ms_per_frame / 1000.0
        while self.accumulator >= 1.0 and not env.is_over:
            env.tick()
            self.accumulator -= 1.0
            ticks += 1
            if not self.fixed_step and ticks % 4 == 0 and time.perf_counter() - started > budget:
                # This machine can't keep up at this speed. Run slower rather than freeze
                # the UI: time is dropped, never ticks, so the race itself is unaffected.
                self.accumulator = min(self.accumulator, 1.0)
                break
        if dt > 0:
            measured = ticks / (dt * cfg.physics.tick_rate)
            self.real_speed += (measured - self.real_speed) * 0.08
        self.alpha = float(min(self.accumulator, 1.0))
        self._process_events(env.drain_events())

    def _process_events(self, events: list) -> None:
        env = self.env
        lively = self.speed <= 4.0
        for event in events:
            if event.kind == "crash":
                self.effects.wreck(event.car)
                if lively:
                    big = self.reels is not None and event.car == self.leader
                    self.particles.sparks(event.x, event.y, event.speed, count=44 if big else 12)
            elif event.kind in ("timeout", "wrong_way"):
                self.effects.wreck(event.car)
            elif event.kind == "checkpoint":
                reached = int(env.progress.visited_upto[event.car])
                if reached > self.best_gate:
                    self.best_gate = reached
                    self.effects.checkpoint(event.value)
            elif event.kind == "lap":
                if event.value > self.best_lap_shown:
                    self.best_lap_shown = event.value
                    seconds = event.ticks * env.dt
                    title = f"LAP {event.value} COMPLETE"
                    livery = self.liveries[event.car % len(self.liveries)]
                    if not self.reels:
                        self.banners.append(menus.Banner(title, f"#{self._genome_id(event.car)}  ·  {seconds:.2f} seconds", livery))
                    self.effects.checkpoint(0, GOLD)
                    if lively:
                        self.particles.confetti(event.x, event.y)
            elif event.kind == "finish" and not self.flag_shown and not self.reels:
                self.flag_shown = True
                self.banners.append(menus.Banner("CHEQUERED FLAG", f"#{self._genome_id(event.car)} finishes the race", GOLD, life=3.4))

    def _genome_id(self, car: int):
        return getattr(self.genomes[car], "key", car)

    # ------------------------------------------------------------------ visuals

    def _refresh_leader(self) -> None:
        env = self.env
        self.fitness_now = env.current_fitness()
        if self.reels:
            return  # the edit chooses which car to follow
        moving = (env.status == CarStatus.RUNNING) | (env.status == CarStatus.FINISHED)
        pool = np.where(moving, self.fitness_now, -1.0)
        best = int(np.argmax(pool)) if moving.any() else int(np.argmax(self.fitness_now))
        # A small margin stops the highlight flickering between two close cars.
        if not moving[self.leader] or self.fitness_now[best] > self.fitness_now[self.leader] * 1.01 + 1.0:
            self.leader = best

    def _pose(self, car: int) -> tuple[float, float, float]:
        env = self.env
        a = self.alpha
        x = env.prev_x[car] + (env.car.x[car] - env.prev_x[car]) * a
        y = env.prev_y[car] + (env.car.y[car] - env.prev_y[car]) * a
        turn = (env.car.heading[car] - env.prev_heading[car] + math.pi) % (2 * math.pi) - math.pi
        return float(x), float(y), float(env.prev_heading[car] + turn * a)

    def _update_visuals(self, dt: float, camera_dt: float | None = None) -> None:
        env = self.env
        self.leader_timer += dt
        if self.leader_timer >= 0.2 or self.leader_timer == 0.0:
            self.leader_timer = 0.0
            self._refresh_leader()

        self.effects.update(dt)
        self.particles.update(dt)
        for banner in self.banners:
            banner.update(dt if dt > 0 else 0.0)
        self.banners = [b for b in self.banners if b.alive]

        if self.speed <= 4.0 and dt > 0 and not self.env.is_over:
            sliding = np.nonzero(
                (np.abs(env.car.lateral_speed) > self.settings.physics.skid_threshold)
                & (env.status == CarStatus.RUNNING))[0]
            for car in sliding[:8].tolist():
                c, s = math.cos(env.car.heading[car]), math.sin(env.car.heading[car])
                vx = env.car.forward_speed[car] * c
                vy = env.car.forward_speed[car] * s
                for side in (-1.0, 1.0):
                    self.particles.smoke(env.car.x[car] - c * 17.0 - s * side * 9.0,
                                         env.car.y[car] - s * 17.0 + c * side * 9.0, vx, vy)

        if self.options.presentation and dt > 0 and not self.reels:
            self.director.update(dt)

        step = dt if camera_dt is None else camera_dt
        self.camera.update(step if step > 0 else 1e-4, self._camera_scene())

    def _camera_scene(self) -> CameraScene:
        env = self.env
        x, y, heading = self._pose(self.leader)
        speed = float(env.car.forward_speed[self.leader])
        s = float(env.progress.last_s[self.leader])
        moving = np.nonzero((env.status == CarStatus.RUNNING) | (env.status == CarStatus.FINISHED))[0]
        top = moving[np.argsort(-self.fitness_now[moving])[:8]] if moving.size else moving
        others = np.column_stack([env.car.x[top], env.car.y[top]]) if top.size else np.zeros((0, 2))
        return CameraScene((x, y), (math.cos(heading) * speed, math.sin(heading) * speed), s, speed, others)

    def _speed_label(self) -> str:
        if self.paused:
            return "PAUSED"
        if self.speed < 1.0:
            return f"{self.speed:g}× SLOW"
        # Tell the truth if the machine can't actually sustain the requested speed.
        if self.speed > 1.0 and not self.fixed_step and self.real_speed < self.speed * 0.85:
            return f"{self.real_speed:.1f}×"
        return f"{self.speed:g}×"

    def _update_hud(self, generation: int, race_time: float) -> None:
        env = self.env
        hud = self.hud
        manager = self.manager
        hud.generation = generation
        hud.generation_label = "GENERATION" if manager else "RUN"
        hud.population = env.count
        hud.active = env.active_count
        hud.eliminated = env.eliminated_count
        hud.finished = env.finished_count
        hud.speed_label = self._speed_label()
        hud.status = "PAUSED" if self.paused else self.status
        hud.camera = self.camera.mode
        hud.best_fitness = float(self.fitness_now.max()) if self.fitness_now.size else 0.0
        hud.average_fitness = float(self.fitness_now.mean()) if self.fitness_now.size else 0.0
        hud.best_distance_m = max(float(env.progress.distance.max()), 0.0) * 0.1
        hud.laps = int(env.progress.laps.sum())
        hud.race_time = race_time
        hud.race_limit = self.settings.training.generation_seconds
        elapsed = race_time if self.fixed_step else time.perf_counter() - self.generation_started
        hud.training_seconds = (manager.training_seconds if manager else 0.0) + (elapsed if race_time > 0 else 0.0)
        if manager:
            previous = manager.previous
            hud.previous_best = previous.best_fitness if previous else None
            hud.previous_average = previous.average_fitness if previous else None
            hud.fitness_history = [r.best_fitness for r in manager.history]
            hud.average_history = [r.average_fitness for r in manager.history]

        car = self.leader
        genome = self.genomes[car] if car < len(self.genomes) else None
        view = CarView(env, car, genome)
        hud.genome_id = view.genome_id
        hud.species = self.species_of.get(hud.genome_id)
        hud.car_status = view.status_label
        hud.speed_kmh = view.speed_kmh
        hud.car_laps = view.laps
        hud.car_number = car + 1
        hud.car_distance_m = view.distance_m
        hud.lap_fraction = view.lap_fraction
        hud.best_lap = view.best_lap_seconds
        hud.livery = self.liveries[car % len(self.liveries)]
        hud.readings = view.sensor_readings.tolist()
        hud.steer = view.steer
        hud.throttle = view.throttle
        if genome is not None and self.config is not None:
            key = (hud.genome_id, id(genome))
            if self.topology_for != key:
                self.topology_for = key
                self.topology = layout_topology(genome, self.config)
            hud.topology = self.topology
            hud.node_values = self.controller.node_values(car) if self.controller else {}

    # ------------------------------------------------------------------ drawing

    def _draw_scene(self) -> None:
        canvas = self.canvas
        camera = self.camera
        view = self.renderer.draw(canvas, camera.left, camera.top, camera.zoom)
        camera.snap_view(*view)
        self.effects.draw_gates(canvas, camera)
        self._draw_cars()
        self.particles.draw(canvas, camera)
        # With the HUD hidden for a Reel's end card, the markers on the car go too.
        marked = self.show_hud or not self.reels
        if self.show_sensors and marked:
            self._draw_sensors()
        if marked:
            self._draw_leader_tag()
        if self.reels:
            self.reels.update(self.frame_dt, self.hud)
            if self.show_hud:
                self.reels.draw(canvas, self.hud, self.speed)
            return
        if self.show_hud:
            presentation = self.options.presentation
            self.dashboard.update(self.frame_dt, self.hud)
            self.dashboard.draw_top(canvas, self.hud, presentation)
            self.dashboard.draw_bottom(canvas, self.hud)
            self.dashboard.draw_side(canvas, self.hud, presentation)
        center = self.renderer.free_rect.centerx if self.show_hud else self.size[0] // 2
        y = self.settings.display.hud_top + 22
        for banner in self.banners[-2:]:
            banner.draw(canvas, self.theme, center, y)
            y += 108

    def _draw_pause_if_needed(self) -> None:
        if self.paused:
            center = self.renderer.free_rect.center if self.show_hud else (self.size[0] // 2, self.size[1] // 2)
            menus.draw_pause(self.canvas, self.theme, center)

    def _on_screen(self, sx: float, sy: float, margin: float = 80.0) -> bool:
        return -margin < sx < self.size[0] + margin and -margin < sy < self.size[1] + margin

    def _draw_cars(self) -> None:
        env = self.env
        canvas = self.canvas
        camera = self.camera
        zoom = camera.view_zoom
        cars = self.cars
        shadow_offset = (3.0 * zoom, 4.0 * zoom)

        turned = self.renderer.heading_offset  # sprites face the rotated view, not the world
        wreck_age = {w.car: w for w in self.effects.wrecks}
        for wreck in self.effects.wrecks:
            x, y = env.car.x[wreck.car], env.car.y[wreck.car]
            sx, sy = camera.to_screen(x, y)
            if not self._on_screen(sx, sy):
                continue
            # Eliminated cars lose their colour and fade away.
            sprite = cars.sprite(wreck.car, env.car.heading[wreck.car] + turned, zoom)
            t = wreck.age / wreck.life
            faded = sprite.copy()
            faded.fill((150, 150, 150, 255), special_flags=pygame.BLEND_RGBA_MULT)
            faded.set_alpha(int(255 * (1.0 - t) ** 1.5))
            canvas.blit(faded, faded.get_rect(center=(sx, sy)))

        moving = np.nonzero((env.status == CarStatus.RUNNING) | (env.status == CarStatus.FINISHED))[0]
        parked = np.nonzero(env.status == CarStatus.PARKED)[0]
        ordered = list(parked.tolist()) + moving[np.argsort(self.fitness_now[moving])].tolist()
        glow_from = len(ordered) - self.settings.display.glow_cars
        detail_from = len(ordered) - self.settings.display.detail_cars
        a = self.alpha
        px, py, ph = env.prev_x, env.prev_y, env.prev_heading
        cx, cy, ch = env.car.x, env.car.y, env.car.heading
        for rank, car in enumerate(ordered):
            if car in wreck_age:
                continue
            x = px[car] + (cx[car] - px[car]) * a
            y = py[car] + (cy[car] - py[car]) * a
            sx, sy = camera.to_screen(x, y)
            if not self._on_screen(sx, sy):
                continue
            turn = (ch[car] - ph[car] + math.pi) % (2 * math.pi) - math.pi
            heading = ph[car] + turn * a + turned
            shadow = cars.shadow(heading, zoom)
            canvas.blit(shadow, shadow.get_rect(center=(sx + shadow_offset[0], sy + shadow_offset[1])))
            if rank >= glow_from:
                self.effects.headlight(canvas, sx, sy, heading, zoom)
            if car == self.leader:
                self.effects.halo(canvas, sx, sy, zoom, SIGNAL)
            # Only the leading few cars animate wheels and steering. Those are the ones
            # the eye follows, and it keeps the sprite cache from churning.
            if rank >= detail_from:
                phase = int(env.tick_count * env.car.forward_speed[car] / 240.0)
                sprite = cars.sprite(car, heading, zoom, env.car.steer[car], phase, detail=True)
            else:
                sprite = cars.sprite(car, heading, zoom)
            canvas.blit(sprite, sprite.get_rect(center=(sx, sy)))

    def _sensor_cars(self) -> list[int]:
        env = self.env
        count = self.settings.display.sensor_cars
        if count <= 1:
            return [self.leader] if env.status[self.leader] in (CarStatus.RUNNING, CarStatus.FINISHED) else []
        moving = np.nonzero(env.status == CarStatus.RUNNING)[0]
        return moving[np.argsort(-self.fitness_now[moving])[:count]].tolist()

    def _draw_sensors(self) -> None:
        env = self.env
        for car in self._sensor_cars():
            x, y, heading = self._pose(car)
            ends = env.sensors.endpoints(x, y, heading, env.readings[car])
            draw_sensor_rays(self.canvas, self.camera, (x, y), ends, env.readings[car])

    def _draw_reels_tag(self, x: float, y: float, heading: float) -> None:
        """The followed car's number. It trails behind the car, so it never hides the road ahead."""
        reach = 62.0
        tx, ty = self.camera.to_screen(x - math.cos(heading) * reach, y - math.sin(heading) * reach)
        label = self.theme.text(f"#{self.leader + 1}", 34, INK, "Bold")
        width, height = label.get_width() + 26, 44
        panel = self.theme.panel(width, height, alpha=232, radius=10)
        rect = panel.get_rect(center=(tx, ty))
        self.canvas.blit(panel, rect)
        pygame.draw.rect(self.canvas, self.liveries[self.leader % len(self.liveries)],
                         (rect.x + 5, rect.y + 9, 5, height - 18), border_radius=2)
        self.canvas.blit(label, label.get_rect(center=(rect.centerx + 3, rect.centery)))

    def _draw_leader_tag(self) -> None:
        env = self.env
        if env.status[self.leader] not in (CarStatus.RUNNING, CarStatus.FINISHED):
            return
        x, y, heading = self._pose(self.leader)
        sx, sy = self.camera.to_screen(x, y)
        if not self._on_screen(sx, sy):
            return
        if self.reels:
            self._draw_reels_tag(x, y, heading)
            return
        offset = 40 * max(self.camera.view_zoom, 0.72)
        text, size, height = f"P1  #{self._genome_id(self.leader)}", 20, 26
        label = self.theme.text(text, size, INK, "Bold")
        width = label.get_width() + 18
        panel = self.theme.panel(width, height, alpha=220, radius=6, color=(120, 34, 16), border=30)
        rect = panel.get_rect(midbottom=(sx, sy - offset))
        self.canvas.blit(panel, rect)
        self.canvas.blit(label, label.get_rect(center=rect.center))
