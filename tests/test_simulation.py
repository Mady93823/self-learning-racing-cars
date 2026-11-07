"""Checks that the simulation is correct and that learning is real.

Run from the project root:  .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import settings as paths
from ai.evolution import EvolutionManager, load_genome, load_neat_config, summarize_generation
from ai.neural_network import NetworkController, layout_topology
from settings import Settings
from simulation.environment import CarStatus, RaceEnvironment, build_track

SETTINGS = Settings()
TRACK, FIELD = build_track(SETTINGS)


def centre_line_driver(env_getter, speed_gain=1.0, throttle=0.45):
    """A scripted test driver that steers towards the centre line. Used only by tests."""

    def control(ids, observations):
        env = env_getter()
        steer = np.zeros(ids.size)
        for position, car in enumerate(ids):
            index = int(env.track_index[car])
            look = (index + 8) % TRACK.count
            target = TRACK.points[look]
            wanted = math.atan2(target[1] - env.car.y[car], target[0] - env.car.x[car])
            error = (wanted - env.car.heading[car] + math.pi) % (2 * math.pi) - math.pi
            steer[position] = np.clip(error * 3.0 * speed_gain, -1.0, 1.0)
        return steer, np.full(ids.size, throttle)

    return control


def constant(steer=0.0, throttle=1.0):
    def control(ids, observations):
        return np.full(ids.size, float(steer)), np.full(ids.size, float(throttle))

    return control


def new_env(settings=None):
    return RaceEnvironment(settings or SETTINGS, TRACK, FIELD)


class TrackGeometry(unittest.TestCase):
    def test_closed_loop_with_uniform_spacing(self):
        gaps = np.linalg.norm(np.roll(TRACK.points, -1, axis=0) - TRACK.points, axis=1)
        self.assertLess(gaps.max(), TRACK.spacing * 1.1, "the loop must close with no gap")
        self.assertGreater(gaps.min(), TRACK.spacing * 0.85)

    def test_sections_never_touch(self):
        barrier = 10.0
        self.assertGreater(TRACK.min_clearance(), 2 * (TRACK.half_width + barrier))

    def test_corners_are_wider_than_the_road_so_edges_stay_valid(self):
        self.assertGreater(TRACK.min_radius(), TRACK.half_width)


class CollisionField(unittest.TestCase):
    def test_sdf_matches_geometry(self):
        for fraction in np.linspace(0.0, 0.98, 60):
            i = int(fraction * TRACK.count)
            p, n = TRACK.points[i], TRACK.normals[i]
            # Distance has a sharp V at the exact centre, where interpolation reads slightly
            # low. That only shortens a safe ray step. Accuracy matters near the barriers.
            centre = FIELD.sample_sdf(p[:1], p[1:])[0]
            self.assertLessEqual(centre, TRACK.half_width + 0.1)
            self.assertGreater(centre, TRACK.half_width - SETTINGS.track.field_cell * 0.75)
            for lateral, expected in ((TRACK.half_width - 20.0, 20.0), (TRACK.half_width - 4.0, 4.0)):
                q = p + n * lateral
                value = FIELD.sample_sdf(np.array([q[0]]), np.array([q[1]]))[0]
                self.assertAlmostEqual(value, expected, delta=0.5)
            outside = p + n * (TRACK.half_width + 12.0)
            self.assertLess(FIELD.sample_sdf(np.array([outside[0]]), np.array([outside[1]]))[0], 0.0)

    def test_projection_recovers_arc_length(self):
        indices = np.arange(0, TRACK.count, 17)
        s, _ = FIELD.project(TRACK.points[indices, 0], TRACK.points[indices, 1])
        error = np.abs((s - indices * TRACK.spacing + TRACK.length / 2) % TRACK.length - TRACK.length / 2)
        self.assertLess(error.max(), 1.0)

    def test_raycast_agrees_with_exact_segment_intersection(self):
        """Sphere tracing is checked against a brute-force ray/boundary intersection."""
        rng = np.random.default_rng(1)
        count = 300
        i = rng.integers(0, TRACK.count, count)
        lateral = rng.uniform(-50, 50, count)
        start = TRACK.points[i] + TRACK.normals[i] * lateral[:, None]
        angle = rng.uniform(-math.pi, math.pi, count)
        direction = np.column_stack([np.cos(angle), np.sin(angle)])
        limit = SETTINGS.sensors.max_range
        result = FIELD.raycast(start[:, 0], start[:, 1], direction[:, 0], direction[:, 1], limit,
                               SETTINGS.sensors.march_steps, SETTINGS.sensors.hit_epsilon)

        edges = []
        for side in (-1.0, 1.0):
            line = TRACK.points + TRACK.normals * side * TRACK.half_width
            edges.append((line, np.roll(line, -1, axis=0)))
        a = np.vstack([e[0] for e in edges])
        b = np.vstack([e[1] for e in edges])
        exact = np.full(count, limit)
        for ray in range(count):
            d = direction[ray]
            segment = b - a
            denominator = d[0] * segment[:, 1] - d[1] * segment[:, 0]
            delta = a - start[ray]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (delta[:, 0] * segment[:, 1] - delta[:, 1] * segment[:, 0]) / denominator
                u = (delta[:, 0] * d[1] - delta[:, 1] * d[0]) / denominator
            hits = t[(t > 0) & (u >= 0) & (u <= 1) & np.isfinite(t)]
            if hits.size:
                exact[ray] = min(limit, hits.min())
        # Allowed error: hit tolerance plus one grid cell. Grazing rays may stop slightly early.
        self.assertLess(np.percentile(np.abs(result - exact), 95), 8.0)
        self.assertLess(np.median(np.abs(result - exact)), 2.5)


class Physics(unittest.TestCase):
    def test_reaches_expected_top_speed_and_never_reverses(self):
        env = new_env()
        env.reset(1, centre_line_driver(lambda: env, throttle=1.0))
        peak = 0.0
        for _ in range(240):
            env.tick()
            peak = max(peak, env.car.forward_speed[0])
        self.assertGreater(peak, 300.0)
        self.assertLessEqual(peak, SETTINGS.physics.max_speed * 1.001)

        env.reset(1, constant(0.0, -1.0))
        for _ in range(120):
            env.tick()
        self.assertEqual(env.car.forward_speed[0], 0.0, "braking from rest must not reverse")

    def test_car_driven_into_a_wall_crashes(self):
        env = new_env()
        env.reset(1, constant(1.0, 1.0))  # full lock, full throttle
        env.run_to_end()
        self.assertEqual(int(env.status[0]), CarStatus.CRASHED)
        self.assertEqual(env.car.forward_speed[0], 0.0)

    def test_all_grid_slots_start_on_the_road(self):
        env = new_env()
        env.reset(100, constant(0.0, 0.0))
        env.tick()
        self.assertFalse(np.any(env.status == CarStatus.CRASHED))


class ProgressAndExploits(unittest.TestCase):
    def test_a_real_lap_counts_as_exactly_one_lap(self):
        settings = Settings()
        settings.training.target_laps = 1
        settings.training.generation_seconds = 120.0
        env = new_env(settings)
        env.reset(1, centre_line_driver(lambda: env, throttle=0.38))
        env.run_to_end()
        self.assertEqual(int(env.status[0]), CarStatus.FINISHED, CarStatus.LABELS[int(env.status[0])])
        self.assertEqual(int(env.progress.laps[0]), 1)
        self.assertEqual(int(env.progress.checkpoints_earned[0]), TRACK.checkpoint_count)
        self.assertAlmostEqual(float(env.progress.distance[0]), TRACK.length + 30.0, delta=20.0)

    def test_going_back_and_forth_earns_no_extra_progress(self):
        env = new_env()
        env.reset(1, constant(0.0, 0.0))
        tracker = env.progress
        ids = np.array([0])
        spacing = TRACK.length / TRACK.checkpoint_count
        forward = spacing * 2.5
        for tick, s in enumerate(np.arange(0.0, forward, 20.0), start=1):
            tracker.update(ids, np.array([s % TRACK.length]), tick)
        distance = float(tracker.distance[0])
        earned = int(tracker.checkpoints_earned[0])
        tick = 1000
        for _ in range(4):
            for s in np.arange(forward, forward - spacing * 1.5, -20.0):
                tick += 1
                tracker.update(ids, np.array([s % TRACK.length]), tick)
            for s in np.arange(forward - spacing * 1.5, forward, 20.0):
                tick += 1
                tracker.update(ids, np.array([s % TRACK.length]), tick)
        self.assertLessEqual(float(tracker.distance[0]), distance + 20.0)
        self.assertEqual(int(tracker.checkpoints_earned[0]), earned, "no checkpoint can be earned twice")
        self.assertGreaterEqual(int(tracker.revisits[0]), 4)
        self.assertGreater(float(tracker.backward_distance[0]), 0.0)

    def test_stationary_car_is_eliminated_for_inactivity_with_zero_fitness(self):
        env = new_env()
        env.reset(1, constant(0.0, 0.0))
        env.run_to_end()
        self.assertEqual(int(env.status[0]), CarStatus.TIMED_OUT)
        self.assertEqual(float(env.current_fitness()[0]), 0.0)
        self.assertAlmostEqual(env.time, SETTINGS.fitness.inactivity_timeout, delta=0.2)

    def test_wrong_way_car_is_eliminated_and_scores_nothing(self):
        env = new_env()
        env.reset(1, constant(0.0, 0.0))
        env.car.heading[0] += math.pi
        env.controller = centre_line_wrong_way(env)
        env.run_to_end()
        self.assertIn(int(env.status[0]), (CarStatus.WRONG_WAY, CarStatus.CRASHED))
        self.assertLess(float(env.progress.high_water[0] - env.progress.start_relative[0]), 30.0)
        self.assertEqual(float(env.current_fitness()[0]), 0.0)

    def test_spinning_on_the_spot_scores_less_than_driving_forward(self):
        env = new_env()
        env.reset(2, constant(0.0, 0.0), slots=[2, 2])
        driver = centre_line_driver(lambda: env, throttle=0.35)

        def mixed(ids, observations):
            steer, throttle = driver(ids, observations)
            for position, car in enumerate(ids):
                if car == 1:
                    steer[position], throttle[position] = 1.0, 0.12
            return steer, throttle

        env.controller = mixed
        for _ in range(600):
            if env.is_over:
                break
            env.tick()
        fitness = env.current_fitness()
        self.assertGreater(fitness[0], fitness[1] * 2 + 50.0)

    def test_crashing_further_along_always_scores_higher(self):
        """Regression test: a flat crash penalty once scored every short run as zero."""
        scores = []
        for ticks in (150, 300, 600):
            env = new_env()
            env.reset(1, centre_line_driver(lambda e=env: e, throttle=0.4))
            for _ in range(ticks):
                env.tick()
            env._eliminate(np.array([0]), CarStatus.CRASHED, "crash")
            scores.append(float(env.current_fitness()[0]))
        self.assertGreater(scores[0], 0.0)
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])

    def test_finishing_beats_crashing_at_the_same_point(self):
        settings = Settings()
        settings.training.target_laps = 1
        settings.training.generation_seconds = 150.0
        env = new_env(settings)
        env.reset(1, centre_line_driver(lambda e=env: e, throttle=0.38))
        env.run_to_end()
        finished = float(env.current_fitness()[0])
        env.status[0] = CarStatus.CRASHED
        self.assertGreater(finished, float(env.current_fitness()[0]))

    def test_finishing_faster_scores_higher(self):
        scores = {}
        for throttle in (0.3, 0.42):
            settings = Settings()
            settings.training.target_laps = 1
            settings.training.generation_seconds = 150.0
            env = new_env(settings)
            env.reset(1, centre_line_driver(lambda e=env: e, throttle=throttle))
            env.run_to_end()
            self.assertEqual(int(env.status[0]), CarStatus.FINISHED)
            scores[throttle] = float(env.current_fitness()[0])
        self.assertGreater(scores[0.42], scores[0.3])


def centre_line_wrong_way(env):
    def control(ids, observations):
        steer = np.zeros(ids.size)
        for position, car in enumerate(ids):
            index = int(env.track_index[car])
            target = TRACK.points[(index - 8) % TRACK.count]
            wanted = math.atan2(target[1] - env.car.y[car], target[0] - env.car.x[car])
            error = (wanted - env.car.heading[car] + math.pi) % (2 * math.pi) - math.pi
            steer[position] = np.clip(error * 3.0, -1.0, 1.0)
        return steer, np.full(ids.size, 0.4)

    return control


class Determinism(unittest.TestCase):
    def _population(self, seed=7, count=20):
        random.seed(seed)
        config = load_neat_config(population=count)
        import neat
        population = neat.Population(config, seed=seed)
        return [g for g in population.population.values()], config

    def test_same_genomes_produce_identical_races(self):
        genomes, config = self._population()
        env = new_env()
        env.reset(len(genomes), NetworkController(genomes, config))
        env.run_to_end()
        first = env.current_fitness().copy()
        env.reset(len(genomes), NetworkController(genomes, config))
        env.run_to_end()
        np.testing.assert_array_equal(first, env.current_fitness())

    def test_visual_app_gets_the_same_result_as_headless_at_any_speed(self):
        """Rendering, fast-forward and frame timing must not change the outcome."""
        import pygame
        from ui.app import AppOptions, SimulatorApp

        genomes, config = self._population(count=12)
        env = new_env()
        env.reset(len(genomes), NetworkController(genomes, config))
        env.run_to_end()
        expected = env.current_fitness().copy()

        app = SimulatorApp(Settings(), AppOptions(show_title=False, speed=8.0, uncapped=True))
        try:
            app.config = config
            app.genomes = genomes
            app.controller = NetworkController(genomes, config)
            app.env.reset(len(genomes), app.controller)
            app._reset_race_visuals()
            app._race()
            np.testing.assert_array_equal(expected, app.env.current_fitness())
        finally:
            pygame.quit()


class Controls(unittest.TestCase):
    """Every documented key, sent through the same event path as a real key press."""

    @classmethod
    def setUpClass(cls):
        from ui.app import AppOptions, SimulatorApp
        cls.app = SimulatorApp(Settings(), AppOptions(show_title=False, uncapped=True))
        # Spend all banked time. The live frame budget would otherwise drop some on slow machines.
        cls.app.fixed_step = True

    @classmethod
    def tearDownClass(cls):
        import pygame
        pygame.quit()

    def press(self, key):
        import pygame
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
        self.app._handle_events()

    def test_keys(self):
        import pygame
        from rendering.camera import CameraMode
        from ui.app import QuitRequested, RestartRequested
        app = self.app

        self.press(pygame.K_SPACE)
        self.assertTrue(app.paused)
        self.press(pygame.K_SPACE)
        self.assertFalse(app.paused)

        speeds = []
        for _ in range(5):
            self.press(pygame.K_f)
            speeds.append(app.speed)
        self.assertEqual(speeds, [2.0, 4.0, 8.0, 16.0, 1.0])

        self.press(pygame.K_s)
        self.assertEqual(app.speed, 0.25)
        self.press(pygame.K_s)
        self.assertEqual(app.speed, 1.0)

        modes = []
        start = app.camera.mode
        for _ in range(3):
            self.press(pygame.K_c)
            modes.append(app.camera.mode)
        self.assertEqual(modes[-1], start)
        self.assertEqual(set(modes), set(CameraMode.ORDER))

        before = app.show_sensors
        self.press(pygame.K_v)
        self.assertNotEqual(app.show_sensors, before)
        self.press(pygame.K_h)
        self.assertFalse(app.show_hud)
        self.press(pygame.K_h)
        self.assertTrue(app.show_hud)

        with self.assertRaises(RestartRequested):
            self.press(pygame.K_r)
        with self.assertRaises(QuitRequested):
            self.press(pygame.K_ESCAPE)

    def test_pause_freezes_the_simulation(self):
        import pygame
        app = self.app
        app.env.reset(1, constant(0.0, 1.0))
        app._reset_race_visuals()
        app.camera.set_mode(app.camera.mode, instant=True)
        self.press(pygame.K_SPACE)
        ticks = app.env.tick_count
        # The race loop only advances the simulation when not paused.
        if not app.paused:
            app._advance_simulation(0.5)
        self.assertEqual(app.env.tick_count, ticks)
        self.press(pygame.K_SPACE)
        app._advance_simulation(0.5)
        self.assertEqual(app.env.tick_count, ticks + 30)

    def test_slow_motion_and_fast_forward_scale_time_only(self):
        app = self.app
        app.env.reset(1, constant(0.0, 1.0))
        app.accumulator = 0.0
        app.speed = 0.25
        app._advance_simulation(1.0)
        self.assertEqual(app.env.tick_count, 15)
        app.speed = 1.0


class VerticalVideo(unittest.TestCase):
    """The Reels pipeline, without encoding a video."""

    def test_camera_keeps_the_followed_car_in_the_clear_zone(self):
        """The car must sit below the cards and above Instagram's caption, not merely on screen."""
        import pygame
        from ui.app import AppOptions, SimulatorApp

        tmp = tempfile.TemporaryDirectory()
        saved = (paths.CHECKPOINT_DIR, paths.RESULT_DIR, paths.BEST_GENOME_PATH)
        root = Path(tmp.name)
        paths.CHECKPOINT_DIR, paths.RESULT_DIR, paths.BEST_GENOME_PATH = root / "c", root / "r", root / "b.pkl"
        try:
            settings = Settings()
            settings.apply_reels_profile()
            manager = EvolutionManager(settings, session="reel", population=30, seed=33396)
            app = SimulatorApp(settings, AppOptions(reels=True, presentation=True, uncapped=True))
            app.fixed_step = True
            frames = [0, 0]

            def end_frame():
                app.frame_number += 1
                sx, sy = app.camera.to_screen(*app._pose(app.leader)[:2])
                frames[0] += 1
                frames[1] += 65 <= sx <= 1015 and 430 <= sy <= 1250

            def draw_scene():
                app.camera.snap_view(app.camera.left, app.camera.top, app.camera.zoom)
                app.reels.update(app.frame_dt, app.hud)

            app._end_frame = end_frame
            app._draw_scene = draw_scene
            app.run_training(manager, 6)
            self.assertGreater(frames[0], 100)
            self.assertGreater(frames[1] / frames[0], 0.95)
            self.assertEqual(app.camera.mode, "ACTION")
        finally:
            paths.CHECKPOINT_DIR, paths.RESULT_DIR, paths.BEST_GENOME_PATH = saved
            pygame.quit()
            tmp.cleanup()

    def test_portrait_view_is_a_true_rotation_of_the_landscape_world(self):
        import pygame
        from rendering.camera import Camera
        from rendering.track_renderer import TrackRenderer
        from ui.theme import Theme

        settings = Settings()
        settings.apply_reels_profile()
        pygame.init()
        pygame.display.set_mode((settings.display.width, settings.display.height))
        try:
            renderer = TrackRenderer(TRACK, FIELD, settings, Theme())
            self.assertEqual(renderer.mips[-1][1].get_width(), renderer.pixel_h)
            self.assertEqual(renderer.mips[-1][1].get_height(), renderer.pixel_w)
            camera = Camera(renderer, 1.25, TRACK, (1080, 1920))
            # The whole circuit must fit on a portrait screen in the full view.
            for x, y in TRACK.points[::50]:
                sx, sy = camera.to_screen(x, y)
                self.assertTrue(0 <= sx <= 1080 and 0 <= sy <= 1920, (sx, sy))
            # A car moving along +x in the world moves down the rotated screen.
            a = camera.to_screen(1000.0, 800.0)
            b = camera.to_screen(1100.0, 800.0)
            self.assertAlmostEqual(a[0], b[0])
            self.assertGreater(b[1], a[1])
        finally:
            pygame.quit()

    def test_a_statistics_failure_cannot_lose_a_finished_video(self):
        """Regression test: the summary used to be written first, and could stop the video closing."""
        import pygame
        from ui.app import AppOptions, SimulatorApp

        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "clip.mp4"
            app = SimulatorApp(Settings(), AppOptions(record_path=video, show_title=False))
            for _ in range(5):
                app.canvas.fill((40, 60, 80))
                app.recorder.add_frame(app.canvas)

            class BrokenManager:
                history = [object()]
                result_dir = Path(folder) / "does" / "not" / "exist"

                def write_summary(self):
                    raise OSError("disk trouble")

            app.manager = BrokenManager()
            app.plots.request = lambda *args: None
            app.close()
            self.assertTrue(video.exists())
            self.assertGreater(video.stat().st_size, 0)
        pygame.quit()

    def test_a_crash_is_never_fast_forwarded(self):
        """The approach to a crash runs at real time, and the impact in slow motion."""
        import pygame
        from ui.app import AppOptions, ReelPlan, SimulatorApp

        settings = Settings()
        settings.apply_reels_profile()
        app = SimulatorApp(settings, AppOptions(reels=True, presentation=True, uncapped=True))
        try:
            app.env.reset(1, constant(0.0, 0.5))
            plan = ReelPlan(0, 10.0, True, False, None, 0.0)
            speeds = {}
            for tick in range(600):
                app.env.tick_count = tick
                speeds[app.env.time] = app._reels_speed(plan, "hero", 10.0, True, 5000.0, False)
            self.assertEqual(max(speeds.values()), 2.0, "dull stretches may run at up to 2x")
            self.assertTrue(all(v <= 1.0 for t, v in speeds.items() if t >= 10.0 - app.REELS_BUILD_UP_SECONDS))
            self.assertTrue(all(v == 0.5 for t, v in speeds.items() if t >= 10.0 - app.REELS_IMPACT_SECONDS))
            self.assertTrue(all(v == 1.0 for t, v in speeds.items() if t < 3.0))
        finally:
            pygame.quit()

    def test_reels_profile_leaves_physics_alone(self):
        settings = Settings()
        settings.apply_reels_profile()
        self.assertEqual(settings.physics, Settings().physics)
        self.assertEqual(settings.sensors, Settings().sensors)
        self.assertEqual((settings.display.width, settings.display.height), (1080, 1920))


class RealLearning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._saved = (paths.CHECKPOINT_DIR, paths.RESULT_DIR, paths.BEST_GENOME_PATH)
        paths.CHECKPOINT_DIR = root / "checkpoints"
        paths.RESULT_DIR = root / "results"
        paths.BEST_GENOME_PATH = root / "best_genome.pkl"

    def tearDown(self):
        paths.CHECKPOINT_DIR, paths.RESULT_DIR, paths.BEST_GENOME_PATH = self._saved
        self.tmp.cleanup()

    def _evaluator(self, env, manager):
        def evaluate(pairs, config):
            genomes = [g for _, g in pairs]
            env.reset(len(genomes), NetworkController(genomes, config))
            env.run_to_end()
            fitness = env.current_fitness()
            for genome, value in zip(genomes, fitness.tolist()):
                genome.fitness = float(value)
            return summarize_generation(env, genomes, fitness, manager.generation, manager.species_count(),
                                        manager.species_map(), 0.0)
        return evaluate

    def test_evolution_improves_saves_and_resumes(self):
        settings = Settings()
        settings.training.checkpoint_every = 2
        env = new_env(settings)
        manager = EvolutionManager(settings, session="learn", population=30, seed=4)
        evaluate = self._evaluator(env, manager)
        results = [manager.run_generation(evaluate) for _ in range(8)]

        early = max(r.best_fitness for r in results[:2])
        late = max(r.best_fitness for r in results[-3:])
        self.assertGreater(late, early, "selection must improve on the first random generations")
        self.assertEqual(manager.generation, 8)

        # The saved model must reproduce a genuine score when raced again.
        saved = paths.BEST_GENOME_PATH
        self.assertTrue(saved.exists())
        genome = load_genome(saved)
        config = load_neat_config()
        env.reset(1, NetworkController([genome], config))
        env.run_to_end()
        self.assertGreater(float(env.progress.distance[0]), 0.0)
        self.assertGreater(len(layout_topology(genome, config).nodes), 0)

        # Checkpoints resume at the right generation, with history trimmed to match.
        checkpoint = paths.CHECKPOINT_DIR / "learn" / "neat-checkpoint-6"
        self.assertTrue(checkpoint.exists())
        resumed = EvolutionManager(settings, resume_from=checkpoint)
        self.assertEqual(resumed.generation, 6)
        self.assertEqual(len(resumed.history), 6)
        self.assertEqual(resumed.best_fitness_ever, manager.best_fitness_ever)
        result = resumed.run_generation(self._evaluator(env, resumed))
        self.assertEqual(result.generation, 6)
        self.assertEqual(resumed.generation, 7)

        # A forked run reads the original's history but must not modify it.
        original = (paths.RESULT_DIR / "learn" / "history.csv").read_text()
        forked = EvolutionManager(settings, session="fork", resume_from=checkpoint)
        forked.run_generation(self._evaluator(env, forked))
        self.assertEqual((paths.RESULT_DIR / "learn" / "history.csv").read_text(), original)
        self.assertEqual(len(forked.history), 7)

    def test_resume_prefers_training_sessions_over_filming_forks(self):
        import os as _os
        import time as _time
        from ai.evolution import latest_session
        root = Path(self.tmp.name) / "sessions"
        for index, name in enumerate(("20260101-000000", "20260102-000000", "record-20260103-000000")):
            (root / name).mkdir(parents=True)
            file = root / name / "neat-checkpoint-5"
            file.write_text("x")
            stamp = _time.time() + index
            _os.utime(file, (stamp, stamp))
        self.assertEqual(latest_session(root).name, "20260102-000000")
        for session in root.iterdir():
            if not session.name.startswith("record-"):
                (session / "neat-checkpoint-5").unlink()
        self.assertEqual(latest_session(root).name, "record-20260103-000000")

    def test_networks_really_control_the_cars(self):
        """Different genomes must drive differently; nothing is hardcoded."""
        import neat
        config = load_neat_config(population=10)
        population = neat.Population(config, seed=9)
        genomes = list(population.population.values())
        env = new_env()
        env.reset(len(genomes), NetworkController(genomes, config))
        for _ in range(90):
            env.tick()
        self.assertGreater(np.unique(np.round(env.car.x, 3)).size, 3)


if __name__ == "__main__":
    unittest.main()
