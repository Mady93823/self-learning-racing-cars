#!/usr/bin/env python3
"""AI Racing Lab: neural networks learn to race through NEAT.

    python main.py                      train and watch
    python main.py train --headless     train at full speed with no window
    python main.py evaluate             watch the best saved AI drive
    python main.py record               render a training video to recordings/
    python main.py reels                render a vertical Instagram Reel from scratch

Add --help after a command for all of its options.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from ai.evolution import EvolutionManager, latest_checkpoint, latest_session, load_genome, load_neat_config, summarize_generation
from ai.neural_network import NetworkController
from settings import BEST_GENOME_PATH, RECORDING_DIR, Settings
from simulation.environment import CarStatus, RaceEnvironment, build_track

CAMERAS = {"full": "FULL TRACK", "follow": "FOLLOW BEST", "cinematic": "CINEMATIC"}


def resolve_resume(value: str | None) -> Path | None:
    """``--resume`` takes a checkpoint path, or no value for the newest one available."""
    if value is None:
        return None
    if value != "latest":
        path = Path(value)
        if path.is_dir():
            found = latest_checkpoint(path)
            if found is None:
                sys.exit(f"No checkpoints in {path}")
            return found
        if not path.exists():
            sys.exit(f"Checkpoint not found: {path}")
        return path
    session = latest_session()
    found = latest_checkpoint(session) if session else None
    if found is None:
        sys.exit("No checkpoint to resume. Train for a few generations first.")
    return found


def make_manager(args, settings: Settings, resume: Path | None = None, session: str | None = None) -> EvolutionManager:
    return EvolutionManager(settings, session=session, resume_from=resume, population=args.population, seed=args.seed)


def load_settings(args) -> Settings:
    try:
        settings = Settings.load(args.settings)
    except (KeyError, ValueError, OSError) as error:
        sys.exit(f"Could not use settings file {args.settings}: {error}")
    if args.population:
        settings.training.population = args.population
    if getattr(args, "laps", None):
        settings.training.target_laps = args.laps
    if getattr(args, "fps", None):
        settings.display.target_fps = args.fps
    return settings


# ---------------------------------------------------------------------- headless


def train_headless(args, settings: Settings) -> int:
    """Identical training logic to the visual app, with no pygame involved."""
    track, field = build_track(settings)
    env = RaceEnvironment(settings, track, field)
    manager = make_manager(args, settings, resolve_resume(args.resume))
    print(f"Session {manager.session}: generation {manager.generation}, "
          f"{manager.config.pop_size} cars, lap length {track.length * 0.1:.0f} m")

    def evaluate(pairs, config):
        started = time.perf_counter()
        genomes = [genome for _, genome in pairs]
        env.reset(len(genomes), NetworkController(genomes, config))
        env.run_to_end()
        env.drain_events()
        fitness = env.current_fitness()
        for genome, value in zip(genomes, fitness.tolist()):
            genome.fitness = float(value)
        return summarize_generation(env, genomes, fitness, manager.generation, manager.species_count(),
                                    manager.species_map(), time.perf_counter() - started)

    from ui.statistics import save_training_plot

    every = settings.training.stats_plot_every
    target = manager.generation + args.generations
    try:
        while manager.generation < target:
            result = manager.run_generation(evaluate)
            lap = f"{result.best_lap_seconds:5.2f}s" if result.best_lap_seconds else "   -  "
            print(f"gen {result.generation:4d} | best {result.best_fitness:7.0f} | avg {result.average_fitness:6.0f} | "
                  f"dist {result.best_distance_m:6.0f} m | laps {result.best_laps} | finishers {result.finishers:3d} | "
                  f"crashed {result.crashed:3d} | lap {lap} | species {result.species:2d} | "
                  f"{result.best_nodes:2d}n {result.best_connections:2d}c | {result.wall_seconds:5.1f}s", flush=True)
            if every and len(manager.history) % every == 0:
                save_training_plot(manager.history, manager.result_dir / "training_stats.png")
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved in the latest checkpoint.")
    if manager.history:
        save_training_plot(manager.history, manager.result_dir / "training_stats.png")
        manager.write_summary()
    print(f"Best fitness: {manager.best_fitness_ever:,.0f}")
    print(f"Best model:   {BEST_GENOME_PATH}")
    print(f"Checkpoints:  {manager.checkpoint_dir}")
    print(f"Statistics:   {manager.result_dir}")
    return 0


def evaluate_headless(args, settings: Settings) -> int:
    """Measure a saved genome from every grid lane. Reports results, never estimates."""
    genome = load_genome(Path(args.model))
    config = load_neat_config()
    track, field = build_track(settings)
    settings.training.generation_seconds = max(settings.training.generation_seconds, args.laps * 80.0)
    env = RaceEnvironment(settings, track, field)
    print(f"Model: {args.model}  ({len(genome.nodes)} nodes, "
          f"{sum(1 for c in genome.connections.values() if c.enabled)} active links), "
          f"{args.laps} lap target from each lane")
    reports = []
    for lane in range(settings.training.grid_lanes):
        env.reset(1, NetworkController([genome], config), slots=[lane])
        env.run_to_end()
        status = CarStatus.LABELS[int(env.status[0])]
        laps = int(env.progress.laps[0])
        best = int(env.progress.best_lap_ticks[0])
        report = {
            "lane": lane,
            "outcome": status,
            "laps": laps,
            "distance_m": round(float(env.progress.distance[0]) * 0.1, 1),
            "best_lap_seconds": round(best * env.dt, 2) if best > 0 else None,
            "race_seconds": round(env.time, 2),
            "fitness": round(float(env.current_fitness()[0]), 1),
        }
        reports.append(report)
        lap_text = f"{report['best_lap_seconds']:.2f}s" if report["best_lap_seconds"] else "-"
        print(f"  lane {lane}: {status:9s} laps {laps}  best lap {lap_text:>7}  {report['distance_m']:7.0f} m  "
              f"fitness {report['fitness']:7.0f}")
    finishes = sum(r["outcome"] == "finished" for r in reports)
    print(f"Finished the distance from {finishes} of {len(reports)} lanes.")
    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2))
    return 0 if finishes == len(reports) else 1


# ---------------------------------------------------------------------- visual


def build_options(args, **extra):
    from ui.app import AppOptions
    window = None
    if getattr(args, "window", None):
        try:
            width, height = args.window.lower().split("x")
            window = (int(width), int(height))
        except ValueError:
            sys.exit(f"--window expects WIDTHxHEIGHT, got {args.window!r}")
    values = dict(
        show_title=not getattr(args, "no_title", False),
        speed=getattr(args, "speed", 1.0),
        camera=CAMERAS[getattr(args, "camera", "full")],
        fullscreen=getattr(args, "fullscreen", False),
        window=window,
        frame_limit=getattr(args, "frame_limit", None),
    )
    if getattr(args, "shots", None):
        values["screenshot_frames"] = tuple(int(n) for n in args.shots.split(","))
        values["screenshot_dir"] = Path(args.shot_dir)
    values.update(extra)
    return AppOptions(**values)


def train_visual(args, settings: Settings) -> int:
    from ui.app import SimulatorApp

    resume = resolve_resume(args.resume)
    manager = make_manager(args, settings, resume)
    app = SimulatorApp(settings, build_options(args))
    manager = app.run_training(manager, args.generations, new_manager=lambda: make_manager(args, settings))
    if manager.history:
        print(f"Best fitness: {manager.best_fitness_ever:,.0f}  |  Statistics: {manager.result_dir}")
    else:
        print("Stopped before any generation finished, so there is nothing new to report.")
    return 0


def evaluate_visual(args, settings: Settings) -> int:
    from ui.app import SimulatorApp

    genome = load_genome(Path(args.model))
    settings.training.generation_seconds = max(settings.training.generation_seconds, args.laps * 80.0)
    settings.display.show_sensors = True
    options = build_options(args, show_title=False)
    app = SimulatorApp(settings, options)
    app.run_evaluation(genome, load_neat_config(), runs=args.runs)
    return 0


def record(args, settings: Settings) -> int:
    from ui.app import SimulatorApp

    output = Path(args.out) if args.out else RECORDING_DIR / f"ai-racing-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    settings.recording.fps = args.fps or settings.recording.fps
    # Recordings train in their own session, so filming from an old checkpoint can never
    # disturb the history of the run that produced it.
    manager = make_manager(args, settings, resolve_resume(args.resume), session=f"record-{time.strftime('%Y%m%d-%H%M%S')}")
    options = build_options(
        args, presentation=True, record_path=output, record_seconds=args.seconds,
        show_every=args.show_every, speed=args.speed, camera=CAMERAS["full"],
        show_title=not args.no_title,
    )
    app = SimulatorApp(settings, options)
    app.run_training(manager, args.generations)
    return 0


def scout_seed(settings: Settings, population: int, low: int, high: int, attempts: int = 10) -> int:
    """Find the most watchable run: the first lap arrives in range, with records along the way.

    This doesn't alter training. Evolution is deterministic for a seed, so the video replays
    the same genuine run from random networks, and we simply choose which run to film.
    """
    import neat

    track, field = build_track(settings)
    env = RaceEnvironment(settings, track, field)
    start = int(time.time()) % 100000

    class FirstLap(Exception):
        pass

    best_seed, best_score = None, -1.0
    print(f"Comparing {attempts} runs for one whose first lap comes in generation {low}-{high}...")
    for seed in range(start, start + attempts):
        config = load_neat_config(population=population)
        random.seed(seed)
        evolution = neat.Population(config, seed=seed)
        first, record, records = None, 0.0, 0

        def evaluate(pairs, config):
            nonlocal record, records
            genomes = [genome for _, genome in pairs]
            env.reset(len(genomes), NetworkController(genomes, config))
            env.run_to_end()
            if env.progress.laps.max() >= 1:
                raise FirstLap()
            distance = float(env.progress.distance.max())
            if distance > record * 1.05:
                records += 1
            record = max(record, distance)
            for genome, value in zip(genomes, env.current_fitness().tolist()):
                genome.fitness = float(value)

        try:
            for generation in range(high + 1):
                first = generation
                evolution.run(evaluate, 1)
            first = None
        except FirstLap:
            pass
        if first is None or not low <= first <= high:
            outcome = "no lap in time" if first is None else f"first lap in generation {first}, out of range"
            print(f"  seed {seed}: {outcome}")
            continue
        # A run keeps people watching when it keeps making visible progress.
        score = records / (first + 1)
        print(f"  seed {seed}: first lap in generation {first}, {records} records")
        if score > best_score:
            best_seed, best_score = seed, score
    if best_seed is None:
        print("No run landed in that range; filming the next seed anyway.")
        return start + attempts
    return best_seed


def reels(args) -> int:
    from ui.app import SimulatorApp

    settings = Settings.load(args.settings)
    settings.apply_reels_profile()
    if args.fps:
        settings.recording.fps = args.fps
    population = args.population or settings.training.population
    settings.training.population = population
    try:
        low, high = (int(part) for part in args.story.split("-"))
    except ValueError:
        sys.exit(f"--story expects a range such as 8-16, got {args.story!r}")

    seed = args.seed if args.seed is not None else scout_seed(settings, population, low, high)
    print(f"Filming seed {seed}, starting from random networks in generation 0.")
    output = Path(args.out) if args.out else RECORDING_DIR / f"reel-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    args.seed = seed
    manager = make_manager(args, settings, session=f"record-reel-{time.strftime('%Y%m%d-%H%M%S')}")
    options = build_options(args, reels=True, presentation=True, record_path=output, record_seconds=args.max_seconds,
                            show_title=False)
    app = SimulatorApp(settings, options)
    app.run_training(manager, args.generations)
    return 0


# ---------------------------------------------------------------------- command line


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="AI Racing Lab: cars that teach themselves to race.")
    commands = root.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--settings", help="JSON file overriding values in settings.py")
    common.add_argument("--population", type=int, help="number of cars per generation (default 50)")
    common.add_argument("--seed", type=int, help="random seed for reproducible evolution")

    window = argparse.ArgumentParser(add_help=False)
    window.add_argument("--window", help="window size, e.g. 1280x720 (rendering stays 1920x1080)")
    window.add_argument("--fullscreen", action="store_true")
    window.add_argument("--no-title", action="store_true", help="skip the title screen")
    window.add_argument("--offscreen", action="store_true", help="no visible window")
    window.add_argument("--frame-limit", type=int, help=argparse.SUPPRESS)
    window.add_argument("--shots", help=argparse.SUPPRESS)
    window.add_argument("--shot-dir", default="results/shots", help=argparse.SUPPRESS)

    train = commands.add_parser("train", parents=[common, window], help="train the AI (default command)")
    train.add_argument("--headless", action="store_true", help="no graphics, maximum speed")
    train.add_argument("--generations", type=int, help="stop after this many (required with --headless)")
    train.add_argument("--resume", nargs="?", const="latest", help="continue from a checkpoint (default: newest)")
    train.add_argument("--speed", type=float, default=1.0, help="initial simulation speed")
    train.add_argument("--camera", choices=CAMERAS, default="full")
    train.add_argument("--fps", type=int, help="visual frame-rate cap (default 60)")

    evaluate = commands.add_parser("evaluate", parents=[common, window], help="test a saved model")
    evaluate.add_argument("--model", default=str(BEST_GENOME_PATH))
    evaluate.add_argument("--headless", action="store_true", help="print measured results from every lane")
    evaluate.add_argument("--laps", type=int, default=3)
    evaluate.add_argument("--runs", type=int, help="stop after this many runs (visual mode)")
    evaluate.add_argument("--speed", type=float, default=1.0)
    evaluate.add_argument("--camera", choices=CAMERAS, default="follow")
    evaluate.add_argument("--json", help="also write the headless results to this file")

    recording = commands.add_parser("record", parents=[common, window], help="render a training video")
    recording.add_argument("--out", help="output .mp4 (default: recordings/ai-racing-<time>.mp4)")
    recording.add_argument("--generations", type=int, default=12)
    recording.add_argument("--show-every", type=int, default=1,
                           help="show every Nth generation; the rest still train, off camera")
    recording.add_argument("--seconds", type=float, help="stop once the video reaches this length")
    recording.add_argument("--speed", type=float, default=1.0, help="race speed in the video")
    recording.add_argument("--fps", type=int, help="video frame rate (default 60)")
    recording.add_argument("--resume", nargs="?", const="latest", help="start from a checkpoint")
    reel = commands.add_parser("reels", parents=[common, window],
                               help="render a vertical Instagram Reel, training from scratch")
    reel.add_argument("--out", help="output .mp4 (default: recordings/reel-<time>.mp4)")
    reel.add_argument("--story", default="5-9", help="which generation the first lap should come in (default 5-9)")
    reel.add_argument("--max-seconds", type=float, default=170.0, help="end the video by this length (default 170)")
    reel.add_argument("--generations", type=int, help="stop after this many generations")
    reel.add_argument("--fps", type=int, help="video frame rate (default 30)")
    reel.add_argument("--preview", action="store_true", help="show a window while rendering")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    root = parser()
    known = {"train", "evaluate", "record", "reels", "-h", "--help"}
    if not arguments or arguments[0] not in known:
        arguments = ["train", *arguments]
    args = root.parse_args(arguments)

    # Rendering a Reel needs no window, unless you want to watch it being made.
    if getattr(args, "offscreen", False) or (args.command == "reels" and not args.preview):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    if args.command == "reels":
        return reels(args)

    settings = load_settings(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.command == "train":
        if args.headless:
            if not args.generations:
                root.error("train --headless needs --generations N")
            return train_headless(args, settings)
        return train_visual(args, settings)
    if args.command == "evaluate":
        if not Path(args.model).exists():
            sys.exit(f"Model not found: {args.model}\nTrain first: python main.py train --headless --generations 40")
        return evaluate_headless(args, settings) if args.headless else evaluate_visual(args, settings)
    return record(args, settings)


if __name__ == "__main__":
    sys.exit(main())
