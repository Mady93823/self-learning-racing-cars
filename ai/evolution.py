"""Generation management around neat-python: checkpoints, best genome, training history.

Evolution itself (speciation, crossover, mutation, elitism) is entirely neat-python's.
This module drives it one generation at a time so that the same code can run behind
the interactive app, the recorder or the headless trainer.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import pickle
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import neat
import numpy as np

import settings as paths
from settings import Settings
from simulation.environment import CarStatus, RaceEnvironment

HISTORY_FIELDS = (
    "generation", "population", "species", "best_fitness", "average_fitness", "best_distance_m",
    "best_laps", "total_laps", "finishers", "crashed", "timed_out", "wrong_way",
    "best_lap_seconds", "best_nodes", "best_connections", "sim_seconds", "wall_seconds",
)


@dataclass
class GenerationResult:
    generation: int
    population: int
    species: int
    best_fitness: float
    average_fitness: float
    best_distance_m: float
    best_laps: int
    total_laps: int
    finishers: int
    crashed: int
    timed_out: int
    wrong_way: int
    best_lap_seconds: float
    best_nodes: int
    best_connections: int
    sim_seconds: float
    wall_seconds: float
    best_genome_id: int = -1
    leaderboard: tuple = ()


def load_neat_config(path: Path = paths.NEAT_CONFIG_PATH, population: int | None = None) -> neat.Config:
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, str(path))
    if population is not None:
        config.pop_size = int(population)
    return config


def latest_session(root: Path | None = None) -> Path | None:
    """The most recently active training session. Filming forks are only used as a last resort."""
    root = root or paths.CHECKPOINT_DIR
    if not root.exists():
        return None
    sessions = [p for p in root.iterdir() if p.is_dir() and any(p.glob("neat-checkpoint-*"))]
    if not sessions:
        return None
    training = [p for p in sessions if not p.name.startswith("record-")]

    def last_saved(session: Path) -> float:
        return max(f.stat().st_mtime for f in session.glob("neat-checkpoint-*"))

    return max(training or sessions, key=last_saved)


def latest_checkpoint(session_dir: Path) -> Path | None:
    files = list(session_dir.glob("neat-checkpoint-*"))
    return max(files, key=lambda p: int(p.name.rsplit("-", 1)[1])) if files else None


def save_genome(genome, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(genome, handle)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def load_genome(path: Path):
    # Models are Python pickles, so only load files you trust (the README says so too).
    with open(path, "rb") as handle:
        return pickle.load(handle)


def summarize_generation(env: RaceEnvironment, genomes: list, fitness: np.ndarray, generation: int,
                         species_count: int, species_of: dict, wall_seconds: float) -> GenerationResult:
    """Everything worth reporting about a finished generation, measured from the race."""
    progress = env.progress
    best = int(np.argmax(fitness))
    lap_ticks = progress.best_lap_ticks[progress.best_lap_ticks > 0]
    order = np.argsort(-fitness)[:5]
    board = tuple(
        {
            "place": place + 1,
            "car": int(car),
            "genome": int(genomes[car].key),
            "species": species_of.get(genomes[car].key, 0),
            "fitness": float(fitness[car]),
            "laps": int(progress.laps[car]),
            "distance_m": float(max(progress.distance[car], 0.0) * paths.UNITS_TO_METRES),
            "status": CarStatus.LABELS[int(env.status[car])],
        }
        for place, car in enumerate(order)
    )
    genome = genomes[best]
    return GenerationResult(
        generation=generation,
        population=len(genomes),
        species=species_count,
        best_fitness=float(fitness[best]),
        average_fitness=float(fitness.mean()),
        best_distance_m=float(max(progress.distance.max(), 0.0) * paths.UNITS_TO_METRES),
        best_laps=int(progress.laps.max()),
        total_laps=int(progress.laps.sum()),
        finishers=env.finished_count,
        crashed=int(np.count_nonzero(env.status == CarStatus.CRASHED)),
        timed_out=int(np.count_nonzero(env.status == CarStatus.TIMED_OUT)),
        wrong_way=int(np.count_nonzero(env.status == CarStatus.WRONG_WAY)),
        best_lap_seconds=float(lap_ticks.min() * env.dt) if lap_ticks.size else 0.0,
        best_nodes=len(genome.nodes),
        best_connections=sum(1 for c in genome.connections.values() if c.enabled),
        sim_seconds=env.time,
        wall_seconds=wall_seconds,
        best_genome_id=int(genome.key),
        leaderboard=board,
    )


class EvolutionManager:
    """One training run: a NEAT population plus everything saved along the way."""

    def __init__(self, settings: Settings, session: str | None = None, resume_from: Path | None = None,
                 population: int | None = None, seed: int | None = None, quiet_checkpoints: bool = True):
        self.settings = settings
        self.population_override = population
        self.seed = seed
        self.quiet = quiet_checkpoints
        self.history: list[GenerationResult] = []
        self.training_seconds = 0.0
        self.best_fitness_ever = float("-inf")
        self.best_genome = None
        self.on_generation: list[Callable[[GenerationResult], None]] = []
        self._start(session, resume_from)

    def _start(self, session: str | None, resume_from: Path | None) -> None:
        count = self.population_override or self.settings.training.population
        source_session = None
        if resume_from is not None:
            resume_from = Path(resume_from)
            source_session = resume_from.parent.name
            # Naming a different session forks the run and leaves the original untouched.
            self.session = session or source_session
            self.population = neat.Checkpointer.restore_checkpoint(str(resume_from))
            if self.population_override:
                self.population.config.pop_size = self.population_override
            self.config = self.population.config
        else:
            self.session = session or time.strftime("%Y%m%d-%H%M%S")
            self.config = load_neat_config(population=count)
            if self.seed is not None:
                random.seed(self.seed)
            self.population = neat.Population(self.config, seed=self.seed)

        self.checkpoint_dir = paths.CHECKPOINT_DIR / self.session
        self.result_dir = paths.RESULT_DIR / self.session
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv = self.result_dir / "history.csv"
        (self.result_dir / "settings.json").write_text(json.dumps(self.settings.to_dict(), indent=2))

        self.checkpointer = neat.Checkpointer(self.settings.training.checkpoint_every,
                                              filename_prefix=str(self.checkpoint_dir / "neat-checkpoint-"))
        # Count the interval from where this run begins, not from generation zero.
        self.checkpointer.last_generation_checkpoint = self.population.generation
        self.population.add_reporter(self.checkpointer)

        if source_session is not None:
            self._restore_history(paths.RESULT_DIR / source_session / "history.csv", self.population.generation)
            self._load_session_best(paths.CHECKPOINT_DIR / source_session)

    def _load_session_best(self, checkpoint_dir: Path) -> None:
        """A resumed run must not overwrite its best genome with a weaker one."""
        best_path = checkpoint_dir / "best_genome.pkl"
        meta_path = best_path.with_suffix(".json")
        if best_path.exists() and meta_path.exists():
            self.best_genome = load_genome(best_path)
            self.best_fitness_ever = json.loads(meta_path.read_text()).get("fitness", float("-inf"))

    def _restore_history(self, source: Path, resume_generation: int) -> None:
        if not source.exists():
            return
        with open(source, newline="") as handle:
            all_rows = list(csv.DictReader(handle))
        rows = [row for row in all_rows if int(row["generation"]) < resume_generation]
        if len(rows) < len(all_rows) and source == self.history_csv:
            # Those generations are about to be re-run. Keep the old timeline, don't lose it.
            shutil.copy(source, source.with_name(f"history.before-resume-{resume_generation}.csv"))
        for row in rows:
            values = {k: (int(row[k]) if k in _INT_FIELDS else float(row[k])) for k in HISTORY_FIELDS}
            self.history.append(GenerationResult(**values))
        self.training_seconds = sum(r.wall_seconds for r in self.history)
        with open(self.history_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, HISTORY_FIELDS)
            writer.writeheader()
            for result in self.history:
                writer.writerow({k: getattr(result, k) for k in HISTORY_FIELDS})

    @property
    def generation(self) -> int:
        return self.population.generation

    @property
    def previous(self) -> GenerationResult | None:
        return self.history[-1] if self.history else None

    def species_map(self) -> dict[int, int]:
        return dict(self.population.species.genome_to_species)

    def species_count(self) -> int:
        return len(self.population.species.species)

    def run_generation(self, evaluate: Callable[[list, neat.Config], GenerationResult]) -> GenerationResult:
        """Evaluate one generation, then let NEAT breed the next.

        ``evaluate`` races the genomes, sets each ``genome.fitness`` and returns a
        GenerationResult.
        """
        outcome: dict[str, GenerationResult] = {}

        def fitness_function(genomes, config):
            outcome["result"] = evaluate(genomes, config)
            self._record(outcome["result"], [g for _, g in genomes])

        sink = io.StringIO() if self.quiet else None
        with contextlib.redirect_stdout(sink) if sink else contextlib.nullcontext():
            self.population.run(fitness_function, 1)
        return outcome["result"]

    def _record(self, result: GenerationResult, genomes: list) -> None:
        self.history.append(result)
        self.training_seconds += result.wall_seconds
        new_file = not self.history_csv.exists()
        with open(self.history_csv, "a", newline="") as handle:
            writer = csv.DictWriter(handle, HISTORY_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow({k: getattr(result, k) for k in HISTORY_FIELDS})

        best = max(genomes, key=lambda g: g.fitness)
        if best.fitness > self.best_fitness_ever:
            self.best_fitness_ever = float(best.fitness)
            self.best_genome = best
            metadata = {
                "fitness": float(best.fitness),
                "generation": result.generation,
                "session": self.session,
                "laps": result.best_laps,
                "best_lap_seconds": result.best_lap_seconds,
                "nodes": result.best_nodes,
                "connections": result.best_connections,
            }
            save_genome(best, self.checkpoint_dir / "best_genome.pkl", metadata)
            self._maybe_publish_global_best(best, metadata)

        for callback in self.on_generation:
            callback(result)

    @staticmethod
    def _maybe_publish_global_best(genome, metadata: dict) -> None:
        """``models/best_genome.pkl`` only ever improves, whichever session produced it."""
        meta_path = paths.BEST_GENOME_PATH.with_suffix(".json")
        if meta_path.exists():
            with contextlib.suppress(ValueError, OSError):
                if json.loads(meta_path.read_text()).get("fitness", float("-inf")) >= metadata["fitness"]:
                    return
        save_genome(genome, paths.BEST_GENOME_PATH, metadata)

    def write_summary(self) -> None:
        payload = {
            "session": self.session,
            "generations": len(self.history),
            "training_seconds": self.training_seconds,
            "best_fitness": self.best_fitness_ever,
            "history": [{k: getattr(r, k) for k in HISTORY_FIELDS} for r in self.history],
        }
        (self.result_dir / "summary.json").write_text(json.dumps(payload, indent=2))


_INT_FIELDS = {"generation", "population", "species", "best_laps", "total_laps", "finishers", "crashed",
               "timed_out", "wrong_way", "best_nodes", "best_connections"}
