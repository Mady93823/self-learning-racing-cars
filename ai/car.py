"""Read-only per-car view used by the UI, over the simulation's batched arrays.

Physics runs on numpy arrays for speed, which makes "the car" awkward to talk about.
``CarView`` gives the presentation layer one tidy object without any per-object
state that could drift from the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

from settings import MS_TO_KMH, UNITS_TO_METRES
from simulation.environment import CarStatus, RaceEnvironment


@dataclass(frozen=True)
class CarView:
    env: RaceEnvironment
    index: int
    genome: object = None
    species: int | None = None

    @property
    def genome_id(self) -> int | None:
        return getattr(self.genome, "key", None)

    @property
    def is_running(self) -> bool:
        return int(self.env.status[self.index]) == CarStatus.RUNNING

    @property
    def status_label(self) -> str:
        return CarStatus.LABELS[int(self.env.status[self.index])]

    @property
    def speed_kmh(self) -> float:
        return float(self.env.car.forward_speed[self.index]) * UNITS_TO_METRES * MS_TO_KMH

    @property
    def distance_m(self) -> float:
        return max(float(self.env.progress.distance[self.index]), 0.0) * UNITS_TO_METRES

    @property
    def laps(self) -> int:
        return int(self.env.progress.laps[self.index])

    @property
    def lap_fraction(self) -> float:
        relative = float(self.env.progress.high_water[self.index])
        return max(relative, 0.0) / self.env.track.length % 1.0

    @property
    def best_lap_seconds(self) -> float | None:
        ticks = int(self.env.progress.best_lap_ticks[self.index])
        return ticks * self.env.dt if ticks > 0 else None

    @property
    def sensor_readings(self):
        return self.env.readings[self.index]

    @property
    def steer(self) -> float:
        return float(self.env.car.steer[self.index])

    @property
    def throttle(self) -> float:
        return float(self.env.car.throttle[self.index])
