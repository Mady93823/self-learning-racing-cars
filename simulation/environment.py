"""The race itself: advances all cars with a fixed timestep and reports what happened.

The environment knows nothing about pygame or wall-clock time, so it behaves the
same whether it is driven headless at full speed or in slow motion on screen. The
renderer only ever reads from it. ``tests/test_simulation.py`` checks that identical
inputs give identical results however the ticks are batched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ai.fitness import FitnessTracker
from ai.sensors import SensorArray
from settings import Settings
from simulation import physics
from simulation.checkpoints import ProgressTracker
from simulation.collision import TrackField
from simulation.physics import CarState
from simulation.track import Track


class CarStatus:
    RUNNING = 0
    CRASHED = 1
    TIMED_OUT = 2
    WRONG_WAY = 3
    FINISHED = 4      # reached race distance; score frozen, still circulating
    PARKED = 5        # finished, then stopped

    LABELS = {0: "running", 1: "crashed", 2: "timed out", 3: "wrong way",
              4: "finished", 5: "finished"}


@dataclass
class RaceEvent:
    """Something the presentation layer may want to react to. Never feeds back into the sim."""

    kind: str           # crash | timeout | wrong_way | checkpoint | lap | finish
    car: int
    x: float
    y: float
    value: int = 0      # checkpoint index or lap number
    ticks: int = 0      # lap duration where relevant
    speed: float = 0.0


# A controller maps (car ids, observations) to (steering, throttle), both in [-1, 1].
Controller = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def build_track(settings: Settings) -> tuple[Track, TrackField]:
    cfg = settings.track
    track = Track(cfg.half_width, cfg.sample_spacing, cfg.checkpoint_count)
    return track, TrackField(track, cfg.field_cell, cfg.field_margin)


class RaceEnvironment:
    OBSERVATION_SIZE = 9  # seven sensors, speed, heading relative to the track

    def __init__(self, settings: Settings, track: Track, field: TrackField):
        self.settings = settings
        self.track = track
        self.field = field
        self.dt = 1.0 / settings.physics.tick_rate
        self.sensors = SensorArray(settings.sensors, field)
        self.tick_count = 0
        self.count = 0

    @property
    def time(self) -> float:
        return self.tick_count * self.dt

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.status == CarStatus.RUNNING))

    @property
    def eliminated_count(self) -> int:
        return int(np.count_nonzero((self.status >= CarStatus.CRASHED)
                                    & (self.status <= CarStatus.WRONG_WAY)))

    @property
    def finished_count(self) -> int:
        return int(np.count_nonzero(self.status >= CarStatus.FINISHED))

    @property
    def is_over(self) -> bool:
        limit_reached = self.time >= self.settings.training.generation_seconds
        return limit_reached or not np.any(self.status == CarStatus.RUNNING)

    def grid_slot(self, index: int) -> tuple[float, float]:
        """Arc length and lateral offset of a grid slot."""
        cfg = self.settings.training
        row = index // cfg.grid_lanes
        lane = index % cfg.grid_lanes - (cfg.grid_lanes - 1) / 2.0
        return -30.0 - row * cfg.grid_row_spacing, lane * cfg.grid_lane_spacing

    def reset(self, count: int, controller: Controller, slots: list[int] | None = None) -> None:
        """Line up ``count`` cars on the grid. ``slots`` overrides which slot each car takes."""
        self.count = count
        self.controller = controller
        self.tick_count = 0
        self.car = CarState.zeros(count)
        self.status = np.zeros(count, dtype=np.int64)
        self.end_tick = np.full(count, -1, dtype=np.int64)
        self.steer_cmd = np.zeros(count)
        self.throttle_cmd = np.zeros(count)
        self.events: list[RaceEvent] = []

        s = np.zeros(count)
        lateral = np.zeros(count)
        for i in range(count):
            s[i], lateral[i] = self.grid_slot(slots[i] if slots is not None else i)
        s = np.mod(s, self.track.length)
        index = self.track.index_at(s)
        position = self.track.points[index] + self.track.normals[index] * lateral[:, None]
        self.car.x[:] = position[:, 0]
        self.car.y[:] = position[:, 1]
        self.car.heading[:] = self.track.headings[index]

        # Keep the previous pose so the renderer can interpolate between ticks.
        self.prev_x = self.car.x.copy()
        self.prev_y = self.car.y.copy()
        self.prev_heading = self.car.heading.copy()

        s, index = self.field.project(self.car.x, self.car.y)
        self.progress = ProgressTracker(self.track, s)
        self.fitness = FitnessTracker(self.settings, self.track.length, count)
        self.heading_relative = np.zeros(count)
        self.readings = np.ones((count, self.sensors.count))  # normalised distances
        self.observations = np.zeros((count, self.OBSERVATION_SIZE))
        self.track_index = index
        # Real readings from the grid, so sensors are correct before the first decision.
        # (No randomness is involved, so this cannot affect the race.)
        self.readings[:] = self.sensors.read(self.car.x, self.car.y, self.car.heading)

    def _sense_and_decide(self, ids: np.ndarray) -> None:
        cfg = self.settings
        readings = self.sensors.read(self.car.x[ids], self.car.y[ids], self.car.heading[ids])
        self.readings[ids] = readings
        observation = np.empty((ids.size, self.OBSERVATION_SIZE))
        observation[:, : self.sensors.count] = readings
        observation[:, self.sensors.count] = self.car.forward_speed[ids] / cfg.physics.max_speed
        observation[:, self.sensors.count + 1] = self.heading_relative[ids] / np.pi
        self.observations[ids] = observation
        steer, throttle = self.controller(ids, observation)
        self.fitness.record_steering(ids, steer - self.steer_cmd[ids])
        self.steer_cmd[ids] = steer
        self.throttle_cmd[ids] = throttle

    def tick(self) -> None:
        """Advance the race by exactly one fixed timestep."""
        cfg = self.settings
        phys = cfg.physics

        # Cars that finished keep lapping for the cameras, but no longer score.
        moving = np.nonzero((self.status == CarStatus.RUNNING)
                            | (self.status == CarStatus.FINISHED))[0]
        self.prev_x[:] = self.car.x
        self.prev_y[:] = self.car.y
        self.prev_heading[:] = self.car.heading

        if moving.size and self.tick_count % cfg.training.ai_interval_ticks == 0:
            self._sense_and_decide(moving)

        physics.step(self.car, moving, self.steer_cmd, self.throttle_cmd, self.dt, phys)
        self.tick_count += 1
        if moving.size == 0:
            return

        hit = self.field.car_hits_barrier(
            self.car.x[moving], self.car.y[moving],
            np.cos(self.car.heading[moving]), np.sin(self.car.heading[moving]),
            phys.collision_half_length, phys.collision_half_width)

        s, index = self.field.project(self.car.x[moving], self.car.y[moving])
        self.track_index[moving] = index
        offset = self.car.heading[moving] - self.track.headings[index]
        relative = (offset + np.pi) % (2.0 * np.pi) - np.pi
        previous_relative = self.heading_relative[moving]
        self.heading_relative[moving] = relative

        running = self.status[moving] == CarStatus.RUNNING
        ids = moving[running]
        if ids.size:
            self._update_running(ids, s[running], relative[running],
                                 previous_relative[running], hit[running])

        # A finished car that later touches a barrier just parks; its score stands.
        parked = moving[~running & hit]
        if parked.size:
            self.status[parked] = CarStatus.PARKED
            self.car.forward_speed[parked] = 0.0
            self.car.lateral_speed[parked] = 0.0

    def _update_running(self, ids, s, relative, previous_relative, hit) -> None:
        cfg = self.settings
        tick = self.tick_count
        checkpoints, laps = self.progress.update(ids, s, tick)
        for car, index in checkpoints:
            self.events.append(RaceEvent("checkpoint", car, self.car.x[car], self.car.y[car], index))
        for car, lap, ticks in laps:
            self.events.append(RaceEvent("lap", car, self.car.x[car], self.car.y[car], lap, ticks))

        turned = (relative - previous_relative + np.pi) % (2.0 * np.pi) - np.pi
        self.fitness.record_motion(ids, self.car.forward_speed[ids], np.abs(turned), self.dt)

        self._eliminate(ids[hit], CarStatus.CRASHED, "crash")

        survivors = ids[~hit]
        idle_ticks = tick - self.progress.last_new_ground_tick[survivors]
        timeout = cfg.fitness.inactivity_timeout * cfg.physics.tick_rate
        self._eliminate(survivors[idle_ticks > timeout], CarStatus.TIMED_OUT, "timeout")

        survivors = survivors[idle_ticks <= timeout]
        retreat = self.progress.high_water[survivors] - self.progress.relative[survivors]
        self._eliminate(survivors[retreat > cfg.fitness.wrong_way_limit],
                        CarStatus.WRONG_WAY, "wrong_way")

        survivors = survivors[retreat <= cfg.fitness.wrong_way_limit]
        done = survivors[self.progress.laps[survivors] >= cfg.training.target_laps]
        if done.size:
            self.status[done] = CarStatus.FINISHED
            self.end_tick[done] = tick
            for car in done:
                self.events.append(RaceEvent("finish", int(car), self.car.x[car], self.car.y[car],
                                             int(self.progress.laps[car])))

    def _eliminate(self, ids: np.ndarray, status: int, kind: str) -> None:
        if ids.size == 0:
            return
        for car in ids:
            self.events.append(RaceEvent(kind, int(car), self.car.x[car], self.car.y[car],
                                         speed=float(self.car.forward_speed[car])))
        self.status[ids] = status
        self.end_tick[ids] = self.tick_count
        self.car.forward_speed[ids] = 0.0
        self.car.lateral_speed[ids] = 0.0

    def drain_events(self) -> list[RaceEvent]:
        events, self.events = self.events, []
        return events

    def current_fitness(self) -> np.ndarray:
        """Live score of every car. Uses the same formula as the final score."""
        end = np.where(self.end_tick >= 0, self.end_tick, self.tick_count)
        return self.fitness.compute(self.progress, self.status, end * self.dt)

    def run_to_end(self) -> None:
        while not self.is_over:
            self.tick()
