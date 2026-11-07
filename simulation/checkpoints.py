"""Lap and checkpoint bookkeeping.

Progress is the integrated change in arc length along the centre line, never a
position test. That closes the usual exploits:

* Driving back and forth over a gate earns nothing, because only a new high-water
  mark counts as progress and each checkpoint can only be earned once.
* Circling is pointless, since a full circle has zero net arc-length change.
* A lap cannot be shortcut, because barriers make the arc length continuous.
"""

from __future__ import annotations

import numpy as np

from simulation.track import Track


class ProgressTracker:
    def __init__(self, track: Track, start_s: np.ndarray):
        count = len(start_s)
        self.track = track
        # Cars on the grid start before the line, so their position relative to the
        # line is negative. ``relative`` is that signed position, continuous across laps.
        self.start_relative = np.where(start_s > track.length / 2, start_s - track.length, start_s)
        self.relative = self.start_relative.copy()
        self.high_water = self.relative.copy()
        self.last_s = start_s.copy()

        self.checkpoint_spacing = track.length / track.checkpoint_count
        self.checkpoints_earned = np.zeros(count, dtype=np.int64)
        self.visited_upto = np.floor(self.relative / self.checkpoint_spacing).astype(np.int64)
        self.position_cell = self.visited_upto.copy()
        self.revisits = np.zeros(count, dtype=np.int64)
        self.backward_distance = np.zeros(count)
        self.laps = np.zeros(count, dtype=np.int64)
        self.last_new_ground_tick = np.zeros(count, dtype=np.int64)
        self.lap_start_tick = np.zeros(count, dtype=np.int64)
        self.best_lap_ticks = np.full(count, -1, dtype=np.int64)
        self.last_lap_ticks = np.full(count, -1, dtype=np.int64)
        self.crossed_line = np.zeros(count, dtype=bool)

    @property
    def distance(self) -> np.ndarray:
        """Furthest ground covered from the car's own grid slot."""
        return self.high_water - self.start_relative

    def update(self, ids: np.ndarray, s: np.ndarray, tick: int):
        """Update after physics. ``s`` holds the current arc length of each car in ``ids``.

        Returns ``(checkpoint_events, lap_events)``, each a list of tuples:
        ``(car, checkpoint_index)`` and ``(car, lap_number, lap_ticks)``.
        """
        length = self.track.length
        delta = s - self.last_s[ids]
        delta = (delta + length / 2.0) % length - length / 2.0  # shortest way round
        self.last_s[ids] = s

        relative = self.relative[ids] + delta
        self.relative[ids] = relative
        self.backward_distance[ids] += np.maximum(-delta, 0.0)

        previous_high = self.high_water[ids]
        new_high = np.maximum(previous_high, relative)
        # Require a little real movement so jitter on the spot can't reset the timer.
        moved = new_high > previous_high + 1e-6
        self.high_water[ids] = new_high
        self.last_new_ground_tick[ids[moved]] = tick

        # Moving forward into a checkpoint cell that was already earned is a revisit.
        cell = np.floor(relative / self.checkpoint_spacing).astype(np.int64)
        previous_cell = self.position_cell[ids]
        self.position_cell[ids] = cell
        forward_cross = cell > previous_cell
        already_earned = cell <= self.visited_upto[ids]
        if np.any(forward_cross & already_earned):
            self.revisits[ids[forward_cross & already_earned]] += 1

        checkpoint_events: list[tuple[int, int]] = []
        lap_events: list[tuple[int, int, int]] = []

        for position in np.nonzero(forward_cross & ~already_earned)[0]:
            car = int(ids[position])
            reached = int(cell[position])
            self.visited_upto[car] = reached
            if reached <= 0:
                # Crossing cells on the way from the grid to the line earns nothing.
                if reached == 0 and not self.crossed_line[car]:
                    self.crossed_line[car] = True
                    self.lap_start_tick[car] = tick
                    checkpoint_events.append((car, 0))
                continue
            self.checkpoints_earned[car] += 1
            checkpoint_index = reached % self.track.checkpoint_count
            checkpoint_events.append((car, checkpoint_index))
            if checkpoint_index == 0:
                lap = reached // self.track.checkpoint_count
                self.laps[car] = lap
                ticks = tick - int(self.lap_start_tick[car])
                self.lap_start_tick[car] = tick
                self.last_lap_ticks[car] = ticks
                if self.best_lap_ticks[car] < 0 or ticks < self.best_lap_ticks[car]:
                    self.best_lap_ticks[car] = ticks
                lap_events.append((car, lap, ticks))
        return checkpoint_events, lap_events
