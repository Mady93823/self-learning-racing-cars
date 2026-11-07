"""The fitness function: what "driving well" means to evolution.

    fitness = progress + new checkpoints + completed laps + finish
              + pace x progress + smoothness x progress
              - crash (a share of the above) - timeout - wrong way
              - backward distance - stationary time - gate revisits - excess spin

Why it can't easily be gamed:

* Progress is a high-water mark on arc length, so only new ground scores.
* Pace and smoothness are multiplied by laps of progress, so a car that blasts into
  the first wall flat out gets almost nothing from them.
* Pace is ``ground covered / time alive``. Crawling scores low and circling scores zero.
* Spin is measured relative to the track direction, so following a corner costs
  nothing while spinning on the spot is charged immediately.
"""

from __future__ import annotations

import numpy as np

from settings import Settings


class FitnessTracker:
    def __init__(self, settings: Settings, lap_length: float, count: int):
        self.cfg = settings.fitness
        self.lap_length = lap_length
        self.max_speed = settings.physics.max_speed
        self.finish_status = 4  # CarStatus.FINISHED; avoids a circular import
        self.stationary_time = np.zeros(count)
        self.spin_total = np.zeros(count)
        self.steering_change = np.zeros(count)
        self.decisions = np.zeros(count, dtype=np.int64)

    def record_steering(self, ids: np.ndarray, change: np.ndarray) -> None:
        self.steering_change[ids] += np.abs(change)
        self.decisions[ids] += 1

    def record_motion(self, ids: np.ndarray, speed: np.ndarray, turned: np.ndarray, dt: float) -> None:
        self.stationary_time[ids] += (speed < self.cfg.stationary_speed) * dt
        self.spin_total[ids] += turned

    def breakdown(self, progress, status: np.ndarray, elapsed: np.ndarray) -> dict[str, np.ndarray]:
        """Every term of the score, so the training log can explain a result."""
        cfg = self.cfg
        laps_done = np.maximum(progress.distance, 0.0) / self.lap_length
        pace = np.clip(progress.distance / np.maximum(elapsed, 1e-3) / self.max_speed, 0.0, 1.0)
        chatter = self.steering_change / np.maximum(self.decisions, 1)
        smoothness = 1.0 / (1.0 + cfg.jerk_scale * chatter)
        excess_spin = np.maximum(self.spin_total - cfg.spin_allowance_rad * (1.0 + laps_done), 0.0)

        rewards = {
            "progress": cfg.progress_per_lap * laps_done,
            "checkpoints": cfg.checkpoint_bonus * progress.checkpoints_earned,
            "laps": cfg.lap_bonus * progress.laps,
            "finish": cfg.finish_bonus * (status >= self.finish_status),
            "speed": cfg.speed_bonus * laps_done * pace,
            "consistency": cfg.consistency_bonus * laps_done * smoothness,
        }
        # Proportional, so a crash never erases the difference between going far and
        # going nowhere. Early generations depend on that signal: a flat penalty would
        # score every short run as zero and leave evolution nothing to select on.
        earned = sum(rewards.values())
        return {
            **rewards,
            "crash": -(cfg.crash_penalty + cfg.crash_fraction * earned) * (status == 1),
            "timeout": -cfg.timeout_penalty * (status == 2),
            "wrong_way": -cfg.wrong_way_penalty * (status == 3),
            "backward": -cfg.backward_penalty_per_lap * progress.backward_distance / self.lap_length,
            "stationary": -cfg.stationary_penalty_per_s * self.stationary_time,
            "revisits": -cfg.revisit_penalty * progress.revisits,
            "spin": -cfg.spin_penalty_per_rad * excess_spin,
        }

    def compute(self, progress, status: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
        total = sum(self.breakdown(progress, status, elapsed).values())
        return np.maximum(total, 0.0)  # keeps NEAT's fitness sharing well behaved
