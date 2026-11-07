"""The seven ray-cast distance sensors each car perceives the track with."""

from __future__ import annotations

import numpy as np

from settings import SensorSettings
from simulation.collision import TrackField


class SensorArray:
    def __init__(self, cfg: SensorSettings, field: TrackField):
        self.cfg = cfg
        self.field = field
        self.angles = np.radians(np.array(cfg.angles_deg, dtype=float))
        self.count = len(self.angles)

    def read(self, x: np.ndarray, y: np.ndarray, heading: np.ndarray) -> np.ndarray:
        """Normalised barrier distances, shape (cars, sensors): 1 = clear, 0 = touching."""
        angles = heading[:, None] + self.angles
        ray_x = np.repeat(x, self.count)
        ray_y = np.repeat(y, self.count)
        flat = angles.ravel()
        distance = self.field.raycast(ray_x, ray_y, np.cos(flat), np.sin(flat),
                                      self.cfg.max_range, self.cfg.march_steps, self.cfg.hit_epsilon)
        return (distance / self.cfg.max_range).reshape(-1, self.count)

    def endpoints(self, x: float, y: float, heading: float, readings: np.ndarray) -> np.ndarray:
        """World coordinates where each ray ended, for drawing. Shape (sensors, 2)."""
        angles = heading + self.angles
        distance = readings * self.cfg.max_range
        return np.column_stack([x + np.cos(angles) * distance, y + np.sin(angles) * distance])
