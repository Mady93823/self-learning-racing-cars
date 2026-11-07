"""Signed-distance field of the road, used for collisions and ray casting.

Rather than intersecting every car and ray with every boundary segment, the road
is baked once into a grid holding ``half_width - distance_to_centre_line``:

    sdf > 0  inside the road, the value being the distance to the nearest barrier
    sdf < 0  beyond the barrier

Collision checks then become grid lookups, and rays are sphere traced: from any
point inside the road you can safely advance by the SDF value without crossing a
barrier. Both operations are vectorised over all cars, so their cost does not
depend on how many segments the circuit has.
"""

from __future__ import annotations

import numpy as np

from simulation.track import Track


class TrackField:
    def __init__(self, track: Track, cell: float, margin: float):
        self.track = track
        self.cell = float(cell)
        self.margin = float(margin)

        x0, y0, x1, y1 = track.bounds(padding=margin)
        self.origin = np.array([x0, y0], dtype=float)
        self.cols = int(np.ceil((x1 - x0) / cell)) + 2
        self.rows = int(np.ceil((y1 - y0) / cell)) + 2

        distance = np.full((self.rows, self.cols), 1e4, dtype=np.float32)
        arc = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Fill each segment's neighbourhood instead of comparing the whole grid with
        # the whole centre line, which cuts the work by two orders of magnitude.
        points = track.points
        following = np.roll(points, -1, axis=0)
        reach = int(np.ceil(margin / cell))
        for i in range(track.count):
            a, b = points[i], following[i]
            ca = np.floor((np.minimum(a, b) - self.origin) / cell).astype(int) - reach
            cb = np.ceil((np.maximum(a, b) - self.origin) / cell).astype(int) + reach
            col0, row0 = np.maximum(ca, 0)
            col1, row1 = np.minimum(cb, (self.cols - 1, self.rows - 1)) + 1
            xs = self.origin[0] + np.arange(col0, col1) * cell
            ys = self.origin[1] + np.arange(row0, row1) * cell
            px, py = np.meshgrid(xs, ys)
            ab = b - a
            t = np.clip(((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / np.dot(ab, ab), 0.0, 1.0)
            d = np.hypot(px - (a[0] + t * ab[0]), py - (a[1] + t * ab[1]))
            view = distance[row0:row1, col0:col1]
            mask = d < view
            view[mask] = d[mask]
            arc[row0:row1, col0:col1][mask] = ((i + t[mask]) * track.spacing).astype(np.float32)

        self.distance = distance
        self.sdf = (track.half_width - distance).astype(np.float32)
        self.arc = arc
        # Arc length wraps at the start line, so anything that interpolates it (the
        # background shader) works with its angle on the unit circle instead.
        phase = arc / track.length * (2.0 * np.pi)
        self.phase_cos = np.cos(phase).astype(np.float32)
        self.phase_sin = np.sin(phase).astype(np.float32)

    def sample_sdf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear SDF lookup. Points off the grid read as far outside the road."""
        gx = (x - self.origin[0]) / self.cell
        gy = (y - self.origin[1]) / self.cell
        col = gx.astype(np.intp)
        row = gy.astype(np.intp)
        valid = (gx >= 0) & (gy >= 0) & (col < self.cols - 1) & (row < self.rows - 1)
        col = np.where(valid, col, 0)
        row = np.where(valid, row, 0)
        fx = gx - col
        fy = gy - row
        grid = self.sdf
        value = (grid[row, col] * (1 - fx) + grid[row, col + 1] * fx) * (1 - fy) \
            + (grid[row + 1, col] * (1 - fx) + grid[row + 1, col + 1] * fx) * fy
        return np.where(valid, value, -1e4)

    def project(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact arc length for points on the road: returns ``(s, sample_index)``."""
        col = np.clip(((x - self.origin[0]) / self.cell + 0.5).astype(np.intp), 0, self.cols - 1)
        row = np.clip(((y - self.origin[1]) / self.cell + 0.5).astype(np.intp), 0, self.rows - 1)
        track = self.track
        s = self.arc[row, col].astype(np.float64)
        index = (s / track.spacing).astype(np.intp) % track.count
        # The grid value is quantised to a cell; one projection step removes that error.
        dx = x - track.points[index, 0]
        dy = y - track.points[index, 1]
        s = (index * track.spacing + dx * track.tangents[index, 0] + dy * track.tangents[index, 1])
        s = np.mod(s, track.length)
        return s, (s / track.spacing).astype(np.intp) % track.count

    def car_hits_barrier(self, x, y, cos_h, sin_h, half_length, half_width) -> np.ndarray:
        """True for each car with any corner of its collision box outside the road."""
        lx, ly = half_length * cos_h, half_length * sin_h
        wx, wy = -half_width * sin_h, half_width * cos_h
        hit = self.sample_sdf(x + lx + wx, y + ly + wy) <= 0.0
        hit |= self.sample_sdf(x + lx - wx, y + ly - wy) <= 0.0
        hit |= self.sample_sdf(x - lx + wx, y - ly + wy) <= 0.0
        hit |= self.sample_sdf(x - lx - wx, y - ly - wy) <= 0.0
        return hit

    def raycast(self, x, y, dir_x, dir_y, max_range: float, steps: int, epsilon: float):
        """Sphere trace rays through the road. All arguments are flat arrays."""
        travelled = np.zeros_like(x, dtype=np.float64)
        active = np.arange(x.size)
        for _ in range(steps):
            px = x[active] + dir_x[active] * travelled[active]
            py = y[active] + dir_y[active] * travelled[active]
            clearance = self.sample_sdf(px, py)
            # A barrier met at a shallow angle would need many tiny steps; never
            # stepping below epsilon bounds the work with negligible error.
            travelled[active] += np.maximum(clearance, epsilon)
            active = active[(clearance > epsilon) & (travelled[active] < max_range)]
            if active.size == 0:
                break
        return np.minimum(travelled, max_range)
