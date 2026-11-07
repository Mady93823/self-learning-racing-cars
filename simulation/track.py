"""Circuit geometry: a closed spline resampled at uniform arc length.

Arc length ``s`` runs along the direction of travel with ``s = 0`` at the
start/finish line.
"""

from __future__ import annotations

import numpy as np

# Control points in travel order. The first long run is the main straight, and
# the circuit then works through a sweeper, a hairpin, esses and a back straight.
CONTROL_POINTS = np.array([
    (1050, 1450), (1650, 1450), (2200, 1440), (2600, 1370), (2820, 1160),
    (2800, 900), (2570, 760), (2290, 830), (2090, 1010), (1820, 1090),
    (1560, 960), (1570, 690), (1850, 540), (2300, 500), (2640, 430),
    (2810, 280), (2680, 110), (2380, 80), (1900, 100), (1400, 120),
    (1090, 220), (1010, 420), (1190, 640), (1210, 900), (1010, 1100),
    (740, 1020), (660, 760), (720, 480), (560, 230), (280, 240),
    (130, 520), (140, 900), (170, 1230), (420, 1430),
], dtype=float)

# Arc length of the start line, as a fraction of the way between control points
# 1 and 2 so the grid sits on the main straight.
_START_POINT = np.array((1650.0, 1450.0))


def _catmull_rom_closed(points: np.ndarray, per_segment: int = 48) -> np.ndarray:
    """Centripetal closed Catmull-Rom spline, which avoids cusps and overshoot."""
    count = len(points)
    output = []
    for i in range(count):
        p0, p1, p2, p3 = (points[(i + k) % count] for k in (-1, 0, 1, 2))
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5
        t = np.linspace(t1, t2, per_segment, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        output.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(output)


def _smooth_closed(points: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Gaussian filter around a closed polyline, computed in the frequency domain."""
    spectrum = np.fft.rfft(points, axis=0)
    frequency = np.fft.rfftfreq(len(points))
    gain = np.exp(-2.0 * (np.pi * frequency * sigma_samples) ** 2)
    return np.fft.irfft(spectrum * gain[:, None], n=len(points), axis=0)


class Track:
    """Uniformly sampled closed centre line with tangents, normals and curvature."""

    def __init__(self, half_width: float, spacing: float, checkpoint_count: int,
                 smoothing: float = 45.0):
        self.half_width = float(half_width)
        self.checkpoint_count = int(checkpoint_count)

        dense = _catmull_rom_closed(CONTROL_POINTS)
        closed = np.vstack([dense, dense[:1]])
        lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total = cumulative[-1]

        count = int(round(total / spacing))
        self.spacing = total / count
        self.length = float(total)
        s = np.arange(count) * self.spacing
        points = np.column_stack([
            np.interp(s, cumulative, closed[:, 0]),
            np.interp(s, cumulative, closed[:, 1]),
        ])

        # Catmull-Rom splines have curvature spikes at their control points. A circular
        # low-pass filter irons those out, giving corners a natural entry and exit.
        points = _smooth_closed(points, sigma_samples=smoothing / self.spacing)
        step_lengths = np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)
        smooth_cumulative = np.concatenate([[0.0], np.cumsum(step_lengths)])
        total = smooth_cumulative[-1]
        count = int(round(total / spacing))
        self.spacing = total / count
        self.length = float(total)
        s = np.arange(count) * self.spacing
        wrapped = np.vstack([points, points[:1]])
        points = np.column_stack([
            np.interp(s, smooth_cumulative, wrapped[:, 0]),
            np.interp(s, smooth_cumulative, wrapped[:, 1]),
        ])

        # Rotate the arrays so index 0 is the start/finish line.
        start_index = int(np.argmin(np.linalg.norm(points - _START_POINT, axis=1)))
        self.points = np.roll(points, -start_index, axis=0)
        self.count = count

        forward = np.roll(self.points, -1, axis=0) - np.roll(self.points, 1, axis=0)
        self.tangents = forward / np.linalg.norm(forward, axis=1, keepdims=True)
        # Screen coordinates have y pointing down, so this normal is the car's right.
        self.normals = np.column_stack([-self.tangents[:, 1], self.tangents[:, 0]])
        self.headings = np.arctan2(self.tangents[:, 1], self.tangents[:, 0])

        turn = np.angle(np.exp(1j * (np.roll(self.headings, -1) - self.headings)))
        raw_curvature = turn / self.spacing
        kernel = np.ones(9) / 9.0
        self.curvature = np.convolve(
            np.concatenate([raw_curvature[-8:], raw_curvature, raw_curvature[:8]]),
            kernel, mode="same")[8:-8]

        self.checkpoint_s = np.arange(self.checkpoint_count) * (self.length / self.checkpoint_count)

    def index_at(self, s) -> np.ndarray:
        return (np.asarray(s) / self.spacing).astype(int) % self.count

    def point_at(self, s, lateral=0.0) -> np.ndarray:
        """World position at arc length ``s``, offset ``lateral`` towards the car's right."""
        i = self.index_at(s)
        return self.points[i] + self.normals[i] * np.asarray(lateral)[..., None] \
            if np.ndim(lateral) else self.points[i] + self.normals[i] * lateral

    def bounds(self, padding: float = 0.0) -> tuple[float, float, float, float]:
        """Axis-aligned (x0, y0, x1, y1) box around the road surface plus ``padding``."""
        reach = self.half_width + padding
        low = self.points.min(axis=0) - reach
        high = self.points.max(axis=0) + reach
        return float(low[0]), float(low[1]), float(high[0]), float(high[1])

    def min_clearance(self, skip: float | None = None) -> float:
        """Smallest gap between centre-line sections that are not neighbours along the lap.

        Must exceed the road width plus barriers, otherwise two parts of the circuit
        would touch and a car could cross between them.
        """
        skip = 3.0 * self.half_width if skip is None else skip
        skip_samples = int(skip / self.spacing)
        best = float("inf")
        for i in range(0, self.count, 2):
            distance = np.linalg.norm(self.points - self.points[i], axis=1)
            offset = np.abs(np.arange(self.count) - i)
            separated = np.minimum(offset, self.count - offset) > skip_samples
            # Only count samples that are genuinely on another section of track.
            if separated.any():
                best = min(best, float(distance[separated].min()))
        return best

    def min_radius(self) -> float:
        peak = float(np.abs(self.curvature).max())
        return 1.0 / peak if peak > 0 else float("inf")
