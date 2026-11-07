"""Paints the circuit once into a large static image, then only blits it.

The road surface is shaded per pixel from the same distance field the physics uses,
like a fragment shader run on the CPU. Every edge is therefore antialiased, and the
paint lines up exactly with the collision boundary. The finished image is cached on
disk because it takes several seconds to produce and never changes.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable

import numpy as np
import pygame

from settings import Settings, TEXTURE_DIR
from simulation.collision import TrackField
from simulation.track import CONTROL_POINTS, Track
from ui.theme import SIGNAL, Theme

ART_VERSION = 5  # bump to invalidate the cached background after changing the artwork

# Areas occupied by buildings: (x0, y0, x1, y1) in world units.
KEEP_OUT = (
    (1230, 1190, 2280, 1370),     # pit complex
    (1090, 1500, 2290, 1740),     # grandstand
    (250, 710, 550, 1010),        # plaza
)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _band(x: np.ndarray, low: float, high: float, aa: float) -> np.ndarray:
    """1 inside [low, high], with an antialiased edge roughly one pixel wide."""
    return np.clip((x - low) / aa + 0.5, 0.0, 1.0) * np.clip((high - x) / aa + 0.5, 0.0, 1.0)


def _mix(base: np.ndarray, colour, amount: np.ndarray) -> np.ndarray:
    return base + (np.asarray(colour, dtype=np.float32) - base) * amount[..., None]


def _value_noise(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth, tileable noise in [-1, 1]."""
    grid = rng.uniform(-1.0, 1.0, (cells, cells)).astype(np.float32)
    position = np.arange(size) * cells / size
    i0 = position.astype(int) % cells
    i1 = (i0 + 1) % cells
    f = position - np.floor(position)
    f = (f * f * (3.0 - 2.0 * f)).astype(np.float32)
    # Bilinear blend over both axes, using smoothed weights.
    top = grid[i0][:, i0] * (1 - f)[None, :] + grid[i0][:, i1] * f[None, :]
    bottom = grid[i1][:, i0] * (1 - f)[None, :] + grid[i1][:, i1] * f[None, :]
    return top * (1 - f)[:, None] + bottom * f[:, None]


class TrackRenderer:
    def __init__(self, track: Track, field: TrackField, settings: Settings, theme: Theme,
                 progress: Callable[[float, str], None] | None = None):
        self.track = track
        self.field = field
        self.settings = settings
        self.theme = theme
        self.progress = progress or (lambda fraction, label: None)
        self.scale = settings.track.render_scale

        display = settings.display
        self.screen_size = (display.width, display.height)
        self.portrait = display.portrait
        # Portrait views turn the world 90 degrees clockwise, which lets a wide circuit fill
        # a tall screen. Only the picture rotates: the simulation is identical either way.
        # "View space" below means world coordinates after that turn: (u, v) = (-y, x).
        self.heading_offset = math.pi / 2.0 if self.portrait else 0.0
        x0, y0, x1, y1 = track.bounds(padding=30.0)
        if self.portrait:
            x0, y0, x1, y1 = -y1, x0, -y0, x1
            # Mostly inside Instagram's safe area. The tips of the circuit may pass under
            # its interface, but the HUD itself stays clear.
            self.free_rect = pygame.Rect(30, 250, display.width - 60, 1400)
            # The clear space below the cards and above the caption, where the action belongs.
            self.action_rect = pygame.Rect(65, 470, display.width - 130, 760)
        else:
            # The full-track shot must clear the HUD, so fit into the area it leaves free.
            self.free_rect = pygame.Rect(14, display.hud_top + 18,
                                         display.width - display.panel_width - 44,
                                         display.height - display.hud_top - display.hud_bottom - 38)
            self.action_rect = self.free_rect
        self.fit_zoom = min(self.free_rect.width / (x1 - x0), self.free_rect.height / (y1 - y0))
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # View-space point shown at the centre of the screen in the full-track view.
        self.fit_center = (cx + (display.width / 2 - self.free_rect.centerx) / self.fit_zoom,
                           cy + (display.height / 2 - self.free_rect.centery) / self.fit_zoom)

        half_w = display.width / 2 / self.fit_zoom + 8
        half_h = display.height / 2 / self.fit_zoom + 8
        if self.portrait:
            # A chase camera looks well past the circuit's edges, so the ground must extend
            # further. Otherwise the view hits the edge and pushes the car out of position.
            half_w += 550.0
            half_h += 950.0
        self.view_x = self.fit_center[0] - half_w
        self.view_y = self.fit_center[1] - half_h
        self.view_w = half_w * 2
        self.view_h = half_h * 2

        # The artwork is always painted in world orientation, then rotated as a whole.
        if self.portrait:
            self.world_x, self.world_w = self.view_y, self.view_h
            self.world_y, self.world_h = -(self.view_x + self.view_w), self.view_w
        else:
            self.world_x, self.world_y, self.world_w, self.world_h = (self.view_x, self.view_y,
                                                                       self.view_w, self.view_h)
        self.pixel_w = int(self.world_w * self.scale)
        self.pixel_h = int(self.world_h * self.scale)

        self.base = self._load_or_paint()
        if self.portrait:
            self.base = pygame.transform.rotate(self.base, -90).convert()
        self.mips = self._build_mips()

    def to_view(self, x: float, y: float) -> tuple[float, float]:
        return (-y, x) if self.portrait else (x, y)

    # ------------------------------------------------------------------ caching

    def _cache_path(self):
        fingerprint = hashlib.sha1()
        fingerprint.update(CONTROL_POINTS.tobytes())
        fingerprint.update(repr((ART_VERSION, self.scale, self.track.half_width, self.pixel_w,
                                 self.pixel_h, self.track.length, self.screen_size)).encode())
        orientation = "portrait" if self.portrait else "landscape"
        return TEXTURE_DIR / f"world_{orientation}_{fingerprint.hexdigest()[:12]}.rgb"

    def _load_or_paint(self) -> pygame.Surface:
        path = self._cache_path()
        if path.exists() and path.stat().st_size == self.pixel_w * self.pixel_h * 3:
            self.progress(0.5, "Loading circuit")
            return pygame.image.frombytes(path.read_bytes(), (self.pixel_w, self.pixel_h), "RGB").convert()
        surface = pygame.Surface((self.pixel_w, self.pixel_h)).convert()
        self._shade(surface)
        self._decorate(surface)
        path.parent.mkdir(parents=True, exist_ok=True)
        for old in TEXTURE_DIR.glob(f"{path.name.rsplit('_', 1)[0]}_*.rgb"):
            old.unlink()  # outdated artwork of this orientation; this app regenerates it
        path.write_bytes(pygame.image.tobytes(surface, "RGB"))
        return surface

    def _build_mips(self) -> list[tuple[float, pygame.Surface]]:
        """Pre-scaled copies, sorted small to large, so drawing never downscales far."""
        self.progress(0.95, "Preparing cameras")
        levels: list[tuple[float, pygame.Surface]] = []
        for zoom in sorted({self.fit_zoom, (self.fit_zoom + self.scale) / 2.0}):
            size = (int(self.view_w * zoom), int(self.view_h * zoom))
            levels.append((zoom, pygame.transform.smoothscale(self.base, size).convert()))
        levels.append((self.scale, self.base))
        return levels

    # ------------------------------------------------------------------ shading

    def _kerb_profile(self) -> np.ndarray:
        """How much kerb each arc-length sample gets: only corners get kerbs."""
        track = self.track
        strength = (np.abs(track.curvature) > 1.0 / 420.0).astype(float)
        # Extend kerbs past the apex and soften their ends.
        reach = int(70.0 / track.spacing)
        kernel = np.ones(2 * reach + 1) / (2 * reach + 1)
        wrapped = np.concatenate([strength[-reach:], strength, strength[:reach]])
        spread = np.convolve(wrapped, kernel, mode="same")[reach:-reach]
        return np.clip((spread - 0.3) * 14.0, 0.0, 1.0)

    def _light_positions(self) -> np.ndarray:
        track = self.track
        spots = []
        for fraction in np.linspace(0.04, 0.96, 9):
            index = int(fraction * track.count)
            side = -np.sign(track.curvature[index]) or 1.0
            spot = track.points[index] + track.normals[index] * side * (track.half_width + 62.0)
            if not any(a <= spot[0] <= c and b <= spot[1] <= d for a, b, c, d in KEEP_OUT):
                spots.append(spot)
        return np.array(spots)

    def _shade(self, surface: pygame.Surface) -> None:
        field = self.field
        track = self.track
        hw = track.half_width
        scale = self.scale
        aa = 1.1 / scale
        rng = np.random.default_rng(11)

        grain = rng.normal(0.0, 1.0, (1024, 1024)).astype(np.float32)
        patches = (_value_noise(1024, 16, rng) + 0.6 * _value_noise(1024, 40, rng)
                   + 0.35 * _value_noise(1024, 96, rng))
        fine = _value_noise(1024, 200, rng)

        kerb_profile = self._kerb_profile().astype(np.float32)
        self.lights = self._light_positions()

        xs = self.world_x + (np.arange(self.pixel_w) + 0.5) / scale
        gx = (xs - field.origin[0]) / field.cell
        col0 = np.clip(np.floor(gx).astype(int), 0, field.cols - 2)
        fx = np.clip(gx - col0, 0.0, 1.0).astype(np.float32)[None, :]
        tile_x = np.arange(self.pixel_w) % 1024

        lights_x = [np.exp(-((xs - lx) ** 2) / (2.0 * 400.0 ** 2)).astype(np.float32) for lx in self.lights[:, 0]]

        pixels = pygame.surfarray.pixels3d(surface)
        band_rows = 200
        for start in range(0, self.pixel_h, band_rows):
            self.progress(0.05 + 0.75 * start / self.pixel_h, "Painting circuit")
            stop = min(start + band_rows, self.pixel_h)
            ys = self.world_y + (np.arange(start, stop) + 0.5) / scale
            gy = (ys - field.origin[1]) / field.cell
            row0 = np.clip(np.floor(gy).astype(int), 0, field.rows - 2)
            fy = np.clip(gy - row0, 0.0, 1.0).astype(np.float32)[:, None]
            tile_y = np.arange(start, stop) % 1024
            ix = np.ix_(tile_y, tile_x)

            def lookup(grid: np.ndarray) -> np.ndarray:
                top = grid[row0][:, col0] * (1 - fx) + grid[row0][:, col0 + 1] * fx
                bottom = grid[row0 + 1][:, col0] * (1 - fx) + grid[row0 + 1][:, col0 + 1] * fx
                return top * (1 - fy) + bottom * fy

            d = lookup(field.distance)
            arc = (np.arctan2(lookup(field.phase_sin), lookup(field.phase_cos)) / (2 * np.pi)
                   % 1.0) * track.length
            kerb = kerb_profile[((arc / track.spacing).astype(int)) % track.count]

            noise_grain = grain[ix]
            noise_patch = patches[ix]
            noise_fine = fine[ix]

            # Ground: dark, desaturated turf with mown bands that follow the circuit.
            ground = np.zeros((stop - start, self.pixel_w, 3), dtype=np.float32)
            ground[..., 0] = 29.0
            ground[..., 1] = 35.0
            ground[..., 2] = 32.0
            mow = np.sin(np.minimum(d, 900.0) * 0.052) * _smoothstep(hw + 30.0, hw + 110.0, d)
            shade = 1.0 + 0.13 * noise_patch + 0.10 * noise_fine + 0.05 * mow + 0.018 * noise_grain
            image = ground * shade[..., None]

            # Paved run-off strip just beyond the barriers.
            runoff = _band(d, hw + 8.0, hw + 38.0 + 2.5 * noise_patch, aa)
            runoff_tone = 44.0 + 4.0 * noise_patch + 2.2 * noise_grain
            image = image + (np.stack([runoff_tone, runoff_tone + 1.0, runoff_tone + 3.0], axis=-1) - image) * runoff[..., None]

            # Barrier shadow falling outward onto whatever lies beyond it.
            shadow = 1.0 - 0.62 * _band(d, hw + 8.0, hw + 34.0, 18.0) * (1.0 - _smoothstep(hw + 10.0, hw + 34.0, d))
            image = image * shadow[..., None]

            # Asphalt with fine grain, large patches and a darker rubbered line.
            asphalt_value = 47.0 + 3.2 * noise_patch + 3.6 * noise_grain + 2.0 * noise_fine
            groove = np.exp(-((d / 30.0) ** 2))
            asphalt_value = asphalt_value * (1.0 - 0.17 * groove)
            asphalt = np.stack([asphalt_value, asphalt_value * 1.03, asphalt_value * 1.09], axis=-1)
            on_road = np.clip((hw + 0.5 - d) / aa + 0.5, 0.0, 1.0)
            image = image + (asphalt - image) * on_road[..., None]

            # Kerbs: alternating red and white blocks, only through corners.
            stripe_wave = np.sin(arc * (2.0 * np.pi / 42.0))
            stripe = np.clip(stripe_wave * 6.0 + 0.5, 0.0, 1.0)
            kerb_body = _band(d, hw - 15.0, hw + 0.5, aa) * kerb
            kerb_red = np.stack([214.0 + 0 * d, 46.0 + 0 * d, 50.0 + 0 * d], axis=-1)
            kerb_white = np.stack([236.0 + 0 * d, 236.0 + 0 * d, 234.0 + 0 * d], axis=-1)
            kerb_colour = kerb_red + (kerb_white - kerb_red) * stripe[..., None]
            bevel = 1.0 - 0.28 * _smoothstep(hw - 6.0, hw, d)  # slopes down towards the barrier
            image = image + (kerb_colour * bevel[..., None] - image) * kerb_body[..., None]

            # Track limit: a thin white line inside the kerbs along the whole lap.
            line = _band(d, hw - 18.0, hw - 15.0, aa) * 0.92
            image = _mix(image, (222, 224, 226), line)

            # Barrier: white and graphite safety blocks with a lit top edge.
            block = np.clip(np.sin(arc * (2.0 * np.pi / 150.0)) * 6.0 + 0.5, 0.0, 1.0)
            barrier_a = np.stack([158.0 + 0 * d, 163.0 + 0 * d, 172.0 + 0 * d], axis=-1)
            barrier_b = np.stack([54.0 + 0 * d, 60.0 + 0 * d, 72.0 + 0 * d], axis=-1)
            barrier_colour = barrier_a + (barrier_b - barrier_a) * block[..., None]
            barrier = _band(d, hw, hw + 10.0, aa)
            image = image + (barrier_colour - image) * barrier[..., None]
            image = _mix(image, (240, 242, 246), _band(d, hw + 3.0, hw + 4.4, aa) * 0.34)
            image = _mix(image, (10, 11, 13), _band(d, hw - 1.0, hw + 0.6, aa) * 0.75)

            # Contact shade along the foot of the barrier, on the road side.
            contact = 1.0 - 0.3 * np.clip((d - (hw - 6.0)) / 6.0, 0.0, 1.0) * on_road * (1.0 - kerb)
            image = image * contact[..., None]

            # Floodlight pools over a dim ambient level: the night-race look.
            light = np.full_like(d, 0.70)
            for (lx, ly), line_x in zip(self.lights, lights_x):
                falloff_y = np.exp(-((ys - ly) ** 2) / (2.0 * 400.0 ** 2)).astype(np.float32)[:, None]
                light += 0.48 * falloff_y * line_x[None, :]
            image = image * np.minimum(light, 1.25)[..., None]

            pixels[:, start:stop, :] = np.clip(image, 0.0, 255.0).astype(np.uint8).swapaxes(0, 1)
        del pixels

    # ------------------------------------------------------------------ decoration

    def _to_pixels(self, points) -> np.ndarray:
        return (np.asarray(points, dtype=float) - (self.world_x, self.world_y)) * self.scale

    def _polygon(self, surface: pygame.Surface, points, colour, alpha: int = 255) -> None:
        """Antialiased polygon, drawn large and scaled down."""
        px = self._to_pixels(points)
        low = np.floor(px.min(axis=0)).astype(int) - 2
        high = np.ceil(px.max(axis=0)).astype(int) + 2
        size = high - low
        if size.min() <= 0 or size.max() > 2000:
            return
        factor = 3
        big = pygame.Surface(tuple(size * factor), pygame.SRCALPHA)
        pygame.draw.polygon(big, (*colour, alpha), [tuple(p) for p in (px - low) * factor])
        surface.blit(pygame.transform.smoothscale(big, tuple(size)), tuple(low))

    def _rect_along(self, surface, center, heading, length, width, colour, alpha=255) -> None:
        c, s = math.cos(heading), math.sin(heading)
        hl, hw = length / 2.0, width / 2.0
        center = np.asarray(center)
        corners = [center + (c * dx - s * dy, s * dx + c * dy)
                   for dx, dy in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))]
        self._polygon(surface, corners, colour, alpha)

    def _clear_of_road(self, x: float, y: float, margin: float) -> bool:
        clearance = self.field.sample_sdf(np.array([x]), np.array([y]))[0]
        return clearance < -margin

    def _decorate(self, surface: pygame.Surface) -> None:
        self.progress(0.82, "Placing scenery")
        self._paint_road_markings(surface)
        self._paint_pit_complex(surface)
        self._paint_grandstand(surface)
        self._paint_plaza(surface)
        self._paint_tyre_walls(surface)
        self._paint_trees(surface)
        self._paint_light_towers(surface)

    def _paint_road_markings(self, surface: pygame.Surface) -> None:
        track = self.track
        hw = track.half_width
        settings = self.settings.training

        # Direction chevrons, faint enough to read as paint rather than as UI.
        for s in np.arange(420.0, track.length - 200.0, 640.0):
            i = int(s / track.spacing)
            p, t, n = track.points[i], track.tangents[i], track.normals[i]
            for lateral in (-26.0, 26.0):
                tip = p + n * lateral + t * 16.0
                self._polygon(surface, [tip, tip - t * 14 - n * 13, tip - t * 24 - n * 13,
                                        tip - t * 10, tip - t * 24 + n * 13, tip - t * 14 + n * 13],
                              (235, 238, 242), 46)

        # Checkpoint gates: a dotted line across the road plus a marker post on each side.
        for index, s in enumerate(track.checkpoint_s):
            if index == 0:
                continue
            i = int(s / track.spacing)
            p, t, n = track.points[i], track.tangents[i], track.normals[i]
            for lateral in np.arange(-hw + 22.0, hw - 18.0, 12.0):
                self._rect_along(surface, p + n * lateral, track.headings[i], 2.4, 6.0, (240, 244, 248), 36)
            for side in (-1.0, 1.0):
                self._rect_along(surface, p + n * side * (hw - 17.0), track.headings[i], 5.0, 5.0, (70, 214, 190), 170)

        # Start/finish: a chequered band across the full width.
        i = 0
        p, t, n = track.points[i], track.tangents[i], track.normals[i]
        cell = 9.0
        columns = int(2 * (hw - 1) / cell)
        for row in range(3):
            for column in range(columns):
                lateral = -hw + 1 + (column + 0.5) * cell
                colour = (240, 240, 240) if (row + column) % 2 == 0 else (16, 16, 18)
                self._rect_along(surface, p + t * (row - 1) * cell + n * lateral,
                                 track.headings[i], cell, cell, colour)

        # Grid slot markers for the largest supported field.
        for slot in range(100):
            row = slot // settings.grid_lanes
            lane = slot % settings.grid_lanes - (settings.grid_lanes - 1) / 2.0
            s = (-30.0 - row * settings.grid_row_spacing + 25.0) % track.length
            i = int(s / track.spacing) % track.count
            position = track.points[i] + track.normals[i] * lane * settings.grid_lane_spacing
            self._rect_along(surface, position, track.headings[i], 2.0, 14.0, (240, 240, 240), 70)

        # Circuit name painted on the road past the line.
        label = self.theme.text("AI RACING LAB", int(40 * self.scale), (240, 242, 246), "ExtraBold", tracking=6)
        i = int(330.0 / track.spacing)
        angle = -math.degrees(track.headings[i])
        rotated = pygame.transform.rotozoom(label, angle, 1.0)
        rotated.set_alpha(48)
        center = self._to_pixels(track.points[i])
        surface.blit(rotated, rotated.get_rect(center=tuple(center)))

    def _paint_pit_complex(self, surface: pygame.Surface) -> None:
        """Pit lane and garages in the infield beside the main straight."""
        y_road = 1290.0
        left, right = 1270.0, 2240.0
        self._polygon(surface, [(left, y_road - 74), (right, y_road - 74), (right, y_road + 54), (left, y_road + 54)],
                      (34, 36, 41))
        self._polygon(surface, [(left, y_road + 6), (right, y_road + 6), (right, y_road + 50), (left, y_road + 50)],
                      (56, 58, 64))
        self._polygon(surface, [(left, y_road + 8), (right, y_road + 8), (right, y_road + 10), (left, y_road + 10)],
                      (235, 235, 235), 200)
        bays = 16
        width = (right - left - 40.0) / bays
        for bay in range(bays):
            x = left + 20.0 + bay * width
            tone = (58, 62, 72) if bay % 2 == 0 else (48, 52, 60)
            self._polygon(surface, [(x + 2, y_road - 66), (x + width - 2, y_road - 66),
                                    (x + width - 2, y_road + 2), (x + 2, y_road + 2)], tone)
            self._polygon(surface, [(x + 2, y_road - 66), (x + width - 2, y_road - 66),
                                    (x + width - 2, y_road - 58), (x + 2, y_road - 58)], (92, 98, 110))
            self._polygon(surface, [(x + 8, y_road - 4), (x + width - 8, y_road - 4),
                                    (x + width - 8, y_road + 2), (x + 8, y_road + 2)], SIGNAL if bay % 4 == 0 else (120, 126, 138))
        label = self.theme.text("PIT LANE", int(26 * self.scale), (216, 220, 226), "Bold", tracking=8)
        faded = label.copy()
        faded.set_alpha(150)
        surface.blit(faded, faded.get_rect(center=tuple(self._to_pixels(((left + right) / 2, y_road + 29)))))

    def _paint_grandstand(self, surface: pygame.Surface) -> None:
        """Stepped seating with a crowd of tiny dots, outside the main straight."""
        rng = np.random.default_rng(5)
        x0, x1 = 1120.0, 2240.0
        y0 = 1450.0 + self.track.half_width + 50.0
        depth = 118.0
        self._polygon(surface, [(x0 - 8, y0 + 8), (x1 + 8, y0 + 8), (x1 + 16, y0 + depth + 24), (x0, y0 + depth + 24)],
                      (8, 9, 11), 150)
        rows = 8
        for row in range(rows):
            top = y0 + row * depth / rows
            bottom = top + depth / rows - 1.5
            tone = int(40 + 3 * row)
            self._polygon(surface, [(x0, top), (x1, top), (x1, bottom), (x0, bottom)], (tone, tone + 2, tone + 7))
        palette = np.array([(226, 228, 232), (255, 84, 46), (58, 214, 190), (250, 212, 110),
                            (120, 150, 210), (200, 90, 120), (150, 156, 168)])
        for _ in range(6200):
            row = rng.integers(0, rows)
            x = rng.uniform(x0 + 6, x1 - 6)
            y = y0 + (row + rng.uniform(0.15, 0.7)) * depth / rows
            colour = palette[rng.integers(0, len(palette))]
            px = self._to_pixels((x, y)).astype(int)
            surface.set_at(tuple(px), tuple(int(c) for c in colour))
            surface.set_at((px[0] + 1, px[1]), tuple(int(c * 0.72) for c in colour))
        self._polygon(surface, [(x0 - 6, y0 - 22), (x1 + 6, y0 - 22), (x1 + 6, y0 - 2), (x0 - 6, y0 - 2)], (24, 26, 31))
        step = (x1 - x0) / 5.0
        for i in range(5):
            label = self.theme.text("AI RACING LAB", int(13 * self.scale), (244, 244, 246), "Bold", tracking=3)
            surface.blit(label, label.get_rect(center=tuple(self._to_pixels((x0 + step * (i + 0.5), y0 - 12)))))

    def _paint_plaza(self, surface: pygame.Surface) -> None:
        """A circular hospitality plaza in the big infield, carrying the lab's mark."""
        center = np.array((400.0, 860.0))
        for radius, colour, alpha in ((132, (6, 7, 9), 110), (122, (36, 38, 44), 255), (112, (46, 49, 56), 255),
                                      (70, (28, 30, 35), 255), (66, (255, 84, 46), 255), (61, (20, 22, 26), 255)):
            angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
            offset = (10, 12) if radius == 132 else (0, 0)
            self._polygon(surface, center + offset + np.column_stack([np.cos(angles), np.sin(angles)]) * radius, colour, alpha)
        for k in range(16):
            angle = k * np.pi / 8
            direction = np.array((math.cos(angle), math.sin(angle)))
            self._rect_along(surface, center + direction * 92.0, angle, 30.0, 3.0, (84, 88, 98))
        label = self.theme.text("AI", int(46 * self.scale), (244, 244, 246), "Black", tracking=2)
        surface.blit(label, label.get_rect(center=tuple(self._to_pixels(center))))

    def _paint_tyre_walls(self, surface: pygame.Surface) -> None:
        """Stacks of tyres on the outside of the sharpest corners."""
        track = self.track
        rng = np.random.default_rng(23)
        peaks = [i for i in range(track.count) if abs(track.curvature[i]) > 1.0 / 210.0]
        for i in peaks:
            side = -np.sign(track.curvature[i]) or 1.0
            base = track.points[i] + track.normals[i] * side * (track.half_width + 22.0)
            if not self._clear_of_road(base[0], base[1], 14.0):
                continue
            for row in range(2):
                position = base + track.normals[i] * side * row * 8.5 + track.tangents[i] * rng.uniform(-1, 1)
                tint = (64, 67, 74) if (i // 3 + row) % 4 else (150, 62, 58)
                self._dot(surface, position, 4.6, tint)
                self._dot(surface, position, 1.9, (10, 10, 12))

    def _dot(self, surface: pygame.Surface, position, radius: float, colour, alpha: int = 255) -> None:
        pixel_radius = radius * self.scale
        size = int(pixel_radius * 2 + 4)
        big = pygame.Surface((size * 3, size * 3), pygame.SRCALPHA)
        pygame.draw.circle(big, (*colour, alpha), (size * 3 // 2, size * 3 // 2), int(pixel_radius * 3))
        small = pygame.transform.smoothscale(big, (size, size))
        surface.blit(small, small.get_rect(center=tuple(self._to_pixels(position))))

    def _tree_sprites(self, rng: np.random.Generator) -> list[pygame.Surface]:
        sprites = []
        for size in (34, 44, 56, 70):
            lumps = _value_noise(256, 12, rng)[:size, :size]
            axis = (np.arange(size) + 0.5) / size * 2.0 - 1.0
            xx, yy = np.meshgrid(axis, axis)
            radius = np.hypot(xx, yy) + lumps * 0.13
            alpha = np.clip((0.92 - radius) * 7.0, 0.0, 1.0)
            # Lit from the upper left, matching the direction of the car shadows.
            light = np.clip(0.62 - 0.34 * (xx + yy) + lumps * 0.34, 0.2, 1.05)
            surface = pygame.Surface((size, size), pygame.SRCALPHA)
            rgb = pygame.surfarray.pixels3d(surface)
            rgb[..., 0] = (light * 44).astype(np.uint8).T
            rgb[..., 1] = (light * 68).astype(np.uint8).T
            rgb[..., 2] = (light * 54).astype(np.uint8).T
            del rgb
            pixels_alpha = pygame.surfarray.pixels_alpha(surface)
            pixels_alpha[:] = (alpha * 255).astype(np.uint8).T
            del pixels_alpha
            sprites.append(surface)
        return sprites

    def _paint_trees(self, surface: pygame.Surface) -> None:
        rng = np.random.default_rng(3)
        sprites = self._tree_sprites(rng)
        shadows = []
        for sprite in sprites:
            shadow = sprite.copy()
            shadow.fill((0, 0, 0, 120), special_flags=pygame.BLEND_RGBA_MIN)
            shadows.append(shadow)

        placed = 0
        attempts = 0
        spots: list[tuple[float, float, int]] = []
        area_factor = self.world_w * self.world_h / 7.4e6 if self.portrait else 1.0
        while placed < int(330 * area_factor) and attempts < int(9000 * area_factor):
            attempts += 1
            # Clusters look far more natural than an even scatter.
            if spots and rng.random() < 0.6:
                base = spots[rng.integers(0, len(spots))]
                x = base[0] + rng.normal(0, 50)
                y = base[1] + rng.normal(0, 50)
            else:
                x = rng.uniform(self.world_x, self.world_x + self.world_w)
                y = rng.uniform(self.world_y, self.world_y + self.world_h)
            kind = int(rng.integers(0, len(sprites)))
            radius = sprites[kind].get_width() / self.scale / 2.0
            if not self._clear_of_road(x, y, 52.0 + radius):
                continue
            if any(a <= x <= c and b <= y <= d for a, b, c, d in KEEP_OUT):
                continue
            spots.append((x, y, kind))
            placed += 1

        spots.sort(key=lambda spot: spot[1])  # paint back to front
        for x, y, kind in spots:
            center = self._to_pixels((x, y))
            surface.blit(shadows[kind], shadows[kind].get_rect(center=(center[0] + 7, center[1] + 9)))
            surface.blit(sprites[kind], sprites[kind].get_rect(center=tuple(center)))

    def _paint_light_towers(self, surface: pygame.Surface) -> None:
        for lx, ly in self.lights:
            center = self._to_pixels((lx, ly))
            size = 150
            glow = np.zeros((size, size), dtype=np.float32)
            axis = (np.arange(size) + 0.5) / size * 2.0 - 1.0
            xx, yy = np.meshgrid(axis, axis)
            glow = np.exp(-(xx * xx + yy * yy) * 7.0)
            sprite = pygame.Surface((size, size))
            rgb = pygame.surfarray.pixels3d(sprite)
            rgb[..., 0] = (glow * 58).astype(np.uint8).T
            rgb[..., 1] = (glow * 58).astype(np.uint8).T
            rgb[..., 2] = (glow * 52).astype(np.uint8).T
            del rgb
            surface.blit(sprite, sprite.get_rect(center=tuple(center)), special_flags=pygame.BLEND_RGB_ADD)
            self._dot(surface, (lx + 4, ly + 5), 7.0, (0, 0, 0), 130)
            self._rect_along(surface, (lx, ly), 0.0, 13.0, 8.0, (72, 76, 86))
            for dx in (-4.0, 0.0, 4.0):
                self._rect_along(surface, (lx + dx, ly), 0.0, 2.6, 5.0, (255, 248, 224))

    # ------------------------------------------------------------------ per frame

    def draw(self, target: pygame.Surface, left: float, top: float, zoom: float) -> tuple[float, float, float]:
        """Draw the view at (left, top, zoom), returning the exact view used.

        Source rectangles snap to whole pixels. Returning the snapped view keeps cars
        locked to the road instead of swimming against it by a fraction of a pixel.
        """
        width, height = target.get_size()
        level_zoom, level = self.mips[-1]
        for candidate_zoom, candidate in self.mips:
            if candidate_zoom >= zoom - 1e-4:
                level_zoom, level = candidate_zoom, candidate
                break

        level_w, level_h = level.get_size()
        source_w = min(level_w, max(1, int(round(width * level_zoom / zoom))))
        source_h = min(level_h, max(1, int(round(height * level_zoom / zoom))))
        source_x = int(round((left - self.view_x) * level_zoom))
        source_y = int(round((top - self.view_y) * level_zoom))
        source_x = max(0, min(source_x, level_w - source_w))
        source_y = max(0, min(source_y, level_h - source_h))

        if source_w == width and source_h == height:
            target.blit(level, (0, 0), (source_x, source_y, width, height))
        else:
            region = level.subsurface((source_x, source_y, source_w, source_h))
            pygame.transform.smoothscale(region, (width, height), target)
        return (self.view_x + source_x / level_zoom, self.view_y + source_y / level_zoom,
                width / (source_w / level_zoom))
