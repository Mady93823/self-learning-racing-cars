"""Procedurally drawn top-down open-wheel race cars, with a rotation cache.

Each car is drawn once as vector shapes. Rotated and scaled copies are created on
first use and reused afterwards, so 100 cars cost little more than 100 blits.
"""

from __future__ import annotations

import math

import pygame

from settings import PhysicsSettings

# (body, accent, secondary) colours per livery. All original team colours.
LIVERIES = [
    ((236, 62, 52), (250, 250, 250), (24, 24, 28)),
    ((30, 120, 232), (255, 214, 66), (16, 22, 40)),
    ((24, 196, 164), (12, 34, 44), (232, 244, 240)),
    ((250, 148, 34), (24, 28, 44), (255, 240, 220)),
    ((168, 86, 240), (250, 214, 250), (30, 18, 50)),
    ((240, 240, 244), (232, 44, 80), (40, 44, 54)),
    ((246, 206, 44), (22, 24, 28), (250, 250, 240)),
    ((236, 66, 148), (34, 22, 44), (255, 226, 240)),
    ((46, 206, 88), (250, 250, 250), (14, 40, 26)),
    ((70, 216, 240), (20, 40, 110), (235, 250, 255)),
    ((150, 158, 176), (255, 84, 46), (34, 36, 44)),
    ((196, 40, 60), (246, 190, 60), (36, 14, 20)),
    ((64, 88, 216), (240, 244, 255), (16, 20, 56)),
    ((134, 214, 56), (24, 44, 22), (240, 250, 220)),
    ((255, 108, 88), (40, 30, 54), (255, 236, 226)),
    ((36, 150, 148), (250, 196, 96), (10, 36, 40)),
]

DESIGN_SCALE = 3.0      # pixels per world unit of the master sprite
SUPERSAMPLE = 2
ZOOM_BUCKETS = (0.5, 0.7, 0.95, 1.25, 1.6)
FINE_ZOOM_BUCKETS = tuple(round(0.4 + 0.05 * i, 2) for i in range(19))
CARBON = (22, 23, 27)
RUBBER = (13, 13, 15)
ALLOY = (150, 155, 165)


def _car_surface(livery_index: int, steer: int, wheel_phase: int, cfg: PhysicsSettings) -> pygame.Surface:
    """Paint one master sprite, facing +x. Coordinates below are in world units."""
    body, accent, secondary = LIVERIES[livery_index % len(LIVERIES)]
    scale = DESIGN_SCALE * SUPERSAMPLE
    length = cfg.car_length
    half_length = length / 2.0
    half_width = cfg.car_width / 2.0 + 2.0
    size = (int((half_length + 2) * 2 * scale), int(half_width * 2 * scale))
    surface = pygame.Surface(size, pygame.SRCALPHA)
    ox, oy = size[0] / 2.0, size[1] / 2.0

    def to_px(x: float, y: float) -> tuple[float, float]:
        return ox + x * scale, oy + y * scale

    def poly(points, colour) -> None:
        pygame.draw.polygon(surface, colour, [to_px(x, y) for x, y in points])

    def mirrored(points, colour) -> None:
        poly(points, colour)
        poly([(x, -y) for x, y in points], colour)

    def box(x0, y0, x1, y1, colour, radius: float = 0.0) -> None:
        a = to_px(x0, y0)
        b = to_px(x1, y1)
        pygame.draw.rect(surface, colour, (a[0], a[1], b[0] - a[0], b[1] - a[1]),
                         border_radius=int(radius * scale))

    def wheel(cx: float, cy: float, wheel_length: float, wheel_width: float, angle: float = 0.0) -> None:
        c, s = math.cos(angle), math.sin(angle)

        def corner(dx, dy):
            return cx + c * dx - s * dy, cy + s * dx + c * dy

        hl, hw = wheel_length / 2.0, wheel_width / 2.0
        poly([corner(-hl, -hw), corner(hl, -hw), corner(hl, hw), corner(-hl, hw)], RUBBER)
        # Tread bars slide along the tyre as it rolls.
        bars = 5
        for k in range(bars):
            offset = -hl + ((k + wheel_phase / 3.0) % bars + 0.5) * wheel_length / bars
            poly([corner(offset - 0.28, -hw + 0.2), corner(offset + 0.28, -hw + 0.2),
                  corner(offset + 0.28, hw - 0.2), corner(offset - 0.28, hw - 0.2)], (40, 41, 46))
        poly([corner(-hl * 0.45, -hw * 0.34), corner(hl * 0.45, -hw * 0.34),
              corner(hl * 0.45, hw * 0.34), corner(-hl * 0.45, hw * 0.34)], ALLOY)

    # Suspension arms, under everything else.
    for x_outer, x_in_a, x_in_b in ((-16.5, -13.0, -8.0), (15.0, 12.0, 6.5)):
        for sign in (-1, 1):
            pygame.draw.line(surface, (70, 72, 80), to_px(x_outer, sign * 9.0), to_px(x_in_a, sign * 3.2), int(0.5 * scale))
            pygame.draw.line(surface, (70, 72, 80), to_px(x_outer, sign * 9.0), to_px(x_in_b, sign * 3.0), int(0.5 * scale))

    # Wheels. Front pair turn with the steering.
    steer_angle = steer * math.radians(18.0)
    for sign in (-1, 1):
        wheel(-16.5, sign * 9.6, 11.0, 5.2)
        wheel(15.0, sign * 9.6, 9.5, 4.4, steer_angle)

    # Carbon floor.
    mirrored([(-20, 0), (-20, 6.4), (-9, 8.2), (4, 8.2), (9, 3.4), (22, 2.2), (22, 0)], CARBON)

    # Sidepods with an undercut and a highlight stripe.
    mirrored([(-15, 2.4), (-15, 5.2), (-8, 7.3), (3.2, 7.3), (6.2, 5.3), (6.2, 2.4)], body)
    mirrored([(-11, 5.7), (2.4, 6.6), (3.2, 7.3), (-8, 7.3)], secondary)
    mirrored([(2.0, 3.2), (5.6, 3.2), (5.6, 6.0), (3.2, 6.8)], (12, 12, 14))  # air intake

    # Engine cover and spine.
    poly([(-20, -1.5), (-6, -3.3), (3, -3.6), (3, 3.6), (-6, 3.3), (-20, 1.5)], body)
    poly([(-19, -0.5), (3, -0.8), (3, 0.8), (-19, 0.5)], accent)

    # Nose, tapering to the front wing.
    poly([(3, -3.6), (9, -3.2), (22, -1.5), (23.4, 0), (22, 1.5), (9, 3.2), (3, 3.6)], body)
    poly([(9, -0.8), (22.5, -0.45), (22.5, 0.45), (9, 0.8)], accent)

    # Cockpit, helmet and halo.
    poly([(1.0, -2.7), (5.2, -2.9), (8.6, -1.5), (8.6, 1.5), (5.2, 2.9), (1.0, 2.7)], (8, 8, 10))
    pygame.draw.circle(surface, accent, to_px(3.3, 0), int(1.85 * scale))
    pygame.draw.circle(surface, secondary, to_px(3.9, 0), int(0.8 * scale))
    pygame.draw.lines(surface, (40, 42, 48), False,
                      [to_px(1.6, -2.7), to_px(6.4, -2.3), to_px(9.0, 0), to_px(6.4, 2.3), to_px(1.6, 2.7)],
                      int(0.55 * scale))

    # Front wing: main plane, flap, endplates.
    box(21.0, -11.8, 24.8, 11.8, CARBON, 0.5)
    box(21.8, -11.2, 23.2, 11.2, accent, 0.3)
    box(23.5, -11.4, 24.5, 11.4, body, 0.2)
    for sign in (-1, 1):
        box(20.6, min(sign * 11.2, sign * 12.4), 25.4, max(sign * 11.2, sign * 12.4), body, 0.2)

    # Rear wing: main plane and endplates, overhanging the engine cover.
    box(-25.8, -10.6, -21.2, 10.6, CARBON, 0.4)
    box(-25.0, -10.0, -22.0, 10.0, body, 0.3)
    box(-24.2, -7.0, -22.8, 7.0, accent, 0.2)
    for sign in (-1, 1):
        box(-26.2, min(sign * 10.0, sign * 11.2), -20.8, max(sign * 10.0, sign * 11.2), secondary, 0.2)

    # A soft highlight down the spine gives the bodywork some volume.
    sheen = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.polygon(sheen, (255, 255, 255, 34), [to_px(x, y) for x, y in
                        ((-17, -1.6), (2, -2.4), (20, -0.6), (20, -0.1), (2, -1.0), (-17, -0.6))])
    surface.blit(sheen, (0, 0))

    return pygame.transform.smoothscale(surface, (size[0] // SUPERSAMPLE, size[1] // SUPERSAMPLE))


class CarRenderer:
    MAX_CACHED = 2600

    def __init__(self, cfg: PhysicsSettings, fine_zoom: bool = False):
        self.cfg = cfg
        # Continuously zooming video needs closely spaced sizes, or the cars visibly pop.
        self.zoom_buckets = FINE_ZOOM_BUCKETS if fine_zoom else ZOOM_BUCKETS
        self._masters: dict[tuple, pygame.Surface] = {}
        self._rotated: dict[tuple, pygame.Surface] = {}
        self._shadows: dict[tuple, pygame.Surface] = {}
        self._shadow_master: pygame.Surface | None = None

    def bucket_zoom(self, zoom: float) -> float:
        return min(self.zoom_buckets, key=lambda z: abs(z - zoom))

    def _master(self, livery: int, steer: int, phase: int) -> pygame.Surface:
        key = (livery, steer, phase)
        master = self._masters.get(key)
        if master is None:
            master = _car_surface(livery, steer, phase, self.cfg)
            self._masters[key] = master
        return master

    def sprite(self, livery: int, heading: float, zoom: float, steer: float = 0.0, phase: int = 0,
               detail: bool = False) -> pygame.Surface:
        """Rotated sprite for a heading in radians. Detail is reduced when far away."""
        zoom_bucket = self.bucket_zoom(zoom)
        close = detail and zoom_bucket >= 0.9
        step = 2 if close else (3 if zoom_bucket >= 0.9 else 4)
        angle = int(round(math.degrees(heading) / step)) * step % 360
        steer_bucket = 0
        wheel = 0
        if close:
            steer_bucket = 1 if steer > 0.3 else (-1 if steer < -0.3 else 0)
            wheel = phase % 3 if zoom_bucket >= 1.2 else 0
        key = (livery % len(LIVERIES), angle, zoom_bucket, steer_bucket, wheel)
        cached = self._rotated.get(key)
        if cached is not None:
            return cached
        master = self._master(key[0], steer_bucket, wheel)
        rotated = pygame.transform.rotozoom(master, -angle, 1.0)
        factor = zoom_bucket / DESIGN_SCALE
        size = (max(2, int(rotated.get_width() * factor)), max(2, int(rotated.get_height() * factor)))
        sprite = pygame.transform.smoothscale(rotated, size)
        if len(self._rotated) >= self.MAX_CACHED:
            # Drop the oldest quarter. Dicts keep insertion order, which approximates LRU here.
            for old in list(self._rotated)[: self.MAX_CACHED // 4]:
                del self._rotated[old]
        self._rotated[key] = sprite
        return sprite

    def shadow(self, heading: float, zoom: float) -> pygame.Surface:
        """Soft, dark silhouette. Cached separately because all liveries share it."""
        zoom_bucket = self.bucket_zoom(zoom)
        angle = int(round(math.degrees(heading) / 6)) * 6 % 360
        key = (angle, zoom_bucket)
        cached = self._shadows.get(key)
        if cached is not None:
            return cached
        if self._shadow_master is None:
            master = _car_surface(0, 0, 0, self.cfg)
            master.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MIN)
            self._shadow_master = master
        rotated = pygame.transform.rotozoom(self._shadow_master, -angle, 1.0)
        factor = zoom_bucket / DESIGN_SCALE
        width, height = max(2, int(rotated.get_width() * factor)), max(2, int(rotated.get_height() * factor))
        # Downscale then upscale: an inexpensive blur.
        small = pygame.transform.smoothscale(rotated, (max(2, width // 2), max(2, height // 2)))
        shadow = pygame.transform.smoothscale(small, (width, height))
        shadow.fill((255, 255, 255, 120), special_flags=pygame.BLEND_RGBA_MULT)
        self._shadows[key] = shadow
        return shadow

    def save_preview(self, path, count: int = 8) -> None:
        """Writes a contact sheet of the liveries, useful for checking the artwork."""
        width = 170
        sheet = pygame.Surface((width * 4, 90 * ((count + 3) // 4)), pygame.SRCALPHA)
        for i in range(count):
            master = self._master(i, 0, 0)
            sheet.blit(master, master.get_rect(center=(width * (i % 4) + width // 2, 90 * (i // 4) + 45)))
        pygame.image.save(sheet, str(path))
