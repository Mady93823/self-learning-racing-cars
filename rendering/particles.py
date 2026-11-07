"""A small fixed-size particle pool: tyre smoke, collision sparks and lap confetti.

Purely cosmetic. Particles use their own random generator, so spawning them can't
disturb the random stream NEAT uses for evolution.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pygame

SMOKE, SPARK, CONFETTI = 0, 1, 2
CONFETTI_COLOURS = ((255, 84, 46), (250, 212, 110), (58, 214, 190), (240, 242, 246))


class ParticleSystem:
    def __init__(self, limit: int):
        self.limit = limit
        self.x = np.zeros(limit)
        self.y = np.zeros(limit)
        self.vx = np.zeros(limit)
        self.vy = np.zeros(limit)
        self.age = np.ones(limit)
        self.life = np.ones(limit)
        self.size = np.ones(limit)
        self.kind = np.zeros(limit, dtype=np.int64)
        self.tint = np.zeros(limit, dtype=np.int64)
        self.alive = np.zeros(limit, dtype=bool)
        self._cursor = 0
        self._rng = random.Random(99)
        self._sprites: dict[tuple, pygame.Surface] = {}

    def clear(self) -> None:
        self.alive[:] = False

    def _spawn(self, x, y, vx, vy, life, size, kind, tint=0) -> None:
        i = self._cursor
        self._cursor = (i + 1) % self.limit
        self.x[i], self.y[i], self.vx[i], self.vy[i] = x, y, vx, vy
        self.age[i], self.life[i], self.size[i] = 0.0, life, size
        self.kind[i], self.tint[i], self.alive[i] = kind, tint, True

    def smoke(self, x: float, y: float, vx: float, vy: float) -> None:
        r = self._rng
        self._spawn(x, y, vx * 0.12 + r.uniform(-12, 12), vy * 0.12 + r.uniform(-12, 12),
                    r.uniform(0.45, 0.85), r.uniform(5.0, 8.0), SMOKE)

    def sparks(self, x: float, y: float, speed: float, count: int = 12) -> None:
        r = self._rng
        for _ in range(count):
            angle = r.uniform(0, math.tau)
            pace = r.uniform(60, 180 + speed * 0.45)
            self._spawn(x, y, math.cos(angle) * pace, math.sin(angle) * pace,
                        r.uniform(0.2, 0.55), r.uniform(1.6, 2.8), SPARK)

    def confetti(self, x: float, y: float, count: int = 34) -> None:
        r = self._rng
        for _ in range(count):
            angle = r.uniform(0, math.tau)
            pace = r.uniform(70, 330)
            self._spawn(x, y, math.cos(angle) * pace, math.sin(angle) * pace,
                        r.uniform(0.7, 1.5), r.uniform(2.4, 4.2), CONFETTI, r.randrange(len(CONFETTI_COLOURS)))

    def update(self, dt: float) -> None:
        ids = np.nonzero(self.alive)[0]
        if ids.size == 0:
            return
        self.age[ids] += dt
        self.x[ids] += self.vx[ids] * dt
        self.y[ids] += self.vy[ids] * dt
        drag = np.where(self.kind[ids] == SMOKE, 1.5, 2.4)
        factor = np.exp(-drag * dt)
        self.vx[ids] *= factor
        self.vy[ids] *= factor
        self.alive[ids] = self.age[ids] < self.life[ids]

    def _sprite(self, kind: int, radius: int, alpha_step: int, tint: int) -> pygame.Surface:
        key = (kind, radius, alpha_step, tint)
        cached = self._sprites.get(key)
        if cached is not None:
            return cached
        size = radius * 2 + 2
        big = pygame.Surface((size * 3, size * 3), pygame.SRCALPHA)
        alpha = int(alpha_step * 255 / 8)
        center = (size * 3 // 2, size * 3 // 2)
        if kind == SMOKE:
            # Layered discs make a soft puff with no per-pixel maths.
            for k in range(4, 0, -1):
                pygame.draw.circle(big, (196, 198, 204, alpha // 5), center, int(radius * 3 * k / 4))
        elif kind == SPARK:
            pygame.draw.circle(big, (255, 150, 60, alpha), center, radius * 3)
            pygame.draw.circle(big, (255, 240, 200, alpha), center, max(1, radius * 2))
        else:
            r, g, b = CONFETTI_COLOURS[tint]
            pygame.draw.rect(big, (r, g, b, alpha), (center[0] - radius * 3, center[1] - radius * 2, radius * 6, radius * 4))
        sprite = pygame.transform.smoothscale(big, (size, size))
        self._sprites[key] = sprite
        return sprite

    def draw(self, target: pygame.Surface, camera) -> None:
        ids = np.nonzero(self.alive)[0]
        if ids.size == 0:
            return
        zoom = camera.view_zoom
        for i in ids.tolist():
            sx, sy = camera.to_screen(self.x[i], self.y[i])
            if not (-30 < sx < target.get_width() + 30 and -30 < sy < target.get_height() + 30):
                continue
            t = self.age[i] / self.life[i]
            growth = 1.0 + 1.4 * t if self.kind[i] == SMOKE else 1.0
            radius = max(1, min(30, int(self.size[i] * zoom * growth)))
            alpha_step = max(1, min(8, int((1.0 - t) * 8) + 1))
            sprite = self._sprite(int(self.kind[i]), radius, alpha_step, int(self.tint[i]))
            target.blit(sprite, (sx - radius, sy - radius))
