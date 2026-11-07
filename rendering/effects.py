"""Short-lived world effects: gate flashes, sensor rays, highlights, headlights, wrecks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame

from simulation.track import Track
from ui.theme import AMBER, DATA, DANGER, SIGNAL, mix


@dataclass
class Flash:
    index: int
    age: float = 0.0
    color: tuple = DATA
    life: float = 0.8


@dataclass
class Wreck:
    car: int
    age: float = 0.0
    life: float = 1.3


class EffectLayer:
    def __init__(self, track: Track):
        self.track = track
        self.flashes: list[Flash] = []
        self.wrecks: list[Wreck] = []
        self._glow_sprites: dict[tuple, pygame.Surface] = {}
        self._halo_sprites: dict[tuple, pygame.Surface] = {}
        self._gate_cache: dict[int, np.ndarray] = {}
        self.time = 0.0

    def clear(self) -> None:
        self.flashes.clear()
        self.wrecks.clear()

    def checkpoint(self, index: int, color=DATA) -> None:
        self.flashes.append(Flash(index, color=color))

    def wreck(self, car: int) -> None:
        self.wrecks.append(Wreck(car))

    def update(self, dt: float) -> None:
        self.time += dt
        for flash in self.flashes:
            flash.age += dt
        for wreck in self.wrecks:
            wreck.age += dt
        self.flashes = [f for f in self.flashes if f.age < f.life]
        self.wrecks = [w for w in self.wrecks if w.age < w.life]

    def _gate(self, index: int) -> np.ndarray:
        gate = self._gate_cache.get(index)
        if gate is None:
            track = self.track
            i = int(track.checkpoint_s[index] / track.spacing) % track.count
            p, n = track.points[i], track.normals[i]
            reach = track.half_width - 2.0
            gate = np.array((p - n * reach, p + n * reach))
            self._gate_cache[index] = gate
        return gate

    def draw_gates(self, target: pygame.Surface, camera) -> None:
        """Checkpoint activation: the gate lights up and fades as a thin band of light."""
        zoom = camera.view_zoom
        for flash in self.flashes:
            gate = self._gate(flash.index)
            a = camera.to_screen(*gate[0])
            b = camera.to_screen(*gate[1])
            t = flash.age / flash.life
            strength = (1.0 - t) ** 2
            color = mix((20, 22, 26), flash.color, strength)
            width = max(1, int((2.0 + 9.0 * t) * zoom))
            box = pygame.Rect(min(a[0], b[0]) - width, min(a[1], b[1]) - width,
                              abs(a[0] - b[0]) + width * 2 + 1, abs(a[1] - b[1]) + width * 2 + 1)
            if box.width <= 0 or box.height <= 0 or not box.colliderect(target.get_rect()):
                continue
            layer = pygame.Surface(box.size)
            start = (a[0] - box.x, a[1] - box.y)
            end = (b[0] - box.x, b[1] - box.y)
            pygame.draw.line(layer, color, start, end, width)
            target.blit(layer, box, special_flags=pygame.BLEND_RGB_ADD)
            pygame.draw.aaline(target, mix((40, 40, 40), (255, 255, 255), strength), a, b)

    def headlight(self, target: pygame.Surface, sx: float, sy: float, heading: float, zoom: float) -> None:
        """Additive beam ahead of the car. Rotated copies are cached."""
        step = 6
        angle = int(round(math.degrees(heading) / step)) * step % 360
        size_bucket = round(zoom, 1)
        key = (angle, size_bucket)
        sprite = self._glow_sprites.get(key)
        if sprite is None:
            sprite = self._make_beam(angle, size_bucket)
            self._glow_sprites[key] = sprite
        target.blit(sprite, sprite.get_rect(center=(sx, sy)), special_flags=pygame.BLEND_RGB_ADD)

    @staticmethod
    def _make_beam(angle: int, zoom: float) -> pygame.Surface:
        extent = max(16, int(92 * zoom))
        axis = np.linspace(-1.0, 1.0, extent * 2)
        xx, yy = np.meshgrid(axis, axis)
        theta = np.radians(angle)
        along = (xx * np.cos(theta) + yy * np.sin(theta)) * extent
        across = (-xx * np.sin(theta) + yy * np.cos(theta)) * extent
        forward = np.clip(along, 0.0, None) / extent
        spread = 0.06 + forward * 0.42
        beam = np.exp(-((across / extent / spread) ** 2)) * np.clip(1.0 - forward, 0.0, 1.0) ** 1.4
        beam = beam * np.clip(along / (extent * 0.16), 0.0, 1.0)
        surface = pygame.Surface((extent * 2, extent * 2))
        rgb = pygame.surfarray.pixels3d(surface)
        rgb[..., 0] = (beam * 72).astype(np.uint8).T
        rgb[..., 1] = (beam * 70).astype(np.uint8).T
        rgb[..., 2] = (beam * 58).astype(np.uint8).T
        del rgb
        return surface

    def halo(self, target: pygame.Surface, sx: float, sy: float, zoom: float, color=SIGNAL) -> None:
        """Pulsing ring that marks the current best car."""
        pulse = 0.5 + 0.5 * math.sin(self.time * 4.2)
        radius = int((36 + 5 * pulse) * max(zoom, 0.72))
        step = 2
        radius -= radius % step
        alpha_bucket = int(pulse * 4)
        key = (radius, alpha_bucket, color)
        sprite = self._halo_sprites.get(key)
        if sprite is None:
            size = radius * 2 + 8
            big = pygame.Surface((size * 3, size * 3), pygame.SRCALPHA)
            center = (size * 3 // 2, size * 3 // 2)
            pygame.draw.circle(big, (*color, 26 + alpha_bucket * 6), center, radius * 3)
            pygame.draw.circle(big, (*color, 170 + alpha_bucket * 18), center, radius * 3, width=max(3, int(2.0 * 3)))
            sprite = pygame.transform.smoothscale(big, (size, size))
            self._halo_sprites[key] = sprite
        target.blit(sprite, sprite.get_rect(center=(sx, sy)))


def draw_sensor_rays(target: pygame.Surface, camera, origin: tuple[float, float],
                     endpoints: np.ndarray, readings: np.ndarray, emphasis: float = 1.0) -> None:
    """Each ray runs from green (clear) to red (barrier close), with a dot at the hit point."""
    start = camera.to_screen(*origin)
    for (ex, ey), reading in zip(endpoints, readings):
        danger = 1.0 - min(1.0, reading / 0.42)
        base = mix(DATA, AMBER, min(1.0, danger * 2.0)) if danger < 0.5 else mix(AMBER, DANGER, (danger - 0.5) * 2.0)
        color = mix((26, 28, 32), base, emphasis)
        end = camera.to_screen(ex, ey)
        pygame.draw.aaline(target, color, start, end)
        pygame.draw.circle(target, color, (int(end[0]), int(end[1])), 3)
