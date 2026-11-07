"""Three camera modes, plus the automatic director used for recordings.

Cameras never read a clock, only the frame ``dt`` they are given, so a video rendered
offline moves exactly the same as one watched live.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from simulation.track import Track
from ui.theme import ease_in_out


class CameraMode:
    FULL = "FULL TRACK"
    FOLLOW = "FOLLOW BEST"
    CINEMATIC = "CINEMATIC"
    ACTION = "ACTION"       # keeps the leading pack framed; used for vertical video
    ORDER = (FULL, FOLLOW, CINEMATIC)


@dataclass
class CameraScene:
    """What the camera may look at this frame."""

    focus: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    focus_s: float = 0.0
    speed: float = 0.0
    others: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


class Camera:
    TRANSITION_SECONDS = 1.5
    MODE_SMOOTHING = 3.2   # higher = tighter tracking
    ZOOM_SMOOTHING = 1.8
    ACTION_ZOOM_CLOSE = 1.25
    ACTION_ZOOM_WIDE = 1.0
    ACTION_FULL_SPEED = 640.0
    ACTION_LOOKAHEAD = 0.7  # seconds of travel; the car sits behind the centre of the shot

    def __init__(self, renderer, follow_zoom: float, track: Track, screen_size: tuple[int, int],
                 seed: int = 4):
        self.renderer = renderer
        self.track = track
        self.follow_zoom = follow_zoom
        self.screen_w, self.screen_h = screen_size
        self.free_rect = renderer.free_rect
        self.mode = CameraMode.FULL
        self.x, self.y = renderer.fit_center
        self.zoom = renderer.fit_zoom
        self.left = self.x - self.screen_w / 2 / self.zoom
        self.top = self.y - self.screen_h / 2 / self.zoom
        self.view_zoom = self.zoom
        self.center_panels = True
        # Live shots hold exact pre-rendered zoom levels, which cost a blit rather than a
        # full-screen rescale. Offline recording has no time limit, so it can zoom freely.
        self.continuous_zoom = False
        self._blend = 1.0
        self._from = (self.x, self.y, self.zoom)
        self._rng = random.Random(seed)
        self._shot = "chase"
        self._shot_time = 0.0
        self._shot_length = 7.0
        self._apex: tuple[float, float] | None = None
        self._apex_s = 0.0
        self._drift = 0.0

    def set_mode(self, mode: str, instant: bool = False) -> None:
        if mode == self.mode and not instant:
            return
        self.mode = mode
        self._from = (self.x, self.y, self.zoom)
        self._blend = 1.0 if instant else 0.0
        self._shot_time = self._shot_length  # the director picks a new shot immediately

    def next_mode(self) -> str:
        order = CameraMode.ORDER
        current = order.index(self.mode) if self.mode in order else -1
        self.set_mode(order[(current + 1) % len(order)])
        return self.mode

    # ------------------------------------------------------------------ targets

    def _frame_on(self, world_x: float, world_y: float, zoom: float) -> tuple[float, float]:
        """View centre that places a world point in the middle of the unobstructed area."""
        if not self.center_panels:
            return world_x, world_y
        return (world_x + (self.screen_w / 2 - self.free_rect.centerx) / zoom,
                world_y + (self.screen_h / 2 - self.free_rect.centery) / zoom)

    def _target_follow(self, scene: CameraScene) -> tuple[float, float, float]:
        zoom = self.follow_zoom
        ahead_x = scene.focus[0] + scene.velocity[0] * 0.42
        ahead_y = scene.focus[1] + scene.velocity[1] * 0.42
        x, y = self._frame_on(ahead_x, ahead_y, zoom)
        return x, y, zoom

    def _pick_shot(self, scene: CameraScene) -> None:
        options = [shot for shot in ("chase", "pack", "apex") if shot != self._shot]
        if scene.others.shape[0] < 3 and "pack" in options:
            options.remove("pack")
        self._shot = self._rng.choice(options)
        self._shot_time = 0.0
        self._shot_length = self._rng.uniform(6.0, 9.0)
        self._drift = self._rng.uniform(-1.0, 1.0)
        if self._shot == "apex":
            self._choose_apex(scene)

    def _choose_apex(self, scene: CameraScene) -> None:
        """Wait at the next proper corner, a couple of seconds ahead of the leader."""
        track = self.track
        ahead = max(scene.speed, 200.0) * 1.8
        start = int((scene.focus_s + ahead) / track.spacing)
        indices = (np.arange(start, start + 110) % track.count)
        best = int(indices[int(np.argmax(np.abs(track.curvature[indices])))])
        self._apex = self.renderer.to_view(float(track.points[best, 0]), float(track.points[best, 1]))
        self._apex_s = best * track.spacing

    def _target_cinematic(self, scene: CameraScene, dt: float) -> tuple[float, float, float]:
        self._shot_time += dt
        if self._shot == "apex":
            # Hold until the leader has gone clearly past the corner.
            gone = (scene.focus_s - self._apex_s) % self.track.length
            if 500.0 < gone < self.track.length / 2 or self._shot_time > 9.0:
                self._pick_shot(scene)
        elif self._shot_time >= self._shot_length:
            self._pick_shot(scene)

        if self._shot == "apex" and self._apex is not None:
            zoom = 1.05
            x, y = self._frame_on(*self._apex, zoom)
            return x, y, zoom
        if self._shot == "pack" and scene.others.shape[0] >= 2:
            low = scene.others.min(axis=0)
            high = scene.others.max(axis=0)
            span = np.maximum(high - low, 1.0) + 320.0
            zoom = min(self.free_rect.width / span[0], self.free_rect.height / span[1])
            zoom = float(np.clip(zoom, self.renderer.fit_zoom * 1.25, self.follow_zoom))
            center = (low + high) / 2.0
            x, y = self._frame_on(float(center[0]), float(center[1]), zoom)
            return x, y, zoom
        # Chase: a slow push-in with the car drifting slightly off centre.
        progress = min(self._shot_time / self._shot_length, 1.0)
        zoom = 1.0 + 0.25 * ease_in_out(progress)
        x = scene.focus[0] + scene.velocity[0] * 0.3 + self._drift * 60.0
        y = scene.focus[1] + scene.velocity[1] * 0.3 - self._drift * 40.0
        x, y = self._frame_on(x, y, zoom)
        return x, y, zoom

    # ------------------------------------------------------------------ update

    def _target_action(self, scene: CameraScene) -> tuple[float, float, float]:
        """Follow one car with the road ahead in view, framed between the cards and the caption.

        Zoom depends only on speed, like a chase camera: close on the grid, easing back at pace
        to show more of what's coming. It never reacts to other cars, so it never hunts.
        """
        pace = min(scene.speed / self.ACTION_FULL_SPEED, 1.0)
        zoom = self.ACTION_ZOOM_CLOSE + (self.ACTION_ZOOM_WIDE - self.ACTION_ZOOM_CLOSE) * pace
        ahead_x = scene.focus[0] + scene.velocity[0] * self.ACTION_LOOKAHEAD
        ahead_y = scene.focus[1] + scene.velocity[1] * self.ACTION_LOOKAHEAD
        rect = self.renderer.action_rect
        return (ahead_x + (self.screen_w / 2 - rect.centerx) / zoom,
                ahead_y + (self.screen_h / 2 - rect.centery) / zoom, zoom)

    def cut_to(self, scene: CameraScene) -> None:
        """Jump straight to the framing for this scene, with no camera move."""
        self.x, self.y, self.zoom = self._target_action(self._to_view_scene(scene))
        self._blend = 1.0
        self._clamp()

    def _to_view_scene(self, scene: CameraScene) -> CameraScene:
        """Express a scene in view space. Scene data arrives in world coordinates."""
        if not self.renderer.portrait:
            return scene
        others = scene.others
        if others.size:
            others = np.column_stack([-others[:, 1], others[:, 0]])
        return CameraScene((-scene.focus[1], scene.focus[0]), (-scene.velocity[1], scene.velocity[0]),
                           scene.focus_s, scene.speed, others)

    def update(self, dt: float, scene: CameraScene) -> None:
        scene = self._to_view_scene(scene)
        if self.mode == CameraMode.ACTION:
            target = self._target_action(scene)
        elif self.mode == CameraMode.FULL:
            x, y = self.renderer.fit_center
            target = (x, y, self.renderer.fit_zoom)
        elif self.mode == CameraMode.FOLLOW:
            target = self._target_follow(scene)
        else:
            target = self._target_cinematic(scene, dt)
            if not self.continuous_zoom:
                wanted = min((level for level, _ in self.renderer.mips), key=lambda z: abs(z - target[2]))
                # Re-frame for the zoom actually used, so the subject stays centred.
                target = (*self._reframe(target, wanted), wanted)

        if self._blend < 1.0:
            # Mode change: an eased move, working in log-zoom so it feels uniform.
            self._blend = min(1.0, self._blend + dt / self.TRANSITION_SECONDS)
            t = ease_in_out(self._blend)
            self.x = self._from[0] + (target[0] - self._from[0]) * t
            self.y = self._from[1] + (target[1] - self._from[1]) * t
            self.zoom = math.exp(math.log(self._from[2]) + (math.log(target[2]) - math.log(self._from[2])) * t)
        else:
            position_k = 1.0 - math.exp(-self.MODE_SMOOTHING * dt)
            zoom_k = 1.0 - math.exp(-self.ZOOM_SMOOTHING * dt)
            self.x += (target[0] - self.x) * position_k
            self.y += (target[1] - self.y) * position_k
            self.zoom += (target[2] - self.zoom) * zoom_k
            if abs(self.zoom - target[2]) < 0.004:
                self.zoom = target[2]  # land exactly on the level, enabling the fast path
        self._clamp()

    def _reframe(self, target: tuple[float, float, float], zoom: float) -> tuple[float, float]:
        """Convert a view centre framed for ``target``'s zoom into one framed for ``zoom``."""
        if not self.center_panels:
            return target[0], target[1]
        dx = self.screen_w / 2 - self.free_rect.centerx
        dy = self.screen_h / 2 - self.free_rect.centery
        return (target[0] + dx * (1.0 / zoom - 1.0 / target[2]),
                target[1] + dy * (1.0 / zoom - 1.0 / target[2]))

    def _clamp(self) -> None:
        r = self.renderer
        half_w = self.screen_w / 2 / self.zoom
        half_h = self.screen_h / 2 / self.zoom
        self.x = float(np.clip(self.x, r.view_x + half_w, r.view_x + r.view_w - half_w))
        self.y = float(np.clip(self.y, r.view_y + half_h, r.view_y + r.view_h - half_h))
        self.left = self.x - half_w
        self.top = self.y - half_h

    def snap_view(self, left: float, top: float, zoom: float) -> None:
        """Adopt the exact pixel-snapped view chosen by the track renderer."""
        self.left, self.top, self.view_zoom = left, top, zoom

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Screen position of a world point."""
        if self.renderer.portrait:
            x, y = -y, x
        return (x - self.left) * self.view_zoom, (y - self.top) * self.view_zoom


class Director:
    """Cuts between camera modes on a schedule, so recordings need no input at all."""

    SEQUENCE = ((CameraMode.FULL, 7.0), (CameraMode.CINEMATIC, 16.0), (CameraMode.FOLLOW, 11.0))

    def __init__(self, camera: Camera):
        self.camera = camera
        self.index = 0
        self.elapsed = 0.0

    def begin_generation(self) -> None:
        self.index = 0
        self.elapsed = 0.0
        self.camera.set_mode(self.SEQUENCE[0][0])

    def update(self, dt: float) -> None:
        self.elapsed += dt
        if self.elapsed >= self.SEQUENCE[self.index][1]:
            self.elapsed = 0.0
            self.index = (self.index + 1) % len(self.SEQUENCE)
            self.camera.set_mode(self.SEQUENCE[self.index][0])
