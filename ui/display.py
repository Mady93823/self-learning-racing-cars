"""Shows the fixed 1920x1080 canvas in a window of any size.

Everything draws to one offscreen canvas, so the layout and any recording are always
1080p regardless of the monitor. Showing it uses, in order of preference:

* GPU: a separate SDL window and renderer, which scale the canvas with no CPU cost.
  pygame's own SCALED flag can't do this on small screens, because it refuses to make
  the window smaller than the canvas, leaving it hanging off a 1366x768 panel.
* Software: a plain window, scaling on the CPU when the screen is too small.
* None: with SDL's dummy driver the canvas is simply never shown (``--offscreen``).
"""

from __future__ import annotations

import os

import pygame


class Display:
    TITLE = "AI Racing Lab"

    def __init__(self, size: tuple[int, int], fullscreen: bool = False, window: tuple[int, int] | None = None):
        self.size = size
        self.is_fullscreen = False
        self._window = None
        self._renderer = None
        self._texture = None
        self._surface: pygame.Surface | None = None
        self.mode = "offscreen"

        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            self.canvas = pygame.display.set_mode(size)
            return

        os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "best")
        window_size = window or self._fit_to_desktop()
        try:
            self._open_gpu(window_size)
            self.mode = "gpu"
        except (pygame.error, ImportError, AttributeError) as error:
            print(f"GPU display unavailable ({error}); using software scaling.")
            self._open_software(window_size)
            self.mode = "software"
        if fullscreen:
            self.toggle_fullscreen()

    def _fit_to_desktop(self) -> tuple[int, int]:
        try:
            desktop = pygame.display.get_desktop_sizes()[0]
        except (pygame.error, IndexError):
            return self.size
        # Leave room for the title bar and the desktop's own panels.
        factor = min(1.0, (desktop[0] - 40) / self.size[0], (desktop[1] - 96) / self.size[1])
        return int(self.size[0] * factor), int(self.size[1] * factor)

    def _open_gpu(self, window_size: tuple[int, int]) -> None:
        from pygame._sdl2.video import Renderer, Texture, Window

        # convert() needs a display mode even though this hidden one is never shown.
        pygame.display.set_mode((1, 1), pygame.HIDDEN)
        self.canvas = pygame.Surface(self.size).convert()
        self._window = Window(self.TITLE, size=window_size, resizable=True)
        self._renderer = Renderer(self._window)
        # Logical size scales and letterboxes automatically at any window shape.
        self._renderer.logical_size = self.size
        self._texture = Texture(self._renderer, self.size, streaming=True)
        self._window.focus()

    def _open_software(self, window_size: tuple[int, int]) -> None:
        pygame.display.set_caption(self.TITLE)
        if window_size == self.size:
            self._surface = pygame.display.set_mode(self.size)
            self.canvas = self._surface
        else:
            self._surface = pygame.display.set_mode(window_size)
            self.canvas = pygame.Surface(self.size).convert()

    def present(self) -> None:
        if self._renderer is not None:
            self._texture.update(self.canvas)
            self._renderer.clear()
            self._renderer.blit(self._texture)
            self._renderer.present()
        elif self._surface is not None and self._surface is not self.canvas:
            pygame.transform.smoothscale(self.canvas, self._surface.get_size(), self._surface)
            pygame.display.flip()
        else:
            pygame.display.flip()

    def toggle_fullscreen(self) -> None:
        if self._window is not None:
            if self.is_fullscreen:
                self._window.set_windowed()
            else:
                self._window.set_fullscreen(True)
            self.is_fullscreen = not self.is_fullscreen
        elif self.mode == "software":
            pygame.display.toggle_fullscreen()
            self.is_fullscreen = not self.is_fullscreen
