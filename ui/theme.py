"""The visual design system: colours, type scale, cached text and panels."""

from __future__ import annotations

import pygame

from settings import FONT_DIR

# Palette: graphite surfaces, one hot signal colour, one cool data colour.
INK = (238, 240, 244)
MUTED = (146, 154, 167)
FAINT = (92, 99, 111)
PANEL = (13, 15, 19)
HAIRLINE = (255, 255, 255)
SIGNAL = (255, 84, 46)      # primary accent: branding, best fitness, the leading car
DATA = (58, 214, 190)       # secondary accent: averages, healthy values
AMBER = (255, 190, 72)      # warnings, slow motion, lap celebrations
DANGER = (236, 66, 76)      # eliminations
GOLD = (250, 212, 110)

PANEL_ALPHA = 218
RADIUS = 12

WEIGHTS = ("Light", "Regular", "Medium", "SemiBold", "Bold", "ExtraBold", "Black")


class Theme:
    """Loads fonts once and caches rendered text, since text rendering is the main HUD cost."""

    def __init__(self) -> None:
        self._fonts: dict[tuple[str, int], pygame.font.Font] = {}
        self._text: dict[tuple, pygame.Surface] = {}
        self._panels: dict[tuple, pygame.Surface] = {}

    def font(self, size: int, weight: str = "Medium") -> pygame.font.Font:
        key = (weight, size)
        if key not in self._fonts:
            path = FONT_DIR / f"SairaCondensed-{weight}.ttf"
            # Fall back to pygame's built-in font if the assets folder was not copied.
            self._fonts[key] = pygame.font.Font(str(path) if path.exists() else None, size)
        return self._fonts[key]

    def text(self, value: str, size: int, color=INK, weight: str = "Medium",
             tracking: int = 0) -> pygame.Surface:
        key = (value, size, color, weight, tracking)
        cached = self._text.get(key)
        if cached is not None:
            return cached
        font = self.font(size, weight)
        if tracking == 0 or not value:
            surface = font.render(value, True, color)
        else:
            # Letterspaced labels read as premium signage; pygame has no native tracking.
            glyphs = [font.render(ch, True, color) for ch in value]
            width = sum(g.get_width() for g in glyphs) + tracking * (len(glyphs) - 1)
            surface = pygame.Surface((max(width, 1), font.get_height()), pygame.SRCALPHA)
            x = 0
            for glyph in glyphs:
                surface.blit(glyph, (x, 0))
                x += glyph.get_width() + tracking
        if len(self._text) > 2400:
            self._text.clear()
        self._text[key] = surface
        return surface

    def draw_text(self, target: pygame.Surface, value: str, pos, size: int, color=INK,
                  weight: str = "Medium", tracking: int = 0, anchor: str = "topleft") -> pygame.Rect:
        surface = self.text(value, size, color, weight, tracking)
        rect = surface.get_rect(**{anchor: pos})
        target.blit(surface, rect)
        return rect

    def panel(self, width: int, height: int, alpha: int = PANEL_ALPHA, radius: int = RADIUS,
              color=PANEL, border: int = 22) -> pygame.Surface:
        """Antialiased rounded panel with a hairline border and a faint top sheen."""
        key = (width, height, alpha, radius, color, border)
        cached = self._panels.get(key)
        if cached is not None:
            return cached
        scale = 3
        big = pygame.Surface((width * scale, height * scale), pygame.SRCALPHA)
        rect = big.get_rect()
        pygame.draw.rect(big, (*color, alpha), rect, border_radius=radius * scale)
        if border:
            pygame.draw.rect(big, (*HAIRLINE, border), rect, width=scale, border_radius=radius * scale)
        surface = pygame.transform.smoothscale(big, (width, height))
        self._panels[key] = surface
        return surface


def mix(a, b, t: float):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def format_distance(metres: float) -> str:
    return f"{metres / 1000.0:.2f} km" if metres >= 1000.0 else f"{metres:.0f} m"
