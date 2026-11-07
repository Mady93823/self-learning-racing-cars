"""Full-screen moments: loading, title, generation intro, results, banners, pause."""

from __future__ import annotations

import math

import pygame

from ui.theme import (AMBER, DANGER, DATA, GOLD, INK, MUTED, PANEL, SIGNAL, Theme, ease_in_out, ease_out,
                      format_distance, format_duration, mix)

CONTROL_LIST = (
    ("SPACE", "Pause / resume"), ("F", "Fast-forward"), ("S", "Slow motion"), ("C", "Switch camera"),
    ("V", "Sensor rays"), ("H", "Toggle HUD"), ("P", "Screenshot"), ("R", "Restart training"),
    ("F11", "Fullscreen"), ("ESC", "Exit"),
)


def _dim(target: pygame.Surface, alpha: int, cache: dict = {}) -> None:
    """Darken the whole frame. The overlay is reused to avoid allocating every frame."""
    size = target.get_size()
    overlay = cache.get(size)
    if overlay is None:
        overlay = pygame.Surface(size)
        overlay.fill((6, 7, 9))
        cache[size] = overlay
    overlay.set_alpha(max(0, min(255, alpha)))
    target.blit(overlay, (0, 0))


def _wordmark(target, theme: Theme, center, size: int, alpha: float = 1.0) -> pygame.Rect:
    """The product mark: two canted bars beside the name."""
    label = theme.text("AI RACING LAB", size, INK, "Black", tracking=max(2, size // 22))
    bar_h = int(size * 0.78)
    slant = int(bar_h * 0.22)
    bars = int(size * 0.5)
    total = label.get_width() + bars + 12
    x = center[0] - total // 2
    y = center[1]
    tint = mix((10, 10, 12), SIGNAL, alpha)
    tint2 = mix((10, 10, 12), mix(SIGNAL, (20, 20, 20), 0.45), alpha)
    bar_w = max(4, size // 7)
    pygame.draw.polygon(target, tint, [(x, y + bar_h // 2), (x + slant, y - bar_h // 2),
                                       (x + slant + bar_w, y - bar_h // 2), (x + bar_w, y + bar_h // 2)])
    second = x + bar_w + bar_w // 2
    pygame.draw.polygon(target, tint2, [(second, y + bar_h // 2), (second + slant, y - bar_h // 2),
                                        (second + slant + bar_w, y - bar_h // 2), (second + bar_w, y + bar_h // 2)])
    rect = label.get_rect(midleft=(x + bars + 12, y))
    if alpha < 1.0:
        faded = label.copy()
        faded.set_alpha(int(alpha * 255))
        target.blit(faded, rect)
    else:
        target.blit(label, rect)
    return rect


def draw_loading(target: pygame.Surface, theme: Theme, fraction: float, label: str) -> None:
    target.fill((14, 16, 20))
    width, height = target.get_size()
    _wordmark(target, theme, (width // 2, height // 2 - 40), 72)
    bar = pygame.Rect(width // 2 - 240, height // 2 + 30, 480, 4)
    pygame.draw.rect(target, (44, 47, 54), bar, border_radius=2)
    fill = bar.copy()
    fill.width = int(bar.width * max(0.0, min(1.0, fraction)))
    if fill.width > 2:
        pygame.draw.rect(target, SIGNAL, fill, border_radius=2)
    theme.draw_text(target, label, (width // 2, height // 2 + 50), 20, MUTED, "Medium", tracking=1, anchor="midtop")


def draw_title(target: pygame.Surface, theme: Theme, time: float, prompt: bool, tagline: str) -> None:
    """Opening title, over the circuit that the caller has already drawn."""
    width, height = target.get_size()
    _dim(target, 150)
    reveal = ease_out(time / 0.9)
    y = int(height / 2 - 70 + (1.0 - reveal) * 26)
    _wordmark(target, theme, (width // 2, y), 132, reveal)
    tag_reveal = ease_out((time - 0.5) / 0.8)
    if tag_reveal > 0:
        tag = theme.text(tagline, 34, INK, "Light", tracking=2)
        faded = tag.copy()
        faded.set_alpha(int(tag_reveal * 235))
        target.blit(faded, faded.get_rect(center=(width // 2, y + 96)))
    line = ease_in_out((time - 0.2) / 1.0)
    half = int(line * 300)
    if half > 0:
        pygame.draw.line(target, SIGNAL, (width // 2 - half, y + 66), (width // 2 + half, y + 66), 2)
    if prompt and time > 1.0:
        pulse = 0.5 + 0.5 * math.sin(time * 3.0)
        theme.draw_text(target, "PRESS ENTER TO START TRAINING", (width // 2, y + 180), 24,
                        mix(MUTED, INK, pulse), "SemiBold", tracking=4, anchor="center")
        _draw_controls_row(target, theme, height - 58)


def _draw_controls_row(target, theme: Theme, y: int) -> None:
    items = [theme.text(f"{key}  ", 16, INK, "Bold") for key, _ in CONTROL_LIST]
    names = [theme.text(name, 16, MUTED, "Medium") for _, name in CONTROL_LIST]
    total = sum(a.get_width() + b.get_width() + 26 for a, b in zip(items, names)) - 26
    x = (target.get_width() - total) // 2
    for key, name in zip(items, names):
        target.blit(key, (x, y))
        x += key.get_width()
        target.blit(name, (x, y))
        x += name.get_width() + 26


def draw_generation_intro(target: pygame.Surface, theme: Theme, t: float, duration: float, generation: int,
                          previous_best: float | None, previous_laps: int | None, population: int,
                          evaluating: bool = False) -> None:
    """A diagonal band sweeps in with the generation number, then the start lights run."""
    width, height = target.get_size()
    enter = ease_out(t / 0.45)
    leave = ease_in_out((t - (duration - 0.35)) / 0.35)
    band_h = 208
    center_y = height // 2 - 20
    offset = int((1.0 - enter) * -width + leave * width * 1.1)

    slant = 70
    panel_points = [(offset - slant, center_y + band_h // 2), (offset + slant, center_y - band_h // 2),
                    (offset + width + slant + 200, center_y - band_h // 2),
                    (offset + width - slant + 200, center_y + band_h // 2)]
    band = pygame.Surface((width, band_h + 4), pygame.SRCALPHA)
    local = [(px, py - (center_y - band_h // 2) + 2) for px, py in panel_points]
    pygame.draw.polygon(band, (*PANEL, 236), local)
    target.blit(band, (0, center_y - band_h // 2 - 2))
    accent = [(offset - slant, center_y + band_h // 2 + 6), (offset - slant + 3, center_y + band_h // 2),
              (offset + width + 200, center_y + band_h // 2), (offset + width + 197, center_y + band_h // 2 + 6)]
    pygame.draw.polygon(target, SIGNAL, accent)

    left = width // 2 + offset
    heading = "EVALUATION RUN" if evaluating else f"GENERATION {generation}"
    theme.draw_text(target, heading, (left, center_y - 34), 92, INK, "Black", tracking=4, anchor="center")
    if not evaluating:
        if previous_best is None:
            note = f"First attempt: {population} random neural networks"
        else:
            laps = f"  ·  {previous_laps} laps" if previous_laps else ""
            note = f"Previous best {previous_best:,.0f}{laps}"
        theme.draw_text(target, note, (left, center_y + 28), 30, MUTED, "Medium", tracking=1, anchor="center")

    # Five start lights come on one by one, then go out together for the race to start.
    lights_y = center_y + 74
    lights_start = 0.45
    span = max(duration - lights_start - 0.35, 0.2)
    lit = int((t - lights_start) / span * 5) + 1 if t > lights_start else 0
    for i in range(5):
        lx = left + (i - 2) * 46
        pygame.draw.circle(target, (24, 26, 30), (lx, lights_y), 15)
        on = i < lit and t < duration - 0.35
        pygame.draw.circle(target, (240, 44, 44) if on else (54, 26, 28), (lx, lights_y), 11)
        if on:
            pygame.draw.circle(target, (255, 170, 160), (lx - 3, lights_y - 3), 3)


def draw_results(target: pygame.Surface, theme: Theme, t: float, result, previous, liveries: list,
                 summary_line: str | None = None) -> None:
    """End-of-generation card: headline numbers, deltas and the top five."""
    width, height = target.get_size()
    _dim(target, int(170 * ease_out(t / 0.3)))
    slide = ease_out(t / 0.45)
    card_w, card_h = 1120, 700
    x = (width - card_w) // 2
    y = (height - card_h) // 2 + int((1.0 - slide) * 44)
    target.blit(theme.panel(card_w, card_h, alpha=242, radius=16), (x, y))

    theme.draw_text(target, "RESULTS", (x + 44, y + 28), 20, SIGNAL, "Bold", tracking=5)
    theme.draw_text(target, f"GENERATION {result.generation}", (x + 44, y + 48), 62, INK, "Black", tracking=2)
    summary = summary_line or f"{result.species} species  ·  {result.population} cars  ·  race lasted {result.sim_seconds:.0f}s"
    theme.draw_text(target, summary, (x + card_w - 44, y + 74), 22, MUTED, "Medium", anchor="topright")

    # Headline metrics count up as the card arrives.
    count = ease_out((t - 0.15) / 0.8)
    def delta(now, then, fmt="{:+,.0f}"):
        if then is None:
            return None
        return fmt.format(now - then), (now - then) >= 0

    metrics = [
        ("BEST FITNESS", f"{result.best_fitness * count:,.0f}", SIGNAL,
         delta(result.best_fitness, previous.best_fitness if previous else None)),
        ("AVERAGE FITNESS", f"{result.average_fitness * count:,.0f}", DATA,
         delta(result.average_fitness, previous.average_fitness if previous else None)),
        ("BEST DISTANCE", format_distance(result.best_distance_m * count), INK,
         delta(result.best_distance_m, previous.best_distance_m if previous else None, "{:+,.0f} m")),
        ("BEST LAP", f"{result.best_lap_seconds:.2f}s" if result.best_lap_seconds else "—", GOLD, None),
    ]
    column = (card_w - 88) / len(metrics)
    for i, (label, value, color, change) in enumerate(metrics):
        mx = int(x + 44 + i * column)
        theme.draw_text(target, label, (mx, y + 140), 15, MUTED, "SemiBold", tracking=2)
        theme.draw_text(target, value, (mx, y + 158), 52, color, "Bold")
        if change is not None:
            text, good = change
            theme.draw_text(target, text, (mx, y + 222), 19, DATA if good else DANGER, "SemiBold")

    # What happened to the field, as a single proportional bar.
    bar = pygame.Rect(x + 44, y + 266, card_w - 88, 12)
    population = max(result.population, 1)
    running = population - result.crashed - result.timed_out - result.wrong_way
    segments = [(running, DATA, "finished or still running"), (result.crashed, DANGER, "crashed"),
                (result.timed_out + result.wrong_way, AMBER, "timed out or wrong way")]
    grow = ease_in_out((t - 0.2) / 0.6)
    bx = bar.x
    for value, color, _ in segments:
        segment = int(bar.width * value / population * grow)
        if segment > 0:
            pygame.draw.rect(target, color, (bx, bar.y, max(segment - 2, 1), bar.height), border_radius=3)
        bx += segment
    lx = bar.x
    for value, color, name in segments:
        pygame.draw.circle(target, color, (lx + 6, y + 299), 5)
        rect = theme.draw_text(target, f"{value} {name}", (lx + 18, y + 288), 19, MUTED, "Medium")
        lx = rect.right + 28

    # Top five.
    table_y = y + 336
    headers = (("POS", 44), ("AI", 118), ("SPECIES", 356), ("LAPS", 516), ("DISTANCE", 646), ("OUTCOME", 806))
    for label, offset_x in headers:
        theme.draw_text(target, label, (x + offset_x, table_y), 14, MUTED, "SemiBold", tracking=2)
    theme.draw_text(target, "FITNESS", (x + card_w - 44, table_y), 14, MUTED, "SemiBold", tracking=2, anchor="topright")
    pygame.draw.line(target, (54, 58, 66), (x + 44, table_y + 24), (x + card_w - 44, table_y + 24))

    for i, row in enumerate(result.leaderboard):
        appear = ease_out((t - 0.3 - i * 0.09) / 0.35)
        if appear <= 0:
            continue
        row_y = table_y + 34 + i * 62 + int((1.0 - appear) * 16)
        color = INK if i else GOLD
        theme.draw_text(target, f"{row['place']}", (x + 44, row_y), 38, color, "Bold")
        pygame.draw.rect(target, liveries[row["car"] % len(liveries)], (x + 118, row_y + 8, 6, 34), border_radius=3)
        theme.draw_text(target, f"#{row['genome']}", (x + 134, row_y + 2), 32, INK, "SemiBold")
        theme.draw_text(target, f"{row['species']}", (x + 356, row_y + 2), 32, MUTED, "Medium")
        theme.draw_text(target, f"{row['laps']}", (x + 516, row_y + 2), 32, INK, "Medium")
        theme.draw_text(target, format_distance(row["distance_m"]), (x + 646, row_y + 2), 32, INK, "Medium")
        status_color = DANGER if row["status"] == "crashed" else (DATA if row["status"] in ("finished", "running") else AMBER)
        theme.draw_text(target, row["status"].upper(), (x + 806, row_y + 8), 22, status_color, "SemiBold", tracking=1)
        theme.draw_text(target, f"{row['fitness']:,.0f}", (x + card_w - 44, row_y + 2), 32, INK, "SemiBold", anchor="topright")


class Banner:
    """Timed message that slides down from under the top bar."""

    def __init__(self, title: str, subtitle: str, color=AMBER, life: float = 2.6):
        self.title = title
        self.subtitle = subtitle
        self.color = color
        self.life = life
        self.age = 0.0

    def update(self, dt: float) -> None:
        self.age += dt

    @property
    def alive(self) -> bool:
        return self.age < self.life

    def draw(self, target: pygame.Surface, theme: Theme, center_x: int, y: int) -> None:
        enter = ease_out(self.age / 0.3)
        leave = ease_in_out((self.age - (self.life - 0.3)) / 0.3)
        offset = (1.0 - enter) * -40.0 - leave * 40.0
        alpha = int(255 * enter * (1.0 - leave))
        if alpha <= 0:
            return
        title = theme.text(self.title, 44, INK, "Black", tracking=3)
        subtitle = theme.text(self.subtitle, 24, self.color, "SemiBold", tracking=1)
        width = max(title.get_width(), subtitle.get_width()) + 90
        height = 96
        layer = pygame.Surface((width, height), pygame.SRCALPHA)
        layer.blit(theme.panel(width, height, alpha=232), (0, 0))
        pygame.draw.rect(layer, self.color, (0, 18, 5, height - 36), border_radius=2)
        layer.blit(title, title.get_rect(midtop=(width // 2, 6)))
        layer.blit(subtitle, subtitle.get_rect(midtop=(width // 2, 58)))
        layer.set_alpha(alpha)
        target.blit(layer, layer.get_rect(midtop=(center_x, int(y + offset))))


def draw_pause(target: pygame.Surface, theme: Theme, free_center: tuple[int, int]) -> None:
    """Pause card listing the controls. The simulation is frozen beneath it."""
    _dim(target, 90)
    width, height = 520, 520
    x = free_center[0] - width // 2
    y = free_center[1] - height // 2
    target.blit(theme.panel(width, height, alpha=240, radius=16), (x, y))
    theme.draw_text(target, "PAUSED", (x + width // 2, y + 22), 58, INK, "Black", tracking=5, anchor="midtop")
    row_y = y + 104
    for key, name in CONTROL_LIST:
        theme.draw_text(target, key, (x + 150, row_y), 24, SIGNAL, "Bold", anchor="topright")
        theme.draw_text(target, name, (x + 176, row_y), 24, INK, "Medium")
        row_y += 39


def draw_end_card(target: pygame.Surface, theme: Theme, t: float, generations: int, best_fitness: float,
                  best_distance: float, best_lap: float, seconds: float) -> None:
    width, height = target.get_size()
    _dim(target, int(200 * ease_out(t / 0.5)))
    reveal = ease_out(t / 0.8)
    y = height // 2 - 170 + int((1.0 - reveal) * 22)
    _wordmark(target, theme, (width // 2, y), 96, reveal)
    theme.draw_text(target, "No driving rules were programmed. Every lap was learned.", (width // 2, y + 80), 32, MUTED,
                    "Medium", anchor="center")
    columns = [("GENERATIONS", f"{generations}"), ("BEST FITNESS", f"{best_fitness:,.0f}"),
               ("BEST DISTANCE", format_distance(best_distance)),
               ("BEST LAP", f"{best_lap:.2f}s" if best_lap else "—"), ("TRAINING TIME", format_duration(seconds))]
    column = 300
    start = width // 2 - column * (len(columns) - 1) // 2
    for i, (label, value) in enumerate(columns):
        appear = ease_out((t - 0.4 - i * 0.12) / 0.5)
        if appear <= 0:
            continue
        cx = start + i * column
        theme.draw_text(target, label, (cx, y + 160), 18, MUTED, "SemiBold", tracking=3, anchor="center")
        theme.draw_text(target, value, (cx, y + 208), 60, INK if i else SIGNAL, "Bold", anchor="center")
