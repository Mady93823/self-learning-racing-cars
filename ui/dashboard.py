"""The live dashboard: top bar, bottom metrics and the side panel."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pygame

from ai.neural_network import INPUT_LABELS, OUTPUT_LABELS, TopologyLayout
from settings import Settings
from ui.theme import (AMBER, DANGER, DATA, FAINT, GOLD, INK, MUTED, SIGNAL, Theme, format_distance,
                      format_duration, mix)


class AnimatedValue:
    """A number that glides to its target instead of snapping."""

    def __init__(self, value: float = 0.0, rate: float = 7.0):
        self.value = value
        self.target = value
        self.rate = rate

    def update(self, dt: float, target: float) -> float:
        self.target = target
        self.value += (target - self.value) * (1.0 - math.exp(-self.rate * dt))
        if abs(self.value - target) < 0.5 and abs(target) >= 10:
            self.value = target
        return self.value


@dataclass
class HudData:
    generation: int = 0
    generation_label: str = "GENERATION"
    population: int = 0
    active: int = 0
    eliminated: int = 0
    finished: int = 0
    speed_label: str = "1×"
    status: str = "TRAINING"
    camera: str = ""
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    best_distance_m: float = 0.0
    laps: int = 0
    training_seconds: float = 0.0
    race_time: float = 0.0
    race_limit: float = 90.0
    previous_best: float | None = None
    previous_average: float | None = None
    fitness_history: list = field(default_factory=list)
    average_history: list = field(default_factory=list)
    # Current best car
    genome_id: int | None = None
    species: int | None = None
    car_status: str = ""
    speed_kmh: float = 0.0
    car_laps: int = 0
    car_number: int = 1
    car_distance_m: float = 0.0
    lap_fraction: float = 0.0
    best_lap: float | None = None
    livery: tuple = SIGNAL
    readings: list = field(default_factory=list)
    steer: float = 0.0
    throttle: float = 0.0
    topology: TopologyLayout | None = None
    node_values: dict = field(default_factory=dict)


class Dashboard:
    SIDE_GAP = 10

    def __init__(self, theme: Theme, settings: Settings):
        self.theme = theme
        self.display = settings.display
        self.sensor_names = settings.sensors.names
        self.best = AnimatedValue()
        self.average = AnimatedValue()
        self.distance = AnimatedValue()
        self.active = AnimatedValue(rate=12.0)
        self.eliminated = AnimatedValue(rate=12.0)
        self.speed = AnimatedValue(rate=10.0)
        self.time = 0.0
        self._network_surface: pygame.Surface | None = None
        self._network_timer = 0.0
        self._network_genome = None
        self._readings = [AnimatedValue(1.0, rate=14.0) for _ in self.sensor_names]
        self._steer = AnimatedValue(rate=14.0)
        self._throttle = AnimatedValue(rate=14.0)

    def update(self, dt: float, data: HudData) -> None:
        self.time += dt
        self.best.update(dt, data.best_fitness)
        self.average.update(dt, data.average_fitness)
        self.distance.update(dt, data.best_distance_m)
        self.active.update(dt, data.active)
        self.eliminated.update(dt, data.eliminated)
        self.speed.update(dt, data.speed_kmh)
        for animated, reading in zip(self._readings, data.readings):
            animated.update(dt, float(reading))
        self._steer.update(dt, data.steer)
        self._throttle.update(dt, data.throttle)
        self._network_timer += dt

    # ------------------------------------------------------------------ helpers

    def _stat(self, target, x, y, label, value, color=INK, value_size=25) -> int:
        """A small caption above a value. Returns the width used."""
        theme = self.theme
        caption = theme.draw_text(target, label, (x, y), 12, MUTED, "SemiBold", tracking=2)
        number = theme.draw_text(target, value, (x, y + 12), value_size, color, "SemiBold")
        return max(caption.width, number.width)

    def _section_title(self, target, x, y, label) -> None:
        self.theme.draw_text(target, label, (x, y), 13, MUTED, "SemiBold", tracking=3)

    # ------------------------------------------------------------------ top bar

    def draw_top(self, target: pygame.Surface, data: HudData, presentation: bool) -> None:
        theme = self.theme
        width = target.get_width() - 28
        height = self.display.hud_top - 12
        target.blit(theme.panel(width, height), (14, 12))

        # A canted marker gives the wordmark its motorsport character.
        pygame.draw.polygon(target, SIGNAL, [(32, 54), (38, 26), (46, 26), (40, 54)])
        pygame.draw.polygon(target, mix(SIGNAL, (20, 20, 20), 0.45), [(44, 54), (50, 26), (56, 26), (50, 54)])
        theme.draw_text(target, "AI RACING LAB", (64, 39), 34, INK, "ExtraBold", tracking=3, anchor="midleft")

        y = 18
        status_color = {"TRAINING": DATA, "PAUSED": AMBER, "EVALUATING": DATA, "RESULTS": AMBER}.get(data.status, DATA)
        columns: list[tuple[str, str, tuple]] = [
            (data.generation_label, f"{data.generation}", INK),
            ("ACTIVE", f"{int(round(self.active.value))} / {data.population}", DATA),
            ("ELIMINATED", f"{int(round(self.eliminated.value))}", DANGER if data.eliminated else MUTED),
            ("FINISHED", f"{data.finished}", GOLD if data.finished else MUTED),
            ("SIM SPEED", data.speed_label, AMBER if data.speed_label != "1×" else INK),
        ]
        if not presentation:
            columns.append(("CAMERA", data.camera, INK))

        x = 460
        spacing = 196 if presentation else 170
        for label, value, color in columns:
            self._stat(target, x, y, label, value, color)
            x += spacing

        # The status chip sits on its own at the far right.
        chip_x = target.get_width() - 208
        self.theme.draw_text(target, "STATUS", (chip_x, y), 12, MUTED, "SemiBold", tracking=2)
        pulse = 0.55 + 0.45 * math.sin(self.time * 4.0)
        dot = mix((30, 30, 34), status_color, pulse if data.status != "PAUSED" else 1.0)
        pygame.draw.circle(target, dot, (chip_x + 7, y + 28), 5)
        theme.draw_text(target, data.status, (chip_x + 20, y + 12), 25, status_color, "SemiBold")

    # ------------------------------------------------------------------ bottom bar

    def draw_bottom(self, target: pygame.Surface, data: HudData) -> None:
        theme = self.theme
        x0 = 14
        width = target.get_width() - self.display.panel_width - 28 - self.SIDE_GAP
        height = self.display.hud_bottom - 14
        y0 = target.get_height() - height - 14
        target.blit(theme.panel(width, height), (x0, y0))

        columns = [
            ("BEST FITNESS", f"{self.best.value:,.0f}", SIGNAL, self._delta(data.best_fitness, data.previous_best)),
            ("AVERAGE FITNESS", f"{self.average.value:,.0f}", DATA, self._delta(data.average_fitness, data.previous_average)),
            ("BEST DISTANCE", format_distance(self.distance.value), INK, ""),
            ("COMPLETED LAPS", f"{data.laps}", INK, ""),
            ("TRAINING TIME", format_duration(data.training_seconds), INK, ""),
        ]
        column = width / len(columns)
        for i, (label, value, color, note) in enumerate(columns):
            x = int(x0 + 22 + column * i)
            if i:
                pygame.draw.line(target, (56, 60, 68), (x - 16, y0 + 16), (x - 16, y0 + height - 16))
            theme.draw_text(target, label, (x, y0 + 9), 13, MUTED, "SemiBold", tracking=2)
            rect = theme.draw_text(target, value, (x, y0 + 22), 36, color, "Bold")
            if note:
                theme.draw_text(target, note, (rect.right + 10, rect.centery - 10), 14, MUTED, "Medium")

    @staticmethod
    def _delta(value: float, previous: float | None) -> str:
        return "" if previous is None else f"prev {previous:,.0f}"

    # ------------------------------------------------------------------ side panel

    def draw_side(self, target: pygame.Surface, data: HudData, presentation: bool) -> None:
        width = self.display.panel_width
        x = target.get_width() - width - 14
        y = self.display.hud_top + self.SIDE_GAP
        bottom = target.get_height() - 14
        available = bottom - y

        best_h = 144
        if presentation:
            progress_h = 300
            network_h = available - best_h - progress_h - 2 * self.SIDE_GAP
            sensor_h = 0
        else:
            sensor_h = 214
            progress_h = 216
            network_h = available - best_h - sensor_h - progress_h - 3 * self.SIDE_GAP

        self._draw_best_card(target, x, y, width, best_h, data)
        y += best_h + self.SIDE_GAP
        self._draw_network(target, x, y, width, network_h, data)
        y += network_h + self.SIDE_GAP
        if sensor_h:
            self._draw_sensors(target, x, y, width, sensor_h, data)
            y += sensor_h + self.SIDE_GAP
        self._draw_progress(target, x, y, width, progress_h, data)

    def _draw_best_card(self, target, x, y, width, height, data: HudData) -> None:
        theme = self.theme
        target.blit(theme.panel(width, height), (x, y))
        self._section_title(target, x + 18, y + 11, "CURRENT BEST AI")
        if data.genome_id is None:
            theme.draw_text(target, "Waiting for the green light", (x + 18, y + 44), 22, MUTED)
            return

        pygame.draw.rect(target, data.livery, (x + 18, y + 38, 6, 42), border_radius=3)
        theme.draw_text(target, f"#{data.genome_id}", (x + 32, y + 30), 38, INK, "Bold")
        species = f"SPECIES {data.species}" if data.species else data.car_status.upper()
        theme.draw_text(target, species, (x + 32, y + 72), 14, MUTED, "SemiBold", tracking=2)

        theme.draw_text(target, f"{self.speed.value:.0f}", (x + width - 66, y + 30), 38, INK, "Bold", anchor="topright")
        theme.draw_text(target, "KM/H", (x + width - 18, y + 46), 15, MUTED, "SemiBold", tracking=1, anchor="topright")
        best_lap = f"BEST LAP {data.best_lap:.2f}s" if data.best_lap else "NO LAP YET"
        theme.draw_text(target, best_lap, (x + width - 18, y + 72), 14, MUTED, "SemiBold", tracking=1, anchor="topright")

        # Lap progress, drawn as a gate-marked bar.
        bar = pygame.Rect(x + 18, y + 106, width - 36, 7)
        pygame.draw.rect(target, (38, 41, 47), bar, border_radius=3)
        filled = bar.copy()
        filled.width = bar.width if data.car_status == "finished" else int(bar.width * data.lap_fraction)
        if filled.width > 3:
            pygame.draw.rect(target, SIGNAL, filled, border_radius=3)
        if data.car_status == "finished":
            theme.draw_text(target, f"FINISHED  ·  {data.car_laps} LAPS", (x + 18, y + 118), 14, GOLD, "SemiBold", tracking=1)
            return
        theme.draw_text(target, f"LAP {data.car_laps + 1}", (x + 18, y + 118), 14, MUTED, "SemiBold", tracking=1)
        theme.draw_text(target, f"{data.lap_fraction * 100:.0f}%", (x + width - 18, y + 118), 14, MUTED,
                        "SemiBold", anchor="topright")

    def _draw_network(self, target, x, y, width, height, data: HudData) -> None:
        theme = self.theme
        target.blit(theme.panel(width, height), (x, y))
        self._section_title(target, x + 18, y + 11, "LIVE NEURAL NETWORK")
        layout = data.topology
        if layout is None:
            return
        summary = f"{len(layout.nodes)} nodes · {layout.connection_count} links"
        theme.draw_text(target, summary, (x + width - 18, y + 9), 15, MUTED, "Medium", anchor="topright")

        # Redrawing the diagram at about 20 Hz is plenty, and much cheaper than per frame.
        area = pygame.Rect(x + 62, y + 40, width - 136, height - 56)
        if self._network_surface is None or self._network_surface.get_size() != (width, height):
            self._network_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            self._network_timer = 1.0
        if self._network_timer >= 0.05 or self._network_genome != data.genome_id:
            self._network_timer = 0.0
            self._network_genome = data.genome_id
            self._network_surface.fill((0, 0, 0, 0))
            local = area.move(-x, -y)
            self._paint_network(self._network_surface, local, layout, data.node_values)
        target.blit(self._network_surface, (x, y))

    def _paint_network(self, surface, area: pygame.Rect, layout: TopologyLayout, values: dict) -> None:
        theme = self.theme

        def place(key: int) -> tuple[float, float]:
            nx, ny = layout.nodes[key]
            return area.x + nx * area.width, area.y + ny * area.height

        for source, target_key, weight in layout.connections:
            if source not in layout.nodes or target_key not in layout.nodes:
                continue
            signal = values.get(source, 0.0) * weight
            strength = min(1.0, abs(signal) * 0.8)
            hue = SIGNAL if weight < 0 else DATA
            color = mix((44, 47, 54), hue, 0.18 + 0.82 * strength)
            pygame.draw.aaline(surface, color, place(source), place(target_key))

        input_index = output_index = 0
        for key, kind in layout.node_kinds.items():
            px, py = place(key)
            value = float(values.get(key, 0.0))
            if kind == "input":
                level = min(1.0, abs(value))
                color = mix((52, 56, 64), DATA, level)
                pygame.draw.circle(surface, color, (px, py), 5)
                label = INPUT_LABELS[input_index] if input_index < len(INPUT_LABELS) else ""
                input_index += 1
                theme.draw_text(surface, label, (px - 11, py), 11, MUTED, "Medium", anchor="midright")
            elif kind == "output":
                level = min(1.0, abs(value))
                color = mix((60, 44, 40), SIGNAL, level)
                pygame.draw.circle(surface, color, (px, py), 8)
                pygame.draw.circle(surface, (235, 238, 242), (px, py), 8, 1)
                label = OUTPUT_LABELS[output_index] if output_index < len(OUTPUT_LABELS) else ""
                output_index += 1
                theme.draw_text(surface, label, (px + 13, py - 8), 11, INK, "SemiBold")
                theme.draw_text(surface, f"{value:+.2f}", (px + 13, py + 3), 12, MUTED, "Medium")
            else:
                level = min(1.0, abs(value))
                pygame.draw.circle(surface, mix((58, 54, 72), AMBER, level), (px, py), 6)

    def _draw_sensors(self, target, x, y, width, height, data: HudData) -> None:
        theme = self.theme
        target.blit(theme.panel(width, height), (x, y))
        self._section_title(target, x + 18, y + 11, "SENSORS")
        row_y = y + 34
        for name, animated in zip(self.sensor_names, self._readings):
            value = max(0.0, min(1.0, animated.value))
            theme.draw_text(target, name, (x + 18, row_y - 2), 13, MUTED, "Medium")
            bar = pygame.Rect(x + 86, row_y + 2, width - 148, 7)
            pygame.draw.rect(target, (38, 41, 47), bar, border_radius=3)
            fill = bar.copy()
            fill.width = max(2, int(bar.width * value))
            color = DANGER if value < 0.15 else (AMBER if value < 0.32 else DATA)
            pygame.draw.rect(target, color, fill, border_radius=3)
            theme.draw_text(target, f"{value * 52:.0f} m", (x + width - 18, row_y - 3), 14, INK, "Medium", anchor="topright")
            row_y += 19

        # Steering and pedals: the network's outputs, as the actuators follow them.
        row_y += 6
        pygame.draw.line(target, (48, 52, 60), (x + 18, row_y - 4), (x + width - 18, row_y - 4))
        half = (width - 36 - 14) // 2
        self._centre_bar(target, x + 18, row_y + 18, half, self._steer.value, "STEER", DATA, DATA)
        self._centre_bar(target, x + 18 + half + 14, row_y + 18, half, self._throttle.value, "THROTTLE", DATA, DANGER)

    def _centre_bar(self, target, x, y, width, value, label, positive, negative) -> None:
        self.theme.draw_text(target, label, (x, y - 18), 12, MUTED, "SemiBold", tracking=1)
        bar = pygame.Rect(x, y, width, 6)
        pygame.draw.rect(target, (38, 41, 47), bar, border_radius=3)
        middle = bar.centerx
        extent = int(max(-1.0, min(1.0, value)) * bar.width / 2)
        rect = pygame.Rect(min(middle, middle + extent), y, abs(extent), 6)
        if rect.width:
            pygame.draw.rect(target, positive if value >= 0 else negative, rect, border_radius=3)
        pygame.draw.line(target, (120, 126, 138), (middle, y - 2), (middle, y + 7))

    def _draw_progress(self, target, x, y, width, height, data: HudData) -> None:
        theme = self.theme
        target.blit(theme.panel(width, height), (x, y))
        self._section_title(target, x + 18, y + 11, "TRAINING PROGRESS")

        previous = f"{data.previous_best:,.0f}" if data.previous_best is not None else "—"
        theme.draw_text(target, "PREV GEN BEST", (x + 18, y + 34), 12, MUTED, "SemiBold", tracking=1)
        theme.draw_text(target, previous, (x + 18, y + 45), 25, INK, "SemiBold")
        theme.draw_text(target, "THIS RACE", (x + width - 18, y + 34), 12, MUTED, "SemiBold", tracking=1, anchor="topright")
        theme.draw_text(target, f"{data.race_time:0.1f}s", (x + width - 18, y + 45), 25, INK, "SemiBold", anchor="topright")

        bar = pygame.Rect(x + 18, y + 80, width - 36, 4)
        pygame.draw.rect(target, (38, 41, 47), bar, border_radius=2)
        fill = bar.copy()
        fill.width = int(bar.width * min(1.0, data.race_time / max(data.race_limit, 1.0)))
        if fill.width > 1:
            pygame.draw.rect(target, (126, 132, 144), fill, border_radius=2)

        chart = pygame.Rect(x + 18, y + 94, width - 36, height - 118)
        hint = "Builds as generations finish" if data.generation_label == "GENERATION" else "No training in this run"
        self._chart(target, chart, data.fitness_history, data.average_history, hint)
        legend_y = y + height - 21
        pygame.draw.line(target, SIGNAL, (x + 18, legend_y + 7), (x + 34, legend_y + 7), 2)
        theme.draw_text(target, "BEST", (x + 39, legend_y), 12, MUTED, "SemiBold", tracking=1)
        pygame.draw.line(target, DATA, (x + 84, legend_y + 7), (x + 100, legend_y + 7), 2)
        theme.draw_text(target, "AVERAGE", (x + 105, legend_y), 12, MUTED, "SemiBold", tracking=1)

    def _chart(self, target, rect: pygame.Rect, best: list, average: list, hint: str) -> None:
        for fraction in (0.0, 0.5, 1.0):
            line_y = rect.y + rect.height * fraction
            pygame.draw.line(target, (40, 43, 50), (rect.x, line_y), (rect.right, line_y))
        if len(best) < 2:
            self.theme.draw_text(target, hint, rect.center, 14, FAINT, anchor="center")
            return
        peak = max(max(best), 1.0)
        self.theme.draw_text(target, f"{peak:,.0f}", (rect.right, rect.y + 2), 12, FAINT, anchor="topright")

        def points(series):
            step = rect.width / max(len(series) - 1, 1)
            return [(rect.x + i * step, rect.bottom - min(v, peak) / peak * rect.height) for i, v in enumerate(series)]

        pygame.draw.aalines(target, DATA, False, points(average))
        pygame.draw.aalines(target, SIGNAL, False, points(best))
        end = points(best)[-1]
        pygame.draw.circle(target, SIGNAL, (int(end[0]), int(end[1])), 3)
