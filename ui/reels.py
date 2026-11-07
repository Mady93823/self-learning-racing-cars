"""The vertical-video presentation: one row of number cards, and a clear view of the race.

Instagram draws its interface over a Reel: status text at the very top, the account and
caption across the bottom, and buttons up the right edge. So every card sits in one slim
row just below the top zone, with short messages directly beneath it. Nothing is ever
drawn over the middle of the screen, where the camera keeps the action.
"""

from __future__ import annotations

import pygame

from ui.dashboard import AnimatedValue, HudData
from ui.theme import AMBER, DANGER, DATA, GOLD, INK, MUTED, SIGNAL, Theme, ease_in_out, ease_out, format_distance, mix

MARGIN = 65                 # keeps cards off the screen edges
ROW_Y = 262                 # just below the zone Instagram uses at the top
ROW_WIDTH = 950
CARD_HEIGHT = 132
CARD_GAP = 12
BAR_Y = ROW_Y + CARD_HEIGHT + 10
MESSAGE_Y = BAR_Y + 22
CENTER_X = MARGIN + ROW_WIDTH // 2
END_CARD_Y = 400


class Message:
    def __init__(self, text: str, color, life: float):
        self.text = text
        self.color = color
        self.life = life
        self.age = 0.0


class ReelsOverlay:
    def __init__(self, theme: Theme, lap_metres: float):
        self.theme = theme
        self.lap_metres = lap_metres
        self.message: Message | None = None
        self.hook_age: float | None = None
        self.record_m = 0.0
        self.generation_pop = 99.0
        self.alive = AnimatedValue(rate=16.0)
        self.distance = AnimatedValue(rate=10.0)
        self._last_alive = 0
        self._drop_age = 99.0

    def begin_generation(self, generation: int, record_m: float, population: int, opening: bool) -> None:
        self.record_m = record_m
        self.generation_pop = 0.0
        self.alive.value = float(population)
        self._last_alive = population
        self.distance.value = 0.0
        self.message = None
        if opening:
            self.hook_age = 0.0

    def caption(self, text: str, color=GOLD, life: float = 1.4) -> None:
        self.message = Message(text, color, life)

    def update(self, dt: float, data: HudData) -> None:
        self.generation_pop += dt
        self._drop_age += dt
        if data.active < self._last_alive:
            self._drop_age = 0.0
        self._last_alive = data.active
        if self.hook_age is not None:
            self.hook_age += dt
            if self.hook_age > 3.0:
                self.hook_age = None
        if self.message is not None:
            self.message.age += dt
            if self.message.age >= self.message.life:
                self.message = None
        self.alive.update(dt, data.active)
        self.distance.update(dt, data.car_distance_m)

    # ------------------------------------------------------------------ cards

    def draw(self, target: pygame.Surface, data: HudData, speed: float = 1.0) -> None:
        self._draw_cards(target, data)
        self._draw_lap_bar(target)
        if self.hook_age is not None:
            self._draw_hook(target, self.hook_age, data.population)
        else:
            self._draw_message(target)
        if speed != 1.0:
            self._draw_speed(target, speed)

    def _card(self, target, x: int, width: int, accent, label: str, label_color=MUTED) -> None:
        theme = self.theme
        target.blit(theme.panel(width, CARD_HEIGHT, alpha=232, radius=18), (x, ROW_Y))
        pygame.draw.rect(target, accent, (x + 18, ROW_Y, width - 36, 5), border_bottom_left_radius=3,
                         border_bottom_right_radius=3)
        theme.draw_text(target, label, (x + 22, ROW_Y + 14), 28, label_color, "SemiBold", tracking=2)

    def _draw_cards(self, target, data: HudData) -> None:
        theme = self.theme
        widths = (210, 310, ROW_WIDTH - 210 - 310 - 2 * CARD_GAP)
        x_gen = MARGIN
        x_cars = x_gen + widths[0] + CARD_GAP
        x_car = x_cars + widths[1] + CARD_GAP
        value_center = ROW_Y + 86  # every big number shares one baseline inside its card

        # Generation: the number pops when a new attempt starts.
        self._card(target, x_gen, widths[0], SIGNAL, "GEN")
        pop = 1.0 + (1.0 - ease_out(self.generation_pop / 0.3)) * 0.35
        number = theme.text(f"{data.generation}", 74, INK, "Black")
        if pop > 1.01:
            number = pygame.transform.smoothscale(number, (int(number.get_width() * pop), int(number.get_height() * pop)))
        target.blit(number, number.get_rect(midleft=(x_gen + 20, value_center)))

        # Cars left: flashes red each time another car goes out.
        alive = int(round(self.alive.value))
        base = DATA if alive > data.population * 0.25 else (AMBER if alive > 0 else DANGER)
        color = mix(DANGER, base, ease_out(self._drop_age / 0.35))
        self._card(target, x_cars, widths[1], base, "CARS LEFT")
        rect = theme.draw_text(target, f"{alive}", (x_cars + 20, value_center), 74, color, "Black", anchor="midleft")
        theme.draw_text(target, f"/ {data.population}", (rect.right + 12, value_center + 8), 36, MUTED, "SemiBold",
                        anchor="midleft")

        # The car being followed: its number, its colour, its distance against the record.
        self._card(target, x_car, widths[2], data.livery, f"CAR #{data.car_number}", INK)
        beaten = self.record_m > 0.0 and data.car_distance_m > self.record_m
        theme.draw_text(target, format_distance(self.distance.value), (x_car + 20, value_center), 74,
                        GOLD if beaten else INK, "Black", anchor="midleft")
        if self.record_m > 0.0:
            theme.draw_text(target, f"REC {format_distance(self.record_m)}", (x_car + widths[2] - 20, ROW_Y + 14), 28,
                            GOLD, "SemiBold", anchor="topright")

    def _draw_lap_bar(self, target) -> None:
        """One lap, end to end, with a mark at the record: the goal at a glance."""
        bar = pygame.Rect(MARGIN, BAR_Y, ROW_WIDTH, 10)
        pygame.draw.rect(target, (30, 32, 37), bar, border_radius=5)
        fraction = min(1.0, self.distance.value / max(self.lap_metres, 1.0))
        beaten = self.record_m > 0.0 and self.distance.value > self.record_m
        if int(bar.width * fraction) > 6:
            pygame.draw.rect(target, GOLD if beaten else SIGNAL, (bar.x, bar.y, int(bar.width * fraction), bar.height),
                             border_radius=5)
        if self.record_m > 0.0:
            mark = bar.x + int(bar.width * min(1.0, self.record_m / max(self.lap_metres, 1.0)))
            pygame.draw.rect(target, INK, (mark - 2, bar.y - 4, 4, bar.height + 8), border_radius=2)

    # ------------------------------------------------------------------ messages

    def _draw_message(self, target) -> None:
        message = self.message
        if message is None:
            return
        enter = ease_out(message.age / 0.12)
        leave = ease_in_out((message.age - (message.life - 0.15)) / 0.15)
        alpha = enter * (1.0 - leave)
        if alpha <= 0:
            return
        text = self.theme.text(message.text, 54, INK, "Black", 2)
        width, height = text.get_width() + 56, 76
        layer = pygame.Surface((width, height), pygame.SRCALPHA)
        layer.blit(self.theme.panel(width, height, alpha=236, radius=16), (0, 0))
        pygame.draw.rect(layer, message.color, (0, 16, 7, height - 32), border_radius=3)
        layer.blit(text, text.get_rect(center=(width // 2 + 4, height // 2)))
        layer.set_alpha(int(alpha * 255))
        target.blit(layer, (MARGIN, MESSAGE_Y))

    def _draw_speed(self, target, speed: float) -> None:
        """Playback speed is always labelled when it isn't real time, but stays small."""
        label = f"{speed:g}× SPEED" if speed > 1.0 else "SLOW-MO"
        text = self.theme.text(label, 32, AMBER, "Bold", 1)
        width, height = text.get_width() + 32, 50
        panel = self.theme.panel(width, height, alpha=225, radius=12)
        rect = panel.get_rect(topright=(MARGIN + ROW_WIDTH, MESSAGE_Y))
        target.blit(panel, rect)
        target.blit(text, text.get_rect(center=rect.center))

    def _draw_hook(self, target, t: float, population: int) -> None:
        """The first seconds decide whether someone keeps watching, so state the premise."""
        alpha = ease_out(t / 0.2) * (1.0 - ease_in_out((t - 2.6) / 0.4))
        if alpha <= 0:
            return
        height = 186
        panel = self.theme.panel(ROW_WIDTH, height, alpha=232, radius=18).copy()
        panel.set_alpha(int(alpha * 255))
        target.blit(panel, (MARGIN, MESSAGE_Y))
        self._centered_text(target, "AI LEARNS TO RACE", MESSAGE_Y + 66, 104, SIGNAL, alpha)
        self._centered_text(target, f"{population} cars. Zero driving skills.", MESSAGE_Y + 146, 46, INK, alpha, "SemiBold", 0)

    def _centered_text(self, target, text: str, y: int, size: int, color, alpha: float, weight="Black", tracking=2) -> None:
        surface = self.theme.text(text, size, color, weight, tracking)
        rect = surface.get_rect(center=(CENTER_X, y))
        if alpha >= 1.0:
            target.blit(surface, rect)
        else:
            faded = surface.copy()
            faded.set_alpha(int(alpha * 255))
            target.blit(faded, rect)

    # ------------------------------------------------------------------ ending

    def draw_end(self, target, t: float, generations: int, lap_seconds: float | None, succeeded: bool) -> None:
        """The payoff card, big enough to read in the moment before a Reel loops."""
        dim = pygame.Surface(target.get_size())
        dim.fill((6, 7, 9))
        dim.set_alpha(int(210 * ease_out(t / 0.3)))
        target.blit(dim, (0, 0))
        reveal = ease_out(t / 0.35)
        if reveal <= 0:
            return
        y = END_CARD_Y + int((1.0 - reveal) * 30)
        if succeeded:
            self._centered_text(target, "FIRST CLEAN LAP", y, 92, GOLD, reveal)
            self._centered_text(target, "IN JUST", y + 120, 56, MUTED, reveal, "SemiBold", 4)
            self._centered_text(target, f"{generations}", y + 290, 260, INK, reveal)
            word = "GENERATIONS" if generations != 1 else "GENERATION"
            self._centered_text(target, word, y + 460, 84, INK, reveal)
            if lap_seconds:
                self._centered_text(target, f"Lap time {lap_seconds:.1f}s", y + 550, 54, DATA, reveal, "SemiBold", 0)
        else:
            self._centered_text(target, "STILL LEARNING", y, 92, AMBER, reveal)
            self._centered_text(target, f"{generations}", y + 220, 260, INK, reveal)
            self._centered_text(target, "GENERATIONS SO FAR", y + 390, 72, INK, reveal)
        second = ease_out((t - 0.5) / 0.4)
        if second > 0:
            self._centered_text(target, "Nobody taught it to drive.", y + 660, 58, INK, second, "SemiBold", 0)
            self._centered_text(target, "It learned by crashing.", y + 730, 58, SIGNAL, second, "SemiBold", 0)
