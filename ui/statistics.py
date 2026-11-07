"""Training charts, drawn with Matplotlib from the recorded generation history."""

from __future__ import annotations

import threading
from pathlib import Path


def save_training_plot(history: list, path: Path) -> None:
    """Write a dark-themed summary PNG. Uses Agg directly, so no display is needed."""
    if len(history) < 2:
        return
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    generations = [r.generation for r in history]
    background, panel, grid, ink, muted = "#111317", "#191c21", "#2a2e35", "#eef0f4", "#929aa7"
    signal, data, amber, danger = "#ff542e", "#3ad6be", "#ffbe48", "#ec424c"

    figure = Figure(figsize=(13, 8), dpi=120, facecolor=background)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)

    def style(ax, title: str) -> None:
        ax.set_facecolor(panel)
        ax.set_title(title, color=ink, loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(True, color=grid, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("Generation", color=muted, fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def legend(ax) -> None:
        ax.legend(frameon=True, facecolor=panel, edgecolor="none", framealpha=0.85, labelcolor=ink,
                  fontsize=9, loc="upper left")

    ax = axes[0, 0]
    style(ax, "Fitness")
    ax.plot(generations, [r.best_fitness for r in history], color=signal, linewidth=2, label="Best")
    ax.plot(generations, [r.average_fitness for r in history], color=data, linewidth=2, label="Average")
    legend(ax)

    ax = axes[0, 1]
    style(ax, "Best distance in a generation (m)")
    ax.fill_between(generations, [r.best_distance_m for r in history], color=signal, alpha=0.18)
    ax.plot(generations, [r.best_distance_m for r in history], color=signal, linewidth=2)

    ax = axes[1, 0]
    style(ax, "How each generation's cars ended")
    finishers = [r.finishers for r in history]
    crashed = [r.crashed for r in history]
    other = [r.timed_out + r.wrong_way for r in history]
    ax.stackplot(generations, finishers, crashed, other, colors=[data, danger, amber], alpha=0.85,
                 labels=["Finished or still running", "Crashed", "Timed out / wrong way"])
    legend(ax)

    ax = axes[1, 1]
    style(ax, "Best lap time (s)")
    timed = [(r.generation, r.best_lap_seconds) for r in history if r.best_lap_seconds > 0]
    if timed:
        ax.plot(*zip(*timed), color=amber, linewidth=2, marker="o", markersize=3)
    else:
        ax.text(0.5, 0.5, "No completed laps yet", color=muted, ha="center", transform=ax.transAxes)

    figure.tight_layout(pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=background)


class PlotWriter:
    """Writes charts on a background thread so the simulation never stalls for them."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None

    def request(self, history: list, path: Path) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=save_training_plot, args=(list(history), path), daemon=True)
        self._thread.start()

    def finish(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=20.0)
