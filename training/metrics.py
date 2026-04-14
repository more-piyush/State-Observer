"""
metrics.py — Metric Collection and Aggregation for RL Training.

Tracks per-episode metrics and provides rolling-window statistics
for comparing baseline vs. observer-augmented agents.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics collected for a single episode."""
    episode: int
    episode_return: float        # cumulative reward
    episode_length: int          # number of steps
    action_correct: bool         # did the agent ACT in READY state?
    acted: bool                  # did the agent take ACT at all (vs truncation)?
    premature_action: bool       # ACT taken in non-READY state
    final_hidden_state: str      # hidden state when episode ended
    loss: float | None = None    # mean training loss this episode


@dataclass
class MetricsLogger:
    """
    Accumulates per-episode metrics and provides aggregate statistics.

    Args:
        mode: Identifier string ("baseline" or "observer").
        seed: Random seed for this run.
    """

    mode: str
    seed: int
    _history: list[EpisodeMetrics] = field(default_factory=list, repr=False)

    def log(self, metrics: EpisodeMetrics) -> None:
        """Record metrics for one episode."""
        self._history.append(metrics)

    @property
    def episode_count(self) -> int:
        return len(self._history)

    # -----------------------------------------------------------------------
    # Rolling Statistics
    # -----------------------------------------------------------------------

    def rolling_mean(
        self, key: str, window: int = 100
    ) -> float:
        """Compute rolling mean of a numeric metric over the last `window` episodes."""
        recent = self._history[-window:]
        if not recent:
            return 0.0
        values = [getattr(m, key) for m in recent if getattr(m, key) is not None]
        return float(np.mean(values)) if values else 0.0

    def rolling_stats(
        self, key: str, window: int = 100
    ) -> tuple[float, float]:
        """Return (mean, std) of a metric over the last `window` episodes."""
        recent = self._history[-window:]
        if not recent:
            return 0.0, 0.0
        values = [getattr(m, key) for m in recent if getattr(m, key) is not None]
        if not values:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values))

    def action_accuracy(self, window: int = 100) -> float:
        """Fraction of episodes (in window) where the agent acted correctly."""
        recent = self._history[-window:]
        if not recent:
            return 0.0
        correct = sum(1 for m in recent if m.action_correct)
        return correct / len(recent)

    def premature_action_rate(self, window: int = 100) -> float:
        """Fraction of episodes (in window) with premature ACT."""
        recent = self._history[-window:]
        if not recent:
            return 0.0
        premature = sum(1 for m in recent if m.premature_action)
        return premature / len(recent)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self, window: int = 100) -> dict[str, Any]:
        """Generate a summary dict of rolling metrics."""
        ret_mean, ret_std = self.rolling_stats("episode_return", window)
        len_mean, len_std = self.rolling_stats("episode_length", window)
        return {
            "mode": self.mode,
            "seed": self.seed,
            "episodes": self.episode_count,
            "return_mean": ret_mean,
            "return_std": ret_std,
            "length_mean": len_mean,
            "length_std": len_std,
            "action_accuracy": self.action_accuracy(window),
            "premature_rate": self.premature_action_rate(window),
        }

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def get_series(self, key: str) -> list[float]:
        """Extract a full time-series of a metric for plotting."""
        return [
            float(getattr(m, key)) if getattr(m, key) is not None else 0.0
            for m in self._history
        ]

    def save_csv(self, path: Path) -> None:
        """Export all episode metrics to a CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._history:
            return

        fieldnames = [
            "episode", "episode_return", "episode_length",
            "action_correct", "acted", "premature_action",
            "final_hidden_state", "loss",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in self._history:
                writer.writerow({
                    "episode": m.episode,
                    "episode_return": f"{m.episode_return:.4f}",
                    "episode_length": m.episode_length,
                    "action_correct": int(m.action_correct),
                    "acted": int(m.acted),
                    "premature_action": int(m.premature_action),
                    "final_hidden_state": m.final_hidden_state,
                    "loss": f"{m.loss:.6f}" if m.loss is not None else "",
                })

        logger.info("Saved %d episodes to %s", len(self._history), path)
