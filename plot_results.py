"""
plot_results.py — Post-Hoc Visualization of RL Experiment Results.

Reads CSV files from results/ and generates comparison plots:
  1. Learning curves (episode return over training).
  2. Action accuracy over training.
  3. Premature action rate over training.
  4. Episode length distribution (final policy).
  5. Summary comparison bar chart.

Usage:
    python plot_results.py
    python plot_results.py --results-dir results --window 200
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("plot")

# Consistent colors
COLOR_BASELINE = "#e74c3c"
COLOR_OBSERVER = "#2ecc71"


def load_data(results_dir: Path) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Load all CSV files, grouped by mode."""
    baseline_dfs: list[pd.DataFrame] = []
    observer_dfs: list[pd.DataFrame] = []

    for csv_file in sorted(results_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)
        if csv_file.name.startswith("baseline"):
            baseline_dfs.append(df)
        elif csv_file.name.startswith("observer"):
            observer_dfs.append(df)

    logger.info("Loaded %d baseline runs, %d observer runs", len(baseline_dfs), len(observer_dfs))
    return baseline_dfs, observer_dfs


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_learning_curves(
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    window: int,
    fig_dir: Path,
) -> None:
    """Plot episode return learning curves with confidence bands."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, dfs, color in [
        ("Baseline", baseline_dfs, COLOR_BASELINE),
        ("Observer", observer_dfs, COLOR_OBSERVER),
    ]:
        # Stack smoothed curves from each seed
        smoothed = []
        for df in dfs:
            s = smooth(df["episode_return"].values, window)
            smoothed.append(s)

        # Truncate to shortest length
        min_len = min(len(s) for s in smoothed)
        stacked = np.array([s[:min_len] for s in smoothed])

        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        x = np.arange(min_len) + window

        ax.plot(x, mean, label=label, color=color, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return (smoothed)")
    ax.set_title("Learning Curves: Baseline vs. Observer-Augmented")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved learning_curves.png")


def plot_accuracy(
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    window: int,
    fig_dir: Path,
) -> None:
    """Plot rolling action accuracy over training."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, dfs, color in [
        ("Baseline", baseline_dfs, COLOR_BASELINE),
        ("Observer", observer_dfs, COLOR_OBSERVER),
    ]:
        smoothed = []
        for df in dfs:
            s = smooth(df["action_correct"].values.astype(float), window)
            smoothed.append(s * 100)  # convert to percentage

        min_len = min(len(s) for s in smoothed)
        stacked = np.array([s[:min_len] for s in smoothed])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        x = np.arange(min_len) + window

        ax.plot(x, mean, label=label, color=color, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Action Accuracy (%)")
    ax.set_title("Action Accuracy: Baseline vs. Observer-Augmented")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "action_accuracy.png", dpi=150)
    plt.close(fig)
    logger.info("Saved action_accuracy.png")


def plot_premature_rate(
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    window: int,
    fig_dir: Path,
) -> None:
    """Plot rolling premature action rate."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, dfs, color in [
        ("Baseline", baseline_dfs, COLOR_BASELINE),
        ("Observer", observer_dfs, COLOR_OBSERVER),
    ]:
        smoothed = []
        for df in dfs:
            s = smooth(df["premature_action"].values.astype(float), window)
            smoothed.append(s * 100)

        min_len = min(len(s) for s in smoothed)
        stacked = np.array([s[:min_len] for s in smoothed])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        x = np.arange(min_len) + window

        ax.plot(x, mean, label=label, color=color, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Premature Action Rate (%)")
    ax.set_title("Premature Actions: Baseline vs. Observer-Augmented")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "premature_rate.png", dpi=150)
    plt.close(fig)
    logger.info("Saved premature_rate.png")


def plot_episode_length_dist(
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    fig_dir: Path,
    tail_frac: float = 0.2,
) -> None:
    """Box plot of episode lengths from the final portion of training."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    labels = []
    colors = []

    for label, dfs, color in [
        ("Baseline", baseline_dfs, COLOR_BASELINE),
        ("Observer", observer_dfs, COLOR_OBSERVER),
    ]:
        all_lengths = []
        for df in dfs:
            n = len(df)
            tail = df.iloc[int(n * (1 - tail_frac)):]
            all_lengths.extend(tail["episode_length"].values.tolist())
        data.append(all_lengths)
        labels.append(label)
        colors.append(color)

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length Distribution (Final 20% of Training)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "episode_length_dist.png", dpi=150)
    plt.close(fig)
    logger.info("Saved episode_length_dist.png")


def plot_summary_bars(
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    fig_dir: Path,
    tail_frac: float = 0.2,
) -> None:
    """Bar chart comparing final metrics side by side."""
    def final_metrics(dfs: list[pd.DataFrame]) -> dict[str, tuple[float, float]]:
        returns, accuracies, lengths, prematures = [], [], [], []
        for df in dfs:
            n = len(df)
            tail = df.iloc[int(n * (1 - tail_frac)):]
            returns.append(tail["episode_return"].mean())
            accuracies.append(tail["action_correct"].mean() * 100)
            lengths.append(tail["episode_length"].mean())
            prematures.append(tail["premature_action"].mean() * 100)
        return {
            "Return": (np.mean(returns), np.std(returns)),
            "Accuracy (%)": (np.mean(accuracies), np.std(accuracies)),
            "Ep. Length": (np.mean(lengths), np.std(lengths)),
            "Premature (%)": (np.mean(prematures), np.std(prematures)),
        }

    bl = final_metrics(baseline_dfs)
    ob = final_metrics(observer_dfs)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    metrics = list(bl.keys())

    for ax, metric in zip(axes, metrics):
        bm, bs = bl[metric]
        om, os_ = ob[metric]

        bars = ax.bar(
            ["Baseline", "Observer"],
            [bm, om],
            yerr=[bs, os_],
            color=[COLOR_BASELINE, COLOR_OBSERVER],
            alpha=0.7,
            capsize=5,
        )
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Final Performance Comparison (Last 20% of Training)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "summary_bars.png", dpi=150)
    plt.close(fig)
    logger.info("Saved summary_bars.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RL experiment results")
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory containing CSV files (default: results/)"
    )
    parser.add_argument(
        "--window", type=int, default=200,
        help="Smoothing window for rolling plots (default: 200)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    baseline_dfs, observer_dfs = load_data(results_dir)

    if not baseline_dfs or not observer_dfs:
        logger.error("No data found. Run 'python run_experiment.py' first.")
        return

    plot_learning_curves(baseline_dfs, observer_dfs, args.window, fig_dir)
    plot_accuracy(baseline_dfs, observer_dfs, args.window, fig_dir)
    plot_premature_rate(baseline_dfs, observer_dfs, args.window, fig_dir)
    plot_episode_length_dist(baseline_dfs, observer_dfs, fig_dir)
    plot_summary_bars(baseline_dfs, observer_dfs, fig_dir)

    logger.info("All plots saved to %s", fig_dir)


if __name__ == "__main__":
    main()
