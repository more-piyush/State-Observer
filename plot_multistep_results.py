"""
plot_multistep_results.py — Visualization for Multi-Step RL Experiment Results.

Reads CSV files from results_multistep/ and generates comparison plots:
  1. Learning curves (episode return over training).
  2. Workflow completion rate over training.
  3. Premature execution rate over training.
  4. Per-node completion breakdown (final policy).
  5. Total steps distribution (efficiency).
  6. Summary comparison bar chart.

Usage:
    python plot_multistep_results.py
    python plot_multistep_results.py --results-dir results_multistep --window 200
    python plot_multistep_results.py --workflow checkout
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("plot_multistep")

# Consistent colors
COLOR_BASELINE = "#e74c3c"
COLOR_OBSERVER = "#2ecc71"
COLOR_PALETTE = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c"]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def discover_workflows(results_dir: Path) -> list[str]:
    """Discover all workflow names from CSV filenames."""
    workflows = set()
    for csv_file in results_dir.glob("*.csv"):
        # Pattern: {workflow}_{mode}_seed{n}.csv
        match = re.match(r"^(.+?)_(baseline|observer)_seed\d+\.csv$", csv_file.name)
        if match:
            workflows.add(match.group(1))
    return sorted(workflows)


def load_workflow_data(
    results_dir: Path, workflow: str
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Load all CSV files for a specific workflow, grouped by mode."""
    baseline_dfs: list[pd.DataFrame] = []
    observer_dfs: list[pd.DataFrame] = []

    for csv_file in sorted(results_dir.glob(f"{workflow}_*.csv")):
        df = pd.read_csv(csv_file)
        if f"{workflow}_baseline" in csv_file.name:
            baseline_dfs.append(df)
        elif f"{workflow}_observer" in csv_file.name:
            observer_dfs.append(df)

    logger.info(
        "Workflow '%s': loaded %d baseline, %d observer runs",
        workflow, len(baseline_dfs), len(observer_dfs),
    )
    return baseline_dfs, observer_dfs


# ---------------------------------------------------------------------------
# Smoothing Utility
# ---------------------------------------------------------------------------

def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_smoothed_comparison(
    ax: plt.Axes,
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    column: str,
    window: int,
    ylabel: str,
    title: str,
    scale: float = 1.0,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Generic helper: plot a smoothed metric with confidence bands for both modes.

    Args:
        scale: Multiply values by this (e.g., 100 for percentage).
    """
    for label, dfs, color in [
        ("Baseline", baseline_dfs, COLOR_BASELINE),
        ("Observer", observer_dfs, COLOR_OBSERVER),
    ]:
        smoothed = []
        for df in dfs:
            s = smooth(df[column].values.astype(float) * scale, window)
            smoothed.append(s)

        if not smoothed:
            continue

        min_len = min(len(s) for s in smoothed)
        stacked = np.array([s[:min_len] for s in smoothed])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        x = np.arange(min_len) + window

        ax.plot(x, mean, label=label, color=color, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Plot 1: Learning Curves
# ---------------------------------------------------------------------------

def plot_learning_curves(
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    window: int,
    fig_dir: Path,
    workflow: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_smoothed_comparison(
        ax, baseline_dfs, observer_dfs,
        column="episode_return", window=window,
        ylabel="Episode Return (smoothed)",
        title=f"Learning Curves — '{workflow}' Workflow",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"{workflow}_learning_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s_learning_curves.png", workflow)


# ---------------------------------------------------------------------------
# Plot 2: Completion Rate
# ---------------------------------------------------------------------------

def plot_completion_rate(
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    window: int,
    fig_dir: Path,
    workflow: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_smoothed_comparison(
        ax, baseline_dfs, observer_dfs,
        column="completion_rate", window=window, scale=100.0,
        ylabel="Workflow Completion Rate (%)",
        title=f"Completion Rate — '{workflow}' Workflow",
        ylim=(0, 105),
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"{workflow}_completion_rate.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s_completion_rate.png", workflow)


# ---------------------------------------------------------------------------
# Plot 3: Premature Executions
# ---------------------------------------------------------------------------

def plot_premature_executions(
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    window: int,
    fig_dir: Path,
    workflow: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_smoothed_comparison(
        ax, baseline_dfs, observer_dfs,
        column="premature_executions", window=window,
        ylabel="Premature Executions per Episode",
        title=f"Premature Executions — '{workflow}' Workflow",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"{workflow}_premature_executions.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s_premature_executions.png", workflow)


# ---------------------------------------------------------------------------
# Plot 4: Nodes Completed vs Skipped (Final Policy)
# ---------------------------------------------------------------------------

def plot_node_breakdown(
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    fig_dir: Path,
    workflow: str,
    tail_frac: float = 0.2,
) -> None:
    """Stacked bar: completed vs skipped nodes in the final portion of training."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Baseline", "Observer"]
    completed_means = []
    skipped_means = []
    completed_stds = []
    skipped_stds = []

    for dfs in [baseline_dfs, observer_dfs]:
        all_completed = []
        all_skipped = []
        for df in dfs:
            n = len(df)
            tail = df.iloc[int(n * (1 - tail_frac)):]
            all_completed.append(tail["nodes_completed"].mean())
            all_skipped.append(tail["nodes_skipped"].mean())
        completed_means.append(np.mean(all_completed))
        completed_stds.append(np.std(all_completed))
        skipped_means.append(np.mean(all_skipped))
        skipped_stds.append(np.std(all_skipped))

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(
        x, completed_means, width,
        yerr=completed_stds, capsize=5,
        label="Completed", color=COLOR_OBSERVER, alpha=0.8,
    )
    bars2 = ax.bar(
        x, skipped_means, width, bottom=completed_means,
        yerr=skipped_stds, capsize=5,
        label="Skipped / Timed Out", color=COLOR_BASELINE, alpha=0.6,
    )

    ax.set_ylabel("Nodes per Episode")
    ax.set_title(f"Node Completion Breakdown — '{workflow}' (Final 20%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{workflow}_node_breakdown.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s_node_breakdown.png", workflow)


# ---------------------------------------------------------------------------
# Plot 5: Episode Length Distribution
# ---------------------------------------------------------------------------

def plot_episode_length_dist(
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    fig_dir: Path,
    workflow: str,
    tail_frac: float = 0.2,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    labels_list = []
    colors = []

    for label, dfs, color in [
        ("Baseline", baseline_dfs, COLOR_BASELINE),
        ("Observer", observer_dfs, COLOR_OBSERVER),
    ]:
        all_lengths = []
        for df in dfs:
            n = len(df)
            tail = df.iloc[int(n * (1 - tail_frac)):]
            all_lengths.extend(tail["total_steps"].values.tolist())
        data.append(all_lengths)
        labels_list.append(label)
        colors.append(color)

    bp = ax.boxplot(data, labels=labels_list, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Total Steps per Episode")
    ax.set_title(f"Episode Length Distribution — '{workflow}' (Final 20%)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{workflow}_episode_length_dist.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s_episode_length_dist.png", workflow)


# ---------------------------------------------------------------------------
# Plot 6: Summary Dashboard (2x3 grid)
# ---------------------------------------------------------------------------

def plot_summary_dashboard(
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    window: int,
    fig_dir: Path,
    workflow: str,
    tail_frac: float = 0.2,
) -> None:
    """Single dashboard figure with all key metrics in a 2x3 grid."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Multi-Step RL Benchmark Dashboard — '{workflow}' Workflow",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # (0,0) Learning curve
    ax1 = fig.add_subplot(gs[0, 0])
    plot_smoothed_comparison(
        ax1, baseline_dfs, observer_dfs,
        column="episode_return", window=window,
        ylabel="Return", title="Learning Curve",
    )

    # (0,1) Completion rate
    ax2 = fig.add_subplot(gs[0, 1])
    plot_smoothed_comparison(
        ax2, baseline_dfs, observer_dfs,
        column="completion_rate", window=window, scale=100.0,
        ylabel="Completion %", title="Completion Rate",
        ylim=(0, 105),
    )

    # (0,2) Premature executions
    ax3 = fig.add_subplot(gs[0, 2])
    plot_smoothed_comparison(
        ax3, baseline_dfs, observer_dfs,
        column="premature_executions", window=window,
        ylabel="Count", title="Premature Executions",
    )

    # (1,0) Summary bars
    ax4 = fig.add_subplot(gs[1, 0])
    _plot_summary_bars(ax4, baseline_dfs, observer_dfs, tail_frac)

    # (1,1) Episode length boxplot
    ax5 = fig.add_subplot(gs[1, 1])
    _plot_length_box(ax5, baseline_dfs, observer_dfs, tail_frac)

    # (1,2) Nodes completed vs skipped
    ax6 = fig.add_subplot(gs[1, 2])
    _plot_node_stacked(ax6, baseline_dfs, observer_dfs, tail_frac)

    fig.savefig(fig_dir / f"{workflow}_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s_dashboard.png", workflow)


def _plot_summary_bars(
    ax: plt.Axes,
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    tail_frac: float,
) -> None:
    """Summary bar chart on a given axes."""
    def final_vals(dfs: list[pd.DataFrame], col: str) -> tuple[float, float]:
        vals = []
        for df in dfs:
            n = len(df)
            tail = df.iloc[int(n * (1 - tail_frac)):]
            vals.append(tail[col].mean())
        return float(np.mean(vals)), float(np.std(vals))

    metrics = {
        "Return": "episode_return",
        "Completion": "completion_rate",
        "Steps": "total_steps",
    }

    x = np.arange(len(metrics))
    width = 0.3

    bl_vals = [final_vals(baseline_dfs, c) for c in metrics.values()]
    ob_vals = [final_vals(observer_dfs, c) for c in metrics.values()]

    ax.bar(x - width/2, [v[0] for v in bl_vals], width,
           yerr=[v[1] for v in bl_vals], capsize=3,
           label="Baseline", color=COLOR_BASELINE, alpha=0.7)
    ax.bar(x + width/2, [v[0] for v in ob_vals], width,
           yerr=[v[1] for v in ob_vals], capsize=3,
           label="Observer", color=COLOR_OBSERVER, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_title("Final Metrics (Last 20%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def _plot_length_box(
    ax: plt.Axes,
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    tail_frac: float,
) -> None:
    data = []
    for dfs in [baseline_dfs, observer_dfs]:
        lengths = []
        for df in dfs:
            n = len(df)
            tail = df.iloc[int(n * (1 - tail_frac)):]
            lengths.extend(tail["total_steps"].tolist())
        data.append(lengths)

    bp = ax.boxplot(data, labels=["Baseline", "Observer"], patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], [COLOR_BASELINE, COLOR_OBSERVER]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title("Episode Length")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_node_stacked(
    ax: plt.Axes,
    baseline_dfs: list[pd.DataFrame],
    observer_dfs: list[pd.DataFrame],
    tail_frac: float,
) -> None:
    labels = ["Baseline", "Observer"]
    completed = []
    skipped = []

    for dfs in [baseline_dfs, observer_dfs]:
        c_vals, s_vals = [], []
        for df in dfs:
            n = len(df)
            tail = df.iloc[int(n * (1 - tail_frac)):]
            c_vals.append(tail["nodes_completed"].mean())
            s_vals.append(tail["nodes_skipped"].mean())
        completed.append(np.mean(c_vals))
        skipped.append(np.mean(s_vals))

    x = np.arange(len(labels))
    ax.bar(x, completed, 0.4, label="Completed", color=COLOR_OBSERVER, alpha=0.8)
    ax.bar(x, skipped, 0.4, bottom=completed, label="Skipped", color=COLOR_BASELINE, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Node Breakdown")
    ax.set_ylabel("Nodes / Episode")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-step RL experiment results")
    parser.add_argument(
        "--results-dir", type=str, default="results_multistep",
        help="Directory containing CSV files (default: results_multistep/)",
    )
    parser.add_argument(
        "--window", type=int, default=200,
        help="Smoothing window (default: 200)",
    )
    parser.add_argument(
        "--workflow", type=str, default=None,
        help="Plot a specific workflow only (default: all discovered)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.workflow:
        workflows = [args.workflow]
    else:
        workflows = discover_workflows(results_dir)

    if not workflows:
        logger.error("No data found in %s. Run 'python run_multistep_experiment.py' first.", results_dir)
        return

    logger.info("Discovered workflows: %s", workflows)

    for wf in workflows:
        logger.info("━" * 40)
        logger.info("Plotting workflow: %s", wf)
        logger.info("━" * 40)

        baseline_dfs, observer_dfs = load_workflow_data(results_dir, wf)

        if not baseline_dfs or not observer_dfs:
            logger.warning("Incomplete data for workflow '%s', skipping.", wf)
            continue

        # Individual plots
        plot_learning_curves(baseline_dfs, observer_dfs, args.window, fig_dir, wf)
        plot_completion_rate(baseline_dfs, observer_dfs, args.window, fig_dir, wf)
        plot_premature_executions(baseline_dfs, observer_dfs, args.window, fig_dir, wf)
        plot_node_breakdown(baseline_dfs, observer_dfs, fig_dir, wf)
        plot_episode_length_dist(baseline_dfs, observer_dfs, fig_dir, wf)

        # Combined dashboard
        plot_summary_dashboard(baseline_dfs, observer_dfs, args.window, fig_dir, wf)

    logger.info("All plots saved to %s", fig_dir)


if __name__ == "__main__":
    main()
