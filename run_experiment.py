"""
run_experiment.py — A/B Benchmark: Baseline vs. PGM Observer-Augmented Agent.

Runs the full comparison experiment:
  - For each random seed (default 5):
      - Train a DQN agent with raw observations only (baseline).
      - Train a DQN agent with PGM observer-augmented observations.
  - Save per-episode metrics to CSV files in the results/ directory.
  - Print a final comparison table.

Usage:
    python run_experiment.py
    python run_experiment.py --num-episodes 5000 --num-seeds 3
    python run_experiment.py --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from envs.page_load_env import PageLoadEnv
from agents.dqn_agent import DQNAgent
from training.trainer import Trainer
from training.metrics import MetricsLogger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment")


# ---------------------------------------------------------------------------
# Seed Utility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Single Run
# ---------------------------------------------------------------------------

def run_single(
    mode: str,
    seed: int,
    num_episodes: int,
    output_dir: Path,
    device: str,
) -> MetricsLogger:
    """
    Train one agent for one seed.

    Args:
        mode: "baseline" or "observer".
        seed: Random seed.
        num_episodes: Number of training episodes.
        output_dir: Directory for CSV output.
        device: Torch device string.

    Returns:
        MetricsLogger with all episode data.
    """
    set_seed(seed)

    use_observer = (mode == "observer")
    env = PageLoadEnv(use_observer=use_observer)
    obs_dim = env.observation_space.shape[0]

    agent = DQNAgent(obs_dim=obs_dim, device=device)
    metrics_logger = MetricsLogger(mode=mode, seed=seed)

    trainer = Trainer(
        env=env,
        agent=agent,
        metrics_logger=metrics_logger,
        num_episodes=num_episodes,
    )

    trainer.train()

    # Save results
    csv_path = output_dir / f"{mode}_seed{seed}.csv"
    metrics_logger.save_csv(csv_path)

    return metrics_logger


# ---------------------------------------------------------------------------
# Comparison Table
# ---------------------------------------------------------------------------

def print_comparison(
    baseline_loggers: list[MetricsLogger],
    observer_loggers: list[MetricsLogger],
    window: int = 1000,
) -> None:
    """Print a side-by-side comparison table of final performance."""
    print("\n" + "=" * 76)
    print("  EXPERIMENT RESULTS — Final Performance (last %d episodes)" % window)
    print("=" * 76)

    def aggregate(loggers: list[MetricsLogger]) -> dict[str, tuple[float, float]]:
        """Compute mean ± std across seeds for each metric."""
        summaries = [lg.summary(window=window) for lg in loggers]
        keys = ["return_mean", "length_mean", "action_accuracy", "premature_rate"]
        result = {}
        for k in keys:
            vals = [s[k] for s in summaries]
            result[k] = (float(np.mean(vals)), float(np.std(vals)))
        return result

    bl = aggregate(baseline_loggers)
    ob = aggregate(observer_loggers)

    header = f"  {'Metric':<25s} {'Baseline':>18s} {'Observer':>18s} {'Improvement':>14s}"
    print(header)
    print("  " + "─" * 73)

    rows = [
        ("Episode Return", "return_mean", "+", True),
        ("Episode Length", "length_mean", "-", False),
        ("Action Accuracy", "action_accuracy", "+", True),
        ("Premature Rate", "premature_rate", "-", False),
    ]

    for label, key, better_dir, higher_is_better in rows:
        bm, bs = bl[key]
        om, os_ = ob[key]

        # Format values
        if "accuracy" in key or "rate" in key:
            bl_str = f"{bm*100:5.1f}% ± {bs*100:4.1f}%"
            ob_str = f"{om*100:5.1f}% ± {os_*100:4.1f}%"
            diff = (om - bm) * 100
            imp_str = f"{diff:+.1f}pp"
        else:
            bl_str = f"{bm:6.2f} ± {bs:5.2f}"
            ob_str = f"{om:6.2f} ± {os_:5.2f}"
            diff = om - bm
            imp_str = f"{diff:+.2f}"

        print(f"  {label:<25s} {bl_str:>18s} {ob_str:>18s} {imp_str:>14s}")

    print("=" * 76)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PGM State Observer — RL A/B Benchmark"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10_000,
        help="Training episodes per run (default: 10000)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5,
        help="Number of random seeds (default: 5)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for CSV output (default: results/)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device (default: cpu)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Experiment config: episodes=%d seeds=%d device=%s output=%s",
        args.num_episodes, args.num_seeds, args.device, output_dir,
    )

    baseline_loggers: list[MetricsLogger] = []
    observer_loggers: list[MetricsLogger] = []

    total_start = time.time()

    for seed in range(args.num_seeds):
        logger.info("━" * 60)
        logger.info("Seed %d/%d", seed + 1, args.num_seeds)
        logger.info("━" * 60)

        # --- Baseline run ---
        bl = run_single("baseline", seed, args.num_episodes, output_dir, args.device)
        baseline_loggers.append(bl)

        # --- Observer-augmented run ---
        ob = run_single("observer", seed, args.num_episodes, output_dir, args.device)
        observer_loggers.append(ob)

    total_time = time.time() - total_start
    logger.info("Total experiment time: %.1fs", total_time)

    # --- Print comparison ---
    print_comparison(baseline_loggers, observer_loggers)

    # --- Trigger plot generation ---
    logger.info("Results saved to %s/. Run 'python plot_results.py' to visualize.", output_dir)


if __name__ == "__main__":
    main()
