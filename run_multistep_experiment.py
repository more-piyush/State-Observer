"""
run_multistep_experiment.py — A/B Benchmark for Multi-Step Workflows.

Compares baseline vs. observer-augmented hierarchical DQN agents on
multi-step web automation tasks (form submission, search, checkout).

For each (workflow, seed, mode) combination:
    1. Instantiate the MultiStepEnv with the chosen TaskGraph.
    2. Train a HierarchicalDQNAgent for N episodes.
    3. Log per-episode metrics including workflow completion rate,
       per-node accuracy, and total reward.

Usage:
    python run_multistep_experiment.py
    python run_multistep_experiment.py --workflow checkout --num-episodes 5000
    python run_multistep_experiment.py --workflow all --num-seeds 3
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
from envs.multi_step_env import MultiStepEnv, WAIT, EXECUTE, SKIP
from envs.task_graph import WORKFLOW_REGISTRY, TaskGraph
from agents.hierarchical_agent import HierarchicalDQNAgent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multistep_experiment")


# ---------------------------------------------------------------------------
# Per-Episode Metrics
# ---------------------------------------------------------------------------

@dataclass
class MultiStepEpisodeMetrics:
    """Metrics for one multi-step episode."""
    episode: int
    episode_return: float
    total_steps: int
    nodes_completed: int
    nodes_skipped: int
    workflow_size: int
    workflow_complete: bool          # all nodes processed
    completion_rate: float           # nodes_completed / workflow_size
    premature_executions: int        # EXECUTE when not in required state
    correct_executions: int          # EXECUTE when in required state
    ctrl_loss: float | None = None
    meta_loss: float | None = None


@dataclass
class MultiStepMetricsLogger:
    """Accumulates multi-step episode metrics."""
    mode: str
    seed: int
    workflow: str
    _history: list[MultiStepEpisodeMetrics] = field(default_factory=list, repr=False)

    def log(self, m: MultiStepEpisodeMetrics) -> None:
        self._history.append(m)

    @property
    def episode_count(self) -> int:
        return len(self._history)

    def rolling_stats(self, key: str, window: int = 100) -> tuple[float, float]:
        recent = self._history[-window:]
        if not recent:
            return 0.0, 0.0
        vals = [getattr(m, key) for m in recent if getattr(m, key) is not None]
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals))

    def summary(self, window: int = 100) -> dict[str, Any]:
        ret_m, ret_s = self.rolling_stats("episode_return", window)
        cr_m, cr_s = self.rolling_stats("completion_rate", window)
        len_m, len_s = self.rolling_stats("total_steps", window)
        prem_m, _ = self.rolling_stats("premature_executions", window)
        return {
            "mode": self.mode,
            "seed": self.seed,
            "workflow": self.workflow,
            "episodes": self.episode_count,
            "return_mean": ret_m, "return_std": ret_s,
            "completion_rate_mean": cr_m, "completion_rate_std": cr_s,
            "length_mean": len_m, "length_std": len_s,
            "premature_mean": prem_m,
        }

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._history:
            return
        fieldnames = [
            "episode", "episode_return", "total_steps",
            "nodes_completed", "nodes_skipped", "workflow_size",
            "workflow_complete", "completion_rate",
            "premature_executions", "correct_executions",
            "ctrl_loss", "meta_loss",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in self._history:
                writer.writerow({
                    "episode": m.episode,
                    "episode_return": f"{m.episode_return:.4f}",
                    "total_steps": m.total_steps,
                    "nodes_completed": m.nodes_completed,
                    "nodes_skipped": m.nodes_skipped,
                    "workflow_size": m.workflow_size,
                    "workflow_complete": int(m.workflow_complete),
                    "completion_rate": f"{m.completion_rate:.4f}",
                    "premature_executions": m.premature_executions,
                    "correct_executions": m.correct_executions,
                    "ctrl_loss": f"{m.ctrl_loss:.6f}" if m.ctrl_loss is not None else "",
                    "meta_loss": f"{m.meta_loss:.6f}" if m.meta_loss is not None else "",
                })
        logger.info("Saved %d episodes to %s", len(self._history), path)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_multistep(
    env: MultiStepEnv,
    agent: HierarchicalDQNAgent,
    metrics_logger: MultiStepMetricsLogger,
    num_episodes: int,
    log_interval: int = 500,
) -> MultiStepMetricsLogger:
    """
    Train the hierarchical agent on the multi-step environment.

    Episode loop:
        For each node in the workflow:
            While not advanced to next node:
                1. Agent selects action (WAIT/EXECUTE/SKIP).
                2. Environment applies action, returns reward.
                3. Store (s, a, r, s', done) in appropriate buffer.
                4. Update controller and meta-controller networks.
    """
    mode = metrics_logger.mode
    seed = metrics_logger.seed
    logger.info("Starting multi-step training: mode=%s seed=%d episodes=%d", mode, seed, num_episodes)
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        episode_return = 0.0
        ctrl_losses: list[float] = []
        meta_losses: list[float] = []
        premature_count = 0
        correct_count = 0

        prev_node_idx = 0
        prev_meta_obs = obs.copy()
        meta_cumulative_reward = 0.0

        done = False
        while not done:
            node_step = info.get("node_step", 0)
            action = agent.select_action(obs, node_step)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward

            # Track execution quality
            if action == EXECUTE:
                if info.get("execute_correct", False):
                    correct_count += 1
                else:
                    premature_count += 1

            # --- Controller buffer (every step) ----------------------------
            ctrl_action = min(action, 1)  # map SKIP→EXECUTE for controller buffer
            if action != SKIP:
                agent.ctrl_buffer.push(obs, ctrl_action, reward, next_obs, done)
                loss = agent.update_controller()
                if loss is not None:
                    ctrl_losses.append(loss)

            # --- Meta-controller buffer (on node transitions) ---------------
            current_node_idx = info.get("current_node_idx", 0)
            meta_cumulative_reward += reward

            if current_node_idx != prev_node_idx or done:
                # A node transition happened — store meta-experience
                meta_action = 1 if action == SKIP else 0  # PROCEED=0, SKIP=1
                agent.meta_buffer.push(
                    prev_meta_obs, meta_action, meta_cumulative_reward, next_obs, done
                )
                meta_loss = agent.update_meta()
                if meta_loss is not None:
                    meta_losses.append(meta_loss)

                prev_node_idx = current_node_idx
                prev_meta_obs = next_obs.copy()
                meta_cumulative_reward = 0.0

            obs = next_obs

        # --- Record episode metrics ----------------------------------------
        wf_size = info.get("workflow_size", 1)
        metrics = MultiStepEpisodeMetrics(
            episode=episode,
            episode_return=episode_return,
            total_steps=info.get("total_step", 0),
            nodes_completed=info.get("nodes_completed", 0),
            nodes_skipped=info.get("nodes_skipped", 0),
            workflow_size=wf_size,
            workflow_complete=info.get("workflow_complete", False),
            completion_rate=info.get("nodes_completed", 0) / max(1, wf_size),
            premature_executions=premature_count,
            correct_executions=correct_count,
            ctrl_loss=float(np.mean(ctrl_losses)) if ctrl_losses else None,
            meta_loss=float(np.mean(meta_losses)) if meta_losses else None,
        )
        metrics_logger.log(metrics)

        # --- Periodic logging -----------------------------------------------
        if episode % log_interval == 0:
            elapsed = time.time() - start_time
            s = metrics_logger.summary(window=log_interval)
            logger.info(
                "[%s seed=%d] Episode %d/%d | "
                "Return: %.2f±%.2f | "
                "Completion: %.1f%% | "
                "Steps: %.1f | "
                "Premature: %.1f | "
                "Epsilon: %.3f | "
                "Time: %.1fs",
                mode, seed, episode, num_episodes,
                s["return_mean"], s["return_std"],
                s["completion_rate_mean"] * 100,
                s["length_mean"],
                s["premature_mean"],
                agent.epsilon,
                elapsed,
            )

    total_time = time.time() - start_time
    final = metrics_logger.summary(window=1000)
    logger.info(
        "[%s seed=%d] Training complete in %.1fs | "
        "Final return: %.2f | Completion: %.1f%%",
        mode, seed, total_time,
        final["return_mean"], final["completion_rate_mean"] * 100,
    )
    return metrics_logger


# ---------------------------------------------------------------------------
# Comparison Table
# ---------------------------------------------------------------------------

def print_multistep_comparison(
    baseline_loggers: list[MultiStepMetricsLogger],
    observer_loggers: list[MultiStepMetricsLogger],
    workflow_name: str,
    window: int = 1000,
) -> None:
    """Print a comparison table for multi-step results."""
    print(f"\n{'=' * 80}")
    print(f"  MULTI-STEP RESULTS — '{workflow_name}' Workflow (last {window} episodes)")
    print(f"{'=' * 80}")

    def agg(loggers: list[MultiStepMetricsLogger]) -> dict[str, tuple[float, float]]:
        summaries = [lg.summary(window=window) for lg in loggers]
        result = {}
        for k in ["return_mean", "completion_rate_mean", "length_mean", "premature_mean"]:
            vals = [s[k] for s in summaries]
            result[k] = (float(np.mean(vals)), float(np.std(vals)))
        return result

    bl = agg(baseline_loggers)
    ob = agg(observer_loggers)

    header = f"  {'Metric':<28s} {'Baseline':>18s} {'Observer':>18s} {'Improvement':>14s}"
    print(header)
    print("  " + "─" * 76)

    rows = [
        ("Episode Return", "return_mean", False),
        ("Completion Rate", "completion_rate_mean", True),
        ("Total Steps", "length_mean", False),
        ("Premature Executions", "premature_mean", False),
    ]

    for label, key, is_pct in rows:
        bm, bs = bl[key]
        om, os_ = ob[key]

        if is_pct:
            bl_str = f"{bm*100:5.1f}% ± {bs*100:4.1f}%"
            ob_str = f"{om*100:5.1f}% ± {os_*100:4.1f}%"
            diff = (om - bm) * 100
            imp_str = f"{diff:+.1f}pp"
        else:
            bl_str = f"{bm:6.2f} ± {bs:5.2f}"
            ob_str = f"{om:6.2f} ± {os_:5.2f}"
            diff = om - bm
            imp_str = f"{diff:+.2f}"

        print(f"  {label:<28s} {bl_str:>18s} {ob_str:>18s} {imp_str:>14s}")

    print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PGM State Observer — Multi-Step RL Benchmark"
    )
    parser.add_argument(
        "--workflow", type=str, default="form_submission",
        choices=list(WORKFLOW_REGISTRY.keys()) + ["all"],
        help="Workflow to benchmark (default: form_submission)",
    )
    parser.add_argument("--num-episodes", type=int, default=10_000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="results_multistep")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflows = (
        list(WORKFLOW_REGISTRY.keys()) if args.workflow == "all"
        else [args.workflow]
    )

    logger.info(
        "Experiment config: workflows=%s episodes=%d seeds=%d device=%s",
        workflows, args.num_episodes, args.num_seeds, args.device,
    )

    total_start = time.time()

    for wf_name in workflows:
        logger.info("━" * 60)
        logger.info("Workflow: %s", wf_name)
        logger.info("━" * 60)

        task_graph_factory = WORKFLOW_REGISTRY[wf_name]
        baseline_loggers: list[MultiStepMetricsLogger] = []
        observer_loggers: list[MultiStepMetricsLogger] = []

        for seed in range(args.num_seeds):
            logger.info("── Seed %d/%d ──", seed + 1, args.num_seeds)

            for mode in ["baseline", "observer"]:
                set_seed(seed)
                use_obs = (mode == "observer")

                task_graph = task_graph_factory()
                env = MultiStepEnv(task_graph=task_graph, use_observer=use_obs)
                obs_dim = env.observation_space.shape[0]

                agent = HierarchicalDQNAgent(obs_dim=obs_dim, device=args.device)
                metrics_log = MultiStepMetricsLogger(
                    mode=mode, seed=seed, workflow=wf_name
                )

                train_multistep(
                    env=env,
                    agent=agent,
                    metrics_logger=metrics_log,
                    num_episodes=args.num_episodes,
                )

                csv_path = output_dir / f"{wf_name}_{mode}_seed{seed}.csv"
                metrics_log.save_csv(csv_path)

                if mode == "baseline":
                    baseline_loggers.append(metrics_log)
                else:
                    observer_loggers.append(metrics_log)

        print_multistep_comparison(baseline_loggers, observer_loggers, wf_name)

    total_time = time.time() - total_start
    logger.info("Total experiment time: %.1fs", total_time)
    logger.info("Results saved to %s/", output_dir)


if __name__ == "__main__":
    main()
