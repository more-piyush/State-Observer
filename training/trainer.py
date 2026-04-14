"""
trainer.py — Training Loop Orchestrator.

Runs a configurable number of episodes for a given (env, agent) pair,
collecting per-episode metrics and printing periodic progress updates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from envs.page_load_env import PageLoadEnv, ACT
from agents.dqn_agent import DQNAgent
from training.metrics import MetricsLogger, EpisodeMetrics

logger = logging.getLogger(__name__)


@dataclass
class Trainer:
    """
    Orchestrates the RL training loop.

    Args:
        env:            The Gymnasium environment.
        agent:          The DQN agent.
        metrics_logger: Logger for per-episode metrics.
        num_episodes:   Total number of training episodes.
        log_interval:   Print progress every N episodes.
    """

    env: PageLoadEnv
    agent: DQNAgent
    metrics_logger: MetricsLogger
    num_episodes: int = 10_000
    log_interval: int = 500

    def train(self) -> MetricsLogger:
        """
        Run the full training loop.

        For each episode:
          1. Reset environment.
          2. Agent selects actions until terminated or truncated.
          3. Store transitions in replay buffer.
          4. Perform DQN updates after each step.
          5. Log episode metrics.

        Returns:
            The MetricsLogger with all recorded episode data.
        """
        mode = self.metrics_logger.mode
        seed = self.metrics_logger.seed
        logger.info(
            "Starting training: mode=%s seed=%d episodes=%d",
            mode, seed, self.num_episodes,
        )
        start_time = time.time()

        for episode in range(1, self.num_episodes + 1):
            ep_return, ep_length, ep_losses = self._run_episode(episode)

            # Log progress
            if episode % self.log_interval == 0:
                elapsed = time.time() - start_time
                summary = self.metrics_logger.summary(window=self.log_interval)
                logger.info(
                    "[%s seed=%d] Episode %d/%d | "
                    "Return: %.2f±%.2f | "
                    "Length: %.1f | "
                    "Accuracy: %.1f%% | "
                    "Premature: %.1f%% | "
                    "Epsilon: %.3f | "
                    "Time: %.1fs",
                    mode, seed, episode, self.num_episodes,
                    summary["return_mean"], summary["return_std"],
                    summary["length_mean"],
                    summary["action_accuracy"] * 100,
                    summary["premature_rate"] * 100,
                    self.agent.epsilon,
                    elapsed,
                )

        total_time = time.time() - start_time
        final_summary = self.metrics_logger.summary(window=1000)
        logger.info(
            "[%s seed=%d] Training complete in %.1fs | "
            "Final return: %.2f | Accuracy: %.1f%%",
            mode, seed, total_time,
            final_summary["return_mean"],
            final_summary["action_accuracy"] * 100,
        )

        return self.metrics_logger

    def _run_episode(self, episode_num: int) -> tuple[float, int, list[float]]:
        """
        Execute a single episode.

        Returns:
            (episode_return, episode_length, losses)
        """
        obs, info = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        losses: list[float] = []
        last_info = info

        done = False
        while not done:
            action = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.agent.replay_buffer.push(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
            )

            # Learn from replay buffer
            loss = self.agent.update()
            if loss is not None:
                losses.append(loss)

            episode_return += reward
            episode_length += 1
            obs = next_obs
            last_info = info

        # Record episode metrics
        acted = last_info.get("action") == "ACT"
        action_correct = last_info.get("action_correct", False)
        premature = acted and not last_info.get("is_ready", False)

        metrics = EpisodeMetrics(
            episode=episode_num,
            episode_return=episode_return,
            episode_length=episode_length,
            action_correct=action_correct,
            acted=acted,
            premature_action=premature,
            final_hidden_state=last_info.get("hidden_state", "UNKNOWN"),
            loss=float(np.mean(losses)) if losses else None,
        )
        self.metrics_logger.log(metrics)

        return episode_return, episode_length, losses
