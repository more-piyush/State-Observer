"""
hierarchical_agent.py — Hierarchical DQN Agent for Multi-Step Web Workflows.

Uses a two-level architecture inspired by the Options framework:

    ┌─────────────────────────────────────────────────────────────┐
    │  META-CONTROLLER (high-level)                               │
    │  "Which node should I focus on? Should I skip or proceed?"  │
    │  Input:  workflow context + aggregated page belief           │
    │  Output: meta-action ∈ {PROCEED, SKIP}                     │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  CONTROLLER (low-level)                                     │
    │  "Given this observation, should I WAIT or EXECUTE now?"    │
    │  Input:  full observation vector (raw + observer if enabled)│
    │  Output: action ∈ {WAIT, EXECUTE}                          │
    └─────────────────────────────────────────────────────────────┘

The meta-controller periodically evaluates whether to continue waiting
on the current node or skip it. The controller makes per-step WAIT/EXECUTE
decisions. Both share the same replay buffer and training schedule.

In practice, the meta-controller fires every `meta_decision_interval` steps.
Between meta-decisions, the controller runs autonomously.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray

from agents.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Q-Networks
# ---------------------------------------------------------------------------

class ControllerQNetwork(nn.Module):
    """
    Low-level controller: obs → Q(WAIT) / Q(EXECUTE).

    Architecture: Input(obs_dim) → 128 → ReLU → 64 → ReLU → 2
    Larger than the single-step DQN because the observation space is richer
    (includes workflow context features).
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),   # WAIT=0, EXECUTE=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaQNetwork(nn.Module):
    """
    High-level meta-controller: obs → Q(PROCEED) / Q(SKIP).

    Architecture: Input(obs_dim) → 64 → ReLU → 32 → ReLU → 2
    Smaller network — meta-decisions are less frequent and simpler.
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),   # PROCEED=0, SKIP=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Hierarchical DQN Agent
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalDQNAgent:
    """
    Two-level DQN agent for multi-step web workflows.

    The agent uses two Q-networks:
      1. Controller: makes per-step WAIT/EXECUTE decisions.
      2. Meta-controller: periodically decides PROCEED/SKIP for the current node.

    Args:
        obs_dim:               Full observation dimension.
        lr:                    Learning rate.
        gamma:                 Discount factor.
        epsilon_start:         Initial exploration rate.
        epsilon_end:           Minimum exploration rate.
        epsilon_decay_steps:   Linear decay schedule length.
        target_sync_freq:      Steps between target network updates.
        meta_decision_interval: Steps between meta-controller evaluations.
        batch_size:            Mini-batch size.
        buffer_capacity:       Replay buffer size.
        device:                Torch device.
    """

    obs_dim: int
    lr: float = 5e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    target_sync_freq: int = 200
    meta_decision_interval: int = 10
    batch_size: int = 64
    buffer_capacity: int = 100_000
    device: str = "cpu"

    # --- Internal state -----------------------------------------------------
    _ctrl_net: ControllerQNetwork = field(init=False, repr=False)
    _ctrl_target: ControllerQNetwork = field(init=False, repr=False)
    _ctrl_optim: optim.Adam = field(init=False, repr=False)

    _meta_net: MetaQNetwork = field(init=False, repr=False)
    _meta_target: MetaQNetwork = field(init=False, repr=False)
    _meta_optim: optim.Adam = field(init=False, repr=False)

    _loss_fn: nn.MSELoss = field(init=False, repr=False)

    ctrl_buffer: ReplayBuffer = field(init=False, repr=False)
    meta_buffer: ReplayBuffer = field(init=False, repr=False)

    _step_count: int = field(default=0, init=False)
    _ctrl_updates: int = field(default=0, init=False)
    _meta_updates: int = field(default=0, init=False)
    _current_meta_action: int = field(default=0, init=False)  # 0=PROCEED

    def __post_init__(self) -> None:
        # Controller networks
        self._ctrl_net = ControllerQNetwork(self.obs_dim).to(self.device)
        self._ctrl_target = ControllerQNetwork(self.obs_dim).to(self.device)
        self._ctrl_target.load_state_dict(self._ctrl_net.state_dict())
        self._ctrl_target.eval()
        self._ctrl_optim = optim.Adam(self._ctrl_net.parameters(), lr=self.lr)

        # Meta-controller networks
        self._meta_net = MetaQNetwork(self.obs_dim).to(self.device)
        self._meta_target = MetaQNetwork(self.obs_dim).to(self.device)
        self._meta_target.load_state_dict(self._meta_net.state_dict())
        self._meta_target.eval()
        self._meta_optim = optim.Adam(self._meta_net.parameters(), lr=self.lr)

        self._loss_fn = nn.MSELoss()
        self.ctrl_buffer = ReplayBuffer(capacity=self.buffer_capacity)
        self.meta_buffer = ReplayBuffer(capacity=self.buffer_capacity // 4)

    # -----------------------------------------------------------------------
    # Exploration
    # -----------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Current epsilon (shared between both levels)."""
        progress = min(1.0, self._step_count / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    # -----------------------------------------------------------------------
    # Action Selection
    # -----------------------------------------------------------------------

    def select_action(
        self,
        obs: NDArray[np.float64],
        node_step: int,
    ) -> int:
        """
        Hierarchical action selection.

        Every `meta_decision_interval` steps within a node, the meta-controller
        re-evaluates whether to PROCEED (keep trying) or SKIP this node.

        Between meta-decisions, the controller chooses WAIT or EXECUTE.

        Args:
            obs:       Current observation vector.
            node_step: Number of steps spent on the current node.

        Returns:
            Action for the environment: WAIT(0), EXECUTE(1), or SKIP(2).
        """
        self._step_count += 1

        # --- Meta-controller decision (periodic) ----------------------------
        if node_step > 0 and node_step % self.meta_decision_interval == 0:
            meta_action = self._select_meta_action(obs)
            self._current_meta_action = meta_action

            if meta_action == 1:  # SKIP
                return 2  # env SKIP action

        # --- Controller decision (every step) --------------------------------
        ctrl_action = self._select_ctrl_action(obs)
        return ctrl_action  # 0=WAIT, 1=EXECUTE (maps directly to env actions)

    def _select_ctrl_action(self, obs: NDArray[np.float64]) -> int:
        """Epsilon-greedy action from the controller Q-network."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(2))

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self._ctrl_net(obs_t)
            return int(q_values.argmax(dim=1).item())

    def _select_meta_action(self, obs: NDArray[np.float64]) -> int:
        """Epsilon-greedy action from the meta-controller Q-network."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(2))

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self._meta_net(obs_t)
            return int(q_values.argmax(dim=1).item())

    # -----------------------------------------------------------------------
    # Learning
    # -----------------------------------------------------------------------

    def update_controller(self) -> float | None:
        """Gradient step for the low-level controller."""
        return self._update_network(
            self._ctrl_net, self._ctrl_target, self._ctrl_optim,
            self.ctrl_buffer, "ctrl"
        )

    def update_meta(self) -> float | None:
        """Gradient step for the meta-controller."""
        return self._update_network(
            self._meta_net, self._meta_target, self._meta_optim,
            self.meta_buffer, "meta"
        )

    def _update_network(
        self,
        q_net: nn.Module,
        target_net: nn.Module,
        optimizer: optim.Adam,
        buffer: ReplayBuffer,
        name: str,
    ) -> float | None:
        """
        Shared DQN update logic for both levels.

        L = E[(r + γ · max_a' Q_target(s', a') - Q(s, a))²]
        """
        if len(buffer) < self.batch_size:
            return None

        batch = buffer.sample(self.batch_size)

        states = torch.tensor(
            np.array([t.state for t in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device
        )

        q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_max = target_net(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q_max * (1.0 - dones)

        loss = self._loss_fn(q_values, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
        optimizer.step()

        # Target sync tracking
        if name == "ctrl":
            self._ctrl_updates += 1
            if self._ctrl_updates % self.target_sync_freq == 0:
                target_net.load_state_dict(q_net.state_dict())
        else:
            self._meta_updates += 1
            if self._meta_updates % self.target_sync_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

        return float(loss.item())

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path_prefix: str) -> None:
        """Save both networks."""
        torch.save(self._ctrl_net.state_dict(), f"{path_prefix}_ctrl.pt")
        torch.save(self._meta_net.state_dict(), f"{path_prefix}_meta.pt")

    def load(self, path_prefix: str) -> None:
        """Load both networks."""
        self._ctrl_net.load_state_dict(
            torch.load(f"{path_prefix}_ctrl.pt", map_location=self.device)
        )
        self._ctrl_target.load_state_dict(self._ctrl_net.state_dict())
        self._meta_net.load_state_dict(
            torch.load(f"{path_prefix}_meta.pt", map_location=self.device)
        )
        self._meta_target.load_state_dict(self._meta_net.state_dict())
