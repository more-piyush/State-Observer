"""
dqn_agent.py — Deep Q-Network Agent.

A simple DQN with:
  - Two-layer MLP Q-network (64 hidden units each).
  - Epsilon-greedy exploration with linear decay.
  - Target network with periodic hard sync.
  - Experience replay (external ReplayBuffer).

The same architecture is used for both baseline (obs_dim=3) and
observer-augmented (obs_dim=9) modes — only the input dimension changes.
This eliminates architectural confounds so any performance difference
is attributable to the quality of the observation signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray

from agents.replay_buffer import ReplayBuffer, Transition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Two-layer MLP mapping observations to Q-values.

    Architecture:
        Input(obs_dim) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(action_dim)
    """

    def __init__(self, obs_dim: int, action_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

@dataclass
class DQNAgent:
    """
    DQN agent with epsilon-greedy exploration and target network.

    Args:
        obs_dim:           Observation vector dimension (3 or 9).
        action_dim:        Number of discrete actions (default 2: WAIT, ACT).
        lr:                Learning rate for Adam optimizer.
        gamma:             Discount factor.
        epsilon_start:     Initial exploration rate.
        epsilon_end:       Minimum exploration rate.
        epsilon_decay_steps: Number of steps over which epsilon decays linearly.
        target_sync_freq:  Steps between target network hard updates.
        batch_size:        Mini-batch size for replay sampling.
        buffer_capacity:   Maximum replay buffer size.
        device:            Torch device ("cpu" or "cuda").
    """

    obs_dim: int
    action_dim: int = 2
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    target_sync_freq: int = 100
    batch_size: int = 64
    buffer_capacity: int = 50_000
    device: str = "cpu"

    # --- Internal state (initialized in __post_init__) ----------------------
    _q_net: QNetwork = field(init=False, repr=False)
    _target_net: QNetwork = field(init=False, repr=False)
    _optimizer: optim.Adam = field(init=False, repr=False)
    _loss_fn: nn.MSELoss = field(init=False, repr=False)
    replay_buffer: ReplayBuffer = field(init=False, repr=False)
    _step_count: int = field(default=0, init=False)
    _update_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._q_net = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        self._target_net = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._target_net.eval()

        self._optimizer = optim.Adam(self._q_net.parameters(), lr=self.lr)
        self._loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)

    # -----------------------------------------------------------------------
    # Exploration Schedule
    # -----------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Current epsilon value (linear decay)."""
        progress = min(1.0, self._step_count / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    # -----------------------------------------------------------------------
    # Action Selection
    # -----------------------------------------------------------------------

    def select_action(self, obs: NDArray[np.float64]) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon, sample a random action.
        Otherwise, select the action with the highest Q-value.

        Args:
            obs: Current observation vector.

        Returns:
            Action index (0 = WAIT, 1 = ACT).
        """
        self._step_count += 1

        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self._q_net(obs_t)
            return int(q_values.argmax(dim=1).item())

    # -----------------------------------------------------------------------
    # Learning
    # -----------------------------------------------------------------------

    def update(self) -> float | None:
        """
        Perform one gradient step on a mini-batch from the replay buffer.

        DQN loss:
            L = E[(r + γ · max_a' Q_target(s', a') - Q(s, a))²]

        Returns:
            Loss value, or None if buffer doesn't have enough samples.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        # Unpack batch into tensors
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

        # Q(s, a) for the actions that were actually taken
        q_values = self._q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + γ · max_a' Q_target(s', a') · (1 - done)
        with torch.no_grad():
            next_q_max = self._target_net(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q_max * (1.0 - dones)

        loss = self._loss_fn(q_values, targets)

        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self._q_net.parameters(), max_norm=1.0)
        self._optimizer.step()

        self._update_count += 1

        # Periodic target network sync
        if self._update_count % self.target_sync_freq == 0:
            self._sync_target()

        return float(loss.item())

    def _sync_target(self) -> None:
        """Hard-copy Q-network weights to the target network."""
        self._target_net.load_state_dict(self._q_net.state_dict())
        logger.debug("Target network synced at update %d", self._update_count)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save Q-network weights to disk."""
        torch.save(self._q_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load Q-network weights from disk."""
        self._q_net.load_state_dict(torch.load(path, map_location=self.device))
        self._sync_target()
