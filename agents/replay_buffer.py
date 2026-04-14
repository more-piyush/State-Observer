"""
replay_buffer.py — Experience Replay Buffer for DQN Training.

Stores (state, action, reward, next_state, done) transitions in a circular
buffer and supports uniform random sampling for mini-batch updates.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class Transition(NamedTuple):
    """A single experience tuple."""
    state: NDArray[np.float64]
    action: int
    reward: float
    next_state: NDArray[np.float64]
    done: bool


class ReplayBuffer:
    """
    Fixed-size circular replay buffer with uniform random sampling.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: NDArray[np.float64],
        action: int,
        reward: float,
        next_state: NDArray[np.float64],
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self._buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample a random mini-batch of transitions."""
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)
