"""
page_load_env.py — Gymnasium Environment Simulating Web Page Load Dynamics.

Implements a faithful simulation of the page-loading process using the same
Dynamic Bayesian Network parameters defined in config.py. The environment
maintains a hidden latent state S_t that transitions according to
config.TRANSITION_MATRIX, and emits noisy observations sampled from the
three emission matrices (pixel_delta, dom_signal, network_latency).

Two observation modes:
  - Baseline:  agent sees only raw discretized observations (dim=3).
  - Augmented: agent sees raw obs + PGM belief vector + entropy +
               should_proceed flag (dim=9).

Action space: Discrete(2) — WAIT (0) or ACT (1).
Episode ends when the agent takes ACT or MAX_OBSERVATION_STEPS is reached.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

import config
from state_observer import StateObserver, ObservationVector

logger = logging.getLogger(__name__)

# Action constants
WAIT: int = 0
ACT: int = 1


class PageLoadEnv(gym.Env):
    """
    Simulated web page loading environment for RL training.

    The hidden state evolves via the DBN transition matrix from config.py.
    Observations are sampled from the emission matrices each step.

    Args:
        use_observer: If True, augment observations with PGM belief state.
        max_steps:    Maximum steps per episode before truncation.
        render_mode:  "human" prints to console, "ansi" returns string.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # --- Reward constants ---------------------------------------------------
    # ACT in READY state: correct action timing
    REWARD_CORRECT_ACT: float = 10.0
    # ACT in non-READY state: premature / incorrect action
    REWARD_WRONG_ACT: float = -5.0
    # WAIT when READY: missed opportunity cost
    REWARD_WAIT_READY: float = -0.1
    # WAIT when not READY: patience is free
    REWARD_WAIT_NOT_READY: float = 0.0
    # Per-step time penalty to encourage efficiency
    REWARD_TIME_PENALTY: float = -0.05
    # Truncation penalty (hit max steps without acting)
    REWARD_TRUNCATION: float = -3.0

    def __init__(
        self,
        use_observer: bool = False,
        max_steps: int = config.MAX_OBSERVATION_STEPS,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.use_observer = use_observer
        self.max_steps = max_steps
        self.render_mode = render_mode

        # --- Action space: WAIT (0) or ACT (1) -----------------------------
        self.action_space = spaces.Discrete(2)

        # --- Observation space ----------------------------------------------
        # Baseline: 3 discrete bins normalized to [0, 1]
        #   pixel_delta_bin ∈ {0, 1}  → normalized to [0, 1]
        #   dom_signal_bin  ∈ {0, 1}  → normalized to [0, 1]
        #   latency_bin     ∈ {0, 1, 2} → normalized to [0, 1]
        #
        # Augmented adds 6 more dimensions:
        #   belief[0..3] ∈ [0, 1]   (4 state probabilities)
        #   entropy      ∈ [0, 1]   (normalized by max entropy = 2.0 bits)
        #   should_proceed ∈ {0, 1}
        self._obs_dim = 9 if use_observer else 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float64
        )

        # --- Model parameters (from config) ---------------------------------
        self._transition = config.TRANSITION_MATRIX.copy()
        self._emit_pixel = config.EMISSION_PIXEL_DELTA.copy()
        self._emit_dom = config.EMISSION_DOM_SIGNAL.copy()
        self._emit_latency = config.EMISSION_NETWORK_LATENCY.copy()

        # --- Internal state -------------------------------------------------
        self._hidden_state: int = 0
        self._step_count: int = 0
        self._observer: Optional[StateObserver] = None
        self._rng: np.random.Generator = np.random.default_rng()

        # Initial hidden state distribution: biased toward LOADING
        # P(S_0) = [0.0, 0.7, 0.2, 0.1] — pages typically start loading
        self._initial_state_dist: NDArray[np.float64] = np.array(
            [0.0, 0.7, 0.2, 0.1], dtype=np.float64
        )

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Samples a hidden state from the initial distribution and returns
        the first observation (no action taken yet).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample initial hidden state
        self._hidden_state = int(
            self._rng.choice(config.NUM_STATES, p=self._initial_state_dist)
        )
        self._step_count = 0

        # Reset the PGM observer if in augmented mode
        if self.use_observer:
            self._observer = StateObserver()

        # Generate initial observation from the hidden state
        obs_vec = self._sample_observation()
        obs = self._build_observation(obs_vec)

        info = self._build_info(obs_vec)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        """
        Execute one environment step.

        1. Evaluate the agent's action against the current hidden state.
        2. Transition the hidden state via the DBN transition matrix.
        3. Sample new observations from emission matrices.
        4. Optionally feed observations through the PGM observer.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        is_ready = (self._hidden_state == config.STATE_INDEX["READY"])

        # --- Compute reward -------------------------------------------------
        if action == ACT:
            reward = self.REWARD_CORRECT_ACT if is_ready else self.REWARD_WRONG_ACT
            terminated = True
        else:
            # WAIT
            reward = self.REWARD_WAIT_READY if is_ready else self.REWARD_WAIT_NOT_READY
            reward += self.REWARD_TIME_PENALTY
            terminated = False

        # --- Check truncation -----------------------------------------------
        truncated = (self._step_count >= self.max_steps) and not terminated
        if truncated:
            reward += self.REWARD_TRUNCATION

        # --- Transition hidden state ----------------------------------------
        # S_t ~ P(S_t | S_{t-1}) = row self._hidden_state of transition matrix
        if not terminated and not truncated:
            transition_probs = self._transition[self._hidden_state]
            self._hidden_state = int(
                self._rng.choice(config.NUM_STATES, p=transition_probs)
            )

        # --- Sample observation from new hidden state -----------------------
        obs_vec = self._sample_observation()
        obs = self._build_observation(obs_vec)
        info = self._build_info(obs_vec, action=action, reward=reward)

        if self.render_mode == "human":
            self._render_human(info)

        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _sample_observation(self) -> ObservationVector:
        """
        Sample discrete observations from emission matrices given hidden state.

        For each signal channel, we draw from the categorical distribution
        P(O_k | S_t = self._hidden_state).
        """
        s = self._hidden_state

        # pixel_delta_bin ~ P(pixel | S_t)
        pixel_bin = int(self._rng.choice(
            len(config.PIXEL_DELTA_BINS), p=self._emit_pixel[s]
        ))

        # dom_signal_bin ~ P(dom | S_t)
        dom_bin = int(self._rng.choice(
            len(config.DOM_SIGNAL_VALUES), p=self._emit_dom[s]
        ))

        # latency_bin ~ P(latency | S_t)
        lat_bin = int(self._rng.choice(
            len(config.NETWORK_LATENCY_BINS), p=self._emit_latency[s]
        ))

        return ObservationVector(pixel_bin, dom_bin, lat_bin)

    def _build_observation(
        self, obs_vec: ObservationVector
    ) -> NDArray[np.float64]:
        """
        Convert an ObservationVector into a normalized float array.

        Baseline (dim=3):
            [pixel_bin/1, dom_bin/1, latency_bin/2]

        Augmented (dim=9):
            [pixel_bin/1, dom_bin/1, latency_bin/2,
             belief[0], belief[1], belief[2], belief[3],
             entropy/2.0, should_proceed]
        """
        # Normalize discrete bins to [0, 1]
        raw = np.array([
            obs_vec.pixel_delta_bin / max(1, len(config.PIXEL_DELTA_BINS) - 1),
            obs_vec.dom_signal_bin / max(1, len(config.DOM_SIGNAL_VALUES) - 1),
            obs_vec.latency_bin / max(1, len(config.NETWORK_LATENCY_BINS) - 1),
        ], dtype=np.float64)

        if not self.use_observer:
            return raw

        # Feed observation to PGM observer and extract augmented features
        assert self._observer is not None
        snapshot = self._observer.update_belief(obs_vec)

        augmented = np.concatenate([
            raw,
            snapshot.belief,                                    # 4 floats
            np.array([snapshot.entropy / 2.0], dtype=np.float64),  # normalized
            np.array([1.0 if snapshot.confidence > config.READY_CONFIDENCE_THRESHOLD
                       and snapshot.most_likely_state == "READY"
                       else 0.0], dtype=np.float64),           # should_proceed
        ])
        return augmented

    def _build_info(
        self,
        obs_vec: ObservationVector,
        action: Optional[int] = None,
        reward: Optional[float] = None,
    ) -> dict[str, Any]:
        """Build the info dict with ground-truth data for logging."""
        info: dict[str, Any] = {
            "hidden_state": config.STATES[self._hidden_state],
            "hidden_state_idx": self._hidden_state,
            "is_ready": self._hidden_state == config.STATE_INDEX["READY"],
            "step": self._step_count,
            "observation_raw": obs_vec,
        }
        if action is not None:
            info["action"] = "ACT" if action == ACT else "WAIT"
            info["action_correct"] = (
                (action == ACT and info["is_ready"]) or
                (action == WAIT and not info["is_ready"])
            )
        if reward is not None:
            info["reward"] = reward
        if self._observer is not None:
            state, conf = self._observer.most_likely_state()
            info["observer_state"] = state
            info["observer_confidence"] = conf
            info["observer_entropy"] = self._observer.entropy()
            info["observer_should_proceed"] = self._observer.should_proceed()
        return info

    def _render_human(self, info: dict[str, Any]) -> None:
        """Print a single-line status to stdout."""
        obs = info["observation_raw"]
        pixel_lbl = config.PIXEL_DELTA_BINS[obs.pixel_delta_bin]
        dom_lbl = "PRESENT" if obs.dom_signal_bin == 0 else "ABSENT"
        lat_lbl = config.NETWORK_LATENCY_BINS[obs.latency_bin]
        action_lbl = info.get("action", "---")
        hidden = info["hidden_state"]
        print(
            f"  Step {info['step']:>3d} | "
            f"Hidden: {hidden:<17s} | "
            f"Obs: [{pixel_lbl:<14s} {dom_lbl:<8s} {lat_lbl:<8s}] | "
            f"Action: {action_lbl}"
        )
