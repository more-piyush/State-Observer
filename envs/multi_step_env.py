"""
multi_step_env.py — Gymnasium Environment for Sequential Multi-Action Workflows.

Extends the single-decision PageLoadEnv to support multi-step web automation
tasks defined by a TaskGraph. The agent must execute a sequence of actions
(navigate → fill → submit → verify), each requiring the page to be in the
correct latent state before proceeding.

Architecture: Semi-MDP (Options Framework)
─────────────────────────────────────────────
Each TaskNode acts like an "option" in the Options framework:

  For each node in the workflow:
    ┌─ SUB-EPISODE ──────────────────────────────────────────────┐
    │                                                            │
    │  While page_state ≠ node.required_state:                   │
    │      Agent chooses: WAIT (0) | EXECUTE (1) | SKIP (2)      │
    │      - WAIT:    observe, let page state evolve              │
    │      - EXECUTE: attempt the action (rewarded/penalized)     │
    │      - SKIP:    abandon this node (emergency escape)        │
    │                                                            │
    │  If EXECUTE succeeds:                                      │
    │      Page state resets via node.post_state_dist             │
    │      Observer belief resets (new page context)              │
    │      → Advance to next node                                │
    └────────────────────────────────────────────────────────────┘

Observation Vector:
    Baseline (dim=6):
        [pixel_bin, dom_bin, latency_bin,
         current_node / max_nodes,    ← workflow progress
         node_steps / max_wait,       ← time pressure within node
         required_state_encoding]     ← what state the node needs

    Augmented (dim=12):
        [baseline dims...,
         belief[0..3],                ← PGM posterior
         entropy / 2.0,              ← uncertainty
         should_proceed]             ← observer gate

Action Space: Discrete(3)
    0 = WAIT    — observe one more step, let the page evolve
    1 = EXECUTE — attempt the current action
    2 = SKIP    — abandon the current node (partial credit, move on)
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
from envs.task_graph import (
    TaskGraph,
    TaskNode,
    create_form_submission_workflow,
)

logger = logging.getLogger(__name__)

# Action constants
WAIT: int = 0
EXECUTE: int = 1
SKIP: int = 2

# State encoding map for the observation vector
_STATE_ENCODING: dict[str, float] = {
    "READY": 0.0,
    "LOADING": 0.33,
    "STALLED": 0.66,
    "SUCCESS_PENDING": 1.0,
}


class MultiStepEnv(gym.Env):
    """
    Multi-step web automation environment.

    The agent progresses through an ordered TaskGraph, deciding when to
    EXECUTE each action based on noisy page observations. Each action
    triggers a new page state transition, requiring the agent to re-assess
    readiness before the next action.

    Args:
        task_graph:   The workflow to execute (default: form_submission).
        use_observer: Augment observations with PGM belief state.
        render_mode:  "human" prints to console.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # --- Reward Constants ---------------------------------------------------
    # Correctly executing an action when the page is in the required state
    REWARD_CORRECT_EXECUTE: float = 10.0
    # Attempting to execute when the page is NOT in the required state
    REWARD_WRONG_EXECUTE: float = -5.0
    # Completing the entire workflow
    REWARD_WORKFLOW_COMPLETE: float = 20.0
    # Skipping a node (partial task failure)
    REWARD_SKIP: float = -8.0
    # Waiting when the page IS in the required state (missed opportunity)
    REWARD_WAIT_WHEN_READY: float = -0.1
    # Waiting when the page is NOT in the required state (patience)
    REWARD_WAIT_WHEN_NOT_READY: float = 0.0
    # Per-step time penalty
    REWARD_TIME_PENALTY: float = -0.03
    # Per-node timeout penalty
    REWARD_NODE_TIMEOUT: float = -4.0
    # Per-step workflow progress bonus (dense shaping)
    # Incentivizes forward progress through the workflow
    REWARD_PROGRESS_BONUS: float = 2.0

    def __init__(
        self,
        task_graph: Optional[TaskGraph] = None,
        use_observer: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.task_graph = task_graph or create_form_submission_workflow()
        self.use_observer = use_observer
        self.render_mode = render_mode

        # --- Action space: WAIT (0), EXECUTE (1), SKIP (2) -----------------
        self.action_space = spaces.Discrete(3)

        # --- Observation space ----------------------------------------------
        # Baseline: 3 (raw obs) + 3 (workflow context) = 6
        # Augmented: 6 + 4 (belief) + 1 (entropy) + 1 (should_proceed) = 12
        self._obs_dim = 12 if use_observer else 6
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float64
        )

        # --- Model parameters -----------------------------------------------
        self._transition = config.TRANSITION_MATRIX.copy()
        self._emit_pixel = config.EMISSION_PIXEL_DELTA.copy()
        self._emit_dom = config.EMISSION_DOM_SIGNAL.copy()
        self._emit_latency = config.EMISSION_NETWORK_LATENCY.copy()

        # --- Internal state -------------------------------------------------
        self._hidden_state: int = 0
        self._current_node_idx: int = 0
        self._node_step_count: int = 0       # steps within current node
        self._total_step_count: int = 0      # steps across entire episode
        self._nodes_completed: int = 0
        self._nodes_skipped: int = 0
        self._observer: Optional[StateObserver] = None
        self._rng: np.random.Generator = np.random.default_rng()

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def current_node(self) -> Optional[TaskNode]:
        """The TaskNode the agent is currently working on."""
        return self.task_graph.get_node(self._current_node_idx)

    @property
    def workflow_complete(self) -> bool:
        """Whether all nodes have been processed (completed or skipped)."""
        return self._current_node_idx >= self.task_graph.num_steps

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Reset to the beginning of the workflow."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Start in LOADING state (typical for page navigation)
        self._hidden_state = config.STATE_INDEX["LOADING"]
        self._current_node_idx = 0
        self._node_step_count = 0
        self._total_step_count = 0
        self._nodes_completed = 0
        self._nodes_skipped = 0

        # Fresh observer for each episode
        if self.use_observer:
            self._observer = StateObserver()

        obs_vec = self._sample_observation()
        obs = self._build_observation(obs_vec)
        info = self._build_info(obs_vec)

        return obs, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the multi-step workflow.

        The step logic depends on the action:
          - WAIT:    Page state evolves via transition matrix. Agent observes.
          - EXECUTE: Agent attempts the current node's action.
                     If page is in the required state → success → advance.
                     If page is NOT in the required state → failure → penalized.
          - SKIP:    Agent abandons the current node and advances.

        After EXECUTE (success) or SKIP, the page state is reset according to
        the node's post_state_dist, and the observer belief is reset (new context).

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._total_step_count += 1
        self._node_step_count += 1

        node = self.current_node
        assert node is not None, "step() called after workflow complete"

        is_required_state = (
            config.STATES[self._hidden_state] == node.required_state
        )
        reward = self.REWARD_TIME_PENALTY
        terminated = False
        truncated = False
        advance_node = False

        # --- Process action -------------------------------------------------
        if action == WAIT:
            # Just observe — page state evolves naturally
            if is_required_state:
                reward += self.REWARD_WAIT_WHEN_READY
            else:
                reward += self.REWARD_WAIT_WHEN_NOT_READY

            # Check node-level timeout
            if self._node_step_count >= node.max_wait_steps:
                reward += self.REWARD_NODE_TIMEOUT
                advance_node = True
                self._nodes_skipped += 1
                logger.debug(
                    "Node '%s' timed out after %d steps",
                    node.name, self._node_step_count,
                )

        elif action == EXECUTE:
            if is_required_state:
                # Correct execution — the page was ready for this action
                reward += self.REWARD_CORRECT_EXECUTE
                reward += self.REWARD_PROGRESS_BONUS
                advance_node = True
                self._nodes_completed += 1
                logger.debug("Node '%s' executed successfully", node.name)
            else:
                # Premature execution — page was not in the required state
                reward += self.REWARD_WRONG_EXECUTE
                # Don't advance — agent stays on this node and must retry
                logger.debug(
                    "Node '%s' executed prematurely (state=%s, required=%s)",
                    node.name, config.STATES[self._hidden_state], node.required_state,
                )

        elif action == SKIP:
            reward += self.REWARD_SKIP
            advance_node = True
            self._nodes_skipped += 1
            logger.debug("Node '%s' skipped by agent", node.name)

        # --- Advance to next node if needed ---------------------------------
        if advance_node:
            # Apply the post-action state distribution
            # This models the page reacting to the action (e.g., form submit → LOADING)
            self._hidden_state = int(
                self._rng.choice(config.NUM_STATES, p=node.post_state_dist)
            )
            self._current_node_idx += 1
            self._node_step_count = 0

            # Reset observer belief — new page context after action
            if self._observer is not None:
                self._observer.reset()

            # Check workflow completion
            if self.workflow_complete:
                reward += self.REWARD_WORKFLOW_COMPLETE
                terminated = True
        else:
            # Normal page state evolution via transition matrix
            transition_probs = self._transition[self._hidden_state]
            self._hidden_state = int(
                self._rng.choice(config.NUM_STATES, p=transition_probs)
            )

        # --- Global safety truncation ---------------------------------------
        max_total_steps = sum(n.max_wait_steps for n in self.task_graph.nodes) + 50
        if self._total_step_count >= max_total_steps and not terminated:
            truncated = True
            reward += self.REWARD_NODE_TIMEOUT

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
        """Sample noisy observations from emission matrices given hidden state."""
        s = self._hidden_state
        pixel_bin = int(self._rng.choice(
            len(config.PIXEL_DELTA_BINS), p=self._emit_pixel[s]
        ))
        dom_bin = int(self._rng.choice(
            len(config.DOM_SIGNAL_VALUES), p=self._emit_dom[s]
        ))
        lat_bin = int(self._rng.choice(
            len(config.NETWORK_LATENCY_BINS), p=self._emit_latency[s]
        ))
        return ObservationVector(pixel_bin, dom_bin, lat_bin)

    def _build_observation(
        self, obs_vec: ObservationVector
    ) -> NDArray[np.float64]:
        """
        Build the full observation vector.

        Baseline (dim=6):
            [pixel, dom, latency, workflow_progress, node_time_pressure, required_state]

        Augmented (dim=12):
            [baseline..., belief[0:4], entropy, should_proceed]
        """
        node = self.current_node
        num_nodes = max(1, self.task_graph.num_steps)

        # Workflow context features (normalized to [0, 1])
        workflow_progress = self._current_node_idx / num_nodes
        node_time_pressure = (
            self._node_step_count / node.max_wait_steps if node else 1.0
        )
        required_state_enc = _STATE_ENCODING.get(
            node.required_state if node else "READY", 0.0
        )

        raw = np.array([
            obs_vec.pixel_delta_bin / max(1, len(config.PIXEL_DELTA_BINS) - 1),
            obs_vec.dom_signal_bin / max(1, len(config.DOM_SIGNAL_VALUES) - 1),
            obs_vec.latency_bin / max(1, len(config.NETWORK_LATENCY_BINS) - 1),
            workflow_progress,
            min(1.0, node_time_pressure),
            required_state_enc,
        ], dtype=np.float64)

        if not self.use_observer:
            return raw

        # PGM observer augmentation
        assert self._observer is not None
        snapshot = self._observer.update_belief(obs_vec)

        augmented = np.concatenate([
            raw,
            snapshot.belief,
            np.array([snapshot.entropy / 2.0], dtype=np.float64),
            np.array([1.0 if snapshot.confidence > config.READY_CONFIDENCE_THRESHOLD
                       and snapshot.most_likely_state == (
                           node.required_state if node else "READY"
                       )
                       else 0.0], dtype=np.float64),
        ])
        return augmented

    def _build_info(
        self,
        obs_vec: ObservationVector,
        action: Optional[int] = None,
        reward: Optional[float] = None,
    ) -> dict[str, Any]:
        """Build info dict with ground-truth diagnostics."""
        node = self.current_node
        info: dict[str, Any] = {
            "hidden_state": config.STATES[self._hidden_state],
            "hidden_state_idx": self._hidden_state,
            "current_node": node.name if node else "DONE",
            "current_node_idx": self._current_node_idx,
            "node_step": self._node_step_count,
            "total_step": self._total_step_count,
            "nodes_completed": self._nodes_completed,
            "nodes_skipped": self._nodes_skipped,
            "workflow_size": self.task_graph.num_steps,
            "workflow_complete": self.workflow_complete,
            "observation_raw": obs_vec,
        }

        if node is not None:
            is_required = config.STATES[self._hidden_state] == node.required_state
            info["is_required_state"] = is_required
            info["required_state"] = node.required_state

        if action is not None:
            action_names = {WAIT: "WAIT", EXECUTE: "EXECUTE", SKIP: "SKIP"}
            info["action"] = action_names.get(action, str(action))
            if action == EXECUTE and node is not None:
                info["execute_correct"] = (
                    config.STATES[self._hidden_state] == node.required_state
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
        """Print a single-line step status to console."""
        obs = info["observation_raw"]
        pixel_lbl = config.PIXEL_DELTA_BINS[obs.pixel_delta_bin]
        dom_lbl = "PRESENT" if obs.dom_signal_bin == 0 else "ABSENT"
        lat_lbl = config.NETWORK_LATENCY_BINS[obs.latency_bin]
        action_lbl = info.get("action", "---")
        node_lbl = info.get("current_node", "DONE")
        hidden = info["hidden_state"]
        progress = f"{info['nodes_completed']}/{info['workflow_size']}"

        print(
            f"  Step {info['total_step']:>3d} | "
            f"Node: {node_lbl:<15s} [{progress}] | "
            f"Hidden: {hidden:<17s} | "
            f"Obs: [{pixel_lbl:<14s} {dom_lbl:<8s} {lat_lbl:<8s}] | "
            f"Action: {action_lbl}"
        )
