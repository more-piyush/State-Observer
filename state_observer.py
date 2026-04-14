"""
state_observer.py — Dynamic Bayesian Network State Observer.

Implements a forward-algorithm belief tracker over the latent state space
of a web application. At each time step the observer:

  1. **Predicts** — propagates the belief through the transition model:
         b̄(s_t) = Σ_{s_{t-1}} P(s_t | s_{t-1}) · b(s_{t-1})

  2. **Updates** — incorporates the observation likelihood:
         b(s_t) ∝ P(o_t | s_t) · b̄(s_t)

  3. **Normalizes** — ensures the belief sums to 1.

Shannon entropy H = -Σ b(s) log₂ b(s) quantifies uncertainty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class ObservationVector(NamedTuple):
    """Discretized observation at time t."""
    pixel_delta_bin: int   # 0 = STATIC, 1 = HIGH_ACTIVITY
    dom_signal_bin: int    # 0 = True (present), 1 = False (absent)
    latency_bin: int       # 0 = LOW, 1 = HIGH, 2 = TIMEOUT


@dataclass
class BeliefSnapshot:
    """Immutable snapshot of the observer's state at one time step."""
    step: int
    belief: NDArray[np.float64]
    most_likely_state: str
    confidence: float
    entropy: float
    high_uncertainty: bool
    observation: ObservationVector | None


# ---------------------------------------------------------------------------
# StateObserver
# ---------------------------------------------------------------------------

@dataclass
class StateObserver:
    """
    Probabilistic state observer using a Hidden Markov Model (forward algorithm).

    The model is parameterized by:
      - Transition matrix A:  P(S_t | S_{t-1})          [config.TRANSITION_MATRIX]
      - Emission matrices B:  P(O_t^k | S_t) per signal [config.EMISSION_*]
      - Initial belief π₀                                [config.INITIAL_BELIEF]

    Attributes:
        belief: Current probability distribution over latent states.
        step:   Number of belief updates performed so far.
    """

    belief: NDArray[np.float64] = field(
        default_factory=lambda: config.INITIAL_BELIEF.copy()
    )
    step: int = 0

    # -- Pre-loaded model parameters (class-level, shared) ------------------
    _transition: NDArray[np.float64] = field(
        default_factory=lambda: config.TRANSITION_MATRIX.copy(), repr=False
    )
    _emit_pixel: NDArray[np.float64] = field(
        default_factory=lambda: config.EMISSION_PIXEL_DELTA.copy(), repr=False
    )
    _emit_dom: NDArray[np.float64] = field(
        default_factory=lambda: config.EMISSION_DOM_SIGNAL.copy(), repr=False
    )
    _emit_latency: NDArray[np.float64] = field(
        default_factory=lambda: config.EMISSION_NETWORK_LATENCY.copy(), repr=False
    )

    # -----------------------------------------------------------------------
    # Observation Discretization
    # -----------------------------------------------------------------------

    @staticmethod
    def discretize_observations(
        pixel_delta: float,
        dom_signal: bool,
        latency_ms: float,
    ) -> ObservationVector:
        """
        Map raw continuous / boolean sensor readings to discrete bins.

        Args:
            pixel_delta: Mean absolute pixel difference between consecutive frames.
            dom_signal:  Whether the target DOM element is present.
            latency_ms:  Network round-trip time in milliseconds.

        Returns:
            ObservationVector with integer bin indices.
        """
        # pixel_delta → {0: STATIC, 1: HIGH_ACTIVITY}
        pd_bin = 0 if pixel_delta <= config.PIXEL_DELTA_THRESHOLD else 1

        # dom_signal → {0: present (True), 1: absent (False)}
        ds_bin = 0 if dom_signal else 1

        # latency → {0: LOW, 1: HIGH, 2: TIMEOUT}
        if latency_ms <= config.LATENCY_LOW_THRESHOLD:
            lat_bin = 0
        elif latency_ms <= config.LATENCY_HIGH_THRESHOLD:
            lat_bin = 1
        else:
            lat_bin = 2

        return ObservationVector(pd_bin, ds_bin, lat_bin)

    # -----------------------------------------------------------------------
    # Belief Propagation (Forward Algorithm)
    # -----------------------------------------------------------------------

    def update_belief(self, obs: ObservationVector) -> BeliefSnapshot:
        """
        Perform one predict-update cycle of the forward algorithm.

        Math:
            Predict:  b̄(s_t) = A^T · b(s_{t-1})
                      where A[i,j] = P(S_t=j | S_{t-1}=i)
                      so A^T @ b gives Σ_i A[i,j] · b[i] for each j.

            Update:   b(s_t) ∝ P(pixel|s_t) · P(dom|s_t) · P(latency|s_t) · b̄(s_t)

            Normalize: b(s_t) /= Σ b(s_t)

        Args:
            obs: Discretized observation vector.

        Returns:
            BeliefSnapshot capturing the post-update state.
        """
        self.step += 1

        # -- 1. PREDICT: temporal propagation through transition model ------
        # b̄ = A^T @ b  (matrix-vector product)
        predicted: NDArray[np.float64] = self._transition.T @ self.belief

        # -- 2. UPDATE: incorporate observation likelihoods -----------------
        # Observation likelihood for each state is the product of independent
        # emission probabilities (conditional independence assumption):
        #   P(o_t | s_t) = P(pixel|s_t) · P(dom|s_t) · P(latency|s_t)
        likelihood: NDArray[np.float64] = (
            self._emit_pixel[:, obs.pixel_delta_bin]
            * self._emit_dom[:, obs.dom_signal_bin]
            * self._emit_latency[:, obs.latency_bin]
        )

        # Element-wise product: unnormalized posterior
        unnormalized: NDArray[np.float64] = likelihood * predicted

        # -- 3. NORMALIZE ---------------------------------------------------
        total = unnormalized.sum()
        if total < 1e-12:
            # Numerical safeguard: if all likelihoods collapse, reset to prior
            logger.warning("Belief collapsed to zero — resetting to uniform prior.")
            self.belief = config.INITIAL_BELIEF.copy()
        else:
            self.belief = unnormalized / total

        # -- Build snapshot -------------------------------------------------
        entropy = self.entropy()
        idx = int(np.argmax(self.belief))
        snapshot = BeliefSnapshot(
            step=self.step,
            belief=self.belief.copy(),
            most_likely_state=config.STATES[idx],
            confidence=float(self.belief[idx]),
            entropy=entropy,
            high_uncertainty=entropy > config.ENTROPY_THRESHOLD,
            observation=obs,
        )

        logger.debug(
            "Step %d | State=%-17s Conf=%.2f%% H=%.3f %s",
            self.step,
            snapshot.most_likely_state,
            snapshot.confidence * 100,
            entropy,
            "⚠ HIGH UNCERTAINTY" if snapshot.high_uncertainty else "",
        )

        return snapshot

    # -----------------------------------------------------------------------
    # Entropy & Decision Functions
    # -----------------------------------------------------------------------

    def entropy(self) -> float:
        """
        Shannon entropy of the current belief distribution.

        H(b) = -Σ_s b(s) · log₂(b(s))

        Uses base-2 logarithm so entropy is measured in bits.
        For 4 states, maximum entropy = log₂(4) = 2.0 bits.

        Returns:
            Entropy value in bits.
        """
        # Mask zeros to avoid log(0)
        b = self.belief[self.belief > 0]
        return float(-np.sum(b * np.log2(b)))

    def most_likely_state(self) -> tuple[str, float]:
        """
        Return the MAP (Maximum A Posteriori) state and its probability.

        Returns:
            (state_name, confidence) where confidence ∈ [0, 1].
        """
        idx = int(np.argmax(self.belief))
        return config.STATES[idx], float(self.belief[idx])

    def should_proceed(self) -> bool:
        """
        Action gate: returns True only when P(READY) exceeds the threshold.

        This prevents downstream agents from acting on the page before it
        has stabilized into the READY state with sufficient confidence.

        Returns:
            True if P(S_t = READY) > config.READY_CONFIDENCE_THRESHOLD.
        """
        ready_prob = float(self.belief[config.STATE_INDEX["READY"]])
        return ready_prob > config.READY_CONFIDENCE_THRESHOLD

    def reset(self) -> None:
        """Reset belief to the uniform prior and zero the step counter."""
        self.belief = config.INITIAL_BELIEF.copy()
        self.step = 0
        logger.info("Observer reset to uniform prior.")
