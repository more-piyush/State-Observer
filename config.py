"""
config.py — Model Parameters for the PGM State Observer.

Defines the Dynamic Bayesian Network (DBN) structure:
  - Latent state space S_t ∈ {READY, LOADING, STALLED, SUCCESS_PENDING}
  - Observable signals O_t = (pixel_delta, dom_signal, network_latency)
  - Transition matrix P(S_t | S_{t-1})
  - Emission / observation matrices P(O_t | S_t)

All probability tables are row-stochastic (rows sum to 1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Latent State Definitions
# ---------------------------------------------------------------------------
STATES: list[str] = ["READY", "LOADING", "STALLED", "SUCCESS_PENDING"]
NUM_STATES: int = len(STATES)
STATE_INDEX: dict[str, int] = {s: i for i, s in enumerate(STATES)}

# ---------------------------------------------------------------------------
# Observable Signal Definitions
# ---------------------------------------------------------------------------

# pixel_delta is discretized into two bins:
PIXEL_DELTA_BINS: list[str] = ["STATIC", "HIGH_ACTIVITY"]
PIXEL_DELTA_THRESHOLD: float = 5.0  # mean abs diff > threshold => HIGH_ACTIVITY

# dom_signal: presence of the target element
DOM_SIGNAL_VALUES: list[bool] = [True, False]

# network_latency bins (milliseconds)
NETWORK_LATENCY_BINS: list[str] = ["LOW", "HIGH", "TIMEOUT"]
LATENCY_LOW_THRESHOLD: float = 200.0    # ms; <= threshold => LOW
LATENCY_HIGH_THRESHOLD: float = 2000.0  # ms; <= threshold => HIGH, else TIMEOUT

# ---------------------------------------------------------------------------
# Observer Loop Parameters
# ---------------------------------------------------------------------------
OBSERVATION_INTERVAL_MS: int = 200       # belief update cadence
ENTROPY_THRESHOLD: float = 0.8          # Shannon entropy flag threshold
READY_CONFIDENCE_THRESHOLD: float = 0.85 # P(READY) must exceed this
MAX_OBSERVATION_STEPS: int = 150         # safety cap (~30 s at 200 ms)

# ---------------------------------------------------------------------------
# Initial Belief (uniform prior — no information at t=0)
# ---------------------------------------------------------------------------
# π₀ = [0.25, 0.25, 0.25, 0.25]
INITIAL_BELIEF: NDArray[np.float64] = np.array(
    [0.25, 0.25, 0.25, 0.25], dtype=np.float64
)

# ---------------------------------------------------------------------------
# Transition Matrix  P(S_t | S_{t-1})
# ---------------------------------------------------------------------------
# Rows = S_{t-1}, Columns = S_t
#
#                        READY   LOADING  STALLED  SUCCESS_PENDING
# READY                  0.90     0.05     0.02     0.03
# LOADING                0.15     0.55     0.15     0.15
# STALLED                0.05     0.10     0.75     0.10
# SUCCESS_PENDING        0.60     0.05     0.05     0.30
#
# Rationale:
#   - READY is mostly self-sustaining (0.90 stay).
#   - LOADING has moderate self-loop; can resolve to READY or degrade to STALLED.
#   - STALLED is semi-absorbing (0.75 stay) — represents a hung page.
#   - SUCCESS_PENDING quickly transitions to READY (0.60) once assets arrive.
TRANSITION_MATRIX: NDArray[np.float64] = np.array([
    [0.90, 0.05, 0.02, 0.03],  # from READY
    [0.15, 0.55, 0.15, 0.15],  # from LOADING
    [0.05, 0.10, 0.75, 0.10],  # from STALLED
    [0.60, 0.05, 0.05, 0.30],  # from SUCCESS_PENDING
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Emission / Observation Matrices  P(O_t | S_t)
# ---------------------------------------------------------------------------
# Each matrix: rows = latent state, columns = discrete observation value.

# P(pixel_delta_bin | S_t)
#                        STATIC  HIGH_ACTIVITY
# READY                   0.85     0.15
# LOADING                 0.20     0.80
# STALLED                 0.70     0.30
# SUCCESS_PENDING         0.40     0.60
#
# Rationale: LOADING pages show high visual activity (spinners, reflows).
# READY pages are mostly static. STALLED may have a frozen animation.
EMISSION_PIXEL_DELTA: NDArray[np.float64] = np.array([
    [0.85, 0.15],  # READY
    [0.20, 0.80],  # LOADING
    [0.70, 0.30],  # STALLED
    [0.40, 0.60],  # SUCCESS_PENDING
], dtype=np.float64)

# P(dom_signal | S_t)  — target element present?
#                        True    False
# READY                  0.95    0.05
# LOADING                0.10    0.90
# STALLED                0.05    0.95
# SUCCESS_PENDING        0.70    0.30
#
# Rationale: The target element appears when the page is fully loaded (READY)
# or nearly so (SUCCESS_PENDING). It's absent during LOADING / STALLED.
EMISSION_DOM_SIGNAL: NDArray[np.float64] = np.array([
    [0.95, 0.05],  # READY
    [0.10, 0.90],  # LOADING
    [0.05, 0.95],  # STALLED
    [0.70, 0.30],  # SUCCESS_PENDING
], dtype=np.float64)

# P(network_latency_bin | S_t)
#                        LOW     HIGH    TIMEOUT
# READY                  0.80    0.15    0.05
# LOADING                0.30    0.55    0.15
# STALLED                0.10    0.30    0.60
# SUCCESS_PENDING        0.50    0.40    0.10
#
# Rationale: READY implies the server responds quickly. STALLED correlates
# with timeouts. LOADING has elevated latency. SUCCESS_PENDING is improving.
EMISSION_NETWORK_LATENCY: NDArray[np.float64] = np.array([
    [0.80, 0.15, 0.05],  # READY
    [0.30, 0.55, 0.15],  # LOADING
    [0.10, 0.30, 0.60],  # STALLED
    [0.50, 0.40, 0.10],  # SUCCESS_PENDING
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Screenshot Capture Settings
# ---------------------------------------------------------------------------
SCREENSHOT_SCALE: float = 0.25  # downscale factor for pixel-delta computation
