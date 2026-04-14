# PGM State Observer

A probabilistic state observer for web-automation AI agents, with a full
reinforcement-learning benchmark that quantifies the value of belief-based
state estimation over raw observation signals.

The observer sits between a browser (Playwright) and a decision-making agent,
maintaining a Dynamic Bayesian Network (DBN) belief over the latent state of a
web page and gating actions until the page is confidently "READY".

---

## What's in this Project

This repository contains three things:

1. **A Dynamic Bayesian Network state observer** — consumes noisy browser
   signals (pixel deltas, DOM presence, network latency) and outputs a
   calibrated posterior over four latent page states:
   `READY`, `LOADING`, `STALLED`, `SUCCESS_PENDING`.

2. **A simulated RL environment** — a Gymnasium environment that reproduces
   the same DBN dynamics faithfully, so an agent can be trained offline at
   ~50,000 steps/sec without a real browser. Both single-step and multi-step
   (TaskGraph-based) variants are provided.

3. **A matched A/B benchmark** — identical DQN / Hierarchical-DQN agents are
   trained with and without the observer's belief features, across multiple
   seeds, producing plots and CSV artifacts that measure the observer's
   contribution to learning speed, final accuracy, and catastrophic-error
   rate.

---

## Why This Matters

Raw browser signals are noisy. A page can *look* static for a frame while still
loading; a target DOM element may flicker in and out during hydration; network
latency is bursty. An agent that reacts to single-frame observations will click
too early and fail silently.

The PGM observer converts noisy, per-step evidence into a **calibrated belief
over a small set of semantically meaningful states**, plus a confidence score
and an entropy-based uncertainty measure. The benchmark shows this produces:

- **~6.5× faster convergence** on single-step tasks,
- **Near-elimination of premature-action failures** on multi-step workflows,
- **Higher final return** at the same step budget.

---

## Repository Layout

```
PGM State Observer/
├── config.py                      # DBN parameters: transition & emission matrices
├── state_observer.py              # StateObserver class (forward algorithm)
│
├── envs/
│   ├── page_load_env.py           # Single-step Gymnasium env (obs_dim 3 or 9)
│   ├── multi_step_env.py          # Multi-action workflow env (obs_dim 6 or 12)
│   ├── task_graph.py              # TaskNode / TaskGraph + 3 workflow definitions
│   └── __init__.py
│
├── agents/
│   ├── dqn_agent.py               # Vanilla DQN (64→64 MLP)
│   ├── hierarchical_agent.py      # Two-level DQN (controller + meta-controller)
│   ├── replay_buffer.py           # Circular experience buffer
│   └── __init__.py
│
├── training/
│   ├── trainer.py                 # Episode loop + update schedule
│   ├── metrics.py                 # Metric dataclasses, rolling stats, CSV export
│   └── __init__.py
│
├── run_experiment.py              # Single-step A/B benchmark (5 seeds × 2 modes)
├── run_multistep_experiment.py    # Multi-step A/B benchmark
├── plot_results.py                # Single-step plots (learning curves, etc.)
├── plot_multistep_results.py      # Multi-step plots (6-panel dashboard)
├── analyze_results.py             # Tabular summary across seeds
│
├── main.py                        # Live Playwright demo of the observer
├── demo_page.html                 # Local HTML page used by main.py
│
├── results/                       # CSV outputs from run_experiment.py
├── results_multistep/             # CSV outputs from run_multistep_experiment.py
│
├── requirements.txt
└── README.md                      # (you are here)
```

---

## Installation

```bash
# 1. Create and activate a virtual environment (Python 3.11+ recommended)
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate        # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional — only needed for main.py live demo)
playwright install chromium
```

### Dependencies

- `numpy` — matrix math for the forward algorithm and RL tensors
- `gymnasium` — environment API
- `torch` — DQN networks
- `matplotlib`, `pandas` — plotting and CSV analysis
- `playwright` — live browser demo (optional)

---

## Quick Start

### 1. Run the live Playwright demo

See the observer update its belief in real time against a simulated loading
page:

```bash
python main.py
```

You'll see a console dashboard printing every 200 ms with the current
observations, the inferred state, the belief vector, and the entropy. The loop
exits when `should_proceed()` becomes `True`.

### 2. Run the single-step RL benchmark

```bash
python run_experiment.py
```

This trains a vanilla DQN agent twice for each of 5 seeds — once with raw
observations (dim=3) and once with the observer's augmented features (dim=9)
— and dumps per-episode metrics to `results/`.

Then plot:

```bash
python plot_results.py
```

### 3. Run the multi-step (TaskGraph) benchmark

```bash
python run_multistep_experiment.py
python plot_multistep_results.py
```

This trains a `HierarchicalDQNAgent` on a form-submission workflow with
multiple sequential actions. Outputs land in `results_multistep/`.

---

## How It Works

### The Observer (DBN Forward Algorithm)

At every timestep the observer runs a two-stage Bayes update:

```
Predict:  b'(S_t) = Σ_{S_{t-1}} P(S_t | S_{t-1}) · b(S_{t-1})
Update:   b(S_t) ∝ P(O_t | S_t) · b'(S_t)
```

Where `P(O_t | S_t)` decomposes into three independent emission channels
(pixel_delta, dom_signal, network_latency) defined in `config.py`.

The observer exposes three decision primitives:
- `most_likely_state()` — argmax of the posterior
- `entropy()` — Shannon entropy, flagging uncertainty
- `should_proceed()` — gate that returns True iff `P(READY) > 0.85`

### The Simulated Environment

`PageLoadEnv` (single-step) and `MultiStepEnv` (multi-action) simulate the same
DBN dynamics used by the observer. The hidden state `S_t` evolves via
`config.TRANSITION_MATRIX`, and observations are sampled from the emission
matrices each step. A single `use_observer: bool` flag toggles between two
observation modes:

| Mode | Dim (single-step) | Dim (multi-step) | Contents |
|---|---|---|---|
| Baseline | 3 | 6 | Raw discretized signals (+ workflow context for multi-step) |
| Augmented | 9 | 12 | Baseline + belief[0:4] + entropy + should_proceed |

Both modes share identical network architecture, reward schedule, RNG seeds,
and episode protocol — the **only** difference is the observation channel's
pre-processing. This isolates the causal effect of the observer.

### Reward Shaping

Single-step (`PageLoadEnv`):

| Event | Reward |
|---|---|
| ACT while READY | +10.0 |
| ACT while not-READY | -5.0 |
| WAIT while not-READY | 0.0 |
| WAIT while READY | -0.1 |
| Per-step time penalty | -0.05 |
| Truncation (timeout) | -3.0 |

Multi-step (`MultiStepEnv`) adds:

| Event | Reward |
|---|---|
| Full workflow complete | +20.0 |
| Per-node progress bonus | +2.0 |
| SKIP action | -8.0 |
| Node timeout | -4.0 |

### The Agents

- **`DQNAgent`** — standard DQN with a 64→64 MLP, ε-greedy (1.0 → 0.05 over
  5,000 steps), target network synced every 100 updates, experience replay
  (50K capacity), γ=0.99.
- **`HierarchicalDQNAgent`** — Options-framework-inspired two-level agent:
  - **Controller** (128→64 MLP) makes per-step WAIT/EXECUTE decisions.
  - **Meta-controller** (64→32 MLP) fires every 10 steps and chooses
    PROCEED / SKIP for the current TaskGraph node.
  - Separate replay buffers (100K ctrl, 25K meta), shared ε schedule.

---

## Benchmark Results (Summary)

Single-step (averaged over 5 seeds, 10K episodes):

- Observer-augmented agent reaches 95% correct-action rate in ~1,500
  episodes; baseline requires ~9,700 episodes to reach the same bar.
- Final return advantage: +3.8 points per episode on the augmented agent.

Multi-step (form-submission workflow, 5 seeds, 5K episodes):

- Workflow completion rate: baseline 41% → augmented 87%.
- Premature-EXECUTE rate: baseline 22% → augmented 3%.
- The observer's belief-reset on node transitions eliminates the
  error-cascade pattern seen in the baseline.

Full plots are produced by the two `plot_*.py` scripts.

---

## Configuration

All model parameters live in `config.py`:

- `TRANSITION_MATRIX` (4×4) — P(S_t | S_{t-1})
- `EMISSION_PIXEL_DELTA`, `EMISSION_DOM_SIGNAL`, `EMISSION_NETWORK_LATENCY`
- `READY_CONFIDENCE_THRESHOLD = 0.85`
- `ENTROPY_THRESHOLD = 0.8`
- `MAX_OBSERVATION_STEPS = 50`
- Observer loop interval: `200 ms`

Agent hyperparameters are dataclass defaults in `agents/dqn_agent.py` and
`agents/hierarchical_agent.py` and can be overridden via constructor kwargs.

Training sweep parameters (seeds, episodes, etc.) are module-level constants
at the top of `run_experiment.py` and `run_multistep_experiment.py`.

---

## Extending the Project

- **New workflows**: add a `create_*_workflow()` function in
  `envs/task_graph.py` and register it in `WORKFLOW_REGISTRY`.
- **New agent**: implement the `select_action / update / save / load`
  interface used by `training/trainer.py`.
- **Different emission models**: edit the matrices in `config.py` — the
  observer and the simulator both read from the same source, so changes stay
  consistent.
- **Real-browser evaluation**: swap the sampling loop in `main.py` for a
  Playwright capture of real pixel deltas / DOM states / network timings.
  The observer API is identical.

---

## Project Design Principles

1. **Single source of truth for model parameters** — both the observer and
   the simulator read `config.py`, so there is no sim-to-observer drift.
2. **Matched A/B harness** — every difference between baseline and augmented
   runs is reduced to a single flag, eliminating confounds.
3. **Dense reward shaping** — every step has non-zero reward so Q-learning
   gets signal even when episodes never terminate early.
4. **Reproducibility** — all randomness is seeded; seeds are logged in CSVs
   and printed at the top of each run.
5. **Clear separation of concerns** — belief estimation
   (`state_observer.py`), environment dynamics (`envs/`), learning
   (`agents/` + `training/`), and evaluation (`run_*.py` + `plot_*.py`) are
   independent modules with small, typed interfaces.
