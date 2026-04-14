"""
main.py — Playwright Simulation + PGM State Observer Loop.

Launches a headless Chromium browser, navigates to the demo page, and runs
the observation loop at ~200 ms intervals. Each tick:

  1. Captures a screenshot → computes pixel delta (mean abs diff of grayscale).
  2. Queries the DOM for #target-element → dom_signal.
  3. Reads window.__latencyMs from the page → network latency.
  4. Feeds the observation vector into StateObserver.update_belief().
  5. Prints a real-time console dashboard line.

Exits when should_proceed() returns True (P(READY) > 0.85) or MAX_STEPS.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from playwright.async_api import async_playwright, Page

import config
from state_observer import StateObserver

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Signal Acquisition Helpers
# ---------------------------------------------------------------------------

async def capture_grayscale_thumbnail(page: Page) -> NDArray[np.float64]:
    """
    Capture a screenshot, decode it into a grayscale numpy array scaled down
    by config.SCREENSHOT_SCALE for fast pixel-delta computation.

    Returns:
        2-D float64 array of grayscale pixel intensities ∈ [0, 255].
    """
    png_bytes: bytes = await page.screenshot(type="png")

    # Decode PNG bytes into a numpy array without PIL dependency.
    # Playwright returns full-resolution PNG; we use a fast in-memory decode.
    from io import BytesIO

    try:
        from PIL import Image
    except ImportError:
        # Fallback: treat raw bytes as a signal hash (less precise but functional)
        arr = np.frombuffer(png_bytes, dtype=np.uint8).astype(np.float64)
        size = int(len(arr) * config.SCREENSHOT_SCALE)
        return arr[:size]

    img = Image.open(BytesIO(png_bytes)).convert("L")  # grayscale
    w, h = img.size
    new_size = (max(1, int(w * config.SCREENSHOT_SCALE)),
                max(1, int(h * config.SCREENSHOT_SCALE)))
    img = img.resize(new_size, Image.BILINEAR)
    return np.array(img, dtype=np.float64)


def compute_pixel_delta(
    prev: NDArray[np.float64] | None,
    curr: NDArray[np.float64],
) -> float:
    """
    Mean absolute difference between two consecutive grayscale frames.

    If the frames differ in size (e.g. first frame), returns 0.0.

    Args:
        prev: Previous frame (or None on the first tick).
        curr: Current frame.

    Returns:
        Mean absolute pixel delta ∈ [0, 255].
    """
    if prev is None or prev.shape != curr.shape:
        return 0.0
    return float(np.mean(np.abs(curr - prev)))


async def check_dom_signal(page: Page) -> bool:
    """Check whether the #target-element is visible in the DOM."""
    try:
        el = await page.query_selector("#target-element.visible")
        return el is not None
    except Exception:
        return False


async def measure_network_latency(page: Page) -> float:
    """Read the latest latency measurement from the page's JS context."""
    try:
        latency = await page.evaluate("() => window.__latencyMs || 0")
        return float(latency)
    except Exception:
        return 5000.0  # assume timeout on error


# ---------------------------------------------------------------------------
# Console Dashboard
# ---------------------------------------------------------------------------

def print_dashboard(snapshot) -> None:
    """Print a single-line real-time dashboard to stdout."""
    obs = snapshot.observation
    obs_labels = (
        config.PIXEL_DELTA_BINS[obs.pixel_delta_bin],
        "PRESENT" if obs.dom_signal_bin == 0 else "ABSENT",
        config.NETWORK_LATENCY_BINS[obs.latency_bin],
    )
    uncertainty_flag = " ⚠ HIGH UNCERTAINTY" if snapshot.high_uncertainty else ""

    line = (
        f"  Step {snapshot.step:>3d} │ "
        f"Obs [{obs_labels[0]:<14s} {obs_labels[1]:<8s} {obs_labels[2]:<8s}] → "
        f"State: {snapshot.most_likely_state:<17s} │ "
        f"Conf: {snapshot.confidence * 100:5.1f}% │ "
        f"H: {snapshot.entropy:.3f}{uncertainty_flag}"
    )
    print(line)


def print_belief_bar(belief: NDArray[np.float64]) -> None:
    """Print a horizontal bar chart of the belief distribution."""
    bar_width = 30
    print("  ┌" + "─" * 52 + "┐")
    for i, state in enumerate(config.STATES):
        filled = int(belief[i] * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  │ {state:<17s} {bar} {belief[i]*100:5.1f}% │")
    print("  └" + "─" * 52 + "┘")


# ---------------------------------------------------------------------------
# Main Observation Loop
# ---------------------------------------------------------------------------

async def run_observer() -> None:
    """Launch browser, navigate to demo page, and run the observer loop."""
    demo_path = Path(__file__).parent / "demo_page.html"
    if not demo_path.exists():
        logger.error("demo_page.html not found at %s", demo_path)
        sys.exit(1)

    demo_url = demo_path.as_uri()
    logger.info("Demo URL: %s", demo_url)

    observer = StateObserver()

    print("\n" + "=" * 72)
    print("  PGM STATE OBSERVER — Real-Time Console Dashboard")
    print("=" * 72)
    print(f"  States: {config.STATES}")
    print(f"  Interval: {config.OBSERVATION_INTERVAL_MS} ms")
    print(f"  Ready threshold: P(READY) > {config.READY_CONFIDENCE_THRESHOLD}")
    print(f"  Entropy flag: H > {config.ENTROPY_THRESHOLD}")
    print("=" * 72 + "\n")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 800, "height": 600},
        )
        page = await context.new_page()

        logger.info("Navigating to demo page...")
        await page.goto(demo_url, wait_until="commit")

        prev_frame: NDArray[np.float64] | None = None

        for step in range(1, config.MAX_OBSERVATION_STEPS + 1):
            # -- Acquire raw signals ------------------------------------
            curr_frame = await capture_grayscale_thumbnail(page)
            pixel_delta = compute_pixel_delta(prev_frame, curr_frame)
            dom_signal = await check_dom_signal(page)
            latency_ms = await measure_network_latency(page)
            prev_frame = curr_frame

            # -- Discretize & update belief -----------------------------
            obs = StateObserver.discretize_observations(
                pixel_delta=pixel_delta,
                dom_signal=dom_signal,
                latency_ms=latency_ms,
            )
            snapshot = observer.update_belief(obs)

            # -- Dashboard output ---------------------------------------
            print_dashboard(snapshot)

            # Print belief bar every 5 steps for readability
            if step % 5 == 0:
                print_belief_bar(snapshot.belief)

            # -- Action gate check --------------------------------------
            if observer.should_proceed():
                print("\n" + "─" * 72)
                print(f"  ✓ ACTION GATE OPEN — P(READY) = "
                      f"{snapshot.belief[config.STATE_INDEX['READY']]*100:.1f}% "
                      f"> {config.READY_CONFIDENCE_THRESHOLD*100:.0f}% threshold")
                print(f"  ✓ should_proceed() = True at step {step}")
                print("─" * 72)
                print_belief_bar(snapshot.belief)
                break

            # -- Wait for next observation interval ---------------------
            await asyncio.sleep(config.OBSERVATION_INTERVAL_MS / 1000.0)
        else:
            print("\n" + "─" * 72)
            print(f"  ✗ MAX STEPS ({config.MAX_OBSERVATION_STEPS}) reached "
                  f"without convergence to READY.")
            state, conf = observer.most_likely_state()
            print(f"  Final state: {state} ({conf*100:.1f}%)")
            print("─" * 72)
            print_belief_bar(observer.belief)

        logger.info("Closing browser...")
        await browser.close()

    print("\n  Observer session complete.\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_observer())
