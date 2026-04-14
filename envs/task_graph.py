"""
task_graph.py — Multi-Step Workflow Definitions as Directed Acyclic Graphs.

A web automation task is modeled as a sequence of TaskNodes, where each node
represents one discrete interaction (click, type, submit, navigate, etc.).
Each action can **change the page's latent state** — for example, clicking
"Submit" triggers a LOADING phase before the confirmation page becomes READY.

Architecture:

    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  OPEN    │────▶│  FILL    │────▶│  SUBMIT  │────▶│  VERIFY  │
    │  PAGE    │     │  FORM    │     │  FORM    │     │  RESULT  │
    └──────────┘     └──────────┘     └──────────┘     └──────────┘
    state→LOADING    state→READY     state→LOADING    state→READY
    (page loads)     (form usable)   (server proc.)   (confirm shown)

Each node carries:
  - A precondition:  which page latent state must hold before acting
  - A post-effect:   the distribution over page states AFTER the action
  - A difficulty:    how noisy / unpredictable the transition is

This file provides:
  1. TaskNode dataclass — one step in a workflow
  2. TaskGraph class — an ordered sequence of nodes
  3. Pre-built factory methods for common web scenarios
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TaskNode — One Step in a Multi-Step Workflow
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskNode:
    """
    A single discrete action within a multi-step web workflow.

    Attributes:
        name:            Human-readable label (e.g. "FILL_EMAIL").
        action_id:       Integer ID used as the agent's action output.
        required_state:  Page latent state that MUST hold before executing.
                         The agent should WAIT until the observer confirms this.
        post_state_dist: P(S_{t+1} | action executed) — probability distribution
                         over page states immediately AFTER this action fires.
                         This replaces the normal transition matrix for one step.
        max_wait_steps:  Max observation steps the agent can wait for the
                         precondition before the sub-task times out.
        description:     What this action does in the real browser.
    """
    name: str
    action_id: int
    required_state: str = "READY"
    post_state_dist: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.85, 0.10, 0.05], dtype=np.float64)
    )
    max_wait_steps: int = 50
    description: str = ""


# ---------------------------------------------------------------------------
# TaskGraph — Ordered Sequence of TaskNodes
# ---------------------------------------------------------------------------

@dataclass
class TaskGraph:
    """
    An ordered workflow of TaskNodes representing a complete web automation task.

    The agent must execute nodes in order. For each node:
      1. WAIT until the page reaches the node's required_state.
      2. Execute the action (ACT).
      3. The page transitions according to the node's post_state_dist.
      4. Move to the next node.

    Attributes:
        name:  Workflow identifier (e.g. "form_submission").
        nodes: Ordered list of TaskNodes.
    """
    name: str
    nodes: list[TaskNode] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        """Total number of sequential actions in this workflow."""
        return len(self.nodes)

    def get_node(self, index: int) -> Optional[TaskNode]:
        """Get the TaskNode at a given workflow position."""
        if 0 <= index < len(self.nodes):
            return self.nodes[index]
        return None

    def __repr__(self) -> str:
        node_names = " → ".join(n.name for n in self.nodes)
        return f"TaskGraph('{self.name}': {node_names})"


# ---------------------------------------------------------------------------
# Pre-Built Workflow Factories
# ---------------------------------------------------------------------------

def create_form_submission_workflow() -> TaskGraph:
    """
    Standard HTML form submission: navigate → fill fields → submit → verify.

    Real-world analogy:
        1. Navigate to /register       (page loads)
        2. Fill in form fields          (page must be READY, typing triggers minor reflows)
        3. Click "Submit"               (triggers server POST, page goes LOADING)
        4. Verify confirmation message  (page must reach READY with success element)

    ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
    │ NAVIGATE  │────▶│   FILL    │────▶│  SUBMIT   │────▶│  VERIFY   │
    │           │     │   FORM    │     │           │     │           │
    └───────────┘     └───────────┘     └───────────┘     └───────────┘
    pre: LOADING      pre: READY        pre: READY        pre: READY
    post: →LOADING    post: →READY(90%) post: →LOADING    post: →SUCCESS_P
    """
    return TaskGraph(
        name="form_submission",
        nodes=[
            TaskNode(
                name="NAVIGATE",
                action_id=0,
                required_state="LOADING",
                # After navigation is "executed" (the page starts loading),
                # the page is predominantly in LOADING state.
                post_state_dist=np.array([0.05, 0.80, 0.10, 0.05], dtype=np.float64),
                max_wait_steps=30,
                description="Navigate to the registration page",
            ),
            TaskNode(
                name="FILL_FORM",
                action_id=1,
                required_state="READY",
                # Filling a form is lightweight — page stays READY with minor flickers.
                post_state_dist=np.array([0.90, 0.05, 0.02, 0.03], dtype=np.float64),
                max_wait_steps=50,
                description="Fill in the form fields (name, email, password)",
            ),
            TaskNode(
                name="SUBMIT",
                action_id=2,
                required_state="READY",
                # Clicking Submit triggers a server round-trip → heavy LOADING.
                post_state_dist=np.array([0.02, 0.78, 0.15, 0.05], dtype=np.float64),
                max_wait_steps=60,
                description="Click the Submit button",
            ),
            TaskNode(
                name="VERIFY",
                action_id=3,
                required_state="READY",
                # Verification is a read-only check — page remains stable.
                post_state_dist=np.array([0.70, 0.05, 0.05, 0.20], dtype=np.float64),
                max_wait_steps=50,
                description="Verify the success confirmation message",
            ),
        ],
    )


def create_search_workflow() -> TaskGraph:
    """
    Search + paginated results: navigate → enter query → submit → read results → next page.

    Real-world analogy: Google search with pagination.

    ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ NAVIGATE  │──▶│  TYPE     │──▶│  SEARCH   │──▶│  READ     │──▶│ NEXT PAGE │
    │           │   │  QUERY    │   │           │   │  RESULTS  │   │           │
    └───────────┘   └───────────┘   └───────────┘   └───────────┘   └───────────┘
    """
    return TaskGraph(
        name="search_paginated",
        nodes=[
            TaskNode(
                name="NAVIGATE",
                action_id=0,
                required_state="LOADING",
                post_state_dist=np.array([0.05, 0.80, 0.10, 0.05], dtype=np.float64),
                max_wait_steps=30,
                description="Navigate to the search page",
            ),
            TaskNode(
                name="TYPE_QUERY",
                action_id=1,
                required_state="READY",
                post_state_dist=np.array([0.85, 0.08, 0.02, 0.05], dtype=np.float64),
                max_wait_steps=40,
                description="Type the search query into the input field",
            ),
            TaskNode(
                name="SEARCH",
                action_id=2,
                required_state="READY",
                post_state_dist=np.array([0.03, 0.75, 0.12, 0.10], dtype=np.float64),
                max_wait_steps=60,
                description="Click Search / press Enter",
            ),
            TaskNode(
                name="READ_RESULTS",
                action_id=3,
                required_state="READY",
                post_state_dist=np.array([0.88, 0.05, 0.02, 0.05], dtype=np.float64),
                max_wait_steps=50,
                description="Read and parse the search results",
            ),
            TaskNode(
                name="NEXT_PAGE",
                action_id=4,
                required_state="READY",
                # Clicking "Next" triggers a new page load
                post_state_dist=np.array([0.05, 0.75, 0.10, 0.10], dtype=np.float64),
                max_wait_steps=50,
                description="Click the 'Next Page' pagination link",
            ),
        ],
    )


def create_checkout_workflow() -> TaskGraph:
    """
    E-commerce checkout: cart → shipping → payment → confirm → receipt.

    The most complex workflow — each step is gated by page readiness and
    several steps involve heavy server processing (payment, order creation).

    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ VIEW     │─▶│ FILL     │─▶│ FILL     │─▶│ CONFIRM  │─▶│ VERIFY   │
    │ CART     │  │ SHIPPING │  │ PAYMENT  │  │ ORDER    │  │ RECEIPT  │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
    """
    return TaskGraph(
        name="checkout",
        nodes=[
            TaskNode(
                name="VIEW_CART",
                action_id=0,
                required_state="LOADING",
                post_state_dist=np.array([0.05, 0.75, 0.12, 0.08], dtype=np.float64),
                max_wait_steps=30,
                description="Open the shopping cart page",
            ),
            TaskNode(
                name="FILL_SHIPPING",
                action_id=1,
                required_state="READY",
                post_state_dist=np.array([0.88, 0.05, 0.02, 0.05], dtype=np.float64),
                max_wait_steps=50,
                description="Enter shipping address details",
            ),
            TaskNode(
                name="FILL_PAYMENT",
                action_id=2,
                required_state="READY",
                # Payment form may trigger real-time card validation → some loading
                post_state_dist=np.array([0.60, 0.25, 0.05, 0.10], dtype=np.float64),
                max_wait_steps=60,
                description="Enter credit card / payment information",
            ),
            TaskNode(
                name="CONFIRM_ORDER",
                action_id=3,
                required_state="READY",
                # Order confirmation triggers heavy backend processing
                post_state_dist=np.array([0.02, 0.65, 0.20, 0.13], dtype=np.float64),
                max_wait_steps=80,
                description="Click 'Place Order' — triggers payment processing",
            ),
            TaskNode(
                name="VERIFY_RECEIPT",
                action_id=4,
                required_state="READY",
                post_state_dist=np.array([0.75, 0.05, 0.05, 0.15], dtype=np.float64),
                max_wait_steps=50,
                description="Verify order confirmation and receipt number",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Registry of all available workflows
# ---------------------------------------------------------------------------
WORKFLOW_REGISTRY: dict[str, callable] = {
    "form_submission": create_form_submission_workflow,
    "search_paginated": create_search_workflow,
    "checkout": create_checkout_workflow,
}
