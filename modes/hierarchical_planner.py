"""
modes/hierarchical_planner.py
==============================
Disclosure mode: HIERARCHICAL_PLANNER  [NOT YET IMPLEMENTED]

Strategy (planned)
------------------
A two-stage planning pipeline:

Stage 1 – Domain Planning
    The LLM classifies the objective into a high-level domain
    (e.g. "issue management", "branch workflow", "repository administration").
    The domain determines which sub-catalog of operations to load.

Stage 2 – Operation Planning
    The LLM receives only the domain-specific sub-catalog and produces the
    final execution plan.

This is the most token-efficient mode but requires the most scaffolding
(domain taxonomy, per-domain operation sets, two LLM round-trips).

Dependencies (not yet implemented)
------------------------------------
- registry.classify_domain(objective) → domain name
- registry.get_by_domain(domain) → list[OperationDefinition]
  Both are already stubbed in OperationRegistry.

Public interface (mirrors other modes)
--------------------------------------
    build_plan(state, registry, model_router, config) -> dict
"""

from __future__ import annotations

from typing import Any

from orchestrator.config import OrchestratorConfig


def build_plan(
    state: dict[str, Any],
    registry: Any,
    model_router: Any,
    config: OrchestratorConfig,
) -> dict[str, Any]:
    """Placeholder — not yet implemented."""
    raise NotImplementedError(
        "disclosure_mode 'hierarchical_planner' is not yet implemented. "
        "Use 'all_loaded' or 'category_gated'."
    )
