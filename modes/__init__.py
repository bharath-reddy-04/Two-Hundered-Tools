"""
modes/
======
Disclosure-mode strategy modules for the orchestration agent.

Each module exposes a single function:

    build_plan(state, registry, model_router, config) -> dict

The function mutates *state* in place (following the LangGraph node
convention) and returns it.  It sets:

    state["plan"]                 – Plan | None
    state["replanning_required"]  – bool

Currently implemented
---------------------
- all_loaded       : Entire catalog disclosed to the LLM at planning time.
- category_gated   : Only operations matching the task's category are shown.

Planned (not yet implemented)
------------------------------
- search_then_load : Semantic similarity retrieval selects candidates.
- hierarchical_planner : Two-stage domain→operation planning.
"""

from __future__ import annotations

from modes.all_loaded import build_plan as all_loaded_plan
from modes.category_gated import build_plan as category_gated_plan

__all__ = ["all_loaded_plan", "category_gated_plan"]

# Registry: mode name → build_plan callable
_MODE_REGISTRY: dict[str, object] = {
    "all_loaded": all_loaded_plan,
    "category_gated": category_gated_plan,
}


def get_plan_builder(mode: str):
    """
    Return the build_plan callable for *mode*.

    Raises
    ------
    NotImplementedError
        If the mode exists in config but is not yet implemented here.
    ValueError
        If the mode is completely unknown.
    """
    if mode in _MODE_REGISTRY:
        return _MODE_REGISTRY[mode]

    _deferred = ("search_then_load", "hierarchical_planner")
    if mode in _deferred:
        raise NotImplementedError(
            f"disclosure_mode '{mode}' is not yet implemented"
        )

    raise ValueError(f"Unknown disclosure_mode: '{mode}'")
