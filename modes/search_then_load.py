"""
modes/search_then_load.py
=========================
Disclosure mode: SEARCH_THEN_LOAD  [NOT YET IMPLEMENTED]

Strategy (planned)
------------------
A semantic similarity search retrieves the top-k most relevant operations
for the task objective before the planning call.  Only those k candidates
are disclosed to the LLM.

This keeps prompt size minimal and is well-suited for wide catalogs where
category gating is still too coarse.

Two-phase flow (planned)
------------------------
Phase 1 – Embed & Retrieve
    Embed the task objective.
    Cosine-search against a pre-built operation embedding index.
    Return top-k operation IDs.

Phase 2 – Plan
    Run the standard planning prompt with only the top-k operations.

Dependencies (not yet installed)
---------------------------------
- A vector store or embedding index (e.g. sentence-transformers, faiss)
- registry.search(query, k) — already stubbed in OperationRegistry

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
        "disclosure_mode 'search_then_load' is not yet implemented. "
        "Use 'all_loaded' or 'category_gated'."
    )
