"""
orchestrator/config.py
======================
Configuration for the orchestration agent.

Provides:
- OrchestratorConfig dataclass with all tuneable parameters
- IRREVERSIBLE_OPERATIONS set for approval gating
- ModelRole enum for model routing
- MODEL_ROLE_MAP default mapping
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Model roles
# ---------------------------------------------------------------------------

class ModelRole(str, Enum):
    """Roles that map to potentially different LLM models."""
    TASK_PLANNER = "task_planner"
    TOOL_SELECTOR = "tool_selector"
    ORCHESTRATION_PLANNER = "orchestration_planner"
    VALIDATOR = "validator"


# Default: all roles map to one model (lean default per spec)
MODEL_ROLE_MAP: dict[ModelRole, str] = {
    role: "default-model" for role in ModelRole
}


# ---------------------------------------------------------------------------
# Irreversible operations (approval required)
# ---------------------------------------------------------------------------

IRREVERSIBLE_OPERATIONS: frozenset[str] = frozenset({
    "repos/delete",
    "git/delete-ref",
    "pulls/merge",
    "repos/delete-file",
    "repos/remove-collaborator",
})


# ---------------------------------------------------------------------------
# Supported disclosure modes
# ---------------------------------------------------------------------------

SUPPORTED_DISCLOSURE_MODES = (
    "all_loaded",
    "category_gated",
)

DEFERRED_DISCLOSURE_MODES = (
    "search_then_load",
    "hierarchical_planner",
)


# ---------------------------------------------------------------------------
# Orchestrator config
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    """All tuneable parameters for one orchestration run."""

    max_replans: int = 2
    disclosure_mode: str = "all_loaded"
    auto_approve: bool = True
    default_model: str = field(
        default_factory=lambda: (os.getenv("GEMINI_MODEL") or "gemini-3.1-flash-lite").strip()
    )
    checkpointer: Any = None  # set to MemorySaver() by agent.py if None

    # Model routing: role → model name
    model_role_map: dict[ModelRole, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fill model_role_map with defaults if empty
        if not self.model_role_map:
            self.model_role_map = {
                role: self.default_model for role in ModelRole
            }

    def get_model(self, role: ModelRole) -> str:
        """Return the model name for a given role."""
        return self.model_role_map.get(role, self.default_model)
