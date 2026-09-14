"""
orchestrator/state.py
=====================
LangGraph state definition for the orchestration agent.
"""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict

from orchestrator.schemas import (
    DryRunResult,
    OperationResult,
    OrchestrationError,
    Plan,
    TraceEvent,
    VerificationResult,
)


class OrchestrationState(TypedDict, total=False):
    """
    The full state threaded through every node in the orchestration graph.

    All fields are optional (total=False) so nodes can update only what they own.
    """

    # ── Task identity ─────────────────────────────────────────────────────
    task_id: str
    objective: str
    user_request: str
    disclosure_mode: str              # only "all_loaded" is valid right now

    # ── Task metadata ─────────────────────────────────────────────────────
    task: dict                        # full task dict from tasks.json
    repo: str                         # owner/repo for sandbox operations

    # ── Planning ──────────────────────────────────────────────────────────
    plan: Optional[Plan]
    discovered_schemas: dict          # op_id → schema dict

    # ── Dry-run ───────────────────────────────────────────────────────────
    dry_run_result: Optional[DryRunResult]
    requires_approval: bool
    approval_status: str              # "pending" | "approved" | "rejected"

    # ── Execution ─────────────────────────────────────────────────────────
    execution_results: list[OperationResult]
    successful_operations: list[str]  # op_ids that already succeeded (for replan)
    step_outputs: dict                # forwarded step outputs across attempts

    # ── Verification ──────────────────────────────────────────────────────
    verification_result: Optional[VerificationResult]

    # ── Replanning & Attempts ─────────────────────────────────────────────
    replanning_required: bool
    retry_count: int
    attempt_history: list[dict]       # structured per-attempt execution and duration history
    _attempt_start_time: float        # timestamp when current attempt started

    # ── Terminal ──────────────────────────────────────────────────────────
    status: str                       # "running" | "verified" | "failed"
    errors: list[OrchestrationError]
    trace: list[TraceEvent]
    execution_id: str

    # ── Disclosure-mode observability ─────────────────────────────────────
    # Set by category_gated when it falls back to the full catalog so that
    # downstream analysis can exclude these runs from mode comparisons.
    degraded_to_all_loaded: bool
    # Categories matched during Phase 1 classify — for traceability.
    matched_categories: list[str]
