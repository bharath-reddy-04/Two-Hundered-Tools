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

    # ── Verification ──────────────────────────────────────────────────────
    verification_result: Optional[VerificationResult]

    # ── Replanning ────────────────────────────────────────────────────────
    replanning_required: bool
    retry_count: int

    # ── Terminal ──────────────────────────────────────────────────────────
    status: str                       # "running" | "verified" | "failed"
    errors: list[OrchestrationError]
    trace: list[TraceEvent]
    execution_id: str
