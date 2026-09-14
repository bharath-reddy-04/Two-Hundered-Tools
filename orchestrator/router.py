"""
orchestrator/router.py
======================
Conditional edge functions for the LangGraph orchestration graph.

Each function takes OrchestrationState and returns a string key
that the graph uses to select the next node.
"""

from __future__ import annotations

import logging
from typing import Any

from orchestrator.config import IRREVERSIBLE_OPERATIONS, OrchestratorConfig
from orchestrator.schemas import OrchestrationError

logger = logging.getLogger(__name__)


def route_after_dry_run(state: dict[str, Any], config: OrchestratorConfig | None = None) -> str:
    """
    Decide next step after dry-run completes.

    Returns
    -------
    str
        "approve"  — if any planned operation is irreversible
        "execute"  — if dry-run passed and no approval needed
        "replan"   — if dry-run failed with recoverable errors
        "failed"   — if dry-run failed with unrecoverable errors or retry limit reached
    """
    retry_count = state.get("retry_count", 0)
    max_replans = getattr(config, "max_replans", 2) if config else 2
    if retry_count > max_replans:
        logger.info("Router: retry limit reached (%d/%d) in dry-run → failed", retry_count, max_replans)
        return "failed"

    dry_run_result = state.get("dry_run_result")

    # If replanning was flagged during schema discovery or validation
    if state.get("replanning_required"):
        logger.info("Router: replanning_required flag set → replan")
        return "replan"

    # Dry-run failure
    if dry_run_result and not dry_run_result.passed:
        errors = state.get("errors", [])
        has_unrecoverable = any(
            not e.recoverable for e in errors
            if hasattr(e, "recoverable")
        )
        if has_unrecoverable:
            logger.info("Router: unrecoverable dry-run failure → failed")
            return "failed"
        logger.info("Router: recoverable dry-run failure → replan")
        return "replan"

    # Check if approval is required
    if state.get("requires_approval"):
        logger.info("Router: approval required → approve")
        return "approve"

    logger.info("Router: dry-run passed, no approval needed → execute")
    return "execute"


def route_after_approval(state: dict[str, Any]) -> str:
    """
    Decide next step after approval node.

    Returns
    -------
    str
        "execute"  — approved
        "rejected" — rejected by operator
    """
    status = state.get("approval_status", "pending")
    if status == "approved":
        logger.info("Router: approval granted → execute")
        return "execute"
    logger.info("Router: approval rejected → rejected")
    return "rejected"


def route_after_evaluation(state: dict[str, Any], config: OrchestratorConfig) -> str:
    """
    Decide next step after evaluation node.

    Returns
    -------
    str
        "complete" — verification passed
        "replan"   — verification failed, retries remain
        "failed"   — verification failed, retry limit reached
    """
    verification_result = state.get("verification_result")

    if verification_result and verification_result.verified:
        logger.info("Router: verification passed → complete")
        return "complete"

    retry_count = state.get("retry_count", 0)
    if retry_count >= config.max_replans:
        errors = state.get("errors", [])
        errors.append(OrchestrationError(
            code="REPLAN_LIMIT_EXCEEDED",
            message=f"Maximum replans ({config.max_replans}) exceeded. "
                    f"Verification still failing after {retry_count} attempts.",
            component="router",
            node="evaluate",
            operation_id=None,
            recoverable=False,
        ))
        state["errors"] = errors
        logger.info("Router: replan limit exceeded → failed")
        return "failed"

    state["retry_count"] = retry_count + 1
    logger.info("Router: verification failed, retry %d → replan", state["retry_count"])
    return "replan"
