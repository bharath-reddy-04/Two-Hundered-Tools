"""
modes/all_loaded.py
===================
Disclosure mode: ALL_LOADED

Strategy
--------
The entire operation catalog is disclosed to the LLM in one shot at
planning time.  The model sees every available GitHub operation and
selects/sequences the ones it needs.

Pros:  Simple.  The LLM has full context.
Cons:  Large prompt.  Token-expensive for wide catalogs.

Public interface
----------------
    build_plan(state, registry, model_router, config) -> dict

*state* is mutated in place and returned (LangGraph node convention).
Sets:
    state["plan"]                 – Plan object on success, None on failure
    state["replanning_required"]  – True if planning failed
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from orchestrator.config import ModelRole, OrchestratorConfig
from orchestrator.llm_utils import (
    LLMEmptyResponseError,
    LLMMalformedJSONError,
    call_llm_json,
)
from orchestrator.schemas import (
    ExpectedOutcome,
    OrchestrationError,
    Plan,
    PlannedOperation,
    TraceEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_error(state: dict, code: str, message: str, recoverable: bool = True) -> None:
    """Append an OrchestrationError to state['errors']."""
    errors = state.get("errors", [])
    errors.append(
        OrchestrationError(
            code=code,
            message=message,
            component="modes.all_loaded",
            node="plan",
            recoverable=recoverable,
        )
    )
    state["errors"] = errors


def _trace(state: dict, event: str, metadata: dict | None = None) -> None:
    """Append a TraceEvent to state['trace']."""
    trace = state.get("trace", [])
    trace.append(TraceEvent(
        timestamp=_now_iso(),
        component="modes.all_loaded",
        node="plan",
        event=event,
        task_id=state.get("task_id", ""),
        execution_id=state.get("execution_id", ""),
        disclosure_mode=state.get("disclosure_mode", ""),
        metadata=metadata or {},
    ))
    state["trace"] = trace


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_plan(
    state: dict[str, Any],
    registry: Any,
    model_router: Any,
    config: OrchestratorConfig,
) -> dict[str, Any]:
    """
    Generate an execution plan with the full catalog disclosed.

    Parameters
    ----------
    state        : LangGraph state dict (mutated in place).
    registry     : OperationRegistry — provides get_all().
    model_router : ModelRouter — used to call the LLM.
    config       : OrchestratorConfig.

    Returns
    -------
    The (mutated) state dict.
    """
    # ── Step 1: collect all operations ───────────────────────────────────
    candidates = registry.get_all()
    total_ops = len(candidates)
    candidate_summaries = [
        {
            "operation_id": op.operation_id,
            "method": op.method,
            "path": op.path,
            "summary": op.summary,
            "risk": op.risk,
            "is_irreversible": op.is_irreversible,
        }
        for op in candidates
    ]

    # ── Step 2: build context ─────────────────────────────────────────────
    task = state.get("task", {})
    objective = state.get("objective", "")
    user_request = state.get("user_request", objective)
    repo = state.get("repo", "")
    retry_count = state.get("retry_count", 0)
    successful_ops = state.get("successful_operations", [])
    discovered_schemas = state.get("discovered_schemas", {})

    replan_context = ""
    if retry_count > 0:
        verification = state.get("verification_result")
        if verification:
            replan_context = (
                f"\n\nThis is replan attempt {retry_count}. "
                f"Previous verification result: verified={verification.verified}\n"
                f"Missing conditions: {verification.missing_conditions}\n"
                f"Unexpected conditions: {verification.unexpected_conditions}\n"
                f"Already-succeeded operations (DO NOT repeat): {successful_ops}\n"
                f"Plan only the remaining operations needed to fill the gap."
            )
        # Include schemas from the previous failed attempt so the model can
        # correct specific parameter mistakes rather than guessing again.
        if discovered_schemas:
            schema_hints: dict[str, Any] = {}
            for op_id, schema in discovered_schemas.items():
                input_schema = schema.get("input_schema", {})
                schema_hints[op_id] = {
                    "required": input_schema.get("required", []),
                    "properties": input_schema.get("properties", {}),
                    "starter_request": schema.get("starter_request", {}),
                }
            replan_context += (
                f"\n\nSchemas for operations from the previous attempt "
                f"(use these to correct parameter mistakes):\n"
                f"{json.dumps(schema_hints, indent=2)}"
            )

    expected_state = task.get("expected_state", {})

    # ── Step 3: build prompts ─────────────────────────────────────────────
    system_prompt = (
        "You are a GitHub task planning agent. Given a task objective and a catalog "
        "of available GitHub API operations, produce a JSON execution plan.\n\n"
        "RULES:\n"
        "1. Select ONLY operations from the provided catalog — never invent an operation_id.\n"
        "2. Provide all required parameters for each operation.\n"
        "3. If an operation depends on another's output, list it in depends_on.\n"
        "4. For path parameters like 'owner' and 'repo', use the provided values.\n"
        "5. When creating a pull request that references an issue, include the exact issue title/ID (e.g. 'eval-issue-008') in the pull request 'body' or 'title'.\n"
        "6. Return ONLY valid JSON matching the Plan schema.\n"
    )

    user_content = (
        f"Task: {user_request}\n"
        f"Objective: {objective}\n"
        f"Target repository: {repo}\n"
        f"Expected state after execution: {json.dumps(expected_state)}\n"
        f"{replan_context}\n\n"
        f"Available operations (ALL {total_ops} loaded):\n"
        f"{json.dumps(candidate_summaries, indent=2)}\n\n"
        f"Return a JSON object with this exact schema:\n"
        f'{{\n'
        f'  "task_id": "{state.get("task_id", "")}",\n'
        f'  "disclosure_mode": "all_loaded",\n'
        f'  "operations": [\n'
        f'    {{"operation_id": "...", "parameters": {{...}}, "depends_on": [], "description": "..."}}\n'
        f'  ],\n'
        f'  "assumptions": ["..."],\n'
        f'  "expected_outcomes": [\n'
        f'    {{"resource": "issue|branch|pull_request|repository", "condition": "...", "expected_value": ...}}\n'
        f'  ]\n'
        f'}}'
    )

    # ── Step 4: call the LLM ──────────────────────────────────────────────
    try:
        plan_dict = call_llm_json(
            ModelRole.ORCHESTRATION_PLANNER,
            system_prompt,
            user_content,
            model_router,
        )

        _trace(state, "plan_call_completed", {
            "candidate_count": total_ops,
            "retry_count": retry_count,
        })

        # ── Step 5: validate & build Plan ────────────────────────────────
        if not isinstance(plan_dict, dict):
            raise ValueError(f"Expected a JSON object, got {type(plan_dict).__name__}")

        operations = [
            PlannedOperation(
                operation_id=op_data.get("operation_id", ""),
                parameters=op_data.get("parameters", {}),
                depends_on=op_data.get("depends_on", []),
                description=op_data.get("description", ""),
            )
            for op_data in plan_dict.get("operations", [])
        ]

        # P2.4 — empty plan is a failure, not a completed no-op
        if not operations:
            _add_error(
                state,
                "PLAN_INVALID",
                "LLM returned a plan with zero operations — treating as planning failure",
                recoverable=True,
            )
            state["replanning_required"] = True
            state["plan"] = None
            return state

        expected_outcomes = [
            ExpectedOutcome(
                resource=eo_data.get("resource", ""),
                condition=eo_data.get("condition", ""),
                expected_value=eo_data.get("expected_value"),
            )
            for eo_data in plan_dict.get("expected_outcomes", [])
        ]

        plan = Plan(
            task_id=state.get("task_id", ""),        # always override from state
            disclosure_mode="all_loaded",             # always override from state
            operations=operations,
            assumptions=plan_dict.get("assumptions", []),
            expected_outcomes=expected_outcomes,
        )

        state["plan"] = plan
        state["replanning_required"] = False

        logger.info(
            "all_loaded: plan created — %d operations: %s",
            len(operations),
            [op.operation_id for op in operations],
        )

    except LLMEmptyResponseError as exc:
        _add_error(state, "MODEL_FAILURE", str(exc), recoverable=False)
        state["replanning_required"] = True
        state["plan"] = None

    except LLMMalformedJSONError as exc:
        _add_error(
            state,
            "PLAN_PARSE_ERROR",
            f"Plan response is not valid JSON: {exc}",
            recoverable=True,
        )
        state["replanning_required"] = True
        state["plan"] = None

    except Exception as exc:
        _add_error(
            state,
            "MODEL_FAILURE",
            f"LLM planning call failed: {type(exc).__name__}: {exc}",
            recoverable=False,
        )
        state["replanning_required"] = True
        state["plan"] = None

    return state
