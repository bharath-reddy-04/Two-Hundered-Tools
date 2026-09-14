"""
modes/category_gated.py
=======================
Disclosure mode: CATEGORY_GATED

Strategy
--------
Instead of dumping the entire catalog, the LLM is first asked to classify
the task into one or more operation *categories* (e.g. "issues", "pulls",
"git", "repos").  Only the operations that belong to those categories are
then disclosed for planning.

This dramatically reduces prompt size for wide catalogs while keeping the
LLM focused on the relevant tool surface.

Two-phase flow
--------------
Phase 1 – Classify
    A lightweight LLM call classifies the task objective into category names.
    Input  : task objective, list of available categories (live from registry).
    Output : list[str] of matched category names.

Phase 2 – Plan
    The standard planning call is made, but candidate_summaries only contains
    operations whose category is in the matched set.
    The Plan's disclosure_mode field is set to "category_gated".

Fallback policy (P1.1)
-----------------------
If Phase 1 produces no valid categories, or Phase 2 produces no candidates,
the mode sets ``state["degraded_to_all_loaded"] = True`` and appends a
``PLAN_DEGRADED`` error before falling back.  This is a loud, structured
signal — not a silent logger.warning — so downstream analysis can filter
these runs out of category_gated result sets.

Public interface
----------------
    build_plan(state, registry, model_router, config) -> dict

*state* is mutated in place and returned (LangGraph node convention).
Sets:
    state["plan"]                  – Plan object on success, None on failure
    state["replanning_required"]   – True if planning failed
    state["matched_categories"]    – categories used for this planning attempt
    state["degraded_to_all_loaded"]– True if fallback to full catalog happened
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

# Fallback: core categories used when classification fails entirely.
# These are the minimal set needed for most eval tasks.
_FALLBACK_CATEGORIES: list[str] = ["issues", "pulls", "git", "repos"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_error(
    state: dict,
    code: str,
    message: str,
    recoverable: bool = True,
) -> None:
    """Append an OrchestrationError to state['errors']."""
    errors = state.get("errors", [])
    errors.append(
        OrchestrationError(
            code=code,
            message=message,
            component="modes.category_gated",
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
        component="modes.category_gated",
        node="plan",
        event=event,
        task_id=state.get("task_id", ""),
        execution_id=state.get("execution_id", ""),
        disclosure_mode=state.get("disclosure_mode", ""),
        metadata=metadata or {},
    ))
    state["trace"] = trace


def _set_degraded(state: dict, reason: str) -> None:
    """
    Mark this run as degraded to all_loaded.

    Sets the structured flag and appends a PLAN_DEGRADED error so no
    downstream consumer silently averages degraded runs into category_gated
    metrics.
    """
    state["degraded_to_all_loaded"] = True
    _add_error(
        state,
        "PLAN_DEGRADED",
        f"category_gated degraded to full catalog: {reason}",
        recoverable=True,
    )
    logger.warning("category_gated: DEGRADED to all_loaded — %s", reason)


# ---------------------------------------------------------------------------
# Phase 1: classify
# ---------------------------------------------------------------------------

def _classify_categories(
    objective: str,
    user_request: str,
    known_categories: list[str],
    model_router: Any,
) -> list[str]:
    """
    Ask the LLM which operation categories are needed for this task.

    Parameters
    ----------
    objective        : Task objective string.
    user_request     : Original user request string.
    known_categories : Live category list from registry.get_all_categories().
    model_router     : ModelRouter instance.

    Returns
    -------
    list[str] of validated category names from known_categories.
    Returns _FALLBACK_CATEGORIES on any error — caller is responsible for
    checking whether results are sufficient and setting degraded flag.
    """
    system_prompt = (
        "You are a GitHub API task classifier.\n"
        "Given a task description, return a JSON array of GitHub API category names "
        "that are needed to complete the task.\n\n"
        "CATEGORY GUIDE:\n"
        "- 'git': git references, creating/deleting branches (git/create-ref), tags, git trees, blobs\n"
        "- 'branches': getting branch details (repos/get-branch), listing branches, branch merge\n"
        "- 'issues': creating/updating issues, adding labels, comments, locking\n"
        "- 'pulls': creating/updating/merging pull requests, PR comments, files\n"
        "- 'repos': creating/deleting repos, file contents, readme, topics\n"
        "- 'actions': workflows, workflow runs, logs, dispatching\n"
        "- 'commits': commit details, comparing commits\n"
        "- 'collaborators': repo collaborators\n"
        "- 'users': authenticated user info\n\n"
        "RULES:\n"
        "1. Only return names from the provided list — never invent categories.\n"
        "2. To create a branch or ref, ALWAYS include 'git' (for git/create-ref) and 'branches' (for repos/get-branch).\n"
        "3. Include ALL categories needed to complete every part of the task instruction.\n"
        "4. Return a JSON array of strings, e.g.: [\"git\", \"issues\", \"pulls\"]\n"
    )

    user_content = (
        f"Task: {user_request}\n"
        f"Objective: {objective}\n\n"
        f"Available categories:\n{json.dumps(known_categories)}\n\n"
        f"Return a JSON array of all relevant category names."
    )

    try:
        raw = call_llm_json(
            ModelRole.TOOL_SELECTOR,
            system_prompt,
            user_content,
            model_router,
        )

        # P2.2 — validate shape before filtering
        if not isinstance(raw, list):
            logger.warning(
                "category_gated: classification returned %s instead of list "
                "— treating as classification failure",
                type(raw).__name__,
            )
            return []  # empty — caller decides whether to degrade

        # Validate: only keep names that exist in the live category set
        known_set = set(known_categories)
        valid = [c for c in raw if isinstance(c, str) and c in known_set]

        if not valid:
            logger.warning(
                "category_gated: classification returned no valid categories "
                "(raw=%r) — caller will set degraded flag",
                raw,
            )
            return []  # empty — caller decides whether to degrade

        logger.info("category_gated: classified categories → %s", valid)
        return valid

    except (LLMEmptyResponseError, LLMMalformedJSONError) as exc:
        logger.warning(
            "category_gated: classification LLM error (%s: %s) — "
            "caller will set degraded flag",
            type(exc).__name__, exc,
        )
        return []  # empty — caller decides whether to degrade

    except Exception as exc:
        logger.warning(
            "category_gated: classification unexpected error (%s: %s) — "
            "caller will set degraded flag",
            type(exc).__name__, exc,
        )
        return []


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
    Generate an execution plan with only category-matched operations disclosed.

    Parameters
    ----------
    state        : LangGraph state dict (mutated in place).
    registry     : OperationRegistry — provides get_by_category() and get_all().
    model_router : ModelRouter.
    config       : OrchestratorConfig.

    Returns
    -------
    The (mutated) state dict.
    """
    # Initialise observability flags every run
    state.setdefault("degraded_to_all_loaded", False)
    state.setdefault("matched_categories", [])

    task = state.get("task", {})
    objective = state.get("objective", "")
    user_request = state.get("user_request", objective)
    repo = state.get("repo", "")
    retry_count = state.get("retry_count", 0)
    successful_ops = state.get("successful_operations", [])
    discovered_schemas = state.get("discovered_schemas", {})

    # ── P2.1: live categories from registry ───────────────────────────────
    try:
        known_categories = registry.get_all_categories()
    except Exception:
        known_categories = _FALLBACK_CATEGORIES

    # ── Phase 1: classify ─────────────────────────────────────────────────
    matched_categories = _classify_categories(
        objective, user_request, known_categories, model_router
    )

    # Record classification result in trace (P2.5 — first of two LLM calls)
    _trace(state, "classify_call_completed", {
        "known_category_count": len(known_categories),
        "matched_categories": matched_categories,
        "degraded": not bool(matched_categories),
    })

    # Handle empty classification result (P1.1 — loud, not silent)
    if not matched_categories:
        _set_degraded(state, "classification returned no valid categories")
        matched_categories = _FALLBACK_CATEGORIES

    # Ensure branch creation tools from 'git' are available whenever 'branches' is selected
    if "branches" in matched_categories and "git" in known_categories and "git" not in matched_categories:
        matched_categories.append("git")

    state["matched_categories"] = matched_categories

    # ── Retrieve candidates from matched categories ───────────────────────
    candidates: list[Any] = []
    seen: set[str] = set()

    for cat in matched_categories:
        for op in registry.get_by_category(cat):
            if op.operation_id not in seen:
                seen.add(op.operation_id)
                candidates.append(op)

    total_ops = len(registry.get_all())

    # Empty candidates → loud degradation (P1.1)
    if not candidates:
        _set_degraded(
            state,
            f"get_by_category returned 0 ops for categories {matched_categories}",
        )
        candidates = registry.get_all()

    candidate_count = len(candidates)
    logger.info(
        "category_gated: %d/%d candidates from categories %s%s",
        candidate_count,
        total_ops,
        matched_categories,
        " [DEGRADED]" if state.get("degraded_to_all_loaded") else "",
    )

    candidate_summaries = [
        {
            "operation_id": op.operation_id,
            "method": op.method,
            "path": op.path,
            "summary": op.summary,
            "risk": op.risk,
            "is_irreversible": op.is_irreversible,
            "category": op.category,        # P3 — direct attr, no getattr
        }
        for op in candidates
    ]

    # ── Phase 2: build replan context ────────────────────────────────────
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
        # P1.4 — inject schemas from previous failed attempt so the model
        # corrects specific parameter mistakes rather than guessing again.
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

    # ── Phase 2: build prompts ───────────────────────────────────────────
    system_prompt = (
        "You are a GitHub task planning agent. Given a task objective and a "
        "filtered set of GitHub API operations relevant to the task category, "
        "produce a JSON execution plan.\n\n"
        "RULES:\n"
        "1. Select ONLY operations from the provided list — never invent an operation_id.\n"
        "2. Include ALL operations needed to satisfy every part of the task (e.g. creating branches, issues, pull requests, labels, comments, merging, closing).\n"
        "3. When the task asks to create an issue, ALWAYS start by calling 'issues/create'. You can include labels directly in 'issues/create' (labels parameter) or add comments via 'issues/create-comment' and close via 'issues/update'.\n"
        "4. Provide all required parameters for each operation.\n"
        "5. If an operation depends on another's output, list it in depends_on.\n"
        "6. For path parameters like 'owner' and 'repo', use the provided values.\n"
        "7. When creating a pull request that references an issue, include the exact issue title/ID (e.g. 'eval-issue-009') in the pull request 'body' or 'title'.\n"
        "8. Return ONLY valid JSON matching the Plan schema.\n"
    )

    user_content = (
        f"Task: {user_request}\n"
        f"Objective: {objective}\n"
        f"Target repository: {repo}\n"
        f"Expected state after execution: {json.dumps(expected_state)}\n"
        f"Matched categories: {matched_categories}\n"
        f"{replan_context}\n\n"
        f"Available operations ({candidate_count} from matched categories):\n"
        f"{json.dumps(candidate_summaries, indent=2)}\n\n"
        f"Return a JSON object with this exact schema:\n"
        f'{{\n'
        f'  "task_id": "{state.get("task_id", "")}",\n'
        f'  "disclosure_mode": "category_gated",\n'
        f'  "operations": [\n'
        f'    {{"operation_id": "...", "parameters": {{...}}, "depends_on": [], "description": "..."}}\n'
        f'  ],\n'
        f'  "assumptions": ["..."],\n'
        f'  "expected_outcomes": [\n'
        f'    {{"resource": "issue|branch|pull_request|repository", "condition": "...", "expected_value": ...}}\n'
        f'  ]\n'
        f'}}'
    )

    # ── Phase 2: call LLM ────────────────────────────────────────────────
    try:
        plan_dict = call_llm_json(
            ModelRole.ORCHESTRATION_PLANNER,
            system_prompt,
            user_content,
            model_router,
        )

        # P2.5 — trace second LLM call (classify was the first)
        _trace(state, "plan_call_completed", {
            "candidate_count": candidate_count,
            "total_ops": total_ops,
            "matched_categories": matched_categories,
            "degraded": state.get("degraded_to_all_loaded", False),
            "retry_count": retry_count,
        })

        # ── Validate shape ────────────────────────────────────────────────
        if not isinstance(plan_dict, dict):
            raise ValueError(
                f"Expected a JSON object from planner, got {type(plan_dict).__name__}"
            )

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
            task_id=state.get("task_id", ""),       # always override from state
            disclosure_mode="category_gated",         # always override from state
            operations=operations,
            assumptions=plan_dict.get("assumptions", []),
            expected_outcomes=expected_outcomes,
        )

        state["plan"] = plan
        state["replanning_required"] = False

        logger.info(
            "category_gated: plan created — %d operations: %s",
            len(operations),
            [op.operation_id for op in operations],
        )

    except LLMEmptyResponseError as exc:
        # P2.3 — API/network call itself failed
        _add_error(state, "MODEL_FAILURE", str(exc), recoverable=False)
        state["replanning_required"] = True
        state["plan"] = None

    except LLMMalformedJSONError as exc:
        # P2.3 — response came back but was not valid JSON
        _add_error(
            state,
            "PLAN_PARSE_ERROR",
            f"Plan response is not valid JSON: {exc}",
            recoverable=True,
        )
        state["replanning_required"] = True
        state["plan"] = None

    except Exception as exc:
        # P2.3 — JSON parsed but Pydantic validation or shape check failed
        _add_error(
            state,
            "PLAN_INVALID",
            f"Plan construction failed: {type(exc).__name__}: {exc}",
            recoverable=True,
        )
        state["replanning_required"] = True
        state["plan"] = None

    return state
