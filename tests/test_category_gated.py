"""
tests/test_category_gated.py
============================
Tests for modes/category_gated.py.

All tests run offline — no LLM API key required.
LLM calls are mocked via unittest.mock so these pass in CI without credentials.

Test matrix
-----------
P1.1  test_reduces_candidate_set          — candidates < get_all() for 3 real tasks
P1.1  test_loud_fallback_on_empty_candidates — PLAN_DEGRADED error + flag set
P1.1  test_loud_fallback_on_bad_classification — degraded flag set on bad classify
P1.2  test_no_google_genai_imports         — zero google.genai lines in source
P2.2  test_classification_shape_validation — dict result treated as failure
P2.3  test_split_error_codes               — LLMMalformedJSON → PLAN_PARSE_ERROR
P2.3  test_model_failure_error_code        — LLMEmptyResponseError → MODEL_FAILURE
P2.4  test_empty_plan_is_failure           — zero operations → replanning_required
"""

from __future__ import annotations

import json
import sys
import types as builtin_types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup so we can import from project root
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_HARNESS = _ROOT / "harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

from orchestrator.llm_utils import LLMEmptyResponseError, LLMMalformedJSONError
from orchestrator.operation_registry import OperationRegistry
from orchestrator.schemas import Plan


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def registry() -> OperationRegistry:
    """Real registry backed by the actual catalog and github_operations module."""
    return OperationRegistry()


@pytest.fixture()
def base_state() -> dict[str, Any]:
    return {
        "task_id": "test_task",
        "objective": "create a branch and open a pull request",
        "user_request": "Create a branch called 'feat-x' from 'main', then open a PR.",
        "disclosure_mode": "category_gated",
        "task": {
            "expected_state": {
                "branch": {"exists": True, "name": "feat-x"},
                "pull_request": {"exists": True, "state": "open"},
            }
        },
        "repo": "test-owner/test-repo",
        "retry_count": 0,
        "successful_operations": [],
        "discovered_schemas": {},
        "errors": [],
        "trace": [],
        "execution_id": "test-exec-001",
        "degraded_to_all_loaded": False,
        "matched_categories": [],
    }


def _make_llm_response(payload: dict | list) -> MagicMock:
    """Build a mock GenerateContentResponse returning JSON text."""
    part = MagicMock()
    part.text = json.dumps(payload)
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    return response


def _make_model_router(responses: list[dict | list]) -> MagicMock:
    """
    Mock ModelRouter whose generate() returns each response in sequence.
    First call = classify, second call = plan.
    """
    router = MagicMock()
    router.generate.side_effect = [_make_llm_response(r) for r in responses]
    return router


def _valid_plan_dict(task_id: str = "test_task") -> dict:
    return {
        "task_id": task_id,
        "disclosure_mode": "category_gated",
        "operations": [
            {
                "operation_id": "git/create-ref",
                "parameters": {
                    "owner": "test-owner",
                    "repo": "test-repo",
                    "ref": "refs/heads/feat-x",
                    "sha": "abc123",
                },
                "depends_on": [],
                "description": "Create branch feat-x",
            },
            {
                "operation_id": "pulls/create",
                "parameters": {
                    "owner": "test-owner",
                    "repo": "test-repo",
                    "title": "feat-x PR",
                    "head": "feat-x",
                    "base": "main",
                },
                "depends_on": ["git/create-ref"],
                "description": "Open PR",
            },
        ],
        "assumptions": [],
        "expected_outcomes": [],
    }


# ---------------------------------------------------------------------------
# P1.1 — candidate reduction
# ---------------------------------------------------------------------------

REAL_TASKS_FOR_REDUCTION = [
    {
        "task_id": "task_01",
        "objective": "Create an issue",
        "user_request": "Create an issue titled 'Login bug'.",
        "expected_categories": ["issues"],
    },
    {
        "task_id": "task_07",
        "objective": "Create branch and open PR",
        "user_request": "Create branch 'feat' then open a pull request.",
        "expected_categories": ["git", "pulls"],
    },
    {
        "task_id": "task_03",
        "objective": "Create a branch",
        "user_request": "Create branch 'feature-login' from 'main'.",
        "expected_categories": ["git"],
    },
]


@pytest.mark.parametrize("task_info", REAL_TASKS_FOR_REDUCTION)
def test_reduces_candidate_set(registry: OperationRegistry, task_info: dict) -> None:
    """
    P1.1 — for at least 3 real task types the category-gated candidate set
    must be strictly smaller than get_all().

    Mocks the LLM classify call to return the expected categories so we test
    the registry filtering logic, not the model.
    """
    from modes.category_gated import build_plan

    total_ops = len(registry.get_all())
    assert total_ops > 0, "Registry must load at least one operation"

    # Mock: classify returns expected_categories, plan returns a valid plan
    classify_response = task_info["expected_categories"]
    plan_response = _valid_plan_dict(task_info["task_id"])
    model_router = _make_model_router([classify_response, plan_response])

    state: dict[str, Any] = {
        "task_id": task_info["task_id"],
        "objective": task_info["objective"],
        "user_request": task_info["user_request"],
        "disclosure_mode": "category_gated",
        "task": {"expected_state": {}},
        "repo": "owner/repo",
        "retry_count": 0,
        "successful_operations": [],
        "discovered_schemas": {},
        "errors": [],
        "trace": [],
        "execution_id": "test-exec",
        "degraded_to_all_loaded": False,
        "matched_categories": [],
    }

    with patch("modes.category_gated.call_llm_json") as mock_llm:
        # First call = classify, second = plan
        mock_llm.side_effect = [
            classify_response,
            plan_response,
        ]
        result = build_plan(state, registry, model_router, MagicMock())

    # The matched categories must have produced fewer candidates than all ops
    matched = result.get("matched_categories", [])
    assert matched, "matched_categories must be non-empty"

    candidate_count = sum(len(registry.get_by_category(c)) for c in matched)
    # De-duplicate as build_plan does
    seen: set[str] = set()
    unique_count = 0
    for c in matched:
        for op in registry.get_by_category(c):
            if op.operation_id not in seen:
                seen.add(op.operation_id)
                unique_count += 1

    assert unique_count < total_ops, (
        f"category_gated for task '{task_info['task_id']}' returned {unique_count} "
        f"candidates — must be < {total_ops} (all ops). "
        f"Categories used: {matched}. "
        f"This indicates a silent collapse to all_loaded."
    )
    assert result.get("degraded_to_all_loaded") is False, (
        "Should not be degraded when classification returned valid categories"
    )


# ---------------------------------------------------------------------------
# P1.1 — loud fallback on empty candidates
# ---------------------------------------------------------------------------

def test_loud_fallback_on_empty_candidates(
    registry: OperationRegistry, base_state: dict
) -> None:
    """
    When get_by_category returns an empty list for all matched categories,
    build_plan must set degraded_to_all_loaded=True and append a PLAN_DEGRADED
    error — not silently continue.
    """
    from modes.category_gated import build_plan

    mock_registry = MagicMock()
    mock_registry.get_all.return_value = registry.get_all()
    mock_registry.get_all_categories.return_value = ["issues", "git"]
    mock_registry.get_by_category.return_value = []  # empty for every category

    classify_result = ["issues"]
    plan_result = _valid_plan_dict()

    with patch("modes.category_gated.call_llm_json") as mock_llm:
        mock_llm.side_effect = [classify_result, plan_result]
        result = build_plan(base_state, mock_registry, MagicMock(), MagicMock())

    assert result["degraded_to_all_loaded"] is True, (
        "degraded_to_all_loaded must be True when get_by_category returns empty"
    )
    error_codes = [e.code for e in result.get("errors", [])]
    assert "PLAN_DEGRADED" in error_codes, (
        f"PLAN_DEGRADED error must be present; got codes: {error_codes}"
    )


def test_loud_fallback_on_bad_classification(
    registry: OperationRegistry, base_state: dict
) -> None:
    """
    When classification returns no valid categories, build_plan must set
    degraded_to_all_loaded=True and proceed with fallback — not silently use all ops.
    """
    from modes.category_gated import build_plan

    # Classify returns categories not in known_categories
    classify_result = ["completely_made_up_category"]
    plan_result = _valid_plan_dict()

    with patch("modes.category_gated.call_llm_json") as mock_llm:
        mock_llm.side_effect = [classify_result, plan_result]
        result = build_plan(base_state, registry, MagicMock(), MagicMock())

    assert result["degraded_to_all_loaded"] is True, (
        "degraded_to_all_loaded must be True when classification returns unknown categories"
    )
    error_codes = [e.code for e in result.get("errors", [])]
    assert "PLAN_DEGRADED" in error_codes


# ---------------------------------------------------------------------------
# P1.2 — zero google.genai imports in source
# ---------------------------------------------------------------------------

def test_no_google_genai_imports() -> None:
    """
    category_gated.py and all_loaded.py must contain zero 'from google.genai'
    lines — all provider code lives in orchestrator/llm_utils.py.
    """
    mode_files = [
        _ROOT / "modes" / "category_gated.py",
        _ROOT / "modes" / "all_loaded.py",
    ]
    for path in mode_files:
        source = path.read_text(encoding="utf-8")
        assert "from google.genai" not in source, (
            f"{path.name} contains 'from google.genai' — "
            "all provider imports must live in orchestrator/llm_utils.py"
        )
        assert "import google.genai" not in source, (
            f"{path.name} contains 'import google.genai'"
        )


# ---------------------------------------------------------------------------
# P2.2 — classification shape validation
# ---------------------------------------------------------------------------

def test_classification_shape_validation(
    registry: OperationRegistry, base_state: dict
) -> None:
    """
    If the classify call returns a dict instead of a list, build_plan must
    treat it as a classification failure (degraded) rather than silently
    iterating over the wrong shape.
    """
    from modes.category_gated import build_plan

    # Classify returns a dict (wrong shape)
    classify_result = {"categories": ["issues", "git"]}  # dict, not list
    plan_result = _valid_plan_dict()

    with patch("modes.category_gated.call_llm_json") as mock_llm:
        mock_llm.side_effect = [classify_result, plan_result]
        result = build_plan(base_state, registry, MagicMock(), MagicMock())

    assert result["degraded_to_all_loaded"] is True, (
        "dict classification result must trigger degraded fallback"
    )


# ---------------------------------------------------------------------------
# P2.3 — split error codes
# ---------------------------------------------------------------------------

def test_split_error_code_plan_parse_error(base_state: dict) -> None:
    """
    LLMMalformedJSONError from the plan call must produce PLAN_PARSE_ERROR,
    not MODEL_FAILURE.
    """
    from modes.category_gated import build_plan

    classify_result = ["issues"]
    with patch("modes.category_gated.call_llm_json") as mock_llm:
        mock_llm.side_effect = [
            classify_result,
            LLMMalformedJSONError("bad json", raw_text="not-json"),
        ]
        result = build_plan(base_state, MagicMock(
            get_all_categories=lambda: ["issues"],
            get_by_category=lambda c: [],
            get_all=lambda: [],
        ), MagicMock(), MagicMock())

    error_codes = [e.code for e in result.get("errors", [])]
    assert "PLAN_PARSE_ERROR" in error_codes, (
        f"LLMMalformedJSONError must produce PLAN_PARSE_ERROR; got {error_codes}"
    )
    assert "MODEL_FAILURE" not in error_codes, (
        "PLAN_PARSE_ERROR and MODEL_FAILURE must not be conflated"
    )
    assert result["replanning_required"] is True


def test_split_error_code_model_failure(base_state: dict) -> None:
    """
    LLMEmptyResponseError from the plan call must produce MODEL_FAILURE.
    """
    from modes.category_gated import build_plan

    classify_result = ["issues"]
    with patch("modes.category_gated.call_llm_json") as mock_llm:
        mock_llm.side_effect = [
            classify_result,
            LLMEmptyResponseError("empty response"),
        ]
        result = build_plan(base_state, MagicMock(
            get_all_categories=lambda: ["issues"],
            get_by_category=lambda c: [],
            get_all=lambda: [],
        ), MagicMock(), MagicMock())

    error_codes = [e.code for e in result.get("errors", [])]
    assert "MODEL_FAILURE" in error_codes, (
        f"LLMEmptyResponseError must produce MODEL_FAILURE; got {error_codes}"
    )
    assert result["replanning_required"] is True


# ---------------------------------------------------------------------------
# P2.4 — empty plan is a failure
# ---------------------------------------------------------------------------

def test_empty_plan_is_failure(
    registry: OperationRegistry, base_state: dict
) -> None:
    """
    If the planner returns a valid JSON object with zero operations,
    build_plan must set replanning_required=True and plan=None,
    not treat it as a completed no-op.
    """
    from modes.category_gated import build_plan

    classify_result = ["issues"]
    empty_plan = {
        "task_id": "test_task",
        "disclosure_mode": "category_gated",
        "operations": [],          # zero operations
        "assumptions": [],
        "expected_outcomes": [],
    }

    with patch("modes.category_gated.call_llm_json") as mock_llm:
        mock_llm.side_effect = [classify_result, empty_plan]
        result = build_plan(base_state, registry, MagicMock(), MagicMock())

    assert result["plan"] is None, "Empty operations must set plan=None"
    assert result["replanning_required"] is True, (
        "Empty operations must set replanning_required=True"
    )
    error_codes = [e.code for e in result.get("errors", [])]
    assert "PLAN_INVALID" in error_codes, (
        f"Empty operations must produce PLAN_INVALID; got {error_codes}"
    )


# ---------------------------------------------------------------------------
# P2.5 — two trace events per task
# ---------------------------------------------------------------------------

def test_two_trace_events_per_task(
    registry: OperationRegistry, base_state: dict
) -> None:
    """
    build_plan must append exactly two LLM-call trace events per task:
    one for the classify call and one for the plan call.
    """
    from modes.category_gated import build_plan

    classify_result = ["issues"]
    plan_result = _valid_plan_dict()

    with patch("modes.category_gated.call_llm_json") as mock_llm:
        mock_llm.side_effect = [classify_result, plan_result]
        result = build_plan(base_state, registry, MagicMock(), MagicMock())

    trace_events = [
        e.event if hasattr(e, "event") else e.get("event")
        for e in result.get("trace", [])
        if (hasattr(e, "component") and e.component == "modes.category_gated")
           or (isinstance(e, dict) and e.get("component") == "modes.category_gated")
    ]

    assert "classify_call_completed" in trace_events, (
        f"classify_call_completed trace event missing; got {trace_events}"
    )
    assert "plan_call_completed" in trace_events, (
        f"plan_call_completed trace event missing; got {trace_events}"
    )
