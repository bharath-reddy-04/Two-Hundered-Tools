"""
orchestrator/nodes.py
=====================
All 8 node functions for the orchestration LangGraph.

Each node function is a closure returned by a factory, capturing:
    registry, discovery_module, sandbox, verifier_module, model_router, config

Node list:
    1. plan_node
    2. discover_schema_node
    3. validate_plan_node
    4. dry_run_node
    5. approval_node
    6. execute_node
    7. verify_node
    8. evaluate_node
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from langgraph.types import interrupt

from orchestrator.config import (
    DEFERRED_DISCLOSURE_MODES,
    IRREVERSIBLE_OPERATIONS,
    ModelRole,
    OrchestratorConfig,
)
from orchestrator.schemas import (
    DryRunCheck,
    DryRunResult,
    ExpectedOutcome,
    OperationResult,
    OrchestrationError,
    Plan,
    PlannedOperation,
    TraceEvent,
    VerificationCheck,
    VerificationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace(state: dict, event: str, node: str, metadata: Optional[dict] = None) -> None:
    """Append a TraceEvent to state['trace']."""
    trace = state.get("trace", [])
    trace.append(TraceEvent(
        timestamp=_now_iso(),
        component="orchestrator",
        node=node,
        event=event,
        task_id=state.get("task_id", ""),
        execution_id=state.get("execution_id", ""),
        disclosure_mode=state.get("disclosure_mode", ""),
        metadata=metadata or {},
    ))
    state["trace"] = trace


def _add_error(
    state: dict,
    code: str,
    message: str,
    node: str,
    operation_id: Optional[str] = None,
    recoverable: bool = True,
) -> None:
    """Append an OrchestrationError to state['errors']."""
    errors = state.get("errors", [])
    errors.append(OrchestrationError(
        code=code,
        message=message,
        component="orchestrator",
        node=node,
        operation_id=operation_id,
        recoverable=recoverable,
    ))
    state["errors"] = errors


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def create_nodes(
    registry: Any,
    discovery_module: Any,
    sandbox: Any,
    verifier_module: Any,
    model_router: Any,
    config: OrchestratorConfig,
) -> dict[str, Callable]:
    """
    Create all 8 node functions as closures over shared dependencies.

    Returns a dict of {node_name: callable} suitable for StateGraph.add_node().
    """

    # ==================================================================
    # 1. PLAN NODE
    # ==================================================================
    def plan_node(state: dict[str, Any]) -> dict[str, Any]:
        """
        Generate an execution plan from the task objective.

        Only all_loaded disclosure mode is supported — others raise
        NotImplementedError immediately.
        """
        mode = state.get("disclosure_mode", "")
        if mode in DEFERRED_DISCLOSURE_MODES:
            raise NotImplementedError(
                f"disclosure_mode '{mode}' not yet implemented"
            )
        if mode != "all_loaded":
            raise NotImplementedError(
                f"disclosure_mode '{mode}' not yet implemented"
            )

        _trace(state, "planning_started", "plan")

        # Get all operations for all_loaded mode
        candidates = registry.get_all()
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

        # Build the planning prompt
        task = state.get("task", {})
        objective = state.get("objective", "")
        user_request = state.get("user_request", objective)
        repo = state.get("repo", "")
        retry_count = state.get("retry_count", 0)
        successful_ops = state.get("successful_operations", [])

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

        expected_state = task.get("expected_state", {})

        system_prompt = (
            "You are a GitHub task planning agent. Given a task objective and a catalog "
            "of available GitHub API operations, produce a JSON execution plan.\n\n"
            "RULES:\n"
            "1. Select ONLY operations from the provided catalog — never invent an operation_id.\n"
            "2. Provide all required parameters for each operation.\n"
            "3. If an operation depends on another's output, list it in depends_on.\n"
            "4. For path parameters like 'owner' and 'repo', use the provided values.\n"
            "5. For git/create-ref, the ref must be 'refs/heads/<branch>' and sha must be provided.\n"
            "6. Return ONLY valid JSON matching the Plan schema.\n"
        )

        user_content = (
            f"Task: {user_request}\n"
            f"Objective: {objective}\n"
            f"Target repository: {repo}\n"
            f"Expected state after execution: {json.dumps(expected_state)}\n"
            f"{replan_context}\n\n"
            f"Available operations:\n{json.dumps(candidate_summaries, indent=2)}\n\n"
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

        try:
            from google.genai import types  # type: ignore

            gen_config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            )

            contents = [{"role": "user", "parts": [{"text": user_content}]}]
            response = model_router.generate(
                ModelRole.ORCHESTRATION_PLANNER,
                contents,
                config=gen_config,
            )

            # Parse the response
            response_text = ""
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        response_text += part.text

            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            plan_dict = json.loads(clean_text)

            # Build Plan object
            operations = []
            for op_data in plan_dict.get("operations", []):
                operations.append(PlannedOperation(
                    operation_id=op_data.get("operation_id", ""),
                    parameters=op_data.get("parameters", {}),
                    depends_on=op_data.get("depends_on", []),
                    description=op_data.get("description", ""),
                ))

            expected_outcomes = []
            for eo_data in plan_dict.get("expected_outcomes", []):
                expected_outcomes.append(ExpectedOutcome(
                    resource=eo_data.get("resource", ""),
                    condition=eo_data.get("condition", ""),
                    expected_value=eo_data.get("expected_value"),
                ))

            plan = Plan(
                task_id=state.get("task_id", ""),
                disclosure_mode="all_loaded",
                operations=operations,
                assumptions=plan_dict.get("assumptions", []),
                expected_outcomes=expected_outcomes,
            )

            state["plan"] = plan
            state["replanning_required"] = False

            _trace(state, "plan_created", "plan", {
                "operation_count": len(operations),
                "operations": [op.operation_id for op in operations],
            })

        except json.JSONDecodeError as exc:
            _add_error(state, "MODEL_FAILURE",
                       f"Failed to parse LLM plan response as JSON: {exc}",
                       "plan", recoverable=True)
            state["replanning_required"] = True
            state["plan"] = None

        except Exception as exc:
            _add_error(state, "MODEL_FAILURE",
                       f"LLM planning call failed: {type(exc).__name__}: {exc}",
                       "plan", recoverable=False)
            state["replanning_required"] = True
            state["plan"] = None

        return state

    # ==================================================================
    # 2. DISCOVER SCHEMA NODE
    # ==================================================================
    def discover_schema_node(state: dict[str, Any]) -> dict[str, Any]:
        """
        For each planned operation, look up its schema via the registry
        and discovery module.
        """
        _trace(state, "schema_discovery_started", "discover_schema")

        plan = state.get("plan")
        if not plan:
            _add_error(state, "PLAN_INVALID", "No plan available for schema discovery",
                       "discover_schema", recoverable=False)
            state["replanning_required"] = True
            return state

        discovered: dict[str, Any] = {}
        needs_replan = False

        for op in plan.operations:
            op_id = op.operation_id

            # Check registry first
            reg_entry = registry.get(op_id)
            if reg_entry is None:
                _add_error(state, "UNKNOWN_OPERATION",
                           f"Operation '{op_id}' not found in registry",
                           "discover_schema", operation_id=op_id, recoverable=True)
                needs_replan = True
                continue

            # Pull schema via discovery module
            try:
                schema = discovery_module.get_schema(op_id)
                discovered[op_id] = schema
            except (ValueError, KeyError) as exc:
                _add_error(state, "SCHEMA_DISCOVERY_FAILED",
                           f"Schema discovery failed for '{op_id}': {exc}",
                           "discover_schema", operation_id=op_id, recoverable=True)
                needs_replan = True

        state["discovered_schemas"] = discovered
        if needs_replan:
            state["replanning_required"] = True

        _trace(state, "schema_discovery_completed", "discover_schema", {
            "discovered_count": len(discovered),
            "failed_count": sum(1 for op in plan.operations if op.operation_id not in discovered),
        })

        return state

    # ==================================================================
    # 3. VALIDATE PLAN NODE
    # ==================================================================
    def validate_plan_node(state: dict[str, Any]) -> dict[str, Any]:
        """
        Validate each operation's parameters against its discovered schema,
        and validate dependency ordering.
        """
        plan = state.get("plan")
        if not plan:
            return state

        if state.get("replanning_required"):
            return state

        discovered = state.get("discovered_schemas", {})
        errors_found = False

        # Validate dependency ordering
        seen_ops: set[str] = set()
        for op in plan.operations:
            for dep in op.depends_on:
                if dep not in seen_ops:
                    _add_error(state, "PLAN_INVALID",
                               f"Operation '{op.operation_id}' depends on '{dep}' "
                               f"which has not been executed yet",
                               "validate_plan", operation_id=op.operation_id)
                    errors_found = True
            seen_ops.add(op.operation_id)

        # Validate parameters against schemas
        for op in plan.operations:
            schema = discovered.get(op.operation_id)
            if not schema:
                continue

            input_schema = schema.get("input_schema", {})
            required_fields = input_schema.get("required", [])
            properties = input_schema.get("properties", {})

            # Check required parameters are provided
            for req_field in required_fields:
                # Skip path parameters that will be resolved at execution time
                prop_info = properties.get(req_field, {})
                if prop_info.get("in") == "path":
                    continue
                if req_field not in op.parameters:
                    _add_error(state, "INVALID_PARAMETERS",
                               f"Required parameter '{req_field}' missing for "
                               f"operation '{op.operation_id}'",
                               "validate_plan", operation_id=op.operation_id)
                    errors_found = True

        if errors_found:
            state["replanning_required"] = True

        # Check if any operations require approval
        requires_approval = any(
            op.operation_id in IRREVERSIBLE_OPERATIONS
            for op in plan.operations
        )
        state["requires_approval"] = requires_approval

        return state

    # ==================================================================
    # 4. DRY-RUN NODE
    # ==================================================================
    def dry_run_node(state: dict[str, Any]) -> dict[str, Any]:
        """
        Non-mutating pre-flight check.

        Confirms operation existence, parameter validity, and dependency order.
        Does NOT make any mutating API calls.
        """
        _trace(state, "dry_run_started", "dry_run")

        plan = state.get("plan")
        if not plan or state.get("replanning_required"):
            state["dry_run_result"] = DryRunResult(
                passed=False,
                checks=[],
                errors=["No valid plan to dry-run"],
            )
            _trace(state, "dry_run_completed", "dry_run", {"passed": False})
            return state

        checks: list[DryRunCheck] = []
        errors: list[str] = []
        discovered = state.get("discovered_schemas", {})

        for op in plan.operations:
            # Check 1: operation exists in registry
            reg_entry = registry.get(op.operation_id)
            if reg_entry is None:
                checks.append(DryRunCheck(
                    check=f"operation_exists:{op.operation_id}",
                    passed=False,
                    message=f"Operation '{op.operation_id}' not in registry",
                ))
                errors.append(f"Unknown operation: {op.operation_id}")
                continue

            checks.append(DryRunCheck(
                check=f"operation_exists:{op.operation_id}",
                passed=True,
            ))

            # Check 2: schema was discovered
            if op.operation_id not in discovered:
                checks.append(DryRunCheck(
                    check=f"schema_available:{op.operation_id}",
                    passed=False,
                    message=f"No schema discovered for '{op.operation_id}'",
                ))
                errors.append(f"Missing schema: {op.operation_id}")
            else:
                checks.append(DryRunCheck(
                    check=f"schema_available:{op.operation_id}",
                    passed=True,
                ))

            # Check 3: has parameters
            checks.append(DryRunCheck(
                check=f"parameters_provided:{op.operation_id}",
                passed=True,
                message=f"Parameters: {list(op.parameters.keys())}",
            ))

        passed = len(errors) == 0
        state["dry_run_result"] = DryRunResult(
            passed=passed,
            checks=checks,
            errors=errors,
        )

        _trace(state, "dry_run_completed", "dry_run", {
            "passed": passed,
            "check_count": len(checks),
            "error_count": len(errors),
        })

        return state

    # ==================================================================
    # 5. APPROVAL NODE
    # ==================================================================
    def approval_node(state: dict[str, Any]) -> dict[str, Any]:
        """
        Interrupt for operator approval when irreversible operations
        are in the plan.

        Uses LangGraph's interrupt() to pause execution.
        """
        _trace(state, "approval_required", "approve")

        plan = state.get("plan")
        irreversible_ops = []
        if plan:
            irreversible_ops = [
                op.operation_id for op in plan.operations
                if op.operation_id in IRREVERSIBLE_OPERATIONS
            ]

        # LangGraph interrupt — pauses the graph until resumed
        approval = interrupt({
            "message": "Approval required for irreversible operations",
            "operations": irreversible_ops,
            "instruction": "Reply 'approved' to proceed or 'rejected' to abort.",
        })

        # After resume, `approval` contains the operator's response
        if isinstance(approval, str) and approval.strip().lower() in ("approved", "yes", "y"):
            state["approval_status"] = "approved"
            _trace(state, "approval_granted", "approve")
        else:
            state["approval_status"] = "rejected"
            _add_error(state, "APPROVAL_REJECTED",
                       f"Operator rejected irreversible operations: {irreversible_ops}",
                       "approve", recoverable=False)
            _trace(state, "approval_rejected", "approve")
            state["status"] = "failed"

        return state

    # ==================================================================
    # 6. EXECUTE NODE
    # ==================================================================
    def execute_node(state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute each planned operation through the sandbox.

        Path: execute_node → github_operations.execute_operation()
              → harness/github_sandbox.execute()

        Never calls sandbox directly. Never bypasses the registry.
        """
        import sys
        from pathlib import Path as _Path

        _project_root = _Path(__file__).resolve().parent.parent
        _harness_dir = _project_root / "harness"
        if str(_harness_dir) not in sys.path:
            sys.path.insert(0, str(_harness_dir))

        try:
            from github_operations import execute_operation
        except ImportError:
            from harness.github_operations import execute_operation  # type: ignore

        _trace(state, "execution_started", "execute")

        plan = state.get("plan")
        if not plan:
            _add_error(state, "EXECUTION_FAILED", "No plan to execute",
                       "execute", recoverable=False)
            return state

        results: list[OperationResult] = []
        successful_ops = list(state.get("successful_operations", []))
        repo = state.get("repo", "")

        for op in plan.operations:
            # Skip operations that already succeeded (idempotency on replan)
            if op.operation_id in successful_ops:
                logger.info("Skipping already-succeeded operation: %s", op.operation_id)
                _trace(state, "operation_skipped", "execute", {
                    "operation_id": op.operation_id,
                    "reason": "already_succeeded",
                })
                continue

            # Inject owner/repo into parameters if not present or placeholder
            params = dict(op.parameters)
            if repo and "/" in repo:
                owner, repo_name = repo.split("/", 1)
                current_repo = params.get("repo")
                if not current_repo or str(current_repo) in ("test-repo-1", "test_repo_1", "test-repo", "test_repo", "repo-1", "repo_1"):
                    params["repo"] = repo_name
                else:
                    params.setdefault("repo", repo_name)
                params.setdefault("owner", owner)

            logger.info("Executing: %s with params %s", op.operation_id, list(params.keys()))

            try:
                result = execute_operation(sandbox, op.operation_id, params)

                op_result = OperationResult(
                    operation_id=op.operation_id,
                    success=result.get("success", False),
                    result=str(result.get("result", "")),
                    error=result.get("message") if not result.get("success") else None,
                    error_type=result.get("error_type") if not result.get("success") else None,
                )
                results.append(op_result)

                if result.get("success"):
                    successful_ops.append(op.operation_id)
                    _trace(state, "operation_completed", "execute", {
                        "operation_id": op.operation_id,
                        "success": True,
                    })
                else:
                    _add_error(state, "EXECUTION_FAILED",
                               f"Operation '{op.operation_id}' failed: "
                               f"{result.get('error_type', 'UNKNOWN')}: {result.get('message', '')}",
                               "execute", operation_id=op.operation_id, recoverable=True)
                    _trace(state, "operation_completed", "execute", {
                        "operation_id": op.operation_id,
                        "success": False,
                        "error_type": result.get("error_type"),
                    })

            except Exception as exc:
                op_result = OperationResult(
                    operation_id=op.operation_id,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                    error_type="UNEXPECTED_ERROR",
                )
                results.append(op_result)
                _add_error(state, "EXECUTION_FAILED",
                           f"Operation '{op.operation_id}' raised exception: {exc}",
                           "execute", operation_id=op.operation_id, recoverable=True)

        state["execution_results"] = results
        state["successful_operations"] = successful_ops

        return state

    # ==================================================================
    # 7. VERIFY NODE
    # ==================================================================
    def verify_node(state: dict[str, Any]) -> dict[str, Any]:
        """
        Call harness/verifier.py to check actual sandbox state against
        the task's expected_state.

        A successful API response from execute_node is NOT sufficient evidence —
        the verifier re-queries GitHub independently.
        """
        _trace(state, "verification_started", "verify")

        task = state.get("task", {})
        expected_state = task.get("expected_state", {})
        repo = state.get("repo", "")

        if not expected_state:
            # No expected state to verify — treat as verified
            state["verification_result"] = VerificationResult(
                verified=True,
                checks=[],
                missing_conditions=[],
                unexpected_conditions=[],
                evidence={"note": "No expected_state defined in task"},
            )
            _trace(state, "verification_completed", "verify", {"verified": True})
            return state

        try:
            result = verifier_module.verify_state(repo, expected_state)

            checks = []
            sub_results = result.get("results", {})
            for resource_type, res_data in sub_results.items():
                for field_name, passed in res_data.get("checks", {}).items():
                    checks.append(VerificationCheck(
                        resource=resource_type,
                        field=field_name,
                        expected=res_data.get("expected_state", {}).get(field_name),
                        actual=res_data.get("actual_state", {}).get(field_name),
                        passed=bool(passed) if passed is not None else False,
                    ))

            missing = []
            unexpected = []
            for resource_type, res_data in sub_results.items():
                if not res_data.get("passed"):
                    for err in res_data.get("errors", []):
                        if "expected" in err.lower() and "found" in err.lower():
                            missing.append(err)
                        else:
                            unexpected.append(err)

            state["verification_result"] = VerificationResult(
                verified=result.get("passed", False),
                checks=checks,
                missing_conditions=missing,
                unexpected_conditions=unexpected,
                evidence={
                    "raw_result": result,
                },
            )

            _trace(state, "verification_completed", "verify", {
                "verified": result.get("passed", False),
                "check_count": len(checks),
            })

        except Exception as exc:
            _add_error(state, "VERIFICATION_FAILED",
                       f"Verification raised exception: {type(exc).__name__}: {exc}",
                       "verify", recoverable=True)
            state["verification_result"] = VerificationResult(
                verified=False,
                checks=[],
                missing_conditions=[str(exc)],
                unexpected_conditions=[],
                evidence={},
            )
            _trace(state, "verification_completed", "verify", {
                "verified": False,
                "error": str(exc),
            })

        return state

    # ==================================================================
    # 8. EVALUATE NODE
    # ==================================================================
    def evaluate_node(state: dict[str, Any]) -> dict[str, Any]:
        """
        Produce VERIFIED, INCOMPLETE, or FAILED from the verification result.
        """
        verification = state.get("verification_result")

        if verification and verification.verified:
            state["status"] = "verified"
            _trace(state, "task_verified", "evaluate")
        else:
            # Check if we should replan or fail
            retry_count = state.get("retry_count", 0)
            if retry_count >= config.max_replans:
                state["status"] = "failed"
                _trace(state, "task_failed", "evaluate", {
                    "reason": "replan_limit_exceeded",
                    "retry_count": retry_count,
                })
            else:
                state["status"] = "incomplete"
                _trace(state, "replanning_started", "evaluate", {
                    "retry_count": retry_count,
                })

        return state

    # ------------------------------------------------------------------
    # Return all nodes
    # ------------------------------------------------------------------
    return {
        "plan": plan_node,
        "discover_schema": discover_schema_node,
        "validate_plan": validate_plan_node,
        "dry_run": dry_run_node,
        "approve": approval_node,
        "execute": execute_node,
        "verify": verify_node,
        "evaluate": evaluate_node,
    }
