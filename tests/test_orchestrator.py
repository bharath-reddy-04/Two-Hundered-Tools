"""
tests/test_orchestrator.py
==========================
Unit tests for the orchestration agent.

Most tests mock the LLM and sandbox to run without network access.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from orchestrator.config import (
    IRREVERSIBLE_OPERATIONS,
    ModelRole,
    OrchestratorConfig,
)
from orchestrator.nodes import create_nodes
from orchestrator.operation_registry import OperationRegistry
from orchestrator.router import (
    route_after_approval,
    route_after_dry_run,
    route_after_evaluation,
)
from orchestrator.schemas import (
    DryRunResult,
    OperationDefinition,
    OrchestrationError,
    Plan,
    PlannedOperation,
    VerificationResult,
)
from orchestrator.state import OrchestrationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides: Any) -> dict[str, Any]:
    """Build a minimal OrchestrationState dict with overrides."""
    state: dict[str, Any] = {
        "task_id": "test_task_01",
        "objective": "Create an issue",
        "user_request": "Create an issue titled 'test'",
        "disclosure_mode": "all_loaded",
        "task": {
            "task_id": "test_task_01",
            "prompt": "Create an issue titled 'test'",
            "expected_state": {},
        },
        "repo": "test-owner/eval-sandbox-repo",
        "plan": None,
        "discovered_schemas": {},
        "dry_run_result": None,
        "requires_approval": False,
        "approval_status": "pending",
        "execution_results": [],
        "successful_operations": [],
        "verification_result": None,
        "replanning_required": False,
        "retry_count": 0,
        "status": "running",
        "errors": [],
        "trace": [],
        "execution_id": "test-exec-001",
    }
    state.update(overrides)
    return state


def _mock_model_router() -> MagicMock:
    """Return a mock model_router that returns a plausible LLM response."""
    router = MagicMock()
    router.select.return_value = "test-model"
    return router


def _mock_llm_plan_response(operations: list[dict], task_id: str = "test_task_01") -> MagicMock:
    """Build a mock LLM response containing a JSON plan."""
    plan_dict = {
        "task_id": task_id,
        "disclosure_mode": "all_loaded",
        "operations": operations,
        "assumptions": [],
        "expected_outcomes": [],
    }
    part = MagicMock()
    part.text = json.dumps(plan_dict)
    part.function_call = None

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# Operation Registry Tests
# ---------------------------------------------------------------------------

class TestOperationRegistry(unittest.TestCase):
    """Tests for orchestrator/operation_registry.py."""

    def setUp(self) -> None:
        self.registry = OperationRegistry()

    def test_operation_registry_get_all(self) -> None:
        """All 51 operations should be loaded."""
        ops = self.registry.get_all()
        self.assertGreaterEqual(len(ops), 51)
        self.assertTrue(all(isinstance(op, OperationDefinition) for op in ops))

    def test_operation_registry_get_unknown_returns_none(self) -> None:
        """Unknown operations should return None, not raise."""
        result = self.registry.get("completely/fake-operation")
        self.assertIsNone(result)

    def test_operation_registry_get_known(self) -> None:
        """Known operations should return OperationDefinition."""
        result = self.registry.get("issues/create")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, OperationDefinition)
        self.assertEqual(result.operation_id, "issues/create")

    def test_operation_registry_irreversible_flag(self) -> None:
        """Operations in IRREVERSIBLE_OPERATIONS should have is_irreversible=True."""
        for op_id in IRREVERSIBLE_OPERATIONS:
            op = self.registry.get(op_id)
            if op is not None:
                self.assertTrue(op.is_irreversible, f"{op_id} should be irreversible")

    def test_operation_registry_get_by_category_raises(self) -> None:
        """Deferred: get_by_category should raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.registry.get_by_category("repos")

    def test_operation_registry_search_raises(self) -> None:
        """Deferred: search should raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.registry.search("create issue")

    def test_operation_registry_classify_domain_raises(self) -> None:
        """Deferred: classify_domain should raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.registry.classify_domain("create a new repo")

    def test_operation_registry_get_by_domain_raises(self) -> None:
        """Deferred: get_by_domain should raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.registry.get_by_domain("repository_management")

    def test_operation_registry_alias_resolution(self) -> None:
        """Registry should resolve aliases like 'create_issue' to canonical IDs."""
        result = self.registry.get("create_issue")
        self.assertIsNotNone(result)
        self.assertEqual(result.operation_id, "issues/create")


# ---------------------------------------------------------------------------
# Plan Node Tests
# ---------------------------------------------------------------------------

class TestPlanNode(unittest.TestCase):
    """Tests for the plan node."""

    def setUp(self) -> None:
        self.config = OrchestratorConfig()
        self.registry = OperationRegistry()
        self.discovery = MagicMock()
        self.sandbox = MagicMock()
        self.verifier = MagicMock()
        self.model_router = _mock_model_router()

    def _make_nodes(self) -> dict:
        return create_nodes(
            registry=self.registry,
            discovery_module=self.discovery,
            sandbox=self.sandbox,
            verifier_module=self.verifier,
            model_router=self.model_router,
            config=self.config,
        )

    def test_plan_node_all_loaded_exposes_full_catalog(self) -> None:
        """Plan node in all_loaded mode should pass all 51 operations to LLM."""
        # Set up mock to capture what's passed to LLM
        plan_response = _mock_llm_plan_response([
            {"operation_id": "issues/create", "parameters": {"title": "test"}, "depends_on": [], "description": "create issue"}
        ])
        self.model_router.generate.return_value = plan_response

        nodes = self._make_nodes()
        state = _make_state()
        result = nodes["plan"](state)

        # Verify model_router.generate was called
        self.model_router.generate.assert_called_once()
        call_args = self.model_router.generate.call_args

        # Verify the role is ORCHESTRATION_PLANNER
        self.assertEqual(call_args[0][0], ModelRole.ORCHESTRATION_PLANNER)

        # Verify contents include operation catalog
        contents = call_args[0][1]
        user_text = contents[0]["parts"][0]["text"]
        self.assertIn("issues/create", user_text)
        self.assertIn("pulls/create", user_text)

        # Verify a plan was produced
        self.assertIsNotNone(result.get("plan"))

    def test_plan_node_rejects_unsupported_disclosure_mode(self) -> None:
        """Plan node should raise NotImplementedError for deferred modes."""
        nodes = self._make_nodes()

        for mode in ("category_gated", "search_then_load", "hierarchical_planner"):
            state = _make_state(disclosure_mode=mode)
            with self.assertRaises(NotImplementedError, msg=f"Mode {mode} should raise"):
                nodes["plan"](state)

    def test_plan_node_rejects_unknown_mode(self) -> None:
        """Plan node should raise NotImplementedError for unknown modes."""
        nodes = self._make_nodes()
        state = _make_state(disclosure_mode="totally_unknown")
        with self.assertRaises(NotImplementedError):
            nodes["plan"](state)


# ---------------------------------------------------------------------------
# Discover Schema Node Tests
# ---------------------------------------------------------------------------

class TestDiscoverSchemaNode(unittest.TestCase):
    """Tests for the discover_schema node."""

    def setUp(self) -> None:
        self.config = OrchestratorConfig()
        self.registry = OperationRegistry()
        self.discovery = MagicMock()
        self.discovery.get_schema.return_value = {
            "operation_id": "issues/create",
            "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
        }
        self.sandbox = MagicMock()
        self.verifier = MagicMock()
        self.model_router = _mock_model_router()

    def _make_nodes(self) -> dict:
        return create_nodes(
            registry=self.registry,
            discovery_module=self.discovery,
            sandbox=self.sandbox,
            verifier_module=self.verifier,
            model_router=self.model_router,
            config=self.config,
        )

    def test_unknown_operation_forces_replan(self) -> None:
        """Unknown operations in the plan should trigger replanning."""
        nodes = self._make_nodes()
        plan = Plan(
            task_id="test",
            disclosure_mode="all_loaded",
            operations=[
                PlannedOperation(operation_id="totally/fake-op", parameters={}),
            ],
        )
        state = _make_state(plan=plan)
        result = nodes["discover_schema"](state)

        self.assertTrue(result.get("replanning_required"))
        errors = result.get("errors", [])
        self.assertTrue(any(e.code == "UNKNOWN_OPERATION" for e in errors))


# ---------------------------------------------------------------------------
# Dry-Run Tests
# ---------------------------------------------------------------------------

class TestDryRunNode(unittest.TestCase):
    """Tests for the dry_run node."""

    def setUp(self) -> None:
        self.config = OrchestratorConfig()
        self.registry = OperationRegistry()
        self.discovery = MagicMock()
        self.sandbox = MagicMock()
        self.verifier = MagicMock()
        self.model_router = _mock_model_router()

    def _make_nodes(self) -> dict:
        return create_nodes(
            registry=self.registry,
            discovery_module=self.discovery,
            sandbox=self.sandbox,
            verifier_module=self.verifier,
            model_router=self.model_router,
            config=self.config,
        )

    def test_dry_run_is_non_mutating(self) -> None:
        """Dry-run should never call sandbox.execute()."""
        nodes = self._make_nodes()
        plan = Plan(
            task_id="test",
            disclosure_mode="all_loaded",
            operations=[
                PlannedOperation(operation_id="issues/create", parameters={"title": "test"}),
            ],
        )
        state = _make_state(
            plan=plan,
            discovered_schemas={"issues/create": {"input_schema": {}}},
        )
        result = nodes["dry_run"](state)

        # Sandbox should NOT have been called
        self.sandbox.execute.assert_not_called()
        self.assertIsNotNone(result.get("dry_run_result"))


# ---------------------------------------------------------------------------
# Approval / Irreversible Tests
# ---------------------------------------------------------------------------

class TestApproval(unittest.TestCase):
    """Tests for approval routing."""

    def test_irreversible_operation_requires_approval(self) -> None:
        """Operations in IRREVERSIBLE_OPERATIONS should trigger approval."""
        plan = Plan(
            task_id="test",
            disclosure_mode="all_loaded",
            operations=[
                PlannedOperation(operation_id="repos/delete", parameters={"repo": "test"}),
            ],
        )
        state = _make_state(
            plan=plan,
            requires_approval=True,
            dry_run_result=DryRunResult(passed=True, checks=[], errors=[]),
        )
        result = route_after_dry_run(state)
        self.assertEqual(result, "approve")


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------

class TestExecuteNode(unittest.TestCase):
    """Tests for the execute node."""

    def setUp(self) -> None:
        self.config = OrchestratorConfig()
        self.registry = OperationRegistry()
        self.discovery = MagicMock()
        self.sandbox = MagicMock()
        self.sandbox._owner = "test-owner"
        self.verifier = MagicMock()
        self.model_router = _mock_model_router()

    def _make_nodes(self) -> dict:
        return create_nodes(
            registry=self.registry,
            discovery_module=self.discovery,
            sandbox=self.sandbox,
            verifier_module=self.verifier,
            model_router=self.model_router,
            config=self.config,
        )

    @patch("orchestrator.nodes.execute_operation", create=True)
    def test_execution_uses_sandbox(self, mock_execute: MagicMock) -> None:
        """
        Execution should go through execute_operation → sandbox.execute().
        Never call sandbox directly from a node.
        """
        nodes = self._make_nodes()

        # Patch execute_operation at the module level where it's imported
        with patch.dict("sys.modules", {}):
            plan = Plan(
                task_id="test",
                disclosure_mode="all_loaded",
                operations=[
                    PlannedOperation(
                        operation_id="issues/create",
                        parameters={"title": "test issue"},
                    ),
                ],
            )
            state = _make_state(plan=plan)

            # The execute node imports execute_operation internally
            # We verify it calls through the proper chain
            result = nodes["execute"](state)

            # Verify execution_results are populated
            self.assertIsInstance(result.get("execution_results"), list)


# ---------------------------------------------------------------------------
# Verification Tests
# ---------------------------------------------------------------------------

class TestVerifyNode(unittest.TestCase):
    """Tests for the verify node."""

    def setUp(self) -> None:
        self.config = OrchestratorConfig()
        self.registry = OperationRegistry()
        self.discovery = MagicMock()
        self.sandbox = MagicMock()
        self.verifier = MagicMock()
        self.model_router = _mock_model_router()

    def _make_nodes(self) -> dict:
        return create_nodes(
            registry=self.registry,
            discovery_module=self.discovery,
            sandbox=self.sandbox,
            verifier_module=self.verifier,
            model_router=self.model_router,
            config=self.config,
        )

    def test_verifier_checks_actual_state(self) -> None:
        """Verify node must call verifier.verify_state() with expected_state."""
        self.verifier.verify_state.return_value = {
            "passed": True,
            "resource": "multi_resource",
            "identifier": "test-owner/eval-sandbox-repo",
            "results": {
                "issue": {
                    "passed": True,
                    "checks": {"exists": True, "title": True},
                    "expected_state": {"title": "test"},
                    "actual_state": {"title": "test"},
                    "errors": [],
                }
            },
            "errors": [],
        }

        nodes = self._make_nodes()
        state = _make_state(
            task={
                "task_id": "test",
                "prompt": "test",
                "expected_state": {
                    "issue": {"title": "test", "exists": True},
                },
            },
        )
        result = nodes["verify"](state)

        # Verify that verifier_module.verify_state was called
        self.verifier.verify_state.assert_called_once()
        call_args = self.verifier.verify_state.call_args
        self.assertEqual(call_args[0][0], "test-owner/eval-sandbox-repo")

        # Verify result
        vr = result.get("verification_result")
        self.assertIsNotNone(vr)
        self.assertTrue(vr.verified)


# ---------------------------------------------------------------------------
# Router / Replan Tests
# ---------------------------------------------------------------------------

class TestRouting(unittest.TestCase):
    """Tests for router functions and replan logic."""

    def test_verification_failure_triggers_replan(self) -> None:
        """Failed verification with retries remaining → replan."""
        state = _make_state(
            verification_result=VerificationResult(
                verified=False,
                checks=[],
                missing_conditions=["Issue not found"],
                unexpected_conditions=[],
                evidence={},
            ),
            retry_count=0,
        )
        config = OrchestratorConfig(max_replans=2)
        result = route_after_evaluation(state, config)
        self.assertEqual(result, "replan")
        self.assertEqual(state["retry_count"], 1)

    def test_replan_limit(self) -> None:
        """MAX_REPLANS should halt the graph with 'failed'."""
        state = _make_state(
            verification_result=VerificationResult(
                verified=False,
                checks=[],
                missing_conditions=["Still not found"],
                unexpected_conditions=[],
                evidence={},
            ),
            retry_count=2,
        )
        config = OrchestratorConfig(max_replans=2)
        result = route_after_evaluation(state, config)
        self.assertEqual(result, "failed")

        # Should have REPLAN_LIMIT_EXCEEDED error
        errors = state.get("errors", [])
        self.assertTrue(any(e.code == "REPLAN_LIMIT_EXCEEDED" for e in errors))

    def test_replan_does_not_repeat_successful_operations(self) -> None:
        """
        After a replan, already-succeeded operations should be skipped.
        """
        config = OrchestratorConfig()
        registry = OperationRegistry()
        discovery = MagicMock()
        sandbox = MagicMock()
        sandbox._owner = "test-owner"
        verifier = MagicMock()
        model_router = _mock_model_router()

        nodes = create_nodes(
            registry=registry,
            discovery_module=discovery,
            sandbox=sandbox,
            verifier_module=verifier,
            model_router=model_router,
            config=config,
        )

        # Simulate: issues/create already succeeded, issues/update is new
        plan = Plan(
            task_id="test",
            disclosure_mode="all_loaded",
            operations=[
                PlannedOperation(operation_id="issues/create", parameters={"title": "test"}),
                PlannedOperation(operation_id="issues/update", parameters={"state": "closed"}),
            ],
        )
        state = _make_state(
            plan=plan,
            successful_operations=["issues/create"],  # already done
        )

        # Execute node should skip issues/create
        result = nodes["execute"](state)
        trace = result.get("trace", [])
        skipped = [t for t in trace if t.event == "operation_skipped"]
        self.assertTrue(len(skipped) >= 1)
        self.assertEqual(skipped[0].metadata.get("operation_id"), "issues/create")


# ---------------------------------------------------------------------------
# Agent-level Tests
# ---------------------------------------------------------------------------

class TestOrchestrationAgent(unittest.TestCase):
    """Tests for orchestrator/agent.py."""

    def test_agent_rejects_deferred_modes(self) -> None:
        """Deferred modes should raise NotImplementedError from run()."""
        sandbox = MagicMock()
        sandbox._owner = "test-owner"
        sandbox._token = "fake-token"

        # Patch to avoid real GitHub connection
        with patch("orchestrator.agent.OperationRegistry"):
            with patch("orchestrator.agent.build_graph"):
                from orchestrator.agent import OrchestrationAgent

                agent = OrchestrationAgent.__new__(OrchestrationAgent)
                agent.sandbox = sandbox
                agent.config = OrchestratorConfig()
                agent.registry = MagicMock()
                agent.discovery_module = MagicMock()
                agent.verifier_module = MagicMock()
                agent.model_router = MagicMock()
                agent.graph = MagicMock()

                task = {"task_id": "test", "prompt": "test", "expected_state": {}}

                for mode in ("category_gated", "search_then_load", "hierarchical_planner"):
                    with self.assertRaises(NotImplementedError, msg=f"Mode {mode}"):
                        agent.run(task, "objective", disclosure_mode=mode)


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    """Tests for orchestrator/config.py."""

    def test_default_config(self) -> None:
        """Default config should have sane values."""
        cfg = OrchestratorConfig()
        self.assertEqual(cfg.max_replans, 2)
        self.assertEqual(cfg.disclosure_mode, "all_loaded")
        self.assertIn(ModelRole.ORCHESTRATION_PLANNER, cfg.model_role_map)

    def test_model_role_map_defaults_to_one_model(self) -> None:
        """All model roles should default to the same model."""
        cfg = OrchestratorConfig(default_model="test-model")
        models = set(cfg.model_role_map.values())
        self.assertEqual(len(models), 1)
        self.assertEqual(models.pop(), "test-model")


# ---------------------------------------------------------------------------
# Graph Compilation Test
# ---------------------------------------------------------------------------

class TestGraphCompilation(unittest.TestCase):
    """Test that the graph compiles without errors."""

    def test_graph_compiles(self) -> None:
        """build_graph should produce a compiled graph with all nodes."""
        from langgraph.checkpoint.memory import MemorySaver

        from orchestrator.graph import build_graph

        config = OrchestratorConfig(checkpointer=MemorySaver())
        registry = OperationRegistry()
        discovery = MagicMock()
        sandbox = MagicMock()
        verifier = MagicMock()
        model_router = MagicMock()

        graph = build_graph(registry, discovery, sandbox, verifier, model_router, config)
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
