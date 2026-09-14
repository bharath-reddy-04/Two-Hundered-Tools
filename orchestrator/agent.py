"""
orchestrator/agent.py
=====================
OrchestrationAgent — the top-level entry point for running one task.

Usage
-----
    from orchestrator.agent import OrchestrationAgent
    from harness.github_sandbox import GitHubSandbox

    sandbox = GitHubSandbox()
    agent = OrchestrationAgent(sandbox)
    result = agent.run(task, objective, disclosure_mode="all_loaded")
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from orchestrator.config import (
    DEFERRED_DISCLOSURE_MODES,
    SUPPORTED_DISCLOSURE_MODES,
    OrchestratorConfig,
)
from orchestrator.graph import build_graph
from orchestrator.model_router import ModelRouter
from orchestrator.operation_registry import OperationRegistry
from orchestrator.schemas import OrchestrationError, TraceEvent

logger = logging.getLogger(__name__)


class OrchestrationAgent:
    """
    One-task control loop: plan → dry-run → approve → execute → verify → evaluate.

    Only ``all_loaded`` disclosure mode is supported. Calling ``run()`` with
    any deferred mode raises ``NotImplementedError`` immediately.
    """

    def __init__(
        self,
        sandbox: Any,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        sandbox : GitHubSandbox
            Initialised sandbox instance.
        config : OrchestratorConfig, optional
            Override defaults.
        """
        self.sandbox = sandbox
        self.config = config or OrchestratorConfig()

        # Set up checkpointer if not provided
        if self.config.checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver
            self.config.checkpointer = MemorySaver()

        # Initialise registry (wraps github_operations + catalog)
        self.registry = OperationRegistry()

        # Import discovery and verifier modules (reuse existing, never duplicate)
        _project_root = Path(__file__).resolve().parent.parent
        _harness_dir = _project_root / "harness"
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        if str(_harness_dir) not in sys.path:
            sys.path.insert(0, str(_harness_dir))

        try:
            import discovery as _discovery
        except ImportError:
            from schemas import discovery as _discovery  # type: ignore
        self.discovery_module = _discovery

        try:
            from harness import verifier as _verifier
        except ImportError:
            import verifier as _verifier  # type: ignore
        self.verifier_module = _verifier

        # Model routing
        self.model_router = ModelRouter(self.config)

        # Build the compiled graph
        self.graph = build_graph(
            registry=self.registry,
            discovery_module=self.discovery_module,
            sandbox=self.sandbox,
            verifier_module=self.verifier_module,
            model_router=self.model_router,
            config=self.config,
        )

        logger.info("OrchestrationAgent initialised: registry=%d ops",
                     len(self.registry.get_all()))

    def run(
        self,
        task: dict[str, Any],
        objective: str,
        disclosure_mode: str = "all_loaded",
    ) -> dict[str, Any]:
        """
        Execute one task end-to-end.

        Parameters
        ----------
        task : dict
            Task dict from tasks.json (must have task_id, prompt, expected_state).
        objective : str
            Natural-language objective (produced upstream by task planner or
            directly from task["prompt"]).
        disclosure_mode : str
            Must be "all_loaded". Other modes raise NotImplementedError.

        Returns
        -------
        dict
            Structured result with: status, trace, errors, execution_results,
            verification_result, execution_id, retry_count.
        """
        # ── Gate deferred modes at the top of run() ──────────────────────
        if disclosure_mode in DEFERRED_DISCLOSURE_MODES:
            raise NotImplementedError(
                f"disclosure_mode '{disclosure_mode}' not yet implemented"
            )
        if disclosure_mode not in SUPPORTED_DISCLOSURE_MODES:
            raise NotImplementedError(
                f"disclosure_mode '{disclosure_mode}' not yet implemented"
            )

        # ── Build initial state ──────────────────────────────────────────
        execution_id = uuid.uuid4().hex[:12]
        repo = f"{self.sandbox._owner}/{os.getenv('GITHUB_REPO', 'eval-sandbox-repo')}"

        initial_state = {
            "task_id": task.get("task_id", ""),
            "objective": objective,
            "user_request": task.get("prompt", objective),
            "disclosure_mode": disclosure_mode,
            "task": task,
            "repo": repo,
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
            "execution_id": execution_id,
        }

        logger.info(
            "Starting orchestration: task=%s, mode=%s, execution_id=%s",
            task.get("task_id"), disclosure_mode, execution_id,
        )

        # ── Invoke the graph ────────────────────────────────────────────
        thread_config = {"configurable": {"thread_id": execution_id}}

        try:
            final_state = None
            for event in self.graph.stream(
                initial_state,
                config=thread_config,
                stream_mode="values",
            ):
                final_state = event

            if final_state is None:
                final_state = initial_state
                final_state["status"] = "failed"
                final_state["errors"].append(OrchestrationError(
                    code="EXECUTION_FAILED",
                    message="Graph produced no output events",
                    component="agent",
                    recoverable=False,
                ))

        except NotImplementedError:
            # Re-raise NotImplementedError for deferred modes
            raise

        except Exception as exc:
            logger.error("Graph execution failed: %s", exc, exc_info=True)
            final_state = initial_state
            final_state["status"] = "failed"
            final_state["errors"].append(OrchestrationError(
                code="EXECUTION_FAILED",
                message=f"Graph execution failed: {type(exc).__name__}: {exc}",
                component="agent",
                recoverable=False,
            ))

        # ── Build result ────────────────────────────────────────────────
        return self._build_result(final_state)

    def _build_result(self, state: dict[str, Any]) -> dict[str, Any]:
        """Convert final state into a structured result dict."""
        verification = state.get("verification_result")

        return {
            "task_id": state.get("task_id", ""),
            "execution_id": state.get("execution_id", ""),
            "disclosure_mode": state.get("disclosure_mode", ""),
            "status": state.get("status", "unknown"),
            "verified": verification.verified if verification else False,
            "retry_count": state.get("retry_count", 0),
            "plan": state.get("plan"),
            "execution_results": [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in state.get("execution_results", [])
            ],
            "verification_result": (
                verification.model_dump() if verification and hasattr(verification, "model_dump")
                else verification
            ),
            "errors": [
                e.model_dump() if hasattr(e, "model_dump") else e
                for e in state.get("errors", [])
            ],
            "attempt_history": state.get("attempt_history", []),
            "trace": [
                t.model_dump() if hasattr(t, "model_dump") else t
                for t in state.get("trace", [])
            ],
        }
