"""

orchestrator/graph.py
=====================
Builds and compiles the LangGraph StateGraph for orchestration.

Graph structure:
    START → plan → discover_schema → validate_plan → dry_run
                                                        │
                    ┌───────────────────────────────────┼─────────────┐
                    ▼                                   ▼             ▼
                 approve                             execute    (replan/failed)
                    │                                   │
                    └──────────────┬────────────────────┘
                                   ▼
                                execute → verify → evaluate
                                                     │
                           ┌─────────────────────────┼───────────────┐
                           ▼                         ▼               ▼
                       VERIFIED                  INCOMPLETE        FAILED
                        (END)                   (→ plan)           (END)
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from orchestrator.config import OrchestratorConfig
from orchestrator.nodes import create_nodes
from orchestrator.router import (
    route_after_approval,
    route_after_dry_run,
    route_after_evaluation,
)
from orchestrator.state import OrchestrationState

logger = logging.getLogger(__name__)


def build_graph(
    registry: Any,
    discovery_module: Any,
    sandbox: Any,
    verifier_module: Any,
    model_router: Any,
    config: OrchestratorConfig,
) -> Any:
    """
    Build and compile the orchestration StateGraph.

    Parameters
    ----------
    registry : OperationRegistry
    discovery_module : module with get_schema(), list_operations()
    sandbox : GitHubSandbox
    verifier_module : module with verify_state()
    model_router : ModelRouter
    config : OrchestratorConfig

    Returns
    -------
    CompiledGraph
    """
    # Create all node functions
    nodes = create_nodes(
        registry=registry,
        discovery_module=discovery_module,
        sandbox=sandbox,
        verifier_module=verifier_module,
        model_router=model_router,
        config=config,
    )

    # Build the graph
    builder = StateGraph(OrchestrationState)

    # Add all 8 nodes
    builder.add_node("plan", nodes["plan"])
    builder.add_node("discover_schema", nodes["discover_schema"])
    builder.add_node("validate_plan", nodes["validate_plan"])
    builder.add_node("dry_run", nodes["dry_run"])
    builder.add_node("approve", nodes["approve"])
    builder.add_node("execute", nodes["execute"])
    builder.add_node("verify", nodes["verify"])
    builder.add_node("evaluate", nodes["evaluate"])

    # Set entry point
    builder.set_entry_point("plan")

    # Linear edges
    builder.add_edge("plan", "discover_schema")
    builder.add_edge("discover_schema", "validate_plan")
    builder.add_edge("validate_plan", "dry_run")

    # Conditional edges after dry_run
    def _dry_run_router(state: dict[str, Any]) -> str:
        return route_after_dry_run(state, config=config)

    builder.add_conditional_edges(
        "dry_run",
        _dry_run_router,
        {
            "approve": "approve",
            "execute": "execute",
            "replan": "plan",
            "failed": END,
        },
    )

    # Conditional edges after approval
    builder.add_conditional_edges(
        "approve",
        route_after_approval,
        {
            "execute": "execute",
            "rejected": END,
        },
    )

    # Execute → verify → evaluate
    builder.add_edge("execute", "verify")
    builder.add_edge("verify", "evaluate")

    # Conditional edges after evaluation
    # Use a closure to bind config (partial causes RunnableConfig type warnings)
    def _eval_router(state: dict[str, Any]) -> str:
        return route_after_evaluation(state, config=config)

    builder.add_conditional_edges(
        "evaluate",
        _eval_router,
        {
            "complete": END,
            "replan": "plan",
            "failed": END,
        },
    )

    # Compile with checkpointer
    compile_kwargs: dict[str, Any] = {}
    if config.checkpointer is not None:
        compile_kwargs["checkpointer"] = config.checkpointer

    compiled = builder.compile(**compile_kwargs)

    logger.info(
        "Graph compiled: nodes=%s",
        list(nodes.keys()),
    )

    return compiled
