"""
orchestrator/schemas.py
=======================
Pydantic models for the orchestration agent's data structures.

All schema models used across state, nodes, and trace are defined here.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Operation definition (registry output)
# ---------------------------------------------------------------------------

class OperationDefinition(BaseModel):
    """A single operation's metadata as exposed by the registry."""
    operation_id: str
    name: str
    method: str
    path: str
    category: str = ""
    subcategory: str = ""
    risk: str = "medium"
    summary: str = ""
    description: str = ""
    input_schema: dict = Field(default_factory=dict)
    starter_request: dict = Field(default_factory=dict)
    is_irreversible: bool = False


# ---------------------------------------------------------------------------
# Plan models
# ---------------------------------------------------------------------------

class PlannedOperation(BaseModel):
    """A single operation within a plan."""
    operation_id: str
    parameters: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    description: str = ""


class ExpectedOutcome(BaseModel):
    """An expected post-execution state assertion."""
    resource: str           # "issue", "branch", "pull_request", "repository"
    condition: str          # human-readable condition description
    expected_value: Any = None


class Plan(BaseModel):
    """The LLM-generated execution plan for one task."""
    task_id: str
    disclosure_mode: str
    operations: list[PlannedOperation] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    expected_outcomes: list[ExpectedOutcome] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Execution results
# ---------------------------------------------------------------------------

class OperationResult(BaseModel):
    """Result of executing a single operation."""
    operation_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Dry-run results
# ---------------------------------------------------------------------------

class DryRunCheck(BaseModel):
    """A single check performed during dry-run."""
    check: str
    passed: bool
    message: str = ""


class DryRunResult(BaseModel):
    """Aggregate dry-run result."""
    passed: bool
    checks: list[DryRunCheck] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Verification results
# ---------------------------------------------------------------------------

class VerificationCheck(BaseModel):
    """A single verification check."""
    resource: str
    field: str
    expected: Any = None
    actual: Any = None
    passed: bool = False


class VerificationResult(BaseModel):
    """Aggregate verification result from harness/verifier.py."""
    verified: bool
    checks: list[VerificationCheck] = Field(default_factory=list)
    missing_conditions: list[str] = Field(default_factory=list)
    unexpected_conditions: list[str] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OrchestrationError(BaseModel):
    """A typed, structured error from the orchestration pipeline."""
    code: str
    # Valid codes:
    #   UNKNOWN_OPERATION | SCHEMA_DISCOVERY_FAILED | INVALID_PARAMETERS |
    #   PLAN_INVALID | DRY_RUN_FAILED | APPROVAL_REJECTED |
    #   EXECUTION_FAILED | VERIFICATION_FAILED | REPLAN_LIMIT_EXCEEDED |
    #   MODEL_FAILURE | UNSUPPORTED_DISCLOSURE_MODE
    message: str
    component: str
    node: Optional[str] = None
    operation_id: Optional[str] = None
    recoverable: bool = True


# ---------------------------------------------------------------------------
# Trace / observability
# ---------------------------------------------------------------------------

class TraceEvent(BaseModel):
    """A structured observability event."""
    timestamp: str
    component: str
    node: Optional[str] = None
    event: str
    task_id: str
    execution_id: str
    disclosure_mode: str
    metadata: dict = Field(default_factory=dict)
