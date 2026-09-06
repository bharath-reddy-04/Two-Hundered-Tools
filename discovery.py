"""
discovery.py
============
Read-only schema and starter request discovery layer for the GitHub operations framework.

Delegates schema and starter request retrieval to the initialized operation system
in github_operations.py. Does not duplicate schemas or maintain a second registry.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent
_HARNESS_DIR = _PROJECT_ROOT / "harness"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

try:
    import github_operations
except ImportError:
    from harness import github_operations  # type: ignore


def get_schema(operation_name: str) -> Dict[str, Any]:
    """
    Retrieve the discovery metadata, exact input schema, and starter request for a GitHub operation.

    Parameters
    ----------
    operation_name : str
        The operation identifier (e.g. 'create_pull_request', 'create_issue', 'pulls/create').

    Returns
    -------
    dict
        A structure containing:
        - 'operation': The requested operation name.
        - 'operation_id': Canonical operation ID.
        - 'method': HTTP method.
        - 'path': Endpoint path.
        - 'summary': Summary documentation.
        - 'description': Detailed description.
        - 'input_schema': Complete input schema from the operation definition.
        - 'starter_request': Valid starter request satisfying required fields.
        - 'parameters': List of resolved path/query parameters.
        - 'path_parameters': Resolved path parameters.
        - 'query_parameters': Resolved query parameters.

    Raises
    ------
    ValueError
        If operation_name is not recognized or not loaded.
    """
    meta = github_operations.get_operation_definition(operation_name)

    result = {
        "operation": operation_name,
        "operation_id": meta["operation_id"],
        "method": meta["method"],
        "path": meta["path"],
        "summary": meta.get("summary", ""),
        "description": meta.get("description", ""),
        "risk": meta.get("risk", "medium"),
        "input_schema": meta["input_schema"],
        "starter_request": meta["starter_request"],
        "parameters": meta.get("parameters", []),
        "path_parameters": meta.get("path_parameters", []),
        "query_parameters": meta.get("query_parameters", []),
    }

    return copy.deepcopy(result)


def list_operations() -> List[str]:
    """Return a sorted list of all available canonical operation IDs."""
    return github_operations.list_operations()
