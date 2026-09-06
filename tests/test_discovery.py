"""
tests/test_discovery.py
=======================
Unit tests for discovery.py.
Verifies that github_operations initializes the operation system,
and discovery.py acts as a read-only discovery interface querying those operations.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
except ImportError:
    pytest = None

# Step 1: Ensure github_operations is imported and initialized first
import github_operations

# Step 2: Import discovery interface
from discovery import get_schema, list_operations


class TestDiscovery(unittest.TestCase):
    """Test suite for the discovery layer."""

    def test_github_operations_initialized(self):
        """Verify github_operations holds the initialized definitions."""
        self.assertTrue(hasattr(github_operations, "OPERATION_DEFINITIONS"))
        self.assertEqual(len(github_operations.OPERATION_DEFINITIONS), 153)  # 51 * 3 aliases
        self.assertTrue(hasattr(github_operations, "get_operation_definition"))

    def test_create_pull_request(self):
        result = get_schema("create_pull_request")

        self.assertEqual(result["operation"], "create_pull_request")
        self.assertEqual(result["method"], "POST")
        self.assertEqual(result["path"], "/repos/{owner}/{repo}/pulls")
        self.assertIn("input_schema", result)
        self.assertIn("starter_request", result)

        starter = result["starter_request"]
        schema = result["input_schema"]

        # Verify starter request satisfies all required fields defined by the OpenAPI schema
        required_fields = schema.get("required", [])
        self.assertIn("head", required_fields)
        self.assertIn("base", required_fields)
        for req in required_fields:
            self.assertIn(req, starter, f"Required field '{req}' missing from starter_request")

        # Verify starter request values
        self.assertEqual(starter["head"], "octocat:new-feature")
        self.assertEqual(starter["base"], "master")

    def test_create_issue(self):
        result = get_schema("create_issue")

        self.assertEqual(result["operation"], "create_issue")
        self.assertEqual(result["method"], "POST")
        self.assertEqual(result["path"], "/repos/{owner}/{repo}/issues")
        self.assertIn("input_schema", result)
        self.assertIn("starter_request", result)

        starter = result["starter_request"]
        schema = result["input_schema"]

        required_fields = schema.get("required", [])
        self.assertIn("title", required_fields)
        for req in required_fields:
            self.assertIn(req, starter, f"Required field '{req}' missing from starter_request")

        self.assertEqual(starter["title"], "Found a bug")

    def test_unknown_operation_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            get_schema("does_not_exist")
        self.assertIn("Unknown GitHub operation", str(ctx.exception))

    def test_empty_or_none_operation_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_schema("")
        with self.assertRaises(ValueError):
            get_schema(None)  # type: ignore

    def test_mutation_safety(self):
        result1 = get_schema("create_pull_request")
        original_title = result1["starter_request"]["title"]

        # Mutate result1
        result1["starter_request"]["title"] = "MUTATED_TITLE"
        result1["starter_request"]["extra_field"] = "CORRUPTED"
        result1["input_schema"]["required"].append("new_fake_required")

        # Fetch fresh schema
        result2 = get_schema("create_pull_request")
        self.assertEqual(result2["starter_request"]["title"], original_title)
        self.assertNotIn("extra_field", result2["starter_request"])
        self.assertNotIn("new_fake_required", result2["input_schema"]["required"])

        # Also verify underlying github_operations definition was not mutated
        canon_def = github_operations.OPERATION_DEFINITIONS["pulls/create"]
        self.assertEqual(canon_def["starter_request"]["title"], original_title)
        self.assertNotIn("extra_field", canon_def["starter_request"])

    def test_ref_resolution_in_parameters(self):
        result = get_schema("create_pull_request")
        params = result["parameters"]
        self.assertTrue(len(params) > 0)

        for p in params:
            # Must NOT contain raw $ref
            self.assertNotIn("$ref", p, f"Parameter {p} still contains unresolved $ref")
            self.assertIn("name", p)
            self.assertIn("in", p)

        # Check path parameters separation
        path_param_names = [p["name"] for p in result["path_parameters"]]
        self.assertIn("owner", path_param_names)
        self.assertIn("repo", path_param_names)

    def test_all_catalog_operations_retrievable(self):
        all_ops = list_operations()
        self.assertEqual(len(all_ops), 51, "Catalog should have 51 operations")

        for op_id in all_ops:
            res = get_schema(op_id)
            self.assertEqual(res["operation_id"], op_id)
            self.assertIn(res["method"], ["GET", "POST", "PUT", "PATCH", "DELETE"])
            self.assertTrue(res["path"].startswith("/"))
            self.assertIn("input_schema", res)
            self.assertIn("starter_request", res)
            self.assertIsInstance(res["starter_request"], dict)

    def test_alternative_name_formats(self):
        # Canonical ID
        res_canon = get_schema("pulls/create")
        self.assertEqual(res_canon["operation_id"], "pulls/create")

        # Tool format (double underscore)
        res_tool = get_schema("pulls__create")
        self.assertEqual(res_tool["operation_id"], "pulls/create")

        # Snake format
        res_snake = get_schema("pulls_create")
        self.assertEqual(res_snake["operation_id"], "pulls/create")

        # Human alias
        res_alias = get_schema("create_pull_request")
        self.assertEqual(res_alias["operation_id"], "pulls/create")


# Standalone functions for pytest compatibility and exact requirement assertions
def test_create_pull_request_standalone():
    result = get_schema("create_pull_request")
    assert result["operation"] == "create_pull_request"
    assert result["method"] == "POST"
    assert result["path"] == "/repos/{owner}/{repo}/pulls"
    assert "input_schema" in result
    assert "starter_request" in result
    for req in result["input_schema"].get("required", []):
        assert req in result["starter_request"]


def test_create_issue_standalone():
    result = get_schema("create_issue")
    assert result["operation"] == "create_issue"
    assert result["method"] == "POST"
    assert result["path"] == "/repos/{owner}/{repo}/issues"
    assert "input_schema" in result
    assert "starter_request" in result
    for req in result["input_schema"].get("required", []):
        assert req in result["starter_request"]


def test_unknown_operation_pytest():
    if pytest:
        with pytest.raises(ValueError):
            get_schema("does_not_exist")


if __name__ == "__main__":
    unittest.main()
