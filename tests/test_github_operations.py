"""
tests/test_github_operations.py
===============================
Unit tests for harness/github_operations.py.
Verifies registry coverage, modular handlers, repository resolution,
and dispatch behavior via execute_operation.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
HARNESS_DIR = PROJECT_ROOT / "harness"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import github_operations as gho
from github_operations import (
    OPERATIONS,
    execute_operation,
    resolve_repo,
)


def _make_mock_sandbox():
    """Build a mock GitHubSandbox."""
    sb = MagicMock()
    sb._owner = "test-owner"
    sb._gh = MagicMock()
    # By default, sb.execute runs the callable passed to it and returns a standard success dict
    def fake_execute(op_name, fn):
        try:
            res = fn()
            return {"success": True, "operation": op_name, "result": res}
        except Exception as exc:
            return {"success": False, "operation": op_name, "error_type": type(exc).__name__, "message": str(exc)}
    sb.execute.side_effect = fake_execute
    return sb


class TestOperationRegistry(unittest.TestCase):
    """Test the OPERATIONS dictionary and catalog alignment."""

    def test_catalog_operations_all_registered(self):
        catalog_path = PROJECT_ROOT / "schemas" / "operation_catalog.json"
        if not catalog_path.exists():
            self.skipTest("Catalog not found")

        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        operations = catalog.get("operations", catalog) if isinstance(catalog, dict) else catalog
        expected_ids = {item["operation_id"] for item in operations}
        self.assertEqual(len(expected_ids), 51, "Catalog should contain exactly 51 operations")

        for op_id in expected_ids:
            self.assertIn(op_id, OPERATIONS, f"Operation '{op_id}' must be in OPERATIONS registry")

    def test_tool_format_and_snake_format_aliases(self):
        # E.g. issues/create -> issues__create, issues_create
        self.assertIn("issues/create", OPERATIONS)
        self.assertIn("issues__create", OPERATIONS)
        self.assertIn("issues_create", OPERATIONS)

        self.assertIn("git/create-ref", OPERATIONS)
        self.assertIn("git__create_ref", OPERATIONS)
        self.assertIn("git_create_ref", OPERATIONS)


class TestRepositoryResolution(unittest.TestCase):
    """Test repo resolution logic."""

    def test_resolve_repo_simple(self):
        sb = _make_mock_sandbox()
        resolve_repo(sb, {"repo": "owner/my-repo"})
        sb.get_repo.assert_called_with("owner/my-repo")

    def test_resolve_repo_with_owner_prefixing(self):
        sb = _make_mock_sandbox()
        resolve_repo(sb, {"repo": "my-repo", "owner": "my-org"})
        sb.get_repo.assert_called_with("my-org/my-repo")

    def test_resolve_repo_repository_param(self):
        sb = _make_mock_sandbox()
        resolve_repo(sb, {"repository": "owner/other-repo"})
        sb.get_repo.assert_called_with("owner/other-repo")

    def test_resolve_repo_missing_raises_value_error(self):
        sb = _make_mock_sandbox()
        with self.assertRaises(ValueError):
            resolve_repo(sb, {})


class TestExecuteOperationRouting(unittest.TestCase):
    """Test execute_operation across core categories."""

    def setUp(self):
        self.sb = _make_mock_sandbox()
        self.mock_repo = MagicMock()
        self.sb.get_repo.return_value = self.mock_repo

    def test_unknown_operation_returns_error(self):
        res = execute_operation(self.sb, "fake/unknown_op", {"repo": "owner/repo"})
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "UNKNOWN_OPERATION")
        self.assertEqual(res["operation"], "fake/unknown_op")

    def test_issues_create(self):
        mock_issue = MagicMock()
        mock_issue.number = 42
        self.mock_repo.create_issue.return_value = mock_issue

        res = execute_operation(
            self.sb,
            "issues/create",
            {"repo": "owner/repo", "title": "New Bug", "body": "Details", "labels": ["bug"]},
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["operation"], "issues/create")
        self.mock_repo.create_issue.assert_called_once_with(
            title="New Bug", body="Details", labels=["bug"]
        )

    def test_issues_create_tool_format(self):
        mock_issue = MagicMock()
        self.mock_repo.create_issue.return_value = mock_issue

        res = execute_operation(
            self.sb,
            "issues__create",
            {"repo": "owner/repo", "title": "Tool Bug"},
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["operation"], "issues/create")

    def test_issues_update(self):
        mock_issue = MagicMock()
        self.mock_repo.get_issue.return_value = mock_issue

        res = execute_operation(
            self.sb,
            "issues/update",
            {"repo": "owner/repo", "issue_number": 1, "state": "closed"},
        )
        self.assertTrue(res["success"])
        self.mock_repo.get_issue.assert_called_with(1)
        mock_issue.edit.assert_called_once_with(state="closed")

    def test_repos_get(self):
        res = execute_operation(self.sb, "repos/get", {"repo": "owner/repo"})
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], self.mock_repo)

    def test_git_create_ref(self):
        full_sha = "a" * 40  # valid 40-char hex SHA
        res = execute_operation(
            self.sb,
            "git/create-ref",
            {"repo": "owner/repo", "ref": "refs/heads/new-branch", "sha": full_sha},
        )
        self.assertTrue(res["success"])
        self.mock_repo.create_git_ref.assert_called_once_with(
            ref="refs/heads/new-branch", sha=full_sha
        )

    def test_pulls_create(self):
        mock_pr = MagicMock()
        mock_pr.number = 7
        self.mock_repo.create_pull.return_value = mock_pr

        res = execute_operation(
            self.sb,
            "pulls/create",
            {"repo": "owner/repo", "title": "PR Title", "head": "feature", "base": "main"},
        )
        self.assertTrue(res["success"])
        self.mock_repo.create_pull.assert_called_once_with(
            title="PR Title",
            body="",
            head="feature",
            base="main",
            draft=False,
            maintainer_can_modify=True,
        )

    def test_contents_create_or_update(self):
        self.mock_repo.default_branch = "main"
        res = execute_operation(
            self.sb,
            "repos/create-or-update-file-contents",
            {"repo": "owner/repo", "path": "README.md", "content": "Hello", "message": "Add README"},
        )
        self.assertTrue(res["success"])
        self.mock_repo.create_file.assert_called_once_with(
            "README.md", "Add README", "Hello", branch="main"
        )

    def test_users_get_authenticated(self):
        mock_user = MagicMock()
        self.sb._gh.get_user.return_value = mock_user

        res = execute_operation(self.sb, "users/get-authenticated", {})
        self.assertTrue(res["success"])
        self.sb._gh.get_user.assert_called_once()

    def test_actions_list_repo_workflows(self):
        mock_wf = MagicMock()
        self.mock_repo.get_workflows.return_value = [mock_wf]

        res = execute_operation(self.sb, "actions/list-repo-workflows", {"repo": "owner/repo"})
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], [mock_wf])


if __name__ == "__main__":
    unittest.main()
