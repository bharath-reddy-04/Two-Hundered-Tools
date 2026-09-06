"""
tests/test_verifier.py
======================
Unit and integration tests for verifier.py.

Unit tests run with no GitHub credentials — all PyGithub calls are mocked.

Integration tests (marked @pytest.mark.integration) require:
    export GITHUB_TOKEN=<pat-with-repo-scope>
    export GITHUB_OWNER=<your-github-username-or-org>

Run unit tests only (no credentials needed):
    python3 -m <runner>  tests/test_verifier.py

Run integration tests (needs live sandbox):
    pytest tests/test_verifier.py -v -m integration
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

try:
    import pytest  # type: ignore
except ModuleNotFoundError:
    import types as _types

    pytest = _types.ModuleType("pytest")

    def _raises(exc, match=None):
        import re

        class _Ctx:
            def __init__(self):
                self.value = None

            def __enter__(self):
                return self

            def __exit__(self, tp, val, tb):
                if tp is None:
                    raise AssertionError(f"Expected {exc!r} to be raised")
                if not issubclass(tp, exc):
                    return False
                self.value = val
                if match and not re.search(match, str(val)):
                    raise AssertionError(
                        f"Exception message {str(val)!r} did not match {match!r}"
                    )
                return True

        return _Ctx()

    pytest.raises = _raises

    class _mark:
        @staticmethod
        def integration(fn_or_cls):
            msg = "integration — needs GITHUB_TOKEN/GITHUB_OWNER and live sandbox"
            skip_deco = unittest.skip(msg)
            if isinstance(fn_or_cls, type):
                setattr(fn_or_cls, "__unittest_skip__", True)
                setattr(fn_or_cls, "__unittest_skip_why__", msg)
                for attr in list(vars(fn_or_cls)):
                    if attr.startswith("test"):
                        setattr(fn_or_cls, attr, skip_deco(getattr(fn_or_cls, attr)))
                return fn_or_cls
            return skip_deco(fn_or_cls)

    pytest.mark = _mark

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))

import verifier
from verifier import (
    VERIFIER_MAP,
    VerifierConfigError,
    _reset_client,
    _infra_fail,
    format_result,
    verify,
    verify_issue,
    verify_issue_absent,
    verify_repository,
    verify_repository_absent,
    verify_repository_exists,
    verify_branch,
    verify_branch_absent,
    verify_pull_request,
    verify_state,
)

from github import GithubException, RateLimitExceededException, UnknownObjectException


# ── shared mock factories ────────────────────────────────────────────────────

def _make_repo_mock(
    name="test-repo",
    full_name="owner/test-repo",
    description="A sandbox repo",
    private=True,
    default_branch="main",
    archived=False,
    fork=False,
    has_issues=True,
    has_wiki=True,
    has_projects=True,
    has_discussions=False,
    allow_squash_merge=True,
    allow_merge_commit=True,
    allow_rebase_merge=True,
    allow_auto_merge=False,
    delete_branch_on_merge=False,
    topics=None,
):
    r = MagicMock()
    r.name = name
    r.full_name = full_name
    r.description = description
    r.private = private
    r.default_branch = default_branch
    r.archived = archived
    r.fork = fork
    r.has_issues = has_issues
    r.has_wiki = has_wiki
    r.has_projects = has_projects
    r.has_discussions = has_discussions
    r.allow_squash_merge = allow_squash_merge
    r.allow_merge_commit = allow_merge_commit
    r.allow_rebase_merge = allow_rebase_merge
    r.allow_auto_merge = allow_auto_merge
    r.delete_branch_on_merge = delete_branch_on_merge
    r.get_topics.return_value = topics or []
    return r


def _make_label_mock(name):
    lb = MagicMock()
    lb.name = name
    return lb


def _make_user_mock(login):
    u = MagicMock()
    u.login = login
    return u


def _make_issue_mock(
    number=1,
    title="Fix login bug",
    body="Some body text",
    state="open",
    locked=False,
    active_lock_reason=None,
    comments=0,
    labels=None,
    assignees=None,
    milestone=None,
    user_login="octocat",
):
    issue = MagicMock()
    issue.number = number
    issue.title = title
    issue.body = body
    issue.state = state
    issue.locked = locked
    issue.active_lock_reason = active_lock_reason
    issue.comments = comments
    issue.labels = [_make_label_mock(lb) for lb in (labels or [])]
    issue.assignees = [_make_user_mock(a) for a in (assignees or [])]
    issue.milestone = None if milestone is None else SimpleNamespace(title=milestone)
    issue.user = _make_user_mock(user_login)
    return issue


def _patch_fetch_repo(repo_mock=None, error=None):
    """
    Context manager that patches verifier._fetch_repo to return either
    a repo mock or an error tuple.
    """
    def _side(*args, **kwargs):
        if error:
            return None, error
        return repo_mock, None

    return patch("verifier._fetch_repo", side_effect=_side)


def _patch_fetch_issue(issue_mock=None, repo_error=None, issue_error=None):
    """Patch verifier._fetch_issue."""
    def _side(*args, **kwargs):
        if repo_error:
            return None, repo_error
        if issue_error:
            return None, issue_error
        return issue_mock, None

    return patch("verifier._fetch_issue", side_effect=_side)


# ── tests ────────────────────────────────────────────────────────────────────

class TestVerifierMap(unittest.TestCase):
    """Verify that VERIFIER_MAP contains the expected catalog operations."""

    def test_issues_create_mapped(self):
        self.assertIn("issues/create", VERIFIER_MAP)

    def test_issues_update_mapped(self):
        self.assertIn("issues/update", VERIFIER_MAP)

    def test_issues_lock_mapped(self):
        self.assertIn("issues/lock", VERIFIER_MAP)

    def test_repos_create_mapped(self):
        self.assertIn("repos/create-for-authenticated-user", VERIFIER_MAP)

    def test_repos_update_mapped(self):
        self.assertIn("repos/update", VERIFIER_MAP)

    def test_repos_delete_mapped(self):
        self.assertIn("repos/delete", VERIFIER_MAP)

    def test_repos_delete_maps_to_absent(self):
        self.assertEqual(VERIFIER_MAP["repos/delete"], "verify_repository_absent")

    def test_repos_create_maps_to_exists(self):
        self.assertEqual(
            VERIFIER_MAP["repos/create-for-authenticated-user"],
            "verify_repository_exists",
        )

    def test_issues_create_maps_to_verify_issue(self):
        self.assertEqual(VERIFIER_MAP["issues/create"], "verify_issue")


# ---------------------------------------------------------------------------
# Repository verification — unit tests
# ---------------------------------------------------------------------------


class TestVerifyRepositoryExists(unittest.TestCase):
    """Test 1 – repository exists."""

    def test_passes_when_repo_found(self):
        mock_repo = _make_repo_mock()
        with _patch_fetch_repo(repo_mock=mock_repo):
            result = verify_repository_exists("owner/test-repo")
        self.assertTrue(result["passed"])
        self.assertEqual(result["resource"], "repository")
        self.assertTrue(result["checks"]["exists"])
        self.assertEqual(result["errors"], [])

    def test_fails_when_repo_not_found(self):
        with _patch_fetch_repo(error=("NOT_FOUND", "Repo not found.")):
            result = verify_repository_exists("owner/missing-repo")
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["exists"])
        self.assertTrue(len(result["errors"]) > 0)

    def test_infra_error_not_treated_as_found(self):
        with _patch_fetch_repo(error=("AUTHENTICATION_ERROR", "Bad credentials.")):
            result = verify_repository_exists("owner/test-repo")
        self.assertFalse(result["passed"])
        self.assertEqual(result["error_kind"], "AUTHENTICATION_ERROR")


class TestVerifyRepositoryState(unittest.TestCase):
    """Test 2 – repository state fields."""

    def _run(self, expected, **repo_kwargs):
        mock_repo = _make_repo_mock(**repo_kwargs)
        with _patch_fetch_repo(repo_mock=mock_repo):
            return verify_repository("owner/test-repo", expected)

    def test_name_match(self):
        result = self._run({"name": "test-repo"}, name="test-repo")
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["name"])

    def test_name_mismatch(self):
        result = self._run({"name": "wrong-name"}, name="test-repo")
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["name"])
        self.assertTrue(any("name" in e for e in result["errors"]))

    def test_private_true_passes(self):
        result = self._run({"private": True}, private=True)
        self.assertTrue(result["passed"])

    def test_private_mismatch_fails(self):
        """Test 4 – incorrect expected state → FAIL."""
        result = self._run({"private": True}, private=False)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["private"])

    def test_description_passes(self):
        result = self._run({"description": "Sandbox"}, description="Sandbox")
        self.assertTrue(result["passed"])

    def test_default_branch_passes(self):
        result = self._run({"default_branch": "main"}, default_branch="main")
        self.assertTrue(result["passed"])

    def test_multiple_fields_all_pass(self):
        result = self._run(
            {"name": "test-repo", "private": True, "default_branch": "main"},
            name="test-repo", private=True, default_branch="main",
        )
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))

    def test_multiple_fields_one_fails(self):
        result = self._run(
            {"name": "test-repo", "private": True, "default_branch": "develop"},
            name="test-repo", private=True, default_branch="main",
        )
        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"]["name"])
        self.assertTrue(result["checks"]["private"])
        self.assertFalse(result["checks"]["default_branch"])

    def test_empty_expected_state_passes_if_repo_exists(self):
        result = self._run({})
        self.assertTrue(result["passed"])

    def test_unsupported_field_recorded_not_hard_fail(self):
        result = self._run({"totally_fake_field": "value"})
        # Should not raise; the unknown field is noted in errors but
        # there are no False checks so it still passes
        self.assertIn("totally_fake_field", result["checks"])


class TestVerifyRepositoryAbsent(unittest.TestCase):
    """Test 3 – repository absent."""

    def test_passes_when_repo_not_found(self):
        with _patch_fetch_repo(error=("NOT_FOUND", "Repo not found.")):
            result = verify_repository_absent("owner/deleted-repo")
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["absent"])

    def test_fails_when_repo_still_exists(self):
        mock_repo = _make_repo_mock()
        with _patch_fetch_repo(repo_mock=mock_repo):
            result = verify_repository_absent("owner/test-repo")
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["absent"])
        self.assertTrue(len(result["errors"]) > 0)

    def test_auth_error_not_treated_as_absent(self):
        with _patch_fetch_repo(error=("AUTHENTICATION_ERROR", "Bad credentials.")):
            result = verify_repository_absent("owner/test-repo")
        self.assertFalse(result["passed"])
        self.assertEqual(result["error_kind"], "AUTHENTICATION_ERROR")

    def test_rate_limit_not_treated_as_absent(self):
        with _patch_fetch_repo(error=("RATE_LIMITED", "Rate limit exceeded.")):
            result = verify_repository_absent("owner/test-repo")
        self.assertFalse(result["passed"])
        self.assertEqual(result["error_kind"], "RATE_LIMITED")


# ---------------------------------------------------------------------------
# Issue verification — unit tests
# ---------------------------------------------------------------------------


class TestVerifyIssueExists(unittest.TestCase):
    """Test 5 – issue exists with correct state."""

    def _run(self, expected, issue_number=1, **issue_kwargs):
        mock_issue = _make_issue_mock(number=issue_number, **issue_kwargs)
        with _patch_fetch_issue(issue_mock=mock_issue):
            return verify_issue(
                "owner/test-repo",
                expected_state=expected,
                issue_number=issue_number,
            )

    def test_title_match_passes(self):
        result = self._run({"title": "Fix login bug"}, title="Fix login bug")
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["title"])

    def test_state_open_passes(self):
        result = self._run({"state": "open"}, state="open")
        self.assertTrue(result["passed"])

    def test_body_match_passes(self):
        result = self._run({"body": "hello"}, body="hello")
        self.assertTrue(result["passed"])

    def test_body_strips_whitespace(self):
        """Body comparison is whitespace-normalised."""
        result = self._run({"body": "hello"}, body="hello\n")
        self.assertTrue(result["passed"])

    def test_labels_subset_passes_by_default(self):
        """Default: expected labels must be present; extra labels allowed."""
        result = self._run(
            {"labels": ["bug"]},
            labels=["bug", "backend"],
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["labels"])

    def test_labels_strict_extra_fails(self):
        """strict_labels=True: exact match required."""
        mock_issue = _make_issue_mock(number=1, labels=["bug", "backend"])
        with _patch_fetch_issue(issue_mock=mock_issue):
            result = verify_issue(
                "owner/test-repo",
                expected_state={"labels": ["bug"]},
                issue_number=1,
                strict_labels=True,
            )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["labels"])

    def test_multiple_fields_pass(self):
        result = self._run(
            {"title": "Fix login bug", "state": "open", "labels": ["bug"]},
            title="Fix login bug",
            state="open",
            labels=["bug", "priority"],
        )
        self.assertTrue(result["passed"])

    def test_empty_expected_passes(self):
        result = self._run({})
        self.assertTrue(result["passed"])


class TestVerifyIssueTitleMismatch(unittest.TestCase):
    """Test 6 – title mismatch → FAIL."""

    def test_wrong_title_fails(self):
        mock_issue = _make_issue_mock(title="Authentication bug")
        with _patch_fetch_issue(issue_mock=mock_issue):
            result = verify_issue(
                "owner/test-repo",
                expected_state={"title": "Login bug"},
                issue_number=1,
            )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["title"])
        self.assertTrue(any("Login bug" in e for e in result["errors"]))


class TestVerifyIssueMissingLabel(unittest.TestCase):
    """Test 7 – missing label → FAIL."""

    def test_missing_label_fails(self):
        mock_issue = _make_issue_mock(labels=[])
        with _patch_fetch_issue(issue_mock=mock_issue):
            result = verify_issue(
                "owner/test-repo",
                expected_state={"labels": ["bug"]},
                issue_number=1,
            )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["labels"])
        self.assertTrue(any("bug" in e for e in result["errors"]))


class TestVerifyIssueAbsent(unittest.TestCase):
    """Test 8 – verify_issue_absent."""

    def test_passes_on_404(self):
        with _patch_fetch_issue(issue_error=("NOT_FOUND", "Issue #99 not found.")):
            result = verify_issue_absent("owner/test-repo", 99)
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["absent"])

    def test_fails_when_issue_exists(self):
        mock_issue = _make_issue_mock(number=1)
        with _patch_fetch_issue(issue_mock=mock_issue):
            result = verify_issue_absent("owner/test-repo", 1)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["absent"])

    def test_auth_error_not_treated_as_absent(self):
        with _patch_fetch_issue(repo_error=("AUTHENTICATION_ERROR", "Bad creds.")):
            result = verify_issue_absent("owner/test-repo", 1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["error_kind"], "AUTHENTICATION_ERROR")

    def test_rate_limit_not_treated_as_absent(self):
        with _patch_fetch_issue(issue_error=("RATE_LIMITED", "Rate limit.")):
            result = verify_issue_absent("owner/test-repo", 1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["error_kind"], "RATE_LIMITED")


class TestVerifyIssueLookupByTitle(unittest.TestCase):
    """Title-based lookup edge cases."""

    def test_single_match_uses_that_issue(self):
        mock_issue = _make_issue_mock(number=3, title="Exact title")
        with patch("verifier._search_issue_by_title", return_value=(mock_issue, None)):
            result = verify_issue(
                "owner/test-repo",
                expected_state={"title": "Exact title"},
                title="Exact title",
            )
        self.assertTrue(result["passed"])

    def test_ambiguous_title_returns_fail(self):
        with patch(
            "verifier._search_issue_by_title",
            return_value=(
                None,
                ("AMBIGUOUS_RESOURCE", "Multiple issues with that title found."),
            ),
        ):
            result = verify_issue(
                "owner/test-repo",
                expected_state={"title": "Duplicate"},
                title="Duplicate",
            )
        self.assertFalse(result["passed"])
        self.assertEqual(result["error_kind"], "AMBIGUOUS_RESOURCE")

    def test_no_match_returns_fail(self):
        with patch(
            "verifier._search_issue_by_title",
            return_value=(None, ("NOT_FOUND", "No issue found.")),
        ):
            result = verify_issue(
                "owner/test-repo",
                expected_state={},
                title="Ghost issue",
            )
        self.assertFalse(result["passed"])

    def test_no_lookup_key_returns_infra_fail(self):
        result = verify_issue("owner/test-repo", expected_state={})
        self.assertFalse(result["passed"])
        self.assertEqual(result["error_kind"], "INVALID_ARGUMENTS")


class TestVerifyIssueStateMismatch(unittest.TestCase):
    """State field checks."""

    def test_closed_vs_open_fails(self):
        mock_issue = _make_issue_mock(state="closed")
        with _patch_fetch_issue(issue_mock=mock_issue):
            result = verify_issue(
                "owner/test-repo",
                expected_state={"state": "open"},
                issue_number=1,
            )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["state"])

    def test_locked_true_passes(self):
        mock_issue = _make_issue_mock(locked=True)
        with _patch_fetch_issue(issue_mock=mock_issue):
            result = verify_issue(
                "owner/test-repo",
                expected_state={"locked": True},
                issue_number=1,
            )
        self.assertTrue(result["passed"])


# ---------------------------------------------------------------------------
# Generic verify() entry point
# ---------------------------------------------------------------------------


class TestGenericVerify(unittest.TestCase):
    """verify() routing tests."""

    def test_unknown_operation_id(self):
        result = verify("nonexistent/operation", {}, {"repo": "owner/repo"})
        self.assertFalse(result["passed"])
        self.assertEqual(result["error_kind"], "UNKNOWN_OPERATION")

    def test_missing_repo_in_context(self):
        result = verify("issues/create", {}, {})
        self.assertFalse(result["passed"])
        self.assertEqual(result["error_kind"], "INVALID_ARGUMENTS")

    def test_routes_issues_create_to_verify_issue(self):
        mock_issue = _make_issue_mock()
        with _patch_fetch_issue(issue_mock=mock_issue):
            result = verify(
                "issues/create",
                {"title": "Fix login bug"},
                {"repo": "owner/test-repo", "issue_number": 1},
            )
        self.assertTrue(result["passed"])
        self.assertEqual(result["resource"], "issue")

    def test_routes_repos_delete_to_absent(self):
        with _patch_fetch_repo(error=("NOT_FOUND", "Gone.")):
            result = verify(
                "repos/delete",
                {},
                {"repo": "owner/test-repo"},
            )
        self.assertTrue(result["passed"])
        self.assertEqual(result["resource"], "repository")
        self.assertTrue(result["checks"]["absent"])

    def test_routes_repos_create_to_exists(self):
        mock_repo = _make_repo_mock()
        with _patch_fetch_repo(repo_mock=mock_repo):
            result = verify(
                "repos/create-for-authenticated-user",
                {},
                {"repo": "owner/test-repo"},
            )
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["exists"])

    def test_routes_repos_update_to_verify_repository(self):
        mock_repo = _make_repo_mock(description="New description")
        with _patch_fetch_repo(repo_mock=mock_repo):
            result = verify(
                "repos/update",
                {"description": "New description"},
                {"repo": "owner/test-repo"},
            )
        self.assertTrue(result["passed"])


class TestVerifyBranch(unittest.TestCase):
    def test_verify_branch_success(self):
        mock_branch = MagicMock()
        mock_branch.name = "feature-login"
        mock_branch.protected = False
        with patch("verifier._fetch_branch", return_value=(mock_branch, None)):
            res = verify_branch("owner/repo", "feature-login", {"name": "feature-login"})
        self.assertTrue(res["passed"])
        self.assertEqual(res["resource"], "branch")

    def test_verify_branch_not_found(self):
        with patch("verifier._fetch_branch", return_value=(None, ("NOT_FOUND", "Not found"))):
            res = verify_branch("owner/repo", "nonexistent")
        self.assertFalse(res["passed"])
        self.assertFalse(res["checks"]["exists"])


class TestVerifyPullRequest(unittest.TestCase):
    def test_verify_pr_success(self):
        mock_pr = MagicMock()
        mock_pr.number = 10
        mock_pr.title = "Add login feature"
        mock_pr.head.ref = "feature-login"
        mock_pr.base.ref = "main"
        mock_pr.state = "open"
        mock_pr.merged = False
        mock_pr.body = "Implements feature"
        with patch("verifier._fetch_pull", return_value=(mock_pr, None)):
            res = verify_pull_request(
                "owner/repo",
                title="Add login feature",
                expected_state={"head": "feature-login", "base": "main", "state": "open"},
            )
        self.assertTrue(res["passed"])
        self.assertEqual(res["resource"], "pull_request")

    def test_verify_pr_merged(self):
        mock_pr = MagicMock()
        mock_pr.number = 10
        mock_pr.title = "Merged PR"
        mock_pr.head.ref = "feature-login"
        mock_pr.base.ref = "main"
        mock_pr.state = "closed"
        mock_pr.merged = True
        with patch("verifier._fetch_pull", return_value=(mock_pr, None)):
            res = verify_pull_request(
                "owner/repo",
                title="Merged PR",
                expected_state={"merged": True},
            )
        self.assertTrue(res["passed"])


class TestVerifyState(unittest.TestCase):
    def test_verify_state_multi_resource(self):
        mock_repo = _make_repo_mock(name="eval-repo")
        mock_issue = _make_issue_mock(title="eval-issue")
        with patch("verifier._fetch_repo", return_value=(mock_repo, None)), \
             patch("verifier._fetch_issue", return_value=(mock_issue, None)), \
             patch("verifier._search_issue_by_title", return_value=(mock_issue, None)):
            res = verify_state("owner/eval-repo", {
                "repository": {"exists": True, "name": "eval-repo"},
                "issue": {"exists": True, "title": "eval-issue", "state": "open"},
            })
        self.assertTrue(res["passed"])
        self.assertEqual(res["resource"], "multi_resource")



# ---------------------------------------------------------------------------
# Error kind propagation
# ---------------------------------------------------------------------------


class TestErrorKindPropagation(unittest.TestCase):
    def test_repo_rate_limit_propagated(self):
        with _patch_fetch_repo(error=("RATE_LIMITED", "Rate limit.")):
            result = verify_repository("owner/repo", {})
        self.assertFalse(result["passed"])
        self.assertEqual(result["error_kind"], "RATE_LIMITED")

    def test_issue_auth_error_propagated(self):
        with _patch_fetch_issue(repo_error=("AUTHENTICATION_ERROR", "Bad creds.")):
            result = verify_issue("owner/repo", {}, issue_number=1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["error_kind"], "AUTHENTICATION_ERROR")

    def test_infra_fail_never_passes(self):
        r = _infra_fail("issue", 1, "something broke", "INFRASTRUCTURE_ERROR")
        self.assertFalse(r["passed"])
        self.assertEqual(r["error_kind"], "INFRASTRUCTURE_ERROR")


# ---------------------------------------------------------------------------
# format_result
# ---------------------------------------------------------------------------


class TestFormatResult(unittest.TestCase):
    def _pass_result(self):
        return {
            "passed": True,
            "resource": "issue",
            "identifier": 1,
            "checks": {"title": True, "state": True},
            "expected_state": {"title": "Bug", "state": "open"},
            "actual_state": {"title": "Bug", "state": "open"},
            "errors": [],
        }

    def _fail_result(self):
        return {
            "passed": False,
            "resource": "issue",
            "identifier": 1,
            "checks": {"title": False},
            "expected_state": {"title": "Login bug"},
            "actual_state": {"title": "Authentication bug"},
            "errors": ["Field 'title': expected 'Login bug', found 'Authentication bug'."],
            "error_kind": "VERIFICATION_FAILURE",
        }

    def test_pass_output_contains_pass(self):
        out = format_result(self._pass_result())
        self.assertIn("PASS", out)

    def test_fail_output_contains_fail(self):
        out = format_result(self._fail_result())
        self.assertIn("FAIL", out)

    def test_fail_output_contains_error_message(self):
        out = format_result(self._fail_result())
        self.assertIn("Login bug", out)

    def test_json_output_is_valid_json(self):
        import json
        out = format_result(self._pass_result(), json_output=True)
        parsed = json.loads(out)
        self.assertTrue(parsed["passed"])

    def test_json_fail_output_is_valid_json(self):
        import json
        out = format_result(self._fail_result(), json_output=True)
        parsed = json.loads(out)
        self.assertFalse(parsed["passed"])


# ---------------------------------------------------------------------------
# Label semantics
# ---------------------------------------------------------------------------


class TestLabelSemantics(unittest.TestCase):
    """Verify non-strict and strict label comparison in isolation."""

    def _run(self, expected_labels, actual_labels, strict=False):
        mock_issue = _make_issue_mock(labels=actual_labels)
        with _patch_fetch_issue(issue_mock=mock_issue):
            return verify_issue(
                "owner/repo",
                expected_state={"labels": expected_labels},
                issue_number=1,
                strict_labels=strict,
            )

    def test_exact_match_non_strict_passes(self):
        self.assertTrue(self._run(["bug"], ["bug"])["passed"])

    def test_subset_non_strict_passes(self):
        self.assertTrue(self._run(["bug"], ["bug", "backend"])["passed"])

    def test_missing_label_non_strict_fails(self):
        self.assertFalse(self._run(["bug"], [])["passed"])

    def test_exact_match_strict_passes(self):
        self.assertTrue(self._run(["bug"], ["bug"], strict=True)["passed"])

    def test_extra_label_strict_fails(self):
        self.assertFalse(self._run(["bug"], ["bug", "extra"], strict=True)["passed"])

    def test_missing_label_strict_fails(self):
        self.assertFalse(self._run(["bug", "feature"], ["bug"], strict=True)["passed"])

    def test_empty_expected_non_strict_passes_any_labels(self):
        self.assertTrue(self._run([], ["bug", "backend"])["passed"])

    def test_empty_expected_strict_fails_with_actual_labels(self):
        self.assertFalse(self._run([], ["bug"], strict=True)["passed"])


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration(unittest.TestCase):
    """
    End-to-end tests that hit the real GitHub API.

    Requires:
        GITHUB_TOKEN=<pat-with-repo-scope>
        GITHUB_OWNER=<your-github-username-or-org>

    These tests use github_sandbox.py to set up fixtures and then call
    verifier functions directly — verifier never touches sandbox internals.

    Run:
        pytest tests/test_verifier.py -v -m integration
    """

    @classmethod
    def setUpClass(cls):
        # Import sandbox only for integration test setup
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
        from github_sandbox import GitHubSandbox

        cls.sandbox = GitHubSandbox()
        cls.owner = cls.sandbox._owner
        # Reset gives us two clean, seeded repos
        cls.repos = cls.sandbox.reset_sandbox()
        cls.repo_full = f"{cls.owner}/{cls.repos[0].name}"
        _reset_client()  # ensure verifier uses its own fresh client

    @classmethod
    def tearDownClass(cls):
        cls.sandbox.cleanup()

    # Test 1 – verify_repository_exists on a known sandbox repo
    def test_repo_exists(self):
        result = verify_repository_exists(self.repo_full)
        self.assertTrue(result["passed"], msg=str(result["errors"]))

    # Test 2 – verify_repository field checks
    def test_repo_state_private_and_branch(self):
        result = verify_repository(
            self.repo_full,
            expected_state={
                "name": self.repos[0].name,
                "private": True,
                "default_branch": "main",
            },
        )
        self.assertTrue(result["passed"], msg=str(result["errors"]))

    # Test 3 – verify_repository_absent for a nonexistent repo
    def test_repo_absent(self):
        fake = f"{self.owner}/definitely-does-not-exist-xyz-999"
        result = verify_repository_absent(fake)
        self.assertTrue(result["passed"], msg=str(result["errors"]))

    # Test 4 – incorrect expected state → FAIL
    def test_repo_wrong_private_fails(self):
        # Sandbox repos are private; expect public → should fail
        result = verify_repository(
            self.repo_full,
            expected_state={"private": False},
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["private"])

    # Test 5 – create issue via sandbox, verify via verifier (end-to-end)
    def test_issue_create_and_verify(self):
        repo_obj = self.sandbox.get_repo(self.repos[0].name)
        op_result = self.sandbox.execute(
            "issues/create",
            lambda: repo_obj.create_issue(
                title="Integration test issue",
                body="Created by test_verifier.py integration suite",
            ),
        )
        self.assertTrue(op_result["success"], msg=op_result)
        issue_number = op_result["result"].number

        result = verify_issue(
            self.repo_full,
            expected_state={
                "title": "Integration test issue",
                "body": "Created by test_verifier.py integration suite",
                "state": "open",
            },
            issue_number=issue_number,
        )
        self.assertTrue(result["passed"], msg=str(result["errors"]))

    # Test 6 – title mismatch → FAIL
    def test_issue_wrong_title_fails(self):
        repo_obj = self.sandbox.get_repo(self.repos[0].name)
        op_result = self.sandbox.execute(
            "issues/create",
            lambda: repo_obj.create_issue(title="Authentication bug", body=""),
        )
        self.assertTrue(op_result["success"])
        issue_number = op_result["result"].number

        result = verify_issue(
            self.repo_full,
            expected_state={"title": "Login bug"},
            issue_number=issue_number,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["title"])

    # Test 7 – missing label → FAIL
    def test_issue_missing_label_fails(self):
        repo_obj = self.sandbox.get_repo(self.repos[0].name)
        op_result = self.sandbox.execute(
            "issues/create",
            lambda: repo_obj.create_issue(title="No label issue", body=""),
        )
        self.assertTrue(op_result["success"])
        issue_number = op_result["result"].number

        result = verify_issue(
            self.repo_full,
            expected_state={"labels": ["bug"]},
            issue_number=issue_number,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["labels"])

    # Test 8 – verify_issue_absent (GitHub doesn't delete issues, use a large number)
    def test_issue_absent_for_nonexistent_number(self):
        result = verify_issue_absent(self.repo_full, 99999)
        self.assertTrue(result["passed"], msg=str(result["errors"]))
