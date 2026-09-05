"""
tests/test_github_sandbox.py
============================
Unit tests for github_sandbox.py that do NOT require live GitHub access.

Integration tests (marked with @pytest.mark.integration) need:
    GITHUB_TOKEN and GITHUB_OWNER set in the environment.
    They perform real network calls and are excluded from the default run.

Run unit tests only:
    pytest tests/test_github_sandbox.py -v

Run including integration tests (requires valid credentials):
    pytest tests/test_github_sandbox.py -v -m integration
"""

import os
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

try:
    import pytest  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # Minimal shim so this file is importable and runnable with plain unittest.
    import types

    pytest = types.ModuleType("pytest")

    def _raises(exc, match=None):
        """Poor-man's pytest.raises as a context manager."""
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
                    return False  # re-raise
                self.value = val
                if match and not re.search(match, str(val)):
                    raise AssertionError(
                        f"Exception message {str(val)!r} did not match {match!r}"
                    )
                return True  # suppress

        return _Ctx()

    pytest.raises = _raises

    class _mark:
        @staticmethod
        def integration(fn_or_cls):
            """Apply unittest.skip to a test function or a whole test class."""
            msg = "integration test — set GITHUB_TOKEN/GITHUB_OWNER and use pytest -m integration"
            skip_deco = unittest.skip(msg)
            if isinstance(fn_or_cls, type):
                # Decorate each test method on the class
                for attr in list(vars(fn_or_cls)):
                    if attr.startswith("test"):
                        setattr(fn_or_cls, attr, skip_deco(getattr(fn_or_cls, attr)))
                return fn_or_cls
            return skip_deco(fn_or_cls)

    pytest.mark = _mark

# Make sure `harness/` is on the path when tests run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))

from github_sandbox import (
    GitHubSandbox,
    RateLimitExhaustedError,
    RateLimitSnapshot,
    SandboxConfigError,
    SEED_FILES,
    SEED_ISSUES,
    REPO_COUNT,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_rl(limit=5000, remaining=4987, reset_dt=None):
    """Build a mock PyGithub rate-limit object."""
    if reset_dt is None:
        reset_dt = datetime(2026, 9, 5, 1, 0, 0)  # naive UTC
    rl = SimpleNamespace(limit=limit, remaining=remaining, reset=reset_dt)
    return rl


def _mock_gh(token="tok", owner="myowner"):
    """
    Return a (patched) GitHubSandbox instance whose PyGithub client is
    entirely mocked — no network calls will be made.
    """
    mock_gh_instance = MagicMock()
    # get_user().login — called during __init__ for auth verification
    mock_gh_instance.get_user.return_value.login = "test-user"
    # get_rate_limit().core — called during _log_rate_limit
    mock_gh_instance.get_rate_limit.return_value.core = _make_rl()

    with patch("github_sandbox.Github", return_value=mock_gh_instance):
        with patch.dict(os.environ, {"GITHUB_TOKEN": token, "GITHUB_OWNER": owner}):
            sb = GitHubSandbox()

    sb._gh = mock_gh_instance  # keep reference for assertions
    return sb


# ---------------------------------------------------------------------------
# 1. Missing-token handling
# ---------------------------------------------------------------------------


class TestMissingCredentials:
    def test_missing_token_raises(self):
        """SandboxConfigError is raised when GITHUB_TOKEN is absent."""
        env = {"GITHUB_OWNER": "some-owner"}
        # Remove GITHUB_TOKEN from env entirely
        env_without_token = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
        env_without_token["GITHUB_OWNER"] = "some-owner"

        with patch.dict(os.environ, env_without_token, clear=True):
            with pytest.raises(SandboxConfigError, match="GITHUB_TOKEN"):
                GitHubSandbox(token="", owner="some-owner")

    def test_missing_owner_raises(self):
        """SandboxConfigError is raised when GITHUB_OWNER is absent."""
        env_without_owner = {k: v for k, v in os.environ.items() if k != "GITHUB_OWNER"}

        mock_gh_instance = MagicMock()
        mock_gh_instance.get_user.return_value.login = "test-user"
        mock_gh_instance.get_rate_limit.return_value.core = _make_rl()

        with patch("github_sandbox.Github", return_value=mock_gh_instance):
            with patch.dict(os.environ, env_without_owner, clear=True):
                with pytest.raises(SandboxConfigError, match="GITHUB_OWNER"):
                    GitHubSandbox(token="valid-token", owner="")

    def test_explicit_token_and_owner_bypass_env(self):
        """Explicit constructor args are preferred over env vars."""
        mock_gh_instance = MagicMock()
        mock_gh_instance.get_user.return_value.login = "test-user"
        mock_gh_instance.get_rate_limit.return_value.core = _make_rl()

        with patch("github_sandbox.Github", return_value=mock_gh_instance):
            # No env vars set, but explicit args provided
            sb = GitHubSandbox(token="mytoken", owner="myowner")
        assert sb._token == "mytoken"
        assert sb._owner == "myowner"


# ---------------------------------------------------------------------------
# 2. Run ID generation
# ---------------------------------------------------------------------------


class TestRunId:
    def test_run_id_is_8_hex_chars(self):
        sb = _mock_gh()
        assert len(sb.run_id) == 8
        assert all(c in "0123456789abcdef" for c in sb.run_id)

    def test_run_ids_are_unique(self):
        sb1 = _mock_gh()
        sb2 = _mock_gh()
        assert sb1.run_id != sb2.run_id

    def test_run_id_is_exposed(self):
        sb = _mock_gh()
        assert hasattr(sb, "run_id")
        assert isinstance(sb.run_id, str)


# ---------------------------------------------------------------------------
# 3. Repository naming
# ---------------------------------------------------------------------------


class TestRepoNaming:
    def test_repo_name_format(self):
        sb = _mock_gh()
        assert sb._repo_name(1) == f"eval-{sb.run_id}-repo-1"
        assert sb._repo_name(2) == f"eval-{sb.run_id}-repo-2"

    def test_repo_prefix_format(self):
        sb = _mock_gh()
        assert sb._repo_prefix() == f"eval-{sb.run_id}-"

    def test_repo_names_contain_run_id(self):
        sb = _mock_gh()
        for n in range(1, REPO_COUNT + 1):
            assert sb.run_id in sb._repo_name(n)

    def test_repo_count_constant(self):
        assert REPO_COUNT == 2


# ---------------------------------------------------------------------------
# 4. Run-ID prefix matching (used by cleanup)
# ---------------------------------------------------------------------------


class TestPrefixMatching:
    def test_prefix_matches_own_repos(self):
        sb = _mock_gh()
        prefix = sb._repo_prefix()
        for n in range(1, REPO_COUNT + 1):
            assert sb._repo_name(n).startswith(prefix)

    def test_prefix_does_not_match_foreign_repos(self):
        sb = _mock_gh()
        prefix = sb._repo_prefix()
        assert not "my-production-repo".startswith(prefix)
        assert not "eval-deadbeef-repo-1".startswith(prefix)  # different run
        assert not "".startswith(prefix)

    def test_cleanup_only_deletes_prefixed_repos(self):
        sb = _mock_gh()
        prefix = sb._repo_prefix()

        own_repo = MagicMock()
        own_repo.name = sb._repo_name(1)
        foreign_repo = MagicMock()
        foreign_repo.name = "totally-unrelated-repo"

        sb._gh.get_user.return_value.get_repos.return_value = [own_repo, foreign_repo]

        sb.cleanup()

        own_repo.delete.assert_called_once()
        foreign_repo.delete.assert_not_called()


# ---------------------------------------------------------------------------
# 5. RateLimitSnapshot
# ---------------------------------------------------------------------------


class TestRateLimitSnapshot:
    def _make_snap(self, limit=5000, remaining=4987):
        reset = datetime(2026, 9, 5, 2, 0, 0, tzinfo=timezone.utc)
        return RateLimitSnapshot(limit=limit, remaining=remaining, reset=reset)

    def test_as_dict_keys(self):
        snap = self._make_snap()
        d = snap.as_dict()
        assert set(d.keys()) == {"limit", "remaining", "reset"}

    def test_as_dict_values(self):
        snap = self._make_snap(limit=5000, remaining=4987)
        d = snap.as_dict()
        assert d["limit"] == 5000
        assert d["remaining"] == 4987
        assert "2026-09-05" in d["reset"]

    def test_str_contains_remaining_and_limit(self):
        snap = self._make_snap(limit=5000, remaining=123)
        s = str(snap)
        assert "123" in s
        assert "5000" in s

    def test_get_rate_limit_returns_snapshot(self):
        sb = _mock_gh()
        sb._gh.get_rate_limit.return_value.core = _make_rl(limit=5000, remaining=999)
        snap = sb.get_rate_limit()
        assert isinstance(snap, RateLimitSnapshot)
        assert snap.limit == 5000
        assert snap.remaining == 999

    def test_get_rate_limit_attaches_utc(self):
        sb = _mock_gh()
        naive_dt = datetime(2026, 9, 5, 3, 0, 0)  # no tzinfo
        sb._gh.get_rate_limit.return_value.core = _make_rl(reset_dt=naive_dt)
        snap = sb.get_rate_limit()
        assert snap.reset.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# 6. check_rate_limit
# ---------------------------------------------------------------------------


class TestCheckRateLimit:
    def test_raises_when_below_minimum(self):
        sb = _mock_gh()
        sb._gh.get_rate_limit.return_value.core = _make_rl(remaining=3)
        with pytest.raises(RateLimitExhaustedError) as exc_info:
            sb.check_rate_limit(minimum_remaining=10)
        assert exc_info.value.remaining == 3
        assert exc_info.value.minimum == 10

    def test_returns_snapshot_when_above_minimum(self):
        sb = _mock_gh()
        sb._gh.get_rate_limit.return_value.core = _make_rl(remaining=500)
        snap = sb.check_rate_limit(minimum_remaining=10)
        assert isinstance(snap, RateLimitSnapshot)
        assert snap.remaining == 500

    def test_exact_minimum_passes(self):
        sb = _mock_gh()
        sb._gh.get_rate_limit.return_value.core = _make_rl(remaining=10)
        # remaining == minimum_remaining should NOT raise
        snap = sb.check_rate_limit(minimum_remaining=10)
        assert snap.remaining == 10

    def test_one_below_minimum_raises(self):
        sb = _mock_gh()
        sb._gh.get_rate_limit.return_value.core = _make_rl(remaining=9)
        with pytest.raises(RateLimitExhaustedError):
            sb.check_rate_limit(minimum_remaining=10)


# ---------------------------------------------------------------------------
# 7. execute() — structured error handling
# ---------------------------------------------------------------------------


class TestExecuteWrapper:
    def _make_sb(self):
        sb = _mock_gh()
        sb._gh.get_rate_limit.return_value.core = _make_rl(remaining=500)
        return sb

    def test_success_result_shape(self):
        sb = self._make_sb()
        result = sb.execute("repos/get", lambda: "payload")
        assert result["success"] is True
        assert result["operation"] == "repos/get"
        assert result["result"] == "payload"
        assert "rate_limit_before" in result
        assert "rate_limit_after" in result

    def test_failure_result_shape_on_exception(self):
        from github import GithubException

        sb = self._make_sb()

        def boom():
            raise GithubException(status=404, data={"message": "Not Found"})

        result = sb.execute("repos/get", boom)
        assert result["success"] is False
        assert result["operation"] == "repos/get"
        assert "error_type" in result
        assert "status" in result
        assert "message" in result
        assert "rate_limit_before" in result
        assert "rate_limit_after" in result

    def test_404_classified_as_not_found(self):
        from github import GithubException

        sb = self._make_sb()

        def boom():
            raise GithubException(status=404, data={"message": "Not Found"})

        result = sb.execute("issues/get", boom)
        assert result["error_type"] == "NOT_FOUND"
        assert result["status"] == 404

    def test_401_classified_as_unauthorized(self):
        from github import GithubException

        sb = self._make_sb()

        def boom():
            raise GithubException(status=401, data={"message": "Bad credentials"})

        result = sb.execute("repos/list", boom)
        assert result["error_type"] == "UNAUTHORIZED"

    def test_403_classified_as_forbidden(self):
        from github import GithubException

        sb = self._make_sb()

        def boom():
            raise GithubException(status=403, data={"message": "Forbidden"})

        result = sb.execute("repos/delete", boom)
        assert result["error_type"] == "FORBIDDEN"

    def test_422_classified_as_validation_error(self):
        from github import GithubException

        sb = self._make_sb()

        def boom():
            raise GithubException(status=422, data={"message": "Validation Failed"})

        result = sb.execute("issues/create", boom)
        assert result["error_type"] == "VALIDATION_ERROR"

    def test_rate_limited_pre_flight(self):
        sb = _mock_gh()
        sb._gh.get_rate_limit.return_value.core = _make_rl(remaining=2)
        called = []
        result = sb.execute("issues/create", lambda: called.append(1), minimum_remaining=5)
        assert result["success"] is False
        assert result["error_type"] == "RATE_LIMITED"
        assert result["status"] == 429
        assert called == []  # fn must NOT have been called

    def test_unexpected_exception_handled(self):
        sb = self._make_sb()

        def boom():
            raise ValueError("Something unexpected")

        result = sb.execute("repos/create", boom)
        assert result["success"] is False
        assert result["error_type"] == "UNEXPECTED_ERROR"
        assert "ValueError" in result["message"]

    def test_rate_limit_captured_around_call(self):
        from github import GithubException

        sb = _mock_gh()
        call_count = [0]

        # Return decreasing remaining to simulate consumption
        def side_effect():
            call_count[0] += 1
            remaining = 500 - call_count[0]
            ns = SimpleNamespace(core=_make_rl(remaining=remaining))
            return ns

        sb._gh.get_rate_limit.side_effect = side_effect

        result = sb.execute("test/op", lambda: "ok")
        assert result["success"] is True
        # Before and after should be different (remaining decremented)
        assert result["rate_limit_before"]["remaining"] != result["rate_limit_after"]["remaining"]


# ---------------------------------------------------------------------------
# 8. Seed data constants
# ---------------------------------------------------------------------------


class TestSeedData:
    def test_seed_files_present(self):
        assert "README.md" in SEED_FILES
        assert "config.json" in SEED_FILES

    def test_seed_files_non_empty(self):
        for path, content in SEED_FILES.items():
            assert content.strip(), f"{path} seed content should not be empty"

    def test_config_json_is_valid_json(self):
        import json

        parsed = json.loads(SEED_FILES["config.json"])
        assert isinstance(parsed, dict)

    def test_seed_issues_present(self):
        titles = [i["title"] for i in SEED_ISSUES]
        assert "Fix login bug" in titles
        assert "Add dark mode" in titles

    def test_seed_issues_have_body(self):
        for issue in SEED_ISSUES:
            assert issue.get("body", "").strip(), "Each seed issue should have a non-empty body"


# ---------------------------------------------------------------------------
# 9. classify_github_error
# ---------------------------------------------------------------------------


class TestClassifyGithubError:
    def _exc(self, status, data=None):
        from github import GithubException

        return GithubException(status=status, data=data or {})

    def test_401(self):
        assert GitHubSandbox._classify_github_error(self._exc(401)) == "UNAUTHORIZED"

    def test_403(self):
        assert GitHubSandbox._classify_github_error(self._exc(403)) == "FORBIDDEN"

    def test_404(self):
        assert GitHubSandbox._classify_github_error(self._exc(404)) == "NOT_FOUND"

    def test_422(self):
        assert GitHubSandbox._classify_github_error(self._exc(422)) == "VALIDATION_ERROR"

    def test_500(self):
        assert GitHubSandbox._classify_github_error(self._exc(500)) == "GITHUB_ERROR"

    def test_rate_limited_exception(self):
        from github import RateLimitExceededException

        exc = RateLimitExceededException(status=403, data={})
        assert GitHubSandbox._classify_github_error(exc) == "RATE_LIMITED"


# ---------------------------------------------------------------------------
# 10. repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_run_id(self):
        sb = _mock_gh()
        r = repr(sb)
        assert sb.run_id in r

    def test_repr_contains_owner(self):
        sb = _mock_gh(owner="myowner")
        r = repr(sb)
        assert "myowner" in r


# ---------------------------------------------------------------------------
# Integration tests (require GITHUB_TOKEN + GITHUB_OWNER in env)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    """
    These tests make real GitHub API calls.

    Prerequisites:
        export GITHUB_TOKEN=<pat-with-repo-scope>
        export GITHUB_OWNER=<your-github-username-or-org>

    Run:
        pytest tests/test_github_sandbox.py -v -m integration
    """

    def test_auth_and_run_id(self):
        sb = GitHubSandbox()
        assert sb.run_id and len(sb.run_id) == 8

    def test_get_rate_limit_live(self):
        sb = GitHubSandbox()
        snap = sb.get_rate_limit()
        assert snap.limit > 0
        assert 0 <= snap.remaining <= snap.limit

    def test_reset_and_cleanup(self):
        sb = GitHubSandbox()
        try:
            repos = sb.reset_sandbox()
            assert len(repos) == REPO_COUNT
            for repo in repos:
                assert sb.run_id in repo.name
                assert repo.private is True
        finally:
            sb.cleanup()

    def test_repo_exists_after_reset(self):
        sb = GitHubSandbox()
        try:
            repos = sb.reset_sandbox()
            for repo in repos:
                assert sb.repo_exists(repo.name) is True
        finally:
            sb.cleanup()

    def test_cleanup_removes_repos(self):
        sb = GitHubSandbox()
        repos = sb.reset_sandbox()
        names = [r.name for r in repos]
        sb.cleanup()
        for name in names:
            assert sb.repo_exists(name) is False

    def test_execute_wrapper_live(self):
        sb = GitHubSandbox()
        try:
            repos = sb.reset_sandbox()
            repo = sb.get_repo(repos[0].name)
            result = sb.execute(
                "issues/create",
                lambda: repo.create_issue(title="Integration test issue", body="auto"),
            )
            assert result["success"] is True
            assert result["result"] is not None
        finally:
            sb.cleanup()
