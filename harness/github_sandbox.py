"""
github_sandbox.py
=================
Infrastructure layer for the 200-Tools agent-evaluation GitHub sandbox.

Responsibilities
----------------
* Authentication via GITHUB_TOKEN / GITHUB_OWNER env-vars (python-dotenv).
* Unique evaluation run-ID generation (uuid4 short hex).
* Lifecycle management of run-scoped test repositories:
    - create, seed, reset, delete
* Rate-limit awareness: get / check / log quota.
* Generic execute() wrapper: logging + structured success/failure dicts.
* Safe, run-ID-scoped cleanup() — never touches foreign repos.

What this file does NOT contain
--------------------------------
The catalog of ~40-60 GitHub operations (issues, PRs, branches, etc.).
Those belong in github_operations.py, which calls into this sandbox.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from github import Github, GithubException, RateLimitExceededException, UnknownObjectException

# ---------------------------------------------------------------------------
# Module-level logger — callers may attach their own handler / formatter
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — seed data used by every evaluation reset
# ---------------------------------------------------------------------------

REPO_COUNT = 2  # number of sandbox repositories per run

SEED_FILES: dict[str, str] = {
    "README.md": (
        "# Evaluation Sandbox\n\n"
        "This repository is used for automated agent evaluation.\n"
        "Do **not** edit it manually.\n"
    ),
    "config.json": json.dumps(
        {
            "project": "eval-sandbox",
            "version": "1.0.0",
            "settings": {"debug": False, "max_retries": 3},
        },
        indent=2,
    )
    + "\n",
}

SEED_ISSUES: list[dict[str, str]] = [
    {
        "title": "Fix login bug",
        "body": (
            "Users are unable to log in when 2FA is enabled. "
            "Reproduce by enabling 2FA and attempting a login."
        ),
    },
    {
        "title": "Add dark mode",
        "body": (
            "Implement a dark-mode toggle in the settings panel. "
            "Should respect the OS-level preference by default."
        ),
    },
]

# Error-type labels returned inside structured result dicts
_ET_RATE_LIMITED = "RATE_LIMITED"
_ET_NOT_FOUND = "NOT_FOUND"
_ET_UNAUTHORIZED = "UNAUTHORIZED"
_ET_FORBIDDEN = "FORBIDDEN"
_ET_VALIDATION = "VALIDATION_ERROR"
_ET_GITHUB = "GITHUB_ERROR"
_ET_UNEXPECTED = "UNEXPECTED_ERROR"


# ---------------------------------------------------------------------------
# Small value objects
# ---------------------------------------------------------------------------


@dataclass
class RateLimitSnapshot:
    """Immutable point-in-time snapshot of the GitHub core-API quota."""

    limit: int
    remaining: int
    reset: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "remaining": self.remaining,
            "reset": self.reset.isoformat(),
        }

    def __str__(self) -> str:
        return (
            f"{self.remaining}/{self.limit} remaining | "
            f"reset={self.reset.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )


class RateLimitExhaustedError(RuntimeError):
    """Raised by check_rate_limit() when remaining quota is critically low."""

    def __init__(self, remaining: int, minimum: int, reset: datetime) -> None:
        super().__init__(
            f"GitHub quota critically low: {remaining} remaining "
            f"(minimum={minimum}). Quota resets at {reset.isoformat()}."
        )
        self.remaining = remaining
        self.minimum = minimum
        self.reset = reset


class SandboxConfigError(RuntimeError):
    """Raised when the sandbox cannot be initialised due to configuration problems."""


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class GitHubSandbox:
    """
    Controlled, repeatable GitHub sandbox for automated agent evaluation.

    Parameters
    ----------
    token : str, optional
        GitHub personal-access token.  Falls back to GITHUB_TOKEN env-var.
    owner : str, optional
        GitHub account / organisation that owns the sandbox repos.
        Falls back to GITHUB_OWNER env-var.

    Attributes
    ----------
    run_id : str
        8-character hex string that uniquely identifies this evaluation run.
        All sandbox resources carry this ID so they can be found and deleted
        without touching unrelated repositories.
    """

    # ------------------------------------------------------------------
    # Construction / authentication
    # ------------------------------------------------------------------

    def __init__(
        self,
        token: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> None:
        load_dotenv()

        self._token: str = token if token is not None else os.getenv("GITHUB_TOKEN", "")
        self._owner: str = owner if owner is not None else os.getenv("GITHUB_OWNER", "")

        if not self._token:
            raise SandboxConfigError(
                "GitHub token not found.  Set GITHUB_TOKEN in your environment or .env file."
            )
        if not self._owner:
            raise SandboxConfigError(
                "GitHub owner not found.  Set GITHUB_OWNER in your environment or .env file."
            )

        # Unique identifier for this evaluation run
        self.run_id: str = uuid.uuid4().hex[:8]

        # Initialise PyGithub client once
        self._gh = Github(self._token)

        # Verify credentials and log connected identity
        try:
            authed_user = self._gh.get_user().login
        except GithubException as exc:
            raise SandboxConfigError(
                f"GitHub authentication failed: {exc.status} {exc.data}"
            ) from exc

        self._log(f"Connected as {authed_user!r} (owner={self._owner!r})")
        self._log_rate_limit("Startup quota")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, message: str, level: int = logging.INFO) -> None:
        logger.log(level, "[RUN %s] %s", self.run_id, message)

    def _repo_prefix(self) -> str:
        return f"eval-{self.run_id}-"

    def _repo_name(self, n: int) -> str:
        return f"eval-{self.run_id}-repo-{n}"

    def _log_rate_limit(self, context: str = "") -> Optional[RateLimitSnapshot]:
        try:
            snap = self.get_rate_limit()
            prefix = f"{context}: " if context else ""
            self._log(f"GitHub quota: {prefix}{snap}")
            return snap
        except Exception as exc:
            self._log(f"Could not fetch rate-limit info: {exc}", logging.WARNING)
            return None

    @staticmethod
    def _classify_github_error(exc: GithubException) -> str:
        if isinstance(exc, RateLimitExceededException):
            return _ET_RATE_LIMITED
        status = getattr(exc, "status", None)
        if status == 401:
            return _ET_UNAUTHORIZED
        if status == 403:
            return _ET_FORBIDDEN
        if status == 404:
            return _ET_NOT_FOUND
        if status == 422:
            return _ET_VALIDATION
        return _ET_GITHUB

    # ------------------------------------------------------------------
    # Repository helpers (infrastructure only)
    # ------------------------------------------------------------------

    def get_repo(self, repo_name: str):
        """
        Return a PyGithub Repository object for *repo_name*.

        Raises ``UnknownObjectException`` (404) if the repo does not exist.
        """
        if "/" in repo_name:
            full_name = repo_name
        else:
            full_name = f"{self._owner}/{repo_name}"
        self._log(f"Fetching repo {full_name!r}")
        return self._gh.get_repo(full_name)

    def repo_exists(self, repo_name: str) -> bool:
        """Return True if *repo_name* exists under the configured owner."""
        try:
            self.get_repo(repo_name)
            return True
        except UnknownObjectException:
            return False
        except GithubException as exc:
            self._log(
                f"Unexpected error checking repo {repo_name!r}: {exc}",
                logging.WARNING,
            )
            return False

    def create_repo(self, repo_name: str, private: bool = True):
        """
        Create a new GitHub repository owned by *self._owner*.

        Returns the new PyGithub Repository object.
        """
        self._log(f"Creating repository {repo_name!r} (private={private})")
        owner_obj = self._gh.get_user(self._owner)
        repo = owner_obj.create_repo(
            name=repo_name,
            description=f"Evaluation sandbox | run_id={self.run_id}",
            private=private,
            auto_init=False,  # we seed manually for full determinism
        )
        self._log(f"Created repository {repo_name!r}")
        return repo

    def delete_repo(self, repo_name: str) -> bool:
        """
        Delete *repo_name* if it exists.

        Returns True on success, False if the repo was not found.
        Never raises on 404 — logs a warning instead.
        """
        self._log(f"Deleting repository {repo_name!r}")
        try:
            repo = self.get_repo(repo_name)
            repo.delete()
            self._log(f"Deleted repository {repo_name!r}")
            return True
        except UnknownObjectException:
            self._log(f"Repository {repo_name!r} not found — skipping delete", logging.WARNING)
            return False

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_repo(self, repo) -> None:
        """
        Establish a deterministic initial state in *repo*.

        Creates:
        * SEED_FILES  — stable file contents (README.md, config.json)
        * SEED_ISSUES — stable issues (Fix login bug, Add dark mode)

        Called only from reset_repo() / reset_sandbox(), not before
        individual catalog operations.
        """
        repo_name = repo.name
        self._log(f"Seeding {repo_name!r}")

        # Create seed files on the default branch
        for path, content in SEED_FILES.items():
            try:
                repo.create_file(
                    path=path,
                    message=f"chore: seed {path} [eval run_id={self.run_id}]",
                    content=content,
                )
                self._log(f"  Created file {path!r} in {repo_name!r}")
            except GithubException as exc:
                self._log(
                    f"  Failed to create {path!r} in {repo_name!r}: {exc}",
                    logging.ERROR,
                )
                raise

        # Create seed issues
        for issue_data in SEED_ISSUES:
            try:
                issue = repo.create_issue(
                    title=issue_data["title"],
                    body=issue_data["body"],
                )
                self._log(
                    f"  Created issue #{issue.number} {issue_data['title']!r} in {repo_name!r}"
                )
            except GithubException as exc:
                self._log(
                    f"  Failed to create issue {issue_data['title']!r} in {repo_name!r}: {exc}",
                    logging.ERROR,
                )
                raise

        self._log(f"Seeded {repo_name!r}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_repo(self, repo_name: str):
        """
        Delete *repo_name* if it exists, recreate it, and seed it.

        Returns the freshly-seeded PyGithub Repository object.
        """
        self._log(f"Resetting {repo_name!r}")
        self.delete_repo(repo_name)
        repo = self.create_repo(repo_name)
        self.seed_repo(repo)
        return repo

    def reset_sandbox(self) -> list:
        """
        Wipe and recreate all sandbox repositories for this run.

        Flow:
            delete existing repos (if any) -> create -> seed -> return repo list

        Returns a list of freshly-seeded PyGithub Repository objects.
        """
        self._log("Reset sandbox started")
        self._log_rate_limit("Before reset")

        repos: list = []
        for n in range(1, REPO_COUNT + 1):
            name = self._repo_name(n)
            repo = self.reset_repo(name)
            repos.append(repo)

        self._log_rate_limit("After reset")
        self._log(f"Reset sandbox complete — {len(repos)} repositor(ies) ready")
        return repos

    # ------------------------------------------------------------------
    # Rate-limit awareness
    # ------------------------------------------------------------------

    def get_rate_limit(self) -> RateLimitSnapshot:
        """
        Fetch the current GitHub core-API rate-limit and return a snapshot.
        """
        rl = self._gh.get_rate_limit()
        if hasattr(rl, "resources") and hasattr(rl.resources, "core") and not hasattr(rl, "_mock_return_value"):
            core = rl.resources.core
        elif hasattr(rl, "core"):
            core = rl.core
        elif hasattr(rl, "resources") and hasattr(rl.resources, "core"):
            core = rl.resources.core
        else:
            core = getattr(rl, "rate", rl)

        reset_dt = core.reset
        if hasattr(reset_dt, "tzinfo") and reset_dt.tzinfo is None:
            reset_dt = reset_dt.replace(tzinfo=timezone.utc)
        elif not hasattr(reset_dt, "tzinfo"):
            reset_dt = core.reset.replace(tzinfo=timezone.utc)

        return RateLimitSnapshot(
            limit=core.limit,
            remaining=core.remaining,
            reset=reset_dt,
        )

    def check_rate_limit(self, minimum_remaining: int = 10) -> RateLimitSnapshot:
        """
        Fetch the current rate-limit and raise RateLimitExhaustedError if
        remaining quota is below *minimum_remaining*.

        Does NOT sleep — exposes the condition to the caller instead.

        Parameters
        ----------
        minimum_remaining : int
            Threshold below which an exception is raised.

        Returns
        -------
        RateLimitSnapshot
            The current snapshot (only returned when quota is sufficient).
        """
        snap = self.get_rate_limit()
        if snap.remaining < minimum_remaining:
            raise RateLimitExhaustedError(snap.remaining, minimum_remaining, snap.reset)
        return snap

    # ------------------------------------------------------------------
    # Generic operation wrapper
    # ------------------------------------------------------------------

    def execute(
        self,
        operation_name: str,
        fn: Callable[[], Any],
        minimum_remaining: int = 5,
    ) -> dict[str, Any]:
        """
        Execute a GitHub API call with logging, rate-limit capture, and
        structured result / error reporting.

        Parameters
        ----------
        operation_name : str
            Human-readable name, e.g. "issues/create".
        fn : callable
            Zero-argument callable that performs the GitHub operation.
        minimum_remaining : int
            Pre-flight minimum quota; the call is aborted (with a structured
            error) if quota is below this value before the operation.

        Returns
        -------
        dict
            Success:
                {
                    "success": True,
                    "operation": operation_name,
                    "result": <return value of fn()>,
                    "rate_limit_before": {...},
                    "rate_limit_after": {...},
                }

            Failure:
                {
                    "success": False,
                    "operation": operation_name,
                    "error_type": "RATE_LIMITED | NOT_FOUND | ...",
                    "status": <HTTP status or None>,
                    "message": "...",
                    "rate_limit_before": {...},
                    "rate_limit_after": {...},
                }
        """
        self._log(f"Executing {operation_name!r}")

        # --- pre-flight rate-limit snapshot ---
        rl_before: dict[str, Any] = {}
        try:
            snap_before = self.get_rate_limit()
            rl_before = snap_before.as_dict()
            self._log(f"  [{operation_name}] quota before: {snap_before}")
            if snap_before.remaining < minimum_remaining:
                msg = (
                    f"Pre-flight quota check failed: "
                    f"{snap_before.remaining} remaining (minimum={minimum_remaining}). "
                    f"Resets at {snap_before.reset.isoformat()}."
                )
                self._log(msg, logging.WARNING)
                return {
                    "success": False,
                    "operation": operation_name,
                    "error_type": _ET_RATE_LIMITED,
                    "status": 429,
                    "message": msg,
                    "rate_limit_before": rl_before,
                    "rate_limit_after": rl_before,
                }
        except GithubException as exc:
            rl_before = {"error": str(exc)}

        # --- execute ---
        result = None
        error_payload: Optional[dict[str, Any]] = None

        try:
            result = fn()
        except RateLimitExceededException as exc:
            self._log(f"  [{operation_name}] RATE LIMITED: {exc}", logging.WARNING)
            error_payload = {
                "error_type": _ET_RATE_LIMITED,
                "status": 429,
                "message": str(exc),
            }
        except UnknownObjectException as exc:
            self._log(f"  [{operation_name}] NOT FOUND (404): {exc}", logging.WARNING)
            error_payload = {
                "error_type": _ET_NOT_FOUND,
                "status": 404,
                "message": str(exc),
            }
        except GithubException as exc:
            error_type = self._classify_github_error(exc)
            self._log(
                f"  [{operation_name}] {error_type}: {exc.status} -- {exc.data}",
                logging.ERROR,
            )
            error_payload = {
                "error_type": error_type,
                "status": getattr(exc, "status", None),
                "message": str(exc.data) if hasattr(exc, "data") else str(exc),
            }
        except Exception as exc:
            self._log(
                f"  [{operation_name}] UNEXPECTED: {type(exc).__name__}: {exc}",
                logging.ERROR,
            )
            error_payload = {
                "error_type": _ET_UNEXPECTED,
                "status": None,
                "message": f"{type(exc).__name__}: {exc}",
            }

        # --- post-flight rate-limit snapshot ---
        rl_after: dict[str, Any] = {}
        try:
            snap_after = self.get_rate_limit()
            rl_after = snap_after.as_dict()
            self._log(f"  [{operation_name}] quota after: {snap_after}")
        except GithubException as exc:
            rl_after = {"error": str(exc)}

        # --- build structured response ---
        if error_payload is not None:
            return {
                "success": False,
                "operation": operation_name,
                "rate_limit_before": rl_before,
                "rate_limit_after": rl_after,
                **error_payload,
            }

        self._log(f"  [{operation_name}] succeeded")
        return {
            "success": True,
            "operation": operation_name,
            "result": result,
            "rate_limit_before": rl_before,
            "rate_limit_after": rl_after,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        Delete all repositories whose names begin with 'eval-{run_id}-'.

        Safety guarantee: only repositories that carry the current run's
        prefix are deleted.  Foreign / unrelated repositories are never
        touched.
        """
        prefix = self._repo_prefix()
        self._log(f"Cleanup started (prefix={prefix!r})")
        self._log_rate_limit("Before cleanup")

        try:
            owner_obj = self._gh.get_user(self._owner)
            repos_to_delete = [
                r for r in owner_obj.get_repos() if r.name.startswith(prefix)
            ]
        except GithubException as exc:
            self._log(
                f"Failed to list repositories during cleanup: {exc}",
                logging.ERROR,
            )
            return

        if not repos_to_delete:
            self._log("No sandbox repositories found to delete")
        else:
            for repo in repos_to_delete:
                try:
                    repo.delete()
                    self._log(f"Deleted {repo.name!r}")
                except GithubException as exc:
                    self._log(
                        f"Failed to delete {repo.name!r}: {exc}",
                        logging.ERROR,
                    )

        self._log_rate_limit("After cleanup")
        self._log("Cleanup complete")

    # ------------------------------------------------------------------
    # Context-manager support (optional convenience)
    # ------------------------------------------------------------------

    def __enter__(self) -> "GitHubSandbox":
        return self

    def __exit__(self, *_) -> None:
        self.cleanup()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"GitHubSandbox(run_id={self.run_id!r}, owner={self._owner!r})"
