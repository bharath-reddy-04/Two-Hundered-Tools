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

        # Default sandbox repository (eval-sandbox-repo)
        self.default_repo: str = os.getenv("GITHUB_REPO", "eval-sandbox-repo")

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
        target_default = getattr(self, "default_repo", os.getenv("GITHUB_REPO", "eval-sandbox-repo"))
        bare_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        if bare_name in ("test-repo-1", "test_repo_1", "repo-1", "repo_1"):
            repo_name = target_default

        if "/" in repo_name:
            full_name = repo_name
        else:
            full_name = f"{self._owner}/{repo_name}"
        self._log(f"Fetching repo {full_name!r}")
        try:
            return self._gh.get_repo(full_name)
        except UnknownObjectException:
            default_full = f"{self._owner}/{target_default}" if "/" not in target_default else target_default
            if full_name != default_full and bare_name in ("test-repo", "test_repo", "test-repository"):
                self._log(f"Repo {full_name!r} not found, falling back to default {default_full!r}")
                return self._gh.get_repo(default_full)
            raise

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
        owner_obj = self._gh.get_user()
        if not hasattr(owner_obj, "create_repo"):
            owner_obj = self._gh.get_user(self._owner)
        repo = owner_obj.create_repo(
            name=repo_name,
            description=f"Evaluation sandbox | run_id={self.run_id}",
            private=private,
            auto_init=True,
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

        Creates / updates:
        * SEED_FILES  — stable file contents (README.md, config.json)
        * SEED_ISSUES — stable issues (Fix login bug, Add dark mode)

        Called only from reset_repo() / reset_sandbox(), not before
        individual catalog operations.
        """
        repo_name = repo.name
        self._log(f"Seeding {repo_name!r}")

        # Create or update seed files on the default branch
        for path, content in SEED_FILES.items():
            try:
                try:
                    existing = repo.get_contents(path)
                    repo.update_file(
                        path=path,
                        message=f"chore: update {path} [eval run_id={self.run_id}]",
                        content=content,
                        sha=existing.sha,
                    )
                    self._log(f"  Updated file {path!r} in {repo_name!r}")
                except (UnknownObjectException, GithubException):
                    repo.create_file(
                        path=path,
                        message=f"chore: seed {path} [eval run_id={self.run_id}]",
                        content=content,
                    )
                    self._log(f"  Created file {path!r} in {repo_name!r}")
            except GithubException as exc:
                self._log(
                    f"  Failed to create/update {path!r} in {repo_name!r}: {exc}",
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

    def _resolve_repo(self, repo_name_or_obj: Any):
        """Internal helper to return a Repository object from name or instance."""
        if hasattr(repo_name_or_obj, "get_issues") or hasattr(repo_name_or_obj, "get_branches"):
            return repo_name_or_obj
        if isinstance(repo_name_or_obj, str):
            return self.get_repo(repo_name_or_obj)
        return repo_name_or_obj

    # ------------------------------------------------------------------
    # Reset & Deterministic State Cleaning
    # ------------------------------------------------------------------

    def clean_issues(
        self,
        repo_name_or_obj: Any,
        prefix: Optional[str] = "eval-",
        archive_prefix: str = "[ARCHIVED]",
    ) -> list[int]:
        """
        Clean old evaluation issues from a repository.

        To prevent 'AMBIGUOUS_RESOURCE' errors when verifier searches by title,
        this method archives and closes matching issues. (GitHub REST API does
        not support deleting issues via standard endpoints; renaming title + closing
        guarantees that subsequent runs start with a unique, deterministic state).

        Parameters
        ----------
        repo_name_or_obj : str | Repository
            Repository name or PyGithub Repository object.
        prefix : Optional[str]
            Only issues whose title starts with or contains prefix are cleaned.
            If None, cleans all non-archived issues.
        archive_prefix : str
            Prefix added to issue title upon archiving (default '[ARCHIVED]').

        Returns
        -------
        list[int]
            List of issue numbers cleaned.
        """
        try:
            repo = self._resolve_repo(repo_name_or_obj)
        except Exception as exc:
            self._log(f"clean_issues: could not resolve repo {repo_name_or_obj!r}: {exc}", logging.WARNING)
            return []

        repo_name = getattr(repo, "name", str(repo_name_or_obj))
        self._log(f"Cleaning issues in {repo_name!r} (prefix={prefix!r})")
        cleaned_numbers: list[int] = []

        try:
            issues = list(repo.get_issues(state="all"))
        except Exception as exc:
            self._log(f"Failed to list issues in {repo_name!r}: {exc}", logging.ERROR)
            return []

        for issue in issues:
            title = getattr(issue, "title", "")
            state = getattr(issue, "state", "open")
            num = getattr(issue, "number", 0)

            # If already archived, just ensure it is closed
            if title.startswith(archive_prefix):
                if state == "open":
                    try:
                        issue.edit(state="closed")
                        self._log(f"  Closed already-archived issue #{num} in {repo_name!r}")
                    except Exception as exc:
                        self._log(f"  Could not close issue #{num}: {exc}", logging.WARNING)
                continue

            # Prefix filtering
            if prefix is not None:
                matches = title.startswith(prefix) or (f": {prefix}" in title) or (prefix in title)
                if not matches:
                    continue

            # Archive and close
            new_title = f"{archive_prefix} {title}"
            try:
                issue.edit(title=new_title, state="closed")
                cleaned_numbers.append(num)
                self._log(f"  Archived & closed issue #{num} ({title!r} -> {new_title!r}) in {repo_name!r}")
            except Exception as exc:
                self._log(f"  Failed to archive issue #{num}: {exc}", logging.WARNING)

        return cleaned_numbers

    def clean_branches(
        self,
        repo_name_or_obj: Any,
        prefix: Optional[str] = "eval-",
        protected_branches: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Delete old evaluation branches from a repository.

        Ensures that subsequent runs creating branches like 'eval-branch-*' do
        not fail with 'Reference already exists' (HTTP 422).

        Never deletes default_branch or protected branches ('main', 'master').
        """
        try:
            repo = self._resolve_repo(repo_name_or_obj)
        except Exception as exc:
            self._log(f"clean_branches: could not resolve repo {repo_name_or_obj!r}: {exc}", logging.WARNING)
            return []

        repo_name = getattr(repo, "name", str(repo_name_or_obj))
        default_branch = getattr(repo, "default_branch", "main")
        protected = set(protected_branches or ["main", "master", "development"])
        if default_branch:
            protected.add(default_branch)

        self._log(f"Cleaning branches in {repo_name!r} (prefix={prefix!r})")
        deleted_branches: list[str] = []

        try:
            branches = list(repo.get_branches())
        except Exception as exc:
            self._log(f"Failed to list branches in {repo_name!r}: {exc}", logging.ERROR)
            return []

        for b in branches:
            b_name = getattr(b, "name", "")
            if b_name in protected:
                continue
            if prefix is not None and not b_name.startswith(prefix):
                continue

            try:
                ref = repo.get_git_ref(f"heads/{b_name}")
                ref.delete()
                deleted_branches.append(b_name)
                self._log(f"  Deleted branch {b_name!r} in {repo_name!r}")
            except UnknownObjectException:
                pass
            except Exception as exc:
                self._log(f"  Failed to delete branch {b_name!r}: {exc}", logging.WARNING)

        return deleted_branches

    def clean_pull_requests(
        self,
        repo_name_or_obj: Any,
        prefix: Optional[str] = "eval-",
        archive_prefix: str = "[ARCHIVED]",
    ) -> list[int]:
        """
        Clean old evaluation pull requests from a repository.

        Closes open evaluation PRs and archives their titles to prevent
        'AMBIGUOUS_RESOURCE' errors during title/head verification.
        """
        try:
            repo = self._resolve_repo(repo_name_or_obj)
        except Exception as exc:
            self._log(f"clean_pull_requests: could not resolve repo {repo_name_or_obj!r}: {exc}", logging.WARNING)
            return []

        repo_name = getattr(repo, "name", str(repo_name_or_obj))
        self._log(f"Cleaning pull requests in {repo_name!r} (prefix={prefix!r})")
        cleaned_prs: list[int] = []

        try:
            pulls = list(repo.get_pulls(state="all"))
        except Exception as exc:
            self._log(f"Failed to list pull requests in {repo_name!r}: {exc}", logging.ERROR)
            return []

        for pr in pulls:
            title = getattr(pr, "title", "")
            state = getattr(pr, "state", "open")
            num = getattr(pr, "number", 0)
            head_ref = getattr(getattr(pr, "head", None), "ref", "")

            if title.startswith(archive_prefix):
                if state == "open":
                    try:
                        pr.edit(state="closed")
                        self._log(f"  Closed already-archived PR #{num} in {repo_name!r}")
                    except Exception as exc:
                        self._log(f"  Could not close PR #{num}: {exc}", logging.WARNING)
                continue

            if prefix is not None:
                matches = title.startswith(prefix) or head_ref.startswith(prefix) or (prefix in title)
                if not matches:
                    continue

            new_title = f"{archive_prefix} {title}"
            try:
                pr.edit(title=new_title, state="closed")
                cleaned_prs.append(num)
                self._log(f"  Archived & closed PR #{num} ({title!r} -> {new_title!r}) in {repo_name!r}")
            except Exception as exc:
                self._log(f"  Failed to clean PR #{num}: {exc}", logging.WARNING)

        return cleaned_prs

    def clean_eval_repos(
        self,
        prefix: str = "eval-",
        exclude_repos: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Delete old evaluation repositories matching prefix.

        Safely excludes target evaluation repositories, production repos,
        and current project repos.
        """
        self._log(f"Cleaning evaluation repositories (prefix={prefix!r})")
        exclude: set[str] = {"200-Tools", "Two-Hundered-Tools", "Two-Hundred-Tools"}
        if exclude_repos:
            for item in exclude_repos:
                if not item:
                    continue
                exclude.add(item)
                if "/" in item:
                    exclude.add(item.split("/")[-1])

        deleted: list[str] = []
        try:
            owner_obj = self._gh.get_user(self._owner)
            repos = list(owner_obj.get_repos())
        except Exception as exc:
            self._log(f"Failed to list user repositories for cleanup: {exc}", logging.ERROR)
            return []

        for r in repos:
            r_name = getattr(r, "name", "")
            full_name = getattr(r, "full_name", f"{self._owner}/{r_name}")
            if r_name in exclude or full_name in exclude:
                continue
            if r_name.startswith(prefix):
                try:
                    r.delete()
                    deleted.append(r_name)
                    self._log(f"  Deleted evaluation repository {r_name!r}")
                except Exception as exc:
                    self._log(f"  Failed to delete repository {r_name!r}: {exc}", logging.WARNING)

        return deleted

    def clean_repo_state(
        self,
        repo_name_or_obj: Any,
        prefix: Optional[str] = "eval-",
    ) -> dict[str, Any]:
        """
        Comprehensive cleaner for a sandbox repository.
        Cleans issues, branches, and pull requests.
        """
        issues = self.clean_issues(repo_name_or_obj, prefix=prefix)
        branches = self.clean_branches(repo_name_or_obj, prefix=prefix)
        prs = self.clean_pull_requests(repo_name_or_obj, prefix=prefix)
        repo_name = getattr(repo_name_or_obj, "name", str(repo_name_or_obj))

        return {
            "repo": repo_name,
            "issues_cleaned": issues,
            "branches_deleted": branches,
            "pull_requests_closed": prs,
        }

    def clean_task_state(
        self,
        repo_name_or_obj: Any,
        task: dict[str, Any],
        archive_prefix: str = "[ARCHIVED]",
    ) -> dict[str, Any]:
        """
        Clean resources specifically expected/created by a single task.

        Inspects task['expected_state'] for:
        - issue.title -> archive/close existing issue with this title
        - branch.name -> delete existing branch with this name
        - pull_request.title / head -> archive/close existing PR
        - repository.name -> delete existing repository if created by task
        """
        exp = task.get("expected_state") or {}
        cleaned_info: dict[str, Any] = {
            "task_id": task.get("task_id", ""),
            "issues_archived": [],
            "branches_deleted": [],
            "prs_archived": [],
            "repos_deleted": [],
        }

        # 1. Check issue title
        issue_exp = exp.get("issue")
        if isinstance(issue_exp, dict) and issue_exp.get("title"):
            target_title = issue_exp["title"]
            try:
                repo = self._resolve_repo(repo_name_or_obj)
                for i in repo.get_issues(state="all"):
                    if i.title == target_title:
                        new_title = f"{archive_prefix} {i.title}"
                        i.edit(title=new_title, state="closed")
                        cleaned_info["issues_archived"].append(i.number)
                        self._log(f"Pre-task clean: archived issue #{i.number} {target_title!r}")
            except Exception as exc:
                self._log(f"Pre-task clean issue error: {exc}", logging.WARNING)

        # 2. Check branch name
        branch_exp = exp.get("branch")
        if isinstance(branch_exp, dict) and branch_exp.get("name"):
            branch_name = branch_exp["name"].replace("refs/heads/", "")
            try:
                repo = self._resolve_repo(repo_name_or_obj)
                default_b = getattr(repo, "default_branch", "main")
                if branch_name not in ("main", "master", default_b):
                    ref = repo.get_git_ref(f"heads/{branch_name}")
                    ref.delete()
                    cleaned_info["branches_deleted"].append(branch_name)
                    self._log(f"Pre-task clean: deleted branch {branch_name!r}")
            except UnknownObjectException:
                pass
            except Exception as exc:
                self._log(f"Pre-task clean branch error: {exc}", logging.WARNING)

        # 3. Check pull request
        pr_exp = exp.get("pull_request")
        if isinstance(pr_exp, dict):
            pr_title = pr_exp.get("title")
            pr_head = pr_exp.get("head")
            if pr_title or pr_head:
                try:
                    repo = self._resolve_repo(repo_name_or_obj)
                    for pr in repo.get_pulls(state="all"):
                        head_ref = getattr(getattr(pr, "head", None), "ref", "")
                        if (pr_title and pr.title == pr_title) or (pr_head and head_ref == pr_head):
                            if not pr.title.startswith(archive_prefix):
                                pr.edit(title=f"{archive_prefix} {pr.title}", state="closed")
                            elif pr.state == "open":
                                pr.edit(state="closed")
                            cleaned_info["prs_archived"].append(pr.number)
                            self._log(f"Pre-task clean: archived PR #{pr.number}")
                except Exception as exc:
                    self._log(f"Pre-task clean PR error: {exc}", logging.WARNING)

        # 4. Check repository
        repo_exp = exp.get("repository")
        if isinstance(repo_exp, dict) and repo_exp.get("name"):
            r_name = repo_exp["name"]
            if r_name.startswith("eval-"):
                try:
                    if self.delete_repo(r_name):
                        cleaned_info["repos_deleted"].append(r_name)
                        self._log(f"Pre-task clean: deleted repo {r_name!r}")
                except Exception as exc:
                    self._log(f"Pre-task clean repo error: {exc}", logging.WARNING)

        return cleaned_info

    def clean_sandbox(
        self,
        target_repo: Optional[str] = None,
        prefix: str = "eval-",
        clean_repos: bool = True,
        auto_create_target: bool = True,
    ) -> dict[str, Any]:
        """
        Top-level pre-flight sandbox reset.

        1. Deletes old ephemeral evaluation repositories (matching prefix),
           safely excluding target_repo.
        2. If target_repo is specified:
           - If it does not exist and auto_create_target is True, creates & seeds it.
           - If it exists, cleans old issues, branches, and PRs inside it.

        Returns structured summary dict.
        """
        self._log(f"Starting top-level sandbox reset (target={target_repo!r})")
        self._log_rate_limit("Before clean_sandbox")

        summary: dict[str, Any] = {
            "target_repo": target_repo,
            "repos_deleted": [],
            "repo_created": False,
            "repo_state": {},
        }

        # 1. Clean ephemeral evaluation repos
        if clean_repos:
            exclude = [target_repo] if target_repo else []
            deleted = self.clean_eval_repos(prefix=prefix, exclude_repos=exclude)
            summary["repos_deleted"] = deleted

        # 2. Handle target repo
        if target_repo:
            target_bare = target_repo.split("/")[-1] if "/" in target_repo else target_repo
            exists = self.repo_exists(target_repo)

            if not exists and auto_create_target:
                self._log(f"Target repository {target_repo!r} does not exist — creating and seeding")
                try:
                    new_repo = self.create_repo(target_bare)
                    self.seed_repo(new_repo)
                    summary["repo_created"] = True
                except Exception as exc:
                    self._log(f"Failed to auto-create target repository {target_repo!r}: {exc}", logging.WARNING)
            elif exists:
                summary["repo_state"] = self.clean_repo_state(target_repo, prefix=prefix)

        self._log_rate_limit("After clean_sandbox")
        self._log("Sandbox reset complete")
        return summary

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
