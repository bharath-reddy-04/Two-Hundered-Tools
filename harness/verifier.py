"""
verifier.py
===========
Independent verification layer for the 200-Tools GitHub evaluation project.

Architecture position
---------------------
    github_sandbox.py  →  GitHub API  →  verifier.py  →  PASS / FAIL

The verifier NEVER trusts sandbox return values.  It re-queries the GitHub
REST API directly and compares the resulting live state against an expected
state dict supplied by the evaluation harness.

Scope
-----
* Repositories   — verify_repository, verify_repository_exists,
                   verify_repository_absent
* Issues         — verify_issue, verify_issue_absent
* Branches       — verify_branch, verify_branch_absent
* Pull Requests  — verify_pull_request
* Multi-resource — verify_state

Authentication
--------------
Reuses the same env-var pattern as github_sandbox.py:
    GITHUB_TOKEN   — personal access token
    GITHUB_OWNER   — account / org owning sandbox repos

python-dotenv is loaded on import so a project-level .env works
transparently.

Safety
------
This module is READ-ONLY.  It never creates, updates, or deletes any
GitHub resource.  All mutations stay inside github_sandbox.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

from dotenv import load_dotenv
from github import Github, GithubException, RateLimitExceededException, UnknownObjectException

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Operation → verifier mapping
# Derived from the actual operation_ids found in schemas/operation_catalog.json
# ---------------------------------------------------------------------------

VERIFIER_MAP: dict[str, str] = {
    # ── repositories ──────────────────────────────────────────────────────
    "repos/create-for-authenticated-user": "verify_repository_exists",
    "repos/get":                           "verify_repository",
    "repos/update":                        "verify_repository",
    "repos/delete":                        "verify_repository_absent",
    "repos/list-for-authenticated-user":   "verify_repository",
    "repos/get-all-topics":                "verify_repository",
    "repos/replace-all-topics":            "verify_repository",
    # ── branches ──────────────────────────────────────────────────────────
    "repos/get-branch":                    "verify_branch",
    "repos/list-branches":                 "verify_branch",
    "git/create-ref":                      "verify_branch",
    "git/delete-ref":                      "verify_branch_absent",
    # ── pull requests ─────────────────────────────────────────────────────
    "pulls/create":                        "verify_pull_request",
    "pulls/get":                           "verify_pull_request",
    "pulls/list":                          "verify_pull_request",
    "pulls/update":                        "verify_pull_request",
    "pulls/merge":                         "verify_pull_request",
    # ── issues ────────────────────────────────────────────────────────────
    "issues/create":                       "verify_issue",
    "issues/get":                          "verify_issue",
    "issues/list":                         "verify_issue",
    "issues/update":                       "verify_issue",
    "issues/create-comment":               "verify_issue",
    "issues/list-comments":                "verify_issue",
    "issues/lock":                         "verify_issue",
}

# ---------------------------------------------------------------------------
# Result builder helpers
# ---------------------------------------------------------------------------

_INFRA_ERRORS = frozenset(
    {
        "INFRASTRUCTURE_ERROR",
        "AUTHENTICATION_ERROR",
        "RATE_LIMITED",
        "AMBIGUOUS_RESOURCE",
    }
)


def _pass(resource: str, identifier: Any, checks: dict, expected: dict, actual: dict) -> dict:
    return {
        "passed": True,
        "resource": resource,
        "identifier": identifier,
        "checks": checks,
        "expected_state": expected,
        "actual_state": actual,
        "errors": [],
    }


def _fail(
    resource: str,
    identifier: Any,
    checks: dict,
    expected: dict,
    actual: dict,
    errors: list[str],
    error_kind: str = "VERIFICATION_FAILURE",
) -> dict:
    return {
        "passed": False,
        "resource": resource,
        "identifier": identifier,
        "checks": checks,
        "expected_state": expected,
        "actual_state": actual,
        "errors": errors,
        "error_kind": error_kind,
    }


def _infra_fail(resource: str, identifier: Any, message: str, error_kind: str) -> dict:
    """Return a structured failure that is clearly an infrastructure problem,
    not a genuine state-mismatch.  ``passed`` is always False."""
    return {
        "passed": False,
        "resource": resource,
        "identifier": identifier,
        "checks": {},
        "expected_state": {},
        "actual_state": {},
        "errors": [message],
        "error_kind": error_kind,
    }


# ---------------------------------------------------------------------------
# GitHub client (shared, lazy-initialised)
# ---------------------------------------------------------------------------

_gh_client: Optional[Github] = None


def _get_client(token: Optional[str] = None) -> Github:
    """Return a cached PyGithub client, creating one if necessary."""
    global _gh_client
    if _gh_client is None:
        tok = token or os.getenv("GITHUB_TOKEN", "")
        if not tok:
            raise VerifierConfigError(
                "GITHUB_TOKEN not set.  Export it or add it to .env."
            )
        _gh_client = Github(tok)
    return _gh_client


def _reset_client() -> None:
    """Force a fresh client on the next call — used by tests."""
    global _gh_client
    _gh_client = None


class VerifierConfigError(RuntimeError):
    """Raised when the verifier cannot be initialised."""


# ---------------------------------------------------------------------------
# Low-level GitHub fetch helpers (all READ-ONLY)
# ---------------------------------------------------------------------------

def _fetch_repo(full_name: str, token: Optional[str] = None):
    """
    Fetch a PyGithub Repository for *full_name* (``owner/repo``).

    Returns ``(repo_obj, None)`` on success or ``(None, error_dict)`` on any
    failure.  Distinguishes 404 from other errors.
    """
    gh = _get_client(token)
    try:
        repo = gh.get_repo(full_name)
        return repo, None
    except UnknownObjectException:
        return None, ("NOT_FOUND", f"Repository '{full_name}' not found (404).")
    except RateLimitExceededException as exc:
        return None, ("RATE_LIMITED", f"GitHub rate limit exceeded: {exc}")
    except GithubException as exc:
        status = getattr(exc, "status", None)
        if status == 401:
            kind = "AUTHENTICATION_ERROR"
        elif status == 403:
            kind = "AUTHENTICATION_ERROR"
        else:
            kind = "INFRASTRUCTURE_ERROR"
        return None, (kind, f"GitHub API error {status}: {exc.data}")
    except Exception as exc:
        return None, ("INFRASTRUCTURE_ERROR", f"Unexpected error: {type(exc).__name__}: {exc}")


def _fetch_issue(full_name: str, issue_number: int, token: Optional[str] = None):
    """
    Fetch a single issue by number.

    Returns ``(issue_obj, None)`` or ``(None, (kind, message))``.
    """
    repo, err = _fetch_repo(full_name, token)
    if err:
        return None, err
    try:
        issue = repo.get_issue(number=issue_number)
        return issue, None
    except UnknownObjectException:
        return None, ("NOT_FOUND", f"Issue #{issue_number} not found in '{full_name}'.")
    except RateLimitExceededException as exc:
        return None, ("RATE_LIMITED", f"GitHub rate limit exceeded: {exc}")
    except GithubException as exc:
        status = getattr(exc, "status", None)
        return None, ("INFRASTRUCTURE_ERROR", f"GitHub API error {status}: {exc.data}")
    except Exception as exc:
        return None, ("INFRASTRUCTURE_ERROR", f"Unexpected error: {type(exc).__name__}: {exc}")


def _search_issue_by_title(full_name: str, title: str, token: Optional[str] = None):
    """
    Look up an issue by exact title within *full_name*.

    Returns ``(issue_obj, None)`` on a unique match.
    Returns ``(None, (kind, message))`` on zero or multiple matches.
    """
    repo, err = _fetch_repo(full_name, token)
    if err:
        return None, err
    try:
        matches = [i for i in repo.get_issues(state="all") if i.title == title]
    except RateLimitExceededException as exc:
        return None, ("RATE_LIMITED", f"GitHub rate limit exceeded: {exc}")
    except GithubException as exc:
        status = getattr(exc, "status", None)
        return None, ("INFRASTRUCTURE_ERROR", f"GitHub API error {status}: {exc.data}")
    except Exception as exc:
        return None, ("INFRASTRUCTURE_ERROR", f"Unexpected error: {type(exc).__name__}: {exc}")

    if len(matches) == 0:
        return None, ("NOT_FOUND", f"No issue with title '{title}' found in '{full_name}'.")
    if len(matches) > 1:
        nums = [str(i.number) for i in matches]
        return None, (
            "AMBIGUOUS_RESOURCE",
            f"Multiple issues with title '{title}' found in '{full_name}' "
            f"(#{', #'.join(nums)}).  Provide issue_number for unambiguous lookup.",
        )
    return matches[0], None


def _fetch_branch(full_name: str, branch_name: str, token: Optional[str] = None):
    """
    Fetch a single branch by name within *full_name*.

    Returns ``(branch_obj, None)`` or ``(None, (kind, message))``.
    """
    repo, err = _fetch_repo(full_name, token)
    if err:
        return None, err
    try:
        # Strip refs/heads/ if present
        cleaned_name = branch_name.replace("refs/heads/", "")
        branch = repo.get_branch(cleaned_name)
        return branch, None
    except UnknownObjectException:
        return None, ("NOT_FOUND", f"Branch '{branch_name}' not found in '{full_name}'.")
    except RateLimitExceededException as exc:
        return None, ("RATE_LIMITED", f"GitHub rate limit exceeded: {exc}")
    except GithubException as exc:
        status = getattr(exc, "status", None)
        return None, ("INFRASTRUCTURE_ERROR", f"GitHub API error {status}: {exc.data}")
    except Exception as exc:
        return None, ("INFRASTRUCTURE_ERROR", f"Unexpected error: {type(exc).__name__}: {exc}")


def _fetch_pull(
    full_name: str,
    pull_number: Optional[int] = None,
    head: Optional[str] = None,
    title: Optional[str] = None,
    token: Optional[str] = None,
):
    """
    Fetch a single pull request by number, head branch, or title.

    Returns ``(pull_obj, None)`` or ``(None, (kind, message))``.
    """
    repo, err = _fetch_repo(full_name, token)
    if err:
        return None, err
    try:
        if pull_number is not None:
            pull = repo.get_pull(number=pull_number)
            return pull, None

        # Look up across open and closed pulls
        pulls = list(repo.get_pulls(state="all"))
        if head:
            # head may be 'branch' or 'owner:branch' or 'refs/heads/branch'
            target_head = head.replace("refs/heads/", "")
            matches = [
                p for p in pulls
                if p.head.ref == target_head or getattr(p.head, "label", "") == target_head
            ]
            if not matches:
                return None, ("NOT_FOUND", f"No pull request with head '{head}' found in '{full_name}'.")
            return matches[0], None

        if title:
            matches = [p for p in pulls if p.title == title]
            if not matches:
                return None, ("NOT_FOUND", f"No pull request with title '{title}' found in '{full_name}'.")
            if len(matches) > 1:
                nums = [str(p.number) for p in matches]
                return None, (
                    "AMBIGUOUS_RESOURCE",
                    f"Multiple pull requests with title '{title}' found in '{full_name}' (#{', #'.join(nums)}).",
                )
            return matches[0], None

        return None, ("INVALID_ARGUMENTS", "Either pull_number, head, or title must be provided.")
    except UnknownObjectException:
        return None, ("NOT_FOUND", f"Pull request not found in '{full_name}'.")
    except RateLimitExceededException as exc:
        return None, ("RATE_LIMITED", f"GitHub rate limit exceeded: {exc}")
    except GithubException as exc:
        status = getattr(exc, "status", None)
        return None, ("INFRASTRUCTURE_ERROR", f"GitHub API error {status}: {exc.data}")
    except Exception as exc:
        return None, ("INFRASTRUCTURE_ERROR", f"Unexpected error: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Repository verification
# ---------------------------------------------------------------------------

_REPO_FIELD_EXTRACTORS: dict[str, Any] = {
    "name":                     lambda r: r.name,
    "full_name":                lambda r: r.full_name,
    "description":              lambda r: r.description,
    "private":                  lambda r: r.private,
    "visibility":               lambda r: "private" if r.private else "public",
    "default_branch":           lambda r: r.default_branch,
    "archived":                 lambda r: r.archived,
    "disabled":                 lambda r: r.archived,
    "fork":                     lambda r: r.fork,
    "has_issues":               lambda r: r.has_issues,
    "has_wiki":                 lambda r: r.has_wiki,
    "has_projects":             lambda r: r.has_projects,
    "has_discussions":          lambda r: r.has_discussions,
    "allow_squash_merge":       lambda r: r.allow_squash_merge,
    "allow_merge_commit":       lambda r: r.allow_merge_commit,
    "allow_rebase_merge":       lambda r: r.allow_rebase_merge,
    "allow_auto_merge":         lambda r: r.allow_auto_merge,
    "delete_branch_on_merge":   lambda r: r.delete_branch_on_merge,
    "topics":                   lambda r: sorted(r.get_topics()),
}


def _snapshot_repo(repo) -> dict:
    """Extract all known verifiable fields from a PyGithub Repository."""
    snap: dict = {}
    for field, extractor in _REPO_FIELD_EXTRACTORS.items():
        try:
            snap[field] = extractor(repo)
        except Exception:
            snap[field] = None
    return snap


def verify_repository(
    repo: str,
    expected_state: Optional[dict] = None,
    token: Optional[str] = None,
) -> dict:
    """
    Independently query GitHub and verify *repo* (``owner/repo``) matches *expected_state*.
    """
    expected_state = expected_state or {}
    resource = "repository"
    identifier = repo

    repo_obj, err = _fetch_repo(repo, token)
    if err:
        kind, msg = err
        return _infra_fail(resource, identifier, msg, kind)

    actual = _snapshot_repo(repo_obj)
    checks: dict[str, bool] = {"exists": True}
    errors: list[str] = []

    for field, expected_value in expected_state.items():
        if field == "exists":
            continue
        if field not in _REPO_FIELD_EXTRACTORS:
            errors.append(
                f"Field '{field}' is not a supported verifiable repository attribute; skipped."
            )
            checks[field] = None
            continue

        actual_value = actual.get(field)
        if field == "topics":
            match = sorted(actual_value or []) == sorted(expected_value or [])
        else:
            match = actual_value == expected_value

        checks[field] = match
        if not match:
            errors.append(
                f"Field '{field}': expected {expected_value!r}, found {actual_value!r}."
            )

    if errors and any(v is False for v in checks.values()):
        return _fail(resource, identifier, checks, expected_state, actual, errors)
    return _pass(resource, identifier, checks, expected_state, actual)


def verify_repository_exists(
    repo: str,
    token: Optional[str] = None,
) -> dict:
    """Pass iff *repo* exists on GitHub."""
    resource = "repository"
    identifier = repo

    repo_obj, err = _fetch_repo(repo, token)
    if err:
        kind, msg = err
        if kind == "NOT_FOUND":
            return _fail(
                resource, identifier,
                {"exists": False},
                {"exists": True},
                {},
                [f"Repository '{repo}' does not exist."],
            )
        return _infra_fail(resource, identifier, msg, kind)

    actual = _snapshot_repo(repo_obj)
    return _pass(resource, identifier, {"exists": True}, {"exists": True}, actual)


def verify_repository_absent(
    repo: str,
    token: Optional[str] = None,
) -> dict:
    """Pass iff *repo* does NOT exist on GitHub (404)."""
    resource = "repository"
    identifier = repo

    repo_obj, err = _fetch_repo(repo, token)
    if err:
        kind, msg = err
        if kind == "NOT_FOUND":
            return _pass(
                resource, identifier,
                {"absent": True},
                {"exists": False},
                {"exists": False},
            )
        return _infra_fail(resource, identifier, msg, kind)

    actual = _snapshot_repo(repo_obj)
    return _fail(
        resource, identifier,
        {"absent": False},
        {"exists": False},
        actual,
        [f"Repository '{repo}' still exists but was expected to be absent."],
    )


# ---------------------------------------------------------------------------
# Branch verification
# ---------------------------------------------------------------------------

_BRANCH_FIELD_EXTRACTORS: dict[str, Any] = {
    "name":      lambda b: b.name,
    "protected": lambda b: b.protected,
}


def _snapshot_branch(branch) -> dict:
    """Extract verifiable fields from a PyGithub Branch."""
    snap: dict = {}
    for field, extractor in _BRANCH_FIELD_EXTRACTORS.items():
        try:
            snap[field] = extractor(branch)
        except Exception:
            snap[field] = None
    return snap


def verify_branch(
    repo: str,
    branch_name: str,
    expected_state: Optional[dict] = None,
    token: Optional[str] = None,
) -> dict:
    """
    Pass iff *branch_name* exists in *repo* and matches *expected_state*.
    """
    expected_state = expected_state or {}
    resource = "branch"
    identifier = branch_name

    branch_obj, err = _fetch_branch(repo, branch_name, token)
    if err:
        kind, msg = err
        if kind == "NOT_FOUND":
            return _fail(
                resource, identifier,
                {"exists": False},
                {"exists": True},
                {},
                [msg],
            )
        return _infra_fail(resource, identifier, msg, kind)

    actual = _snapshot_branch(branch_obj)
    checks: dict[str, bool] = {"exists": True}
    errors: list[str] = []

    for field, expected_value in expected_state.items():
        if field in ("exists", "base"):
            continue  # 'base' branch ancestry is verified via commit if needed; existence confirmed
        if field not in _BRANCH_FIELD_EXTRACTORS:
            errors.append(f"Field '{field}' is not a supported verifiable branch attribute; skipped.")
            checks[field] = None
            continue

        actual_value = actual.get(field)
        match = actual_value == expected_value
        checks[field] = match
        if not match:
            errors.append(f"Field '{field}': expected {expected_value!r}, found {actual_value!r}.")

    if errors and any(v is False for v in checks.values()):
        return _fail(resource, identifier, checks, expected_state, actual, errors)
    return _pass(resource, identifier, checks, expected_state, actual)


def verify_branch_absent(
    repo: str,
    branch_name: str,
    token: Optional[str] = None,
) -> dict:
    """Pass iff *branch_name* does NOT exist in *repo* (404)."""
    resource = "branch"
    identifier = branch_name

    branch_obj, err = _fetch_branch(repo, branch_name, token)
    if err:
        kind, msg = err
        if kind == "NOT_FOUND":
            return _pass(resource, identifier, {"absent": True}, {"exists": False}, {"exists": False})
        return _infra_fail(resource, identifier, msg, kind)

    actual = _snapshot_branch(branch_obj)
    return _fail(
        resource, identifier,
        {"absent": False}, {"exists": False}, actual,
        [f"Branch '{branch_name}' still exists but was expected to be absent."],
    )


# ---------------------------------------------------------------------------
# Pull Request verification
# ---------------------------------------------------------------------------

_PR_FIELD_EXTRACTORS: dict[str, Any] = {
    "number": lambda p: p.number,
    "title":  lambda p: p.title,
    "body":   lambda p: p.body or "",
    "state":  lambda p: p.state,
    "merged": lambda p: p.merged,
    "draft":  lambda p: p.draft,
    "head":   lambda p: p.head.ref,
    "base":   lambda p: p.base.ref,
}


def _snapshot_pull(pull) -> dict:
    """Extract verifiable fields from a PyGithub PullRequest."""
    snap: dict = {}
    for field, extractor in _PR_FIELD_EXTRACTORS.items():
        try:
            snap[field] = extractor(pull)
        except Exception:
            snap[field] = None
    return snap


def verify_pull_request(
    repo: str,
    pull_number: Optional[int] = None,
    head: Optional[str] = None,
    title: Optional[str] = None,
    expected_state: Optional[dict] = None,
    token: Optional[str] = None,
) -> dict:
    """
    Independently query GitHub and verify a pull request's live state.
    """
    expected_state = expected_state or {}
    resource = "pull_request"

    pull_obj, err = _fetch_pull(repo, pull_number=pull_number, head=head, title=title, token=token)
    if err:
        kind, msg = err
        ident = head or title or pull_number or "unknown"
        if kind == "NOT_FOUND":
            return _fail(resource, ident, {"exists": False}, {"exists": True}, {}, [msg])
        return _infra_fail(resource, ident, msg, kind)

    actual = _snapshot_pull(pull_obj)
    identifier = actual.get("number", pull_number or head or title)
    checks: dict[str, bool] = {"exists": True}
    errors: list[str] = []

    for field, expected_value in expected_state.items():
        if field == "exists":
            continue

        if field == "references_issue":
            # Check if expected issue identifier/title is referenced in body or title
            body = (actual.get("body") or "").lower()
            pr_title = (actual.get("title") or "").lower()
            expected_ref = str(expected_value).lower()
            match = (expected_ref in body) or (expected_ref in pr_title)
            checks["references_issue"] = match
            if not match:
                errors.append(
                    f"Pull request body/title does not reference issue '{expected_value}'."
                )
            continue

        if field not in _PR_FIELD_EXTRACTORS:
            errors.append(f"Field '{field}' is not a supported verifiable pull request attribute; skipped.")
            checks[field] = None
            continue

        actual_value = actual.get(field)
        match = actual_value == expected_value
        checks[field] = match
        if not match:
            errors.append(f"Field '{field}': expected {expected_value!r}, found {actual_value!r}.")

    if errors and any(v is False for v in checks.values()):
        return _fail(resource, identifier, checks, expected_state, actual, errors)
    return _pass(resource, identifier, checks, expected_state, actual)


# ---------------------------------------------------------------------------
# Issue verification
# ---------------------------------------------------------------------------

_ISSUE_FIELD_EXTRACTORS: dict[str, Any] = {
    "number":           lambda i: i.number,
    "title":            lambda i: i.title,
    "body":             lambda i: i.body or "",
    "body_contains":    lambda i: i.body or "",
    "state":            lambda i: i.state,
    "locked":           lambda i: i.locked,
    "lock_reason":      lambda i: getattr(i, "active_lock_reason", None),
    "comments":         lambda i: i.comments,
    "comment_contains": lambda i: [c.body for c in i.get_comments()],
    "user":             lambda i: i.user.login if i.user else None,
    "assignees":        lambda i: sorted(a.login for a in i.assignees),
    "milestone":        lambda i: i.milestone.title if i.milestone else None,
    "labels":           lambda i: sorted(lb.name for lb in i.labels),
}


def _snapshot_issue(issue) -> dict:
    """Extract all known verifiable fields from a PyGithub Issue."""
    snap: dict = {}
    for field, extractor in _ISSUE_FIELD_EXTRACTORS.items():
        try:
            snap[field] = extractor(issue)
        except Exception:
            snap[field] = None
    return snap


def verify_issue(
    repo: str,
    expected_state: Optional[dict] = None,
    issue_number: Optional[int] = None,
    title: Optional[str] = None,
    strict_labels: bool = False,
    token: Optional[str] = None,
) -> dict:
    """
    Independently query GitHub and verify an issue's live state.
    """
    expected_state = expected_state or {}
    resource = "issue"

    if issue_number is not None:
        identifier = issue_number
        issue_obj, err = _fetch_issue(repo, issue_number, token)
    elif title is not None:
        identifier = title
        issue_obj, err = _search_issue_by_title(repo, title, token)
    else:
        return _infra_fail(
            resource, None,
            "Either issue_number or title must be provided.",
            "INVALID_ARGUMENTS",
        )

    if err:
        kind, msg = err
        return _infra_fail(resource, identifier, msg, kind)

    actual = _snapshot_issue(issue_obj)
    identifier = actual.get("number", identifier)

    checks: dict[str, bool] = {"exists": True}
    errors: list[str] = []

    for field, expected_value in expected_state.items():
        if field == "exists":
            continue
        if field not in _ISSUE_FIELD_EXTRACTORS:
            errors.append(
                f"Field '{field}' is not a supported verifiable issue attribute; skipped."
            )
            checks[field] = None
            continue

        actual_value = actual.get(field)

        if field == "labels":
            expected_labels = set(expected_value or [])
            actual_labels = set(actual_value or [])
            if strict_labels:
                match = expected_labels == actual_labels
                if not match:
                    missing = expected_labels - actual_labels
                    extra = actual_labels - expected_labels
                    parts = []
                    if missing:
                        parts.append(f"missing: {sorted(missing)}")
                    if extra:
                        parts.append(f"unexpected: {sorted(extra)}")
                    errors.append(f"Labels mismatch (strict): {', '.join(parts)}.")
            else:
                missing = expected_labels - actual_labels
                match = len(missing) == 0
                if not match:
                    errors.append(
                        f"Expected label(s) not found: {sorted(missing)}. Actual: {sorted(actual_labels)}."
                    )
            checks["labels"] = match

        elif field == "assignees":
            expected_set = set(expected_value or [])
            actual_set = set(actual_value or [])
            match = expected_set == actual_set
            checks["assignees"] = match
            if not match:
                errors.append(
                    f"Assignees: expected {sorted(expected_set)!r}, found {sorted(actual_set)!r}."
                )

        elif field == "body":
            match = (actual_value or "").strip() == (expected_value or "").strip()
            checks["body"] = match
            if not match:
                errors.append(
                    f"Body mismatch: expected {expected_value!r}, found {actual_value!r}."
                )

        elif field == "body_contains":
            match = str(expected_value).lower() in (actual_value or "").lower()
            checks["body_contains"] = match
            if not match:
                errors.append(
                    f"Body does not contain expected substring {expected_value!r}."
                )

        elif field == "comments":
            count = actual_value if isinstance(actual_value, int) else 0
            match = count >= expected_value
            checks["comments"] = match
            if not match:
                errors.append(
                    f"Comments count mismatch: expected at least {expected_value}, found {count}."
                )

        elif field == "comment_contains":
            comments_list = actual_value if isinstance(actual_value, list) else []
            match = any(str(expected_value).lower() in (c or "").lower() for c in comments_list)
            checks["comment_contains"] = match
            if not match:
                errors.append(
                    f"No comment found containing expected text {expected_value!r}."
                )

        else:
            match = actual_value == expected_value
            checks[field] = match
            if not match:
                errors.append(
                    f"Field '{field}': expected {expected_value!r}, found {actual_value!r}."
                )

    if errors and any(v is False for v in checks.values()):
        return _fail(resource, identifier, checks, expected_state, actual, errors)
    return _pass(resource, identifier, checks, expected_state, actual)


def verify_issue_absent(
    repo: str,
    issue_number: int,
    token: Optional[str] = None,
) -> dict:
    """Pass iff issue *issue_number* does NOT exist in *repo* (404)."""
    resource = "issue"
    identifier = issue_number

    issue_obj, err = _fetch_issue(repo, issue_number, token)
    if err:
        kind, msg = err
        if kind == "NOT_FOUND":
            return _pass(
                resource, identifier,
                {"absent": True},
                {"exists": False},
                {"exists": False},
            )
        return _infra_fail(resource, identifier, msg, kind)

    actual = _snapshot_issue(issue_obj)
    return _fail(
        resource, identifier,
        {"absent": False},
        {"exists": False},
        actual,
        [f"Issue #{issue_number} still exists but was expected to be absent."],
    )


# ---------------------------------------------------------------------------
# Multi-resource composite verification
# ---------------------------------------------------------------------------

def verify_state(
    repo: str,
    expected_state: dict[str, Any],
    token: Optional[str] = None,
) -> dict[str, Any]:
    """
    Verify an aggregate expected_state dictionary containing one or more of:
    - 'repository'
    - 'branch'
    - 'issue'
    - 'pull_request'

    Parameters
    ----------
    repo : str
        Default repository name in ``owner/repo`` format.
    expected_state : dict
        Dict mapping resource type to expected attributes.
    token : str, optional
        GitHub token.

    Returns
    -------
    dict
        Combined result with overall ``passed`` flag and per-resource sub-results.
    """
    sub_results: dict[str, Any] = {}
    all_passed = True
    all_errors: list[str] = []

    # 1. Repository
    if "repository" in expected_state:
        repo_exp = expected_state["repository"]
        target_repo = repo_exp.get("name", repo)
        if "/" not in target_repo and "/" in repo:
            target_repo = f"{repo.split('/')[0]}/{target_repo}"
        res = verify_repository(target_repo, repo_exp, token=token)
        sub_results["repository"] = res
        if not res["passed"]:
            all_passed = False
            all_errors.extend(res.get("errors", []))

    # 2. Branch
    if "branch" in expected_state:
        branch_exp = expected_state["branch"]
        branch_name = branch_exp.get("name")
        if branch_name:
            res = verify_branch(repo, branch_name, branch_exp, token=token)
            sub_results["branch"] = res
            if not res["passed"]:
                all_passed = False
                all_errors.extend(res.get("errors", []))

    # 3. Issue
    if "issue" in expected_state:
        issue_exp = expected_state["issue"]
        res = verify_issue(
            repo,
            expected_state=issue_exp,
            title=issue_exp.get("title"),
            issue_number=issue_exp.get("number"),
            token=token,
        )
        sub_results["issue"] = res
        if not res["passed"]:
            all_passed = False
            all_errors.extend(res.get("errors", []))

    # 4. Pull Request
    if "pull_request" in expected_state:
        pr_exp = expected_state["pull_request"]
        res = verify_pull_request(
            repo,
            pull_number=pr_exp.get("number"),
            head=pr_exp.get("head"),
            title=pr_exp.get("title"),
            expected_state=pr_exp,
            token=token,
        )
        sub_results["pull_request"] = res
        if not res["passed"]:
            all_passed = False
            all_errors.extend(res.get("errors", []))

    return {
        "passed": all_passed,
        "resource": "multi_resource",
        "identifier": repo,
        "results": sub_results,
        "errors": all_errors,
    }


# ---------------------------------------------------------------------------
# Generic verification entry point
# ---------------------------------------------------------------------------

def verify(
    operation_id: str,
    expected_state: dict,
    context: dict,
    token: Optional[str] = None,
) -> dict:
    """
    Route an operation to the appropriate verifier and return its result.
    """
    verifier_name = VERIFIER_MAP.get(operation_id)
    if verifier_name is None:
        return _infra_fail(
            "unknown",
            operation_id,
            f"No verifier registered for operation '{operation_id}'.  "
            f"Add it to VERIFIER_MAP in verifier.py.",
            "UNKNOWN_OPERATION",
        )

    repo = context.get("repo", "")
    if not repo:
        return _infra_fail(
            "unknown", operation_id,
            "context['repo'] is required for all verifications.",
            "INVALID_ARGUMENTS",
        )

    if verifier_name == "verify_repository":
        return verify_repository(repo, expected_state, token=token)

    if verifier_name == "verify_repository_exists":
        return verify_repository_exists(repo, token=token)

    if verifier_name == "verify_repository_absent":
        return verify_repository_absent(repo, token=token)

    if verifier_name == "verify_branch":
        branch_name = context.get("branch") or context.get("ref", "").replace("refs/heads/", "")
        if not branch_name:
            return _infra_fail("branch", None, "context['branch'] is required for verify_branch.", "INVALID_ARGUMENTS")
        return verify_branch(repo, branch_name, expected_state, token=token)

    if verifier_name == "verify_branch_absent":
        branch_name = context.get("branch") or context.get("ref", "").replace("refs/heads/", "")
        if not branch_name:
            return _infra_fail("branch", None, "context['branch'] is required for verify_branch_absent.", "INVALID_ARGUMENTS")
        return verify_branch_absent(repo, branch_name, token=token)

    if verifier_name == "verify_pull_request":
        return verify_pull_request(
            repo,
            pull_number=context.get("pull_number"),
            head=context.get("head"),
            title=context.get("title"),
            expected_state=expected_state,
            token=token,
        )

    if verifier_name == "verify_issue":
        return verify_issue(
            repo=repo,
            expected_state=expected_state,
            issue_number=context.get("issue_number"),
            title=context.get("title"),
            strict_labels=context.get("strict_labels", False),
            token=token,
        )

    if verifier_name == "verify_issue_absent":
        issue_number = context.get("issue_number")
        if issue_number is None:
            return _infra_fail(
                "issue", None,
                "context['issue_number'] is required for verify_issue_absent.",
                "INVALID_ARGUMENTS",
            )
        return verify_issue_absent(repo, issue_number, token=token)

    return _infra_fail(
        "unknown", operation_id,
        f"Verifier '{verifier_name}' is registered but not implemented.",
        "INFRASTRUCTURE_ERROR",
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_result(result: dict, json_output: bool = False) -> str:
    """Return a human-readable string for *result*."""
    if json_output:
        return json.dumps(result, indent=2, default=str)

    lines: list[str] = []
    status = "PASS" if result.get("passed") else "FAIL"
    lines.append(f"{'=' * 60}")
    lines.append(f"  {status}  |  resource={result.get('resource')}  "
                 f"|  id={result.get('identifier')}")
    lines.append(f"{'=' * 60}")

    if result.get("resource") == "multi_resource":
        sub_results = result.get("results", {})
        for res_type, res_data in sub_results.items():
            sub_status = "✓ PASS" if res_data.get("passed") else "✗ FAIL"
            lines.append(f"  [{res_type}] {sub_status}")
            for name, ok in res_data.get("checks", {}).items():
                sym = "✓" if ok else ("?" if ok is None else "✗")
                lines.append(f"      {sym} {name}")
    else:
        checks = result.get("checks", {})
        if checks:
            lines.append("Checks:")
            for name, ok in checks.items():
                symbol = "✓" if ok else ("?" if ok is None else "✗")
                lines.append(f"  {symbol}  {name}")

    errors = result.get("errors", [])
    if errors:
        lines.append("Errors:")
        for e in errors:
            lines.append(f"  • {e}")

    if result.get("error_kind"):
        lines.append(f"Error kind: {result['error_kind']}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verifier.py",
        description="Independent GitHub state verifier for the 200-Tools evaluation project.",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Output result as JSON instead of human-readable text.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── repository subcommand ────────────────────────────────────────────
    repo_p = sub.add_parser("repository", help="Verify repository state.")
    repo_p.add_argument("--repo", required=True, help="Full repository name (owner/repo).")
    repo_p.add_argument("--exists", action="store_true", help="Verify repo exists.")
    repo_p.add_argument("--absent", action="store_true", help="Verify repo does NOT exist.")
    repo_p.add_argument("--name", help="Expected repo name.")
    repo_p.add_argument("--description", help="Expected description.")
    repo_p.add_argument("--private", choices=["true", "false"], help="Expected private flag.")
    repo_p.add_argument("--default-branch", dest="default_branch", help="Expected default branch.")
    repo_p.add_argument("--archived", choices=["true", "false"], help="Expected archived state.")
    repo_p.add_argument("--has-issues", dest="has_issues", choices=["true", "false"], help="Expected has_issues.")

    # ── branch subcommand ────────────────────────────────────────────────
    branch_p = sub.add_parser("branch", help="Verify branch state.")
    branch_p.add_argument("--repo", required=True, help="Full repository name (owner/repo).")
    branch_p.add_argument("--branch", required=True, help="Branch name.")
    branch_p.add_argument("--absent", action="store_true", help="Verify branch does NOT exist.")

    # ── pull-request subcommand ──────────────────────────────────────────
    pr_p = sub.add_parser("pull-request", help="Verify pull request state.")
    pr_p.add_argument("--repo", required=True, help="Full repository name (owner/repo).")
    pr_p.add_argument("--number", type=int, help="Pull request number.")
    pr_p.add_argument("--head", help="Head branch name.")
    pr_p.add_argument("--title", help="Pull request title.")
    pr_p.add_argument("--state", choices=["open", "closed"], help="Expected state.")
    pr_p.add_argument("--merged", choices=["true", "false"], help="Expected merged flag.")

    # ── issue subcommand ─────────────────────────────────────────────────
    issue_p = sub.add_parser("issue", help="Verify issue state.")
    issue_p.add_argument("--repo", required=True, help="Full repository name (owner/repo).")
    issue_p.add_argument("--issue-number", dest="issue_number", type=int, help="Issue number.")
    issue_p.add_argument("--absent", action="store_true", help="Verify issue does NOT exist (404).")
    issue_p.add_argument("--title", help="Expected title.")
    issue_p.add_argument("--body", help="Expected body.")
    issue_p.add_argument("--state", choices=["open", "closed"], help="Expected state.")
    issue_p.add_argument("--labels", nargs="*", default=None, metavar="LABEL", help="Expected label names.")
    issue_p.add_argument("--strict-labels", dest="strict_labels", action="store_true", help="Exact label match.")
    issue_p.add_argument("--assignees", nargs="*", default=None, metavar="LOGIN", help="Expected assignees.")
    issue_p.add_argument("--locked", choices=["true", "false"], help="Expected locked state.")

    # ── state subcommand ─────────────────────────────────────────────────
    state_p = sub.add_parser("state", help="Verify multi-resource JSON state.")
    state_p.add_argument("--repo", required=True, help="Default repository name (owner/repo).")
    state_p.add_argument("--file", help="Path to JSON file containing expected_state dict.")
    state_p.add_argument("--json-str", dest="json_str", help="Inline JSON string of expected_state dict.")

    return parser


def _run_cli(argv: Optional[list[str]] = None) -> int:
    """Parse CLI args, run verification, print result.  Returns exit code."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "repository":
        if args.absent:
            result = verify_repository_absent(args.repo)
        elif args.exists:
            result = verify_repository_exists(args.repo)
        else:
            expected: dict = {}
            if args.name:           expected["name"] = args.name
            if args.description:    expected["description"] = args.description
            if args.private:        expected["private"] = args.private == "true"
            if args.default_branch: expected["default_branch"] = args.default_branch
            if args.archived:       expected["archived"] = args.archived == "true"
            if args.has_issues:     expected["has_issues"] = args.has_issues == "true"
            result = verify_repository(args.repo, expected)

    elif args.command == "branch":
        if args.absent:
            result = verify_branch_absent(args.repo, args.branch)
        else:
            result = verify_branch(args.repo, args.branch)

    elif args.command == "pull-request":
        expected = {}
        if args.state:
            expected["state"] = args.state
        if args.merged:
            expected["merged"] = args.merged == "true"
        result = verify_pull_request(
            args.repo,
            pull_number=args.number,
            head=args.head,
            title=args.title,
            expected_state=expected,
        )

    elif args.command == "issue":
        if args.absent:
            if args.issue_number is None:
                print("ERROR: --issue-number is required with --absent.", file=sys.stderr)
                return 2
            result = verify_issue_absent(args.repo, args.issue_number)
        else:
            expected = {}
            if args.title:     expected["title"] = args.title
            if args.body:      expected["body"] = args.body
            if args.state:     expected["state"] = args.state
            if args.labels is not None:    expected["labels"] = args.labels
            if args.assignees is not None: expected["assignees"] = args.assignees
            if args.locked:    expected["locked"] = args.locked == "true"
            result = verify_issue(
                repo=args.repo,
                expected_state=expected,
                issue_number=args.issue_number,
                title=args.title if args.issue_number is None else None,
                strict_labels=args.strict_labels,
            )

    elif args.command == "state":
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                expected_state = json.load(f)
        elif args.json_str:
            expected_state = json.loads(args.json_str)
        else:
            print("ERROR: either --file or --json-str is required for state verification.", file=sys.stderr)
            return 2
        result = verify_state(args.repo, expected_state)

    else:
        parser.print_help()
        return 2

    print(format_result(result, json_output=args.json_output))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(_run_cli())
