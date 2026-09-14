"""
harness/github_operations.py
============================
Modular GitHub operation handlers, schema definitions, and execution dispatcher for 200-Tools.

Owns the operation definitions, schemas, and starter request initialization:
    schemas/operation_catalog.json
          ↓
    github_operations.py (initializes operation definitions, schemas, starter requests, and execution handlers)
          ↓
    discovery.py / task_runner.py / single_prompt_agent.py
          ↓
    github_sandbox.py
          ↓
    GitHub REST API / Sandbox Repositories
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repository Resolution Helper
# ---------------------------------------------------------------------------

def resolve_repo(sandbox: Any, parameters: Dict[str, Any], name: Optional[str] = None) -> Any:
    """
    Resolve a PyGithub Repository object from parameters and sandbox context.

    Parameters
    ----------
    sandbox : GitHubSandbox
        Initialised sandbox instance.
    parameters : dict
        Call arguments containing 'repo' or 'repository', and optionally 'owner'.
    name : str, optional
        Explicit repository name override.

    Returns
    -------
    github.Repository.Repository
    """
    repo_name = name or parameters.get("repo") or parameters.get("repository")
    if not repo_name:
        default_repo = getattr(sandbox, "default_repo", None)
        if isinstance(default_repo, str) and default_repo:
            repo_name = default_repo
        else:
            raise ValueError("Repository name ('repo' or 'repository') is required.")

    bare_name = str(repo_name).split("/")[-1]
    default_sandbox_repo = getattr(sandbox, "default_repo", None)
    if not isinstance(default_sandbox_repo, str):
        default_sandbox_repo = "eval-sandbox-repo"
    if bare_name in ("test-repo-1", "test_repo_1", "repo-1", "repo_1"):
        repo_name = default_sandbox_repo

    owner_name = parameters.get("owner") or getattr(sandbox, "_owner", "")
    if owner_name and "/" not in str(repo_name):
        repo_name = f"{owner_name}/{repo_name}"
    return sandbox.get_repo(repo_name)


def _get_params(parameters: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(parameters or {})
    merged.update(kwargs)
    return merged


def _resolve_issue_number(repo_obj: Any, params: Dict[str, Any]) -> int:
    """
    Extract and validate issue_number from params.

    If the value is missing (e.g. stripped by the template sanitizer) or not
    a valid integer, fall back to the most recently created issue in the
    repo. This handles the common case where the LLM emits a template
    reference like ``${issues/create.number}`` for a chained operation.
    """
    raw = params.get("issue_number")

    # Happy path: already a valid int
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw == int(raw):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw)
        except (ValueError, TypeError):
            pass

    # Auto-resolve: pick the most recently created open issue, fallback to all states
    logger.warning(
        "issue_number is missing or invalid (%r) — auto-resolving from latest issue",
        raw,
    )
    try:
        issues = repo_obj.get_issues(state="open", sort="created", direction="desc")
        if hasattr(issues, "__getitem__") and len(issues) > 0:
            resolved = issues[0].number
        else:
            resolved = next(iter(issues)).number
        logger.info("Auto-resolved issue_number to %d", resolved)
        return resolved
    except (IndexError, StopIteration, AttributeError, Exception):
        pass

    try:
        issues = repo_obj.get_issues(state="all", sort="created", direction="desc")
        if hasattr(issues, "__getitem__") and len(issues) > 0:
            resolved = issues[0].number
        else:
            resolved = next(iter(issues)).number
        logger.info("Auto-resolved issue_number to %d (all states)", resolved)
        return resolved
    except (IndexError, StopIteration, AttributeError, Exception):
        pass

    raise ValueError(
        "issue_number is required but was not provided and no open issues "
        "exist in the repository to auto-resolve from."
    )


def _resolve_pull_number(repo_obj: Any, params: Dict[str, Any]) -> int:
    """
    Extract and validate pull_number from params.

    If the value is missing (e.g. stripped by the template sanitizer) or not
    a valid integer, fall back to the most recently created pull request in the
    repo. This handles the common case where the LLM emits a template
    reference like ``${pulls/create.number}`` for a chained operation.
    """
    raw = (
        params.get("pull_number")
        or params.get("pull_request_number")
        or params.get("pr_number")
        or params.get("number")
    )

    # Happy path: already a valid int
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw == int(raw):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw)
        except (ValueError, TypeError):
            pass

    # Auto-resolve: pick the most recently created open PR, fallback to all states
    logger.warning(
        "pull_number is missing or invalid (%r) — auto-resolving from latest pull request",
        raw,
    )
    try:
        pulls = repo_obj.get_pulls(state="open", sort="created", direction="desc")
        if hasattr(pulls, "__getitem__") and len(pulls) > 0:
            resolved = pulls[0].number
        else:
            resolved = next(iter(pulls)).number
        logger.info("Auto-resolved pull_number to %d", resolved)
        return resolved
    except (IndexError, StopIteration, AttributeError, Exception):
        pass

    try:
        pulls = repo_obj.get_pulls(state="all", sort="created", direction="desc")
        if hasattr(pulls, "__getitem__") and len(pulls) > 0:
            resolved = pulls[0].number
        else:
            resolved = next(iter(pulls)).number
        logger.info("Auto-resolved pull_number to %d (all states)", resolved)
        return resolved
    except (IndexError, StopIteration, AttributeError, Exception):
        pass

    raise ValueError(
        "pull_number is required but was not provided and no open pull requests "
        "exist in the repository to auto-resolve from."
    )


# ---------------------------------------------------------------------------
# 1. Repositories Handlers
# ---------------------------------------------------------------------------

def repos_create_for_authenticated_user(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    user = sandbox._gh.get_user()
    if not hasattr(user, "create_repo"):
        user = sandbox._gh.get_user(sandbox._owner)
    return user.create_repo(
        name=params["name"],
        description=params.get("description", ""),
        private=params.get("private", True),
        auto_init=params.get("auto_init", True),
        has_issues=params.get("has_issues", True),
        has_wiki=params.get("has_wiki", True),
        has_projects=params.get("has_projects", True),
    )


def repos_get(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params)


def repos_update(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    r = resolve_repo(sandbox, params)
    edit_kwargs = {}
    for field in [
        "name", "description", "homepage", "private",
        "has_issues", "has_wiki", "has_projects",
        "default_branch", "archived", "allow_squash_merge",
        "allow_merge_commit", "allow_rebase_merge",
        "allow_auto_merge", "delete_branch_on_merge",
    ]:
        if field in params:
            edit_kwargs[field] = params[field]
    r.edit(**edit_kwargs)
    return r


def repos_delete(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).delete()


def repos_list_for_authenticated_user(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    owner = sandbox._owner
    return list(sandbox._gh.get_user(owner).get_repos())


def repos_list_collaborators(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return list(resolve_repo(sandbox, params).get_collaborators())


def repos_add_collaborator(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).add_to_collaborators(
        params["username"], permission=params.get("permission", "push")
    )


def repos_remove_collaborator(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).remove_from_collaborators(params["username"])


def repos_get_all_topics(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).get_topics()


def repos_replace_all_topics(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).replace_topics(params.get("names", []))


# ---------------------------------------------------------------------------
# 2. Branches and Git References Handlers
# ---------------------------------------------------------------------------

def repos_list_branches(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return list(resolve_repo(sandbox, params).get_branches())


def repos_get_branch(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    branch_name = params["branch"]
    try:
        return repo_obj.get_branch(branch_name)
    except Exception as exc:
        if branch_name in ("main", "master"):
            try:
                repo_obj.create_file(
                    path="README.md",
                    message="chore: initial repository commit",
                    content=f"# {repo_obj.name}\n",
                    branch=branch_name,
                )
                return repo_obj.get_branch(branch_name)
            except Exception:
                pass
        raise exc


def repos_rename_branch(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).rename_branch(params["branch"], params["new_name"])


def repos_merge(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).merge(
        params["base"],
        params["head"],
        commit_message=params.get("commit_message", ""),
    )


def repos_compare_commits(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).compare(params["base"], params["head"])


def _extract_git_ref(params: Dict[str, Any]) -> Optional[str]:
    return (
        params.get("ref")
        or params.get("git-ref-only")
        or params.get("git_ref_only")
        or params.get("commit-ref")
        or params.get("commit_ref")
        or params.get("branch")
        or params.get("ref_name")
    )


def git_create_ref(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    ref = _extract_git_ref(params)
    if not ref:
        raise ValueError("Missing required parameter 'ref' for git/create-ref")
    if not ref.startswith("refs/"):
        if ref.startswith("heads/") or ref.startswith("tags/"):
            ref = f"refs/{ref}"
        else:
            ref = f"refs/heads/{ref}"
    repo_obj = resolve_repo(sandbox, params)
    sha = params.get("sha")
    # GitHub requires a full 40-char hex SHA.  The LLM sometimes fabricates
    # short or invalid hashes, so we validate and discard bad values.
    if sha and not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        sha = None  # force auto-resolve from base branch
    if not sha:
        base_branch = params.get("base") or getattr(repo_obj, "default_branch", "main") or "main"
        try:
            sha = repo_obj.get_branch(base_branch).commit.sha
        except Exception:
            try:
                sha = repo_obj.get_git_ref(f"heads/{base_branch}").object.sha
            except Exception:
                # If repository is empty with no commits, seed initial commit on base_branch
                try:
                    logger.warning("Repository '%s' has no commits on '%s' — initializing README", repo_obj.name, base_branch)
                    repo_obj.create_file(
                        path="README.md",
                        message="chore: initial repository commit",
                        content=f"# {repo_obj.name}\n",
                        branch=base_branch,
                    )
                    sha = repo_obj.get_branch(base_branch).commit.sha
                except Exception as init_exc:
                    logger.warning("Could not auto-seed initial commit on '%s': %s", repo_obj.name, init_exc)
    return repo_obj.create_git_ref(ref=ref, sha=sha)


def git_delete_ref(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    ref = _extract_git_ref(params)
    if not ref:
        raise ValueError("Missing required parameter 'ref' for git/delete-ref")
    if ref.startswith("refs/"):
        ref = ref[5:]
    elif not ref.startswith("heads/") and not ref.startswith("tags/"):
        ref = f"heads/{ref}"
    return resolve_repo(sandbox, params).get_git_ref(ref).delete()


def git_get_ref(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    ref = _extract_git_ref(params)
    if not ref:
        raise ValueError("Missing required parameter 'ref' for git/get-ref")
    if ref.startswith("refs/"):
        ref = ref[5:]
    elif not ref.startswith("heads/") and not ref.startswith("tags/"):
        ref = f"heads/{ref}"
    return resolve_repo(sandbox, params).get_git_ref(ref)


def git_list_matching_refs(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    ref = _extract_git_ref(params) or ""
    return list(resolve_repo(sandbox, params).get_git_matching_refs(ref))


def git_get_tree(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).get_git_tree(
        params["tree_sha"], recursive=params.get("recursive")
    )


def git_get_blob(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).get_git_blob(params["file_sha"])


# ---------------------------------------------------------------------------
# 3. Contents & Commits Handlers
# ---------------------------------------------------------------------------

def repos_get_content(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).get_contents(
        params["path"], ref=params.get("ref")
    )


def repos_create_or_update_file_contents(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    r = resolve_repo(sandbox, params)
    path = params["path"]
    message = params.get("message", "Update file")
    content = params.get("content", "")
    branch = params.get("branch") or getattr(r, "default_branch", "main")
    sha = params.get("sha")

    if sha:
        try:
            return r.update_file(path, message, content, sha, branch=branch)
        except TypeError:
            return r.update_file(path, message, content, sha)

    try:
        return r.create_file(path, message, content, branch=branch)
    except Exception as exc:
        err_str = str(getattr(exc, "data", exc))
        # If file already exists and needs sha:
        if "sha" in err_str.lower() or "422" in err_str:
            try:
                existing = r.get_contents(path, ref=branch)
                try:
                    return r.update_file(path, message, content, existing.sha, branch=branch)
                except TypeError:
                    return r.update_file(path, message, content, existing.sha)
            except Exception:
                pass
        # If branch not found (404), create branch and retry
        if "branch" in err_str.lower() or "404" in err_str:
            if branch and branch not in ("main", "master"):
                try:
                    base_sha = r.get_branch(r.default_branch).commit.sha
                    r.create_git_ref(ref=f"refs/heads/{branch}", sha=base_sha)
                    return r.create_file(path, message, content, branch=branch)
                except Exception:
                    pass
        raise exc


def repos_delete_file(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).delete_file(
        params["path"],
        params.get("message", "Delete file"),
        params["sha"],
        branch=params.get("branch"),
    )


def repos_get_readme(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).get_readme()


def repos_list_commits(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return list(resolve_repo(sandbox, params).get_commits(
        sha=params.get("sha"), path=params.get("path")
    ))


def repos_get_commit(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).get_commit(params["commit_sha"])


# ---------------------------------------------------------------------------
# 4. Pull Requests Handlers
# ---------------------------------------------------------------------------

def pulls_create(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    head = params["head"]
    base = params["base"]

    def _create_pr():
        body = params.get("body") or params.get("description", "")
        return repo_obj.create_pull(
            title=params.get("title", ""),
            body=body,
            head=head,
            base=base,
            draft=params.get("draft", False),
            maintainer_can_modify=params.get("maintainer_can_modify", True),
        )

    try:
        return _create_pr()
    except Exception as exc:
        # GitHub returns 422 "No commits between X and Y" when the head branch
        # points to the same commit as base (freshly created branch).
        # Auto-seed a commit on the head branch so the PR can be created.
        err_str = str(getattr(exc, "data", exc))
        if "No commits between" in err_str or "no commits between" in err_str.lower():
            logger.warning(
                "No commits between '%s' and '%s' — seeding a commit on '%s'",
                base, head, head,
            )
            seed_file_path = f".eval-seed-{head.replace('/', '-')}.md"
            seed_content = f"# Evaluation seed commit\n\nBranch: {head}\nTimestamp: {time.time()}\n"
            try:
                # Check if seed file already exists on the branch
                existing = None
                try:
                    existing = repo_obj.get_contents(seed_file_path, ref=head)
                except Exception:
                    pass

                if existing is not None:
                    repo_obj.update_file(
                        path=seed_file_path,
                        message=f"chore: seed commit for branch '{head}'",
                        content=seed_content,
                        sha=existing.sha,
                        branch=head,
                    )
                else:
                    repo_obj.create_file(
                        path=seed_file_path,
                        message=f"chore: seed commit for branch '{head}'",
                        content=seed_content,
                        branch=head,
                    )
            except Exception as seed_exc:
                # Fallback to unique filename if conflict occurs
                try:
                    unique_path = f".eval-seed-{head.replace('/', '-')}-{uuid.uuid4().hex[:8]}.md"
                    repo_obj.create_file(
                        path=unique_path,
                        message=f"chore: seed commit for branch '{head}'",
                        content=seed_content,
                        branch=head,
                    )
                except Exception as final_seed_exc:
                    logger.error("Failed to seed commit on '%s': %s", head, final_seed_exc)
                    raise exc from final_seed_exc
            # Retry PR creation after seeding
            return _create_pr()
        raise


def pulls_get(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    return repo_obj.get_pull(_resolve_pull_number(repo_obj, params))


def pulls_list(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return list(resolve_repo(sandbox, params).get_pulls(
        state=params.get("state", "open"),
        head=params.get("head"),
        base=params.get("base"),
        sort=params.get("sort", "created"),
        direction=params.get("direction", "desc"),
    ))


def pulls_update(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    pr = repo_obj.get_pull(_resolve_pull_number(repo_obj, params))
    edit_kwargs = {}
    for field in ["title", "body", "state", "base", "maintainer_can_modify"]:
        if field in params:
            edit_kwargs[field] = params[field]
    if "description" in params and "body" not in edit_kwargs:
        edit_kwargs["body"] = params["description"]
    pr.edit(**edit_kwargs)
    return pr


def pulls_merge(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    return repo_obj.get_pull(
        _resolve_pull_number(repo_obj, params)
    ).merge(
        commit_title=params.get("commit_title", ""),
        commit_message=params.get("commit_message", ""),
        merge_method=params.get("merge_method", "merge"),
    )


def pulls_list_files(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    return list(repo_obj.get_pull(_resolve_pull_number(repo_obj, params)).get_files())


def pulls_list_commits(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    return list(repo_obj.get_pull(_resolve_pull_number(repo_obj, params)).get_commits())


def pulls_list_review_comments(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    return list(repo_obj.get_pull(_resolve_pull_number(repo_obj, params)).get_review_comments())


# ---------------------------------------------------------------------------
# 5. Issues Handlers
# ---------------------------------------------------------------------------

def issues_create(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    r = resolve_repo(sandbox, params)
    create_kwargs: Dict[str, Any] = {"title": params["title"]}
    if "body" in params:
        create_kwargs["body"] = params["body"]
    elif "description" in params:
        create_kwargs["body"] = params["description"]
    if "labels" in params:
        create_kwargs["labels"] = params["labels"]
    if "assignees" in params:
        create_kwargs["assignees"] = params["assignees"]
    if "milestone" in params:
        create_kwargs["milestone"] = params["milestone"]
    return r.create_issue(**create_kwargs)


def issues_get(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    return repo_obj.get_issue(_resolve_issue_number(repo_obj, params))


def issues_list(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return list(resolve_repo(sandbox, params).get_issues(
        state=params.get("state", "open"),
        labels=params.get("labels", []),
        assignee=params.get("assignee"),
        milestone=params.get("milestone"),
    ))


def issues_update(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    issue = repo_obj.get_issue(_resolve_issue_number(repo_obj, params))
    edit_kwargs = {}
    for field in ["title", "body", "state", "labels", "assignees", "milestone"]:
        if field in params:
            edit_kwargs[field] = params[field]
    if "description" in params and "body" not in edit_kwargs:
        edit_kwargs["body"] = params["description"]
    issue.edit(**edit_kwargs)
    return issue


def issues_create_comment(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    return repo_obj.get_issue(
        _resolve_issue_number(repo_obj, params)
    ).create_comment(params["body"])


def issues_list_comments(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    return list(repo_obj.get_issue(_resolve_issue_number(repo_obj, params)).get_comments())


def issues_lock(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    repo_obj = resolve_repo(sandbox, params)
    return repo_obj.get_issue(
        _resolve_issue_number(repo_obj, params)
    ).lock(params.get("lock_reason", "off-topic"))


# ---------------------------------------------------------------------------
# 6. Actions Handlers
# ---------------------------------------------------------------------------

def actions_list_repo_workflows(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return list(resolve_repo(sandbox, params).get_workflows())


def actions_list_workflow_runs_for_repo(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return list(resolve_repo(sandbox, params).get_workflow_runs())


def actions_get_workflow_run(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).get_workflow_run(params["run_id"])


def actions_create_workflow_dispatch(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).get_workflow(
        params["workflow_id"]
    ).create_dispatch(
        ref=params.get("ref", "main"),
        inputs=params.get("inputs", {}),
    )


def actions_cancel_workflow_run(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).get_workflow_run(params["run_id"]).cancel()


def actions_re_run_workflow(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).get_workflow_run(params["run_id"]).rerun()


def actions_re_run_workflow_failed_jobs(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).get_workflow_run(params["run_id"]).rerun_failed_jobs()


def actions_download_workflow_run_logs(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    params = _get_params(parameters, kwargs)
    return resolve_repo(sandbox, params).get_workflow_run(params["run_id"]).logs_url


# ---------------------------------------------------------------------------
# 7. Users Handlers
# ---------------------------------------------------------------------------

def users_get_authenticated(sandbox: Any, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    return sandbox._gh.get_user()


# ---------------------------------------------------------------------------
# Execution Registry (51 Operations)
# ---------------------------------------------------------------------------

_CANONICAL_OPERATIONS: Dict[str, Callable[..., Any]] = {
    # repos
    "repos/create-for-authenticated-user": repos_create_for_authenticated_user,
    "repos/get": repos_get,
    "repos/update": repos_update,
    "repos/delete": repos_delete,
    "repos/list-for-authenticated-user": repos_list_for_authenticated_user,
    "repos/list-collaborators": repos_list_collaborators,
    "repos/add-collaborator": repos_add_collaborator,
    "repos/remove-collaborator": repos_remove_collaborator,
    "repos/get-all-topics": repos_get_all_topics,
    "repos/replace-all-topics": repos_replace_all_topics,

    # branches / git refs
    "repos/list-branches": repos_list_branches,
    "repos/get-branch": repos_get_branch,
    "repos/rename-branch": repos_rename_branch,
    "repos/merge": repos_merge,
    "repos/compare-commits": repos_compare_commits,
    "git/create-ref": git_create_ref,
    "git/delete-ref": git_delete_ref,
    "git/get-ref": git_get_ref,
    "git/list-matching-refs": git_list_matching_refs,
    "git/get-tree": git_get_tree,
    "git/get-blob": git_get_blob,

    # contents / commits
    "repos/get-content": repos_get_content,
    "repos/create-or-update-file-contents": repos_create_or_update_file_contents,
    "repos/delete-file": repos_delete_file,
    "repos/get-readme": repos_get_readme,
    "repos/list-commits": repos_list_commits,
    "repos/get-commit": repos_get_commit,

    # pull requests
    "pulls/create": pulls_create,
    "pulls/get": pulls_get,
    "pulls/list": pulls_list,
    "pulls/update": pulls_update,
    "pulls/merge": pulls_merge,
    "pulls/list-files": pulls_list_files,
    "pulls/list-commits": pulls_list_commits,
    "pulls/list-review-comments": pulls_list_review_comments,

    # issues
    "issues/create": issues_create,
    "issues/get": issues_get,
    "issues/list": issues_list,
    "issues/update": issues_update,
    "issues/create-comment": issues_create_comment,
    "issues/list-comments": issues_list_comments,
    "issues/lock": issues_lock,

    # actions
    "actions/list-repo-workflows": actions_list_repo_workflows,
    "actions/list-workflow-runs-for-repo": actions_list_workflow_runs_for_repo,
    "actions/get-workflow-run": actions_get_workflow_run,
    "actions/create-workflow-dispatch": actions_create_workflow_dispatch,
    "actions/cancel-workflow-run": actions_cancel_workflow_run,
    "actions/re-run-workflow": actions_re_run_workflow,
    "actions/re-run-workflow-failed-jobs": actions_re_run_workflow_failed_jobs,
    "actions/download-workflow-run-logs": actions_download_workflow_run_logs,

    # users
    "users/get-authenticated": users_get_authenticated,
}

OPERATIONS: Dict[str, Callable[..., Any]] = {}

for op_id, handler in _CANONICAL_OPERATIONS.items():
    OPERATIONS[op_id] = handler
    tool_format = op_id.replace("/", "__").replace("-", "_")
    OPERATIONS[tool_format] = handler
    snake_format = op_id.replace("/", "_").replace("-", "_")
    OPERATIONS[snake_format] = handler


# ---------------------------------------------------------------------------
# OpenAPI Parameter Components Definition Map
# ---------------------------------------------------------------------------

_PARAM_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "#/components/parameters/owner": {
        "name": "owner",
        "description": "The account owner of the repository. The name is not case sensitive.",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
    },
    "#/components/parameters/repo": {
        "name": "repo",
        "description": "The name of the repository without the `.git` extension. The name is not case sensitive.",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
    },
    "#/components/parameters/issue-number": {
        "name": "issue_number",
        "description": "The number that identifies the issue.",
        "in": "path",
        "required": True,
        "schema": {"type": "integer"},
    },
    "#/components/parameters/pull-number": {
        "name": "pull_number",
        "description": "The number that identifies the pull request.",
        "in": "path",
        "required": True,
        "schema": {"type": "integer"},
    },
    "#/components/parameters/branch": {
        "name": "branch",
        "description": "The name of the branch.",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
        "x-multi-segment": True,
    },
    "#/components/parameters/per-page": {
        "name": "per_page",
        "description": "The number of results per page (max 100).",
        "in": "query",
        "required": False,
        "schema": {"type": "integer", "default": 30},
    },
    "#/components/parameters/page": {
        "name": "page",
        "description": "The page number of the results to fetch.",
        "in": "query",
        "required": False,
        "schema": {"type": "integer", "default": 1},
    },
    "#/components/parameters/username": {
        "name": "username",
        "description": "The handle for the GitHub user account.",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
    },
    "#/components/parameters/commit-sha": {
        "name": "commit_sha",
        "description": "The SHA of the commit.",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
    },
    "#/components/parameters/ref": {
        "name": "ref",
        "description": "The git reference.",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
    },
    "#/components/parameters/git-ref-only": {
        "name": "ref",
        "description": "The Git reference.",
        "in": "path",
        "required": True,
        "example": "heads/feature-a",
        "schema": {"type": "string"},
        "x-multi-segment": True,
    },
    "#/components/parameters/commit-ref": {
        "name": "ref",
        "description": "The commit reference. Can be a commit SHA, branch name, or tag name.",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
        "x-multi-segment": True,
    },
    "#/components/parameters/basehead": {
        "name": "basehead",
        "description": "Base and head to compare: base...head.",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
    },
    "#/components/parameters/file-sha": {
        "name": "file_sha",
        "description": "The SHA of the blob (file).",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
    },
    "#/components/parameters/tree-sha": {
        "name": "tree_sha",
        "description": "The SHA1 value or ref name of the tree.",
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
    },
    "#/components/parameters/run-id": {
        "name": "run_id",
        "description": "The unique identifier of the workflow run.",
        "in": "path",
        "required": True,
        "schema": {"type": "integer"},
    },
    "#/components/parameters/workflow-id": {
        "name": "workflow_id",
        "description": "The ID of the workflow or workflow file name.",
        "in": "path",
        "required": True,
        "schema": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
    },
    "#/components/parameters/since": {
        "name": "since",
        "description": "Only show results that were last updated after the given time (ISO 8601).",
        "in": "query",
        "required": False,
        "schema": {"type": "string", "format": "date-time"},
    },
    "#/components/parameters/since-repo-date": {
        "name": "since",
        "description": "Only show repositories updated after this ISO 8601 timestamp.",
        "in": "query",
        "required": False,
        "schema": {"type": "string"},
    },
    "#/components/parameters/before-repo-date": {
        "name": "before",
        "description": "Only show repositories updated before this ISO 8601 timestamp.",
        "in": "query",
        "required": False,
        "schema": {"type": "string"},
    },
    "#/components/parameters/direction": {
        "name": "direction",
        "description": "The direction to sort the results by.",
        "in": "query",
        "required": False,
        "schema": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
    },
    "#/components/parameters/sort": {
        "name": "sort",
        "description": "The property to sort the results by.",
        "in": "query",
        "required": False,
        "schema": {"type": "string", "enum": ["created", "updated"], "default": "created"},
    },
    "#/components/parameters/labels": {
        "name": "labels",
        "description": "A list of comma separated label names.",
        "in": "query",
        "required": False,
        "schema": {"type": "string"},
    },
    "#/components/parameters/actor": {
        "name": "actor",
        "description": "Returns someone's workflow runs (login name).",
        "in": "query",
        "required": False,
        "schema": {"type": "string"},
    },
    "#/components/parameters/created": {
        "name": "created",
        "description": "Returns workflow runs created within the given date-time range.",
        "in": "query",
        "required": False,
        "schema": {"type": "string", "format": "date-time"},
    },
    "#/components/parameters/event": {
        "name": "event",
        "description": "Returns workflow run triggered by the event specified.",
        "in": "query",
        "required": False,
        "schema": {"type": "string"},
    },
    "#/components/parameters/exclude-pull-requests": {
        "name": "exclude_pull_requests",
        "description": "If true pull requests are omitted from the response.",
        "in": "query",
        "required": False,
        "schema": {"type": "boolean", "default": False},
    },
    "#/components/parameters/workflow-run-branch": {
        "name": "branch",
        "description": "Returns workflow runs associated with a branch.",
        "in": "query",
        "required": False,
        "schema": {"type": "string"},
    },
    "#/components/parameters/workflow-run-check-suite-id": {
        "name": "check_suite_id",
        "description": "Returns workflow runs with the check_suite_id specified.",
        "in": "query",
        "required": False,
        "schema": {"type": "integer", "format": "int64"},
    },
    "#/components/parameters/workflow-run-head-sha": {
        "name": "head_sha",
        "description": "Only returns workflow runs associated with head_sha.",
        "in": "query",
        "required": False,
        "schema": {"type": "string"},
    },
    "#/components/parameters/workflow-run-status": {
        "name": "status",
        "description": "Returns workflow runs with check run status specified.",
        "in": "query",
        "required": False,
        "schema": {
            "type": "string",
            "enum": [
                "completed", "action_required", "cancelled", "failure",
                "neutral", "skipped", "stale", "success", "timed_out",
                "in_progress", "queued", "requested", "waiting", "pending",
            ],
        },
    },
}


def _resolve_param_ref(ref_str: str) -> Dict[str, Any]:
    """Resolve a parameter $ref stub into its complete dictionary."""
    if ref_str in _PARAM_COMPONENTS:
        return copy.deepcopy(_PARAM_COMPONENTS[ref_str])
    param_name = ref_str.split("/")[-1].replace("-", "_")
    return {"name": param_name, "in": "path", "required": True, "schema": {"type": "string"}}


# ---------------------------------------------------------------------------
# Operation Aliases & Resolution Map
# ---------------------------------------------------------------------------

OPERATION_ALIASES: Dict[str, str] = {
    # Pull requests
    "create_pull_request": "pulls/create",
    "create_pull": "pulls/create",
    "create_pr": "pulls/create",
    "get_pull_request": "pulls/get",
    "get_pull": "pulls/get",
    "get_pr": "pulls/get",
    "list_pull_requests": "pulls/list",
    "list_pulls": "pulls/list",
    "list_prs": "pulls/list",
    "update_pull_request": "pulls/update",
    "update_pull": "pulls/update",
    "update_pr": "pulls/update",
    "merge_pull_request": "pulls/merge",
    "merge_pull": "pulls/merge",
    "merge_pr": "pulls/merge",
    "list_pull_files": "pulls/list-files",
    "list_pull_request_files": "pulls/list-files",
    "list_pull_commits": "pulls/list-commits",
    "list_pull_request_commits": "pulls/list-commits",
    "list_pull_review_comments": "pulls/list-review-comments",
    "list_review_comments": "pulls/list-review-comments",

    # Issues
    "create_issue": "issues/create",
    "get_issue": "issues/get",
    "list_issues": "issues/list",
    "update_issue": "issues/update",
    "create_issue_comment": "issues/create-comment",
    "create_comment": "issues/create-comment",
    "list_issue_comments": "issues/list-comments",
    "list_comments": "issues/list-comments",
    "lock_issue": "issues/lock",

    # Repositories
    "create_repository": "repos/create-for-authenticated-user",
    "create_repo": "repos/create-for-authenticated-user",
    "get_repository": "repos/get",
    "get_repo": "repos/get",
    "update_repository": "repos/update",
    "update_repo": "repos/update",
    "delete_repository": "repos/delete",
    "delete_repo": "repos/delete",
    "list_repositories": "repos/list-for-authenticated-user",
    "list_repos": "repos/list-for-authenticated-user",
    "list_collaborators": "repos/list-collaborators",
    "add_collaborator": "repos/add-collaborator",
    "remove_collaborator": "repos/remove-collaborator",
    "get_all_topics": "repos/get-all-topics",
    "get_topics": "repos/get-all-topics",
    "replace_all_topics": "repos/replace-all-topics",
    "replace_topics": "repos/replace-all-topics",

    # Branches & Git Refs
    "list_branches": "repos/list-branches",
    "get_branch": "repos/get-branch",
    "rename_branch": "repos/rename-branch",
    "merge_branch": "repos/merge",
    "merge": "repos/merge",
    "compare_commits": "repos/compare-commits",
    "create_branch": "git/create-ref",
    "create_git_ref": "git/create-ref",
    "create_ref": "git/create-ref",
    "delete_branch": "git/delete-ref",
    "delete_git_ref": "git/delete-ref",
    "delete_ref": "git/delete-ref",
    "get_git_ref": "git/get-ref",
    "get_ref": "git/get-ref",
    "list_matching_refs": "git/list-matching-refs",
    "get_tree": "git/get-tree",
    "get_blob": "git/get-blob",

    # Repository Contents
    "get_content": "repos/get-content",
    "get_contents": "repos/get-content",
    "get_file": "repos/get-content",
    "create_or_update_file_contents": "repos/create-or-update-file-contents",
    "create_or_update_file": "repos/create-or-update-file-contents",
    "create_file": "repos/create-or-update-file-contents",
    "update_file": "repos/create-or-update-file-contents",
    "delete_file": "repos/delete-file",
    "get_readme": "repos/get-readme",
    "list_commits": "repos/list-commits",
    "get_commit": "repos/get-commit",

    # Actions
    "list_repo_workflows": "actions/list-repo-workflows",
    "list_workflows": "actions/list-repo-workflows",
    "list_workflow_runs_for_repo": "actions/list-workflow-runs-for-repo",
    "list_workflow_runs": "actions/list-workflow-runs-for-repo",
    "get_workflow_run": "actions/get-workflow-run",
    "create_workflow_dispatch": "actions/create-workflow-dispatch",
    "cancel_workflow_run": "actions/cancel-workflow-run",
    "rerun_workflow": "actions/re-run-workflow",
    "re_run_workflow": "actions/re-run-workflow",
    "rerun_workflow_failed_jobs": "actions/re-run-workflow-failed-jobs",
    "re_run_workflow_failed_jobs": "actions/re-run-workflow-failed-jobs",
    "download_workflow_run_logs": "actions/download-workflow-run-logs",
    "download_workflow_logs": "actions/download-workflow-run-logs",

    # Users
    "get_authenticated_user": "users/get-authenticated",
    "get_user": "users/get-authenticated",
}


# ---------------------------------------------------------------------------
# Operation Definitions & Schemas Store
# ---------------------------------------------------------------------------

OPERATION_DEFINITIONS: Dict[str, Dict[str, Any]] = {}


def _init_operation_definitions() -> None:
    """
    Initialize and cache all 51 operation definitions, schemas, and starter requests
    from schemas/operation_catalog.json upon module loading.
    """
    this_dir = Path(__file__).resolve().parent
    catalog_path = this_dir.parent / "schemas" / "operation_catalog.json"
    if not catalog_path.exists():
        # Fallback path if loaded from different cwd
        catalog_path = Path("schemas/operation_catalog.json").resolve()
        if not catalog_path.exists():
            return

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    for op in catalog.get("operations", []):
        op_id = op["operation_id"]

        # Resolve parameters
        resolved_parameters: List[Dict[str, Any]] = []
        for raw_param in op.get("parameters", []):
            if isinstance(raw_param, dict) and "$ref" in raw_param:
                resolved_parameters.append(_resolve_param_ref(raw_param["$ref"]))
            elif isinstance(raw_param, dict):
                resolved_parameters.append(copy.deepcopy(raw_param))

        path_params = [p for p in resolved_parameters if p.get("in") == "path"]
        query_params = [p for p in resolved_parameters if p.get("in") == "query"]

        # Resolve input schema
        rb = op.get("request_body")
        if rb and "content" in rb and "application/json" in rb["content"]:
            input_schema = copy.deepcopy(rb["content"]["application/json"].get("schema", {}))
        else:
            props = {
                p["name"]: {**p.get("schema", {}), "description": p.get("description", ""), "in": p.get("in", "")}
                for p in resolved_parameters if "name" in p
            }
            reqs = [p["name"] for p in resolved_parameters if p.get("required") and "name" in p]
            input_schema = {"type": "object", "properties": props, "required": reqs}

        input_schema["parameters"] = copy.deepcopy(resolved_parameters)
        input_schema["path_parameters"] = copy.deepcopy(path_params)
        input_schema["query_parameters"] = copy.deepcopy(query_params)

        # Resolve starter request
        starter_request: Dict[str, Any] = {}
        if rb:
            content = rb.get("content", {})
            app_json = content.get("application/json", {})
            examples = app_json.get("examples", {})
            if "default" in examples and "value" in examples["default"]:
                val = examples["default"]["value"]
                if isinstance(val, dict):
                    starter_request = copy.deepcopy(val)

            if not starter_request and "example" in app_json and isinstance(app_json["example"], dict):
                starter_request = copy.deepcopy(app_json["example"])

            # If no example value, synthesize starter fields for required properties
            if not starter_request:
                properties = input_schema.get("properties", {})
                required_fields = input_schema.get("required", [])
                for field in required_fields:
                    prop = properties.get(field, {})
                    if "example" in prop:
                        starter_request[field] = copy.deepcopy(prop["example"])
                    elif "default" in prop:
                        starter_request[field] = copy.deepcopy(prop["default"])
                    else:
                        starter_request[field] = f"sample_{field}"

        definition = {
            "operation_id": op_id,
            "name": op.get("name", op_id.replace("/", "_")),
            "method": op.get("method", "GET").upper(),
            "path": op.get("path", ""),
            "summary": op.get("summary", ""),
            "description": op.get("description", ""),
            "risk": op.get("risk", "medium"),
            "parameters": resolved_parameters,
            "path_parameters": path_params,
            "query_parameters": query_params,
            "input_schema": input_schema,
            "starter_request": starter_request,
        }

        OPERATION_DEFINITIONS[op_id] = definition
        # Register tool and snake aliases
        OPERATION_DEFINITIONS[op_id.replace("/", "__").replace("-", "_")] = definition
        OPERATION_DEFINITIONS[op_id.replace("/", "_").replace("-", "_")] = definition


# Run initialization on import
_init_operation_definitions()


def _resolve_operation_id(name: str) -> str:
    """Resolve an operation name to its canonical operation_id."""
    name_lower = name.lower()

    if name in OPERATION_DEFINITIONS:
        return OPERATION_DEFINITIONS[name]["operation_id"]

    if name_lower in OPERATION_ALIASES:
        return OPERATION_ALIASES[name_lower]

    for op_id in _CANONICAL_OPERATIONS:
        if name_lower == op_id.lower():
            return op_id
        if name_lower == op_id.replace("/", "__").replace("-", "_").lower():
            return op_id
        if name_lower == op_id.replace("/", "_").replace("-", "_").lower():
            return op_id

    normalized = name_lower.replace("-", "_").replace("/", "_").replace("__", "_")
    for op_id in _CANONICAL_OPERATIONS:
        if normalized == op_id.replace("-", "_").replace("/", "_").lower():
            return op_id

    raise ValueError(f"Unknown GitHub operation: {name}")


def get_operation_definition(name: str) -> Dict[str, Any]:
    """
    Retrieve the initialized definition, schema, and starter request for an operation.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Unknown GitHub operation: {name}")

    canonical_id = _resolve_operation_id(name)
    if canonical_id not in OPERATION_DEFINITIONS:
        raise ValueError(f"Operation schema for '{name}' has not been loaded.")

    return copy.deepcopy(OPERATION_DEFINITIONS[canonical_id])


def list_operations() -> List[str]:
    """Return a sorted list of all available canonical operation IDs."""
    return sorted(list(_CANONICAL_OPERATIONS.keys()))


# ---------------------------------------------------------------------------
# Operation Lookup for execute_operation
# ---------------------------------------------------------------------------

def _lookup_operation(name: str) -> Optional[Tuple[str, Callable[..., Any]]]:
    """
    Resolve an operation name to its (canonical_operation_id, handler) tuple.
    """
    if name in _CANONICAL_OPERATIONS:
        return name, _CANONICAL_OPERATIONS[name]

    if name in OPERATIONS:
        for canon_id, fn in _CANONICAL_OPERATIONS.items():
            if fn == OPERATIONS[name]:
                return canon_id, fn

    normalised = name.replace("__", "/").replace("_", "-")
    if normalised in _CANONICAL_OPERATIONS:
        return normalised, _CANONICAL_OPERATIONS[normalised]

    try:
        canon_id = _resolve_operation_id(name)
        if canon_id in _CANONICAL_OPERATIONS:
            return canon_id, _CANONICAL_OPERATIONS[canon_id]
    except ValueError:
        pass

    return None


# ---------------------------------------------------------------------------
# Central Dispatcher: execute_operation
# ---------------------------------------------------------------------------

def execute_operation(
    sandbox: Any,
    name: str,
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Execute a GitHub operation through the sandbox audit and rate-limiting wrapper.

    Parameters
    ----------
    sandbox : GitHubSandbox
        Initialised sandbox instance.
    name : str
        The operation identifier (e.g. 'issues/create', 'issues__create', 'issues_create').
    parameters : dict, optional
        Arguments dictionary for the operation.
    **kwargs
        Additional arguments passed as keyword parameters.

    Returns
    -------
    dict
        The structured sandbox execute() result dictionary containing:
        {'success': bool, 'operation': str, ...}
    """
    params = _get_params(parameters, kwargs)
    resolved = _lookup_operation(name)

    if resolved is None:
        return {
            "success": False,
            "operation": name,
            "error_type": "UNKNOWN_OPERATION",
            "message": (
                f"Operation '{name}' is in the catalog but has no "
                "execution handler in single_prompt_agent.py."
            ),
        }

    # Sanitize parameters: strip unresolved template references the LLM
    # may have emitted (e.g. "${issues/create.number}", "$$.0.number").
    params = _sanitize_template_refs(params)

    canonical_id, handler = resolved
    return sandbox.execute(canonical_id, lambda: handler(sandbox, params))


# ---------------------------------------------------------------------------
# Template-reference sanitizer
# ---------------------------------------------------------------------------

# Patterns the LLM may emit as placeholder references to prior step outputs.
_TEMPLATE_REF_PATTERN = re.compile(
    r"^(<|\{)?\$"    # starts with $, <$, or {$
    r"("
    r"\{?[^}>]+\}?"  # ${issues/create.number} or <$.issues/create.number>
    r"|"
    r"\$\.[0-9]"     # $$.0.number
    r"|"
    r"\[[0-9]+\]"    # $[0].number
    r")"
    r"|^\s*\{\{.*\}\}\s*$"  # {{operations.issues-create.outputs.number}}
)


def _sanitize_template_refs(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect and remove unresolved template-style references in parameter values.

    The LLM sometimes generates JSONPath/template references like
    ``${issues/create.number}`` or ``$$.0.number`` expecting them to be
    dynamically resolved, but there is no template engine in the pipeline.
    These raw strings cause downstream AssertionError/TypeError when the
    handler expects an int or other concrete type.

    This function removes such values so that the operation handlers can
    fall back to auto-resolution (e.g. looking up the latest issue).
    """
    cleaned = {}
    for key, value in params.items():
        if isinstance(value, str) and _TEMPLATE_REF_PATTERN.match(value):
            logger.warning(
                "Stripped unresolved template reference for param '%s': %s",
                key, value,
            )
            # Don't include the key — let the handler auto-resolve
            continue
        cleaned[key] = value
    return cleaned
