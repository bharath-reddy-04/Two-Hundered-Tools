#!/usr/bin/env python3
"""
task_runner.py
==============
Production task runner and evaluation harness for the 200-Tools project.

Architecture:
    tasks.json
        ↓
    task_runner.py
        ↓
    single_prompt_agent.py (LLM tool selection)
        ↓
    github_sandbox.py (tool execution against sandbox)
        ↓
    verifier.py (independent end-state verification)
        ↓
    results/results.json + Console Summary

Key Features:
- Sequential execution across all tasks in tasks.json
- Independent verification: does NOT trust the LLM's self-reported "success"
- Captures agent response, tool calls, execution metadata, and verifier checks
- Safe isolation: requires sandbox configuration, prevents production writes
- Full CLI support (--task-id, --limit, --start, --retry, --repo, --dry-run)
- Detailed results saved to results/results.json and logs to results/task_runner.log
- Formatted console progress and execution summary
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load .env on import so configuration is immediately available
load_dotenv()

# ---------------------------------------------------------------------------
# Path setup & Module imports
# ---------------------------------------------------------------------------

_this_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _this_dir if (_this_dir / "tasks.json").exists() else _this_dir.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
HARNESS_DIR = PROJECT_ROOT / "harness"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

DEFAULT_TASKS_FILE = PROJECT_ROOT / "tasks.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_RESULTS_FILE = DEFAULT_RESULTS_DIR / "results.json"
DEFAULT_LOG_FILE = DEFAULT_RESULTS_DIR / "task_runner.log"

RUNNER_VERSION = "1.0.0"

# Module-level imports from harness for easier patching in tests
try:
    from verifier import verify_state  # type: ignore
except ImportError:
    verify_state = None  # type: ignore

try:
    from github_operations import execute_operation, OPERATIONS  # type: ignore
except ImportError:
    try:
        from harness.github_operations import execute_operation, OPERATIONS  # type: ignore
    except ImportError:
        execute_operation = None  # type: ignore
        OPERATIONS = {}  # type: ignore


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def setup_logging(log_file: Optional[Path] = None, verbose: bool = False) -> logging.Logger:
    """Configure file and console logging."""
    logger = logging.getLogger("task_runner")
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        h.close()
    logger.handlers.clear()

    # Formatters
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Environment & Sandbox validation
# ---------------------------------------------------------------------------

def check_environment(require_sandbox: bool = True) -> Dict[str, Any]:
    """
    Validate that required sandbox and LLM credentials are configured.
    Stops execution with a clear error if required sandbox config is missing.
    """
    github_token = os.getenv("GITHUB_TOKEN", "")
    github_owner = os.getenv("GITHUB_OWNER", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    missing: List[str] = []
    if require_sandbox:
        if not github_token:
            missing.append("GITHUB_TOKEN (GitHub personal access token)")
        if not github_owner:
            missing.append("GITHUB_OWNER (Sandbox account / org owner)")
        if not gemini_key and not openai_key:
            missing.append("GEMINI_API_KEY (or OPENAI_API_KEY for LLM access)")

    if missing:
        print("\n" + "=" * 70, file=sys.stderr)
        print("ERROR: Missing Required Sandbox / LLM Configuration", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        for item in missing:
            print(f"  • {item}", file=sys.stderr)
        print("\nPlease add the missing variables to your .env file or environment.", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    # Obfuscate token for display
    token_display = (
        f"{github_token[:4]}...{github_token[-3:]}"
        if len(github_token) > 8
        else ("configured" if github_token else "missing")
    )
    llm_provider = "Google Gemini" if gemini_key else ("OpenAI" if openai_key else "None")

    return {
        "github_owner": github_owner,
        "github_token_preview": token_display,
        "llm_provider": llm_provider,
        "model": os.getenv("GEMINI_MODEL") or os.getenv("OPENAI_MODEL") or "gemini-3.6-flash",
    }


def print_env_check(
    env_info: Dict[str, Any],
    target_repo: str,
    total_tasks: int,
    agent_type: str = "single_prompt",
    disclosure_mode: str = "all_loaded",
) -> None:
    """Print a clean environment check summary before starting."""
    print("=" * 70)
    print("200-TOOLS AGENT EVALUATION RUNNER")
    print(f"Runner Version:      v{RUNNER_VERSION}")
    print(f"Agent Architecture:  {agent_type}")
    print(f"Disclosure Mode:     {disclosure_mode}")
    print(f"LLM Provider:        {env_info['llm_provider']}")
    print(f"Model:               {env_info['model']}")
    print(f"GitHub Sandbox User: {env_info['github_owner']}")
    print(f"Target Repository:   {target_repo}")
    print(f"Tasks to Execute:    {total_tasks}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Task loading & validation
# ---------------------------------------------------------------------------

def load_tasks(tasks_path: Path) -> List[Dict[str, Any]]:
    """
    Load tasks from tasks.json.
    Supports both a top-level list of tasks and an object with a 'tasks' list.
    """
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found at: {tasks_path}")

    try:
        with tasks_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in tasks file {tasks_path}: {exc}") from exc

    if isinstance(data, list):
        tasks = data
    elif isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
    else:
        raise ValueError(
            f"Invalid structure in {tasks_path}: root must be a list of tasks or an object with a 'tasks' array."
        )

    return tasks


def validate_task(task: Dict[str, Any], index: int) -> None:
    """Validate that a single task has a unique task_id and an instruction/prompt field."""
    if not isinstance(task, dict):
        raise ValueError(f"Task #{index} must be a dictionary.")

    task_id = task.get("task_id")
    if not task_id or not isinstance(task_id, str):
        raise ValueError(f"Task #{index} is missing a valid 'task_id'.")

    instruction = task.get("instruction") or task.get("prompt")
    if not instruction or not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Task '{task_id}' is missing a non-empty 'instruction' or 'prompt' field.")


# ---------------------------------------------------------------------------
# Agent execution wrapper
# ---------------------------------------------------------------------------

def execute_agent(
    task_instruction: str,
    repo: str,
    sandbox=None,
    model: Optional[str] = None,
    mock: bool = False,
    agent_type: str = "single_prompt",
    task: Optional[Dict[str, Any]] = None,
    disclosure_mode: str = "all_loaded",
) -> Dict[str, Any]:
    """
    Execute the agent on a single task instruction.
    Supports both 'single_prompt' (baseline) and 'orchestrator' (LangGraph control loop).
    """
    if mock:
        # Mock mode for dry-run or unit testing
        return {
            "success": True,
            "task": task_instruction,
            "model": model or "mock-model",
            "tool_calls": [{"operation_id": "issues/create", "arguments": {"title": "Mock Issue"}}],
            "results": [{"success": True, "operation": "issues/create"}],
            "final_response": "Mock execution completed.",
        }

    if agent_type in ("orchestrator", "orchestration"):
        # Import dynamically so orchestrator is loaded lazily
        try:
            from orchestrator.agent import OrchestrationAgent
            from orchestrator.config import OrchestratorConfig
        except ImportError:
            from agent import OrchestrationAgent  # type: ignore
            from config import OrchestratorConfig  # type: ignore

        cfg = OrchestratorConfig(default_model=model) if model else None
        agent = OrchestrationAgent(sandbox=sandbox, config=cfg)
        raw_result = agent.run(
            task=task or {"task_id": "unknown", "prompt": task_instruction, "expected_state": {}},
            objective=task_instruction,
            disclosure_mode=disclosure_mode,
        )

        tool_calls = []
        plan = raw_result.get("plan")
        if plan and hasattr(plan, "operations"):
            for op in plan.operations:
                tool_calls.append({"operation_id": op.operation_id, "arguments": getattr(op, "parameters", {})})

        error_msg = None
        if raw_result.get("errors"):
            error_msg = "; ".join(
                e.get("message", "") if isinstance(e, dict) else getattr(e, "message", str(e))
                for e in raw_result["errors"]
            )

        return {
            "success": raw_result.get("status") == "verified" or len(raw_result.get("execution_results", [])) > 0,
            "task": task_instruction,
            "model": model or "gemini",
            "tool_calls": tool_calls,
            "results": raw_result.get("execution_results", []),
            "final_response": f"Status: {raw_result.get('status')}, Verified: {raw_result.get('verified')}",
            "error": error_msg,
            "raw_orchestration": raw_result,
        }

    from single_prompt_agent import run_agent  # type: ignore

    return run_agent(
        task=task_instruction,
        repo=repo,
        sandbox=sandbox,
        model=model,
    )


# ---------------------------------------------------------------------------
# Verification wrapper
# ---------------------------------------------------------------------------

def verify_task(
    task: Dict[str, Any],
    default_repo: str,
    agent_result: Dict[str, Any],
    token: Optional[str] = None,
    mock: bool = False,
) -> Tuple[str, bool, Dict[str, Any]]:
    """
    Independently verify the actual sandbox state for a completed task.

    Returns:
        (status, passed, verification_details)
        where status is 'PASS', 'FAIL', 'NO_VERIFIER', or 'ERROR'.
    """
    expected_state = task.get("expected_state")
    if not expected_state:
        return "NO_VERIFIER", False, {
            "passed": False,
            "status": "NO_VERIFIER",
            "message": "Task does not specify an expected_state dictionary for verification.",
        }

    if mock:
        return "PASS", True, {"passed": True, "checks": {"mock": True}, "errors": []}

    target_repo = default_repo
    if "repository" in expected_state:
        repo_name = expected_state["repository"].get("name")
        if repo_name:
            if "/" in repo_name:
                target_repo = repo_name
            elif "/" in default_repo:
                target_repo = f"{default_repo.split('/')[0]}/{repo_name}"
            else:
                target_repo = repo_name

    # If the agent executed a resource creation that returned a created object, extract identifier
    # to facilitate direct, replication-lag-free verification
    effective_expected = json.loads(json.dumps(expected_state))
    for res in agent_result.get("results", []):
        if res.get("success") and res.get("operation") == "issues/create":
            created_issue = res.get("result")
            if created_issue and hasattr(created_issue, "number") and "issue" in effective_expected:
                if not effective_expected["issue"].get("number"):
                    effective_expected["issue"]["number"] = created_issue.number
        elif res.get("success") and res.get("operation") == "pulls/create":
            created_pr = res.get("result")
            if created_pr and hasattr(created_pr, "number") and "pull_request" in effective_expected:
                if not effective_expected["pull_request"].get("number"):
                    effective_expected["pull_request"]["number"] = created_pr.number

    try:
        verification = verify_state(target_repo, effective_expected, token=token)
        passed = bool(verification.get("passed", False))
        status = "PASS" if passed else "FAIL"
        return status, passed, verification
    except Exception as exc:
        return "ERROR", False, {
            "passed": False,
            "status": "ERROR",
            "errors": [f"Verification exception: {type(exc).__name__}: {exc}"],
        }


# ---------------------------------------------------------------------------
# Task execution orchestrator (with optional retry)
# ---------------------------------------------------------------------------

def is_transient_error(error_msg: str) -> bool:
    """Heuristic check for transient network, timeout, or rate-limit errors."""
    if not error_msg:
        return False
    lower = error_msg.lower()
    transient_indicators = [
        "rate limit",
        "429",
        "timeout",
        "connection reset",
        "connection refused",
        "502",
        "503",
        "504",
        "temporary failure",
    ]
    return any(indicator in lower for indicator in transient_indicators)


def run_task(
    task: Dict[str, Any],
    repo: str,
    sandbox=None,
    model: Optional[str] = None,
    max_retries: int = 0,
    mock: bool = False,
    clean_task: bool = True,
    logger: Optional[logging.Logger] = None,
    agent_type: str = "single_prompt",
    disclosure_mode: str = "all_loaded",
) -> Dict[str, Any]:
    """
    Run a single task through agent execution and independent verification.
    Supports retries for transient infrastructure errors.
    """
    task_id = task["task_id"]
    instruction = task.get("instruction") or task.get("prompt", "")
    difficulty = task.get("difficulty", "unknown")

    start_time = time.time()
    attempts = 0
    final_status = "ERROR"
    final_passed = False
    agent_result: Dict[str, Any] = {}
    verification: Dict[str, Any] = {}
    last_error: Optional[str] = None

    while attempts <= max_retries:
        attempts += 1
        agent_error = None

        if logger:
            logger.info("Executing task %s (attempt %d/%d)", task_id, attempts, max_retries + 1)

        # Pre-task resource cleanup to eliminate duplicate state
        if sandbox and not mock and clean_task and hasattr(sandbox, "clean_task_state"):
            try:
                sandbox.clean_task_state(repo_name_or_obj=repo, task=task)
            except Exception as exc:
                if logger:
                    logger.warning("Could not clean task state for %s: %s", task_id, exc)

        # 1. Execute agent
        try:
            agent_result = execute_agent(
                task_instruction=instruction,
                repo=repo,
                sandbox=sandbox,
                model=model,
                mock=mock,
                agent_type=agent_type,
                task=task,
                disclosure_mode=disclosure_mode,
            )
            if not agent_result.get("success"):
                agent_error = agent_result.get("error")
                if not agent_error:
                    failed_ops = []
                    for res in agent_result.get("results", []):
                        if not res.get("success"):
                            op_name = res.get("operation", "tool")
                            err_msg = res.get("message") or res.get("error_type") or res.get("error") or "execution failed"
                            failed_ops.append(f"{op_name}: {err_msg}")
                    if failed_ops:
                        agent_error = "; ".join(failed_ops)
        except Exception as exc:
            agent_error = f"{type(exc).__name__}: {exc}"
            agent_result = {
                "success": False,
                "task": instruction,
                "tool_calls": [],
                "results": [],
                "final_response": "",
                "error": agent_error,
            }

        # 2. Verify state
        status, passed, verif_details = verify_task(
            task=task,
            default_repo=repo,
            agent_result=agent_result,
            token=getattr(sandbox, "_token", None),
            mock=mock,
        )

        final_status = status
        final_passed = passed
        verification = verif_details
        last_error = agent_error or ("; ".join(verification.get("errors", [])) if not passed else None)

        # 3. Check if retry is appropriate
        if passed or status == "NO_VERIFIER":
            break

        # If failure was transient and no side-effects were committed, retry
        tool_calls_made = agent_result.get("tool_calls", [])
        if attempts <= max_retries and is_transient_error(str(last_error)) and len(tool_calls_made) == 0:
            backoff = 2.0 * attempts
            if logger:
                logger.warning("Transient error on %s; retrying in %.1fs: %s", task_id, backoff, last_error)
            time.sleep(backoff)
            continue
        else:
            break

    duration_seconds = round(time.time() - start_time, 2)

    # Classify overall status
    if final_passed:
        final_status = "PASS"
    elif final_status == "NO_VERIFIER":
        final_status = "NO_VERIFIER"
    elif agent_result.get("error") and not verification.get("checks"):
        final_status = "ERROR"
    else:
        final_status = "FAIL"

    return {
        "task_id": task_id,
        "instruction": instruction,
        "difficulty": difficulty,
        "status": final_status,
        "passed": final_passed,
        "agent_response": agent_result.get("final_response", ""),
        "tool_calls": agent_result.get("tool_calls", []),
        "results": agent_result.get("results", []),
        "verification": verification,
        "error": last_error,
        "duration_seconds": duration_seconds,
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# Results persistence & summary
# ---------------------------------------------------------------------------

def save_results(
    output_path: Path,
    run_metadata: Dict[str, Any],
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> None:
    """Save structured evaluation results to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_metadata": run_metadata,
        "summary": summary,
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def print_task_progress(index: int, total: int, result: Dict[str, Any]) -> None:
    """Print live progress output for a single task execution."""
    task_id = result["task_id"]
    difficulty = result.get("difficulty", "unknown").capitalize()
    status = result["status"]
    duration = result["duration_seconds"]
    instruction = result["instruction"]
    short_instruction = (instruction[:75] + "…") if len(instruction) > 75 else instruction
    tool_calls = result.get("tool_calls", [])

    # Status formatting
    status_symbol = {
        "PASS": "✓ PASS",
        "FAIL": "✗ FAIL",
        "ERROR": "! ERROR",
        "NO_VERIFIER": "? NO_VERIFIER",
    }.get(status, status)

    ops_used = [tc.get("operation_id", "?") for tc in tool_calls]
    ops_summary = f"{len(tool_calls)} call(s)" + (f" [{', '.join(ops_used)}]" if ops_used else "")

    print(f"[{index}/{total}] {task_id} ({difficulty})")
    print(f"  Instruction: {short_instruction}")
    print(f"  Agent:       {ops_summary}")
    print(f"  Verifier:    {status_symbol}")
    if result.get("error") and status != "PASS":
        print(f"  Reason:      {result['error']}")
    print(f"  Duration:    {duration:.2f}s\n")


def print_summary(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    """Print final evaluation summary table and failed task diagnostics."""
    print("=" * 45)
    print("TASK RUN SUMMARY")
    print("=" * 45)
    print(f"Total:        {summary['total']}")
    print(f"Passed:       {summary['passed']}")
    print(f"Failed:       {summary['failed']}")
    print(f"Errors:       {summary['errors']}")
    print(f"No verifier:  {summary['no_verifier']}")
    print(f"Success rate: {summary['success_rate_percent']:.2f}%")
    print("=" * 45)

    failed_tasks = [r for r in results if r["status"] in ("FAIL", "ERROR")]
    if failed_tasks:
        print(f"\nFailed Tasks ({len(failed_tasks)}):")
        for f in failed_tasks:
            reason = f.get("error") or "Verification mismatch"
            print(f"  • {f['task_id']}: {reason}")
        print("=" * 45)
    print()


# ---------------------------------------------------------------------------
# Main CLI & Runner Entry Point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="task_runner.py",
        description="Run and independently verify 200-Tools agent evaluation tasks.",
    )
    parser.add_argument(
        "--tasks-file",
        default=str(DEFAULT_TASKS_FILE),
        help="Path to tasks.json dataset (default: tasks.json).",
    )
    parser.add_argument(
        "--task-id",
        nargs="+",
        help="Filter specific task IDs to run (e.g. --task-id task_01_create_issue).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="0-indexed start offset in the task list (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to execute.",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=0,
        help="Number of retries for transient infrastructure errors (default: 0).",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Target sandbox repository in owner/repo format or bare name (default: GITHUB_REPO or sandbox).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (default: GEMINI_MODEL or gemini-3.6-flash).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RESULTS_FILE),
        help="Path to save results JSON (default: results/results.json).",
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="Path to save detailed execution log (default: results/task_runner.log).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run with mock execution (no live API calls made).",
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Skip sandbox initialization.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip pre-run sandbox reset (cleaning old issues, branches, PRs, and evaluation repos).",
    )
    parser.add_argument(
        "--no-clean-task",
        action="store_true",
        help="Skip cleaning task-specific resources right before running each task.",
    )
    parser.add_argument(
        "--agent",
        default="single_prompt",
        choices=["single_prompt", "orchestrator", "orchestration"],
        help="Agent architecture: 'single_prompt' (baseline) or 'orchestrator' (LangGraph control loop).",
    )
    parser.add_argument(
        "--disclosure-mode",
        default="all_loaded",
        choices=["all_loaded", "category_gated", "search_then_load", "hierarchical_planner"],
        help="Tool disclosure mode (default: all_loaded).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose console logging.",
    )

    args = parser.parse_args(argv)

    # Initialize logger
    log_path = Path(args.log_file)
    logger = setup_logging(log_file=log_path, verbose=args.verbose)
    logger.info("Starting task runner session with PID %d", os.getpid())

    # Check environment & sandbox prerequisites
    require_sandbox = not args.dry_run and not args.no_sandbox
    env_info = check_environment(require_sandbox=require_sandbox)

    # Load and validate tasks
    tasks_path = Path(args.tasks_file)
    try:
        all_tasks = load_tasks(tasks_path)
    except Exception as exc:
        print(f"ERROR: Failed to load tasks from {tasks_path}: {exc}", file=sys.stderr)
        return 1

    seen_ids: set[str] = set()
    for idx, t in enumerate(all_tasks, start=1):
        try:
            validate_task(t, idx)
            tid = t["task_id"]
            if tid in seen_ids:
                print(f"ERROR: Duplicate task_id '{tid}' in {tasks_path}.", file=sys.stderr)
                return 1
            seen_ids.add(tid)
        except ValueError as exc:
            print(f"ERROR: Task validation error: {exc}", file=sys.stderr)
            return 1

    # Filter tasks by --task-id, --start, --limit
    selected_tasks = all_tasks
    if args.task_id:
        wanted_ids = set(args.task_id)
        selected_tasks = [t for t in selected_tasks if t["task_id"] in wanted_ids]
        if not selected_tasks:
            print(f"ERROR: No tasks matched the requested ID(s): {args.task_id}", file=sys.stderr)
            return 1

    if args.start > 0:
        selected_tasks = selected_tasks[args.start:]

    if args.limit is not None and args.limit > 0:
        selected_tasks = selected_tasks[:args.limit]

    total_tasks = len(selected_tasks)
    if total_tasks == 0:
        print("No tasks to execute.")
        return 0

    # Initialize sandbox if live
    sandbox = None
    target_repo = args.repo or os.getenv("GITHUB_REPO") or "eval-sandbox-repo"

    if require_sandbox:
        from github_sandbox import GitHubSandbox  # type: ignore
        try:
            sandbox = GitHubSandbox()
            # If target_repo is bare, prefix with owner
            if "/" not in target_repo:
                target_repo = f"{sandbox._owner}/{target_repo}"
            logger.info("Sandbox initialised: owner=%s run_id=%s", sandbox._owner, sandbox.run_id)
        except Exception as exc:
            print(f"ERROR: Could not initialise GitHub Sandbox: {exc}", file=sys.stderr)
            return 1

    # Print startup environment banner
    print_env_check(
        env_info,
        target_repo=target_repo,
        total_tasks=total_tasks,
        agent_type=args.agent,
        disclosure_mode=args.disclosure_mode,
    )

    # Pre-run sandbox reset to establish deterministic starting state
    if require_sandbox and sandbox and not args.no_reset:
        print("Resetting sandbox to deterministic state...")
        try:
            reset_summary = sandbox.clean_sandbox(
                target_repo=target_repo,
                prefix="eval-",
                clean_repos=True,
            )
            repos_del = len(reset_summary.get("repos_deleted", []))
            repo_st = reset_summary.get("repo_state") or {}
            issues_cl = len(repo_st.get("issues_cleaned", []))
            branches_del = len(repo_st.get("branches_deleted", []))
            prs_cl = len(repo_st.get("pull_requests_closed", []))
            print(f"  ✓ Deleted {repos_del} leftover evaluation repo(s)")
            print(f"  ✓ Cleaned {issues_cl} old issue(s) in {target_repo}")
            print(f"  ✓ Deleted {branches_del} old branch(es) in {target_repo}")
            print(f"  ✓ Closed {prs_cl} old pull request(s) in {target_repo}")
            print("Deterministic sandbox state established.\n")
        except Exception as exc:
            logger.warning("Sandbox pre-run reset encountered an error: %s", exc)
            print(f"WARNING: Sandbox reset encountered an error: {exc}\n", file=sys.stderr)

    # Execute all selected tasks sequentially
    start_run_time = datetime.datetime.now(datetime.timezone.utc)
    results: List[Dict[str, Any]] = []

    for index, task in enumerate(selected_tasks, start=1):
        res = run_task(
            task=task,
            repo=target_repo,
            sandbox=sandbox,
            model=args.model or env_info["model"],
            max_retries=args.retry,
            mock=args.dry_run,
            clean_task=not args.no_clean_task,
            logger=logger,
            agent_type=args.agent,
            disclosure_mode=args.disclosure_mode,
        )
        results.append(res)
        print_task_progress(index=index, total=total_tasks, result=res)

    end_run_time = datetime.datetime.now(datetime.timezone.utc)
    total_duration = round((end_run_time - start_run_time).total_seconds(), 2)

    # Compute summary metrics
    passed_count = sum(1 for r in results if r["status"] == "PASS")
    failed_count = sum(1 for r in results if r["status"] == "FAIL")
    errors_count = sum(1 for r in results if r["status"] == "ERROR")
    no_verifier_count = sum(1 for r in results if r["status"] == "NO_VERIFIER")
    success_rate = (passed_count / total_tasks * 100.0) if total_tasks > 0 else 0.0

    summary = {
        "total": total_tasks,
        "passed": passed_count,
        "failed": failed_count,
        "errors": errors_count,
        "no_verifier": no_verifier_count,
        "success_rate_percent": round(success_rate, 2),
    }

    run_metadata = {
        "runner_version": RUNNER_VERSION,
        "start_time": start_run_time.isoformat(),
        "end_time": end_run_time.isoformat(),
        "total_duration_seconds": total_duration,
        "model": args.model or env_info["model"],
        "target_repo": target_repo,
        "task_order": [r["task_id"] for r in results],
    }

    # Save results to disk
    output_path = Path(args.output)
    save_results(output_path, run_metadata, summary, results)
    print(f"Results saved to: {output_path}")

    # Print summary banner
    print_summary(summary, results)

    # Return exit code: non-zero if there were failures or errors
    return 0 if (failed_count == 0 and errors_count == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
