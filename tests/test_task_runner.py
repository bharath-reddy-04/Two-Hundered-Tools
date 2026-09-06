"""
tests/test_task_runner.py
=========================
Unit tests for task_runner.py.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
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

import task_runner
from task_runner import (
    check_environment,
    execute_agent,
    is_transient_error,
    load_tasks,
    main,
    run_task,
    save_results,
    validate_task,
    verify_task,
)


class TestTaskLoading(unittest.TestCase):
    def test_load_tasks_from_list(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"task_id": "t1", "prompt": "do something"}], f)
            temp_path = Path(f.name)
        try:
            tasks = load_tasks(temp_path)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["task_id"], "t1")
        finally:
            temp_path.unlink()

    def test_load_tasks_from_dict_wrapper(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"tasks": [{"task_id": "t2", "instruction": "do something else"}]}, f)
            temp_path = Path(f.name)
        try:
            tasks = load_tasks(temp_path)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["task_id"], "t2")
        finally:
            temp_path.unlink()

    def test_load_tasks_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_tasks(Path("/nonexistent/path/tasks.json"))

    def test_load_tasks_invalid_json_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not-valid-json {")
            temp_path = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                load_tasks(temp_path)
        finally:
            temp_path.unlink()

    def test_load_tasks_invalid_structure_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump("just a string", f)
            temp_path = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                load_tasks(temp_path)
        finally:
            temp_path.unlink()


class TestTaskValidation(unittest.TestCase):
    def test_valid_task_with_prompt_passes(self):
        validate_task({"task_id": "t1", "prompt": "create an issue"}, index=1)

    def test_valid_task_with_instruction_passes(self):
        validate_task({"task_id": "t2", "instruction": "create an issue"}, index=2)

    def test_missing_task_id_raises(self):
        with self.assertRaises(ValueError):
            validate_task({"prompt": "create an issue"}, index=1)

    def test_missing_prompt_and_instruction_raises(self):
        with self.assertRaises(ValueError):
            validate_task({"task_id": "t3"}, index=3)

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValueError):
            validate_task({"task_id": "t4", "prompt": "   "}, index=4)


class TestTransientErrorDetection(unittest.TestCase):
    def test_rate_limit_is_transient(self):
        self.assertTrue(is_transient_error("GitHub rate limit exceeded: 429"))

    def test_timeout_is_transient(self):
        self.assertTrue(is_transient_error("Connection timeout from server"))

    def test_connection_reset_is_transient(self):
        self.assertTrue(is_transient_error("Connection reset by peer"))

    def test_syntax_error_not_transient(self):
        self.assertFalse(is_transient_error("SyntaxError: unexpected token"))

    def test_not_found_not_transient(self):
        self.assertFalse(is_transient_error("404 Not Found"))


class TestTaskExecution(unittest.TestCase):
    def test_run_task_pass(self):
        task = {
            "task_id": "test_pass",
            "prompt": "Create an issue",
            "difficulty": "easy",
            "expected_state": {"issue": {"exists": True, "title": "Test"}},
        }
        with patch("task_runner.execute_agent") as mock_exec, \
             patch("task_runner.verify_state") as mock_verify:
            mock_exec.return_value = {
                "success": True,
                "tool_calls": [{"operation_id": "issues/create"}],
                "final_response": "Done",
            }
            mock_verify.return_value = {"passed": True, "checks": {"exists": True}}

            res = run_task(task, repo="owner/repo", sandbox=None)
            self.assertEqual(res["status"], "PASS")
            self.assertTrue(res["passed"])
            self.assertEqual(len(res["tool_calls"]), 1)

    def test_run_task_fail_when_verifier_fails(self):
        task = {
            "task_id": "test_fail",
            "prompt": "Create an issue",
            "difficulty": "easy",
            "expected_state": {"issue": {"exists": True, "title": "Test"}},
        }
        with patch("task_runner.execute_agent") as mock_exec, \
             patch("task_runner.verify_state") as mock_verify:
            mock_exec.return_value = {
                "success": True,
                "tool_calls": [{"operation_id": "issues/create"}],
                "final_response": "Done",
            }
            mock_verify.return_value = {
                "passed": False,
                "checks": {"exists": False},
                "errors": ["Issue not found"],
            }

            res = run_task(task, repo="owner/repo", sandbox=None)
            self.assertEqual(res["status"], "FAIL")
            self.assertFalse(res["passed"])
            self.assertIn("Issue not found", str(res["error"]))

    def test_run_task_no_verifier(self):
        task = {
            "task_id": "test_no_verifier",
            "prompt": "Stripe charge customer",
            "difficulty": "medium",
            # No expected_state defined
        }
        res = run_task(task, repo="owner/repo", sandbox=None, mock=True)
        self.assertEqual(res["status"], "NO_VERIFIER")
        self.assertFalse(res["passed"])

    def test_run_task_agent_error(self):
        task = {
            "task_id": "test_agent_error",
            "prompt": "Create issue",
            "difficulty": "easy",
            "expected_state": {"issue": {"exists": True}},
        }
        with patch("task_runner.execute_agent") as mock_exec, \
             patch("task_runner.verify_state") as mock_verify:
            mock_exec.side_effect = RuntimeError("API service completely down")
            mock_verify.return_value = {"passed": False, "checks": {}, "errors": ["Not found"]}

            res = run_task(task, repo="owner/repo", sandbox=None)
            self.assertEqual(res["status"], "ERROR")
            self.assertFalse(res["passed"])
            self.assertIn("API service completely down", res["error"])


class TestResultsPersistence(unittest.TestCase):
    def test_save_results(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)
        try:
            metadata = {"runner_version": "1.0.0", "start_time": "2026-09-05T00:00:00Z"}
            summary = {"total": 1, "passed": 1, "failed": 0, "errors": 0, "no_verifier": 0, "success_rate_percent": 100.0}
            results = [{
                "task_id": "t1",
                "instruction": "Test",
                "status": "PASS",
                "passed": True,
                "agent_response": "Done",
                "tool_calls": [],
                "verification": {"passed": True},
                "error": None,
                "duration_seconds": 0.5,
            }]
            save_results(temp_path, metadata, summary, results)

            with temp_path.open("r") as f:
                saved = json.load(f)
            self.assertEqual(saved["summary"]["passed"], 1)
            self.assertEqual(len(saved["results"]), 1)
            self.assertEqual(saved["results"][0]["task_id"], "t1")
        finally:
            temp_path.unlink()


class TestCLIAndFiltering(unittest.TestCase):
    def test_cli_dry_run_all(self):
        exit_code = main(["--dry-run", "--limit", "2"])
        self.assertEqual(exit_code, 0)

    def test_cli_dry_run_specific_id(self):
        exit_code = main(["--dry-run", "--task-id", "task_01_create_issue"])
        self.assertEqual(exit_code, 0)

    def test_cli_unknown_task_id_fails(self):
        exit_code = main(["--dry-run", "--task-id", "nonexistent_task_id"])
        self.assertEqual(exit_code, 1)

    def test_cli_dry_run_with_no_reset_and_no_clean_task_flags(self):
        exit_code = main(["--dry-run", "--no-reset", "--no-clean-task", "--limit", "1"])
        self.assertEqual(exit_code, 0)

    def test_run_task_invokes_clean_task_state(self):
        task = {
            "task_id": "test_clean",
            "prompt": "Create issue",
            "expected_state": {"issue": {"title": "Test"}},
        }
        mock_sandbox = MagicMock()
        with patch("task_runner.execute_agent") as mock_exec, \
             patch("task_runner.verify_state") as mock_verify:
            mock_exec.return_value = {"success": True, "tool_calls": []}
            mock_verify.return_value = {"passed": True, "checks": {}}

            run_task(task, repo="owner/repo", sandbox=mock_sandbox, mock=False, clean_task=True)
            mock_sandbox.clean_task_state.assert_called_once_with(repo_name_or_obj="owner/repo", task=task)

    def test_run_task_skips_clean_task_when_disabled(self):
        task = {
            "task_id": "test_no_clean",
            "prompt": "Create issue",
            "expected_state": {"issue": {"title": "Test"}},
        }
        mock_sandbox = MagicMock()
        with patch("task_runner.execute_agent") as mock_exec, \
             patch("task_runner.verify_state") as mock_verify:
            mock_exec.return_value = {"success": True, "tool_calls": []}
            mock_verify.return_value = {"passed": True, "checks": {}}

            run_task(task, repo="owner/repo", sandbox=mock_sandbox, mock=False, clean_task=False)
            mock_sandbox.clean_task_state.assert_not_called()

    def test_cli_live_clean_sandbox_called(self):
        mock_sb = MagicMock()
        mock_sb._owner = "testowner"
        mock_sb.clean_sandbox.return_value = {
            "repos_deleted": ["eval-repo-old"],
            "repo_state": {"issues_cleaned": [1], "branches_deleted": ["b1"], "pull_requests_closed": [10]},
        }

        with patch.dict(os.environ, {"GITHUB_TOKEN": "tok", "GITHUB_OWNER": "testowner", "GEMINI_API_KEY": "key"}), \
             patch("github_sandbox.GitHubSandbox", return_value=mock_sb), \
             patch("task_runner.run_task") as mock_run_task:
            mock_run_task.return_value = {
                "task_id": "task_01_create_issue",
                "instruction": "Test",
                "status": "PASS",
                "passed": True,
                "duration_seconds": 0.1,
            }

            exit_code = main(["--repo", "testowner/eval-sandbox-repo", "--limit", "1"])
            self.assertEqual(exit_code, 0)
            mock_sb.clean_sandbox.assert_called_once_with(
                target_repo="testowner/eval-sandbox-repo",
                prefix="eval-",
                clean_repos=True,
            )

    def test_cli_dry_run_with_orchestrator_agent(self):
        exit_code = main(["--dry-run", "--agent", "orchestrator", "--limit", "1"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
