"""
tests/test_single_prompt_agent.py
==================================
Unit tests for single_prompt_agent.py (Gemini backend).

All tests that touch the LLM or GitHub API are mocked.
One optional integration test (marked @pytest.mark.integration) runs a
real agent session against the GitHub sandbox.

Run unit tests only (no credentials needed):
    myvenv1/bin/python3 -c "... custom runner ..."

Run integration tests:
    GEMINI_API_KEY=... GITHUB_TOKEN=... GITHUB_OWNER=...
    pytest tests/test_single_prompt_agent.py -v -m integration
"""

from __future__ import annotations

import json
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import pytest  # type: ignore
except ModuleNotFoundError:
    import types as _types
    pytest = _types.ModuleType("pytest")

    def _raises(exc, match=None):
        import re
        class _Ctx:
            def __init__(self): self.value = None
            def __enter__(self): return self
            def __exit__(self, tp, val, tb):
                if tp is None: raise AssertionError(f"Expected {exc!r} raised")
                if not issubclass(tp, exc): return False
                self.value = val
                if match and not re.search(match, str(val)):
                    raise AssertionError(f"Message {str(val)!r} did not match {match!r}")
                return True
        return _Ctx()

    pytest.raises = _raises

    class _mark:
        @staticmethod
        def integration(fn_or_cls):
            msg = "integration — needs GEMINI_API_KEY + GITHUB_TOKEN + GITHUB_OWNER"
            skip = unittest.skip(msg)
            if isinstance(fn_or_cls, type):
                for attr in list(vars(fn_or_cls)):
                    if attr.startswith("test"):
                        setattr(fn_or_cls, attr, skip(getattr(fn_or_cls, attr)))
                return fn_or_cls
            return skip(fn_or_cls)

    pytest.mark = _mark

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))

import single_prompt_agent as spa
from single_prompt_agent import (
    AgentSession,
    _build_gemini_function_declaration,
    _execute_tool_call,
    _resolve_param,
    load_catalog,
    load_tool_schemas,
)


# ── google-genai stub ─────────────────────────────────────────────────────────
# Build a minimal stub so tests run without google-genai installed.

def _make_google_genai_stub():
    """Return a sys.modules entry for google.genai that is just enough for tests."""
    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    types_mod = types.ModuleType("google.genai.types")

    class _FunctionDeclaration:
        def __init__(self, **kw): self.__dict__.update(kw)

    class _Tool:
        def __init__(self, function_declarations=None):
            self.function_declarations = function_declarations or []

    class _FunctionCallingConfig:
        def __init__(self, mode="AUTO"): self.mode = mode

    class _ToolConfig:
        def __init__(self, function_calling_config=None): pass

    class _GenerateContentConfig:
        def __init__(self, **kw): self.__dict__.update(kw)

    class _Client:
        def __init__(self, api_key=None): pass
        class models:
            @staticmethod
            def generate_content(**kw): return MagicMock()

    types_mod.FunctionDeclaration = _FunctionDeclaration
    types_mod.Tool = _Tool
    types_mod.FunctionCallingConfig = _FunctionCallingConfig
    types_mod.ToolConfig = _ToolConfig
    types_mod.GenerateContentConfig = _GenerateContentConfig

    genai_mod.Client = _Client
    genai_mod.types = types_mod
    google_mod.genai = genai_mod

    return google_mod, genai_mod, types_mod


_google_stub, _genai_stub, _gtypes_stub = _make_google_genai_stub()
sys.modules.setdefault("google", _google_stub)
sys.modules.setdefault("google.genai", _genai_stub)
sys.modules.setdefault("google.genai.types", _gtypes_stub)


# ── shared mock factories ─────────────────────────────────────────────────────

def _make_sandbox(owner="test-owner"):
    sb = MagicMock()
    sb._owner = owner
    sb.execute.side_effect = lambda op_id, fn: {"success": True, "operation": op_id, "result": fn()}
    return sb


def _make_fc(name: str, args: dict):
    """Build a fake Gemini FunctionCall (has .name and .args)."""
    fc = SimpleNamespace(name=name, args=args)
    return fc


def _make_gemini_response(function_calls: list[tuple[str, dict]], text: str = ""):
    """
    Build a fake Gemini GenerateContentResponse.

    Gemini response structure:
        response.candidates[0].content.parts[i]
            .function_call  — FunctionCall with .name/.args
            .text           — str
    """
    parts = []
    if text:
        parts.append(SimpleNamespace(text=text, function_call=None))
    for name, args in function_calls:
        parts.append(SimpleNamespace(
            text=None,
            function_call=_make_fc(name, args),
        ))

    content = SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


def _make_gemini_response_text_only(text="I cannot help with that."):
    return _make_gemini_response([], text=text)


# ── 1. Tool schema loading ────────────────────────────────────────────────────

class TestLoadToolSchemas(unittest.TestCase):
    """Tool schemas are loaded correctly from the catalog."""

    def test_returns_list_and_index(self):
        fn_decls, index = load_tool_schemas()
        self.assertIsInstance(fn_decls, list)
        self.assertIsInstance(index, dict)

    def test_all_51_tools_loaded(self):
        fn_decls, _ = load_tool_schemas()
        self.assertEqual(len(fn_decls), 51)

    def test_each_decl_has_required_keys(self):
        fn_decls, _ = load_tool_schemas()
        for d in fn_decls:
            self.assertIn("name", d)
            self.assertIn("description", d)
            self.assertIn("parameters", d)

    def test_parameters_is_object_type(self):
        fn_decls, _ = load_tool_schemas()
        for d in fn_decls:
            self.assertEqual(d["parameters"]["type"], "OBJECT")
            self.assertIn("properties", d["parameters"])

    def test_index_keys_match_decl_names(self):
        fn_decls, index = load_tool_schemas()
        for d in fn_decls:
            self.assertIn(d["name"], index)

    def test_issues_create_in_index(self):
        _, index = load_tool_schemas()
        self.assertIn("issues__create", index)

    def test_git_create_ref_in_index(self):
        _, index = load_tool_schemas()
        self.assertIn("git__create_ref", index)

    def test_repos_delete_in_index(self):
        _, index = load_tool_schemas()
        self.assertIn("repos__delete", index)

    def test_operation_id_preserved_in_index(self):
        _, index = load_tool_schemas()
        op = index["issues__create"]
        self.assertEqual(op["operation_id"], "issues/create")

    def test_catalog_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_tool_schemas("/nonexistent/path/catalog.json")

    def test_description_contains_operation_id(self):
        fn_decls, _ = load_tool_schemas()
        issues_create = next(d for d in fn_decls if d["name"] == "issues__create")
        self.assertIn("issues/create", issues_create["description"])

    def test_issues_create_has_title_required(self):
        fn_decls, _ = load_tool_schemas()
        d = next(d for d in fn_decls if d["name"] == "issues__create")
        self.assertIn("title", d["parameters"].get("required", []))

    def test_risk_level_in_description(self):
        fn_decls, _ = load_tool_schemas()
        delete_tool = next(d for d in fn_decls if d["name"] == "repos__delete")
        self.assertIn("high", delete_tool["description"].lower())

    def test_gemini_type_strings_uppercase(self):
        """All property types must be uppercase Gemini type strings."""
        fn_decls, _ = load_tool_schemas()
        for d in fn_decls:
            for prop_name, prop in d["parameters"].get("properties", {}).items():
                t = prop.get("type", "")
                self.assertEqual(t, t.upper(), msg=f"{d['name']}.{prop_name} type should be uppercase")


# ── 2. $ref resolution ───────────────────────────────────────────────────────

class TestParamRefResolution(unittest.TestCase):
    def test_owner_ref_resolves(self):
        raw = {"$ref": "#/components/parameters/owner"}
        result = _resolve_param(raw)
        self.assertEqual(result["name"], "owner")
        self.assertTrue(result.get("required", False))

    def test_repo_ref_resolves(self):
        raw = {"$ref": "#/components/parameters/repo"}
        result = _resolve_param(raw)
        self.assertEqual(result["name"], "repo")

    def test_issue_number_ref_resolves(self):
        raw = {"$ref": "#/components/parameters/issue-number"}
        result = _resolve_param(raw)
        self.assertEqual(result["name"], "issue_number")
        self.assertEqual(result["schema"]["type"], "integer")

    def test_plain_param_passed_through(self):
        raw = {"name": "state", "in": "query", "required": False,
               "schema": {"type": "string"}}
        result = _resolve_param(raw)
        self.assertEqual(result["name"], "state")

    def test_non_dict_returns_empty(self):
        result = _resolve_param("not a dict")
        self.assertEqual(result, {})

    def test_unknown_ref_returns_fallback(self):
        raw = {"$ref": "#/components/parameters/unknown-weird"}
        result = _resolve_param(raw)
        self.assertIn("name", result)


# ── 3. AgentSession: single LLM call ─────────────────────────────────────────

class TestAgentSessionLLMCall(unittest.TestCase):
    def _make_session(self):
        sb = _make_sandbox()
        with patch("single_prompt_agent._get_gemini_client", return_value=MagicMock()):
            session = AgentSession(sb)
        return session

    def test_llm_called_once(self):
        session = self._make_session()
        response = _make_gemini_response_text_only()
        with patch.object(session, "_llm_call", return_value=response) as mock_call:
            session.run("List all repos")
        mock_call.assert_called_once()

    def test_llm_receives_tool_object(self):
        """_llm_call is called with a tool argument (not None)."""
        session = self._make_session()
        captured = []

        def capture(contents, tool):
            captured.append(tool)
            return _make_gemini_response_text_only()

        with patch.object(session, "_llm_call", side_effect=capture):
            session.run("Do something")

        self.assertEqual(len(captured), 1)
        self.assertIsNotNone(captured[0])

    def test_user_content_contains_task(self):
        session = self._make_session()
        captured_contents = []

        def capture(contents, tool):
            captured_contents.extend(contents)
            return _make_gemini_response_text_only()

        with patch.object(session, "_llm_call", side_effect=capture):
            session.run("Create an issue titled 'Login bug'")

        combined = " ".join(
            p["text"] for c in captured_contents for p in c.get("parts", [])
        )
        self.assertIn("Login bug", combined)

    def test_repo_context_injected_when_provided(self):
        session = self._make_session()
        captured_contents = []

        def capture(contents, tool):
            captured_contents.extend(contents)
            return _make_gemini_response_text_only()

        with patch.object(session, "_llm_call", side_effect=capture):
            session.run("Create issue", repo="myowner/myrepo")

        combined = " ".join(
            p["text"] for c in captured_contents for p in c.get("parts", [])
        )
        self.assertIn("myowner/myrepo", combined)


# ── 4. Tool calls correctly converted to sandbox calls ───────────────────────

class TestToolCallExecution(unittest.TestCase):
    def _make_session(self):
        sb = _make_sandbox()
        with patch("single_prompt_agent._get_gemini_client", return_value=MagicMock()):
            session = AgentSession(sb)
        session.sandbox = sb
        return session, sb

    def test_issues_create_routed_to_sandbox(self):
        session, sb = self._make_session()
        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = MagicMock()
        sb.get_repo.return_value = mock_repo
        response = _make_gemini_response([
            ("issues__create", {"repo": "test-repo", "title": "Login bug"})
        ])
        with patch.object(session, "_llm_call", return_value=response):
            result = session.run("Create issue")
        sb.execute.assert_called_once()
        self.assertEqual(sb.execute.call_args[0][0], "issues/create")

    def test_operation_id_preserved_in_result(self):
        session, sb = self._make_session()
        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = MagicMock()
        sb.get_repo.return_value = mock_repo
        response = _make_gemini_response([
            ("issues__create", {"repo": "test-repo", "title": "Test"})
        ])
        with patch.object(session, "_llm_call", return_value=response):
            result = session.run("Create issue")
        self.assertEqual(result["tool_calls"][0]["operation_id"], "issues/create")

    def test_multiple_tool_calls_all_executed(self):
        session, sb = self._make_session()
        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = MagicMock()
        sb.get_repo.return_value = mock_repo
        response = _make_gemini_response([
            ("issues__create", {"repo": "r", "title": "Issue 1"}),
            ("issues__create", {"repo": "r", "title": "Issue 2"}),
        ])
        with patch.object(session, "_llm_call", return_value=response):
            result = session.run("Create two issues")
        self.assertEqual(sb.execute.call_count, 2)
        self.assertEqual(len(result["tool_calls"]), 2)
        self.assertEqual(len(result["results"]), 2)

    def test_tool_call_result_captured(self):
        session, sb = self._make_session()
        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = MagicMock(number=42)
        sb.get_repo.return_value = mock_repo
        response = _make_gemini_response([
            ("issues__create", {"repo": "r", "title": "Login bug"})
        ])
        with patch.object(session, "_llm_call", return_value=response):
            result = session.run("Create issue")
        self.assertEqual(len(result["results"]), 1)
        self.assertTrue(result["results"][0]["success"])

    def test_repos_delete_routed_to_sandbox(self):
        session, sb = self._make_session()
        mock_repo = MagicMock()
        sb.get_repo.return_value = mock_repo
        response = _make_gemini_response([
            ("repos__delete", {"repo": "old-repo"})
        ])
        with patch.object(session, "_llm_call", return_value=response):
            session.run("Delete repo")
        sb.execute.assert_called_once()


# ── 5. Execution results captured ────────────────────────────────────────────

class TestExecutionResults(unittest.TestCase):
    def _make_session(self):
        sb = _make_sandbox()
        with patch("single_prompt_agent._get_gemini_client", return_value=MagicMock()):
            session = AgentSession(sb)
        session.sandbox = sb
        return session, sb

    def test_success_flag_true_on_all_pass(self):
        session, sb = self._make_session()
        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = MagicMock()
        sb.get_repo.return_value = mock_repo
        response = _make_gemini_response([
            ("issues__create", {"repo": "r", "title": "T"})
        ])
        with patch.object(session, "_llm_call", return_value=response):
            result = session.run("Task")
        self.assertTrue(result["success"])

    def test_success_flag_false_on_any_fail(self):
        session, sb = self._make_session()
        sb.execute.side_effect = None
        sb.execute.return_value = {
            "success": False, "operation": "issues/create",
            "error_type": "GITHUB_ERROR", "message": "Boom",
        }
        sb.get_repo.return_value = MagicMock()
        response = _make_gemini_response([
            ("issues__create", {"repo": "r", "title": "T"})
        ])
        with patch.object(session, "_llm_call", return_value=response):
            result = session.run("Task")
        self.assertFalse(result["success"])

    def test_task_preserved_in_result(self):
        session, sb = self._make_session()
        with patch.object(session, "_llm_call", return_value=_make_gemini_response_text_only()):
            result = session.run("My special task")
        self.assertEqual(result["task"], "My special task")

    def test_model_preserved_in_result(self):
        session, sb = self._make_session()
        with patch.object(session, "_llm_call", return_value=_make_gemini_response_text_only()):
            result = session.run("Task")
        self.assertIn("model", result)
        self.assertIn("gemini", result["model"])


# ── 6. Invalid tool calls handled ────────────────────────────────────────────

class TestInvalidToolCalls(unittest.TestCase):
    def _make_session(self):
        sb = _make_sandbox()
        with patch("single_prompt_agent._get_gemini_client", return_value=MagicMock()):
            session = AgentSession(sb)
        return session, sb

    def test_unknown_function_name_handled(self):
        session, sb = self._make_session()
        response = _make_gemini_response([
            ("totally__unknown__operation", {"foo": "bar"})
        ])
        with patch.object(session, "_llm_call", return_value=response):
            result = session.run("Do something weird")
        self.assertFalse(result["success"])
        self.assertEqual(result["tool_calls"][0]["operation_id"], "unknown")

    def test_unknown_operation_execution_returns_error(self):
        sb = _make_sandbox()
        op = {"operation_id": "fake/nonexistent", "method": "GET", "path": "/fake"}
        result = _execute_tool_call(sb, op, {})
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "UNKNOWN_OPERATION")

    def test_no_tool_calls_returns_text_response(self):
        session, sb = self._make_session()
        response = _make_gemini_response_text_only("Sorry, I can't do that.")
        with patch.object(session, "_llm_call", return_value=response):
            result = session.run("Impossible task")
        self.assertEqual(result["tool_calls"], [])
        self.assertIn("can't do that", result["final_response"])


# ── 7. LLM / API errors handled ──────────────────────────────────────────────

class TestLLMErrors(unittest.TestCase):
    def _make_session(self):
        sb = _make_sandbox()
        with patch("single_prompt_agent._get_gemini_client", return_value=MagicMock()):
            session = AgentSession(sb)
        return session, sb

    def test_llm_exception_returns_structured_failure(self):
        session, sb = self._make_session()
        with patch.object(session, "_llm_call", side_effect=RuntimeError("API down")):
            result = session.run("Do anything")
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("API down", result["error"])

    def test_result_has_empty_tool_calls_on_llm_error(self):
        session, sb = self._make_session()
        with patch.object(session, "_llm_call", side_effect=ValueError("Bad request")):
            result = session.run("Task")
        self.assertEqual(result["tool_calls"], [])

    def test_no_gemini_key_raises_env_error(self):
        env_without_key = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            with pytest.raises(EnvironmentError):
                spa._get_gemini_client()

    def test_google_genai_not_installed_raises_import_error(self):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("google", "google.genai"):
                raise ImportError("No module named 'google'")
            return real_import(name, *args, **kwargs)

        # Temporarily remove stubs from sys.modules
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("google")}
        try:
            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(ImportError, match="google-genai"):
                    spa._get_gemini_client()
        finally:
            sys.modules.update(saved)


# ── 8. No credentials in logs ────────────────────────────────────────────────

class TestNoSecretsInLogs(unittest.TestCase):
    def test_token_argument_redacted(self):
        import io, logging
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logging.getLogger("single_prompt_agent").addHandler(handler)
        logging.getLogger("single_prompt_agent").setLevel(logging.DEBUG)

        sb = _make_sandbox()
        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = MagicMock()
        sb.get_repo.return_value = mock_repo

        with patch("single_prompt_agent._get_gemini_client", return_value=MagicMock()):
            session = AgentSession(sb)

        response = _make_gemini_response([
            ("issues__create", {"repo": "r", "title": "T", "api_token": "secret-abc"})
        ])
        with patch.object(session, "_llm_call", return_value=response):
            session.run("Task")

        log_output = log_stream.getvalue()
        logging.getLogger("single_prompt_agent").removeHandler(handler)
        self.assertNotIn("secret-abc", log_output)
        self.assertIn("***", log_output)


# ── 9. Catalog loading ────────────────────────────────────────────────────────

class TestCatalogLoading(unittest.TestCase):
    def test_catalog_has_correct_version(self):
        catalog = load_catalog()
        self.assertEqual(catalog["catalog_version"], "1.0")

    def test_catalog_has_51_operations(self):
        catalog = load_catalog()
        self.assertEqual(catalog["operation_count"], 51)

    def test_all_operation_ids_unique(self):
        catalog = load_catalog()
        ids = [op["operation_id"] for op in catalog["operations"]]
        self.assertEqual(len(ids), len(set(ids)))


# ── 10. Trial 1: simple issue creation ───────────────────────────────────────

class TestTrialTaskSimpleIssue(unittest.TestCase):
    def test_expected_operation_selected(self):
        sb = _make_sandbox()
        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = MagicMock(number=1, title="Login bug")
        sb.get_repo.return_value = mock_repo

        with patch("single_prompt_agent._get_gemini_client", return_value=MagicMock()):
            session = AgentSession(sb)

        response = _make_gemini_response([
            ("issues__create", {"repo": "sandbox-repo", "title": "Login bug", "labels": ["bug"]})
        ])
        with patch.object(session, "_llm_call", return_value=response):
            result = session.run("Create an issue titled 'Login bug' with the 'bug' label.")

        self.assertTrue(result["success"])
        self.assertEqual(result["tool_calls"][0]["operation_id"], "issues/create")
        args = result["tool_calls"][0]["arguments"]
        self.assertEqual(args["title"], "Login bug")
        self.assertIn("bug", args.get("labels", []))


# ── 11. Trial 2: branch creation ─────────────────────────────────────────────

class TestTrialTaskBranchCreation(unittest.TestCase):
    def test_expected_operation_selected(self):
        sb = _make_sandbox()
        mock_repo = MagicMock()
        mock_repo.create_git_ref.return_value = MagicMock()
        sb.get_repo.return_value = mock_repo

        with patch("single_prompt_agent._get_gemini_client", return_value=MagicMock()):
            session = AgentSession(sb)

        response = _make_gemini_response([
            ("git__create_ref", {
                "repo": "sandbox-repo",
                "ref": "refs/heads/feature-login",
                "sha": "abc123def456",
            })
        ])
        with patch.object(session, "_llm_call", return_value=response):
            result = session.run("Create a branch called 'feature-login' from 'main'.")

        self.assertTrue(result["success"])
        self.assertEqual(result["tool_calls"][0]["operation_id"], "git/create-ref")
        args = result["tool_calls"][0]["arguments"]
        self.assertIn("feature-login", args["ref"])


# ── 12. Trial 3: chained operations ──────────────────────────────────────────

class TestTrialTaskChained(unittest.TestCase):
    def test_multiple_operations_selected(self):
        sb = _make_sandbox()
        mock_repo = MagicMock()
        mock_repo.create_issue.return_value = MagicMock(number=5)
        mock_repo.create_pull.return_value = MagicMock(number=1)
        mock_repo.create_git_ref.return_value = MagicMock()
        sb.get_repo.return_value = mock_repo

        with patch("single_prompt_agent._get_gemini_client", return_value=MagicMock()):
            session = AgentSession(sb)

        response = _make_gemini_response([
            ("git__create_ref", {"repo": "r", "ref": "refs/heads/bug-fix", "sha": "abc123"}),
            ("issues__create", {"repo": "r", "title": "Login bug", "body": "Login fails."}),
            ("pulls__create", {"repo": "r", "title": "Fix login", "head": "bug-fix", "base": "main", "body": "Fixes #5"}),
        ])
        with patch.object(session, "_llm_call", return_value=response):
            result = session.run(
                "Create a branch 'bug-fix' from main, create an issue, and open a PR."
            )

        op_ids = [tc["operation_id"] for tc in result["tool_calls"]]
        self.assertIn("git/create-ref", op_ids)
        self.assertIn("issues/create", op_ids)
        self.assertIn("pulls/create", op_ids)
        self.assertEqual(len(result["tool_calls"]), 3)


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.integration
class TestIntegration(unittest.TestCase):
    """
    End-to-end test against real GitHub sandbox + real Gemini LLM.

    Prerequisites:
        export GEMINI_API_KEY=...
        export GITHUB_TOKEN=ghp_...
        export GITHUB_OWNER=your-username
        pip install google-genai
    """

    @classmethod
    def setUpClass(cls):
        from github_sandbox import GitHubSandbox
        cls.sandbox = GitHubSandbox()
        cls.repos = cls.sandbox.reset_sandbox()
        cls.repo_full = f"{cls.sandbox._owner}/{cls.repos[0].name}"

    @classmethod
    def tearDownClass(cls):
        cls.sandbox.cleanup()

    def test_simple_issue_creation(self):
        result = spa.run_agent(
            task=f"In the repository '{self.repo_full}', create an issue titled 'Integration test issue' with the body 'This is a test'.",
            sandbox=self.sandbox,
        )
        self.assertTrue(result["success"], msg=result.get("error") or result)
        op_ids = [tc["operation_id"] for tc in result["tool_calls"]]
        self.assertIn("issues/create", op_ids)

    def test_branch_creation(self):
        result = spa.run_agent(
            task=f"In the repository '{self.repo_full}', create a branch called 'test-branch' from 'main'.",
            sandbox=self.sandbox,
        )
        self.assertTrue(result["success"], msg=result.get("error") or result)
        op_ids = [tc["operation_id"] for tc in result["tool_calls"]]
        self.assertTrue(
            "git/create-ref" in op_ids or "repos/get-branch" in op_ids,
            msg=f"Expected branch op, got: {op_ids}",
        )
