"""
single_prompt_agent.py
======================
Baseline single-prompt GitHub agent for the 200-Tools evaluation project.

Architecture
------------
    User Task
        ↓
    single_prompt_agent.py
        ↓  (ONE LLM call with all 51 tool schemas)
    LLM (Google Gemini function-calling)
        ↓  function_call parts
    github_sandbox.py  (execution layer)
        ↓
    GitHub Sandbox

What this file deliberately does NOT do
----------------------------------------
* No planning loop
* No ReAct / reflection loop
* No retry loop
* No verification (verifier.py is called externally)
* No multi-agent orchestration
* No LangChain / LangGraph

The single LLM call occurs inside ``AgentSession.run()``,
specifically at the ``self._llm_call(contents, tools)`` line.
Search for ``# ── THE SINGLE LLM CALL ──`` to find it.

Configuration
-------------
Required environment variables (load from .env or shell):
    GEMINI_API_KEY    — Google AI Studio / Vertex AI API key
    GITHUB_TOKEN      — GitHub personal access token
    GITHUB_OWNER      — GitHub account / org owning sandbox repos

Optional:
    GEMINI_MODEL      — defaults to "gemini-3.6-flash"
    CATALOG_PATH      — path to operation_catalog.json
                        (defaults to schemas/operation_catalog.json
                         relative to this file's directory)

Install:
    pip install google-genai>=1.0.0
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

try:
    from github_operations import execute_operation
except ImportError:
    from harness.github_operations import execute_operation

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent  # harness/ → project root

_DEFAULT_CATALOG = _PROJECT_ROOT / "schemas" / "operation_catalog.json"

# ---------------------------------------------------------------------------
# $ref → inline param resolution
# ---------------------------------------------------------------------------

_PARAM_REF_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "#/components/parameters/owner": {
        "name": "owner",
        "in": "path",
        "required": True,
        "description": "The account owner of the repository (username or org name).",
        "schema": {"type": "string"},
    },
    "#/components/parameters/repo": {
        "name": "repo",
        "in": "path",
        "required": True,
        "description": "The name of the repository (without the owner prefix).",
        "schema": {"type": "string"},
    },
    "#/components/parameters/issue-number": {
        "name": "issue_number",
        "in": "path",
        "required": True,
        "description": "The number that identifies the issue.",
        "schema": {"type": "integer"},
    },
    "#/components/parameters/pull-number": {
        "name": "pull_number",
        "in": "path",
        "required": True,
        "description": "The number that identifies the pull request.",
        "schema": {"type": "integer"},
    },
    "#/components/parameters/branch": {
        "name": "branch",
        "in": "path",
        "required": True,
        "description": "The name of the branch.",
        "schema": {"type": "string"},
    },
    "#/components/parameters/per-page": {
        "name": "per_page",
        "in": "query",
        "required": False,
        "description": "The number of results per page (max 100).",
        "schema": {"type": "integer", "default": 30},
    },
    "#/components/parameters/page": {
        "name": "page",
        "in": "query",
        "required": False,
        "description": "The page number of the results to fetch.",
        "schema": {"type": "integer", "default": 1},
    },
    "#/components/parameters/collaborator": {
        "name": "username",
        "in": "path",
        "required": True,
        "description": "The handle for the GitHub user account.",
        "schema": {"type": "string"},
    },
    "#/components/parameters/commit-sha": {
        "name": "commit_sha",
        "in": "path",
        "required": True,
        "description": "The SHA of the commit.",
        "schema": {"type": "string"},
    },
    "#/components/parameters/ref": {
        "name": "ref",
        "in": "path",
        "required": True,
        "description": "The git reference (branch, tag, or full ref like refs/heads/main).",
        "schema": {"type": "string"},
    },
    "#/components/parameters/git-ref-only": {
        "name": "ref",
        "in": "path",
        "required": True,
        "description": "The git reference (e.g. heads/main or main).",
        "schema": {"type": "string"},
    },
    "#/components/parameters/commit-ref": {
        "name": "ref",
        "in": "path",
        "required": True,
        "description": "The commit reference (commit SHA, branch name, or tag name).",
        "schema": {"type": "string"},
    },
    "#/components/parameters/basehead": {
        "name": "basehead",
        "in": "path",
        "required": True,
        "description": "Base and head to compare: base...head.",
        "schema": {"type": "string"},
    },
    "#/components/parameters/file-sha": {
        "name": "file_sha",
        "in": "path",
        "required": True,
        "description": "The SHA of the blob (file).",
        "schema": {"type": "string"},
    },
    "#/components/parameters/tree-sha": {
        "name": "tree_sha",
        "in": "path",
        "required": True,
        "description": "The SHA1 value or ref name of the tree.",
        "schema": {"type": "string"},
    },
    "#/components/parameters/run-id": {
        "name": "run_id",
        "in": "path",
        "required": True,
        "description": "The unique identifier of the workflow run.",
        "schema": {"type": "integer"},
    },
    "#/components/parameters/workflow-id": {
        "name": "workflow_id",
        "in": "path",
        "required": True,
        "description": "The ID or filename of the workflow.",
        "schema": {"type": "string"},
    },
    "#/components/parameters/since-repo-date": {
        "name": "since",
        "in": "query",
        "required": False,
        "description": "Only show repositories updated after this ISO 8601 timestamp.",
        "schema": {"type": "string"},
    },
    "#/components/parameters/before-repo-date": {
        "name": "before",
        "in": "query",
        "required": False,
        "description": "Only show repositories updated before this ISO 8601 timestamp.",
        "schema": {"type": "string"},
    },
    "#/components/parameters/comment-id": {
        "name": "comment_id",
        "in": "path",
        "required": True,
        "description": "The unique identifier of the comment.",
        "schema": {"type": "integer"},
    },
}


def _resolve_param(param: dict | Any) -> dict:
    """Replace a $ref stub with its inline description, or pass through."""
    if not isinstance(param, dict):
        return {}
    ref = param.get("$ref")
    if ref:
        return _PARAM_REF_DESCRIPTIONS.get(
            ref, {"name": ref.split("/")[-1], "in": "path", "required": True}
        )
    return param


# ---------------------------------------------------------------------------
# Gemini schema type mapping
# ---------------------------------------------------------------------------

# Gemini's Schema type field only accepts these STRING values (not Python types).
# Map OpenAPI type names → Gemini type strings.
_OPENAPI_TO_GEMINI_TYPE = {
    "string":  "STRING",
    "integer": "INTEGER",
    "number":  "NUMBER",
    "boolean": "BOOLEAN",
    "array":   "ARRAY",
    "object":  "OBJECT",
}


def _openapi_prop_to_gemini_schema(prop: dict) -> dict:
    """
    Convert a single OpenAPI property dict to a Gemini Schema-compatible dict.

    Returns a plain dict (not a genai.types.Schema object) — we pass the whole
    FunctionDeclaration parameters as a plain dict to ``genai.types.FunctionDeclaration``,
    which accepts raw dicts for the schema argument.
    """
    result: dict[str, Any] = {}

    oa_type = prop.get("type", "string")
    gemini_type = _OPENAPI_TO_GEMINI_TYPE.get(oa_type, "STRING")
    result["type"] = gemini_type

    desc = prop.get("description", "")
    if desc:
        result["description"] = textwrap.shorten(desc, width=200, placeholder=" …")

    if prop.get("enum"):
        result["enum"] = [str(v) for v in prop["enum"]]

    if gemini_type == "ARRAY" and "items" in prop:
        items = prop["items"]
        item_type = _OPENAPI_TO_GEMINI_TYPE.get(items.get("type", "string"), "STRING")
        result["items"] = {"type": item_type}

    return result


# ---------------------------------------------------------------------------
# Tool schema loading  (Gemini format)
# ---------------------------------------------------------------------------

def load_catalog(catalog_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Load the operation catalog JSON."""
    path = Path(catalog_path) if catalog_path else _DEFAULT_CATALOG
    if not path.exists():
        raise FileNotFoundError(
            f"Operation catalog not found at {path}. "
            "Set CATALOG_PATH env-var or pass catalog_path explicitly."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_gemini_function_declaration(op: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one catalog operation into a Gemini FunctionDeclaration dict.

    Returns a plain dict with keys ``name``, ``description``, ``parameters``
    that can be passed to ``genai.types.FunctionDeclaration(**decl)``.

    Function name: operation_id with "/" → "__" and "-" → "_".
    E.g. "issues/create" → "issues__create".
    """
    operation_id = op["operation_id"]
    fn_name = operation_id.replace("/", "__").replace("-", "_")

    # ── description ──────────────────────────────────────────────────────
    summary = op.get("summary", "")
    description_raw = op.get("description", "")
    description_short = textwrap.shorten(description_raw, width=250, placeholder=" …")
    description = f"[{operation_id}] {summary}"
    if description_short:
        description += f"\n{description_short}"
    description += f"\nHTTP: {op['method']} {op['path']}  Risk: {op['risk']}"

    # ── build properties and required list ────────────────────────────────
    properties: dict[str, Any] = {}
    required: list[str] = []

    # Path/query parameters
    for raw_param in op.get("parameters", []):
        param = _resolve_param(raw_param)
        name = param.get("name")
        if not name:
            continue
        schema = param.get("schema", {})
        prop = _openapi_prop_to_gemini_schema(schema)
        prop["description"] = param.get("description", f"Parameter: {name}")
        if schema.get("enum"):
            prop["enum"] = [str(v) for v in schema["enum"]]
        properties[name] = prop
        if param.get("required", False):
            required.append(name)

    # Request body properties
    rb = op.get("request_body")
    if rb:
        try:
            schema_obj = rb["content"]["application/json"]["schema"]
            body_props = schema_obj.get("properties", {})
            body_required = schema_obj.get("required", [])
            for prop_name, prop_schema in body_props.items():
                prop = _openapi_prop_to_gemini_schema(prop_schema)
                desc = prop_schema.get("description", f"Body property: {prop_name}")
                prop["description"] = textwrap.shorten(desc, width=200, placeholder=" …")
                properties[prop_name] = prop
                if prop_name in body_required:
                    required.append(prop_name)
        except (KeyError, TypeError):
            pass

    # Gemini parameters schema (OpenAPI-style object schema as a plain dict)
    parameters_schema: dict[str, Any] = {
        "type": "OBJECT",
        "properties": properties,
    }
    if required:
        parameters_schema["required"] = list(dict.fromkeys(required))

    return {
        "name": fn_name,
        "description": description,
        "parameters": parameters_schema,
    }


def load_tool_schemas(
    catalog_path: Optional[str | Path] = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Load the catalog and build Gemini FunctionDeclaration dicts.

    Returns
    -------
    fn_decls : list[dict]
        List of FunctionDeclaration dicts (plain dicts, not genai objects).
        Pass these to ``_build_gemini_tool(fn_decls)`` to get the Tool object.
    op_index : dict[str, dict]
        Mapping of  fn_name  →  catalog operation dict, for execution routing.
    """
    catalog = load_catalog(catalog_path)
    fn_decls: list[dict[str, Any]] = []
    op_index: dict[str, dict[str, Any]] = {}

    for op in catalog["operations"]:
        decl = _build_gemini_function_declaration(op)
        fn_name = decl["name"]
        fn_decls.append(decl)
        op_index[fn_name] = op

    logger.info("Loaded %d function declarations from catalog", len(fn_decls))
    return fn_decls, op_index


def _build_gemini_tool(fn_decls: list[dict[str, Any]]) -> Any:
    """
    Wrap a list of FunctionDeclaration dicts into a single Gemini Tool object.

    Imported lazily so ``google-genai`` is not required at module-import time
    (lets unit tests stub the module).
    """
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    declarations = [
        types.FunctionDeclaration(**decl) for decl in fn_decls
    ]
    return types.Tool(function_declarations=declarations)


# ---------------------------------------------------------------------------
# Sandbox execution bridge  (unchanged from OpenAI version)
# ---------------------------------------------------------------------------

def _execute_tool_call(
    sandbox,
    op: dict[str, Any],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Route a single tool call through the GitHub sandbox.

    Parameters
    ----------
    sandbox : GitHubSandbox
        Initialised sandbox instance.
    op : dict
        The catalog operation entry.
    arguments : dict
        Arguments chosen by the LLM.

    Returns
    -------
    dict
        The sandbox ``execute()`` result dict.
    """
    operation_id = op["operation_id"]
    return execute_operation(sandbox, operation_id, arguments)


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def _get_gemini_client():
    """Lazily import and initialise the google-genai client."""
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'google-genai' package is not installed.\n"
            "Run: pip install google-genai"
        ) from exc

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.  "
            "Export it or add it to your .env file.\n"
            "Get a key at https://aistudio.google.com/app/apikey"
        )
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a GitHub task execution agent.

Your job is to select and call the appropriate GitHub API tools to complete \
the user's task. You have access to a catalog of GitHub REST API operations \
exposed as tools.

Guidelines:
- Select only the tools needed to accomplish the task.
- Provide all required arguments for each tool call.
- For operations that require 'owner' and 'repo', use the values provided \
in the task context.
- For 'git__create_ref', the 'ref' must be in the format 'refs/heads/<branch-name>' \
and 'sha' must be the commit SHA of the base branch.
- If a task requires multiple operations, call all the needed tools in one response.
- Do not perform operations not requested by the user.
- Prefer the minimum set of operations that satisfy the task.
"""


# ---------------------------------------------------------------------------
# Agent session
# ---------------------------------------------------------------------------

class AgentSession:
    """
    A single evaluation run of the single-prompt GitHub agent.

    Parameters
    ----------
    sandbox : GitHubSandbox
        Initialised sandbox instance used for tool execution.
    catalog_path : str | Path, optional
        Override catalog location.
    model : str, optional
        Gemini model name.  Defaults to GEMINI_MODEL env-var or "gemini-3.6-flash".
    """

    def __init__(
        self,
        sandbox,
        catalog_path: Optional[str | Path] = None,
        model: Optional[str] = None,
    ) -> None:
        self.sandbox = sandbox
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.fn_decls, self.op_index = load_tool_schemas(catalog_path)
        self._client = _get_gemini_client()
        logger.info(
            "AgentSession ready — model=%s  tools=%d",
            self.model,
            len(self.fn_decls),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, task: str, repo: Optional[str] = None) -> dict[str, Any]:
        """
        Execute a natural-language GitHub task with a single Gemini call.

        Parameters
        ----------
        task : str
            The user's natural-language task description.
        repo : str, optional
            The target repository in ``owner/repo`` format.

        Returns
        -------
        dict
            Structured result:
            - ``success``       : bool
            - ``task``          : str
            - ``model``         : str
            - ``tool_calls``    : list of tool call records
            - ``results``       : list of execution results
            - ``final_response``: str
            - ``error``         : str (only on failure)
        """
        # Build content string (Gemini uses a single 'contents' list)
        context_prefix = ""
        if repo:
            context_prefix = f"Target repository: {repo}\n\n"
        elif self.sandbox:
            context_prefix = f"GitHub owner: {self.sandbox._owner}\n\n"

        user_content = f"{context_prefix}Task: {task}"

        # Gemini contents: system instruction + user turn
        contents = [
            {"role": "user", "parts": [{"text": user_content}]},
        ]

        # Build the Gemini Tool object (wraps all FunctionDeclarations)
        gemini_tool = _build_gemini_tool(self.fn_decls)

        logger.info("Task: %s", task)
        logger.info("Available tools: %d", len(self.fn_decls))
        logger.debug("User content:\n%s", user_content)

        # ── THE SINGLE LLM CALL ──────────────────────────────────────────
        # This is the ONLY call to the LLM in this baseline agent.
        # No loop. No reflection. No retry.
        try:
            response = self._llm_call(contents, gemini_tool)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return {
                "success": False,
                "task": task,
                "model": self.model,
                "tool_calls": [],
                "results": [],
                "final_response": "",
                "error": f"LLM API error: {type(exc).__name__}: {exc}",
            }
        # ── END OF SINGLE LLM CALL ───────────────────────────────────────

        return self._process_response(task, response)

    # ------------------------------------------------------------------
    # Internal: LLM call (thin wrapper — easy to mock in tests)
    # ------------------------------------------------------------------

    def _llm_call(self, contents: list[dict], tool: Any) -> Any:
        """
        Send contents + tools to the Gemini API and return the raw response.

        Uses system_instruction for the agent persona and AUTO tool mode so
        Gemini decides which tools (if any) to call.
        """
        from google.genai import types  # type: ignore

        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="AUTO",
                )
            ),
        )
        return self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

    # ------------------------------------------------------------------
    # Internal: parse Gemini response
    # ------------------------------------------------------------------

    def _process_response(self, task: str, response: Any) -> dict[str, Any]:
        """
        Parse the Gemini GenerateContentResponse and execute all function calls.

        Gemini returns function calls as parts with a ``function_call`` field
        (type ``google.genai.types.FunctionCall``) alongside optional text parts.
        """
        tool_call_records: list[dict[str, Any]] = []
        execution_results: list[dict[str, Any]] = []
        overall_success = True
        text_parts: list[str] = []
        function_call_parts: list[Any] = []

        # Gather all parts from all candidates (usually 1 candidate)
        try:
            candidates = response.candidates or []
        except AttributeError:
            candidates = []

        for candidate in candidates:
            try:
                parts = candidate.content.parts or []
            except AttributeError:
                parts = []
            for part in parts:
                # Text part
                try:
                    if part.text:
                        text_parts.append(part.text)
                except AttributeError:
                    pass
                # Function call part
                try:
                    if part.function_call is not None:
                        function_call_parts.append(part.function_call)
                except AttributeError:
                    pass

        final_response = "\n".join(text_parts)
        logger.info("Gemini returned %d function call(s)", len(function_call_parts))

        for fc in function_call_parts:
            # fc is a google.genai.types.FunctionCall with .name and .args (dict)
            fn_name = fc.name
            arguments = dict(fc.args) if fc.args else {}

            # Resolve fn_name → catalog operation
            op = self.op_index.get(fn_name)
            if op is None:
                err_msg = f"Gemini called unknown function '{fn_name}'."
                logger.warning(err_msg)
                tool_call_records.append({
                    "function_name": fn_name,
                    "operation_id": "unknown",
                    "arguments": arguments,
                    "error": err_msg,
                })
                execution_results.append({"success": False, "error": err_msg})
                overall_success = False
                continue

            operation_id = op["operation_id"]
            logger.info(
                "Executing tool: %s  arguments=%s",
                operation_id,
                json.dumps(
                    {k: ("***" if "token" in k.lower() or "key" in k.lower() else v)
                     for k, v in arguments.items()},
                ),
            )

            tool_call_records.append({
                "function_name": fn_name,
                "operation_id": operation_id,
                "arguments": arguments,
            })

            # Execute through sandbox
            result = _execute_tool_call(self.sandbox, op, arguments)
            execution_results.append(result)

            if not result.get("success", False):
                overall_success = False
                logger.warning(
                    "Tool '%s' failed: %s",
                    operation_id,
                    result.get("message", result.get("error", "unknown")),
                )
            else:
                logger.info("Tool '%s' succeeded.", operation_id)

        if not function_call_parts and not final_response:
            final_response = "(The model returned no tool calls and no text response.)"

        return {
            "success": overall_success,
            "task": task,
            "model": self.model,
            "tool_calls": tool_call_records,
            "results": execution_results,
            "final_response": final_response,
        }


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_agent(
    task: str,
    repo: Optional[str] = None,
    sandbox=None,
    catalog_path: Optional[str | Path] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """
    Top-level convenience function for running the single-prompt agent.

    Parameters
    ----------
    task : str
        Natural-language GitHub task.
    repo : str, optional
        Target repository in ``owner/repo`` format.
    sandbox : GitHubSandbox, optional
        Reuse an existing sandbox instance.  A new one is created if None.
    catalog_path : str | Path, optional
        Override catalog location.
    model : str, optional
        Override Gemini model name.
    """
    if sandbox is None:
        _harness = Path(__file__).resolve().parent
        if str(_harness) not in sys.path:
            sys.path.insert(0, str(_harness))
        from github_sandbox import GitHubSandbox  # type: ignore
        sandbox = GitHubSandbox()

    session = AgentSession(sandbox, catalog_path=catalog_path, model=model)
    return session.run(task, repo=repo)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(result: dict[str, Any], debug: bool = False) -> None:
    """Pretty-print the agent result to stdout."""
    print()
    print("=" * 70)
    print(f"Task:  {result['task']}")
    print(f"Model: {result.get('model', '?')}")
    print("=" * 70)
    print()

    tool_calls = result.get("tool_calls", [])
    print(f"Available tools: 51")
    print()

    if tool_calls:
        print(f"Selected operations ({len(tool_calls)}):")
        for tc in tool_calls:
            status = "  ← ERROR" if "error" in tc else ""
            print(f"  - {tc['operation_id']}{status}")
            args = tc.get("arguments", {})
            if args:
                print(f"      arguments: {json.dumps(args)}")
        print()

    results = result.get("results", [])
    if results:
        print("Execution results:")
        for i, r in enumerate(results, 1):
            ok = r.get("success", False)
            symbol = "✓" if ok else "✗"
            op = tool_calls[i - 1]["operation_id"] if i <= len(tool_calls) else f"#{i}"
            print(f"  {symbol}  {op}")
            if not ok:
                msg = r.get("message") or r.get("error", "unknown error")
                print(f"      {msg}")
                if "get-a-repository" in msg or "Not Found" in msg:
                    repo_arg = tool_calls[i - 1].get("arguments", {}).get("repo") if i <= len(tool_calls) else None
                    if repo_arg:
                        print(f"      HINT: Repository '{repo_arg}' was not found on GitHub. Specify an existing repo with --repo <name> or create it first.")
            elif debug:
                print(f"      result type: {type(r.get('result')).__name__}")
        print()

    resp = result.get("final_response", "")
    if resp:
        print("Model response:")
        print(textwrap.indent(resp, "  "))
        print()

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        print()

    overall = "SUCCESS" if result.get("success") else "FAIL"
    print(f"Final result: {overall}")
    print("=" * 70)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="single_prompt_agent.py",
        description="Baseline single-prompt GitHub agent (Gemini, 200-Tools evaluation).",
    )
    parser.add_argument("task", help="Natural-language GitHub task to execute.")
    parser.add_argument("--repo", "-r", help="Target repository in owner/repo format.")
    parser.add_argument(
        "--model", "-m", default=None,
        help="Gemini model name (default: GEMINI_MODEL env-var or gemini-3.6-flash).",
    )
    parser.add_argument("--catalog", default=None,
                        help="Path to operation_catalog.json.")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Show full argument and result details.")
    parser.add_argument("--json", action="store_true",
                        help="Output full result as JSON.")
    parser.add_argument("--no-sandbox", action="store_true",
                        help="Skip sandbox init (tool execution will fail).")
    parser.add_argument("--list-tools", action="store_true",
                        help="List all available tool names and exit.")
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.list_tools:
        fn_decls, _ = load_tool_schemas(args.catalog)
        print(f"Available tools ({len(fn_decls)}):")
        for d in fn_decls:
            first_line = d["description"].splitlines()[0][:60]
            print(f"  {d['name']:45s}  {first_line}")
        return 0

    sandbox = None
    if not args.no_sandbox:
        _harness = Path(__file__).resolve().parent
        if str(_harness) not in sys.path:
            sys.path.insert(0, str(_harness))
        try:
            from github_sandbox import GitHubSandbox  # type: ignore
            sandbox = GitHubSandbox()
        except Exception as exc:
            print(f"ERROR: Could not initialise GitHub sandbox: {exc}", file=sys.stderr)
            return 1

    try:
        result = run_agent(
            task=args.task,
            repo=args.repo,
            sandbox=sandbox,
            catalog_path=args.catalog,
            model=args.model,
        )
    except Exception as exc:
        print(f"ERROR: Agent failed: {exc}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    if args.json:
        safe_result = json.loads(json.dumps(result, default=lambda o: str(o)))
        print(json.dumps(safe_result, indent=2))
    else:
        _print_result(result, debug=args.debug)

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
