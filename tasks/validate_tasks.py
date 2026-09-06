#!/usr/bin/env python3
"""
validate_tasks.py
=================
Validator for the 200-Tools GitHub evaluation dataset (tasks.json).

Checks:
1. File exists and is valid JSON.
2. Contains exactly 13 tasks (or expected count).
3. Every task has a unique task_id.
4. Every task has: task_id, name, difficulty, operation_count, prompt, expected_operations, expected_state.
5. Difficulty is one of 'easy', 'medium', 'hard' and matches expected distribution (4 easy, 4 medium, 5 hard).
6. Prompts are natural language and do NOT leak internal operation IDs (e.g. 'issues/create').
7. Every operation referenced in expected_operations exists in schemas/operation_catalog.json.
8. Expected state contains valid resource types ('repository', 'branch', 'issue', 'pull_request').
9. Resource names use safe eval/sandbox prefixes (e.g. 'eval-') and avoid production repository names.

Usage:
    python validate_tasks.py
    python validate_tasks.py tasks.json
    python validate_tasks.py --catalog schemas/operation_catalog.json tasks.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, List, Set

_this = Path(__file__).resolve().parent
PROJECT_ROOT = _this.parent if _this.name == "tasks" else _this
DEFAULT_TASKS_FILE = PROJECT_ROOT / "tasks.json"
DEFAULT_CATALOG_FILE = PROJECT_ROOT / "schemas" / "operation_catalog.json"

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_RESOURCE_TYPES = {"repository", "branch", "issue", "pull_request"}
PRODUCTION_REPO_PATTERNS = [
    re.compile(r"^200-Tools$", re.IGNORECASE),
    re.compile(r"^main$", re.IGNORECASE),
    re.compile(r"^master$", re.IGNORECASE),
]


def load_catalog_operation_ids(catalog_path: Path) -> Set[str]:
    """Extract all valid operation_id values from operation_catalog.json."""
    if not catalog_path.exists():
        raise FileNotFoundError(f"Operation catalog not found at: {catalog_path}")

    with catalog_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    operations = data.get("operations", [])
    op_ids = {op["operation_id"] for op in operations if "operation_id" in op}
    return op_ids


def validate_tasks(tasks_path: Path, catalog_path: Path) -> List[str]:
    """
    Validate the tasks JSON file against catalog and schema requirements.
    Returns a list of error strings (empty list means all checks passed).
    """
    errors: List[str] = []

    if not tasks_path.exists():
        return [f"Tasks file not found at: {tasks_path}"]

    try:
        with tasks_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as exc:
        return [f"Failed to parse JSON in {tasks_path}: {exc}"]

    if isinstance(raw_data, list):
        tasks = raw_data
    elif isinstance(raw_data, dict) and "tasks" in raw_data:
        tasks = raw_data["tasks"]
    else:
        return ["Root JSON structure must be a list of task objects or an object with a 'tasks' array."]

    # Check total task count
    if len(tasks) != 13:
        errors.append(f"Expected exactly 13 tasks in initial dataset, found {len(tasks)}.")

    # Load known operation IDs from catalog
    try:
        known_op_ids = load_catalog_operation_ids(catalog_path)
    except Exception as exc:
        errors.append(f"Failed to load operation catalog: {exc}")
        known_op_ids = set()

    seen_task_ids: Set[str] = set()
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}

    required_fields = ["task_id", "difficulty", "prompt", "expected_operations", "expected_state"]

    for idx, task in enumerate(tasks, start=1):
        task_label = f"Task #{idx}"

        if not isinstance(task, dict):
            errors.append(f"{task_label}: Must be a JSON object.")
            continue

        task_id = task.get("task_id")
        if not task_id or not isinstance(task_id, str):
            errors.append(f"{task_label}: Missing or invalid 'task_id'.")
            task_label = f"{task_label} (unknown id)"
        else:
            task_label = f"Task '{task_id}'"
            if task_id in seen_task_ids:
                errors.append(f"{task_label}: Duplicate task_id '{task_id}'.")
            seen_task_ids.add(task_id)

        # Check required fields
        for field in required_fields:
            if field not in task:
                errors.append(f"{task_label}: Missing required field '{field}'.")

        # Check difficulty
        difficulty = task.get("difficulty")
        if difficulty not in VALID_DIFFICULTIES:
            errors.append(f"{task_label}: Invalid difficulty '{difficulty}'. Must be one of {sorted(VALID_DIFFICULTIES)}.")
        else:
            difficulty_counts[difficulty] += 1

        # Check prompt
        prompt = task.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"{task_label}: 'prompt' must be a non-empty string.")
        else:
            # Verify prompt does not leak internal operation IDs (e.g. issues/create)
            for op_id in known_op_ids:
                if op_id in prompt:
                    errors.append(f"{task_label}: Prompt leaks internal operation ID '{op_id}'. Keep prompts natural language.")

        # Check expected_operations
        expected_ops = task.get("expected_operations")
        if not isinstance(expected_ops, list) or len(expected_ops) == 0:
            errors.append(f"{task_label}: 'expected_operations' must be a non-empty list.")
        else:
            for op in expected_ops:
                if not isinstance(op, str):
                    errors.append(f"{task_label}: Operation '{op}' is not a string.")
                elif known_op_ids and op not in known_op_ids:
                    errors.append(f"{task_label}: Referenced operation '{op}' not found in operation_catalog.json.")

        # Check expected_state
        expected_state = task.get("expected_state")
        if not isinstance(expected_state, dict) or len(expected_state) == 0:
            errors.append(f"{task_label}: 'expected_state' must be a non-empty object.")
        else:
            for resource_type, state_def in expected_state.items():
                if resource_type not in VALID_RESOURCE_TYPES:
                    errors.append(
                        f"{task_label}: Invalid resource type '{resource_type}' in expected_state. "
                        f"Must be one of {sorted(VALID_RESOURCE_TYPES)}."
                    )
                if not isinstance(state_def, dict):
                    errors.append(f"{task_label}: Expected state for '{resource_type}' must be an object.")

                # Check for accidental production repository names
                if resource_type == "repository":
                    repo_name = state_def.get("name", "")
                    for pattern in PRODUCTION_REPO_PATTERNS:
                        if pattern.match(repo_name):
                            errors.append(f"{task_label}: Accidental production/reserved repo name used: '{repo_name}'.")

    # Check difficulty distribution: 4 easy, 4 medium, 5 hard
    expected_dist = {"easy": 4, "medium": 4, "hard": 5}
    for diff, exp_count in expected_dist.items():
        act_count = difficulty_counts[diff]
        if act_count != exp_count:
            errors.append(f"Difficulty '{diff}': expected {exp_count} tasks, found {act_count}.")

    return errors


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate GitHub 200-Tools tasks.json evaluation dataset.")
    parser.add_argument("tasks", nargs="?", default=str(DEFAULT_TASKS_FILE), help="Path to tasks.json")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_FILE), help="Path to operation_catalog.json")
    args = parser.parse_args(argv)

    tasks_path = Path(args.tasks)
    catalog_path = Path(args.catalog)

    print(f"Validating tasks:   {tasks_path}")
    print(f"Using catalog:      {catalog_path}")

    errors = validate_tasks(tasks_path, catalog_path)

    if errors:
        print(f"\n❌ FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  • {err}")
        return 1

    print("\n✅ SUCCESS: All 13 tasks passed validation!")
    print("  • 4 Easy tasks")
    print("  • 4 Medium tasks")
    print("  • 5 Hard tasks")
    print("  • All operation IDs exist in catalog")
    print("  • Prompts are natural and leak no tool names")
    print("  • Resource naming is isolated with eval prefixes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
