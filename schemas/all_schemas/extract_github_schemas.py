#!/usr/bin/env python3
"""
Extract exactly 50 GitHub REST API operations from the official GitHub OpenAPI spec.

Input:
    github_openapi.json

Output:
    github_operation_catalog.json

Usage:
    python extract_github_schemas.py github_openapi.json
    python extract_github_schemas.py github_openapi.json github_operation_catalog.json

The extractor keeps the complete OpenAPI operation definition for each selected
operation, including parameters, requestBody, responses, operationId, and
documentation metadata. This makes the output suitable for building an
agent/tool operation catalog.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# Exactly 50 UNIQUE GitHub REST API operationIds.
# These are selected to cover read, create, update, merge, permission,
# branch, content, issue, pull-request, commit, and Actions operations.
TARGET_OPERATION_IDS = [
    # ---------------- Repository: 10 ----------------
    "repos/create-for-authenticated-user",
    "repos/get",
    "repos/update",
    "repos/delete",
    "repos/list-for-authenticated-user",
    "repos/list-collaborators",
    "repos/add-collaborator",
    "repos/remove-collaborator",
    "repos/get-all-topics",
    "repos/replace-all-topics",

    # ---------------- Branch/Git refs: 7 ----------------
    "repos/list-branches",
    "repos/get-branch",
    "git/create-ref",
    "git/delete-ref",
    "repos/rename-branch",
    "repos/merge",
    "repos/compare-commits",

    # ---------------- Repository contents: 6 ----------------
    "repos/get-content",
    "repos/create-or-update-file-contents",
    "repos/delete-file",
    "repos/get-readme",
    "repos/list-commits",
    "repos/get-commit",

    # ---------------- Git database: 4 ----------------
    "git/get-tree",
    "git/get-blob",
    "git/get-ref",
    "git/list-matching-refs",

    # ---------------- Pull requests: 8 ----------------
    "pulls/create",
    "pulls/get",
    "pulls/list",
    "pulls/update",
    "pulls/merge",
    "pulls/list-files",
    "pulls/list-commits",
    "pulls/list-review-comments",

    # ---------------- Issues: 7 ----------------
    "issues/create",
    "issues/get",
    "issues/list",
    "issues/update",
    "issues/create-comment",
    "issues/list-comments",
    "issues/lock",

    # ---------------- GitHub Actions: 8 ----------------
    "actions/list-repo-workflows",
    "actions/list-workflow-runs-for-repo",
    "actions/get-workflow-run",
    "actions/create-workflow-dispatch",
    "actions/cancel-workflow-run",
    "actions/re-run-workflow",
    "actions/re-run-workflow-failed-jobs",
    "actions/download-workflow-run-logs",

    # ---------------- User: 1 ----------------
    "users/get-authenticated",
]

# Risk is application-level metadata for your agent's approval policy.
# It is NOT taken from GitHub's OpenAPI specification.
RISK_BY_METHOD = {
    "get": "low",
    "post": "medium",
    "put": "medium",
    "patch": "medium",
    "delete": "high",
}


def load_openapi(path: Path) -> dict[str, Any]:
    """Load a JSON OpenAPI document."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("OpenAPI document must be a JSON object.")

    if "paths" not in data:
        raise ValueError("OpenAPI document does not contain a 'paths' object.")

    return data


def collect_operations(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Flatten spec['paths'] into:
        operationId -> operation metadata

    Non-operation path keys such as parameters are ignored.
    """
    operations: dict[str, dict[str, Any]] = {}

    for path, path_item in spec.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue

        for method, operation in path_item.items():
            if method.lower() not in {
                "get", "post", "put", "patch", "delete", "head", "options", "trace"
            }:
                continue

            if not isinstance(operation, dict):
                continue

            operation_id = operation.get("operationId")
            if not operation_id:
                continue

            if operation_id in operations:
                raise ValueError(
                    f"Duplicate operationId found in OpenAPI spec: {operation_id}"
                )

            operations[operation_id] = {
                "method": method.upper(),
                "path": path,
                "operation": operation,
            }

    return operations


def classify_risk(method: str, operation_id: str) -> str:
    """
    Add useful approval-policy metadata.

    DELETE operations are high risk.
    Explicit merge/delete/permission-changing actions are also treated as high.
    Everything else follows HTTP-method defaults.
    """
    oid = operation_id.lower()

    high_risk_terms = (
        "delete",
        "remove-collaborator",
        "merge",
        "cancel-workflow-run",
        "rerun-workflow",
        "rerun-failed-jobs",
        "lock",
    )

    if method.upper() == "DELETE" or any(term in oid for term in high_risk_terms):
        return "high"

    return RISK_BY_METHOD.get(method.lower(), "medium")


def build_catalog(spec: dict[str, Any]) -> dict[str, Any]:
    operations = collect_operations(spec)

    missing = [
        operation_id
        for operation_id in TARGET_OPERATION_IDS
        if operation_id not in operations
    ]

    if missing:
        print("\nERROR: The following requested operationIds were not found:\n")
        for operation_id in missing:
            print(f"  - {operation_id}")

        print(
            "\nThis usually means the GitHub OpenAPI file is a different version. "
            "Do not silently substitute endpoints; update TARGET_OPERATION_IDS "
            "after checking the operationIds in your spec."
        )
        raise SystemExit(2)

    selected = []

    for index, operation_id in enumerate(TARGET_OPERATION_IDS, start=1):
        item = operations[operation_id]
        operation = item["operation"]

        selected.append(
            {
                "id": index,
                "name": operation_id.replace("/", "_"),
                "operation_id": operation_id,
                "method": item["method"],
                "path": item["path"],
                "summary": operation.get("summary", ""),
                "description": operation.get("description", ""),
                "category": (
                    operation.get("x-github", {}).get("category")
                    if isinstance(operation.get("x-github"), dict)
                    else None
                ),
                "subcategory": (
                    operation.get("x-github", {}).get("subcategory")
                    if isinstance(operation.get("x-github"), dict)
                    else None
                ),
                "risk": classify_risk(item["method"], operation_id),

                # Keep the actual OpenAPI pieces needed to call the operation.
                "parameters": operation.get("parameters", []),
                "request_body": operation.get("requestBody"),
                "responses": operation.get("responses", {}),

                # Useful for your agent to expose official documentation.
                "external_docs": operation.get("externalDocs"),
            }
        )

    return {
        "catalog_version": "1.0",
        "source": {
            "title": spec.get("info", {}).get("title"),
            "version": spec.get("info", {}).get("version"),
            "servers": spec.get("servers", []),
        },
        "operation_count": len(selected),
        "operations": selected,
    }

def main() -> None:
    INPUT_FILE = Path(
        "/Users/bharathreddy/200-Tools/schemas/all_schemas/api.github.com.json"
    )

    OUTPUT_FILE = Path(
        "/Users/bharathreddy/200-Tools/schemas/operation_catalog.json"
    )

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    spec = load_openapi(INPUT_FILE)

    

    catalog = build_catalog(spec)

    # Make sure output directory exists.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Extracted {catalog['operation_count']} operations.")
    print(f"Written to: {OUTPUT_FILE}")

    print("\nSelected operations:")
    for op in catalog["operations"]:
        print(
            f"{op['id']:02d}. {op['method']:6s} "
            f"{op['path']:65s} "
            f"{op['operation_id']} "
            f"[{op['risk']}]"
        )

if __name__ == "__main__":
    main()
