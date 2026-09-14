"""
orchestrator/operation_registry.py
===================================
Wraps operation_catalog.json + github_operations.py into a typed registry.

This is a read-only view over the already-initialised operation system.
It does NOT duplicate schemas, definitions, or execution handlers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from orchestrator.config import IRREVERSIBLE_OPERATIONS
from orchestrator.schemas import OperationDefinition

logger = logging.getLogger(__name__)


class OperationRegistry:
    """
    Registry of all GitHub operations available to the orchestration agent.

    Wraps harness/github_operations.py (execution handlers + definitions)
    and schemas/operation_catalog.json (category, subcategory metadata).
    """

    def __init__(
        self,
        catalog_path: Optional[str | Path] = None,
        operations_module: Any = None,
    ) -> None:
        """
        Parameters
        ----------
        catalog_path : str | Path, optional
            Path to operation_catalog.json. Auto-detected if omitted.
        operations_module : module, optional
            The github_operations module. Imported automatically if omitted.
        """
        # Lazy import to avoid circular deps at module level
        if operations_module is None:
            import sys
            _project_root = Path(__file__).resolve().parent.parent
            _harness_dir = _project_root / "harness"
            if str(_harness_dir) not in sys.path:
                sys.path.insert(0, str(_harness_dir))
            try:
                from harness import github_operations as _ops
            except ImportError:
                import github_operations as _ops  # type: ignore
            operations_module = _ops

        self._ops_module = operations_module

        # Load catalog for category/subcategory metadata
        if catalog_path is None:
            _project_root = Path(__file__).resolve().parent.parent
            catalog_path = _project_root / "schemas" / "operation_catalog.json"

        self._catalog_path = Path(catalog_path)
        self._catalog_index: dict[str, dict[str, Any]] = {}
        self._load_catalog()

        # Cache OperationDefinition objects
        self._definitions: dict[str, OperationDefinition] = {}
        self._build_definitions()

    def _load_catalog(self) -> None:
        """Load category/subcategory metadata from the catalog JSON."""
        if not self._catalog_path.exists():
            logger.warning("Catalog not found at %s", self._catalog_path)
            return
        with open(self._catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        for op in catalog.get("operations", []):
            self._catalog_index[op["operation_id"]] = {
                "category": op.get("category", ""),
                "subcategory": op.get("subcategory", ""),
            }

    def _build_definitions(self) -> None:
        """Build OperationDefinition objects from the operations module."""
        for op_id in self._ops_module.list_operations():
            try:
                defn = self._ops_module.get_operation_definition(op_id)
            except (ValueError, KeyError):
                logger.warning("Failed to load definition for %s", op_id)
                continue

            catalog_meta = self._catalog_index.get(op_id, {})

            self._definitions[op_id] = OperationDefinition(
                operation_id=op_id,
                name=defn.get("name", op_id),
                method=defn.get("method", "GET"),
                path=defn.get("path", ""),
                category=catalog_meta.get("category", ""),
                subcategory=catalog_meta.get("subcategory", ""),
                risk=defn.get("risk", "medium"),
                summary=defn.get("summary", ""),
                description=defn.get("description", ""),
                input_schema=defn.get("input_schema", {}),
                starter_request=defn.get("starter_request", {}),
                is_irreversible=(op_id in IRREVERSIBLE_OPERATIONS),
            )

    # ------------------------------------------------------------------
    # Public API — implemented now (all_loaded)
    # ------------------------------------------------------------------

    def get_all(self) -> list[OperationDefinition]:
        """Every operation in the catalog. Used by all_loaded mode."""
        return list(self._definitions.values())

    def get(self, operation_id: str) -> Optional[OperationDefinition]:
        """
        Exact lookup. Returns None if unknown — caller raises UNKNOWN_OPERATION.

        Tries canonical ID first, then alias resolution.
        """
        if operation_id in self._definitions:
            return self._definitions[operation_id]

        # Try alias resolution via the operations module
        try:
            if hasattr(self._ops_module, "_resolve_operation_id"):
                canonical = self._ops_module._resolve_operation_id(operation_id)
            else:
                meta = self._ops_module.get_operation_definition(operation_id)
                canonical = meta.get("operation_id")
            return self._definitions.get(canonical)
        except (ValueError, AttributeError):
            return None

    # ------------------------------------------------------------------
    # category_gated support — implemented
    # ------------------------------------------------------------------

    def get_by_category(self, category: str) -> list[OperationDefinition]:
        """
        Return all operations whose category matches *category* (case-insensitive).

        Used by category_gated mode.  Derived from the already-loaded
        _definitions dict — no extra I/O required.

        Returns an empty list if no operations belong to the category;
        callers must treat an empty result as an error, not a silent no-op.
        """
        needle = category.lower()
        return [
            op for op in self._definitions.values()
            if op.category.lower() == needle
        ]

    def get_all_categories(self) -> list[str]:
        """
        Return a sorted list of unique non-empty category names across all
        loaded operations.

        Derived live from _definitions so it can never drift out of sync
        with the catalog.
        """
        return sorted(
            {op.category for op in self._definitions.values() if op.category}
        )

    # ------------------------------------------------------------------
    # Deferred — do NOT implement bodies. Signatures only.
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 8) -> list[OperationDefinition]:
        """Semantic search over operations. Used by search_then_load mode."""
        raise NotImplementedError("search_then_load not yet implemented")

    def classify_domain(self, objective: str) -> str:
        """Classify task objective into a domain. Used by hierarchical_planner."""
        raise NotImplementedError("hierarchical_planner not yet implemented")

    def get_by_domain(self, domain: str) -> list[OperationDefinition]:
        """Return operations for a domain. Used by hierarchical_planner."""
        raise NotImplementedError("hierarchical_planner not yet implemented")
