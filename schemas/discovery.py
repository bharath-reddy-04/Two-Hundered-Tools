"""
schemas/discovery.py
====================
Re-exports the discovery module from discovery.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from discovery import get_schema, list_operations

__all__ = ["get_schema", "list_operations"]
