"""
github_operations.py
====================
Root re-export of harness/github_operations.py.
Provides direct access to all GitHub operations, definitions, and execute_operation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parent / "harness"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

from harness.github_operations import *  # noqa: F401, F403
from harness.github_operations import _resolve_operation_id  # noqa: F401
