"""OMEGA-THDSE integration test suite (Phase 2 + Phase 3).

Ensures ``shared/`` and ``bridges/`` are on ``sys.path`` when the
test files are collected by pytest regardless of the invocation
directory, so the ``shared`` and ``bridges`` package imports inside
each test module resolve cleanly.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
