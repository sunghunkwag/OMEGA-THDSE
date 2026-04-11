"""Phase 6 — THDSE-side Rule 3 compliance audit.

PLAN.md Rule 3 forbids any direct call to ``hdc_core.FhrrArena(...)``
outside ``shared/arena_manager.py``. This audit walks every Python
file under ``thdse/`` and confirms that:

1. Zero ``FhrrArena(`` call sites survive (Rule 3).
2. Every ``import hdc_core`` is wrapped in a ``try:`` block so missing
   Rust backends do not crash legacy code paths (Rule 5).
3. The ``thdse/src/utils/arena_factory.py`` shim exists and exports
   ``make_arena`` and ``_PyFhrrArenaExtended``.
4. The Python fallback supports the full surface area THDSE expects
   (allocate, inject_phases, extract_phases, bind, bundle,
   compute_correlation, correlate_matrix, bind_bundle_fusion,
   expand_dimension, get_op_counts).

These tests run in any environment with just Python + numpy. They do
NOT require the Rust crate or Z3.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Tuple

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_THDSE = _REPO_ROOT / "thdse"


def _iter_thdse_py() -> List[Path]:
    return [
        p
        for p in sorted(_THDSE.rglob("*.py"))
        if "__pycache__" not in p.parts
    ]


# --------------------------------------------------------------------------- #
# Rule 3 — no direct FhrrArena calls
# --------------------------------------------------------------------------- #


_FHRR_CALL = re.compile(r"FhrrArena\(")


def test_no_direct_fhrr_arena_call_in_thdse():
    hits: List[Tuple[Path, int, str]] = []
    for path in _iter_thdse_py():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _FHRR_CALL.search(line):
                hits.append((path, lineno, line.strip()))
    assert hits == [], f"Rule 3 violations in thdse/: {hits}"


# --------------------------------------------------------------------------- #
# Rule 5 — every ``import hdc_core`` is inside a try: block
# --------------------------------------------------------------------------- #


def _hdc_core_imports_in_try_block(path: Path) -> Tuple[int, int]:
    """Return ``(total, in_try)`` counts for ``import hdc_core`` statements."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return (0, 0)

    # Collect node IDs of imports inside try blocks.
    in_try_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    in_try_ids.add(id(child))

    total = 0
    in_try = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "hdc_core":
                    total += 1
                    if id(node) in in_try_ids:
                        in_try += 1
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "hdc_core":
                total += 1
                if id(node) in in_try_ids:
                    in_try += 1
    return (total, in_try)


def test_every_hdc_core_import_is_try_wrapped():
    bare: List[Tuple[Path, int]] = []
    total_imports = 0
    for path in _iter_thdse_py():
        total, in_try = _hdc_core_imports_in_try_block(path)
        if total > in_try:
            bare.append((path, total - in_try))
        total_imports += total
    assert bare == [], (
        f"Rule 5 violations: bare ``import hdc_core`` outside try: blocks "
        f"in {bare}"
    )
    # Make sure the test actually inspected something — if zero imports
    # exist anywhere, the audit is vacuous.
    assert total_imports >= 1


# --------------------------------------------------------------------------- #
# arena_factory shim exists and exports the expected names
# --------------------------------------------------------------------------- #


def test_arena_factory_module_present_and_exports():
    factory_path = _THDSE / "src" / "utils" / "arena_factory.py"
    assert factory_path.is_file(), "thdse/src/utils/arena_factory.py missing"
    text = factory_path.read_text(encoding="utf-8")
    assert "def make_arena(" in text
    assert "class _PyFhrrArenaExtended" in text
    # Rule 3: factory must not import hdc_core or call FhrrArena.
    assert "import hdc_core" not in text
    assert "FhrrArena(" not in text


def test_python_fallback_arena_supports_full_api():
    import sys

    sys.path.insert(0, str(_THDSE))
    from src.utils.arena_factory import _PyFhrrArenaExtended  # noqa: E402

    arena = _PyFhrrArenaExtended(capacity=64, dimension=32)
    expected_methods = (
        "allocate",
        "inject_phases",
        "extract_phases",
        "get_phases",
        "bind",
        "bundle",
        "compute_correlation",
        "correlate_matrix",
        "bind_bundle_fusion",
        "expand_dimension",
        "get_op_counts",
        "record_bind_cost",
        "record_bundle_cost",
        "reset",
    )
    missing = [m for m in expected_methods if not hasattr(arena, m)]
    assert missing == [], f"_PyFhrrArenaExtended missing methods: {missing}"


def test_python_fallback_bind_is_correct():
    import math
    import sys

    sys.path.insert(0, str(_THDSE))
    from src.utils.arena_factory import _PyFhrrArenaExtended  # noqa: E402

    arena = _PyFhrrArenaExtended(capacity=8, dimension=4)
    h_a = arena.allocate()
    h_b = arena.allocate()
    h_out = arena.allocate()
    arena.inject_phases(h_a, [0.1, 0.2, 0.3, 0.4])
    arena.inject_phases(h_b, [1.0, 1.1, 1.2, 1.3])
    arena.bind(h_a, h_b, h_out)
    expected = [(0.1 + 1.0) % (2 * math.pi)] * 1 + [
        (0.2 + 1.1) % (2 * math.pi),
        (0.3 + 1.2) % (2 * math.pi),
        (0.4 + 1.3) % (2 * math.pi),
    ]
    bound = arena.extract_phases(h_out)
    for got, want in zip(bound, expected):
        assert got == pytest.approx(want, abs=1e-6)


def test_python_fallback_correlation_self_is_one():
    import sys

    sys.path.insert(0, str(_THDSE))
    from src.utils.arena_factory import _PyFhrrArenaExtended  # noqa: E402

    arena = _PyFhrrArenaExtended(capacity=4, dimension=8)
    h = arena.allocate()
    arena.inject_phases(h, [0.7] * 8)
    assert arena.compute_correlation(h, h) == pytest.approx(1.0, abs=1e-6)


def test_python_fallback_correlate_matrix_is_symmetric():
    import sys

    sys.path.insert(0, str(_THDSE))
    from src.utils.arena_factory import _PyFhrrArenaExtended  # noqa: E402

    arena = _PyFhrrArenaExtended(capacity=8, dimension=16)
    handles = []
    for value in (0.1, 0.5, 1.2):
        h = arena.allocate()
        arena.inject_phases(h, [value] * 16)
        handles.append(h)
    matrix = arena.correlate_matrix(handles)
    n = len(handles)
    for i in range(n):
        for j in range(n):
            assert matrix[i][j] == pytest.approx(matrix[j][i], abs=1e-9)
            if i == j:
                assert matrix[i][j] == pytest.approx(1.0, abs=1e-6)


def test_make_arena_returns_python_fallback_without_manager():
    import sys

    sys.path.insert(0, str(_THDSE))
    from src.utils.arena_factory import (  # noqa: E402
        _PyFhrrArenaExtended,
        make_arena,
    )

    arena = make_arena(capacity=10, dimension=4)
    assert isinstance(arena, _PyFhrrArenaExtended)
    assert arena.capacity == 10
    assert arena.dimension == 4


def test_make_arena_uses_manager_arena_when_rust_backend_available():
    """If the manager backend is rust AND dim matches, return the Rust arena."""
    import sys

    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_THDSE))
    from shared.arena_manager import ArenaManager  # noqa: E402
    from shared.constants import THDSE_ARENA_DIM  # noqa: E402
    from src.utils.arena_factory import make_arena  # noqa: E402

    mgr = ArenaManager(master_seed=4242)
    if mgr.backend != "rust":
        pytest.skip("Rust backend not installed; manager-borrow path not exercised")
    arena = make_arena(
        capacity=mgr.thdse_capacity,
        dimension=THDSE_ARENA_DIM,
        arena_manager=mgr,
    )
    # The factory should hand back the manager's actual Rust arena
    # object — same identity as ``mgr._thdse_arena``.
    assert arena is mgr._thdse_arena
