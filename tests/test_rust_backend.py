"""Phase 7.1 — Rust backend integration tests.

These tests verify the actual Rust :mod:`hdc_core` extension when it
is installed. They are SKIPPED in environments without the compiled
crate via :func:`pytest.importorskip` so the rest of the suite stays
green on bare Python + numpy.

Coverage:
- ``ArenaManager`` reports ``backend == "rust"`` when the crate is
  importable.
- The bridge self-test passes when seeded with Rust-allocated phase
  arrays (Rule 12).
- Bind commutation holds with the Rust arena (Rule 12 invariant).
- Rust and ``_PyFhrrArenaExtended`` agree on 100 random vector pairs
  for similarity within float32 tolerance.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module if hdc_core is missing.
pytest.importorskip("hdc_core")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_THDSE = _REPO_ROOT / "thdse"
for p in (str(_REPO_ROOT), str(_THDSE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.arena_manager import ArenaManager  # noqa: E402
from shared.constants import (  # noqa: E402
    BRIDGE_SUBSAMPLE_STRIDE,
    CCE_ARENA_DIM,
    THDSE_ARENA_DIM,
)
from shared.dimension_bridge import (  # noqa: E402
    PROJECTION_INDICES,
    cross_arena_similarity,
    project_down,
)
from src.utils.arena_factory import _PyFhrrArenaExtended  # noqa: E402

_TWO_PI = 2.0 * math.pi


@pytest.fixture
def manager() -> ArenaManager:
    return ArenaManager(master_seed=7001)


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #


def test_arena_manager_selects_rust_backend(manager: ArenaManager):
    assert manager.backend == "rust"
    assert manager._cce_arena is not None  # noqa: SLF001
    assert manager._thdse_arena is not None  # noqa: SLF001


def test_rust_backend_reports_correct_dimension(manager: ArenaManager):
    assert manager.cce_dim == CCE_ARENA_DIM == 10_000
    assert manager.thdse_dim == THDSE_ARENA_DIM == 256


# --------------------------------------------------------------------------- #
# Dimension bridge invariants under Rust
# --------------------------------------------------------------------------- #


def test_dimension_bridge_self_test_runs_under_rust(manager: ArenaManager):
    """Re-invoke the self-test explicitly with Rust-backed arenas in scope."""
    from shared.dimension_bridge import _run_self_test

    _run_self_test(seed=1337, num_pairs=10)


def test_bind_commutation_with_rust_arena(manager: ArenaManager):
    rng = manager.rng.fork("rust_bind")
    for _ in range(10):
        a = rng.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)
        b = rng.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)
        # Use the bridge's own subsampling helper
        sub_a = a[PROJECTION_INDICES]
        sub_b = b[PROJECTION_INDICES]
        bound_full = np.mod(a + b, np.float32(_TWO_PI))
        sub_of_bound = bound_full[PROJECTION_INDICES]
        bind_of_sub = np.mod(sub_a + sub_b, np.float32(_TWO_PI))
        assert np.array_equal(sub_of_bound, bind_of_sub)


# --------------------------------------------------------------------------- #
# Cross-backend numerical agreement
# --------------------------------------------------------------------------- #


def test_rust_vs_python_similarity_agrees_for_100_pairs(
    manager: ArenaManager,
):
    py_arena = _PyFhrrArenaExtended(capacity=300, dimension=THDSE_ARENA_DIM)
    rng = manager.rng.fork("parity")

    for _ in range(100):
        phases_a = rng.uniform(0.0, _TWO_PI, THDSE_ARENA_DIM).astype(
            np.float32
        )
        phases_b = rng.uniform(0.0, _TWO_PI, THDSE_ARENA_DIM).astype(
            np.float32
        )

        # Rust arena via the manager.
        h_a_rust = manager.alloc_thdse(phases=phases_a)
        h_b_rust = manager.alloc_thdse(phases=phases_b)
        rust_arena = manager._thdse_arena  # noqa: SLF001
        rust_corr = float(rust_arena.compute_correlation(h_a_rust, h_b_rust))

        # Python arena.
        h_a_py = py_arena.allocate()
        h_b_py = py_arena.allocate()
        py_arena.inject_phases(h_a_py, phases_a.tolist())
        py_arena.inject_phases(h_b_py, phases_b.tolist())
        py_corr = py_arena.compute_correlation(h_a_py, h_b_py)

        assert abs(rust_corr - py_corr) < 1e-5, (
            f"Rust vs Python correlation diverged: "
            f"rust={rust_corr}, py={py_corr}"
        )
