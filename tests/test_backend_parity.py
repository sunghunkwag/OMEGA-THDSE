"""Phase 7.2 — Cross-backend parity tests.

Verify that ``_PyFhrrArenaExtended`` produces numerically equivalent
results to the Rust ``hdc_core.FhrrArena`` for every operation THDSE
relies on. The tests run with the Python fallback always; the Rust
arm is skipped when the crate is missing via :func:`pytest.importorskip`.

Each operation is exercised over a deterministic sweep of 100 random
phase vectors. The accepted absolute tolerance is ``1e-5`` to account
for SIMD float32 ordering differences in the Rust arena.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_THDSE = _REPO_ROOT / "thdse"
for p in (str(_REPO_ROOT), str(_THDSE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.deterministic_rng import DeterministicRNG  # noqa: E402
from src.utils.arena_factory import _PyFhrrArenaExtended  # noqa: E402

_TWO_PI = 2.0 * math.pi
_DIM = 64
_CAP = 400
_TOL = 1e-5


def _make_rng(namespace: str) -> np.random.Generator:
    return DeterministicRNG(master_seed=7100).fork(namespace)


def _alloc_with_phases(arena, phases) -> int:
    handle = arena.allocate()
    arena.inject_phases(handle, list(phases))
    return handle


@pytest.fixture
def py_arena() -> _PyFhrrArenaExtended:
    return _PyFhrrArenaExtended(capacity=_CAP, dimension=_DIM)


@pytest.fixture
def rust_arena():
    hdc_core = pytest.importorskip("hdc_core")
    return hdc_core.FhrrArena(_CAP, _DIM)


# --------------------------------------------------------------------------- #
# allocate / inject / extract
# --------------------------------------------------------------------------- #


def test_allocate_returns_increasing_handles(
    py_arena: _PyFhrrArenaExtended, rust_arena
):
    py_handles = [py_arena.allocate() for _ in range(5)]
    rust_handles = [rust_arena.allocate() for _ in range(5)]
    assert py_handles == [0, 1, 2, 3, 4]
    assert list(rust_handles) == [0, 1, 2, 3, 4]


def test_inject_extract_round_trip_matches(
    py_arena: _PyFhrrArenaExtended, rust_arena
):
    rng = _make_rng("inject_extract")
    for _ in range(10):
        phases = rng.uniform(0.0, _TWO_PI, _DIM).astype(np.float32)
        py_h = _alloc_with_phases(py_arena, phases)
        rust_h = _alloc_with_phases(rust_arena, phases)
        py_back = np.asarray(py_arena.extract_phases(py_h))
        rust_back = np.asarray(rust_arena.extract_phases(rust_h))
        # Both should match the input within float32 epsilon.
        assert np.allclose(py_back, phases, atol=_TOL)
        assert np.allclose(rust_back, phases, atol=_TOL)


# --------------------------------------------------------------------------- #
# bind / bundle parity
# --------------------------------------------------------------------------- #


def test_bind_results_match_within_tolerance(
    py_arena: _PyFhrrArenaExtended, rust_arena
):
    rng = _make_rng("bind_parity")
    for _ in range(100):
        a = rng.uniform(0.0, _TWO_PI, _DIM).astype(np.float32)
        b = rng.uniform(0.0, _TWO_PI, _DIM).astype(np.float32)
        py_a = _alloc_with_phases(py_arena, a)
        py_b = _alloc_with_phases(py_arena, b)
        py_out = py_arena.allocate()
        py_arena.bind(py_a, py_b, py_out)
        py_phases = np.asarray(py_arena.extract_phases(py_out))

        rust_a = _alloc_with_phases(rust_arena, a)
        rust_b = _alloc_with_phases(rust_arena, b)
        rust_out = rust_arena.allocate()
        rust_arena.bind(rust_a, rust_b, rust_out)
        rust_phases = np.asarray(rust_arena.extract_phases(rust_out))

        # Compare via cosine of phase difference (mod 2π safe).
        cos_diff = np.cos(py_phases - rust_phases)
        assert float(np.min(cos_diff)) > 1.0 - _TOL


def test_bundle_results_match_within_tolerance(
    py_arena: _PyFhrrArenaExtended, rust_arena
):
    rng = _make_rng("bundle_parity")
    for _ in range(50):
        bundles = []
        for _ in range(4):
            phases = rng.uniform(0.0, _TWO_PI, _DIM).astype(np.float32)
            bundles.append(phases)
        py_handles = [_alloc_with_phases(py_arena, p) for p in bundles]
        rust_handles = [_alloc_with_phases(rust_arena, p) for p in bundles]
        py_out = py_arena.allocate()
        rust_out = rust_arena.allocate()
        py_arena.bundle(py_handles, py_out)
        rust_arena.bundle(list(rust_handles), rust_out)

        py_phases = np.asarray(py_arena.extract_phases(py_out))
        rust_phases = np.asarray(rust_arena.extract_phases(rust_out))
        cos_diff = np.cos(py_phases - rust_phases)
        assert float(np.min(cos_diff)) > 1.0 - _TOL


# --------------------------------------------------------------------------- #
# correlation parity
# --------------------------------------------------------------------------- #


def test_compute_correlation_self_is_one(
    py_arena: _PyFhrrArenaExtended, rust_arena
):
    rng = _make_rng("self_corr")
    phases = rng.uniform(0.0, _TWO_PI, _DIM).astype(np.float32)
    py_h = _alloc_with_phases(py_arena, phases)
    rust_h = _alloc_with_phases(rust_arena, phases)
    assert py_arena.compute_correlation(py_h, py_h) == pytest.approx(
        1.0, abs=_TOL
    )
    assert rust_arena.compute_correlation(rust_h, rust_h) == pytest.approx(
        1.0, abs=_TOL
    )


def test_compute_correlation_pairwise_matches(
    py_arena: _PyFhrrArenaExtended, rust_arena
):
    rng = _make_rng("pair_corr")
    for _ in range(100):
        a = rng.uniform(0.0, _TWO_PI, _DIM).astype(np.float32)
        b = rng.uniform(0.0, _TWO_PI, _DIM).astype(np.float32)
        py_a = _alloc_with_phases(py_arena, a)
        py_b = _alloc_with_phases(py_arena, b)
        rust_a = _alloc_with_phases(rust_arena, a)
        rust_b = _alloc_with_phases(rust_arena, b)
        py_corr = py_arena.compute_correlation(py_a, py_b)
        rust_corr = rust_arena.compute_correlation(rust_a, rust_b)
        assert abs(py_corr - rust_corr) < _TOL


# --------------------------------------------------------------------------- #
# Python-only fallback sanity (always runs)
# --------------------------------------------------------------------------- #


def test_python_fallback_is_deterministic_across_instances():
    a = _PyFhrrArenaExtended(capacity=4, dimension=8)
    b = _PyFhrrArenaExtended(capacity=4, dimension=8)
    h_a = a.allocate()
    h_b = b.allocate()
    a.inject_phases(h_a, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    b.inject_phases(h_b, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert a.extract_phases(h_a) == b.extract_phases(h_b)


def test_python_fallback_capacity_exhaustion_raises():
    arena = _PyFhrrArenaExtended(capacity=2, dimension=4)
    arena.allocate()
    arena.allocate()
    with pytest.raises(ValueError, match="exhausted"):
        arena.allocate()
