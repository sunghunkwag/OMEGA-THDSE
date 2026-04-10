"""Tests for :mod:`shared.arena_manager` (PLAN.md Phase 2).

Covers:

- Dimension/capacity properties match PLAN.md Section D constants.
- ``alloc_cce``, ``alloc_thdse``, ``alloc_bridge`` return unique
  monotonically increasing integer handles per arena.
- Injected phases are read back wrapped into ``[0, 2π)``.
- Phase-shape validation raises :class:`DimensionMismatchError`.
- Per-arena handle counts track allocation.
- ``ArenaManager`` refuses pickle and deepcopy (Rule 11 / Risk 1).
- Provenance tags are recorded with the correct arena and dimension.
"""

from __future__ import annotations

import copy
import math
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.arena_manager import ArenaManager, HandleTag  # noqa: E402
from shared.constants import (  # noqa: E402
    BRIDGE_ARENA_CAP,
    BRIDGE_ARENA_DIM,
    CCE_ARENA_CAP,
    CCE_ARENA_DIM,
    THDSE_ARENA_CAP,
    THDSE_ARENA_DIM,
)
from shared.deterministic_rng import DeterministicRNG  # noqa: E402
from shared.exceptions import DimensionMismatchError  # noqa: E402

_TWO_PI = 2.0 * math.pi


@pytest.fixture
def mgr() -> ArenaManager:
    return ArenaManager(master_seed=42)


# --------------------------------------------------------------------------- #
# Structural invariants
# --------------------------------------------------------------------------- #


def test_dimension_properties_match_constants(mgr: ArenaManager):
    assert mgr.cce_dim == CCE_ARENA_DIM == 10_000
    assert mgr.thdse_dim == THDSE_ARENA_DIM == 256
    assert mgr.bridge_dim == BRIDGE_ARENA_DIM == 10_000


def test_capacity_properties_match_constants(mgr: ArenaManager):
    assert mgr.cce_capacity == CCE_ARENA_CAP == 100_000
    assert mgr.thdse_capacity == THDSE_ARENA_CAP == 2_000_000
    assert mgr.bridge_capacity == BRIDGE_ARENA_CAP == 50_000


def test_manager_owns_deterministic_rng(mgr: ArenaManager):
    assert isinstance(mgr.rng, DeterministicRNG)
    assert mgr.rng.master_seed == 42


def test_count_starts_at_zero(mgr: ArenaManager):
    assert mgr.count("cce") == 0
    assert mgr.count("thdse") == 0
    assert mgr.count("bridge") == 0


def test_count_rejects_unknown_arena(mgr: ArenaManager):
    with pytest.raises(ValueError, match="unknown arena"):
        mgr.count("not_an_arena")


# --------------------------------------------------------------------------- #
# Allocation semantics
# --------------------------------------------------------------------------- #


def test_alloc_cce_returns_unique_monotonic_handles(mgr: ArenaManager):
    handles = [mgr.alloc_cce() for _ in range(5)]
    assert handles == [0, 1, 2, 3, 4]
    assert mgr.count("cce") == 5
    assert mgr.count("thdse") == 0
    assert mgr.count("bridge") == 0


def test_alloc_thdse_independent_from_cce(mgr: ArenaManager):
    h_cce = mgr.alloc_cce()
    h_thdse = mgr.alloc_thdse()
    h_bridge = mgr.alloc_bridge()
    # Each arena numbers its handles independently starting at zero.
    assert h_cce == 0
    assert h_thdse == 0
    assert h_bridge == 0
    assert mgr.count("cce") == 1
    assert mgr.count("thdse") == 1
    assert mgr.count("bridge") == 1


def test_alloc_with_phases_injects_and_wraps(mgr: ArenaManager):
    # 2*pi should wrap to zero; values in [0, 2*pi) remain unchanged.
    raw = np.full(THDSE_ARENA_DIM, 2.0 * math.pi + 0.25, dtype=np.float32)
    handle = mgr.alloc_thdse(phases=raw)
    stored = mgr.get_thdse_phases(handle)
    assert stored.shape == (THDSE_ARENA_DIM,)
    assert np.allclose(stored, 0.25, atol=1e-5)


def test_alloc_with_phases_preserves_unchanged_values(mgr: ArenaManager):
    rng = mgr.rng.fork("test_phases")
    raw = rng.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)
    handle = mgr.alloc_cce(phases=raw)
    stored = mgr.get_cce_phases(handle)
    assert np.allclose(stored, raw, atol=1e-5)


def test_alloc_rejects_wrong_dim_phases(mgr: ArenaManager):
    wrong = np.zeros(CCE_ARENA_DIM, dtype=np.float32)  # 10k into THDSE
    with pytest.raises(DimensionMismatchError) as exc:
        mgr.alloc_thdse(phases=wrong)
    assert exc.value.expected == (THDSE_ARENA_DIM,)
    assert exc.value.actual == (CCE_ARENA_DIM,)
    assert exc.value.operation == "alloc_thdse"


def test_alloc_bridge_stays_at_10k_dim(mgr: ArenaManager):
    # Regression guard: bridge arena must be 10k-dim, NOT 256.
    raw = np.zeros(BRIDGE_ARENA_DIM, dtype=np.float32)
    handle = mgr.alloc_bridge(phases=raw)
    stored = mgr.get_bridge_phases(handle)
    assert stored.shape == (BRIDGE_ARENA_DIM,)
    assert BRIDGE_ARENA_DIM == 10_000


# --------------------------------------------------------------------------- #
# Handle provenance tagging
# --------------------------------------------------------------------------- #


def test_handle_tags_record_arena_origin(mgr: ArenaManager):
    h_cce = mgr.alloc_cce()
    h_thdse = mgr.alloc_thdse()
    h_bridge = mgr.alloc_bridge()

    tag_cce = mgr.tag_of("cce", h_cce)
    assert isinstance(tag_cce, HandleTag)
    assert tag_cce.arena == "cce"
    assert tag_cce.dimension == CCE_ARENA_DIM
    assert tag_cce.handle == h_cce

    tag_thdse = mgr.tag_of("thdse", h_thdse)
    assert tag_thdse.arena == "thdse"
    assert tag_thdse.dimension == THDSE_ARENA_DIM

    tag_bridge = mgr.tag_of("bridge", h_bridge)
    assert tag_bridge.arena == "bridge"
    assert tag_bridge.dimension == BRIDGE_ARENA_DIM


def test_tag_of_raises_for_unknown_handle(mgr: ArenaManager):
    with pytest.raises(KeyError):
        mgr.tag_of("cce", 9999)


# --------------------------------------------------------------------------- #
# Process-isolation guards (PLAN.md Rule 11 / Risk 1)
# --------------------------------------------------------------------------- #


def test_pickle_dumps_raises_runtime_error(mgr: ArenaManager):
    with pytest.raises(RuntimeError, match="Rust FFI"):
        pickle.dumps(mgr)


def test_getstate_raises_runtime_error(mgr: ArenaManager):
    with pytest.raises(RuntimeError, match="Rust FFI"):
        mgr.__getstate__()


def test_setstate_raises_runtime_error(mgr: ArenaManager):
    with pytest.raises(RuntimeError, match="Rust FFI"):
        mgr.__setstate__({})


def test_deepcopy_raises_runtime_error(mgr: ArenaManager):
    with pytest.raises(RuntimeError, match="Rust FFI"):
        copy.deepcopy(mgr)


# --------------------------------------------------------------------------- #
# Determinism: two managers with same seed share RNG derivation
# --------------------------------------------------------------------------- #


def test_same_master_seed_yields_identical_forks():
    a = ArenaManager(master_seed=7)
    b = ArenaManager(master_seed=7)
    seq_a = a.rng.fork("serl").uniform(0.0, 1.0, 32)
    seq_b = b.rng.fork("serl").uniform(0.0, 1.0, 32)
    assert np.array_equal(seq_a, seq_b)


def test_different_master_seed_yields_different_forks():
    a = ArenaManager(master_seed=1).rng.fork("cce").uniform(0.0, 1.0, 16)
    b = ArenaManager(master_seed=2).rng.fork("cce").uniform(0.0, 1.0, 16)
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Backend sanity: python fallback must still satisfy the contract
# --------------------------------------------------------------------------- #


def test_backend_is_known_value(mgr: ArenaManager):
    assert mgr.backend in {"rust", "python"}


def test_repr_contains_backend_and_counts(mgr: ArenaManager):
    mgr.alloc_cce()
    mgr.alloc_thdse()
    text = repr(mgr)
    assert mgr.backend in text
    assert "cce=1/100000@10000" in text
    assert "thdse=1/2000000@256" in text
    assert "bridge=0/50000@10000" in text
