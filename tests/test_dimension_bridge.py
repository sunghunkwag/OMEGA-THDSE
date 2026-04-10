"""Tests for :mod:`shared.dimension_bridge` (PLAN.md Phase 2).

Covers:

- The fixed projection index set is exactly the first 256 values of
  ``np.arange(0, 10_000, 39)`` and is immutable.
- ``project_down`` returns a 256-dim vector plus provenance metadata.
- ``cross_arena_similarity`` returns 1.0 for self-similarity through
  projection and near 0.0 for a random 256-dim vector.
- Bind commutes with subsampling EXACTLY:
  ``sub(bind(A,B)) == bind(sub(A), sub(B))``.
- Dimension violations raise :class:`DimensionMismatchError`.
- A forced bad projection index set triggers
  :class:`BridgeIntegrityError` from a direct ``_run_self_test`` call.
- ``DimensionBridge`` class delegates to the module functions.
- Importing the module (which runs the self-test) does not raise.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared import dimension_bridge as db  # noqa: E402
from shared.constants import (  # noqa: E402
    BRIDGE_RANDOM_SIMILARITY_MAX,
    BRIDGE_SELF_SIMILARITY_MIN,
    BRIDGE_SUBSAMPLE_STRIDE,
    CCE_ARENA_DIM,
    THDSE_ARENA_DIM,
)
from shared.dimension_bridge import (  # noqa: E402
    DimensionBridge,
    PROJECTION_INDICES,
    cross_arena_similarity,
    project_down,
)
from shared.exceptions import BridgeIntegrityError, DimensionMismatchError  # noqa: E402

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Projection index set
# --------------------------------------------------------------------------- #


def test_projection_index_count_is_256():
    assert PROJECTION_INDICES.shape == (THDSE_ARENA_DIM,)
    assert len(PROJECTION_INDICES) == 256


def test_projection_indices_are_stride_39():
    expected = np.arange(0, CCE_ARENA_DIM, BRIDGE_SUBSAMPLE_STRIDE)[
        :THDSE_ARENA_DIM
    ]
    assert np.array_equal(PROJECTION_INDICES, expected)


def test_projection_indices_first_and_last_values():
    # PLAN.md Section C: stride 39 starting at 0.
    # 0, 39, 78, then stride-39 onwards up to index 255 → 255 * 39 = 9945.
    assert int(PROJECTION_INDICES[0]) == 0
    assert int(PROJECTION_INDICES[1]) == 39
    assert int(PROJECTION_INDICES[-1]) == 255 * 39 == 9945


def test_projection_indices_are_readonly():
    with pytest.raises(ValueError):
        PROJECTION_INDICES[0] = 1234


# --------------------------------------------------------------------------- #
# project_down
# --------------------------------------------------------------------------- #


def test_project_down_output_shape_and_dtype():
    vec = np.linspace(0.0, _TWO_PI, CCE_ARENA_DIM, endpoint=False).astype(
        np.float32
    )
    result = project_down(vec)
    out = result["vector"]
    assert out.shape == (THDSE_ARENA_DIM,)
    assert out.dtype == np.float32


def test_project_down_preserves_subsampled_values():
    vec = np.arange(CCE_ARENA_DIM, dtype=np.float32)
    out = project_down(vec)["vector"]
    # The k-th output must be the 39*k-th input for k < 256.
    for k in (0, 1, 2, 37, 100, 255):
        assert out[k] == float(k * BRIDGE_SUBSAMPLE_STRIDE)


def test_project_down_metadata_has_provenance():
    vec = np.zeros(CCE_ARENA_DIM, dtype=np.float32)
    result = project_down(vec)
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "project_down"
    assert prov["source_arena"] == "cce"
    assert prov["target_arena"] == "thdse"
    assert prov["source_dim"] == CCE_ARENA_DIM
    assert prov["target_dim"] == THDSE_ARENA_DIM
    assert prov["stride"] == BRIDGE_SUBSAMPLE_STRIDE
    assert prov["index_count"] == 256


def test_project_down_rejects_wrong_dim():
    bad = np.zeros(THDSE_ARENA_DIM, dtype=np.float32)
    with pytest.raises(DimensionMismatchError) as exc:
        project_down(bad)
    assert exc.value.expected == (CCE_ARENA_DIM,)
    assert exc.value.actual == (THDSE_ARENA_DIM,)
    assert exc.value.operation == "project_down"


def test_project_down_rejects_non_1d():
    bad = np.zeros((100, 100), dtype=np.float32)
    with pytest.raises(DimensionMismatchError):
        project_down(bad)


# --------------------------------------------------------------------------- #
# cross_arena_similarity
# --------------------------------------------------------------------------- #


def test_self_similarity_is_exactly_one():
    g = np.random.default_rng(11)
    vec = g.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)
    projected = project_down(vec)["vector"]
    result = cross_arena_similarity(vec, projected)
    # cos(x - x) = 1 for every component, so the mean is exactly 1.
    assert result["similarity"] == pytest.approx(1.0, abs=1e-6)


def test_self_similarity_exceeds_threshold_across_many_seeds():
    for seed in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        g = np.random.default_rng(seed)
        vec = g.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)
        sim = cross_arena_similarity(vec, project_down(vec)["vector"])[
            "similarity"
        ]
        assert sim > BRIDGE_SELF_SIMILARITY_MIN


def test_random_similarity_is_near_zero_on_self_test_seed():
    # Seed 1337 is what the import-time self-test uses; all 10 trials
    # must pass under BRIDGE_RANDOM_SIMILARITY_MAX.
    g = np.random.default_rng(1337)
    observed = []
    for _ in range(10):
        # Match the draw order used by _run_self_test so we cover the
        # same pairs it does.
        g.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)  # a
        g.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)  # b (unused here)
        vec = g.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)
        rnd = g.uniform(0.0, _TWO_PI, THDSE_ARENA_DIM).astype(np.float32)
        sim = cross_arena_similarity(vec, rnd)["similarity"]
        observed.append(abs(sim))
    assert max(observed) < BRIDGE_RANDOM_SIMILARITY_MAX


def test_random_similarity_mean_is_near_zero_over_large_sample():
    # For a large sample, the expected random similarity is 0 and the
    # standard deviation of the sample mean shrinks as 1/sqrt(N*256).
    g = np.random.default_rng(20250410)
    sims = []
    for _ in range(200):
        vec = g.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)
        rnd = g.uniform(0.0, _TWO_PI, THDSE_ARENA_DIM).astype(np.float32)
        sims.append(cross_arena_similarity(vec, rnd)["similarity"])
    mean = float(np.mean(sims))
    # Theoretical std of one trial: sqrt(0.5 / 256) ≈ 0.0442.
    # Std of the mean over 200 trials: 0.0442 / sqrt(200) ≈ 0.00313.
    # 0.02 is comfortably > 6 sigma.
    assert abs(mean) < 0.02


def test_cross_arena_similarity_metadata_has_provenance():
    vec = np.zeros(CCE_ARENA_DIM, dtype=np.float32)
    tgt = np.zeros(THDSE_ARENA_DIM, dtype=np.float32)
    prov = cross_arena_similarity(vec, tgt)["metadata"]["provenance"]
    assert prov["operation"] == "cross_arena_similarity"
    assert prov["source_arenas"] == ("cce", "thdse")
    assert prov["compared_in_dim"] == THDSE_ARENA_DIM
    assert prov["stride"] == BRIDGE_SUBSAMPLE_STRIDE


def test_cross_arena_similarity_rejects_swapped_dims():
    big = np.zeros(CCE_ARENA_DIM, dtype=np.float32)
    small = np.zeros(THDSE_ARENA_DIM, dtype=np.float32)
    with pytest.raises(DimensionMismatchError):
        cross_arena_similarity(small, small)
    with pytest.raises(DimensionMismatchError):
        cross_arena_similarity(big, big)


def test_cross_arena_similarity_antiparallel_returns_minus_one():
    vec = np.full(CCE_ARENA_DIM, 0.5, dtype=np.float32)
    projected = project_down(vec)["vector"]
    flipped = projected + np.float32(math.pi)
    sim = cross_arena_similarity(vec, flipped)["similarity"]
    assert sim == pytest.approx(-1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Bind commutation (PLAN.md Section C, Rule 12(a))
# --------------------------------------------------------------------------- #


def test_subsample_commutes_with_bind_exactly():
    g = np.random.default_rng(99)
    for trial in range(10):
        a = g.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)
        b = g.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)
        bound_full = np.mod(a + b, np.float32(_TWO_PI))
        sub_of_bound = project_down(bound_full)["vector"]
        sub_a = project_down(a)["vector"]
        sub_b = project_down(b)["vector"]
        bind_of_sub = np.mod(sub_a + sub_b, np.float32(_TWO_PI))
        assert np.array_equal(sub_of_bound, bind_of_sub), (
            f"trial {trial}: commutation violated"
        )


# --------------------------------------------------------------------------- #
# Self-test on import (Rule 12)
# --------------------------------------------------------------------------- #


def test_module_import_ran_self_test_without_raising():
    # If _run_self_test had raised at import, this import would have
    # failed at module-load time. Confirm a direct re-invocation with
    # the same seed still succeeds.
    db._run_self_test(seed=1337, num_pairs=10)


def test_self_test_detects_broken_commutation(monkeypatch):
    # Force the subsampler to scramble the result so the bind commutation
    # check fails; _run_self_test must raise BridgeIntegrityError.
    real_subsample = db._subsample

    def broken(vec):
        out = real_subsample(vec)
        # Destroy an element — small but enough to break array_equal.
        out = out.copy()
        out[0] = np.float32(out[0] + 0.5)
        return out

    monkeypatch.setattr(db, "_subsample", broken)
    with pytest.raises(BridgeIntegrityError) as exc:
        db._run_self_test(seed=1337, num_pairs=10)
    assert exc.value.check == "bind_commutation"


def test_self_test_detects_broken_self_similarity(monkeypatch):
    # Make cross_arena_similarity always return 0 so self-sim fails.
    def broken_cross(vec_10k, vec_256):
        return {
            "similarity": 0.0,
            "metadata": {"provenance": {"operation": "broken"}},
        }

    monkeypatch.setattr(db, "cross_arena_similarity", broken_cross)
    with pytest.raises(BridgeIntegrityError) as exc:
        db._run_self_test(seed=1337, num_pairs=10)
    assert exc.value.check == "self_similarity"


# --------------------------------------------------------------------------- #
# DimensionBridge class wrapper
# --------------------------------------------------------------------------- #


def test_dimension_bridge_class_delegates_to_module():
    bridge = DimensionBridge(label="unit-test")
    assert bridge.label == "unit-test"
    vec = np.full(CCE_ARENA_DIM, 0.1, dtype=np.float32)
    out = bridge.project_down(vec)["vector"]
    assert out.shape == (THDSE_ARENA_DIM,)
    assert np.allclose(out, 0.1, atol=1e-6)
    sim = bridge.cross_arena_similarity(vec, out)["similarity"]
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_dimension_bridge_exposes_readonly_projection_indices():
    bridge = DimensionBridge()
    idx = bridge.projection_indices
    assert idx.shape == (THDSE_ARENA_DIM,)
    with pytest.raises(ValueError):
        idx[0] = 999
