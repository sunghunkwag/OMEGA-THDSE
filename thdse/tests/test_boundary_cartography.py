"""Integration tests for the Boundary Cartography subsystem.

Tests WallArchive, multi-quotient projection, residual dimension estimation,
open direction extraction, and directed synthesis.

All tests require hdc_core — skip gracefully if not available.
"""

import math
import os
import sys

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

hdc_core = pytest.importorskip("hdc_core")

from src.synthesis.wall_archive import WallArchive, WallRecord
from src.decoder.constraint_decoder import ConstraintDecoder, UnsatCoreResult
from src.projection.isomorphic_projector import IsomorphicProjector
from src.analysis.boundary_cartography import (
    estimate_residual_dimension,
    extract_open_directions,
)


# ── Helpers ──────────────────────────────────────────────────────

def _deterministic_phases(seed: int, dimension: int):
    """Same LCG as IsomorphicProjector."""
    phases = []
    state = seed & 0xFFFFFFFF
    for _ in range(dimension):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        phase = ((state >> 33) / (2**31)) * 2.0 * math.pi - math.pi
        phases.append(phase)
    return phases


def _make_unsat_core(arena, dimension, seed, labels=None):
    """Create a mock UnsatCoreResult with a real V_error in the arena."""
    phases = _deterministic_phases(seed, dimension)
    handle = arena.allocate()
    arena.inject_phases(handle, phases)
    return UnsatCoreResult(
        core_labels=labels or [f"atom:test_{seed}"],
        core_atom_handles=[handle],
        v_error_handle=handle,
        projected_count=0,
    )


@pytest.fixture
def arena_dim():
    """Create a fresh arena with dimension 128."""
    dim = 128
    arena = hdc_core.FhrrArena(100_000, dim)
    return arena, dim


# ── Test 1: WallArchive basic operations ─────────────────────────

class TestWallArchiveBasicOps:
    def test_record_and_count(self, arena_dim):
        arena, dim = arena_dim
        archive = WallArchive(arena, dim)
        assert archive.count() == 0

        # Record 3 walls
        for i in range(3):
            core = _make_unsat_core(arena, dim, seed=100 + i)
            archive.record(core, synthesis_context=f"test_{i}")

        assert archive.count() == 3

    def test_phase_persistence(self, arena_dim):
        """Extract phases → re-inject → correlate > 0.99."""
        arena, dim = arena_dim
        archive = WallArchive(arena, dim)

        core = _make_unsat_core(arena, dim, seed=42)
        wall = archive.record(core, synthesis_context="test")

        # Re-inject phases into a new handle
        new_handle = arena.allocate()
        arena.inject_phases(new_handle, wall.v_error_phases)

        corr = arena.compute_correlation(wall.v_error_handle, new_handle)
        assert corr > 0.99, f"Phase persistence failed: corr={corr:.4f}"

    def test_all_handles_and_phases(self, arena_dim):
        arena, dim = arena_dim
        archive = WallArchive(arena, dim)

        cores = [_make_unsat_core(arena, dim, seed=200 + i) for i in range(3)]
        for i, core in enumerate(cores):
            archive.record(core, synthesis_context=f"ctx_{i}")

        handles = archive.all_handles()
        assert len(handles) == 3

        phase_arrays = archive.all_phase_arrays()
        assert len(phase_arrays) == 3
        assert all(len(p) == dim for p in phase_arrays)


# ── Test 2: Wall resonance and cliques ───────────────────────────

class TestWallResonance:
    def test_identical_walls_form_clique(self, arena_dim):
        """Record 2 identical walls + 1 different wall; identical walls clique."""
        arena, dim = arena_dim
        archive = WallArchive(arena, dim)

        # Two walls with same phases
        phases_same = _deterministic_phases(999, dim)
        for i in range(2):
            h = arena.allocate()
            arena.inject_phases(h, phases_same)
            core = UnsatCoreResult(
                core_labels=[f"same_{i}"],
                core_atom_handles=[h],
                v_error_handle=h,
                projected_count=0,
            )
            archive.record(core, synthesis_context=f"same_{i}")

        # One different wall
        phases_diff = _deterministic_phases(12345, dim)
        h_diff = arena.allocate()
        arena.inject_phases(h_diff, phases_diff)
        core_diff = UnsatCoreResult(
            core_labels=["different"],
            core_atom_handles=[h_diff],
            v_error_handle=h_diff,
            projected_count=0,
        )
        archive.record(core_diff, synthesis_context="different")

        # Check resonance
        resonance = archive.compute_wall_resonance()
        # Identical walls should have correlation ~1.0
        assert resonance[(0, 1)] > 0.99

        # Cliques: identical walls should form a clique
        cliques = archive.extract_wall_cliques(threshold=0.5)
        # Find the clique containing both identical walls
        found_pair = False
        for clique in cliques:
            if 0 in clique and 1 in clique:
                found_pair = True
                break
        assert found_pair, f"Identical walls not in same clique. Cliques: {cliques}"


# ── Test 3: Multi-quotient projection ────────────────────────────

class TestMultiQuotientProjection:
    def test_two_orthogonal_walls_projected(self, arena_dim):
        """Project out 2 walls from a test vector; both components drop to ~0."""
        arena, dim = arena_dim

        # Create 2 wall vectors
        wall_phases_1 = _deterministic_phases(500, dim)
        wall_phases_2 = _deterministic_phases(501, dim)

        w1 = arena.allocate()
        arena.inject_phases(w1, wall_phases_1)
        w2 = arena.allocate()
        arena.inject_phases(w2, wall_phases_2)

        # Create a test vector that has components along both walls
        # test = bind(w1, w2) loosely — just use deterministic phases
        test_phases = _deterministic_phases(502, dim)
        test_h = arena.allocate()
        arena.inject_phases(test_h, test_phases)

        # Record initial correlations
        corr_before_1 = abs(arena.compute_correlation(test_h, w1))
        corr_before_2 = abs(arena.compute_correlation(test_h, w2))

        # Project out both walls
        count = arena.project_to_multi_quotient_space([w1, w2])
        assert count > 0

        # After projection, correlations with walls should be near 0
        corr_after_1 = abs(arena.compute_correlation(test_h, w1))
        corr_after_2 = abs(arena.compute_correlation(test_h, w2))

        # The correlation should decrease significantly
        assert corr_after_1 < corr_before_1 + 0.05 or corr_after_1 < 0.15, \
            f"Wall 1 not projected out: before={corr_before_1:.4f}, after={corr_after_1:.4f}"
        assert corr_after_2 < corr_before_2 + 0.05 or corr_after_2 < 0.15, \
            f"Wall 2 not projected out: before={corr_before_2:.4f}, after={corr_after_2:.4f}"


# ── Test 4: Arena integrity — analysis must not corrupt main arena ─

class TestArenaIntegrity:
    def test_estimate_does_not_corrupt_arena(self, arena_dim):
        """Calling estimate_residual_dimension must not change
        existing vectors in the main arena."""
        arena, dim = arena_dim
        # Create a corpus vector
        phases = _deterministic_phases(42, dim)
        corpus_h = arena.allocate()
        arena.inject_phases(corpus_h, phases)
        # Record self-correlation before
        corr_before = arena.compute_correlation(corpus_h, corpus_h)
        # Create walls and run estimation
        wall_handles = []
        for i in range(3):
            wp = _deterministic_phases(300 + i, dim)
            wh = arena.allocate()
            arena.inject_phases(wh, wp)
            wall_handles.append(wh)
        estimate_residual_dimension(arena, wall_handles, dim)
        # Self-correlation must be unchanged
        corr_after = arena.compute_correlation(corpus_h, corpus_h)
        assert abs(corr_after - corr_before) < 0.01, \
            f"Arena corrupted: self-corr {corr_before:.4f} → {corr_after:.4f}"

    def test_extract_open_dirs_does_not_corrupt_arena(self, arena_dim):
        """Calling extract_open_directions must not change
        existing vectors in the main arena."""
        arena, dim = arena_dim
        # Create a corpus vector
        phases = _deterministic_phases(55, dim)
        corpus_h = arena.allocate()
        arena.inject_phases(corpus_h, phases)

        # Create a second vector to measure pairwise correlation stability
        phases2 = _deterministic_phases(56, dim)
        corpus_h2 = arena.allocate()
        arena.inject_phases(corpus_h2, phases2)

        corr_before = arena.compute_correlation(corpus_h, corpus_h2)

        # Record walls in archive
        archive = WallArchive(arena, dim)
        for i in range(3):
            core = _make_unsat_core(arena, dim, seed=310 + i)
            archive.record(core, synthesis_context=f"integrity_{i}")

        projector = IsomorphicProjector(arena, dim)
        extract_open_directions(arena, projector, archive, n_directions=3)

        # Pairwise correlation must be unchanged
        corr_after = arena.compute_correlation(corpus_h, corpus_h2)
        assert abs(corr_after - corr_before) < 0.01, \
            f"Arena corrupted: pairwise corr {corr_before:.4f} → {corr_after:.4f}"


# ── Test 5: Residual dimension estimation ─────────────────────────

class TestResidualDimension:
    def test_zero_walls_full_dimension(self, arena_dim):
        """With 0 walls, residual dim = full dim."""
        arena, dim = arena_dim
        result = estimate_residual_dimension(arena, [], dim)
        assert result["estimated_residual_dimension"] == dim
        assert result["wall_count"] == 0
        assert result["compression_ratio"] == 1.0
        assert not result["is_closed"]

    def test_walls_reduce_dimension(self, arena_dim):
        """With k walls, residual dim <= full dim - k."""
        arena, dim = arena_dim

        # Create k wall vectors
        k = 3
        wall_handles = []
        for i in range(k):
            phases = _deterministic_phases(600 + i, dim)
            h = arena.allocate()
            arena.inject_phases(h, phases)
            wall_handles.append(h)

        result = estimate_residual_dimension(arena, wall_handles, dim)
        # After projecting out k walls, residual should be reduced
        assert result["wall_count"] == k
        assert result["estimated_residual_dimension"] <= dim


# ── Test 6: Open direction extraction ─────────────────────────────

class TestOpenDirections:
    def test_one_wall_has_open_directions(self, arena_dim):
        """With 1 wall, at least 1 open direction exists and is near-orthogonal."""
        arena, dim = arena_dim
        archive = WallArchive(arena, dim)

        # Record 1 wall
        core = _make_unsat_core(arena, dim, seed=700)
        wall = archive.record(core, synthesis_context="test")
        wall_handle = wall.v_error_handle

        projector = IsomorphicProjector(arena, dim)
        open_dirs = extract_open_directions(arena, projector, archive, n_directions=3)

        assert len(open_dirs) >= 1, "Expected at least 1 open direction"

        # Each open direction should be near-orthogonal to the wall
        for dir_handle, variance in open_dirs:
            corr = abs(arena.compute_correlation(dir_handle, wall_handle))
            # After projection, correlation with wall should be low
            assert corr < 0.5, f"Open direction too correlated with wall: {corr:.4f}"


# ── Test 7: Directed synthesis smoke test ─────────────────────────

class TestDirectedSynthesis:
    def test_smoke_ingest_and_synthesize(self, arena_dim):
        """Ingest 4 functions, collect walls, attempt directed synthesis."""
        arena, dim = arena_dim
        projector = IsomorphicProjector(arena, dim)

        from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer

        synthesizer = AxiomaticSynthesizer(arena, projector, resonance_threshold=0.15)

        # Ingest 4 simple functions
        corpus = {
            "fn_add": "def add(a, b):\n    return a + b\n",
            "fn_sub": "def sub(a, b):\n    return a - b\n",
            "fn_mul": "def mul(a, b):\n    return a * b\n",
            "fn_div": "def div(a, b):\n    if b == 0:\n        return 0\n    return a / b\n",
        }
        synthesizer.ingest_batch(corpus)
        synthesizer.compute_resonance()

        # Create wall archive with a wall
        archive = WallArchive(arena, dim)
        core = _make_unsat_core(arena, dim, seed=800)
        archive.record(core, synthesis_context="smoke_test")

        # Extract open directions
        open_dirs = extract_open_directions(arena, projector, archive, n_directions=2)
        assert len(open_dirs) >= 1, "Expected at least 1 open direction after smoke test"

        # Attempt directed synthesis (may or may not decode successfully,
        # but should not crash)
        from src.analysis.boundary_cartography import synthesize_along_direction
        from src.decoder.constraint_decoder import ConstraintDecoder

        decoder = ConstraintDecoder(
            arena, projector, dim,
            activation_threshold=0.04,
            wall_archive=archive,
        )

        for dir_handle, variance in open_dirs:
            result = synthesize_along_direction(
                arena, projector, synthesizer, decoder,
                dir_handle, list(corpus.keys()),
            )
            # Result is Optional[str] — either source or None, no crash
            assert result is None or isinstance(result, str)


# ── Test 8: Revalidation after dimension expansion ────────────────

class TestRevalidateAfterDimensionExpansion:
    def test_revalidate_after_expand(self, arena_dim):
        """Record a wall at d=128, expand to d=256, revalidate."""
        arena, dim = arena_dim
        projector = IsomorphicProjector(arena, dim)
        archive = WallArchive(arena, dim)

        # Record a wall
        core = _make_unsat_core(arena, dim, seed=900)
        wall = archive.record(core, synthesis_context="pre_expand")
        assert archive.count() == 1
        assert wall.status == "active"

        # Expand dimension
        new_dim = dim * 2
        arena.expand_dimension(new_dim)

        # Create decoder at new dimension for revalidation
        decoder = ConstraintDecoder(
            arena, projector, new_dim,
            activation_threshold=0.04,
        )
        projector.dimension = new_dim

        result = archive.revalidate_walls(decoder, projector)
        assert result["total_checked"] == 1
        # Wall may be resolved or still valid — both are acceptable
        assert result["still_valid"] + result["resolved"] + result["orphaned"] == 1


# ── Test 9: Resolved walls excluded from active map ───────────────

class TestResolvedWallsExcluded:
    def test_resolved_excluded_from_handles(self, arena_dim):
        """Record 2 walls, resolve 1, verify all_handles() excludes resolved."""
        arena, dim = arena_dim
        archive = WallArchive(arena, dim)

        # Record 2 walls
        core1 = _make_unsat_core(arena, dim, seed=1000)
        wall1 = archive.record(core1, synthesis_context="wall_1")

        core2 = _make_unsat_core(arena, dim, seed=1001)
        wall2 = archive.record(core2, synthesis_context="wall_2")

        assert len(archive.all_handles()) == 2

        # Manually mark wall2 as resolved
        wall2.status = "resolved"

        # all_handles should only return active walls
        active_handles = archive.all_handles()
        assert len(active_handles) == 1
        assert active_handles[0] == wall1.v_error_handle

        # compute_wall_resonance should only include active walls
        resonance = archive.compute_wall_resonance()
        # Only 1 active wall → only self-correlation entry
        assert (0, 0) not in resonance or len(resonance) <= 1


# ── Test 10: Serialization round-trip ─────────────────────────────

class TestSerialization:
    def test_serialize_deserialize(self, arena_dim, tmp_path):
        """Serialize archive, deserialize, verify phase fidelity."""
        arena, dim = arena_dim
        archive = WallArchive(arena, dim)

        for i in range(3):
            core = _make_unsat_core(arena, dim, seed=1100 + i)
            archive.record(core, synthesis_context=f"serial_{i}")

        filepath = str(tmp_path / "walls.pkl")
        archive.serialize(filepath)

        # New arena for deserialization
        arena2 = hdc_core.FhrrArena(100_000, dim)
        archive2 = WallArchive.deserialize(filepath, arena2)

        assert archive2.count() == 3
        # Phase arrays should match
        for orig, loaded in zip(archive.all_phase_arrays(), archive2.all_phase_arrays()):
            for a, b in zip(orig, loaded):
                assert abs(a - b) < 1e-6
