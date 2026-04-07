"""Regression tests for batch correlation and refactored consumers.

Tests:
  1. correlate_matrix correctness vs individual compute_correlation
  2. correlate_matrix_subset correctness vs individual compute_correlation
  3. ResonanceMatrix behavioral equivalence
  4. RefactoringDetector behavioral equivalence
  5. correlate_matrix edge cases (empty, single)
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

hdc_core = pytest.importorskip("hdc_core")

from src.projection.isomorphic_projector import IsomorphicProjector
from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer
from src.analysis.refactoring_detector import RefactoringDetector, _get_directory


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


@pytest.fixture
def arena_dim():
    dim = 128
    arena = hdc_core.FhrrArena(100_000, dim)
    return arena, dim


# ── Test 1: correlate_matrix correctness ─────────────────────────

class TestCorrelateMatrixCorrectness:
    def test_10_vectors(self, arena_dim):
        """Create 10 vectors, verify correlate_matrix matches 45 individual calls."""
        arena, dim = arena_dim
        n = 10
        handles = []
        for i in range(n):
            phases = _deterministic_phases(1000 + i, dim)
            h = arena.allocate()
            arena.inject_phases(h, phases)
            handles.append(h)

        flat = arena.correlate_matrix(handles)
        assert len(flat) == n * (n - 1) // 2, f"Expected {n*(n-1)//2} pairs, got {len(flat)}"

        # Compare with individual calls
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                individual = arena.compute_correlation(handles[i], handles[j])
                assert abs(flat[k] - individual) < 1e-5, \
                    f"Mismatch at ({i},{j}): batch={flat[k]:.6f}, individual={individual:.6f}"
                k += 1


# ── Test 2: correlate_matrix_subset correctness ──────────────────

class TestCorrelateMatrixSubsetCorrectness:
    def test_5_queries_8_targets(self, arena_dim):
        """Create 5 query + 8 target vectors, verify subset matches 40 individual calls."""
        arena, dim = arena_dim
        queries = []
        for i in range(5):
            phases = _deterministic_phases(2000 + i, dim)
            h = arena.allocate()
            arena.inject_phases(h, phases)
            queries.append(h)

        targets = []
        for i in range(8):
            phases = _deterministic_phases(3000 + i, dim)
            h = arena.allocate()
            arena.inject_phases(h, phases)
            targets.append(h)

        flat = arena.correlate_matrix_subset(queries, targets)
        assert len(flat) == 5 * 8, f"Expected 40 entries, got {len(flat)}"

        k = 0
        for i in range(5):
            for j in range(8):
                individual = arena.compute_correlation(queries[i], targets[j])
                assert abs(flat[k] - individual) < 1e-5, \
                    f"Mismatch at q={i},t={j}: batch={flat[k]:.6f}, individual={individual:.6f}"
                k += 1


# ── Test 3: ResonanceMatrix behavioral equivalence ───────────────

class TestResonanceMatrixEquivalence:
    def test_20_functions(self, arena_dim):
        """Ingest 20 functions. Compare batch resonance matrix with
        individually computed correlations."""
        arena, dim = arena_dim
        projector = IsomorphicProjector(arena, dim)
        synthesizer = AxiomaticSynthesizer(arena, projector, resonance_threshold=0.15)

        corpus = {}
        for i in range(20):
            corpus[f"proj/file_{i}.py"] = (
                f"def func_{i}(x):\n"
                f"    y = x + {i}\n"
                f"    if y > {i * 2}:\n"
                f"        return y * {i + 1}\n"
                f"    return y\n"
            )
        synthesizer.ingest_batch(corpus)

        # Compute resonance using the (now batched) implementation
        resonance_dict = synthesizer.compute_resonance()
        ids = synthesizer.store.all_ids()
        n = len(ids)

        # Independently verify with individual calls
        for i in range(n):
            for j in range(i + 1, n):
                id_a = ids[i]
                id_b = ids[j]
                ax_a = synthesizer.store.axioms[id_a]
                ax_b = synthesizer.store.axioms[id_b]
                expected = arena.compute_correlation(ax_a.handle, ax_b.handle)
                actual = resonance_dict.get((id_a, id_b), 0.0)
                assert abs(actual - expected) < 1e-5, \
                    f"Resonance mismatch for ({id_a}, {id_b}): " \
                    f"batch={actual:.6f}, individual={expected:.6f}"


# ── Test 4: RefactoringDetector behavioral equivalence ───────────

class TestRefactoringDetectorEquivalence:
    def test_identical_results(self, arena_dim):
        """Ingest 20 functions across 2 directories. Verify new scan()
        produces the same candidates as the old _scan_fallback()."""
        arena, dim = arena_dim
        projector = IsomorphicProjector(arena, dim)
        synthesizer = AxiomaticSynthesizer(arena, projector, resonance_threshold=0.15)

        corpus = {}
        # Two directories with structurally similar functions
        for i in range(10):
            corpus[f"project_a/module_{i}.py"] = (
                f"def process_{i}(data):\n"
                f"    result = data + {i}\n"
                f"    if result > 0:\n"
                f"        return result\n"
                f"    return -result\n"
            )
            corpus[f"project_b/module_{i}.py"] = (
                f"def handle_{i}(data):\n"
                f"    result = data + {i}\n"
                f"    if result > 0:\n"
                f"        return result\n"
                f"    return -result\n"
            )
        synthesizer.ingest_batch(corpus)

        detector = RefactoringDetector(
            arena, projector, synthesizer,
            dup_ast_threshold=0.7,
            dup_cfg_threshold=0.7,
            unify_data_threshold=0.8,
            unify_ast_ceiling=0.3,
        )

        # New (batch) implementation
        new_candidates = detector.scan()

        # Old (fallback) implementation
        ids = synthesizer.store.all_ids()
        old_candidates = detector._scan_fallback(ids)

        # Compare: same set of (kind, file_a, file_b) triples
        new_triples = set(
            (c.kind, c.file_a, c.file_b) for c in new_candidates
        )
        old_triples = set(
            (c.kind, c.file_a, c.file_b) for c in old_candidates
        )

        assert new_triples == old_triples, (
            f"Candidate mismatch!\n"
            f"  New only: {new_triples - old_triples}\n"
            f"  Old only: {old_triples - new_triples}"
        )

        # Also verify similarity values are close for matching candidates
        old_by_key = {(c.kind, c.file_a, c.file_b): c for c in old_candidates}
        for nc in new_candidates:
            key = (nc.kind, nc.file_a, nc.file_b)
            if key in old_by_key:
                oc = old_by_key[key]
                assert abs(nc.sim_ast - oc.sim_ast) < 1e-4, \
                    f"AST sim mismatch for {key}: {nc.sim_ast} vs {oc.sim_ast}"
                if nc.sim_cfg is not None and oc.sim_cfg is not None:
                    assert abs(nc.sim_cfg - oc.sim_cfg) < 1e-4, \
                        f"CFG sim mismatch for {key}: {nc.sim_cfg} vs {oc.sim_cfg}"


# ── Test 5: Edge cases ──────────────────────────────────────────

class TestCorrelateMatrixEdgeCases:
    def test_empty_input(self, arena_dim):
        """correlate_matrix([]) returns empty."""
        arena, _ = arena_dim
        assert arena.correlate_matrix([]) == []

    def test_single_handle(self, arena_dim):
        """correlate_matrix([h0]) returns empty (no pairs)."""
        arena, dim = arena_dim
        h = arena.allocate()
        arena.inject_phases(h, _deterministic_phases(42, dim))
        assert arena.correlate_matrix([h]) == []

    def test_two_handles(self, arena_dim):
        """correlate_matrix([h0, h1]) returns Vec of length 1."""
        arena, dim = arena_dim
        h0 = arena.allocate()
        arena.inject_phases(h0, _deterministic_phases(50, dim))
        h1 = arena.allocate()
        arena.inject_phases(h1, _deterministic_phases(51, dim))
        result = arena.correlate_matrix([h0, h1])
        assert len(result) == 1
        expected = arena.compute_correlation(h0, h1)
        assert abs(result[0] - expected) < 1e-6

    def test_subset_empty_queries(self, arena_dim):
        """correlate_matrix_subset([], targets) returns empty."""
        arena, dim = arena_dim
        h = arena.allocate()
        arena.inject_phases(h, _deterministic_phases(60, dim))
        assert arena.correlate_matrix_subset([], [h]) == []

    def test_subset_empty_targets(self, arena_dim):
        """correlate_matrix_subset(queries, []) returns empty."""
        arena, dim = arena_dim
        h = arena.allocate()
        arena.inject_phases(h, _deterministic_phases(70, dim))
        assert arena.correlate_matrix_subset([h], []) == []
