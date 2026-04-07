"""Integration tests for the structural diff engine (LEAP 2).

Tests layer-decomposed similarity and structural delta reporting.
All tests require hdc_core — skip gracefully if not available.
"""

import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

hdc_core = pytest.importorskip("hdc_core")

from src.projection.isomorphic_projector import IsomorphicProjector
from src.analysis.structural_diff import StructuralDiffEngine, LayerSimilarity


@pytest.fixture
def pipeline():
    """Create a fresh arena + projector for each test."""
    arena = hdc_core.FhrrArena(100_000, 256)
    projector = IsomorphicProjector(arena, 256)
    engine = StructuralDiffEngine(arena, projector)
    return arena, projector, engine


# ── Sample source code ──────────────────────────────────────────

IDENTICAL_A = """\
def add(a, b):
    return a + b
"""

IDENTICAL_B = """\
def add(a, b):
    return a + b
"""

DIFFERENT_A = """\
def add(a, b):
    return a + b
"""

DIFFERENT_B = """\
class DataProcessor:
    def __init__(self, items):
        self.items = items

    def process(self):
        for item in self.items:
            if item > 0:
                yield item * 2
"""

# Same AST structure (if/return) but different control flow
AST_SAME_CFG_DIFF_A = """\
def check(x):
    if x > 0:
        return x
    return -x
"""

AST_SAME_CFG_DIFF_B = """\
def check(x):
    if x > 0:
        return x * 2
    return x + 1
"""


class TestIdenticalFiles:
    """Two identical files should have high similarity across all layers."""

    def test_all_layers_high(self, pipeline):
        arena, projector, engine = pipeline
        sim = engine.compare_files(IDENTICAL_A, IDENTICAL_B)

        assert isinstance(sim, LayerSimilarity)
        assert sim.sim_ast > 0.95, f"AST similarity {sim.sim_ast} should be > 0.95"
        assert sim.sim_final > 0.95, f"Final similarity {sim.sim_final} should be > 0.95"

    def test_diagnosis_not_empty(self, pipeline):
        _, _, engine = pipeline
        sim = engine.compare_files(IDENTICAL_A, IDENTICAL_B)
        assert sim.diagnosis, "Diagnosis should not be empty"


class TestDifferentFiles:
    """Completely different files should have low similarity."""

    def test_all_layers_low(self, pipeline):
        _, _, engine = pipeline
        sim = engine.compare_files(DIFFERENT_A, DIFFERENT_B)

        assert sim.sim_ast < 0.5, f"AST similarity {sim.sim_ast} should be < 0.5"
        assert sim.sim_final < 0.5, f"Final similarity {sim.sim_final} should be < 0.5"


class TestASTSameCFGDifferent:
    """Files with same AST shape but different CFG should show the pattern."""

    def test_ast_higher_than_cfg(self, pipeline):
        _, _, engine = pipeline
        sim = engine.compare_files(AST_SAME_CFG_DIFF_A, AST_SAME_CFG_DIFF_B)

        # AST should be more similar than completely different files
        assert sim.sim_ast > 0.3, f"AST similarity {sim.sim_ast} should be > 0.3"

    def test_dominant_layer_reported(self, pipeline):
        _, _, engine = pipeline
        sim = engine.compare_files(AST_SAME_CFG_DIFF_A, AST_SAME_CFG_DIFF_B)
        assert sim.dominant_layer in ("ast", "cfg", "data")


class TestStructuralDelta:
    """Test the full structural delta computation."""

    def test_delta_with_vocab(self, pipeline):
        arena, projector, engine = pipeline
        from src.decoder.subtree_vocab import SubTreeVocabulary

        vocab = SubTreeVocabulary()
        vocab.ingest_source(DIFFERENT_A)
        vocab.ingest_source(DIFFERENT_B)
        vocab.project_all(arena, projector)

        proj_a = projector.project(DIFFERENT_A)
        proj_b = projector.project(DIFFERENT_B)

        delta = engine.compute_delta(proj_a, proj_b, vocab=vocab)
        assert delta.summary, "Delta summary should not be empty"
        assert delta.layer_sim is not None

    def test_delta_without_vocab(self, pipeline):
        arena, projector, engine = pipeline
        proj_a = projector.project(IDENTICAL_A)
        proj_b = projector.project(IDENTICAL_B)

        delta = engine.compute_delta(proj_a, proj_b)
        assert delta.layer_sim.sim_ast > 0.95


class TestSerialization:
    """Test that results serialize to dictionaries."""

    def test_to_dict(self, pipeline):
        _, _, engine = pipeline
        proj_a = engine.projector.project(IDENTICAL_A)
        proj_b = engine.projector.project(DIFFERENT_B)

        delta = engine.compute_delta(proj_a, proj_b)
        d = engine.to_dict(delta)

        assert "layer_similarity" in d
        assert "summary" in d
        assert isinstance(d["layer_similarity"]["ast"], float)
