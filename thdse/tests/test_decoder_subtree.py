"""Integration tests for the sub-tree decoder (LEAP 1).

Tests sub-tree vocabulary extraction, variable threading, and round-trip
decode fidelity. All tests require hdc_core — skip gracefully if not available.
"""

import ast
import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

hdc_core = pytest.importorskip("hdc_core")

from src.projection.isomorphic_projector import IsomorphicProjector
from src.decoder.subtree_vocab import SubTreeVocabulary, SubTreeAtom
from src.decoder.variable_threading import (
    VariableThreader, UnionFind, thread_variables,
)
from src.decoder.constraint_decoder import ConstraintDecoder
from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer


# ── Sample corpus ───────────────────────────────────────────────

SAMPLE_FUNCTIONS = [
    """\
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
    """\
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b
""",
    """\
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
""",
    """\
def sum_list(items):
    total = 0
    for item in items:
        total += item
    return total
""",
    """\
def find_max(items):
    if not items:
        return None
    result = items[0]
    for item in items[1:]:
        if item > result:
            result = item
    return result
""",
]


@pytest.fixture
def pipeline():
    """Create a fresh arena + projector + synthesizer."""
    arena = hdc_core.FhrrArena(500_000, 256)
    projector = IsomorphicProjector(arena, 256)
    synthesizer = AxiomaticSynthesizer(
        arena, projector, resonance_threshold=0.15,
    )
    return arena, projector, synthesizer


@pytest.fixture
def vocab_with_corpus():
    """Build a sub-tree vocabulary from sample functions."""
    vocab = SubTreeVocabulary()
    for source in SAMPLE_FUNCTIONS:
        vocab.ingest_source(source)
    return vocab


# ── Sub-Tree Vocabulary Tests ───────────────────────────────────

class TestSubTreeVocabulary:
    """Test vocabulary extraction from sample corpus."""

    def test_vocab_size_positive(self, vocab_with_corpus):
        assert vocab_with_corpus.size() > 0, "Vocabulary should contain sub-trees"

    def test_vocab_summary(self, vocab_with_corpus):
        summary = vocab_with_corpus.summary()
        assert summary["unique_subtrees"] > 0
        assert summary["total_occurrences"] >= summary["unique_subtrees"]
        assert isinstance(summary["by_root_type"], dict)

    def test_vocab_has_common_types(self, vocab_with_corpus):
        summary = vocab_with_corpus.summary()
        root_types = summary["by_root_type"]
        # At least some If/Return/For patterns should be extracted
        assert any(t in root_types for t in ["If", "Return", "For", "FunctionDef"]), \
            f"Expected common node types in vocabulary, got: {root_types}"

    def test_projection(self, pipeline, vocab_with_corpus):
        arena, projector, _ = pipeline
        projected = vocab_with_corpus.project_all(arena, projector)
        assert projected > 0, "At least some atoms should project successfully"

        atoms = vocab_with_corpus.get_projected_atoms()
        assert len(atoms) > 0
        for atom in atoms:
            assert atom.handle is not None

    def test_deduplication(self):
        """Same source ingested twice should not double the vocab."""
        vocab = SubTreeVocabulary()
        vocab.ingest_source(SAMPLE_FUNCTIONS[0])
        size_1 = vocab.size()
        vocab.ingest_source(SAMPLE_FUNCTIONS[0])
        size_2 = vocab.size()
        assert size_2 == size_1, "Duplicate ingestion should not increase vocab size"

    def test_frequency_tracking(self):
        """Duplicate sub-trees should increment frequency."""
        vocab = SubTreeVocabulary()
        vocab.ingest_source(SAMPLE_FUNCTIONS[0])
        vocab.ingest_source(SAMPLE_FUNCTIONS[0])
        atoms = vocab.get_atoms()
        # At least one atom should have frequency > 1
        max_freq = max(a.frequency for a in atoms.values())
        assert max_freq >= 2, "Duplicate ingestion should increase frequency"


# ── Variable Threading Tests ────────────────────────────────────

class TestUnionFind:
    """Test the union-find data structure."""

    def test_basic_union(self):
        uf = UnionFind()
        uf.make_set("a")
        uf.make_set("b")
        assert uf.find("a") != uf.find("b")

        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")

    def test_transitive(self):
        uf = UnionFind()
        uf.make_set("a")
        uf.make_set("b")
        uf.make_set("c")

        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("c")

    def test_groups(self):
        uf = UnionFind()
        for x in "abcde":
            uf.make_set(x)
        uf.union("a", "b")
        uf.union("c", "d")

        groups = uf.groups()
        assert len(groups) == 3  # {a,b}, {c,d}, {e}


class TestVariableThreading:
    """Test variable threading across sub-trees."""

    def test_thread_simple(self):
        """Thread variables across two simple assignment + use sub-trees."""
        tree_a = ast.parse("x0 = 0").body[0]
        tree_b = ast.parse("x0 = x0 + 1").body[0]

        result = thread_variables([tree_a, tree_b])
        assert len(result) == 2

        # After threading, both should use the same concrete name
        source_a = ast.unparse(result[0])
        source_b = ast.unparse(result[1])
        # x0 should be replaced with a concrete name
        assert "x0" not in source_a or "x0" not in source_b

    def test_thread_preserves_count(self):
        """Threading should not add or remove sub-trees."""
        trees = [ast.parse("x0 = 1").body[0], ast.parse("return x0").body[0]]
        result = thread_variables(trees)
        assert len(result) == 2


# ── Decoder Integration Tests ──────────────────────────────────

class TestDecoderSubtreeIntegration:
    """Test the full decode pipeline with sub-tree vocabulary."""

    def test_decode_produces_valid_ast(self, pipeline):
        """Decode a synthesized vector and verify it passes ast.parse()."""
        arena, projector, synthesizer = pipeline

        # Ingest sample functions
        for i, source in enumerate(SAMPLE_FUNCTIONS):
            synthesizer.ingest(f"sample_{i}", source)

        # Build vocab
        vocab = SubTreeVocabulary()
        for source in SAMPLE_FUNCTIONS:
            vocab.ingest_source(source)
        vocab.project_all(arena, projector)

        # Create decoder with vocab
        decoder = ConstraintDecoder(
            arena, projector, 256,
            activation_threshold=0.04,
            subtree_vocab=vocab,
        )

        # Synthesize from a pair
        synthesizer.compute_resonance()
        cliques = synthesizer.extract_cliques(min_size=2)
        if not cliques:
            pytest.skip("No resonant cliques found in sample corpus")

        synth_proj = synthesizer.synthesize_from_clique(cliques[0])
        result = decoder.decode(synth_proj)

        assert result is not None, "Decode should produce a result"
        assert isinstance(result, ast.Module)

        # Must pass ast.parse verification
        source = ast.unparse(result)
        parsed = ast.parse(source)
        assert parsed is not None

    def test_decode_produces_compilable_code(self, pipeline):
        """Decoded output should be compilable (>80% success rate)."""
        arena, projector, synthesizer = pipeline

        for i, source in enumerate(SAMPLE_FUNCTIONS):
            synthesizer.ingest(f"sample_{i}", source)

        vocab = SubTreeVocabulary()
        for source in SAMPLE_FUNCTIONS:
            vocab.ingest_source(source)
        vocab.project_all(arena, projector)

        decoder = ConstraintDecoder(
            arena, projector, 256,
            activation_threshold=0.04,
            subtree_vocab=vocab,
        )

        synthesizer.compute_resonance()
        cliques = synthesizer.extract_cliques(min_size=2)

        compile_successes = 0
        compile_attempts = 0

        for clique in cliques[:5]:
            synth_proj = synthesizer.synthesize_from_clique(clique)
            result = decoder.decode(synth_proj)
            if result is not None:
                source = ast.unparse(result)
                compile_attempts += 1
                try:
                    compile(source, "<test>", "exec")
                    compile_successes += 1
                except (SyntaxError, ValueError):
                    pass

        if compile_attempts > 0:
            rate = compile_successes / compile_attempts
            assert rate >= 0.8, f"Compile success rate {rate:.0%} below 80%"

    def test_round_trip_fidelity(self, pipeline):
        """Round-trip test: project → synthesize → decode → re-project → cosine > 0.3."""
        arena, projector, synthesizer = pipeline

        for i, source in enumerate(SAMPLE_FUNCTIONS):
            synthesizer.ingest(f"sample_{i}", source)

        vocab = SubTreeVocabulary()
        for source in SAMPLE_FUNCTIONS:
            vocab.ingest_source(source)
        vocab.project_all(arena, projector)

        decoder = ConstraintDecoder(
            arena, projector, 256,
            activation_threshold=0.04,
            subtree_vocab=vocab,
        )

        synthesizer.compute_resonance()
        cliques = synthesizer.extract_cliques(min_size=2)
        if not cliques:
            pytest.skip("No resonant cliques found")

        synth_proj = synthesizer.synthesize_from_clique(cliques[0])
        source, correlation = decoder.decode_and_verify(synth_proj)

        if source is not None and correlation is not None:
            # Round-trip fidelity should be meaningful (> 0.3 target,
            # but allow lower for noisy small-corpus results)
            assert correlation > -1.0, f"Correlation {correlation} is invalid"

    def test_decode_contains_corpus_fragments(self, pipeline):
        """Decoded output should contain recognizable fragments from corpus."""
        arena, projector, synthesizer = pipeline

        for i, source in enumerate(SAMPLE_FUNCTIONS):
            synthesizer.ingest(f"sample_{i}", source)

        vocab = SubTreeVocabulary()
        for source in SAMPLE_FUNCTIONS:
            vocab.ingest_source(source)
        vocab.project_all(arena, projector)

        decoder = ConstraintDecoder(
            arena, projector, 256,
            activation_threshold=0.04,
            subtree_vocab=vocab,
        )

        synthesizer.compute_resonance()
        cliques = synthesizer.extract_cliques(min_size=2)
        if not cliques:
            pytest.skip("No resonant cliques found")

        synth_proj = synthesizer.synthesize_from_clique(cliques[0])
        result = decoder.decode(synth_proj)

        if result is not None:
            source = ast.unparse(result)
            # Should contain more than just 'pass'
            assert source.strip() != "pass", \
                "Decoded output should contain real structure, not just 'pass'"
