"""Tests for the SERL Loop (Step 3).

5 tests covering cycle history, graceful termination, vocabulary expansion,
garbage rejection, and stagnation detection.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

hdc_core = pytest.importorskip("hdc_core")

from src.projection.isomorphic_projector import IsomorphicProjector
from src.decoder.subtree_vocab import SubTreeVocabulary
from src.decoder.constraint_decoder import ConstraintDecoder
from src.decoder.vocab_expander import VocabularyExpander
from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer
from src.synthesis.serl import SERLLoop, SERLResult
from src.execution.sandbox import ExecutionSandbox, ExecutionProfile


DIM = 256
CAPACITY = 2_000_000


def _build_pipeline(corpus: dict, resonance_threshold: float = -1.0):
    """Build a full pipeline from a corpus dict."""
    arena = hdc_core.FhrrArena(CAPACITY, DIM)
    projector = IsomorphicProjector(arena, DIM)
    synthesizer = AxiomaticSynthesizer(arena, projector, resonance_threshold=resonance_threshold)
    synthesizer.ingest_batch(corpus)

    # Build subtree vocab from corpus
    vocab = SubTreeVocabulary()
    for source in corpus.values():
        vocab.ingest_source(source)
    vocab.project_all(arena, projector)

    decoder = ConstraintDecoder(
        arena, projector, DIM,
        activation_threshold=0.04,
        subtree_vocab=vocab,
    )
    sandbox = ExecutionSandbox(timeout_s=3.0)
    expander = VocabularyExpander()
    loop = SERLLoop()

    return arena, projector, synthesizer, decoder, sandbox, expander, vocab, loop


# Test 1: 3 cycles on 4-function corpus → cycle_history has 3 entries
def test_cycle_history():
    """Run 3 cycles and verify cycle_history structure."""
    corpus = {
        "identity": "def f(x): return x",
        "double":   "def g(x): return x * 2",
        "negate":   "def h(x): return -x",
        "square":   "def k(x): return x * x",
    }
    arena, projector, synthesizer, decoder, sandbox, expander, vocab, loop = _build_pipeline(corpus)

    result = loop.run(
        arena, projector, synthesizer, decoder,
        sandbox, expander, vocab,
        max_cycles=3,
        stagnation_limit=10,  # Don't halt early
        fitness_threshold=0.4,
    )

    assert result.cycles_completed == 3
    assert len(result.cycle_history) == 3
    for entry in result.cycle_history:
        assert hasattr(entry, 'cycle_index')
        assert hasattr(entry, 'syntheses_attempted')
        assert hasattr(entry, 'new_vocab_atoms')
        assert hasattr(entry, 'f_eff_expanded')
        assert hasattr(entry, 'best_fitness')


# Test 2: single-function corpus (no cliques) → terminates gracefully
def test_single_function_corpus():
    """Single function → no cliques → terminates gracefully."""
    corpus = {"only": "def f(x): return x"}
    arena, projector, synthesizer, decoder, sandbox, expander, vocab, loop = _build_pipeline(corpus)

    result = loop.run(
        arena, projector, synthesizer, decoder,
        sandbox, expander, vocab,
        max_cycles=3,
        stagnation_limit=2,
    )

    # Should complete without error
    assert result.cycles_completed >= 1
    # All cycles should have 0 syntheses attempted (no cliques with 1 axiom)
    for entry in result.cycle_history:
        assert entry.syntheses_attempted == 0


# Test 3: simple corpus that MUST produce vocabulary expansion
def test_vocabulary_expansion():
    """The 4-function corpus MUST produce at least one new vocab atom."""
    corpus = {
        "identity": "def f(x): return x",
        "double":   "def g(x): return x * 2",
        "negate":   "def h(x): return -x",
        "square":   "def k(x): return x * x",
    }
    arena, projector, synthesizer, decoder, sandbox, expander, vocab, loop = _build_pipeline(corpus)

    result = loop.run(
        arena, projector, synthesizer, decoder,
        sandbox, expander, vocab,
        max_cycles=5,
        stagnation_limit=10,
        fitness_threshold=0.4,
    )

    assert result.vocab_size_final >= result.vocab_size_initial, (
        f"Vocabulary must not shrink: {result.vocab_size_initial} → {result.vocab_size_final}"
    )
    # At least one cycle should produce new atoms
    assert result.total_new_atoms > 0 or result.vocab_size_final > result.vocab_size_initial, (
        f"4-function corpus must produce vocabulary expansion. "
        f"Initial: {result.vocab_size_initial}, Final: {result.vocab_size_final}, "
        f"New atoms: {result.total_new_atoms}. "
        f"Cycle details: {[(c.syntheses_decoded, c.best_fitness, c.new_vocab_atoms) for c in result.cycle_history]}"
    )


# Test 4: mock sandbox returning fitness=0 → garbage not reingested
def test_garbage_not_reingested():
    """When sandbox returns fitness=0 for everything, no atoms are added."""
    corpus = {
        "identity": "def f(x): return x",
        "double":   "def g(x): return x * 2",
        "negate":   "def h(x): return -x",
        "square":   "def k(x): return x * x",
    }
    arena, projector, synthesizer, decoder, _, expander, vocab, loop = _build_pipeline(corpus)

    # Mock sandbox that always returns fitness = 0
    mock_sandbox = MagicMock(spec=ExecutionSandbox)
    mock_profile = ExecutionProfile(source="", fitness=0.0)
    mock_sandbox.execute.return_value = mock_profile

    result = loop.run(
        arena, projector, synthesizer, decoder,
        mock_sandbox, expander, vocab,
        max_cycles=3,
        stagnation_limit=10,
    )

    for entry in result.cycle_history:
        assert entry.new_vocab_atoms == 0, (
            f"Cycle {entry.cycle_index}: zero-fitness code should not produce new atoms"
        )


# Test 5: stagnation detection
def test_stagnation_detection():
    """With mock zero-fitness sandbox and stagnation_limit=2, halts after 2 cycles."""
    corpus = {
        "identity": "def f(x): return x",
        "double":   "def g(x): return x * 2",
        "negate":   "def h(x): return -x",
        "square":   "def k(x): return x * x",
    }
    arena, projector, synthesizer, decoder, _, expander, vocab, loop = _build_pipeline(corpus)

    # Mock sandbox that always returns fitness = 0
    mock_sandbox = MagicMock(spec=ExecutionSandbox)
    mock_profile = ExecutionProfile(source="", fitness=0.0)
    mock_sandbox.execute.return_value = mock_profile

    result = loop.run(
        arena, projector, synthesizer, decoder,
        mock_sandbox, expander, vocab,
        max_cycles=20,
        stagnation_limit=2,
    )

    assert result.space_closed is True
    assert result.cycles_completed == 2, (
        f"Should halt after 2 cycles (stagnation_limit=2), got {result.cycles_completed}"
    )
