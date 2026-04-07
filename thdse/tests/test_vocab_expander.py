"""Tests for the Vocabulary Expander (SERL Step 2).

6 tests verifying fitness-gated vocabulary expansion, deduplication,
projection, and activation of new atoms.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

hdc_core = pytest.importorskip("hdc_core")

from src.projection.isomorphic_projector import IsomorphicProjector
from src.decoder.subtree_vocab import SubTreeVocabulary
from src.decoder.vocab_expander import VocabularyExpander
from src.decoder.constraint_decoder import ConstraintDecoder
from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer


DIM = 256


@pytest.fixture
def pipeline():
    """Set up arena, projector, vocab, and expander."""
    arena = hdc_core.FhrrArena(500_000, DIM)
    projector = IsomorphicProjector(arena, DIM)
    vocab = SubTreeVocabulary()
    # Seed vocab with a small corpus
    vocab.ingest_source("def baseline(x): return x")
    vocab.project_all(arena, projector)
    expander = VocabularyExpander()
    return arena, projector, vocab, expander


# Test 1: fitness below threshold → 0 new atoms
def test_below_threshold(pipeline):
    """Source with fitness below threshold is NOT added to vocabulary."""
    arena, projector, vocab, expander = pipeline
    size_before = vocab.size()
    added = expander.expand(
        "def f(x): return x + 1",
        fitness=0.2,  # Below 0.4 threshold
        vocab=vocab,
        arena=arena,
        projector=projector,
    )
    assert added == 0
    assert vocab.size() == size_before


# Test 2: high fitness → at least 1 new atom
def test_above_threshold_adds_atoms(pipeline):
    """Source with fitness above threshold adds at least 1 new atom."""
    arena, projector, vocab, expander = pipeline
    added = expander.expand(
        "def f(x):\n    if x > 0:\n        return x * 2\n    return -x",
        fitness=0.6,
        vocab=vocab,
        arena=arena,
        projector=projector,
    )
    assert added >= 1


# Test 3: same source twice → second call returns 0 (dedup)
def test_idempotent(pipeline):
    """Expanding the same source twice adds 0 atoms on second call."""
    arena, projector, vocab, expander = pipeline
    source = "def g(x):\n    result = x + 10\n    return result"
    first = expander.expand(source, 0.6, vocab, arena, projector)
    assert first >= 1
    second = expander.expand(source, 0.6, vocab, arena, projector)
    assert second == 0


# Test 4: vocab.size() increases after expansion
def test_vocab_size_increases(pipeline):
    """Vocabulary size increases after successful expansion."""
    arena, projector, vocab, expander = pipeline
    size_before = vocab.size()
    expander.expand(
        "def h(x):\n    for i in range(x):\n        x = x + i\n    return x",
        fitness=0.7,
        vocab=vocab,
        arena=arena,
        projector=projector,
    )
    assert vocab.size() > size_before


# Test 5: new atoms are projected (have non-None handles)
def test_new_atoms_projected(pipeline):
    """All atoms including newly added ones have arena handles."""
    arena, projector, vocab, expander = pipeline
    expander.expand(
        "def p(x):\n    y = x * x\n    return y + 1",
        fitness=0.6,
        vocab=vocab,
        arena=arena,
        projector=projector,
    )
    projected = vocab.get_projected_atoms()
    for atom in projected:
        assert atom.handle is not None


# Test 6: new atoms are usable in probe — decoder can activate them
def test_new_atoms_activatable(pipeline):
    """Decoder's probe_subtrees can activate newly added atoms."""
    arena, projector, vocab, expander = pipeline

    # Add a distinctive pattern
    source = "def q(x):\n    if x > 0:\n        return x\n    return 0"
    expander.expand(source, 0.6, vocab, arena, projector)

    # Create a decoder with the expanded vocab
    decoder = ConstraintDecoder(
        arena, projector, DIM,
        activation_threshold=0.04,
        subtree_vocab=vocab,
    )

    # Project the same source and probe — should activate some atoms
    synth = AxiomaticSynthesizer(arena, projector, resonance_threshold=-1.0)
    projection = projector.project(source)
    activated = decoder.probe_subtrees(projection)

    # At least some atoms should activate (the source's own patterns)
    assert len(activated) > 0
