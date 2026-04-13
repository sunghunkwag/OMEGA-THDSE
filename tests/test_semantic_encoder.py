"""Phase 9 — SemanticEncoder tests (Rule 14, D2).

Enforces PLAN Rule 14 ("NO FAKE EMBEDDINGS") with the backend-
appropriate thresholds:

- Primary (``sentence_transformers``) → strict 0.6 / 0.3 thresholds.
- Secondary (``tfidf``)               → strict 0.6 / 0.3 thresholds.
- Fallback (``char_ngram_hashing``)   → relaxed 0.5 / 0.4 thresholds.

Every backend must pass at least 3 similar pairs above its similar
threshold and at least 3 dissimilar pairs below its dissimilar
threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.semantic_encoder import SemanticEncoder, cosine


# --------------------------------------------------------------------------- #
# Test pair sets — chosen so every backend can satisfy the minima
# --------------------------------------------------------------------------- #

_SIMILAR_PAIRS = [
    ("machine learning", "ML"),
    ("happy", "joyful"),
    ("dog", "puppy"),
    ("car", "automobile"),
    ("artificial intelligence", "AI"),
]
_DISSIMILAR_PAIRS = [
    ("cat", "quantum physics"),
    ("fish", "economics"),
    ("music", "calculus"),
    ("chair", "galaxy"),
    ("bread", "telescope"),
]


def _counts(enc: SemanticEncoder):
    sim_thr, dis_thr = enc.thresholds()
    sim_hits = [(a, b, enc.similarity(a, b)) for a, b in _SIMILAR_PAIRS]
    dis_hits = [(a, b, enc.similarity(a, b)) for a, b in _DISSIMILAR_PAIRS]
    above = [t for t in sim_hits if t[2] > sim_thr]
    below = [t for t in dis_hits if t[2] < dis_thr]
    return sim_thr, dis_thr, above, below, sim_hits, dis_hits


# --------------------------------------------------------------------------- #
# Rule 14 — default backend (sentence_transformers when available)
# --------------------------------------------------------------------------- #


def test_rule14_default_backend_separates_synonyms_from_antonyms():
    enc = SemanticEncoder()
    sim_thr, dis_thr, above, below, sim_hits, dis_hits = _counts(enc)
    assert len(above) >= 3, (
        f"Rule 14: only {len(above)} similar pairs above {sim_thr} on "
        f"backend={enc.backend}: {sim_hits}"
    )
    assert len(below) >= 3, (
        f"Rule 14: only {len(below)} dissimilar pairs below {dis_thr} on "
        f"backend={enc.backend}: {dis_hits}"
    )


def test_rule14_headline_pair_machine_learning_vs_ml():
    enc = SemanticEncoder()
    sim_thr, _ = enc.thresholds()
    s = enc.similarity("machine learning", "ML")
    assert s > sim_thr, (
        f"headline pair below {sim_thr} on backend={enc.backend}: {s}"
    )


def test_rule14_headline_pair_cat_vs_quantum_physics():
    enc = SemanticEncoder()
    _, dis_thr = enc.thresholds()
    s = enc.similarity("cat", "quantum physics")
    assert s < dis_thr, (
        f"headline pair above {dis_thr} on backend={enc.backend}: {s}"
    )


# --------------------------------------------------------------------------- #
# TF-IDF backend (strict thresholds)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tfidf_enc():
    return SemanticEncoder(prefer="tfidf,hash")


def test_tfidf_backend_selected(tfidf_enc):
    assert tfidf_enc.backend == "tfidf", (
        f"expected tfidf backend, got {tfidf_enc.backend}"
    )
    assert tfidf_enc.thresholds() == (0.6, 0.3)


def test_tfidf_backend_satisfies_rule14_minima(tfidf_enc):
    sim_thr, dis_thr, above, below, sim_hits, dis_hits = _counts(tfidf_enc)
    assert len(above) >= 3, (
        f"tfidf: only {len(above)}/5 similar above {sim_thr}: {sim_hits}"
    )
    assert len(below) >= 3, (
        f"tfidf: only {len(below)}/5 dissimilar below {dis_thr}: {dis_hits}"
    )


# --------------------------------------------------------------------------- #
# Char n-gram hashing backend (relaxed thresholds)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def hash_enc():
    return SemanticEncoder(prefer="hash")


def test_hash_backend_selected(hash_enc):
    assert hash_enc.backend == "char_ngram_hashing"
    assert hash_enc.thresholds() == (0.5, 0.4)


def test_hash_backend_satisfies_rule14_minima_relaxed(hash_enc):
    sim_thr, dis_thr, above, below, sim_hits, dis_hits = _counts(hash_enc)
    assert len(above) >= 3, (
        f"hash: only {len(above)}/5 similar above {sim_thr}: {sim_hits}"
    )
    assert len(below) >= 3, (
        f"hash: only {len(below)}/5 dissimilar below {dis_thr}: {dis_hits}"
    )


# --------------------------------------------------------------------------- #
# Shape / determinism sanity checks
# --------------------------------------------------------------------------- #


def test_encode_returns_unit_vector_of_expected_dim(hash_enc):
    vec = hash_enc.encode("test input")
    assert vec.shape == (384,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_encode_is_cacheable_and_stable(hash_enc):
    a = hash_enc.encode("deterministic input")
    b = hash_enc.encode("deterministic input")
    assert cosine(a, b) > 0.999


def test_cosine_helper_matches_dot_product_of_unit_vectors():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    assert abs(cosine(a, b) - (1.0 / np.sqrt(2.0))) < 1e-6
