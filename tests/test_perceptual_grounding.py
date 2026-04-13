"""Phase 9 — PerceptualGrounder tests (D1).

Proves the grounder turns text / structured data / files into valid
10,000-dim CCE phase vectors whose geometry tracks the underlying
semantic cosine (not zlib.crc32 noise).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shared.constants import CCE_ARENA_DIM, SEMANTIC_ENCODER_DIM
from shared.perceptual_grounding import PerceptualGrounder
from shared.semantic_encoder import SemanticEncoder


@pytest.fixture(scope="module")
def grounder():
    # Force the hashing backend so tests run without downloading the
    # transformer model. Rule 14 thresholds are relaxed for this
    # backend but the grounder tests care about *shape + provenance +
    # ordering*, not the strict Rule 14 separation.
    enc = SemanticEncoder(prefer="hash")
    return PerceptualGrounder(encoder=enc)


def test_ingest_text_returns_cce_phase_vector(grounder):
    result = grounder.ingest_text("hello world")
    phases = result["phases"]
    assert phases.shape == (CCE_ARENA_DIM,)
    assert phases.dtype == np.float32
    assert float(phases.min()) >= 0.0
    assert float(phases.max()) < 2.0 * np.pi


def test_ingest_text_produces_semantic_vector_of_encoder_dim(grounder):
    result = grounder.ingest_text("hello world")
    assert result["semantic_vector"].shape == (SEMANTIC_ENCODER_DIM,)


def test_ingest_text_is_deterministic(grounder):
    r1 = grounder.ingest_text("repeated input")
    r2 = grounder.ingest_text("repeated input")
    assert np.allclose(r1["phases"], r2["phases"])
    assert np.allclose(r1["semantic_vector"], r2["semantic_vector"])


def test_ingest_text_provenance_required(grounder):
    result = grounder.ingest_text("provenance check")
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "ingest_text"
    assert prov["source_arena"] == "perceptual"
    assert prov["target_arena"] == "cce"
    assert prov["encoder_dim"] == SEMANTIC_ENCODER_DIM
    assert prov["target_dim"] == CCE_ARENA_DIM


def test_similar_texts_have_higher_grounded_similarity(grounder):
    sim_pair = grounder.similarity(
        grounder.ingest_text("machine learning"),
        grounder.ingest_text("ML"),
    )
    dis_pair = grounder.similarity(
        grounder.ingest_text("cat"),
        grounder.ingest_text("quantum physics"),
    )
    assert sim_pair > dis_pair, (
        f"semantic pair ({sim_pair}) should exceed dissimilar pair "
        f"({dis_pair}) — grounder is blind to meaning"
    )


def test_ingest_structured_flattens_dicts(grounder):
    data = {
        "name": "widget",
        "tags": ["alpha", "beta"],
        "count": 7,
    }
    result = grounder.ingest_structured(data)
    assert result["phases"].shape == (CCE_ARENA_DIM,)
    assert result["metadata"]["token_count"] >= 4
    assert result["metadata"]["provenance"]["operation"] == "ingest_structured"


def test_ingest_structured_is_order_independent_for_dicts(grounder):
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"c": 3, "a": 1, "b": 2}
    r1 = grounder.ingest_structured(d1)
    r2 = grounder.ingest_structured(d2)
    assert np.allclose(r1["phases"], r2["phases"])


def test_ingest_file_reads_text(tmp_path, grounder):
    p: Path = tmp_path / "sample.txt"
    p.write_text("this is a text sample", encoding="utf-8")
    result = grounder.ingest_file(p)
    assert result["metadata"]["source_path"] == str(p)
    assert result["metadata"]["file_extension"] == "txt"
    assert result["phases"].shape == (CCE_ARENA_DIM,)


def test_ingest_file_reads_json(tmp_path, grounder):
    p: Path = tmp_path / "sample.json"
    p.write_text(json.dumps({"kind": "test", "value": 42}), encoding="utf-8")
    result = grounder.ingest_file(p)
    assert result["metadata"]["file_extension"] == "json"
    assert result["metadata"]["provenance"]["operation"] == "ingest_structured"


def test_ingest_file_missing_raises(tmp_path, grounder):
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        grounder.ingest_file(missing)


def test_ingest_increments_counter(grounder):
    before = grounder.ingest_count
    grounder.ingest_text("counter probe")
    grounder.ingest_text("counter probe 2")
    assert grounder.ingest_count == before + 2


def test_non_dummy_grounding_text_bytes_differ():
    """D1 fix: different inputs must produce different phase vectors.

    The pre-Phase-9 system only ingested .py files and produced hash
    collisions for anything else. After Phase 9, two different text
    inputs must yield phase vectors that differ in >40% of components.
    """
    enc = SemanticEncoder(prefer="hash")
    g = PerceptualGrounder(encoder=enc)
    a = g.ingest_text("the quick brown fox").get("phases")
    b = g.ingest_text("algorithmic abstract proof").get("phases")
    diff_frac = float(np.mean(np.abs(a - b) > 1e-6))
    assert diff_frac > 0.4, (
        f"perceptual grounding collapses distinct inputs "
        f"(diff_frac={diff_frac:.3f})"
    )
