"""Phase 10 — ContinuousLearningBridge tests (D3, Rule 13, Rule 20)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.arena_manager import ArenaManager
from shared.constants import SEMANTIC_ENCODER_DIM
from shared.semantic_encoder import SemanticEncoder
from bridges.continuous_learning_bridge import ContinuousLearningBridge
from bridges.semantic_concept_bridge import SemanticConceptBridge


@pytest.fixture()
def bridge():
    mgr = ArenaManager(master_seed=1010)
    scb = SemanticConceptBridge(mgr, encoder=SemanticEncoder(prefer="hash"))
    return ContinuousLearningBridge(
        mgr,
        semantic_bridge=scb,
        input_dim=SEMANTIC_ENCODER_DIM,
        output_dim=2,
        hidden_dims=[32],
        lr=0.02,
    )


def test_add_experience_grows_buffer(bridge):
    x = np.zeros(SEMANTIC_ENCODER_DIM, dtype=np.float32)
    y = np.array([1.0, 0.0], dtype=np.float32)
    res = bridge.add_experience(x, y)
    assert res["buffer_size"] == 1
    assert res["metadata"]["provenance"]["operation"] == "add_experience"


def test_train_step_reports_rule13_compliance(bridge):
    # Build a small regression dataset on the semantic encoder's space.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((16, SEMANTIC_ENCODER_DIM)).astype(np.float32)
    y = rng.standard_normal((16, 2)).astype(np.float32) * 0.1
    # Baseline loss, then run many steps — Rule 13 requires loss_after
    # to drop by >= (1 - MIN_LOSS_DECREASE_RATIO).
    initial = bridge.learner.compute_loss(x, y)
    for _ in range(40):
        bridge.train_step(x, y)
    final = bridge.learner.compute_loss(x, y)
    assert final < initial * 0.8


def test_learn_text_mapping_reduces_loss(bridge):
    samples = [
        ("machine learning", np.array([1.0, 0.0])),
        ("deep learning", np.array([1.0, 0.0])),
        ("neural network", np.array([1.0, 0.0])),
        ("quantum physics", np.array([0.0, 1.0])),
        ("general relativity", np.array([0.0, 1.0])),
        ("particle accelerator", np.array([0.0, 1.0])),
    ]
    result = bridge.learn_text_mapping(samples, epochs=50)
    assert result["final_loss"] < result["initial_loss"]
    assert result["metadata"]["provenance"]["operation"] == "learn_text_mapping"
    assert len(result["cce_handles"]) == len(samples)


def test_predict_from_text_returns_vector(bridge):
    samples = [
        ("alpha", np.array([1.0, 0.0])),
        ("beta", np.array([0.0, 1.0])),
    ]
    bridge.learn_text_mapping(samples, epochs=10)
    pred = bridge.predict_from_text("alpha")
    assert pred["prediction"].shape == (2,)
    assert pred["metadata"]["provenance"]["operation"] == "predict_from_text"


def test_rule20_bridge_imports_phase9_semantic_bridge():
    import bridges.continuous_learning_bridge as clb
    assert hasattr(clb, "SemanticConceptBridge")


def test_rule9_provenance_on_train_batch(bridge):
    # Populate buffer from text samples.
    for text in ("alpha", "beta", "gamma", "delta"):
        bridge.ingest_text_experience(text, np.array([0.1, -0.1]))
    res = bridge.train_batch(batch_size=3)
    prov = res["metadata"]["provenance"]
    assert prov["operation"] == "train_batch"
    assert prov["source_arena"] == "replay_buffer"
