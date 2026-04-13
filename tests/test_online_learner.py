"""Phase 10 — OnlineLearner tests (Rule 13, D3)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.constants import MIN_LOSS_DECREASE_RATIO
from shared.deterministic_rng import DeterministicRNG
from shared.online_learner import (
    ExperienceReplayBuffer,
    OnlineLearner,
    loss_decreased,
)


def _regression_dataset(n: int = 64, d_in: int = 8, d_out: int = 2):
    rng = np.random.default_rng(2024)
    x = rng.standard_normal((n, d_in)).astype(np.float32)
    # Non-trivial non-linear target: ReLU(x @ W) @ V
    w = rng.standard_normal((d_in, 16)).astype(np.float32) * 0.5
    v = rng.standard_normal((16, d_out)).astype(np.float32) * 0.5
    y = np.maximum(x @ w, 0.0) @ v
    return x.astype(np.float32), y.astype(np.float32)


# --------------------------------------------------------------------------- #
# Rule 13 — real gradient updates must reduce loss on non-trivial data
# --------------------------------------------------------------------------- #


def test_rule13_train_step_decreases_loss_on_nontrivial_data():
    x, y = _regression_dataset()
    learner = OnlineLearner(
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        hidden_dims=[32, 16],
        lr=0.01,
        rng=DeterministicRNG(master_seed=2025),
    )
    # Average over 30 steps to avoid flaky single-step minibatch noise.
    initial = learner.compute_loss(x, y)
    for _ in range(30):
        learner.train_step(x, y)
    final = learner.compute_loss(x, y)
    assert final < initial * MIN_LOSS_DECREASE_RATIO, (
        f"Rule 13: loss did not drop enough: {initial:.4f} -> {final:.4f}"
    )


def test_rule13_single_train_step_returns_both_losses():
    x, y = _regression_dataset(n=16, d_in=4, d_out=2)
    learner = OnlineLearner(
        input_dim=4, output_dim=2, hidden_dims=[8], lr=0.05,
        rng=DeterministicRNG(master_seed=7),
    )
    info = learner.train_step(x, y)
    assert "loss_before" in info and "loss_after" in info
    assert info["gradient_norm"] > 0.0
    # The very first step on a freshly-initialised net should reduce loss.
    assert info["loss_after"] < info["loss_before"]


def test_loss_decreased_helper_honors_rule13_ratio():
    assert loss_decreased({"loss_before": 1.0, "loss_after": 0.5})
    assert not loss_decreased({"loss_before": 1.0, "loss_after": 0.9})


# --------------------------------------------------------------------------- #
# Weight-update verification — Rule 13 "real parameter update"
# --------------------------------------------------------------------------- #


def test_train_step_actually_mutates_weights():
    learner = OnlineLearner(
        input_dim=4, output_dim=2, hidden_dims=[6], lr=0.01,
        rng=DeterministicRNG(master_seed=99),
    )
    before = learner.param_snapshot()
    x, y = _regression_dataset(n=8, d_in=4, d_out=2)
    learner.train_step(x, y)
    after = learner.param_snapshot()
    # At least one weight matrix must differ by more than numerical noise.
    assert any(
        not np.allclose(b, a, atol=1e-6) for b, a in zip(before, after)
    ), "Rule 13: train_step did not modify any parameter"


# --------------------------------------------------------------------------- #
# Cross-entropy path
# --------------------------------------------------------------------------- #


def test_cross_entropy_training_reduces_loss_on_classification_toy():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((40, 6)).astype(np.float32)
    # Binary class based on the sign of a linear combo.
    labels = (x @ np.array([1.0, -1.0, 0.5, 0.0, -0.5, 1.0]) > 0).astype(int)
    y = np.zeros((len(labels), 2), dtype=np.float32)
    y[np.arange(len(labels)), labels] = 1.0

    learner = OnlineLearner(
        input_dim=6, output_dim=2, hidden_dims=[16], lr=0.05, loss="ce",
        rng=DeterministicRNG(master_seed=33),
    )
    initial = learner.compute_loss(x, y)
    for _ in range(50):
        learner.train_step(x, y)
    final = learner.compute_loss(x, y)
    assert final < initial * 0.7


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #


def test_replay_buffer_respects_capacity():
    buf = ExperienceReplayBuffer(capacity=3)
    for i in range(10):
        buf.add(np.array([i], dtype=np.float32), np.array([-i], dtype=np.float32))
    assert len(buf) == 3
    # Only the last 3 survive.
    last_values = [float(e.x[0]) for e in list(buf._buffer)]
    assert last_values == [7.0, 8.0, 9.0]


def test_replay_buffer_sampling_is_deterministic_given_rng():
    rng = DeterministicRNG(master_seed=1234)
    buf = ExperienceReplayBuffer(capacity=100, rng=rng)
    for i in range(50):
        buf.add(np.array([float(i)]), np.array([0.0]))
    # Reset fork so both samples start from the same state.
    rng.reset("experience_replay")
    s1 = [float(e.x[0]) for e in buf.sample(5)]
    rng.reset("experience_replay")
    s2 = [float(e.x[0]) for e in buf.sample(5)]
    assert s1 == s2


def test_train_batch_reduces_loss():
    rng = DeterministicRNG(master_seed=17)
    buf = ExperienceReplayBuffer(capacity=200, rng=rng)
    x, y = _regression_dataset(n=64, d_in=6, d_out=2)
    for xi, yi in zip(x, y):
        buf.add(xi, yi)
    learner = OnlineLearner(
        input_dim=6, output_dim=2, hidden_dims=[24], lr=0.02,
        rng=rng,
    )
    initial = learner.compute_loss(x, y)
    for _ in range(40):
        learner.train_batch(buf, batch_size=16)
    final = learner.compute_loss(x, y)
    assert final < initial * 0.8


# --------------------------------------------------------------------------- #
# Determinism of initialisation (Rule 10)
# --------------------------------------------------------------------------- #


def test_weight_init_is_reproducible():
    a = OnlineLearner(4, 2, [6], rng=DeterministicRNG(master_seed=55))
    b = OnlineLearner(4, 2, [6], rng=DeterministicRNG(master_seed=55))
    for wa, wb in zip(a.param_snapshot(), b.param_snapshot()):
        assert np.allclose(wa, wb)


def test_rejects_wrong_dim_input():
    learner = OnlineLearner(4, 2, [6], rng=DeterministicRNG(master_seed=1))
    with pytest.raises(ValueError):
        learner.train_step(
            np.zeros((3, 5), dtype=np.float32),
            np.zeros((3, 2), dtype=np.float32),
        )
