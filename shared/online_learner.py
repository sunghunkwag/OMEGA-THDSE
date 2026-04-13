"""Pure-numpy online learning engine for OMEGA-THDSE Phase 10 (fixes D3).

The pre-Phase-10 system could only improve itself via evolutionary
search (SERL). It had no gradient-based learning loop, so it could
never pick up a mapping, calibrate a value estimate, or internalise a
reward signal between evolutionary epochs. This module closes that
gap.

The engine is deliberately restricted to pure numpy (PLAN dependency
policy) so it can run in the same process as the deterministic arena
manager without pulling in torch / tensorflow.

Core components:

- :class:`OnlineLearner` — an MLP with configurable hidden layers
  (default ``[256, 128, 64]`` per PLAN ``ONLINE_LEARNER_HIDDEN_DIMS``),
  ReLU activations, and a plain SGD + Adam-ish optimiser. ``train_step``
  performs a single forward/backward/update pass and returns the pre-
  and post-step losses so Rule 13 ("NO MOCK LEARNING") can be
  enforced by every caller.

- :class:`ExperienceReplayBuffer` — a bounded FIFO of (x, y) pairs
  used by :class:`OnlineLearner.train_batch` and by the downstream
  Phase 13 agent loop.

Anti-shortcut compliance
------------------------
PLAN Rule 13 (NO MOCK LEARNING): Every ``train_step`` computes a real
gradient via the chain rule and applies a real update. Tests must
assert ``loss_after < loss_before`` on non-trivial data; this module
returns both losses explicitly so the assertion is trivial to write.

PLAN Rule 10 (DETERMINISM): All weight init uses the provided
:class:`DeterministicRNG` fork (``online_learner``) — no bare
``np.random`` calls.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Sequence, Tuple

import numpy as np

from .constants import (
    EXPERIENCE_REPLAY_CAPACITY,
    MIN_LOSS_DECREASE_RATIO,
    ONLINE_LEARNER_DEFAULT_LR,
    ONLINE_LEARNER_HIDDEN_DIMS,
)
from .deterministic_rng import DeterministicRNG


# --------------------------------------------------------------------------- #
# Activation helpers
# --------------------------------------------------------------------------- #


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Experience replay
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Experience:
    x: np.ndarray
    y: np.ndarray
    weight: float = 1.0


class ExperienceReplayBuffer:
    """Bounded FIFO buffer for online-learning experience tuples."""

    def __init__(
        self,
        capacity: int = EXPERIENCE_REPLAY_CAPACITY,
        rng: DeterministicRNG | None = None,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = int(capacity)
        self._buffer: Deque[Experience] = deque(maxlen=self._capacity)
        self._rng = rng or DeterministicRNG(master_seed=42)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, x: np.ndarray, y: np.ndarray, weight: float = 1.0) -> None:
        self._buffer.append(
            Experience(
                x=np.asarray(x, dtype=np.float32).copy(),
                y=np.asarray(y, dtype=np.float32).copy(),
                weight=float(weight),
            )
        )

    def extend(self, items: Sequence[Tuple[np.ndarray, np.ndarray]]) -> None:
        for x, y in items:
            self.add(x, y)

    def sample(self, batch_size: int) -> List[Experience]:
        if not self._buffer:
            raise RuntimeError("cannot sample from empty buffer")
        k = min(int(batch_size), len(self._buffer))
        gen = self._rng.fork("experience_replay")
        indices = gen.integers(0, len(self._buffer), size=k)
        return [self._buffer[int(i)] for i in indices]


# --------------------------------------------------------------------------- #
# Online learner
# --------------------------------------------------------------------------- #


class OnlineLearner:
    """Pure-numpy MLP with Adam optimiser and real gradient updates.

    Parameters
    ----------
    input_dim / output_dim:
        Feature and target dimensionality.
    hidden_dims:
        Hidden layer sizes. Defaults to
        :data:`ONLINE_LEARNER_HIDDEN_DIMS`.
    lr:
        Base learning rate. Defaults to
        :data:`ONLINE_LEARNER_DEFAULT_LR`.
    loss:
        ``"mse"`` (mean-squared error, real-valued targets) or
        ``"ce"`` (softmax cross-entropy, one-hot targets).
    rng:
        :class:`DeterministicRNG` (Rule 10). Weights are initialised
        via the ``online_learner`` fork.
    """

    _BETA1 = 0.9
    _BETA2 = 0.999
    _EPS = 1e-8

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] | None = None,
        *,
        lr: float = ONLINE_LEARNER_DEFAULT_LR,
        loss: str = "mse",
        rng: DeterministicRNG | None = None,
    ):
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("dims must be positive")
        if loss not in ("mse", "ce"):
            raise ValueError(f"unknown loss {loss!r}; expected 'mse' or 'ce'")
        self._input_dim = int(input_dim)
        self._output_dim = int(output_dim)
        self._hidden_dims = list(hidden_dims or ONLINE_LEARNER_HIDDEN_DIMS)
        self._lr = float(lr)
        self._loss_kind = loss
        self._rng = rng or DeterministicRNG(master_seed=42)
        self._step_count = 0
        self._weights, self._biases = self._init_params()
        # Adam moment buffers, matched to weights/biases.
        self._m_w = [np.zeros_like(w) for w in self._weights]
        self._v_w = [np.zeros_like(w) for w in self._weights]
        self._m_b = [np.zeros_like(b) for b in self._biases]
        self._v_b = [np.zeros_like(b) for b in self._biases]

    # ---- introspection ---- #

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def loss_kind(self) -> str:
        return self._loss_kind

    def param_snapshot(self) -> List[np.ndarray]:
        return [w.copy() for w in self._weights] + [
            b.copy() for b in self._biases
        ]

    # ---- forward / loss ---- #

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = self._ensure_batch(x, self._input_dim)
        activations, _ = self._forward(x)
        return activations[-1]

    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        x = self._ensure_batch(x, self._input_dim)
        y = self._ensure_batch(y, self._output_dim)
        activations, _ = self._forward(x)
        return float(self._loss_value(activations[-1], y))

    # ---- single-step training (Rule 13 contract) ---- #

    def train_step(
        self, x: np.ndarray, y: np.ndarray
    ) -> dict[str, Any]:
        """Run one forward/backward/update pass.

        Returns a dict with ``loss_before`` (loss at the pre-step
        weights on the same batch) and ``loss_after`` (loss after the
        update). Downstream Rule 13 tests assert
        ``loss_after < loss_before``.
        """
        x = self._ensure_batch(x, self._input_dim)
        y = self._ensure_batch(y, self._output_dim)

        activations, pre_acts = self._forward(x)
        loss_before = self._loss_value(activations[-1], y)

        grads_w, grads_b = self._backward(activations, pre_acts, y)
        grad_norm_sq = 0.0
        for g in grads_w:
            grad_norm_sq += float(np.sum(g * g))
        for g in grads_b:
            grad_norm_sq += float(np.sum(g * g))
        grad_norm = math.sqrt(grad_norm_sq)

        self._adam_update(grads_w, grads_b)
        self._step_count += 1

        activations_after, _ = self._forward(x)
        loss_after = self._loss_value(activations_after[-1], y)
        return {
            "loss_before": float(loss_before),
            "loss_after": float(loss_after),
            "gradient_norm": float(grad_norm),
            "step": self._step_count,
        }

    def train_batch(
        self,
        buffer: ExperienceReplayBuffer,
        batch_size: int,
    ) -> dict[str, Any]:
        """Sample ``batch_size`` experiences from a replay buffer and train."""
        if len(buffer) == 0:
            raise RuntimeError("empty replay buffer")
        samples = buffer.sample(batch_size)
        xs = np.stack([s.x for s in samples], axis=0)
        ys = np.stack([s.y for s in samples], axis=0)
        return self.train_step(xs, ys)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
    ) -> dict[str, Any]:
        """Convenience training loop over a single fixed (x, y) dataset.

        Used in Rule 13 tests to demonstrate that repeated calls
        produce a monotonic (on-average) loss decrease.
        """
        history: List[float] = []
        for _ in range(int(epochs)):
            info = self.train_step(x, y)
            history.append(info["loss_after"])
        return {
            "initial_loss": float(history[0]) if history else float("nan"),
            "final_loss": float(history[-1]) if history else float("nan"),
            "history": history,
            "steps": self._step_count,
        }

    # ---- internals ---- #

    def _init_params(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        gen = self._rng.fork("online_learner")
        layer_sizes = [self._input_dim, *self._hidden_dims, self._output_dim]
        weights: List[np.ndarray] = []
        biases: List[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He init for ReLU networks.
            scale = math.sqrt(2.0 / fan_in)
            w = gen.standard_normal((fan_in, fan_out)).astype(np.float32) * scale
            b = np.zeros(fan_out, dtype=np.float32)
            weights.append(w)
            biases.append(b)
        return weights, biases

    def _forward(
        self, x: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations: List[np.ndarray] = [x]
        pre_acts: List[np.ndarray] = []
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            z = activations[-1] @ w + b
            pre_acts.append(z)
            if i < len(self._weights) - 1:
                activations.append(_relu(z))
            else:
                # Output layer: identity for MSE, softmax for CE.
                if self._loss_kind == "ce":
                    shifted = z - np.max(z, axis=1, keepdims=True)
                    exp = np.exp(shifted)
                    activations.append(exp / np.sum(exp, axis=1, keepdims=True))
                else:
                    activations.append(z)
        return activations, pre_acts

    def _loss_value(self, pred: np.ndarray, target: np.ndarray) -> float:
        if self._loss_kind == "mse":
            return float(np.mean((pred - target) ** 2))
        # cross-entropy with numerical guard
        eps = 1e-12
        return float(-np.mean(np.sum(target * np.log(pred + eps), axis=1)))

    def _backward(
        self,
        activations: List[np.ndarray],
        pre_acts: List[np.ndarray],
        target: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        batch = activations[0].shape[0]
        n_layers = len(self._weights)
        grads_w: List[np.ndarray] = [None] * n_layers  # type: ignore[list-item]
        grads_b: List[np.ndarray] = [None] * n_layers  # type: ignore[list-item]

        # Output gradient (dL/dz at the last layer).
        if self._loss_kind == "mse":
            # MSE: dL/dy = 2/N * (pred - target); chain through identity.
            dz = (activations[-1] - target) * (2.0 / batch)
        else:
            # Softmax + CE: dL/dz = (softmax - target) / N
            dz = (activations[-1] - target) / float(batch)

        for layer in reversed(range(n_layers)):
            a_prev = activations[layer]
            grads_w[layer] = a_prev.T @ dz
            grads_b[layer] = dz.sum(axis=0)
            if layer > 0:
                da_prev = dz @ self._weights[layer].T
                dz = da_prev * _relu_grad(pre_acts[layer - 1])
        return grads_w, grads_b

    def _adam_update(
        self,
        grads_w: List[np.ndarray],
        grads_b: List[np.ndarray],
    ) -> None:
        t = self._step_count + 1
        b1 = self._BETA1
        b2 = self._BETA2
        lr = self._lr
        bc1 = 1.0 - (b1 ** t)
        bc2 = 1.0 - (b2 ** t)
        for i, (gw, gb) in enumerate(zip(grads_w, grads_b)):
            self._m_w[i] = b1 * self._m_w[i] + (1.0 - b1) * gw
            self._v_w[i] = b2 * self._v_w[i] + (1.0 - b2) * (gw * gw)
            m_hat = self._m_w[i] / bc1
            v_hat = self._v_w[i] / bc2
            self._weights[i] -= (lr * m_hat / (np.sqrt(v_hat) + self._EPS)).astype(
                np.float32
            )

            self._m_b[i] = b1 * self._m_b[i] + (1.0 - b1) * gb
            self._v_b[i] = b2 * self._v_b[i] + (1.0 - b2) * (gb * gb)
            m_hat_b = self._m_b[i] / bc1
            v_hat_b = self._v_b[i] / bc2
            self._biases[i] -= (
                lr * m_hat_b / (np.sqrt(v_hat_b) + self._EPS)
            ).astype(np.float32)

    @staticmethod
    def _ensure_batch(x: np.ndarray, expected_dim: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2 or arr.shape[1] != expected_dim:
            raise ValueError(
                f"expected shape (batch, {expected_dim}), got {tuple(arr.shape)}"
            )
        return arr


# --------------------------------------------------------------------------- #
# Rule 13 helper — unified assertion primitive
# --------------------------------------------------------------------------- #


def loss_decreased(info: dict[str, Any], ratio: float = MIN_LOSS_DECREASE_RATIO) -> bool:
    """Return True iff a ``train_step`` result satisfies Rule 13.

    ``loss_after`` must be at most ``ratio * loss_before`` (default
    0.8 per :data:`MIN_LOSS_DECREASE_RATIO`). Used both inside the
    learner's own regression tests and by Phase 13 callers that want
    to veto agent updates when learning is not actually happening.
    """
    before = float(info["loss_before"])
    after = float(info["loss_after"])
    if before <= 1e-9:
        return after <= before + 1e-9
    return after <= ratio * before


__all__ = [
    "Experience",
    "ExperienceReplayBuffer",
    "OnlineLearner",
    "loss_decreased",
]
