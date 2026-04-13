"""Phase 10 bridge — Continuous Learning → Arena (fixes D3).

Wires the Phase 10 :class:`OnlineLearner` into the unified arena so
that CCE / bridge phase vectors can be used as features / targets for
gradient-based learning, and the resulting predictions can be stored
back into the arena as fresh handles.

Rule 13 contract
----------------
Every training method on the bridge returns the raw
``OnlineLearner.train_step`` / ``train_batch`` result so callers can
assert ``loss_after < loss_before`` without ambiguity.

Rule 20 wiring
--------------
This bridge imports :class:`SemanticConceptBridge` from Phase 9 so
that a single API call (:meth:`ContinuousLearningBridge.learn_text_mapping`)
can ground raw text, train the learner on the grounded vectors, and
return provenance-stamped results. In turn, the bridge is imported
by :mod:`bridges.self_model_bridge` in Phase 11 integration (by
making self-model updates call ``train_step`` on the shared learner).

Every return carries ``metadata["provenance"]`` (Rule 9).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np

from shared.arena_manager import ArenaManager
from shared.constants import (
    CCE_ARENA_DIM,
    EXPERIENCE_REPLAY_CAPACITY,
    ONLINE_LEARNER_DEFAULT_LR,
    ONLINE_LEARNER_HIDDEN_DIMS,
    SEMANTIC_ENCODER_DIM,
)
from shared.online_learner import (
    ExperienceReplayBuffer,
    OnlineLearner,
    loss_decreased,
)
from shared.semantic_encoder import SemanticEncoder
from bridges.semantic_concept_bridge import SemanticConceptBridge


class ContinuousLearningBridge:
    """Couple an :class:`OnlineLearner` to the shared arena.

    The default configuration learns a mapping from the 384-dim
    semantic encoder output to a caller-provided target dimension,
    which is the right shape for Phase 11 episodic → semantic
    consolidation and Phase 12 heuristic regression.
    """

    def __init__(
        self,
        arena_manager: ArenaManager,
        *,
        semantic_bridge: SemanticConceptBridge | None = None,
        input_dim: int = SEMANTIC_ENCODER_DIM,
        output_dim: int = 1,
        hidden_dims: Sequence[int] | None = None,
        lr: float = ONLINE_LEARNER_DEFAULT_LR,
        loss: str = "mse",
        replay_capacity: int = EXPERIENCE_REPLAY_CAPACITY,
    ):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._semantic_bridge = semantic_bridge or SemanticConceptBridge(
            arena_manager, encoder=SemanticEncoder(rng=arena_manager.rng)
        )
        self._learner = OnlineLearner(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims or ONLINE_LEARNER_HIDDEN_DIMS,
            lr=lr,
            loss=loss,
            rng=arena_manager.rng,
        )
        self._buffer = ExperienceReplayBuffer(
            capacity=replay_capacity, rng=arena_manager.rng
        )
        self._train_log: List[Dict[str, Any]] = []

    # ---- introspection ---- #

    @property
    def learner(self) -> OnlineLearner:
        return self._learner

    @property
    def buffer(self) -> ExperienceReplayBuffer:
        return self._buffer

    @property
    def training_events(self) -> int:
        return len(self._train_log)

    # ---- ingest experiences ---- #

    def add_experience(
        self, x: np.ndarray, y: np.ndarray, weight: float = 1.0
    ) -> Dict[str, Any]:
        """Append an ``(x, y)`` pair to the replay buffer."""
        self._buffer.add(x, y, weight=weight)
        return {
            "buffer_size": len(self._buffer),
            "metadata": {
                "provenance": {
                    "operation": "add_experience",
                    "source_arena": "external",
                    "target_arena": "replay_buffer",
                    "timestamp": time.time(),
                }
            },
        }

    def ingest_text_experience(
        self, text: str, target: np.ndarray, weight: float = 1.0
    ) -> Dict[str, Any]:
        """Encode ``text`` via the semantic bridge and add it as a sample."""
        grounded = self._semantic_bridge.ground_text(text)
        sem = np.asarray(grounded["semantic_vector"], dtype=np.float32)
        self._buffer.add(sem, target, weight=weight)
        return {
            "cce_handle": grounded["cce_handle"],
            "buffer_size": len(self._buffer),
            "metadata": {
                "grounding_provenance": grounded["metadata"]["provenance"],
                "provenance": {
                    "operation": "ingest_text_experience",
                    "source_arena": "text",
                    "target_arena": "replay_buffer",
                    "timestamp": time.time(),
                },
            },
        }

    # ---- training ---- #

    def train_step(
        self, x: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """Single gradient step; surfaces before/after loss for Rule 13."""
        info = self._learner.train_step(x, y)
        decreased = loss_decreased(info)
        self._train_log.append(info)
        return {
            "loss_before": info["loss_before"],
            "loss_after": info["loss_after"],
            "gradient_norm": info["gradient_norm"],
            "loss_decreased": decreased,
            "step": info["step"],
            "metadata": {
                "provenance": {
                    "operation": "train_step",
                    "source_arena": "external",
                    "target_arena": "online_learner",
                    "loss_kind": self._learner.loss_kind,
                    "timestamp": time.time(),
                }
            },
        }

    def train_batch(self, batch_size: int) -> Dict[str, Any]:
        """Sample from the replay buffer and run one gradient step."""
        info = self._learner.train_batch(self._buffer, batch_size)
        decreased = loss_decreased(info)
        self._train_log.append(info)
        return {
            "loss_before": info["loss_before"],
            "loss_after": info["loss_after"],
            "gradient_norm": info["gradient_norm"],
            "loss_decreased": decreased,
            "step": info["step"],
            "batch_size": batch_size,
            "metadata": {
                "provenance": {
                    "operation": "train_batch",
                    "source_arena": "replay_buffer",
                    "target_arena": "online_learner",
                    "timestamp": time.time(),
                }
            },
        }

    def learn_text_mapping(
        self,
        samples: Sequence[tuple[str, np.ndarray]],
        epochs: int = 10,
    ) -> Dict[str, Any]:
        """Train on a list of ``(text, target)`` pairs for ``epochs`` passes.

        The method (1) grounds every text through the Phase 9
        semantic bridge, (2) appends the resulting sample to the
        replay buffer, (3) trains with mini-batches of the whole set
        for ``epochs`` passes, and (4) returns the initial / final
        losses so callers can assert Rule 13.
        """
        if not samples:
            raise ValueError("samples must be non-empty")
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        cce_handles: List[int] = []
        for text, target in samples:
            grounded = self._semantic_bridge.ground_text(text)
            sem = np.asarray(grounded["semantic_vector"], dtype=np.float32)
            tgt = np.asarray(target, dtype=np.float32).reshape(-1)
            xs.append(sem)
            ys.append(tgt)
            cce_handles.append(grounded["cce_handle"])
            self._buffer.add(sem, tgt)

        x_batch = np.stack(xs, axis=0)
        y_batch = np.stack(ys, axis=0)

        initial_loss = self._learner.compute_loss(x_batch, y_batch)
        history: List[float] = []
        for _ in range(int(epochs)):
            info = self._learner.train_step(x_batch, y_batch)
            history.append(info["loss_after"])
        final_loss = history[-1] if history else initial_loss
        self._train_log.append(
            {
                "loss_before": initial_loss,
                "loss_after": final_loss,
                "step": self._learner.step_count,
            }
        )
        return {
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "loss_decreased": final_loss < initial_loss,
            "history": history,
            "cce_handles": cce_handles,
            "metadata": {
                "provenance": {
                    "operation": "learn_text_mapping",
                    "source_arena": "text",
                    "target_arena": "online_learner",
                    "samples": len(samples),
                    "epochs": int(epochs),
                    "timestamp": time.time(),
                }
            },
        }

    def predict_from_text(self, text: str) -> Dict[str, Any]:
        """Ground ``text`` and run the learner forward pass."""
        grounded = self._semantic_bridge.ground_text(text)
        sem = np.asarray(grounded["semantic_vector"], dtype=np.float32)
        prediction = self._learner.predict(sem)[0]
        return {
            "prediction": prediction,
            "cce_handle": grounded["cce_handle"],
            "metadata": {
                "grounding_provenance": grounded["metadata"]["provenance"],
                "provenance": {
                    "operation": "predict_from_text",
                    "source_arena": "text",
                    "target_arena": "online_learner",
                    "timestamp": time.time(),
                },
            },
        }


__all__ = ["ContinuousLearningBridge"]
