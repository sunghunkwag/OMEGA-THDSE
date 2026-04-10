"""Gap 10 — CCE RSI pipeline ↔ THDSE SERL bridge.

CCE has an RSI pipeline that quarantines, compiles, and registers
self-modifying code produced by the agent. THDSE runs SERL, an
evolutionary loop over axiom programs. Both pipelines produce
candidate programs but on their own they never see each other's
outputs. This bridge provides the bidirectional hand-off:

- :meth:`serl_candidate_to_rsi` takes a SERL candidate and, if it
  clears the ``SERL_FITNESS_GATE``, reports whether the candidate is
  eligible to enter the RSI pipeline together with a cross-arena
  similarity measurement.
- :meth:`rsi_skill_to_serl_feedback` takes a registered skill's
  execution performance history and produces a summary record that
  SERL can fold back into its fitness-shaping layer.

Both methods return structured dicts with ``metadata.provenance`` so
that an audit trail is preserved across the handoff.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from shared.arena_manager import ArenaManager
from shared.constants import (
    CCE_ARENA_DIM,
    SERL_FITNESS_GATE,
    THDSE_ARENA_DIM,
)
from shared.dimension_bridge import cross_arena_similarity


class RsiSerlBridge:
    """Bidirectional RSI ↔ SERL translation layer."""

    def __init__(self, arena_manager: ArenaManager):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._candidates_seen = 0
        self._eligible_count = 0
        self._feedback_records: list[dict[str, Any]] = []

    # ---- SERL → RSI ---- #

    def serl_candidate_to_rsi(
        self,
        serl_program_source: str,
        serl_fitness: float,
        thdse_handle: int,
    ) -> dict[str, Any]:
        """Report whether a SERL candidate is eligible for RSI intake.

        Eligibility requires: (1) fitness above ``SERL_FITNESS_GATE``,
        (2) a non-empty program source, (3) a valid THDSE handle whose
        phases can be read back. Cross-arena similarity is computed
        against a zero CCE reference so the caller can gauge how far
        the candidate is from the ambient baseline.
        """
        if not isinstance(serl_program_source, str):
            raise TypeError(
                f"serl_program_source must be str, got "
                f"{type(serl_program_source).__name__}"
            )
        if not isinstance(serl_fitness, (int, float)):
            raise TypeError(
                f"serl_fitness must be numeric, got "
                f"{type(serl_fitness).__name__}"
            )
        if not isinstance(thdse_handle, int) or thdse_handle < 0:
            raise TypeError(
                f"thdse_handle must be a non-negative int, "
                f"got {thdse_handle!r}"
            )

        fitness = float(serl_fitness)
        self._candidates_seen += 1
        timestamp = time.time()

        # Quick reject path: fitness below the gate short-circuits
        # before we touch the arena.
        if fitness < SERL_FITNESS_GATE:
            return {
                "eligible": False,
                "rsi_compatible": False,
                "fitness": fitness,
                "cross_similarity": 0.0,
                "reason": (
                    f"fitness {fitness:.3f} < SERL_FITNESS_GATE "
                    f"{SERL_FITNESS_GATE}"
                ),
                "metadata": self._build_candidate_metadata(
                    thdse_handle, fitness, timestamp, eligible=False
                ),
            }

        if not serl_program_source.strip():
            return {
                "eligible": False,
                "rsi_compatible": False,
                "fitness": fitness,
                "cross_similarity": 0.0,
                "reason": "empty program source",
                "metadata": self._build_candidate_metadata(
                    thdse_handle, fitness, timestamp, eligible=False
                ),
            }

        thdse_phases = self._mgr.get_thdse_phases(thdse_handle)
        if thdse_phases.shape != (THDSE_ARENA_DIM,):
            return {
                "eligible": False,
                "rsi_compatible": False,
                "fitness": fitness,
                "cross_similarity": 0.0,
                "reason": (
                    f"THDSE phases shape {tuple(thdse_phases.shape)} "
                    f"invalid"
                ),
                "metadata": self._build_candidate_metadata(
                    thdse_handle, fitness, timestamp, eligible=False
                ),
            }

        zero_cce: list[float] = [0.0] * self._mgr.cce_dim
        sim = cross_arena_similarity(zero_cce, thdse_phases)
        cross_similarity = float(sim["similarity"])

        self._eligible_count += 1
        return {
            "eligible": True,
            "rsi_compatible": True,
            "fitness": fitness,
            "cross_similarity": cross_similarity,
            "reason": "fitness gate passed",
            "metadata": self._build_candidate_metadata(
                thdse_handle,
                fitness,
                timestamp,
                eligible=True,
                similarity_provenance=sim["metadata"]["provenance"],
            ),
        }

    # ---- RSI → SERL ---- #

    def rsi_skill_to_serl_feedback(
        self,
        skill_id: str,
        performance_scores: list[float],
    ) -> dict[str, Any]:
        """Summarize a registered skill's performance for SERL fitness shaping."""
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ValueError("skill_id must be a non-empty string")
        if not isinstance(performance_scores, (list, tuple)):
            raise TypeError(
                f"performance_scores must be a list, got "
                f"{type(performance_scores).__name__}"
            )
        if not performance_scores:
            raise ValueError(
                "performance_scores must contain at least one sample"
            )

        numeric: list[float] = []
        for i, score in enumerate(performance_scores):
            if not isinstance(score, (int, float)):
                raise TypeError(
                    f"performance_scores[{i}] must be numeric, got "
                    f"{type(score).__name__}"
                )
            numeric.append(float(score))

        mean = sum(numeric) / len(numeric)
        exceeds = mean >= SERL_FITNESS_GATE
        max_score = max(numeric)
        min_score = min(numeric)
        timestamp = time.time()

        record = {
            "skill_id": skill_id,
            "mean_performance": mean,
            "sample_count": len(numeric),
            "exceeds_gate": exceeds,
            "max_score": max_score,
            "min_score": min_score,
            "timestamp": timestamp,
        }
        self._feedback_records.append(record)

        return {
            "skill_id": skill_id,
            "mean_performance": mean,
            "sample_count": len(numeric),
            "exceeds_gate": exceeds,
            "metadata": {
                "max_score": max_score,
                "min_score": min_score,
                "gate_threshold": SERL_FITNESS_GATE,
                "feedback_index": len(self._feedback_records),
                "cce_dim": CCE_ARENA_DIM,
                "thdse_dim": THDSE_ARENA_DIM,
                "provenance": {
                    "operation": "rsi_skill_to_serl_feedback",
                    "source_arena": "cce",
                    "target_arena": "thdse",
                    "timestamp": timestamp,
                    "skill_hash": hashlib.blake2b(
                        skill_id.encode("utf-8"), digest_size=6
                    ).hexdigest(),
                },
            },
        }

    # ---- introspection ---- #

    @property
    def candidates_seen(self) -> int:
        return self._candidates_seen

    @property
    def eligible_count(self) -> int:
        return self._eligible_count

    @property
    def feedback_count(self) -> int:
        return len(self._feedback_records)

    def get_feedback_history(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._feedback_records]

    # ---- internals ---- #

    def _build_candidate_metadata(
        self,
        thdse_handle: int,
        fitness: float,
        timestamp: float,
        eligible: bool,
        similarity_provenance: dict | None = None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "thdse_handle": int(thdse_handle),
            "fitness": fitness,
            "gate_threshold": SERL_FITNESS_GATE,
            "eligible": eligible,
            "candidate_index": self._candidates_seen,
            "provenance": {
                "operation": "serl_candidate_to_rsi",
                "source_arena": "thdse",
                "target_arena": "cce",
                "timestamp": timestamp,
            },
        }
        if similarity_provenance is not None:
            meta["similarity_provenance"] = dict(similarity_provenance)
        return meta


__all__ = ["RsiSerlBridge"]
