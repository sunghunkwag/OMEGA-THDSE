"""Gap 7 — World Model ↔ Swarm bridge (PLAN.md Phase 4).

CCE's :class:`WorldModel` predicts state transitions and Q-values
inside a 10,000-dim semantic space. THDSE's swarm explores parameter
space in 256-dim FHRR. Without a translator the swarm has no way to
exploit what the world model already believes about a domain — it
explores blind. This bridge gives the swarm advisory guidance vectors
projected from the current observation + Q-values, and conversely
lets a swarm consensus vector flow back into the world model as a
similarity-weighted update signal.

Routing rules:

- 10k → 256 always uses :func:`shared.dimension_bridge.project_down`.
- 256 ↔ 10k similarity always uses
  :func:`shared.dimension_bridge.cross_arena_similarity`.

The ``SWARM_CONSENSUS_THRESHOLD`` constant from PLAN.md Section D is
used as the gating threshold for whether the world model should adopt
a swarm consensus.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Dict, List

from shared.arena_manager import ArenaManager
from shared.constants import (
    CCE_ARENA_DIM,
    SWARM_CONSENSUS_THRESHOLD,
    SWARM_NUM_AGENTS,
    THDSE_ARENA_DIM,
)
from shared.dimension_bridge import cross_arena_similarity, project_down
from shared.exceptions import DimensionMismatchError

_TWO_PI = 2.0 * math.pi


class WorldModelSwarmBridge:
    """Bidirectional bridge between CCE world model and THDSE swarm."""

    def __init__(self, arena_manager: ArenaManager):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._projection_count = 0
        self._consensus_count = 0
        self._adopted_count = 0

    # ---- deterministic obs/Q-value encoding ---- #

    def _encode_state_phases(
        self, obs: Dict[str, Any], q_values: Dict[str, float]
    ) -> List[float]:
        """Encode obs+Q-values as a 10k-dim phase vector deterministically.

        The encoding is content-addressed via BLAKE2b: identical inputs
        always produce identical phases regardless of dict ordering, so
        the bridge is reproducible across runs without storing per-key
        random seeds.
        """
        # Build a canonical signature from sorted (key, value) pairs.
        obs_items = sorted(
            (str(k), str(obs[k])) for k in obs.keys()
        )
        q_items = sorted(
            (str(k), float(q_values[k])) for k in q_values.keys()
        )
        canonical = "|".join(f"{k}={v}" for k, v in obs_items) + "::" + \
            "|".join(f"{k}={v:.6f}" for k, v in q_items)
        # Deterministic phases: derive a fixed seed from the canonical
        # input and fan it out through BLAKE2b bytes. We avoid
        # ``self._mgr.rng.fork`` here because forks cache their
        # generator state across calls, which would make repeated
        # invocations with the same obs produce DIFFERENT phases as
        # the generator advances.
        phases: List[float] = []
        counter = 0
        while len(phases) < CCE_ARENA_DIM:
            chunk = hashlib.blake2b(
                f"{canonical}|{counter}".encode("utf-8"), digest_size=64
            ).digest()
            for i in range(0, len(chunk), 4):
                if len(phases) >= CCE_ARENA_DIM:
                    break
                word = int.from_bytes(chunk[i:i + 4], "big")
                # Map uint32 → [0, 2π).
                phases.append((word / 4294967296.0) * _TWO_PI)
            counter += 1
        return phases

    # ---- public API ---- #

    def project_world_state_for_swarm(
        self, obs: Dict[str, Any], q_values: Dict[str, float]
    ) -> Dict[str, Any]:
        """Encode CCE state into a 256-dim guidance vector for the swarm.

        The confidence component summarizes how peaked the Q-value
        distribution is — a near-uniform distribution is treated as
        low confidence (the swarm should explore broadly), a
        high-variance distribution is high confidence (the swarm
        should exploit the projected guidance more aggressively).
        """
        if not isinstance(obs, dict):
            raise TypeError(f"obs must be a dict, got {type(obs).__name__}")
        if not isinstance(q_values, dict):
            raise TypeError(
                f"q_values must be a dict, got {type(q_values).__name__}"
            )

        phases = self._encode_state_phases(obs, q_values)
        projection = project_down(phases)

        # Confidence: stddev of Q-values, capped to [0, 1].
        if q_values:
            mean_q = sum(q_values.values()) / len(q_values)
            var_q = sum(
                (v - mean_q) ** 2 for v in q_values.values()
            ) / len(q_values)
            confidence = min(1.0, math.sqrt(var_q))
        else:
            confidence = 0.0

        self._projection_count += 1
        timestamp = time.time()
        return {
            "thdse_guidance_vector": projection["vector"],
            "confidence": confidence,
            "metadata": {
                "obs_keys": sorted(obs.keys()),
                "q_value_count": len(q_values),
                "swarm_num_agents": SWARM_NUM_AGENTS,
                "underlying_projection": projection["metadata"]["provenance"],
                "projection_index": self._projection_count,
                "provenance": {
                    "operation": "world_to_swarm",
                    "source_arena": "cce",
                    "target_arena": "thdse",
                    "timestamp": timestamp,
                    "source_dim": CCE_ARENA_DIM,
                    "target_dim": THDSE_ARENA_DIM,
                },
            },
        }

    def incorporate_swarm_consensus(
        self, consensus_vector_256, threshold: float = None
    ) -> Dict[str, Any]:
        """Decide whether the world model should adopt a swarm consensus.

        Computes the cross-arena similarity between a freshly encoded
        "ambient world state" (zero-CCE reference) and the incoming
        swarm consensus vector. If the similarity exceeds ``threshold``
        (defaulting to ``SWARM_CONSENSUS_THRESHOLD``), the bridge
        signals adoption.
        """
        try:
            consensus_list = list(consensus_vector_256)
        except TypeError as exc:
            raise DimensionMismatchError(
                f"consensus_vector_256 not iterable: {exc}",
                operation="incorporate_swarm_consensus",
            ) from exc
        if len(consensus_list) != THDSE_ARENA_DIM:
            raise DimensionMismatchError(
                "consensus_vector_256 must be 256-dim",
                expected=(THDSE_ARENA_DIM,),
                actual=(len(consensus_list),),
                operation="incorporate_swarm_consensus",
            )

        if threshold is None:
            threshold = SWARM_CONSENSUS_THRESHOLD
        if not isinstance(threshold, (int, float)):
            raise TypeError(
                f"threshold must be numeric, got {type(threshold).__name__}"
            )

        zero_cce = [0.0] * CCE_ARENA_DIM
        sim_result = cross_arena_similarity(zero_cce, consensus_list)
        similarity = float(sim_result["similarity"])
        should_adopt = similarity >= float(threshold)

        self._consensus_count += 1
        if should_adopt:
            self._adopted_count += 1
        timestamp = time.time()
        return {
            "similarity": similarity,
            "should_adopt": should_adopt,
            "threshold": float(threshold),
            "metadata": {
                "consensus_index": self._consensus_count,
                "adopted_index": self._adopted_count,
                "underlying_similarity_provenance": sim_result["metadata"][
                    "provenance"
                ],
                "provenance": {
                    "operation": "swarm_to_world",
                    "source_arena": "thdse",
                    "target_arena": "cce",
                    "timestamp": timestamp,
                },
            },
        }

    def compare_two_swarm_consensuses(
        self, consensus_a, consensus_b
    ) -> Dict[str, Any]:
        """Pairwise FHRR similarity between two 256-dim consensus vectors."""
        a = list(consensus_a)
        b = list(consensus_b)
        if len(a) != THDSE_ARENA_DIM or len(b) != THDSE_ARENA_DIM:
            raise DimensionMismatchError(
                "both consensus vectors must be 256-dim",
                expected=(THDSE_ARENA_DIM,),
                actual=(len(a), len(b)),
                operation="compare_two_swarm_consensuses",
            )
        mean_cos = sum(
            math.cos(float(a[i]) - float(b[i]))
            for i in range(THDSE_ARENA_DIM)
        ) / THDSE_ARENA_DIM
        return {
            "similarity": float(mean_cos),
            "metadata": {
                "compared_in_dim": THDSE_ARENA_DIM,
                "provenance": {
                    "operation": "compare_two_swarm_consensuses",
                    "source_arena": "thdse",
                    "timestamp": time.time(),
                },
            },
        }

    def summarize_swarm_state(self) -> Dict[str, Any]:
        """Return aggregate counters tracking the bridge's lifetime activity."""
        adoption_rate = (
            self._adopted_count / self._consensus_count
            if self._consensus_count
            else 0.0
        )
        return {
            "projection_count": self._projection_count,
            "consensus_count": self._consensus_count,
            "adopted_count": self._adopted_count,
            "adoption_rate": adoption_rate,
            "metadata": {
                "swarm_num_agents": SWARM_NUM_AGENTS,
                "consensus_threshold": SWARM_CONSENSUS_THRESHOLD,
                "provenance": {
                    "operation": "summarize_swarm_state",
                    "source_arena": "both",
                    "timestamp": time.time(),
                },
            },
        }

    def project_action_distribution(
        self, action_q_values: Dict[str, float]
    ) -> Dict[str, Any]:
        """Encode an action-conditioned Q-value distribution as a 256-dim vector."""
        if not isinstance(action_q_values, dict):
            raise TypeError(
                f"action_q_values must be a dict, got "
                f"{type(action_q_values).__name__}"
            )
        if not action_q_values:
            raise ValueError("action_q_values must contain at least one entry")
        # Normalize Q-values into a softmax-like distribution for the
        # encoded phase signal.
        items = sorted(action_q_values.items())
        max_q = max(v for _, v in items)
        exp_vals = [math.exp(float(v) - max_q) for _, v in items]
        total = sum(exp_vals) or 1.0
        normalized = [e / total for e in exp_vals]

        # Use the normalized distribution as a deterministic seed.
        canonical = "|".join(
            f"{k}={p:.6f}" for (k, _), p in zip(items, normalized)
        )
        phases: List[float] = []
        counter = 0
        while len(phases) < CCE_ARENA_DIM:
            chunk = hashlib.blake2b(
                f"{canonical}|{counter}".encode("utf-8"), digest_size=64
            ).digest()
            for i in range(0, len(chunk), 4):
                if len(phases) >= CCE_ARENA_DIM:
                    break
                word = int.from_bytes(chunk[i:i + 4], "big")
                phases.append((word / 4294967296.0) * _TWO_PI)
            counter += 1
        projection = project_down(phases)
        timestamp = time.time()
        return {
            "thdse_action_vector": projection["vector"],
            "action_distribution": dict(zip([k for k, _ in items], normalized)),
            "metadata": {
                "action_count": len(items),
                "underlying_projection": projection["metadata"]["provenance"],
                "provenance": {
                    "operation": "project_action_distribution",
                    "source_arena": "cce",
                    "target_arena": "thdse",
                    "timestamp": timestamp,
                },
            },
        }

    # ---- introspection ---- #

    @property
    def projection_count(self) -> int:
        return self._projection_count

    @property
    def consensus_count(self) -> int:
        return self._consensus_count

    @property
    def adopted_count(self) -> int:
        return self._adopted_count


__all__ = ["WorldModelSwarmBridge"]
