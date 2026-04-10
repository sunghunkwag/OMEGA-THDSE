"""Gap 5 — Memory ↔ Hypothesis bridge (PLAN.md Phase 4).

CCE's :class:`SharedMemory` retrieves memory items via 10,000-dim FHRR
associative search. THDSE's wall archive scores hypotheses by raw
fitness numbers and lives in 256-dim space. Without a translator the
two stores cannot reinforce each other: a memory cannot bias which
hypothesis to evolve next, and a high-fitness hypothesis cannot find
its closest memory anchor.

This bridge wires the two stores together. Memory titles + tags are
encoded into a deterministic 10k-dim phase vector via the unified
:class:`ArenaManager`, then projected down to 256 dims via
:func:`shared.dimension_bridge.project_down` so the result lives in
THDSE space. Hypothesis relevance is then scored as a weighted blend
of THDSE fitness and memory similarity, returning a structured dict
with full provenance metadata (PLAN.md Rule 9).
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Dict, List

from shared.arena_manager import ArenaManager
from shared.constants import (
    CCE_ARENA_DIM,
    MEMORY_SIMILARITY_THRESHOLD,
    MEMORY_TITLE_WEIGHT,
    THDSE_ARENA_DIM,
)
from shared.dimension_bridge import cross_arena_similarity, project_down
from shared.exceptions import DimensionMismatchError

_TWO_PI = 2.0 * math.pi


class MemoryHypothesisBridge:
    """Bidirectional bridge between CCE memory and THDSE hypothesis space."""

    def __init__(self, arena_manager: ArenaManager):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._encoded_count = 0
        self._scored_count = 0
        # Cache token-phase vectors so repeated encoding of the same
        # token produces identical 10k-dim vectors (deterministic).
        self._token_cache: Dict[str, list[float]] = {}

    # ---- token encoding (deterministic, no random calls) ---- #

    def _token_phases(self, token: str) -> list[float]:
        if token in self._token_cache:
            return self._token_cache[token]
        # Deterministic seed: 4-byte BLAKE2b digest → uint32 seed.
        digest = hashlib.blake2b(
            token.encode("utf-8"), digest_size=4
        ).digest()
        seed = int.from_bytes(digest, "big")
        # Use the manager's deterministic RNG fork keyed off the seed
        # so two bridges constructed with the same master_seed produce
        # identical token vectors.
        rng = self._mgr.rng.fork(f"mhb_token_{seed}")
        phases = [float(rng.uniform(0.0, _TWO_PI)) for _ in range(CCE_ARENA_DIM)]
        self._token_cache[token] = phases
        return phases

    def _bundle_phases(self, vectors: List[list[float]]) -> list[float]:
        """Circular-mean bundle of phase vectors (FHRR bundle semantics)."""
        if not vectors:
            return [0.0] * CCE_ARENA_DIM
        bundled: list[float] = []
        for i in range(CCE_ARENA_DIM):
            sin_sum = 0.0
            cos_sum = 0.0
            for vec in vectors:
                sin_sum += math.sin(vec[i])
                cos_sum += math.cos(vec[i])
            angle = math.atan2(sin_sum, cos_sum)
            if angle < 0.0:
                angle += _TWO_PI
            bundled.append(angle)
        return bundled

    # ---- public API ---- #

    def encode_memory_for_hypothesis(
        self, memory_title: str, memory_tags: List[str]
    ) -> Dict[str, Any]:
        """Encode a memory title + tags into a 256-dim THDSE-space vector.

        The title is weighted ``MEMORY_TITLE_WEIGHT`` times the
        contribution of any single tag, matching the existing
        title-weighted bundling in :class:`SharedMemory`.
        """
        if not isinstance(memory_title, str) or not memory_title.strip():
            raise ValueError("memory_title must be a non-empty string")
        if not isinstance(memory_tags, (list, tuple)):
            raise TypeError(
                f"memory_tags must be a list, got {type(memory_tags).__name__}"
            )

        title_vec = self._token_phases(f"title:{memory_title}")
        contributors: List[list[float]] = [
            title_vec for _ in range(MEMORY_TITLE_WEIGHT)
        ]
        for tag in memory_tags:
            contributors.append(self._token_phases(f"tag:{tag}"))

        bundled_10k = self._bundle_phases(contributors)
        projection = project_down(bundled_10k)
        thdse_vector = projection["vector"]

        self._encoded_count += 1
        timestamp = time.time()
        return {
            "thdse_vector": thdse_vector,
            "original_dim": CCE_ARENA_DIM,
            "projected_dim": THDSE_ARENA_DIM,
            "tag_count": len(memory_tags),
            "encoded_index": self._encoded_count,
            "metadata": {
                "memory_title": memory_title,
                "memory_tags": list(memory_tags),
                "title_weight": MEMORY_TITLE_WEIGHT,
                "underlying_projection": projection["metadata"]["provenance"],
                "provenance": {
                    "operation": "memory_to_hypothesis",
                    "source_arena": "cce",
                    "target_arena": "thdse",
                    "timestamp": timestamp,
                    "encoded_index": self._encoded_count,
                },
            },
        }

    def score_hypothesis_relevance(
        self, hypothesis_fitness: float, memory_similarity: float
    ) -> Dict[str, Any]:
        """Combine THDSE fitness and CCE memory similarity into a relevance score.

        Composite score: ``0.6 * fitness + 0.4 * memory_similarity`` —
        fitness dominates because hypothesis viability matters more
        than memory anchoring, but a memory anchor still meaningfully
        boosts the result. The memory similarity term is gated against
        ``MEMORY_SIMILARITY_THRESHOLD`` so weak memory matches do not
        bias the score upward.
        """
        if not isinstance(hypothesis_fitness, (int, float)):
            raise TypeError(
                f"hypothesis_fitness must be numeric, got "
                f"{type(hypothesis_fitness).__name__}"
            )
        if not isinstance(memory_similarity, (int, float)):
            raise TypeError(
                f"memory_similarity must be numeric, got "
                f"{type(memory_similarity).__name__}"
            )

        fitness = float(hypothesis_fitness)
        sim = float(memory_similarity)
        gated_sim = sim if sim >= MEMORY_SIMILARITY_THRESHOLD else 0.0
        relevance = 0.6 * fitness + 0.4 * gated_sim
        # Clip to [0, 1] for downstream consumers that expect bounded scores.
        if relevance < 0.0:
            relevance = 0.0
        elif relevance > 1.0:
            relevance = 1.0

        self._scored_count += 1
        timestamp = time.time()
        return {
            "relevance_score": relevance,
            "components": {
                "fitness": fitness,
                "memory_similarity": sim,
                "gated_memory_similarity": gated_sim,
            },
            "metadata": {
                "fitness_weight": 0.6,
                "memory_weight": 0.4,
                "memory_threshold": MEMORY_SIMILARITY_THRESHOLD,
                "scored_index": self._scored_count,
                "provenance": {
                    "operation": "hypothesis_scoring",
                    "source_arena": "both",
                    "timestamp": timestamp,
                },
            },
        }

    def compare_memory_to_hypothesis(
        self, encoded_memory: Dict[str, Any], hypothesis_vector_256
    ) -> Dict[str, Any]:
        """Compute similarity between an encoded memory and a hypothesis vector.

        ``encoded_memory`` must be the dict returned by
        :meth:`encode_memory_for_hypothesis`. ``hypothesis_vector_256``
        is any 1-D phase vector of length 256 (numpy array or list).
        """
        if not isinstance(encoded_memory, dict):
            raise TypeError("encoded_memory must be a dict")
        if "thdse_vector" not in encoded_memory:
            raise KeyError("encoded_memory missing 'thdse_vector'")

        memory_vec = encoded_memory["thdse_vector"]
        # Use the bridge primitive: similarity in the shared 256-dim
        # space. We rebuild a 10k vector by repeating the encoded
        # memory's title token vector so we can route through the
        # cross_arena_similarity API and pick up its provenance.
        memory_title = encoded_memory["metadata"]["memory_title"]
        ten_k = self._token_phases(f"title:{memory_title}")
        sim = cross_arena_similarity(ten_k, list(memory_vec))
        # The above measures memory→memory; we also need
        # memory→hypothesis. Compute that directly via mean cosine of
        # phase differences in 256 space (FHRR similarity).
        try:
            import builtins as _b
            it_a = list(memory_vec)
            it_b = list(hypothesis_vector_256)
            if len(it_a) != THDSE_ARENA_DIM or len(it_b) != THDSE_ARENA_DIM:
                raise DimensionMismatchError(
                    "compare_memory_to_hypothesis vectors must be 256-dim",
                    expected=(THDSE_ARENA_DIM,),
                    actual=(len(it_a), len(it_b)),
                    operation="compare_memory_to_hypothesis",
                )
            mean_cos = sum(
                math.cos(float(it_a[i]) - float(it_b[i]))
                for i in range(THDSE_ARENA_DIM)
            ) / THDSE_ARENA_DIM
            _ = _b  # silence unused-import lint
        except (TypeError, ValueError) as exc:
            raise DimensionMismatchError(
                f"hypothesis_vector_256 not iterable: {exc}",
                operation="compare_memory_to_hypothesis",
            ) from exc

        timestamp = time.time()
        return {
            "memory_to_hypothesis_similarity": float(mean_cos),
            "memory_self_similarity": float(sim["similarity"]),
            "metadata": {
                "memory_title": memory_title,
                "underlying_self_similarity_provenance": sim["metadata"][
                    "provenance"
                ],
                "provenance": {
                    "operation": "compare_memory_to_hypothesis",
                    "source_arena": "both",
                    "compared_in_dim": THDSE_ARENA_DIM,
                    "timestamp": timestamp,
                },
            },
        }

    def rank_hypotheses_by_memory(
        self, hypotheses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort hypotheses by descending relevance_score (Rule 9 metadata).

        Each input dict must carry ``hypothesis_fitness`` and
        ``memory_similarity`` (or be the output of
        :meth:`score_hypothesis_relevance`). The returned list contains
        new dicts with a ``rank`` field added.
        """
        if not isinstance(hypotheses, list):
            raise TypeError("hypotheses must be a list")

        scored: List[tuple[float, Dict[str, Any]]] = []
        for idx, h in enumerate(hypotheses):
            if not isinstance(h, dict):
                raise TypeError(f"hypotheses[{idx}] must be a dict")
            if "relevance_score" in h:
                score = float(h["relevance_score"])
                annotated = dict(h)
            else:
                if "hypothesis_fitness" not in h or "memory_similarity" not in h:
                    raise KeyError(
                        f"hypotheses[{idx}] missing required scoring keys"
                    )
                computed = self.score_hypothesis_relevance(
                    h["hypothesis_fitness"], h["memory_similarity"]
                )
                score = computed["relevance_score"]
                annotated = dict(h)
                annotated["relevance_score"] = score
                annotated["scoring_metadata"] = computed["metadata"]
            scored.append((score, annotated))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        timestamp = time.time()
        ranked: List[Dict[str, Any]] = []
        for rank, (score, item) in enumerate(scored, start=1):
            ranked_item = dict(item)
            ranked_item["rank"] = rank
            ranked_item["rank_score"] = score
            existing = dict(ranked_item.get("metadata", {}))
            existing.setdefault(
                "provenance",
                {
                    "operation": "rank_hypotheses_by_memory",
                    "source_arena": "both",
                    "timestamp": timestamp,
                },
            )
            existing["ranking_provenance"] = {
                "operation": "rank_hypotheses_by_memory",
                "source_arena": "both",
                "rank": rank,
                "score": score,
                "timestamp": timestamp,
            }
            ranked_item["metadata"] = existing
            ranked.append(ranked_item)
        return ranked

    def reset_token_cache(self) -> Dict[str, Any]:
        """Drop the cached token vectors and report how many were freed."""
        freed = len(self._token_cache)
        self._token_cache.clear()
        return {
            "freed": freed,
            "metadata": {
                "provenance": {
                    "operation": "reset_token_cache",
                    "source_arena": "cce",
                    "timestamp": time.time(),
                }
            },
        }

    # ---- introspection ---- #

    @property
    def encoded_count(self) -> int:
        return self._encoded_count

    @property
    def scored_count(self) -> int:
        return self._scored_count

    @property
    def cached_token_count(self) -> int:
        return len(self._token_cache)


__all__ = ["MemoryHypothesisBridge"]
