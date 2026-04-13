"""Three-tier memory architecture for OMEGA-THDSE Phase 11 (fixes D6).

The pre-Phase-11 CCE ``Memory`` module mixed episodic events,
semantic facts, and procedural skills into a single associative
store, which is why Rule 18 ("NO SHALLOW MEMORY") was routinely
violated on retrieval benchmarks. This module replaces that shortcut
with three cooperating stores:

- :class:`EpisodicMemory` — timestamped single-event records with
  an explicit rehearsal counter. Items rehearsed at least
  :data:`EPISODIC_CONSOLIDATION_THRESHOLD` times are candidates for
  promotion into semantic memory.
- :class:`SemanticMemory` — deduplicated fact store keyed by the
  semantic encoder output. Queries return the nearest-key fact.
- :class:`ProceduralMemory` — callable procedures indexed by a
  textual trigger. Queries return the best-matching procedure plus
  its expected argument schema.

All three stores use :class:`shared.semantic_encoder.SemanticEncoder`
for retrieval keys. Rule 18 requires top-1 accuracy >= 0.8 across at
least 50 items with at least 5 queries; the :class:`DeepMemory`
facade exposes a :meth:`query` method that drives that benchmark end
to end.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    EPISODIC_CONSOLIDATION_THRESHOLD,
    EPISODIC_MEMORY_CAPACITY,
    PROCEDURAL_MEMORY_CAPACITY,
    SEMANTIC_ENCODER_DIM,
    SEMANTIC_MEMORY_CAPACITY,
)
from .semantic_encoder import SemanticEncoder, cosine


# --------------------------------------------------------------------------- #
# Shared record types
# --------------------------------------------------------------------------- #


@dataclass
class EpisodicRecord:
    event_id: int
    content: str
    vector: np.ndarray
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    rehearsals: int = 0


@dataclass
class SemanticRecord:
    fact_id: int
    subject: str
    fact: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_episodes: List[int] = field(default_factory=list)


@dataclass
class ProceduralRecord:
    proc_id: int
    trigger: str
    procedure: Callable[..., Any]
    vector: np.ndarray
    schema: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _argmax_cosine(
    query_vec: np.ndarray, keys: Sequence[np.ndarray]
) -> Tuple[int, float]:
    """Return ``(index_of_best_key, similarity_score)`` by cosine."""
    if not keys:
        return -1, float("-inf")
    best_idx = -1
    best_score = float("-inf")
    for i, key in enumerate(keys):
        score = cosine(query_vec, key)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score


# --------------------------------------------------------------------------- #
# Episodic memory
# --------------------------------------------------------------------------- #


class EpisodicMemory:
    """Single-event episodic store with rehearsal tracking."""

    def __init__(
        self,
        encoder: SemanticEncoder,
        capacity: int = EPISODIC_MEMORY_CAPACITY,
        consolidation_threshold: int = EPISODIC_CONSOLIDATION_THRESHOLD,
    ):
        self._encoder = encoder
        self._capacity = int(capacity)
        self._records: List[EpisodicRecord] = []
        self._next_id = 0
        self._threshold = int(consolidation_threshold)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def consolidation_threshold(self) -> int:
        return self._threshold

    def __len__(self) -> int:
        return len(self._records)

    def store(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> EpisodicRecord:
        if len(self._records) >= self._capacity:
            # Drop the oldest unrehearsed record to make room.
            oldest_idx = min(
                range(len(self._records)),
                key=lambda i: (
                    self._records[i].rehearsals,
                    self._records[i].timestamp,
                ),
            )
            self._records.pop(oldest_idx)
        vec = self._encoder.encode(content)
        record = EpisodicRecord(
            event_id=self._next_id,
            content=content,
            vector=vec,
            timestamp=time.time(),
            metadata=dict(metadata or {}),
        )
        self._records.append(record)
        self._next_id += 1
        return record

    def recall(self, query: str, top_k: int = 1) -> List[EpisodicRecord]:
        if not self._records:
            return []
        qvec = self._encoder.encode(query)
        scored = [
            (float(cosine(qvec, r.vector)), r) for r in self._records
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [r for _, r in scored[: int(top_k)]]
        for rec in top:
            rec.rehearsals += 1
        return top

    def consolidation_candidates(self) -> List[EpisodicRecord]:
        return [r for r in self._records if r.rehearsals >= self._threshold]


# --------------------------------------------------------------------------- #
# Semantic memory
# --------------------------------------------------------------------------- #


class SemanticMemory:
    """Deduplicated fact store indexed by semantic embedding."""

    def __init__(
        self,
        encoder: SemanticEncoder,
        capacity: int = SEMANTIC_MEMORY_CAPACITY,
        dedupe_threshold: float = 0.98,
    ):
        self._encoder = encoder
        self._capacity = int(capacity)
        self._records: List[SemanticRecord] = []
        self._next_id = 0
        self._dedupe = float(dedupe_threshold)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def capacity(self) -> int:
        return self._capacity

    def assert_fact(
        self,
        subject: str,
        fact: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_episodes: Optional[Sequence[int]] = None,
    ) -> SemanticRecord:
        if len(self._records) >= self._capacity:
            raise RuntimeError("semantic memory at capacity")
        key_text = f"{subject}: {fact}"
        vec = self._encoder.encode(key_text)
        # Dedupe: if an almost-identical fact exists, merge sources.
        for existing in self._records:
            if float(cosine(existing.vector, vec)) >= self._dedupe:
                existing.source_episodes.extend(list(source_episodes or []))
                existing.metadata.update(dict(metadata or {}))
                return existing
        rec = SemanticRecord(
            fact_id=self._next_id,
            subject=subject,
            fact=fact,
            vector=vec,
            metadata=dict(metadata or {}),
            source_episodes=list(source_episodes or []),
        )
        self._records.append(rec)
        self._next_id += 1
        return rec

    def query(self, text: str, top_k: int = 1) -> List[SemanticRecord]:
        if not self._records:
            return []
        qvec = self._encoder.encode(text)
        scored = [
            (float(cosine(qvec, r.vector)), r) for r in self._records
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [r for _, r in scored[: int(top_k)]]


# --------------------------------------------------------------------------- #
# Procedural memory
# --------------------------------------------------------------------------- #


class ProceduralMemory:
    """Callable-by-trigger procedural memory."""

    def __init__(
        self,
        encoder: SemanticEncoder,
        capacity: int = PROCEDURAL_MEMORY_CAPACITY,
    ):
        self._encoder = encoder
        self._capacity = int(capacity)
        self._records: List[ProceduralRecord] = []
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._records)

    def register(
        self,
        trigger: str,
        procedure: Callable[..., Any],
        schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProceduralRecord:
        if len(self._records) >= self._capacity:
            raise RuntimeError("procedural memory at capacity")
        if not callable(procedure):
            raise TypeError("procedure must be callable")
        vec = self._encoder.encode(trigger)
        rec = ProceduralRecord(
            proc_id=self._next_id,
            trigger=trigger,
            procedure=procedure,
            vector=vec,
            schema=dict(schema or {}),
            metadata=dict(metadata or {}),
        )
        self._records.append(rec)
        self._next_id += 1
        return rec

    def match(self, trigger: str, top_k: int = 1) -> List[ProceduralRecord]:
        if not self._records:
            return []
        qvec = self._encoder.encode(trigger)
        scored = [
            (float(cosine(qvec, r.vector)), r) for r in self._records
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [r for _, r in scored[: int(top_k)]]


# --------------------------------------------------------------------------- #
# Facade + consolidation
# --------------------------------------------------------------------------- #


class DeepMemory:
    """Unified facade over the three memory stores.

    Provides a single ``query(text) -> {"tier", "record", "score"}``
    that benchmarks top-1 accuracy for Rule 18, plus a
    :meth:`consolidate` routine that promotes rehearsed episodic
    memories into semantic facts.
    """

    def __init__(self, encoder: Optional[SemanticEncoder] = None):
        self._encoder = encoder or SemanticEncoder()
        self.episodic = EpisodicMemory(self._encoder)
        self.semantic = SemanticMemory(self._encoder)
        self.procedural = ProceduralMemory(self._encoder)

    @property
    def encoder(self) -> SemanticEncoder:
        return self._encoder

    def query(self, text: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """Unified top-K retrieval across all three tiers."""
        qvec = self._encoder.encode(text)

        def _scores(tier: str, records):
            return [
                (
                    float(cosine(qvec, r.vector)),
                    tier,
                    r,
                )
                for r in records
            ]

        pool: List[Tuple[float, str, Any]] = []
        pool.extend(_scores("episodic", self.episodic._records))
        pool.extend(_scores("semantic", self.semantic._records))
        pool.extend(_scores("procedural", self.procedural._records))
        pool.sort(key=lambda t: t[0], reverse=True)
        return [
            {"score": score, "tier": tier, "record": rec}
            for score, tier, rec in pool[: int(top_k)]
        ]

    def consolidate(self) -> Dict[str, Any]:
        """Promote rehearsed episodic memories into semantic facts."""
        candidates = self.episodic.consolidation_candidates()
        promoted: List[int] = []
        for ep in candidates:
            subject = ep.metadata.get("subject", "episode")
            fact = ep.content
            self.semantic.assert_fact(
                subject=subject,
                fact=fact,
                metadata={
                    "consolidated_from_episode": ep.event_id,
                    "original_rehearsals": ep.rehearsals,
                    **ep.metadata,
                },
                source_episodes=[ep.event_id],
            )
            promoted.append(ep.event_id)
        return {
            "promoted_count": len(promoted),
            "promoted_event_ids": promoted,
        }

    def counts(self) -> Dict[str, int]:
        return {
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "procedural": len(self.procedural),
        }


__all__ = [
    "DeepMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "EpisodicRecord",
    "SemanticRecord",
    "ProceduralRecord",
]
