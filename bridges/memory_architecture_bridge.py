"""Phase 11 bridge — DeepMemory ↔ Arena (fixes D6).

Wires the three-tier :class:`DeepMemory` architecture into the
unified :class:`ArenaManager`. Every stored event / fact / procedure
is mirrored in the CCE arena as a phase vector handle so downstream
bridges (concept→axiom, memory→hypothesis, reasoning) can address
memory items via the same handle vocabulary as every other concept.

Rule 18 entry point: :meth:`MemoryArchitectureBridge.query_top1` is
the method that Phase 11 tests call to verify the 80% top-1 accuracy
threshold. It returns the best match across all three tiers plus
provenance.

Rule 20 wiring: imports :class:`SemanticConceptBridge` (Phase 9) and
:class:`MemoryHypothesisBridge` (Phase 4) so Phase 11 memory updates
flow into Phase 4's hypothesis generation without the caller having
to remember to call both.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from shared.arena_manager import ArenaManager
from shared.constants import (
    CCE_ARENA_DIM,
    MEMORY_TOP1_ACCURACY_MIN,
    SEMANTIC_ENCODER_DIM,
)
from shared.deep_memory import DeepMemory
from shared.perceptual_grounding import PerceptualGrounder
from shared.semantic_encoder import SemanticEncoder
from bridges.memory_hypothesis_bridge import MemoryHypothesisBridge
from bridges.semantic_concept_bridge import SemanticConceptBridge


class MemoryArchitectureBridge:
    """Arena-integrated Episodic / Semantic / Procedural memory.

    Maintains a mapping from every stored memory record to a CCE handle
    so memory retrievals can be composed with Phase 9 groundings and
    Phase 12 reasoning without leaving the arena.
    """

    def __init__(
        self,
        arena_manager: ArenaManager,
        *,
        encoder: Optional[SemanticEncoder] = None,
        semantic_bridge: Optional[SemanticConceptBridge] = None,
    ):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._encoder = encoder or SemanticEncoder(rng=arena_manager.rng)
        self._semantic_bridge = semantic_bridge or SemanticConceptBridge(
            arena_manager, encoder=self._encoder
        )
        self._grounder: PerceptualGrounder = self._semantic_bridge.grounder
        self._memory = DeepMemory(encoder=self._encoder)
        # Record-id → CCE handle maps per tier, so tests + reasoning
        # can round-trip from memory records back into arena handles.
        self._episodic_handles: Dict[int, int] = {}
        self._semantic_handles: Dict[int, int] = {}
        self._procedural_handles: Dict[int, int] = {}
        self._hypothesis_bridge = MemoryHypothesisBridge(arena_manager)

    # ---- introspection ---- #

    @property
    def memory(self) -> DeepMemory:
        return self._memory

    @property
    def encoder(self) -> SemanticEncoder:
        return self._encoder

    def counts(self) -> Dict[str, int]:
        return self._memory.counts()

    # ---- store / register ---- #

    def remember_event(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        record = self._memory.episodic.store(content, metadata=metadata)
        cce_handle = self._allocate_cce_for_vector(record.vector)
        self._episodic_handles[record.event_id] = cce_handle
        return {
            "tier": "episodic",
            "event_id": record.event_id,
            "cce_handle": cce_handle,
            "metadata": self._provenance("remember_event", tier="episodic"),
        }

    def assert_fact(
        self,
        subject: str,
        fact: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = self._memory.semantic.assert_fact(
            subject=subject, fact=fact, metadata=metadata
        )
        if record.fact_id not in self._semantic_handles:
            handle = self._allocate_cce_for_vector(record.vector)
            self._semantic_handles[record.fact_id] = handle
        return {
            "tier": "semantic",
            "fact_id": record.fact_id,
            "cce_handle": self._semantic_handles[record.fact_id],
            "metadata": self._provenance("assert_fact", tier="semantic"),
        }

    def register_procedure(
        self,
        trigger: str,
        procedure: Callable[..., Any],
        schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = self._memory.procedural.register(
            trigger=trigger,
            procedure=procedure,
            schema=schema,
            metadata=metadata,
        )
        handle = self._allocate_cce_for_vector(record.vector)
        self._procedural_handles[record.proc_id] = handle
        return {
            "tier": "procedural",
            "proc_id": record.proc_id,
            "cce_handle": handle,
            "metadata": self._provenance(
                "register_procedure", tier="procedural"
            ),
        }

    # ---- query ---- #

    def query_top1(self, text: str) -> Dict[str, Any]:
        """Unified top-1 retrieval across all tiers (Rule 18 entry point)."""
        hits = self._memory.query(text, top_k=1)
        if not hits:
            return {
                "found": False,
                "tier": None,
                "content": None,
                "score": None,
                "metadata": self._provenance("query_top1", tier="empty"),
            }
        hit = hits[0]
        record = hit["record"]
        content = self._extract_content(hit["tier"], record)
        return {
            "found": True,
            "tier": hit["tier"],
            "score": float(hit["score"]),
            "content": content,
            "record_id": self._record_id(hit["tier"], record),
            "cce_handle": self._handle_for(hit["tier"], record),
            "metadata": self._provenance("query_top1", tier=hit["tier"]),
        }

    def benchmark_top1(
        self,
        queries: Sequence[tuple[str, str]],
    ) -> Dict[str, Any]:
        """Rule 18 benchmark: expected_content must be top-1 for >=80%."""
        correct = 0
        misses: List[Dict[str, Any]] = []
        for query, expected in queries:
            result = self.query_top1(query)
            got = (result.get("content") or "").strip().lower()
            exp = expected.strip().lower()
            if got == exp:
                correct += 1
            else:
                misses.append(
                    {"query": query, "expected": expected, "got": got}
                )
        accuracy = correct / float(len(queries)) if queries else 0.0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(queries),
            "passes_rule18": accuracy >= MEMORY_TOP1_ACCURACY_MIN,
            "misses": misses,
            "metadata": self._provenance("benchmark_top1", tier="all"),
        }

    # ---- consolidation (episodic → semantic) ---- #

    def consolidate(self) -> Dict[str, Any]:
        result = self._memory.consolidate()
        # Ensure newly consolidated facts have arena handles.
        for rec in self._memory.semantic._records:
            if rec.fact_id not in self._semantic_handles:
                self._semantic_handles[rec.fact_id] = (
                    self._allocate_cce_for_vector(rec.vector)
                )
        result["metadata"] = self._provenance(
            "consolidate", tier="episodic→semantic"
        )
        return result

    # ---- Phase-4 hypothesis wire-up (Rule 20) ---- #

    def memory_summary_for_hypothesis(
        self, subject: str, tags: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """Produce a MemoryHypothesisBridge entry from current memory state.

        Pulls the top-scoring item across all tiers, then hands it to
        the Phase 4 :class:`MemoryHypothesisBridge` so that Phase 11
        memory updates can feed hypothesis generation directly.
        """
        summary_tags = list(tags or [])
        result = self.query_top1(subject)
        if result["found"]:
            summary_tags.append(result["tier"])
        encoded = self._hypothesis_bridge.encode_memory_for_hypothesis(
            subject, summary_tags
        )
        encoded["memory_lookup"] = result
        return encoded

    # ---- internals ---- #

    def _allocate_cce_for_vector(self, vec384: np.ndarray) -> int:
        """Lift a 384-dim encoder vector into a CCE (10k) handle."""
        phases = self._lift(vec384)
        return self._mgr.alloc_cce(phases=phases)

    def _lift(self, vec384: np.ndarray) -> np.ndarray:
        from shared.perceptual_grounding import _semantic_to_phases  # type: ignore

        if vec384.shape != (SEMANTIC_ENCODER_DIM,):
            raise ValueError(
                f"expected semantic vector of dim {SEMANTIC_ENCODER_DIM}"
            )
        return _semantic_to_phases(vec384, self._grounder._lift)  # type: ignore[attr-defined]

    def _extract_content(self, tier: str, record: Any) -> str:
        if tier == "episodic":
            return record.content
        if tier == "semantic":
            return f"{record.subject}: {record.fact}"
        if tier == "procedural":
            return record.trigger
        return ""

    def _record_id(self, tier: str, record: Any) -> int:
        if tier == "episodic":
            return record.event_id
        if tier == "semantic":
            return record.fact_id
        if tier == "procedural":
            return record.proc_id
        return -1

    def _handle_for(self, tier: str, record: Any) -> Optional[int]:
        if tier == "episodic":
            return self._episodic_handles.get(record.event_id)
        if tier == "semantic":
            return self._semantic_handles.get(record.fact_id)
        if tier == "procedural":
            return self._procedural_handles.get(record.proc_id)
        return None

    def _provenance(self, operation: str, *, tier: str) -> Dict[str, Any]:
        return {
            "tier": tier,
            "counts": self.counts(),
            "provenance": {
                "operation": operation,
                "source_arena": "deep_memory",
                "target_arena": "cce",
                "tier": tier,
                "timestamp": time.time(),
            },
        }


__all__ = ["MemoryArchitectureBridge"]
