"""Phase 12 bridge — Multi-step reasoning ↔ Arena (fixes D5, D7).

The reasoning bridge hands the Phase 12 :class:`ChainOfThoughtReasoner`
and :class:`AnalogyEngine` the memory + grounding context they need
to run depth->=3 chains against the shared arena. It also exposes the
Rule 17 and Rule 19 contract methods in a single place so the Phase 12
test suite can exercise them without reaching into private state.

Rule 20 wiring: imports :class:`MemoryArchitectureBridge` (Phase 11)
and :class:`SemanticConceptBridge` (Phase 9) so reasoning can retrieve
remembered facts and ground free-text premises. Existing Phase 3
:class:`ConceptAxiomBridge` is re-used to project reasoning-derived
premises into the THDSE arena.

Every return carries ``metadata["provenance"]`` (Rule 9).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from shared.arena_manager import ArenaManager
from shared.constants import (
    ANALOGY_SIMILARITY_MIN,
    REASONING_BEAM_WIDTH,
    REASONING_DEFAULT_DEPTH,
)
from shared.reasoning_engine import (
    AnalogyEngine,
    ChainOfThoughtReasoner,
    ReasoningStep,
    verify_chain_linkage,
)
from shared.semantic_encoder import SemanticEncoder
from bridges.memory_architecture_bridge import MemoryArchitectureBridge
from bridges.semantic_concept_bridge import SemanticConceptBridge


class ReasoningBridge:
    """Arena-aware facade over reasoning + analogy.

    The bridge holds references to the Phase 9 grounding bridge, the
    Phase 11 memory bridge, and a :class:`ChainOfThoughtReasoner`
    instance configured with a pluggable operator dictionary. Callers
    can register their own operators via :meth:`register_operator`.
    """

    def __init__(
        self,
        arena_manager: ArenaManager,
        *,
        encoder: Optional[SemanticEncoder] = None,
        semantic_bridge: Optional[SemanticConceptBridge] = None,
        memory_bridge: Optional[MemoryArchitectureBridge] = None,
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
        self._memory_bridge = memory_bridge or MemoryArchitectureBridge(
            arena_manager,
            encoder=self._encoder,
            semantic_bridge=self._semantic_bridge,
        )
        self._operators: Dict[str, Callable[[Any], Sequence[Tuple[Any, float, Dict[str, Any]]]]] = {}
        self._analogy = AnalogyEngine(
            encoder=self._encoder, similarity_min=ANALOGY_SIMILARITY_MIN
        )

    # ---- configuration ---- #

    @property
    def analogy(self) -> AnalogyEngine:
        return self._analogy

    def register_operator(
        self,
        name: str,
        operator: Callable[[Any], Sequence[Tuple[Any, float, Dict[str, Any]]]],
    ) -> None:
        if not callable(operator):
            raise TypeError("operator must be callable")
        self._operators[name] = operator

    # ---- reasoning ---- #

    def reason(
        self,
        initial_premise: Any,
        goal_fn: Callable[[Any], float],
        *,
        max_depth: int = REASONING_DEFAULT_DEPTH,
        beam_width: int = REASONING_BEAM_WIDTH,
        goal_threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """Run a chain-of-thought search of depth ``>= 3``.

        Returns the full step list plus linkage-verification flag.
        Rule 17 tests call this method and assert both
        ``depth >= 3`` and ``linkage_ok``.
        """
        if not self._operators:
            raise RuntimeError(
                "no operators registered — call register_operator first"
            )
        reasoner = ChainOfThoughtReasoner(
            operators=self._operators,
            goal_fn=goal_fn,
            max_depth=max_depth,
            beam_width=beam_width,
        )
        trace = reasoner.run(initial_premise, goal_threshold=goal_threshold)
        steps: List[ReasoningStep] = trace["steps"]
        linkage_ok = verify_chain_linkage(steps)
        serialised = [
            {
                "index": s.index,
                "premise": s.premise,
                "operator": s.operator,
                "conclusion": s.conclusion,
                "score": s.score,
                "metadata": s.metadata,
            }
            for s in steps
        ]
        return {
            "steps": serialised,
            "depth": trace["depth"],
            "final_premise": trace["final_premise"],
            "final_score": trace["final_score"],
            "reached_goal": trace["reached_goal"],
            "linkage_ok": linkage_ok,
            "metadata": {
                "provenance": {
                    "operation": "reason",
                    "source_arena": "cce",
                    "target_arena": "reasoning",
                    "operators": list(self._operators.keys()),
                    "timestamp": time.time(),
                }
            },
        }

    # ---- analogy (Rule 19) ---- #

    def extract_and_transfer(
        self,
        source_examples: Sequence[str],
        target_examples: Sequence[str],
        pattern_name: str = "analogy_pattern",
    ) -> Dict[str, Any]:
        """Rule 19 contract: pattern from source lifts score on target."""
        result = self._analogy.transfer_score(
            source_examples=source_examples,
            target_examples=target_examples,
            pattern_name=pattern_name,
        )
        return {
            **result,
            "metadata": {
                "provenance": {
                    "operation": "extract_and_transfer",
                    "source_arena": "analogy",
                    "target_arena": "cce",
                    "source_n": len(source_examples),
                    "target_n": len(target_examples),
                    "timestamp": time.time(),
                }
            },
        }

    # ---- memory-driven reasoning ---- #

    def reason_with_memory(
        self,
        question: str,
        *,
        max_depth: int = REASONING_DEFAULT_DEPTH,
    ) -> Dict[str, Any]:
        """Use memory recall as the chain's initial premise.

        Exists so Phase 11 + Phase 12 can be exercised end-to-end in
        a single integration test.
        """
        mem_hit = self._memory_bridge.query_top1(question)
        premise = mem_hit.get("content") or question

        goal_vec = self._encoder.encode(question)

        def _goal_fn(p: Any) -> float:
            if not isinstance(p, str):
                p = str(p)
            from shared.semantic_encoder import cosine as _cos
            return float(_cos(self._encoder.encode(p), goal_vec))

        if not self._operators:
            # Install a default identity operator so downstream tests
            # can call reason_with_memory without configuration.
            self.register_operator(
                "expand",
                lambda p: [(str(p) + " .", 0.1, {"note": "identity_expand"})],
            )
        trace = self.reason(
            premise, goal_fn=_goal_fn, max_depth=max_depth
        )
        trace["memory_seed"] = mem_hit
        return trace


__all__ = ["ReasoningBridge"]
