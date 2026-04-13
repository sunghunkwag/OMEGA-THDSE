"""Gap 2 — CCE Concept → THDSE Axiom bridge (PLAN.md Section B).

A *concept* in CCE is a 10,000-dim FHRR vector carrying semantic
content encoded by the agent's perception / memory subsystems. An
*axiom* in THDSE is a 256-dim FHRR vector that feeds the axiomatic
synthesizer and SERL evolutionary loops. This module converts a
concept handle into an axiom handle by projecting the phase vector
through :mod:`shared.dimension_bridge`.

Every conversion records provenance so downstream observability can
trace a THDSE axiom back to the originating CCE concept and the
concept's metadata (title, source, timestamp, etc.).

Usage::

    mgr = ArenaManager(master_seed=42)
    bridge = ConceptAxiomBridge(mgr)
    h_cce = mgr.alloc_cce(phases=my_concept_phases)
    result = bridge.concept_to_axiom(h_cce, {"title": "red apple"})
    axiom_handle = result["thdse_handle"]
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from shared.arena_manager import ArenaManager
from shared.constants import CCE_ARENA_DIM, THDSE_ARENA_DIM
from shared.dimension_bridge import cross_arena_similarity, project_down
from shared.exceptions import DimensionMismatchError

# Phase 9 — Rule 20 wiring: the semantic-concept bridge is the new
# upstream feeder for concept→axiom. Importing it here ensures every
# new Phase 9 module has at least one existing-bridge consumer.
from bridges.semantic_concept_bridge import SemanticConceptBridge


class ConceptAxiomBridge:
    """Project CCE concept vectors into THDSE axiom space."""

    def __init__(
        self,
        arena_manager: ArenaManager,
        semantic_bridge: SemanticConceptBridge | None = None,
    ):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._conversion_count = 0
        # Phase 9: optional semantic-grounding front-end. Built lazily
        # on first use so existing callers that allocate their own
        # concept phases continue to work without downloading any
        # encoder model.
        self._semantic_bridge = semantic_bridge

    # ---- Phase 9 convenience: ground text then project to axiom ---- #

    def ground_and_project(
        self, text: str, concept_metadata: dict | None = None
    ) -> dict[str, Any]:
        """Ground raw text into CCE then project into a THDSE axiom.

        Uses :class:`SemanticConceptBridge` under the hood so callers
        can go from perception to axiom in a single call. The returned
        dict carries both ``cce_handle`` and ``thdse_handle`` plus the
        merged provenance chain.
        """
        if self._semantic_bridge is None:
            self._semantic_bridge = SemanticConceptBridge(self._mgr)
        grounded = self._semantic_bridge.ground_text(
            text, context=concept_metadata
        )
        cce_handle = int(grounded["cce_handle"])
        projection = self.concept_to_axiom(
            cce_handle,
            {
                **(concept_metadata or {}),
                "semantic_preview": grounded["metadata"]["ingest_metadata"][
                    "content_preview"
                ],
            },
        )
        projection["cce_handle"] = cce_handle
        projection["metadata"]["grounding_provenance"] = grounded["metadata"][
            "provenance"
        ]
        return projection

    # ---- primary conversion ---- #

    def concept_to_axiom(
        self, concept_handle: int, concept_metadata: dict | None = None
    ) -> dict[str, Any]:
        """Project a CCE concept handle into a new THDSE axiom handle.

        Returns a dict with keys:

        - ``thdse_handle``: int — handle into ``arena_manager.thdse_arena``.
        - ``similarity_to_source``: float — self-similarity of the
          projection (exactly ``1.0`` modulo floating-point noise, since
          ``cross_arena_similarity(v, project_down(v))`` is identity).
        - ``metadata``: dict carrying ``provenance``, the concept's
          original metadata, and a deterministic axiom ID.
        """
        if not isinstance(concept_handle, int) or concept_handle < 0:
            raise TypeError(
                f"concept_handle must be non-negative int, got "
                f"{concept_handle!r}"
            )

        cce_phases = self._mgr.get_cce_phases(concept_handle)
        if cce_phases.shape != (CCE_ARENA_DIM,):
            raise DimensionMismatchError(
                "retrieved CCE phases have wrong shape",
                expected=(CCE_ARENA_DIM,),
                actual=tuple(cce_phases.shape),
                operation="concept_to_axiom",
            )

        projection = project_down(cce_phases)
        projected_vector = projection["vector"]

        thdse_handle = self._mgr.alloc_thdse(phases=projected_vector)
        sim_result = cross_arena_similarity(cce_phases, projected_vector)
        similarity = float(sim_result["similarity"])

        axiom_id = self._compute_axiom_id(
            concept_handle, thdse_handle, projected_vector
        )
        self._conversion_count += 1

        return {
            "thdse_handle": thdse_handle,
            "similarity_to_source": similarity,
            "metadata": {
                "axiom_id": axiom_id,
                "source_concept_handle": concept_handle,
                "source_concept_metadata": dict(concept_metadata or {}),
                "projection_provenance": projection["metadata"]["provenance"],
                "provenance": {
                    "operation": "concept_to_axiom",
                    "source_arena": "cce",
                    "target_arena": "thdse",
                    "source_dim": CCE_ARENA_DIM,
                    "target_dim": THDSE_ARENA_DIM,
                    "timestamp": time.time(),
                    "conversion_index": self._conversion_count,
                },
            },
        }

    def axiom_to_concept_similarity(
        self, axiom_handle: int, concept_handle: int
    ) -> dict[str, Any]:
        """Compare a THDSE axiom to a CCE concept in 256-dim space."""
        thdse_phases = self._mgr.get_thdse_phases(axiom_handle)
        cce_phases = self._mgr.get_cce_phases(concept_handle)
        sim = cross_arena_similarity(cce_phases, thdse_phases)
        return {
            "similarity": float(sim["similarity"]),
            "metadata": {
                "axiom_handle": int(axiom_handle),
                "concept_handle": int(concept_handle),
                "underlying_similarity_provenance": sim["metadata"][
                    "provenance"
                ],
                "provenance": {
                    "operation": "axiom_to_concept_similarity",
                    "source_arenas": ("cce", "thdse"),
                    "compared_in_dim": THDSE_ARENA_DIM,
                    "timestamp": time.time(),
                },
            },
        }

    def batch_project(
        self, concept_handles: list[int]
    ) -> list[dict[str, Any]]:
        """Apply :meth:`concept_to_axiom` to every handle in the list."""
        if not isinstance(concept_handles, (list, tuple)):
            raise TypeError(
                f"concept_handles must be a list, got "
                f"{type(concept_handles).__name__}"
            )
        results: list[dict[str, Any]] = []
        batch_start = time.time()
        for idx, handle in enumerate(concept_handles):
            item = self.concept_to_axiom(handle, {"batch_index": idx})
            item["metadata"]["batch_index"] = idx
            item["metadata"]["batch_timestamp"] = batch_start
            results.append(item)
        return results

    # ---- metadata / introspection ---- #

    @property
    def conversion_count(self) -> int:
        return self._conversion_count

    # ---- internals ---- #

    @staticmethod
    def _compute_axiom_id(
        concept_handle: int, thdse_handle: int, vector
    ) -> str:
        """Deterministic axiom ID derived from the concept+thdse handles."""
        payload = (
            f"cce:{int(concept_handle)}|"
            f"thdse:{int(thdse_handle)}|"
            f"sum:{float(vector.sum()):.6f}|"
            f"head:{float(vector[0]):.6f}"
        ).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
        return f"axiom-{digest}"


__all__ = ["ConceptAxiomBridge"]
