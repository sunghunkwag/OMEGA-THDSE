"""Phase 9 bridge — Perceptual/Semantic → CCE Concept (fixes D1, D2).

This bridge is the missing link between raw perception and the
unified arena. It owns a :class:`PerceptualGrounder`, encodes
incoming text / structured data / files into 10,000-dim phase vectors,
and allocates a CCE handle for each input via
:class:`shared.arena_manager.ArenaManager`.

Downstream, :mod:`bridges.concept_axiom_bridge` imports this module
to satisfy Phase 9's Rule 20 requirement (every new module must be
imported by an existing bridge). Other bridges can use it directly:
the returned dict exposes ``cce_handle`` so callers can pass the
handle into ``ConceptAxiomBridge.concept_to_axiom`` without ever
touching raw phases.

Every return value carries ``metadata["provenance"]`` (Rule 9).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from shared.arena_manager import ArenaManager
from shared.constants import CCE_ARENA_DIM, SEMANTIC_ENCODER_DIM
from shared.perceptual_grounding import PerceptualGrounder
from shared.semantic_encoder import SemanticEncoder


class SemanticConceptBridge:
    """Ingest raw perception and produce allocated CCE concept handles.

    Parameters
    ----------
    arena_manager:
        Central :class:`ArenaManager`. The bridge uses its deterministic
        RNG fork (``semantic_concept_bridge``) so repeat ingests of the
        same input always produce identical phase vectors.
    encoder:
        Optional pre-built :class:`SemanticEncoder`. When ``None``, a
        new encoder is constructed with the manager's RNG.
    grounder:
        Optional pre-built :class:`PerceptualGrounder`. When ``None``,
        a new one is constructed with the bridge's encoder.
    """

    def __init__(
        self,
        arena_manager: ArenaManager,
        *,
        encoder: SemanticEncoder | None = None,
        grounder: PerceptualGrounder | None = None,
    ):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._encoder = encoder or SemanticEncoder(
            dim=SEMANTIC_ENCODER_DIM, rng=arena_manager.rng
        )
        self._grounder = grounder or PerceptualGrounder(
            encoder=self._encoder, rng=arena_manager.rng
        )
        self._concepts: Dict[int, Dict[str, Any]] = {}

    # ---- properties ---- #

    @property
    def encoder(self) -> SemanticEncoder:
        return self._encoder

    @property
    def grounder(self) -> PerceptualGrounder:
        return self._grounder

    @property
    def concept_count(self) -> int:
        return len(self._concepts)

    # ---- primary API ---- #

    def ground_text(
        self, text: str, *, context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Ground a text input and allocate a CCE handle for it."""
        ingest = self._grounder.ingest_text(text)
        return self._allocate_and_wrap(
            ingest, operation="ground_text", source="text", context=context
        )

    def ground_structured(
        self, data: Any, *, context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Ground a JSON-compatible structured input."""
        ingest = self._grounder.ingest_structured(data)
        return self._allocate_and_wrap(
            ingest,
            operation="ground_structured",
            source="structured",
            context=context,
        )

    def ground_file(
        self, path: str | Path, *, context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Ground a file from disk (.txt / .md / .json / .py / ...)."""
        ingest = self._grounder.ingest_file(path)
        return self._allocate_and_wrap(
            ingest,
            operation="ground_file",
            source="file",
            context=context,
        )

    def concept_similarity(self, handle_a: int, handle_b: int) -> Dict[str, Any]:
        """Compare two previously grounded CCE concepts.

        Falls back to FHRR phase similarity when one of the handles
        was not produced by this bridge (e.g. concepts allocated by
        legacy code paths). When both handles are known, the cached
        semantic cosine is returned, which is a stronger signal than
        the FHRR cosine alone.
        """
        import numpy as np  # local to keep module import cheap

        rec_a = self._concepts.get(int(handle_a))
        rec_b = self._concepts.get(int(handle_b))
        if rec_a is not None and rec_b is not None:
            from shared.semantic_encoder import cosine as _cos

            sim = _cos(rec_a["semantic_vector"], rec_b["semantic_vector"])
            source = "semantic_cache"
        else:
            phases_a = self._mgr.get_cce_phases(int(handle_a))
            phases_b = self._mgr.get_cce_phases(int(handle_b))
            sim = float(np.mean(np.cos(phases_a - phases_b)))
            source = "fhrr_phase"
        return {
            "similarity": float(sim),
            "metadata": {
                "handle_a": int(handle_a),
                "handle_b": int(handle_b),
                "provenance": {
                    "operation": "concept_similarity",
                    "source_arena": "cce",
                    "comparison_source": source,
                    "timestamp": time.time(),
                },
            },
        }

    def list_concepts(self) -> List[Dict[str, Any]]:
        """Return a shallow view of every concept allocated so far."""
        return [
            {
                "cce_handle": handle,
                "preview": rec["preview"],
                "modality": rec["modality"],
            }
            for handle, rec in sorted(self._concepts.items())
        ]

    # ---- internals ---- #

    def _allocate_and_wrap(
        self,
        ingest: Dict[str, Any],
        *,
        operation: str,
        source: str,
        context: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        phases = ingest["phases"]
        if phases.shape != (CCE_ARENA_DIM,):
            raise ValueError(
                f"perceptual grounder produced shape {phases.shape}, "
                f"expected ({CCE_ARENA_DIM},)"
            )
        handle = self._mgr.alloc_cce(phases=phases)
        record = {
            "semantic_vector": ingest["semantic_vector"],
            "preview": ingest["metadata"].get("content_preview", "")[:80],
            "modality": ingest["metadata"].get("modality", source),
        }
        self._concepts[handle] = record
        return {
            "cce_handle": handle,
            "semantic_vector": ingest["semantic_vector"],
            "phases": phases,
            "metadata": {
                "ingest_metadata": ingest["metadata"],
                "context": dict(context or {}),
                "provenance": {
                    "operation": operation,
                    "source_arena": source,
                    "target_arena": "cce",
                    "encoder_backend": self._encoder.backend,
                    "timestamp": time.time(),
                },
            },
        }


__all__ = ["SemanticConceptBridge"]
