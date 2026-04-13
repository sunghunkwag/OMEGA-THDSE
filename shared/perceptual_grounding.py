"""Perceptual grounding layer for OMEGA-THDSE Phase 9 (fixes D1).

Before Phase 9 the ingest surface was limited to ``*.py`` source files
parsed by the RSI pipeline. No text document, JSON record, or
structured data blob could become a CCE concept — there was simply no
path from raw perception to the 10,000-dim FHRR space. This module
closes that gap.

``PerceptualGrounder`` accepts three input modalities:

- **text** — free-form natural-language strings.
- **structured data** — dicts / lists / JSON-compatible values, which
  are flattened into a semantically-meaningful "path: leaf" text form
  before encoding.
- **files** — ``.txt``, ``.md``, ``.json``, ``.py`` (and any other
  UTF-8 text) are read and routed to one of the above.

Each ingest call returns a deterministic CCE phase vector
(10,000-dim, values in ``[0, 2π)``) plus metadata (including
``provenance``, per PLAN Rule 9). The phase encoding is produced by
lifting a dense 384-dim semantic embedding into 10k-dim phase space
via a fixed deterministic random projection seeded by
:class:`DeterministicRNG`. The mapping is content-addressable — two
semantically similar texts yield CCE vectors whose FHRR similarity
exceeds the low-level encoder's raw cosine (so downstream HDC
operations operate on genuine semantic geometry, not crc32 noise).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .constants import CCE_ARENA_DIM, SEMANTIC_ENCODER_DIM
from .deterministic_rng import DeterministicRNG
from .semantic_encoder import SemanticEncoder, cosine


_TWO_PI = 2.0 * np.pi


def _build_lifting_matrix(
    rng: DeterministicRNG, in_dim: int, out_dim: int
) -> np.ndarray:
    """Deterministic Gaussian lifting matrix (semantic → phase space).

    Seeded from :class:`DeterministicRNG` fork ``perceptual.lift`` so
    two processes with the same master seed build identical matrices.
    """
    seed = rng.child_seed("perceptual.lift")
    gen = np.random.default_rng(seed)
    mat = gen.standard_normal((out_dim, in_dim)).astype(np.float32)
    # Row-normalise so projections have roughly unit scale regardless
    # of ``in_dim`` — prevents phase aliasing in the modulo below.
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return mat / norms


def _semantic_to_phases(
    semantic_vec: np.ndarray, lift: np.ndarray
) -> np.ndarray:
    """Lift a 384-dim unit vector into a 10k-dim phase vector."""
    raw = lift @ semantic_vec.astype(np.float32)  # shape (out_dim,)
    # Map to phase space through a sigmoid-like wrap so equal-semantic
    # vectors produce near-identical phases. ``arctan2(sin, cos)`` of
    # the scaled values keeps phases in ``(-π, π]``; shifting by π
    # gives ``[0, 2π)`` consistent with the FHRR convention.
    phases = np.arctan2(np.sin(raw * np.pi), np.cos(raw * np.pi)) + np.pi
    return np.mod(phases.astype(np.float32), np.float32(_TWO_PI))


def _flatten_structured(value: Any, prefix: str = "") -> List[str]:
    """Flatten a JSON-compatible value into ``"path: leaf"`` strings.

    The output is deterministic (sorted keys) so the same dict always
    produces the same token stream — a requirement for reproducible
    perceptual grounding.
    """
    tokens: List[str] = []
    if isinstance(value, dict):
        for key in sorted(value.keys(), key=str):
            sub = f"{prefix}.{key}" if prefix else str(key)
            tokens.extend(_flatten_structured(value[key], sub))
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            sub = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            tokens.extend(_flatten_structured(item, sub))
    elif value is None:
        tokens.append(f"{prefix}: null" if prefix else "null")
    else:
        leaf = str(value)
        tokens.append(f"{prefix}: {leaf}" if prefix else leaf)
    return tokens


def _extension(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


class PerceptualGrounder:
    """Ground raw perceptual inputs into CCE-compatible phase vectors.

    The grounder owns an internal :class:`SemanticEncoder` plus a
    deterministic lifting matrix that maps the encoder's 384-dim
    output to CCE's 10,000-dim phase space.
    """

    #: File extensions the grounder parses as UTF-8 text.
    TEXT_EXTENSIONS: frozenset[str] = frozenset(
        {"txt", "md", "rst", "py", "toml", "yaml", "yml", "cfg", "ini"}
    )
    #: File extensions the grounder parses as structured JSON.
    JSON_EXTENSIONS: frozenset[str] = frozenset({"json"})

    def __init__(
        self,
        encoder: SemanticEncoder | None = None,
        rng: DeterministicRNG | None = None,
    ):
        self._rng = rng or DeterministicRNG(master_seed=42)
        self._encoder = encoder or SemanticEncoder(rng=self._rng)
        if self._encoder.dim != SEMANTIC_ENCODER_DIM:
            raise ValueError(
                f"SemanticEncoder dim must be {SEMANTIC_ENCODER_DIM}, "
                f"got {self._encoder.dim}"
            )
        self._lift = _build_lifting_matrix(
            self._rng, self._encoder.dim, CCE_ARENA_DIM
        )
        self._ingest_count = 0

    # ---- metadata ---- #

    @property
    def encoder(self) -> SemanticEncoder:
        return self._encoder

    @property
    def ingest_count(self) -> int:
        return self._ingest_count

    # ---- primary ingest methods ---- #

    def ingest_text(self, text: str) -> Dict[str, Any]:
        """Encode ``text`` into a CCE phase vector."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        sem = self._encoder.encode(text)
        phases = _semantic_to_phases(sem, self._lift)
        self._ingest_count += 1
        return {
            "phases": phases,
            "semantic_vector": sem,
            "metadata": self._build_metadata(
                operation="ingest_text",
                modality="text",
                content_preview=text[:120],
                payload_hash=self._hash_payload(text),
            ),
        }

    def ingest_structured(self, data: Any) -> Dict[str, Any]:
        """Flatten + encode a JSON-compatible structured value."""
        tokens = _flatten_structured(data)
        text = " ; ".join(tokens) if tokens else "∅"
        sem = self._encoder.encode(text)
        phases = _semantic_to_phases(sem, self._lift)
        self._ingest_count += 1
        payload_repr = json.dumps(data, sort_keys=True, default=str)
        return {
            "phases": phases,
            "semantic_vector": sem,
            "metadata": self._build_metadata(
                operation="ingest_structured",
                modality="structured",
                content_preview=text[:120],
                payload_hash=self._hash_payload(payload_repr),
                token_count=len(tokens),
            ),
        }

    def ingest_file(self, path: str | Path) -> Dict[str, Any]:
        """Read a file from disk and route it to the matching ingest."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"perceptual ingest target missing: {p}")
        ext = _extension(p)
        raw = p.read_text(encoding="utf-8")
        if ext in self.JSON_EXTENSIONS:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = raw
            if isinstance(data, (dict, list, tuple)):
                result = self.ingest_structured(data)
            else:
                result = self.ingest_text(str(data))
            result["metadata"]["source_path"] = str(p)
            result["metadata"]["file_extension"] = ext
            return result
        result = self.ingest_text(raw)
        result["metadata"]["source_path"] = str(p)
        result["metadata"]["file_extension"] = ext
        return result

    def similarity(self, a_result: Dict[str, Any], b_result: Dict[str, Any]) -> float:
        """Cosine similarity between two ingest results (semantic space)."""
        return cosine(a_result["semantic_vector"], b_result["semantic_vector"])

    # ---- internals ---- #

    def _build_metadata(self, **fields: Any) -> Dict[str, Any]:
        return {
            "ingest_index": self._ingest_count,
            **fields,
            "provenance": {
                "operation": fields.get("operation", "perceptual_ingest"),
                "source_arena": "perceptual",
                "target_arena": "cce",
                "encoder_backend": self._encoder.backend,
                "encoder_dim": self._encoder.dim,
                "target_dim": CCE_ARENA_DIM,
                "timestamp": time.time(),
            },
        }

    @staticmethod
    def _hash_payload(payload: str) -> str:
        return hashlib.blake2b(
            payload.encode("utf-8", errors="replace"), digest_size=8
        ).hexdigest()


__all__ = ["PerceptualGrounder"]
