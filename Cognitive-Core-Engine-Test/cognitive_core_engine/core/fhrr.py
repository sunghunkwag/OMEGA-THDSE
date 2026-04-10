"""FHRR Vectors routed through the unified ArenaManager (PLAN.md Phase 4).

Phase 2 introduced ``shared.arena_manager.ArenaManager`` as the single
owner of every FHRR arena in the unified engine. This module used to
own its own ``hdc_core.FhrrArena`` instance at module scope, which
violated PLAN.md Rule 3. The arena is now injected via
:func:`set_arena_manager`, and every algebraic op routes through
``arena_manager.cce_arena`` (DIM=10000, CAP=100000).

The :class:`FhrrVector` public API (``from_seed``, ``zero``, ``bind``,
``bundle``, ``similarity``, ``cosine_similarity``, ``permute``,
``fractional_bind``) is unchanged.
"""
from __future__ import annotations
import zlib
import numpy as np
from typing import List, Optional, Any

from shared.arena_manager import ArenaManager

# Backwards-compat constants — kept so legacy callers that read these
# names continue to import cleanly. They MUST NOT be used to construct
# a new arena anywhere; the only legitimate arena owner is ArenaManager.
GLOBAL_ARENA_CAPACITY = 100000
GLOBAL_ARENA_DIMENSION = 10000

_ARENA_MANAGER: Optional[ArenaManager] = None


def set_arena_manager(mgr: ArenaManager) -> None:
    """Wire this module to a central :class:`ArenaManager` instance.

    Called by :class:`Orchestrator` during construction. Replaces the
    legacy module-level ``ARENA = hdc_core.FhrrArena(...)`` global with
    a manager-owned arena, satisfying PLAN.md Rule 3.
    """
    global _ARENA_MANAGER
    if not isinstance(mgr, ArenaManager):
        raise TypeError(
            f"set_arena_manager expected ArenaManager, got {type(mgr).__name__}"
        )
    _ARENA_MANAGER = mgr


def get_arena_manager() -> ArenaManager:
    """Return the active :class:`ArenaManager` or raise if not wired."""
    if _ARENA_MANAGER is None:
        raise RuntimeError(
            "ArenaManager not set. Call set_arena_manager() first."
        )
    return _ARENA_MANAGER


def _cce_arena():
    """Internal helper: return the CCE-side arena from the active manager."""
    return get_arena_manager()._cce_arena  # noqa: SLF001 — sanctioned access


class FhrrVector:
    """Wrapper that mimics the old HyperVector interface but routes to ArenaManager's CCE arena."""
    DIM = GLOBAL_ARENA_DIMENSION

    def __init__(self, handle: Optional[int] = None) -> None:
        if handle is None:
            mgr = get_arena_manager()
            # Use the manager's deterministic RNG fork instead of the
            # global numpy RNG so this constructor is reproducible
            # under PLAN.md Rule 10.
            rng = mgr.rng.fork("fhrr_default")
            phases = (rng.random(self.DIM) * 2 * np.pi).astype(np.float32)
            self.handle = mgr.alloc_cce(phases=phases)
        else:
            self.handle = handle

    @classmethod
    def from_seed(cls, seed_obj: Any) -> "FhrrVector":
        s = str(seed_obj)
        h = zlib.crc32(s.encode('utf-8')) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        phases = (rng.random(cls.DIM) * 2 * np.pi).astype(np.float32)
        mgr = get_arena_manager()
        handle = mgr.alloc_cce(phases=phases)
        return cls(handle)

    @classmethod
    def zero(cls) -> "FhrrVector":
        mgr = get_arena_manager()
        zero_phases = np.zeros(cls.DIM, dtype=np.float32)
        handle = mgr.alloc_cce(phases=zero_phases)
        return cls(handle)

    def bind(self, other: "FhrrVector") -> "FhrrVector":
        arena = _cce_arena()
        out_handle = arena.allocate()
        arena.bind(self.handle, other.handle, out_handle)
        return FhrrVector(out_handle)

    def fractional_bind(self, other: "FhrrVector", role_index: int) -> "FhrrVector":
        # Match legacy behavior: permute the role first then bind.
        shifted = other.permute(role_index * 7 + 1)
        return self.bind(shifted)

    def permute(self, shifts: int = 1) -> "FhrrVector":
        arena = _cce_arena()
        # The Rust arena exposes ``extract_phases``; the python fallback
        # exposes ``get_phases``. Try the legacy name first then fall
        # back, so this method works on both backends.
        if hasattr(arena, "extract_phases"):
            phases = np.asarray(arena.extract_phases(self.handle))
        else:
            phases = arena.get_phases(self.handle)
        rolled = np.roll(phases, shifts).astype(np.float32)
        out_handle = arena.allocate()
        arena.inject_phases(out_handle, rolled)
        return FhrrVector(out_handle)

    def similarity(self, other: "FhrrVector") -> float:
        arena = _cce_arena()
        if hasattr(arena, "compute_correlation"):
            raw_corr = arena.compute_correlation(self.handle, other.handle)
        else:
            raw_corr = arena.similarity(self.handle, other.handle)
        # Map [-1, 1] cosine similarity to [0, 1] like the legacy API.
        return float((raw_corr + 1.0) / 2.0)

    def cosine_similarity(self, other: "FhrrVector") -> float:
        arena = _cce_arena()
        if hasattr(arena, "compute_correlation"):
            return float(arena.compute_correlation(self.handle, other.handle))
        return float(arena.similarity(self.handle, other.handle))

    @staticmethod
    def bundle(vectors: List["FhrrVector"]) -> "FhrrVector":
        if not vectors:
            return FhrrVector.zero()
        arena = _cce_arena()
        out_handle = arena.allocate()
        handles = [v.handle for v in vectors]
        arena.bundle(handles, out_handle)
        return FhrrVector(out_handle)
