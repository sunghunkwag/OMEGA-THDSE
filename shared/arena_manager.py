"""Central owner of every FHRR arena in the unified OMEGA-THDSE engine.

PLAN.md Rule 3: "No file except ``arena_manager.py`` may call
``hdc_core.FhrrArena()`` directly. All arena access goes through
ArenaManager."

PLAN.md Section A target: replace the three fragmented arenas (CCE at
10k-dim, THDSE at 256-dim, orphaned bridge at 10k-dim) with a single
:class:`ArenaManager` that owns exactly two working arenas plus a
bridge arena for cross-engine correlation.

This module is the ONLY place in the codebase permitted to import
``hdc_core`` or instantiate ``hdc_core.FhrrArena``. When the Rust
backend is unavailable, the manager falls back to :class:`_PyFhrrArena`
— a pure-Python arena that implements the same public contract over
numpy phase buffers.

PLAN.md Section F Risk 1 requires pickling an ArenaManager to raise
``RuntimeError`` because the underlying Rust ``FhrrArena`` contains
non-serializable FFI state. Both :meth:`__reduce__` and
:meth:`__getstate__` enforce this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .constants import (
    BRIDGE_ARENA_CAP,
    BRIDGE_ARENA_DIM,
    CCE_ARENA_CAP,
    CCE_ARENA_DIM,
    THDSE_ARENA_CAP,
    THDSE_ARENA_DIM,
)
from .deterministic_rng import DeterministicRNG
from .exceptions import DimensionMismatchError

# --------------------------------------------------------------------------- #
# Rust backend detection (Rule 3: only this file may touch hdc_core.FhrrArena)
# --------------------------------------------------------------------------- #

try:  # pragma: no cover — environment-dependent
    import hdc_core  # type: ignore

    _RustFhrrArena = hdc_core.FhrrArena
    _HAS_RUST_BACKEND = True
except Exception:  # pragma: no cover — Rust wheel absent in dev env
    _RustFhrrArena = None
    _HAS_RUST_BACKEND = False


_TWO_PI = np.float32(2.0 * math.pi)


class _PyFhrrArena:
    """Pure-Python FHRR arena (fallback when ``hdc_core`` is unavailable).

    Stores phase vectors as a contiguous float32 buffer of shape
    ``(capacity, dimension)``. All phases are wrapped into ``[0, 2π)``
    on write. The public surface mirrors the subset of the Rust
    :class:`FhrrArena` methods that Phase 2 exercises: ``allocate``,
    ``inject_phases``, ``get_phases``, ``bind``, ``bundle``,
    ``similarity``, and metadata accessors.
    """

    def __init__(self, capacity: int, dimension: int):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity must be a positive int, got {capacity!r}")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"dimension must be a positive int, got {dimension!r}")
        self._capacity = capacity
        self._dimension = dimension
        self._head = 0
        self._buffer = np.zeros((capacity, dimension), dtype=np.float32)

    # ---- metadata ---- #

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def dimension(self) -> int:
        return self._dimension

    def get_head(self) -> int:
        return self._head

    def get_dimension(self) -> int:
        return self._dimension

    def get_capacity(self) -> int:
        return self._capacity

    # ---- allocation ---- #

    def allocate(self) -> int:
        if self._head >= self._capacity:
            raise RuntimeError(
                f"_PyFhrrArena capacity exhausted (dim={self._dimension}, "
                f"cap={self._capacity})"
            )
        handle = self._head
        self._head += 1
        return handle

    def reset(self) -> None:
        self._head = 0
        self._buffer.fill(0.0)

    # ---- I/O ---- #

    def inject_phases(self, handle: int, phases: Iterable[float]) -> None:
        self._validate_handle(handle)
        arr = np.asarray(phases, dtype=np.float32)
        if arr.shape != (self._dimension,):
            raise ValueError(
                f"Phase length mismatch: expected ({self._dimension},), "
                f"got {tuple(arr.shape)}"
            )
        self._buffer[handle] = np.mod(arr, _TWO_PI)

    def get_phases(self, handle: int) -> np.ndarray:
        self._validate_handle(handle)
        return self._buffer[handle].copy()

    # ---- FHRR algebra ---- #

    def bind(self, h1: int, h2: int, out: int) -> None:
        self._validate_handle(h1)
        self._validate_handle(h2)
        self._validate_handle(out)
        self._buffer[out] = np.mod(
            self._buffer[h1] + self._buffer[h2], _TWO_PI
        )

    def bundle(self, handles: list[int], out: int) -> None:
        if not handles:
            raise ValueError("bundle requires at least one input handle")
        for h in handles:
            self._validate_handle(h)
        self._validate_handle(out)
        sin_sum = np.zeros(self._dimension, dtype=np.float32)
        cos_sum = np.zeros(self._dimension, dtype=np.float32)
        for h in handles:
            sin_sum += np.sin(self._buffer[h])
            cos_sum += np.cos(self._buffer[h])
        self._buffer[out] = np.mod(np.arctan2(sin_sum, cos_sum), _TWO_PI)

    def similarity(self, h1: int, h2: int) -> float:
        self._validate_handle(h1)
        self._validate_handle(h2)
        diff = self._buffer[h1] - self._buffer[h2]
        return float(np.mean(np.cos(diff)))

    # ---- internal ---- #

    def _validate_handle(self, handle: int) -> None:
        if not isinstance(handle, (int, np.integer)):
            raise TypeError(
                f"handle must be int, got {type(handle).__name__}"
            )
        if not 0 <= int(handle) < self._head:
            raise IndexError(
                f"invalid handle {handle}: valid range is [0, {self._head})"
            )


@dataclass(frozen=True)
class HandleTag:
    """Provenance tag recorded for every allocated handle.

    Attached so that :mod:`shared.dimension_bridge` and downstream
    bridge modules can verify which arena a handle came from before
    mixing it with vectors from another arena (PLAN.md Rule 6).
    """

    arena: str
    dimension: int
    handle: int


class ArenaManager:
    """Central owner of the CCE, THDSE, and bridge FHRR arenas.

    Only one instance is expected per process. Downstream code should
    receive the manager (or one of its ``alloc_*`` methods) by dependency
    injection rather than importing a global, so tests can construct
    isolated managers on demand.
    """

    _BACKEND_RUST = "rust"
    _BACKEND_PY = "python"

    def __init__(self, master_seed: int = 42):
        if _HAS_RUST_BACKEND and _RustFhrrArena is not None:  # pragma: no cover
            # NOTE: PLAN.md Rule 3 — this is the ONE AND ONLY call site
            # for hdc_core.FhrrArena in the entire repository.
            self._cce_arena: Any = _RustFhrrArena(CCE_ARENA_CAP, CCE_ARENA_DIM)
            self._thdse_arena: Any = _RustFhrrArena(
                THDSE_ARENA_CAP, THDSE_ARENA_DIM
            )
            self._bridge_arena: Any = _RustFhrrArena(
                BRIDGE_ARENA_CAP, BRIDGE_ARENA_DIM
            )
            self._backend: str = self._BACKEND_RUST
        else:
            self._cce_arena = _PyFhrrArena(CCE_ARENA_CAP, CCE_ARENA_DIM)
            self._thdse_arena = _PyFhrrArena(
                THDSE_ARENA_CAP, THDSE_ARENA_DIM
            )
            self._bridge_arena = _PyFhrrArena(
                BRIDGE_ARENA_CAP, BRIDGE_ARENA_DIM
            )
            self._backend = self._BACKEND_PY

        self._rng = DeterministicRNG(master_seed=master_seed)
        self._tags: dict[tuple[str, int], HandleTag] = {}

    # ---- identity / metadata ---- #

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def rng(self) -> DeterministicRNG:
        return self._rng

    @property
    def cce_dim(self) -> int:
        return CCE_ARENA_DIM

    @property
    def thdse_dim(self) -> int:
        return THDSE_ARENA_DIM

    @property
    def bridge_dim(self) -> int:
        return BRIDGE_ARENA_DIM

    @property
    def cce_capacity(self) -> int:
        return CCE_ARENA_CAP

    @property
    def thdse_capacity(self) -> int:
        return THDSE_ARENA_CAP

    @property
    def bridge_capacity(self) -> int:
        return BRIDGE_ARENA_CAP

    def count(self, arena: str) -> int:
        """Return the number of handles currently allocated in ``arena``."""
        return self._select_arena(arena).get_head()

    def tag_of(self, arena: str, handle: int) -> HandleTag:
        """Return the :class:`HandleTag` recorded for ``(arena, handle)``."""
        key = (arena, int(handle))
        if key not in self._tags:
            raise KeyError(
                f"no handle tagged for arena={arena!r}, handle={handle!r}"
            )
        return self._tags[key]

    # ---- allocation API ---- #

    def alloc_cce(self, phases: np.ndarray | None = None) -> int:
        """Allocate a CCE (10k-dim) handle, optionally seeded with ``phases``."""
        return self._alloc("cce", self._cce_arena, CCE_ARENA_DIM, phases)

    def alloc_thdse(self, phases: np.ndarray | None = None) -> int:
        """Allocate a THDSE (256-dim) handle, optionally seeded with ``phases``."""
        return self._alloc("thdse", self._thdse_arena, THDSE_ARENA_DIM, phases)

    def alloc_bridge(self, phases: np.ndarray | None = None) -> int:
        """Allocate a bridge (10k-dim) handle, optionally seeded with ``phases``."""
        return self._alloc("bridge", self._bridge_arena, BRIDGE_ARENA_DIM, phases)

    # ---- read-back helpers ---- #

    def get_cce_phases(self, handle: int) -> np.ndarray:
        return self._cce_arena.get_phases(int(handle))

    def get_thdse_phases(self, handle: int) -> np.ndarray:
        return self._thdse_arena.get_phases(int(handle))

    def get_bridge_phases(self, handle: int) -> np.ndarray:
        return self._bridge_arena.get_phases(int(handle))

    # ---- process-isolation guards (Rule 11 / Risk 1) ---- #

    def __reduce__(self):
        raise RuntimeError(
            "Cannot serialize ArenaManager: owns Rust FFI objects "
            "(hdc_core.FhrrArena). Pass integer handle IDs across process "
            "boundaries instead (PLAN.md Rule 11 / Section F Risk 1)."
        )

    def __getstate__(self):
        raise RuntimeError(
            "Cannot serialize ArenaManager: owns Rust FFI objects "
            "(hdc_core.FhrrArena). Pass integer handle IDs across process "
            "boundaries instead (PLAN.md Rule 11 / Section F Risk 1)."
        )

    def __setstate__(self, state):
        raise RuntimeError(
            "Cannot deserialize ArenaManager: owns Rust FFI objects "
            "(hdc_core.FhrrArena). Reconstruct locally in each process."
        )

    def __deepcopy__(self, memo):
        raise RuntimeError(
            "Cannot deepcopy ArenaManager: owns Rust FFI objects "
            "(hdc_core.FhrrArena)."
        )

    # ---- internals ---- #

    def _alloc(
        self,
        arena_name: str,
        arena: Any,
        expected_dim: int,
        phases: np.ndarray | None,
    ) -> int:
        if phases is not None:
            arr = np.asarray(phases, dtype=np.float32)
            if arr.shape != (expected_dim,):
                raise DimensionMismatchError(
                    f"alloc_{arena_name}: phase vector has wrong shape",
                    expected=(expected_dim,),
                    actual=tuple(arr.shape),
                    operation=f"alloc_{arena_name}",
                )
        handle = int(arena.allocate())
        if phases is not None:
            arena.inject_phases(handle, arr)
        self._tags[(arena_name, handle)] = HandleTag(
            arena=arena_name, dimension=expected_dim, handle=handle
        )
        return handle

    def _select_arena(self, arena: str) -> Any:
        if arena == "cce":
            return self._cce_arena
        if arena == "thdse":
            return self._thdse_arena
        if arena == "bridge":
            return self._bridge_arena
        raise ValueError(
            f"unknown arena name {arena!r}; expected 'cce', 'thdse', or 'bridge'"
        )

    def __repr__(self) -> str:
        return (
            f"ArenaManager(backend={self._backend}, "
            f"cce={self.count('cce')}/{CCE_ARENA_CAP}@{CCE_ARENA_DIM}, "
            f"thdse={self.count('thdse')}/{THDSE_ARENA_CAP}@{THDSE_ARENA_DIM}, "
            f"bridge={self.count('bridge')}/{BRIDGE_ARENA_CAP}@{BRIDGE_ARENA_DIM})"
        )


__all__ = ["ArenaManager", "HandleTag"]
