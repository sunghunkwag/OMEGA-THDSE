"""THDSE arena factory — single chokepoint for FHRR arena allocation.

PLAN.md Rule 3 forbids direct Rust-arena constructor calls anywhere
except :mod:`shared.arena_manager`. THDSE-side code historically
constructed its own arena instances per-class; this module is the
migration target. Every THDSE caller now invokes :func:`make_arena`
which returns either:

1. The Rust-backed arena borrowed from an injected
   :class:`shared.arena_manager.ArenaManager` (preferred, zero-copy).
2. A pure-Python ``_PyFhrrArenaExtended`` that implements the full
   public surface that THDSE expects (``allocate``, ``inject_phases``,
   ``extract_phases``, ``bind``, ``bundle``, ``compute_correlation``,
   ``correlate_matrix``, ``bind_bundle_fusion``, ``expand_dimension``,
   ``get_op_counts``, …). This is the fallback when no manager is
   available.

This module **never** imports the Rust crate directly. Rule 3 is
preserved: even our Python fallback is a Python class, not a wrapper
around a Rust constructor.

Rule 16 backward compatibility: every THDSE Python file that used to
construct a Rust arena directly should now call
``make_arena(capacity, dimension, arena_manager=...)``. With
``arena_manager=None`` (legacy mode) the Python fallback is used so
the file still runs standalone.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

_TWO_PI = 2.0 * math.pi


class _PyFhrrArenaExtended:
    """Pure-Python FHRR arena exposing the full ``hdc_core.FhrrArena`` API.

    The Rust crate stores complex (re, im) pairs internally; this class
    stores raw phases in ``[0, 2π)`` and recomputes complex
    representations on demand. The behavioural contract matches the
    Rust crate within float32 tolerance.
    """

    def __init__(self, capacity: int, dimension: int):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity must be a positive int, got {capacity!r}")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(
                f"dimension must be a positive int, got {dimension!r}"
            )
        self._capacity = capacity
        self._dimension = dimension
        self._head = 0
        # Phase storage indexed by handle. We allocate lazily so empty
        # capacity is cheap.
        self._phases: List[Optional[List[float]]] = [None] * capacity
        self._op_counts: List[Tuple[int, int]] = [(0, 0)] * capacity

    # ---- metadata ---- #

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def dimension(self) -> int:
        return self._dimension

    def get_dimension(self) -> int:
        return self._dimension

    def get_head(self) -> int:
        return self._head

    def get_capacity(self) -> int:
        return self._capacity

    # ---- allocation ---- #

    def allocate(self) -> int:
        if self._head >= self._capacity:
            raise ValueError(
                f"_PyFhrrArenaExtended capacity exhausted "
                f"(dim={self._dimension}, cap={self._capacity})"
            )
        handle = self._head
        self._head += 1
        self._phases[handle] = [0.0] * self._dimension
        return handle

    def reset(self) -> None:
        self._head = 0
        for i in range(len(self._phases)):
            self._phases[i] = None
            self._op_counts[i] = (0, 0)

    # ---- I/O ---- #

    def inject_phases(self, handle: int, phases: Sequence[float]) -> None:
        self._validate_handle(handle)
        seq = list(phases)
        if len(seq) != self._dimension:
            raise ValueError(
                f"Phase length mismatch: expected {self._dimension}, "
                f"got {len(seq)}"
            )
        self._phases[handle] = [float(p) % _TWO_PI for p in seq]

    def extract_phases(self, handle: int) -> List[float]:
        self._validate_handle(handle)
        stored = self._phases[handle]
        return list(stored) if stored is not None else [0.0] * self._dimension

    # Some Rust callers use ``get_phases``; alias for compatibility
    # with the simpler ``shared._PyFhrrArena``.
    def get_phases(self, handle: int) -> List[float]:
        return self.extract_phases(handle)

    # ---- FHRR algebra ---- #

    def bind(self, h1: int, h2: int, out: int) -> None:
        self._validate_handle(h1)
        self._validate_handle(h2)
        self._validate_handle(out)
        a = self._phases[h1] or []
        b = self._phases[h2] or []
        self._phases[out] = [
            (a[i] + b[i]) % _TWO_PI for i in range(self._dimension)
        ]
        bind_count, bundle_count = self._op_counts[out]
        self._op_counts[out] = (bind_count + 1, bundle_count)

    def bundle(self, handles: Sequence[int], out: int) -> None:
        if not handles:
            raise ValueError("bundle requires at least one input handle")
        for h in handles:
            self._validate_handle(h)
        self._validate_handle(out)
        d = self._dimension
        sin_sum = [0.0] * d
        cos_sum = [0.0] * d
        for h in handles:
            phases = self._phases[h] or [0.0] * d
            for i in range(d):
                sin_sum[i] += math.sin(phases[i])
                cos_sum[i] += math.cos(phases[i])
        result = [
            math.atan2(sin_sum[i], cos_sum[i]) % _TWO_PI for i in range(d)
        ]
        self._phases[out] = result
        bind_count, bundle_count = self._op_counts[out]
        self._op_counts[out] = (bind_count, bundle_count + len(handles))

    def compute_correlation(self, h1: int, h2: int) -> float:
        self._validate_handle(h1)
        self._validate_handle(h2)
        a = self._phases[h1] or []
        b = self._phases[h2] or []
        d = self._dimension
        return float(sum(math.cos(a[i] - b[i]) for i in range(d)) / d)

    # The Rust arena exposes ``correlate_matrix`` for batched scoring.
    def correlate_matrix(self, handles: Sequence[int]) -> List[List[float]]:
        n = len(handles)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self.compute_correlation(
                        handles[i], handles[j]
                    )
        return matrix

    def bind_bundle_fusion(
        self,
        h_bind: int,
        handles_bundle: Sequence[int],
        out: int,
    ) -> None:
        if not handles_bundle:
            raise ValueError("bind_bundle_fusion requires non-empty handles")
        # Allocate a temp slot for the intermediate bundle.
        tmp = self.allocate()
        self.bundle(handles_bundle, tmp)
        self.bind(h_bind, tmp, out)
        # Track the fused cost.
        bind_count, bundle_count = self._op_counts[out]
        self._op_counts[out] = (bind_count + 1, bundle_count + 1)

    # ---- meta-grammar emergence ---- #

    def expand_dimension(self, new_dim: int) -> None:
        if new_dim <= self._dimension:
            raise ValueError(
                f"new_dim {new_dim} must exceed current {self._dimension}"
            )
        old_dim = self._dimension
        for h in range(self._head):
            existing = self._phases[h]
            if existing is None:
                self._phases[h] = [0.0] * new_dim
                continue
            extended = list(existing)
            for j in range(old_dim, new_dim):
                # Conjugate reflection of the periodic component, mod 2π.
                extended.append((-existing[j % old_dim]) % _TWO_PI)
            self._phases[h] = extended
        self._dimension = new_dim

    # ---- thermodynamic cost tracking ---- #

    def get_op_counts(self, handle: int) -> Tuple[int, int]:
        self._validate_handle(handle)
        return self._op_counts[handle]

    def record_bind_cost(self, handle: int) -> None:
        self._validate_handle(handle)
        bind_count, bundle_count = self._op_counts[handle]
        self._op_counts[handle] = (bind_count + 1, bundle_count)

    def record_bundle_cost(self, handle: int, fan_in: int) -> None:
        self._validate_handle(handle)
        bind_count, bundle_count = self._op_counts[handle]
        self._op_counts[handle] = (bind_count, bundle_count + int(fan_in))

    # ---- quotient space projection (PLAN.md Phase 8B / Rule 22) ---- #

    def project_to_quotient_space(self, v_error_id: int) -> int:
        """Project every other live handle away from ``v_error_id``.

        Phase-space port of the Rust ``project_to_quotient_space``
        scalar implementation. For each handle ``h`` (except v_error
        itself), we:

        1. Convert ``v_error`` and ``X`` from phases to complex (cos, sin).
        2. Compute the complex inner product ``<V, X>``.
        3. Compute coeff = ``<V, X> / |V|²``.
        4. Subtract ``V * coeff`` from ``X``.
        5. Renormalize each component to unit magnitude.
        6. Convert back to phases via ``atan2``.

        Returns the number of handles that were modified. The output
        matches the Rust crate within float32 tolerance.
        """
        self._validate_handle(v_error_id)
        v_phases = self._phases[v_error_id]
        if v_phases is None:
            return 0

        v_re = [math.cos(p) for p in v_phases]
        v_im = [math.sin(p) for p in v_phases]
        norm_sq = sum(
            v_re[j] * v_re[j] + v_im[j] * v_im[j]
            for j in range(self._dimension)
        )
        if norm_sq < 1e-12:
            return 0
        inv_norm_sq = 1.0 / norm_sq

        projected_count = 0
        for h in range(self._head):
            if h == v_error_id:
                continue
            x_phases = self._phases[h]
            if x_phases is None:
                continue

            x_re = [math.cos(p) for p in x_phases]
            x_im = [math.sin(p) for p in x_phases]

            dot_re = sum(
                v_re[j] * x_re[j] + v_im[j] * x_im[j]
                for j in range(self._dimension)
            )
            dot_im = sum(
                v_re[j] * x_im[j] - v_im[j] * x_re[j]
                for j in range(self._dimension)
            )
            coeff_re = dot_re * inv_norm_sq
            coeff_im = dot_im * inv_norm_sq
            if coeff_re * coeff_re + coeff_im * coeff_im < 1e-16:
                continue

            new_phases: List[float] = []
            for j in range(self._dimension):
                proj_re = v_re[j] * coeff_re - v_im[j] * coeff_im
                proj_im = v_re[j] * coeff_im + v_im[j] * coeff_re
                xr = x_re[j] - proj_re
                xi = x_im[j] - proj_im
                mag = math.sqrt(xr * xr + xi * xi)
                if mag > 1e-12:
                    xr /= mag
                    xi /= mag
                new_phases.append(math.atan2(xi, xr) % _TWO_PI)
            self._phases[h] = new_phases
            projected_count += 1
        return projected_count

    def project_to_multi_quotient_space(
        self, v_error_ids: Sequence[int]
    ) -> int:
        """Repeatedly project away from each error vector in turn."""
        total = 0
        for v_id in v_error_ids:
            total += self.project_to_quotient_space(int(v_id))
        return total

    # ---- internal ---- #

    def _validate_handle(self, handle: int) -> None:
        if not isinstance(handle, int):
            raise TypeError(
                f"handle must be int, got {type(handle).__name__}"
            )
        if not 0 <= handle < self._head:
            raise IndexError(
                f"invalid handle {handle}: valid range is [0, {self._head})"
            )


def make_arena(
    capacity: int,
    dimension: int,
    arena_manager: Optional[Any] = None,
) -> Any:
    """Return an arena suitable for THDSE work at ``(capacity, dimension)``.

    When ``arena_manager`` is provided AND its backend is the Rust
    crate AND the requested dimension matches the manager's CCE or
    THDSE arena, the manager-owned arena is returned directly so the
    caller benefits from zero-copy access to the Rust SIMD kernels.

    In every other case (no manager, Python fallback backend,
    mismatched dimension) a fresh :class:`_PyFhrrArenaExtended` is
    constructed. Rule 3 is preserved because this module never imports
    or instantiates ``hdc_core.FhrrArena`` itself; the only legitimate
    Rust path goes through :class:`shared.arena_manager.ArenaManager`.
    """
    if arena_manager is not None:
        backend = getattr(arena_manager, "backend", None)
        if backend == "rust":
            cce_dim = getattr(arena_manager, "cce_dim", None)
            thdse_dim = getattr(arena_manager, "thdse_dim", None)
            if dimension == thdse_dim:
                return arena_manager._thdse_arena  # noqa: SLF001
            if dimension == cce_dim:
                return arena_manager._cce_arena  # noqa: SLF001
    return _PyFhrrArenaExtended(capacity, dimension)


__all__ = ["_PyFhrrArenaExtended", "make_arena"]
