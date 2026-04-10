"""Asymmetric CCE↔THDSE dimension bridge (PLAN.md Section C).

FHRR vectors in CCE live in 10,000-dim phase space; FHRR vectors in
THDSE live in 256-dim phase space. There is NO valid algebraic
isomorphism between the two spaces, so the bridge is deliberately
asymmetric:

- **CCE → THDSE**: phase subsampling on a stride of 39. The 256
  indices ``np.arange(0, 10000, 39)[:256]`` are a fixed, stateless
  subset of the 10k axes. Because bind is element-wise phase addition,
  subsampling commutes with bind EXACTLY::

      subsample(bind(A, B)) == bind(subsample(A), subsample(B))

- **THDSE ↔ CCE similarity**: the 10k vector is projected DOWN to 256
  dimensions; the comparison happens in the shared 256-dim space. We
  never attempt to lift a 256-dim vector to 10k-dim — that would
  require fabricating information that was never encoded in the source.

Every public bridge call returns a dict whose ``metadata["provenance"]``
field records the operation, source arena, and bridge parameters.
PLAN.md Rule 9 requires this on every bridge operation result.

On import, this module runs a self-test covering ten random CCE vector
pairs and verifying three invariants (bind commutation, self-similarity
above 0.95, random similarity below 0.1). If any check fails, the
module raises :class:`BridgeIntegrityError` during import — downstream
modules that depend on the bridge must never load against a broken
implementation (PLAN.md Rule 12).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import (
    BRIDGE_RANDOM_SIMILARITY_MAX,
    BRIDGE_SELF_SIMILARITY_MIN,
    BRIDGE_SUBSAMPLE_STRIDE,
    CCE_ARENA_DIM,
    THDSE_ARENA_DIM,
)
from .exceptions import BridgeIntegrityError, DimensionMismatchError

# --------------------------------------------------------------------------- #
# Fixed projection index set
# --------------------------------------------------------------------------- #

#: Indices selected from a 10,000-dim CCE vector to produce a 256-dim
#: THDSE projection. Computed once at import time and frozen — the
#: indices MUST NOT change across calls or the bridge commutation
#: property would silently drift between runs.
PROJECTION_INDICES: np.ndarray = np.arange(
    0, CCE_ARENA_DIM, BRIDGE_SUBSAMPLE_STRIDE
)[:THDSE_ARENA_DIM].copy()

if PROJECTION_INDICES.shape != (THDSE_ARENA_DIM,):
    raise BridgeIntegrityError(
        "projection index construction produced the wrong shape",
        check="index_construction",
        observed=float(PROJECTION_INDICES.shape[0]),
        threshold=float(THDSE_ARENA_DIM),
    )

PROJECTION_INDICES.setflags(write=False)

_TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# Low-level pure math (no metadata, no dict wrapping)
# --------------------------------------------------------------------------- #


def _subsample(vec_10k: np.ndarray) -> np.ndarray:
    """Return the 256-dim phase-subsampled view of a 10k vector (copy)."""
    return vec_10k[PROJECTION_INDICES].copy()


def _fhrr_similarity(vec_a_256: np.ndarray, vec_b_256: np.ndarray) -> float:
    """FHRR similarity: mean cosine of element-wise phase differences."""
    return float(np.mean(np.cos(vec_a_256 - vec_b_256)))


def _validate_cce_vector(vec: np.ndarray, *, operation: str) -> np.ndarray:
    arr = np.asarray(vec)
    if arr.ndim != 1 or arr.shape[0] != CCE_ARENA_DIM:
        raise DimensionMismatchError(
            f"{operation}: expected 1-D vector of length {CCE_ARENA_DIM}",
            expected=(CCE_ARENA_DIM,),
            actual=tuple(arr.shape),
            operation=operation,
        )
    return arr


def _validate_thdse_vector(vec: np.ndarray, *, operation: str) -> np.ndarray:
    arr = np.asarray(vec)
    if arr.ndim != 1 or arr.shape[0] != THDSE_ARENA_DIM:
        raise DimensionMismatchError(
            f"{operation}: expected 1-D vector of length {THDSE_ARENA_DIM}",
            expected=(THDSE_ARENA_DIM,),
            actual=tuple(arr.shape),
            operation=operation,
        )
    return arr


# --------------------------------------------------------------------------- #
# Public API — returns dicts with provenance metadata (Rule 9)
# --------------------------------------------------------------------------- #


def project_down(vec_10k: np.ndarray) -> dict[str, Any]:
    """Project a 10k-dim CCE vector down to a 256-dim THDSE-space vector.

    Returns a dict with keys:

    - ``vector``: numpy array of shape ``(256,)`` containing the
      subsampled phases.
    - ``metadata``: dict whose ``"provenance"`` entry records the
      operation name, source and target arenas, stride, and index
      range used.

    Raises :class:`DimensionMismatchError` if ``vec_10k`` is not a
    1-D array of length 10,000.
    """
    arr = _validate_cce_vector(vec_10k, operation="project_down")
    subsampled = _subsample(arr)
    return {
        "vector": subsampled,
        "metadata": {
            "provenance": {
                "operation": "project_down",
                "source_arena": "cce",
                "target_arena": "thdse",
                "source_dim": CCE_ARENA_DIM,
                "target_dim": THDSE_ARENA_DIM,
                "stride": BRIDGE_SUBSAMPLE_STRIDE,
                "index_count": int(PROJECTION_INDICES.shape[0]),
            }
        },
    }


def cross_arena_similarity(
    vec_10k: np.ndarray, vec_256: np.ndarray
) -> dict[str, Any]:
    """Compute FHRR similarity between a CCE and a THDSE vector.

    The 10k vector is projected DOWN to 256 dims and compared in the
    shared 256-dim space (mean cosine of phase differences). We never
    lift 256 → 10k because that direction is mathematically invalid
    for FHRR (see PLAN.md Section C, "Rejected Approaches").

    Returns a dict with keys:

    - ``similarity``: float in ``[-1.0, 1.0]``.
    - ``metadata``: dict whose ``"provenance"`` entry records the
      operation and the arenas involved.
    """
    arr_10k = _validate_cce_vector(vec_10k, operation="cross_arena_similarity")
    arr_256 = _validate_thdse_vector(
        vec_256, operation="cross_arena_similarity"
    )
    projected = _subsample(arr_10k)
    sim = _fhrr_similarity(projected, arr_256)
    return {
        "similarity": sim,
        "metadata": {
            "provenance": {
                "operation": "cross_arena_similarity",
                "source_arenas": ("cce", "thdse"),
                "compared_in_dim": THDSE_ARENA_DIM,
                "stride": BRIDGE_SUBSAMPLE_STRIDE,
            }
        },
    }


class DimensionBridge:
    """Stateless namespace wrapper around the module-level bridge API.

    Provided so callers who prefer object-style dispatch (``bridge.project_down``)
    can use a single object carried through a dependency-injection chain
    while still routing through the same underlying functions. All
    methods are thin pass-throughs — there is no per-instance state.
    """

    def __init__(self, *, label: str = "omega-thdse"):
        self._label = str(label)

    @property
    def label(self) -> str:
        return self._label

    @property
    def projection_indices(self) -> np.ndarray:
        """Return a read-only view of the fixed stride-39 index set."""
        return PROJECTION_INDICES

    @staticmethod
    def project_down(vec_10k: np.ndarray) -> dict[str, Any]:
        return project_down(vec_10k)

    @staticmethod
    def cross_arena_similarity(
        vec_10k: np.ndarray, vec_256: np.ndarray
    ) -> dict[str, Any]:
        return cross_arena_similarity(vec_10k, vec_256)

    def __repr__(self) -> str:
        return (
            f"DimensionBridge(label={self._label!r}, "
            f"cce_dim={CCE_ARENA_DIM}, thdse_dim={THDSE_ARENA_DIM}, "
            f"stride={BRIDGE_SUBSAMPLE_STRIDE})"
        )


# --------------------------------------------------------------------------- #
# Import-time self-test (PLAN.md Rule 12)
# --------------------------------------------------------------------------- #


def _run_self_test(*, seed: int = 1337, num_pairs: int = 10) -> None:
    """Verify bind commutation, self-similarity, random orthogonality.

    Runs on module import. Failure raises :class:`BridgeIntegrityError`
    so that downstream modules never load against a broken bridge.
    """
    generator = np.random.default_rng(seed)

    for trial in range(num_pairs):
        a = generator.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)
        b = generator.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)

        # (a) subsample(bind(A, B)) == bind(subsample(A), subsample(B))
        bound_full = np.mod(a + b, np.float32(_TWO_PI))
        sub_of_bound = _subsample(bound_full)

        sub_a = _subsample(a)
        sub_b = _subsample(b)
        bind_of_sub = np.mod(sub_a + sub_b, np.float32(_TWO_PI))

        if not np.array_equal(sub_of_bound, bind_of_sub):
            max_diff = float(np.max(np.abs(sub_of_bound - bind_of_sub)))
            raise BridgeIntegrityError(
                "bind commutation check failed: "
                "subsample(bind(A,B)) != bind(subsample(A), subsample(B))",
                check="bind_commutation",
                trial=trial,
                observed=max_diff,
                threshold=0.0,
            )

        # (b) self-similarity > 0.95 (via cross_arena_similarity(A, sub(A)))
        self_result = cross_arena_similarity(a, sub_a)
        self_sim = self_result["similarity"]
        if self_sim <= BRIDGE_SELF_SIMILARITY_MIN:
            raise BridgeIntegrityError(
                "self-similarity below required minimum",
                check="self_similarity",
                trial=trial,
                observed=self_sim,
                threshold=BRIDGE_SELF_SIMILARITY_MIN,
            )

        # (c) random similarity < 0.1
        random_256 = generator.uniform(
            0.0, _TWO_PI, THDSE_ARENA_DIM
        ).astype(np.float32)
        random_result = cross_arena_similarity(a, random_256)
        random_sim = abs(random_result["similarity"])
        if random_sim >= BRIDGE_RANDOM_SIMILARITY_MAX:
            raise BridgeIntegrityError(
                "random similarity exceeded allowed maximum",
                check="random_orthogonality",
                trial=trial,
                observed=random_sim,
                threshold=BRIDGE_RANDOM_SIMILARITY_MAX,
            )


_run_self_test()


__all__ = [
    "PROJECTION_INDICES",
    "project_down",
    "cross_arena_similarity",
    "DimensionBridge",
]
