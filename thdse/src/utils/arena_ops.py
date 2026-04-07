"""
Arena operations — pure-Python utilities for FHRR phase algebra.

These utilities operate on phase arrays (List[float]) alongside the Rust arena,
enabling operations the arena doesn't expose (conjugate, phase extraction).

No raw buffer access required. All operations derive from the algebraic
properties of FHRR (Fourier Holographic Reduced Representation):
  - bind(A, B) phases  = A_phases + B_phases  (element-wise addition)
  - bundle({Aᵢ}) phases = atan2(Σ sin(θᵢ), Σ cos(θᵢ))  per dimension
  - conjugate(A) phases = -A_phases  (negation)

In FHRR, every vector has unit magnitude per component: (cos θ, sin θ).
The conjugate (= inverse for unit-magnitude) is (cos(-θ), sin(-θ)).
bind(V, conjugate(V)) = identity (all phases ≈ 0).
"""

from typing import Any, List

import numpy as np


def bind_phases(phases_a: List[float], phases_b: List[float]) -> List[float]:
    """Compute the phase array of bind(A, B) = element-wise complex multiply.

    In FHRR: θ_result[d] = θ_A[d] + θ_B[d]
    """
    return [a + b for a, b in zip(phases_a, phases_b)]


def bundle_phases(phase_arrays: List[List[float]]) -> List[float]:
    """Compute the phase array of bundle({V₁, ..., Vₙ}) = sum + normalize.

    In FHRR: θ_result[d] = atan2(Σᵢ sin(θᵢ[d]), Σᵢ cos(θᵢ[d]))

    This is exact (not approximate) because the arena's normalize step
    projects each component back to unit magnitude, which is equivalent
    to taking the angle of the complex sum.
    """
    if not phase_arrays:
        return []
    dim = len(phase_arrays[0])
    cos_sum = np.zeros(dim)
    sin_sum = np.zeros(dim)
    for phases in phase_arrays:
        arr = np.asarray(phases)
        cos_sum += np.cos(arr)
        sin_sum += np.sin(arr)
    return np.arctan2(sin_sum, cos_sum).tolist()


def negate_phases(phases: List[float]) -> List[float]:
    """Compute the conjugate phases: -θ per dimension.

    bind(V, conjugate(V)) produces the identity vector (all phases ≈ 0).
    """
    return [-p for p in phases]


def conjugate_into(arena: Any, phases: List[float], dst_handle: int):
    """Inject the complex conjugate of a phase vector into an arena slot.

    The conjugate of FHRR vector with phases [θ₁, θ₂, ...] is [-θ₁, -θ₂, ...].
    This is the multiplicative inverse for unit-magnitude FHRR vectors:
        bind(V, conjugate(V)) ≈ identity (phases ≈ 0, i.e., all-real ≈ 1+0j)
    """
    arena.inject_phases(dst_handle, negate_phases(phases))


# ── Meta-Grammar Emergence: Fusion Operator ──────────────────────

def bind_bundle_fusion_phases(
    phases_bind: List[float], phase_arrays_bundle: List[List[float]]
) -> List[float]:
    """Phase-domain fusion: fuse(A, {B₁,...,Bₖ}) = A ⊗ (B₁ ⊕ ... ⊕ Bₖ).

    Computes bundle of phase_arrays_bundle, then binds with phases_bind.
    This is the new algebraic grammar rule synthesized from ⊗ and ⊕.
    """
    bundled = bundle_phases(phase_arrays_bundle)
    return bind_phases(phases_bind, bundled)


# ── Meta-Grammar Emergence: Dimension Expansion ─────────────────

def expand_phases(phases: List[float], new_dim: int) -> List[float]:
    """Expand a phase vector from dim d to new_dim by conjugate reflection.

    For j in [d, new_dim): new_phase[j] = -phases[j % d]
    This mirrors the Rust arena's expand_dimension logic exactly.
    """
    old_dim = len(phases)
    if new_dim <= old_dim:
        return phases[:]
    extended = phases[:]
    for j in range(old_dim, new_dim):
        extended.append(-phases[j % old_dim])
    return extended


# ── Topological Thermodynamics: Entropy Computation ──────────────

def weighted_bundle_phases(
    phase_arrays: List[List[float]],
    weights: List[float],
) -> List[float]:
    """Weighted bundle: each member's contribution is scaled by its weight.

    In FHRR, weighting is applied to the complex components before summation:
      centroid[d] = atan2(
          Σ_i w_i * sin(θ_i[d]),
          Σ_i w_i * cos(θ_i[d])
      )

    This is the correct algebraic generalization of bundle (⊕) to non-uniform
    superposition. When all weights are equal, this reduces to standard bundle.

    Higher-weight members pull the centroid phase angle toward their direction
    more strongly. After normalization, each component returns to unit magnitude
    (preserving the FHRR invariant), but the ANGLE is biased toward
    high-resonance members.

    Args:
        phase_arrays: List of phase vectors to bundle.
        weights: Parallel list of non-negative weights (one per vector).
                 Need not sum to 1 — only relative magnitudes matter.

    Returns:
        Weighted centroid phase array.
    """
    if not phase_arrays:
        return []
    if len(phase_arrays) != len(weights):
        raise ValueError("phase_arrays and weights must have equal length")
    dim = len(phase_arrays[0])
    cos_sum = np.zeros(dim)
    sin_sum = np.zeros(dim)
    for phases, w in zip(phase_arrays, weights):
        arr = np.asarray(phases)
        cos_sum += w * np.cos(arr)
        sin_sum += w * np.sin(arr)
    return np.arctan2(sin_sum, cos_sum).tolist()


def compute_phase_entropy(phases: List[float]) -> float:
    """Compute the structural entropy of a phase vector.

    Uses the circular variance as a deterministic entropy measure:
      S = -ln(R) where R = |mean(e^{iθ})|

    R = 1 → all phases aligned → zero entropy (maximally compressed)
    R → 0 → phases uniformly spread → maximum entropy (maximally complex)

    This is O(d) and deterministic.
    """
    arr = np.asarray(phases)
    mean_re = float(np.mean(np.cos(arr)))
    mean_im = float(np.mean(np.sin(arr)))
    R = np.sqrt(mean_re ** 2 + mean_im ** 2)
    if R < 1e-12:
        return float(np.log(len(phases)))  # Maximum entropy
    return float(-np.log(R))


def compute_operation_entropy(bind_count: int, bundle_fan_in: int) -> float:
    """Compute the thermodynamic entropy cost of VSA operations.

    S_ops = bind_count × ln(2) + bundle_fan_in × ln(2)

    Each bind doubles the algebraic complexity (ln(2) per bind).
    Each bundle fan-in adds ln(2) per superposed vector.
    """
    ln2 = float(np.log(2.0))
    return bind_count * ln2 + bundle_fan_in * ln2
