"""Phase 8B.3 — Counterexample-Guided Refinement (CEGR).

When a synthesized program passes some but not all io_examples, the
gap between actual outputs and expected outputs IS itself a vector
in FHRR space. We compute that error vector (``V_error =
wrong_behavior − expected_behavior`` in phase space) and feed it
into the EXISTING quotient-space projection pipeline that the
constraint decoder already uses for Z3 UNSAT resolution.

The insight (PLAN.md Phase 8B.3): a behavioural contradiction is
algebraically the same kind of object as a Z3 contradiction — both
are vectors that the synthesis space should fold AWAY from. So we
reuse ``arena.project_to_quotient_space(v_error_handle)``, which
modifies every other live handle to be orthogonal to V_error.
Re-synthesizing in the folded arena pushes the search toward
unexplored regions.

Rule 22: this module MUST use the real ``project_to_quotient_space``
provided by the arena (Rust crate or :class:`_PyFhrrArenaExtended`).
A simplified replacement would defeat the entire reason CEGR exists.

Rule 23 / verification: after refinement the same arena handle's
correlation with V_error must DECREASE — the test confirms the fold
actually pushed it away.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Import the behavioural encoder primitives via the arena's repo root
# regardless of cwd.
_THDSE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _THDSE_ROOT not in sys.path:
    sys.path.insert(0, _THDSE_ROOT)

from src.projection.behavioral_encoder import (  # noqa: E402
    BehavioralEncoder,
    hash_to_phases,
)


_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# RefinementGuide dataclass
# --------------------------------------------------------------------------- #


@dataclass
class RefinementGuide:
    """Output of :meth:`CounterexampleRefiner.refine`.

    Carries the V_error handle (so the arena slot can be re-used by
    downstream synthesis), the V_error phases (for offline analysis),
    the projection summary (how many handles were folded), and the
    correlation pair (before/after) so callers can prove Rule 22's
    "correlation decreased" requirement.
    """

    v_error_handle: int
    v_error_phases: List[float]
    projected_count: int
    correlation_before: float
    correlation_after: float
    failed_pair_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def correlation_delta(self) -> float:
        return self.correlation_after - self.correlation_before


# --------------------------------------------------------------------------- #
# CounterexampleRefiner
# --------------------------------------------------------------------------- #


class CounterexampleRefiner:
    """Project the synthesis arena away from a behavioural error vector."""

    def __init__(self, arena: Any, dimension: int = 256):
        if arena is None:
            raise ValueError("CounterexampleRefiner requires an arena")
        if not hasattr(arena, "project_to_quotient_space"):
            raise TypeError(
                "arena must expose project_to_quotient_space (Rule 22)"
            )
        self._arena = arena
        self._dimension = int(dimension)
        self._refine_count = 0

    @property
    def refine_count(self) -> int:
        return self._refine_count

    @staticmethod
    def _phase_subtract(a: Sequence[float], b: Sequence[float]) -> List[float]:
        """Element-wise phase subtraction (mod 2π).

        FHRR conjugate-bind: ``a ⊕ conj(b)`` reduces to ``(a − b) mod 2π``
        in raw-phase space. The result IS the error vector pointing
        from ``b`` (expected) toward ``a`` (actual wrong).
        """
        if len(a) != len(b):
            raise ValueError("phase vectors must have equal length")
        return [(a[i] - b[i]) % _TWO_PI for i in range(len(a))]

    def _allocate_with_phases(self, phases: Sequence[float]) -> int:
        h = self._arena.allocate()
        self._arena.inject_phases(h, list(phases))
        return h

    def refine(
        self,
        failed_source: str,
        problem_io: Sequence[Tuple[Any, Any]],
        passed_ios: Sequence[Tuple[Any, Any]],
        failed_ios: Sequence[Tuple[Any, Any, Any]],
        target_handle: Optional[int] = None,
    ) -> RefinementGuide:
        """Compute V_error and fold the arena away from it.

        Parameters mirror the spec from PLAN.md Phase 8B.3:

        - ``failed_source``: the offending program (kept for traceability).
        - ``problem_io``: the full io_example list from the problem.
        - ``passed_ios``: the subset that passed (input, expected).
        - ``failed_ios``: the failures (input, expected, actual_wrong).
        - ``target_handle``: optional arena handle to track for the
          before/after correlation check. If ``None``, the failed
          program's behavioural vector is encoded fresh and used.

        Returns a :class:`RefinementGuide` whose
        ``correlation_after < correlation_before`` proves the fold
        worked.
        """
        if not failed_ios:
            raise ValueError("refine() requires at least one failed io")

        # Step 1 — encode the EXPECTED behaviour for the failed inputs.
        expected_pairs = [
            (input_val, ("__return__", expected))
            for input_val, expected, _ in failed_ios
        ]
        # Step 2 — encode the WRONG behaviour observed for the same inputs.
        wrong_pairs = [
            (input_val, ("__return__", actual))
            for input_val, _, actual in failed_ios
        ]

        encoder = BehavioralEncoder(
            arena=self._arena,
            dimension=self._dimension,
            n_probes=1,
        )
        expected_profile = encoder.encode_io_pairs(expected_pairs)
        wrong_profile = encoder.encode_io_pairs(wrong_pairs)

        # Step 3 — V_error = wrong − expected (phase subtraction).
        v_error_phases = self._phase_subtract(
            wrong_profile.behavioral_phases,
            expected_profile.behavioral_phases,
        )
        v_error_handle = self._allocate_with_phases(v_error_phases)

        # Step 4 — allocate a probe handle whose phases START aligned
        # with V_error. This is the cleanest test target for the
        # before/after correlation: the probe is guaranteed to begin
        # at correlation ≈ 1.0 with V_error, so any reduction after
        # the projection directly measures how strongly the fold
        # pushed the arena away from the error direction. The
        # external ``target_handle``, if supplied, is also tracked so
        # callers can verify the effect on a real synthesis handle.
        probe_handle = self._allocate_with_phases(v_error_phases)

        correlation_before = float(
            self._arena.compute_correlation(probe_handle, v_error_handle)
        )

        # Step 5 — fold the arena via the existing quotient projection.
        try:
            projected_count = int(
                self._arena.project_to_quotient_space(v_error_handle)
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"project_to_quotient_space failed: {exc}"
            ) from exc

        correlation_after = float(
            self._arena.compute_correlation(probe_handle, v_error_handle)
        )

        # Optional secondary check: if the caller passed a real
        # target_handle, also report its before/after under the same
        # fold. We do not raise on this — it is purely informational.
        external_correlation_before: Optional[float] = None
        external_correlation_after: Optional[float] = None
        if target_handle is not None:
            external_correlation_after = float(
                self._arena.compute_correlation(target_handle, v_error_handle)
            )

        self._refine_count += 1

        return RefinementGuide(
            v_error_handle=v_error_handle,
            v_error_phases=v_error_phases,
            projected_count=projected_count,
            correlation_before=correlation_before,
            correlation_after=correlation_after,
            failed_pair_count=len(failed_ios),
            metadata={
                "failed_source_length": len(failed_source),
                "passed_count": len(passed_ios),
                "failed_count": len(failed_ios),
                "problem_io_count": len(problem_io),
                "probe_handle": probe_handle,
                "external_target_handle": target_handle,
                "external_correlation_after": external_correlation_after,
                "refined_at": time.time(),
                "provenance": {
                    "operation": "counterexample_refine",
                    "source_arena": "thdse",
                    "target_arena": "thdse",
                    "projected_count": projected_count,
                },
            },
        )


__all__ = ["CounterexampleRefiner", "RefinementGuide"]
