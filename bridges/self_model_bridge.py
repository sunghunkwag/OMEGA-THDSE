"""Gap 9 — Self-Referential Model bridge (PLAN.md Phase 4).

CCE's :class:`SelfReferentialModel` carries four 10,000-dim FHRR
vectors representing the agent's belief, goal, capability, and emotion
states. THDSE has no native pathway to read these — the synthesizer
runs in 256-dim and cannot lift the 10k vectors directly. This bridge
projects each component down through
:func:`shared.dimension_bridge.project_down`, bundles the four
projected vectors into a single 256-dim self-model summary, and
exposes wireheading-detection + drift-tracking helpers that combine
THDSE fitness deltas with the projected self-model.

Three immutable thresholds govern the bridge:

- ``SELF_MODEL_COMPONENTS = 4`` — must equal the number of bundled
  components, otherwise we'd silently drop or duplicate state.
- ``WIREHEADING_THRESHOLD`` — fitness self-similarity above this
  triggers a wireheading flag.
- ``CONTINUITY_THRESHOLD`` — drift below this counts as identity
  continuity preserved.
- ``MAX_CREDIBLE_LEAP`` — fitness deltas above this are flagged as
  suspicious for further audit.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List

from shared.arena_manager import ArenaManager
from shared.constants import (
    CCE_ARENA_DIM,
    CONTINUITY_THRESHOLD,
    MAX_CREDIBLE_LEAP,
    SELF_MODEL_COMPONENTS,
    THDSE_ARENA_DIM,
    WIREHEADING_THRESHOLD,
)
from shared.dimension_bridge import cross_arena_similarity, project_down
from shared.exceptions import DimensionMismatchError

_TWO_PI = 2.0 * math.pi


def _validate_cce_vector(vec, *, name: str) -> List[float]:
    try:
        as_list = list(vec)
    except TypeError as exc:
        raise DimensionMismatchError(
            f"{name} not iterable: {exc}",
            operation="self_model_export",
        ) from exc
    if len(as_list) != CCE_ARENA_DIM:
        raise DimensionMismatchError(
            f"{name} must be {CCE_ARENA_DIM}-dim",
            expected=(CCE_ARENA_DIM,),
            actual=(len(as_list),),
            operation="self_model_export",
        )
    return as_list


def _bundle_256(vectors: List[List[float]]) -> List[float]:
    """Circular-mean bundle of 256-dim phase vectors."""
    if not vectors:
        return [0.0] * THDSE_ARENA_DIM
    bundled: List[float] = []
    for i in range(THDSE_ARENA_DIM):
        sin_sum = 0.0
        cos_sum = 0.0
        for vec in vectors:
            sin_sum += math.sin(float(vec[i]))
            cos_sum += math.cos(float(vec[i]))
        angle = math.atan2(sin_sum, cos_sum)
        if angle < 0.0:
            angle += _TWO_PI
        bundled.append(angle)
    return bundled


class SelfModelBridge:
    """Project the 4-component CCE self-model into 256-dim THDSE space."""

    def __init__(self, arena_manager: ArenaManager):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._exports: int = 0
        self._wireheading_flags: int = 0
        self._drift_observations: List[float] = []

    # ---- export ---- #

    def export_self_model_state(
        self,
        belief_vec,
        goal_vec,
        capability_vec,
        emotion_vec,
    ) -> Dict[str, Any]:
        """Project + bundle the four self-model components into 256-dim space."""
        components = {
            "belief": _validate_cce_vector(belief_vec, name="belief_vec"),
            "goal": _validate_cce_vector(goal_vec, name="goal_vec"),
            "capability": _validate_cce_vector(
                capability_vec, name="capability_vec"
            ),
            "emotion": _validate_cce_vector(emotion_vec, name="emotion_vec"),
        }
        if len(components) != SELF_MODEL_COMPONENTS:
            raise ValueError(
                f"expected exactly {SELF_MODEL_COMPONENTS} components, "
                f"got {len(components)}"
            )

        component_projections: Dict[str, Any] = {}
        projected_lists: List[List[float]] = []
        for name, vec in components.items():
            projection = project_down(vec)
            arr = projection["vector"]
            component_projections[name] = arr
            projected_lists.append(list(arr))

        bundled = _bundle_256(projected_lists)
        self._exports += 1
        timestamp = time.time()
        return {
            "thdse_self_model": bundled,
            "component_projections": component_projections,
            "metadata": {
                "component_count": SELF_MODEL_COMPONENTS,
                "component_names": list(components.keys()),
                "export_index": self._exports,
                "provenance": {
                    "operation": "self_model_export",
                    "source_arena": "cce",
                    "target_arena": "thdse",
                    "timestamp": timestamp,
                    "source_dim": CCE_ARENA_DIM,
                    "target_dim": THDSE_ARENA_DIM,
                },
            },
        }

    # ---- wireheading detection ---- #

    def detect_wireheading_from_thdse(
        self, proposed_delta: float, thdse_fitness_delta: float
    ) -> Dict[str, Any]:
        """Cross-check a CCE-side proposed self-model delta against THDSE fitness.

        Suspicion criteria:
        1. Either delta exceeds ``MAX_CREDIBLE_LEAP`` in magnitude.
        2. The two deltas have opposite signs while their magnitudes
           are non-trivial (CCE is claiming improvement while THDSE
           reports decline, or vice versa).
        """
        if not isinstance(proposed_delta, (int, float)):
            raise TypeError(
                f"proposed_delta must be numeric, got "
                f"{type(proposed_delta).__name__}"
            )
        if not isinstance(thdse_fitness_delta, (int, float)):
            raise TypeError(
                f"thdse_fitness_delta must be numeric, got "
                f"{type(thdse_fitness_delta).__name__}"
            )
        cce = float(proposed_delta)
        thd = float(thdse_fitness_delta)
        reasons: List[str] = []
        if abs(cce) > MAX_CREDIBLE_LEAP:
            reasons.append(
                f"CCE delta {cce:+.3f} exceeds MAX_CREDIBLE_LEAP "
                f"{MAX_CREDIBLE_LEAP}"
            )
        if abs(thd) > MAX_CREDIBLE_LEAP:
            reasons.append(
                f"THDSE delta {thd:+.3f} exceeds MAX_CREDIBLE_LEAP "
                f"{MAX_CREDIBLE_LEAP}"
            )
        sign_clash = (cce > 0.05 and thd < -0.05) or (cce < -0.05 and thd > 0.05)
        if sign_clash:
            reasons.append(
                f"sign clash: CCE delta {cce:+.3f} vs THDSE delta {thd:+.3f}"
            )
        is_suspicious = bool(reasons)
        if is_suspicious:
            self._wireheading_flags += 1
        timestamp = time.time()
        return {
            "is_suspicious": is_suspicious,
            "reason": "; ".join(reasons) if reasons else "no anomaly detected",
            "metadata": {
                "proposed_delta": cce,
                "thdse_fitness_delta": thd,
                "max_credible_leap": MAX_CREDIBLE_LEAP,
                "wireheading_threshold": WIREHEADING_THRESHOLD,
                "flag_count": self._wireheading_flags,
                "provenance": {
                    "operation": "wireheading_detection",
                    "source_arena": "both",
                    "timestamp": timestamp,
                },
            },
        }

    # ---- drift tracking ---- #

    def compute_self_model_drift(
        self, previous_export: Dict[str, Any], current_export: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Quantify drift between two self-model exports."""
        if not isinstance(previous_export, dict) or not isinstance(
            current_export, dict
        ):
            raise TypeError("both exports must be dicts")
        if "thdse_self_model" not in previous_export:
            raise KeyError("previous_export missing 'thdse_self_model'")
        if "thdse_self_model" not in current_export:
            raise KeyError("current_export missing 'thdse_self_model'")

        prev = list(previous_export["thdse_self_model"])
        curr = list(current_export["thdse_self_model"])
        if len(prev) != THDSE_ARENA_DIM or len(curr) != THDSE_ARENA_DIM:
            raise DimensionMismatchError(
                "self_model vectors must be 256-dim",
                expected=(THDSE_ARENA_DIM,),
                actual=(len(prev), len(curr)),
                operation="compute_self_model_drift",
            )

        # Compare in 256-dim space using mean cosine of phase
        # differences (FHRR similarity). Drift is 1 - similarity so it
        # increases as the model changes.
        mean_cos = sum(
            math.cos(float(prev[i]) - float(curr[i]))
            for i in range(THDSE_ARENA_DIM)
        ) / THDSE_ARENA_DIM
        similarity = float(mean_cos)
        drift_score = 1.0 - similarity

        # Route through the bridge primitive too so the result carries
        # cross_arena_similarity provenance for downstream auditing.
        zero_cce = [0.0] * CCE_ARENA_DIM
        sim_record = cross_arena_similarity(zero_cce, curr)

        if drift_score >= 1.0 - WIREHEADING_THRESHOLD:
            severity = "wireheading_suspect"
        elif similarity >= CONTINUITY_THRESHOLD:
            severity = "continuous"
        elif drift_score < 0.5:
            severity = "minor"
        else:
            severity = "major"

        self._drift_observations.append(drift_score)
        timestamp = time.time()
        return {
            "drift_score": drift_score,
            "similarity": similarity,
            "severity": severity,
            "metadata": {
                "wireheading_threshold": WIREHEADING_THRESHOLD,
                "continuity_threshold": CONTINUITY_THRESHOLD,
                "history_length": len(self._drift_observations),
                "underlying_similarity_provenance": sim_record["metadata"][
                    "provenance"
                ],
                "provenance": {
                    "operation": "compute_self_model_drift",
                    "source_arena": "both",
                    "timestamp": timestamp,
                },
            },
        }

    # ---- aggregate audit helpers ---- #

    def summarize_drift_history(self) -> Dict[str, Any]:
        """Return aggregate stats over the bridge's lifetime drift observations."""
        if not self._drift_observations:
            mean = 0.0
            max_drift = 0.0
            min_drift = 0.0
        else:
            mean = sum(self._drift_observations) / len(self._drift_observations)
            max_drift = max(self._drift_observations)
            min_drift = min(self._drift_observations)
        return {
            "mean_drift": mean,
            "max_drift": max_drift,
            "min_drift": min_drift,
            "sample_count": len(self._drift_observations),
            "metadata": {
                "wireheading_flag_count": self._wireheading_flags,
                "export_count": self._exports,
                "provenance": {
                    "operation": "summarize_drift_history",
                    "source_arena": "both",
                    "timestamp": time.time(),
                },
            },
        }

    def reset_drift_history(self) -> Dict[str, Any]:
        """Drop the drift observation buffer."""
        cleared = len(self._drift_observations)
        self._drift_observations.clear()
        return {
            "cleared": cleared,
            "metadata": {
                "provenance": {
                    "operation": "reset_drift_history",
                    "source_arena": "cce",
                    "timestamp": time.time(),
                }
            },
        }

    # ---- introspection ---- #

    @property
    def export_count(self) -> int:
        return self._exports

    @property
    def wireheading_flag_count(self) -> int:
        return self._wireheading_flags

    @property
    def drift_observation_count(self) -> int:
        return len(self._drift_observations)


__all__ = ["SelfModelBridge"]
