"""Gap 8 — CCE GoalGenerator → THDSE Synthesis bridge.

The CCE agent's :class:`GoalGenerator` produces goals as natural
language descriptions plus an associated 10k-dim FHRR "goal vector"
describing the semantic target in concept space. The THDSE axiomatic
synthesizer, however, operates in 256-dim axiom space and needs a
concrete target handle to seed its search. This bridge projects goal
vectors down through :mod:`shared.dimension_bridge` and attaches a
priority-weighted ranking so the synthesizer can pick the highest
expected-value target first.

Nothing in this bridge calls CCE or THDSE directly — both sides of
the wire are represented as plain integer handles and phase arrays
owned by the shared :class:`ArenaManager`.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from shared.arena_manager import ArenaManager
from shared.constants import CCE_ARENA_DIM, THDSE_ARENA_DIM
from shared.dimension_bridge import cross_arena_similarity, project_down
from shared.exceptions import DimensionMismatchError


class GoalSynthesisBridge:
    """Translate CCE goals into THDSE synthesis targets."""

    def __init__(self, arena_manager: ArenaManager):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._projection_count = 0

    # ---- primary conversion ---- #

    def goal_to_synthesis_target(
        self,
        goal_description: str,
        goal_vector_handle: int,
        priority: float,
    ) -> dict[str, Any]:
        """Project a goal vector down and wrap it in a target spec.

        ``goal_description`` is a non-empty string carried into the
        target metadata for human traceability. ``goal_vector_handle``
        is an integer handle into the CCE arena. ``priority`` is a
        non-negative float; downstream ranking multiplies priority by
        projected similarity.
        """
        if not isinstance(goal_description, str) or not goal_description.strip():
            raise ValueError("goal_description must be a non-empty string")
        if not isinstance(goal_vector_handle, int) or goal_vector_handle < 0:
            raise TypeError(
                f"goal_vector_handle must be a non-negative int, "
                f"got {goal_vector_handle!r}"
            )
        if not isinstance(priority, (int, float)):
            raise TypeError(
                f"priority must be numeric, got {type(priority).__name__}"
            )
        priority_val = float(priority)
        if priority_val < 0.0:
            raise ValueError(f"priority must be >= 0, got {priority_val}")

        cce_phases = self._mgr.get_cce_phases(goal_vector_handle)
        if cce_phases.shape != (CCE_ARENA_DIM,):
            raise DimensionMismatchError(
                "goal vector has wrong shape",
                expected=(CCE_ARENA_DIM,),
                actual=tuple(cce_phases.shape),
                operation="goal_to_synthesis_target",
            )

        projection = project_down(cce_phases)
        projected_vector = projection["vector"]
        thdse_handle = self._mgr.alloc_thdse(phases=projected_vector)

        sim = cross_arena_similarity(cce_phases, projected_vector)
        projected_similarity = float(sim["similarity"])

        self._projection_count += 1
        target_id = self._compute_target_id(
            goal_description, goal_vector_handle, thdse_handle
        )
        timestamp = time.time()

        return {
            "thdse_target_handle": thdse_handle,
            "projected_similarity": projected_similarity,
            "priority": priority_val,
            "metadata": {
                "target_id": target_id,
                "goal_description": goal_description,
                "goal_vector_handle": goal_vector_handle,
                "expected_value": priority_val * projected_similarity,
                "projection_provenance": projection["metadata"]["provenance"],
                "cce_dim": CCE_ARENA_DIM,
                "thdse_dim": THDSE_ARENA_DIM,
                "projection_index": self._projection_count,
                "provenance": {
                    "operation": "goal_to_synthesis_target",
                    "source_arena": "cce",
                    "target_arena": "thdse",
                    "timestamp": timestamp,
                },
            },
        }

    # ---- ranking ---- #

    def rank_goals(self, goals: list[dict]) -> list[dict[str, Any]]:
        """Sort a list of goal-target dicts by descending expected value.

        Each input goal dict is expected to be the result of a prior
        :meth:`goal_to_synthesis_target` call (or any dict that has
        ``priority`` and ``projected_similarity`` keys). The returned
        list is a new list of dicts augmented with a ``rank`` key
        (1-indexed) and provenance metadata.
        """
        if not isinstance(goals, list):
            raise TypeError(
                f"goals must be a list, got {type(goals).__name__}"
            )

        scored: list[tuple[float, dict]] = []
        for idx, goal in enumerate(goals):
            if not isinstance(goal, dict):
                raise TypeError(
                    f"goals[{idx}] must be a dict, got {type(goal).__name__}"
                )
            if "priority" not in goal or "projected_similarity" not in goal:
                raise KeyError(
                    f"goals[{idx}] missing 'priority' or "
                    f"'projected_similarity'"
                )
            score = float(goal["priority"]) * float(
                goal["projected_similarity"]
            )
            scored.append((score, goal))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        ranked: list[dict[str, Any]] = []
        timestamp = time.time()
        for rank, (score, goal) in enumerate(scored, start=1):
            item = dict(goal)
            existing_meta = dict(item.get("metadata", {}))
            existing_meta["rank"] = rank
            existing_meta["rank_score"] = score
            # Preserve any existing provenance while adding the rank op.
            existing_meta["ranking_provenance"] = {
                "operation": "rank_goals",
                "source_arena": "cce",
                "target_arena": "thdse",
                "rank": rank,
                "score": score,
                "timestamp": timestamp,
            }
            # Ensure the outer provenance key still exists for callers
            # that only look at the top-level `metadata.provenance`.
            if "provenance" not in existing_meta:
                existing_meta["provenance"] = existing_meta[
                    "ranking_provenance"
                ]
            item["metadata"] = existing_meta
            item["rank"] = rank
            item["rank_score"] = score
            ranked.append(item)
        return ranked

    # ---- introspection ---- #

    @property
    def projection_count(self) -> int:
        return self._projection_count

    # ---- internals ---- #

    @staticmethod
    def _compute_target_id(
        description: str, cce_handle: int, thdse_handle: int
    ) -> str:
        payload = (
            f"{description[:64]}|cce:{int(cce_handle)}|"
            f"thdse:{int(thdse_handle)}"
        ).encode("utf-8")
        return f"goal-{hashlib.blake2b(payload, digest_size=8).hexdigest()}"


__all__ = ["GoalSynthesisBridge"]
