"""Planner – lookahead over world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from cognitive_core_engine.core.world_model import WorldModel


@dataclass
class PlanCandidate:
    actions: List[str]
    score: float


class Planner:
    def __init__(self, wm: WorldModel, depth: int = 3,
                 width: int = 6, gamma: float = 0.9) -> None:
        self.wm = wm
        self.depth = depth
        self.width = width
        self.gamma = gamma

    def propose(self, obs: Dict[str, Any], action_space: List[str],
                risk_pref: float) -> List[PlanCandidate]:
        # Robustness: Safety check
        if not action_space:
            return []

        beam: List[PlanCandidate] = [PlanCandidate(actions=[], score=0.0)]

        for d in range(self.depth):
            new_beam: List[PlanCandidate] = []
            for cand in beam:
                for a in action_space:
                    q = self.wm.q_value(obs, a)
                    uncertainty = 1.0 - self.wm.confidence(obs, a)
                    adjusted = q - (1.0 - risk_pref) * uncertainty
                    sc = cand.score + (self.gamma ** d) * adjusted
                    new_beam.append(PlanCandidate(actions=cand.actions + [a], score=sc))

            # Robustness: Sort and Prune
            if not new_beam:
                break

            new_beam.sort(key=lambda c: c.score, reverse=True)
            beam = new_beam[: self.width]

        return beam
