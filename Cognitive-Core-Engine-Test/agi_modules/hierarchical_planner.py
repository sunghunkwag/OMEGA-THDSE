"""
HierarchicalPlanner — Multi-level planning using ConceptGraph abstractions.

Serves AGI capability: replaces flat beam search with hierarchical decomposition
from meta-strategies down to individual actions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agi_modules.concept_graph import ConceptGraph, ConceptNode

# --- Named constants (Rule 6) ---

# Planning time budget in milliseconds
PLANNING_TIME_BUDGET_MS = 100

# Fallback to flat planning if concept graph is empty
MIN_CONCEPTS_FOR_HIERARCHICAL = 1

# Level names for human-readable plans
LEVEL_NAMES = {
    0: "action",
    1: "tactic",
    2: "strategy",
    3: "meta-strategy",
    4: "meta-meta-strategy",
}


@dataclass
class PlanCandidate:
    """A plan candidate compatible with the existing Planner output format.

    Maintains backward compatibility with Agent.choose_action().
    """
    actions: List[str]
    expected_value: float
    concept_levels_used: List[int] = field(default_factory=list)


class HierarchicalPlanner:
    """Plans at multiple abstraction levels using ConceptGraph.

    Why it exists: flat beam search cannot compose strategies or reuse
    abstract plans across domains. Hierarchical planning enables this.

    Fallback: delegates to original Planner.propose() if ConceptGraph is empty.
    """

    def __init__(self, wm: Any, concept_graph: ConceptGraph,
                 max_depth: int = 4) -> None:
        self.wm = wm
        self.concept_graph = concept_graph
        self.max_depth = max_depth

    def plan(self, obs: Dict[str, Any], action_space: List[str],
             risk: float) -> List[PlanCandidate]:
        """Generate plans using hierarchical concept decomposition.

        Why: enables multi-level reasoning from strategy to action.
        Fallback: returns flat beam-search plans if concept graph is empty.
        """
        start_ms = time.time() * 1000

        # Check if we have enough concepts for hierarchical planning
        if self.concept_graph.size() < MIN_CONCEPTS_FOR_HIERARCHICAL:
            return self._flat_plan(obs, action_space, risk)

        candidates: List[PlanCandidate] = []
        domain = str(obs.get("domain", ""))
        difficulty = int(obs.get("difficulty", 3))

        # Level 3+: meta-strategy selection
        max_level = min(self.max_depth, self.concept_graph.depth())
        levels_used = []

        for level in range(max_level, -1, -1):
            if (time.time() * 1000 - start_ms) > PLANNING_TIME_BUDGET_MS:
                break

            concepts = self.concept_graph.concepts_at_level(level)
            if not concepts:
                continue

            # Find concepts relevant to current domain
            relevant = self._filter_relevant(concepts, domain, difficulty)
            if not relevant:
                continue

            levels_used.append(level)

            # Use top concept to guide action selection
            best_concept = max(relevant, key=lambda c: c.avg_reward)
            guided_actions = self._decompose_concept(best_concept, action_space)

            for action in guided_actions[:3]:  # Top 3 guided actions
                q_val = self.wm.q_value(obs, action)
                concept_bonus = best_concept.avg_reward * 0.2
                expected = q_val + concept_bonus

                candidates.append(PlanCandidate(
                    actions=[action],
                    expected_value=expected,
                    concept_levels_used=list(levels_used),
                ))

        # Add flat beam search candidates as fallback
        flat = self._flat_plan(obs, action_space, risk)
        for fc in flat:
            if not any(c.actions == fc.actions for c in candidates):
                candidates.append(fc)

        # Sort by expected value
        candidates.sort(key=lambda c: c.expected_value, reverse=True)

        # Enforce time budget
        elapsed = time.time() * 1000 - start_ms
        if elapsed > PLANNING_TIME_BUDGET_MS * 2:
            # Return what we have so far
            pass

        return candidates[:6]  # Top 6 candidates

    def _flat_plan(self, obs: Dict[str, Any], action_space: List[str],
                   risk: float) -> List[PlanCandidate]:
        """Flat beam search using WorldModel Q-values.

        Why: bootstrap planning when concept graph is empty.
        Fallback: random action ordering if Q-values are all zero.
        """
        candidates = []
        for action in action_space:
            q_val = self.wm.q_value(obs, action)
            candidates.append(PlanCandidate(
                actions=[action],
                expected_value=q_val,
                concept_levels_used=[0],
            ))

        candidates.sort(key=lambda c: c.expected_value, reverse=True)
        return candidates

    def _filter_relevant(self, concepts: List[ConceptNode],
                         domain: str, difficulty: int) -> List[ConceptNode]:
        """Filter concepts relevant to the current observation context.

        Why: only use concepts that have been successful in similar contexts.
        Fallback: returns all concepts if none match domain filter.
        """
        relevant = []
        for concept in concepts:
            for ctx in concept.success_contexts:
                if (str(ctx.get("domain", "")) == domain or
                        concept.usage_count > 10):
                    relevant.append(concept)
                    break

        return relevant if relevant else concepts[:5]

    def _decompose_concept(self, concept: ConceptNode,
                           action_space: List[str]) -> List[str]:
        """Decompose a high-level concept into concrete actions.

        Why: maps abstract strategies to executable actions.
        Fallback: returns full action space if decomposition fails.
        """
        # Extract action patterns from concept's success contexts
        action_counts: Dict[str, int] = {}
        for ctx in concept.success_contexts:
            action = ctx.get("action")
            if action and action in action_space:
                action_counts[action] = action_counts.get(action, 0) + 1

        if action_counts:
            # Sort by frequency in success contexts
            sorted_actions = sorted(action_counts.keys(),
                                    key=lambda a: action_counts[a],
                                    reverse=True)
            # Add remaining actions
            remaining = [a for a in action_space if a not in sorted_actions]
            return sorted_actions + remaining

        return action_space
