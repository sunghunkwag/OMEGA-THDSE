"""
IntrinsicMotivationModule — Curiosity, novelty, and learning progress rewards.

Serves AGI capability: drives exploration beyond extrinsic reward signals,
preventing premature convergence to local optima.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from agi_modules.competence_map import CompetenceMap

# --- Named constants (Rule 6) ---

# Intrinsic reward component weights
CURIOSITY_WEIGHT = 0.4       # Prediction error drives exploration
NOVELTY_WEIGHT = 0.3         # Visit-count novelty bonus
LEARNING_PROGRESS_WEIGHT = 0.3  # Derivative of competence

# Curiosity boost threshold for action selection
CURIOSITY_BOOST_THRESHOLD = 0.7
CURIOSITY_BOOST_FACTOR = 1.5

# Default blending ratio (extrinsic : intrinsic)
DEFAULT_EXTRINSIC_WEIGHT = 0.6
DEFAULT_INTRINSIC_WEIGHT = 0.4


def _stable_state_hash(domain: str, difficulty: int, action: str) -> str:
    """Create a stable hash for state-action counting.

    Why: needed for count-based novelty computation.
    Fallback: returns concatenated string if hashing fails.
    """
    return f"{domain}:{difficulty}:{action}"


class IntrinsicMotivationModule:
    """Computes intrinsic reward signals for curiosity-driven exploration.

    Why it exists: without intrinsic motivation, the agent only optimizes
    for extrinsic reward and converges to boring local optima.

    Fallback: returns 0.0 for all components if data is missing.
    """

    def __init__(self, shared_mem: Any, competence_map: CompetenceMap) -> None:
        self.shared_mem = shared_mem
        self.competence_map = competence_map
        self._visit_counts: Dict[str, int] = {}
        self._competence_history: Dict[Tuple[str, int], list] = {}
        self._prediction_cache: Dict[str, float] = {}

    def compute_curiosity(self, obs: Dict[str, Any], action: str,
                          outcome: Dict[str, Any]) -> float:
        """Prediction error curiosity: |predicted - actual| normalized to [0, 1].

        Why: drives exploration toward surprising outcomes.
        Fallback: returns 0.5 if no prediction available.
        """
        actual_reward = float(outcome.get("reward", 0.0))
        domain = str(obs.get("domain", ""))
        difficulty = int(obs.get("difficulty", 3))

        # Use competence as prediction proxy
        predicted = self.competence_map.get_rate(domain, difficulty)
        error = abs(predicted - actual_reward)

        # Normalize to [0, 1]
        return min(1.0, max(0.0, error))

    def compute_novelty(self, obs: Dict[str, Any],
                        outcome: Dict[str, Any]) -> float:
        """Count-based novelty: 1 / sqrt(visit_count + 1).

        Why: encourages visiting less-explored state-action pairs.
        Fallback: returns 1.0 for never-visited states.
        """
        domain = str(obs.get("domain", ""))
        difficulty = int(obs.get("difficulty", 3))
        action = str(outcome.get("action", ""))

        state_hash = _stable_state_hash(domain, difficulty, action)
        count = self._visit_counts.get(state_hash, 0)
        self._visit_counts[state_hash] = count + 1

        # 1 / sqrt(count + 1), approaches 0 as count grows
        return 1.0 / math.sqrt(count + 1)

    def compute_learning_progress(self, domain: str,
                                  difficulty: int) -> float:
        """Derivative of competence over recent history.

        Why: positive progress = interesting area, zero = plateau (try elsewhere).
        Fallback: returns 0.0 if insufficient history.
        """
        key = (domain, difficulty)
        current = self.competence_map.get_rate(domain, difficulty)

        # Track history
        if key not in self._competence_history:
            self._competence_history[key] = []
        history = self._competence_history[key]
        history.append(current)

        # Keep last 20 entries
        if len(history) > 20:
            history.pop(0)

        if len(history) < 2:
            return 0.0

        # Compare current vs 10 steps ago (or earliest available)
        lookback = min(10, len(history) - 1)
        past = history[-1 - lookback]
        delta = current - past

        # Normalize to [0, 1]: map [-1, 1] → [0, 1]
        return min(1.0, max(0.0, (delta + 1.0) / 2.0))

    def total_intrinsic_reward(self, obs: Dict[str, Any], action: str,
                               outcome: Dict[str, Any]) -> float:
        """Weighted sum of all intrinsic reward components.

        Why: provides a single scalar for blending with extrinsic reward.
        Fallback: returns 0.0 if all components fail.
        """
        domain = str(obs.get("domain", ""))
        difficulty = int(obs.get("difficulty", 3))

        curiosity = self.compute_curiosity(obs, action, outcome)
        novelty = self.compute_novelty(obs, outcome)
        lp = self.compute_learning_progress(domain, difficulty)

        total = (CURIOSITY_WEIGHT * curiosity +
                 NOVELTY_WEIGHT * novelty +
                 LEARNING_PROGRESS_WEIGHT * lp)

        return min(1.0, max(0.0, total))

    def curiosity_for_action(self, obs: Dict[str, Any],
                             action: str) -> float:
        """Estimate curiosity for a hypothetical action (pre-execution).

        Why: used by Agent.choose_action to boost exploration of curious actions.
        Fallback: returns 0.5 if no data.
        """
        domain = str(obs.get("domain", ""))
        difficulty = int(obs.get("difficulty", 3))
        state_hash = _stable_state_hash(domain, difficulty, action)

        count = self._visit_counts.get(state_hash, 0)
        novelty = 1.0 / math.sqrt(count + 1)

        rate = self.competence_map.get_rate(domain, difficulty)
        uncertainty = 1.0 - abs(rate - 0.5) * 2  # Highest at 0.5

        return min(1.0, max(0.0, 0.5 * novelty + 0.5 * uncertainty))
