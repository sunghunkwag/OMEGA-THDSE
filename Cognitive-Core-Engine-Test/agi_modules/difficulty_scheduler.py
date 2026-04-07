"""
DifficultyScheduler — Curriculum learning via adaptive difficulty management.

Serves AGI capability: prevents permanent stagnation by dynamically adjusting
task difficulty based on competence and learning velocity.
"""

from __future__ import annotations

import random as stdlib_random
from typing import Dict, Optional

from agi_modules.competence_map import CompetenceMap

# --- Named constants (Rule 6) ---

# Difficulty adjustment thresholds
INCREASE_COMPETENCE = 0.80   # Above this: increase difficulty
DECREASE_COMPETENCE = 0.20   # Below this for N rounds: decrease difficulty
DECREASE_ROUNDS = 5          # Rounds below threshold before decrease

# Stagnation escape: random perturbation after this many stagnant rounds
STAGNATION_ROUNDS = 10
STAGNATION_PERTURBATION_RANGE = 2  # ±2 difficulty levels

# Difficulty bounds
MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 10

# Chaos injection probability
DEFAULT_CHAOS_PROBABILITY = 0.1
CHAOS_PERTURBATION_MAX = 3


class DifficultyScheduler:
    """Manages adaptive difficulty per domain based on competence trajectory.

    Why it exists: without curriculum learning, agents either get stuck on
    easy tasks (no learning) or impossible tasks (wasted compute).

    Fallback: returns current difficulty if insufficient data.
    """

    def __init__(self, competence_map: CompetenceMap,
                 rng: Optional[stdlib_random.Random] = None) -> None:
        self.competence_map = competence_map
        self.rng = rng or stdlib_random.Random(42)
        self._current_difficulties: Dict[str, int] = {}
        self._rounds_below: Dict[str, int] = {}
        self._rounds_stagnant: Dict[str, int] = {}
        self._last_rates: Dict[str, float] = {}
        self._chaos_fired_count: int = 0
        self._total_calls: int = 0

    def schedule(self, round_idx: int) -> Dict[str, int]:
        """Return target difficulty per domain based on competence analysis.

        Why: implements curriculum learning — progressively harder tasks.
        Fallback: returns difficulty 3 for unknown domains.
        """
        self._total_calls += 1
        domains = self.competence_map.all_domains()
        result: Dict[str, int] = {}

        for domain in domains:
            current_diff = self._current_difficulties.get(domain, 3)
            rate = self.competence_map.get_rate(domain, current_diff)

            # Track stagnation
            last_rate = self._last_rates.get(domain, rate)
            if abs(rate - last_rate) < 0.01:
                self._rounds_stagnant[domain] = (
                    self._rounds_stagnant.get(domain, 0) + 1
                )
            else:
                self._rounds_stagnant[domain] = 0
            self._last_rates[domain] = rate

            # Rule: mastered → increase difficulty
            if rate > INCREASE_COMPETENCE:
                current_diff = min(MAX_DIFFICULTY, current_diff + 1)
                self._rounds_below[domain] = 0

            # Rule: struggling for too long → decrease difficulty
            elif rate < DECREASE_COMPETENCE:
                self._rounds_below[domain] = (
                    self._rounds_below.get(domain, 0) + 1
                )
                if self._rounds_below[domain] >= DECREASE_ROUNDS:
                    current_diff = max(MIN_DIFFICULTY, current_diff - 1)
                    self._rounds_below[domain] = 0
            else:
                self._rounds_below[domain] = 0

            # Rule: stagnation escape via random perturbation
            stagnant = self._rounds_stagnant.get(domain, 0)
            if stagnant >= STAGNATION_ROUNDS:
                perturbation = self.rng.randint(
                    -STAGNATION_PERTURBATION_RANGE,
                    STAGNATION_PERTURBATION_RANGE
                )
                current_diff = max(MIN_DIFFICULTY,
                                   min(MAX_DIFFICULTY, current_diff + perturbation))
                self._rounds_stagnant[domain] = 0

            # Enforce bounds
            current_diff = max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, current_diff))
            self._current_difficulties[domain] = current_diff
            result[domain] = current_diff

        return result

    def inject_chaos(self, probability: float = DEFAULT_CHAOS_PROBABILITY) -> bool:
        """Randomly perturb all difficulties to prevent comfort plateau.

        Why: chaos injection escapes local optima in difficulty space.
        Fallback: no-op if probability is 0.
        """
        if self.rng.random() >= probability:
            return False

        self._chaos_fired_count += 1
        for domain in list(self._current_difficulties):
            perturbation = self.rng.randint(-CHAOS_PERTURBATION_MAX,
                                            CHAOS_PERTURBATION_MAX)
            current = self._current_difficulties[domain]
            new_diff = max(MIN_DIFFICULTY,
                           min(MAX_DIFFICULTY, current + perturbation))
            self._current_difficulties[domain] = new_diff

        return True

    def get_difficulty(self, domain: str) -> int:
        """Get current difficulty for a domain.

        Why: needed by environment and goal generator.
        Fallback: returns 3 for unknown domains.
        """
        return self._current_difficulties.get(domain, 3)

    def chaos_fired_count(self) -> int:
        """Return how many times chaos was injected.

        Why: validation that chaos fires approximately 10% of the time.
        Fallback: returns 0.
        """
        return self._chaos_fired_count

    def total_calls(self) -> int:
        """Return total schedule() calls.

        Why: needed for chaos rate computation.
        Fallback: returns 0.
        """
        return self._total_calls
