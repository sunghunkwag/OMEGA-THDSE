"""
CompetenceMap — Tracks agent competence across (domain, difficulty) pairs.

Serves AGI capability: Autonomous goal generation via zone-of-proximal-development
detection. Without this, the system cannot identify what it should learn next.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# --- Named constants (Rule 6) ---

# Zone of proximal development bounds: tasks in this range are learnable
ZPD_LOWER = 0.15  # Below this: incompetent, needs new capabilities
ZPD_UPPER = 0.70  # Above this: mastered, should move to harder tasks

# Mastery threshold: above this, the agent has reliably solved this level
# Calibrated for environment where normalized rewards converge around 0.5
MASTERY_THRESHOLD = 0.65

# Exponential moving average rate for competence updates
EMA_ALPHA = 0.1

# Default success rate for unseen (domain, difficulty) pairs
DEFAULT_SUCCESS_RATE = 0.5

# Novelty decay denominator: novelty = 1 - attempts / (attempts + this)
NOVELTY_HALF_LIFE = 10


class CompetenceMap:
    """Tracks running success rate per (domain, difficulty) pair.

    Why it exists: enables frontier-based goal generation by identifying
    the zone of proximal development — tasks that are neither too easy
    nor too hard for the current agent.

    Fallback: if no data, returns defaults that encourage exploration.
    """

    def __init__(self) -> None:
        self._rates: Dict[Tuple[str, int], float] = {}
        self._attempts: Dict[Tuple[str, int], int] = {}

    def update(self, domain: str, difficulty: int, reward: float) -> None:
        """Update running success rate with exponential moving average.

        Why: tracks competence trajectory per (domain, difficulty).
        Fallback: initializes from DEFAULT_SUCCESS_RATE on first call.
        """
        key = (domain, difficulty)
        prev = self._rates.get(key, DEFAULT_SUCCESS_RATE)
        self._rates[key] = (1 - EMA_ALPHA) * prev + EMA_ALPHA * reward
        self._attempts[key] = self._attempts.get(key, 0) + 1

    def get_rate(self, domain: str, difficulty: int) -> float:
        """Return current success rate for a (domain, difficulty) pair.

        Why: used by other modules to query competence.
        Fallback: returns DEFAULT_SUCCESS_RATE if never observed.
        """
        return self._rates.get((domain, difficulty), DEFAULT_SUCCESS_RATE)

    def get_attempts(self, domain: str, difficulty: int) -> int:
        """Return attempt count for a (domain, difficulty) pair.

        Why: used by novelty scoring.
        Fallback: returns 0 if never observed.
        """
        return self._attempts.get((domain, difficulty), 0)

    def frontier(self) -> List[Tuple[str, int]]:
        """Return (domain, difficulty) pairs in the zone of proximal development.

        Why: these are the most learnable tasks — not too easy, not too hard.
        Fallback: if no data recorded, returns empty list.
        """
        return [
            k for k, rate in self._rates.items()
            if ZPD_LOWER <= rate <= ZPD_UPPER
        ]

    def gaps(self) -> List[Tuple[str, int]]:
        """Return (domain, difficulty) pairs where competence is very low.

        Why: identifies areas needing new capabilities or remediation.
        Fallback: returns empty list if no data.
        """
        return [
            k for k, rate in self._rates.items()
            if rate < ZPD_LOWER
        ]

    def mastered(self) -> List[Tuple[str, int]]:
        """Return (domain, difficulty) pairs where competence is high.

        Why: identifies areas ready for difficulty escalation or transfer source.
        Fallback: returns empty list if no data.
        """
        return [
            k for k, rate in self._rates.items()
            if rate > MASTERY_THRESHOLD
        ]

    def novelty_score(self, domain: str, difficulty: int) -> float:
        """Return novelty score in [0.0, 1.0] — higher for less-attempted combos.

        Why: encourages exploration of under-sampled regions.
        Fallback: returns 1.0 for never-attempted combinations.
        """
        attempts = self._attempts.get((domain, difficulty), 0)
        return 1.0 - (attempts / (attempts + NOVELTY_HALF_LIFE))

    def all_domains(self) -> List[str]:
        """Return all known domains.

        Why: needed by GoalGenerator for cross-domain exploration.
        Fallback: returns empty list if no data.
        """
        return list({k[0] for k in self._rates})

    def all_keys(self) -> List[Tuple[str, int]]:
        """Return all tracked (domain, difficulty) pairs.

        Why: needed for iteration by other modules.
        Fallback: returns empty list if no data.
        """
        return list(self._rates.keys())

    def history_window(self, domain: str, difficulty: int,
                       window: int = 10) -> List[float]:
        """Return the current rate (no full history stored, returns single value).

        Why: used for learning progress computation.
        Fallback: returns [DEFAULT_SUCCESS_RATE] if no data.
        """
        rate = self._rates.get((domain, difficulty), DEFAULT_SUCCESS_RATE)
        return [rate]
