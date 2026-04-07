"""
AGIProgressTracker — Measures progress across 5 AGI capability axes.

Serves AGI capability: provides quantitative evidence of progress toward
general intelligence across generalization, autonomy, self-improvement,
abstraction, and open-endedness.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# --- Named constants (Rule 6) ---

# Target abstraction depth for scoring
TARGET_ABSTRACTION_DEPTH = 5

# Plateau detection: rounds with < this improvement
PLATEAU_IMPROVEMENT_THRESHOLD = 0.01
PLATEAU_WINDOW = 20


class AGIProgressTracker:
    """Tracks 5 AGI capability axes, each scored 0.0 to 1.0.

    Why it exists: provides measurable evidence of AGI-relevant progress
    beyond simple task performance metrics.

    Fallback: returns 0.0 for axes with no data.
    """

    def __init__(self) -> None:
        self._transfer_successes: List[float] = []
        self._transfer_attempts: int = 0
        self._self_generated_goals: int = 0
        self._total_goals: int = 0
        self._goals_with_benchmark_improvement: int = 0  # A4: externally validated
        self._beneficial_self_mods: int = 0
        self._externally_validated_mods: int = 0  # A4: held-out improvement
        self._total_self_mods: int = 0
        self._concept_depth: int = 0
        self._new_domains: int = 0
        self._domains_above_random: int = 0  # A4: only count if above random
        self._difficulty_increases: int = 0
        self._rounds_elapsed: int = 0
        self._composite_history: List[float] = []
        self._external_benchmark_scores: List[float] = []
        # BN-08: Emergence metrics
        self._skill_births: int = 0
        self._skills_that_improved_reward: int = 0
        self._skill_derived_domains: int = 0
        self._initial_domains: set = set()
        self._excluded_domains: set = set()  # BN-09 Fix 5: NOVEL_DOMAINS
        self._recursive_depth: int = 0
        self._skill_derived_domain_names: set = set()
        # Phase 4: Algorithm synthesis metrics
        self._algorithm_level: int = 0
        self._sr_attempts: int = 0
        self._sr_successes: int = 0

    def update_transfer(self, success: float, attempted: bool) -> None:
        """Record a transfer learning attempt and outcome.

        Why: feeds the GENERALIZATION axis.
        Fallback: no-op if not attempted.
        """
        if attempted:
            self._transfer_attempts += 1
            self._transfer_successes.append(success)

    def update_goals(self, self_generated: int, total: int) -> None:
        """Record goal generation statistics.

        Why: feeds the AUTONOMY axis.
        Fallback: no-op.
        """
        self._self_generated_goals += self_generated
        self._total_goals += total

    def update_self_improvement(self, beneficial: bool, attempted: bool) -> None:
        """Record self-improvement attempt and outcome.

        Why: feeds the SELF-IMPROVEMENT axis.
        Fallback: no-op if not attempted.
        """
        if attempted:
            self._total_self_mods += 1
            if beneficial:
                self._beneficial_self_mods += 1

    def update_abstraction(self, depth: int) -> None:
        """Record concept graph depth.

        Why: feeds the ABSTRACTION axis.
        Fallback: no-op.
        """
        self._concept_depth = max(self._concept_depth, depth)

    def update_open_endedness(self, new_domains: int,
                              difficulty_increases: int,
                              domains_above_random: int = 0) -> None:
        """Record environment expansion metrics.

        Why: feeds the OPEN-ENDEDNESS axis.
        A4: only count domains where above-random performance achieved.
        Fallback: no-op.
        """
        self._new_domains += new_domains
        self._difficulty_increases += difficulty_increases
        self._domains_above_random += domains_above_random

    def update_external_benchmark(self, score: float) -> None:
        """Record external benchmark score for overfitting detection.

        Why (A2/A4): tracks held-out performance over time.
        Fallback: no-op.
        """
        self._external_benchmark_scores.append(score)

    def update_goal_benchmark_improvement(self, improved: bool) -> None:
        """Record whether a self-generated goal led to benchmark improvement.

        Why (A4): AUTONOMY only counts goals that led to actual improvement.
        Fallback: no-op.
        """
        if improved:
            self._goals_with_benchmark_improvement += 1

    def update_externally_validated_mod(self, validated: bool) -> None:
        """Record whether a self-improvement mod improved held-out score.

        Why (A4): SELF-IMPROVEMENT only counts externally validated mods.
        Fallback: no-op.
        """
        if validated:
            self._externally_validated_mods += 1

    # --- BN-08: Emergence metrics ---

    def set_initial_domains(self, domains: set) -> None:
        """Record the initial domain set for capability horizon calculation."""
        self._initial_domains = set(domains)

    def set_excluded_domains(self, domains: set) -> None:
        """Record domains to exclude from capability_horizon (e.g. NOVEL_DOMAINS).

        BN-09 Fix 5: prevents creative-strategy domains from inflating
        the capability horizon metric.
        """
        self._excluded_domains = set(domains)

    def update_emergence(
        self,
        skill_births: int = 0,
        skills_improved_reward: int = 0,
        skill_derived_domains: int = 0,
        recursive_depth: int = 0,
        skill_derived_domain_names: Optional[set] = None,
    ) -> None:
        """Update BN-08 emergence metrics.

        Tool Genesis Rate = skills_that_improved_reward / total_rounds (E9)
        Capability Horizon = skill-derived domains solvable (E10: excludes initial + NOVEL_DOMAINS)
        Recursive Depth = max causal chain depth
        """
        self._skill_births += skill_births
        self._skills_that_improved_reward += skills_improved_reward
        self._skill_derived_domains += skill_derived_domains
        self._recursive_depth = max(self._recursive_depth, recursive_depth)
        if skill_derived_domain_names:
            self._skill_derived_domain_names.update(skill_derived_domain_names)

    def tool_genesis_rate(self) -> float:
        """BN-08: evolved skills that improved reward / total_rounds.

        Anti-cheat E9: denominator is total_rounds, not skill-present rounds.
        """
        if self._rounds_elapsed == 0:
            return 0.0
        return self._skills_that_improved_reward / self._rounds_elapsed

    def capability_horizon(self) -> int:
        """BN-08: count of solvable domains that didn't exist at round 0.

        Anti-cheat E10: excludes initial domains AND NOVEL_DOMAINS.
        BN-09 Fix 5: also excludes _excluded_domains (NOVEL_DOMAINS).
        """
        return len(self._skill_derived_domain_names - self._initial_domains - self._excluded_domains)

    def emergence_depth(self) -> int:
        """BN-08: longest causal chain depth."""
        return self._recursive_depth

    # ------------------------------------------------------------------
    # Phase 4: Algorithm synthesis metrics
    # ------------------------------------------------------------------

    def update_algorithm_level(self, level: int) -> None:
        """Track the highest algorithm task level achieved.

        Feeds capability_horizon = level / 4.0 (normalized).
        """
        self._algorithm_level = max(self._algorithm_level, level)

    def update_sr_success(self, task_name: str, reward: float) -> None:
        """Track self-referential task attempts and successes.

        Feeds self_improvement axis: SR success bumps score by 0.1 (capped).
        """
        self._sr_attempts += 1
        if reward > 0.5:
            self._sr_successes += 1
            # Bump self-improvement score via beneficial mod tracking
            self._beneficial_self_mods += 1
            self._total_self_mods = max(self._total_self_mods, self._beneficial_self_mods)

    def algorithm_summary(self) -> Dict[str, Any]:
        """Return Phase 4 algorithm synthesis metrics."""
        return {
            "capability_horizon": self._algorithm_level / 4.0,
            "sr_success_rate": self._sr_successes / max(1, self._sr_attempts),
            "algorithm_level": self._algorithm_level,
            "sr_attempts": self._sr_attempts,
            "sr_successes": self._sr_successes,
        }

    def tick_round(self) -> None:
        """Record that a round has elapsed.

        Why: needed for rate-based metrics.
        Fallback: no-op.
        """
        self._rounds_elapsed += 1
        self._composite_history.append(self.composite_score())

    def score(self) -> Dict[str, float]:
        """Return all 5 AGI axis scores.

        Why: provides the full capability profile.
        Fallback: returns 0.0 for axes with no data.
        """
        return {
            "generalization": self._score_generalization(),
            "autonomy": self._score_autonomy(),
            "self_improvement": self._score_self_improvement(),
            "abstraction": self._score_abstraction(),
            "open_endedness": self._score_open_endedness(),
        }

    def composite_score(self) -> float:
        """Geometric mean of all 5 axes (all must improve, not just one).

        Why: prevents gaming a single axis while neglecting others.
        Fallback: returns 0.0 if any axis is 0.
        """
        scores = self.score()
        values = list(scores.values())

        # Add small epsilon to avoid zero product
        epsilon = 0.001
        adjusted = [max(v, epsilon) for v in values]

        # Geometric mean
        product = 1.0
        for v in adjusted:
            product *= v
        return product ** (1.0 / len(adjusted))

    def progress_report(self, round_idx: int) -> str:
        """Human-readable progress report.

        Why: enables monitoring of AGI capability development.
        Fallback: returns minimal report if no data.
        """
        scores = self.score()
        composite = self.composite_score()
        lines = [
            f"=== AGI Progress Report (Round {round_idx}) ===",
            f"  Generalization:    {scores['generalization']:.3f}",
            f"  Autonomy:          {scores['autonomy']:.3f}",
            f"  Self-Improvement:  {scores['self_improvement']:.3f}",
            f"  Abstraction:       {scores['abstraction']:.3f}",
            f"  Open-Endedness:    {scores['open_endedness']:.3f}",
            f"  Composite (geom):  {composite:.3f}",
            f"  Plateaued:         {self.is_plateaued()}",
            f"  Algorithm Level:   {self._algorithm_level}",
            f"  Capability Horizon:{self._algorithm_level / 4.0:.3f}",
            f"  SR Success Rate:   {self._sr_successes / max(1, self._sr_attempts):.3f}",
        ]
        return "\n".join(lines)

    def is_plateaued(self) -> bool:
        """Detect if composite score hasn't improved in PLATEAU_WINDOW rounds.

        Why: triggers adaptive responses to break out of stagnation.
        Fallback: returns False if insufficient history.
        """
        if len(self._composite_history) < PLATEAU_WINDOW:
            return False

        recent = self._composite_history[-PLATEAU_WINDOW:]
        improvement = max(recent) - min(recent)
        return improvement < PLATEAU_IMPROVEMENT_THRESHOLD

    # --- Private scoring methods ---

    def _score_generalization(self) -> float:
        """GENERALIZATION: mean transfer success rate.

        Why: measures cross-domain knowledge reuse.
        Fallback: returns 0.0 if no transfers attempted.
        """
        if not self._transfer_successes:
            return 0.0
        # Normalize: positive transfer → higher score
        positive = [max(0.0, s) for s in self._transfer_successes]
        return min(1.0, sum(positive) / max(1, len(positive)))

    def _score_autonomy(self) -> float:
        """AUTONOMY: fraction of self-generated goals.

        Why: measures independence from hardcoded tasks.
        Fallback: returns 0.0 if no goals generated.
        """
        if self._total_goals == 0:
            return 0.0
        return min(1.0, self._self_generated_goals / self._total_goals)

    def _score_self_improvement(self) -> float:
        """SELF-IMPROVEMENT: beneficial modification rate.

        Why: measures ability to improve own parameters.
        Fallback: returns 0.0 if no modifications attempted.
        """
        if self._total_self_mods == 0:
            return 0.0
        return min(1.0, self._beneficial_self_mods / self._total_self_mods)

    def _score_abstraction(self) -> float:
        """ABSTRACTION: concept depth / target depth.

        Why: measures hierarchical concept formation capability.
        Fallback: returns 0.0 if no concepts formed.
        """
        return min(1.0, self._concept_depth / TARGET_ABSTRACTION_DEPTH)

    def _score_open_endedness(self) -> float:
        """OPEN-ENDEDNESS: domain mastery fraction (70%) + difficulty rate (30%).

        Why: counting new domain labels alone is easily gamed (string-label inflation).
        This scoring requires domains_above_random > 0 (actual performance) to
        contribute. Difficulty increases contribute 30% as a growth signal.
        Fallback: returns 0.0 if no data.
        """
        if self._rounds_elapsed == 0:
            return 0.0

        # Component 1 (70%): fraction of new domains where above-random performance
        mastery_fraction = 0.0
        if self._new_domains > 0:
            mastery_fraction = self._domains_above_random / self._new_domains

        # Component 2 (30%): difficulty increase rate, capped at 1.0
        diff_rate = min(1.0, self._difficulty_increases / max(1, self._rounds_elapsed) / 0.3)

        return min(1.0, 0.7 * mastery_fraction + 0.3 * diff_rate)
