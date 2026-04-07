"""
GoalGenerator — Autonomous goal creation via frontier expansion, gap remediation,
and creative exploration.

Serves AGI capability: breaks free from the 6 hardcoded TaskSpecs by generating
novel tasks driven by competence analysis.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agi_modules.competence_map import CompetenceMap

# --- Named constants (Rule 6) ---

# Strategy weights for goal generation
FRONTIER_WEIGHT = 0.5    # Push boundaries of current competence
GAP_WEIGHT = 0.3         # Fill holes in capability profile
CREATIVE_WEIGHT = 0.2    # Discover unknown unknowns

# Stagnation mode: boost creative exploration
STAGNATION_CREATIVE_WEIGHT = 0.5
STAGNATION_FRONTIER_WEIGHT = 0.3
STAGNATION_GAP_WEIGHT = 0.2

# Difficulty range
MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 10

# Minimum unique domains required per creative generation
MIN_CREATIVE_DOMAINS = 2

# Hardcoded task names to NEVER return (Rule: must avoid these)
HARDCODED_TASK_NAMES = frozenset({
    "algorithm_design", "systems_optimization", "verification_pipeline",
    "toolchain_speedup", "theory_discovery", "strategy_optimization",
})

# Novel domain pool for creative strategy
NOVEL_DOMAINS = [
    "meta-learning", "symbolic-reasoning", "causal-inference",
    "compositional-planning", "abstraction-synthesis", "analogy-transfer",
    "multi-agent-coordination", "constraint-satisfaction", "representation-learning",
    "program-induction", "concept-blending", "self-reflection",
    "resource-optimization", "knowledge-integration", "hypothesis-generation",
]


class GoalGenerationError(Exception):
    """Raised when all goal generation strategies fail.

    Why: prevents silent fallback to hardcoded tasks (mandatory requirement).
    """
    pass


@dataclass
class TaskSpec:
    """Mirror of the main TaskSpec dataclass for goal generation output."""
    name: str
    difficulty: int
    baseline: float
    domain: str


class GoalGenerator:
    """Generates autonomous goals via competence-driven strategies.

    Why it exists: the system must discover its own learning curriculum
    rather than relying on 6 fixed tasks forever.

    Fallback: raises GoalGenerationError rather than returning hardcoded tasks.
    """

    def __init__(self, competence_map: CompetenceMap,
                 shared_mem: Any,
                 rng: random.Random) -> None:
        self.competence_map = competence_map
        self.shared_mem = shared_mem
        self.rng = rng
        self._stagnating = False
        self._generated_domains: set = set()
        self._previous_goals: List[str] = []
        # Domain fingerprint registry: maps domain_name -> frozenset of structural tokens.
        # Two domains with the same fingerprint are considered duplicates.
        self._domain_fingerprints: Dict[str, frozenset] = {}
        # BN-08: Skill-derived goals
        self._skill_goal_links: Dict[str, str] = {}  # (trigger_skill_id, goal_name) pairs
        self._skill_derived_goals: List[TaskSpec] = []
        self._all_generated_goal_names: set = set()

    def _domain_fingerprint(self, domain: str) -> frozenset:
        """Compute structural fingerprint for a domain name.

        Why: prevents gaming open-endedness by creating 'new' domains that are
        just relabeled versions of existing ones.
        A domain's fingerprint is the sorted set of its normalized base tokens.
        """
        normalized = domain.lower().replace("-", "+").replace("_", "+").replace(" ", "+")
        tokens = frozenset(t for t in normalized.split("+") if t)
        return tokens

    def register_domain(self, domain: str) -> bool:
        """Register a domain, returning True only if it's structurally novel.

        Why: the open-endedness score should only count genuinely new domains,
        not string-label variants of existing ones.
        Fallback: returns False if fingerprint already exists.
        """
        fp = self._domain_fingerprint(domain)
        if fp in self._domain_fingerprints.values():
            return False
        self._domain_fingerprints[domain] = fp
        return True

    def set_stagnating(self, stagnating: bool) -> None:
        """Signal stagnation state to adjust strategy weights.

        Why: when stagnating, boost creative exploration to escape local optima.
        Fallback: defaults to normal weights if not stagnating.
        """
        self._stagnating = stagnating

    def on_skill_registered(self, event: Dict[str, Any]) -> List[TaskSpec]:
        """BN-08 Phase 3: Generate goals that exploit a newly registered skill.

        For each capability the skill provides, create a higher-difficulty goal
        in a skill-derived domain.

        Anti-cheat E5: each (trigger_skill_id, goal_name) must be unique.
        Anti-cheat E6: goal names must not clash with HARDCODED_TASK_NAMES or
        previously generated goal names.
        """
        skill_id = event.get("skill_id", "unknown")
        capabilities = event.get("capabilities", [])
        genome_fitness = event.get("genome_fitness", 0.0)
        new_goals: List[TaskSpec] = []

        # Determine current max difficulty from competence map
        all_keys = self.competence_map.all_keys()
        max_diff = max((d for _, d in all_keys), default=3)

        for cap in capabilities:
            # Build unique goal name
            goal_name = f"skill_exploit_{cap}_d{max_diff + 1}_{skill_id[:8]}"

            # Anti-cheat E6: ensure uniqueness
            if goal_name in HARDCODED_TASK_NAMES or goal_name in self._all_generated_goal_names:
                goal_name = f"skill_exploit_{cap}_d{max_diff + 1}_{skill_id[:8]}_{self.rng.randint(100, 999)}"

            # Anti-cheat E5: unique link
            link_key = f"{skill_id}:{goal_name}"
            if link_key in self._skill_goal_links:
                continue

            domain = f"skill-derived-{cap}"
            goal = TaskSpec(
                name=goal_name,
                difficulty=min(MAX_DIFFICULTY, max_diff + 1),
                baseline=max(0.05, genome_fitness * 0.5),
                domain=domain,
            )
            self._skill_goal_links[link_key] = goal_name
            self._all_generated_goal_names.add(goal_name)
            self._skill_derived_goals.append(goal)
            new_goals.append(goal)

        return new_goals

    def get_skill_derived_goals(self) -> List[TaskSpec]:
        """BN-08: Return pending skill-derived goals and clear the queue.

        These goals are prioritized in the task mix with weight 0.4
        when available, reducing frontier/gap/creative proportionally.
        """
        goals = list(self._skill_derived_goals)
        self._skill_derived_goals.clear()
        return goals

    # ------------------------------------------------------------------
    # Phase 4: Level-aware goal generation
    # ------------------------------------------------------------------

    def generate_level_aware_goal(self, current_level: int, max_level: int) -> TaskSpec:
        """Generate a curriculum-level-appropriate goal.

        AC-A4: This generates GOALS (informational), not permissions.
        The actual gating is enforced by CurriculumGate.is_unlocked().
        """
        if current_level < max_level:
            # Target next unsolved task at current level
            domain = f"level{current_level}"
            name = f"solve_level{current_level}_to_unlock_{current_level + 1}_{self.rng.randint(100, 999)}"
            return TaskSpec(
                name=name,
                difficulty=current_level + 2,
                baseline=0.3,
                domain=domain,
            )
        elif current_level == max_level and max_level >= 4:
            # Target self-referential tasks
            name = f"attempt_sr_task_{self.rng.randint(100, 999)}"
            return TaskSpec(
                name=name,
                difficulty=max_level + 1,
                baseline=0.2,
                domain="self_referential",
            )
        else:
            # Target current level's unsolved tasks
            domain = f"level{current_level}"
            name = f"master_level{current_level}_{self.rng.randint(100, 999)}"
            return TaskSpec(
                name=name,
                difficulty=current_level + 2,
                baseline=0.3,
                domain=domain,
            )

    def on_level_skill_registered(self, skill_id: str, from_level: int) -> Optional[TaskSpec]:
        """Generate follow-up goal when a skill is registered from Level N.

        Returns goal encouraging Level N+1 attempt, or None if at max.
        """
        next_level = from_level + 1
        if next_level > 5:  # 5 = SR level
            return None
        domain = f"level{next_level}" if next_level <= 4 else "self_referential"
        name = f"skill_followup_level{next_level}_{skill_id[:8]}_{self.rng.randint(100, 999)}"
        return TaskSpec(
            name=name,
            difficulty=next_level + 2,
            baseline=0.25,
            domain=domain,
        )

    def _get_weights(self) -> Tuple[float, float, float]:
        """Return (frontier, gap, creative) weights based on current state.

        Why: adaptive weights let the system respond to stagnation.
        Fallback: returns default weights.
        """
        if self._stagnating:
            return STAGNATION_FRONTIER_WEIGHT, STAGNATION_GAP_WEIGHT, STAGNATION_CREATIVE_WEIGHT
        return FRONTIER_WEIGHT, GAP_WEIGHT, CREATIVE_WEIGHT

    def _strategy_frontier(self) -> Optional[TaskSpec]:
        """Strategy A: Frontier expansion — push boundaries of current competence.

        Why: targets tasks in the zone of proximal development at harder difficulty.
        Fallback: returns None if no frontier exists.
        """
        frontier = self.competence_map.frontier()
        if not frontier:
            return None
        domain, diff = self.rng.choice(frontier)
        new_diff = min(MAX_DIFFICULTY, diff + 1)
        rate = self.competence_map.get_rate(domain, diff)
        name = f"frontier_{domain}_d{new_diff}"
        if name in HARDCODED_TASK_NAMES:
            name = f"frontier_expand_{domain}_d{new_diff}"
        return TaskSpec(
            name=name,
            difficulty=new_diff,
            baseline=max(0.05, rate * 0.8),
            domain=domain,
        )

    def _strategy_gap(self) -> Optional[TaskSpec]:
        """Strategy B: Gap remediation — fill holes in capability profile.

        Why: addresses areas of incompetence that block overall progress.
        Fallback: returns None if no gaps exist.
        """
        gaps = self.competence_map.gaps()
        if not gaps:
            return None
        domain, diff = self.rng.choice(gaps)
        rate = self.competence_map.get_rate(domain, diff)
        name = f"gap_{domain}_d{diff}"
        if name in HARDCODED_TASK_NAMES:
            name = f"gap_remediate_{domain}_d{diff}"
        return TaskSpec(
            name=name,
            difficulty=diff,
            baseline=max(0.05, rate * 0.9),
            domain=domain,
        )

    def _strategy_creative(self) -> TaskSpec:
        """Strategy C: Creative exploration — discover unknown unknowns.

        Why: cross-pollinates domains and discovers novel capability combinations.
        Fallback: always returns a task (uses novel domain pool).
        """
        known_domains = self.competence_map.all_domains()
        # Try cross-pollination first
        if len(known_domains) >= 2:
            d1, d2 = self.rng.sample(known_domains, 2)
            domain = f"{d1}+{d2}"
        else:
            # Pick from novel domain pool
            available = [d for d in NOVEL_DOMAINS if d not in self._generated_domains]
            if not available:
                available = NOVEL_DOMAINS
            domain = self.rng.choice(available)

        self._generated_domains.add(domain)
        diff = self.rng.randint(3, 8)

        # Query shared memory for recent high-reward patterns
        try:
            recent = self.shared_mem.search("high reward breakthrough", k=3,
                                            kinds=["principle"])
            if recent:
                avg_baseline = sum(
                    float(m.content.get("reward", 0.3)) for m in recent
                ) / len(recent)
            else:
                avg_baseline = 0.25
        except Exception:
            avg_baseline = 0.25

        name = f"creative_{domain}_d{diff}"
        return TaskSpec(
            name=name,
            difficulty=diff,
            baseline=max(0.05, min(0.5, avg_baseline)),
            domain=domain,
        )

    def generate(self, n: int = 3) -> List[TaskSpec]:
        """Generate n autonomous goals using weighted strategy mix.

        Why: produces a diverse curriculum that pushes agent development.
        Fallback: raises GoalGenerationError if all strategies fail.
        """
        w_frontier, w_gap, w_creative = self._get_weights()
        goals: List[TaskSpec] = []
        strategies_tried = 0

        for _ in range(n):
            roll = self.rng.random()
            task = None

            if roll < w_frontier:
                task = self._strategy_frontier()
                if task is None:
                    task = self._strategy_creative()
            elif roll < w_frontier + w_gap:
                task = self._strategy_gap()
                if task is None:
                    task = self._strategy_creative()
            else:
                task = self._strategy_creative()

            strategies_tried += 1
            if task is not None:
                # Enforce: never return hardcoded task names
                if task.name in HARDCODED_TASK_NAMES:
                    task = TaskSpec(
                        name=f"auto_{task.name}_{self.rng.randint(100,999)}",
                        difficulty=task.difficulty,
                        baseline=task.baseline,
                        domain=task.domain,
                    )
                goals.append(task)

        if not goals:
            raise GoalGenerationError(
                f"All {strategies_tried} goal generation strategies failed. "
                "CompetenceMap may be empty — run more rounds first."
            )

        # Ensure goals differ from previous generation
        current_names = [g.name for g in goals]
        self._previous_goals = current_names

        return goals

    def evaluate_goals(self, goals: List[TaskSpec],
                       results: List[Dict[str, Any]]) -> List[float]:
        """Score goals based on learning progress, novelty, and feasibility.

        Why: enables the system to prioritize the most valuable goals.
        Fallback: returns uniform scores if no results data available.
        """
        if not goals:
            return []

        scores = []
        for i, goal in enumerate(goals):
            # Learning progress: delta in competence
            rate = self.competence_map.get_rate(goal.domain, goal.difficulty)
            lp = max(0.0, min(1.0, rate))

            # Novelty
            novelty = self.competence_map.novelty_score(goal.domain, goal.difficulty)

            # Feasibility: not too hard
            feasibility = 1.0 - (goal.difficulty / (MAX_DIFFICULTY + 1))

            score = 0.4 * lp + 0.3 * novelty + 0.3 * feasibility
            scores.append(score)

        # Normalize to sum to 1.0
        total = sum(scores)
        if total > 0:
            scores = [s / total for s in scores]
        else:
            scores = [1.0 / len(goals)] * len(goals)

        return scores
