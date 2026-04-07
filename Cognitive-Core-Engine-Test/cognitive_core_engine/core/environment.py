"""Environment (research/engineering playground)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RuleProposal:
    proposal_id: str
    level: str  # "L0" | "L1" | "L2"
    payload: Dict[str, Any]
    creator_key: str
    created_ms: int
    evidence: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None


@dataclass
class TaskSpec:
    name: str
    difficulty: int
    baseline: float
    domain: str   # "algorithm" | "systems" | "theory" | "strategy" ...


class ResearchEnvironment:
    """
    Abstract multi-domain environment.
    - Each step is "run one agent on one project node for a given task/budget"
    - Reward ~ improvement over task baseline + infra gain
    - Global qualities (tool/kb/org) mediate acceleration
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.tasks: List[TaskSpec] = [
            TaskSpec("algorithm_design", difficulty=3, baseline=0.35, domain="algorithm"),
            TaskSpec("systems_optimization", difficulty=4, baseline=0.30, domain="systems"),
            TaskSpec("verification_pipeline", difficulty=2, baseline=0.40, domain="verification"),
            TaskSpec("toolchain_speedup", difficulty=5, baseline=0.25, domain="engineering"),
            TaskSpec("theory_discovery", difficulty=5, baseline=0.28, domain="theory"),
            TaskSpec("strategy_optimization", difficulty=3, baseline=0.32, domain="strategy"),
        ]
        self.global_tool_quality = 0.10
        self.global_kb_quality = 0.10
        self.global_org_quality = 0.10
        # BN-10 Fix 1: domain reward tracking for adaptive difficulty
        self._domain_reward_history: Dict[str, List[float]] = {}

    def sample_task(self) -> TaskSpec:
        return self.rng.choice(self.tasks)

    def make_observation(self, task: TaskSpec, budget: int,
                         phase: str = "research") -> Dict[str, Any]:
        return {
            "task": task.name,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "baseline": task.baseline,
            "budget": budget,
            "phase": phase,
        }

    def step(self, obs: Dict[str, Any], action: str,
             payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        diff = int(obs["difficulty"])
        base = float(obs["baseline"])
        budget = int(obs["budget"])
        domain = str(obs.get("domain", ""))

        tq = self.global_tool_quality
        kq = self.global_kb_quality
        oq = self.global_org_quality

        infra_scale = 1.0 / (1.0 + 0.4 * diff)
        leverage = 0.30 * tq + 0.30 * kq + 0.30 * oq
        diminishing = 1.0 / (1.0 + 2.0 * leverage)

        domain_bonus = {
            "algorithm": 0.04 if action == "attempt_breakthrough" else 0.01,
            "theory": 0.05 if action == "attempt_breakthrough" else 0.01,
            "systems": 0.04 if action in ("build_tool", "tune_orchestration") else 0.01,
            "engineering": 0.05 if action == "build_tool" else 0.01,
            "verification": 0.05 if action == "write_verified_note" else 0.01,
            "strategy": 0.04 if action == "tune_orchestration" else 0.01,
        }.get(domain, 0.01)

        if action == "build_tool":
            invest = float(payload.get("invest", 1.0))
            gain = (0.03 + 0.12 * tq) * invest * infra_scale * diminishing
            self.global_tool_quality = min(1.0, self.global_tool_quality + gain)
            raw = 0.02 * invest + domain_bonus
        elif action == "write_verified_note":
            invest = float(payload.get("invest", 1.0))
            gain = (0.03 + 0.10 * kq) * invest * infra_scale * diminishing
            self.global_kb_quality = min(1.0, self.global_kb_quality + gain)
            raw = 0.018 * invest + domain_bonus
        elif action == "tune_orchestration":
            invest = float(payload.get("invest", 1.0))
            gain = (0.03 + 0.10 * oq) * invest * infra_scale * diminishing
            self.global_org_quality = min(1.0, self.global_org_quality + gain)
            raw = 0.016 * invest + domain_bonus
        elif action == "attempt_breakthrough":
            effort = (1.0 + math.log(1 + budget) / 4.0)
            raw = (0.04 + 0.32 * leverage) * effort * (1.0 / (1.0 + 0.30 * diff)) + domain_bonus
        else:
            raw = 0.0

        noise = self.rng.uniform(-0.02, 0.02)

        # BN-10 Fix 1: skill_bonus based on RSI skill output proximity to target
        skill_bonus = 0.0
        rsi_skill_output = payload.get("rsi_skill_output")
        if rsi_skill_output is not None:
            target = diff * 0.1 + base * 2.0
            skill_bonus = max(0.0, 0.15 - abs(float(rsi_skill_output) - target) * 0.1)

        performance = max(0.0, min(1.0, base + raw + skill_bonus + noise))
        delta = performance - base
        infra_bonus = 0.025 * (tq + kq + oq) / 3.0
        reward = delta + infra_bonus

        # BN-10 Fix 1: Track domain rewards for adaptive difficulty
        if domain not in self._domain_reward_history:
            self._domain_reward_history[domain] = []
        self._domain_reward_history[domain].append(reward)
        self.adaptive_difficulty(domain)

        next_obs = dict(obs)
        next_obs["phase"] = "integrate"
        info = {
            "task": obs.get("task"),
            "performance": performance,
            "delta": delta,
            "tq": self.global_tool_quality,
            "kq": self.global_kb_quality,
            "oq": self.global_org_quality,
            "skill_bonus": skill_bonus,
        }
        return next_obs, reward, info

    def adaptive_difficulty(self, domain: str) -> None:
        """BN-10 Fix 1: Increase difficulty when domain performance is consistently high."""
        history = self._domain_reward_history.get(domain, [])
        if len(history) < 5:
            return
        recent = history[-5:]
        mean_reward = sum(recent) / len(recent)
        if mean_reward > 0.08:
            for task in self.tasks:
                if task.domain == domain and task.difficulty < 10:
                    task.difficulty = min(10, task.difficulty + 1)
                    task.baseline = max(0.05, task.baseline * 0.90)
                    # Reset history after adaptation
                    self._domain_reward_history[domain] = []
                    break

    def get_state_vector(self) -> List[float]:
        """BN-10 Fix 1: Return environment state as 8-element float vector."""
        difficulties = [t.difficulty for t in self.tasks] if self.tasks else [0]
        baselines = [t.baseline for t in self.tasks] if self.tasks else [0]
        domains = set(t.domain for t in self.tasks)
        return [
            self.global_tool_quality,
            self.global_kb_quality,
            self.global_org_quality,
            len(self.tasks) / 50.0,
            sum(difficulties) / max(1, len(difficulties)) / 10.0,
            sum(baselines) / max(1, len(baselines)),
            len(domains) / 20.0,
            max(difficulties) / 10.0,
        ]

    def add_domain(self, name: str, difficulty: int, baseline: float) -> TaskSpec:
        """Dynamically add a new domain to the environment.

        Why: enables open-ended learning beyond the initial 6 tasks.
        Fallback: reuses existing task if domain already exists.
        """
        # Check for duplicates
        for t in self.tasks:
            if t.domain == name and t.difficulty == difficulty:
                return t
        task = TaskSpec(
            name=f"dynamic_{name}_d{difficulty}",
            difficulty=max(1, min(10, difficulty)),
            baseline=max(0.05, min(0.9, baseline)),
            domain=name,
        )
        self.tasks.append(task)
        return task

    def evolve_task(self, task: TaskSpec, competence: float) -> TaskSpec:
        """Create a harder variant of an existing task.

        Why: prevents plateau by increasing challenge as competence grows.
        Fallback: returns original if already at max difficulty.
        """
        new_diff = min(10, task.difficulty + 1)
        new_baseline = max(0.05, task.baseline * 0.85)
        evolved = TaskSpec(
            name=f"{task.name}_evolved_d{new_diff}",
            difficulty=new_diff,
            baseline=new_baseline,
            domain=task.domain,
        )
        self.tasks.append(evolved)
        return evolved

    def compose_tasks(self, task_a: TaskSpec, task_b: TaskSpec) -> TaskSpec:
        """Create a multi-domain task requiring capabilities from both domains.

        Why: creates pressure for transfer learning across domains.
        Fallback: returns task_a if both are same domain.
        """
        composed = TaskSpec(
            name=f"composed_{task_a.domain}+{task_b.domain}",
            difficulty=min(10, max(task_a.difficulty, task_b.difficulty) + 1),
            baseline=max(0.05, min(task_a.baseline, task_b.baseline) * 0.8),
            domain=f"{task_a.domain}+{task_b.domain}",
        )
        self.tasks.append(composed)
        return composed
