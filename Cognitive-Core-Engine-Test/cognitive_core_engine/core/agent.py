"""Agent — B-type core with BN-06 adaptive meta-depth ceiling.

BN-06:
  _meta_depth_ceiling() computes the agent's recent calibration error
  (mean |Q_predicted - actual_reward| over the last 8 transitions) and
  caps planner depth at:
    depth=1  if calibration_error > 0.30  (low confidence, conservative)
    depth=2  if calibration_error in [0.15, 0.30]
    depth=3  if calibration_error in [0.05, 0.15)
    depth=4  if calibration_error < 0.05   (high confidence)
  Theorist and strategist roles receive a +1 ceiling bonus.
  The agent stores _transition_log (max 8 entries) filled from world-model
  Q predictions vs actual rewards.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cognitive_core_engine.core.world_model import WorldModel
from cognitive_core_engine.core.planner import Planner
from cognitive_core_engine.core.skills import Skill, SkillStep, SkillLibrary
from cognitive_core_engine.core.tools import ToolRegistry
from cognitive_core_engine.core.memory import SharedMemory
from cognitive_core_engine.core.environment import ResearchEnvironment
from cognitive_core_engine.core.project_graph import ProjectNode

from agi_modules.intrinsic_motivation import IntrinsicMotivationModule
from agi_modules.self_model import SelfModel
from agi_modules.concept_graph import ConceptGraph
from agi_modules.hierarchical_planner import HierarchicalPlanner

# ---------------------------------------------------------------------------
# Constants for BN-06 depth ceiling
# ---------------------------------------------------------------------------

_TRANSITION_LOG_MAX = 8       # rolling window for calibration error
_CAL_HIGH_CONFIDENCE = 0.05   # below this → depth 4
_CAL_MEDIUM_HIGH = 0.15       # below this → depth 3
_CAL_MEDIUM_LOW = 0.30        # below this → depth 2; above → depth 1
_ROLE_DEPTH_BONUS = frozenset({"theorist", "strategist"})  # +1 bonus roles


# ---------------------------------------------------------------------------
# BN-06 helper: calibration-based depth ceiling
# ---------------------------------------------------------------------------

def _meta_depth_ceiling(
    transition_log: List[Dict[str, float]],
    role: str,
    config_depth: int,
) -> int:
    """Compute the calibration-bounded planner depth for this step.

    Args:
        transition_log: list of {"q_pred": float, "reward": float} dicts
                        (most recent _TRANSITION_LOG_MAX entries).
        role:           agent role string ("theorist" gets +1 bonus).
        config_depth:   the agent's configured planner_depth (hard ceiling).

    Returns:
        Effective planner depth, capped at config_depth.
    """
    if not transition_log:
        # No history yet → conservative default
        base = 2
    else:
        cal_error = sum(
            abs(t["q_pred"] - t["reward"]) for t in transition_log
        ) / len(transition_log)

        if cal_error < _CAL_HIGH_CONFIDENCE:
            base = 4
        elif cal_error < _CAL_MEDIUM_HIGH:
            base = 3
        elif cal_error < _CAL_MEDIUM_LOW:
            base = 2
        else:
            base = 1

    bonus = 1 if role in _ROLE_DEPTH_BONUS else 0
    return min(config_depth, base + bonus)


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    name: str
    role: str = "general"  # theorist | builder | experimenter | verifier | strategist
    planner_depth: int = 3
    planner_width: int = 6
    risk: float = 0.2
    extrinsic_weight: float = 0.6
    intrinsic_weight: float = 0.4


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    B-type core:
    - WorldModel + Planner
    - SharedMemory + SkillLibrary + ToolRegistry
    - No self-modifying code; only state/memory/skills evolve.
    """

    def __init__(
        self,
        cfg: AgentConfig,
        tools: ToolRegistry,
        shared_mem: SharedMemory,
        skills: SkillLibrary,
        intrinsic_motivation: Optional[IntrinsicMotivationModule] = None,
        self_model: Optional[SelfModel] = None,
        concept_graph: Optional[ConceptGraph] = None,
    ) -> None:
        self.cfg = cfg
        self.tools = tools
        self.mem = shared_mem
        self.skills = skills

        self.wm = WorldModel()
        self.planner = Planner(self.wm, depth=cfg.planner_depth,
                               width=cfg.planner_width)
        self.domain_expertise: Dict[str, float] = {}

        # AGI modules
        self.intrinsic_motivation = intrinsic_motivation
        self.self_model = self_model
        self.concept_graph = concept_graph
        self.hierarchical_planner: Optional[HierarchicalPlanner] = None
        if concept_graph is not None:
            self.hierarchical_planner = HierarchicalPlanner(self.wm, concept_graph)
        self._skip_requested = False

        # BN-06: rolling transition log for calibration error
        self._transition_log: List[Dict[str, float]] = []

        # BN-09 Fix 2: Track RSI skill consultation for reward feedback
        self._last_consulted_rsi_skill_id: Optional[str] = None
        self._rsi_skill_accepted: bool = False
        # BN-10 Fix 4: Store raw skill output for env.step() skill_bonus
        self._last_rsi_skill_output: Optional[float] = None

    # ------------------------------------------------------------------
    # BN-06: transition log management
    # ------------------------------------------------------------------

    def _record_transition(self, obs: Dict[str, Any], action: str, reward: float) -> None:
        """Append a (Q_predicted, actual_reward) entry to the transition log."""
        try:
            q_pred = float(self.wm.q_value(obs, action))
        except Exception:
            q_pred = 0.0
        self._transition_log.append({"q_pred": q_pred, "reward": reward})
        # Keep only the most recent entries
        if len(self._transition_log) > _TRANSITION_LOG_MAX:
            self._transition_log = self._transition_log[-_TRANSITION_LOG_MAX:]

    def calibration_error(self) -> float:
        """Return mean |Q_pred - reward| over the transition log."""
        if not self._transition_log:
            return float("inf")
        return sum(
            abs(t["q_pred"] - t["reward"]) for t in self._transition_log
        ) / len(self._transition_log)

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------

    def action_space(self) -> List[str]:
        base = ["attempt_breakthrough", "build_tool", "write_verified_note", "tune_orchestration"]
        r = self.cfg.role
        if r == "verifier":
            return ["write_verified_note", "build_tool", "tune_orchestration", "attempt_breakthrough"]
        if r == "builder":
            return ["build_tool", "attempt_breakthrough", "write_verified_note", "tune_orchestration"]
        if r == "theorist":
            return ["attempt_breakthrough", "write_verified_note", "build_tool", "tune_orchestration"]
        if r == "experimenter":
            return ["build_tool", "attempt_breakthrough", "write_verified_note", "tune_orchestration"]
        if r == "strategist":
            return ["tune_orchestration", "attempt_breakthrough", "build_tool", "write_verified_note"]
        # Phase 4: Adversarial roles
        if r == "challenger":
            return ["generate_challenge", "submit_program", "attempt_breakthrough", "build_tool"]
        if r == "meta_optimizer":
            return ["submit_program", "compose_skills", "tune_orchestration", "attempt_breakthrough"]
        return base

    # ------------------------------------------------------------------
    # Phase 4: New action implementations
    # ------------------------------------------------------------------

    def _act_submit_program(self, obs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Submit the best genome from OmegaForge population (AC-A2)."""
        forge = obs.get("forge_engine")
        if forge is None or not hasattr(forge, "population") or not forge.population:
            return ("submit_program", {"genome": None})
        ranked = sorted(forge.population, key=lambda g: g.last_score, reverse=True)
        best = ranked[0]
        return ("submit_program", {
            "genome": best,
            "task_name": obs.get("current_task", "L0_SUM"),
        })

    def _act_compose_skills(self, obs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Compose top 2 skills by performance."""
        skill_lib = obs.get("skill_library")
        if skill_lib is None:
            return ("compose_skills", {"skill_ids": []})
        vm_skills = skill_lib.vm_skills()
        if len(vm_skills) < 2:
            return ("compose_skills", {"skill_ids": []})
        # Sort by mean performance
        def _mean_perf(sk):
            perf = skill_lib.skill_performance_log.get(sk.id, [])
            return sum(perf) / max(1, len(perf)) if perf else 0.0
        ranked = sorted(vm_skills, key=_mean_perf, reverse=True)
        return ("compose_skills", {"skill_ids": [ranked[0].id, ranked[1].id]})

    def _act_generate_challenge(self, obs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate a challenge task with oracle (AC-A1: >= 3 unique inputs)."""
        from cognitive_core_engine.omega_forge.instructions import Instruction, ProgramGenome
        from cognitive_core_engine.omega_forge.stage1 import TaskMacroLibrary

        base_seed = obs.get("step", 0)
        for retry in range(5):
            rng = random.Random(base_seed + retry * 1000)
            inputs_list = []
            for _ in range(5):
                n = rng.randint(2, 6)
                inp = [float(rng.randint(-9, 9)) for _ in range(n)]
                inputs_list.append(inp)
            # AC-A1: Check uniqueness
            unique = set(str(inp) for inp in inputs_list)
            if len(unique) >= 3:
                break
        else:
            return ("generate_challenge", {"error": "insufficient_unique_inputs"})

        expected_list = [sum(inp) for inp in inputs_list]

        # Build oracle genome from known-good SUM instructions
        oracle_insts = TaskMacroLibrary.sum_skeleton()
        oracle_insts.append(Instruction("HALT", 0, 0, 0))
        oracle_genome = ProgramGenome(gid="challenge_oracle", instructions=oracle_insts)

        return ("generate_challenge", {
            "inputs_list": inputs_list,
            "expected_outputs_list": expected_list,
            "oracle_genome": oracle_genome,
        })

    # ------------------------------------------------------------------
    # Action selection (BN-06 integrated)
    # ------------------------------------------------------------------

    def choose_action(self, obs: Dict[str, Any]) -> str:
        """Select an action using calibration-bounded planning depth.

        BN-06: planner depth is constrained by _meta_depth_ceiling() rather
        than being set directly from task difficulty.  Task difficulty can
        still widen the planner (more branches) but cannot exceed the
        calibration ceiling.
        """
        difficulty = int(obs.get("difficulty", 3))

        # BN-06: compute calibration-bounded depth
        cal_depth = _meta_depth_ceiling(
            self._transition_log, self.cfg.role, self.cfg.planner_depth
        )
        # Width still adapts to task difficulty, but depth is bounded by calibration
        adaptive_width = min(12, max(4, 4 + difficulty // 2))
        self.planner = Planner(self.wm, depth=cal_depth, width=adaptive_width)

        try:
            candidates = self.planner.propose(obs, self.action_space(), self.cfg.risk)
        except Exception:
            candidates = []

        if not candidates:
            return random.choice(self.action_space())

        draft_action = candidates[0].actions[0]
        task = obs.get("task", "")

        # System 2: memory-assisted success/failure analysis
        past_episodes = self.mem.search(
            query=f"{task} {draft_action}",
            k=8,
            kinds=["episode"],
            tags=[task, draft_action],
        )

        success_count = 0
        failure_count = 0
        total_reward = 0.0
        for mem in past_episodes:
            if mem.content.get("action") == draft_action:
                r = float(mem.content.get("reward", 0.0))
                total_reward += r
                if r >= 0.25:
                    success_count += 1
                elif r < 0.10:
                    failure_count += 1

        if success_count + failure_count > 0:
            success_rate = success_count / (success_count + failure_count)
            if success_rate < 0.3 and failure_count >= 3:
                if random.random() < 0.6 and len(candidates) > 1:
                    return candidates[1].actions[0]

        # Curiosity-boosted exploration
        if self.intrinsic_motivation is not None:
            best_curiosity = 0.0
            best_curious_action = None
            for action in self.action_space():
                curiosity = self.intrinsic_motivation.curiosity_for_action(obs, action)
                if curiosity > 0.7 and curiosity > best_curiosity and action != draft_action:
                    best_curiosity = curiosity
                    best_curious_action = action
            if best_curious_action is not None:
                h = hash(str(obs.get("task", "")) + str(obs.get("difficulty", 0)))
                if h % 5 < 2:
                    return best_curious_action

        # BN-08/09/10: Consult RSI skills if available
        vm_skills = self.skills.vm_skills()
        if vm_skills and hasattr(obs, '__getitem__'):
            try:
                difficulty = int(obs.get("difficulty", 3))
                budget = int(obs.get("budget", 12))
                tq = float(obs.get("tq", 0.5))
                kq = float(obs.get("kq", 0.3))
                oq = float(obs.get("oq", 0.2))
                skill_input = [tq, kq, oq, float(difficulty), float(budget)]
                actions = self.action_space()
                for sk in vm_skills[:3]:  # consult up to 3 skills
                    if hasattr(sk, "fn") and callable(getattr(sk, "fn", None)):
                        try:
                            raw_output = sk.fn(skill_input)
                            action_idx = int(abs(raw_output)) % len(actions)
                            suggested = actions[action_idx]
                            if suggested != draft_action:
                                # BN-10 Fix 7: Performance-based acceptance scaling
                                perf = self.skills.skill_performance_log.get(sk.id, [])
                                if perf and sum(perf) / len(perf) > 0:
                                    mean_perf = sum(perf) / len(perf)
                                    max_perf = max(abs(p) for p in perf) or 1.0
                                    acceptance_prob = min(0.9, 0.6 + 0.3 * mean_perf / max_perf)
                                else:
                                    acceptance_prob = 0.6
                                if random.random() < acceptance_prob:
                                    self._last_consulted_rsi_skill_id = sk.id
                                    self._rsi_skill_accepted = True
                                    # BN-10 Fix 4: Store raw skill output
                                    self._last_rsi_skill_output = float(raw_output)
                                    return suggested
                        except Exception:
                            pass
            except Exception:
                pass

        if random.random() > self.cfg.risk:
            return draft_action
        return random.choice(self.action_space())

    # ------------------------------------------------------------------
    # Skill synthesis
    # ------------------------------------------------------------------

    def maybe_synthesize_skill(self, obs: Dict[str, Any]) -> Optional[str]:
        task = obs.get("task", "")
        if task == "verification_pipeline" and random.random() < 0.30:
            sk = Skill(
                name=f"{self.cfg.name}_verify_pipeline",
                purpose="Evaluate candidate and write verified note if passing.",
                steps=[
                    SkillStep(kind="call", tool="evaluate_candidate",
                              args_template={"task": "${task}", "candidate": "${candidate}"}),
                    SkillStep(
                        kind="if",
                        condition={"key": "last_verdict", "op": "eq", "value": "pass"},
                        steps=[SkillStep(kind="call", tool="write_note",
                                        args_template={"title": "verified_result",
                                                       "payload": "${step_0}"})],
                        else_steps=[SkillStep(kind="call", tool="write_note",
                                             args_template={"title": "needs_revision",
                                                            "payload": "${step_0}"})],
                    ),
                ],
                tags=["verification", "meta"],
            )
            return self.skills.add(sk)
        if task == "toolchain_speedup" and random.random() < 0.30:
            sk = Skill(
                name=f"{self.cfg.name}_toolchain_upgrade",
                purpose="Propose toolchain improvement artifact for each hint.",
                steps=[
                    SkillStep(
                        kind="foreach", list_key="hint_titles", item_key="hint",
                        steps=[
                            SkillStep(kind="call", tool="tool_build_report",
                                      args_template={"task": "${task}",
                                                     "idea": {"hint": "${hint}"}}),
                            SkillStep(kind="call", tool="write_artifact",
                                      args_template={"title": "tool_artifact",
                                                     "payload": "${last}"}),
                        ],
                    )
                ],
                tags=["toolchain", "artifact"],
            )
            return self.skills.add(sk)
        return None

    # ------------------------------------------------------------------
    # Main act loop
    # ------------------------------------------------------------------

    def act_on_project(
        self,
        env: ResearchEnvironment,
        proj_node: ProjectNode,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute one step: plan → act → update world model → log."""
        # BN-09/10: Reset RSI skill tracking at start of every step
        self._last_consulted_rsi_skill_id = None
        self._rsi_skill_accepted = False
        self._last_rsi_skill_output = None
        # Self-model: check if we should attempt this task
        self._skip_requested = False
        if self.self_model is not None:
            task_proxy = type("T", (), {
                "domain": obs.get("domain", ""),
                "difficulty": obs.get("difficulty", 3),
                "baseline": obs.get("baseline", 0.3),
            })()
            should, reason = self.self_model.should_attempt(task_proxy)
            if not should:
                self._skip_requested = True
                return {
                    "agent": self.cfg.name, "role": self.cfg.role,
                    "project_id": proj_node.id, "project_name": proj_node.name,
                    "action": "skip", "reward": 0.0, "mem_id": "",
                    "info": {
                        "task": obs.get("task", ""),
                        "domain": obs.get("domain", ""),
                        "skip_reason": reason, "skipped": True,
                        "tq": env.global_tool_quality,
                        "kq": env.global_kb_quality,
                        "oq": env.global_org_quality,
                    },
                }

        hints = self.mem.search(
            f"{obs.get('task', '')} difficulty {obs.get('difficulty', 0)}",
            k=6,
            kinds=["principle", "artifact", "note"],
        )

        context = {
            "task": obs.get("task"),
            "domain": obs.get("domain"),
            "difficulty": obs.get("difficulty"),
            "budget": obs.get("budget"),
            "project": {"id": proj_node.id, "name": proj_node.name},
            "candidate": {
                "type": "proposal", "from": self.cfg.name,
                "role": self.cfg.role, "hints": [h.title for h in hints],
            },
            "idea": {
                "from": self.cfg.name,
                "summary": "incremental improvement on project using accumulated tools/kb/org.",
            },
            "hint_titles": [h.title for h in hints],
        }

        sid = self.maybe_synthesize_skill(obs)
        if sid:
            self.mem.add("artifact", f"skill_added:{sid}",
                         {"agent": self.cfg.name, "skill_id": sid}, tags=["skill"])

        action = self.choose_action(obs)
        invest = max(1.0, float(obs.get("budget", 1)) / 10.0)
        payload = {
            "invest": invest, "agent": self.cfg.name,
            "role": self.cfg.role, "task": obs.get("task"),
            "project_id": proj_node.id,
        }
        # BN-10 Fix 4: Pass raw RSI skill output to env for skill_bonus
        if self._last_rsi_skill_output is not None:
            payload["rsi_skill_output"] = self._last_rsi_skill_output

        next_obs, reward, info = env.step(obs, action, payload)

        # BN-09 Fix 2: Log actual env.step() reward for consulted RSI skill
        if self._last_consulted_rsi_skill_id is not None and self._rsi_skill_accepted:
            self.skills.log_skill_performance(self._last_consulted_rsi_skill_id, reward)

        # BN-06: record transition before world model update
        self._record_transition(obs, action, reward)

        # Intrinsic motivation blend
        combined_reward = reward
        intrinsic_val = 0.0
        if self.intrinsic_motivation is not None:
            outcome = {"reward": reward, "action": action, "info": info}
            intrinsic_val = self.intrinsic_motivation.total_intrinsic_reward(
                obs, action, outcome)
            combined_reward = (self.cfg.extrinsic_weight * reward
                               + self.cfg.intrinsic_weight * intrinsic_val)

        self.wm.update(obs, action, combined_reward, next_obs, self.action_space())

        # Self-model update
        if self.self_model is not None:
            result_entry = {"domain": obs.get("domain", ""), "reward": reward,
                            "action": action, "info": info}
            self.self_model.update(result_entry)
            self.self_model.record_actual(reward)
            if reward < 0.3:
                task_proxy = type("T", (), {
                    "domain": obs.get("domain", ""),
                    "difficulty": obs.get("difficulty", 3),
                    "baseline": obs.get("baseline", 0.3),
                })()
                self.self_model.diagnose_failure(task_proxy, result_entry)

        # Concept formation
        domain = str(obs.get("domain", ""))
        difficulty = int(obs.get("difficulty", 3))
        if self.concept_graph is not None and reward > 0.05:
            concept_ctx = {"domain": domain, "difficulty": difficulty,
                           "action": action, "reward": reward}
            cid = self.concept_graph.add_concept(
                name=f"{action}@{domain}", level=0, children=[],
                context=concept_ctx, creation_round=0)
            self.concept_graph.record_usage(cid, reward, concept_ctx, success=True)
            self.mem.add("note", f"concept_formed:{cid}",
                         {"concept_id": cid, "action": action, "domain": domain},
                         tags=["concept"])

        mem_id = self.mem.add(
            "episode",
            f"{self.cfg.name}:{action}:{obs.get('task')}:{proj_node.name}",
            {
                "obs": obs, "action": action, "payload": payload,
                "reward": reward, "intrinsic_reward": intrinsic_val,
                "combined_reward": combined_reward, "info": info,
                "project_id": proj_node.id,
                "hints_used": [h.id for h in hints],
                # BN-06: log calibration state for diagnostics
                "calibration_error": self.calibration_error(),
                "planner_depth_used": _meta_depth_ceiling(
                    self._transition_log, self.cfg.role, self.cfg.planner_depth),
            },
            tags=["episode", self.cfg.role, obs.get("task", "task")],
        )

        if random.random() < 0.35:
            tag = "verification" if action == "write_verified_note" else "toolchain"
            candidates = self.skills.list(tag=tag)
            if candidates:
                sk = random.choice(candidates)
                out = sk.run(self.tools, context)
                self.mem.add("note", f"{self.cfg.name}:skill_run:{sk.name}",
                             {"skill_id": sk.id, "out": out},
                             tags=["skill_run", tag])

        return {
            "agent": self.cfg.name, "role": self.cfg.role,
            "project_id": proj_node.id, "project_name": proj_node.name,
            "action": action, "reward": reward, "mem_id": mem_id, "info": info,
            "rsi_skill_used": self._last_consulted_rsi_skill_id,
        }
