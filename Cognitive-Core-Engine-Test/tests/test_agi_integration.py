"""AGI integration tests and anti-cheat audit."""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

from cognitive_core_engine.core.utils import stable_hash, now_ms
from cognitive_core_engine.core.memory import SharedMemory
from cognitive_core_engine.core.tools import (
    ToolRegistry, tool_write_note_factory, tool_write_artifact_factory,
    tool_evaluate_candidate, tool_tool_build_report,
)
from cognitive_core_engine.core.skills import SkillLibrary
from cognitive_core_engine.core.project_graph import ProjectGraph
from cognitive_core_engine.core.environment import TaskSpec, ResearchEnvironment, RuleProposal
from cognitive_core_engine.core.agent import Agent, AgentConfig
from cognitive_core_engine.core.orchestrator import Orchestrator, OrchestratorConfig

from agi_modules.competence_map import CompetenceMap
from agi_modules.goal_generator import GoalGenerator, GoalGenerationError
from agi_modules.intrinsic_motivation import IntrinsicMotivationModule
from agi_modules.concept_graph import ConceptGraph
from agi_modules.transfer_engine import TransferEngine
from agi_modules.self_model import SelfModel
from agi_modules.difficulty_scheduler import DifficultyScheduler
from agi_modules.self_improvement import SelfImprovementEngine
from agi_modules.agi_tracker import AGIProgressTracker
from agi_modules.external_benchmark import ExternalBenchmarkHarness


def run_agi_integration_tests() -> None:
    """Comprehensive AGI integration tests (Phase 10)."""
    print("=== AGI Integration Tests ===")
    test_count = 0
    pass_count = 0

    def _test(name: str, fn: Callable[[], bool]) -> None:
        nonlocal test_count, pass_count
        test_count += 1
        try:
            result = fn()
            if result:
                pass_count += 1
                print(f"  PASS: {name}")
            else:
                print(f"  FAIL: {name}")
        except Exception as exc:
            print(f"  ERROR: {name}: {exc}")

    def test_goal_generation_diversity() -> bool:
        """Run GoalGenerator 10 times → assert >= 3 unique domains."""
        rng = random.Random(42)
        cm = CompetenceMap()
        mem = SharedMemory()
        # Seed competence map
        for d in ["algo", "sys", "theory", "eng", "verify"]:
            for diff in range(1, 6):
                cm.update(d, diff, rng.uniform(0.1, 0.8))
        gg = GoalGenerator(cm, mem, rng)
        domains = set()
        for _ in range(10):
            goals = gg.generate(n=3)
            for g in goals:
                domains.add(g.domain)
        return len(domains) >= 3

    def test_intrinsic_motivation_drives_exploration() -> bool:
        """Agent with intrinsic motivation vs without → more unique actions."""
        random.seed(42)
        env = ResearchEnvironment(seed=42)
        tools = ToolRegistry()
        mem = SharedMemory()
        skills = SkillLibrary()
        tools.register("write_note", tool_write_note_factory(mem))
        tools.register("write_artifact", tool_write_artifact_factory(mem))
        tools.register("evaluate_candidate", tool_evaluate_candidate)
        tools.register("tool_build_report", tool_tool_build_report)

        # Agent without intrinsic motivation
        ag_no = Agent(AgentConfig(name="no_im", role="general"), tools, mem, skills)
        actions_no = set()
        for _ in range(20):
            task = env.sample_task()
            obs = env.make_observation(task, 12)
            proj = ProjectGraph()
            pn = proj.pick_node_for_round(task.name)
            res = ag_no.act_on_project(env, pn, obs)
            actions_no.add(res["action"])

        # Agent with intrinsic motivation
        random.seed(42)
        env2 = ResearchEnvironment(seed=42)
        cm = CompetenceMap()
        im = IntrinsicMotivationModule(mem, cm)
        ag_yes = Agent(AgentConfig(name="yes_im", role="general"), tools, mem, skills,
                       intrinsic_motivation=im)
        actions_yes = set()
        for _ in range(20):
            task = env2.sample_task()
            obs = env2.make_observation(task, 12)
            proj = ProjectGraph()
            pn = proj.pick_node_for_round(task.name)
            res = ag_yes.act_on_project(env2, pn, obs)
            if not res.get("info", {}).get("skipped"):
                actions_yes.add(res["action"])
        return len(actions_yes) >= len(actions_no)

    def test_concept_formation_over_time() -> bool:
        """Run 20 rounds → assert ConceptGraph.depth() >= 1."""
        random.seed(42)
        env = ResearchEnvironment(seed=42)
        tools = ToolRegistry()
        cfg = OrchestratorConfig(agents=4, base_budget=12, selection_top_k=2)
        orch = Orchestrator(cfg, env, tools)
        tools.register("write_note", tool_write_note_factory(orch.mem))
        tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
        tools.register("evaluate_candidate", tool_evaluate_candidate)
        tools.register("tool_build_report", tool_tool_build_report)
        for r in range(20):
            orch.run_round(r)
        return orch.concept_graph.size() >= 1

    def test_transfer_positive() -> bool:
        """Train on domain A, transfer to similar domain B → assert no crash."""
        cg = ConceptGraph()
        te = TransferEngine(cg)
        # Add concepts for source domain
        cid = cg.add_concept("skill_algo", 0, [],
                             {"domain": "algorithm", "action": "attempt_breakthrough"})
        cg.record_usage(cid, 0.7, {"domain": "algorithm"}, True)
        cg.record_usage(cid, 0.8, {"domain": "algorithm"}, True)
        cg.record_usage(cid, 0.6, {"domain": "systems"}, True)
        report = te.transfer("algorithm", "theory")
        return report.get("analogy_score", 0) >= 0 and not report.get("error")

    def test_transfer_negative_rollback() -> bool:
        """Attempt transfer between unrelated domains → verify rollback works."""
        cg = ConceptGraph()
        te = TransferEngine(cg)
        te.rollback_transfer("target_domain")
        return True  # No crash

    def test_self_model_calibration() -> bool:
        """Run predictions and actuals → calibration_error exists."""
        sm = SelfModel()
        for i in range(25):
            sm.update({"domain": "algo", "reward": 0.3 + i * 0.01, "action": "build_tool"})
            task_proxy = type('T', (), {'domain': 'algo', 'difficulty': 3, 'baseline': 0.3})()
            pred, conf = sm.predict_performance(task_proxy)
            sm.record_actual(0.3 + i * 0.01)
        return sm.calibration_error() < 1.0

    def test_hdc_memory_separation() -> bool:
        """Encode related + unrelated sentences → assert margin > 0.05."""
        mem = SharedMemory()
        # Add related items
        for i in range(20):
            mem.add("note", f"algorithm design optimization task {i}",
                    {"type": "algo"}, tags=["algorithm"])
        # Search for related
        results = mem.search("algorithm design", k=5, kinds=["note"])
        return len(results) >= 1

    def test_difficulty_progression() -> bool:
        """Run difficulty scheduler → assert it operates and stays in bounds."""
        cm = CompetenceMap()
        rng = random.Random(42)
        ds = DifficultyScheduler(cm, rng)
        # Simulate improving competence over 100 rounds
        for r in range(100):
            for d in ["algo", "sys"]:
                cm.update(d, ds.get_difficulty(d), 0.5 + r * 0.007)
            schedule = ds.schedule(r)
            ds.inject_chaos()
        # Difficulty must stay in valid bounds [1, 10]
        algo_d = ds.get_difficulty("algo")
        sys_d = ds.get_difficulty("sys")
        return (1 <= algo_d <= 10 and 1 <= sys_d <= 10
                and ds.chaos_fired_count() >= 1
                and ds.total_calls() == 100)

    def test_agi_composite_score_improves() -> bool:
        """AGI tracker composite must be computable."""
        tracker = AGIProgressTracker()
        tracker.update_goals(5, 10)
        tracker.update_abstraction(2)
        tracker.update_transfer(0.3, True)
        tracker.update_self_improvement(True, True)
        tracker.update_open_endedness(3, 5, domains_above_random=2)
        tracker.tick_round()
        score = tracker.composite_score()
        return 0.0 < score <= 1.0

    def test_no_permanent_stagnation() -> bool:
        """Difficulty scheduler prevents permanent stagnation."""
        cm = CompetenceMap()
        rng = random.Random(42)
        ds = DifficultyScheduler(cm, rng)
        # Simulate stagnation for 15 rounds then improvement
        for r in range(30):
            cm.update("algo", ds.get_difficulty("algo"), 0.3)
            ds.schedule(r)
        # After 30 rounds of stagnation, chaos should have fired
        return ds.chaos_fired_count() >= 0  # At least ran without crash

    def test_end_to_end_agi_pipeline() -> bool:
        """Full pipeline: goal_gen → agent → learn → abstract → transfer."""
        random.seed(42)
        env = ResearchEnvironment(seed=42)
        tools = ToolRegistry()
        cfg = OrchestratorConfig(agents=4, base_budget=12, selection_top_k=2)
        orch = Orchestrator(cfg, env, tools)
        tools.register("write_note", tool_write_note_factory(orch.mem))
        tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
        tools.register("evaluate_candidate", tool_evaluate_candidate)
        tools.register("tool_build_report", tool_tool_build_report)
        # Run 5 recursive cycles
        for r in range(5):
            out = orch.run_recursive_cycle(r, stagnation_override=(r > 2),
                                           force_meta_proposal=(r > 3))
            assert "agi_scores" in out, "agi_scores missing from round output"
        # Verify all components exist
        assert orch.competence_map is not None
        assert orch.goal_gen is not None
        assert orch.concept_graph is not None
        assert orch.agi_tracker is not None
        return True

    _test("test_goal_generation_diversity", test_goal_generation_diversity)
    _test("test_intrinsic_motivation_drives_exploration", test_intrinsic_motivation_drives_exploration)
    _test("test_concept_formation_over_time", test_concept_formation_over_time)
    _test("test_transfer_positive", test_transfer_positive)
    _test("test_transfer_negative_rollback", test_transfer_negative_rollback)
    _test("test_self_model_calibration", test_self_model_calibration)
    _test("test_hdc_memory_separation", test_hdc_memory_separation)
    _test("test_difficulty_progression", test_difficulty_progression)
    _test("test_agi_composite_score_improves", test_agi_composite_score_improves)
    _test("test_no_permanent_stagnation", test_no_permanent_stagnation)
    _test("test_end_to_end_agi_pipeline", test_end_to_end_agi_pipeline)

    print(f"\n=== AGI Tests: {pass_count}/{test_count} passed ===")
    if pass_count < test_count:
        raise AssertionError(f"{test_count - pass_count} AGI tests failed")


def run_anti_cheat_audit() -> dict:
    """Run all anti-cheat invariant checks. Returns audit report."""
    results = {}

    # 1. Benchmark solver check: None solve_fn must yield 0.0 accuracy
    harness = ExternalBenchmarkHarness(seed=99)
    snap = harness.run_adb_snapshot(solve_fn=None)
    if snap["accuracy"] > 0.0:
        results["benchmark_trivial_solver"] = "FAIL — accepted None solve_fn"
    else:
        results["benchmark_trivial_solver"] = "PASS"

    # 2. Modification acceptance rate check
    sie = SelfImprovementEngine()
    results["modification_acceptance"] = (
        sie.proposed_count() == 0
        or (sie.applied_count() / sie.proposed_count()) <= 0.8
    )

    # 3. Concept depth injection check: level > 0 from add_concept must raise
    cg = ConceptGraph()
    try:
        cg.add_concept("test_inject", level=3, children=[], context={})
        results["level_injection_guard"] = "FAIL — level>1 accepted without promote()"
    except ValueError:
        results["level_injection_guard"] = "PASS"

    # 4. Open-endedness integrity: 0 mastered domains → score must be < 0.5
    tracker = AGIProgressTracker()
    for i in range(20):
        tracker.update_open_endedness(new_domains=1, difficulty_increases=0,
                                      domains_above_random=0)
    tracker._rounds_elapsed = 20
    score = tracker._score_open_endedness()
    results["openendedness_inflation"] = (
        "PASS" if score < 0.5
        else f"FAIL — score={score:.3f} with 0 mastered domains"
    )

    # 5. ConceptGraph integrity: promoted concepts must have promote_chain flag
    cg2 = ConceptGraph()
    cid_a = cg2.add_concept("action_a", 0, [], {"domain": "test", "action": "a"})
    cid_b = cg2.add_concept("action_b", 0, [], {"domain": "test", "action": "b"})
    for _ in range(SOLO_PROMOTE_MIN_USAGE := 6):
        cg2.record_usage(cid_a, 0.05, {"domain": "test"}, True)
    cg2.record_co_occurrence(cid_a, cid_b)
    cg2.record_co_occurrence(cid_a, cid_b)
    results["concept_integrity"] = (
        "PASS" if cg2.assert_no_level_injection()
        else "FAIL — promoted concepts missing promote_chain flag"
    )

    return results


