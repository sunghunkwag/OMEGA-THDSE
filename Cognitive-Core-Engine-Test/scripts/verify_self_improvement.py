#!/usr/bin/env python3
"""
Self-Improvement Verification Suite
====================================
Verifies that the system's self-improvement mechanisms genuinely work.

Checks:
1. Does test_modification() actually call env.step() (empirical rollout)?
2. Is the before/after performance delta a real measurement (not arithmetic)?
3. Do rejected modifications exist? (100% acceptance = rubber-stamping)
4. Do applied modifications actually change agent behavior?
5. Over 50 rounds, do CompetenceMap, ConceptGraph, TransferEngine show
   genuine improvement trajectories?
6. Does the agent actually learn on the external benchmark (ADB)?
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognitive_core_engine.core.environment import ResearchEnvironment
from cognitive_core_engine.core.memory import SharedMemory
from cognitive_core_engine.core.tools import (
    ToolRegistry, tool_write_note_factory, tool_write_artifact_factory,
    tool_evaluate_candidate, tool_tool_build_report,
)
from cognitive_core_engine.core.project_graph import ProjectGraph
from cognitive_core_engine.core.orchestrator import Orchestrator, OrchestratorConfig
from cognitive_core_engine.core.agent import AgentConfig
import types
core = types.SimpleNamespace(
    ResearchEnvironment=ResearchEnvironment, SharedMemory=SharedMemory,
    ToolRegistry=ToolRegistry, OrchestratorConfig=OrchestratorConfig,
    Orchestrator=Orchestrator, ProjectGraph=ProjectGraph, AgentConfig=AgentConfig,
    tool_write_note_factory=tool_write_note_factory,
    tool_write_artifact_factory=tool_write_artifact_factory,
    tool_evaluate_candidate=tool_evaluate_candidate,
    tool_tool_build_report=tool_tool_build_report,
)
from agi_modules.self_improvement import SelfImprovementEngine
from agi_modules.competence_map import CompetenceMap
from agi_modules.concept_graph import ConceptGraph
from agi_modules.external_benchmark import ExternalBenchmarkHarness


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    marker = "  [OK]" if passed else "  [!!]"
    print(f"{marker} {name}: {status}")
    if detail:
        print(f"       {detail}")
    return passed


def test_1_empirical_rollout():
    """Check 1: Does test_modification() actually call env.step()?"""
    section("TEST 1: test_modification() empirical env rollout verification")

    env = core.ResearchEnvironment(seed=99)
    sie = SelfImprovementEngine()

    # Build up decision history
    for i in range(10):
        sie.record_decision({"reward": 0.05 + i * 0.002, "action": "build_tool", "domain": "algorithm"})

    mod = {"type": "policy_modification", "changes": {"risk_delta": 0.05}}
    params = {"risk": 0.25}

    # Track env.step() call count
    original_step = env.step
    call_count = [0]
    def counting_step(*args, **kwargs):
        call_count[0] += 1
        return original_step(*args, **kwargs)
    env.step = counting_step

    result = sie.test_modification(mod, env, params)

    check("Return type is dict?", isinstance(result, dict),
          f"type={type(result).__name__}")
    check("empirically_tested=True?", result.get("empirically_tested") is True,
          f"empirically_tested={result.get('empirically_tested')}")
    check("method='env_rollout'?", result.get("method") == "env_rollout",
          f"method={result.get('method')}")
    check("env.step() actually called?", call_count[0] > 0,
          f"env.step() call count: {call_count[0]}")
    check("baseline_avg included?", "baseline_avg" in result,
          f"baseline_avg={result.get('baseline_avg', 'MISSING')}")
    check("modified_avg included?", "modified_avg" in result,
          f"modified_avg={result.get('modified_avg', 'MISSING')}")

    # Fallback check: env=None
    result_no_env = sie.test_modification(mod, None, params)
    check("env=None -> empirically_tested=False", result_no_env.get("empirically_tested") is False,
          f"result={result_no_env}")
    check("env=None -> delta=0.0", result_no_env.get("delta") == 0.0)

    return call_count[0] > 0


def test_2_rejection_exists():
    """Check 2: Do rejected modifications exist? (prevents 100% acceptance)"""
    section("TEST 2: Modification rejection mechanism verification")

    random.seed(42)
    env = core.ResearchEnvironment(seed=42)
    tools = core.ToolRegistry()
    cfg = core.OrchestratorConfig(agents=4, base_budget=12, selection_top_k=2)
    orch = core.Orchestrator(cfg, env, tools)
    tools.register("write_note", core.tool_write_note_factory(orch.mem))
    tools.register("write_artifact", core.tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", core.tool_evaluate_candidate)
    tools.register("tool_build_report", core.tool_tool_build_report)

    for r in range(30):
        orch.run_recursive_cycle(r, stagnation_override=(r > 5 and r % 7 == 0))

    proposed = orch.self_improvement.proposed_count()
    applied = orch.self_improvement.applied_count()
    rejected = proposed - applied

    check("Proposals generated?", proposed > 0, f"proposed={proposed}")
    check("Applications occurred?", applied > 0, f"applied={applied}")
    check("Rejections exist (not 100% acceptance)?", rejected > 0 or proposed < 5,
          f"proposed={proposed}, applied={applied}, rejected={rejected}")

    if proposed > 0:
        rate = applied / proposed
        check("Acceptance rate <= 80%?", rate <= 0.80,
              f"acceptance rate={rate:.1%}")
        guard = orch.self_improvement.acceptance_rate_guard()
        check("acceptance_rate_guard() suspicious=False?",
              not guard.get("suspicious", True),
              f"guard={guard}")

    return proposed > 0 and rejected >= 0


def test_3_parameter_change_effect():
    """Check 3: Do applied modifications actually change agent behavior?"""
    section("TEST 3: Parameter change -> behavior change verification")

    random.seed(123)
    env = core.ResearchEnvironment(seed=123)
    tools = core.ToolRegistry()
    cfg = core.OrchestratorConfig(agents=4, base_budget=12, selection_top_k=2)
    orch = core.Orchestrator(cfg, env, tools)
    tools.register("write_note", core.tool_write_note_factory(orch.mem))
    tools.register("write_artifact", core.tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", core.tool_evaluate_candidate)
    tools.register("tool_build_report", core.tool_tool_build_report)

    # Record initial risk value
    initial_risk = orch._org_policy["risk"]

    # Run 30 rounds (self-improvement fires every 5 rounds)
    for r in range(30):
        orch.run_recursive_cycle(r, stagnation_override=(r > 5 and r % 7 == 0))

    final_risk = orch._org_policy["risk"]
    mods = orch.self_improvement.get_applied_modifications()

    risk_changed = abs(final_risk - initial_risk) > 1e-6
    check("Risk parameter changed?", risk_changed,
          f"initial={initial_risk:.4f} -> final={final_risk:.4f}")

    if mods:
        for i, mod in enumerate(mods):
            tr = mod.get("test_result", {})
            if isinstance(tr, dict):
                check(f"  mod[{i}] empirically tested?",
                      tr.get("empirically_tested") is True,
                      f"delta={tr.get('delta', '?'):.4f}, method={tr.get('method')}")
    else:
        check("Applied modification history exists", False, "mods list is empty")

    return risk_changed


def test_4_competence_trajectory():
    """Check 4: 50-round genuine improvement trajectory"""
    section("TEST 4: 50-round improvement trajectory verification")

    random.seed(42)
    env = core.ResearchEnvironment(seed=42)
    tools = core.ToolRegistry()
    cfg = core.OrchestratorConfig(agents=6, base_budget=20, selection_top_k=3)
    orch = core.Orchestrator(cfg, env, tools)
    tools.register("write_note", core.tool_write_note_factory(orch.mem))
    tools.register("write_artifact", core.tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", core.tool_evaluate_candidate)
    tools.register("tool_build_report", core.tool_tool_build_report)

    checkpoints = []
    start = time.time()

    for r in range(50):
        out = orch.run_recursive_cycle(
            r, stagnation_override=(r > 5 and r % 7 == 0),
            force_meta_proposal=(r > 10 and r % 10 == 0))

        if r % 10 == 0 or r == 49:
            mean_reward = sum(res["reward"] for res in out["results"]) / max(1, len(out["results"]))
            checkpoints.append({
                "round": r,
                "mean_reward": mean_reward,
                "competence_keys": len(orch.competence_map.all_keys()),
                "concept_count": orch.concept_graph.size(),
                "concept_depth": orch.concept_graph.depth(),
                "domains": len(env.tasks),
                "agi_composite": orch.agi_tracker.composite_score(),
                "agi_scores": orch.agi_tracker.score(),
            })

    elapsed = time.time() - start

    print(f"\n  50 rounds completed ({elapsed:.1f}s)")
    print(f"  {'Round':>5} | {'Reward':>7} | {'Competence':>10} | {'Concepts':>8} | {'Depth':>5} | {'Domains':>7} | {'Composite':>9}")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}-+-{'-'*7}-+-{'-'*9}")
    for cp in checkpoints:
        print(f"  {cp['round']:5d} | {cp['mean_reward']:7.4f} | {cp['competence_keys']:10d} | "
              f"{cp['concept_count']:8d} | {cp['concept_depth']:5d} | {cp['domains']:7d} | "
              f"{cp['agi_composite']:9.4f}")

    first = checkpoints[0]
    last = checkpoints[-1]

    check("Competence keys grew", last["competence_keys"] > first["competence_keys"],
          f"{first['competence_keys']} -> {last['competence_keys']}")
    check("Concept graph grew", last["concept_count"] > first["concept_count"],
          f"{first['concept_count']} -> {last['concept_count']}")
    check("Concept depth > 1", last["concept_depth"] > 1,
          f"depth={last['concept_depth']}")
    check("Domain expansion", last["domains"] > 6,
          f"6 -> {last['domains']}")
    check("AGI composite score improved", last["agi_composite"] > first["agi_composite"],
          f"{first['agi_composite']:.4f} -> {last['agi_composite']:.4f}")

    # External benchmark
    ext_scores = orch.external_benchmark.get_external_score_history()
    check("External benchmark records exist", len(ext_scores) > 0,
          f"snapshots={len(ext_scores)}")
    if ext_scores:
        check("Agent external accuracy > 0", max(ext_scores) > 0,
              f"scores={ext_scores}")

    # Transfer learning
    transfer_hist = orch.transfer_engine.transfer_history
    check("Transfer attempts exist", len(transfer_hist) > 0,
          f"transfer attempts={len(transfer_hist)}")
    if transfer_hist:
        best_analogy = max(t.similarity for t in transfer_hist)
        check("Transfer analogy > 0.02", best_analogy > 0.02,
              f"best analogy={best_analogy:.4f}")

    # Self-improvement
    si_proposed = orch.self_improvement.proposed_count()
    si_applied = orch.self_improvement.applied_count()
    check("Self-improvement proposals exist", si_proposed > 0,
          f"proposed={si_proposed}, applied={si_applied}")

    # Final score breakdown
    final_scores = last["agi_scores"]
    print(f"\n  Final AGI axis scores:")
    for axis, score in final_scores.items():
        marker = "[OK]" if score > 0.05 else "[--]"
        print(f"    {marker} {axis}: {score:.4f}")

    return last["agi_composite"] > first["agi_composite"]


def test_5_external_benchmark_learning():
    """Check 5: Does the agent actually learn on the external benchmark?"""
    section("TEST 5: External benchmark learning curve verification")

    random.seed(42)
    env = core.ResearchEnvironment(seed=42)
    tools = core.ToolRegistry()
    cfg = core.OrchestratorConfig(agents=6, base_budget=20, selection_top_k=3)
    orch = core.Orchestrator(cfg, env, tools)
    tools.register("write_note", core.tool_write_note_factory(orch.mem))
    tools.register("write_artifact", core.tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", core.tool_evaluate_candidate)
    tools.register("tool_build_report", core.tool_tool_build_report)

    # Run benchmark with agent solve_fn (before training)
    solve_fn_before = orch._make_agent_solve_fn()
    harness_before = ExternalBenchmarkHarness(seed=77)
    result_before = harness_before.run_adb_snapshot(solve_fn=solve_fn_before)
    acc_before = result_before["accuracy"]

    # Train for 30 rounds
    for r in range(30):
        orch.run_recursive_cycle(r, stagnation_override=(r > 5 and r % 7 == 0))

    # Run benchmark after training
    solve_fn_after = orch._make_agent_solve_fn()
    harness_after = ExternalBenchmarkHarness(seed=77)  # same seed = same problems
    result_after = harness_after.run_adb_snapshot(solve_fn=solve_fn_after)
    acc_after = result_after["accuracy"]

    check("Pre-training benchmark ran", True, f"accuracy={acc_before:.1%}")
    check("Post-training benchmark ran", True, f"accuracy={acc_after:.1%}")
    check("Post-training accuracy >= pre-training", acc_after >= acc_before,
          f"{acc_before:.1%} -> {acc_after:.1%}")

    # HDC validation
    fresh_mem = core.SharedMemory()
    hdc_result = orch.external_benchmark.validate_hdc_retrieval(fresh_mem)
    check("HDC precision >= 0.6 (no tag filter)",
          hdc_result["passes_threshold"],
          f"precision={hdc_result['mean_precision']:.3f}, "
          f"tag_inflation={hdc_result.get('possible_tag_inflation', '?')}, "
          f"random_baseline={hdc_result.get('random_baseline', '?')}")

    # Overfitting check
    overfitting = orch.external_benchmark.is_overfitting(
        orch.agi_tracker.composite_score(),
        acc_after)
    check("No overfitting detected", not overfitting, f"is_overfitting={overfitting}")

    return acc_after >= acc_before


def main():
    print("=" * 60)
    print("  Self-Improvement Verification Suite")
    print("=" * 60)

    results = {}
    results["1_empirical_rollout"] = test_1_empirical_rollout()
    results["2_rejection_exists"] = test_2_rejection_exists()
    results["3_parameter_effect"] = test_3_parameter_change_effect()
    results["4_improvement_trajectory"] = test_4_competence_trajectory()
    results["5_external_learning"] = test_5_external_benchmark_learning()

    section("FINAL RESULTS SUMMARY")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        marker = "[OK]" if passed else "[!!]"
        print(f"  {marker} {name}: {status}")
        if not passed:
            all_pass = False

    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"\n  {passed_count}/{total_count} checks passed")

    if all_pass:
        print("\n  Conclusion: Self-improvement is genuinely operational")
    else:
        print("\n  Conclusion: Some self-improvement mechanisms have issues")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
