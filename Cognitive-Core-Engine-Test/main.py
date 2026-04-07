#!/usr/bin/env python3
"""
Cognitive Core Engine — Main entry point.

Usage:
  python main.py selftest
  python main.py audit
  python main.py --rounds 40 --agents 8
  python main.py benchmark --suite ADB_v1 --seed 0 --trials 20
  python main.py arc-benchmark --suite arc_agi2_public_eval --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys

from cognitive_core_engine.core.tools import (
    ToolRegistry, tool_write_note_factory, tool_write_artifact_factory,
    tool_evaluate_candidate, tool_tool_build_report,
)
from cognitive_core_engine.core.environment import ResearchEnvironment
from cognitive_core_engine.core.orchestrator import Orchestrator, OrchestratorConfig

from tests.test_selftest import run_full_system_selftest, run_contract_negative_tests
from tests.test_agi_integration import run_agi_integration_tests, run_anti_cheat_audit
from tests.test_benchmarks import run_benchmark_suite, run_arc_benchmark


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        print("=== Running selftest suite ===")
        run_full_system_selftest()
        print("--- core selftest passed ---")
        run_contract_negative_tests()
        print("--- negative contract tests passed ---")
        run_agi_integration_tests()
        print("=== ALL SELFTESTS PASSED ===")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "audit":
        report = run_anti_cheat_audit()
        all_pass = True
        for check, result in report.items():
            is_pass = "PASS" in str(result) or result is True
            status = "PASS" if is_pass else "FAIL"
            print(f"  {status}: {check}: {result}")
            if not is_pass:
                all_pass = False
        sys.exit(0 if all_pass else 1)

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        ap = argparse.ArgumentParser()
        ap.add_argument("benchmark")
        ap.add_argument("--suite", required=True)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--trials", type=int, default=20)
        args = ap.parse_args()
        result = run_benchmark_suite(args.suite, args.seed, args.trials)
        print(json.dumps(result, ensure_ascii=False))
        return

    if len(sys.argv) > 1 and sys.argv[1] == "arc-benchmark":
        ap = argparse.ArgumentParser()
        ap.add_argument("arc-benchmark")
        ap.add_argument("--suite", required=True)
        ap.add_argument("--seed", type=int, default=0)
        args = ap.parse_args()
        result = run_arc_benchmark(args.suite, args.seed)
        print(json.dumps(result, ensure_ascii=False))
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    env = ResearchEnvironment(seed=args.seed)
    tools = ToolRegistry()

    orch_cfg = OrchestratorConfig(
        agents=args.agents,
        base_budget=20,
        selection_top_k=max(3, args.agents // 2),
    )
    orch = Orchestrator(orch_cfg, env, tools)

    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    print("=== NON-RSI AGI CORE v5 (Neuro-Symbolic): RUN START ===")
    for r in range(args.rounds):
        out = orch.run_round(r)
        top = sorted(out["results"], key=lambda x: x["reward"], reverse=True)[:3]
        print(
            f"[Round {r:02d}] tasks={','.join(out['tasks']):<35} "
            f"tq={out['env']['tq']:.3f} kq={out['env']['kq']:.3f} oq={out['env']['oq']:.3f} "
            f"risk={out['policy']['risk']:.2f} infra={out['policy']['infra_focus']:.2f} "
            f"top_rewards={[round(x['reward'],4) for x in top]}"
        )

    print("=== RUN END ===")
    print("Recent memory summary:")
    for it in orch.mem.dump_summary(k=15):
        print(it)


if __name__ == "__main__":
    main()
