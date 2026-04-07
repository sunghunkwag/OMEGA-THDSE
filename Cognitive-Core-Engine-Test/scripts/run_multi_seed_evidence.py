#!/usr/bin/env python3
"""
Task 5: Multi-seed statistical evidence.

Runs run_recursive_cycle() for 30 rounds across 20 seeds [0..19],
recording composite_score, skill_births, max_chain_depth,
domains_discovered, and self_improvement_applied_count per seed.

Why: provides statistical confidence that system behavior is not
seed-dependent. Mean ± std across seeds shows reproducibility.
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognitive_core_engine.core.environment import ResearchEnvironment
from cognitive_core_engine.core.tools import (
    ToolRegistry, tool_write_note_factory, tool_write_artifact_factory,
    tool_evaluate_candidate, tool_tool_build_report,
)
from cognitive_core_engine.core.orchestrator import Orchestrator, OrchestratorConfig


def run_single_seed(seed: int, rounds: int = 30) -> dict:
    """Run a complete independent trial with the given seed."""
    random.seed(seed)
    env = ResearchEnvironment(seed=seed)
    tools = ToolRegistry()
    cfg = OrchestratorConfig(agents=4, base_budget=15, selection_top_k=2)
    orch = Orchestrator(cfg, env, tools)
    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    for r in range(rounds):
        orch.run_recursive_cycle(
            r,
            stagnation_override=(r > 3 and r % 6 == 0),
            force_meta_proposal=(r > 10 and r % 15 == 0),
        )

    return {
        "seed": seed,
        "composite_score": orch.agi_tracker.composite_score(),
        "skill_births": orch.causal_tracker.skill_birth_count(),
        "max_chain_depth": orch.causal_tracker.max_chain_depth(),
        "domains_discovered": len(env.tasks) - 6,
        "self_improvement_applied": orch.self_improvement.applied_count(),
    }


def main():
    seeds = list(range(20))
    results = []
    crashes = 0

    print("=== Multi-Seed Statistical Evidence ===")
    print(f"Seeds: {seeds}")
    print(f"Rounds per seed: 30\n")

    start = time.time()
    for seed in seeds:
        try:
            t0 = time.time()
            result = run_single_seed(seed)
            elapsed = time.time() - t0
            result["elapsed"] = elapsed
            result["crashed"] = False
            results.append(result)
            print(f"  seed={seed:2d}: composite={result['composite_score']:.4f} "
                  f"births={result['skill_births']} depth={result['max_chain_depth']} "
                  f"domains={result['domains_discovered']} ({elapsed:.1f}s)")
        except Exception as e:
            crashes += 1
            results.append({"seed": seed, "crashed": True, "error": str(e)})
            print(f"  seed={seed:2d}: CRASHED — {e}")
            traceback.print_exc()

    total_time = time.time() - start
    valid = [r for r in results if not r.get("crashed")]

    # Compute statistics
    def stats(key):
        vals = [r[key] for r in valid]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
        std = var ** 0.5
        return mean, std

    print(f"\n=== Summary ({len(valid)}/{len(seeds)} seeds completed, {crashes} crashes) ===")
    print(f"  Total time: {total_time:.1f}s")
    for metric in ["composite_score", "skill_births", "max_chain_depth",
                    "domains_discovered", "self_improvement_applied"]:
        m, s = stats(metric)
        print(f"  {metric:30s}: {m:.4f} ± {s:.4f}")

    # Save to JSON
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "multi_seed_evidence.json"
    with open(log_path, "w") as f:
        json.dump({"seeds": seeds, "results": results, "total_time": total_time,
                    "crashes": crashes}, f, indent=2, default=str)
    print(f"\n  Results saved to {log_path}")


if __name__ == "__main__":
    main()
