#!/usr/bin/env python3
"""
Task 6: L0 solve evidence from AlgorithmSynthesisEnvironment.

Runs OmegaForgeV13 for 50 generations with pop_size=60, then submits
top-5 genomes to each L0 task (SUM, MAX, MIN, COUNT) and reports
holdout pass rates. Run across 5 seeds [0..4].

Why: provides honest evidence of what OmegaForge evolution can achieve
on real algorithmic tasks, without any formula-based proxy metrics.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognitive_core_engine.core.algorithm_env import AlgorithmSynthesisEnvironment
from cognitive_core_engine.core.environment import TaskSpec
from cognitive_core_engine.omega_forge.engine import OmegaForgeV13
from cognitive_core_engine.omega_forge.benchmark import EnvironmentCoupledFitness
from cognitive_core_engine.omega_forge.evidence import EngineConfig
from cognitive_core_engine.omega_forge.instructions import ProgramGenome


L0_TASKS = ["L0_SUM", "L0_MAX", "L0_MIN", "L0_COUNT"]


def run_seed(seed: int) -> dict:
    """Evolve genomes and evaluate on L0 tasks for one seed."""
    cfg = EngineConfig(pop_size=60, elite_keep=20, children_per_elite=3)
    engine = OmegaForgeV13(seed=seed, config=cfg)
    ef = EnvironmentCoupledFitness()
    engine.env_fitness = ef
    engine.init_population()

    for _ in range(50):
        engine.step()

    # Rank by score
    ranked = sorted(engine.population, key=lambda g: g.last_score, reverse=True)
    top5 = ranked[:5]

    env = AlgorithmSynthesisEnvironment(seed=seed)
    results = {}

    for task_name in L0_TASKS:
        task_results = []
        algo_task = env.get_algo_task(task_name)
        if algo_task is None:
            results[task_name] = {"best_holdout": 0.0, "genomes_tested": 0}
            continue

        for g in top5:
            obs = env.make_observation(
                TaskSpec(name=task_name, difficulty=1, baseline=0.3, domain="level0"),
                budget=10)
            _, reward, info = env.step(obs, "submit_program", {"genome": g})
            task_results.append({
                "gid": g.gid[:16],
                "holdout_rate": info.get("holdout_rate", 0.0),
            })

        best = max(task_results, key=lambda x: x["holdout_rate"])
        results[task_name] = {
            "best_holdout": best["holdout_rate"],
            "genomes_tested": len(task_results),
            "all_results": task_results,
        }

    solved = [name for name, r in results.items() if r["best_holdout"] >= 0.6]
    curriculum_level = env._curriculum.max_level

    return {
        "seed": seed,
        "results": results,
        "solved_tasks": solved,
        "curriculum_level": curriculum_level,
        "l1_unlocked": curriculum_level >= 1,
    }


def main():
    seeds = list(range(5))
    all_results = []

    print("=== L0 Solve Evidence from AlgorithmSynthesisEnvironment ===")
    print(f"Seeds: {seeds}")
    print(f"Generations: 50, Pop size: 60\n")

    start = time.time()
    for seed in seeds:
        t0 = time.time()
        result = run_seed(seed)
        elapsed = time.time() - t0
        all_results.append(result)

        print(f"  seed={seed}: solved={result['solved_tasks']} "
              f"L1_unlocked={result['l1_unlocked']} ({elapsed:.1f}s)")
        for task_name in L0_TASKS:
            r = result["results"][task_name]
            print(f"    {task_name}: best_holdout={r['best_holdout']:.3f}")

    total_time = time.time() - start

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Total time: {total_time:.1f}s")
    for task_name in L0_TASKS:
        rates = [r["results"][task_name]["best_holdout"] for r in all_results]
        mean_rate = sum(rates) / max(1, len(rates))
        solved_count = sum(1 for rate in rates if rate >= 0.6)
        print(f"  {task_name}: mean_best={mean_rate:.3f}, solved_in={solved_count}/{len(seeds)} seeds")

    total_solved = sum(len(r["solved_tasks"]) for r in all_results)
    l1_unlocked = sum(1 for r in all_results if r["l1_unlocked"])
    print(f"\n  Total L0 tasks solved: {total_solved}/{len(seeds) * 4}")
    print(f"  Seeds with L1 unlocked: {l1_unlocked}/{len(seeds)}")

    if total_solved == 0:
        print("\n=== Why L0 is Hard ===")
        print("  OmegaForge evolves VM programs via random mutation + crossover.")
        print("  L0 tasks require programs that correctly accumulate values in")
        print("  memory loops — a specific control flow pattern that random search")
        print("  finds with low probability in 50 generations. The evolutionary")
        print("  pressure from TaskBenchmark guides toward correct output values,")
        print("  but the exact tolerance (1e-6) on holdout cases is much stricter")
        print("  than the training signal (partial credit). Longer evolution or")
        print("  smarter initialization could close this gap.")

    # Save
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "algo_evidence.json"
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {log_path}")


if __name__ == "__main__":
    main()
