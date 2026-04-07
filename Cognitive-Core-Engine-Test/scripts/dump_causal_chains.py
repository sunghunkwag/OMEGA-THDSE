#!/usr/bin/env python3
"""
Task 7: Detailed causal chain dump.

Runs 30 rounds of run_recursive_cycle() with seed=12345 (the seed that
produced depth-4 chains in BN-10 testing), then dumps all causal chains
of depth >= 2 with full event details and verification status.

Why: provides transparent evidence of recursive self-improvement chains
by showing the actual events, not just aggregate counts.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognitive_core_engine.core.environment import ResearchEnvironment
from cognitive_core_engine.core.tools import (
    ToolRegistry, tool_write_note_factory, tool_write_artifact_factory,
    tool_evaluate_candidate, tool_tool_build_report,
)
from cognitive_core_engine.core.orchestrator import Orchestrator, OrchestratorConfig


def main():
    seed = 12345
    rounds = 30

    print(f"=== Causal Chain Dump (seed={seed}, {rounds} rounds) ===\n")

    random.seed(seed)
    env = ResearchEnvironment(seed=seed)
    tools = ToolRegistry()
    cfg = OrchestratorConfig(agents=4, base_budget=15, selection_top_k=2)
    orch = Orchestrator(cfg, env, tools)
    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    start = time.time()
    for r in range(rounds):
        orch.run_recursive_cycle(
            r,
            stagnation_override=(r > 2 and r % 6 == 0),
            force_meta_proposal=(r > 10 and r % 15 == 0),
        )
    elapsed = time.time() - start

    tracker = orch.causal_tracker
    chains = tracker.chains
    events_by_id = tracker._events_by_id

    # Print all chains of depth >= 2
    depth_counts = {2: 0, 3: 0, "4+": 0}
    verified_count = 0
    total_chains = len(chains)

    print(f"Total events: {len(tracker.events)}")
    print(f"Total chains: {total_chains}")
    print(f"Skill births: {tracker.skill_birth_count()}")
    print(f"Goals created: {tracker.goal_created_count()}")
    print()

    for chain_id, event_ids in sorted(chains.items()):
        depth = len(event_ids)
        if depth < 2:
            continue

        verified = tracker.verify_chain(chain_id)
        if verified:
            verified_count += 1

        if depth == 2:
            depth_counts[2] += 1
        elif depth == 3:
            depth_counts[3] += 1
        else:
            depth_counts["4+"] += 1

        print(f"--- Chain {chain_id} (depth={depth}, verified={verified}) ---")
        for eid in event_ids:
            event = events_by_id.get(eid)
            if event is None:
                print(f"  [MISSING EVENT: {eid}]")
                continue

            line = f"  [{event.event_type}] round={event.timestamp_round}"
            data = event.data

            if event.event_type == "skill_born":
                line += f" skill_id={data.get('skill_id', '?')}"
                line += f" fitness={data.get('genome_fitness', 0):.3f}"
            elif event.event_type == "goal_created":
                line += f" goal={data.get('goal_name', '?')}"
                line += f" trigger={data.get('trigger_skill_id', '?')}"
            elif event.event_type == "goal_achieved":
                line += f" goal={data.get('goal_name', '?')}"
                line += f" reward={data.get('reward', 0):.3f}"
                line += f" skills={data.get('contributing_skill_ids', [])}"
            elif event.event_type == "skill_used":
                line += f" skill={data.get('skill_id', '?')}"
                line += f" accepted={data.get('was_accepted', False)}"
                line += f" reward={data.get('reward', 0):.3f}"
            elif event.event_type == "level_unlocked":
                line += f" level={data.get('level', '?')}"
            elif event.event_type == "program_submitted":
                line += f" task={data.get('task_name', '?')}"
                line += f" reward={data.get('reward', 0):.3f}"
            elif event.event_type == "challenge_created":
                line += f" challenge={data.get('challenge_name', '?')}"

            if event.cause_event_id:
                line += f" caused_by={event.cause_event_id}"
            print(line)
        print()

    # Summary
    print(f"=== Summary ===")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Max chain depth: {tracker.max_chain_depth()}")
    print(f"  Chains depth 2: {depth_counts[2]}")
    print(f"  Chains depth 3: {depth_counts[3]}")
    print(f"  Chains depth 4+: {depth_counts['4+']}")
    deep_chains = sum(1 for _, eids in chains.items() if len(eids) >= 2)
    print(f"  Verification pass rate: {verified_count}/{deep_chains}")

    if tracker.max_chain_depth() < 3:
        print("\n  Note: No depth-3+ chains in this run. Depth-3 requires:")
        print("  skill_born → goal_created → goal_achieved, which needs")
        print("  the skill-derived goal to be attempted AND produce reward > 0.1")

    # Save
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    dump_data = {
        "seed": seed,
        "rounds": rounds,
        "total_events": len(tracker.events),
        "total_chains": total_chains,
        "max_depth": tracker.max_chain_depth(),
        "depth_counts": depth_counts,
        "verification_rate": f"{verified_count}/{deep_chains}" if deep_chains else "N/A",
    }
    log_path = log_dir / "causal_chain_dump.json"
    with open(log_path, "w") as f:
        json.dump(dump_data, f, indent=2)
    print(f"\n  Summary saved to {log_path}")


if __name__ == "__main__":
    main()
