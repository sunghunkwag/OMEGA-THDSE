#!/usr/bin/env python3
"""Reproduce evidence runs and write logs/ output."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognitive_core_engine.core.environment import ResearchEnvironment
from cognitive_core_engine.core.tools import (
    ToolRegistry, tool_write_note_factory, tool_write_artifact_factory,
    tool_evaluate_candidate, tool_tool_build_report,
)
from cognitive_core_engine.core.orchestrator import Orchestrator, OrchestratorConfig
import types
core = types.SimpleNamespace(
    ResearchEnvironment=ResearchEnvironment, ToolRegistry=ToolRegistry,
    OrchestratorConfig=OrchestratorConfig, Orchestrator=Orchestrator,
    tool_write_note_factory=tool_write_note_factory,
    tool_write_artifact_factory=tool_write_artifact_factory,
    tool_evaluate_candidate=tool_evaluate_candidate,
    tool_tool_build_report=tool_tool_build_report,
)
import cognitive_core_engine.omega_forge.cli as omega


def run_core_baseline(log_dir: Path) -> dict:
    seed = 11
    rounds = 30
    log_path = log_dir / "core_baseline.txt"

    random.seed(seed)
    env = core.ResearchEnvironment(seed=seed)
    tools = core.ToolRegistry()
    orch_cfg = core.OrchestratorConfig(
        agents=6,
        base_budget=20,
        selection_top_k=3,
    )
    orch = core.Orchestrator(orch_cfg, env, tools)

    tools.register("write_note", core.tool_write_note_factory(orch.mem))
    tools.register("write_artifact", core.tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", core.tool_evaluate_candidate)
    tools.register("tool_build_report", core.tool_tool_build_report)

    lines = []
    lines.append(f"seed={seed} rounds={rounds}")
    stagnation_rounds = 0
    for r in range(rounds):
        out = orch.run_round(r)
        orch._record_round_rewards(out["results"])
        mean_reward = orch._recent_rewards[-1]
        stagnation = orch._detect_stagnation(window=5, threshold=0.01)
        if stagnation:
            stagnation_rounds += 1
        top = sorted(out["results"], key=lambda x: x["reward"], reverse=True)[:3]
        lines.append(
            f"[Round {r:02d}] tasks={','.join(out['tasks']):<35} "
            f"mean_reward={mean_reward:.4f} stagnation={stagnation} "
            f"top_rewards={[round(x['reward'],4) for x in top]}"
        )

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"seed": seed, "rounds": rounds, "stagnation_rounds": stagnation_rounds}


def run_omega_evidence(log_dir: Path) -> dict:
    args = SimpleNamespace(
        out=str(log_dir / "omega_evidence.jsonl"),
        target=6,
        max_generations=2000,
        seed=42,
        report_every=50,
    )
    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()
    omega.cmd_evidence_run(args)

    evidence_lines = 0
    unique_cfg = set()
    loop_counts = Counter()
    scc_counts = Counter()
    with open(args.out, "r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry.get("type") != "evidence":
                continue
            evidence_lines += 1
            cfg_hash = entry.get("diag", {}).get("cfg_hash")
            if cfg_hash:
                unique_cfg.add(cfg_hash)
            loop_counts[entry.get("metrics", {}).get("loops")] += 1
            scc_counts[entry.get("diag", {}).get("scc_n")] += 1

    return {
        "seed": args.seed,
        "target": args.target,
        "evidence_lines": evidence_lines,
        "unique_cfg": len(unique_cfg),
        "loop_counts": dict(loop_counts),
        "scc_counts": dict(scc_counts),
    }


def run_critic_eval(log_dir: Path) -> dict:
    critic = core.load_unified_critic_module()
    log_path = log_dir / "critic_eval.jsonl"
    records = []

    random.seed(21)
    env = core.ResearchEnvironment(seed=21)
    tools = core.ToolRegistry()
    orch_cfg = core.OrchestratorConfig(agents=4, base_budget=12, selection_top_k=2)
    orch = core.Orchestrator(orch_cfg, env, tools)

    engine = omega.Stage1Engine(seed=123)
    engine.init_population()
    for _ in range(5):
        engine.step()
        if engine.candidates:
            break

    omega_candidates = engine.candidates[:1]

    for idx, cand in enumerate(omega_candidates):
        packet = {
            "proposal": {
                "proposal_id": f"omega_{idx}",
                "level": "L0",
                "payload": {"candidate": cand, "gap_spec": {"seed": 123}},
                "evidence": {"metrics": cand.get("metrics", {})},
            },
            "evaluation_rules": dict(orch.evaluation_rules),
            "invariants": dict(orch.invariants),
        }
        verdict = critic.critic_evaluate_candidate_packet(packet, invariants=orch.invariants)
        failed = [k for k, ok in verdict.items() if k.endswith("_ok") and ok is False]
        records.append(
            {
                "source": "omega_stage1",
                "proposal_id": packet["proposal"]["proposal_id"],
                "gid": cand.get("gid"),
                "verdict": verdict.get("verdict"),
                "guardrails_ok": verdict.get("guardrails_ok"),
                "failed_checks": failed,
                "verdict_detail": verdict,
            }
        )

    reject_candidate = {
        "gid": "missing_holdout",
        "metrics": {"train_pass_rate": 0.33},
    }
    packet = {
        "proposal": {
            "proposal_id": "reject_missing_holdout",
            "level": "L0",
            "payload": {"candidate": reject_candidate, "gap_spec": {"seed": 123}},
            "evidence": {"metrics": reject_candidate["metrics"]},
        },
        "evaluation_rules": dict(orch.evaluation_rules),
        "invariants": dict(orch.invariants),
    }
    verdict = critic.critic_evaluate_candidate_packet(packet, invariants=orch.invariants)
    failed = [k for k, ok in verdict.items() if k.endswith("_ok") and ok is False]
    records.append(
        {
            "source": "synthetic_guardrail",
            "proposal_id": packet["proposal"]["proposal_id"],
            "gid": reject_candidate.get("gid"),
            "verdict": verdict.get("verdict"),
            "guardrails_ok": verdict.get("guardrails_ok"),
            "failed_checks": failed,
            "verdict_detail": verdict,
        }
    )

    approved_candidate = {
        "gid": "synthetic_approve",
        "metrics": {
            "train_pass_rate": 0.37,
            "holdout_pass_rate": 0.36,
            "adversarial_pass_rate": 0.30,
            "distribution_shift": {"holdout_pass_rate": 0.31},
            "discovery_cost": {"holdout": 0.5},
        },
    }
    packet = {
        "proposal": {
            "proposal_id": "synthetic_pass",
            "level": "L0",
            "payload": {"candidate": approved_candidate, "gap_spec": {"seed": 123}},
            "evidence": {"metrics": approved_candidate["metrics"]},
        },
        "evaluation_rules": dict(orch.evaluation_rules),
        "invariants": dict(orch.invariants),
    }
    verdict = critic.critic_evaluate_candidate_packet(packet, invariants=orch.invariants)
    failed = [k for k, ok in verdict.items() if k.endswith("_ok") and ok is False]
    records.append(
        {
            "source": "synthetic",
            "proposal_id": packet["proposal"]["proposal_id"],
            "gid": approved_candidate.get("gid"),
            "verdict": verdict.get("verdict"),
            "guardrails_ok": verdict.get("guardrails_ok"),
            "failed_checks": failed,
            "verdict_detail": verdict,
        }
    )

    with log_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    verdict_counts = Counter(r["verdict"] for r in records)
    reject_reasons = Counter()
    for rec in records:
        if rec.get("verdict") == "reject":
            for reason in rec.get("failed_checks", []):
                reject_reasons[reason] += 1

    return {
        "verdict_counts": dict(verdict_counts),
        "reject_reasons": dict(reject_reasons),
    }


def run_end_to_end(log_dir: Path) -> dict:
    seed = 33
    rounds = 2
    log_path = log_dir / "blackboard.jsonl"

    random.seed(seed)
    env = core.ResearchEnvironment(seed=seed)
    tools = core.ToolRegistry()
    orch_cfg = core.OrchestratorConfig(agents=4, base_budget=12, selection_top_k=2)
    orch = core.Orchestrator(orch_cfg, env, tools)

    tools.register("write_note", core.tool_write_note_factory(orch.mem))
    tools.register("write_artifact", core.tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", core.tool_evaluate_candidate)
    tools.register("tool_build_report", core.tool_tool_build_report)

    with log_path.open("w", encoding="utf-8") as handle:
        for r in range(rounds):
            stagnation_override = True if r == 0 else None
            out = orch.run_recursive_cycle(r, stagnation_override=stagnation_override, force_meta_proposal=True)
            handle.write(
                json.dumps(
                    {
                        "event": "round_complete",
                        "round": r,
                        "stagnation": out.get("stagnation"),
                        "gap_spec": out.get("gap_spec"),
                        "critic_results": out.get("critic_results"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for item in out.get("critic_results", []):
                handle.write(
                    json.dumps(
                        {
                            "event": "critic_decision",
                            "round": r,
                            "proposal_id": item.get("proposal_id"),
                            "level": item.get("level"),
                            "verdict": item.get("verdict"),
                            "adopted": item.get("adopted"),
                            "result": "REGISTERED" if item.get("adopted") else "REJECTED",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    critic_events = 0
    levels = Counter()
    verdicts = Counter()
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry.get("event") != "critic_decision":
                continue
            critic_events += 1
            levels[entry.get("level")] += 1
            verdicts[entry.get("verdict")] += 1

    return {
        "rounds": rounds,
        "critic_events": critic_events,
        "levels": dict(levels),
        "verdicts": dict(verdicts),
    }


def main() -> int:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    core_metrics = run_core_baseline(log_dir)
    omega_metrics = run_omega_evidence(log_dir)
    critic_metrics = run_critic_eval(log_dir)
    e2e_metrics = run_end_to_end(log_dir)

    print("=== RESULTS SUMMARY ===")
    print(
        "Core baseline: seed={seed} rounds={rounds} stagnation_rounds={stagnation_rounds}".format(
            **core_metrics
        )
    )
    print(
        "Omega evidence: seed={seed} evidence_lines={evidence_lines} unique_cfg={unique_cfg}".format(
            **omega_metrics
        )
    )
    print(
        "Critic eval: verdict_counts={verdict_counts} reject_reasons={reject_reasons}".format(
            **critic_metrics
        )
    )
    print(
        "End-to-end: rounds={rounds} critic_events={critic_events} levels={levels} verdicts={verdicts}".format(
            **e2e_metrics
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
