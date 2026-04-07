"""CLI commands and parser for unified governance module."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cognitive_core_engine.governance.utils import now_ms, sha256, safe_mkdir, read_json, write_json


def cmd_selftest(args):
    print("[selftest] Validating...")
    assert validate_expr("sin(x) + x*x")[0]
    assert not validate_expr("__import__('os')")[0]

    g = seed_genome(random.Random(42))
    t = TaskSpec()
    b = sample_batch(random.Random(42), t)
    assert b is not None
    r = evaluate(g, b, t.name)
    assert isinstance(r.score, float)

    hint = TaskDetective.detect_pattern(b)
    lg = seed_learner_genome(random.Random(42), hint)
    lr = evaluate_learner(lg, b, t.name)
    assert isinstance(lr.score, float)

    algo_code = "def run(inp):\n    return inp\n"
    assert validate_algo_program(algo_code)[0]

    print("[selftest] OK")
    return 0

def cmd_autopatch_probe(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    score = _autopatch_probe_score(
        mode=args.mode,
        task_name=args.task,
        seed=args.seed,
        generations=args.generations,
        population=args.population,
        universes=args.universes,
        freeze_eval=args.freeze_eval,
    )
    print(f"{score:.6f}")
    return 0

def cmd_evolve(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    resume = bool(args.resume) and (not args.fresh)
    mode = args.mode or ("algo" if args.task in ALGO_TASK_NAMES else "solver")
    run_multiverse(
        args.seed,
        TaskSpec(name=args.task),
        args.generations,
        args.population,
        args.universes,
        resume=resume,
        save_every=args.save_every,
        mode=mode,
        freeze_eval=args.freeze_eval,
    )
    print(f"\n[OK] State saved to {STATE_DIR / 'state.json'}")
    return 0

def cmd_learner_evolve(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    resume = bool(args.resume) and (not args.fresh)
    run_multiverse(
        args.seed,
        TaskSpec(name=args.task),
        args.generations,
        args.population,
        args.universes,
        resume=resume,
        save_every=args.save_every,
        mode="learner",
        freeze_eval=args.freeze_eval,
    )
    print(f"\n[OK] State saved to {STATE_DIR / 'state.json'}")
    return 0

def cmd_best(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    gs = load_state()
    if not gs:
        print("No state.")
        return 1
    u = next((s for s in gs.universes if s.get("uid") == gs.selected_uid), gs.universes[0] if gs.universes else {})
    best = u.get("best")
    if best:
        if gs.mode == "learner":
            g = LearnerGenome(**best)
        else:
            g = Genome(**best)
        print(g.code)
    print(f"Score: {u.get('best_score')} | Hold: {u.get('best_hold')} | Stress: {u.get('best_stress')} | Test: {u.get('best_test')}")
    print(f"Generations: {gs.generations_done}")
    return 0

def cmd_rsi_loop(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    levels = [int(l) for l in args.levels.split(",") if l.strip()]
    run_rsi_loop(
        args.generations,
        args.rounds,
        levels,
        args.population,
        args.universes,
        mode=args.mode,
        freeze_eval=args.freeze_eval,
        meta_meta=args.meta_meta,
        update_rule_rounds=args.update_rule_rounds,
    )
    return 0


def cmd_update_rule_search(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    run_update_rule_search(
        seed=args.seed,
        rounds=args.rounds,
        gens_per_round=args.generations,
        pop=args.population,
        freeze_eval=args.freeze_eval,
        state_dir=STATE_DIR,
    )
    return 0

def cmd_duo_loop(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    run_duo_loop(
        rounds=args.rounds,
        slice_seconds=args.slice_seconds,
        blackboard_path=Path(args.blackboard),
        k_full=args.k_full,
        seed=args.seed,
        mode=args.mode,
        freeze_eval=args.freeze_eval,
        population=args.population,
        max_candidates=args.max_candidates,
    )
    return 0

def cmd_meta_meta(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    run_meta_meta(
        seed=args.seed,
        episodes=args.episodes,
        gens_per_episode=args.gens_per_episode,
        pop=args.population,
        n_univ=args.universes,
        policy_pop=args.policy_pop,
        freeze_eval=args.freeze_eval,
        state_dir=STATE_DIR,
        eval_every=args.eval_every,
        few_shot_gens=args.few_shot_gens,
    )
    return 0

def cmd_task_switch(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    result = run_task_switch(
        seed=args.seed,
        task_a=TaskSpec(name=args.task_a),
        task_b=TaskSpec(name=args.task_b),
        gens_a=args.gens_a,
        gens_b=args.gens_b,
        pop=args.population,
        n_univ=args.universes,
        freeze_eval=args.freeze_eval,
        state_dir=STATE_DIR,
    )
    print(json.dumps(result, indent=2))
    return 0

def cmd_report(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    report = generate_report(STATE_DIR / "run_log.jsonl", args.few_shot_gens)
    print(json.dumps(report, indent=2))
    return 0


def cmd_transfer_bench(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    result = transfer_bench(args.task_from, args.task_to, args.budget, args.seed, freeze_eval=not args.no_freeze_eval)
    print(json.dumps(result, indent=2))
    return 0

def build_parser():
    p = argparse.ArgumentParser(prog="UNIFIED_RSI_EXTENDED", description="True RSI Engine with hard gates and rollback")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)

    e = sub.add_parser("evolve")
    e.add_argument("--seed", type=int, default=1337)
    e.add_argument("--generations", type=int, default=80)
    e.add_argument("--population", type=int, default=128)
    e.add_argument("--universes", type=int, default=4)
    e.add_argument("--task", default="poly2")
    e.add_argument("--resume", action="store_true")
    e.add_argument("--fresh", action="store_true")
    e.add_argument("--save-every", type=int, default=5)
    e.add_argument("--state-dir", default=".rsi_state")
    e.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    e.add_argument("--mode", default="", choices=["", "solver", "algo"])
    e.set_defaults(fn=cmd_evolve)

    le = sub.add_parser("learner-evolve")
    le.add_argument("--seed", type=int, default=1337)
    le.add_argument("--generations", type=int, default=80)
    le.add_argument("--population", type=int, default=128)
    le.add_argument("--universes", type=int, default=4)
    le.add_argument("--task", default="poly2")
    le.add_argument("--resume", action="store_true")
    le.add_argument("--fresh", action="store_true")
    le.add_argument("--save-every", type=int, default=5)
    le.add_argument("--state-dir", default=".rsi_state")
    le.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    le.set_defaults(fn=cmd_learner_evolve)

    b = sub.add_parser("best")
    b.add_argument("--state-dir", default=".rsi_state")
    b.set_defaults(fn=cmd_best)

    r = sub.add_parser("rsi-loop")
    r.add_argument("--generations", type=int, default=50)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--population", type=int, default=64)
    r.add_argument("--universes", type=int, default=2)
    r.add_argument("--state-dir", default=".rsi_state")
    r.add_argument("--mode", default="solver", choices=["solver", "learner", "algo"])
    r.add_argument("--levels", default="1,2,3", help="Comma-separated autopatch levels (e.g., 1,3)")
    r.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--meta-meta", action="store_true", help="Run meta-meta loop instead of standard RSI rounds")
    r.add_argument("--update-rule-rounds", type=int, default=0, help="Rounds of update-rule search per RSI round")
    r.set_defaults(fn=cmd_rsi_loop)

    dl = sub.add_parser("duo-loop")
    dl.add_argument("--rounds", type=int, default=5)
    dl.add_argument("--slice-seconds", type=float, default=0.0)
    dl.add_argument("--blackboard", default=".rsi_blackboard.jsonl")
    dl.add_argument("--k-full", type=int, default=6)
    dl.add_argument("--seed", type=int, default=1337)
    dl.add_argument("--mode", default="solver", choices=["solver", "algo", "program"])
    dl.add_argument("--population", type=int, default=64)
    dl.add_argument("--max-candidates", type=int, default=512)
    dl.add_argument("--state-dir", default=".rsi_state")
    dl.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    dl.set_defaults(fn=cmd_duo_loop)

    ap = sub.add_parser("autopatch-probe")
    ap.add_argument("--mode", default="solver", choices=["solver", "learner", "algo"])
    ap.add_argument("--task", default="poly2")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--universes", type=int, default=1)
    ap.add_argument("--state-dir", default=".rsi_state")
    ap.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    ap.set_defaults(fn=cmd_autopatch_probe)

    mm = sub.add_parser("meta-meta")
    mm.add_argument("--seed", type=int, default=1337)
    mm.add_argument("--episodes", type=int, default=20)
    mm.add_argument("--gens-per-episode", type=int, default=20)
    mm.add_argument("--population", type=int, default=64)
    mm.add_argument("--universes", type=int, default=2)
    mm.add_argument("--policy-pop", type=int, default=4)
    mm.add_argument("--state-dir", default=".rsi_state")
    mm.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    mm.add_argument("--eval-every", type=int, default=4)
    mm.add_argument("--few-shot-gens", type=int, default=10)
    mm.set_defaults(fn=cmd_meta_meta)

    ts = sub.add_parser("task-switch")
    ts.add_argument("--seed", type=int, default=1337)
    ts.add_argument("--task-a", default="poly2")
    ts.add_argument("--task-b", default="piecewise")
    ts.add_argument("--gens-a", type=int, default=10)
    ts.add_argument("--gens-b", type=int, default=10)
    ts.add_argument("--population", type=int, default=64)
    ts.add_argument("--universes", type=int, default=2)
    ts.add_argument("--state-dir", default=".rsi_state")
    ts.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    ts.set_defaults(fn=cmd_task_switch)

    tb = sub.add_parser("transfer-bench")
    tb.add_argument("--from", dest="task_from", required=True)
    tb.add_argument("--to", dest="task_to", required=True)
    tb.add_argument("--budget", type=int, default=12)
    tb.add_argument("--seed", type=int, default=1337)
    tb.add_argument("--state-dir", default=".rsi_state")
    tb.add_argument("--no-freeze-eval", action="store_true")
    tb.set_defaults(fn=cmd_transfer_bench)

    rp = sub.add_parser("report")
    rp.add_argument("--state-dir", default=".rsi_state")
    rp.add_argument("--few-shot-gens", type=int, default=10)
    rp.set_defaults(fn=cmd_report)

    inv = sub.add_parser("invention")
    inv.add_argument("--seed", type=int, default=0)
    inv.add_argument("--iterations", type=int, default=6)
    inv.set_defaults(fn=cmd_invention)

    ur = sub.add_parser("update-rule")
    ur.add_argument("--seed", type=int, default=1337)
    ur.add_argument("--rounds", type=int, default=4)
    ur.add_argument("--generations", type=int, default=6)
    ur.add_argument("--population", type=int, default=32)
    ur.add_argument("--state-dir", default=".rsi_state")
    ur.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    ur.set_defaults(fn=cmd_update_rule_search)

    return p

def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        sys.argv.append("selftest")
    args = parser.parse_args()
    return args.fn(args)

if __name__ == "__main__":
    raise SystemExit(main())
