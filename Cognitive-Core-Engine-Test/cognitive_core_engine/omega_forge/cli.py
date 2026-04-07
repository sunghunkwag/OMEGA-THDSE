"""CLI commands and entry points for omega_forge."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from cognitive_core_engine.omega_forge.instructions import (
    Instruction, ProgramGenome, OPS, rand_inst,
)
from cognitive_core_engine.omega_forge.concepts import (
    Concept, ConceptLibrary, _inst_tuple_list, _concept_hash_from_insts,
)
from cognitive_core_engine.omega_forge.benchmark import TaskBenchmark, StrictStructuralDetector, DetectorParams
from cognitive_core_engine.omega_forge.evidence import EvidenceWriter, EngineConfig
from cognitive_core_engine.omega_forge.engine import OmegaForgeV13


def run_concept_selftests() -> None:
    vm = VirtualMachine(max_steps=50)
    lib = ConceptLibrary(max_size=10)
    concept = Concept(
        cid="c_copy_first",
        name="copy_first",
        kind="macro",
        payload={"instructions": [("LOAD", 0, 0, 0)]},
        compile_fn_id="macro_v1",
        discovered_gen=0,
        parents=[],
    )
    cid = lib.add_concept(concept, dedup=True)
    assert cid is not None, "concept add failed"

    tmp_path = "concept_selftest.json"
    lib.save(tmp_path)
    lib2 = ConceptLibrary(max_size=10)
    lib2.load(tmp_path)
    assert lib2.get("c_copy_first") is not None, "concept load failed"

    insts = lib2.compile(concept)
    assert insts, "concept compile failed"
    g_concept = ProgramGenome(gid="g_concept", instructions=insts)
    g_base = ProgramGenome(gid="g_base", instructions=[Instruction("HALT", 0, 0, 0)])

    metrics_base = ConceptDiscoveryBenchmark.evaluate(g_base, vm)
    metrics_concept = ConceptDiscoveryBenchmark.evaluate(g_concept, vm)
    assert metrics_concept["holdout_pass_rate"] > metrics_base["holdout_pass_rate"], "holdout did not improve"

    # Negative: train improves but holdout regresses on synthetic split
    def _eval_custom(genome: ProgramGenome, train: List[List[float]], holdout: List[List[float]]) -> Tuple[float, float]:
        def _score(dataset: List[List[float]]) -> float:
            passed = 0
            for inputs in dataset:
                st = vm.execute(genome, inputs)
                if st.error or not st.halted_cleanly:
                    continue
                if abs(st.regs[0]) < 0.01:
                    passed += 1
            return passed / max(1, len(dataset))

        return _score(train), _score(holdout)

    g_overfit = ProgramGenome(gid="g_overfit", instructions=[Instruction("SET", 0, 0, 0)])
    train = [[0.0], [0.0, 1.0]]
    holdout = [[1.0], [2.0, 2.0]]
    train_rate, holdout_rate = _eval_custom(g_overfit, train, holdout)
    assert train_rate > holdout_rate, "gap check failed"

    # Negative: adversarial/shift break detection
    adv = [[-1.0], [-2.0]]
    adv_rate, _ = _eval_custom(g_overfit, adv, holdout)
    assert adv_rate <= train_rate, "adversarial regression not detected"

    # VM step limit regression check
    g_loop = ProgramGenome(gid="g_loop", instructions=[Instruction("JMP", 0, 0, 0)])
    st = vm.execute(g_loop, [1.0])
    assert st.steps <= vm.max_steps, "step limit regression"

def cmd_selftest(args: argparse.Namespace) -> int:
    """
    Selftest validates:
      - engine executes for N generations without crashing
      - evidence file is created and non-empty (at least header line)
    It does NOT require successes in a short horizon.
    """
    out = args.out or "v13_selftest.jsonl"
    if os.path.exists(out):
        try:
            os.remove(out)
        except Exception:
            pass

    # For selftest, relax params a bit so it is more likely to see at least one success,
    # but still keep anti-cheat and logging correctness.
    p = DetectorParams(
        K_initial=4,
        L_initial=7,
        C_coverage=0.45,
        f_rarity=0.01,
        N_repro=3,
        require_both=True,
        min_loops=1,
        min_scc=1,
    )
    det = StrictStructuralDetector(p)
    eng = OmegaForgeV13(seed=args.seed, detector=det)
    eng.init_population()
    w = EvidenceWriter(out)

    gens = int(args.generations or 200)
    total_success_lines = 0
    try:
        for _ in range(gens):
            succ, _ = eng.step(writer=w)
            # progress
            # Count evidence lines roughly by success count (header already present)
            total_success_lines += succ
            if eng.generation % 10 == 0:
                # "total_evidence_lines" includes only evidence, not header
                print(f"[gen {eng.generation}] successes_this_gen={succ} total_evidence_lines={total_success_lines}", flush=True)
    finally:
        w.close()

    # Validate file exists and has at least 1 line (header)
    if not os.path.exists(out):
        print("SELFTEST_FAIL: evidence file missing", flush=True)
        return 1

    try:
        sz = os.path.getsize(out)
    except Exception:
        sz = 0

    if sz <= 0:
        print("SELFTEST_FAIL: evidence file empty (should contain header)", flush=True)
        return 1

    # Pass criteria: file non-empty + ran gens
    print(f"SELFTEST_OK: ran_gens={gens} evidence_file_bytes={sz} evidence_successes={total_success_lines}", flush=True)
    try:
        run_concept_selftests()
        print("CONCEPT_SELFTEST_OK", flush=True)
    except Exception as e:
        print(f"CONCEPT_SELFTEST_FAIL: {e}", flush=True)
        return 1
    return 0

def cmd_evidence_run(args: argparse.Namespace) -> int:
    out = args.out or "evidence_v13.jsonl"
    target = int(args.target or 6)
    max_g = int(args.max_generations or 2000)

    # Use default strict params unless user overrides via flags later
    eng = OmegaForgeV13(seed=args.seed)
    eng.init_population()
    w = EvidenceWriter(out)

    found = 0
    try:
        while eng.generation < max_g and found < target:
            succ, _ = eng.step(writer=w)
            found += succ
            if eng.generation % max(1, int(args.report_every or 10)) == 0:
                print(f"[gen {eng.generation}] found={found}/{target} out={out}", flush=True)
    finally:
        w.close()

    print(f"EVIDENCE_RUN_DONE: gens={eng.generation} found={found} out={out}", flush=True)
    return 0

def cmd_run(args: argparse.Namespace) -> int:
    log = args.log or "v13_run.jsonl"
    gens = int(args.generations or 5000)
    eng = OmegaForgeV13(seed=args.seed)
    eng.init_population()
    w = EvidenceWriter(log)
    try:
        for _ in range(gens):
            succ, _ = eng.step(writer=w)
            if eng.generation % max(1, int(args.report_every or 50)) == 0:
                print(f"[gen {eng.generation}] successes_this_gen={succ} log={log}", flush=True)
    finally:
        w.close()
    print(f"RUN_DONE: gens={gens} log={log}", flush=True)
    return 0

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="OMEGA_FORGE V13 CLEAN (streaming evidence)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("selftest", help="Run crash-safe logging selftest")
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--generations", type=int, default=200)
    p1.add_argument("--out", type=str, default="v13_selftest.jsonl")
    p1.set_defaults(func=cmd_selftest)

    p2 = sub.add_parser("evidence_run", help="Run until N evidence lines are found")
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--target", type=int, default=6)
    p2.add_argument("--max_generations", type=int, default=2000)
    p2.add_argument("--out", type=str, default="evidence_v13.jsonl")
    p2.add_argument("--report_every", type=int, default=10)
    p2.set_defaults(func=cmd_evidence_run)

    p3 = sub.add_parser("run", help="Long run (writes all evidence to log)")
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--generations", type=int, default=5000)
    p3.add_argument("--log", type=str, default="v13_run.jsonl")
    p3.add_argument("--report_every", type=int, default=50)
    p3.set_defaults(func=cmd_run)

    return ap

def main() -> int:
    ap = build_cli()
    args = ap.parse_args()
    return int(args.func(args))


# ==============================================================================
# Two-Stage Evolution Engine V4 + Feedback Loop (inlined)
# ==============================================================================

"""
OMEGA_FORGE Two-Stage Evolution Engine V4
==========================================
SUM Fix Patches Applied:
1. Diverse SUM cases (24 deterministic cases)
2. Full-sum dominant scoring (prefix is small tie-breaker)
3. SUM strict-pass gate after curriculum switch
4. Curriculum timing adjusted (250)
5. Accurate per-genome strict-pass benchmark
6. Debug output at gen 1

Usage:
  python two_stage_engine.py full --stage1_gens 300 --stage2_gens 500
"""

import argparse
import json
import random as global_random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Import from main engine
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Engine V4 (SUM Fix)")
    subparsers = parser.add_subparsers(dest="command")
    
    pf = subparsers.add_parser("full", help="Run full pipeline")
    pf.add_argument("--stage1_gens", type=int, default=300)
    pf.add_argument("--stage2_gens", type=int, default=500)
    pf.add_argument("--feedback_in", type=str, default="", help="Optional Stage2 feedback JSON to bias Stage1 opcode sampling")
    pf.add_argument("--feedback_out", type=str, default="stage2_feedback.json", help="Where to write Stage2 feedback JSON")
    pf.add_argument("--feedback_topk", type=int, default=20, help="Top-K genomes used to compute feedback biases")
    pf.add_argument("--concepts_on", action="store_true", help="Enable concept invention layer in Stage1")
    pf.add_argument("--concept_budget", type=int, default=80, help="Max concepts in library")
    pf.add_argument("--concept_library_path", type=str, default="concept_library.json", help="Path to concept library JSON")

    pf.add_argument("--seed", type=int, default=42)
    pf.add_argument("--agg", type=str, default="gmean", choices=["gmean", "min", "avg"])
    pf.add_argument("--curriculum_switch", type=int, default=250)
    
    args = parser.parse_args()
    
    if args.command == "full":
        global AGG_MODE, CURRICULUM_SWITCH_GEN
        AGG_MODE = args.agg
        CURRICULUM_SWITCH_GEN = args.curriculum_switch
        
        print("=" * 60)
        print("TWO-STAGE EVOLUTION V4 (SUM Fix Patches Applied)")
        print("=" * 60)
        print(f"Config: AGG={AGG_MODE}, SWITCH_GEN={CURRICULUM_SWITCH_GEN}, SUM_GATE={SUM_GATE_AFTER_SWITCH}")
        print()
        
        
        # Optional: apply prior feedback to bias Stage1 opcode sampling
        if args.feedback_in:
            fb = load_feedback_json(args.feedback_in)
            apply_feedback_to_stage1(fb)

        s1 = Stage1Engine(
            seed=args.seed,
            concepts_on=args.concepts_on,
            concept_budget=args.concept_budget,
            concept_library_path=args.concept_library_path,
        )
        candidates = s1.run(args.stage1_gens, "stage1_candidates.jsonl")
        
        print()
        
        s2 = Stage2Engine(candidates, seed=args.seed)
        s2.load_population()
        s2.run(args.stage2_gens)

        # Write Stage2->Stage1 feedback biases
        try:
            fb = extract_stage2_feedback(
                s2.population,
                s2.vm,
                n_top=args.feedback_topk,
                require_sum_pass=True,
                concept_library=s1.concept_library if args.concepts_on else None,
            )
            save_feedback_json(fb, args.feedback_out)
            print(f"\n[Feedback] Wrote Stage2 feedback to {args.feedback_out}")
        except Exception as e:
            print(f"\n[Feedback] WARNING: failed to write feedback: {e}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
