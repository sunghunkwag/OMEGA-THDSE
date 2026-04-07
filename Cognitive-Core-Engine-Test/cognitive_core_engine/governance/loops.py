"""Main search loops: duo_loop and rsi_loop."""
from __future__ import annotations

import ast
import copy
import json
import math
import os
import random
import re
import textwrap
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from cognitive_core_engine.governance.utils import now_ms, sha256, safe_mkdir, read_json, write_json
from cognitive_core_engine.governance.sandbox import (
    safe_exec, safe_exec_algo, validate_code, validate_program, safe_exec_engine,
    SAFE_BUILTINS,
)


def run_duo_loop(
    rounds: int,
    slice_seconds: float,
    blackboard_path: Path,
    k_full: int,
    seed: int,
    mode: str = "solver",
    freeze_eval: bool = True,
    population: int = 64,
    max_candidates: int = 512,
) -> None:
    task = TaskSpec()
    task.ensure_descriptor()
    rng = random.Random(seed)
    gs = load_state()
    if gs and gs.mode == mode and gs.universes:
        selected = next((u for u in gs.universes if u.get("uid") == gs.selected_uid), gs.universes[0])
        universe = Universe.from_snapshot(selected)
    else:
        batch0 = get_task_batch(task, seed, freeze_eval=freeze_eval)
        hint = TaskDetective.detect_pattern(batch0)
        universe = Universe(
            uid=0,
            seed=seed,
            meta=MetaState(),
            pool=[seed_genome(random.Random(seed + i), hint) for i in range(population)],
            library=FunctionLibrary(),
            eval_mode="program" if mode == "program" else ("algo" if mode == "algo" else "solver"),
        )

    creator_policy = CREATOR_POLICY
    critic_policy = CRITIC_POLICY
    reseed_templates: List[List[str]] = []
    fixed_batch = get_task_batch(task, seed, freeze_eval=freeze_eval, gen=0)
    if fixed_batch is None:
        print("[DUO] No batch available; aborting.")
        return

    for r in range(rounds):
        round_seed = seed + r * 9973
        round_rng = random.Random(round_seed)
        batch = get_task_batch(task, seed, freeze_eval=freeze_eval, gen=r)
        if batch is None:
            print("[DUO] No batch available; aborting.")
            break
        helper_env = universe.library.get_helpers()
        hint = TaskDetective.detect_pattern(batch)
        if hint:
            print(f"[DUO] Detected pattern: {hint}")

        creator_slice = slice_seconds if slice_seconds > 0 else creator_policy.slice_seconds
        critic_slice = slice_seconds if slice_seconds > 0 else critic_policy.slice_seconds

        print(f"\n{'='*60}\n[DUO ROUND {r+1}/{rounds}] Creator\n{'='*60}")
        creator_candidates: List[Genome] = [
            seed_genome(round_rng, hint),
            _fallback_template_genome(round_rng, hint),
        ]
        if universe.best:
            creator_candidates.append(_repair_genome(universe.best))
        creator_start = time.time()
        while time.time() - creator_start < creator_slice:
            if len(creator_candidates) >= max_candidates:
                break
            mode_choice = creator_policy.generator_mode
            if mode_choice == "template":
                if reseed_templates:
                    stmts = round_rng.choice(reseed_templates)
                    g = Genome(statements=list(stmts), op_tag="reseed")
                else:
                    g = seed_genome(round_rng, hint)
            elif mode_choice == "mutate":
                parent = round_rng.choice(universe.pool) if universe.pool else seed_genome(round_rng, hint)
                g = _mutate_genome_with_meta(round_rng, parent, universe.meta, universe.library)
            else:
                g = _synthesize_genome(round_rng, universe.pool, hint, universe.library)
            creator_candidates.append(g)

        print(f"[DUO] Creator proposed {len(creator_candidates)} candidates")

        print(f"\n{'='*60}\n[DUO ROUND {r+1}/{rounds}] Critic\n{'='*60}")
        critic_start = time.time()
        gate_fail_reasons: collections.Counter = collections.Counter()
        validator_fail_reasons: collections.Counter = collections.Counter()
        scored_empty_count = 0
        prefiltered: List[Tuple[Genome, EvalResult]] = []
        seen_hashes: Set[str] = set()
        duplicate_count = 0
        total_checked = 0
        gate_pass = 0

        for g in creator_candidates:
            if time.time() - critic_start > critic_slice:
                break
            total_checked += 1
            code_hash = _candidate_hash(g.code)
            if code_hash in seen_hashes:
                duplicate_count += 1
            seen_hashes.add(code_hash)

            ok, reason, pre_res = _prefilter_eval(
                g,
                batch,
                universe.eval_mode,
                task.name,
                extra_env=helper_env,
                validator=validate_program if universe.eval_mode == "program" else validate_code,
            )
            record = {
                "timestamp": now_ms(),
                "agent_id": "critic",
                "generation": r,
                "candidate_hash": code_hash,
                "gate_ok": ok,
                "gate_reason": "" if ok else reason,
                "score_train": pre_res.train if pre_res else None,
                "score_holdout": pre_res.hold if pre_res else None,
                "score_stress": pre_res.stress if pre_res else None,
                "selected": False,
                "note": "prefilter",
            }
            append_blackboard(blackboard_path, record)

            if not ok:
                if reason.startswith("hard_gate:"):
                    gate_fail_reasons[reason.split("hard_gate:", 1)[1]] += 1
                elif reason.startswith("validator:"):
                    validator_fail_reasons[reason.split("validator:", 1)[1]] += 1
                continue
            gate_pass += 1
            if pre_res:
                prefiltered.append((g, pre_res))

        if not prefiltered:
            scored_empty_count += 1
            reseed_templates = [_fallback_template_genome(round_rng, hint).statements]
            append_blackboard(
                blackboard_path,
                {
                    "timestamp": now_ms(),
                    "agent_id": "critic",
                    "generation": r,
                    "candidate_hash": "none",
                    "gate_ok": False,
                    "gate_reason": "scored_empty",
                    "score_train": None,
                    "score_holdout": None,
                    "score_stress": None,
                    "selected": False,
                    "note": "reseed",
                },
            )
            gate_pass_rate = gate_pass / max(1, total_checked)
            creator_policy = _adjust_creator_policy(creator_policy, gate_pass_rate, gate_fail_reasons)
            print("[DUO] No candidates passed prefilter; reseeding templates.")
            _print_critic_summary(
                gate_pass=gate_pass,
                total_checked=total_checked,
                adopted=False,
                full_results_count=0,
                duplicate_count=duplicate_count,
                scored_empty_count=scored_empty_count,
                gate_fail_reasons=gate_fail_reasons,
                validator_fail_reasons=validator_fail_reasons,
            )
            continue

        prefiltered.sort(key=lambda t: _critic_rank_score(t[1], critic_policy))
        selected = prefiltered[: max(1, k_full)]
        prefilter_map = {_candidate_hash(g.code): res for g, res in prefiltered}
        baseline_candidates = creator_candidates[:2]
        baseline_hashes = {_candidate_hash(c.code) for c in baseline_candidates}
        selected_hashes = {_candidate_hash(g.code) for g, _ in selected}
        for base in baseline_candidates:
            base_hash = _candidate_hash(base.code)
            if base_hash not in selected_hashes:
                base_res = prefilter_map.get(base_hash)
                if base_res:
                    selected.append((base, base_res))
        for g, pre_res in selected:
            append_blackboard(
                blackboard_path,
                {
                    "timestamp": now_ms(),
                    "agent_id": "critic",
                    "generation": r,
                    "candidate_hash": _candidate_hash(g.code),
                    "gate_ok": True,
                    "gate_reason": "",
                    "score_train": pre_res.train,
                    "score_holdout": pre_res.hold,
                    "score_stress": pre_res.stress,
                    "selected": True,
                    "note": "prefilter_selected",
                },
            )

        full_results: List[Tuple[Genome, EvalResult]] = []
        forced_eval = set(baseline_hashes)
        for g, _ in selected:
            if time.time() - critic_start > critic_slice and _candidate_hash(g.code) not in forced_eval:
                break
            refined = _critic_refine(round_rng, g, universe.meta, universe.library)
            for candidate in [g] + refined:
                if time.time() - critic_start > critic_slice and _candidate_hash(candidate.code) not in forced_eval:
                    break
                res = _evaluate_candidate(
                    candidate,
                    _merge_stress(fixed_batch, batch),
                    universe.eval_mode,
                    task.name,
                    extra_env=helper_env,
                    validator=validate_program if universe.eval_mode == "program" else validate_code,
                )
                if res.ok:
                    full_results.append((candidate, res))
                else:
                    if res.err:
                        if res.err.startswith("hard_gate:"):
                            gate_fail_reasons[res.err.split("hard_gate:", 1)[1]] += 1
                        else:
                            validator_fail_reasons[res.err] += 1
                append_blackboard(
                    blackboard_path,
                    {
                        "timestamp": now_ms(),
                        "agent_id": "critic",
                        "generation": r,
                        "candidate_hash": _candidate_hash(candidate.code),
                        "gate_ok": res.ok,
                        "gate_reason": "" if res.ok else (res.err or ""),
                        "score_train": res.train if res.ok else None,
                        "score_holdout": res.hold if res.ok else None,
                        "score_stress": res.stress if res.ok else None,
                        "selected": False,
                        "note": candidate.op_tag,
                    },
                )

        adopted = False
        full_results_count = len(full_results)
        if not full_results:
            scored_empty_count += 1
            reseed_templates = [_fallback_template_genome(round_rng, hint).statements]
            append_blackboard(
                blackboard_path,
                {
                    "timestamp": now_ms(),
                    "agent_id": "critic",
                    "generation": r,
                    "candidate_hash": "none",
                    "gate_ok": False,
                    "gate_reason": "scored_empty",
                    "score_train": None,
                    "score_holdout": None,
                    "score_stress": None,
                    "selected": False,
                    "note": "reseed",
                },
            )
            print("[DUO] No candidates survived full evaluation; reseeding templates.")
        else:
            full_results.sort(key=lambda t: t[1].score)
            best_g, best_res = full_results[0]
            if best_res.score < universe.best_score:
                adopted = True
                universe.best = best_g
                universe.best_score = best_res.score
                universe.best_train = best_res.train
                universe.best_hold = best_res.hold
                universe.best_stress = best_res.stress
                universe.best_test = best_res.test
                append_blackboard(
                    blackboard_path,
                    {
                        "timestamp": now_ms(),
                        "agent_id": "critic",
                        "generation": r,
                        "candidate_hash": _candidate_hash(best_g.code),
                        "gate_ok": True,
                        "gate_reason": "",
                        "score_train": best_res.train,
                        "score_holdout": best_res.hold,
                        "score_stress": best_res.stress,
                        "selected": True,
                        "note": "adopted",
                    },
                )
            universe.pool = [g for g, _ in full_results[: max(8, population // 4)]]
            if len(universe.pool) < population:
                universe.pool.extend([seed_genome(round_rng, hint) for _ in range(population - len(universe.pool))])

        gate_pass_rate = gate_pass / max(1, total_checked)
        creator_policy = _adjust_creator_policy(creator_policy, gate_pass_rate, gate_fail_reasons)
        _print_critic_summary(
            gate_pass=gate_pass,
            total_checked=total_checked,
            adopted=adopted,
            full_results_count=full_results_count,
            duplicate_count=duplicate_count,
            scored_empty_count=scored_empty_count,
            gate_fail_reasons=gate_fail_reasons,
            validator_fail_reasons=validator_fail_reasons,
        )

        gs = GlobalState(
            "RSI_EXTENDED_v2",
            now_ms(),
            now_ms(),
            seed,
            asdict(task),
            [universe.snapshot()],
            universe.uid,
            r + 1,
            mode=mode,
        )
        save_state(gs)

def run_rsi_loop(
    gens_per_round: int,
    rounds: int,
    levels: List[int],
    pop: int,
    n_univ: int,
    mode: str,
    freeze_eval: bool = True,
    meta_meta: bool = False,
    update_rule_rounds: int = 0,
):
    task = TaskSpec()
    seed = int(time.time()) % 100000
    if meta_meta:
        run_meta_meta(
            seed=seed,
            episodes=rounds,
            gens_per_episode=gens_per_round,
            pop=pop,
            n_univ=n_univ,
            freeze_eval=freeze_eval,
            state_dir=STATE_DIR,
            eval_every=1,
            few_shot_gens=max(3, gens_per_round // 2),
        )
        print(f"\n[RSI LOOP COMPLETE] {rounds} meta-meta rounds finished")
        return

    archive_path = STATE_DIR / "rsi_archive.json"
    archive = _load_rsi_archive(archive_path)
    if archive.get("current") and "genome" not in archive["current"]:
        archive = {"entries": [], "current": None, "consecutive": 0}
    fixed_batch = get_task_batch(task, seed, freeze_eval=True, gen=0)
    if fixed_batch is None:
        print("[RSI] No batch available; aborting.")
        return

    for r in range(rounds):
        print(f"\n{'='*60}\n[RSI ROUND {r+1}/{rounds}]\n{'='*60}")
        print(f"[EVOLVE] {gens_per_round} generations...")
        gs = run_multiverse(seed, task, gens_per_round, pop, n_univ, resume=(r > 0), mode=mode, freeze_eval=freeze_eval)
        best_snapshot = next((u for u in gs.universes if u.get("uid") == gs.selected_uid), None)
        best_data = (best_snapshot or {}).get("best")
        best_code = None
        if isinstance(best_data, dict):
            if mode == "learner":
                best_code = LearnerGenome(**best_data).code
            else:
                best_code = Genome(**best_data).code
        if best_code and best_code != "none":
            gate_ok, gate_reason = _hard_gate_ok(best_code, fixed_batch, mode, task.name)
            if not gate_ok:
                print(f"[RSI] Hard gate failed for best candidate ({gate_reason}); rejecting before scoring/autopatch.")
                archive["current"] = None
                archive["consecutive"] = 0
                archive["entries"] = []
                _save_rsi_archive(archive_path, archive)
                continue
        recent_scores = load_recent_scores(STATE_DIR / "run_log.jsonl", 5)
        forced_applied = False
        if is_300s_stagnation(recent_scores):
            print("[STAGNATION] 300s plateau detected for >=5 gens. Forcing L1/L3 autopatch.")
            forced = run_deep_autopatch([1, 3], candidates=4, apply=True, mode=mode)
            forced_applied = bool(forced.get("applied"))
            if forced_applied:
                print("[RSI] Self-modified via forced L1/L3 patch.")
            else:
                print("[STAGNATION] Forced patch rejected. Launching meta-meta acceleration.")
                run_meta_meta(
                    seed=seed,
                    episodes=1,
                    gens_per_episode=gens_per_round,
                    pop=pop,
                    n_univ=n_univ,
                    freeze_eval=freeze_eval,
                    state_dir=STATE_DIR,
                    eval_every=1,
                    few_shot_gens=max(3, gens_per_round // 2),
                )
                print("[STAGNATION] Meta-meta episode completed.")
        if not forced_applied:
            print(f"[AUTOPATCH] Trying L{levels}...")
            result = run_deep_autopatch(levels, candidates=4, apply=True, mode=mode)
            if result.get("applied"):
                print("[RSI] Self-modified! Reloading...")
        if update_rule_rounds > 0:
            print(f"[META] Running update-rule search for {update_rule_rounds} rounds...")
            run_update_rule_search(
                seed=seed + r * 127,
                rounds=update_rule_rounds,
                gens_per_round=max(3, gens_per_round // 2),
                pop=max(16, pop // 2),
                freeze_eval=freeze_eval,
                state_dir=STATE_DIR,
            )

    print(f"\n[RSI LOOP COMPLETE] {rounds} rounds finished")


# ---------------------------
# CLI Commands
# ---------------------------

