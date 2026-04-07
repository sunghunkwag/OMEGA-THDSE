"""AutoPatch, scoring, filtering, and policy management."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from cognitive_core_engine.governance.utils import now_ms, sha256, safe_mkdir, read_json, write_json, unified_diff
from cognitive_core_engine.governance.sandbox import (
    safe_exec, safe_exec_algo, validate_code, validate_program,
    validate_algo_program, safe_exec_engine, safe_load_module,
    SAFE_BUILTINS, SAFE_VARS,
)


def policy_stats_from_history(history: List[Dict[str, Any]], window: int = 5) -> Dict[str, float]:
    if not history:
        return {"delta_best": 0.0, "auc_window": 0.0, "timeout_rate": 0.0, "avg_nodes": 0.0}
    holds = [h.get("hold", 0.0) for h in history]
    recent = holds[-window:] if len(holds) >= window else holds
    auc_window = sum(recent) / max(1, len(recent))
    if len(holds) >= window:
        delta_best = holds[-1] - holds[-window]
    else:
        delta_best = holds[-1] - holds[0]
    timeout_rate = history[-1].get("timeout_rate", 0.0)
    avg_nodes = history[-1].get("avg_nodes", 0.0)
    return {
        "delta_best": delta_best,
        "auc_window": auc_window,
        "timeout_rate": timeout_rate,
        "avg_nodes": avg_nodes,
    }


def run_policy_episode(
    seed: int,
    task: TaskSpec,
    policy: MetaPolicy,
    gens: int,
    pop: int,
    n_univ: int,
    freeze_eval: bool,
    library_archive: LibraryArchive,
    logger: Optional[RunLogger],
    mode: str,
    update_archive: bool = True,
) -> Tuple[List[Dict[str, Any]], Universe]:
    batch = get_task_batch(task, seed, freeze_eval=freeze_eval)
    hint = TaskDetective.detect_pattern(batch)
    descriptor = task.ensure_descriptor()
    base_lib = FunctionLibrary()
    for lib in library_archive.select(descriptor):
        base_lib.merge(lib)
    update_rule = load_update_rule(STATE_DIR / "update_rule.json")
    universes = [
        Universe(
            uid=i,
            seed=seed + i * 9973,
            meta=MetaState(update_rule=update_rule),
            pool=[seed_genome(random.Random(seed + i), hint) for _ in range(pop)],
            library=FunctionLibrary.from_snapshot(base_lib.snapshot()),
        )
        for i in range(n_univ)
    ]
    for gen in range(gens):
        start_ms = now_ms()
        stats = policy_stats_from_history(universes[0].history)
        controls = policy.act(descriptor, stats)
        for u in universes:
            u.step(gen, task, pop, batch, policy_controls=controls)
        universes.sort(key=lambda u: u.best_score)
        best = universes[0]
        if logger:
            best_code = best.best.code if best.best else "none"
            code_hash = sha256(best_code)
            novelty = 1.0 if code_hash not in logger.seen_hashes else 0.0
            logger.seen_hashes.add(code_hash)
            logger.log(
                gen=gen,
                task_id=task.name,
                mode=mode,
                score_hold=best.best_hold,
                score_stress=best.best_stress,
                score_test=best.best_test,
                runtime_ms=now_ms() - start_ms,
                nodes=node_count(best_code),
                code_hash=code_hash,
                accepted=bool(best.history[-1]["accepted"]) if best.history else False,
                novelty=novelty,
                meta_policy_params={"pid": policy.pid, "weights": policy.weights, "bias": policy.bias, "controls": controls},
                task_descriptor=descriptor.snapshot(),
            )
    universes.sort(key=lambda u: u.best_score)
    best = universes[0]
    if update_archive:
        library_archive.add(descriptor, best.best_hold, best.library)
    return best.history, best


def compute_transfer_metrics(history: List[Dict[str, Any]], window: int) -> Dict[str, float]:
    if not history:
        return {"auc": float("inf"), "regret": float("inf"), "gap": float("inf"), "recovery_time": float("inf")}
    holds = [h.get("hold", float("inf")) for h in history[:window]]
    tests = [h.get("test", float("inf")) for h in history[:window]]
    auc = sum(holds) / max(1, len(holds))
    best = min(holds)
    regret = sum(h - best for h in holds) / max(1, len(holds))
    gap = (tests[-1] - holds[-1]) if holds and tests else float("inf")
    threshold = best * 1.1 if math.isfinite(best) else float("inf")
    recovery_time = float("inf")
    for i, h in enumerate(holds):
        if h <= threshold:
            recovery_time = i + 1
            break
    return {"auc": auc, "regret": regret, "gap": gap, "recovery_time": recovery_time}


def run_meta_meta(
    seed: int,
    episodes: int,
    gens_per_episode: int,
    pop: int,
    n_univ: int,
    policy_pop: int,
    freeze_eval: bool,
    state_dir: Path,
    eval_every: int,
    few_shot_gens: int,
) -> None:
    rng = random.Random(seed)
    meta_train, meta_test = split_meta_tasks(seed)
    n_inputs = len(TaskSpec().ensure_descriptor().vector()) + 4
    policies = [MetaPolicy.seed(rng, n_outputs=5, n_inputs=n_inputs) for _ in range(policy_pop)]
    policy_scores = {p.pid: float("inf") for p in policies}
    archive = LibraryArchive(k=2)
    logger = RunLogger(state_dir / "run_log.jsonl")

    for episode in range(episodes):
        task = rng.choice(meta_train)
        policy = policies[episode % len(policies)]
        history, best = run_policy_episode(
            seed + episode * 31,
            task,
            policy,
            gens_per_episode,
            pop,
            n_univ,
            freeze_eval,
            archive,
            logger,
            mode="meta-train",
            update_archive=True,
        )
        metrics = compute_transfer_metrics(history, window=min(few_shot_gens, len(history)))
        reward = metrics["auc"]
        policy_scores[policy.pid] = min(policy_scores[policy.pid], reward)

        if (episode + 1) % eval_every == 0:
            transfer_scores = []
            for task_test in meta_test:
                warmup_task = rng.choice(meta_train) if meta_train else task_test
                warmup_gens = max(1, few_shot_gens // 2)
                run_policy_episode(
                    seed + episode * 73,
                    warmup_task,
                    policy,
                    warmup_gens,
                    pop,
                    n_univ,
                    freeze_eval,
                    archive,
                    logger,
                    mode="meta-transfer-train",
                    update_archive=True,
                )
                hist, _ = run_policy_episode(
                    seed + episode * 73 + 1,
                    task_test,
                    policy,
                    few_shot_gens,
                    pop,
                    n_univ,
                    freeze_eval,
                    archive,
                    logger,
                    mode="meta-transfer-test",
                    update_archive=False,
                )
                transfer_scores.append(compute_transfer_metrics(hist, window=few_shot_gens)["auc"])
            if transfer_scores:
                policy_scores[policy.pid] = sum(transfer_scores) / len(transfer_scores)
            policies.sort(key=lambda p: policy_scores.get(p.pid, float("inf")))
            best_policy = policies[0]
            policies = [best_policy] + [best_policy.mutate(rng, scale=0.05) for _ in range(policy_pop - 1)]


def evaluate_update_rule(
    seed: int,
    task: TaskSpec,
    rule: UpdateRuleGenome,
    gens: int,
    pop: int,
    freeze_eval: bool,
) -> Tuple[float, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    batch = get_task_batch(task, seed, freeze_eval=freeze_eval)
    hint = TaskDetective.detect_pattern(batch)
    universe = Universe(
        uid=0,
        seed=seed,
        meta=MetaState(update_rule=rule),
        pool=[seed_genome(rng, hint) for _ in range(pop)],
        library=FunctionLibrary(),
    )
    for gen in range(gens):
        universe.step(gen, task, pop, batch)
    return universe.best_score, universe.history


def run_update_rule_search(
    seed: int,
    rounds: int,
    gens_per_round: int,
    pop: int,
    freeze_eval: bool,
    state_dir: Path,
) -> UpdateRuleGenome:
    rng = random.Random(seed)
    task = TaskSpec(name="self_audit", n_train=64, n_hold=48, n_test=48, noise=0.0)
    current = load_update_rule(state_dir / "update_rule.json")
    rules = [current] + [current.mutate(rng) for _ in range(3)]
    best_rule = current
    best_score = float("inf")
    for r in range(rounds):
        scored: List[Tuple[float, UpdateRuleGenome]] = []
        for idx, rule in enumerate(rules):
            score, history = evaluate_update_rule(seed + r * 101 + idx, task, rule, gens_per_round, pop, freeze_eval)
            scored.append((score, rule))
            if history:
                last = history[-1]
                if last.get("code"):
                    SURROGATE.train(history)
        scored.sort(key=lambda item: item[0])
        score, best = scored[0]
        if score < best_score:
            best_score = score
            best_rule = best
            save_update_rule(state_dir / "update_rule.json", best_rule)
        rules = [best] + [best.mutate(rng) for _ in range(max(1, len(rules) - 1))]
    return best_rule


def run_task_switch(
    seed: int,
    task_a: TaskSpec,
    task_b: TaskSpec,
    gens_a: int,
    gens_b: int,
    pop: int,
    n_univ: int,
    freeze_eval: bool,
    state_dir: Path,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    n_inputs = len(TaskSpec().ensure_descriptor().vector()) + 4
    transfer_policy = MetaPolicy.seed(rng, n_outputs=5, n_inputs=n_inputs)
    archive = LibraryArchive(k=2)
    logger = RunLogger(state_dir / "run_log.jsonl")
    baseline = MetaPolicy.seed(random.Random(seed + 999), n_outputs=5, n_inputs=n_inputs)

    history_a, _ = run_policy_episode(
        seed,
        task_a,
        transfer_policy,
        gens_a,
        pop,
        n_univ,
        freeze_eval,
        archive,
        logger,
        mode="switch-train",
        update_archive=True,
    )
    history_transfer, _ = run_policy_episode(
        seed + 1,
        task_b,
        transfer_policy,
        gens_b,
        pop,
        n_univ,
        freeze_eval,
        archive,
        logger,
        mode="switch-transfer",
        update_archive=False,
    )
    history_baseline, _ = run_policy_episode(
        seed + 2,
        task_b,
        baseline,
        gens_b,
        pop,
        n_univ,
        freeze_eval,
        LibraryArchive(k=0),
        logger,
        mode="switch-baseline",
        update_archive=False,
    )
    metrics_transfer = compute_transfer_metrics(history_transfer, window=gens_b)
    metrics_baseline = compute_transfer_metrics(history_baseline, window=gens_b)
    delta_auc = metrics_baseline["auc"] - metrics_transfer["auc"]
    delta_recovery = metrics_baseline["recovery_time"] - metrics_transfer["recovery_time"]
    return {
        "transfer": metrics_transfer,
        "baseline": metrics_baseline,
        "delta_auc": delta_auc,
        "delta_recovery_time": delta_recovery,
    }


def generate_report(path: Path, few_shot_gens: int) -> Dict[str, Any]:
    if not path.exists():
        return {"error": "run_log.jsonl not found"}
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        key = f"{rec['task_id']}::{rec.get('mode', 'unknown')}"
        by_task.setdefault(key, []).append(rec)
    report = {"tasks": {}, "few_shot_gens": few_shot_gens}
    for key, recs in by_task.items():
        recs.sort(key=lambda r: r["gen"])
        holds = [r["score_hold"] for r in recs[:few_shot_gens]]
        tests = [r["score_test"] for r in recs[:few_shot_gens]]
        auc = sum(holds) / max(1, len(holds))
        best = min(holds) if holds else float("inf")
        regret = sum(h - best for h in holds) / max(1, len(holds))
        gap = (tests[-1] - holds[-1]) if holds and tests else float("inf")
        threshold = best * 1.1 if math.isfinite(best) else float("inf")
        recovery_time = float("inf")
        for i, h in enumerate(holds):
            if h <= threshold:
                recovery_time = i + 1
                break
        few_shot_delta = (holds[0] - holds[-1]) if len(holds) > 1 else 0.0
        report["tasks"][key] = {
            "auc": auc,
            "regret": regret,
            "generalization_gap": gap,
            "recovery_time": recovery_time,
            "few_shot_delta": few_shot_delta,
        }
    return report


def transfer_bench(
    task_from: str,
    task_to: str,
    budget: int,
    seed: int,
    freeze_eval: bool = True,
) -> Dict[str, Any]:
    task_a = TaskSpec(name=task_from)
    task_b = TaskSpec(name=task_to)
    mode = "algo" if task_from in ALGO_TASK_NAMES else "solver"
    u = Universe(uid=0, seed=seed, meta=MetaState(), pool=[], library=FunctionLibrary(), eval_mode=mode)
    rng = random.Random(seed)

    for g in range(budget):
        batch = get_task_batch(task_a, seed, freeze_eval=freeze_eval, gen=g)
        if batch is None:
            break
        u.step(g, task_a, 24, batch)

    holds: List[float] = []
    for g in range(budget):
        batch = get_task_batch(task_b, seed + 17, freeze_eval=freeze_eval, gen=g)
        if batch is None:
            break
        u.step(g, task_b, 24, batch)
        holds.append(u.best_hold)

    auc = sum(holds) / max(1, len(holds))
    best = min(holds) if holds else float("inf")
    threshold = best * 1.1 if math.isfinite(best) else float("inf")
    recovery_time = float("inf")
    for i, h in enumerate(holds):
        if h <= threshold:
            recovery_time = i + 1
            break

    record = {
        "from": task_from,
        "to": task_to,
        "budget": budget,
        "seed": seed,
        "auc_N": auc,
        "recovery_time": recovery_time,
        "holds": holds,
    }
    out = STATE_DIR / "transfer_bench.jsonl"
    safe_mkdir(out.parent)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


# ---------------------------
# RSI Loop (Hard Gates + Rollback)
# ---------------------------

STRESS_MAX = 1_000_000.0
OUTPUT_VARIANCE_EPS = 1e-6
RSI_CONFIRM_ROUNDS = 2

def _outputs_constant(outputs: List[Any], tol: float = 1e-9) -> bool:
    if not outputs:
        return True
    first = outputs[0]
    if isinstance(first, (int, float)):
        return all(isinstance(o, (int, float)) and abs(o - first) <= tol for o in outputs[1:])
    return all(_algo_equal(o, first) for o in outputs[1:])

def _unique_output_count(outputs: List[Any]) -> int:
    uniques: List[Any] = []
    for out in outputs:
        if not any(_algo_equal(out, seen) for seen in uniques):
            uniques.append(out)
    return len(uniques)

def _piecewise_constant(outputs: List[Any], max_unique: int = 2) -> bool:
    if not outputs:
        return True
    return _unique_output_count(outputs) <= max_unique

def _variance_low(outputs: List[Any], eps: float = OUTPUT_VARIANCE_EPS) -> bool:
    if not outputs or not all(isinstance(o, (int, float)) for o in outputs):
        return False
    vals = [float(o) for o in outputs]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
    return var <= eps

def _collect_outputs(
    code: str,
    xs: List[Any],
    mode: str,
    extra_env: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[Any], str]:
    outputs: List[Any] = []
    if mode == "learner":
        env = safe_load_module(code)
        if not env:
            return False, [], "load_failed"
        required = ["init_mem", "encode", "predict"]
        if not all(name in env and callable(env[name]) for name in required):
            return False, [], "missing_funcs"
        mem = env["init_mem"]()
        encode = env["encode"]
        predict = env["predict"]
        for x in xs:
            try:
                z = encode(x, mem)
                out = predict(z, mem)
            except Exception:
                return False, [], "exec_error"
            outputs.append(out)
        return True, outputs, ""
    if mode == "algo":
        for x in xs:
            out, _, timeout = safe_exec_algo(code, x)
            if timeout:
                return False, [], "timeout"
            outputs.append(out)
        return True, outputs, ""
    for x in xs:
        out = safe_exec(code, x, extra_env=extra_env)
        if out is None:
            return False, [], "no_output"
        outputs.append(out)
    return True, outputs, ""

def _hard_gate_ok(
    code: str,
    batch: Batch,
    mode: str,
    task_name: str,
    extra_env: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    xs = batch.x_ho[:8] if batch.x_ho else batch.x_tr[:8]
    if not xs:
        return False, "no_inputs"
    ok, outputs, err = _collect_outputs(code, xs, mode, extra_env=extra_env)
    if not ok:
        return False, err
    # Hard gate: reject any non-finite numeric output (timeouts/NaNs are disqualifying).
    for out in outputs:
        if isinstance(out, (int, float)) and not math.isfinite(out):
            return False, "non_finite_output"
    # Hard gate: reject constant or near-constant outputs to enforce input dependence.
    if _outputs_constant(outputs):
        return False, "constant_output"
    # Hard gate: prevent piecewise-constant or low-diversity output hacks.
    if _piecewise_constant(outputs):
        return False, "piecewise_constant"
    # Hard gate: reject numerically low-variance responses (e.g., tiny jitter around a constant).
    if _variance_low(outputs):
        return False, "low_variance_output"
    return True, ""

def _evaluate_candidate(
    g: Union[Genome, LearnerGenome],
    batch: Batch,
    mode: str,
    task_name: str,
    extra_env: Optional[Dict[str, Any]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_code,
) -> EvalResult:
    gate_ok, gate_reason = _hard_gate_ok(g.code, batch, mode, task_name, extra_env=extra_env)
    if not gate_ok:
        return EvalResult(
            False,
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            node_count(g.code),
            float("inf"),
            f"hard_gate:{gate_reason}",
        )
    if mode == "learner":
        return evaluate_learner(g, batch, task_name)
    if mode == "algo":
        return evaluate_algo(g, batch, task_name)
    return evaluate(g, batch, task_name, extra_env=extra_env, validator=validator)

def _merge_stress(fixed: Batch, resampled: Batch) -> Batch:
    return Batch(
        x_tr=resampled.x_tr,
        y_tr=resampled.y_tr,
        x_ho=resampled.x_ho,
        y_ho=resampled.y_ho,
        x_st=fixed.x_st + resampled.x_st,
        y_st=fixed.y_st + resampled.y_st,
        x_te=resampled.x_te,
        y_te=resampled.y_te,
    )

def _load_rsi_archive(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"entries": [], "current": None, "consecutive": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"entries": [], "current": None, "consecutive": 0}

def _save_rsi_archive(path: Path, archive: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(archive, indent=2), encoding="utf-8")

def _load_state_snapshot(state_dir: Path) -> Optional[Dict[str, Any]]:
    state_path = state_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _write_state_snapshot(state_dir: Path, snapshot: Dict[str, Any]) -> None:
    safe_mkdir(state_dir)
    (state_dir / "state.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

def _current_best_score(snapshot: Optional[Dict[str, Any]]) -> float:
    if not snapshot:
        return float("inf")
    universes = snapshot.get("universes", [])
    selected_uid = snapshot.get("selected_uid", None)
    best = float("inf")
    for u in universes:
        score = float(u.get("best_score", float("inf")))
        if selected_uid is not None and u.get("uid") == selected_uid:
            return score
        best = min(best, score)
    return best

def _clone_state_dir(src: Path, dest: Path) -> None:
    safe_mkdir(dest)
    if not src.exists():
        return
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

def _autopatch_evolve_score(
    script: Path,
    state_dir: Path,
    mode: str,
    task_name: str,
    seed: int,
    generations: int,
    population: int,
    universes: int,
    resume: bool,
    freeze_eval: bool = True,
) -> float:
    if mode == "learner":
        cmd = [
            sys.executable,
            str(script),
            "learner-evolve",
            "--seed",
            str(seed),
            "--generations",
            str(generations),
            "--population",
            str(population),
            "--universes",
            str(universes),
            "--task",
            task_name,
            "--state-dir",
            str(state_dir),
        ]
    else:
        cmd = [
            sys.executable,
            str(script),
            "evolve",
            "--seed",
            str(seed),
            "--generations",
            str(generations),
            "--population",
            str(population),
            "--universes",
            str(universes),
            "--task",
            task_name,
            "--state-dir",
            str(state_dir),
        ]
        if mode:
            cmd.extend(["--mode", mode])
    if resume:
        cmd.append("--resume")
    if not freeze_eval:
        cmd.append("--no-freeze-eval")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        return float("inf")
    snapshot = _load_state_snapshot(state_dir)
    if not snapshot:
        return float("inf")
    return _current_best_score(snapshot)

def _autopatch_probe_score(
    mode: str,
    task_name: str,
    seed: int = 1337,
    generations: int = 6,
    population: int = 32,
    universes: int = 1,
    freeze_eval: bool = True,
) -> float:
    task = TaskSpec(name=task_name)
    gs = run_multiverse(
        seed,
        task,
        generations,
        population,
        universes,
        resume=False,
        save_every=0,
        mode=mode,
        freeze_eval=freeze_eval,
    )
    if not gs.universes:
        return float("inf")
    best_snapshot = next((u for u in gs.universes if u.get("uid") == gs.selected_uid), gs.universes[0])
    return float(best_snapshot.get("best_score", float("inf")))

def _probe_score(script: Path, mode: str, task_name: str) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "autopatch-probe",
                "--mode",
                mode,
                "--task",
                task_name,
                "--state-dir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode != 0:
        return float("inf")
    output = result.stdout.strip().splitlines()
    if not output:
        return float("inf")
    try:
        return float(output[-1].strip())
    except Exception:
        return float("inf")

def _replace_source_segment(source: str, old: str, new: str) -> str:
    if old not in source:
        return source
    return source.replace(old, new, 1)

def _mutate_hyperparameter(
    tree: ast.AST,
    source: str,
    param_name: str,
    rng: random.Random,
) -> Tuple[str, str, Optional[float]]:
    ranges: Dict[str, Tuple[float, float]] = {
        "mutation_rate": (0.05, 0.95),
        "crossover_rate": (0.0, 0.9),
        "complexity_lambda": (1e-5, 1e-2),
        "epsilon_explore": (0.05, 0.5),
    }
    int_ranges: Dict[str, Tuple[int, int]] = {
        "adapt_steps": (4, 16),
    }
    target_node: Optional[ast.AnnAssign] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MetaState":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.target.id == param_name:
                        target_node = stmt
                        break
    if not target_node:
        return source, "", None
    old_segment = ast.get_source_segment(source, target_node.value) or ""
    if param_name in int_ranges:
        low, high = int_ranges[param_name]
        new_value = rng.randint(low, high)
        new_segment = str(new_value)
    else:
        low, high = ranges.get(param_name, (0.0, 1.0))
        new_value = rng.uniform(low, high)
        new_segment = f"{new_value:.6f}"
    new_source = _replace_source_segment(source, old_segment, new_segment)
    return new_source, f"L1:{param_name}", float(new_value)

def _mutate_operator(tree: ast.AST, source: str, rng: random.Random) -> Tuple[str, str]:
    target_assign: Optional[ast.Assign] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "act":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                        if stmt.targets[0].id == "op_weights" and isinstance(stmt.value, ast.Dict):
                            target_assign = stmt
                            break
    if not target_assign or not isinstance(target_assign.value, ast.Dict):
        return source, ""
    new_source = source
    for key_node, value_node in zip(target_assign.value.keys, target_assign.value.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        key = key_node.value
        if key not in ("insert_assign", "list_manip"):
            continue
        old_segment = ast.get_source_segment(source, value_node) or ""
        offset = rng.uniform(0.0, 0.5)
        if key == "list_manip":
            offset = rng.uniform(0.0, 0.3)
        if "op_scale" not in old_segment:
            continue
        if "op_scale +" in old_segment:
            new_segment = re.sub(r"op_scale\s*\+\s*[-+]?\d*\.?\d+", f"op_scale + {offset:.3f}", old_segment)
        else:
            new_segment = re.sub(r"op_scale\s*-\s*[-+]?\d*\.?\d+", f"op_scale - {offset:.3f}", old_segment)
        if new_segment == old_segment:
            continue
        new_source = _replace_source_segment(new_source, old_segment, new_segment)
    return new_source, "L2:op_weights"

def _mutate_evaluation(tree: ast.AST, source: str, rng: random.Random) -> Tuple[str, str]:
    weights = {
        "SCORE_W_HOLD": rng.uniform(0.45, 0.7),
        "SCORE_W_STRESS": rng.uniform(0.2, 0.6),
        "SCORE_W_TRAIN": rng.uniform(0.0, 0.2),
    }
    new_source = source
    changed = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in weights:
                old_segment = ast.get_source_segment(source, node.value) or ""
                new_segment = f"{weights[name]:.6f}"
                if old_segment:
                    new_source = _replace_source_segment(new_source, old_segment, new_segment)
                    changed = True
    return new_source, "L3:score_weights" if changed else ""

def _evaluate_patch_candidate(
    patch_code: str,
    baseline_score: float,
    mode: str,
    task_name: str,
) -> Tuple[bool, float, float]:
    """Evaluate a patch candidate. Returns (accepted, improvement, new_score)."""
    min_improvement_threshold = 0.03
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(patch_code)
        tmp = Path(f.name)
    try:
        new_score = _probe_score(tmp, mode, task_name)
    finally:
        tmp.unlink(missing_ok=True)
    if not math.isfinite(baseline_score) or baseline_score <= 0:
        return False, 0.0, new_score
    improvement = (baseline_score - new_score) / baseline_score
    return improvement >= min_improvement_threshold, improvement, new_score

def _select_best_patch(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [c for c in candidates if c.get("improvement", 0.0) > 0.0]
    if not valid:
        return None
    return max(valid, key=lambda c: (c["improvement"], -c["diff_size"]))

def _safe_apply_patch(self_path: Path, new_code: str) -> bool:
    backup_path = self_path.with_suffix(".py.bak")
    shutil.copy(self_path, backup_path)
    try:
        ast.parse(new_code)
        self_path.write_text(new_code, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(self_path), "selftest"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError("Selftest failed")
        return True
    except Exception:
        shutil.copy(backup_path, self_path)
        return False

def _log_autopatch_attempt(record: Dict[str, Any]) -> None:
    log_path = STATE_DIR / "autopatch_log.jsonl"
    safe_mkdir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def run_deep_autopatch(
    levels: List[int],
    candidates: int = 4,
    apply: bool = True,
    mode: str = "solver",
) -> Dict[str, Any]:
    """
    True RSI self-modification system with fitness-gated acceptance and rollback safety.
    Core change: evolve after mutation instead of re-evaluating the same code.
    """
    script = Path(__file__).resolve()
    source = script.read_text(encoding="utf-8")
    state_snapshot = _load_state_snapshot(STATE_DIR)
    task_name = (state_snapshot or {}).get("task", {}).get("name", TaskSpec().name)
    seed = int((state_snapshot or {}).get("base_seed", 1337))
    universes = max(1, len((state_snapshot or {}).get("universes", [])) or 1)
    pool_len = 0
    if state_snapshot and state_snapshot.get("universes"):
        pool_len = len(state_snapshot["universes"][0].get("pool", []))
    population = max(64, pool_len, 32)
    baseline = _current_best_score(state_snapshot)
    if not math.isfinite(baseline) or baseline <= 0:
        baseline = _probe_score(script, mode, task_name)
    print(f"[AUTOPATCH L{levels}] Baseline: {baseline:.4f}")

    rng = random.Random(int(time.time()) % 100000)
    patch_candidates: List[Dict[str, Any]] = []
    attempt_idx = 0

    for level in levels:
        for _ in range(candidates):
            attempt_idx += 1
            tree = ast.parse(source)
            patch_type = ""
            mutated_source = source
            mutated_state = copy.deepcopy(state_snapshot) if state_snapshot else None
            mutated_params: Dict[str, Any] = {}
            if level == 1:
                param = rng.choice(["mutation_rate", "crossover_rate", "complexity_lambda", "epsilon_explore", "adapt_steps"])
                mutated_source, patch_type, new_value = _mutate_hyperparameter(tree, source, param, rng)
                if new_value is None:
                    continue
                if param == "adapt_steps":
                    new_value = int(round(new_value))
                mutated_params[param] = new_value
                if mutated_state:
                    for u in mutated_state.get("universes", []):
                        meta = u.get("meta", {})
                        meta[param] = new_value
                        u["meta"] = meta
            elif level == 2:
                mutated_source, patch_type = _mutate_operator(tree, source, rng)
            elif level == 3:
                mutated_source, patch_type = _mutate_evaluation(tree, source, rng)
            if not patch_type or (mutated_source == source and not mutated_params):
                continue
            diff = unified_diff(source, mutated_source, str(script))
            diff_size = len(diff.splitlines())
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_state_dir = Path(tmpdir)
                if state_snapshot:
                    _clone_state_dir(STATE_DIR, tmp_state_dir)
                    if mutated_state:
                        _write_state_snapshot(tmp_state_dir, mutated_state)
                script_path = script
                if mutated_source != source:
                    script_path = tmp_state_dir / script.name
                    script_path.write_text(mutated_source, encoding="utf-8")
                attempt_seed = seed + attempt_idx
                print(f"[DEBUG] Running evolution with params: {mutated_params}")
                new_score = _autopatch_evolve_score(
                    script_path,
                    tmp_state_dir,
                    mode,
                    task_name,
                    attempt_seed,
                    generations=15,
                    population=population,
                    universes=universes,
                    resume=state_snapshot is not None,
                    freeze_eval=True,
                )
                print(f"[DEBUG] Evolution returned best_score: {new_score}")
            improvement = baseline - new_score
            accepted = improvement > 0
            record = {
                "level": level,
                "patch_type": patch_type,
                "old_score": baseline,
                "new_score": new_score,
                "improvement": improvement,
                "diff_size": diff_size,
                "accepted": accepted,
                "params": mutated_params,
            }
            _log_autopatch_attempt(record)
            if accepted:
                print(f"[AUTOPATCH] {patch_type} -> {new_score:.4f} (ACCEPT +{improvement:.2f})")
            else:
                print(f"[AUTOPATCH] {patch_type} -> {new_score:.4f} (REJECT)")
            patch_candidates.append(
                {
                    **record,
                    "diff": diff,
                    "code": mutated_source,
                    "state": mutated_state,
                }
            )

    best = _select_best_patch(patch_candidates)
    if not best:
        return {
            "applied": False,
            "improvement": 0.0,
            "old_score": baseline,
            "new_score": baseline,
            "patch_type": "",
            "diff": "",
    }

    if apply:
        applied = True
        if best["code"] != source:
            applied = _safe_apply_patch(script, best["code"])
        if applied and best.get("state"):
            _write_state_snapshot(STATE_DIR, best["state"])
        if applied:
            print(f"[RSI] Self-modified! Score: {best['old_score']:.4f} -> {best['new_score']:.4f}")
        return {
            "applied": applied,
            "improvement": best["improvement"],
            "old_score": best["old_score"],
            "new_score": best["new_score"],
            "patch_type": best["patch_type"],
            "diff": best["diff"],
        }

    return {
        "applied": False,
        "improvement": best["improvement"],
        "old_score": best["old_score"],
        "new_score": best["new_score"],
        "patch_type": best["patch_type"],
        "diff": best["diff"],
    }


def load_recent_scores(log_path: Path, n: int) -> List[float]:
    scores = []
    if not log_path.exists():
        return scores
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-n:]:
                try:
                    data = json.loads(line)
                    if "score" in data:
                        scores.append(float(data["score"]))
                except:
                    pass
    except Exception:
        pass
    return scores


def is_300s_stagnation(scores: List[float]) -> bool:
    if len(scores) < 5:
        return False
    return all(s > 300.0 for s in scores)


def _candidate_hash(code: str) -> str:
    return sha256(code)


def _slice_pair(xs: List[Any], ys: List[Any], n: int) -> Tuple[List[Any], List[Any]]:
    if not xs or not ys:
        return [], []
    k = min(n, len(xs), len(ys))
    return xs[:k], ys[:k]


def _prefilter_batch(batch: Batch, max_samples: int = 3) -> Batch:
    x_tr, y_tr = _slice_pair(batch.x_tr, batch.y_tr, max_samples)
    x_ho, y_ho = _slice_pair(batch.x_ho, batch.y_ho, max_samples)
    x_st, y_st = _slice_pair(batch.x_st, batch.y_st, max_samples)
    x_te, y_te = _slice_pair(batch.x_te, batch.y_te, max_samples)
    if not x_tr and x_ho:
        x_tr, y_tr = x_ho, y_ho
    if not x_ho and x_tr:
        x_ho, y_ho = x_tr, y_tr
    if not x_st and x_tr:
        x_st, y_st = x_tr, y_tr
    if not x_te and x_tr:
        x_te, y_te = x_tr, y_tr
    return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)


def _prefilter_eval(
    g: Genome,
    batch: Batch,
    mode: str,
    task_name: str,
    extra_env: Optional[Dict[str, Any]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_code,
) -> Tuple[bool, str, Optional[EvalResult]]:
    gate_ok, gate_reason = _hard_gate_ok(g.code, batch, mode, task_name, extra_env=extra_env)
    if not gate_ok:
        return False, f"hard_gate:{gate_reason}", None
    if mode in ("solver", "program"):
        ok, err = validator(g.code)
        if not ok:
            return False, f"validator:{err}", None
    mini_batch = _prefilter_batch(batch, max_samples=4)
    if mode == "learner":
        res = evaluate_learner(g, mini_batch, task_name)
    elif mode == "algo":
        res = evaluate_algo(g, mini_batch, task_name)
    else:
        res = evaluate(g, mini_batch, task_name, extra_env=extra_env, validator=validator)
    return res.ok, res.err or "", res


def _mutate_genome_with_meta(
    rng: random.Random,
    g: Genome,
    meta: MetaState,
    library: FunctionLibrary,
    op_bias: Optional[str] = None,
) -> Genome:
    stmts = g.statements[:]
    op_tag = "mutate"
    use_synth = rng.random() < 0.3 and bool(OPERATORS_LIB)
    if use_synth:
        synth_name = rng.choice(list(OPERATORS_LIB.keys()))
        steps = OPERATORS_LIB[synth_name].get("steps", [])
        stmts = apply_synthesized_op(rng, stmts, steps)
        op_tag = f"synth:{synth_name}"
    else:
        op = op_bias or meta.sample_op(rng)
        if op in OPERATORS:
            stmts = OPERATORS[op](rng, stmts)
        op_tag = f"mut:{op}"
    stmts = inject_helpers_into_statements(rng, list(stmts), library)
    return Genome(statements=stmts, parents=[g.gid], op_tag=op_tag)


def _synthesize_genome(
    rng: random.Random,
    pool: List[Genome],
    hint: Optional[str],
    library: FunctionLibrary,
) -> Genome:
    if not pool:
        return seed_genome(rng, hint)
    p1 = rng.choice(pool)
    p2 = rng.choice(pool)
    if len(p1.statements) <= 1 or len(p2.statements) <= 1:
        stmts = (p1.statements or []) + (p2.statements or [])
    else:
        cut1 = max(1, len(p1.statements) // 2)
        cut2 = max(1, len(p2.statements) // 2)
        stmts = p1.statements[:cut1] + p2.statements[-cut2:]
    if not stmts:
        stmts = ["return x"]
    stmts = inject_helpers_into_statements(rng, list(stmts), library)
    return Genome(statements=stmts, parents=[p1.gid, p2.gid], op_tag="synthesize")


def _fallback_template_genome(rng: random.Random, hint: Optional[str]) -> Genome:
    if hint:
        return seed_genome(rng, hint)
    return Genome(statements=["v0 = x", "return v0"], op_tag="fallback")


def _simplify_genome(rng: random.Random, g: Genome) -> Optional[Genome]:
    if len(g.statements) <= 1:
        return None
    stmts = g.statements[:]
    removable = [i for i, s in enumerate(stmts) if not s.strip().startswith("return ")]
    if not removable:
        return None
    idx = rng.choice(removable)
    del stmts[idx]
    return Genome(statements=stmts, parents=[g.gid], op_tag="simplify")


def _repair_genome(g: Genome) -> Genome:
    stmts = g.statements[:]
    has_return = any(s.strip().startswith("return ") for s in stmts)
    if not has_return:
        stmts.append("return x")
    else:
        if not any("x" in s for s in stmts if s.strip().startswith("return ")):
            stmts.append("return x")
    return Genome(statements=stmts, parents=[g.gid], op_tag="repair")


def _critic_refine(
    rng: random.Random,
    g: Genome,
    meta: MetaState,
    library: FunctionLibrary,
) -> List[Genome]:
    refined: List[Genome] = []
    simplified = _simplify_genome(rng, g)
    if simplified:
        refined.append(simplified)
    refined.append(_repair_genome(g))
    refined.append(_mutate_genome_with_meta(rng, g, meta, library, op_bias="modify_return"))
    return refined


def _adjust_creator_policy(
    policy: AgentPolicy,
    gate_pass_rate: float,
    gate_fail_reasons: collections.Counter,
) -> AgentPolicy:
    new_search = dict(policy.search_bias)
    generator_mode = policy.generator_mode
    if gate_pass_rate < policy.gate_target:
        generator_mode = "template"
        new_search["simplicity"] = clamp(new_search.get("simplicity", 0.5) + 0.3, 0.1, 2.0)
    if gate_fail_reasons.get("constant_output", 0) > 0:
        new_search["robustness"] = clamp(new_search.get("robustness", 0.5) + 0.2, 0.1, 2.0)
    return AgentPolicy(
        generator_mode=generator_mode,
        search_bias=new_search,
        gate_target=policy.gate_target,
        slice_seconds=policy.slice_seconds,
    )


def _critic_rank_score(res: EvalResult, policy: AgentPolicy) -> float:
    simplicity = policy.search_bias.get("simplicity", 0.0)
    robustness = policy.search_bias.get("robustness", 0.0)
    generalization = policy.search_bias.get("generalization", 0.0)
    perf = policy.search_bias.get("perf", 0.0)
    return (
        res.score
        + simplicity * 0.0005 * res.nodes
        + robustness * res.stress
        + generalization * res.hold
        + perf * res.train
    )


def _print_critic_summary(
    gate_pass: int,
    total_checked: int,
    adopted: bool,
    full_results_count: int,
    duplicate_count: int,
    scored_empty_count: int,
    gate_fail_reasons: collections.Counter,
    validator_fail_reasons: collections.Counter,
) -> None:
    gate_pass_rate = gate_pass / max(1, total_checked)
    adoption_rate = (1.0 if adopted else 0.0) / max(1, full_results_count)
    duplicate_ratio = duplicate_count / max(1, total_checked)
    top_gate = gate_fail_reasons.most_common(5)
    print(
        f"[Critic] gate_pass_rate={gate_pass_rate:.2f} adoption_rate={adoption_rate:.2f} "
        f"duplicate_ratio={duplicate_ratio:.2f} scored_empty={scored_empty_count}"
    )
    if top_gate:
        print("[Critic] top gate failures:", ", ".join(f"{k}:{v}" for k, v in top_gate))
    else:
        print("[Critic] top gate failures: none")
    if validator_fail_reasons:
        top_validator = validator_fail_reasons.most_common(3)
        print("[Critic] top validator failures:", ", ".join(f"{k}:{v}" for k, v in top_validator))


