"""Benchmark suites extracted from NON_RSI_AGI_CORE_v5.py."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cognitive_core_engine.core.utils import stable_hash, now_ms
from cognitive_core_engine.core.memory import SharedMemory
from cognitive_core_engine.core.tools import (
    ToolRegistry, tool_write_note_factory, tool_write_artifact_factory,
    tool_evaluate_candidate, tool_tool_build_report,
)
from cognitive_core_engine.core.skills import SkillLibrary
from cognitive_core_engine.core.project_graph import ProjectGraph
from cognitive_core_engine.core.environment import TaskSpec, ResearchEnvironment
from cognitive_core_engine.core.agent import Agent, AgentConfig


def _make_benchmark_stack(seed: int) -> Tuple[ResearchEnvironment, Agent, ProjectGraph]:
    random.seed(seed)
    env = ResearchEnvironment(seed=seed)
    tools = ToolRegistry()
    mem = SharedMemory()
    skills = SkillLibrary()
    agent = Agent(AgentConfig(name="bench_agent", role="general"), tools, mem, skills)

    tools.register("write_note", tool_write_note_factory(mem))
    tools.register("write_artifact", tool_write_artifact_factory(mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    projects = ProjectGraph()
    return env, agent, projects


def _run_benchmark_step(
    env: ResearchEnvironment,
    agent: Agent,
    projects: ProjectGraph,
    task: TaskSpec,
    budget: int,
) -> Dict[str, Any]:
    obs = env.make_observation(task, budget)
    proj_node = projects.pick_node_for_round(task.name)
    return agent.act_on_project(env, proj_node, obs)


def _adb_apply_rule(rule: str, params: Dict[str, int], seq: List[int]) -> List[int]:
    if rule == "reverse":
        return list(reversed(seq))
    if rule == "sort_unique":
        out: List[int] = []
        seen = set()
        for val in sorted(seq):
            if val not in seen:
                seen.add(val)
                out.append(val)
        return out
    if rule == "add_then_filter":
        delta = params.get("delta", 0)
        threshold = params.get("threshold", 0)
        return [val + delta for val in seq if val + delta >= threshold]
    if rule == "window_sum":
        width = max(1, params.get("width", 2))
        return [sum(seq[i:i + width]) for i in range(0, len(seq), width)]
    return seq


def _generate_adb_task(rng: random.Random) -> Dict[str, Any]:
    rule = rng.choice(["reverse", "sort_unique", "add_then_filter", "window_sum"])
    params = {}
    if rule == "add_then_filter":
        params = {"delta": rng.randint(-3, 3), "threshold": rng.randint(0, 6)}
    if rule == "window_sum":
        params = {"width": rng.randint(2, 3)}

    train_pairs = []
    for _ in range(3):
        length = rng.randint(3, 6)
        inp = [rng.randint(-4, 9) for _ in range(length)]
        out = _adb_apply_rule(rule, params, inp)
        train_pairs.append({"input": inp, "output": out})

    test_length = rng.randint(6, 9)
    test_input = [rng.randint(-6, 12) for _ in range(test_length)]
    adversarial = test_input[:]
    rng.shuffle(adversarial)
    if rule == "add_then_filter":
        adversarial = [val - params.get("delta", 0) for val in adversarial]
    test_output = _adb_apply_rule(rule, params, test_input)
    adversarial_output = _adb_apply_rule(rule, params, adversarial)
    return {
        "train": train_pairs,
        "test": {"input": test_input, "output": test_output},
        "adversarial": {"input": adversarial, "output": adversarial_output},
    }


def _solve_adb(task: Dict[str, Any], test_input: List[int]) -> Tuple[Any, int]:
    attempts = 0
    train_pairs = task.get("train", [])
    if train_pairs and all(
        pair.get("output") == list(reversed(pair.get("input", []))) for pair in train_pairs
    ):
        attempts += 1
        return list(reversed(test_input)), attempts
    if train_pairs and all(
        pair.get("output") == sorted(pair.get("input", [])) for pair in train_pairs
    ):
        attempts += 1
        return sorted(test_input), attempts
    attempts += 1
    return [], attempts


def _run_adb_suite_split(seed: int, trials: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    passes = 0
    robust_passes = 0
    total_attempts = 0
    runtimes_ms: List[int] = []

    for _ in range(trials):
        task = _generate_adb_task(rng)
        start = now_ms()
        base_input = task["test"]["input"]
        prediction, attempts = _solve_adb(task, base_input)
        commit_hash = stable_hash({"pred": prediction})
        end = now_ms()
        runtimes_ms.append(end - start)
        total_attempts += attempts
        base_ok = prediction == task["test"]["output"]

        robust_ok = False
        if base_ok:
            adv_input = task["adversarial"]["input"]
            adv_prediction, _ = _solve_adb(task, adv_input)
            robust_ok = adv_prediction == task["adversarial"]["output"]
            _ = commit_hash
        if base_ok:
            passes += 1
        if base_ok and robust_ok:
            robust_passes += 1

    trials_count = max(1, trials)
    return {
        "pass_rate": passes / trials_count,
        "robust_pass_rate": robust_passes / trials_count,
        "discovery_cost": total_attempts / max(1, passes),
        "avg_runtime_ms_per_trial": sum(runtimes_ms) / max(1, len(runtimes_ms)),
    }


def run_adb_benchmark_suite(seed: int, trials: int) -> Dict[str, Any]:
    train_result = _run_adb_suite_split(seed, trials)
    holdout_seed = _derive_holdout_seed(seed)
    holdout_result = _run_adb_suite_split(holdout_seed, trials)
    return {
        "suite": "ADB_v1",
        "seed": seed,
        "trials": trials,
        "train_pass_rate": train_result["pass_rate"],
        "holdout_pass_rate": holdout_result["pass_rate"],
        "discovery_cost": {
            "train": train_result["discovery_cost"],
            "holdout": holdout_result["discovery_cost"],
        },
        "robust_pass_rate": {
            "train": train_result["robust_pass_rate"],
            "holdout": holdout_result["robust_pass_rate"],
        },
        "avg_runtime_ms_per_trial": {
            "train": train_result["avg_runtime_ms_per_trial"],
            "holdout": holdout_result["avg_runtime_ms_per_trial"],
        },
    }


def _derive_holdout_seed(base_seed: int) -> int:
    nonce = "holdout-seed-v1"
    file_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    mix = f"{base_seed}:{file_hash}:{nonce}".encode("utf-8")
    return int(hashlib.sha256(mix).hexdigest()[:8], 16)


def _generate_program_synthesis_task(rng: random.Random) -> Dict[str, Any]:
    rule = rng.choice(["reverse", "sort", "dedup"])
    train_pairs = []
    for _ in range(3):
        length = rng.randint(3, 6)
        inp = [rng.randint(-3, 9) for _ in range(length)]
        if rule == "reverse":
            out = list(reversed(inp))
        elif rule == "sort":
            out = sorted(inp)
        else:
            out = []
            seen = set()
            for val in inp:
                if val not in seen:
                    seen.add(val)
                    out.append(val)
        train_pairs.append({"input": inp, "output": out})
    test_length = rng.randint(3, 6)
    test_input = [rng.randint(-3, 9) for _ in range(test_length)]
    if rule == "reverse":
        test_output = list(reversed(test_input))
    elif rule == "sort":
        test_output = sorted(test_input)
    else:
        test_output = []
        seen = set()
        for val in test_input:
            if val not in seen:
                seen.add(val)
                test_output.append(val)
    return {"train": train_pairs, "test": {"input": test_input, "output": test_output}}


def _generate_algo_micro_task(rng: random.Random) -> Dict[str, Any]:
    rule = rng.choice(["sum", "max", "count_even"])
    train_pairs = []
    for _ in range(3):
        length = rng.randint(3, 7)
        inp = [rng.randint(-5, 12) for _ in range(length)]
        if rule == "sum":
            out = sum(inp)
        elif rule == "max":
            out = max(inp)
        else:
            out = sum(1 for v in inp if v % 2 == 0)
        train_pairs.append({"input": inp, "output": out})
    test_length = rng.randint(3, 7)
    test_input = [rng.randint(-5, 12) for _ in range(test_length)]
    if rule == "sum":
        test_output = sum(test_input)
    elif rule == "max":
        test_output = max(test_input)
    else:
        test_output = sum(1 for v in test_input if v % 2 == 0)
    return {"train": train_pairs, "test": {"input": test_input, "output": test_output}}


def _generate_robustness_task(rng: random.Random) -> Dict[str, Any]:
    length = rng.randint(4, 8)
    base_input = [rng.randint(-4, 9) for _ in range(length)]
    base_output = sum(base_input)
    return {"base_input": base_input, "base_output": base_output}


def _solve_program_synthesis(task: Dict[str, Any]) -> Tuple[Any, int]:
    train_pairs = task.get("train", [])
    attempts = 0
    if train_pairs and all(pair.get("input") == pair.get("output") for pair in train_pairs):
        attempts += 1
        return task["test"]["input"], attempts
    if train_pairs and all(
        pair.get("output") == list(reversed(pair.get("input", []))) for pair in train_pairs
    ):
        attempts += 1
        return list(reversed(task["test"]["input"])), attempts
    attempts += 1
    return [], attempts


def _solve_algo_micro(task: Dict[str, Any]) -> Tuple[Any, int]:
    attempts = 1
    return 0, attempts


def _run_hard_suite_split(suite: str, seed: int, trials: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    passes = 0
    total_attempts = 0
    runtimes_ms: List[int] = []

    for _ in range(trials):
        start = now_ms()
        if suite == "program_synthesis_hard_v1":
            task = _generate_program_synthesis_task(rng)
            prediction, attempts = _solve_program_synthesis(task)
            expected = task["test"]["output"]
            solved = prediction == expected
        elif suite == "algo_micro_hard_v1":
            task = _generate_algo_micro_task(rng)
            prediction, attempts = _solve_algo_micro(task)
            expected = task["test"]["output"]
            solved = prediction == expected
        elif suite == "robustness_hard_v1":
            attempts = 1
            base_task = _generate_robustness_task(rng)
            base_input = base_task["base_input"]
            expected = base_task["base_output"]
            prediction, _ = _solve_algo_micro({"input": base_input})
            solved = prediction == expected
            if solved:
                shuffled = base_input[:]
                rng.shuffle(shuffled)
                prediction, _ = _solve_algo_micro({"input": shuffled})
                solved = prediction == expected
            if solved:
                noisy = base_input[:] + [0, 0]
                rng.shuffle(noisy)
                prediction, _ = _solve_algo_micro({"input": noisy})
                solved = prediction == expected
        else:
            raise ValueError(f"unknown suite: {suite}")
        end = now_ms()
        runtimes_ms.append(end - start)
        total_attempts += attempts
        if solved:
            passes += 1

    pass_rate = passes / max(1, trials)
    return {
        "pass_rate": pass_rate,
        "proposals_evaluated_per_solve": total_attempts / max(1, passes),
        "avg_runtime_ms_per_trial": sum(runtimes_ms) / max(1, len(runtimes_ms)),
    }


def run_hard_benchmark_suite(suite: str, seed: int, trials: int) -> Dict[str, Any]:
    train_result = _run_hard_suite_split(suite, seed, trials)
    holdout_seed = _derive_holdout_seed(seed)
    holdout_result = _run_hard_suite_split(suite, holdout_seed, trials)
    result = {
        "suite": suite,
        "seed": seed,
        "trials": trials,
        "train_pass_rate": train_result["pass_rate"],
        "holdout_pass_rate": holdout_result["pass_rate"],
        "avg_runtime_ms_per_trial": {
            "train": train_result["avg_runtime_ms_per_trial"],
            "holdout": holdout_result["avg_runtime_ms_per_trial"],
        },
    }
    if suite == "program_synthesis_hard_v1":
        result["proposals_evaluated_per_solve"] = {
            "train": train_result["proposals_evaluated_per_solve"],
            "holdout": holdout_result["proposals_evaluated_per_solve"],
        }
    return result


def run_benchmark_suite(suite: str, seed: int, trials: int) -> Dict[str, Any]:
    if suite == "ADB_v1":
        return run_adb_benchmark_suite(seed, trials)
    if suite in {"program_synthesis_hard_v1", "algo_micro_hard_v1", "robustness_hard_v1"}:
        return run_hard_benchmark_suite(suite, seed, trials)
    passes = 0
    total_rewards: List[float] = []
    skill_successes = 0
    attempts = 0

    for idx in range(trials):
        env, agent, projects = _make_benchmark_stack(seed + idx)

        if suite == "algo_micro_v1":
            task = next(t for t in env.tasks if t.domain == "algorithm")
            res = _run_benchmark_step(env, agent, projects, task, budget=12)
            reward = float(res.get("reward", 0.0))
            total_rewards.append(reward)
            if reward >= 0.02:
                passes += 1
        elif suite == "robustness_v1":
            rewards: List[float] = []
            for budget in (8, 12, 16):
                task = env.sample_task()
                res = _run_benchmark_step(env, agent, projects, task, budget=budget)
                rewards.append(float(res.get("reward", 0.0)))
            total_rewards.extend(rewards)
            if min(rewards) >= -0.01:
                passes += 1
        elif suite == "program_synthesis_v1":
            for _ in range(5):
                task = next(
                    t for t in env.tasks if t.name in ("verification_pipeline", "toolchain_speedup")
                )
                res = _run_benchmark_step(env, agent, projects, task, budget=12)
                total_rewards.append(float(res.get("reward", 0.0)))
                attempts += 1
            if agent.skills.list():
                passes += 1
                skill_successes += 1
        else:
            raise ValueError(f"unknown suite: {suite}")

    pass_rate = passes / max(1, trials)
    result = {
        "suite": suite,
        "seed": seed,
        "trials": trials,
        "pass_rate": pass_rate,
        "avg_reward": sum(total_rewards) / max(1, len(total_rewards)),
    }
    if suite == "program_synthesis_v1":
        proposals_per_solve = attempts / max(1, skill_successes)
        result["proposals_evaluated_per_solve"] = proposals_per_solve
    return result


def _load_arc_tasks(data_root: Path, suite: str) -> List[Dict[str, Any]]:
    if suite != "arc_agi2_public_eval":
        raise ValueError(f"unknown suite: {suite}")
    candidates = [
        data_root / "public_eval",
        data_root / "public",
        data_root / "evaluation",
        data_root / "eval",
        data_root / "public_eval_tasks",
    ]
    task_dir = next((p for p in candidates if p.exists()), None)
    if task_dir is None:
        raise FileNotFoundError(f"ARC public eval dataset not found under {data_root}")
    tasks = []
    for path in sorted(task_dir.glob("*.json")):
        tasks.append(json.loads(path.read_text(encoding="utf-8")))
    if not tasks:
        raise ValueError(f"no ARC tasks found in {task_dir}")
    return tasks


def _arc_constant_output(train_pairs: List[Dict[str, Any]]) -> Optional[List[List[int]]]:
    if not train_pairs:
        return None
    first = train_pairs[0].get("output")
    if first is None:
        return None
    for pair in train_pairs[1:]:
        if pair.get("output") != first:
            return None
    return first


def _arc_color_map(train_pairs: List[Dict[str, Any]]) -> Optional[Dict[int, int]]:
    mapping: Dict[int, int] = {}
    for pair in train_pairs:
        inp = pair.get("input")
        out = pair.get("output")
        if inp is None or out is None or len(inp) != len(out):
            return None
        if any(len(inp[r]) != len(out[r]) for r in range(len(inp))):
            return None
        for r in range(len(inp)):
            for c in range(len(inp[r])):
                src = int(inp[r][c])
                dst = int(out[r][c])
                if src in mapping and mapping[src] != dst:
                    return None
                mapping[src] = dst
    return mapping if mapping else None


def _arc_apply_color_map(grid: List[List[int]], mapping: Dict[int, int]) -> List[List[int]]:
    return [[mapping.get(int(cell), int(cell)) for cell in row] for row in grid]


def solve_arc_task(task: Dict[str, Any]) -> Tuple[List[List[int]], int]:
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])
    test_input = test_pairs[0].get("input") if test_pairs else None
    attempts = 0

    constant_output = _arc_constant_output(train_pairs)
    if constant_output is not None:
        attempts += 1
        return constant_output, attempts

    color_map = _arc_color_map(train_pairs)
    if color_map is not None and test_input is not None:
        attempts += 1
        return _arc_apply_color_map(test_input, color_map), attempts

    attempts += 1
    return test_input if test_input is not None else [], attempts


def run_arc_benchmark(suite: str, seed: int) -> Dict[str, Any]:
    data_root = Path(os.environ.get("ARC_GYM_PATH", ""))
    if not str(data_root):
        raise EnvironmentError("ARC_GYM_PATH is not set")
    tasks = _load_arc_tasks(data_root, suite)
    random.seed(seed)

    tasks_solved = 0
    total_attempts = 0
    runtimes_ms: List[int] = []

    for task in tasks:
        start = now_ms()
        prediction, attempts = solve_arc_task(task)
        end = now_ms()
        runtimes_ms.append(end - start)
        total_attempts += attempts
        test_pairs = task.get("test", [])
        expected = test_pairs[0].get("output") if test_pairs else None
        if expected is not None and prediction == expected:
            tasks_solved += 1

    tasks_total = len(tasks)
    accuracy = tasks_solved / max(1, tasks_total)
    return {
        "suite": suite,
        "seed": seed,
        "tasks_total": tasks_total,
        "tasks_solved": tasks_solved,
        "accuracy": accuracy,
        "avg_attempts_per_task": total_attempts / max(1, tasks_total),
        "avg_runtime_ms_per_task": sum(runtimes_ms) / max(1, tasks_total),
    }
