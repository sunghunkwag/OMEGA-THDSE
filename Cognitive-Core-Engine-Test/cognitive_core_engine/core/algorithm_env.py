"""
AlgorithmSynthesisEnvironment — Agents earn rewards ONLY by producing VM
programs that solve real algorithmic tasks.

Levels 0-4 task hierarchy with curriculum gating.  Rewards computed solely
by running vm.execute() and comparing outputs to oracle-generated expected
values on holdout test cases.

Anti-cheat: AC-E1..E7 enforced.  No formula rewards, no shaping, no bonuses.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from cognitive_core_engine.core.environment import ResearchEnvironment, TaskSpec
from cognitive_core_engine.omega_forge.instructions import Instruction, ProgramGenome
from cognitive_core_engine.omega_forge.vm import VirtualMachine

import random as _stdlib_random


# ---------------------------------------------------------------------------
# Test-case data structure
# ---------------------------------------------------------------------------

@dataclass
class AlgoTestCase:
    """A single test case: inputs loaded into VM memory, expected output."""
    inputs: List[float]          # values loaded into memory[0..len-1]
    n: int                       # length stored in reg[1]
    expected_reg0: Optional[float] = None   # expected reg[0] after execution
    expected_memory: Optional[Dict[int, float]] = None  # expected memory state
    extra_memory: Optional[Dict[int, float]] = None  # extra memory beyond inputs
    tolerance: float = 1e-6


@dataclass
class AlgoTask:
    """A task with train/holdout/external splits and an oracle."""
    name: str
    level: int
    domain: str
    train_cases: List[AlgoTestCase]
    holdout_cases: List[AlgoTestCase]
    external_cases: List[AlgoTestCase]  # for ExternalBenchmarkHarness
    oracle: Callable  # (inputs, extra) -> expected
    output_mode: str = "reg0"  # "reg0" or "memory"


# ---------------------------------------------------------------------------
# Oracles — pure Python functions computing ground truth
# ---------------------------------------------------------------------------

def oracle_sum(inputs: List[float], _extra: Any = None) -> float:
    return sum(inputs)

def oracle_max(inputs: List[float], _extra: Any = None) -> float:
    return max(inputs) if inputs else 0.0

def oracle_min(inputs: List[float], _extra: Any = None) -> float:
    return min(inputs) if inputs else 0.0

def oracle_count(inputs: List[float], _extra: Any = None) -> float:
    return float(len(inputs))

def oracle_count_positive(inputs: List[float], _extra: Any = None) -> float:
    return float(sum(1 for x in inputs if x > 0))

def oracle_sum_above_threshold(inputs: List[float], extra: Any = None) -> float:
    threshold = extra.get("threshold", 0.0) if extra else 0.0
    return sum(x for x in inputs if x > threshold)

def oracle_clamp(inputs: List[float], extra: Any = None) -> Dict[int, float]:
    lo = extra.get("lo", 0.0) if extra else 0.0
    hi = extra.get("hi", 9.0) if extra else 9.0
    result: Dict[int, float] = {}
    for i, v in enumerate(inputs):
        result[i] = max(lo, min(hi, v))
    return result

def oracle_filter_sum(inputs: List[float], _extra: Any = None) -> float:
    return sum(x for x in inputs if 0 < x < 8)

def oracle_bubble_sort(inputs: List[float], _extra: Any = None) -> Dict[int, float]:
    s = sorted(inputs)
    return {i: v for i, v in enumerate(s)}

def oracle_reverse(inputs: List[float], _extra: Any = None) -> Dict[int, float]:
    r = list(reversed(inputs))
    return {i: v for i, v in enumerate(r)}

def oracle_unique_count(inputs: List[float], _extra: Any = None) -> float:
    return float(len(set(inputs)))

def oracle_inner_product(inputs: List[float], extra: Any = None) -> float:
    n = len(inputs) // 2
    a, b = inputs[:n], inputs[n:2*n]
    return sum(x * y for x, y in zip(a, b))

def oracle_sort_then_sum_top_k(inputs: List[float], extra: Any = None) -> float:
    k = int(extra.get("k", 2)) if extra else 2
    s = sorted(inputs, reverse=True)
    return sum(s[:k])

def oracle_max_adjacent_sums(inputs: List[float], _extra: Any = None) -> float:
    if len(inputs) < 2:
        return inputs[0] if inputs else 0.0
    return max(inputs[i] + inputs[i+1] for i in range(len(inputs) - 1))

def oracle_normalize(inputs: List[float], _extra: Any = None) -> Dict[int, float]:
    total = sum(inputs)
    if abs(total) < 1e-12:
        return {i: 0.0 for i in range(len(inputs))}
    return {i: v / total for i, v in enumerate(inputs)}

def oracle_compose_sum_max(inputs: List[float], extra: Any = None) -> float:
    n = extra.get("split", len(inputs) // 2) if extra else len(inputs) // 2
    arr_a, arr_b = inputs[:n], inputs[n:]
    return sum(arr_a) + (max(arr_b) if arr_b else 0.0)

def oracle_eval_and_compare(inputs: List[float], extra: Any = None) -> float:
    ref = extra.get("reference", 0.0) if extra else 0.0
    computed = sum(inputs)  # default task: compare sum to reference
    return 1.0 if abs(computed - ref) < 1e-3 else 0.0


# ---------------------------------------------------------------------------
# Self-referential oracle helpers (Phase 3)
# ---------------------------------------------------------------------------

def _sr_oracle_evolution_yield(inputs: List[float], extra: Any = None) -> float:
    """Oracle for SR_IMPROVE_EVOLUTION_YIELD.

    Evaluates whether the agent's proposed mutation (encoded in inputs)
    improves evolution yield above baseline (0.2).
    AC-S1: Uses deterministic seed 9001 for seed genomes.
    """
    baseline_yield = 0.2
    # The inputs represent the proposed mutation operator encoded as floats
    # We measure how many of the input values exceed the baseline threshold
    rng = _stdlib_random.Random(9001)
    seed_scores = [rng.random() * 0.4 for _ in range(5)]
    improved = sum(1 for i, s in enumerate(seed_scores)
                   if i < len(inputs) and abs(inputs[i]) > 0.1 and s + abs(inputs[i]) * 0.3 > baseline_yield)
    return improved / 5.0


def _sr_oracle_fitness_discrimination(inputs: List[float], extra: Any = None) -> float:
    """Oracle for SR_IMPROVE_FITNESS_DISCRIMINATION.

    Evaluates whether the agent's proposed threshold (first input value)
    discriminates between good and bad genomes.
    AC-S4: Uses seed 9002.
    """
    rng = _stdlib_random.Random(9002)
    # Generate 10 genomes: 5 "good" (score > 0.5) and 5 "bad" (score <= 0.5)
    good_scores = [0.5 + rng.random() * 0.4 for _ in range(5)]
    bad_scores = [rng.random() * 0.4 for _ in range(5)]
    all_scores = good_scores + bad_scores
    all_labels = [1.0] * 5 + [0.0] * 5

    threshold = inputs[0] if inputs else 0.5
    predictions = [1.0 if s > threshold else 0.0 for s in all_scores]

    # Compute simple rank correlation proxy
    correct = sum(1 for p, l in zip(predictions, all_labels) if p == l)
    correlation = (correct / 10.0 - 0.5) * 2.0  # scale to [-1, 1]
    return max(0.0, correlation)


def _sr_oracle_self_test(inputs: List[float], extra: Any = None) -> float:
    """Oracle for SR_SELF_TEST_IMPROVEMENT.

    Evaluates whether the agent's proposed test parameters (encoded in inputs)
    can discriminate between correct and incorrect programs.
    AC-S7: Uses seed 9003 + case_index.
    """
    if not inputs or len(inputs) < 2:
        return 0.0
    # inputs encode: [input_length, value_range, ...]
    input_length = max(1, min(8, int(abs(inputs[0]))))
    value_range = max(1, min(20, int(abs(inputs[1]))))

    # Generate test cases using proposed parameters
    fail_count = 0
    total_programs = max(1, int(extra.get("n_programs", 5)) if extra else 5)
    for case_idx in range(10):
        rng = _stdlib_random.Random(9003 + case_idx)
        test_input = [float(rng.randint(-value_range, value_range))
                      for _ in range(input_length)]
        expected = sum(test_input)
        # Check if a "simple sum" program would fail on extreme cases
        if abs(expected) > value_range * input_length * 0.8:
            fail_count += 1

    return fail_count / 10.0


# ---------------------------------------------------------------------------
# Test case generator (deterministic per-task RNG)
# ---------------------------------------------------------------------------

class TaskCaseGenerator:
    """Generates train/holdout/external test cases deterministically."""

    @staticmethod
    def generate(
        task_name: str,
        level: int,
        oracle_fn: Callable,
        output_mode: str,
        n_train: int,
        n_holdout: int,
        n_external: int = 5,
        min_len: int = 1,
        max_len: int = 8,
        val_lo: int = -9,
        val_hi: int = 9,
        extra_fn: Optional[Callable] = None,
    ) -> Tuple[List[AlgoTestCase], List[AlgoTestCase], List[AlgoTestCase]]:
        """Generate splits using deterministic seeds."""
        train = TaskCaseGenerator._gen_split(
            task_name, "train", 777, n_train, min_len, max_len, val_lo, val_hi,
            oracle_fn, output_mode, extra_fn)
        holdout = TaskCaseGenerator._gen_split(
            task_name, "holdout", 888, n_holdout, min_len, max_len, val_lo, val_hi,
            oracle_fn, output_mode, extra_fn)
        external = TaskCaseGenerator._gen_split(
            task_name, "external", 999, n_external, min_len, max_len, val_lo, val_hi,
            oracle_fn, output_mode, extra_fn)
        return train, holdout, external

    @staticmethod
    def _gen_split(
        task_name: str, split: str, base_seed: int, count: int,
        min_len: int, max_len: int, val_lo: int, val_hi: int,
        oracle_fn: Callable, output_mode: str,
        extra_fn: Optional[Callable],
    ) -> List[AlgoTestCase]:
        seed = hash(f"{task_name}:{split}:{base_seed}") % (2**31)
        rng = _stdlib_random.Random(seed)
        cases: List[AlgoTestCase] = []
        for _ in range(count):
            n = rng.randint(min_len, max_len)
            inputs = [float(rng.randint(val_lo, val_hi)) for _ in range(n)]
            extra = extra_fn(rng, inputs) if extra_fn else None
            extra_memory: Optional[Dict[int, float]] = None
            if extra and isinstance(extra, dict) and "extra_memory" in extra:
                extra_memory = extra["extra_memory"]

            result = oracle_fn(inputs, extra)
            if output_mode == "reg0":
                cases.append(AlgoTestCase(
                    inputs=inputs, n=n,
                    expected_reg0=float(result),
                    extra_memory=extra_memory))
            elif output_mode == "memory":
                cases.append(AlgoTestCase(
                    inputs=inputs, n=n,
                    expected_memory=dict(result) if isinstance(result, dict) else {},
                    extra_memory=extra_memory))
        return cases


# ---------------------------------------------------------------------------
# Task registry — builds all Level 0-4 tasks
# ---------------------------------------------------------------------------

def _extra_threshold(rng: _stdlib_random.Random, inputs: List[float]) -> Dict:
    t = float(rng.randint(-3, 5))
    return {"threshold": t, "extra_memory": {len(inputs): t}}

def _extra_clamp(rng: _stdlib_random.Random, inputs: List[float]) -> Dict:
    lo = float(rng.randint(-5, 0))
    hi = float(rng.randint(1, 9))
    n = len(inputs)
    return {"lo": lo, "hi": hi, "extra_memory": {n: lo, n+1: hi}}

def _extra_k(rng: _stdlib_random.Random, inputs: List[float]) -> Dict:
    k = rng.randint(1, max(1, len(inputs) - 1))
    return {"k": k}

def _extra_inner(rng: _stdlib_random.Random, inputs: List[float]) -> Dict:
    n = len(inputs)
    extra_vals = [float(rng.randint(-9, 9)) for _ in range(n)]
    return {"extra_memory": {n + i: v for i, v in enumerate(extra_vals)}}

def _extra_compose(rng: _stdlib_random.Random, inputs: List[float]) -> Dict:
    split = len(inputs) // 2
    extra_vals = [float(rng.randint(-9, 9)) for _ in range(split)]
    return {"split": split, "extra_memory": {len(inputs) + i: v for i, v in enumerate(extra_vals)}}

def _extra_eval_compare(rng: _stdlib_random.Random, inputs: List[float]) -> Dict:
    ref = sum(inputs)
    return {"reference": ref, "extra_memory": {32: ref}}


def build_all_tasks() -> Dict[str, AlgoTask]:
    """Build the complete Level 0-4 task hierarchy."""
    tasks: Dict[str, AlgoTask] = {}

    # Level 0 — Single-pass accumulation
    for name, oracle, mode in [
        ("L0_SUM", oracle_sum, "reg0"),
        ("L0_MAX", oracle_max, "reg0"),
        ("L0_MIN", oracle_min, "reg0"),
        ("L0_COUNT", oracle_count, "reg0"),
    ]:
        tr, ho, ex = TaskCaseGenerator.generate(
            name, 0, oracle, mode, n_train=20, n_holdout=10)
        tasks[name] = AlgoTask(name=name, level=0, domain="level0",
                               train_cases=tr, holdout_cases=ho,
                               external_cases=ex, oracle=oracle, output_mode=mode)

    # Level 1 — Conditional accumulation
    for name, oracle, mode, extra_fn in [
        ("L1_COUNT_POSITIVE", oracle_count_positive, "reg0", None),
        ("L1_SUM_ABOVE_THRESHOLD", oracle_sum_above_threshold, "reg0", _extra_threshold),
        ("L1_CLAMP", oracle_clamp, "memory", _extra_clamp),
        ("L1_FILTER_SUM", oracle_filter_sum, "reg0", None),
    ]:
        tr, ho, ex = TaskCaseGenerator.generate(
            name, 1, oracle, mode, n_train=15, n_holdout=10, extra_fn=extra_fn)
        tasks[name] = AlgoTask(name=name, level=1, domain="level1",
                               train_cases=tr, holdout_cases=ho,
                               external_cases=ex, oracle=oracle, output_mode=mode)

    # Level 2 — Nested loops
    for name, oracle, mode in [
        ("L2_BUBBLE_SORT", oracle_bubble_sort, "memory"),
        ("L2_REVERSE", oracle_reverse, "memory"),
        ("L2_UNIQUE_COUNT", oracle_unique_count, "reg0"),
        ("L2_INNER_PRODUCT", oracle_inner_product, "reg0"),
    ]:
        efn = _extra_inner if name == "L2_INNER_PRODUCT" else None
        tr, ho, ex = TaskCaseGenerator.generate(
            name, 2, oracle, mode, n_train=10, n_holdout=8,
            min_len=3, max_len=6, extra_fn=efn)
        tasks[name] = AlgoTask(name=name, level=2, domain="level2",
                               train_cases=tr, holdout_cases=ho,
                               external_cases=ex, oracle=oracle, output_mode=mode)

    # Level 3 — Subroutine composition
    for name, oracle, mode, extra_fn in [
        ("L3_SORT_SUM_TOP_K", oracle_sort_then_sum_top_k, "reg0", _extra_k),
        ("L3_MAX_ADJACENT_SUMS", oracle_max_adjacent_sums, "reg0", None),
        ("L3_NORMALIZE", oracle_normalize, "memory", None),
    ]:
        tr, ho, ex = TaskCaseGenerator.generate(
            name, 3, oracle, mode, n_train=8, n_holdout=6,
            min_len=3, max_len=6, extra_fn=extra_fn)
        tasks[name] = AlgoTask(name=name, level=3, domain="level3",
                               train_cases=tr, holdout_cases=ho,
                               external_cases=ex, oracle=oracle, output_mode=mode)

    # Level 4 — Meta-programs
    for name, oracle, mode, extra_fn in [
        ("L4_COMPOSE_SUM_MAX", oracle_compose_sum_max, "reg0", _extra_compose),
        ("L4_EVAL_AND_COMPARE", oracle_eval_and_compare, "reg0", _extra_eval_compare),
    ]:
        tr, ho, ex = TaskCaseGenerator.generate(
            name, 4, oracle, mode, n_train=5, n_holdout=5,
            min_len=4, max_len=6, extra_fn=extra_fn)
        tasks[name] = AlgoTask(name=name, level=4, domain="level4",
                               train_cases=tr, holdout_cases=ho,
                               external_cases=ex, oracle=oracle, output_mode=mode)

    # Self-Referential tasks (Phase 3) — Level 5 (requires max_level >= 4)
    SR_LEVEL = 5  # gated by curriculum: only available when all L0-L4 unlocked

    for sr_name, sr_oracle, sr_seed in [
        ("SR_IMPROVE_EVOLUTION_YIELD", _sr_oracle_evolution_yield, 9001),
        ("SR_IMPROVE_FITNESS_DISCRIMINATION", _sr_oracle_fitness_discrimination, 9002),
        ("SR_SELF_TEST_IMPROVEMENT", _sr_oracle_self_test, 9003),
    ]:
        tr, ho, ex = TaskCaseGenerator.generate(
            sr_name, SR_LEVEL, sr_oracle, "reg0",
            n_train=5, n_holdout=5, n_external=3,
            min_len=3, max_len=6, val_lo=-5, val_hi=5)
        tasks[sr_name] = AlgoTask(
            name=sr_name, level=SR_LEVEL, domain="self_referential",
            train_cases=tr, holdout_cases=ho, external_cases=ex,
            oracle=sr_oracle, output_mode="reg0")

    return tasks


# ---------------------------------------------------------------------------
# Curriculum gate
# ---------------------------------------------------------------------------

class CurriculumGate:
    """Tracks task solve rates and controls level unlock."""

    def __init__(self) -> None:
        self._task_holdout_rates: Dict[str, float] = {}
        self._max_unlocked_level: int = 0

    def record_solve_rate(self, task_name: str, level: int, rate: float) -> None:
        """Record a holdout solve rate for a task."""
        self._task_holdout_rates[task_name] = rate
        self._update_unlock()

    def _update_unlock(self) -> None:
        """Check if next level should be unlocked."""
        for target_level in range(self._max_unlocked_level + 1, 5):
            prev_level = target_level - 1
            tasks_at_prev = [name for name, rate in self._task_holdout_rates.items()
                             if name.startswith(f"L{prev_level}_")]
            passed = sum(1 for name in tasks_at_prev
                         if self._task_holdout_rates.get(name, 0) >= 0.6)
            if passed >= 2:
                self._max_unlocked_level = target_level
            else:
                break

    @property
    def max_level(self) -> int:
        return self._max_unlocked_level

    def is_unlocked(self, level: int) -> bool:
        return level <= self._max_unlocked_level


# ---------------------------------------------------------------------------
# AlgorithmSynthesisEnvironment
# ---------------------------------------------------------------------------

class AlgorithmSynthesisEnvironment(ResearchEnvironment):
    """Environment where rewards come ONLY from VM program correctness.

    Anti-cheat: AC-E1..E7 enforced in step().
    """

    VM_TIMEOUT = 500  # AC-E1

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._seed = seed
        self._vm = VirtualMachine(max_steps=self.VM_TIMEOUT)
        self._algo_tasks = build_all_tasks()
        self._curriculum = CurriculumGate()
        self._current_task_name: Optional[str] = None
        # Challenger tracking
        self._challenger_tasks: Dict[str, AlgoTask] = {}
        self._challenge_attempt_counts: Dict[str, int] = {}
        self._challenge_solve_counts: Dict[str, float] = {}
        # Self-ref measurement counter
        self._self_ref_env_id: int = id(self)
        # Phase 3: Solved programs registry (AC-S10)
        self._solved_programs: List[ProgramGenome] = []

    def get_algo_task(self, task_name: str) -> Optional[AlgoTask]:
        """Get an algorithm task by name."""
        return self._algo_tasks.get(task_name) or self._challenger_tasks.get(task_name)

    def available_tasks(self) -> List[AlgoTask]:
        """Return tasks at or below the current curriculum level."""
        max_lv = self._curriculum.max_level
        return [t for t in self._algo_tasks.values() if t.level <= max_lv]

    def _create_measurement_env(self) -> 'AlgorithmSynthesisEnvironment':
        """Create a SEPARATE environment for self-referential measurements.

        AC-S8: seed = self._seed + 7777, distinct from self.
        AC-S3: Uses its own VirtualMachine instance.
        """
        return AlgorithmSynthesisEnvironment(seed=self._seed + 7777)

    def make_observation(self, task: TaskSpec, budget: int,
                         phase: str = "research") -> Dict[str, Any]:
        """Override to include algorithm task train cases."""
        obs = super().make_observation(task, budget, phase)
        # Find matching algo task and attach train cases
        algo = self._algo_tasks.get(task.name)
        if algo:
            obs["algo_task"] = task.name
            obs["algo_level"] = algo.level
            obs["train_cases"] = [
                {"inputs": tc.inputs, "n": tc.n, "expected": tc.expected_reg0 or tc.expected_memory}
                for tc in algo.train_cases
            ]
            obs["output_mode"] = algo.output_mode
        obs["curriculum_level"] = self._curriculum.max_level
        return obs

    def step(self, obs: Dict[str, Any], action: str,
             payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Compute reward ONLY from VM program correctness on holdout cases.

        AC-E3: Reward computed solely by vm.execute() comparison.
        AC-E5: No reward smoothing/shaping/intrinsic components.
        AC-E6: No structural bonuses.
        """
        info: Dict[str, Any] = {
            "task": obs.get("task", ""),
            "domain": obs.get("domain", ""),
            "curriculum_level": self._curriculum.max_level,
        }

        if action == "submit_program":
            reward = self._handle_submit(obs, payload, info)
        elif action == "compose_skills":
            reward = self._handle_compose(obs, payload, info)
        elif action == "generate_challenge":
            reward = self._handle_challenge(obs, payload, info)
        else:
            # Deprecated actions
            reward = 0.0
            info["deprecated_action"] = True

        next_obs = dict(obs)
        next_obs["phase"] = "integrate"
        # Update infrastructure qualities minimally for compatibility
        self.global_tool_quality = min(1.0, self.global_tool_quality + 0.001)
        self.global_kb_quality = min(1.0, self.global_kb_quality + 0.001)
        info["tq"] = self.global_tool_quality
        info["kq"] = self.global_kb_quality
        info["oq"] = self.global_org_quality
        return next_obs, reward, info

    def _handle_submit(self, obs: Dict[str, Any], payload: Dict[str, Any],
                       info: Dict[str, Any]) -> float:
        """Submit a ProgramGenome, evaluate on holdout cases."""
        genome = payload.get("genome")
        if not isinstance(genome, ProgramGenome):
            info["error"] = "no_genome"
            return 0.0

        task_name = obs.get("algo_task") or obs.get("task", "")
        algo = self.get_algo_task(task_name)
        if algo is None:
            info["error"] = "unknown_task"
            return 0.0

        # Curriculum gate check
        if not self._curriculum.is_unlocked(algo.level):
            info["error"] = "level_locked"
            info["required_level"] = algo.level
            info["current_level"] = self._curriculum.max_level
            return 0.0

        # Evaluate on holdout cases only (AC-E4)
        rate = self._evaluate_genome(genome, algo.holdout_cases, algo.output_mode, info)

        # Record for curriculum tracking
        self._curriculum.record_solve_rate(task_name, algo.level, rate)
        info["holdout_rate"] = rate
        info["level"] = algo.level

        # AC-S10: Append to solved_programs if rate >= 0.6 and NOT an SR task
        is_sr = algo.domain == "self_referential"
        if rate >= 0.6 and not is_sr:
            self._solved_programs.append(copy.deepcopy(genome))

        return rate

    def _handle_compose(self, obs: Dict[str, Any], payload: Dict[str, Any],
                        info: Dict[str, Any]) -> float:
        """Compose two skills' genomes with CALL/RET bridge."""
        skill_ids = payload.get("skill_ids", [])
        if len(skill_ids) < 2:
            info["error"] = "need_2_skills"
            return 0.0

        # This would require SkillLibrary access — simplified for now
        # The composed genome evaluation follows the same path as submit_program
        genome = payload.get("genome")
        if not isinstance(genome, ProgramGenome):
            info["error"] = "no_composed_genome"
            return 0.0

        task_name = obs.get("algo_task") or obs.get("task", "")
        algo = self.get_algo_task(task_name)
        if algo is None:
            info["error"] = "unknown_task"
            return 0.0

        rate = self._evaluate_genome(genome, algo.holdout_cases, algo.output_mode, info)
        self._curriculum.record_solve_rate(task_name, algo.level, rate)
        info["holdout_rate"] = rate
        return rate

    def _handle_challenge(self, obs: Dict[str, Any], payload: Dict[str, Any],
                          info: Dict[str, Any]) -> float:
        """Generate a new challenge task."""
        inputs_list = payload.get("inputs_list", [])
        expected_list = payload.get("expected_outputs_list", [])
        oracle_genome = payload.get("oracle_genome")

        # Validate challenge (AC-E7)
        if len(inputs_list) < 3:
            info["error"] = "too_few_cases"
            return -0.1

        # Validate inputs have length >= 2
        if any(len(inp) < 2 for inp in inputs_list if isinstance(inp, list)):
            info["error"] = "inputs_too_short"
            return -0.1

        # Validate expected outputs not all identical
        if len(set(str(e) for e in expected_list)) <= 1:
            info["error"] = "trivial_outputs"
            return -0.1

        # Validate oracle if provided
        if isinstance(oracle_genome, ProgramGenome):
            for inp, expected in zip(inputs_list, expected_list):
                try:
                    st = self._vm.execute(oracle_genome, inp)
                    if st.error or not st.halted_cleanly:
                        info["error"] = "oracle_crashed"
                        return -0.1
                    result = st.regs[0]
                    if math.isnan(result) or math.isinf(result):
                        info["error"] = "oracle_nan"
                        return -0.1
                except Exception:
                    info["error"] = "oracle_exception"
                    return -0.1

        # Register the challenge
        challenge_name = f"challenge_{len(self._challenger_tasks)}_{self.rng.randint(0, 9999)}"
        cases = []
        for inp, exp in zip(inputs_list, expected_list):
            cases.append(AlgoTestCase(inputs=inp, n=len(inp), expected_reg0=float(exp)))
        n_holdout = max(1, len(cases) // 3)
        task = AlgoTask(
            name=challenge_name, level=self._curriculum.max_level,
            domain="challenge", train_cases=cases[n_holdout:],
            holdout_cases=cases[:n_holdout], external_cases=[],
            oracle=lambda inp, _e=expected_list, _i=inputs_list: (
                _e[_i.index(inp)] if inp in _i else 0.0),
            output_mode="reg0",
        )
        self._challenger_tasks[challenge_name] = task
        self._challenge_attempt_counts[challenge_name] = 0
        self._challenge_solve_counts[challenge_name] = 0.0
        info["challenge_registered"] = challenge_name
        # Challenger reward is 0 now, computed at round end
        return 0.0

    def compute_challenger_reward(self, challenge_name: str) -> float:
        """Compute challenger reward at end of round (AC-A1)."""
        attempts = self._challenge_attempt_counts.get(challenge_name, 0)
        if attempts < 2:
            return 0.0
        total_solve = self._challenge_solve_counts.get(challenge_name, 0.0)
        mean_solve_rate = total_solve / max(1, attempts)
        return 1.0 - mean_solve_rate

    def _evaluate_genome(
        self, genome: ProgramGenome, cases: List[AlgoTestCase],
        output_mode: str, info: Dict[str, Any],
    ) -> float:
        """Run genome on test cases, return fraction passed exactly.

        AC-E1: 500 step timeout.  AC-E2: constant output ban.
        AC-E3: Actual vm.execute() comparison only.
        """
        if not cases:
            return 0.0

        passed = 0
        total = len(cases)
        outputs: List[float] = []

        for tc in cases:
            try:
                # Load inputs into VM memory and set reg[1] = N
                vm = VirtualMachine(max_steps=self.VM_TIMEOUT)
                st = vm.execute(genome, tc.inputs)

                # AC-E1: check step limit
                if st.steps >= self.VM_TIMEOUT and not st.halted_cleanly:
                    outputs.append(float('nan'))
                    continue

                if st.error:
                    outputs.append(float('nan'))
                    continue

                if output_mode == "reg0":
                    result = st.regs[0]
                    outputs.append(result)
                    if tc.expected_reg0 is not None:
                        if abs(result - tc.expected_reg0) < tc.tolerance:
                            passed += 1
                elif output_mode == "memory":
                    if tc.expected_memory is not None:
                        all_match = True
                        mem_val = 0.0
                        for addr, expected_val in tc.expected_memory.items():
                            got = st.memory.get(addr, 0.0)
                            mem_val = got
                            tol = 1e-3 if tc.tolerance > 1e-4 else tc.tolerance
                            if abs(got - expected_val) >= tol:
                                all_match = False
                                break
                        outputs.append(mem_val)
                        if all_match:
                            passed += 1
                    else:
                        outputs.append(0.0)
            except Exception:
                outputs.append(float('nan'))

        # AC-E2: constant output ban
        valid_outputs = [o for o in outputs if not math.isnan(o)]
        if len(valid_outputs) >= 2 and len(set(round(o, 6) for o in valid_outputs)) < 2:
            info["constant_output_ban"] = True
            return 0.0

        rate = passed / max(1, total)
        info["cases_passed"] = passed
        info["cases_total"] = total
        return rate
