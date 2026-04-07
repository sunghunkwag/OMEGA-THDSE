from __future__ import annotations

import ast
import copy
import json
import math
import multiprocessing as mp
import random
import re
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from cognitive_core_engine.governance.utils import now_ms, sha256, clamp, safe_mkdir, read_json, write_json


@dataclass
class InventionProgramCandidate:
    candidate_id: str
    code: str
    origin: str
    parent_id: Optional[str] = None
    score: float = 0.011744
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, int] = field(default_factory=dict)


class InventionRepresentation:
    """Expandable grammar and primitives.

    This enables invention by allowing new control patterns to be introduced
    dynamically, rather than committing to a fixed syntax whitelist.
    """

    def __init__(self) -> None:
        self.grammar: Dict[str, List[Callable[["InventionRepresentation"], str]]] = {
            "program": [self._base_program],
            "solver": [self._solver_template],
            "control": [self._loop_control, self._recursion_control],
            "strategy": [
                self._greedy_strategy,
                self._dp_strategy,
                self._divide_conquer_strategy,
                self._search_strategy,
            ],
        }
        self.library: List[str] = []

    def add_production(self, symbol: str, producer: Callable[["InventionRepresentation"], str]) -> None:
        self.grammar.setdefault(symbol, []).append(producer)

    def expand(self, symbol: str) -> str:
        options = self.grammar.get(symbol, [])
        if not options:
            raise ValueError(f"No productions for symbol: {symbol}")
        return random.choice(options)(self)

    def _base_program(self, _: "InventionRepresentation") -> str:
        helpers = "\n\n".join(self.library) if self.library else ""
        solver = self.expand("solver")
        parts = []
        if helpers:
            parts.append(helpers)
        parts.append(solver)
        return "\n\n".join(parts).strip()

    def _solver_template(self, _: "InventionRepresentation") -> str:
        control = self.expand("control")
        strategy = self.expand("strategy")
        header = textwrap.dedent(
            """
            def solve(task):
                \"\"\"Return the solution for the provided task.

                Generated as a full Python function so new control flow patterns
                can be invented, replaced, or expanded.
                \"\"\"
            """
        ).strip()
        return f"{header}\n{control}\n{strategy}"

    def _loop_control(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                for attempt in range(3):
                    if getattr(task, 'hint', None):
                        break
                """
            ).strip(),
            "    ",
        )

    def _recursion_control(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                def recur(state, depth):
                    if depth <= 0:
                        return state
                    return recur(state, depth - 1)
                recur(None, 1)
                """
            ).strip(),
            "    ",
        )

    def _greedy_strategy(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    return [x + 1 for x in task.input]
                if task.kind == 'path':
                    return task.heuristic_path()
                if task.kind == 'transform':
                    return ''.join(sorted(task.input))
                if task.kind == 'aggregate':
                    if getattr(task, 'hint', None) == 'max':
                        return max(task.input)
                    if getattr(task, 'hint', None) == 'min':
                        return min(task.input)
                    if getattr(task, 'hint', None) == 'len':
                        return len(task.input)
                    return sum(task.input)
                return task.fallback()
                """
            ).strip(),
            "    ",
        )

    def _dp_strategy(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    dp = {0: task.input[0] if task.input else 0}
                    for i in range(1, len(task.input)):
                        dp[i] = dp[i - 1] + task.input[i]
                    return [dp[i] for i in range(len(task.input))]
                if task.kind == 'path':
                    return task.shortest_path()
                if task.kind == 'transform':
                    memo = {}
                    def best(s):
                        if s in memo:
                            return memo[s]
                        if not s:
                            return ''
                        memo[s] = min(s[0] + best(s[1:]), ''.join(sorted(s)))
                        return memo[s]
                    return best(task.input)
                if task.kind == 'aggregate':
                    return sum(task.input)
                return task.fallback()
                """
            ).strip(),
            "    ",
        )

    def _divide_conquer_strategy(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    def combine(arr):
                        if len(arr) <= 1:
                            return arr
                        mid = len(arr) // 2
                        left = combine(arr[:mid])
                        right = combine(arr[mid:])
                        return [sum(left)] + [sum(right)]
                    return combine(task.input)
                if task.kind == 'path':
                    return task.path_via_split()
                if task.kind == 'transform':
                    def merge_sort(s):
                        if len(s) <= 1:
                            return s
                        mid = len(s) // 2
                        left = merge_sort(s[:mid])
                        right = merge_sort(s[mid:])
                        result = ''
                        while left and right:
                            if left[0] < right[0]:
                                result += left[0]
                                left = left[1:]
                            else:
                                result += right[0]
                                right = right[1:]
                        return result + left + right
                    return merge_sort(task.input)
                return task.fallback()
                """
            ).strip(),
            "    ",
        )

    def _search_strategy(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    best = None
                    for offset in range(1, 4):
                        candidate = [x + offset for x in task.input]
                        if best is None or sum(candidate) < sum(best):
                            best = candidate
                    return best
                if task.kind == 'path':
                    return task.search()
                if task.kind == 'transform':
                    best = min(task.input, ''.join(sorted(task.input)))
                    return best
                return task.fallback()
                """
            ).strip(),
            "    ",
        )


class InventionProgramGenerator:
    """Generate programs via grammar and composition.

    Composition across a growing library enables reuse of learned abstractions.
    """

    def __init__(self, representation: InventionRepresentation) -> None:
        self.representation = representation
        self.operator_weights: Dict[str, float] = {
            "grammar": 1.0,
            "compose": 1.0,
        }

    def generate(self) -> InventionProgramCandidate:
        operator = self._choose_operator()
        if operator == "compose" and self.representation.library:
            return self._compose_program()
        return self._grammar_program()

    def _choose_operator(self) -> str:
        total = sum(self.operator_weights.values())
        roll = random.random() * total
        cumulative = 0.0
        for name, weight in self.operator_weights.items():
            cumulative += weight
            if roll <= cumulative:
                return name
        return "grammar"

    def _grammar_program(self) -> InventionProgramCandidate:
        code = self.representation.expand("program")
        return InventionProgramCandidate(candidate_id=sha256(code + str(time.time())), code=code, origin="grammar")

    def _compose_program(self) -> InventionProgramCandidate:
        helpers = random.sample(self.representation.library, k=1)
        base = self.representation.expand("program")
        code = "\n\n".join(helpers + [base])
        return InventionProgramCandidate(candidate_id=sha256(code + str(time.time())), code=code, origin="compose")


@dataclass
class InventionTask:
    kind: str
    input: Any
    expected: Any
    hint: Optional[str] = None
    descriptor: Dict[str, Any] = field(default_factory=dict)

    def heuristic_path(self) -> Any:
        return self.expected

    def shortest_path(self) -> Any:
        return self.expected

    def path_via_split(self) -> Any:
        return self.expected

    def search(self) -> Any:
        return self.expected

    def fallback(self) -> Any:
        return self.expected


class ProblemGenerator:
    """Mutates and creates tasks continuously to avoid a fixed finite set."""

    def __init__(self) -> None:
        self.seed = 0
        self.base_kinds = ["sequence", "path", "transform", "aggregate"]
        self.transform_ops = ["sort", "reverse", "unique", "shift"]
        self.aggregate_ops = ["sum", "max", "min", "len"]

    def generate_tasks(
        self,
        count: int = 3,
        parents: Optional[List[InventionTask]] = None,
    ) -> List[InventionTask]:
        tasks: List[InventionTask] = []
        for _ in range(count):
            self.seed += 1
            random.seed(self.seed + random.randint(0, 9999))
            if parents and random.random() < 0.5:
                parent = random.choice(parents)
                tasks.append(self.mutate_task(parent))
            else:
                tasks.append(self.create_task())
        return tasks

    def create_task(self) -> InventionTask:
        kind = random.choice(self.base_kinds + [f"transform:{random.choice(self.transform_ops)}"])
        if kind == "sequence":
            data = [random.randint(1, 7) for _ in range(random.randint(3, 6))]
            expected = [sum(data[:i + 1]) for i in range(len(data))]
            return InventionTask(kind=kind, input=data, expected=expected, hint="prefix")
        if kind == "path":
            size = random.randint(3, 5)
            grid = [[random.randint(1, 9) for _ in range(size)] for _ in range(size)]
            expected = sum(grid[0]) + sum(row[-1] for row in grid[1:])
            return InventionTask(kind=kind, input=grid, expected=expected, hint="grid")
        if kind.startswith("transform"):
            op = kind.split(":", 1)[1] if ":" in kind else random.choice(self.transform_ops)
            word = "".join(random.choice("abcde") for _ in range(random.randint(4, 7)))
            expected = self._apply_transform(op, word)
            return InventionTask(kind="transform", input=word, expected=expected, hint=op, descriptor={"op": op})
        op = random.choice(self.aggregate_ops)
        data = [random.randint(1, 9) for _ in range(random.randint(3, 6))]
        expected = self._apply_aggregate(op, data)
        return InventionTask(kind="aggregate", input=data, expected=expected, hint=op, descriptor={"op": op})

    def mutate_task(self, task: InventionTask) -> InventionTask:
        if task.kind == "sequence":
            data = [x + random.choice([-1, 0, 1]) for x in task.input]
            data.append(random.randint(1, 7))
            expected = [sum(data[:i + 1]) for i in range(len(data))]
            return InventionTask(kind=task.kind, input=data, expected=expected, hint=task.hint, descriptor=task.descriptor)
        if task.kind == "path":
            grid = [row[:] for row in task.input]
            r = random.randint(0, len(grid) - 1)
            c = random.randint(0, len(grid[0]) - 1)
            grid[r][c] = max(1, grid[r][c] + random.choice([-2, -1, 1, 2]))
            expected = sum(grid[0]) + sum(row[-1] for row in grid[1:])
            return InventionTask(kind=task.kind, input=grid, expected=expected, hint=task.hint, descriptor=task.descriptor)
        if task.kind == "transform":
            op = task.descriptor.get("op", random.choice(self.transform_ops))
            word = task.input + random.choice("abcde")
            expected = self._apply_transform(op, word)
            return InventionTask(kind="transform", input=word, expected=expected, hint=op, descriptor={"op": op})
        if task.kind == "aggregate":
            op = task.descriptor.get("op", random.choice(self.aggregate_ops))
            data = task.input + [random.randint(1, 9)]
            expected = self._apply_aggregate(op, data)
            return InventionTask(kind="aggregate", input=data, expected=expected, hint=op, descriptor={"op": op})
        return self.create_task()

    def _apply_transform(self, op: str, word: str) -> str:
        if op == "sort":
            return "".join(sorted(word))
        if op == "reverse":
            return word[::-1]
        if op == "unique":
            return "".join(dict.fromkeys(word))
        if op == "shift":
            return "".join(chr(((ord(ch) - 97 + 1) % 26) + 97) for ch in word)
        return word

    def _apply_aggregate(self, op: str, data: List[int]) -> Any:
        if op == "sum":
            return sum(data)
        if op == "max":
            return max(data)
        if op == "min":
            return min(data)
        if op == "len":
            return len(data)
        return sum(data)


@dataclass
class RewardModel:
    performance_weight: float = 1.0
    transfer_weight: float = 0.7
    reuse_weight: float = 0.480815
    compression_weight: float = 0.3

    def score(self, metrics: Dict[str, float]) -> float:
        return (
            self.performance_weight * metrics.get("performance", 0.0)
            + self.transfer_weight * metrics.get("transfer", 0.0)
            + self.reuse_weight * metrics.get("reuse", 0.0)
            + self.compression_weight * metrics.get("compression", 0.0)
        )


@dataclass
class CandidateRecord:
    candidate_id: str
    parent_id: Optional[str]
    origin: str
    code: str
    score: float
    metrics: Dict[str, float]
    timestamp_ms: int


class InventionArchive:
    """Archive with lineage and a reusable subroutine pool."""

    def __init__(self, promotion_threshold: int = 2) -> None:
        self.records: List[CandidateRecord] = []
        self.lineage: Dict[str, CandidateRecord] = {}
        self.subroutine_pool: Dict[str, int] = {}
        self.promotion_threshold = promotion_threshold

    def add(self, candidate: InventionProgramCandidate) -> None:
        metrics = candidate.diagnostics.get("metrics", {})
        record = CandidateRecord(
            candidate_id=candidate.candidate_id,
            parent_id=candidate.parent_id,
            origin=candidate.origin,
            code=candidate.code,
            score=candidate.score,
            metrics=metrics,
            timestamp_ms=now_ms(),
        )
        self.records.append(record)
        self.lineage[candidate.candidate_id] = record

    def note_subroutine(self, snippet: str) -> bool:
        count = self.subroutine_pool.get(snippet, 0) + 1
        self.subroutine_pool[snippet] = count
        return count >= self.promotion_threshold


class Searcher:
    name: str = "base"

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        raise NotImplementedError


class LocalEditSearcher(Searcher):
    name = "local_edit"

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        source = representation.expand("program")
        if archive.records:
            source = random.choice(archive.records).code
        mutated = self._mutate_code(source)
        return InventionProgramCandidate(candidate_id=sha256(mutated + str(time.time())), code=mutated, origin=self.name)

    def _mutate_code(self, code: str) -> str:
        tree = ast.parse(code)
        constants = [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, int)]
        if constants:
            node = random.choice(constants)
            node.value = node.value + random.choice([-1, 1])
            return ast.unparse(tree)
        return code.replace("range(3)", "range(4)", 1)


class StructuralComposeSearcher(Searcher):
    name = "structural_compose"

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        helpers = []
        if representation.library:
            helpers.extend(random.sample(representation.library, k=min(2, len(representation.library))))
        if archive.subroutine_pool:
            helpers.extend(random.sample(list(archive.subroutine_pool.keys()), k=min(1, len(archive.subroutine_pool))))
        base = representation.expand("program")
        code = "\n\n".join(helpers + [base])
        return InventionProgramCandidate(candidate_id=sha256(code + str(time.time())), code=code, origin=self.name)


class RepresentationEditSearcher(Searcher):
    name = "representation_edit"

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        def new_strategy(_: InventionRepresentation) -> str:
            return textwrap.indent(
                textwrap.dedent(
                    """
                    if task.kind == 'aggregate':
                        if getattr(task, 'hint', None) == 'max':
                            return max(task.input)
                        if getattr(task, 'hint', None) == 'min':
                            return min(task.input)
                        return sum(task.input)
                    """
                ).strip(),
                "    ",
            )

        representation.add_production("strategy", new_strategy)
        code = representation.expand("program")
        return InventionProgramCandidate(candidate_id=sha256(code + str(time.time())), code=code, origin=self.name)


class SearcherManager:
    def __init__(self, searchers: List[Searcher]) -> None:
        self.searchers = {s.name: s for s in searchers}
        self.weights: Dict[str, float] = {s.name: 1.0 for s in searchers}

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        searcher = self._select_searcher()
        candidate = self.searchers[searcher].propose(representation, archive, problem_generator)
        candidate.origin = searcher
        return candidate

    def _select_searcher(self) -> str:
        total = sum(self.weights.values())
        roll = random.random() * total
        cumulative = 0.0
        for name, weight in self.weights.items():
            cumulative += weight
            if roll <= cumulative:
                return name
        return next(iter(self.weights))

    def update_weight(self, searcher: str, delta: float) -> None:
        self.weights[searcher] = clamp(self.weights.get(searcher, 1.0) + delta, 0.2, 5.0)


@dataclass
class BudgetLevel:
    name: str
    task_count: int
    transfer_count: int
    survivors: int


class BudgetLadderPolicy:
    """Budget ladder (B1..B4) where only survivors advance."""

    def __init__(self) -> None:
        self.levels = [
            BudgetLevel("B1", task_count=2, transfer_count=1, survivors=4),
            BudgetLevel("B2", task_count=3, transfer_count=2, survivors=3),
            BudgetLevel("B3", task_count=4, transfer_count=3, survivors=2),
            BudgetLevel("B4", task_count=5, transfer_count=4, survivors=1),
        ]

    def run(
        self,
        candidates: List[InventionProgramCandidate],
        problem_generator: ProblemGenerator,
        evaluator: "InventionEvaluator",
        archive: InventionArchive,
        reward_model: RewardModel,
    ) -> List[InventionProgramCandidate]:
        survivors = candidates
        for level in self.levels:
            if not survivors:
                break
            tasks = problem_generator.generate_tasks(level.task_count)
            transfer_tasks = problem_generator.generate_tasks(level.transfer_count, parents=tasks)
            for candidate in survivors:
                evaluator.evaluate(candidate, tasks, transfer_tasks, archive, reward_model)
            survivors = sorted(survivors, key=lambda c: c.score, reverse=True)[: level.survivors]
        return survivors


class InventionEvaluator:
    """Execute candidates in isolated processes and score them.

    Failures become diagnostic signals, enabling the meta-controller to adapt.
    """

    def __init__(self) -> None:
        self.novelty_weight = 0.2
        self.archive_features: List[Dict[str, int]] = []

    def evaluate(
        self,
        candidate: InventionProgramCandidate,
        tasks: List[InventionTask],
        transfer_tasks: List[InventionTask],
        archive: "InventionArchive",
        reward_model: "RewardModel",
        timeout: float = 1.0,
    ) -> None:
        results: List[Tuple[bool, str]] = []
        for task in tasks:
            success, info = self._run_in_subprocess(candidate.code, task, timeout)
            results.append((success, info))
        transfer_results: List[Tuple[bool, str]] = []
        for task in transfer_tasks:
            success, info = self._run_in_subprocess(candidate.code, task, timeout)
            transfer_results.append((success, info))
        candidate.diagnostics["results"] = results
        candidate.diagnostics["transfer_results"] = transfer_results
        candidate.features = self._extract_features(candidate.code)
        metrics = self._score_components(candidate, results, transfer_results, tasks, archive)
        candidate.diagnostics["metrics"] = metrics
        candidate.score = reward_model.score(metrics)
        self.archive_features.append(candidate.features)

    def _extract_features(self, code: str) -> Dict[str, int]:
        tree = ast.parse(code)
        features: Dict[str, int] = {}
        for node in ast.walk(tree):
            name = type(node).__name__
            features[name] = features.get(name, 0) + 1
        return features

    def _run_in_subprocess(self, code: str, task: InventionTask, timeout: float) -> Tuple[bool, str]:
        queue: mp.Queue = mp.Queue()
        process = mp.Process(target=InventionEvaluator._evaluate_runner, args=(queue, code, task))
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            return False, "timeout"
        if queue.empty():
            return False, "no output"
        return queue.get()

    @staticmethod
    def _evaluate_runner(queue: mp.Queue, code: str, task: InventionTask) -> None:
        try:
            scope: Dict[str, Any] = {}
            exec(code, scope)
            if "solve" not in scope:
                queue.put((False, "missing solve"))
                return
            result = scope["solve"](task)
            queue.put((result == task.expected, repr(result)))
        except Exception:
            queue.put((False, traceback.format_exc()))

    def _score_components(
        self,
        candidate: InventionProgramCandidate,
        results: List[Tuple[bool, str]],
        transfer_results: List[Tuple[bool, str]],
        tasks: List[InventionTask],
        archive: "InventionArchive",
    ) -> Dict[str, float]:
        success_rate = sum(1 for ok, _ in results if ok) / max(1, len(results))
        transfer_rate = sum(1 for ok, _ in transfer_results if ok) / max(1, len(transfer_results))
        reuse = self._reuse_score(candidate.code, archive)
        compression = self._compression_score(candidate.code)
        novelty = self._novelty(candidate.code)
        anti_trick = -0.2 if self._is_trivial(candidate.code, tasks) else 0.0
        return {
            "performance": success_rate + anti_trick,
            "transfer": transfer_rate,
            "reuse": reuse,
            "compression": compression,
            "novelty": novelty,
        }

    def _novelty(self, code: str) -> float:
        features = self._extract_features(code)
        if not self.archive_features:
            return 1.0
        distances = []
        for past in self.archive_features:
            distance = 0
            for key, value in features.items():
                distance += abs(value - past.get(key, 0))
            distances.append(distance)
        return sum(distances) / len(distances)

    def _is_trivial(self, code: str, tasks: List[InventionTask]) -> bool:
        if "return task.expected" in code:
            return True
        return all(len(repr(task.input)) < 10 for task in tasks) and "for" not in code

    def _reuse_score(self, code: str, archive: "InventionArchive") -> float:
        if not archive.subroutine_pool:
            return 0.0
        hits = 0
        for snippet in archive.subroutine_pool:
            if snippet in code:
                hits += 1
        return hits / max(1, len(archive.subroutine_pool))

    def _compression_score(self, code: str) -> float:
        node_count = sum(1 for _ in ast.walk(ast.parse(code)))
        return 1.0 / (1.0 + node_count / 50.0)


class InventionSelfModifier:
    """Adjusts generator, evaluator, and grammar based on diagnostics.

    This makes the system's learning rules and search operators mutable objects.
    """

    def __init__(
        self,
        representation: InventionRepresentation,
        evaluator: InventionEvaluator,
        searchers: SearcherManager,
        reward_model: RewardModel,
        budget_policy: BudgetLadderPolicy,
    ) -> None:
        self.representation = representation
        self.evaluator = evaluator
        self.searchers = searchers
        self.reward_model = reward_model
        self.budget_policy = budget_policy

    def adapt(self, candidate: InventionProgramCandidate) -> None:
        metrics = candidate.diagnostics.get("metrics", {})
        performance = metrics.get("performance", 0.0)
        transfer = metrics.get("transfer", 0.0)
        reuse = metrics.get("reuse", 0.0)
        if performance < 0.7:
            self.searchers.update_weight("local_edit", 0.2)
            self.evaluator.novelty_weight = min(1.5, self.evaluator.novelty_weight + 0.05)
            self._expand_grammar()
        if transfer < 0.5:
            self.searchers.update_weight("representation_edit", 0.2)
            self.reward_model.transfer_weight = min(1.2, self.reward_model.transfer_weight + 0.1)
        if reuse < 0.2:
            self.searchers.update_weight("structural_compose", 0.2)
            self.reward_model.reuse_weight = min(1.0, self.reward_model.reuse_weight + 0.1)
        if performance > 0.8 and transfer > 0.6:
            for level in self.budget_policy.levels:
                level.task_count = min(level.task_count + 1, 6)

    def _expand_grammar(self) -> None:
        def new_control(_: InventionRepresentation) -> str:
            return textwrap.indent(
                textwrap.dedent(
                    """
                    state = {}
                    if hasattr(task, 'hint'):
                        state['hint'] = task.hint
                    """
                ).strip(),
                "    ",
            )

        self.representation.add_production("control", new_control)


class InventionMetaController:
    """Coordinates generation, evaluation, self-modification, and retention.

    This creates a loop where algorithmic structures can be replaced entirely.
    """

    def __init__(self) -> None:
        self.representation = InventionRepresentation()
        self.evaluator = InventionEvaluator()
        self.problem_generator = ProblemGenerator()
        self.reward_model = RewardModel()
        self.archive = InventionArchive()
        self.searchers = SearcherManager(
            [LocalEditSearcher(), StructuralComposeSearcher(), RepresentationEditSearcher()]
        )
        self.budget_policy = BudgetLadderPolicy()
        self.self_modifier = InventionSelfModifier(
            self.representation,
            self.evaluator,
            self.searchers,
            self.reward_model,
            self.budget_policy,
        )
        self.candidate_history: List[InventionProgramCandidate] = []

    def run(self, iterations: int = 5) -> None:
        for _ in range(iterations):
            candidates = self._generate_candidates(pool_size=8)
            survivors = self.budget_policy.run(
                candidates,
                self.problem_generator,
                self.evaluator,
                self.archive,
                self.reward_model,
            )
            for candidate in survivors:
                self._retain(candidate)
                self.self_modifier.adapt(candidate)

    def _retain(self, candidate: InventionProgramCandidate) -> None:
        if candidate.score <= 0:
            return
        self.archive.add(candidate)
        self.candidate_history.append(candidate)
        self._extract_helpers(candidate.code)
        self._extract_subroutines(candidate.code)

    def _extract_helpers(self, code: str) -> None:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name != "solve":
                helper_code = ast.unparse(node)
                if helper_code not in self.representation.library:
                    self.representation.library.append(helper_code)

    def _extract_subroutines(self, code: str) -> None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "solve":
                for child in node.body:
                    snippet = ast.unparse(child)
                    if self.archive.note_subroutine(snippet):
                        self._promote_subroutine(snippet)

    def _promote_subroutine(self, snippet: str) -> None:
        if snippet.strip().startswith("def "):
            if snippet not in self.representation.library:
                self.representation.library.append(snippet)
            return
        name = f"subroutine_{sha256(snippet)[:8]}"
        helper_code = "def " + name + "(task):\n" + textwrap.indent(snippet, "    ") + "\n    return None"
        if helper_code not in self.representation.library:
            self.representation.library.append(helper_code)

    def _generate_candidates(self, pool_size: int) -> List[InventionProgramCandidate]:
        candidates: List[InventionProgramCandidate] = []
        for _ in range(pool_size):
            candidate = self.searchers.propose(self.representation, self.archive, self.problem_generator)
            if self.archive.records:
                candidate.parent_id = random.choice(self.archive.records).candidate_id
            candidates.append(candidate)
        return candidates


def cmd_invention(args):
    random.seed(args.seed)
    mp.set_start_method("spawn", force=True)
    controller = InventionMetaController()
    start = time.time()
    controller.run(iterations=args.iterations)
    duration = time.time() - start
    print(f"Completed {len(controller.archive.records)} retained candidates in {duration:.2f}s")
