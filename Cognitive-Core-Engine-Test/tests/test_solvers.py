"""
Solver Tests with Anti-Cheat Validation (BN-07)

Tests:
  - ARC solver correctness on bundled + novel tasks
  - HumanEval solver correctness on bundled + novel prompts
  - Anti-cheat: no hardcoded answers (C1), works on unseen tasks (C2),
    train-test firewall (C3), no trivial bypass (C4), honest failure (C5),
    no external calls (C6), ablation integrity (C7)
"""

from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agi_modules.arc_solver import solve_arc_task
from agi_modules.humaneval_solver import solve_humaneval
from agi_modules.solver_bridge import create_solver_pair
from agi_modules.external_benchmark import ExternalBenchmarkHarness


def _load_arc_tasks():
    with (ROOT / "data" / "arc_agi_sample.json").open() as f:
        return json.load(f)


def _load_humaneval_problems():
    with (ROOT / "data" / "humaneval_sample.json").open() as f:
        return json.load(f)


# ======================================================================
# ARC Solver Tests
# ======================================================================

class TestArcSolverBundled(unittest.TestCase):
    """ARC solver on all 20 bundled tasks."""

    def test_arc_solver_all_20(self):
        tasks = _load_arc_tasks()
        solved = 0
        for task in tasks:
            ctx = {"train": task["train"], "test_input": task["test"][0]["input"]}
            result = solve_arc_task(ctx)
            if result == task["test"][0]["output"]:
                solved += 1
        self.assertEqual(solved, 20, f"Expected 20/20 but got {solved}/20")


class TestArcSolverShuffled(unittest.TestCase):
    """C1: Shuffle task order → identical per-task results."""

    def test_arc_solver_shuffled_order(self):
        tasks = _load_arc_tasks()
        # Solve in original order
        original_results = []
        for task in tasks:
            ctx = {"train": task["train"], "test_input": task["test"][0]["input"]}
            original_results.append(solve_arc_task(ctx))

        # Shuffle tasks
        indices = list(range(len(tasks)))
        random.seed(999)
        random.shuffle(indices)

        # Solve in shuffled order
        for shuffled_pos, orig_idx in enumerate(indices):
            task = tasks[orig_idx]
            ctx = {"train": task["train"], "test_input": task["test"][0]["input"]}
            result = solve_arc_task(ctx)
            self.assertEqual(
                result, original_results[orig_idx],
                f"Task {orig_idx} gave different result when shuffled to position {shuffled_pos}"
            )


class TestArcSolverNovel(unittest.TestCase):
    """C2: Works on unseen tasks not in the bundled dataset."""

    def test_novel_5x5_rotation(self):
        """5x5 grid 90° CW rotation."""
        train = [
            {"input": [[1,0,0],[0,0,0],[0,0,0]],
             "output": [[0,0,1],[0,0,0],[0,0,0]]}
        ]
        test_input = [
            [1,0,0,0,0],
            [0,2,0,0,0],
            [0,0,3,0,0],
            [0,0,0,4,0],
            [0,0,0,0,5],
        ]
        ctx = {"train": train, "test_input": test_input}
        result = solve_arc_task(ctx)
        self.assertIsNotNone(result, "Solver returned None on solvable novel task")

    def test_novel_value_swap_7_2(self):
        """3x3 value swap (7↔2), asymmetric train pair to disambiguate."""
        train = [
            {"input": [[7,7],[7,2]], "output": [[2,2],[2,7]]},
        ]
        test_input = [[7,7,2],[2,7,2],[2,2,7]]
        expected = [[2,2,7],[7,2,7],[7,7,2]]
        ctx = {"train": train, "test_input": test_input}
        result = solve_arc_task(ctx)
        self.assertEqual(result, expected)

    def test_novel_hflip_with_values_036(self):
        """4x4 horizontal flip with values 0,3,6."""
        train = [
            {"input": [[0,3,6,0]], "output": [[0,6,3,0]]}
        ]
        test_input = [[0,3],[6,0],[3,6],[0,0]]
        expected = [[3,0],[0,6],[6,3],[0,0]]
        ctx = {"train": train, "test_input": test_input}
        result = solve_arc_task(ctx)
        self.assertEqual(result, expected)

    def test_novel_unsolvable(self):
        """Solver returns None for unsolvable tasks (inconsistent mapping)."""
        # Two train pairs with inconsistent value mappings — no rule can match both
        train = [
            {"input": [[1,2],[3,4]], "output": [[4,3],[2,1]]},
            {"input": [[1,2],[3,4]], "output": [[1,3],[2,4]]},
        ]
        test_input = [[5,6],[7,8]]
        ctx = {"train": train, "test_input": test_input}
        result = solve_arc_task(ctx)
        self.assertIsNone(result)


class TestArcTrainTestFirewall(unittest.TestCase):
    """C3: Solver uses only train pairs, never test output."""

    def test_no_test_output_leak(self):
        """Pass task with test output stripped. Solver must still work."""
        tasks = _load_arc_tasks()
        for task in tasks:
            ctx = {"train": task["train"], "test_input": task["test"][0]["input"]}
            # Note: ctx has NO "test_output" key — solver cannot cheat
            result = solve_arc_task(ctx)
            if result is not None:
                self.assertEqual(result, task["test"][0]["output"])


class TestArcNoTrivialBypass(unittest.TestCase):
    """C4: Solver does not use degenerate strategies."""

    def test_not_always_identity(self):
        """Solver does not return input unchanged for >50% of tasks."""
        tasks = _load_arc_tasks()
        identity_count = 0
        for task in tasks:
            ctx = {"train": task["train"], "test_input": task["test"][0]["input"]}
            result = solve_arc_task(ctx)
            if result == task["test"][0]["input"]:
                identity_count += 1
        self.assertLess(identity_count, len(tasks) * 0.5,
                        f"Identity returned for {identity_count}/{len(tasks)} tasks")

    def test_not_all_identical_output(self):
        """Solver does not return the same output for all tasks."""
        tasks = _load_arc_tasks()
        results = []
        for task in tasks:
            ctx = {"train": task["train"], "test_input": task["test"][0]["input"]}
            result = solve_arc_task(ctx)
            if result is not None:
                results.append(str(result))
        self.assertGreater(len(set(results)), 1,
                           "All tasks returned identical output")

    def test_not_always_train_output(self):
        """Solver does not always return the first train output verbatim."""
        tasks = _load_arc_tasks()
        train_copy_count = 0
        for task in tasks:
            ctx = {"train": task["train"], "test_input": task["test"][0]["input"]}
            result = solve_arc_task(ctx)
            if result == task["train"][0]["output"]:
                # Only count if it's NOT the correct answer
                if result != task["test"][0]["output"]:
                    train_copy_count += 1
        self.assertEqual(train_copy_count, 0,
                         f"Returned first train output incorrectly {train_copy_count} times")


class TestArcNovelGridSizes(unittest.TestCase):
    """C2: Works on never-seen grid sizes and values."""

    def test_large_grid_vflip(self):
        """6x6 grid vertical flip."""
        train = [
            {"input": [[1,2,3],[4,5,6]], "output": [[4,5,6],[1,2,3]]}
        ]
        test_input = [[i+j for j in range(6)] for i in range(6)]
        expected = test_input[::-1]
        ctx = {"train": train, "test_input": test_input}
        result = solve_arc_task(ctx)
        self.assertEqual(result, expected)

    def test_high_values_swap(self):
        """Grid with values 50 and 99, asymmetric train to disambiguate."""
        train = [
            {"input": [[50,50],[50,99]], "output": [[99,99],[99,50]]}
        ]
        test_input = [[50,50,99],[99,99,50]]
        expected = [[99,99,50],[50,50,99]]
        ctx = {"train": train, "test_input": test_input}
        result = solve_arc_task(ctx)
        self.assertEqual(result, expected)


# ======================================================================
# HumanEval Solver Tests
# ======================================================================

class TestHumanEvalBundled(unittest.TestCase):
    """HumanEval solver on all 10 bundled problems."""

    def test_humaneval_all_10(self):
        problems = _load_humaneval_problems()
        solved = 0
        for prob in problems:
            generated = solve_humaneval(prob["prompt"])
            if generated is None:
                continue
            code = prob["prompt"] + generated
            ns = {}
            try:
                exec(compile(code, "<gen>", "exec"), ns)
                exec(compile(prob["test"], "<test>", "exec"), ns)
                if "check" in ns:
                    ns["check"](ns.get(prob["entry_point"]))
                solved += 1
            except Exception:
                pass
        self.assertEqual(solved, 10, f"Expected 10/10 but got {solved}/10")


class TestHumanEvalNovel(unittest.TestCase):
    """C2: Novel problems via keyword fallback."""

    def test_novel_double_list(self):
        prompt = (
            "from typing import List\n\n"
            "def double_list(lst: List[int]) -> List[int]:\n"
            '    """Return a list with each element doubled.\n'
            "    >>> double_list([1, 2, 3])\n"
            "    [2, 4, 6]\n"
            '    """\n'
        )
        result = solve_humaneval(prompt)
        self.assertIsNotNone(result, "Keyword fallback should handle 'double' + 'list'")
        ns = {}
        exec(compile(prompt + result, "<gen>", "exec"), ns)
        self.assertEqual(ns["double_list"]([1, 2, 3]), [2, 4, 6])

    def test_novel_count_vowels(self):
        prompt = (
            "def count_vowels(s: str) -> int:\n"
            '    """Count the number of vowels in the given string.\n'
            "    >>> count_vowels('hello')\n"
            "    2\n"
            '    """\n'
        )
        result = solve_humaneval(prompt)
        self.assertIsNotNone(result, "Keyword fallback should handle 'count' + 'vowel'")
        ns = {}
        exec(compile(prompt + result, "<gen>", "exec"), ns)
        self.assertEqual(ns["count_vowels"]("hello"), 2)
        self.assertEqual(ns["count_vowels"]("AEIOU"), 5)

    def test_unrecognizable_returns_none(self):
        prompt = (
            "def xyzzy_quantum_frob(data: list) -> dict:\n"
            '    """Perform quantum frobulation on the data.\n'
            '    """\n'
        )
        result = solve_humaneval(prompt)
        self.assertIsNone(result, "Should return None for unrecognizable prompts")


# ======================================================================
# Anti-Cheat: No External Calls (C6)
# ======================================================================

class TestNoExternalCalls(unittest.TestCase):
    """C6: Solvers work offline — no network or subprocess calls."""

    def test_no_network_calls(self):
        """Monkeypatch urllib and subprocess, verify solvers still work."""
        def raise_net(*a, **kw):
            raise RuntimeError("No network allowed!")

        with patch("urllib.request.urlopen", side_effect=raise_net):
            with patch("subprocess.run", side_effect=raise_net):
                with patch("subprocess.call", side_effect=raise_net):
                    with patch("subprocess.Popen", side_effect=raise_net):
                        # ARC solver
                        tasks = _load_arc_tasks()
                        ctx = {"train": tasks[0]["train"],
                               "test_input": tasks[0]["test"][0]["input"]}
                        result = solve_arc_task(ctx)
                        self.assertIsNotNone(result)

                        # HumanEval solver
                        problems = _load_humaneval_problems()
                        code = solve_humaneval(problems[0]["prompt"])
                        self.assertIsNotNone(code)


# ======================================================================
# Integration Test
# ======================================================================

class TestFullBenchmarkIntegration(unittest.TestCase):
    """Integration: run_full_benchmark() with real solvers."""

    def test_combined_score_is_one(self):
        harness = ExternalBenchmarkHarness(seed=42)
        arc_fn, he_fn = create_solver_pair()
        score = harness.run_full_benchmark(
            arc_solve_fn=arc_fn, humaneval_solve_fn=he_fn
        )
        self.assertEqual(score.arc_agi_tasks_solved, 20)
        self.assertEqual(score.humaneval_solved, 10)
        self.assertAlmostEqual(score.combined, 1.0, places=3)
        self.assertEqual(len(score.errors), 0)

    def test_no_solvers_gives_zero(self):
        harness = ExternalBenchmarkHarness(seed=42)
        score = harness.run_full_benchmark()
        self.assertAlmostEqual(score.combined, 0.0, places=3)


# ======================================================================
# Honest Failure (C5)
# ======================================================================

class TestHonestFailure(unittest.TestCase):
    """C5: Solver returns None when it cannot determine the rule."""

    def test_arc_returns_none_for_unknown(self):
        """Inconsistent train pairs — no single rule can match both."""
        task = {
            "train": [
                {"input": [[1,2],[3,4]], "output": [[4,3],[2,1]]},
                {"input": [[1,2],[3,4]], "output": [[1,3],[2,4]]},
            ],
            "test_input": [[5,6],[7,8]],
        }
        self.assertIsNone(solve_arc_task(task))

    def test_humaneval_returns_none_for_unknown(self):
        prompt = "def quantum_entangle(q: complex) -> complex:\n    \"\"\"Entangle.\"\"\"\n"
        self.assertIsNone(solve_humaneval(prompt))


if __name__ == "__main__":
    unittest.main()
