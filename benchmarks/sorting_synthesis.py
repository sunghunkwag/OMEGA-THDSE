"""Phase 8C.1 — Benchmark suite for goal-directed synthesis.

Five problems, each defined by 20+ deterministic input/output examples.
Each problem has a single unambiguous oracle: ``func(input) ==
expected``. PLAN.md Rule 19 forbids "generate any valid Python" type
problems — every spec here has concrete I/O.

The seed corpus is a small library of canonical Python implementations
covering arithmetic, iteration, accumulation, and list manipulation.
The benchmark scripts ingest this corpus into the THDSE synthesizer
and try to recombine its sub-tree atoms into solutions.
"""

from __future__ import annotations

from typing import Dict, List

from src.synthesis.problem_spec import ProblemSpec


# --------------------------------------------------------------------------- #
# Problem definitions
# --------------------------------------------------------------------------- #


def _make_sum_list() -> ProblemSpec:
    examples = [
        ([], 0),
        ([1], 1),
        ([1, 2], 3),
        ([1, 2, 3], 6),
        ([1, 2, 3, 4], 10),
        ([1, 2, 3, 4, 5], 15),
        ([0, 0, 0], 0),
        ([-1, -2, -3], -6),
        ([10, 20, 30], 60),
        ([100], 100),
        ([7, 8, 9], 24),
        ([-5, 5], 0),
        ([1, -1, 2, -2], 0),
        ([42, 0, 1], 43),
        ([2, 4, 6, 8], 20),
        ([1, 2, 3, 4, 5, 6, 7], 28),
        ([-100, 50, 25], -25),
        ([3, 3, 3, 3], 12),
        ([99, 1], 100),
        ([0, 1, -1, 2, -2], 0),
    ]
    return ProblemSpec(
        name="sum_list",
        io_examples=examples,
        description="Return the sum of all elements of a list.",
    )


def _make_max_element() -> ProblemSpec:
    examples = [
        ([1], 1),
        ([1, 2], 2),
        ([2, 1], 2),
        ([1, 2, 3], 3),
        ([3, 2, 1], 3),
        ([5, 5, 5], 5),
        ([-1, -2, -3], -1),
        ([0], 0),
        ([10, 100, 50], 100),
        ([-10, -20, -5], -5),
        ([7, 8, 9, 6, 5], 9),
        ([42, 41, 43], 43),
        ([1, 1, 1, 2], 2),
        ([100, 99, 98], 100),
        ([0, 1, -1], 1),
        ([3, 7, 2, 8, 1], 8),
        ([4, 4, 4, 5, 4], 5),
        ([-100, 0, 100], 100),
        ([6, 5, 4, 3, 2, 1], 6),
        ([1, 2, 3, 4, 5, 4, 3, 2, 1], 5),
    ]
    return ProblemSpec(
        name="max_element",
        io_examples=examples,
        description="Return the maximum element of a non-empty list.",
    )


def _make_reverse_list() -> ProblemSpec:
    examples = [
        ([], []),
        ([1], [1]),
        ([1, 2], [2, 1]),
        ([1, 2, 3], [3, 2, 1]),
        ([1, 2, 3, 4], [4, 3, 2, 1]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
        ([0], [0]),
        ([7, 7, 7], [7, 7, 7]),
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),
        ([10, 20], [20, 10]),
        ([-1, -2, -3], [-3, -2, -1]),
        ([100, 200, 300, 400], [400, 300, 200, 100]),
        ([42], [42]),
        ([1, 1, 2, 2], [2, 2, 1, 1]),
        ([0, 1, 0, 1], [1, 0, 1, 0]),
        ([8, 6, 4, 2], [2, 4, 6, 8]),
        ([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]),
        ([9, 8, 7], [7, 8, 9]),
        ([-5, 5, -5, 5], [5, -5, 5, -5]),
        ([2, 4, 6, 8, 10], [10, 8, 6, 4, 2]),
    ]
    return ProblemSpec(
        name="reverse_list",
        io_examples=examples,
        description="Return the input list in reverse order.",
    )


def _make_count_occurrences() -> ProblemSpec:
    """Count occurrences of value 1 in a list (single-arg form)."""
    examples = [
        ([], 0),
        ([1], 1),
        ([2], 0),
        ([1, 1], 2),
        ([1, 2, 1], 2),
        ([1, 2, 3], 1),
        ([0, 0, 0], 0),
        ([1, 1, 1, 1], 4),
        ([2, 2, 2], 0),
        ([1, 0, 1, 0, 1], 3),
        ([5, 1, 5, 1, 5], 2),
        ([1, 2, 1, 2, 1, 2, 1], 4),
        ([3, 3, 3, 3, 3], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 2, 3, 4, 5], 1),
        ([1, 1, 0, 0, 1], 3),
        ([10, 20, 30, 1, 1], 2),
        ([1, 1, 1, 2, 2, 2], 3),
        ([0, 0, 1, 0, 0, 1, 0, 0, 1], 3),
    ]
    return ProblemSpec(
        name="count_occurrences",
        io_examples=examples,
        description="Count how many times the value 1 appears in a list.",
    )


def _make_flatten_nested() -> ProblemSpec:
    """Flatten a list of lists by one level (no recursion required)."""
    examples = [
        ([], []),
        ([[]], []),
        ([[1]], [1]),
        ([[1, 2]], [1, 2]),
        ([[1], [2]], [1, 2]),
        ([[1, 2], [3, 4]], [1, 2, 3, 4]),
        ([[1, 2, 3], [4, 5]], [1, 2, 3, 4, 5]),
        ([[], [1], [2]], [1, 2]),
        ([[7], [8], [9]], [7, 8, 9]),
        ([[1, 1], [2, 2], [3, 3]], [1, 1, 2, 2, 3, 3]),
        ([[0, 0]], [0, 0]),
        ([[10, 20, 30]], [10, 20, 30]),
        ([[1], [], [3]], [1, 3]),
        ([[5, 6], [7, 8], [9, 10]], [5, 6, 7, 8, 9, 10]),
        ([[1, 2, 3, 4]], [1, 2, 3, 4]),
        ([[1], [2], [3], [4]], [1, 2, 3, 4]),
        ([[100, 200], [300]], [100, 200, 300]),
        ([[-1, -2], [-3, -4]], [-1, -2, -3, -4]),
        ([[42, 43], [44, 45], [46, 47]], [42, 43, 44, 45, 46, 47]),
        ([[1, 0], [0, 1], [1, 0]], [1, 0, 0, 1, 1, 0]),
    ]
    return ProblemSpec(
        name="flatten_nested",
        io_examples=examples,
        description="Flatten a list of lists into a single list (one level).",
    )


def all_problems() -> List[ProblemSpec]:
    return [
        _make_sum_list(),
        _make_max_element(),
        _make_reverse_list(),
        _make_count_occurrences(),
        _make_flatten_nested(),
    ]


# --------------------------------------------------------------------------- #
# Seed corpus — canonical building blocks for synthesis to recombine
# --------------------------------------------------------------------------- #


SEED_CORPUS: Dict[str, str] = {
    "iter_sum": (
        "def iter_sum(arr):\n"
        "    total = 0\n"
        "    for x in arr:\n"
        "        total = total + x\n"
        "    return total\n"
    ),
    "builtin_sum": (
        "def builtin_sum(arr):\n"
        "    return sum(arr)\n"
    ),
    "iter_max": (
        "def iter_max(arr):\n"
        "    best = arr[0]\n"
        "    for x in arr:\n"
        "        if x > best:\n"
        "            best = x\n"
        "    return best\n"
    ),
    "builtin_max": (
        "def builtin_max(arr):\n"
        "    return max(arr)\n"
    ),
    "iter_reverse": (
        "def iter_reverse(arr):\n"
        "    out = []\n"
        "    for x in arr:\n"
        "        out = [x] + out\n"
        "    return out\n"
    ),
    "slice_reverse": (
        "def slice_reverse(arr):\n"
        "    return arr[::-1]\n"
    ),
    "iter_count": (
        "def iter_count(arr):\n"
        "    n = 0\n"
        "    for x in arr:\n"
        "        if x == 1:\n"
        "            n = n + 1\n"
        "    return n\n"
    ),
    "method_count": (
        "def method_count(arr):\n"
        "    return arr.count(1)\n"
    ),
    "iter_flatten": (
        "def iter_flatten(arr):\n"
        "    out = []\n"
        "    for sub in arr:\n"
        "        for x in sub:\n"
        "            out = out + [x]\n"
        "    return out\n"
    ),
    "list_concat_flatten": (
        "def list_concat_flatten(arr):\n"
        "    out = []\n"
        "    for sub in arr:\n"
        "        out = out + sub\n"
        "    return out\n"
    ),
    "len_helper": (
        "def len_helper(arr):\n"
        "    return len(arr)\n"
    ),
    "first_helper": (
        "def first_helper(arr):\n"
        "    return arr[0]\n"
    ),
    "last_helper": (
        "def last_helper(arr):\n"
        "    return arr[-1]\n"
    ),
    "double_helper": (
        "def double_helper(x):\n"
        "    return x + x\n"
    ),
    "increment_helper": (
        "def increment_helper(x):\n"
        "    return x + 1\n"
    ),
    "is_zero_helper": (
        "def is_zero_helper(x):\n"
        "    return x == 0\n"
    ),
    "abs_helper": (
        "def abs_helper(x):\n"
        "    return abs(x)\n"
    ),
    "negate_helper": (
        "def negate_helper(x):\n"
        "    return -x\n"
    ),
    "min_helper": (
        "def min_helper(arr):\n"
        "    return min(arr)\n"
    ),
    "sorted_helper": (
        "def sorted_helper(arr):\n"
        "    return sorted(arr)\n"
    ),
    "builtin_flatten": (
        "def builtin_flatten(arr):\n"
        "    return sum(arr, [])\n"
    ),
    "builtin_reverse": (
        "def builtin_reverse(arr):\n"
        "    return arr[::-1]\n"
    ),
}


__all__ = ["SEED_CORPUS", "all_problems"]
