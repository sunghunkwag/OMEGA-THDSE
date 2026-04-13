"""Phase 14 — SynthesisEngine tests (Rule 15, D4)."""

from __future__ import annotations

import pytest

from shared.synthesis_engine import (
    ProblemSpec,
    SynthesisEngine,
    Template,
    spec,
)


# --------------------------------------------------------------------------- #
# Benchmark problem fixtures — five Phase 8C.1 problems
# --------------------------------------------------------------------------- #


def _sum_list_spec() -> ProblemSpec:
    return spec(
        "sum_list",
        [
            ([], 0),
            ([1], 1),
            ([1, 2, 3], 6),
            ([1, 2, 3, 4, 5], 15),
            ([10, 20, 30], 60),
            ([0, 0, 0], 0),
            ([-1, -2, -3], -6),
            ([7, 8, 9], 24),
        ],
        description="Return the sum of all elements of a list.",
    )


def _max_element_spec() -> ProblemSpec:
    return spec(
        "max_element",
        [
            ([1], 1),
            ([1, 2], 2),
            ([3, 1, 2], 3),
            ([-1, -2], -1),
            ([10, 20, 5, 15], 20),
            ([0, 0, 0], 0),
            ([100, -5, 7], 100),
        ],
        description="Return the maximum element of a non-empty list.",
    )


def _reverse_list_spec() -> ProblemSpec:
    return spec(
        "reverse_list",
        [
            ([], []),
            ([1], [1]),
            ([1, 2], [2, 1]),
            ([1, 2, 3], [3, 2, 1]),
            ([1, 2, 3, 4], [4, 3, 2, 1]),
            ([10, 20], [20, 10]),
            (["a", "b", "c"], ["c", "b", "a"]),
        ],
        description="Return the list in reverse order.",
    )


def _count_occurrences_spec() -> ProblemSpec:
    return spec(
        "count_occurrences",
        [
            ([], 0),
            ([1], 1),
            ([2], 0),
            ([1, 2, 1, 3, 1], 3),
            ([2, 2, 2], 0),
            ([1, 1, 1, 1], 4),
            ([1, 2, 3, 4, 1], 2),
        ],
        description="Count how many times the value 1 appears in the list.",
    )


def _flatten_nested_spec() -> ProblemSpec:
    return spec(
        "flatten_nested",
        [
            ([], []),
            ([[]], []),
            ([[1]], [1]),
            ([[1], [2]], [1, 2]),
            ([[1, 2], [3, 4]], [1, 2, 3, 4]),
            ([[7], [8], [9]], [7, 8, 9]),
            ([[1, 2, 3], [4, 5]], [1, 2, 3, 4, 5]),
        ],
        description="Flatten a list of lists one level deep.",
    )


def _benchmark_problems():
    return [
        _sum_list_spec(),
        _max_element_spec(),
        _reverse_list_spec(),
        _count_occurrences_spec(),
        _flatten_nested_spec(),
    ]


# --------------------------------------------------------------------------- #
# Rule 15 — real execution, no hardcoded scores
# --------------------------------------------------------------------------- #


def test_rule15_wrong_solution_scores_lower_than_correct():
    """Rule 15 (NO HALLUCINATED METRICS): wrong < correct."""
    engine = SynthesisEngine()
    wrong = Template(
        name="always_zero",
        source="lambda arr: 0",
        func=lambda arr: 0,
    )
    engine.register_template(wrong)

    problem = _sum_list_spec()
    # Score each candidate by actually running it.
    wrong_score = engine._score_candidate(wrong, problem).pass_rate
    correct = [t for t in engine.templates if t.name == "sum_list"][0]
    correct_score = engine._score_candidate(correct, problem).pass_rate
    assert correct_score > wrong_score, (
        f"Rule 15: correct={correct_score:.2f} did not exceed wrong={wrong_score:.2f}"
    )
    assert correct_score == 1.0
    assert wrong_score < 1.0


def test_rule15_benchmark_scores_come_from_execution():
    engine = SynthesisEngine()
    problems = _benchmark_problems()
    result = engine.benchmark(problems)
    # Each per-problem record must expose the passed count + total.
    for entry in result["per_problem"]:
        assert entry["total"] == len(
            dict((p.name, p) for p in problems)[entry["problem"]].examples
        )
        assert 0.0 <= entry["pass_rate"] <= 1.0


# --------------------------------------------------------------------------- #
# D4 — the engine solves >= 4/5 benchmark problems
# --------------------------------------------------------------------------- #


def test_d4_engine_solves_at_least_four_of_five():
    engine = SynthesisEngine()
    problems = _benchmark_problems()
    result = engine.benchmark(problems)
    assert result["solved"] >= 4, (
        f"D4 breakthrough: solved only {result['solved']}/5 "
        f"(per_problem={[(p['problem'], p['pass_rate']) for p in result['per_problem']]})"
    )
    assert result["meets_target"] is True


def test_solve_caches_winning_template():
    engine = SynthesisEngine()
    p = _sum_list_spec()
    r1 = engine.solve(p)
    assert r1["pass_rate"] == 1.0
    assert engine.solution_cache[p.name] == r1["winner"]
    r2 = engine.solve(p)
    assert r2["from_cache"]
    assert r2["winner"] == r1["winner"]


def test_empty_examples_rejected():
    with pytest.raises(ValueError):
        ProblemSpec(name="empty", examples=())


def test_decomposition_with_raw_reasoner_returns_depth_chain():
    engine = SynthesisEngine()
    p = _sum_list_spec()
    trace = engine.decompose_with_reasoner(p, reasoner=None, max_depth=3)  # type: ignore[arg-type]
    # With reasoner=None we fall through to the raw ChainOfThoughtReasoner.
    assert trace["depth"] >= 3
