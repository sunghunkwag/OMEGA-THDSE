"""Phase 12 — ChainOfThoughtReasoner + AnalogyEngine (Rule 17, 19, D5, D7)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytest

from shared.reasoning_engine import (
    AnalogyEngine,
    ChainOfThoughtReasoner,
    ReasoningStep,
    verify_chain_linkage,
)
from shared.semantic_encoder import SemanticEncoder


# --------------------------------------------------------------------------- #
# Rule 17 — reasoning depth >= 3, each step feeds the next
# --------------------------------------------------------------------------- #


def _arithmetic_operators():
    """Toy operators for a numeric reasoning task (target = 100)."""

    def add_one(x: int) -> Sequence[Tuple[int, float, Dict[str, Any]]]:
        y = x + 1
        return [(y, 0.5, {"op": "+1"})]

    def times_two(x: int) -> Sequence[Tuple[int, float, Dict[str, Any]]]:
        y = x * 2
        return [(y, 0.8, {"op": "*2"})]

    def square(x: int) -> Sequence[Tuple[int, float, Dict[str, Any]]]:
        y = x * x
        return [(y, 0.4, {"op": "**2"})]

    return {"add_one": add_one, "times_two": times_two, "square": square}


def _goal_fn_target(target: int):
    def _fn(x: int) -> float:
        # Higher is better; peak at 1.0 when x == target.
        if x == target:
            return 1.0
        return 1.0 / (1.0 + abs(target - x))
    return _fn


def test_rule17_chain_has_depth_at_least_3():
    reasoner = ChainOfThoughtReasoner(
        operators=_arithmetic_operators(),
        goal_fn=_goal_fn_target(100),
        max_depth=5,
    )
    result = reasoner.run(initial_premise=3, goal_threshold=2.0)  # never reached
    assert result["depth"] >= 3, f"depth={result['depth']}"


def test_rule17_step_premise_equals_previous_conclusion():
    reasoner = ChainOfThoughtReasoner(
        operators=_arithmetic_operators(),
        goal_fn=_goal_fn_target(50),
        max_depth=5,
    )
    result = reasoner.run(initial_premise=2, goal_threshold=2.0)
    steps: List[ReasoningStep] = result["steps"]
    assert len(steps) >= 3
    for a, b in zip(steps[:-1], steps[1:]):
        assert a.conclusion == b.premise, (
            f"linkage broken at index {a.index}→{b.index}: "
            f"a.conclusion={a.conclusion!r} != b.premise={b.premise!r}"
        )
    assert verify_chain_linkage(steps)


def test_rule17_rejects_depth_below_3():
    with pytest.raises(ValueError):
        ChainOfThoughtReasoner(
            operators=_arithmetic_operators(),
            goal_fn=_goal_fn_target(10),
            max_depth=2,
        )


def test_rule17_respects_max_depth_cap():
    with pytest.raises(ValueError):
        ChainOfThoughtReasoner(
            operators=_arithmetic_operators(),
            goal_fn=_goal_fn_target(10),
            max_depth=999,
        )


def test_reasoner_reaches_exact_goal_when_possible():
    reasoner = ChainOfThoughtReasoner(
        operators=_arithmetic_operators(),
        goal_fn=_goal_fn_target(8),
        max_depth=5,
    )
    result = reasoner.run(initial_premise=2, goal_threshold=1.0)
    # 2 → (times_two=4) → (times_two=8) path should reach the target.
    assert result["reached_goal"]
    assert result["final_premise"] == 8


def test_backtracks_on_plateau():
    # Operators that loop forever without making progress to 1000.
    def noop(x):
        return [(x, 0.1, {"op": "noop"})]

    reasoner = ChainOfThoughtReasoner(
        operators={"noop": noop},
        goal_fn=lambda x: 0.0,
        max_depth=8,
        patience=1,
    )
    result = reasoner.run(0, goal_threshold=1.0)
    # After patience hits, the reasoner stops before max_depth.
    assert result["depth"] < 8


# --------------------------------------------------------------------------- #
# Rule 19 — analogy transfer raises score on unseen domain
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def encoder():
    return SemanticEncoder(prefer="hash")


def test_rule19_transfer_improves_target_score(encoder):
    eng = AnalogyEngine(encoder)
    # Source: mammal examples; Target: mostly-mammal examples with some
    # noise. Pattern from source should lift the target score relative
    # to the target's own noisy centroid.
    source = ["dog", "puppy", "canine", "cat", "feline", "kitten"]
    target = ["dog runs fast", "cat climbs trees", "puppy barks"]
    result = eng.transfer_score(
        source_examples=source,
        target_examples=target,
        pattern_name="mammals",
    )
    # The PLAN Rule 19 test says: score_with_transfer > score_without_transfer.
    # We test on synonyms so transfer should reliably help.
    assert result["with_transfer"] > result["without_transfer"], (
        f"transfer did not improve score: "
        f"without={result['without_transfer']:.3f}, "
        f"with={result['with_transfer']:.3f}"
    )


def test_analogy_extract_and_match(encoder):
    eng = AnalogyEngine(encoder)
    eng.extract_pattern("vehicles", ["car", "automobile", "truck"])
    eng.extract_pattern("animals", ["dog", "cat", "bird"])
    scores = eng.match("motorcycle")
    # The vehicles pattern should rank above animals.
    names = [name for name, _ in scores]
    assert names[0] == "vehicles"


def test_analogy_empty_examples_raises(encoder):
    eng = AnalogyEngine(encoder)
    with pytest.raises(ValueError):
        eng.extract_pattern("empty", [])


def test_analogy_transfer_requires_two_target_examples(encoder):
    eng = AnalogyEngine(encoder)
    with pytest.raises(ValueError):
        eng.transfer_score(source_examples=["x"], target_examples=["y"])
