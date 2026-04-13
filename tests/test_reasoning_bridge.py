"""Phase 12 — ReasoningBridge integration tests (Rule 17, 19, 20)."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import pytest

from shared.arena_manager import ArenaManager
from shared.semantic_encoder import SemanticEncoder
from bridges.reasoning_bridge import ReasoningBridge


@pytest.fixture()
def bridge():
    mgr = ArenaManager(master_seed=1212)
    enc = SemanticEncoder(prefer="hash")
    return ReasoningBridge(mgr, encoder=enc)


def test_reason_runs_depth3_chain_via_bridge(bridge):
    def plus_one(x: int) -> Sequence[Tuple[int, float, Dict[str, Any]]]:
        return [(x + 1, 0.3, {})]

    def times_two(x: int) -> Sequence[Tuple[int, float, Dict[str, Any]]]:
        return [(x * 2, 0.8, {})]

    bridge.register_operator("plus_one", plus_one)
    bridge.register_operator("times_two", times_two)

    def goal_fn(x: int) -> float:
        return 1.0 / (1.0 + abs(16 - x))

    result = bridge.reason(1, goal_fn=goal_fn, max_depth=5, goal_threshold=1.0)
    assert result["depth"] >= 3
    assert result["linkage_ok"]
    # Rule 17 adjacency check via serialised steps.
    steps = result["steps"]
    for a, b in zip(steps[:-1], steps[1:]):
        assert a["conclusion"] == b["premise"]
    assert result["metadata"]["provenance"]["operation"] == "reason"


def test_reasoning_without_operators_raises(bridge):
    with pytest.raises(RuntimeError):
        bridge.reason(1, goal_fn=lambda x: 0.0)


def test_extract_and_transfer_improves_score(bridge):
    result = bridge.extract_and_transfer(
        source_examples=["dog", "puppy", "canine"],
        target_examples=["dog runs", "puppy plays", "canine barks"],
    )
    assert result["improved"] is True
    assert result["with_transfer"] > result["without_transfer"]
    assert result["metadata"]["provenance"]["operation"] == "extract_and_transfer"


def test_reason_with_memory_uses_stored_facts(bridge):
    # Seed memory with a fact that matches the question.
    bridge._memory_bridge.assert_fact(
        "light speed", "light travels at about 300000 kilometres per second"
    )
    result = bridge.reason_with_memory("light speed", max_depth=3)
    assert result["depth"] >= 3
    assert result["memory_seed"]["found"]
    assert result["memory_seed"]["tier"] == "semantic"


def test_rule20_bridge_imports_phase9_and_phase11():
    import bridges.reasoning_bridge as rb
    assert hasattr(rb, "SemanticConceptBridge")
    assert hasattr(rb, "MemoryArchitectureBridge")
