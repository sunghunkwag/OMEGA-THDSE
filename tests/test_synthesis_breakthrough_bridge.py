"""Phase 14 — SynthesisBreakthroughBridge integration tests (D4, Rule 20)."""

from __future__ import annotations

from typing import List

import pytest

from shared.arena_manager import ArenaManager
from shared.semantic_encoder import SemanticEncoder
from shared.synthesis_engine import ProblemSpec, spec
from bridges.synthesis_breakthrough_bridge import SynthesisBreakthroughBridge


def _benchmark() -> List[ProblemSpec]:
    return [
        spec(
            "sum_list",
            [([], 0), ([1, 2, 3], 6), ([10, 20], 30), ([-1, -2], -3), ([0], 0)],
            description="sum of list",
        ),
        spec(
            "max_element",
            [([1, 2, 3], 3), ([3, 1, 2], 3), ([5], 5), ([-1, -2], -1)],
            description="max of list",
        ),
        spec(
            "reverse_list",
            [([], []), ([1, 2], [2, 1]), ([1, 2, 3], [3, 2, 1]), ([5, 6], [6, 5])],
            description="reverse list",
        ),
        spec(
            "count_occurrences",
            [([1, 2, 1], 2), ([], 0), ([1, 1, 1], 3), ([2, 3, 4], 0)],
            description="count ones",
        ),
        spec(
            "flatten_nested",
            [([[1, 2], [3]], [1, 2, 3]), ([], []), ([[1], [2]], [1, 2])],
            description="flatten one level",
        ),
    ]


@pytest.fixture()
def breakthrough():
    mgr = ArenaManager(master_seed=1414)
    enc = SemanticEncoder(prefer="hash")
    return SynthesisBreakthroughBridge(mgr, encoder=enc)


def test_d4_breakthrough_solves_four_of_five(breakthrough):
    result = breakthrough.run_benchmark(_benchmark())
    assert result["meets_target"], (
        f"D4: solved {result['solved']}/5; "
        f"per-problem={[(p['problem'], p['pass_rate']) for p in result['per_problem']]}"
    )
    assert result["solved"] >= 4


def test_benchmark_writes_memory(breakthrough):
    before = breakthrough.memory.counts()
    breakthrough.run_benchmark(_benchmark())
    after = breakthrough.memory.counts()
    # Each problem produces at least one episodic event; solved ones
    # additionally promote a semantic fact.
    assert after["episodic"] > before["episodic"]
    assert after["semantic"] > before["semantic"]


def test_benchmark_provenance(breakthrough):
    result = breakthrough.run_benchmark(_benchmark())
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "run_benchmark"
    assert prov["source_arena"] == "synthesis_engine"
    assert "timestamp" in prov


def test_decompose_with_reasoning(breakthrough):
    prob = _benchmark()[0]
    trace = breakthrough.decompose_with_reasoning(prob, max_depth=3)
    assert trace["depth"] >= 3
    assert "decomposition_provenance" in trace["metadata"]


def test_run_agent_pick_returns_trace(breakthrough):
    # Ensure benchmark ran first so the engine has cached winners,
    # then spin up the bandit agent over the problem set.
    breakthrough.run_benchmark(_benchmark())
    agent_result = breakthrough.run_agent_pick(_benchmark(), episodes=1)
    assert agent_result["num_steps"] > 0
    assert "pick_provenance" in agent_result["metadata"]


def test_rule20_breakthrough_imports_all_prior_bridges():
    import bridges.synthesis_breakthrough_bridge as sbb
    for name in (
        "SemanticConceptBridge",
        "ConceptAxiomBridge",
        "ContinuousLearningBridge",
        "MemoryArchitectureBridge",
        "ReasoningBridge",
        "AgentEnvironmentBridge",
    ):
        assert hasattr(sbb, name), f"missing import: {name}"
