"""Tests for :mod:`bridges.memory_hypothesis_bridge` (Phase 4 Gap 5)."""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bridges.memory_hypothesis_bridge import MemoryHypothesisBridge  # noqa: E402
from shared.arena_manager import ArenaManager  # noqa: E402
from shared.constants import (  # noqa: E402
    CCE_ARENA_DIM,
    MEMORY_SIMILARITY_THRESHOLD,
    MEMORY_TITLE_WEIGHT,
    THDSE_ARENA_DIM,
)
from shared.exceptions import DimensionMismatchError  # noqa: E402


@pytest.fixture
def mgr() -> ArenaManager:
    return ArenaManager(master_seed=701)


@pytest.fixture
def bridge(mgr: ArenaManager) -> MemoryHypothesisBridge:
    return MemoryHypothesisBridge(mgr)


def test_constructor_rejects_non_arena_manager():
    with pytest.raises(TypeError):
        MemoryHypothesisBridge(arena_manager=42)  # type: ignore[arg-type]


def test_encode_memory_returns_256_dim_vector(bridge: MemoryHypothesisBridge):
    result = bridge.encode_memory_for_hypothesis(
        "machine learning research", ["ml", "research"]
    )
    vec = result["thdse_vector"]
    assert vec.shape == (THDSE_ARENA_DIM,)
    assert result["original_dim"] == CCE_ARENA_DIM
    assert result["projected_dim"] == THDSE_ARENA_DIM
    assert result["tag_count"] == 2


def test_encode_memory_metadata_has_provenance(bridge: MemoryHypothesisBridge):
    result = bridge.encode_memory_for_hypothesis("hello world", ["greeting"])
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "memory_to_hypothesis"
    assert prov["source_arena"] == "cce"
    assert prov["target_arena"] == "thdse"
    assert isinstance(prov["timestamp"], float)
    assert result["metadata"]["title_weight"] == MEMORY_TITLE_WEIGHT


def test_encode_memory_is_deterministic_for_same_input(
    bridge: MemoryHypothesisBridge,
):
    a = bridge.encode_memory_for_hypothesis("apple", ["fruit"])
    b = bridge.encode_memory_for_hypothesis("apple", ["fruit"])
    # Token cache should make repeated encodings produce identical phases.
    for i in range(THDSE_ARENA_DIM):
        assert a["thdse_vector"][i] == pytest.approx(
            b["thdse_vector"][i], abs=1e-6
        )


def test_encode_memory_rejects_empty_title(bridge: MemoryHypothesisBridge):
    with pytest.raises(ValueError):
        bridge.encode_memory_for_hypothesis("   ", ["x"])


def test_encode_memory_rejects_non_list_tags(bridge: MemoryHypothesisBridge):
    with pytest.raises(TypeError):
        bridge.encode_memory_for_hypothesis(
            "ok title", "not_a_list"  # type: ignore[arg-type]
        )


def test_score_relevance_combines_components(bridge: MemoryHypothesisBridge):
    result = bridge.score_hypothesis_relevance(0.8, 0.5)
    expected = 0.6 * 0.8 + 0.4 * 0.5
    assert result["relevance_score"] == pytest.approx(expected, abs=1e-6)
    assert result["components"]["fitness"] == 0.8
    assert result["components"]["memory_similarity"] == 0.5


def test_score_relevance_gates_low_memory_similarity(
    bridge: MemoryHypothesisBridge,
):
    # Below threshold → memory contribution drops to 0.
    sim_below = MEMORY_SIMILARITY_THRESHOLD - 0.05
    result = bridge.score_hypothesis_relevance(0.5, sim_below)
    assert result["components"]["gated_memory_similarity"] == 0.0
    assert result["relevance_score"] == pytest.approx(0.6 * 0.5, abs=1e-6)


def test_score_relevance_metadata_has_provenance(
    bridge: MemoryHypothesisBridge,
):
    result = bridge.score_hypothesis_relevance(0.5, 0.5)
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "hypothesis_scoring"
    assert prov["source_arena"] == "both"
    assert isinstance(prov["timestamp"], float)


def test_compare_memory_to_hypothesis_returns_similarity_in_range(
    bridge: MemoryHypothesisBridge,
):
    encoded = bridge.encode_memory_for_hypothesis("topic", ["t"])
    sim_self = bridge.compare_memory_to_hypothesis(
        encoded, list(encoded["thdse_vector"])
    )
    assert sim_self["memory_to_hypothesis_similarity"] == pytest.approx(
        1.0, abs=1e-5
    )
    assert sim_self["metadata"]["provenance"]["operation"] == (
        "compare_memory_to_hypothesis"
    )


def test_compare_memory_rejects_wrong_dim(bridge: MemoryHypothesisBridge):
    encoded = bridge.encode_memory_for_hypothesis("topic", ["t"])
    with pytest.raises(DimensionMismatchError):
        bridge.compare_memory_to_hypothesis(encoded, [0.0] * 100)


def test_rank_hypotheses_orders_by_relevance(bridge: MemoryHypothesisBridge):
    inputs = [
        {"id": "low", "hypothesis_fitness": 0.1, "memory_similarity": 0.9},
        {"id": "high", "hypothesis_fitness": 0.9, "memory_similarity": 0.9},
        {"id": "mid", "hypothesis_fitness": 0.5, "memory_similarity": 0.6},
    ]
    ranked = bridge.rank_hypotheses_by_memory(inputs)
    assert [r["id"] for r in ranked] == ["high", "mid", "low"]
    assert ranked[0]["rank"] == 1
    assert "ranking_provenance" in ranked[0]["metadata"]


def test_reset_token_cache_returns_freed_count(
    bridge: MemoryHypothesisBridge,
):
    bridge.encode_memory_for_hypothesis("apple", ["fruit", "color"])
    cached_before = bridge.cached_token_count
    assert cached_before > 0
    result = bridge.reset_token_cache()
    assert result["freed"] == cached_before
    assert bridge.cached_token_count == 0
    assert result["metadata"]["provenance"]["operation"] == "reset_token_cache"


def test_encoded_count_tracks_calls(bridge: MemoryHypothesisBridge):
    assert bridge.encoded_count == 0
    for i in range(4):
        bridge.encode_memory_for_hypothesis(f"topic_{i}", [f"tag_{i}"])
    assert bridge.encoded_count == 4


def test_independent_managers_yield_independent_state():
    mgr_a = ArenaManager(master_seed=1)
    mgr_b = ArenaManager(master_seed=1)
    a = MemoryHypothesisBridge(mgr_a)
    b = MemoryHypothesisBridge(mgr_b)
    a.encode_memory_for_hypothesis("alpha", ["x"])
    assert a.encoded_count == 1
    assert b.encoded_count == 0
