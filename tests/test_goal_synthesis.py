"""Tests for :mod:`bridges.goal_synthesis_bridge` (PLAN.md Phase 3, Gap 8)."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bridges.goal_synthesis_bridge import GoalSynthesisBridge  # noqa: E402
from shared.arena_manager import ArenaManager  # noqa: E402
from shared.constants import CCE_ARENA_DIM, THDSE_ARENA_DIM  # noqa: E402
from shared.exceptions import DimensionMismatchError  # noqa: E402

_TWO_PI = 2.0 * math.pi


def _seed_goal_vec(mgr: ArenaManager, fork_name: str) -> int:
    rng = mgr.rng.fork(fork_name)
    phases = rng.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)
    return mgr.alloc_cce(phases=phases)


@pytest.fixture
def mgr() -> ArenaManager:
    return ArenaManager(master_seed=505)


@pytest.fixture
def bridge(mgr: ArenaManager) -> GoalSynthesisBridge:
    return GoalSynthesisBridge(mgr)


def test_constructor_rejects_non_manager():
    with pytest.raises(TypeError):
        GoalSynthesisBridge(arena_manager=None)  # type: ignore[arg-type]


def test_goal_to_synthesis_target_projects_and_allocates(
    bridge: GoalSynthesisBridge, mgr: ArenaManager
):
    h_goal = _seed_goal_vec(mgr, "g1")
    before = mgr.count("thdse")
    result = bridge.goal_to_synthesis_target("find red apples", h_goal, 0.8)
    after = mgr.count("thdse")
    assert after == before + 1
    assert isinstance(result["thdse_target_handle"], int)
    assert result["priority"] == 0.8


def test_goal_to_synthesis_target_self_similarity(
    bridge: GoalSynthesisBridge, mgr: ArenaManager
):
    h_goal = _seed_goal_vec(mgr, "g2")
    result = bridge.goal_to_synthesis_target("goal B", h_goal, 0.5)
    # Projection self-similarity is always exactly 1.0.
    assert result["projected_similarity"] == pytest.approx(1.0, abs=1e-6)


def test_goal_to_synthesis_target_preserves_phases(
    bridge: GoalSynthesisBridge, mgr: ArenaManager
):
    h_goal = _seed_goal_vec(mgr, "g3")
    raw_cce = mgr.get_cce_phases(h_goal)
    result = bridge.goal_to_synthesis_target("phase preservation", h_goal, 1.0)
    stored_thdse = mgr.get_thdse_phases(result["thdse_target_handle"])
    for k in range(THDSE_ARENA_DIM):
        assert stored_thdse[k] == pytest.approx(raw_cce[k * 39], abs=1e-5)


def test_goal_to_synthesis_target_result_has_provenance(
    bridge: GoalSynthesisBridge, mgr: ArenaManager
):
    h_goal = _seed_goal_vec(mgr, "g4")
    result = bridge.goal_to_synthesis_target("goal D", h_goal, 0.4)
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "goal_to_synthesis_target"
    assert prov["source_arena"] == "cce"
    assert prov["target_arena"] == "thdse"
    assert result["metadata"]["cce_dim"] == CCE_ARENA_DIM
    assert result["metadata"]["thdse_dim"] == THDSE_ARENA_DIM
    assert result["metadata"]["expected_value"] == pytest.approx(
        0.4 * 1.0, abs=1e-6
    )


def test_goal_to_synthesis_target_rejects_empty_description(
    bridge: GoalSynthesisBridge, mgr: ArenaManager
):
    h_goal = _seed_goal_vec(mgr, "g5")
    with pytest.raises(ValueError):
        bridge.goal_to_synthesis_target("  ", h_goal, 0.5)


def test_goal_to_synthesis_target_rejects_negative_priority(
    bridge: GoalSynthesisBridge, mgr: ArenaManager
):
    h_goal = _seed_goal_vec(mgr, "g6")
    with pytest.raises(ValueError):
        bridge.goal_to_synthesis_target("neg prio", h_goal, -0.1)


def test_goal_to_synthesis_target_rejects_non_numeric_priority(
    bridge: GoalSynthesisBridge, mgr: ArenaManager
):
    h_goal = _seed_goal_vec(mgr, "g7")
    with pytest.raises(TypeError):
        bridge.goal_to_synthesis_target(
            "string prio", h_goal, "high"  # type: ignore[arg-type]
        )


def test_rank_goals_orders_by_priority_times_similarity(
    bridge: GoalSynthesisBridge, mgr: ArenaManager
):
    goals = []
    for i, (desc, prio) in enumerate(
        [("low", 0.1), ("high", 0.9), ("mid", 0.5)]
    ):
        h = _seed_goal_vec(mgr, f"rank{i}")
        goals.append(bridge.goal_to_synthesis_target(desc, h, prio))
    ranked = bridge.rank_goals(goals)
    descriptions = [g["metadata"]["goal_description"] for g in ranked]
    assert descriptions == ["high", "mid", "low"]
    # Ranks are 1-indexed and attached to both the top-level item and
    # its metadata dict.
    assert ranked[0]["rank"] == 1
    assert ranked[0]["metadata"]["rank"] == 1
    assert ranked[-1]["rank"] == 3


def test_rank_goals_adds_ranking_provenance(
    bridge: GoalSynthesisBridge, mgr: ArenaManager
):
    goals = []
    for i in range(2):
        h = _seed_goal_vec(mgr, f"rprov{i}")
        goals.append(
            bridge.goal_to_synthesis_target(f"g{i}", h, 0.5 * (i + 1))
        )
    ranked = bridge.rank_goals(goals)
    for item in ranked:
        assert "ranking_provenance" in item["metadata"]
        assert item["metadata"]["ranking_provenance"]["operation"] == (
            "rank_goals"
        )
        # Original provenance is still present.
        assert "provenance" in item["metadata"]


def test_rank_goals_rejects_bad_inputs(bridge: GoalSynthesisBridge):
    with pytest.raises(TypeError):
        bridge.rank_goals("not a list")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        bridge.rank_goals([1, 2, 3])  # type: ignore[list-item]
    with pytest.raises(KeyError):
        bridge.rank_goals([{"priority": 0.5}])  # missing projected_similarity


def test_projection_count_tracks_calls(
    bridge: GoalSynthesisBridge, mgr: ArenaManager
):
    assert bridge.projection_count == 0
    for i in range(4):
        h = _seed_goal_vec(mgr, f"pc{i}")
        bridge.goal_to_synthesis_target(f"goal{i}", h, 0.1 * (i + 1))
    assert bridge.projection_count == 4
