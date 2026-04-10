"""Tests for :mod:`bridges.world_model_swarm_bridge` (Phase 4 Gap 7)."""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bridges.world_model_swarm_bridge import WorldModelSwarmBridge  # noqa: E402
from shared.arena_manager import ArenaManager  # noqa: E402
from shared.constants import (  # noqa: E402
    SWARM_CONSENSUS_THRESHOLD,
    SWARM_NUM_AGENTS,
    THDSE_ARENA_DIM,
)
from shared.exceptions import DimensionMismatchError  # noqa: E402

_TWO_PI = 2.0 * math.pi


@pytest.fixture
def mgr() -> ArenaManager:
    return ArenaManager(master_seed=801)


@pytest.fixture
def bridge(mgr: ArenaManager) -> WorldModelSwarmBridge:
    return WorldModelSwarmBridge(mgr)


def test_constructor_rejects_non_arena_manager():
    with pytest.raises(TypeError):
        WorldModelSwarmBridge(arena_manager=None)  # type: ignore[arg-type]


def test_project_world_state_returns_256_vector(
    bridge: WorldModelSwarmBridge,
):
    result = bridge.project_world_state_for_swarm(
        {"task": "x", "step": 5}, {"a": 0.8, "b": 0.1, "c": 0.05}
    )
    vec = result["thdse_guidance_vector"]
    assert vec.shape == (THDSE_ARENA_DIM,)
    assert 0.0 <= result["confidence"] <= 1.0


def test_project_world_state_metadata_has_provenance(
    bridge: WorldModelSwarmBridge,
):
    result = bridge.project_world_state_for_swarm(
        {"task": "t"}, {"a": 1.0}
    )
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "world_to_swarm"
    assert prov["source_arena"] == "cce"
    assert prov["target_arena"] == "thdse"
    assert result["metadata"]["swarm_num_agents"] == SWARM_NUM_AGENTS


def test_project_world_state_is_deterministic(
    bridge: WorldModelSwarmBridge,
):
    r1 = bridge.project_world_state_for_swarm(
        {"k": "v"}, {"act": 0.5}
    )
    r2 = bridge.project_world_state_for_swarm(
        {"k": "v"}, {"act": 0.5}
    )
    v1 = r1["thdse_guidance_vector"]
    v2 = r2["thdse_guidance_vector"]
    for i in range(THDSE_ARENA_DIM):
        assert v1[i] == pytest.approx(v2[i], abs=1e-6)


def test_project_world_state_rejects_non_dict(
    bridge: WorldModelSwarmBridge,
):
    with pytest.raises(TypeError):
        bridge.project_world_state_for_swarm(
            "not a dict", {"a": 0.1}  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        bridge.project_world_state_for_swarm(
            {"k": "v"}, "not a dict"  # type: ignore[arg-type]
        )


def test_incorporate_swarm_consensus_returns_bool_should_adopt(
    bridge: WorldModelSwarmBridge,
):
    # Zero-phase vector vs zero-CCE reference → similarity = 1.0 >
    # SWARM_CONSENSUS_THRESHOLD → adoption signaled.
    consensus = [0.0] * THDSE_ARENA_DIM
    result = bridge.incorporate_swarm_consensus(consensus)
    assert result["similarity"] == pytest.approx(1.0, abs=1e-5)
    assert result["should_adopt"] is True
    assert result["threshold"] == SWARM_CONSENSUS_THRESHOLD


def test_incorporate_swarm_consensus_rejects_wrong_dim(
    bridge: WorldModelSwarmBridge,
):
    with pytest.raises(DimensionMismatchError):
        bridge.incorporate_swarm_consensus([0.0] * 10)


def test_incorporate_swarm_consensus_metadata_has_provenance(
    bridge: WorldModelSwarmBridge,
):
    result = bridge.incorporate_swarm_consensus([0.0] * THDSE_ARENA_DIM)
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "swarm_to_world"
    assert prov["source_arena"] == "thdse"
    assert prov["target_arena"] == "cce"


def test_incorporate_swarm_consensus_custom_threshold(
    bridge: WorldModelSwarmBridge,
):
    # Pi-phase vs zero-CCE → cos(pi - 0) = -1 on every coord → similarity ≈ -1.
    pi_vec = [math.pi] * THDSE_ARENA_DIM
    result = bridge.incorporate_swarm_consensus(pi_vec, threshold=-1.1)
    assert result["should_adopt"] is True
    result2 = bridge.incorporate_swarm_consensus(pi_vec, threshold=0.0)
    assert result2["should_adopt"] is False


def test_compare_two_swarm_consensuses_self_similarity(
    bridge: WorldModelSwarmBridge,
):
    vec = [0.5] * THDSE_ARENA_DIM
    sim = bridge.compare_two_swarm_consensuses(vec, vec)
    assert sim["similarity"] == pytest.approx(1.0, abs=1e-6)


def test_compare_two_swarm_consensuses_antiparallel(
    bridge: WorldModelSwarmBridge,
):
    a = [0.0] * THDSE_ARENA_DIM
    b = [math.pi] * THDSE_ARENA_DIM
    sim = bridge.compare_two_swarm_consensuses(a, b)
    assert sim["similarity"] == pytest.approx(-1.0, abs=1e-5)


def test_summarize_swarm_state_after_activity(
    bridge: WorldModelSwarmBridge,
):
    bridge.project_world_state_for_swarm({"k": "a"}, {"x": 0.3})
    bridge.project_world_state_for_swarm({"k": "b"}, {"x": 0.6})
    bridge.incorporate_swarm_consensus([0.0] * THDSE_ARENA_DIM)
    summary = bridge.summarize_swarm_state()
    assert summary["projection_count"] == 2
    assert summary["consensus_count"] == 1
    assert summary["adopted_count"] == 1
    assert summary["adoption_rate"] == pytest.approx(1.0, abs=1e-6)


def test_project_action_distribution_encodes_size_256(
    bridge: WorldModelSwarmBridge,
):
    result = bridge.project_action_distribution(
        {"act_a": 0.8, "act_b": 0.1, "act_c": 0.05}
    )
    vec = result["thdse_action_vector"]
    assert vec.shape == (THDSE_ARENA_DIM,)
    # Softmax normalization → sums to 1 exactly.
    assert sum(result["action_distribution"].values()) == pytest.approx(
        1.0, abs=1e-6
    )


def test_project_action_distribution_rejects_empty(
    bridge: WorldModelSwarmBridge,
):
    with pytest.raises(ValueError):
        bridge.project_action_distribution({})
