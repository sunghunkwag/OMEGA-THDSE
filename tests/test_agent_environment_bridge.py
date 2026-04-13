"""Phase 13 — AgentEnvironmentBridge integration tests (D8, Rule 16, 20)."""

from __future__ import annotations

import pytest

from shared.arena_manager import ArenaManager
from shared.environment import BanditEnvironment, GridWorldEnvironment
from shared.semantic_encoder import SemanticEncoder
from bridges.agent_environment_bridge import AgentEnvironmentBridge


@pytest.fixture()
def bandit_bridge():
    mgr = ArenaManager(master_seed=1313)
    env = BanditEnvironment(k=4, rng=mgr.rng)
    enc = SemanticEncoder(prefer="hash")
    return AgentEnvironmentBridge(mgr, env, encoder=enc, max_steps=60)


@pytest.fixture()
def grid_bridge():
    mgr = ArenaManager(master_seed=1414)
    env = GridWorldEnvironment([[".", ".", "."], [".", ".", "."], [".", ".", "."]])
    enc = SemanticEncoder(prefer="hash")
    return AgentEnvironmentBridge(mgr, env, encoder=enc, max_steps=60)


def test_verify_action_diversity_passes_on_bandit(bandit_bridge):
    result = bandit_bridge.verify_action_diversity()
    assert result["passes_rule16"]
    assert result["metadata"]["provenance"]["operation"] == "verify_action_diversity"


def test_verify_action_diversity_passes_on_gridworld(grid_bridge):
    result = grid_bridge.verify_action_diversity(
        probe_states=[
            {"pos": (1, 1), "step_count": 0, "done": False},
            {"pos": (0, 1), "step_count": 0, "done": False},
            {"pos": (1, 0), "step_count": 0, "done": False},
        ]
    )
    assert result["fraction"] >= 0.4
    assert result["passes_rule16"]


def test_run_episode_writes_episodic_memory(bandit_bridge):
    result = bandit_bridge.run_episode(episodes=1)
    assert result["num_steps"] > 0
    assert result["memory_counts"]["episodic"] >= result["num_steps"] // 2
    assert result["metadata"]["provenance"]["operation"] == "run_episode"


def test_run_episode_produces_learning(bandit_bridge):
    result = bandit_bridge.run_episode(episodes=1)
    assert len(result["losses"]) > 0


def test_rule20_bridge_imports_phases_9_10_11():
    import bridges.agent_environment_bridge as aeb
    assert hasattr(aeb, "SemanticConceptBridge")
    assert hasattr(aeb, "ContinuousLearningBridge")
    assert hasattr(aeb, "MemoryArchitectureBridge")
