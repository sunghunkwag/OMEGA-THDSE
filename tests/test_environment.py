"""Phase 13 — Environment + agent loop (Rule 16, D8)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.agent_loop import AgentLoop, epsilon_greedy_random_policy
from shared.deterministic_rng import DeterministicRNG
from shared.environment import (
    BanditEnvironment,
    GridWorldEnvironment,
    StepResult,
    action_diversity,
)
from shared.online_learner import ExperienceReplayBuffer, OnlineLearner


# --------------------------------------------------------------------------- #
# Rule 16 — different actions must produce different observations
# --------------------------------------------------------------------------- #


def test_rule16_gridworld_actions_produce_distinct_observations():
    env = GridWorldEnvironment([[".", "."], [".", "."]])
    env.reset()
    # Center-of-grid state (1,0) gives different outcomes for each action.
    env.restore_state({"pos": (1, 0), "step_count": 0, "done": False})
    observations = set()
    state = env.snapshot_state()
    for a in env.action_space:
        env.restore_state(state)
        r = env.step(a)
        observations.add(tuple(r.observation.tolist()))
    # At least 2 distinct observations from 4 actions.
    assert len(observations) >= 2


def test_rule16_gridworld_diversity_fraction_meets_threshold():
    env = GridWorldEnvironment([[".", ".", "."], [".", ".", "."], [".", ".", "."]])
    env.reset()
    # Probe a handful of interior states (none against the boundary).
    probe_states = [
        {"pos": (1, 1), "step_count": 0, "done": False},
        {"pos": (0, 1), "step_count": 0, "done": False},
        {"pos": (1, 0), "step_count": 0, "done": False},
        {"pos": (2, 1), "step_count": 0, "done": False},
        {"pos": (1, 2), "step_count": 0, "done": False},
    ]
    result = action_diversity(env, probe_states)
    assert result["fraction"] >= 0.4
    assert result["passes_rule16"]


def test_rule16_bandit_has_full_diversity():
    env = BanditEnvironment(k=4, rng=DeterministicRNG(master_seed=7))
    env.reset()
    probe_states = [
        {"last_action": None, "step_count": 0},
        {"last_action": 0, "step_count": 5},
        {"last_action": 2, "step_count": 10},
    ]
    result = action_diversity(env, probe_states)
    assert result["fraction"] == 1.0


def test_gridworld_rejects_unknown_action():
    env = GridWorldEnvironment([[".", "."]])
    env.reset()
    with pytest.raises(ValueError):
        env.step(42)


# --------------------------------------------------------------------------- #
# GridWorld behaviour
# --------------------------------------------------------------------------- #


def test_gridworld_reaches_goal_terminates_episode():
    env = GridWorldEnvironment(
        [[".", "."], [".", "."]], start=(0, 0), goal=(0, 1)
    )
    env.reset()
    result = env.step(1)  # move right → goal
    assert result.done
    assert result.reward > 0.5
    assert result.info.get("reached_goal")


def test_gridworld_observation_shape():
    env = GridWorldEnvironment([[".", ".", "."], [".", ".", "."]])
    obs = env.reset()
    assert obs.shape == (2 * 3 + 2,)


# --------------------------------------------------------------------------- #
# Bandit reward determinism
# --------------------------------------------------------------------------- #


def test_bandit_rewards_are_deterministic_given_rng():
    rng_a = DeterministicRNG(master_seed=99)
    rng_b = DeterministicRNG(master_seed=99)
    env_a = BanditEnvironment(k=4, rng=rng_a)
    env_b = BanditEnvironment(k=4, rng=rng_b)
    env_a.reset()
    env_b.reset()
    for arm in (0, 1, 2, 3, 2, 1):
        assert env_a.step(arm).reward == env_b.step(arm).reward


def test_bandit_distinct_mean_rewards_per_arm():
    env = BanditEnvironment(k=5, rng=DeterministicRNG(master_seed=5))
    env.reset()
    rewards = [env.step(a).reward for a in env.action_space]
    # All five arms must have distinct rewards (diversity).
    assert len(set(rewards)) == 5


# --------------------------------------------------------------------------- #
# AgentLoop — adapts a learner
# --------------------------------------------------------------------------- #


def test_agent_loop_populates_trace_and_buffer():
    env = BanditEnvironment(k=4, rng=DeterministicRNG(master_seed=11))
    agent = AgentLoop(
        env,
        rng=DeterministicRNG(master_seed=11),
        max_steps=40,
        learn_every=2,
        batch_size=8,
    )
    trace = agent.run(episodes=1)
    assert trace["num_steps"] == 40
    # With batch_size=8 and learn_every=2, learner must train many times.
    assert len(trace["losses"]) > 0
    assert len(agent.buffer) >= 40


def test_agent_loop_actually_reduces_loss_on_average():
    env = BanditEnvironment(k=4, rng=DeterministicRNG(master_seed=22))
    agent = AgentLoop(
        env,
        rng=DeterministicRNG(master_seed=22),
        max_steps=120,
        learn_every=3,
        batch_size=16,
    )
    trace = agent.run(episodes=1)
    # First half vs second half: second half mean loss should be lower.
    losses = [info["loss_after"] for info in trace["losses"]]
    assert len(losses) >= 10
    first = np.mean(losses[: len(losses) // 2])
    second = np.mean(losses[len(losses) // 2 :])
    assert second < first, (
        f"agent did not adapt: first_half={first:.4f} second_half={second:.4f}"
    )


def test_consolidation_callback_fires():
    env = BanditEnvironment(k=3, rng=DeterministicRNG(master_seed=3))
    agent = AgentLoop(
        env,
        rng=DeterministicRNG(master_seed=3),
        max_steps=80,
        consolidation_interval=10,
        learn_every=20,
        batch_size=4,
    )
    calls = []
    agent.set_consolidation_callback(lambda step: {"step": step})
    agent.run(episodes=1)
    # With max_steps=80 and interval=10, at least 7 callbacks fire.
    # The callback writes into `calls` via closure — but we used lambda
    # without side-effects, so instead inspect the returned trace:
    # just re-run with an accumulating callback.
    env.reset()
    agent2 = AgentLoop(
        env,
        rng=DeterministicRNG(master_seed=3),
        max_steps=80,
        consolidation_interval=10,
        learn_every=20,
        batch_size=4,
    )
    agent2.set_consolidation_callback(lambda step: calls.append(step) or {})
    agent2.run(episodes=1)
    assert len(calls) >= 7
