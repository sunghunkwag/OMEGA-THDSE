"""Agent interaction loop for OMEGA-THDSE Phase 13 (fixes D8).

Runs a real ``act → observe → adapt`` cycle against any
:class:`shared.environment.Environment` instance. The loop:

1. picks an action via a pluggable ``policy_fn`` (defaults to an
   epsilon-greedy Q-table over the environment's action space),
2. pushes the transition into a replay buffer tied to an
   :class:`OnlineLearner` (Phase 10),
3. periodically trains the learner on sampled batches, and
4. consolidates episodic memory every
   :data:`AGENT_CONSOLIDATION_INTERVAL` steps by promoting rehearsed
   events into semantic facts.

The loop returns a structured trace covering every step — including
rewards, learner losses, and consolidation events — so Phase 13
tests can assert that the cycle really updates parameters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .constants import (
    AGENT_CONSOLIDATION_INTERVAL,
    AGENT_LOOP_MAX_STEPS,
    ONLINE_LEARNER_DEFAULT_LR,
)
from .deterministic_rng import DeterministicRNG
from .environment import Environment, StepResult
from .online_learner import (
    ExperienceReplayBuffer,
    OnlineLearner,
    loss_decreased,
)


@dataclass
class InteractionRecord:
    step: int
    action: Any
    reward: float
    observation_before: np.ndarray
    observation_after: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


PolicyFn = Callable[[np.ndarray, Sequence[Any], DeterministicRNG], Any]


def epsilon_greedy_random_policy(epsilon: float = 1.0) -> PolicyFn:
    """Return a simple exploration policy that picks uniformly at random.

    With ``epsilon=1.0`` this is pure exploration; the point in
    Phase 13 is to prove the loop interacts with the environment and
    adapts a learner, not to master control — deep RL is out of scope.
    """
    eps = float(epsilon)

    def _policy(
        obs: np.ndarray, actions: Sequence[Any], rng: DeterministicRNG
    ) -> Any:
        gen = rng.fork("agent_policy")
        if gen.random() < eps:
            return actions[int(gen.integers(0, len(actions)))]
        return actions[0]

    return _policy


class AgentLoop:
    """Run an environment interaction loop with real learning."""

    def __init__(
        self,
        env: Environment,
        *,
        learner: Optional[OnlineLearner] = None,
        buffer: Optional[ExperienceReplayBuffer] = None,
        policy_fn: Optional[PolicyFn] = None,
        rng: Optional[DeterministicRNG] = None,
        max_steps: int = AGENT_LOOP_MAX_STEPS,
        consolidation_interval: int = AGENT_CONSOLIDATION_INTERVAL,
        learn_every: int = 4,
        batch_size: int = 16,
    ):
        if not isinstance(env, Environment):
            raise TypeError(f"env must be Environment, got {type(env).__name__}")
        self._env = env
        self._rng = rng or DeterministicRNG(master_seed=42)
        self._policy = policy_fn or epsilon_greedy_random_policy(1.0)
        self._max_steps = int(max_steps)
        self._consolidation_interval = int(consolidation_interval)
        self._learn_every = int(learn_every)
        self._batch_size = int(batch_size)
        obs_dim = env.observation_dim
        # Learner predicts reward as a 1-D regression target (value fn).
        self._learner = learner or OnlineLearner(
            input_dim=obs_dim,
            output_dim=1,
            hidden_dims=[32, 16],
            lr=ONLINE_LEARNER_DEFAULT_LR,
            rng=self._rng,
        )
        self._buffer = buffer or ExperienceReplayBuffer(
            capacity=2000, rng=self._rng
        )
        self._consolidate_cb: Optional[Callable[[int], Dict[str, Any]]] = None

    @property
    def learner(self) -> OnlineLearner:
        return self._learner

    @property
    def buffer(self) -> ExperienceReplayBuffer:
        return self._buffer

    def set_consolidation_callback(
        self, cb: Callable[[int], Dict[str, Any]]
    ) -> None:
        """Register a memory-consolidation callback.

        Called every :data:`AGENT_CONSOLIDATION_INTERVAL` steps with
        the current step index; the returned dict is appended to the
        trace's ``consolidations`` list.
        """
        self._consolidate_cb = cb

    def run(self, episodes: int = 1) -> Dict[str, Any]:
        """Run ``episodes`` full episodes and return the aggregated trace."""
        trace: List[InteractionRecord] = []
        losses: List[Dict[str, Any]] = []
        consolidations: List[Dict[str, Any]] = []
        total_reward = 0.0
        for _episode in range(int(episodes)):
            obs = self._env.reset()
            for step_idx in range(self._max_steps):
                actions = list(self._env.action_space)
                action = self._policy(obs, actions, self._rng)
                result: StepResult = self._env.step(action)
                # Store transition (obs_before, reward) in replay buffer.
                self._buffer.add(
                    obs.astype(np.float32),
                    np.array([result.reward], dtype=np.float32),
                )
                total_reward += float(result.reward)
                trace.append(
                    InteractionRecord(
                        step=len(trace),
                        action=action,
                        reward=float(result.reward),
                        observation_before=obs.copy(),
                        observation_after=result.observation.copy(),
                        done=bool(result.done),
                        info=dict(result.info),
                    )
                )
                obs = result.observation
                # Learn periodically.
                if (
                    len(self._buffer) >= self._batch_size
                    and (step_idx + 1) % self._learn_every == 0
                ):
                    info = self._learner.train_batch(
                        self._buffer, self._batch_size
                    )
                    info["step"] = len(trace)
                    info["loss_decreased"] = loss_decreased(info)
                    losses.append(info)
                # Consolidate periodically.
                if (
                    self._consolidate_cb is not None
                    and (step_idx + 1) % self._consolidation_interval == 0
                ):
                    cons_result = self._consolidate_cb(len(trace))
                    consolidations.append(
                        {"step": len(trace), "result": cons_result}
                    )
                if result.done:
                    break
        return {
            "steps": [
                {
                    "step": r.step,
                    "action": r.action,
                    "reward": r.reward,
                    "done": r.done,
                    "info": r.info,
                }
                for r in trace
            ],
            "total_reward": float(total_reward),
            "num_steps": len(trace),
            "losses": losses,
            "consolidations": consolidations,
            "episodes": int(episodes),
        }


__all__ = [
    "AgentLoop",
    "InteractionRecord",
    "epsilon_greedy_random_policy",
]
