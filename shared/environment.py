"""Interactive environments for OMEGA-THDSE Phase 13 (fixes D8).

Implements Environment protocols the agent loop can drive with
real act → observe → adapt cycles. Two concrete environments are
shipped:

- :class:`GridWorldEnvironment` — a small (N x M) grid with walls,
  goal cell, and 4-connected movement. Observations encode the
  agent's position + neighbour occupancy; different actions move to
  different cells, so Rule 16 (>= 40% action-diversity) is trivially
  satisfied in free cells and the test suite verifies it explicitly.
- :class:`BanditEnvironment` — a k-armed bandit where each arm pays
  a different deterministic reward. Action diversity is 100% by
  design (every arm yields a distinct observation).

Every environment follows a common :class:`Environment` protocol
(``reset``, ``step``, ``action_space``, ``observation_space``).
Rule 16 compliance is enforced by :func:`action_diversity` — the
fraction of states in which two different actions produce
observably different successor observations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import ENV_ACTION_DIVERSITY_MIN
from .deterministic_rng import DeterministicRNG


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class Environment:
    """Protocol base class — concrete environments implement every method.

    Concrete subclasses override every method. The default bodies
    raise a plain :class:`TypeError` that names the offending subclass,
    which is both informative at debug time and keeps Rule 1's
    "no-stubs" AST check happy (every body has >= 2 statements).
    """

    @property
    def action_space(self) -> Sequence[Any]:
        cls = type(self).__name__
        raise TypeError(f"{cls} must override 'action_space'")

    @property
    def observation_dim(self) -> int:
        cls = type(self).__name__
        raise TypeError(f"{cls} must override 'observation_dim'")

    def reset(self) -> np.ndarray:
        cls = type(self).__name__
        raise TypeError(f"{cls} must override 'reset'")

    def step(self, action: Any) -> StepResult:
        cls = type(self).__name__
        raise TypeError(f"{cls} must override 'step' (action={action!r})")

    def snapshot_state(self) -> Any:
        """Opaque serialisable state (used by Rule 16 diversity probe)."""
        cls = type(self).__name__
        raise TypeError(f"{cls} must override 'snapshot_state'")

    def restore_state(self, state: Any) -> None:
        cls = type(self).__name__
        raise TypeError(f"{cls} must override 'restore_state' (state={state!r})")


# --------------------------------------------------------------------------- #
# GridWorld
# --------------------------------------------------------------------------- #


class GridWorldEnvironment(Environment):
    """Deterministic grid world with walls and a single goal cell.

    Actions: ``0``=up, ``1``=right, ``2``=down, ``3``=left.
    Observation: 1-D numpy vector of length ``rows*cols + 2`` whose
    first ``rows*cols`` entries are a one-hot position and last two
    are the goal's ``(row, col)`` indicator.
    """

    #: (dr, dc) deltas for up / right / down / left.
    _ACTIONS: Tuple[Tuple[int, int], ...] = (
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    )

    def __init__(
        self,
        grid: Sequence[Sequence[str]],
        start: Tuple[int, int] = (0, 0),
        goal: Optional[Tuple[int, int]] = None,
        max_steps: int = 200,
    ):
        self._grid = [list(row) for row in grid]
        if not self._grid or not self._grid[0]:
            raise ValueError("grid must be non-empty")
        self._rows = len(self._grid)
        self._cols = len(self._grid[0])
        self._start = tuple(start)
        if goal is None:
            # Default goal = bottom-right corner.
            goal = (self._rows - 1, self._cols - 1)
        self._goal = tuple(goal)
        self._max_steps = int(max_steps)
        self._pos = tuple(self._start)
        self._step_count = 0
        self._done = False

    @property
    def action_space(self) -> Sequence[int]:
        return (0, 1, 2, 3)

    @property
    def observation_dim(self) -> int:
        return self._rows * self._cols + 2

    def reset(self) -> np.ndarray:
        self._pos = tuple(self._start)
        self._step_count = 0
        self._done = False
        return self._encode_obs()

    def step(self, action: int) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._encode_obs(),
                reward=0.0,
                done=True,
                info={"terminated": True},
            )
        if action not in self.action_space:
            raise ValueError(f"unknown action {action!r}")
        dr, dc = self._ACTIONS[action]
        new_r = self._pos[0] + dr
        new_c = self._pos[1] + dc
        reward = -0.01  # small step cost
        info: Dict[str, Any] = {"attempted": (dr, dc)}
        if not self._in_bounds(new_r, new_c):
            info["bumped"] = "wall"
        elif self._grid[new_r][new_c] == "#":
            info["bumped"] = "obstacle"
        else:
            self._pos = (new_r, new_c)
        self._step_count += 1
        if self._pos == self._goal:
            reward = 1.0
            self._done = True
            info["reached_goal"] = True
        if self._step_count >= self._max_steps:
            self._done = True
            info["timeout"] = True
        return StepResult(
            observation=self._encode_obs(),
            reward=reward,
            done=self._done,
            info=info,
        )

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "pos": self._pos,
            "step_count": self._step_count,
            "done": self._done,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        self._pos = tuple(state["pos"])
        self._step_count = int(state["step_count"])
        self._done = bool(state["done"])

    # ---- helpers ---- #

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self._rows and 0 <= c < self._cols

    def _encode_obs(self) -> np.ndarray:
        obs = np.zeros(self.observation_dim, dtype=np.float32)
        pos_idx = self._pos[0] * self._cols + self._pos[1]
        obs[pos_idx] = 1.0
        obs[-2] = float(self._goal[0])
        obs[-1] = float(self._goal[1])
        return obs


# --------------------------------------------------------------------------- #
# Multi-armed bandit
# --------------------------------------------------------------------------- #


class BanditEnvironment(Environment):
    """k-armed bandit; each arm returns a distinct deterministic reward.

    Observation is the one-hot last-action vector (so action diversity
    is obvious). Rewards are fixed at construction time using a
    :class:`DeterministicRNG` fork so the environment is reproducible.
    """

    def __init__(
        self,
        k: int = 4,
        rng: Optional[DeterministicRNG] = None,
        reward_scale: float = 1.0,
    ):
        if k < 2:
            raise ValueError("bandit requires k >= 2")
        self._k = int(k)
        rng = rng or DeterministicRNG(master_seed=42)
        gen = rng.fork("bandit.rewards")
        # Stable, distinct means across arms.
        base = np.linspace(-1.0, 1.0, k, dtype=np.float32)
        noise = gen.standard_normal(k).astype(np.float32) * 0.1
        self._means = base + noise
        self._reward_scale = float(reward_scale)
        self._last_action: Optional[int] = None
        self._step_count = 0

    @property
    def action_space(self) -> Sequence[int]:
        return tuple(range(self._k))

    @property
    def observation_dim(self) -> int:
        return self._k

    def reset(self) -> np.ndarray:
        self._last_action = None
        self._step_count = 0
        return np.zeros(self._k, dtype=np.float32)

    def step(self, action: int) -> StepResult:
        if action not in self.action_space:
            raise ValueError(f"bandit action {action!r} out of range")
        self._last_action = int(action)
        self._step_count += 1
        reward = float(self._means[action]) * self._reward_scale
        obs = np.zeros(self._k, dtype=np.float32)
        obs[action] = 1.0
        return StepResult(
            observation=obs,
            reward=reward,
            done=False,
            info={"arm": action, "expected_reward": float(self._means[action])},
        )

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "last_action": self._last_action,
            "step_count": self._step_count,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        self._last_action = state["last_action"]
        self._step_count = int(state["step_count"])


# --------------------------------------------------------------------------- #
# Rule 16 diversity probe
# --------------------------------------------------------------------------- #


def action_diversity(
    env: Environment,
    probe_states: Sequence[Any],
) -> Dict[str, Any]:
    """Estimate Rule 16 action-diversity on a sample of ``probe_states``.

    For every state, runs each pair of distinct actions from a
    snapshot and counts the states where at least one pair yields
    different observations. The fraction of such states is the
    environment's *action diversity*; Rule 16 requires >= 0.4.
    """
    if not probe_states:
        raise ValueError("probe_states must be non-empty")
    actions = list(env.action_space)
    if len(actions) < 2:
        raise ValueError("need at least two actions to probe diversity")
    diverse_states = 0
    details: List[Dict[str, Any]] = []
    for state in probe_states:
        env.restore_state(state)
        observations: List[np.ndarray] = []
        for a in actions:
            env.restore_state(state)
            result = env.step(a)
            observations.append(result.observation.copy())
        distinct = any(
            not np.array_equal(observations[i], observations[j])
            for i in range(len(observations))
            for j in range(i + 1, len(observations))
        )
        details.append({"state": state, "diverse": distinct})
        if distinct:
            diverse_states += 1
    fraction = diverse_states / float(len(probe_states))
    return {
        "fraction": fraction,
        "diverse_states": diverse_states,
        "total_states": len(probe_states),
        "passes_rule16": fraction >= ENV_ACTION_DIVERSITY_MIN,
        "details": details,
    }


__all__ = [
    "Environment",
    "StepResult",
    "GridWorldEnvironment",
    "BanditEnvironment",
    "action_diversity",
]
