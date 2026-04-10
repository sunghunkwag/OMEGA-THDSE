"""Structural determinism enforcement for OMEGA-THDSE (PLAN.md Section E).

Two classes are exposed:

- :class:`DeterministicRNG` — a seed-forking RNG used by CCE paths that
  need reproducible randomness (Agent exploration, SERL evolution, Swarm
  sampling, memory dropout, etc.).
- :class:`FrozenRNG` — a sentinel RNG that raises on ANY random-number
  call, used by THDSE axiomatic synthesis paths where even a single
  random draw would invalidate Z3 proof reproducibility.

PLAN.md Rule 10 forbids bare ``import random``, ``np.random.rand(``, and
``np.random.random(`` anywhere in ``shared/`` except this file. All
randomness in the unified engine must flow through these classes.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np


def _derive_child_seed(master_seed: int, namespace: str) -> int:
    """Deterministically derive a 32-bit child seed from master+namespace.

    Uses BLAKE2b so the derivation is stable across Python interpreter
    runs (unlike Python's ``hash()``, which is salted by PYTHONHASHSEED
    for strings). A stable derivation is required so that two processes
    seeded with the same master seed produce the same forks — which is
    the whole point of a deterministic RNG.
    """
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    payload = f"{int(master_seed)}::{namespace}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


class DeterministicRNG:
    """Structural determinism enforcement for the unified engine.

    THDSE operations get FROZEN RNG (no randomness allowed) via
    :class:`FrozenRNG`. CCE operations get SEEDED RNG (reproducible
    randomness) via :meth:`fork`.

    Usage::

        rng = DeterministicRNG(master_seed=42)

        # For THDSE (deterministic):
        thdse_rng = FrozenRNG()  # Any call → RuntimeError

        # For CCE (reproducible random):
        cce_rng = rng.fork("cce")            # seeded, reproducible
        serl_rng = rng.fork("serl")          # independent stream
    """

    def __init__(self, master_seed: int = 42):
        if not isinstance(master_seed, int):
            raise TypeError(
                f"master_seed must be int, got {type(master_seed).__name__}"
            )
        self._master_seed = int(master_seed)
        self._forks: dict[str, np.random.Generator] = {}

    def fork(self, namespace: str) -> np.random.Generator:
        """Return (creating if necessary) a deterministic fork for a subsystem.

        The same ``namespace`` always maps to the same initial stream for
        a given ``master_seed``. Repeated calls with the same namespace
        return the SAME generator object, so consumers can advance its
        state across calls without losing determinism.
        """
        if namespace not in self._forks:
            child_seed = _derive_child_seed(self._master_seed, namespace)
            self._forks[namespace] = np.random.default_rng(child_seed)
        return self._forks[namespace]

    def reset(self, namespace: Optional[str] = None) -> None:
        """Reset a single fork (or all forks) to its initial state.

        With ``namespace=None`` all known forks are dropped; subsequent
        :meth:`fork` calls re-create them from the master seed.
        """
        if namespace is None:
            self._forks.clear()
            return
        if namespace in self._forks:
            child_seed = _derive_child_seed(self._master_seed, namespace)
            self._forks[namespace] = np.random.default_rng(child_seed)

    def child_seed(self, namespace: str) -> int:
        """Return the 32-bit seed derived for a namespace.

        Exposed for the bridge modules that need to seed their own
        non-numpy RNGs (e.g. Z3 solver parameter ``random_seed``) using
        the same master key.
        """
        return _derive_child_seed(self._master_seed, namespace)

    @property
    def master_seed(self) -> int:
        return self._master_seed

    @property
    def active_namespaces(self) -> tuple:
        return tuple(sorted(self._forks.keys()))

    def __repr__(self) -> str:
        return (
            f"DeterministicRNG(master_seed={self._master_seed}, "
            f"forks={self.active_namespaces})"
        )


class FrozenRNG:
    """RNG that raises on any attempt to generate random numbers.

    Used for THDSE synthesis paths where ANY randomness would invalidate
    Z3 axiomatic proofs. Designed to be a drop-in replacement for an
    :class:`numpy.random.Generator`: attribute access for any public
    method raises :class:`RuntimeError` immediately, before the call
    even happens.

    PLAN.md Rule 10 requires that THDSE synthesis receives an instance
    of this class (never a seeded generator). PLAN.md Section F Risk 2
    requires that ``FrozenRNG().random()`` raise ``RuntimeError``.
    """

    __slots__ = ("_tag",)

    def __init__(self, tag: str = "thdse"):
        object.__setattr__(self, "_tag", str(tag))

    def __getattr__(self, name: str):  # noqa: D401 — behavioral hook
        # Private/dunder access goes through normal resolution so that
        # repr, pickling checks, hash, etc. continue to behave sanely.
        if name.startswith("_"):
            raise AttributeError(name)
        raise RuntimeError(
            f"FrozenRNG[{self._tag}]: Attempted to call '{name}()' in a "
            f"deterministic context. THDSE synthesis operations must not "
            f"use randomness. If you need randomness, use "
            f"DeterministicRNG.fork('cce') instead."
        )

    def __setattr__(self, name: str, value) -> None:
        raise RuntimeError(
            f"FrozenRNG[{self._tag}]: Attempted to set attribute '{name}'. "
            f"FrozenRNG is immutable."
        )

    def __repr__(self) -> str:
        return f"FrozenRNG(tag={self._tag!r})"


__all__ = ["DeterministicRNG", "FrozenRNG"]
