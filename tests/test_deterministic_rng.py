"""Tests for :mod:`shared.deterministic_rng` (PLAN.md Phase 2).

Covers:

- :class:`DeterministicRNG` seed forking is reproducible across
  instances and across process runs (the derivation uses BLAKE2b, not
  Python's salted ``hash()``).
- Distinct namespaces produce distinct streams.
- ``reset`` with a namespace rewinds only that fork; ``reset`` with
  no argument drops every fork.
- ``active_namespaces`` reflects fork history.
- ``child_seed`` returns a 32-bit integer that matches the seed used
  for the numpy generator.
- :class:`FrozenRNG` raises :class:`RuntimeError` on every public
  attribute access (including ``random``, ``integers``, ``normal``,
  ``uniform``, ``choice``).
- :class:`FrozenRNG` rejects attribute assignment.
- :class:`DeterministicRNG` rejects non-int master seeds and empty
  namespace strings.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.deterministic_rng import (  # noqa: E402
    DeterministicRNG,
    FrozenRNG,
    _derive_child_seed,
)


# --------------------------------------------------------------------------- #
# DeterministicRNG — structural properties
# --------------------------------------------------------------------------- #


def test_master_seed_round_trip():
    rng = DeterministicRNG(master_seed=123)
    assert rng.master_seed == 123


def test_rejects_non_int_master_seed():
    with pytest.raises(TypeError):
        DeterministicRNG(master_seed="42")  # type: ignore[arg-type]


def test_rejects_empty_namespace():
    rng = DeterministicRNG(master_seed=1)
    with pytest.raises(ValueError):
        rng.fork("")


def test_rejects_non_string_namespace():
    rng = DeterministicRNG(master_seed=1)
    with pytest.raises(ValueError):
        rng.fork(42)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Determinism across instances (the whole point of the class)
# --------------------------------------------------------------------------- #


def test_same_seed_and_namespace_yields_identical_stream():
    a = DeterministicRNG(master_seed=42).fork("cce")
    b = DeterministicRNG(master_seed=42).fork("cce")
    seq_a = a.uniform(0.0, 1.0, 64)
    seq_b = b.uniform(0.0, 1.0, 64)
    assert np.array_equal(seq_a, seq_b)


def test_different_namespaces_yield_different_streams():
    rng = DeterministicRNG(master_seed=42)
    seq_cce = rng.fork("cce").uniform(0.0, 1.0, 64)
    seq_serl = rng.fork("serl").uniform(0.0, 1.0, 64)
    assert not np.array_equal(seq_cce, seq_serl)


def test_different_master_seeds_yield_different_streams():
    a = DeterministicRNG(master_seed=1).fork("cce").uniform(0.0, 1.0, 64)
    b = DeterministicRNG(master_seed=2).fork("cce").uniform(0.0, 1.0, 64)
    assert not np.array_equal(a, b)


def test_repeat_fork_returns_same_generator_object():
    rng = DeterministicRNG(master_seed=7)
    first = rng.fork("cce")
    second = rng.fork("cce")
    assert first is second


def test_fork_generator_advances_across_calls():
    rng = DeterministicRNG(master_seed=7)
    gen = rng.fork("cce")
    first = gen.uniform(0.0, 1.0, 8)
    second = gen.uniform(0.0, 1.0, 8)
    # Sequential draws must not repeat — if they did, fork() would
    # be handing out a fresh generator on every call.
    assert not np.array_equal(first, second)


# --------------------------------------------------------------------------- #
# Reset semantics
# --------------------------------------------------------------------------- #


def test_reset_namespace_rewinds_only_that_fork():
    rng = DeterministicRNG(master_seed=42)
    cce = rng.fork("cce")
    serl = rng.fork("serl")
    first_cce = cce.uniform(0.0, 1.0, 16).copy()
    # Advance serl's state so we can detect it being reset (or not).
    serl_state_before = serl.uniform(0.0, 1.0, 16).copy()

    rng.reset("cce")
    cce_after = rng.fork("cce").uniform(0.0, 1.0, 16)
    assert np.array_equal(cce_after, first_cce)

    # The serl generator was NOT reset, so its next draw differs from
    # the earlier draw.
    serl_after = rng.fork("serl").uniform(0.0, 1.0, 16)
    assert not np.array_equal(serl_after, serl_state_before)


def test_reset_all_forks_clears_and_reseeds():
    rng = DeterministicRNG(master_seed=42)
    first = rng.fork("cce").uniform(0.0, 1.0, 16).copy()
    # Advance it.
    rng.fork("cce").uniform(0.0, 1.0, 16)

    rng.reset()
    assert rng.active_namespaces == ()

    second = rng.fork("cce").uniform(0.0, 1.0, 16)
    assert np.array_equal(second, first)


def test_reset_unknown_namespace_is_noop():
    rng = DeterministicRNG(master_seed=42)
    rng.fork("cce")
    rng.reset("nonexistent")
    assert rng.active_namespaces == ("cce",)


def test_active_namespaces_reflects_history():
    rng = DeterministicRNG(master_seed=42)
    assert rng.active_namespaces == ()
    rng.fork("serl")
    rng.fork("cce")
    rng.fork("swarm")
    assert rng.active_namespaces == ("cce", "serl", "swarm")


# --------------------------------------------------------------------------- #
# child_seed derivation
# --------------------------------------------------------------------------- #


def test_child_seed_is_32bit_unsigned():
    rng = DeterministicRNG(master_seed=42)
    for ns in ("cce", "thdse", "serl", "swarm", "memory"):
        seed = rng.child_seed(ns)
        assert 0 <= seed < 2**32


def test_child_seed_matches_internal_derivation():
    rng = DeterministicRNG(master_seed=42)
    for ns in ("cce", "serl"):
        assert rng.child_seed(ns) == _derive_child_seed(42, ns)


def test_child_seed_matches_fork_generator_seed():
    rng = DeterministicRNG(master_seed=42)
    seed = rng.child_seed("cce")
    expected = np.random.default_rng(seed).uniform(0.0, 1.0, 32)
    actual = rng.fork("cce").uniform(0.0, 1.0, 32)
    assert np.array_equal(expected, actual)


# --------------------------------------------------------------------------- #
# FrozenRNG — every random method must raise
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "method_name",
    ["random", "integers", "normal", "uniform", "choice", "standard_normal"],
)
def test_frozen_rng_raises_on_attribute_access(method_name: str):
    f = FrozenRNG()
    with pytest.raises(RuntimeError, match="FrozenRNG"):
        getattr(f, method_name)


def test_frozen_rng_raises_on_random_call():
    f = FrozenRNG()
    with pytest.raises(RuntimeError) as exc:
        f.random()  # type: ignore[call-arg]
    assert "THDSE synthesis" in str(exc.value)


def test_frozen_rng_repr_reflects_tag():
    f = FrozenRNG(tag="synthesis")
    assert repr(f) == "FrozenRNG(tag='synthesis')"


def test_frozen_rng_default_tag_is_thdse():
    f = FrozenRNG()
    assert "thdse" in repr(f)


def test_frozen_rng_rejects_attribute_assignment():
    f = FrozenRNG()
    with pytest.raises(RuntimeError, match="immutable"):
        f.custom_attr = 123  # type: ignore[attr-defined]


def test_frozen_rng_private_access_returns_attribute_error():
    # Private/dunder-style access should NOT trigger the random-call
    # guard — it should behave like normal Python attribute lookup.
    f = FrozenRNG()
    with pytest.raises(AttributeError):
        _ = f._not_a_real_field
