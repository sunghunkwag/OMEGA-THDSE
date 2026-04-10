"""Tests for :mod:`bridges.governance_synthesis_bridge` (Gap 6) and the
Rule 8 invariants enforced by :mod:`bridges.causal_provenance_bridge`
(Gap 4).

This file combines governance-synthesis coverage (parse checks, critic
threshold, credible-leap guard, gate_registration chokepoint) with
the no-silent-failures tests for the causal provenance bridge (every
UNSAT must be counted, chronological ordering preserved, filtering by
event type, Rule 8 audit counter).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bridges.causal_provenance_bridge import CausalProvenanceBridge  # noqa: E402
from bridges.governance_synthesis_bridge import (  # noqa: E402
    GovernanceSynthesisBridge,
)
from shared.arena_manager import ArenaManager  # noqa: E402
from shared.constants import (  # noqa: E402
    CRITIC_THRESHOLD,
    MAX_CREDIBLE_LEAP,
    THDSE_ARENA_DIM,
)

_TWO_PI = 2.0 * math.pi


def _seed_axiom(mgr: ArenaManager, fork_name: str) -> int:
    rng = mgr.rng.fork(fork_name)
    phases = rng.uniform(0.0, _TWO_PI, THDSE_ARENA_DIM).astype(np.float32)
    return mgr.alloc_thdse(phases=phases)


# --------------------------------------------------------------------------- #
# GovernanceSynthesisBridge fixtures + tests
# --------------------------------------------------------------------------- #


@pytest.fixture
def gov_mgr() -> ArenaManager:
    return ArenaManager(master_seed=303)


@pytest.fixture
def gov_bridge(gov_mgr: ArenaManager) -> GovernanceSynthesisBridge:
    return GovernanceSynthesisBridge(gov_mgr)


def test_governance_bridge_rejects_non_manager():
    with pytest.raises(TypeError):
        GovernanceSynthesisBridge(arena_manager=42)  # type: ignore[arg-type]


def test_evaluate_candidate_approves_clean_high_fitness(
    gov_bridge: GovernanceSynthesisBridge, gov_mgr: ArenaManager
):
    h = _seed_axiom(gov_mgr, "clean")
    result = gov_bridge.evaluate_candidate(
        "def f():\n    return 1\n", h, CRITIC_THRESHOLD + 0.05
    )
    assert result["approved"] is True
    assert result["reason"] == "approved"
    assert result["metadata"]["checks"]["parse_ok"] is True
    assert result["metadata"]["checks"]["critic_ok"] is True


def test_evaluate_candidate_rejects_low_fitness(
    gov_bridge: GovernanceSynthesisBridge, gov_mgr: ArenaManager
):
    h = _seed_axiom(gov_mgr, "low")
    result = gov_bridge.evaluate_candidate(
        "x = 1", h, CRITIC_THRESHOLD - 0.2
    )
    assert result["approved"] is False
    assert "CRITIC_THRESHOLD" in result["reason"]
    assert result["metadata"]["checks"]["critic_ok"] is False


def test_evaluate_candidate_rejects_syntax_error(
    gov_bridge: GovernanceSynthesisBridge, gov_mgr: ArenaManager
):
    h = _seed_axiom(gov_mgr, "syntax")
    result = gov_bridge.evaluate_candidate(
        "def broken(:", h, CRITIC_THRESHOLD + 0.1
    )
    assert result["approved"] is False
    assert "SyntaxError" in result["reason"]
    assert result["metadata"]["checks"]["parse_ok"] is False


def test_evaluate_candidate_rejects_empty_source(
    gov_bridge: GovernanceSynthesisBridge, gov_mgr: ArenaManager
):
    h = _seed_axiom(gov_mgr, "empty")
    result = gov_bridge.evaluate_candidate("", h, CRITIC_THRESHOLD + 0.1)
    assert result["approved"] is False
    assert result["metadata"]["checks"]["parse_ok"] is False


def test_evaluate_candidate_enforces_credible_leap(
    gov_bridge: GovernanceSynthesisBridge, gov_mgr: ArenaManager
):
    h = _seed_axiom(gov_mgr, "leap")
    # First pass: baseline just above threshold.
    first = gov_bridge.evaluate_candidate("y = 2", h, CRITIC_THRESHOLD)
    assert first["approved"] is True

    # Second pass: leap by MAX_CREDIBLE_LEAP + 0.05 → must be blocked.
    jump = CRITIC_THRESHOLD + MAX_CREDIBLE_LEAP + 0.05
    second = gov_bridge.evaluate_candidate("y = 3", h, jump)
    assert second["approved"] is False
    assert "MAX_CREDIBLE_LEAP" in second["reason"]
    assert second["metadata"]["leap_delta"] == pytest.approx(
        jump - CRITIC_THRESHOLD, abs=1e-6
    )


def test_evaluate_candidate_result_has_provenance(
    gov_bridge: GovernanceSynthesisBridge, gov_mgr: ArenaManager
):
    h = _seed_axiom(gov_mgr, "prov")
    result = gov_bridge.evaluate_candidate("z = 0", h, CRITIC_THRESHOLD + 0.1)
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "evaluate_candidate"
    assert prov["source_arena"] == "cce"
    assert prov["target_arena"] == "thdse"
    assert isinstance(prov["timestamp"], float)
    assert result["metadata"]["critic_threshold"] == CRITIC_THRESHOLD
    assert result["metadata"]["max_credible_leap"] == MAX_CREDIBLE_LEAP


def test_gate_registration_passes_only_on_approved(
    gov_bridge: GovernanceSynthesisBridge,
):
    assert gov_bridge.gate_registration({"approved": True}) is True
    assert gov_bridge.gate_registration({"approved": False}) is False
    assert gov_bridge.gate_registration({}) is False
    with pytest.raises(TypeError):
        gov_bridge.gate_registration("not a dict")  # type: ignore[arg-type]


def test_history_and_approval_rate_update(
    gov_bridge: GovernanceSynthesisBridge, gov_mgr: ArenaManager
):
    h = _seed_axiom(gov_mgr, "hist")
    gov_bridge.evaluate_candidate("a = 1", h, CRITIC_THRESHOLD + 0.1)
    gov_bridge.evaluate_candidate("a = 2", h, CRITIC_THRESHOLD - 0.5)
    assert gov_bridge.history_length == 2
    # One approved, one rejected → 0.5 approval rate.
    assert gov_bridge.approval_rate() == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# CausalProvenanceBridge — Rule 8 (no silent UNSAT failures)
# --------------------------------------------------------------------------- #


@pytest.fixture
def causal_mgr() -> ArenaManager:
    return ArenaManager(master_seed=404)


@pytest.fixture
def causal_bridge(causal_mgr: ArenaManager) -> CausalProvenanceBridge:
    return CausalProvenanceBridge(causal_mgr)


def test_causal_bridge_rule_8_unsat_count_is_tracked(
    causal_bridge: CausalProvenanceBridge, causal_mgr: ArenaManager
):
    h = _seed_axiom(causal_mgr, "unsat")
    # Record 5 events: 3 UNSAT, 2 SAT. Rule 8 requires get_unsat_count() == 3.
    causal_bridge.record_synthesis_event("unsat", h, {"reason": "contradiction"})
    causal_bridge.record_synthesis_event("sat", h, {"proof": "yes"})
    causal_bridge.record_synthesis_event("unsat", h, {"reason": "timeout"})
    causal_bridge.record_synthesis_event("sat", h, {})
    causal_bridge.record_synthesis_event("unsat", None, {"reason": "deadend"})
    assert causal_bridge.get_unsat_count() == 3
    assert causal_bridge.get_sat_count() == 2
    assert causal_bridge.total_events() == 5


def test_causal_bridge_rejects_invalid_event_type(
    causal_bridge: CausalProvenanceBridge,
):
    with pytest.raises(ValueError):
        causal_bridge.record_synthesis_event("mystery", None, {})


def test_causal_bridge_chain_is_chronological(
    causal_bridge: CausalProvenanceBridge, causal_mgr: ArenaManager
):
    h = _seed_axiom(causal_mgr, "chrono")
    ids: list[str] = []
    for kind in ("sat", "unsat", "serl_cycle", "swarm_consensus"):
        r = causal_bridge.record_synthesis_event(kind, h, {"k": kind})
        ids.append(r["event_id"])
    chain = causal_bridge.get_chain()
    assert [e["event_id"] for e in chain] == ids
    assert [e["sequence_index"] for e in chain] == [0, 1, 2, 3]


def test_causal_bridge_filter_by_type_returns_matching_events(
    causal_bridge: CausalProvenanceBridge, causal_mgr: ArenaManager
):
    h = _seed_axiom(causal_mgr, "filter")
    causal_bridge.record_synthesis_event("sat", h, {})
    causal_bridge.record_synthesis_event("unsat", h, {})
    causal_bridge.record_synthesis_event("unsat", h, {})
    unsat_events = causal_bridge.filter_by_type("unsat")
    assert len(unsat_events) == 2
    for ev in unsat_events:
        assert ev["event_type"] == "unsat"


def test_causal_bridge_event_record_has_provenance(
    causal_bridge: CausalProvenanceBridge, causal_mgr: ArenaManager
):
    h = _seed_axiom(causal_mgr, "provcpb")
    result = causal_bridge.record_synthesis_event("sat", h, {"k": 1})
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "record_synthesis_event"
    assert prov["source_arena"] == "thdse"
    assert prov["target_arena"] == "cce"
    assert prov["event_type"] == "sat"
    assert isinstance(prov["timestamp"], float)


def test_causal_bridge_describe_dimensions_matches_manager(
    causal_bridge: CausalProvenanceBridge, causal_mgr: ArenaManager
):
    dims = causal_bridge.describe_dimensions()
    assert dims["cce_dim"] == 10_000
    assert dims["thdse_dim"] == 256
    assert dims["cce_dim_on_manager"] == causal_mgr.cce_dim
    assert dims["thdse_dim_on_manager"] == causal_mgr.thdse_dim


def test_causal_bridge_rejects_non_int_handle(
    causal_bridge: CausalProvenanceBridge,
):
    with pytest.raises(TypeError):
        causal_bridge.record_synthesis_event("sat", "abc", {})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        causal_bridge.record_synthesis_event("sat", 0, "details")  # type: ignore[arg-type]
