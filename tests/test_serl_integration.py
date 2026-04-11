"""Phase 6 — Tier-1 wiring tests for the SERL evolutionary loop.

These tests construct a real :class:`SERLLoop` with real bridge
instances and exercise the post-decode handler
(:meth:`SERLLoop.handle_decode_result`) directly. They never mock
the SERL class or the bridges. They DO skip running the full
synthesis loop because that requires Z3 and a corpus — those paths
are exercised by the Tier-2 file when Z3 is installed.

The post-decode handler is the central wiring point: every decode
result (SAT or UNSAT) flows through it, then optionally through
:class:`bridges.rsi_serl_bridge.RsiSerlBridge`,
:class:`bridges.governance_synthesis_bridge.GovernanceSynthesisBridge`,
and :class:`bridges.axiom_skill_bridge.AxiomSkillBridge`. Each of
the tests below verifies a specific transition in that pipeline.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_THDSE = _REPO_ROOT / "thdse"
_CCE = _REPO_ROOT / "Cognitive-Core-Engine-Test"
for p in (str(_REPO_ROOT), str(_THDSE), str(_CCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

if "cognitive_core_engine.core.hdc" not in sys.modules:
    _stub = types.ModuleType("cognitive_core_engine.core.hdc")

    class _StubHV:
        DIM = 10_000

    _stub.HyperVector = _StubHV
    sys.modules["cognitive_core_engine.core.hdc"] = _stub


from shared.arena_manager import ArenaManager  # noqa: E402
from shared.constants import (  # noqa: E402
    CRITIC_THRESHOLD,
    SERL_FITNESS_GATE,
    THDSE_ARENA_DIM,
)
from shared.exceptions import GovernanceError  # noqa: E402

from bridges.axiom_skill_bridge import AxiomSkillBridge  # noqa: E402
from bridges.causal_provenance_bridge import CausalProvenanceBridge  # noqa: E402
from bridges.governance_synthesis_bridge import (  # noqa: E402
    GovernanceSynthesisBridge,
)
from bridges.rsi_serl_bridge import RsiSerlBridge  # noqa: E402

from cognitive_core_engine.core.causal_chain import (  # noqa: E402
    CausalChainTracker,
)

from src.synthesis.serl import SERLLoop  # noqa: E402


_VALID_SOURCE = "def evolved():\n    return 42\n"


def _make_thdse_handle(mgr: ArenaManager) -> int:
    """Allocate a real THDSE handle whose phases the bridges can read."""
    return mgr.alloc_thdse(
        phases=np.zeros(THDSE_ARENA_DIM, dtype=np.float32)
    )


@pytest.fixture
def manager() -> ArenaManager:
    return ArenaManager(master_seed=6101)


@pytest.fixture
def fully_wired_loop(manager: ArenaManager) -> SERLLoop:
    return SERLLoop(
        arena_manager=manager,
        rsi_serl_bridge=RsiSerlBridge(manager),
        governance_bridge=GovernanceSynthesisBridge(manager),
        axiom_skill_bridge=AxiomSkillBridge(manager),
        provenance_bridge=CausalProvenanceBridge(manager),
        causal_tracker=CausalChainTracker(),
    )


# --------------------------------------------------------------------------- #
# Constructor (Rule 16 backward compatibility)
# --------------------------------------------------------------------------- #


def test_serl_legacy_constructor_keeps_working():
    loop = SERLLoop()
    assert loop._rsi_serl_bridge is None  # noqa: SLF001
    assert loop._provenance_bridge is None  # noqa: SLF001
    # Even with no bridges wired, the handler returns a structured dict.
    result = loop.handle_decode_result(
        source=None, fitness=0.0, thdse_handle=0, round_idx=0
    )
    assert result["result"] == "unsat"
    assert result["registered"] is False


def test_serl_constructor_stores_all_bridges(manager: ArenaManager):
    rsi = RsiSerlBridge(manager)
    gov = GovernanceSynthesisBridge(manager)
    asb = AxiomSkillBridge(manager)
    pb = CausalProvenanceBridge(manager)
    tracker = CausalChainTracker()
    loop = SERLLoop(
        arena_manager=manager,
        rsi_serl_bridge=rsi,
        governance_bridge=gov,
        axiom_skill_bridge=asb,
        provenance_bridge=pb,
        causal_tracker=tracker,
    )
    assert loop._rsi_serl_bridge is rsi  # noqa: SLF001
    assert loop._governance_bridge is gov  # noqa: SLF001
    assert loop._axiom_skill_bridge is asb  # noqa: SLF001
    assert loop._provenance_bridge is pb  # noqa: SLF001
    assert loop._causal_tracker is tracker  # noqa: SLF001


# --------------------------------------------------------------------------- #
# UNSAT path — Rule 8 double logging
# --------------------------------------------------------------------------- #


def test_unsat_decode_emits_provenance_and_causal_events(
    manager: ArenaManager, fully_wired_loop: SERLLoop
):
    handle = _make_thdse_handle(manager)
    result = fully_wired_loop.handle_decode_result(
        source=None,
        fitness=0.0,
        thdse_handle=handle,
        round_idx=4,
    )
    assert result["result"] == "unsat"
    assert result["registered"] is False
    pb = fully_wired_loop._provenance_bridge  # noqa: SLF001
    tracker = fully_wired_loop._causal_tracker  # noqa: SLF001
    assert pb.get_unsat_count() == 1
    unsat_events = [
        e for e in tracker._events  # noqa: SLF001
        if e.event_type == "synthesis_unsat"
    ]
    assert len(unsat_events) == 1
    assert unsat_events[0].data["logged"] is True


def test_unsat_decode_with_empty_source_is_unsat(
    manager: ArenaManager, fully_wired_loop: SERLLoop
):
    handle = _make_thdse_handle(manager)
    result = fully_wired_loop.handle_decode_result(
        source="   ",  # Whitespace-only counts as UNSAT
        fitness=0.0,
        thdse_handle=handle,
        round_idx=1,
    )
    assert result["result"] == "unsat"


# --------------------------------------------------------------------------- #
# SAT path — RSI/governance/skill pipeline
# --------------------------------------------------------------------------- #


def test_sat_below_gate_emits_sat_but_does_not_register(
    manager: ArenaManager, fully_wired_loop: SERLLoop
):
    handle = _make_thdse_handle(manager)
    fitness = SERL_FITNESS_GATE - 0.1  # Below the gate.
    result = fully_wired_loop.handle_decode_result(
        source=_VALID_SOURCE,
        fitness=fitness,
        thdse_handle=handle,
        round_idx=2,
    )
    assert result["result"] == "sat"
    assert result["eligible"] is False
    assert result["registered"] is False
    pb = fully_wired_loop._provenance_bridge  # noqa: SLF001
    assert pb.get_sat_count() == 1


def test_sat_above_gate_runs_full_pipeline_and_registers(
    manager: ArenaManager, fully_wired_loop: SERLLoop
):
    handle = _make_thdse_handle(manager)
    fitness = max(SERL_FITNESS_GATE, CRITIC_THRESHOLD) + 0.1
    result = fully_wired_loop.handle_decode_result(
        source=_VALID_SOURCE,
        fitness=fitness,
        thdse_handle=handle,
        round_idx=5,
    )
    assert result["result"] == "sat"
    assert result["eligible"] is True
    assert result["approved"] is True
    assert result["registered"] is True
    asb = fully_wired_loop._axiom_skill_bridge  # noqa: SLF001
    assert asb.registered_count >= 1
    pb = fully_wired_loop._provenance_bridge  # noqa: SLF001
    assert pb.get_sat_count() == 1


def test_sat_above_serl_below_critic_blocks_at_governance(
    manager: ArenaManager,
):
    """A candidate above SERL_FITNESS_GATE but below CRITIC_THRESHOLD
    must clear RSI eligibility but fail governance approval."""
    rsi = RsiSerlBridge(manager)
    gov = GovernanceSynthesisBridge(manager)
    asb = AxiomSkillBridge(manager)
    loop = SERLLoop(
        arena_manager=manager,
        rsi_serl_bridge=rsi,
        governance_bridge=gov,
        axiom_skill_bridge=asb,
    )
    handle = _make_thdse_handle(manager)
    # SERL gate is 0.4; critic threshold is 0.7. Pick 0.5.
    fitness = (SERL_FITNESS_GATE + CRITIC_THRESHOLD) / 2.0
    result = loop.handle_decode_result(
        source=_VALID_SOURCE,
        fitness=fitness,
        thdse_handle=handle,
        round_idx=0,
    )
    assert result["eligible"] is True
    assert result["approved"] is False
    assert result["registered"] is False
    assert asb.registered_count == 0


# --------------------------------------------------------------------------- #
# Counters & feedback path
# --------------------------------------------------------------------------- #


def test_handler_invocation_counter_advances(
    manager: ArenaManager, fully_wired_loop: SERLLoop
):
    handle = _make_thdse_handle(manager)
    fully_wired_loop.handle_decode_result(
        source=None, fitness=0.0, thdse_handle=handle, round_idx=0
    )
    fully_wired_loop.handle_decode_result(
        source=_VALID_SOURCE,
        fitness=SERL_FITNESS_GATE - 0.1,
        thdse_handle=handle,
        round_idx=1,
    )
    assert fully_wired_loop.handler_invocations == 2


def test_feedback_skill_performance_routes_through_bridge(
    manager: ArenaManager, fully_wired_loop: SERLLoop
):
    feedback = fully_wired_loop.feedback_skill_performance(
        skill_id="serl_skill_42",
        performance_scores=[0.8, 0.6, 0.7],
    )
    assert feedback["feedback_applied"] is True
    assert feedback["sample_count"] == 3
    assert feedback["mean_performance"] == pytest.approx(
        (0.8 + 0.6 + 0.7) / 3.0, abs=1e-6
    )
    assert feedback["metadata"]["provenance"]["operation"] == (
        "rsi_skill_to_serl_feedback"
    )


def test_feedback_no_op_when_no_rsi_bridge_wired():
    loop = SERLLoop()
    feedback = loop.feedback_skill_performance("test", [0.5])
    assert feedback["feedback_applied"] is False
    assert feedback["metadata"]["reason"] == "no_rsi_serl_bridge_wired"


def test_unwired_governance_skips_registration_silently(
    manager: ArenaManager,
):
    """Without a governance bridge, the SAT path emits provenance but
    does not attempt skill registration even if RSI accepted it."""
    loop = SERLLoop(
        arena_manager=manager,
        rsi_serl_bridge=RsiSerlBridge(manager),
        provenance_bridge=CausalProvenanceBridge(manager),
    )
    handle = _make_thdse_handle(manager)
    result = loop.handle_decode_result(
        source=_VALID_SOURCE,
        fitness=SERL_FITNESS_GATE + 0.2,
        thdse_handle=handle,
        round_idx=0,
    )
    assert result["eligible"] is True
    assert result["approved"] is False
    assert result["registered"] is False
