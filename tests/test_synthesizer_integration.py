"""Phase 6 — Tier-1 wiring tests for the THDSE Axiomatic Synthesizer.

These tests exercise the REAL :class:`AxiomaticSynthesizer` class
against REAL :class:`shared.arena_manager.ArenaManager`,
:class:`shared.deterministic_rng.FrozenRNG`, and
:class:`bridges.causal_provenance_bridge.CausalProvenanceBridge`
instances. They never mock the synthesizer, the bridges, or the
arena. They DO skip the actual Z3 solve because:

1. PLAN.md Phase 6 splits the synthesizer into testable handlers
   (``handle_z3_result``, ``attempt_perturbation``) so the
   pre/post-Z3 wiring can be exercised independently.
2. Real Z3 is exercised by the Tier-2 file
   (``tests/test_synthesizer_z3_integration.py``), which uses
   ``pytest.importorskip("z3")``.

PLAN.md Rule 13 (revised): Tier-1 must run with bare Python + numpy.
PLAN.md Rule 14: every injected dependency must be USED in a real
code path. The tests below confirm that:

- ``arena_manager`` injection actually swaps the synthesizer's arena.
- ``frozen_rng`` injection makes ``attempt_perturbation`` raise.
- ``provenance_bridge`` injection makes ``handle_z3_result("sat")``
  emit an event the bridge counter can see.
- ``causal_tracker`` injection makes ``handle_z3_result("unsat")``
  append a ``record_unsat_event`` entry with ``logged=True``.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_THDSE = _REPO_ROOT / "thdse"
_CCE = _REPO_ROOT / "Cognitive-Core-Engine-Test"
for p in (str(_REPO_ROOT), str(_THDSE), str(_CCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Stub the legacy CCE hdc submodule the package init still references.
if "cognitive_core_engine.core.hdc" not in sys.modules:
    _stub = types.ModuleType("cognitive_core_engine.core.hdc")

    class _StubHV:
        DIM = 10_000

    _stub.HyperVector = _StubHV
    sys.modules["cognitive_core_engine.core.hdc"] = _stub


from shared.arena_manager import ArenaManager  # noqa: E402
from shared.deterministic_rng import DeterministicRNG, FrozenRNG  # noqa: E402

from bridges.causal_provenance_bridge import CausalProvenanceBridge  # noqa: E402

from cognitive_core_engine.core.causal_chain import (  # noqa: E402
    CausalChainTracker,
)

from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer  # noqa: E402
from src.utils.arena_factory import _PyFhrrArenaExtended  # noqa: E402


@pytest.fixture
def manager() -> ArenaManager:
    return ArenaManager(master_seed=6001)


@pytest.fixture
def provenance_bridge(manager: ArenaManager) -> CausalProvenanceBridge:
    return CausalProvenanceBridge(manager)


@pytest.fixture
def causal_tracker() -> CausalChainTracker:
    return CausalChainTracker()


# --------------------------------------------------------------------------- #
# Constructor injection
# --------------------------------------------------------------------------- #


def test_synthesizer_legacy_constructor_still_works():
    """Rule 16: legacy callers that don't pass arena_manager keep working."""
    arena = _PyFhrrArenaExtended(capacity=64, dimension=32)
    synth = AxiomaticSynthesizer(arena=arena, projector=None)
    assert synth.arena is arena
    assert synth._arena_manager is None  # noqa: SLF001
    assert synth._provenance_bridge is None  # noqa: SLF001


def test_synthesizer_arena_manager_injection_borrows_arena(
    manager: ArenaManager,
):
    """Rule 14: arena_manager injection actually replaces the private arena."""
    synth = AxiomaticSynthesizer(arena_manager=manager, projector=None)
    assert synth._arena_manager is manager  # noqa: SLF001
    # The injected arena must expose the standard FHRR API surface so
    # downstream synthesizer code can call ``allocate``, ``bind``, …
    assert hasattr(synth.arena, "allocate")
    assert hasattr(synth.arena, "bind")
    assert callable(synth.arena.allocate)
    if manager.backend == "rust":
        assert synth.arena is manager._thdse_arena  # noqa: SLF001


def test_synthesizer_default_init_constructs_python_fallback():
    """When neither arena nor arena_manager is provided, fall back."""
    synth = AxiomaticSynthesizer(projector=None)
    assert isinstance(synth.arena, _PyFhrrArenaExtended)


# --------------------------------------------------------------------------- #
# FrozenRNG enforcement (Rule 14)
# --------------------------------------------------------------------------- #


def test_attempt_perturbation_raises_under_frozen_rng(
    manager: ArenaManager,
):
    synth = AxiomaticSynthesizer(
        arena_manager=manager,
        projector=None,
        frozen_rng=FrozenRNG(tag="thdse_synth"),
    )
    with pytest.raises(RuntimeError, match="FrozenRNG"):
        synth.attempt_perturbation(magnitude=0.01)


def test_attempt_perturbation_returns_zero_when_no_rng_injected(
    manager: ArenaManager,
):
    synth = AxiomaticSynthesizer(arena_manager=manager, projector=None)
    assert synth.attempt_perturbation(0.5) == 0.0


def test_attempt_perturbation_uses_seeded_rng_when_provided(
    manager: ArenaManager,
):
    drng_fork = DeterministicRNG(master_seed=42).fork("thdse_perturb")
    synth = AxiomaticSynthesizer(
        arena_manager=manager,
        projector=None,
        frozen_rng=drng_fork,
    )
    value_a = synth.attempt_perturbation(0.5)
    assert 0.0 <= value_a < 0.5
    # A second call advances the rng state — the value should differ.
    value_b = synth.attempt_perturbation(0.5)
    assert value_a != value_b


# --------------------------------------------------------------------------- #
# handle_z3_result wiring (Rule 8 + Rule 9 + Rule 14)
# --------------------------------------------------------------------------- #


def test_handle_sat_emits_provenance_event(
    manager: ArenaManager, provenance_bridge: CausalProvenanceBridge
):
    synth = AxiomaticSynthesizer(
        arena_manager=manager,
        projector=None,
        provenance_bridge=provenance_bridge,
    )
    result = synth.handle_z3_result(
        "sat", formula_id="phi_sat_1", round_idx=0
    )
    assert result["result"] == "sat"
    # Bridge mints an event id with a known prefix when wired.
    assert isinstance(result["provenance_event_id"], str)
    assert result["provenance_event_id"].startswith("cpb-")
    assert result["metadata"]["provenance"]["operation"] == "handle_z3_result"
    assert provenance_bridge.get_sat_count() == 1


def test_handle_unsat_double_logs_to_bridge_and_tracker(
    manager: ArenaManager,
    provenance_bridge: CausalProvenanceBridge,
    causal_tracker: CausalChainTracker,
):
    synth = AxiomaticSynthesizer(
        arena_manager=manager,
        projector=None,
        provenance_bridge=provenance_bridge,
        causal_tracker=causal_tracker,
    )
    result = synth.handle_z3_result(
        "unsat",
        formula_id="phi_unsat_1",
        round_idx=2,
        details={"reason": "contradiction"},
    )
    # Bridge counter incremented.
    assert provenance_bridge.get_unsat_count() == 1
    # Causal tracker entry exists with logged=True (Rule 8).
    assert isinstance(result["causal_event_id"], str)
    assert result["causal_event_id"].startswith("us_")
    ev = causal_tracker._events_by_id[result["causal_event_id"]]  # noqa: SLF001
    assert ev.event_type == "synthesis_unsat"
    assert ev.data["logged"] is True
    assert ev.data["formula_id"] == "phi_unsat_1"


def test_handle_z3_result_validates_input():
    synth = AxiomaticSynthesizer(projector=None)
    with pytest.raises(ValueError, match="must be 'sat' or 'unsat'"):
        synth.handle_z3_result("maybe", formula_id="x", round_idx=0)


def test_handle_z3_result_returns_metadata_provenance_when_no_bridges():
    """Rule 9: even without wired bridges the return dict carries provenance."""
    synth = AxiomaticSynthesizer(projector=None)
    result = synth.handle_z3_result(
        "sat", formula_id="standalone", round_idx=1
    )
    assert "metadata" in result
    assert result["metadata"]["provenance"]["operation"] == "handle_z3_result"
    assert result["provenance_event_id"] is None
    assert result["causal_event_id"] is None


def test_handle_unsat_without_causal_tracker_still_emits_provenance(
    manager: ArenaManager, provenance_bridge: CausalProvenanceBridge
):
    """Bridge wiring is independent — emit even without tracker."""
    synth = AxiomaticSynthesizer(
        arena_manager=manager,
        projector=None,
        provenance_bridge=provenance_bridge,
    )
    synth.handle_z3_result(
        "unsat", formula_id="phi_lonely", round_idx=0
    )
    assert provenance_bridge.get_unsat_count() == 1


def test_handle_sat_then_unsat_keeps_per_type_counts(
    manager: ArenaManager, provenance_bridge: CausalProvenanceBridge
):
    synth = AxiomaticSynthesizer(
        arena_manager=manager,
        projector=None,
        provenance_bridge=provenance_bridge,
    )
    for i in range(3):
        synth.handle_z3_result("sat", f"phi_s_{i}", round_idx=i)
    for i in range(2):
        synth.handle_z3_result("unsat", f"phi_u_{i}", round_idx=i)
    assert provenance_bridge.get_sat_count() == 3
    assert provenance_bridge.get_unsat_count() == 2
    assert provenance_bridge.total_events() == 5
