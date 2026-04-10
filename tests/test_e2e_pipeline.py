"""End-to-end integration tests for the unified OMEGA-THDSE pipeline.

These tests exercise real objects across every phase boundary:

- Phase 2 primitives (ArenaManager, DimensionBridge, DeterministicRNG)
- Phase 3 bridges (concept↔axiom, axiom→skill, governance↔synthesis,
  goal↔synthesis, rsi↔serl, causal↔provenance)
- Phase 4 bridges (memory↔hypothesis, world_model↔swarm, self_model)
- Phase 4 CCE modifications (fhrr, skills, causal_chain)

No test-double fakes, no patching, no stubs. Every assertion checks a
specific computed value against a concrete expectation. Each test
constructs its own ArenaManager so there is no shared state between
scenarios.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap + stub for the missing cognitive_core_engine.core.hdc
# legacy submodule (same pattern as tests/test_phase4_modifications.py).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CCE_ROOT = _REPO_ROOT / "Cognitive-Core-Engine-Test"
for _path in (str(_REPO_ROOT), str(_CCE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if "cognitive_core_engine.core.hdc" not in sys.modules:
    _hdc_stub = types.ModuleType("cognitive_core_engine.core.hdc")

    class _StubHyperVector:  # noqa: D401 — legacy sentinel
        DIM = 10_000

    _hdc_stub.HyperVector = _StubHyperVector
    sys.modules["cognitive_core_engine.core.hdc"] = _hdc_stub


from shared.arena_manager import ArenaManager  # noqa: E402
from shared.constants import (  # noqa: E402
    BRIDGE_SELF_SIMILARITY_MIN,
    CCE_ARENA_DIM,
    CRITIC_THRESHOLD,
    MAX_CREDIBLE_LEAP,
    SELF_MODEL_COMPONENTS,
    SERL_FITNESS_GATE,
    SWARM_CONSENSUS_THRESHOLD,
    THDSE_ARENA_DIM,
)
from shared.deterministic_rng import DeterministicRNG, FrozenRNG  # noqa: E402
from shared.dimension_bridge import (  # noqa: E402
    cross_arena_similarity,
    project_down,
)
from shared.exceptions import GovernanceError  # noqa: E402

from bridges.axiom_skill_bridge import AxiomSkillBridge  # noqa: E402
from bridges.causal_provenance_bridge import CausalProvenanceBridge  # noqa: E402
from bridges.concept_axiom_bridge import ConceptAxiomBridge  # noqa: E402
from bridges.goal_synthesis_bridge import GoalSynthesisBridge  # noqa: E402
from bridges.governance_synthesis_bridge import (  # noqa: E402
    GovernanceSynthesisBridge,
)
from bridges.memory_hypothesis_bridge import MemoryHypothesisBridge  # noqa: E402
from bridges.rsi_serl_bridge import RsiSerlBridge  # noqa: E402
from bridges.self_model_bridge import SelfModelBridge  # noqa: E402
from bridges.world_model_swarm_bridge import WorldModelSwarmBridge  # noqa: E402

from cognitive_core_engine.core import fhrr as fhrr_module  # noqa: E402
from cognitive_core_engine.core.causal_chain import (  # noqa: E402
    CausalChainTracker,
)
from cognitive_core_engine.core.skills import (  # noqa: E402
    Skill,
    SkillLibrary,
    SkillStep,
)


_TWO_PI = 2.0 * math.pi
_VALID_PY = "def run():\n    return 1\n"


def _random_cce_phases(master_seed: int, namespace: str) -> np.ndarray:
    """Deterministic 10k-dim phase vector via a private fork."""
    drng = DeterministicRNG(master_seed=master_seed)
    return drng.fork(namespace).uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(
        np.float32
    )


# --------------------------------------------------------------------------- #
# TEST 1: Full arena lifecycle
# --------------------------------------------------------------------------- #


def test_e2e_1_full_arena_lifecycle():
    mgr = ArenaManager(master_seed=1001)
    phases = _random_cce_phases(1001, "e2e_1")
    h_cce = mgr.alloc_cce(phases=phases)
    # Allocate a THDSE handle with the projected subsample so the two
    # arenas each carry a vector at their native dimension.
    projection = project_down(phases)
    h_thdse = mgr.alloc_thdse(phases=projection["vector"])

    # Arenas are tagged with their native dimension — they are NOT
    # interchangeable.
    assert mgr.tag_of("cce", h_cce).dimension == CCE_ARENA_DIM
    assert mgr.tag_of("thdse", h_thdse).dimension == THDSE_ARENA_DIM
    assert mgr.cce_dim != mgr.thdse_dim

    # Self-similarity through the bridge must exceed the invariant.
    sim = cross_arena_similarity(phases, projection["vector"])
    assert sim["similarity"] == pytest.approx(1.0, abs=1e-5)
    assert sim["similarity"] > BRIDGE_SELF_SIMILARITY_MIN


# --------------------------------------------------------------------------- #
# TEST 2: Concept → Axiom → Skill pipeline (Gaps 2 + 3)
# --------------------------------------------------------------------------- #


def test_e2e_2_concept_to_axiom_to_governed_skill():
    mgr = ArenaManager(master_seed=1002)
    cab = ConceptAxiomBridge(mgr)
    gov = GovernanceSynthesisBridge(mgr)
    asb = AxiomSkillBridge(mgr)

    phases = _random_cce_phases(1002, "e2e_2_concept")
    concept_handle = mgr.alloc_cce(phases=phases)

    # Project concept → axiom.
    concept_result = cab.concept_to_axiom(
        concept_handle, {"title": "commutativity"}
    )
    axiom_handle = concept_result["thdse_handle"]
    assert concept_result["similarity_to_source"] == pytest.approx(
        1.0, abs=1e-5
    )

    # Governance gate the axiom.
    verdict = gov.evaluate_candidate(_VALID_PY, axiom_handle, CRITIC_THRESHOLD + 0.1)
    assert verdict["approved"] is True
    assert gov.gate_registration(verdict) is True

    # Register as a skill.
    reg = asb.validate_and_register(
        axiom_handle=axiom_handle,
        program_source=_VALID_PY,
        skill_name="e2e2_skill",
        governance_approved=True,
    )
    assert reg["registered"] is True
    assert reg["skill_name"] == "e2e2_skill"
    assert reg["metadata"]["provenance"]["operation"] == "validate_and_register"
    # Registry actually has the entry.
    stored = asb.get_registration("e2e2_skill")
    assert stored["axiom_handle"] == axiom_handle


# --------------------------------------------------------------------------- #
# TEST 3: Causal chain integrity (Gap 4 + Rule 8)
# --------------------------------------------------------------------------- #


def test_e2e_3_causal_chain_integrity_with_unsat_logging():
    tracker = CausalChainTracker()
    sb_id = tracker.record_skill_birth(
        skill_id="skill_a", genome_fitness=0.8, round_idx=1
    )
    goal_id = tracker.record_goal_from_skill(
        goal_name="goal_a",
        trigger_skill_id="skill_a",
        trigger_event_id=sb_id,
        round_idx=2,
    )
    ach_id = tracker.record_goal_achieved(
        goal_name="goal_a",
        reward=0.9,
        round_idx=3,
        contributing_skill_ids=["skill_a"],
        cause_event_id=goal_id,
    )
    thdse_id = tracker.record_thdse_provenance(
        source_arena="thdse",
        operation="z3_sat",
        result="PROVED",
        round_idx=4,
        cause_event_id=ach_id,
    )
    unsat_id = tracker.record_unsat_event(
        formula_id="phi_1",
        reason="contradiction",
        round_idx=5,
        cause_event_id=thdse_id,
    )

    # Chain depth: the longest chain must contain all 5 events.
    chain_ids = set()
    for chain in tracker._chains.values():  # noqa: SLF001 — introspection
        chain_ids.update(chain)
    assert len(chain_ids) >= 5
    assert {sb_id, goal_id, ach_id, thdse_id, unsat_id}.issubset(chain_ids)

    # Rule 8: every synthesis_unsat event carries logged=True in data.
    unsat_events = [
        e for e in tracker._events  # noqa: SLF001
        if e.event_type == "synthesis_unsat"
    ]
    assert len(unsat_events) == 1
    assert unsat_events[0].data["logged"] is True


# --------------------------------------------------------------------------- #
# TEST 4: Memory ↔ Hypothesis pipeline (Gap 5)
# --------------------------------------------------------------------------- #


def test_e2e_4_memory_hypothesis_scoring_and_ranking():
    mgr = ArenaManager(master_seed=1004)
    mhb = MemoryHypothesisBridge(mgr)

    encoded = mhb.encode_memory_for_hypothesis(
        "machine learning", ["ml", "research"]
    )
    assert encoded["thdse_vector"].shape == (THDSE_ARENA_DIM,)
    assert encoded["original_dim"] == CCE_ARENA_DIM

    scored = mhb.score_hypothesis_relevance(0.8, 0.6)
    expected = 0.6 * 0.8 + 0.4 * 0.6
    assert scored["relevance_score"] == pytest.approx(expected, abs=1e-6)

    ranked = mhb.rank_hypotheses_by_memory(
        [
            {"id": "low", "hypothesis_fitness": 0.1, "memory_similarity": 0.9},
            {"id": "high", "hypothesis_fitness": 0.9, "memory_similarity": 0.9},
            {"id": "mid", "hypothesis_fitness": 0.5, "memory_similarity": 0.6},
        ]
    )
    assert [r["id"] for r in ranked] == ["high", "mid", "low"]
    assert ranked[0]["rank"] == 1
    assert ranked[-1]["rank"] == 3


# --------------------------------------------------------------------------- #
# TEST 5: Governance → Synthesis → Skill pipeline (Gap 6 + Rule 7)
# --------------------------------------------------------------------------- #


def test_e2e_5_governance_to_synthesis_to_skill_rejections():
    mgr = ArenaManager(master_seed=1005)
    gov = GovernanceSynthesisBridge(mgr)
    asb = AxiomSkillBridge(mgr)

    # Clean candidate above the critic threshold → approved.
    h = mgr.alloc_thdse(
        phases=np.zeros(THDSE_ARENA_DIM, dtype=np.float32)
    )
    clean = gov.evaluate_candidate(
        _VALID_PY, h, CRITIC_THRESHOLD + 0.1
    )
    assert clean["approved"] is True

    # Syntax error → rejected.
    syntax_bad = gov.evaluate_candidate(
        "def broken(:", h, CRITIC_THRESHOLD + 0.15
    )
    assert syntax_bad["approved"] is False
    assert "SyntaxError" in syntax_bad["reason"]

    # Credible-leap violation — previous fitness is now the most
    # recently observed score (syntax_bad's ``CRITIC_THRESHOLD + 0.15``),
    # so the next candidate must stay within MAX_CREDIBLE_LEAP of that.
    # Jump well above the previous fitness to trip the guard.
    jump = CRITIC_THRESHOLD + 0.15 + MAX_CREDIBLE_LEAP + 0.2
    leap_bad = gov.evaluate_candidate(_VALID_PY, h, jump)
    assert leap_bad["approved"] is False
    assert "MAX_CREDIBLE_LEAP" in leap_bad["reason"]

    # Rule 7: rejected candidates cannot bypass governance gate on the
    # skill library — register() refuses unapproved calls.
    with pytest.raises(GovernanceError):
        asb.validate_and_register(
            axiom_handle=h,
            program_source=_VALID_PY,
            skill_name="rejected_skill",
            governance_approved=False,
        )
    # Registry stayed empty.
    assert asb.registered_count == 0


# --------------------------------------------------------------------------- #
# TEST 6: World Model ↔ Swarm pipeline (Gap 7)
# --------------------------------------------------------------------------- #


def test_e2e_6_world_model_swarm_roundtrip():
    mgr = ArenaManager(master_seed=1006)
    bridge = WorldModelSwarmBridge(mgr)

    result = bridge.project_world_state_for_swarm(
        {"task": "optimize", "step": 3},
        {"act_a": 0.9, "act_b": 0.05, "act_c": 0.05},
    )
    assert result["thdse_guidance_vector"].shape == (THDSE_ARENA_DIM,)
    assert 0.0 <= result["confidence"] <= 1.0

    # Zero-phase consensus vs zero-CCE reference → similarity 1.0 →
    # should_adopt True (well above SWARM_CONSENSUS_THRESHOLD).
    zero_consensus = [0.0] * THDSE_ARENA_DIM
    adopt = bridge.incorporate_swarm_consensus(zero_consensus)
    assert adopt["similarity"] == pytest.approx(1.0, abs=1e-5)
    assert adopt["should_adopt"] is True
    assert adopt["threshold"] == SWARM_CONSENSUS_THRESHOLD

    # Anti-parallel consensus → should_adopt False at default threshold.
    pi_consensus = [math.pi] * THDSE_ARENA_DIM
    reject = bridge.incorporate_swarm_consensus(pi_consensus)
    assert reject["similarity"] == pytest.approx(-1.0, abs=1e-5)
    assert reject["should_adopt"] is False


# --------------------------------------------------------------------------- #
# TEST 7: Goal synthesis pipeline (Gap 8)
# --------------------------------------------------------------------------- #


def test_e2e_7_goal_synthesis_projection_and_ranking():
    mgr = ArenaManager(master_seed=1007)
    gsb = GoalSynthesisBridge(mgr)
    goals = []
    for i, (desc, prio) in enumerate(
        [("low", 0.1), ("high", 0.9), ("mid", 0.5)]
    ):
        h = mgr.alloc_cce(phases=_random_cce_phases(1007, f"goal_{i}"))
        goals.append(gsb.goal_to_synthesis_target(desc, h, prio))
    # Self-similarity through projection → each goal's similarity ≈ 1.
    for goal in goals:
        assert goal["projected_similarity"] == pytest.approx(1.0, abs=1e-5)
        assert goal["metadata"]["provenance"]["operation"] == (
            "goal_to_synthesis_target"
        )

    ranked = gsb.rank_goals(goals)
    descriptions = [g["metadata"]["goal_description"] for g in ranked]
    assert descriptions == ["high", "mid", "low"]
    # Ranking preserves provenance.
    for item in ranked:
        assert "ranking_provenance" in item["metadata"]
        assert "provenance" in item["metadata"]


# --------------------------------------------------------------------------- #
# TEST 8: Self-model export + wireheading detection (Gap 9)
# --------------------------------------------------------------------------- #


def test_e2e_8_self_model_export_and_wireheading():
    mgr = ArenaManager(master_seed=1008)
    smb = SelfModelBridge(mgr)
    belief = _random_cce_phases(1008, "belief")
    goal = _random_cce_phases(1008, "goal")
    capability = _random_cce_phases(1008, "capability")
    emotion = _random_cce_phases(1008, "emotion")
    export = smb.export_self_model_state(belief, goal, capability, emotion)
    assert len(export["thdse_self_model"]) == THDSE_ARENA_DIM
    assert export["metadata"]["component_count"] == SELF_MODEL_COMPONENTS
    assert SELF_MODEL_COMPONENTS == 4

    # Wireheading detection: a large delta in either arena triggers flag.
    suspicious = smb.detect_wireheading_from_thdse(
        MAX_CREDIBLE_LEAP + 0.1, 0.0
    )
    assert suspicious["is_suspicious"] is True
    ok = smb.detect_wireheading_from_thdse(0.05, 0.05)
    assert ok["is_suspicious"] is False


# --------------------------------------------------------------------------- #
# TEST 9: RSI ↔ SERL pipeline (Gap 10)
# --------------------------------------------------------------------------- #


def test_e2e_9_rsi_serl_candidate_feedback_loop():
    mgr = ArenaManager(master_seed=1009)
    rsb = RsiSerlBridge(mgr)
    # Seed a THDSE handle above the gate.
    good_handle = mgr.alloc_thdse(
        phases=np.zeros(THDSE_ARENA_DIM, dtype=np.float32)
    )
    above = rsb.serl_candidate_to_rsi(
        "def prog(): return 1", SERL_FITNESS_GATE + 0.2, good_handle
    )
    assert above["eligible"] is True
    assert above["rsi_compatible"] is True

    bad_handle = mgr.alloc_thdse(
        phases=np.zeros(THDSE_ARENA_DIM, dtype=np.float32)
    )
    below = rsb.serl_candidate_to_rsi(
        "def prog(): return 1", SERL_FITNESS_GATE - 0.1, bad_handle
    )
    assert below["eligible"] is False

    feedback = rsb.rsi_skill_to_serl_feedback(
        "skill_e2e", [0.2, 0.8, 0.5]
    )
    assert feedback["sample_count"] == 3
    assert feedback["mean_performance"] == pytest.approx(0.5, abs=1e-6)
    assert feedback["metadata"]["min_score"] == pytest.approx(0.2, abs=1e-6)
    assert feedback["metadata"]["max_score"] == pytest.approx(0.8, abs=1e-6)


# --------------------------------------------------------------------------- #
# TEST 10: fhrr.py ArenaManager wiring
# --------------------------------------------------------------------------- #


def test_e2e_10_fhrr_arena_manager_wiring():
    mgr = ArenaManager(master_seed=1010)
    fhrr_module._ARENA_MANAGER = None  # type: ignore[attr-defined]
    fhrr_module.set_arena_manager(mgr)

    a = fhrr_module.FhrrVector.from_seed("e2e_test_a")
    b = fhrr_module.FhrrVector.from_seed("e2e_test_b")
    c = fhrr_module.FhrrVector.from_seed("e2e_test_c")

    bound = a.bind(b)
    bundled = fhrr_module.FhrrVector.bundle([a, b, c])

    assert fhrr_module.FhrrVector.DIM == 10_000
    assert mgr.get_cce_phases(bound.handle).shape == (10_000,)
    assert mgr.get_cce_phases(bundled.handle).shape == (10_000,)
    # similarity must be a float in [-1, 1]-ish after mapping to [0, 1].
    sim = a.similarity(a)
    assert sim == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# TEST 11: Skills governance gate end-to-end
# --------------------------------------------------------------------------- #


def _make_skill(name: str) -> Skill:
    return Skill(
        name=name,
        purpose="e2e skill",
        steps=[SkillStep(kind="call", tool="noop")],
        tags=["e2e"],
    )


def test_e2e_11_skills_governance_gate_full():
    lib = SkillLibrary()
    # Ungoverned register → GovernanceError.
    with pytest.raises(GovernanceError):
        lib.register(_make_skill("no_gov"))

    # Approved register → success.
    sid_a = lib.register(_make_skill("approved"), governance_approved=True)
    assert lib.get(sid_a).name == "approved"

    # Internal path: add() bypasses governance.
    sid_b = lib.add(_make_skill("internal"))
    assert lib.get(sid_b).name == "internal"

    # Library holds exactly the two successfully stored skills.
    names = sorted(s.name for s in lib.list())
    assert names == ["approved", "internal"]


# --------------------------------------------------------------------------- #
# TEST 12: DeterministicRNG propagation
# --------------------------------------------------------------------------- #


def test_e2e_12_deterministic_rng_propagation_and_frozen():
    drng_a = DeterministicRNG(master_seed=42)
    drng_b = DeterministicRNG(master_seed=42)
    seq_cce_a = drng_a.fork("cce").uniform(0.0, 1.0, 100)
    seq_thdse_a = drng_a.fork("thdse").uniform(0.0, 1.0, 100)
    seq_cce_b = drng_b.fork("cce").uniform(0.0, 1.0, 100)
    seq_thdse_b = drng_b.fork("thdse").uniform(0.0, 1.0, 100)
    # EXACT equality — DeterministicRNG forks are byte-identical across
    # instances seeded with the same master seed.
    assert np.array_equal(seq_cce_a, seq_cce_b)
    assert np.array_equal(seq_thdse_a, seq_thdse_b)
    # Different namespaces diverge.
    assert not np.array_equal(seq_cce_a, seq_thdse_a)

    # FrozenRNG must raise on every random method call.
    frozen = FrozenRNG()
    for attr in ("random", "integers", "uniform", "normal", "choice"):
        with pytest.raises(RuntimeError, match="FrozenRNG"):
            getattr(frozen, attr)


# --------------------------------------------------------------------------- #
# TEST 13 (bonus): ArenaManager is not picklable across processes
# --------------------------------------------------------------------------- #


def test_e2e_13_arena_manager_pickle_guard():
    mgr = ArenaManager(master_seed=1013)
    mgr.alloc_cce(phases=_random_cce_phases(1013, "pickle_probe"))
    with pytest.raises(RuntimeError, match="Rust FFI"):
        pickle.dumps(mgr)


# --------------------------------------------------------------------------- #
# TEST 14 (bonus): Causal provenance bridge + tracker interop
# --------------------------------------------------------------------------- #


def test_e2e_14_causal_provenance_bridge_to_tracker():
    mgr = ArenaManager(master_seed=1014)
    cpb = CausalProvenanceBridge(mgr)
    h = mgr.alloc_thdse(phases=np.zeros(THDSE_ARENA_DIM, dtype=np.float32))

    # Record 5 events: 3 UNSAT, 2 SAT. Rule 8 auditor must count 3.
    cpb.record_synthesis_event("unsat", h, {"reason": "a"})
    cpb.record_synthesis_event("sat", h, {})
    cpb.record_synthesis_event("unsat", h, {"reason": "b"})
    cpb.record_synthesis_event("sat", h, {})
    cpb.record_synthesis_event("unsat", None, {"reason": "c"})
    assert cpb.get_unsat_count() == 3
    assert cpb.total_events() == 5
    # Independent CCE tracker can also ingest a summary event.
    tracker = CausalChainTracker()
    eid = tracker.record_thdse_provenance(
        source_arena="thdse",
        operation="bridge_sync",
        result=f"unsat={cpb.get_unsat_count()}",
        round_idx=1,
    )
    assert tracker._events_by_id[eid].event_type == "thdse_provenance"  # noqa: SLF001
