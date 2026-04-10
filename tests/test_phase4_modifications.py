"""Tests for PLAN.md Phase 4 modifications to existing CCE files.

This is the CRITICAL file that verifies the seven hand-modified files
plus the deprecated ``goal_corpus_selector.py`` still behave correctly
after the Phase 4 rewiring. It exercises the rewired public surfaces:

- ``fhrr.py``: ``get_arena_manager`` raise-until-wired, ``FhrrVector``
  API works end-to-end once the orchestrator wires an ArenaManager,
  no ``hdc_core.FhrrArena(`` call survives in the source.
- ``skills.py``: ``register()`` governance gate (Rule 7) with strict
  ``is not True`` identity check; ``add()`` still works.
- ``memory.py``: ``accept_thdse_provenance`` stores a new memory item
  with ``kind="thdse_provenance"``; the ``HyperVector.zero()`` bug is
  fixed (empty-text encoding now returns a real vector).
- ``agent.py``: injecting an rng makes action selection reproducible.
- ``orchestrator.py``: ``arena_manager`` / ``deterministic_rng``
  properties exist and return the right types.
- ``causal_chain.py``: new ``record_thdse_provenance`` and
  ``record_unsat_event`` methods append the right event shapes.
- ``goal_generator.py``: ``generate_from_thdse_synthesis`` returns a
  valid :class:`TaskSpec`.
- ``goal_corpus_selector.py``: import triggers DeprecationWarning.
"""

from __future__ import annotations

import os
import sys
import types
import warnings
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — put shared/, bridges/, and the CCE package on sys.path,
# and stub out the missing cognitive_core_engine.core.hdc submodule so
# the package __init__ can import cleanly.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CCE_ROOT = _REPO_ROOT / "Cognitive-Core-Engine-Test"
for p in (str(_REPO_ROOT), str(_CCE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Pre-stub the missing HDC submodule so `cognitive_core_engine.core`
# package import does not crash during collection.
if "cognitive_core_engine.core.hdc" not in sys.modules:
    _hdc_stub = types.ModuleType("cognitive_core_engine.core.hdc")

    class _StubHyperVector:  # noqa: D401 — legacy sentinel
        DIM = 10_000

    _hdc_stub.HyperVector = _StubHyperVector
    sys.modules["cognitive_core_engine.core.hdc"] = _hdc_stub


# Imports AFTER the stub + path setup so they resolve cleanly.
from shared.arena_manager import ArenaManager  # noqa: E402
from shared.deterministic_rng import DeterministicRNG  # noqa: E402
from shared.exceptions import GovernanceError  # noqa: E402

from cognitive_core_engine.core import fhrr as fhrr_module  # noqa: E402
from cognitive_core_engine.core.causal_chain import (  # noqa: E402
    CausalChainTracker,
)
from cognitive_core_engine.core.memory import SharedMemory  # noqa: E402
from cognitive_core_engine.core.skills import (  # noqa: E402
    Skill,
    SkillLibrary,
    SkillStep,
)

from agi_modules.goal_generator import GoalGenerator, TaskSpec  # noqa: E402


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def fresh_arena_manager() -> ArenaManager:
    """Construct a brand-new ArenaManager and unwire any prior global.

    We deliberately reset ``fhrr_module._ARENA_MANAGER`` on each test so
    the ``get_arena_manager() raises before set_arena_manager()`` case
    can be exercised in isolation.
    """
    fhrr_module._ARENA_MANAGER = None  # type: ignore[attr-defined]
    return ArenaManager(master_seed=42)


@pytest.fixture
def wired_arena_manager(fresh_arena_manager: ArenaManager) -> ArenaManager:
    fhrr_module.set_arena_manager(fresh_arena_manager)
    return fresh_arena_manager


# --------------------------------------------------------------------------- #
# fhrr.py
# --------------------------------------------------------------------------- #


def test_fhrr_get_arena_manager_raises_before_wiring(
    fresh_arena_manager: ArenaManager,
):
    # The fixture cleared the global, so the getter must raise.
    with pytest.raises(RuntimeError, match="ArenaManager not set"):
        fhrr_module.get_arena_manager()


def test_fhrr_from_seed_returns_correct_dim(wired_arena_manager: ArenaManager):
    vec = fhrr_module.FhrrVector.from_seed("apple")
    assert fhrr_module.FhrrVector.DIM == 10_000
    # Validate the handle is alive and references a 10k-phase row.
    phases = wired_arena_manager.get_cce_phases(vec.handle)
    assert phases.shape == (10_000,)


def test_fhrr_bind_and_bundle_still_work(wired_arena_manager: ArenaManager):
    a = fhrr_module.FhrrVector.from_seed("alpha")
    b = fhrr_module.FhrrVector.from_seed("beta")
    bound = a.bind(b)
    bundled = fhrr_module.FhrrVector.bundle([a, b])
    # Every operation must return a valid FhrrVector with a live handle.
    assert isinstance(bound, fhrr_module.FhrrVector)
    assert isinstance(bundled, fhrr_module.FhrrVector)
    assert wired_arena_manager.get_cce_phases(bound.handle).shape == (10_000,)
    assert wired_arena_manager.get_cce_phases(bundled.handle).shape == (10_000,)


def test_fhrr_source_contains_no_direct_fhrr_arena_call():
    src = (_CCE_ROOT / "cognitive_core_engine/core/fhrr.py").read_text()
    # The docstring / comments may legitimately mention the old
    # ``hdc_core.FhrrArena`` name for historical context; what must
    # not survive is a CALL site. Strip comments/docstrings by
    # ignoring any line beginning with ``#`` or contained in triple
    # quotes, then verify no executable line references the name.
    import re

    code_no_docstrings = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
    code_no_comments = "\n".join(
        line
        for line in code_no_docstrings.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "hdc_core.FhrrArena(" not in code_no_comments
    assert "import hdc_core" not in code_no_comments
    # Plain ``FhrrArena(`` constructor calls also forbidden — the
    # module must route everything through ArenaManager.
    assert "FhrrArena(" not in code_no_comments


def test_fhrr_set_arena_manager_rejects_wrong_type(
    fresh_arena_manager: ArenaManager,
):
    with pytest.raises(TypeError):
        fhrr_module.set_arena_manager("not a manager")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# skills.py — Rule 7 governance gate
# --------------------------------------------------------------------------- #


def _make_skill(name: str = "test_skill") -> Skill:
    return Skill(
        name=name,
        purpose="unit test skill",
        steps=[SkillStep(kind="call", tool="noop_tool")],
        tags=["test"],
    )


def test_skills_register_without_governance_raises():
    lib = SkillLibrary()
    with pytest.raises(GovernanceError):
        lib.register(_make_skill("no_gov"))


def test_skills_register_with_governance_true_succeeds():
    lib = SkillLibrary()
    sid = lib.register(_make_skill("approved_skill"), governance_approved=True)
    assert isinstance(sid, str)
    retrieved = lib.get(sid)
    assert retrieved.name == "approved_skill"
    assert retrieved.id == sid


def test_skills_register_with_governance_false_raises():
    lib = SkillLibrary()
    with pytest.raises(GovernanceError):
        lib.register(_make_skill("rejected"), governance_approved=False)


def test_skills_register_with_truthy_non_true_raises():
    lib = SkillLibrary()
    with pytest.raises(GovernanceError):
        lib.register(
            _make_skill("truthy_yes"),
            governance_approved="yes",  # type: ignore[arg-type]
        )
    with pytest.raises(GovernanceError):
        lib.register(
            _make_skill("truthy_one"),
            governance_approved=1,  # type: ignore[arg-type]
        )


def test_skills_add_bypasses_governance():
    lib = SkillLibrary()
    sid = lib.add(_make_skill("internal_path"))
    retrieved = lib.get(sid)
    assert retrieved.name == "internal_path"
    assert len(lib.list()) == 1


# --------------------------------------------------------------------------- #
# memory.py — bug fix + THDSE provenance ingestion
# --------------------------------------------------------------------------- #


def test_memory_hypervector_bug_is_fixed(wired_arena_manager: ArenaManager):
    mem = SharedMemory()
    # Empty text used to reference HyperVector.zero() (undefined); the
    # fix routes through FhrrVector.zero(). Call the private helper
    # and assert we got back a FhrrVector of the right dimension —
    # no NameError and a live handle backed by the wired arena.
    vec = mem._encode_text_bag("")  # noqa: SLF001 — intentional bug check
    assert vec.DIM == 10_000
    phases = wired_arena_manager.get_cce_phases(vec.handle)
    assert phases.shape == (10_000,)


def test_memory_accept_thdse_provenance_stores_item(
    wired_arena_manager: ArenaManager,
):
    mem = SharedMemory()
    event = {
        "source_arena": "thdse",
        "operation": "z3_sat",
        "result_similarity": 0.77,
        "timestamp": 1234567.0,
    }
    item_id = mem.accept_thdse_provenance(event)
    assert isinstance(item_id, str) and len(item_id) > 0
    # Walk the memory items looking for the stored provenance record.
    stored = [it for it in mem._items if it.id == item_id]  # noqa: SLF001
    assert len(stored) == 1
    assert stored[0].kind == "thdse_provenance"
    assert stored[0].content["source_arena"] == "thdse"
    assert (
        stored[0].content["metadata"]["provenance"]["operation"]
        == "accept_thdse_provenance"
    )


def test_memory_accept_thdse_provenance_rejects_non_dict(
    wired_arena_manager: ArenaManager,
):
    mem = SharedMemory()
    with pytest.raises(TypeError):
        mem.accept_thdse_provenance("not a dict")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# agent.py — deterministic rng wiring
# --------------------------------------------------------------------------- #


def test_agent_rand_unit_is_reproducible_with_injected_rng(
    wired_arena_manager: ArenaManager,
):
    # Import here because Agent pulls in a lot of CCE modules; we
    # want the sys.modules stub in place first.
    from cognitive_core_engine.core.agent import Agent, AgentConfig
    from cognitive_core_engine.core.tools import ToolRegistry

    def _build() -> Agent:
        return Agent(
            cfg=AgentConfig(name="agent_a", role="theorist"),
            tools=ToolRegistry(),
            shared_mem=SharedMemory(),
            skills=SkillLibrary(),
            rng=DeterministicRNG(master_seed=7).fork("agent_test"),
        )

    a = _build()
    b = _build()
    seq_a = [a._rand_unit() for _ in range(6)]  # noqa: SLF001
    seq_b = [b._rand_unit() for _ in range(6)]  # noqa: SLF001
    assert seq_a == seq_b


def test_agent_rand_choice_uses_injected_rng(wired_arena_manager: ArenaManager):
    from cognitive_core_engine.core.agent import Agent, AgentConfig
    from cognitive_core_engine.core.tools import ToolRegistry

    def _build() -> Agent:
        return Agent(
            cfg=AgentConfig(name="agent_b", role="builder"),
            tools=ToolRegistry(),
            shared_mem=SharedMemory(),
            skills=SkillLibrary(),
            rng=DeterministicRNG(master_seed=11).fork("agent_choice"),
        )

    options = ["alpha", "beta", "gamma", "delta"]
    picks_a = [_build()._rand_choice(options) for _ in range(3)]  # noqa: SLF001
    picks_b = [_build()._rand_choice(options) for _ in range(3)]  # noqa: SLF001
    assert picks_a == picks_b
    for pick in picks_a:
        assert pick in options


# --------------------------------------------------------------------------- #
# orchestrator.py — properties expose the wired infrastructure
# --------------------------------------------------------------------------- #


def test_orchestrator_has_arena_manager_and_drng_properties():
    # The Orchestrator pulls in the entire CCE agent stack. Build it
    # with the default ``ResearchEnvironment(seed=0)`` (its own
    # constructor populates the task list) and the smallest possible
    # config so the properties added by Phase 4 can be asserted.
    from cognitive_core_engine.core.environment import ResearchEnvironment
    from cognitive_core_engine.core.orchestrator import (
        Orchestrator,
        OrchestratorConfig,
    )
    from cognitive_core_engine.core.tools import ToolRegistry

    env = ResearchEnvironment(seed=0)
    tools = ToolRegistry()
    orch = Orchestrator(
        OrchestratorConfig(agents=1), env=env, tools=tools
    )
    assert isinstance(orch.arena_manager, ArenaManager)
    assert isinstance(orch.deterministic_rng, DeterministicRNG)
    # The fhrr module was wired to the same manager during construction.
    assert fhrr_module.get_arena_manager() is orch.arena_manager


# --------------------------------------------------------------------------- #
# causal_chain.py — THDSE ingestion + Rule 8 UNSAT flag
# --------------------------------------------------------------------------- #


def test_causal_chain_record_thdse_provenance_creates_event():
    tracker = CausalChainTracker()
    eid = tracker.record_thdse_provenance(
        source_arena="thdse",
        operation="z3_sat",
        result="PROVED",
        round_idx=4,
    )
    chain = tracker._events  # noqa: SLF001
    assert len(chain) == 1
    ev = chain[0]
    assert ev.event_id == eid
    assert ev.event_type == "thdse_provenance"
    assert ev.data["source_arena"] == "thdse"
    assert ev.data["metadata"]["provenance"]["target_arena"] == "cce"


def test_causal_chain_record_unsat_event_flags_logged():
    tracker = CausalChainTracker()
    eid = tracker.record_unsat_event(
        formula_id="phi_42", reason="contradiction", round_idx=7
    )
    ev = tracker._events_by_id[eid]  # noqa: SLF001
    assert ev.event_type == "synthesis_unsat"
    # Rule 8: every UNSAT event must carry logged=True in its data.
    assert ev.data["logged"] is True
    assert ev.data["formula_id"] == "phi_42"
    assert ev.data["reason"] == "contradiction"


# --------------------------------------------------------------------------- #
# goal_generator.py — THDSE synthesis → TaskSpec
# --------------------------------------------------------------------------- #


def test_goal_generator_generate_from_thdse_synthesis_returns_taskspec():
    import random as py_random

    from agi_modules.competence_map import CompetenceMap

    gg = GoalGenerator(CompetenceMap(), SharedMemory(), py_random.Random(42))
    task = gg.generate_from_thdse_synthesis(
        {
            "axiom_name": "commutativity",
            "confidence": 0.85,
            "domain": "algebra",
            "difficulty": 6,
        }
    )
    assert isinstance(task, TaskSpec)
    assert task.domain == "algebra"
    assert task.difficulty == 6
    # Baseline should be calibrated from confidence (0.85 * 0.8 = 0.68).
    assert task.baseline == pytest.approx(0.68, abs=1e-6)
    assert "commutativity" in task.name


def test_goal_generator_generate_from_thdse_rejects_missing_keys():
    import random as py_random

    from agi_modules.competence_map import CompetenceMap

    gg = GoalGenerator(CompetenceMap(), SharedMemory(), py_random.Random(3))
    with pytest.raises(KeyError):
        gg.generate_from_thdse_synthesis(
            {"axiom_name": "foo", "confidence": 0.5}
        )


# --------------------------------------------------------------------------- #
# goal_corpus_selector.py — deprecation warning
# --------------------------------------------------------------------------- #


def test_goal_corpus_selector_import_triggers_deprecation_warning():
    # Ensure the module is freshly loaded so the warning fires every time.
    removed = []
    for name in list(sys.modules.keys()):
        if name == "bridge.goal_corpus_selector" or name.startswith(
            "bridge.goal_corpus_selector."
        ):
            removed.append(name)
            sys.modules.pop(name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            # The deprecated bridge imports ``hdc_core`` at module
            # scope; that package is absent in the test environment,
            # so the import itself may raise ImportError. We only
            # care that the DeprecationWarning was emitted BEFORE the
            # crash, so we swallow ImportError here.
            import importlib

            try:
                importlib.import_module("bridge.goal_corpus_selector")
            except ImportError:
                pass
        finally:
            # Clean up whatever we loaded so subsequent tests see
            # the same sys.modules state.
            for name in list(sys.modules.keys()):
                if (
                    name == "bridge.goal_corpus_selector"
                    or name.startswith("bridge.goal_corpus_selector.")
                ):
                    sys.modules.pop(name, None)

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) >= 1
    assert "goal_corpus_selector.py is deprecated" in str(
        deprecation_warnings[0].message
    )
