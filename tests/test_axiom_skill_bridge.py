"""Tests for :mod:`bridges.axiom_skill_bridge` (PLAN.md Phase 3, Gap 3).

Covers construction, governance gate enforcement, program source
validation, duplicate-name rejection, rejection path bookkeeping,
provenance metadata, skill listing, and registry lookup.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bridges.axiom_skill_bridge import AxiomSkillBridge  # noqa: E402
from shared.arena_manager import ArenaManager  # noqa: E402
from shared.constants import THDSE_ARENA_DIM  # noqa: E402
from shared.exceptions import GovernanceError  # noqa: E402

_TWO_PI = 2.0 * math.pi

_VALID_PROGRAM = "def skill():\n    return 42\n"


def _seed_axiom(mgr: ArenaManager, fork_name: str) -> int:
    rng = mgr.rng.fork(fork_name)
    phases = rng.uniform(0.0, _TWO_PI, THDSE_ARENA_DIM).astype(np.float32)
    return mgr.alloc_thdse(phases=phases)


@pytest.fixture
def mgr() -> ArenaManager:
    return ArenaManager(master_seed=202)


@pytest.fixture
def bridge(mgr: ArenaManager) -> AxiomSkillBridge:
    return AxiomSkillBridge(mgr)


def test_constructor_rejects_non_arena_manager():
    with pytest.raises(TypeError):
        AxiomSkillBridge(arena_manager="not a manager")  # type: ignore[arg-type]


def test_validate_without_governance_raises_governance_error(
    bridge: AxiomSkillBridge, mgr: ArenaManager
):
    h = _seed_axiom(mgr, "a1")
    with pytest.raises(GovernanceError) as exc:
        bridge.validate_and_register(h, _VALID_PROGRAM, "skill_a")
    assert exc.value.subject == "skill_a"
    assert exc.value.reason == "unapproved"
    # Failed attempt must be logged in the rejection history.
    assert bridge.rejection_count == 1
    assert bridge.registered_count == 0


def test_validate_with_governance_registers(
    bridge: AxiomSkillBridge, mgr: ArenaManager
):
    h = _seed_axiom(mgr, "a2")
    result = bridge.validate_and_register(
        h, _VALID_PROGRAM, "skill_b", governance_approved=True
    )
    assert result["registered"] is True
    assert result["skill_name"] == "skill_b"
    assert result["skill_id"].startswith("skill-")
    assert isinstance(result["axiom_similarity"], float)
    assert bridge.registered_count == 1


def test_validate_rejects_empty_program_source(
    bridge: AxiomSkillBridge, mgr: ArenaManager
):
    h = _seed_axiom(mgr, "a3")
    with pytest.raises(ValueError, match="non-empty"):
        bridge.validate_and_register(
            h, "   ", "skill_c", governance_approved=True
        )


def test_validate_rejects_unparseable_program(
    bridge: AxiomSkillBridge, mgr: ArenaManager
):
    h = _seed_axiom(mgr, "a4")
    bad_src = "def broken(:\n    return"
    with pytest.raises(SyntaxError):
        bridge.validate_and_register(
            h, bad_src, "skill_d", governance_approved=True
        )


def test_validate_rejects_duplicate_skill_name(
    bridge: AxiomSkillBridge, mgr: ArenaManager
):
    h1 = _seed_axiom(mgr, "dup1")
    h2 = _seed_axiom(mgr, "dup2")
    bridge.validate_and_register(
        h1, _VALID_PROGRAM, "dup_skill", governance_approved=True
    )
    with pytest.raises(ValueError, match="already registered"):
        bridge.validate_and_register(
            h2, _VALID_PROGRAM, "dup_skill", governance_approved=True
        )


def test_validate_result_has_provenance(
    bridge: AxiomSkillBridge, mgr: ArenaManager
):
    h = _seed_axiom(mgr, "a5")
    result = bridge.validate_and_register(
        h, _VALID_PROGRAM, "prov_skill", governance_approved=True
    )
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "validate_and_register"
    assert prov["source_arena"] == "thdse"
    assert prov["target_arena"] == "cce"
    assert prov["governance_approved"] is True
    assert isinstance(prov["timestamp"], float)


def test_reject_unapproved_records_and_returns_metadata(
    bridge: AxiomSkillBridge, mgr: ArenaManager
):
    h = _seed_axiom(mgr, "a6")
    result = bridge.reject_unapproved(h, "critic vetoed")
    assert result["registered"] is False
    assert result["reason"] == "critic vetoed"
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "reject_unapproved"
    assert bridge.rejection_count == 1


def test_reject_unapproved_requires_non_empty_reason(
    bridge: AxiomSkillBridge, mgr: ArenaManager
):
    h = _seed_axiom(mgr, "a7")
    with pytest.raises(ValueError):
        bridge.reject_unapproved(h, "")


def test_get_registration_returns_stored_entry(
    bridge: AxiomSkillBridge, mgr: ArenaManager
):
    h = _seed_axiom(mgr, "a8")
    bridge.validate_and_register(
        h, _VALID_PROGRAM, "lookup_skill", governance_approved=True
    )
    entry = bridge.get_registration("lookup_skill")
    assert entry["axiom_handle"] == h
    assert entry["program_source"] == _VALID_PROGRAM
    assert entry["skill_name"] == "lookup_skill"


def test_get_registration_missing_name_raises(bridge: AxiomSkillBridge):
    with pytest.raises(KeyError):
        bridge.get_registration("unknown")


def test_list_skills_returns_sorted_names(
    bridge: AxiomSkillBridge, mgr: ArenaManager
):
    for name in ("z_last", "a_first", "m_middle"):
        h = _seed_axiom(mgr, f"ls_{name}")
        bridge.validate_and_register(
            h, _VALID_PROGRAM, name, governance_approved=True
        )
    assert bridge.list_skills() == ["a_first", "m_middle", "z_last"]


def test_governance_flag_must_be_exact_true(
    bridge: AxiomSkillBridge, mgr: ArenaManager
):
    h = _seed_axiom(mgr, "truthy1")
    # Truthy values other than exactly True must still be rejected —
    # Rule 7 requires an explicit boolean True, not "1" or 1.
    with pytest.raises(GovernanceError):
        bridge.validate_and_register(
            h, _VALID_PROGRAM, "truthy_skill", governance_approved=1
        )  # type: ignore[arg-type]
