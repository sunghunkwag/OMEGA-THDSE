"""Tests for :mod:`bridges.rsi_serl_bridge` (PLAN.md Phase 3, Gap 10)."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bridges.rsi_serl_bridge import RsiSerlBridge  # noqa: E402
from shared.arena_manager import ArenaManager  # noqa: E402
from shared.constants import (  # noqa: E402
    SERL_FITNESS_GATE,
    THDSE_ARENA_DIM,
)

_TWO_PI = 2.0 * math.pi

_PROGRAM = "def evolved():\n    return 99\n"


def _seed_thdse(mgr: ArenaManager, fork_name: str) -> int:
    rng = mgr.rng.fork(fork_name)
    phases = rng.uniform(0.0, _TWO_PI, THDSE_ARENA_DIM).astype(np.float32)
    return mgr.alloc_thdse(phases=phases)


@pytest.fixture
def mgr() -> ArenaManager:
    return ArenaManager(master_seed=606)


@pytest.fixture
def bridge(mgr: ArenaManager) -> RsiSerlBridge:
    return RsiSerlBridge(mgr)


def test_constructor_rejects_non_manager():
    with pytest.raises(TypeError):
        RsiSerlBridge(arena_manager=[])  # type: ignore[arg-type]


def test_serl_candidate_below_gate_is_not_eligible(
    bridge: RsiSerlBridge, mgr: ArenaManager
):
    h = _seed_thdse(mgr, "below")
    fitness = SERL_FITNESS_GATE - 0.01
    result = bridge.serl_candidate_to_rsi(_PROGRAM, fitness, h)
    assert result["eligible"] is False
    assert result["rsi_compatible"] is False
    assert "SERL_FITNESS_GATE" in result["reason"]
    assert result["fitness"] == pytest.approx(fitness, abs=1e-6)
    assert bridge.eligible_count == 0
    assert bridge.candidates_seen == 1


def test_serl_candidate_above_gate_is_eligible(
    bridge: RsiSerlBridge, mgr: ArenaManager
):
    h = _seed_thdse(mgr, "above")
    fitness = SERL_FITNESS_GATE + 0.2
    result = bridge.serl_candidate_to_rsi(_PROGRAM, fitness, h)
    assert result["eligible"] is True
    assert result["rsi_compatible"] is True
    assert isinstance(result["cross_similarity"], float)
    # Compared against a zero CCE vector, cos(phase - 0) = cos(phase);
    # for uniformly distributed phases the magnitude stays small.
    assert -1.0 <= result["cross_similarity"] <= 1.0
    assert bridge.eligible_count == 1


def test_serl_candidate_result_has_provenance(
    bridge: RsiSerlBridge, mgr: ArenaManager
):
    h = _seed_thdse(mgr, "prov")
    result = bridge.serl_candidate_to_rsi(
        _PROGRAM, SERL_FITNESS_GATE + 0.05, h
    )
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "serl_candidate_to_rsi"
    assert prov["source_arena"] == "thdse"
    assert prov["target_arena"] == "cce"
    assert result["metadata"]["gate_threshold"] == SERL_FITNESS_GATE
    assert result["metadata"]["eligible"] is True


def test_serl_candidate_rejects_empty_program(
    bridge: RsiSerlBridge, mgr: ArenaManager
):
    h = _seed_thdse(mgr, "empty")
    result = bridge.serl_candidate_to_rsi(
        "   ", SERL_FITNESS_GATE + 0.1, h
    )
    assert result["eligible"] is False
    assert "empty" in result["reason"]


def test_serl_candidate_rejects_non_int_handle(bridge: RsiSerlBridge):
    with pytest.raises(TypeError):
        bridge.serl_candidate_to_rsi(_PROGRAM, 0.9, "h")  # type: ignore[arg-type]


def test_serl_candidate_rejects_non_numeric_fitness(
    bridge: RsiSerlBridge, mgr: ArenaManager
):
    h = _seed_thdse(mgr, "badfit")
    with pytest.raises(TypeError):
        bridge.serl_candidate_to_rsi(_PROGRAM, "0.9", h)  # type: ignore[arg-type]


def test_rsi_skill_feedback_mean_and_exceeds(bridge: RsiSerlBridge):
    scores = [0.1, 0.4, 0.5, 0.8]
    result = bridge.rsi_skill_to_serl_feedback("skill-xyz", scores)
    assert result["skill_id"] == "skill-xyz"
    assert result["sample_count"] == 4
    assert result["mean_performance"] == pytest.approx(
        sum(scores) / len(scores), abs=1e-6
    )
    assert result["exceeds_gate"] is (
        result["mean_performance"] >= SERL_FITNESS_GATE
    )


def test_rsi_skill_feedback_records_min_max(bridge: RsiSerlBridge):
    result = bridge.rsi_skill_to_serl_feedback(
        "skill-mm", [0.2, 0.9, 0.5]
    )
    assert result["metadata"]["min_score"] == pytest.approx(0.2, abs=1e-6)
    assert result["metadata"]["max_score"] == pytest.approx(0.9, abs=1e-6)
    assert result["metadata"]["gate_threshold"] == SERL_FITNESS_GATE


def test_rsi_skill_feedback_has_provenance(bridge: RsiSerlBridge):
    result = bridge.rsi_skill_to_serl_feedback("skill-prov", [0.5])
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "rsi_skill_to_serl_feedback"
    assert prov["source_arena"] == "cce"
    assert prov["target_arena"] == "thdse"
    assert isinstance(prov["skill_hash"], str)
    assert len(prov["skill_hash"]) == 12  # 6 bytes hex


def test_rsi_skill_feedback_rejects_empty_scores(bridge: RsiSerlBridge):
    with pytest.raises(ValueError):
        bridge.rsi_skill_to_serl_feedback("skill-empty", [])


def test_rsi_skill_feedback_rejects_non_numeric_scores(
    bridge: RsiSerlBridge,
):
    with pytest.raises(TypeError):
        bridge.rsi_skill_to_serl_feedback(
            "skill-bad", [0.5, "high"]  # type: ignore[list-item]
        )


def test_candidate_counters_track_sequence(
    bridge: RsiSerlBridge, mgr: ArenaManager
):
    # 3 candidates: 2 eligible, 1 below gate.
    h1 = _seed_thdse(mgr, "cs1")
    h2 = _seed_thdse(mgr, "cs2")
    h3 = _seed_thdse(mgr, "cs3")
    bridge.serl_candidate_to_rsi(_PROGRAM, SERL_FITNESS_GATE + 0.1, h1)
    bridge.serl_candidate_to_rsi(_PROGRAM, SERL_FITNESS_GATE - 0.2, h2)
    bridge.serl_candidate_to_rsi(_PROGRAM, SERL_FITNESS_GATE + 0.3, h3)
    assert bridge.candidates_seen == 3
    assert bridge.eligible_count == 2


def test_feedback_history_is_preserved(bridge: RsiSerlBridge):
    bridge.rsi_skill_to_serl_feedback("skill-a", [0.5])
    bridge.rsi_skill_to_serl_feedback("skill-b", [0.7, 0.8])
    assert bridge.feedback_count == 2
    history = bridge.get_feedback_history()
    assert len(history) == 2
    assert history[0]["skill_id"] == "skill-a"
    assert history[1]["skill_id"] == "skill-b"
