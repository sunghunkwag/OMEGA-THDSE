"""Tests for :mod:`bridges.self_model_bridge` (Phase 4 Gap 9)."""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bridges.self_model_bridge import SelfModelBridge  # noqa: E402
from shared.arena_manager import ArenaManager  # noqa: E402
from shared.constants import (  # noqa: E402
    CCE_ARENA_DIM,
    CONTINUITY_THRESHOLD,
    MAX_CREDIBLE_LEAP,
    SELF_MODEL_COMPONENTS,
    THDSE_ARENA_DIM,
    WIREHEADING_THRESHOLD,
)
from shared.exceptions import DimensionMismatchError  # noqa: E402


def _make_phase_vec(value: float) -> list[float]:
    return [value] * CCE_ARENA_DIM


@pytest.fixture
def mgr() -> ArenaManager:
    return ArenaManager(master_seed=901)


@pytest.fixture
def bridge(mgr: ArenaManager) -> SelfModelBridge:
    return SelfModelBridge(mgr)


def test_constructor_rejects_non_arena_manager():
    with pytest.raises(TypeError):
        SelfModelBridge(arena_manager="nope")  # type: ignore[arg-type]


def test_export_self_model_state_returns_256_summary(
    bridge: SelfModelBridge,
):
    result = bridge.export_self_model_state(
        _make_phase_vec(0.1),
        _make_phase_vec(0.2),
        _make_phase_vec(0.3),
        _make_phase_vec(0.4),
    )
    bundled = result["thdse_self_model"]
    assert len(bundled) == THDSE_ARENA_DIM
    assert result["metadata"]["component_count"] == SELF_MODEL_COMPONENTS
    assert result["metadata"]["component_names"] == [
        "belief",
        "goal",
        "capability",
        "emotion",
    ]


def test_export_component_projections_are_256_dim(bridge: SelfModelBridge):
    result = bridge.export_self_model_state(
        _make_phase_vec(0.0),
        _make_phase_vec(0.5),
        _make_phase_vec(1.0),
        _make_phase_vec(1.5),
    )
    for name in ("belief", "goal", "capability", "emotion"):
        assert result["component_projections"][name].shape == (THDSE_ARENA_DIM,)


def test_export_metadata_has_provenance(bridge: SelfModelBridge):
    result = bridge.export_self_model_state(
        _make_phase_vec(0.1),
        _make_phase_vec(0.1),
        _make_phase_vec(0.1),
        _make_phase_vec(0.1),
    )
    prov = result["metadata"]["provenance"]
    assert prov["operation"] == "self_model_export"
    assert prov["source_arena"] == "cce"
    assert prov["target_arena"] == "thdse"
    assert prov["source_dim"] == CCE_ARENA_DIM
    assert prov["target_dim"] == THDSE_ARENA_DIM


def test_export_rejects_wrong_dim_component(bridge: SelfModelBridge):
    with pytest.raises(DimensionMismatchError):
        bridge.export_self_model_state(
            _make_phase_vec(0.0),
            [0.0] * 100,
            _make_phase_vec(0.0),
            _make_phase_vec(0.0),
        )


def test_detect_wireheading_flags_massive_cce_leap(bridge: SelfModelBridge):
    big = MAX_CREDIBLE_LEAP + 0.1
    result = bridge.detect_wireheading_from_thdse(big, 0.01)
    assert result["is_suspicious"] is True
    assert "MAX_CREDIBLE_LEAP" in result["reason"]
    assert result["metadata"]["flag_count"] == 1


def test_detect_wireheading_flags_sign_clash(bridge: SelfModelBridge):
    result = bridge.detect_wireheading_from_thdse(0.10, -0.10)
    assert result["is_suspicious"] is True
    assert "sign clash" in result["reason"]


def test_detect_wireheading_no_flag_for_consistent_deltas(
    bridge: SelfModelBridge,
):
    result = bridge.detect_wireheading_from_thdse(0.05, 0.05)
    assert result["is_suspicious"] is False
    assert result["reason"] == "no anomaly detected"
    assert result["metadata"]["flag_count"] == 0


def test_detect_wireheading_rejects_non_numeric(bridge: SelfModelBridge):
    with pytest.raises(TypeError):
        bridge.detect_wireheading_from_thdse("0.1", 0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        bridge.detect_wireheading_from_thdse(0.1, "0.1")  # type: ignore[arg-type]


def test_compute_self_model_drift_self_is_zero(bridge: SelfModelBridge):
    export = bridge.export_self_model_state(
        _make_phase_vec(0.1),
        _make_phase_vec(0.2),
        _make_phase_vec(0.3),
        _make_phase_vec(0.4),
    )
    drift = bridge.compute_self_model_drift(export, export)
    assert drift["drift_score"] == pytest.approx(0.0, abs=1e-5)
    assert drift["similarity"] == pytest.approx(1.0, abs=1e-5)
    assert drift["severity"] == "continuous"


def test_compute_self_model_drift_detects_major_shift(
    bridge: SelfModelBridge,
):
    a = bridge.export_self_model_state(
        _make_phase_vec(0.0),
        _make_phase_vec(0.0),
        _make_phase_vec(0.0),
        _make_phase_vec(0.0),
    )
    b = bridge.export_self_model_state(
        _make_phase_vec(math.pi),
        _make_phase_vec(math.pi),
        _make_phase_vec(math.pi),
        _make_phase_vec(math.pi),
    )
    drift = bridge.compute_self_model_drift(a, b)
    # Anti-parallel → similarity ≈ -1 → drift_score ≈ 2.
    assert drift["similarity"] == pytest.approx(-1.0, abs=1e-5)
    assert drift["drift_score"] == pytest.approx(2.0, abs=1e-5)
    # Drift above (1 - WIREHEADING_THRESHOLD) → severity flagged.
    assert drift["severity"] == "wireheading_suspect"


def test_compute_self_model_drift_metadata_has_provenance(
    bridge: SelfModelBridge,
):
    exp = bridge.export_self_model_state(
        _make_phase_vec(0.0),
        _make_phase_vec(0.0),
        _make_phase_vec(0.0),
        _make_phase_vec(0.0),
    )
    drift = bridge.compute_self_model_drift(exp, exp)
    prov = drift["metadata"]["provenance"]
    assert prov["operation"] == "compute_self_model_drift"
    assert prov["source_arena"] == "both"
    assert (
        drift["metadata"]["continuity_threshold"] == CONTINUITY_THRESHOLD
    )
    assert drift["metadata"]["wireheading_threshold"] == WIREHEADING_THRESHOLD


def test_compute_self_model_drift_rejects_missing_key(
    bridge: SelfModelBridge,
):
    exp = bridge.export_self_model_state(
        _make_phase_vec(0.1),
        _make_phase_vec(0.1),
        _make_phase_vec(0.1),
        _make_phase_vec(0.1),
    )
    with pytest.raises(KeyError):
        bridge.compute_self_model_drift({}, exp)
    with pytest.raises(KeyError):
        bridge.compute_self_model_drift(exp, {})


def test_summarize_drift_history_aggregates(bridge: SelfModelBridge):
    a = bridge.export_self_model_state(
        _make_phase_vec(0.0),
        _make_phase_vec(0.0),
        _make_phase_vec(0.0),
        _make_phase_vec(0.0),
    )
    b = bridge.export_self_model_state(
        _make_phase_vec(0.5),
        _make_phase_vec(0.5),
        _make_phase_vec(0.5),
        _make_phase_vec(0.5),
    )
    bridge.compute_self_model_drift(a, a)
    bridge.compute_self_model_drift(a, b)
    summary = bridge.summarize_drift_history()
    assert summary["sample_count"] == 2
    assert summary["max_drift"] >= summary["mean_drift"] >= summary["min_drift"]


def test_reset_drift_history_clears(bridge: SelfModelBridge):
    exp = bridge.export_self_model_state(
        _make_phase_vec(0.0),
        _make_phase_vec(0.0),
        _make_phase_vec(0.0),
        _make_phase_vec(0.0),
    )
    bridge.compute_self_model_drift(exp, exp)
    assert bridge.drift_observation_count == 1
    result = bridge.reset_drift_history()
    assert result["cleared"] == 1
    assert bridge.drift_observation_count == 0
