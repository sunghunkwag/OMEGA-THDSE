"""Tests for :mod:`bridges.concept_axiom_bridge` (PLAN.md Phase 3, Gap 2).

Covers construction, single concept projection, self-similarity
guarantees, provenance propagation, batch operations, handle
validation, and error paths (bad handle, wrong type).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bridges.concept_axiom_bridge import ConceptAxiomBridge  # noqa: E402
from shared.arena_manager import ArenaManager  # noqa: E402
from shared.constants import CCE_ARENA_DIM, THDSE_ARENA_DIM  # noqa: E402

_TWO_PI = 2.0 * math.pi


def _seed_concept(mgr: ArenaManager, fork_name: str) -> tuple[int, np.ndarray]:
    rng = mgr.rng.fork(fork_name)
    phases = rng.uniform(0.0, _TWO_PI, CCE_ARENA_DIM).astype(np.float32)
    handle = mgr.alloc_cce(phases=phases)
    return handle, phases


@pytest.fixture
def mgr() -> ArenaManager:
    return ArenaManager(master_seed=101)


@pytest.fixture
def bridge(mgr: ArenaManager) -> ConceptAxiomBridge:
    return ConceptAxiomBridge(mgr)


def test_constructor_rejects_non_arena_manager():
    with pytest.raises(TypeError):
        ConceptAxiomBridge(arena_manager=object())  # type: ignore[arg-type]


def test_concept_to_axiom_allocates_new_thdse_handle(
    bridge: ConceptAxiomBridge, mgr: ArenaManager
):
    h_cce, _ = _seed_concept(mgr, "concept1")
    before = mgr.count("thdse")
    result = bridge.concept_to_axiom(h_cce, {"title": "apple"})
    after = mgr.count("thdse")
    assert after == before + 1
    assert isinstance(result["thdse_handle"], int)
    assert result["thdse_handle"] >= 0


def test_concept_to_axiom_self_similarity_is_one(
    bridge: ConceptAxiomBridge, mgr: ArenaManager
):
    h_cce, _ = _seed_concept(mgr, "concept2")
    result = bridge.concept_to_axiom(h_cce, None)
    # cross_arena_similarity(v, project_down(v)) must be exactly 1.0.
    assert result["similarity_to_source"] == pytest.approx(1.0, abs=1e-6)


def test_concept_to_axiom_projection_preserves_phases(
    bridge: ConceptAxiomBridge, mgr: ArenaManager
):
    h_cce, phases = _seed_concept(mgr, "concept3")
    result = bridge.concept_to_axiom(h_cce, {})
    stored = mgr.get_thdse_phases(result["thdse_handle"])
    # Every 39th element of the 10k source must appear in the axiom.
    for k in range(THDSE_ARENA_DIM):
        assert stored[k] == pytest.approx(phases[k * 39], abs=1e-5)


def test_concept_to_axiom_result_has_provenance(
    bridge: ConceptAxiomBridge, mgr: ArenaManager
):
    h_cce, _ = _seed_concept(mgr, "concept4")
    result = bridge.concept_to_axiom(h_cce, {"title": "cat"})
    meta = result["metadata"]
    assert "provenance" in meta
    prov = meta["provenance"]
    assert prov["operation"] == "concept_to_axiom"
    assert prov["source_arena"] == "cce"
    assert prov["target_arena"] == "thdse"
    assert prov["source_dim"] == CCE_ARENA_DIM
    assert prov["target_dim"] == THDSE_ARENA_DIM
    assert isinstance(prov["timestamp"], float)
    assert meta["source_concept_handle"] == h_cce
    assert meta["source_concept_metadata"]["title"] == "cat"


def test_concept_to_axiom_includes_projection_provenance(
    bridge: ConceptAxiomBridge, mgr: ArenaManager
):
    h_cce, _ = _seed_concept(mgr, "concept5")
    result = bridge.concept_to_axiom(h_cce, None)
    proj_prov = result["metadata"]["projection_provenance"]
    assert proj_prov["operation"] == "project_down"
    assert proj_prov["stride"] == 39


def test_concept_to_axiom_generates_stable_axiom_id(
    bridge: ConceptAxiomBridge, mgr: ArenaManager
):
    h1, _ = _seed_concept(mgr, "stableA")
    h2, _ = _seed_concept(mgr, "stableA")  # same fork name → same seed
    r1 = bridge.concept_to_axiom(h1, {})
    r2 = bridge.concept_to_axiom(h2, {})
    # The THDSE handles differ but the axiom id depends on phases + handle,
    # so the two ids should NOT collide.
    assert r1["metadata"]["axiom_id"] != r2["metadata"]["axiom_id"]
    assert r1["metadata"]["axiom_id"].startswith("axiom-")


def test_concept_to_axiom_rejects_negative_handle(bridge: ConceptAxiomBridge):
    with pytest.raises(TypeError):
        bridge.concept_to_axiom(-1, {})


def test_concept_to_axiom_rejects_non_int_handle(bridge: ConceptAxiomBridge):
    with pytest.raises(TypeError):
        bridge.concept_to_axiom("not_a_handle", {})  # type: ignore[arg-type]


def test_axiom_to_concept_similarity_returns_float(
    bridge: ConceptAxiomBridge, mgr: ArenaManager
):
    h_cce, phases = _seed_concept(mgr, "mix1")
    first = bridge.concept_to_axiom(h_cce, {})
    h_axiom = first["thdse_handle"]
    sim_result = bridge.axiom_to_concept_similarity(h_axiom, h_cce)
    assert isinstance(sim_result["similarity"], float)
    # Same source must produce perfect similarity through the projection.
    assert sim_result["similarity"] == pytest.approx(1.0, abs=1e-6)
    assert sim_result["metadata"]["provenance"]["operation"] == (
        "axiom_to_concept_similarity"
    )


def test_axiom_to_concept_similarity_is_low_for_random_pair(
    bridge: ConceptAxiomBridge, mgr: ArenaManager
):
    h_cce_1, _ = _seed_concept(mgr, "rp1")
    h_cce_2, _ = _seed_concept(mgr, "rp2")
    r1 = bridge.concept_to_axiom(h_cce_1, {})
    sim_result = bridge.axiom_to_concept_similarity(r1["thdse_handle"], h_cce_2)
    # Two independent concepts should be near-orthogonal.
    assert abs(sim_result["similarity"]) < 0.1


def test_batch_project_converts_each_input(
    bridge: ConceptAxiomBridge, mgr: ArenaManager
):
    handles = [_seed_concept(mgr, f"batch{i}")[0] for i in range(5)]
    results = bridge.batch_project(handles)
    assert len(results) == 5
    for idx, item in enumerate(results):
        assert item["metadata"]["batch_index"] == idx
        assert item["metadata"]["provenance"]["operation"] == (
            "concept_to_axiom"
        )
        assert isinstance(item["thdse_handle"], int)


def test_batch_project_rejects_non_list(bridge: ConceptAxiomBridge):
    with pytest.raises(TypeError):
        bridge.batch_project("not a list")  # type: ignore[arg-type]


def test_conversion_count_tracks_calls(
    bridge: ConceptAxiomBridge, mgr: ArenaManager
):
    assert bridge.conversion_count == 0
    for i in range(3):
        h, _ = _seed_concept(mgr, f"cnt{i}")
        bridge.concept_to_axiom(h, None)
    assert bridge.conversion_count == 3


def test_independent_managers_are_isolated():
    mgr_a = ArenaManager(master_seed=1)
    mgr_b = ArenaManager(master_seed=1)
    bridge_a = ConceptAxiomBridge(mgr_a)
    bridge_b = ConceptAxiomBridge(mgr_b)
    h_a, _ = _seed_concept(mgr_a, "iso")
    _ = bridge_a.concept_to_axiom(h_a, None)
    # mgr_b should not have received any allocation from mgr_a's work.
    assert mgr_b.count("thdse") == 0
    assert bridge_b.conversion_count == 0
