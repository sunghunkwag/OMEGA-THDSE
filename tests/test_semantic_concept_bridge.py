"""Phase 9 — SemanticConceptBridge integration tests (D1, D2, Rule 20)."""

from __future__ import annotations

import pytest

from shared.arena_manager import ArenaManager
from shared.constants import CCE_ARENA_DIM
from shared.semantic_encoder import SemanticEncoder
from bridges.semantic_concept_bridge import SemanticConceptBridge


@pytest.fixture()
def bridge():
    mgr = ArenaManager(master_seed=909)
    enc = SemanticEncoder(prefer="hash")  # avoid model download in CI
    return SemanticConceptBridge(mgr, encoder=enc)


def test_ground_text_allocates_cce_handle(bridge):
    result = bridge.ground_text("semantic grounding smoke test")
    h = result["cce_handle"]
    assert isinstance(h, int) and h >= 0
    # verify the handle resolves to a real CCE vector
    phases = bridge._mgr.get_cce_phases(h)
    assert phases.shape == (CCE_ARENA_DIM,)


def test_ground_text_provenance_required_for_rule9(bridge):
    result = bridge.ground_text("rule 9 check")
    meta = result["metadata"]
    prov = meta["provenance"]
    assert prov["operation"] == "ground_text"
    assert prov["source_arena"] == "text"
    assert prov["target_arena"] == "cce"
    # nested provenance from the grounder must also be preserved
    sub = meta["ingest_metadata"]["provenance"]
    assert sub["operation"] == "ingest_text"


def test_ground_structured_allocates_distinct_handle(bridge):
    a = bridge.ground_text("same content")
    b = bridge.ground_structured({"label": "same content"})
    assert a["cce_handle"] != b["cce_handle"]
    assert b["metadata"]["provenance"]["operation"] == "ground_structured"


def test_ground_file_reads_from_disk(tmp_path, bridge):
    p = tmp_path / "fragment.md"
    p.write_text("# Title\nbody", encoding="utf-8")
    result = bridge.ground_file(p)
    assert result["metadata"]["provenance"]["operation"] == "ground_file"
    assert result["metadata"]["ingest_metadata"]["source_path"] == str(p)


def test_concept_similarity_of_two_grounded_handles(bridge):
    r1 = bridge.ground_text("machine learning")
    r2 = bridge.ground_text("ML")
    r3 = bridge.ground_text("quantum physics")
    sim_pair = bridge.concept_similarity(r1["cce_handle"], r2["cce_handle"])
    dis_pair = bridge.concept_similarity(r1["cce_handle"], r3["cce_handle"])
    assert sim_pair["similarity"] > dis_pair["similarity"], (
        f"semantic pair ({sim_pair['similarity']:.3f}) should exceed "
        f"dissimilar pair ({dis_pair['similarity']:.3f})"
    )
    assert sim_pair["metadata"]["provenance"]["operation"] == "concept_similarity"


def test_concept_count_tracks_allocations(bridge):
    before = bridge.concept_count
    bridge.ground_text("a")
    bridge.ground_text("b")
    assert bridge.concept_count == before + 2


# --------------------------------------------------------------------------- #
# Rule 20 — every new module is imported by an existing bridge
# --------------------------------------------------------------------------- #


def test_rule20_concept_axiom_bridge_imports_semantic_concept_bridge():
    import bridges.concept_axiom_bridge as cab
    assert hasattr(cab, "SemanticConceptBridge")


def test_concept_axiom_bridge_ground_and_project_end_to_end():
    from bridges.concept_axiom_bridge import ConceptAxiomBridge

    mgr = ArenaManager(master_seed=910)
    enc = SemanticEncoder(prefer="hash")
    scb = SemanticConceptBridge(mgr, encoder=enc)
    cab = ConceptAxiomBridge(mgr, semantic_bridge=scb)
    result = cab.ground_and_project("end to end smoke")
    assert "cce_handle" in result
    assert "thdse_handle" in result
    assert result["metadata"]["provenance"]["operation"] == "concept_to_axiom"
    assert "grounding_provenance" in result["metadata"]
