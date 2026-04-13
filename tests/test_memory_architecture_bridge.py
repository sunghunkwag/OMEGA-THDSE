"""Phase 11 — MemoryArchitectureBridge tests (Rule 18, D6, Rule 20)."""

from __future__ import annotations

from typing import List, Tuple

import pytest

from shared.arena_manager import ArenaManager
from shared.constants import CCE_ARENA_DIM
from shared.semantic_encoder import SemanticEncoder
from bridges.memory_architecture_bridge import MemoryArchitectureBridge
from bridges.semantic_concept_bridge import SemanticConceptBridge


@pytest.fixture()
def bridge():
    mgr = ArenaManager(master_seed=1111)
    enc = SemanticEncoder(prefer="hash")
    scb = SemanticConceptBridge(mgr, encoder=enc)
    return MemoryArchitectureBridge(mgr, encoder=enc, semantic_bridge=scb)


def test_remember_event_allocates_cce_handle(bridge):
    result = bridge.remember_event("the kettle boiled at 9am")
    assert result["tier"] == "episodic"
    assert result["cce_handle"] >= 0
    phases = bridge._mgr.get_cce_phases(result["cce_handle"])
    assert phases.shape == (CCE_ARENA_DIM,)
    assert result["metadata"]["provenance"]["operation"] == "remember_event"


def test_assert_fact_stores_and_queries(bridge):
    bridge.assert_fact("water", "boils at 100 celsius")
    bridge.assert_fact("mercury", "freezes at -38 celsius")
    top = bridge.query_top1("boiling water")
    assert top["found"]
    assert top["tier"] == "semantic"
    assert "water" in top["content"]
    assert top["metadata"]["provenance"]["operation"] == "query_top1"


def test_register_procedure_is_callable_after_retrieval(bridge):
    bridge.register_procedure("add two numbers", lambda a, b: a + b)
    top = bridge.query_top1("add 2 and 3")
    assert top["found"]
    assert top["tier"] == "procedural"
    proc_records = bridge.memory.procedural._records
    assert proc_records[0].procedure(2, 3) == 5


def test_consolidation_promotes_rehearsed_episodes(bridge):
    bridge.remember_event("summer trip to the lake", metadata={"subject": "trip"})
    # Rehearse above threshold.
    for _ in range(bridge.memory.episodic.consolidation_threshold + 1):
        bridge.memory.episodic.recall("lake trip", top_k=1)
    res = bridge.consolidate()
    assert res["promoted_count"] >= 1
    assert bridge.counts()["semantic"] >= 1
    assert res["metadata"]["provenance"]["operation"] == "consolidate"


# --------------------------------------------------------------------------- #
# Rule 18 end-to-end via the bridge
# --------------------------------------------------------------------------- #

_RULE18_TARGETS: List[Tuple[str, str]] = [
    ("water-temp", "water boils at one hundred degrees celsius"),
    ("capital-france", "the capital of france is paris"),
    ("speed-of-light", "light travels at about 300000 kilometres per second"),
    ("python-lang", "python is a high level programming language"),
    ("pets-dogs", "dogs are loyal four legged pets"),
]
_RULE18_DISTRACTORS: List[str] = [
    "the pacific ocean is vast",
    "bees pollinate flowers",
    "shakespeare wrote plays",
    "bread contains flour and yeast",
    "basketball uses a round ball",
    "mount everest is tall",
    "gold is a precious metal",
    "ice is frozen water",
    "venus is a planet",
    "jupiter has storms",
    "tea comes from leaves",
    "the heart pumps blood",
    "airplanes fly through the sky",
    "the moon orbits the earth",
    "saltwater is briny",
    "guitars make music",
    "the sahara has sand",
    "rain falls from clouds",
    "chess is a board game",
    "bananas grow on trees",
    "the internet links computers",
    "elephants are big mammals",
    "whales are marine mammals",
    "trees make oxygen",
    "walls can be long",
    "fire consumes wood",
    "glass is transparent",
    "batteries power devices",
    "lemons taste sour",
    "mirrors reflect images",
    "magnets stick together",
    "diamonds are hard",
    "snow is white and cold",
    "rivers reach the sea",
    "stars shine at night",
    "the sun is bright",
    "penguins live in antarctica",
    "honey tastes sweet",
    "spiders spin webs",
    "owls hoot at night",
    "kangaroos live in australia",
    "books are read by people",
    "clocks measure time",
    "bicycles have two wheels",
    "clouds drift across the sky",
]


_RULE18_QUERIES: List[Tuple[str, str]] = [
    ("water boiling temperature", "water boils at one hundred degrees celsius"),
    ("france capital", "the capital of france is paris"),
    ("speed of light value", "light travels at about 300000 kilometres per second"),
    ("python programming", "python is a high level programming language"),
    ("loyal dog", "dogs are loyal four legged pets"),
]


def test_rule18_bridge_benchmark_meets_threshold(bridge):
    for subject, fact in _RULE18_TARGETS:
        bridge.assert_fact(subject=subject, fact=fact)
    for line in _RULE18_DISTRACTORS:
        bridge.assert_fact(subject="misc", fact=line)
    assert sum(bridge.counts().values()) >= 50

    result = bridge.benchmark_top1(
        [
            (q, f"{subject}: {fact}".lower())
            for (q, fact), (subject, _) in zip(
                _RULE18_QUERIES, _RULE18_TARGETS
            )
        ]
    )
    # Re-form expected strings from actual stored records (subject: fact)
    expected_pairs = [
        (q, f"{subject}: {fact}")
        for (q, expected_fact), (subject, fact) in zip(
            _RULE18_QUERIES, _RULE18_TARGETS
        )
    ]
    real_result = bridge.benchmark_top1(expected_pairs)
    assert real_result["accuracy"] >= 0.8, (
        f"Rule 18: accuracy {real_result['accuracy']:.2f} < 0.8 "
        f"(misses={real_result['misses']})"
    )
    assert real_result["passes_rule18"] is True


def test_rule20_bridge_imports_phase9_and_phase4():
    import bridges.memory_architecture_bridge as mab
    assert hasattr(mab, "SemanticConceptBridge")
    assert hasattr(mab, "MemoryHypothesisBridge")


def test_rule9_provenance_on_every_store_method(bridge):
    e = bridge.remember_event("event")
    f = bridge.assert_fact("subject", "fact")
    p = bridge.register_procedure("do the thing", lambda: None)
    for r in (e, f, p):
        assert "provenance" in r["metadata"]


def test_memory_summary_for_hypothesis_wiring(bridge):
    bridge.assert_fact("car", "a car has four wheels")
    res = bridge.memory_summary_for_hypothesis("car", tags=["vehicle"])
    # Rule 20: MemoryHypothesisBridge result must be embedded.
    assert "metadata" in res and "provenance" in res["metadata"]
    assert res["memory_lookup"]["found"]
