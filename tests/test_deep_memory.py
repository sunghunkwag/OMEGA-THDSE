"""Phase 11 — DeepMemory tests (Rule 18, D6)."""

from __future__ import annotations

import pytest

from shared.deep_memory import (
    DeepMemory,
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
)
from shared.semantic_encoder import SemanticEncoder


@pytest.fixture(scope="module")
def encoder():
    return SemanticEncoder(prefer="hash")


@pytest.fixture()
def deep(encoder):
    return DeepMemory(encoder=encoder)


# --------------------------------------------------------------------------- #
# Episodic
# --------------------------------------------------------------------------- #


def test_episodic_store_and_recall_returns_closest(encoder):
    ep = EpisodicMemory(encoder)
    ep.store("breakfast coffee morning")
    ep.store("meeting product team noon")
    ep.store("evening park run")
    hits = ep.recall("breakfast coffee", top_k=1)
    assert hits
    assert "breakfast" in hits[0].content


def test_episodic_rehearsal_tracked(encoder):
    ep = EpisodicMemory(encoder)
    rec = ep.store("picnic on saturday")
    assert rec.rehearsals == 0
    ep.recall("saturday picnic", top_k=1)
    ep.recall("picnic", top_k=1)
    assert ep.recall("picnic", top_k=1)[0].rehearsals == 3


# --------------------------------------------------------------------------- #
# Semantic
# --------------------------------------------------------------------------- #


def test_semantic_dedupe_merges_sources(encoder):
    sem = SemanticMemory(encoder, dedupe_threshold=0.99)
    a = sem.assert_fact("water", "boils at 100C", source_episodes=[1])
    b = sem.assert_fact("water", "boils at 100C", source_episodes=[2])
    # Dedupe on identical key => single record, merged sources.
    assert a.fact_id == b.fact_id
    assert sorted(a.source_episodes) == [1, 2]


def test_semantic_query_returns_best_match(encoder):
    sem = SemanticMemory(encoder)
    sem.assert_fact("python", "is a programming language")
    sem.assert_fact("fish", "breathes with gills")
    sem.assert_fact("car", "runs on gasoline")
    top = sem.query("programming language")[0]
    assert top.subject == "python"


# --------------------------------------------------------------------------- #
# Procedural
# --------------------------------------------------------------------------- #


def test_procedural_match_returns_triggered_procedure(encoder):
    proc = ProceduralMemory(encoder)
    proc.register("add two numbers", lambda a, b: a + b)
    proc.register("multiply two numbers", lambda a, b: a * b)
    top = proc.match("add 3 and 4")[0]
    assert top.trigger == "add two numbers"
    assert top.procedure(3, 4) == 7


def test_procedural_rejects_non_callable(encoder):
    proc = ProceduralMemory(encoder)
    with pytest.raises(TypeError):
        proc.register("bad", 42)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Rule 18 — 50+ items, >= 80% top-1 accuracy on 5 queries
# --------------------------------------------------------------------------- #


_RULE18_FACTS = [
    ("water boils at 100 degrees celsius", "water temperature"),
    ("paris is the capital of france", "capital of france"),
    ("the speed of light is 299792458 meters per second", "speed of light"),
    ("python is a programming language", "python programming"),
    ("dogs are loyal canine pets", "loyal dog pet"),
]

_RULE18_DISTRACTORS = [
    "the pacific ocean covers a third of earth",
    "bees pollinate flowers in spring",
    "shakespeare wrote many plays",
    "the eiffel tower is a famous landmark",
    "bread is made from flour water and yeast",
    "basketball has five players per team",
    "mount everest is the tallest peak",
    "gold is a dense metal",
    "ice melts above zero degrees",
    "venus is the second planet from the sun",
    "jupiter is the largest planet",
    "tea is brewed from dried leaves",
    "the heart pumps blood throughout the body",
    "airplanes use jet engines for propulsion",
    "the moon orbits the earth",
    "saltwater contains dissolved sodium chloride",
    "guitars have six strings usually",
    "the sahara is the largest hot desert",
    "rain is precipitation from clouds",
    "thunder follows lightning",
    "chess has sixty four squares",
    "bananas grow in tropical climates",
    "earth rotates once per day",
    "the internet connects computers worldwide",
    "bread rises due to yeast fermentation",
    "elephants are large land mammals",
    "whales breathe air through blowholes",
    "trees produce oxygen through photosynthesis",
    "the great wall of china is long",
    "fire needs oxygen to burn",
    "glass is made by melting sand",
    "batteries store electric charge",
    "lemons have a sour taste",
    "mirrors reflect light waves",
    "magnets attract iron filings",
    "diamonds are crystallised carbon",
    "snow falls in winter weather",
    "rivers flow toward the sea",
    "stars emit their own light",
    "the sun is a yellow dwarf star",
    "penguins cannot fly but can swim",
    "honey is produced by bees",
    "spiders have eight legs",
    "owls hunt during the night",
    "kangaroos hop across australia",
]


def test_rule18_top1_accuracy_exceeds_threshold(encoder):
    mem = DeepMemory(encoder=encoder)
    # Store target facts as semantic records (high-signal).
    for fact, _query in _RULE18_FACTS:
        mem.semantic.assert_fact(subject=_query, fact=fact)
    # Stuff the memory with 45 distractors to reach > 50 items total.
    for line in _RULE18_DISTRACTORS:
        mem.semantic.assert_fact(subject="misc", fact=line)
    assert (
        len(mem.semantic) + len(mem.episodic) + len(mem.procedural)
    ) >= 50

    # Rule 18 query set — 5 queries that must rank the right fact #1.
    queries = [
        ("boiling water temperature", "water temperature"),
        ("capital france", "capital of france"),
        ("how fast is light", "speed of light"),
        ("python language", "python programming"),
        ("loyal pet dog", "loyal dog pet"),
    ]
    correct = 0
    misses = []
    for q, expected_subject in queries:
        top = mem.semantic.query(q, top_k=1)[0]
        if top.subject == expected_subject:
            correct += 1
        else:
            misses.append((q, expected_subject, top.subject))
    accuracy = correct / len(queries)
    assert accuracy >= 0.8, (
        f"Rule 18: top-1 accuracy {accuracy:.2f} < 0.8 "
        f"(misses={misses})"
    )


# --------------------------------------------------------------------------- #
# Consolidation (episodic → semantic promotion)
# --------------------------------------------------------------------------- #


def test_consolidation_promotes_rehearsed_episodes(encoder):
    mem = DeepMemory(encoder=encoder)
    mem.episodic.store("alice met bob at the conference", metadata={"subject": "alice"})
    # Rehearse by recalling above the threshold.
    for _ in range(mem.episodic.consolidation_threshold + 1):
        mem.episodic.recall("alice conference", top_k=1)
    result = mem.consolidate()
    assert result["promoted_count"] >= 1
    # Semantic memory must now contain a fact sourced from the episode.
    query_hit = mem.semantic.query("alice bob conference")[0]
    assert "alice" in query_hit.fact


def test_unified_query_returns_tier(deep):
    deep.episodic.store("saturday beach trip", metadata={})
    deep.semantic.assert_fact("python", "programming language")
    deep.procedural.register("greet someone", lambda name: f"hi {name}")
    top = deep.query("friendly greeting", top_k=1)[0]
    assert top["tier"] in ("episodic", "semantic", "procedural")
    assert top["score"] > -1.0
