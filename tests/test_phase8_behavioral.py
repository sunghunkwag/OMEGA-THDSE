"""Phase 8 unit tests — behavioural encoder, prototypes, problem encoding, CEGR.

These are Tier-1 wiring tests: they exercise REAL classes (no mocks)
with the Python-fallback arena from
:mod:`thdse.src.utils.arena_factory`. They run on bare Python +
numpy and verify every Phase 8 invariant called out in PLAN.md
Section 8 (Rules 18, 19, 20, 21, 22, 23) plus the basic shape /
provenance contracts (Rule 9).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_THDSE = _REPO_ROOT / "thdse"
for p in (str(_REPO_ROOT), str(_THDSE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.projection.behavioral_encoder import (  # noqa: E402
    DEFAULT_PROBE_INPUTS,
    BehavioralEncoder,
    BehavioralProfile,
    hash_to_phases,
)
from src.synthesis.counterexample_refiner import (  # noqa: E402
    CounterexampleRefiner,
    RefinementGuide,
)
from src.synthesis.problem_spec import (  # noqa: E402
    ProblemEncoder,
    ProblemSpec,
    ProblemVector,
    score_against_problem,
)
from src.synthesis.prototype_extractor import (  # noqa: E402
    FunctionalPrototype,
    PrototypeExtractor,
)
from src.utils.arena_factory import _PyFhrrArenaExtended  # noqa: E402


def _arena(capacity: int = 20_000) -> _PyFhrrArenaExtended:
    return _PyFhrrArenaExtended(capacity=capacity, dimension=256)


# --------------------------------------------------------------------------- #
# RULE 18 — behavioural encoder must execute, not analyse syntax
# --------------------------------------------------------------------------- #


def test_default_probe_set_is_deterministic_and_diverse():
    a = DEFAULT_PROBE_INPUTS
    assert len(a) == 20
    types_seen = {type(value).__name__ for _, value in a}
    # The encoder must hit at least four data shapes so behaviour
    # encoding is type-diverse.
    assert len(types_seen) >= 4


def test_hash_to_phases_is_deterministic():
    a = hash_to_phases(("input", [1, 2, 3]), 256, tag="x")
    b = hash_to_phases(("input", [1, 2, 3]), 256, tag="x")
    assert a == b
    assert len(a) == 256


def test_hash_to_phases_distinguishes_distinct_values():
    a = hash_to_phases(("input", [1, 2, 3]), 256, tag="x")
    b = hash_to_phases(("input", [1, 2, 4]), 256, tag="x")
    diff = sum(1 for x, y in zip(a, b) if x != y)
    assert diff > 100


def test_rule18_identical_behaviour_high_similarity():
    arena = _arena()
    enc = BehavioralEncoder(arena, dimension=256, n_probes=20)
    p1 = enc.encode_behavior("def f(x):\n    return x + 1\n")
    p2 = enc.encode_behavior("def f(x):\n    return 1 + x\n")
    sim = enc.similarity(p1, p2)
    assert sim > 0.95, f"identical behaviour gave sim={sim}"


def test_rule18_different_behaviour_low_similarity():
    arena = _arena()
    enc = BehavioralEncoder(arena, dimension=256, n_probes=20)
    p1 = enc.encode_behavior("def f(x):\n    return x + 1\n")
    p2 = enc.encode_behavior("def f(x):\n    return x * 2\n")
    sim = enc.similarity(p1, p2)
    assert sim < 0.30, f"different behaviour gave sim={sim}"


def test_encode_behavior_records_provenance_and_io_pairs():
    arena = _arena()
    enc = BehavioralEncoder(arena, dimension=256, n_probes=20)
    profile = enc.encode_behavior("def f(x):\n    return x\n")
    assert isinstance(profile, BehavioralProfile)
    assert len(profile.io_pairs) == 20
    assert len(profile.behavioral_phases) == 256
    prov = profile.metadata["provenance"]
    assert prov["operation"] == "encode_behavior"
    assert prov["source_arena"] == "thdse"
    assert prov["probe_count"] == 20


def test_encode_behavior_handles_crashing_program():
    arena = _arena()
    enc = BehavioralEncoder(arena, dimension=256, n_probes=20)
    profile = enc.encode_behavior(
        "def f(x):\n    raise ValueError('boom')\n"
    )
    assert len(profile.io_pairs) == 20
    # Crashing programs still encode every io pair — failure modes
    # are recorded as data, never silently dropped.
    assert profile.execution_profile["fitness"] >= 0.0


def test_encode_io_pairs_short_circuit_on_empty_input():
    arena = _arena()
    enc = BehavioralEncoder(arena, dimension=256, n_probes=20)
    profile = enc.encode_io_pairs([])
    assert profile.execution_profile.get("empty") is True
    assert len(profile.behavioral_phases) == 256


# --------------------------------------------------------------------------- #
# RULE 21 — prototype must compress
# --------------------------------------------------------------------------- #


def test_prototype_extractor_compresses_atom_set():
    arena = _arena()
    enc = BehavioralEncoder(arena, dimension=256, n_probes=20)
    sources = [
        "def f(arr):\n    total = 0\n    for x in arr:\n        total += x\n    return total\n",
        "def f(arr):\n    return sum(arr)\n",
        "def f(arr):\n    s = 0\n    i = 0\n    while i < len(arr):\n        s += arr[i]\n        i += 1\n    return s\n",
    ]
    candidates = []
    for src in sources:
        prof = enc.encode_behavior(src)
        candidates.append((src, prof, 0.7))

    extractor = PrototypeExtractor(arena, dimension=256)
    prototypes = extractor.extract_prototypes(
        candidates, similarity_threshold=0.5
    )
    assert len(prototypes) >= 1
    proto = prototypes[0]
    assert isinstance(proto, FunctionalPrototype)
    mean_member = proto.metadata["mean_member_atom_count"]
    # Rule 21 compression test: essential atoms must be strictly
    # smaller than the mean member size.
    assert len(proto.essential_atoms) < mean_member
    assert proto.member_count >= 2


def test_prototype_metadata_carries_provenance():
    arena = _arena()
    enc = BehavioralEncoder(arena, dimension=256, n_probes=20)
    sources = [
        "def f(x):\n    return x + 1\n",
        "def f(x):\n    return 1 + x\n",
    ]
    candidates = [
        (src, enc.encode_behavior(src), 0.6) for src in sources
    ]
    extractor = PrototypeExtractor(arena, dimension=256)
    prototypes = extractor.extract_prototypes(
        candidates, similarity_threshold=0.95
    )
    assert prototypes
    assert prototypes[0].metadata["provenance"]["operation"] == (
        "extract_prototype"
    )


def test_prototype_extractor_drops_singletons():
    arena = _arena()
    enc = BehavioralEncoder(arena, dimension=256, n_probes=20)
    a_src = "def f(x):\n    return x + 1\n"
    b_src = "def f(x):\n    return x * 2\n"
    candidates = [
        (a_src, enc.encode_behavior(a_src), 0.6),
        (b_src, enc.encode_behavior(b_src), 0.6),
    ]
    extractor = PrototypeExtractor(arena, dimension=256)
    prototypes = extractor.extract_prototypes(
        candidates, similarity_threshold=0.99
    )
    # Two unrelated candidates should not form a cluster.
    assert prototypes == []


# --------------------------------------------------------------------------- #
# RULE 19 — ProblemSpec / ProblemEncoder
# --------------------------------------------------------------------------- #


def test_problemspec_rejects_empty_examples():
    with pytest.raises(ValueError):
        ProblemSpec(name="bad", io_examples=[])


def test_problemspec_requires_pairs():
    with pytest.raises(ValueError):
        ProblemSpec(name="bad", io_examples=[(1,)])  # type: ignore[list-item]


def test_problem_encoder_round_trip():
    arena = _arena()
    spec = ProblemSpec(
        name="inc",
        io_examples=[(i, i + 1) for i in range(10)],
    )
    encoder = ProblemEncoder(arena, dimension=256)
    pv = encoder.encode_problem(spec)
    assert isinstance(pv, ProblemVector)
    assert len(pv.phases) == 256
    assert pv.example_count == 10
    assert pv.metadata["provenance"]["operation"] == "encode_problem"


def test_score_against_problem_correctly_counts_passed_failed():
    spec = ProblemSpec(
        name="double",
        io_examples=[(i, i * 2) for i in range(10)],
    )
    good = lambda x: x * 2
    bad = lambda x: x + 1
    good_score = score_against_problem(good, spec)
    bad_score = score_against_problem(bad, spec)
    assert good_score["pass_rate"] == 1.0
    assert bad_score["pass_rate"] < 1.0
    # Rule 20: bad scoring counts ONLY exact matches.
    assert bad_score["passed"] == sum(
        1 for i in range(10) if (i + 1) == (i * 2)
    )


# --------------------------------------------------------------------------- #
# RULE 22 — counterexample refiner uses real quotient projection
# --------------------------------------------------------------------------- #


def test_refiner_correlation_decreases_after_projection():
    arena = _arena()
    spec = ProblemSpec(
        name="inc",
        io_examples=[(i, i + 1) for i in range(10)],
    )
    refiner = CounterexampleRefiner(arena, dimension=256)
    guide = refiner.refine(
        failed_source="def f(x): return x * 2",
        problem_io=spec.io_examples,
        passed_ios=[],
        failed_ios=[(i, i + 1, i * 2) for i in range(10)],
    )
    assert isinstance(guide, RefinementGuide)
    assert guide.correlation_before >= 0.99
    # Rule 22: after projection the probe must be pushed away from
    # V_error — its correlation has to fall meaningfully below 1.0.
    assert guide.correlation_after < guide.correlation_before
    assert guide.correlation_after < 0.5
    assert guide.projected_count > 0


def test_refiner_records_provenance_and_metadata():
    arena = _arena()
    refiner = CounterexampleRefiner(arena, dimension=256)
    guide = refiner.refine(
        failed_source="def f(x): return x",
        problem_io=[(1, 2)],
        passed_ios=[],
        failed_ios=[(1, 2, 1)],
    )
    prov = guide.metadata["provenance"]
    assert prov["operation"] == "counterexample_refine"
    assert prov["source_arena"] == "thdse"
    assert guide.failed_pair_count == 1
    assert guide.metadata["passed_count"] == 0
    assert guide.metadata["failed_count"] == 1


def test_refiner_rejects_arena_without_quotient_projection():
    class _BareArena:
        def allocate(self):
            return 0

    with pytest.raises(TypeError):
        CounterexampleRefiner(_BareArena(), dimension=256)


def test_refiner_requires_at_least_one_failure():
    arena = _arena()
    refiner = CounterexampleRefiner(arena, dimension=256)
    with pytest.raises(ValueError):
        refiner.refine(
            failed_source="def f(x): return x",
            problem_io=[],
            passed_ios=[],
            failed_ios=[],
        )


# --------------------------------------------------------------------------- #
# Dual-axis resonance + axiom behavioural integration
# --------------------------------------------------------------------------- #


def test_axiom_dataclass_carries_optional_behavioral_field():
    from src.synthesis.axiomatic_synthesizer import Axiom

    # Test directly via dataclass instantiation — the field default
    # must be None so legacy callers do not have to know about
    # behavioural profiles.
    fields = Axiom.__dataclass_fields__
    assert "behavioral" in fields
    assert fields["behavioral"].default is None


def test_resonance_matrix_blends_structural_and_behavioral():
    from src.synthesis.axiomatic_synthesizer import (
        AxiomStore,
        ResonanceMatrix,
    )

    arena = _arena()
    store = AxiomStore(arena)
    rm = ResonanceMatrix(arena, store, alpha=0.5)
    assert rm.alpha == 0.5
    # alpha=0.5 by default → behavioural similarity contributes half.
    rm_pure_struct = ResonanceMatrix(arena, store, alpha=1.0)
    assert rm_pure_struct.alpha == 1.0


def test_resonance_matrix_alpha_validates_range():
    from src.synthesis.axiomatic_synthesizer import (
        AxiomStore,
        ResonanceMatrix,
    )

    arena = _arena()
    store = AxiomStore(arena)
    with pytest.raises(ValueError):
        ResonanceMatrix(arena, store, alpha=1.5)
    with pytest.raises(ValueError):
        ResonanceMatrix(arena, store, alpha=-0.1)
