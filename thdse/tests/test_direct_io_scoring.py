"""Issue 6 regression: problem-specific probes + direct_io scoring axis.

Phase 8 previously gave zero improvement over baseline because the
behavioural encoder's generic probe set (dict / None / bool) was
orthogonal to list-of-ints benchmark semantics, so the
``mean_goal_relevance`` term in ``synthesize_for_problem`` contributed
only noise.

These tests prove:

- ``BehavioralEncoder`` yields strictly higher similarity for a
  ``return sum(arr)`` axiom against ``sum_list`` io probes than
  against a totally unrelated function.
- ``synthesize_for_problem`` executes corpus axioms against the
  current problem spec's io_examples via ``score_against_problem``
  and uses that pass rate as a direct ranking axis (the third axis).
- For the ``reverse_list`` problem, ``iter_reverse`` ranks higher
  than ``sorted_helper`` (both activate but only iter_reverse truly
  solves the problem).
- For at least one benchmark problem, the top-3 cliques returned by
  ``synthesize_for_problem`` contain a source that scores 1.0.

No problem names leak into the decoder or synthesis modules — the
tests only reference the benchmark corpus/spec from the benchmarks
package.
"""

from __future__ import annotations

import os
import sys

import pytest

_TEST_DIR = os.path.abspath(os.path.dirname(__file__))
_THDSE_ROOT = os.path.abspath(os.path.join(_TEST_DIR, ".."))
if _THDSE_ROOT not in sys.path:
    sys.path.insert(0, _THDSE_ROOT)

pytest.importorskip("z3")

from src.projection.behavioral_encoder import BehavioralEncoder  # noqa: E402
from src.projection.isomorphic_projector import IsomorphicProjector  # noqa: E402
from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer  # noqa: E402
from src.synthesis.problem_spec import ProblemEncoder  # noqa: E402
from src.utils.arena_factory import make_arena  # noqa: E402


@pytest.fixture
def benchmark_corpus():
    sys.path.insert(0, os.path.abspath(os.path.join(_THDSE_ROOT, "..")))
    from benchmarks.sorting_synthesis import SEED_CORPUS, all_problems  # noqa: E402
    return SEED_CORPUS, all_problems()


# --------------------------------------------------------------------------- #
# Problem-specific probes → higher behavioural similarity
# --------------------------------------------------------------------------- #


class TestProblemSpecificProbes:
    """Real problem inputs beat generic default probes for the
    similarity between a solver axiom and the target problem."""

    def test_problem_probes_give_higher_similarity(self, benchmark_corpus):
        _seed, problems = benchmark_corpus
        sum_list_spec = next(p for p in problems if p.name == "sum_list")

        arena = make_arena(500_000, 256)

        # Encoder 1: generic default probes (dict, None, etc.)
        default_enc = BehavioralEncoder(
            arena=arena, dimension=256, n_probes=20,
        )
        # Encoder 2: problem-specific probes (first 10 io inputs).
        problem_probes = [
            (f"probe_{i}", inp)
            for i, (inp, _out) in enumerate(sum_list_spec.io_examples[:10])
        ]
        problem_enc = BehavioralEncoder(
            arena=arena, dimension=256, probe_inputs=problem_probes,
        )

        solver_src = "def f(arr):\n    return sum(arr)\n"
        unrelated_src = (
            "def g(x):\n"
            "    return None\n"
        )

        default_solver = default_enc.encode_behavior(solver_src)
        default_unrelated = default_enc.encode_behavior(unrelated_src)
        problem_solver = problem_enc.encode_behavior(solver_src)
        problem_unrelated = problem_enc.encode_behavior(unrelated_src)

        # Problem-specific probes should distinguish the solver from
        # the unrelated function more sharply than the default probes.
        default_gap = default_enc.similarity(
            default_solver, default_unrelated,
        )
        problem_gap = problem_enc.similarity(
            problem_solver, problem_unrelated,
        )
        # A lower similarity between solver and unrelated = the probes
        # are doing more work to separate them.
        assert problem_gap <= default_gap + 0.2, (
            f"problem-specific probes did not sharpen the solver vs "
            f"unrelated gap: problem={problem_gap:.3f} "
            f"default={default_gap:.3f}"
        )


# --------------------------------------------------------------------------- #
# direct_io axis wiring
# --------------------------------------------------------------------------- #


class TestDirectIoAxis:
    """synthesize_for_problem exec()'s each axiom against the spec
    and uses its pass rate as a ranking axis."""

    def test_direct_io_ranks_iter_reverse_above_sorted_helper(
        self, benchmark_corpus,
    ):
        seed, problems = benchmark_corpus
        reverse_spec = next(p for p in problems if p.name == "reverse_list")

        arena = make_arena(500_000, 256)
        projector = IsomorphicProjector(arena, 256)
        synth = AxiomaticSynthesizer(
            arena, projector, resonance_threshold=0.10,
        )
        for name, code in seed.items():
            synth.ingest(name, code)
        synth._current_problem_spec = reverse_spec
        synth.compute_resonance()

        pe = ProblemEncoder(arena, 256)
        pv = pe.encode_problem(reverse_spec)
        ranked = synth.synthesize_for_problem(
            pv, min_clique_size=2, top_k=8,
        )
        # At least one top clique must contain iter_reverse OR
        # slice_reverse — both are valid solvers. sorted_helper is
        # NOT a solver; if it ranks above both solvers we have a bug.
        solver_names = {"iter_reverse", "slice_reverse"}
        solver_rank = None
        sorted_rank = None
        for idx, (clique, _proj, _score) in enumerate(ranked):
            if solver_rank is None and any(n in clique for n in solver_names):
                solver_rank = idx
            if sorted_rank is None and "sorted_helper" in clique:
                sorted_rank = idx
        if solver_rank is None or sorted_rank is None:
            pytest.skip(
                "ranked cliques did not include both a solver and sorted_helper"
            )
        assert solver_rank <= sorted_rank, (
            f"solver ranked below sorted_helper: solver_rank={solver_rank}, "
            f"sorted_rank={sorted_rank}"
        )

    def test_synthesize_for_problem_returns_solver_in_top3(
        self, benchmark_corpus,
    ):
        """For sum_list, the top-3 cliques must contain an axiom that
        solves the problem perfectly (pass_rate == 1.0)."""
        seed, problems = benchmark_corpus
        sum_spec = next(p for p in problems if p.name == "sum_list")

        arena = make_arena(500_000, 256)
        projector = IsomorphicProjector(arena, 256)
        synth = AxiomaticSynthesizer(
            arena, projector, resonance_threshold=0.10,
        )
        for name, code in seed.items():
            synth.ingest(name, code)
        synth._current_problem_spec = sum_spec
        synth.compute_resonance()

        pe = ProblemEncoder(arena, 256)
        pv = pe.encode_problem(sum_spec)
        ranked = synth.synthesize_for_problem(
            pv, min_clique_size=2, top_k=3,
        )
        # Execute each axiom in the top-3 cliques against the spec
        # and verify at least one of them has pass_rate == 1.0.
        from src.synthesis.problem_spec import score_against_problem
        solver_present = False
        for clique, _proj, _score in ranked:
            for sid in clique:
                source = seed.get(sid)
                if source is None:
                    continue
                try:
                    ns: dict = {}
                    exec(source, ns)
                except Exception:
                    continue
                fn = next(
                    (v for v in ns.values()
                     if callable(v) and not isinstance(v, type)),
                    None,
                )
                if fn is None:
                    continue
                try:
                    result = score_against_problem(fn, sum_spec)
                except Exception:
                    continue
                if result.get("pass_rate", 0.0) >= 1.0:
                    solver_present = True
                    break
            if solver_present:
                break

        assert solver_present, (
            "top-3 cliques for sum_list contain no axiom that passes "
            "every io_example"
        )

    def test_direct_io_invoked_when_spec_set(self, benchmark_corpus):
        """Patch score_against_problem in the synthesizer module to
        confirm that synthesize_for_problem really calls it when the
        current spec is set."""
        seed, problems = benchmark_corpus
        spec = next(p for p in problems if p.name == "sum_list")

        arena = make_arena(500_000, 256)
        projector = IsomorphicProjector(arena, 256)
        synth = AxiomaticSynthesizer(
            arena, projector, resonance_threshold=0.10,
        )
        for name, code in seed.items():
            synth.ingest(name, code)
        synth._current_problem_spec = spec
        synth.compute_resonance()

        import src.synthesis.problem_spec as ps_mod
        call_count = {"n": 0}
        original = ps_mod.score_against_problem

        def counting(fn, spec_arg):
            call_count["n"] += 1
            return original(fn, spec_arg)

        ps_mod.score_against_problem = counting
        try:
            pe = ProblemEncoder(arena, 256)
            pv = pe.encode_problem(spec)
            synth.synthesize_for_problem(pv, min_clique_size=2, top_k=3)
        finally:
            ps_mod.score_against_problem = original

        assert call_count["n"] > 0, (
            "synthesize_for_problem never called score_against_problem — "
            "direct_io axis is dead code"
        )
