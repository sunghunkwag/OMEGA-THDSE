"""Regression tests for Issue 4: compile_model_subtrees FunctionDef assembly.

The pre-fix assembly logic would place non-FunctionDef statements after
a self-contained FunctionDef as module-level siblings, producing
outputs such as::

    def synthesized_fn(x):
        return sum(x)
    return y[-0]       # ← stray top-level return
    z = n[0]           # ← stray top-level assign

These stray top-level statements crashed at import time; the execution
sandbox masked the bug by extracting only FunctionDef nodes and
re-executing. These tests prove the new assembly never emits a Module
whose top level contains anything other than a FunctionDef / ClassDef /
Import, via real AST walks — no string regexes.
"""

from __future__ import annotations

import ast
import os
import sys

import pytest

_TEST_DIR = os.path.abspath(os.path.dirname(__file__))
_THDSE_ROOT = os.path.abspath(os.path.join(_TEST_DIR, ".."))
if _THDSE_ROOT not in sys.path:
    sys.path.insert(0, _THDSE_ROOT)

pytest.importorskip("z3")

from src.decoder.constraint_decoder import ConstraintDecoder  # noqa: E402
from src.decoder.subtree_vocab import SubTreeVocabulary  # noqa: E402
from src.projection.isomorphic_projector import IsomorphicProjector  # noqa: E402
from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer  # noqa: E402
from src.utils.arena_factory import make_arena  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_TOP_LEVEL_OK = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Import, ast.ImportFrom,
)


def _collect_top_level_non_function(module: ast.Module) -> list:
    return [s for s in module.body if not isinstance(s, _TOP_LEVEL_OK)]


def _all_returns_inside_function(module: ast.Module) -> bool:
    """Walk the module and check every Return is lexically inside a
    FunctionDef or AsyncFunctionDef."""
    for top_stmt in module.body:
        if isinstance(top_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(top_stmt, ast.ClassDef):
            continue
        for child in ast.walk(top_stmt):
            if isinstance(child, ast.Return):
                return False
    return True


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def full_decoder():
    sys.path.insert(0, os.path.abspath(os.path.join(_THDSE_ROOT, "..")))
    from benchmarks.sorting_synthesis import SEED_CORPUS  # noqa: E402

    arena = make_arena(500_000, 256)
    projector = IsomorphicProjector(arena, 256)
    synth = AxiomaticSynthesizer(
        arena, projector, resonance_threshold=0.10,
    )
    for name, code in SEED_CORPUS.items():
        synth.ingest(name, code)

    vocab = SubTreeVocabulary()
    for code in SEED_CORPUS.values():
        vocab.ingest_source(code)
    vocab.project_all(arena, projector)

    decoder = ConstraintDecoder(
        arena, projector, 256,
        activation_threshold=0.10,
        subtree_vocab=vocab,
    )
    synth.compute_resonance()
    cliques = synth.extract_cliques(min_size=2)
    if not cliques:
        pytest.skip("no cliques")
    return arena, projector, synth, decoder, vocab, cliques


# --------------------------------------------------------------------------- #
# Core assembly invariants
# --------------------------------------------------------------------------- #


class TestNoStrayTopLevelStatements:
    """The Module emitted by compile_model_subtrees must never have
    any top-level statement that isn't a FunctionDef/ClassDef/Import."""

    def test_beam_outputs_have_no_stray_statements(self, full_decoder):
        _a, _p, synth, decoder, _v, cliques = full_decoder
        io = [([1, 2, 3], 6), ([], 0)] * 5
        seen_any = False
        for clique in cliques[:3]:
            projection = synth.synthesize_from_clique(clique)
            source, _rate = decoder.beam_decode(projection, io, beam_width=5)
            if source is None:
                continue
            seen_any = True
            module = ast.parse(source)
            strays = _collect_top_level_non_function(module)
            assert strays == [], (
                f"stray top-level statements: "
                f"{[ast.dump(s) for s in strays]} — source={source!r}"
            )
        assert seen_any, "beam_decode never produced a candidate to inspect"

    def test_return_always_inside_function(self, full_decoder):
        _a, _p, synth, decoder, _v, cliques = full_decoder
        io = [([1, 2, 3], 6), ([], 0)] * 5
        checked = 0
        for clique in cliques[:3]:
            projection = synth.synthesize_from_clique(clique)
            source, _ = decoder.beam_decode(projection, io, beam_width=5)
            if source is None:
                continue
            checked += 1
            module = ast.parse(source)
            assert _all_returns_inside_function(module), (
                f"Return escaped FunctionDef: {source!r}"
            )
        assert checked >= 1

    def test_for_and_while_always_inside_function(self, full_decoder):
        _a, _p, synth, decoder, _v, cliques = full_decoder
        io = [([1, 2, 3], 6), ([], 0)] * 5
        inspected = 0
        for clique in cliques[:3]:
            projection = synth.synthesize_from_clique(clique)
            source, _ = decoder.beam_decode(projection, io, beam_width=5)
            if source is None:
                continue
            inspected += 1
            module = ast.parse(source)
            # Walk top-level non-function nodes and ensure no For/While.
            for top in module.body:
                if isinstance(top, _TOP_LEVEL_OK):
                    continue
                for ch in ast.walk(top):
                    assert not isinstance(ch, (ast.For, ast.While)), (
                        f"For/While escaped FunctionDef: {source!r}"
                    )
        assert inspected >= 1


class TestStrayEvictionWalk:
    """Direct unit test of :meth:`_evict_stray_module_statements`.
    Builds a malformed module by hand and verifies the walker moves
    stray statements into the preceding FunctionDef."""

    def test_walker_moves_stray_return(self, full_decoder):
        _a, _p, _s, decoder, _v, _c = full_decoder
        fn = ast.FunctionDef(
            name="f", args=ast.arguments(
                posonlyargs=[], args=[ast.arg(arg="x")],
                kwonlyargs=[], kw_defaults=[], defaults=[],
            ),
            body=[ast.Return(value=ast.Name(id="x", ctx=ast.Load()))],
            decorator_list=[], returns=None,
        )
        stray = ast.Return(value=ast.Constant(value=42))
        bad_module = ast.Module(body=[fn, stray], type_ignores=[])
        ast.fix_missing_locations(bad_module)

        cleaned = decoder._evict_stray_module_statements(bad_module)
        strays_after = _collect_top_level_non_function(cleaned)
        assert strays_after == [], (
            "stray top-level Return was not evicted"
        )
        # The stray must now live inside fn.body.
        assert any(
            isinstance(s, ast.Return) and isinstance(s.value, ast.Constant)
            and s.value.value == 42
            for s in cleaned.body[0].body
        )

    def test_walker_synthesizes_shell_when_no_preceding_function(
        self, full_decoder,
    ):
        _a, _p, _s, decoder, _v, _c = full_decoder
        stray = ast.Assign(
            targets=[ast.Name(id="x", ctx=ast.Store())],
            value=ast.Constant(value=1),
        )
        bad_module = ast.Module(body=[stray], type_ignores=[])
        ast.fix_missing_locations(bad_module)

        cleaned = decoder._evict_stray_module_statements(bad_module)
        # After cleaning the top level is purely FunctionDef.
        assert all(
            isinstance(s, _TOP_LEVEL_OK) for s in cleaned.body
        )
        assert any(
            isinstance(s, ast.FunctionDef) for s in cleaned.body
        )

    def test_walker_preserves_valid_modules(self, full_decoder):
        _a, _p, _s, decoder, _v, _c = full_decoder
        original_src = (
            "def g(x):\n"
            "    return x + 1\n"
        )
        module = ast.parse(original_src)
        cleaned = decoder._evict_stray_module_statements(module)
        assert ast.unparse(cleaned).strip() == original_src.strip()
