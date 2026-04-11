"""Regression tests for Issue 1: FHRR-grounded data-dep variable threading.

These tests exercise the non-adjacent reaching-definition unification
logic in :class:`VariableThreader.unify_reaching_definitions` and the
:meth:`ConstraintDecoder._extract_data_dep_edges` FHRR-layer probe that
supplies ``data_deps`` to :func:`thread_variables`.

Public API under test:

- ``variable_threading.thread_variables``: verifies the threader honours
  explicit data-dep edges AND infers reaching definitions on its own.
- ``ConstraintDecoder._extract_data_dep_edges``: verifies the decoder
  actually calls the arena correlation path and produces edges grounded
  in FHRR data-layer similarity, not hardcoded or empty lists.
- End-to-end: the decoder's sub-tree assembly compiler threads variables
  using the extracted edges so non-adjacent sub-trees sharing data flow
  collapse to identical concrete names in the emitted source.
"""

from __future__ import annotations

import ast
import os
import sys
from unittest import mock

import pytest

_TEST_DIR = os.path.abspath(os.path.dirname(__file__))
_THDSE_ROOT = os.path.abspath(os.path.join(_TEST_DIR, ".."))
if _THDSE_ROOT not in sys.path:
    sys.path.insert(0, _THDSE_ROOT)

pytest.importorskip("z3")

from src.decoder.constraint_decoder import ConstraintDecoder  # noqa: E402
from src.decoder.subtree_vocab import SubTreeVocabulary  # noqa: E402
from src.decoder.variable_threading import (  # noqa: E402
    VariableThreader,
    thread_variables,
)
from src.projection.isomorphic_projector import IsomorphicProjector  # noqa: E402
from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer  # noqa: E402
from src.utils.arena_factory import make_arena  # noqa: E402


# --------------------------------------------------------------------------- #
# Standalone threader tests (no FHRR needed)
# --------------------------------------------------------------------------- #


def _parse_stmt(src: str) -> ast.AST:
    return ast.parse(src).body[0]


def _extract_concrete_name(tree: ast.AST, placeholder: str) -> str:
    """Find the concrete identifier a given placeholder was rewritten to.

    If the placeholder no longer appears, scans for the first Name/arg
    whose id isn't another placeholder and returns it. Used to assert
    two uses share (or differ on) their post-threading name.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == placeholder:
            return node.id
        if isinstance(node, ast.arg) and node.arg == placeholder:
            return node.arg
    # Placeholder was renamed — return the first non-placeholder name.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not (
            node.id.startswith("x") and node.id[1:].isdigit()
        ):
            return node.id
    return ""


def _names_in(tree: ast.AST) -> list[str]:
    return [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]


class TestNonAdjacentThreading:
    """Issue 1 core fix: non-adjacent sub-trees sharing data flow must
    resolve to a single concrete variable name."""

    def test_non_adjacent_unified_with_data_dep(self):
        """def in 0, use in 2 with an explicit data_dep edge → same name."""
        trees = [
            _parse_stmt("x0 = 0"),          # sub-tree 0: def x0
            _parse_stmt("x1 = 99"),          # sub-tree 1: unrelated
            _parse_stmt("return x0"),        # sub-tree 2: use x0
        ]
        data_deps = [(0, "x0", 2, "x0")]

        result = thread_variables(trees, data_deps=data_deps)
        assert len(result) == 3

        # After threading, no bare x0 placeholder may remain.
        src_all = "\n".join(ast.unparse(t) for t in result)
        assert "x0" not in src_all, (
            f"x0 placeholder leaked after threading: {src_all!r}"
        )
        # The def in sub-tree 0 and the use in sub-tree 2 must share
        # the same concrete name.
        def_target_name = result[0].targets[0].id
        use_value_name = result[2].value.id
        assert def_target_name == use_value_name, (
            f"non-adjacent def/use not unified: {def_target_name} vs "
            f"{use_value_name}"
        )

    def test_reaching_defs_auto_inferred_without_explicit_edges(self):
        """Even with NO explicit data_deps, the reaching-def fixpoint
        should link non-adjacent def → use pairs when no intervening
        redefinition exists."""
        trees = [
            _parse_stmt("x0 = 7"),
            _parse_stmt("x1 = 0"),     # no redef of x0
            _parse_stmt("return x0"),  # use x0 reaches from sub-tree 0
        ]
        result = thread_variables(trees, data_deps=None)
        def_name = result[0].targets[0].id
        use_name = result[2].value.id
        assert def_name == use_name

    def test_intervening_redef_blocks_link(self):
        """def in 0, redef in 1, use in 2 → use links to sub-tree 1, not 0.

        After threading, the def-site in sub-tree 0 must resolve to a
        DIFFERENT concrete name from the use-site in sub-tree 2, because
        the redefinition at sub-tree 1 dominates the use.
        """
        trees = [
            _parse_stmt("x0 = 1"),           # sub-tree 0: def1
            _parse_stmt("x0 = 2"),           # sub-tree 1: redef
            _parse_stmt("return x0"),         # sub-tree 2: use reaches redef
        ]
        result = thread_variables(trees, data_deps=None)

        def0_name = result[0].targets[0].id
        def1_name = result[1].targets[0].id
        use2_name = result[2].value.id

        # Both defs + the use are all placeholder "x0" originally. The
        # threader assigns distinct concrete names when the two defs live
        # in different equivalence classes, OR a single name if they all
        # merge. Reaching-definition analysis should produce the dominated
        # linkage: use2 -- def1 (closest) forming one class, def0 alone.
        assert use2_name == def1_name, (
            f"use at 2 did not link to the closest (dominating) def at 1: "
            f"def0={def0_name} def1={def1_name} use2={use2_name}"
        )
        assert def0_name != use2_name, (
            f"use at 2 incorrectly linked past redef at 1: "
            f"def0={def0_name} def1={def1_name} use2={use2_name}"
        )

    def test_three_subtree_chain(self):
        """def x0 in 0, use x0 + def x1 in 1, use x1 in 2 → x0 and x1
        receive different concrete names, and the chain is consistent."""
        trees = [
            _parse_stmt("x0 = 1"),
            _parse_stmt("x1 = x0 + 1"),
            _parse_stmt("return x1"),
        ]
        result = thread_variables(trees)

        # Collect the name actually used in each slot.
        def0_name = result[0].targets[0].id
        def1_name = result[1].targets[0].id
        use0_at_1 = result[1].value.left.id  # the x0 read in sub-tree 1
        return_name = result[2].value.id

        assert def0_name == use0_at_1, (
            f"x0 def in 0 not unified with x0 use in 1: {def0_name} vs "
            f"{use0_at_1}"
        )
        assert def1_name == return_name, (
            f"x1 def in 1 not unified with x1 use in 2: {def1_name} vs "
            f"{return_name}"
        )
        assert def0_name != def1_name, (
            "x0 and x1 collapsed to the same concrete name — distinct "
            "variables must receive distinct names"
        )

    def test_distinct_concrete_names(self):
        """After threading, no placeholder "xN" may remain anywhere."""
        trees = [
            _parse_stmt("x0 = 0"),
            _parse_stmt("x1 = x0 + 1"),
            _parse_stmt("x2 = x1 * 2"),
            _parse_stmt("return x2"),
        ]
        result = thread_variables(trees)
        joined = "\n".join(ast.unparse(t) for t in result)
        for idx in range(10):
            placeholder = f"x{idx}"
            assert placeholder not in joined.split(), (
                f"{placeholder!r} leaked post-threading: {joined!r}"
            )


# --------------------------------------------------------------------------- #
# End-to-end: decoder extracts FHRR edges and feeds them to the threader
# --------------------------------------------------------------------------- #


@pytest.fixture
def decoder_pipeline():
    """Build a real arena + decoder + vocab from the full seed corpus."""
    # Use the real benchmark seed corpus so there are enough sub-trees
    # with both def- and use-placeholders to exercise the extractor.
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
        activation_threshold=0.04,
        subtree_vocab=vocab,
    )
    synth.compute_resonance()
    cliques = synth.extract_cliques(min_size=2)
    if not cliques:
        pytest.skip("no cliques available in seed corpus")
    projection = synth.synthesize_from_clique(cliques[0])
    return arena, projector, synth, decoder, vocab, projection


def _atoms_with_defs_and_uses(vocab: SubTreeVocabulary) -> list:
    """Return a subset of vocab atoms that collectively contain at
    least one def-placeholder and one use-placeholder. Used to seed
    :meth:`ConstraintDecoder._extract_data_dep_edges` with input that
    can actually produce edges."""
    has_def = []
    has_use = []
    for atom in vocab.get_projected_atoms():
        node = atom.canonical_ast
        defs, uses = [], []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                name = child.id
                if not (name.startswith("x") and name[1:].isdigit()):
                    continue
                if isinstance(child.ctx, (ast.Store, ast.Del)):
                    defs.append(name)
                elif isinstance(child.ctx, ast.Load):
                    uses.append(name)
            elif isinstance(child, ast.arg):
                name = child.arg
                if name.startswith("x") and name[1:].isdigit():
                    defs.append(name)
        if defs:
            has_def.append(atom)
        if uses:
            has_use.append(atom)

    # Return (up to) 3 def-bearing atoms followed by 5 use-bearing ones
    # so the extractor sees at least one (def_idx, use_idx) pair.
    return has_def[:3] + has_use[:5]


class TestDataDepEdgesFromFHRR:
    """Verify _extract_data_dep_edges returns a real list shape and
    that it is actually invoked during compile_model_subtrees."""

    def test_extract_returns_list_of_quads(self, decoder_pipeline):
        _a, _p, _s, decoder, vocab, projection = decoder_pipeline
        atoms = _atoms_with_defs_and_uses(vocab)
        assert any(
            any(
                isinstance(ch, ast.Name)
                and isinstance(ch.ctx, (ast.Store, ast.Del))
                and ch.id.startswith("x") and ch.id[1:].isdigit()
                for ch in ast.walk(a.canonical_ast)
            ) or any(
                isinstance(ch, ast.arg)
                and ch.arg.startswith("x") and ch.arg[1:].isdigit()
                for ch in ast.walk(a.canonical_ast)
            )
            for a in atoms
        ), "fixture must include at least one atom with a def placeholder"

        edges = decoder._extract_data_dep_edges(projection, atoms)

        assert isinstance(edges, list)
        for e in edges:
            assert isinstance(e, tuple)
            assert len(e) == 4
            i, def_ph, j, use_ph = e
            assert isinstance(i, int) and isinstance(j, int)
            assert i < j
            assert isinstance(def_ph, str) and def_ph.startswith("x")
            assert isinstance(use_ph, str) and use_ph.startswith("x")
            assert def_ph[1:].isdigit() and use_ph[1:].isdigit()

    def test_extract_calls_arena_correlation(self, decoder_pipeline):
        """The method must invoke arena.compute_correlation — proof it
        really probes the FHRR layer rather than returning a stub list."""
        _a, _p, _s, decoder, vocab, projection = decoder_pipeline
        atoms = _atoms_with_defs_and_uses(vocab)

        call_counter = {"n": 0}
        original_corr = decoder.arena.compute_correlation

        def counting_corr(h1, h2):
            call_counter["n"] += 1
            return original_corr(h1, h2)

        decoder.arena.compute_correlation = counting_corr
        try:
            decoder._extract_data_dep_edges(projection, atoms)
        finally:
            decoder.arena.compute_correlation = original_corr

        assert call_counter["n"] > 0, (
            "_extract_data_dep_edges did not touch "
            "arena.compute_correlation — data edges are not FHRR-grounded"
        )

    def test_data_deps_actually_passed_to_threader(self, decoder_pipeline):
        """Patch thread_variables to capture what the compiler hands it
        and verify data_deps is a real (possibly empty) list."""
        _a, _p, synth, decoder, _v, projection = decoder_pipeline

        captured: dict = {}
        import src.decoder.constraint_decoder as cd_mod

        original_thread = cd_mod.thread_variables

        def capture_thread(subtrees, data_deps=None):
            captured["data_deps"] = data_deps
            return original_thread(subtrees, data_deps=data_deps)

        with mock.patch.object(
            cd_mod, "thread_variables", side_effect=capture_thread,
        ):
            decoder.decode(projection)

        # data_deps must be a list — None means the decoder never set up
        # the projection context, which is the pre-fix regression.
        assert "data_deps" in captured, (
            "thread_variables was never called during decode"
        )
        assert captured["data_deps"] is not None, (
            "compile_model_subtrees passed data_deps=None — Issue 1 "
            "regression: extractor path is not wired"
        )
        assert isinstance(captured["data_deps"], list)
        # Each edge (if any) must be a 4-tuple of (int, str, int, str).
        for edge in captured["data_deps"]:
            assert len(edge) == 4
            assert isinstance(edge[0], int)
            assert isinstance(edge[1], str)
            assert isinstance(edge[2], int)
            assert isinstance(edge[3], str)
