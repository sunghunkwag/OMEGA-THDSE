"""Tests for the improved legacy statement builder (Issue 3).

Exercises :meth:`ConstraintDecoder._make_statement_node_from_vocab`
and the rewritten :meth:`ConstraintDecoder._make_statement_node`:

- Vocab-backed construction picks a deep-copied canonical sub-tree
  when the sub-tree vocabulary has an atom of the requested type.
- The legacy fallback allocates distinct concrete names via the
  per-decode counter, not every variable collapsing to ``"x"``.
- Expression richness scales with ``present_exprs`` (BinOp yields an
  operator, Subscript yields indexing, etc.).
- The un-parsed output has at most two bare ``x`` occurrences.
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
from src.utils.arena_factory import make_arena  # noqa: E402


def _unparse(node: ast.AST) -> str:
    """Unparse a freshly-constructed AST node by wrapping it in a
    Module and fixing missing locations first — individual stmt
    nodes cannot be unparsed directly without this step in Python 3.11."""
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module)


@pytest.fixture
def decoder_vocab():
    """Build a decoder with a corpus-derived sub-tree vocabulary."""
    sys.path.insert(0, os.path.abspath(os.path.join(_THDSE_ROOT, "..")))
    from benchmarks.sorting_synthesis import SEED_CORPUS  # noqa: E402

    arena = make_arena(500_000, 256)
    projector = IsomorphicProjector(arena, 256)
    vocab = SubTreeVocabulary()
    for code in SEED_CORPUS.values():
        vocab.ingest_source(code)
    vocab.project_all(arena, projector)
    decoder = ConstraintDecoder(
        arena, projector, 256,
        activation_threshold=0.10,
        subtree_vocab=vocab,
    )
    decoder._reset_legacy_namer()
    return decoder, vocab


@pytest.fixture
def decoder_no_vocab():
    """Build a decoder with NO sub-tree vocabulary so the legacy
    fallback is the only code path exercised."""
    arena = make_arena(100_000, 256)
    projector = IsomorphicProjector(arena, 256)
    decoder = ConstraintDecoder(
        arena, projector, 256,
        activation_threshold=0.10,
        subtree_vocab=None,
    )
    decoder._reset_legacy_namer()
    return decoder


# --------------------------------------------------------------------------- #
# Vocab-backed construction
# --------------------------------------------------------------------------- #


class TestVocabBackedConstruction:
    """When the vocab has an atom of the requested type, the decoder
    must deep-copy it — not fabricate a generic placeholder."""

    def test_vocab_backed_statement_uses_real_subtree(self, decoder_vocab):
        decoder, vocab = decoder_vocab
        # The benchmark seed corpus contains several real Return sub-trees
        # such as ``return sum(x0)`` — the vocab-backed builder must pick
        # one of those, not the ``return x`` placeholder emitted by the
        # plain legacy builder.
        ret_atoms = vocab.get_atoms_by_type("Return")
        assert ret_atoms, "corpus must contain Return atoms"

        ret_atoms.sort(key=lambda a: (-a.frequency, a.tree_hash))
        expected_source = ret_atoms[0].canonical_source
        expected_tree = ast.parse(expected_source).body[0]

        node = decoder._make_statement_node_from_vocab("Return", set())
        assert node is not None
        assert isinstance(node, ast.Return)

        # The returned node MUST structurally match the top corpus atom
        # — otherwise the builder silently fell back to the legacy stub
        # even though the vocab had a concrete atom available.
        assert ast.dump(node) == ast.dump(expected_tree), (
            f"vocab-backed Return did not deep-copy the top corpus atom: "
            f"{ast.dump(node)!r} vs expected {ast.dump(expected_tree)!r}"
        )

        # And the unparsed source must be a real expression, not the
        # legacy ``return x`` placeholder template.
        actual_source = _unparse(node).strip()
        assert actual_source.startswith("return "), actual_source
        assert actual_source != "return x", (
            "vocab-backed Return returned the legacy placeholder"
        )

    def test_vocab_backed_deep_copies(self, decoder_vocab):
        """Calling the builder twice must return independent AST
        objects — mutating one cannot affect the vocab atom."""
        decoder, vocab = decoder_vocab
        n1 = decoder._make_statement_node_from_vocab("Return", set())
        n2 = decoder._make_statement_node_from_vocab("Return", set())
        assert n1 is not n2
        # Mutate n1 by replacing its body and verify n2 is unchanged.
        original_dump = ast.dump(n2)
        if isinstance(n1, ast.Return):
            n1.value = ast.Constant(value=42)
        assert ast.dump(n2) == original_dump


# --------------------------------------------------------------------------- #
# Legacy fallback improvements
# --------------------------------------------------------------------------- #


class TestLegacyBuilderImprovements:
    """The legacy builder — now called through the vocab-backed wrapper
    whenever no vocab atom matches — must allocate distinct names and
    use the ``present_exprs`` set to drive expression richness."""

    def test_distinct_variable_names(self, decoder_no_vocab):
        decoder = decoder_no_vocab
        present = {"Name", "Constant", "BinOp", "Compare"}

        # Build three statements without resetting the namer between
        # calls — each kind="var_primary" name lookup returns the same
        # concrete name, but kind="loop_var"/"accum" etc. get distinct
        # names. The rendered source must contain ≥3 unique identifiers.
        stmts = []
        for stype in ("Assign", "For", "Return"):
            node = decoder._make_statement_node(stype, present)
            assert node is not None
            stmts.append(node)

        rendered = "\n".join(_unparse(s) for s in stmts)
        # Collect Name identifiers from the three statements.
        names: set = set()
        for stmt in stmts:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Name):
                    names.add(n.id)
                elif isinstance(n, ast.arg):
                    names.add(n.arg)
        # Exclude the "len"/"range" builtins — they're not variables.
        variable_names = {n for n in names if n not in {"len", "range"}}
        assert len(variable_names) >= 3, (
            f"legacy builder collapsed variables: {variable_names!r} "
            f"rendered={rendered!r}"
        )

    def test_expression_richness_binop(self, decoder_no_vocab):
        """When BinOp is in present_exprs, Return must contain an
        operator, not a bare Name."""
        decoder = decoder_no_vocab
        node = decoder._make_statement_node("Return", {"Name", "BinOp"})
        assert isinstance(node, ast.Return)
        assert node.value is not None
        # BinOp variant
        assert any(
            isinstance(c, (ast.BinOp,)) for c in ast.walk(node)
        ), f"expected BinOp in Return, got {_unparse(node)!r}"

    def test_expression_richness_subscript(self, decoder_no_vocab):
        decoder = decoder_no_vocab
        node = decoder._make_statement_node("Return", {"Name", "Subscript"})
        assert isinstance(node, ast.Return)
        assert any(
            isinstance(c, ast.Subscript) for c in ast.walk(node)
        ), f"expected Subscript in Return, got {_unparse(node)!r}"

    def test_no_all_x_variables(self, decoder_no_vocab):
        """The decoded source must not contain more than two bare "x"
        occurrences — the original regression emitted "x" for every
        variable."""
        decoder = decoder_no_vocab
        present = {"Name", "BinOp", "Compare", "Call", "Subscript"}
        stmts = []
        for stype in ("Assign", "AugAssign", "For", "If", "Return"):
            node = decoder._make_statement_node(stype, present)
            if node is not None:
                stmts.append(node)
        rendered = "\n".join(_unparse(s) for s in stmts)
        # Count "x" as a word-identifier (bounded), ignoring occurrences
        # inside larger names (e.g. "result" doesn't match).
        import re
        occurrences = len(re.findall(r"\bx\b", rendered))
        assert occurrences <= 2, (
            f"legacy source contains {occurrences} bare-x uses (>2): "
            f"{rendered!r}"
        )

    def test_namer_resets_per_compile_model(self, decoder_no_vocab):
        """``_reset_legacy_namer`` must be idempotent and actually
        clear state — after reset, the first allocation must return
        the first pool entry."""
        decoder = decoder_no_vocab
        decoder._reset_legacy_namer()
        n1 = decoder._legacy_fresh_name()
        # Advance the counter a few ticks then reset.
        decoder._legacy_fresh_name()
        decoder._legacy_fresh_name()
        decoder._reset_legacy_namer()
        n2 = decoder._legacy_fresh_name()
        assert n1 == n2, (
            f"namer reset did not restore initial state: {n1!r} vs {n2!r}"
        )

    def test_legacy_fresh_name_advances(self, decoder_no_vocab):
        """Successive calls must return different names from the pool
        until the pool is exhausted."""
        decoder = decoder_no_vocab
        decoder._reset_legacy_namer()
        names = [decoder._legacy_fresh_name() for _ in range(4)]
        assert len(set(names)) == 4, (
            f"namer returned duplicates without wrap-around: {names!r}"
        )
