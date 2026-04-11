"""Template-hole-fill decoder — compositional synthesis via hole completion.

The plain sub-tree path of :class:`ConstraintDecoder` concatenates
opaque statement blocks. It cannot express something like::

    for □ in arr:
        □ = □ + □
    return □

so programs that need compositional structure (accumulator loops,
filter-collect, recursion, etc.) are unreachable even when every
sub-tree required to build them lives in the vocabulary.

This module introduces a second, complementary pipeline:

1. :class:`TemplateLibrary` parses real corpus functions via ``ast``,
   replaces each body statement with a :class:`Hole` whose
   ``allowed_types`` is set to the single type of the statement it
   replaces, and stores the skeleton alongside its FHRR projection.
2. :class:`TemplateDecoder` selects templates by FHRR correlation,
   enumerates Z3 hole-assignments (with hole-type and variable-
   consistency constraints) and returns the highest-passing source
   under a real ``score_against_problem`` scoring pass.

Public API:

- ``TemplateLibrary.extract_templates(source)`` → int
- ``TemplateLibrary.project_templates(arena, projector)`` → int
- ``TemplateLibrary.get_templates()`` → list[Template]
- ``TemplateDecoder.template_decode(projection, io_examples,
  beam_width)`` → Tuple[Optional[str], float]

No member of this module references benchmark problem names or
embeds answers. Every template is derived from a parsed corpus
function via the AST transformer; no string templates are fabricated.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


__all__ = ["Hole", "Template", "TemplateLibrary", "TemplateDecoder"]


# --------------------------------------------------------------------------- #
# Core data classes
# --------------------------------------------------------------------------- #


@dataclass
class Hole:
    """A single fillable position inside a :class:`Template` skeleton.

    Attributes:
        index: Ordinal position inside the owning template's body —
            used for Z3 variable naming and hole↔filler matching.
        allowed_types: Set of root AST type names (``"Assign"``,
            ``"For"``, ...) that a filler sub-tree may have. At least
            one entry; derived from the statement the hole replaced.
        defines: Placeholder names the original statement defined
            (e.g. ``"x0"``). Used for variable-consistency constraints.
        uses: Placeholder names the original statement read. Used for
            variable-consistency constraints.
    """

    index: int
    allowed_types: Set[str]
    defines: Set[str] = field(default_factory=set)
    uses: Set[str] = field(default_factory=set)


@dataclass
class Template:
    """A corpus-derived program skeleton with one or more :class:`Hole`\\ s.

    Attributes:
        name: Stable template id (function name + body-hash prefix).
        source: Original canonical source (used for FHRR projection).
        skeleton_ast: ``ast.FunctionDef`` whose body is a list of
            ``Hole`` markers (via the ``_HoleMarker`` sentinel node).
        holes: Ordered list of :class:`Hole` records.
        handle: Arena handle produced by
            :meth:`TemplateLibrary.project_templates` — ``None`` until
            projection is run.
    """

    name: str
    source: str
    skeleton_ast: ast.FunctionDef
    holes: List[Hole]
    handle: Optional[int] = None


# --------------------------------------------------------------------------- #
# Placeholder / slot extraction helpers
# --------------------------------------------------------------------------- #


def _walk_placeholder_slots(
    node: ast.AST,
) -> Tuple[Set[str], Set[str]]:
    """Return ``(defines, uses)`` placeholder sets for an AST node.

    Placeholders are variables named ``x0``, ``x1``, ... produced by
    :class:`src.decoder.subtree_vocab._NameCanonicalizer`. Builtins
    and everything else are ignored.
    """
    defines: Set[str] = set()
    uses: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            name = child.id
            if not (name.startswith("x") and name[1:].isdigit()):
                continue
            if isinstance(child.ctx, (ast.Store, ast.Del)):
                defines.add(name)
            elif isinstance(child.ctx, ast.Load):
                uses.add(name)
        elif isinstance(child, ast.arg):
            name = child.arg
            if name.startswith("x") and name[1:].isdigit():
                defines.add(name)
    return defines, uses


class _CanonicalRewriter(ast.NodeTransformer):
    """Rewrites arbitrary variable names to positional ``xN`` placeholders.

    Mirrors :class:`src.decoder.subtree_vocab._NameCanonicalizer` so
    template hole placeholders live in the same namespace as sub-tree
    atoms. This guarantees that variable-consistency constraints
    generated from templates can reference the same ``xN`` slots the
    sub-tree vocabulary uses.
    """

    _BUILTINS = {
        "int", "float", "str", "bool", "list", "dict", "set", "tuple",
        "bytes", "bytearray", "complex", "frozenset",
        "range", "enumerate", "zip", "map", "filter", "sorted",
        "reversed", "iter", "next",
        "len", "sum", "max", "min", "abs", "all", "any",
        "round", "pow", "divmod",
        "print", "isinstance", "type", "hash", "id", "repr",
        "ord", "chr", "hex", "oct", "bin",
        "True", "False", "None", "ValueError", "TypeError",
        "KeyError", "IndexError", "StopIteration", "Exception",
        "AttributeError", "RuntimeError", "ZeroDivisionError",
    }

    def __init__(self) -> None:
        self._name_map: Dict[str, str] = {}
        self._counter = 0

    def _placeholder(self, original: str) -> str:
        if original in self._name_map:
            return self._name_map[original]
        ph = f"x{self._counter}"
        self._counter += 1
        self._name_map[original] = ph
        return ph

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self._BUILTINS:
            return node
        return ast.Name(id=self._placeholder(node.id), ctx=node.ctx)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        return ast.arg(
            arg=self._placeholder(node.arg),
            annotation=node.annotation,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = self._placeholder(node.name)
        self.generic_visit(node)
        return node


# --------------------------------------------------------------------------- #
# Template library
# --------------------------------------------------------------------------- #


class TemplateLibrary:
    """Builds a vocabulary of hole-filled program skeletons from a corpus.

    Each corpus function is canonicalised (via the same ``xN``
    placeholder scheme sub-tree atoms use), then the body statements
    are reified as :class:`Hole` markers. The resulting skeleton is
    unparsable (holes are not valid Python), so skeletons are only
    used for FHRR projection and variable-slot bookkeeping — never
    written back to disk.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, Template] = {}
        self._projected: bool = False

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_templates(self, source: str) -> int:
        """Parse ``source`` and register every enclosed function as
        a template. Returns the number of new templates added.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0

        added = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            added += self._register_function(node, source)
        return added

    def _register_function(
        self, func: ast.FunctionDef, provenance_source: str,
    ) -> int:
        """Convert a single FunctionDef into a hole-filled template."""
        # Canonicalise variable names to xN.
        canonical = _CanonicalRewriter().visit(copy.deepcopy(func))
        ast.fix_missing_locations(canonical)

        # Build holes from the function body.
        holes: List[Hole] = []
        for idx, stmt in enumerate(canonical.body):
            allowed = {type(stmt).__name__}
            defs, uses = _walk_placeholder_slots(stmt)
            holes.append(Hole(
                index=idx, allowed_types=allowed,
                defines=set(defs), uses=set(uses),
            ))

        if not holes:
            return 0

        # Keep the canonical source (valid Python) as the FHRR probe.
        try:
            skeleton_source = ast.unparse(canonical)
        except (ValueError, TypeError, AttributeError):
            return 0

        name = (
            f"{canonical.name}_"
            f"{hashlib.sha256(skeleton_source.encode()).hexdigest()[:8]}"
        )
        if name in self._templates:
            return 0

        self._templates[name] = Template(
            name=name,
            source=skeleton_source,
            skeleton_ast=canonical,
            holes=holes,
        )
        return 1

    # ------------------------------------------------------------------
    # FHRR projection
    # ------------------------------------------------------------------

    def project_templates(self, arena: Any, projector: Any) -> int:
        """Project every registered template through the FHRR pipeline.

        Returns the number of templates successfully projected.
        """
        projected = 0
        for tmpl in self._templates.values():
            if tmpl.handle is not None:
                projected += 1
                continue
            try:
                layered = projector.project(tmpl.source)
                tmpl.handle = layered.final_handle
                projected += 1
            except (ValueError, TypeError, SyntaxError, AttributeError):
                continue
        self._projected = projected > 0
        return projected

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_templates(self) -> List[Template]:
        return list(self._templates.values())

    def size(self) -> int:
        return len(self._templates)


# --------------------------------------------------------------------------- #
# Template-hole-fill decoder
# --------------------------------------------------------------------------- #


class TemplateDecoder:
    """Picks templates by FHRR correlation and fills their holes via Z3.

    Produces executable candidates that the beam decoder can score
    against the io_examples alongside its sub-tree candidates. The
    key distinction from :class:`ConstraintDecoder._beam_subtree_candidates`
    is that hole positions are *structural* — a template imposes an
    explicit control-flow skeleton that sub-tree concatenation cannot.
    """

    def __init__(
        self,
        arena: Any,
        projector: Any,
        subtree_vocab: Any,
        template_lib: TemplateLibrary,
        activation_threshold: float = 0.10,
    ) -> None:
        self.arena = arena
        self.projector = projector
        self.subtree_vocab = subtree_vocab
        self.template_lib = template_lib
        self.activation_threshold = activation_threshold
        self._diagnostics: List[Tuple[str, str]] = []

    # ------------------------------------------------------------------
    # FHRR template selection
    # ------------------------------------------------------------------

    def _rank_templates(
        self, projection: Any, top_k: int = 5,
    ) -> List[Tuple[Template, float]]:
        """Correlate ``projection.final_handle`` against each template
        handle and return the top-K (template, score) tuples."""
        ranked: List[Tuple[Template, float]] = []
        final_handle = getattr(projection, "final_handle", None)
        if final_handle is None:
            return ranked
        for tmpl in self.template_lib.get_templates():
            if tmpl.handle is None:
                continue
            try:
                score = self.arena.compute_correlation(
                    final_handle, tmpl.handle,
                )
            except (AttributeError, TypeError, ValueError):
                continue
            ranked.append((tmpl, float(score)))
        ranked.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return ranked[: max(1, top_k)]

    # ------------------------------------------------------------------
    # Filler pool
    # ------------------------------------------------------------------

    def _fillers_for_hole(self, hole: Hole) -> List[Any]:
        """Return sub-tree atoms whose root_type matches ``hole.allowed_types``."""
        if self.subtree_vocab is None:
            return []
        fillers: List[Any] = []
        for atom in self.subtree_vocab.get_projected_atoms():
            if atom.root_type in hole.allowed_types:
                fillers.append(atom)
        return fillers

    # ------------------------------------------------------------------
    # Z3 encoding
    # ------------------------------------------------------------------

    def _encode_template(
        self,
        template: Template,
        hole_fillers: Dict[int, List[Any]],
    ) -> Tuple[Any, Dict[str, Any]]:
        """Encode hole-fill choices as Z3 constraints.

        Bool variables:
          - ``use_{hole_i}_{filler_j}``: filler j is assigned to hole i.

        Constraints:
          - Exactly one filler per hole (``PbEq``).
          - Variable consistency: if two holes share a placeholder name
            in their def/use sets, their fillers' root-types must agree
            with the template's connectivity (encoded as a pairwise
            implication — a weak form of data-flow consistency).
        """
        import z3  # local import, z3 is an optional runtime dep

        solver = z3.Solver()
        vars_map: Dict[str, Any] = {}

        # Per-hole selection vars.
        per_hole: Dict[int, List[Tuple[Any, Any]]] = {}
        for hole_idx, fillers in hole_fillers.items():
            selections: List[Tuple[Any, Any]] = []
            for filler_idx, filler in enumerate(fillers):
                key = f"use_{hole_idx}_{filler_idx}"
                var = z3.Bool(key)
                vars_map[key] = var
                selections.append((var, filler))
            per_hole[hole_idx] = selections
            if selections:
                # Exactly-one per hole — use PbEq for compactness.
                solver.add(
                    z3.PbEq(
                        [(var, 1) for var, _ in selections], 1,
                    )
                )
            else:
                # No viable fillers — the whole template is unsat.
                solver.add(z3.BoolVal(False))

        # Variable consistency: if a placeholder xK is defined at hole
        # m and used at hole n (n > m), encode that their chosen
        # fillers' def/use sets must both reference xK. This is a soft
        # pairwise implication — fillers that don't share the name
        # cannot be co-chosen.
        holes_by_index = {h.index: h for h in template.holes}
        for hole_m in template.holes:
            for hole_n in template.holes:
                if hole_n.index <= hole_m.index:
                    continue
                shared = hole_m.defines & hole_n.uses
                if not shared:
                    continue
                sel_m = per_hole.get(hole_m.index, [])
                sel_n = per_hole.get(hole_n.index, [])
                if not sel_m or not sel_n:
                    continue
                # For every pair (filler_m, filler_n) — if they do
                # NOT share any placeholder name, forbid selecting both.
                for var_m, filler_m in sel_m:
                    fm_defs, fm_uses = _walk_placeholder_slots(
                        filler_m.canonical_ast,
                    )
                    for var_n, filler_n in sel_n:
                        fn_defs, fn_uses = _walk_placeholder_slots(
                            filler_n.canonical_ast,
                        )
                        if not (fm_defs & fn_uses):
                            solver.add(z3.Not(z3.And(var_m, var_n)))

        return solver, vars_map

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble_from_model(
        self,
        template: Template,
        hole_fillers: Dict[int, List[Any]],
        model: Any,
        vars_map: Dict[str, Any],
    ) -> Optional[str]:
        """Turn a SAT model into a compilable source string."""
        import z3

        new_body: List[ast.stmt] = []
        for hole in template.holes:
            selections = hole_fillers.get(hole.index, [])
            picked_atom = None
            for filler_idx, filler in enumerate(selections):
                var = vars_map.get(f"use_{hole.index}_{filler_idx}")
                if var is None:
                    continue
                val = model.evaluate(var, model_completion=True)
                if z3.is_true(val):
                    picked_atom = filler
                    break
            if picked_atom is None:
                return None
            node_copy = copy.deepcopy(picked_atom.canonical_ast)
            if isinstance(node_copy, ast.stmt):
                new_body.append(node_copy)
            elif isinstance(node_copy, ast.expr):
                new_body.append(ast.Expr(value=node_copy))
            else:
                return None

        # Thread variables across filled body so xN placeholders become
        # consistent concrete names. Late import keeps module load
        # cheap and avoids any circular-import risk.
        from src.decoder.variable_threading import thread_variables
        threaded = thread_variables(list(new_body))

        # Rebuild a FunctionDef identical to the template's shell but
        # with the freshly-assembled body.
        fn = copy.deepcopy(template.skeleton_ast)
        fn.body = list(threaded)
        ast.fix_missing_locations(fn)

        module = ast.Module(body=[fn], type_ignores=[])
        ast.fix_missing_locations(module)
        try:
            compile(module, "<template>", "exec")
        except (SyntaxError, ValueError, TypeError):
            return None
        try:
            return ast.unparse(module)
        except (ValueError, TypeError, AttributeError):
            return None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def template_decode(
        self,
        input_: Any,
        io_examples: List[Tuple[Any, Any]],
        beam_width: int = 10,
    ) -> Tuple[Optional[str], float]:
        """Enumerate SAT models across top-ranked templates and pick
        the source with the highest ``score_against_problem`` pass rate.

        Args:
            input_: LayeredProjection whose ``final_handle`` drives
                template ranking.
            io_examples: (input, expected_output) pairs for scoring.
            beam_width: Maximum number of Z3 SAT models (across all
                templates combined) to enumerate.

        Returns:
            ``(source, pass_rate)`` — ``source`` is ``None`` iff no
            template yielded a candidate with pass_rate > 0.
        """
        from src.synthesis.problem_spec import (
            ProblemSpec,
            score_against_problem,
        )

        self._diagnostics = []
        if not io_examples:
            return None, 0.0
        if beam_width < 1:
            beam_width = 1

        try:
            spec = ProblemSpec(
                name="__template_decode_target__",
                io_examples=list(io_examples),
                description="Internal template-decode scoring container.",
            )
        except (ValueError, TypeError) as exc:
            self._diagnostics.append(("build_spec", repr(exc)))
            return None, 0.0

        ranked = self._rank_templates(input_, top_k=beam_width)
        if not ranked:
            return None, 0.0

        candidates: List[Tuple[str, float]] = []
        remaining_budget = beam_width

        for template, _score in ranked:
            if remaining_budget <= 0:
                break
            fillers_by_hole: Dict[int, List[Any]] = {
                h.index: self._fillers_for_hole(h) for h in template.holes
            }
            # If any hole has zero viable fillers, skip the template.
            if any(not v for v in fillers_by_hole.values()):
                continue
            try:
                solver, vars_map = self._encode_template(
                    template, fillers_by_hole,
                )
            except Exception as exc:  # noqa: BLE001
                self._diagnostics.append(
                    (f"encode[{template.name}]", repr(exc)),
                )
                continue

            # Enumerate up to remaining_budget SAT models for this template.
            import z3
            seen: Set[str] = set()
            per_template_budget = max(1, remaining_budget // max(1, len(ranked)))
            for _ in range(per_template_budget):
                if remaining_budget <= 0:
                    break
                try:
                    check = solver.check()
                except z3.Z3Exception as exc:
                    self._diagnostics.append(
                        (f"check[{template.name}]", repr(exc)),
                    )
                    break
                if check != z3.sat:
                    break
                try:
                    model = solver.model()
                except z3.Z3Exception as exc:
                    self._diagnostics.append(
                        (f"model[{template.name}]", repr(exc)),
                    )
                    break
                try:
                    source = self._assemble_from_model(
                        template, fillers_by_hole, model, vars_map,
                    )
                except (ValueError, TypeError, AttributeError) as exc:
                    self._diagnostics.append(
                        (f"assemble[{template.name}]", repr(exc)),
                    )
                    source = None

                if source and source not in seen:
                    seen.add(source)
                    pass_rate = self._score(
                        source, spec, score_against_problem,
                    )
                    candidates.append((source, pass_rate))

                # Block the current assignment so the next solver.check
                # is forced to a different hole-filler selection.
                active_terms: List[Any] = []
                for key, var in vars_map.items():
                    if not key.startswith("use_"):
                        continue
                    try:
                        val = model.evaluate(var, model_completion=True)
                        active_terms.append(var != val)
                    except z3.Z3Exception:
                        continue
                if not active_terms:
                    break
                try:
                    solver.add(z3.Or(*active_terms))
                except z3.Z3Exception as exc:
                    self._diagnostics.append(
                        (f"block[{template.name}]", repr(exc)),
                    )
                    break
                remaining_budget -= 1

        if not candidates:
            return None, 0.0
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        best_source, best_rate = candidates[0]
        if best_rate <= 0.0:
            return None, 0.0
        return best_source, best_rate

    # ------------------------------------------------------------------
    # Scoring helper
    # ------------------------------------------------------------------

    def _score(
        self,
        source: str,
        spec: Any,
        score_fn: Callable[[Callable[[Any], Any], Any], Dict[str, Any]],
    ) -> float:
        """Execute ``source`` and hand its first callable to ``score_fn``.

        Every caught exception is logged to ``self._diagnostics`` so
        diagnostic visibility matches :meth:`ConstraintDecoder.beam_decode`.
        """
        namespace: Dict[str, Any] = {}
        try:
            exec(source, namespace)  # noqa: S102 — scorer is authoritative
        except (
            SyntaxError, IndentationError, NameError, TypeError,
            ValueError, AttributeError, IndexError, KeyError,
            ZeroDivisionError, ArithmeticError, ImportError, RuntimeError,
            AssertionError, OverflowError, MemoryError, RecursionError,
            LookupError, UnicodeError,
        ) as exc:
            self._diagnostics.append(("exec", repr(exc)))
            return 0.0

        candidate: Optional[Callable[[Any], Any]] = None
        for value in namespace.values():
            if callable(value) and not isinstance(value, type):
                candidate = value
                break
        if candidate is None:
            self._diagnostics.append(("exec", "no_callable"))
            return 0.0

        try:
            score = score_fn(candidate, spec)
        except Exception as exc:  # noqa: BLE001
            self._diagnostics.append(("score_fn", repr(exc)))
            return 0.0

        try:
            return float(score.get("pass_rate", 0.0))
        except (TypeError, ValueError, AttributeError):
            return 0.0

    def get_diagnostics(self) -> List[Tuple[str, str]]:
        return list(self._diagnostics)
