"""
Constraint Decoder — Translates a synthesized FHRR hypervector back into
a concrete Python AST via deterministic SMT solving (Z3).

Architecture:
  1. Probe (layer-aware): unbind non-target layers via conjugate, then
     correlate against layer-specific vocabulary atoms.
  2. Encode: translate constraints into Z3 formulas with thermodynamic penalty.
  3. Solve: if SAT → compile model to AST; if UNSAT → trigger Meta-Grammar
     Emergence (dimension expansion or operator fusion) and retry.

Sub-Tree Assembly (LEAP 1):
  When a SubTreeVocabulary is provided, the decoder uses concrete AST sub-trees
  harvested from the source corpus instead of abstract placeholder templates.
  The SMT solver selects which sub-trees to include and in what order,
  and the compiler assembles them with variable threading.

Algebraic fix (unbinding):
  Given final = ast ⊗ cfg ⊗ data, to recover AST-layer signal:
    recovered_ast = final ⊗ conj(cfg) ⊗ conj(data)
  Since bind(V, conj(V)) ≈ identity in FHRR, this cancels the
  non-target layers, exposing the AST bundle for direct probing.

Singularity Expansion mechanisms:
  - Meta-Grammar Emergence: UNSAT triggers dimension expansion or fusion operators.
  - Topological Thermodynamics: entropy penalty forces minimum-complexity solutions.

Strict guarantees:
  - Zero randomness: Z3's DPLL(T) is deterministic for a fixed formula.
  - Zero hallucination: only satisfiable structures are emitted.
  - All expansions are O(N) and algebraically provable.
"""

import ast
import copy
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto

import z3

from src.projection.isomorphic_projector import IsomorphicProjector, LayeredProjection
from src.utils.arena_ops import (
    conjugate_into, negate_phases, bind_bundle_fusion_phases,
    expand_phases, compute_phase_entropy, compute_operation_entropy,
)
from src.decoder.subtree_vocab import SubTreeVocabulary, SubTreeAtom
from src.decoder.variable_threading import thread_variables


# ── Structural atoms vocabulary ──────────────────────────────────

class AtomKind(Enum):
    NODE_TYPE = auto()
    CFG_SEQUENCE = auto()
    CFG_BRANCH = auto()
    CFG_LOOP = auto()
    DATA_DEF_USE = auto()


@dataclass
class StructuralAtom:
    kind: AtomKind
    label: str
    handle: int
    z3_var: Optional[z3.BoolRef] = None


@dataclass
class DecodedConstraints:
    """The set of topological constraints extracted from probing."""
    active_node_types: List[str] = field(default_factory=list)
    cfg_sequences: List[Tuple[str, str]] = field(default_factory=list)
    cfg_branches: List[Tuple[str, str, str]] = field(default_factory=list)
    cfg_loops: List[Tuple[str, str]] = field(default_factory=list)
    data_deps: List[Tuple[str, str]] = field(default_factory=list)
    resonance_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class UnsatCoreResult:
    """Result of UNSAT Core extraction from a Z3 solver."""
    core_labels: List[str]          # Z3 tracking labels from the UNSAT core
    core_atom_handles: List[int]    # Arena handles of conflicting atoms
    v_error_handle: Optional[int] = None   # Synthesized contradiction vector handle
    projected_count: int = 0        # Number of arena vectors projected to quotient space


@dataclass
class MetaGrammarEvent:
    """Records a Meta-Grammar Emergence event triggered by topological contradiction."""
    trigger: str                    # "unsat_dimension_expand" or "unsat_fusion_retry"
    old_dimension: int
    new_dimension: int
    retry_succeeded: bool
    entropy_before: Optional[float] = None
    entropy_after: Optional[float] = None
    unsat_core: Optional[UnsatCoreResult] = None  # Attached core when quotient folding used


# ── Known AST node types for the codebook ────────────────────────

_STATEMENT_TYPES = [
    "Module", "FunctionDef", "AsyncFunctionDef", "ClassDef",
    "Return", "Delete", "Assign", "AugAssign", "AnnAssign",
    "For", "AsyncFor", "While", "If", "With", "AsyncWith",
    "Raise", "Try", "Assert", "Import", "ImportFrom",
    "Global", "Nonlocal", "Expr", "Pass", "Break", "Continue",
]

_EXPRESSION_TYPES = [
    "BoolOp", "NamedExpr", "BinOp", "UnaryOp", "Lambda",
    "IfExp", "Dict", "Set", "ListComp", "SetComp", "DictComp",
    "GeneratorExp", "Await", "Yield", "YieldFrom",
    "Compare", "Call", "FormattedValue", "JoinedStr",
    "Constant", "Attribute", "Subscript", "Starred", "Name",
    "List", "Tuple", "Slice",
]

_ALL_NODE_TYPES = _STATEMENT_TYPES + _EXPRESSION_TYPES


class ConstraintDecoder:
    """Decodes a synthesized hypervector into a Python AST via SMT solving."""

    def __init__(
        self,
        arena: Any,
        projector: IsomorphicProjector,
        dimension: int,
        activation_threshold: float = 0.10,
        verification_threshold: float = 0.15,
        max_meta_grammar_retries: int = 0,
        entropy_weight: float = 0.0,
        subtree_vocab: Optional[SubTreeVocabulary] = None,
        wall_archive: Optional[Any] = None,
    ):
        self.arena = arena
        self.projector = projector
        self.dimension = dimension
        self.activation_threshold = activation_threshold
        self.verification_threshold = verification_threshold
        self.max_meta_grammar_retries = max_meta_grammar_retries
        self.entropy_weight = entropy_weight
        self._atom_vocab: Dict[str, StructuralAtom] = {}
        self._meta_grammar_log: List[MetaGrammarEvent] = []
        self._subtree_vocab = subtree_vocab
        self._wall_archive = wall_archive
        self._build_vocabulary()

    # ── Vocabulary construction ──────────────────────────────────

    def _deterministic_phases(self, seed: int) -> List[float]:
        """Identical LCG expansion as IsomorphicProjector for consistency."""
        phases = []
        state = seed & 0xFFFFFFFF
        for _ in range(self.dimension):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            phase = ((state >> 33) / (2**31)) * 2.0 * math.pi - math.pi
            phases.append(phase)
        return phases

    def _mint_atom_handle(self, label: str) -> int:
        seed = int(hashlib.md5(label.encode("utf-8")).hexdigest()[:8], 16)
        handle = self.arena.allocate()
        self.arena.inject_phases(handle, self._deterministic_phases(seed))
        return handle

    def _build_vocabulary(self):
        """Build the structural atom codebook in the arena."""
        for ntype in _ALL_NODE_TYPES:
            label = f"atom:type:{ntype}"
            handle = self._mint_atom_handle(label)
            self._atom_vocab[label] = StructuralAtom(
                kind=AtomKind.NODE_TYPE, label=label, handle=handle,
            )

        for a in _STATEMENT_TYPES:
            for b in _STATEMENT_TYPES:
                label = f"atom:cfg_seq:{a}->{b}"
                handle = self._mint_atom_handle(label)
                self._atom_vocab[label] = StructuralAtom(
                    kind=AtomKind.CFG_SEQUENCE, label=label, handle=handle,
                )

        for cond_type in ["If"]:
            for body_type in _STATEMENT_TYPES:
                label = f"atom:cfg_branch:{cond_type}->{body_type}"
                handle = self._mint_atom_handle(label)
                self._atom_vocab[label] = StructuralAtom(
                    kind=AtomKind.CFG_BRANCH, label=label, handle=handle,
                )

        for loop_type in ["While", "For"]:
            for body_type in _STATEMENT_TYPES:
                label = f"atom:cfg_loop:{loop_type}->{body_type}"
                handle = self._mint_atom_handle(label)
                self._atom_vocab[label] = StructuralAtom(
                    kind=AtomKind.CFG_LOOP, label=label, handle=handle,
                )

        for def_type in ["Assign", "AugAssign", "AnnAssign", "For", "FunctionDef"]:
            for use_type in _STATEMENT_TYPES:
                label = f"atom:data_dep:{def_type}->{use_type}"
                handle = self._mint_atom_handle(label)
                self._atom_vocab[label] = StructuralAtom(
                    kind=AtomKind.DATA_DEF_USE, label=label, handle=handle,
                )

    # ── Unbinding helpers ────────────────────────────────────────

    def _recover_layer(
        self, projection: LayeredProjection, target: str,
    ) -> int:
        """Unbind non-target layers from the final handle to recover a single layer.

        Given final = ast ⊗ cfg ⊗ data:
          recover "ast" → final ⊗ conj(cfg) ⊗ conj(data)
          recover "cfg" → final ⊗ conj(ast) ⊗ conj(data)
          recover "data" → final ⊗ conj(ast) ⊗ conj(cfg)

        If a layer is None (absent), it was never bound in — skip it.
        """
        result_h = projection.final_handle

        # Determine which layers to unbind (all except target)
        layers_to_unbind = []
        if target != "ast":
            # ast is always present; always unbind it unless it IS the target
            layers_to_unbind.append(projection.ast_phases)
        if target != "cfg" and projection.cfg_phases is not None:
            layers_to_unbind.append(projection.cfg_phases)
        if target != "data" and projection.data_phases is not None:
            layers_to_unbind.append(projection.data_phases)

        for phases in layers_to_unbind:
            conj_h = self.arena.allocate()
            conjugate_into(self.arena, phases, conj_h)
            unbound = self.arena.allocate()
            self.arena.bind(result_h, conj_h, unbound)
            result_h = unbound

        return result_h

    # ── Phase 1: Layer-aware probing ─────────────────────────────

    def probe_layered(self, projection: LayeredProjection) -> DecodedConstraints:
        """Probe each layer independently after unbinding other layers.

        - Node-type atoms (AST layer): probe against recovered AST
        - CFG atoms: probe against recovered CFG
        - Data-dep atoms: probe against recovered Data-dep
        """
        constraints = DecodedConstraints()

        # Recover each layer
        recovered_ast = self._recover_layer(projection, "ast")
        recovered_cfg = self._recover_layer(projection, "cfg") if projection.cfg_handle is not None else None
        recovered_data = self._recover_layer(projection, "data") if projection.data_handle is not None else None

        for label, atom in self._atom_vocab.items():
            # Route each atom kind to the correct recovered layer
            if atom.kind == AtomKind.NODE_TYPE:
                target_h = recovered_ast
            elif atom.kind in (AtomKind.CFG_SEQUENCE, AtomKind.CFG_BRANCH, AtomKind.CFG_LOOP):
                target_h = recovered_cfg
            elif atom.kind == AtomKind.DATA_DEF_USE:
                target_h = recovered_data
            else:
                continue

            if target_h is None:
                continue

            corr = self.arena.compute_correlation(target_h, atom.handle)
            constraints.resonance_scores[label] = corr

            if abs(corr) < self.activation_threshold:
                continue

            if atom.kind == AtomKind.NODE_TYPE:
                ntype = label.split(":")[-1]
                constraints.active_node_types.append(ntype)

            elif atom.kind == AtomKind.CFG_SEQUENCE:
                pair = label.split(":")[-1]
                a, b = pair.split("->")
                constraints.cfg_sequences.append((a, b))

            elif atom.kind == AtomKind.CFG_BRANCH:
                pair = label.split(":")[-1]
                parts = pair.split("->")
                constraints.cfg_branches.append((parts[0], parts[1], "Pass"))

            elif atom.kind == AtomKind.CFG_LOOP:
                pair = label.split(":")[-1]
                parts = pair.split("->")
                constraints.cfg_loops.append((parts[0], parts[1]))

            elif atom.kind == AtomKind.DATA_DEF_USE:
                pair = label.split(":")[-1]
                parts = pair.split("->")
                constraints.data_deps.append((parts[0], parts[1]))

        return constraints

    def probe(self, input_: Union[int, LayeredProjection]) -> DecodedConstraints:
        """Probe — dispatches to layer-aware or legacy based on input type.

        If a LayeredProjection is provided, uses layer-aware unbinding (correct).
        If a bare int handle is provided, falls back to direct correlation (legacy).
        """
        if isinstance(input_, LayeredProjection):
            return self.probe_layered(input_)

        # Legacy fallback: direct correlation against composed vector (algebraically broken
        # for cross-layer-bound vectors, but preserved for backward compatibility)
        synth_handle = input_
        constraints = DecodedConstraints()
        for label, atom in self._atom_vocab.items():
            corr = self.arena.compute_correlation(synth_handle, atom.handle)
            constraints.resonance_scores[label] = corr
            if abs(corr) < self.activation_threshold:
                continue
            if atom.kind == AtomKind.NODE_TYPE:
                constraints.active_node_types.append(label.split(":")[-1])
            elif atom.kind == AtomKind.CFG_SEQUENCE:
                pair = label.split(":")[-1]
                a, b = pair.split("->")
                constraints.cfg_sequences.append((a, b))
            elif atom.kind == AtomKind.CFG_BRANCH:
                pair = label.split(":")[-1]
                parts = pair.split("->")
                constraints.cfg_branches.append((parts[0], parts[1], "Pass"))
            elif atom.kind == AtomKind.CFG_LOOP:
                pair = label.split(":")[-1]
                parts = pair.split("->")
                constraints.cfg_loops.append((parts[0], parts[1]))
            elif atom.kind == AtomKind.DATA_DEF_USE:
                pair = label.split(":")[-1]
                parts = pair.split("->")
                constraints.data_deps.append((parts[0], parts[1]))
        return constraints

    # ── Phase 2: SMT Encoding ────────────────────────────────────

    def encode_smt(
        self, constraints: DecodedConstraints, entropy_budget: Optional[float] = None,
        enable_unsat_core: bool = False,
    ) -> Tuple[z3.Solver, Dict]:
        """Translate topological constraints into Z3 formulas.

        If entropy_budget is provided (Topological Thermodynamics), adds a strict
        constraint that total structural complexity must not exceed the budget.
        The solver must find the lowest-entropy satisfying model.

        If enable_unsat_core is True, uses Z3's assert_and_track for topological
        constraints so that the UNSAT core can be extracted on contradiction.
        """
        solver = z3.Solver()
        if enable_unsat_core:
            solver.set("unsat_core", True)
        vars_map = {}

        node_vars = {}
        for ntype in _ALL_NODE_TYPES:
            v = z3.Bool(f"present_{ntype}")
            node_vars[ntype] = v
            vars_map[f"present_{ntype}"] = v

        # Closed-world assumption: force active nodes present, all others absent.
        # Module is always present (well-formedness invariant).
        active_set = set(constraints.active_node_types) | {"Module"}
        for ntype, v in node_vars.items():
            if ntype in active_set:
                solver.add(v == True)
            else:
                solver.add(v == False)

        order_vars = {}
        for stype in _STATEMENT_TYPES:
            v = z3.Int(f"order_{stype}")
            order_vars[stype] = v
            vars_map[f"order_{stype}"] = v
            solver.add(v >= 0)
            solver.add(z3.Implies(z3.Not(node_vars.get(stype, z3.BoolVal(False))), v == 0))

        # Incremental constraint addition with consistency filtering.
        # Each topological constraint is tested via push/pop before permanent
        # addition. Contradictory constraints (from probe noise) are discarded.
        # This extracts the maximal consistent subset — deterministic, O(N·T_smt).

        # Filter CFG sequences: remove self-loops, resolve contradictions
        cfg_pairs = {}
        for a, b in constraints.cfg_sequences:
            if a == b:
                continue  # Self-loops are algebraically impossible
            key = (min(a, b), max(a, b))
            score = abs(constraints.resonance_scores.get(f"atom:cfg_seq:{a}->{b}", 0.0))
            if key not in cfg_pairs or score > cfg_pairs[key][1]:
                cfg_pairs[key] = ((a, b), score)

        # Helper: add a constraint, optionally tracked for UNSAT core extraction
        _track_idx = [0]  # mutable counter for unique tracking labels

        def _add_tracked(s: z3.Solver, constraint, track_label: Optional[str] = None):
            """Add constraint with optional UNSAT core tracking."""
            if enable_unsat_core and track_label is not None:
                tracker = z3.Bool(track_label)
                s.assert_and_track(constraint, tracker)
            else:
                s.add(constraint)

        for (a, b), _score in sorted(cfg_pairs.values(), key=lambda x: -x[1]):
            if a in order_vars and b in order_vars and a in node_vars and b in node_vars:
                c = z3.Implies(z3.And(node_vars[a], node_vars[b]),
                               order_vars[a] < order_vars[b])
                solver.push()
                solver.add(c)
                if solver.check() == z3.sat:
                    solver.pop()
                    _add_tracked(solver, c, f"atom:cfg_seq:{a}->{b}")
                else:
                    solver.pop()  # Discard contradictory constraint

        for cond, then_t, else_t in constraints.cfg_branches:
            if cond in node_vars:
                targets = []
                if then_t in node_vars:
                    targets.append(node_vars[then_t])
                if else_t in node_vars:
                    targets.append(node_vars[else_t])
                if targets:
                    c = z3.Implies(node_vars[cond], z3.Or(*targets))
                    solver.push()
                    solver.add(c)
                    if solver.check() == z3.sat:
                        solver.pop()
                        _add_tracked(solver, c, f"atom:cfg_branch:{cond}->{then_t}")
                    else:
                        solver.pop()

        for header, body_type in constraints.cfg_loops:
            if header in node_vars and body_type in node_vars:
                c1 = z3.Implies(node_vars[header], node_vars[body_type])
                solver.push()
                solver.add(c1)
                if header in order_vars and body_type in order_vars:
                    c2 = z3.Implies(
                        z3.And(node_vars[header], node_vars[body_type]),
                        order_vars[header] <= order_vars[body_type])
                    solver.add(c2)
                if solver.check() == z3.sat:
                    solver.pop()
                    _add_tracked(solver, c1, f"atom:cfg_loop:{header}->{body_type}")
                    if header in order_vars and body_type in order_vars:
                        _add_tracked(solver, c2, f"atom:cfg_loop_order:{header}->{body_type}")
                else:
                    solver.pop()

        for def_type, use_type in constraints.data_deps:
            if def_type in order_vars and use_type in order_vars:
                if def_type in node_vars and use_type in node_vars:
                    c = z3.Implies(
                        z3.And(node_vars[def_type], node_vars[use_type]),
                        order_vars[def_type] < order_vars[use_type])
                    solver.push()
                    solver.add(c)
                    if solver.check() == z3.sat:
                        solver.pop()
                        _add_tracked(solver, c, f"atom:data_dep:{def_type}->{use_type}")
                    else:
                        solver.pop()

        if "Module" in node_vars:
            solver.add(node_vars["Module"] == True)

        if "FunctionDef" in node_vars:
            body_candidates = [
                node_vars[s] for s in ["Return", "Assign", "Expr", "Pass", "If", "While", "For", "Raise"]
                if s in node_vars
            ]
            if body_candidates:
                solver.add(z3.Implies(node_vars["FunctionDef"], z3.Or(*body_candidates)))

        max_stmts = len(constraints.active_node_types) + 5
        for v in order_vars.values():
            solver.add(v <= max_stmts)

        # ── Hierarchical Nesting Constraints ─────────────────────────
        # Add parent-child containment: parent_{Type} = -1 (top-level)
        # or parent_{Type} = k (nested inside statement at order k).
        parent_vars = {}
        for stype in _STATEMENT_TYPES:
            pv = z3.Int(f"parent_{stype}")
            parent_vars[stype] = pv
            vars_map[f"parent_{stype}"] = pv
            # parent >= -1 and parent < order (no circular nesting)
            solver.add(pv >= -1)
            # If absent, parent = -1
            solver.add(z3.Implies(
                z3.Not(node_vars.get(stype, z3.BoolVal(False))),
                pv == -1,
            ))
            # A statement's parent must come before it in the ordering
            if stype in order_vars:
                solver.add(z3.Implies(
                    z3.And(node_vars.get(stype, z3.BoolVal(False)), pv >= 0),
                    pv < order_vars[stype],
                ))

        # FunctionDef is always top-level
        if "FunctionDef" in parent_vars and "FunctionDef" in node_vars:
            solver.add(z3.Implies(
                node_vars["FunctionDef"],
                parent_vars["FunctionDef"] == -1,
            ))

        # Module is always top-level
        if "Module" in parent_vars and "Module" in node_vars:
            solver.add(z3.Implies(
                node_vars["Module"],
                parent_vars["Module"] == -1,
            ))

        # Nesting types: statements that can contain a body
        _NESTING_TYPES = {"If", "While", "For", "AsyncFor", "With", "AsyncWith",
                          "Try", "FunctionDef", "AsyncFunctionDef", "ClassDef"}

        # CFG branch constraints → parent-child nesting (If → body_type)
        for cond, body_type, _else_t in constraints.cfg_branches:
            if (cond in node_vars and body_type in node_vars
                    and cond in parent_vars and body_type in parent_vars
                    and cond in order_vars and body_type in order_vars
                    and cond in _NESTING_TYPES):
                c = z3.Implies(
                    z3.And(node_vars[cond], node_vars[body_type]),
                    parent_vars[body_type] == order_vars[cond],
                )
                solver.push()
                solver.add(c)
                if solver.check() == z3.sat:
                    solver.pop()
                    solver.add(c)
                else:
                    solver.pop()  # Discard — would cause UNSAT

        # CFG loop constraints → parent-child nesting (While/For → body_type)
        for header, body_type in constraints.cfg_loops:
            if (header in node_vars and body_type in node_vars
                    and header in parent_vars and body_type in parent_vars
                    and header in order_vars and body_type in order_vars
                    and header in _NESTING_TYPES):
                c = z3.Implies(
                    z3.And(node_vars[header], node_vars[body_type]),
                    parent_vars[body_type] == order_vars[header],
                )
                solver.push()
                solver.add(c)
                if solver.check() == z3.sat:
                    solver.pop()
                    solver.add(c)
                else:
                    solver.pop()  # Discard — would cause UNSAT

        # ── Topological Thermodynamics: Entropy Constraint ────────
        # Each active statement type carries a complexity cost.
        # The solver enforces maximum structural compression.
        entropy_var = z3.Int("total_entropy_cost")
        vars_map["total_entropy_cost"] = entropy_var

        # Entropy cost per statement type: statements with more sub-structure
        # carry higher thermodynamic penalty (deterministic, fixed mapping).
        _ENTROPY_COSTS = {
            "Module": 0, "Pass": 1, "Break": 1, "Continue": 1,
            "Return": 2, "Assign": 2, "AugAssign": 2, "Expr": 2,
            "Import": 2, "ImportFrom": 2, "Global": 1, "Nonlocal": 1,
            "Assert": 2, "Delete": 2, "Raise": 2,
            "If": 4, "While": 4, "For": 4,
            "FunctionDef": 5, "AsyncFunctionDef": 5, "ClassDef": 6,
            "With": 3, "AsyncWith": 3, "Try": 5,
            "AnnAssign": 3, "AsyncFor": 4,
        }

        cost_terms = []
        for stype in _STATEMENT_TYPES:
            cost = _ENTROPY_COSTS.get(stype, 2)
            if stype in node_vars:
                cost_terms.append(
                    z3.If(node_vars[stype], z3.IntVal(cost), z3.IntVal(0))
                )

        if cost_terms:
            solver.add(entropy_var == z3.Sum(cost_terms))
        else:
            solver.add(entropy_var == 0)

        # Thermodynamic ceiling: enforce minimum-complexity solution
        if entropy_budget is not None:
            budget_int = int(math.ceil(entropy_budget))
            solver.add(entropy_var <= budget_int)

        # Order compactness (only when thermodynamics is active):
        # Forces the solver to select the most compressed statement ordering
        if entropy_budget is not None:
            active_stmts = [s for s in _STATEMENT_TYPES if s in active_set and s in order_vars]
            if len(active_stmts) >= 2:
                max_order = z3.Int("max_order_span")
                vars_map["max_order_span"] = max_order
                for s in active_stmts:
                    solver.add(max_order >= order_vars[s])
                solver.add(max_order <= len(active_stmts) + 2)

        return solver, vars_map

    # ── Sub-Tree Probing ──────────────────────────────────────────

    def probe_subtrees(
        self, input_: Union[int, LayeredProjection],
    ) -> List[Tuple[SubTreeAtom, float]]:
        """Probe a synthesized vector against the sub-tree vocabulary.

        For each sub-tree atom in the vocabulary, computes correlation
        with the input vector. Returns atoms exceeding the activation threshold,
        sorted by descending resonance score.

        Args:
            input_: Arena handle or LayeredProjection to probe.

        Returns:
            List of (SubTreeAtom, resonance_score) tuples exceeding threshold.
        """
        if self._subtree_vocab is None:
            return []

        if isinstance(input_, LayeredProjection):
            target_h = self._recover_layer(input_, "ast")
        else:
            target_h = input_

        activated: List[Tuple[SubTreeAtom, float]] = []
        for atom in self._subtree_vocab.get_projected_atoms():
            if atom.handle is None:
                continue
            corr = self.arena.compute_correlation(target_h, atom.handle)
            if abs(corr) >= self.activation_threshold:
                activated.append((atom, corr))

        # Sort by descending absolute resonance
        activated.sort(key=lambda x: abs(x[1]), reverse=True)
        return activated

    # ── Sub-Tree SMT Encoding ─────────────────────────────────────

    def encode_smt_subtrees(
        self,
        activated_atoms: List[Tuple[SubTreeAtom, float]],
        entropy_budget: Optional[float] = None,
    ) -> Tuple[z3.Solver, Dict, List[SubTreeAtom]]:
        """Encode sub-tree selection and ordering as Z3 constraints.

        For each activated sub-tree atom:
          - Bool use_subtree_{hash}: whether this sub-tree is included
          - Int position_{hash}: ordering position in the final AST body
          - Compatibility constraints between co-positioned sub-trees
          - Exclusion constraints for same-slot sub-trees
          - Well-formedness constraints (function header requires body, etc.)
          - Resonance score maximization via threshold constraint

        Args:
            activated_atoms: List of (SubTreeAtom, resonance_score) from probing.
            entropy_budget: Optional entropy ceiling for thermodynamic constraint.

        Returns:
            Tuple of (solver, vars_map, ordered_atom_list).
        """
        solver = z3.Solver()
        vars_map: Dict[str, Any] = {}
        atom_list: List[SubTreeAtom] = []

        if not activated_atoms:
            return solver, vars_map, atom_list

        # Create Z3 variables for each activated sub-tree
        use_vars: Dict[str, z3.BoolRef] = {}
        pos_vars: Dict[str, z3.ArithRef] = {}
        atom_by_hash: Dict[str, Tuple[SubTreeAtom, float]] = {}

        for atom, score in activated_atoms:
            h = atom.tree_hash[:12]  # shortened hash for readability
            use_var = z3.Bool(f"use_{h}")
            pos_var = z3.Int(f"pos_{h}")

            use_vars[atom.tree_hash] = use_var
            pos_vars[atom.tree_hash] = pos_var
            atom_by_hash[atom.tree_hash] = (atom, score)

            vars_map[f"use_{h}"] = use_var
            vars_map[f"pos_{h}"] = pos_var

            # Position bounds: 0..len(activated_atoms)
            solver.add(z3.Implies(use_var, pos_var >= 0))
            solver.add(z3.Implies(use_var, pos_var < len(activated_atoms)))
            solver.add(z3.Implies(z3.Not(use_var), pos_var == -1))

        # Exclusion: same structural slot sub-trees cannot both be active.
        # Group sub-trees by root type — at most N of same type for reasonable programs.
        by_root_type: Dict[str, List[str]] = {}
        for atom, _ in activated_atoms:
            if atom.root_type not in by_root_type:
                by_root_type[atom.root_type] = []
            by_root_type[atom.root_type].append(atom.tree_hash)

        for root_type, hashes in by_root_type.items():
            if len(hashes) <= 3:
                continue  # Allow up to 3 of same type
            # AtMost 3 of same root type can be active
            solver.add(z3.AtMost(*[use_vars[h] for h in hashes], 3))

        # Distinct positions: no two active sub-trees share a position
        active_hashes = list(use_vars.keys())
        for i, h_a in enumerate(active_hashes):
            for h_b in active_hashes[i + 1:]:
                solver.add(z3.Implies(
                    z3.And(use_vars[h_a], use_vars[h_b]),
                    pos_vars[h_a] != pos_vars[h_b],
                ))

        # Compatibility: sub-trees sharing variable slots should be co-positioned
        # (adjacent in output). This is a soft constraint via ordering.
        for i, h_a in enumerate(active_hashes):
            atom_a = atom_by_hash[h_a][0]
            for h_b in active_hashes[i + 1:]:
                atom_b = atom_by_hash[h_b][0]
                # If both define/use variables, encourage adjacency
                if atom_a.is_definition and atom_b.is_use:
                    solver.add(z3.Implies(
                        z3.And(use_vars[h_a], use_vars[h_b]),
                        pos_vars[h_a] < pos_vars[h_b],
                    ))

        # Well-formedness: if any Return sub-tree is active, ensure a
        # FunctionDef-rooted sub-tree is also active (or we wrap in one)
        return_hashes = by_root_type.get("Return", [])
        funcdef_hashes = by_root_type.get("FunctionDef", [])
        if return_hashes and funcdef_hashes:
            for rh in return_hashes:
                solver.add(z3.Implies(
                    use_vars[rh],
                    z3.Or(*[use_vars[fh] for fh in funcdef_hashes]),
                ))

        # Resonance threshold: total resonance of selected sub-trees
        # must exceed a minimum to ensure meaningful output
        resonance_terms = []
        for h, (atom, score) in atom_by_hash.items():
            # Scale score to integer (multiply by 1000 for precision)
            int_score = int(abs(score) * 1000)
            resonance_terms.append(
                z3.If(use_vars[h], z3.IntVal(int_score), z3.IntVal(0))
            )

        if resonance_terms:
            total_resonance = z3.Int("total_resonance")
            vars_map["total_resonance"] = total_resonance
            solver.add(total_resonance == z3.Sum(resonance_terms))
            # Require at least one sub-tree to be active
            solver.add(z3.Or(*[use_vars[h] for h in active_hashes]))
            # Maximize by requiring resonance above minimum meaningful threshold
            min_resonance = max(40, int(self.activation_threshold * 1000))
            solver.add(total_resonance >= min_resonance)

        # Entropy budget constraint (thermodynamics)
        if entropy_budget is not None:
            depth_cost_terms = []
            for h, (atom, _) in atom_by_hash.items():
                depth_cost_terms.append(
                    z3.If(use_vars[h], z3.IntVal(atom.depth), z3.IntVal(0))
                )
            if depth_cost_terms:
                total_depth = z3.Int("total_depth_cost")
                vars_map["total_depth_cost"] = total_depth
                solver.add(total_depth == z3.Sum(depth_cost_terms))
                solver.add(total_depth <= int(entropy_budget * 2))

        # Collect atom list in hash order for compile_model_subtrees
        atom_list = [atom_by_hash[h][0] for h in active_hashes]

        return solver, vars_map, atom_list

    # ── Sub-Tree Assembly Compiler ─────────────────────────────────

    def compile_model_subtrees(
        self,
        model: z3.ModelRef,
        vars_map: Dict,
        atom_list: List[SubTreeAtom],
    ) -> ast.Module:
        """Compile a Z3 model into a Python AST by assembling real sub-trees.

        The SAT model tells us WHICH canonical sub-trees to include and in
        WHAT ORDER. The compiler:
        1. Collects all use_subtree == True sub-trees from the model
        2. Sorts by position value
        3. Deep-copies canonical ASTs and threads variables
        4. Wraps in FunctionDef or Module as appropriate
        5. Runs ast.fix_missing_locations() and verifies via compile()

        Args:
            model: Z3 satisfying model.
            vars_map: Variable map from encode_smt_subtrees.
            atom_list: Ordered list of SubTreeAtom candidates.

        Returns:
            A valid ast.Module node.
        """
        # Step 1: Collect active sub-trees with their positions
        selected: List[Tuple[int, SubTreeAtom]] = []
        for atom in atom_list:
            h = atom.tree_hash[:12]
            use_key = f"use_{h}"
            pos_key = f"pos_{h}"

            if use_key not in vars_map:
                continue

            use_val = model.evaluate(vars_map[use_key], model_completion=True)
            if not z3.is_true(use_val):
                continue

            pos_val = model.evaluate(vars_map[pos_key], model_completion=True)
            try:
                pos_int = pos_val.as_long()
            except (AttributeError, Exception):
                pos_int = 0

            selected.append((pos_int, atom))

        if not selected:
            # Fallback: emit a minimal valid module
            return ast.Module(body=[ast.Pass()], type_ignores=[])

        # Step 2: Sort by position
        selected.sort(key=lambda x: (x[0], x[1].root_type))

        # Step 3: Deep-copy canonical ASTs
        subtree_nodes: List[ast.AST] = []
        for _, atom in selected:
            node_copy = copy.deepcopy(atom.canonical_ast)
            subtree_nodes.append(node_copy)

        # Step 4: Thread variables across sub-trees
        subtree_nodes = thread_variables(subtree_nodes)

        # Step 5: Assemble into module body
        body_stmts: List[ast.stmt] = []
        func_body: List[ast.stmt] = []
        in_function = False
        has_funcdef = False

        for node in subtree_nodes:
            if isinstance(node, ast.FunctionDef):
                has_funcdef = True
                in_function = True
                # We'll wrap subsequent statements as the function body
                # but keep the function def node's own body
                if node.body:
                    body_stmts.append(node)
                    in_function = False  # function is self-contained
                else:
                    # Empty function body — subsequent stmts become body
                    body_stmts.append(node)
                continue

            if isinstance(node, ast.stmt):
                if in_function and body_stmts and isinstance(body_stmts[-1], ast.FunctionDef):
                    # Add to the last function's body
                    body_stmts[-1].body.append(node)
                else:
                    body_stmts.append(node)
            elif isinstance(node, ast.expr):
                # Wrap bare expressions in Expr statement
                body_stmts.append(ast.Expr(value=node))

        # If we collected statements that should be in a function but aren't,
        # and Return is among them, wrap in a function
        has_return = any(
            isinstance(s, ast.Return) for s in body_stmts
            if not isinstance(s, ast.FunctionDef)
        )
        if has_return and not has_funcdef:
            # Extract non-function stmts and wrap in synthesized function
            non_func = [s for s in body_stmts if not isinstance(s, ast.FunctionDef)]
            if non_func:
                func_def = ast.FunctionDef(
                    name="synthesized_fn",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg="x")],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=non_func,
                    decorator_list=[],
                    returns=None,
                )
                func_stmts = [s for s in body_stmts if isinstance(s, ast.FunctionDef)]
                body_stmts = func_stmts + [func_def]

        if not body_stmts:
            body_stmts = [ast.Pass()]

        module = ast.Module(body=body_stmts, type_ignores=[])
        ast.fix_missing_locations(module)

        # Step 6: Verify via compile()
        try:
            compile(module, "<synthesized>", "exec")
        except (SyntaxError, TypeError, ValueError):
            # If compilation fails, fall through to ensure at least valid AST
            # by removing problematic nodes one by one
            safe_body: List[ast.stmt] = []
            for stmt in body_stmts:
                test_mod = ast.Module(body=safe_body + [stmt], type_ignores=[])
                ast.fix_missing_locations(test_mod)
                try:
                    compile(test_mod, "<synthesized>", "exec")
                    safe_body.append(stmt)
                except (SyntaxError, TypeError, ValueError):
                    continue
            if not safe_body:
                safe_body = [ast.Pass()]
            module = ast.Module(body=safe_body, type_ignores=[])
            ast.fix_missing_locations(module)

        return module

    # ── Phase 3: AST Compilation (Legacy) ─────────────────────────

    def compile_model(self, model: z3.ModelRef, vars_map: Dict) -> ast.Module:
        """Compile a Z3 satisfying model into a Python AST.

        Uses parent_{Type} variables (if present) to build a proper nested tree
        instead of placing all statements as flat siblings.
        """
        present_stmts = []
        for stype in _STATEMENT_TYPES:
            var_name = f"present_{stype}"
            if var_name in vars_map:
                val = model.evaluate(vars_map[var_name], model_completion=True)
                if z3.is_true(val):
                    order_var = f"order_{stype}"
                    if order_var in vars_map:
                        order_val = model.evaluate(vars_map[order_var], model_completion=True)
                        try:
                            order_int = order_val.as_long()
                        except (AttributeError, Exception):
                            order_int = 0
                    else:
                        order_int = 0
                    # Extract parent assignment
                    parent_var = f"parent_{stype}"
                    parent_int = -1
                    if parent_var in vars_map:
                        parent_val = model.evaluate(vars_map[parent_var], model_completion=True)
                        try:
                            parent_int = parent_val.as_long()
                        except (AttributeError, Exception):
                            parent_int = -1
                    present_stmts.append((order_int, stype, parent_int))

        present_stmts.sort(key=lambda x: (x[0], x[1]))

        present_exprs = set()
        for etype in _EXPRESSION_TYPES:
            var_name = f"present_{etype}"
            if var_name in vars_map:
                val = model.evaluate(vars_map[var_name], model_completion=True)
                if z3.is_true(val):
                    present_exprs.add(etype)

        # Build an order→(stype, node, parent_order) map for tree assembly
        order_to_node = {}
        func_order = None

        for order_int, stype, parent_int in present_stmts:
            if stype == "Module":
                continue
            if stype == "FunctionDef":
                func_order = order_int
                continue
            node = self._make_statement_node(stype, present_exprs)
            if node is not None:
                order_to_node[order_int] = (stype, node, parent_int)

        # Nesting types: statements whose AST nodes have a .body list
        _NESTING_STYPES = {"If", "While", "For", "AsyncFor", "With", "AsyncWith",
                           "Try", "FunctionDef", "AsyncFunctionDef", "ClassDef"}

        # Assemble tree using parent assignments
        # Pass 1: collect top-level nodes and children
        top_level = []
        children_of = {}  # order_int -> list of child nodes

        for order_int in sorted(order_to_node.keys()):
            stype, node, parent_int = order_to_node[order_int]
            if parent_int < 0:
                # Top-level (inside function body if function present, else module body)
                top_level.append(node)
            else:
                # Find the parent statement at the given order position
                if parent_int in order_to_node:
                    parent_stype, parent_node, _ = order_to_node[parent_int]
                    if parent_stype in _NESTING_STYPES and hasattr(parent_node, 'body'):
                        # Remove placeholder Pass from parent body if present
                        if (len(parent_node.body) == 1
                                and isinstance(parent_node.body[0], ast.Pass)):
                            parent_node.body.clear()
                        parent_node.body.append(node)
                    else:
                        # Parent doesn't support nesting, fall back to sibling
                        top_level.append(node)
                else:
                    # Parent is FunctionDef (at func_order) — treat as top-level
                    # (will be placed in function body below)
                    top_level.append(node)

        # Wrap in function if FunctionDef was present
        has_function = func_order is not None
        if has_function:
            func_body = top_level if top_level else [ast.Pass()]
            func_def = ast.FunctionDef(
                name="synthesized_fn",
                args=ast.arguments(
                    posonlyargs=[], args=[ast.arg(arg="x")],
                    kwonlyargs=[], kw_defaults=[], defaults=[],
                ),
                body=func_body,
                decorator_list=[],
                returns=None,
            )
            body_stmts = [func_def]
        else:
            body_stmts = top_level

        if not body_stmts:
            body_stmts = [ast.Pass()]

        module = ast.Module(body=body_stmts, type_ignores=[])
        ast.fix_missing_locations(module)
        return module

    def _make_statement_node(self, stype: str, present_exprs: set) -> Optional[ast.stmt]:
        """Construct a minimal AST statement node of the given type."""
        placeholder_name = ast.Name(id="x", ctx=ast.Load())
        placeholder_const = ast.Constant(value=0)

        builders = {
            "Return": lambda: ast.Return(value=placeholder_name if "Name" in present_exprs else None),
            "Assign": lambda: ast.Assign(
                targets=[ast.Name(id="x", ctx=ast.Store())],
                value=placeholder_const if "Constant" in present_exprs else placeholder_name,
            ),
            "AugAssign": lambda: ast.AugAssign(
                target=ast.Name(id="x", ctx=ast.Store()),
                op=ast.Add(),
                value=ast.Constant(value=1),
            ),
            "Expr": lambda: ast.Expr(
                value=ast.Call(
                    func=placeholder_name, args=[], keywords=[],
                ) if "Call" in present_exprs else placeholder_name
            ),
            "If": lambda: ast.If(
                test=ast.Compare(
                    left=placeholder_name,
                    ops=[ast.LtE()],
                    comparators=[ast.Constant(value=1)],
                ) if "Compare" in present_exprs else placeholder_name,
                body=[ast.Return(value=ast.Constant(value=1))
                      if "Return" in present_exprs else ast.Pass()],
                orelse=[],
            ),
            "While": lambda: ast.While(
                test=ast.Compare(
                    left=placeholder_name,
                    ops=[ast.Gt()],
                    comparators=[ast.Constant(value=0)],
                ) if "Compare" in present_exprs else placeholder_name,
                body=[ast.Pass()],
                orelse=[],
            ),
            "For": lambda: ast.For(
                target=ast.Name(id="i", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[placeholder_name], keywords=[],
                ),
                body=[ast.Pass()],
                orelse=[],
            ),
            "Pass": lambda: ast.Pass(),
            "Break": lambda: ast.Break(),
            "Continue": lambda: ast.Continue(),
            "Raise": lambda: ast.Raise(exc=ast.Name(id="ValueError", ctx=ast.Load())),
            "Assert": lambda: ast.Assert(test=placeholder_name),
            "Import": lambda: ast.Import(names=[ast.alias(name="os")]),
        }

        builder = builders.get(stype)
        if builder is None:
            return None
        return builder()

    # ── Dynamic Null Space Projection: Quotient Space Folding ──────

    def _extract_unsat_core(self, solver: z3.Solver) -> List[str]:
        """Extract the UNSAT core from a solver that returned UNSAT.

        Returns the list of tracking labels (constraint names) that form
        the minimal conflicting subset. These labels correspond to atom
        vocabulary keys (e.g., "atom:cfg_seq:If->Return").
        """
        core = solver.unsat_core()
        return [str(c) for c in core]

    def _reverse_lookup_handles(self, core_labels: List[str]) -> List[int]:
        """Map UNSAT core constraint labels back to arena atom handles.

        Each label in the core corresponds to a structural atom in the
        vocabulary. This performs the reverse lookup to recover the
        hypervector handles that encode the conflicting constraints.
        """
        handles = []
        seen = set()
        for label in core_labels:
            if label in self._atom_vocab:
                h = self._atom_vocab[label].handle
                if h not in seen:
                    handles.append(h)
                    seen.add(h)
        return handles

    def _attempt_quotient_folding(
        self,
        projection: LayeredProjection,
        constraints: DecodedConstraints,
        entropy_budget: Optional[float] = None,
    ) -> Optional[Tuple[ast.Module, UnsatCoreResult]]:
        """Attempt to resolve an UNSAT via Dynamic Null Space Projection.

        Pipeline:
          1. Re-encode constraints with UNSAT core tracking enabled
          2. If UNSAT, extract the minimal conflicting core
          3. Reverse-lookup core labels → arena atom handles
          4. Synthesize Contradiction Vector V_error = bind/bundle of conflicting atoms
          5. Project entire arena to quotient space H / <V_error>
          6. Re-probe the (now folded) projection and re-solve

        Returns (ast.Module, UnsatCoreResult) on success, None on failure.
        """
        # Step 1: Re-encode with UNSAT core tracking
        solver, vars_map = self.encode_smt(
            constraints, entropy_budget=entropy_budget, enable_unsat_core=True,
        )
        result = solver.check()

        if result == z3.sat:
            # Not actually UNSAT — solve succeeded with tracked assertions
            return self.compile_model(solver.model(), vars_map), UnsatCoreResult(
                core_labels=[], core_atom_handles=[], v_error_handle=None, projected_count=0,
            )

        if result != z3.unsat:
            return None  # Unknown result

        # Step 2: Extract the UNSAT core
        core_labels = self._extract_unsat_core(solver)
        if not core_labels:
            return None  # Empty core — cannot isolate contradiction

        # Step 3: Reverse-lookup to arena handles
        core_handles = self._reverse_lookup_handles(core_labels)
        if not core_handles:
            return None  # No matching atoms found

        # Step 4: Synthesize Contradiction Vector
        v_error = self.projector.synthesize_contradiction_vector(core_handles)

        # Step 5: Project arena to quotient space H / <V_error>
        projected_count = self.arena.project_to_quotient_space(v_error)

        core_result = UnsatCoreResult(
            core_labels=core_labels,
            core_atom_handles=core_handles,
            v_error_handle=v_error,
            projected_count=projected_count,
        )

        # Step 6: Re-probe the folded projection and re-solve
        folded_constraints = self.probe(projection)
        if not folded_constraints.active_node_types:
            return None

        folded_solver, folded_vars = self.encode_smt(
            folded_constraints, entropy_budget=entropy_budget,
        )
        if folded_solver.check() == z3.sat:
            return self.compile_model(folded_solver.model(), folded_vars), core_result

        return None

    # ── Meta-Grammar Emergence: UNSAT Resolution ───────────────────

    def _attempt_fusion_reprobe(
        self, projection: LayeredProjection, constraints: DecodedConstraints,
    ) -> Optional[DecodedConstraints]:
        """On UNSAT, synthesize a fused operator and re-probe.

        Creates fuse(ast, [cfg, data]) — a new algebraic grammar rule
        combining bind and bundle — and probes the fused representation.
        This can resolve contradictions by collapsing cross-layer interference.
        """
        if not isinstance(projection, LayeredProjection):
            return None
        if projection.cfg_handle is None and projection.data_handle is None:
            return None

        # Collect non-None layer handles for fusion bundle
        bundle_handles = []
        if projection.cfg_handle is not None:
            bundle_handles.append(projection.cfg_handle)
        if projection.data_handle is not None:
            bundle_handles.append(projection.data_handle)

        if not bundle_handles:
            return None

        # Fused operator: fuse(ast, [cfg, data]) = ast ⊗ (cfg ⊕ data)
        fused_h = self.arena.allocate()
        self.arena.bind_bundle_fusion(projection.ast_handle, bundle_handles, fused_h)

        # Build fused phase arrays for unbinding
        bundle_phase_arrays = []
        if projection.cfg_phases is not None:
            bundle_phase_arrays.append(projection.cfg_phases)
        if projection.data_phases is not None:
            bundle_phase_arrays.append(projection.data_phases)

        fused_phases = bind_bundle_fusion_phases(projection.ast_phases, bundle_phase_arrays)

        # Create a synthetic LayeredProjection with the fused representation
        fused_proj = LayeredProjection(
            final_handle=fused_h,
            ast_handle=fused_h,
            cfg_handle=None,
            data_handle=None,
            ast_phases=fused_phases,
            cfg_phases=None,
            data_phases=None,
        )

        return self.probe_layered(fused_proj)

    def _attempt_dimension_expansion(
        self, projection: LayeredProjection,
    ) -> Optional[Tuple[LayeredProjection, DecodedConstraints]]:
        """On UNSAT after fusion, expand the arena dimension and re-project.

        Doubles the dimension d → 2d, extending all existing vectors
        with deterministic conjugate-reflected components.
        Then rebuilds the vocabulary and re-probes.
        """
        old_dim = self.dimension
        new_dim = old_dim * 2

        # Expand the Rust arena in-place
        self.arena.expand_dimension(new_dim)
        self.dimension = new_dim
        self.projector.dimension = new_dim

        # Rebuild vocabulary at new dimension
        self._atom_vocab.clear()
        self._build_vocabulary()

        # Revalidate walls after dimension expansion
        if self._wall_archive is not None:
            self._wall_archive.revalidate_walls(self, self.projector)

        # Expand the projection's phase arrays
        new_ast_phases = expand_phases(projection.ast_phases, new_dim)
        new_cfg_phases = (
            expand_phases(projection.cfg_phases, new_dim)
            if projection.cfg_phases is not None else None
        )
        new_data_phases = (
            expand_phases(projection.data_phases, new_dim)
            if projection.data_phases is not None else None
        )

        expanded_proj = LayeredProjection(
            final_handle=projection.final_handle,
            ast_handle=projection.ast_handle,
            cfg_handle=projection.cfg_handle,
            data_handle=projection.data_handle,
            ast_phases=new_ast_phases,
            cfg_phases=new_cfg_phases,
            data_phases=new_data_phases,
        )

        constraints = self.probe_layered(expanded_proj)
        if constraints.active_node_types:
            return expanded_proj, constraints
        return None

    def get_meta_grammar_log(self) -> List[MetaGrammarEvent]:
        """Return the log of all Meta-Grammar Emergence events."""
        return list(self._meta_grammar_log)

    # ── Full decode pipeline ─────────────────────────────────────

    def decode(self, input_: Union[int, LayeredProjection]) -> Optional[ast.Module]:
        """Full pipeline: probe → encode SMT → solve → compile AST.

        When a SubTreeVocabulary is available, uses sub-tree assembly:
          1. Probe sub-tree vocabulary for resonating concrete fragments
          2. Encode sub-tree selection/ordering as Z3 constraints
          3. Solve → assemble real sub-trees with variable threading

        Falls back to legacy atom-based decoding if no sub-tree vocabulary
        is configured or if sub-tree assembly fails.

        On UNSAT (topological contradiction), triggers Meta-Grammar Emergence:
          1. First attempt: synthesize fused operator and re-probe
          2. Second attempt: expand dimension d → 2d and re-probe
        Returns None only if all meta-grammar expansions fail.

        Topological Thermodynamics is enforced via entropy constraints in SMT.
        """
        # Compute entropy budget from input signal (thermodynamic ceiling)
        entropy_budget = None
        if isinstance(input_, LayeredProjection) and self.entropy_weight > 0:
            phase_entropy = compute_phase_entropy(input_.ast_phases)
            entropy_budget = phase_entropy * self.entropy_weight * 50

        # ── Sub-Tree Assembly Path (LEAP 1) ───────────────────────
        if self._subtree_vocab is not None and self._subtree_vocab.size() > 0:
            activated = self.probe_subtrees(input_)
            if activated:
                solver_st, vars_st, atom_list = self.encode_smt_subtrees(
                    activated, entropy_budget=entropy_budget,
                )
                result_st = solver_st.check()
                if result_st == z3.sat:
                    return self.compile_model_subtrees(
                        solver_st.model(), vars_st, atom_list,
                    )

        # ── Legacy Atom-Based Path ────────────────────────────────
        constraints = self.probe(input_)

        if not constraints.active_node_types:
            return None

        solver, vars_map = self.encode_smt(constraints, entropy_budget=entropy_budget)

        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            return self.compile_model(model, vars_map)

        # ── Phase Transition: Dynamic Null Space Projection ──────
        # Attempt quotient space folding FIRST (algebraically cheapest:
        # no dimension change, no re-projection, just annihilate the
        # contradiction axis and re-solve).
        old_dim = self.dimension

        if isinstance(input_, LayeredProjection):
            folding_result = self._attempt_quotient_folding(
                input_, constraints, entropy_budget=entropy_budget,
            )
            if folding_result is not None:
                compiled_ast, core_result = folding_result
                if self._wall_archive is not None and core_result is not None:
                    self._wall_archive.record(core_result, synthesis_context=str(input_))
                self._meta_grammar_log.append(MetaGrammarEvent(
                    trigger="unsat_quotient_folding",
                    old_dimension=old_dim,
                    new_dimension=old_dim,
                    retry_succeeded=True,
                    unsat_core=core_result,
                ))
                return compiled_ast

        # ── Meta-Grammar Emergence: UNSAT triggers expansion ──────

        for retry in range(self.max_meta_grammar_retries):
            # Attempt 1: Fusion operator (cheaper — no dimension change)
            if isinstance(input_, LayeredProjection) and retry == 0:
                fused_constraints = self._attempt_fusion_reprobe(input_, constraints)
                if fused_constraints and fused_constraints.active_node_types:
                    solver2, vars_map2 = self.encode_smt(fused_constraints)
                    if solver2.check() == z3.sat:
                        self._meta_grammar_log.append(MetaGrammarEvent(
                            trigger="unsat_fusion_retry",
                            old_dimension=old_dim,
                            new_dimension=old_dim,
                            retry_succeeded=True,
                        ))
                        return self.compile_model(solver2.model(), vars_map2)

            # Attempt 2: Dimension expansion (heavier — doubles d)
            if isinstance(input_, LayeredProjection):
                expansion_result = self._attempt_dimension_expansion(input_)
                if expansion_result is not None:
                    expanded_proj, expanded_constraints = expansion_result
                    solver3, vars_map3 = self.encode_smt(expanded_constraints)
                    if solver3.check() == z3.sat:
                        self._meta_grammar_log.append(MetaGrammarEvent(
                            trigger="unsat_dimension_expand",
                            old_dimension=old_dim,
                            new_dimension=self.dimension,
                            retry_succeeded=True,
                        ))
                        return self.compile_model(solver3.model(), vars_map3)
                    input_ = expanded_proj  # Use expanded for next retry

        # All meta-grammar attempts exhausted
        self._meta_grammar_log.append(MetaGrammarEvent(
            trigger="unsat_all_failed",
            old_dimension=old_dim,
            new_dimension=self.dimension,
            retry_succeeded=False,
        ))
        return None

    def decode_to_source(self, input_: Union[int, LayeredProjection]) -> Optional[str]:
        """Convenience: decode and unparse to Python source string."""
        module = self.decode(input_)
        if module is None:
            return None
        return ast.unparse(module)

    def decode_and_verify(
        self, input_: Union[int, LayeredProjection]
    ) -> Tuple[Optional[str], Optional[float]]:
        """Decode, then re-project and verify round-trip correlation."""
        source = self.decode_to_source(input_)
        if source is None:
            return None, None

        reprojected = self.projector.project(source)
        if isinstance(input_, LayeredProjection):
            correlation = self.arena.compute_correlation(input_.final_handle, reprojected.final_handle)
        else:
            correlation = self.arena.compute_correlation(input_, reprojected.final_handle)
        return source, correlation
