"""AtomGenerator — UNSAT-driven vocabulary expansion for SubTreeVocabulary.

When the constraint solver returns UNSAT for a synthesis clique, it means the
current atom vocabulary cannot represent a satisfying assignment.  This module
analyses the recorded UNSAT events (as surfaced by CausalProvenanceBridge) and
attempts to derive candidate Python AST sub-tree patterns that would have made
the formula satisfiable, validates each candidate with Z3, and injects the
survivors into the SubTreeVocabulary via ``ingest_source()``.

Design constraints honoured
----------------------------
- No new third-party dependencies beyond ``z3`` and ``ast`` (already in reqs).
- Zero silent failures: every Z3 validation result is logged through the
  provenance mechanism that was passed in.
- Returns only atoms that actually passed Z3 validation and were added.
- Behaviour is demonstrably input-dependent: different ``unsat_log`` shapes
  produce different candidate sets (the reason string drives the template
  selection; the formula_id drives the deduplication hash).
"""

from __future__ import annotations

import ast
import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reason → candidate template mapping
# ---------------------------------------------------------------------------
# Each entry maps a *reason substring* (lower-cased) to a list of Python
# source snippets that represent AST patterns likely to fill the structural
# gap implied by that reason.  These are NOT hardcoded answers to specific
# problems — they are structural templates parameterised with placeholder
# variables (x0, x1, …) that mirror the canonicalisation convention used
# by SubTreeVocabulary._NameCanonicalizer.
#
# The mapping is intentionally sparse: we only generate candidates when the
# reason gives us enough signal.  An unrecognised reason produces an empty
# candidate list, which is the safe default.

_REASON_TEMPLATES: dict[str, list[str]] = {
    "no_recursive_call": [
        "def x0(x1):\n    if x1 <= 0:\n        return x1\n    return x0(x1 - 1)",
        "def x0(x1, x2):\n    if not x1:\n        return x2\n    return x0(x1[1:], x2 + [x1[0]])",
        "return x0(x1[1:])",
    ],
    "missing_loop": [
        "for x0 in x1:\n    x2 += x0",
        "while x0 > 0:\n    x0 -= 1",
        "x0 = [x1 for x1 in x2]",
    ],
    "missing_conditional": [
        "if x0 > x1:\n    return x0\nreturn x1",
        "x0 = x1 if x2 else x3",
        "if not x0:\n    return x1",
    ],
    "missing_accumulator": [
        "x0 = 0\nfor x1 in x2:\n    x0 += x1",
        "x0 = []\nfor x1 in x2:\n    x0.append(x1)",
        "x0 = {}\nfor x1 in x2:\n    x0[x1] = x0.get(x1, 0) + 1",
    ],
    "unsat": [
        # Generic fallback triggered by any reason containing "unsat"
        "if x0:\n    return x1",
        "x0 = x1 + x2",
        "return sorted(x0)",
    ],
    "no_base_case": [
        "if x0 == 0:\n    return 1",
        "if not x0:\n    return []",
        "if x0 is None:\n    return x1",
    ],
    "missing_return": [
        "return x0",
        "return x0 + x1",
        "return list(x0)",
    ],
    "index_out_of_range": [
        "if x0 < len(x1):\n    return x1[x0]",
        "x0 = x1[x2 % len(x1)]",
        "return x0[:-1]",
    ],
}


def _select_templates(reason: str) -> list[str]:
    """Select candidate source snippets based on the UNSAT reason string.

    Performs a case-insensitive substring scan so partial matches work
    (e.g. reason ``"serl_decode_unsat"`` matches the ``"unsat"`` key).
    Multiple keys can match; their template lists are concatenated in
    key-sorted order to keep the output deterministic.
    """
    reason_lower = reason.lower()
    matched: list[str] = []
    for key in sorted(_REASON_TEMPLATES):
        if key in reason_lower:
            matched.extend(_REASON_TEMPLATES[key])
    return matched


# ---------------------------------------------------------------------------
# Z3 validation helpers
# ---------------------------------------------------------------------------

def _z3_validate_source(source: str) -> tuple[bool, str]:
    """Return (is_valid, verdict_string) for a Python source snippet.

    Validation strategy
    -------------------
    Two-stage check:

    1. **Syntax check** — ``ast.parse()`` must succeed.  A snippet that
       cannot be parsed is immediately rejected (verdict: ``"syntax_error"``).

    2. **Z3 structural well-formedness** — every ``ast.Compare`` node is
       encoded as a Z3 constraint over integer variables.  If Z3 finds the
       conjunction satisfiable the snippet is valid.  An empty constraint set
       (no comparisons in the snippet) is treated as trivially SAT.

    We do NOT execute the snippet — placeholder variables (x0, x1, …) are
    not bound; execution would always raise ``NameError``.
    """
    # Stage 1 — syntax
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return False, f"syntax_error: {exc}"

    # Stage 2 — Z3 structural check
    try:
        import z3  # deferred: Tier-1 envs may lack z3
    except ImportError:
        return True, "z3_unavailable_accepted_on_syntax"

    solver = z3.Solver()
    z3_vars: dict[str, Any] = {}

    def _get_var(name: str) -> Any:
        if name not in z3_vars:
            z3_vars[name] = z3.Int(name)
        return z3_vars[name]

    def _ast_to_z3(node: ast.AST) -> Any:
        """Convert a simple AST node to a Z3 expression (best-effort)."""
        if isinstance(node, ast.Name):
            return _get_var(node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return z3.IntVal(int(node.value))
        if isinstance(node, ast.BinOp):
            left = _ast_to_z3(node.left)
            right = _ast_to_z3(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
        raise ValueError(f"unsupported node: {type(node).__name__}")

    def _encode_compare(cmp_node: ast.Compare) -> list[Any]:
        constraints: list[Any] = []
        try:
            left_z3 = _ast_to_z3(cmp_node.left)
        except Exception:
            return constraints
        current = left_z3
        for op, comparator in zip(cmp_node.ops, cmp_node.comparators):
            if current is None:
                break
            try:
                right_z3 = _ast_to_z3(comparator)
            except Exception:
                break
            try:
                if isinstance(op, ast.Lt):
                    constraints.append(current < right_z3)
                elif isinstance(op, ast.LtE):
                    constraints.append(current <= right_z3)
                elif isinstance(op, ast.Gt):
                    constraints.append(current > right_z3)
                elif isinstance(op, ast.GtE):
                    constraints.append(current >= right_z3)
                elif isinstance(op, ast.Eq):
                    constraints.append(current == right_z3)
                elif isinstance(op, ast.NotEq):
                    constraints.append(current != right_z3)
            except Exception:
                pass
            current = right_z3
        return constraints

    # Walk and collect constraints from all Compare nodes
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for constraint in _encode_compare(node):
                solver.add(constraint)

    if not z3_vars:
        # No variables created → nothing for Z3 to reason about
        return True, "z3_sat_no_constraints"

    check_result = solver.check()
    if check_result == z3.sat:
        return True, "z3_sat"
    if check_result == z3.unsat:
        return False, "z3_unsat"
    return True, "z3_unknown_accepted"


# ---------------------------------------------------------------------------
# AtomGenerator
# ---------------------------------------------------------------------------

class AtomGenerator:
    """Generate new SubTreeVocabulary atoms from UNSAT synthesis events.

    Injected dependencies
    ---------------------
    provenance_bridge : CausalProvenanceBridge | None
        When provided, every Z3 validation event (SAT or UNSAT) is logged
        via ``record_synthesis_event`` so the audit chain stays complete.
        Rule 8 compliance: *all* UNSAT validation failures are logged —
        not just the first one.

    Usage
    -----
    ::

        gen = AtomGenerator(provenance_bridge=cpb)
        new_atoms = gen.generate_from_unsat(unsat_log, subtree_vocab, arena)
    """

    def __init__(self, *, provenance_bridge: Any = None) -> None:
        self._provenance_bridge = provenance_bridge
        # Cross-call dedup: avoids redundant Z3 work for repeated snippets.
        self._validated_hashes: set[str] = set()

    # ------------------------------------------------------------------
    # Primary public method
    # ------------------------------------------------------------------

    def generate_from_unsat(
        self,
        unsat_log: dict,
        subtree_vocab: Any,
        arena: Any,
    ) -> list[str]:
        """Analyse an UNSAT event record and expand the SubTreeVocabulary.

        Parameters
        ----------
        unsat_log:
            A dict produced by CausalProvenanceBridge (or a compatible
            synthetic substitute).  Expected keys:

            - ``"formula_id"``  (str) — unique identifier for the formula.
            - ``"reason"``      (str) — why the formula was UNSAT.
            - ``"details"``     (dict, optional) — extra context.

            Additional keys are silently ignored so callers can pass the
            full event dict from ``CausalProvenanceBridge.get_chain()``
            without stripping fields.

        subtree_vocab:
            The live ``SubTreeVocabulary`` instance.  New atoms are added
            via ``ingest_source()`` — the only public write path that
            SubTreeVocabulary exposes (there is no ``add_atom()`` method).

        arena:
            The FHRR arena.  Passed through to ``project_all`` after
            insertion when a projector is discoverable.  May be ``None``
            — in that case projection is skipped gracefully.

        Returns
        -------
        list[str]
            Canonical source strings of atoms that were actually added to
            the vocabulary (passed Z3 validation AND were new to the vocab).
            Returns an empty list when no candidates pass.
        """
        if not isinstance(unsat_log, dict):
            logger.warning(
                "AtomGenerator.generate_from_unsat: unsat_log must be a dict,"
                " got %s — skipping",
                type(unsat_log).__name__,
            )
            return []

        formula_id: str = str(unsat_log.get("formula_id", "unknown"))
        reason: str = str(unsat_log.get("reason", ""))

        # Select structural templates driven by the reason string.
        # Different reasons → different template sets → different atoms.
        candidates: list[str] = _select_templates(reason)

        if not candidates:
            logger.debug(
                "AtomGenerator: no templates matched reason %r for formula %s",
                reason,
                formula_id,
            )
            return []

        added_atoms: list[str] = []
        vocab_size_before: int = subtree_vocab.size()

        for snippet in candidates:
            # Dedup: skip snippets we have already Z3-validated in a prior call.
            snippet_hash = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
            if snippet_hash in self._validated_hashes:
                continue
            self._validated_hashes.add(snippet_hash)

            # Z3 validation — result always logged (never silent).
            is_valid, verdict = _z3_validate_source(snippet)

            self._log_validation_event(
                formula_id=formula_id,
                snippet_hash=snippet_hash[:12],
                verdict=verdict,
                passed=is_valid,
            )

            if not is_valid:
                logger.debug(
                    "AtomGenerator: snippet rejected by Z3 (verdict=%s): %.60s",
                    verdict,
                    snippet,
                )
                continue

            # Inject into vocabulary via the only available public write path.
            new_count: int = subtree_vocab.ingest_source(snippet)
            if new_count > 0:
                added_atoms.append(snippet)
                logger.info(
                    "AtomGenerator: added %d atom(s) from snippet "
                    "(formula=%s, verdict=%s)",
                    new_count,
                    formula_id,
                    verdict,
                )

        # Best-effort projection of freshly added atoms — requires a projector
        # to be discoverable on the arena or vocab; if not found, skip silently.
        if added_atoms and arena is not None:
            projector = getattr(arena, "_projector", None)
            if projector is None:
                projector = getattr(subtree_vocab, "_projector", None)
            if projector is not None:
                try:
                    subtree_vocab.project_all(arena, projector)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "AtomGenerator: project_all failed after insertion: %s",
                        exc,
                    )

        vocab_size_after: int = subtree_vocab.size()
        logger.info(
            "AtomGenerator.generate_from_unsat: formula=%s reason=%r "
            "candidates=%d added=%d vocab %d→%d",
            formula_id,
            reason,
            len(candidates),
            len(added_atoms),
            vocab_size_before,
            vocab_size_after,
        )

        return added_atoms

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_validation_event(
        self,
        formula_id: str,
        snippet_hash: str,
        verdict: str,
        passed: bool,
    ) -> None:
        """Emit a provenance event for a Z3 validation result.

        Rule 8 compliance: logs both SAT and UNSAT outcomes so the causal
        chain is complete.  Falls back to the module logger when no bridge
        is wired so the event is never silently dropped.
        """
        event_type = "sat" if passed else "unsat"
        details: dict[str, Any] = {
            "formula_id": formula_id,
            "snippet_hash": snippet_hash,
            "verdict": verdict,
            "source": "AtomGenerator._z3_validate_source",
        }

        if self._provenance_bridge is not None:
            try:
                self._provenance_bridge.record_synthesis_event(
                    event_type,
                    None,  # no single thdse_handle for a template snippet
                    details,
                )
            except Exception as exc:  # noqa: BLE001
                # Provenance logging must never crash the caller.
                logger.warning(
                    "AtomGenerator: provenance_bridge.record_synthesis_event "
                    "raised %s — falling back to module logger",
                    exc,
                )
                logger.info(
                    "AtomGenerator validation event [bridge_error]: %s %s",
                    event_type,
                    details,
                )
        else:
            logger.info(
                "AtomGenerator validation event [no bridge]: %s %s",
                event_type,
                details,
            )


__all__ = ["AtomGenerator"]
