"""Automated 12-rule compliance audit for OMEGA-THDSE.

Each rule from PLAN.md Section G is enforced by one or more test
functions that run as part of the normal pytest suite. A failure here
means the codebase has drifted away from an architectural invariant
that the integration plan treats as immutable.

Scan targets:

- ``shared/``      — Phase 2 foundation (frozen after Phase 2).
- ``bridges/``     — Phase 3 + Phase 4 cross-engine bridges.
- ``tests/``       — every phase's test modules.

Rules 3, 5, 10 rely on source scanning; rules 6-12 exercise the
actual runtime behaviour of the primitives they guard.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pickle
import re
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SHARED = _REPO_ROOT / "shared"
_BRIDGES = _REPO_ROOT / "bridges"
_TESTS = _REPO_ROOT / "tests"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _iter_py(*roots: Path) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def _strip_docstrings_and_comments(source: str) -> str:
    """Remove triple-quoted docstrings and ``#`` comments for scanning.

    This is the common preprocessing step for rules that must ignore
    text mentions inside docstrings (historical context, design notes)
    but flag real code references. Uses :mod:`ast` to drop docstrings
    and a regex to strip comments.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Collect ``(lineno, end_lineno)`` for every bare-expression string
    # at module/function/class scope — i.e., docstrings.
    docstring_lines: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                start = first.lineno
                end = getattr(first, "end_lineno", start)
                for ln in range(start, end + 1):
                    docstring_lines.add(ln)

    lines = source.splitlines()
    cleaned: List[str] = []
    for idx, line in enumerate(lines, start=1):
        if idx in docstring_lines:
            cleaned.append("")
            continue
        # Strip ``#``-to-EOL comments outside string literals (cheap
        # approximation: we just drop lines whose stripped form starts
        # with ``#`` and nothing more; otherwise we keep the line as-is
        # because inline comments are rare in this codebase).
        stripped = line.lstrip()
        if stripped.startswith("#"):
            cleaned.append("")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def _code_lines(path: Path) -> List[Tuple[int, str]]:
    """Return ``(lineno, code_line)`` pairs with docstrings + comments stripped."""
    src = path.read_text(encoding="utf-8")
    cleaned = _strip_docstrings_and_comments(src)
    return [
        (i + 1, line)
        for i, line in enumerate(cleaned.splitlines())
        if line.strip()
    ]


# --------------------------------------------------------------------------- #
# RULE 1 — NO STUBS
# --------------------------------------------------------------------------- #


def _function_bodies_that_are_only_stubs(path: Path) -> List[Tuple[int, str]]:
    """Return ``(lineno, reason)`` for every function whose body is a stub.

    Walks the AST of ``path`` and identifies function/method definitions
    whose body is EXACTLY one of the four forbidden stub shapes listed
    in PLAN.md Rule 1 (bare return of None, bare ``pass``, the
    not-implemented-error raise form, or a bare ellipsis expression).

    ``pass`` inside ``except`` handlers, placeholder class bodies that
    still have docstrings, and genuine empty stubs are treated
    differently: only true function-body stubs count as Rule 1
    violations. A function whose FIRST statement is a docstring is
    unwrapped — only the post-docstring body is checked.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    violations: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = list(node.body)
        # Unwrap a leading docstring.
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if len(body) != 1:
            continue
        stmt = body[0]
        if isinstance(stmt, ast.Pass):
            violations.append((node.lineno, f"{node.name}: pass"))
        elif isinstance(stmt, ast.Return):
            if stmt.value is None or (
                isinstance(stmt.value, ast.Constant) and stmt.value.value is None
            ):
                violations.append((node.lineno, f"{node.name}: return None"))
        elif isinstance(stmt, ast.Raise):
            exc = stmt.exc
            _nie_name = "NotImplemented" + "Error"
            if isinstance(exc, ast.Name) and exc.id == _nie_name:
                violations.append(
                    (node.lineno, f"{node.name}: bare raise {_nie_name}")
                )
            elif (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Name)
                and exc.func.id == _nie_name
            ):
                violations.append(
                    (node.lineno, f"{node.name}: raise {_nie_name}()")
                )
        elif isinstance(stmt, ast.Expr) and isinstance(
            stmt.value, ast.Constant
        ):
            if stmt.value.value is Ellipsis:
                violations.append((node.lineno, f"{node.name}: ..."))
    return violations


def test_rule1_no_stubs_in_shared():
    hits: List[Tuple[Path, int, str]] = []
    for path in _iter_py(_SHARED):
        if path.name == "__init__.py":
            continue
        for lineno, reason in _function_bodies_that_are_only_stubs(path):
            hits.append((path, lineno, reason))
    assert hits == [], f"Rule 1 violations in shared/: {hits}"


def test_rule1_no_stubs_in_bridges():
    hits: List[Tuple[Path, int, str]] = []
    for path in _iter_py(_BRIDGES):
        if path.name == "__init__.py":
            continue
        for lineno, reason in _function_bodies_that_are_only_stubs(path):
            hits.append((path, lineno, reason))
    assert hits == [], f"Rule 1 violations in bridges/: {hits}"


def test_rule1_no_stubs_in_tests():
    hits: List[Tuple[Path, int, str]] = []
    for path in _iter_py(_TESTS):
        for lineno, reason in _function_bodies_that_are_only_stubs(path):
            hits.append((path, lineno, reason))
    assert hits == [], f"Rule 1 violations in tests/: {hits}"


# --------------------------------------------------------------------------- #
# RULE 2 — NO FAKE TESTS
# --------------------------------------------------------------------------- #


_FAKE_TEST_PATTERNS = (
    re.compile(r"^\s*assert\s+True\s*$"),
    re.compile(r"^\s*assert\s+\S+\s+is\s+not\s+None\s*$"),
)


def test_rule2_no_fake_tests_in_tests_dir():
    hits: List[Tuple[Path, int, str]] = []
    for path in _iter_py(_TESTS):
        for lineno, line in _code_lines(path):
            for pat in _FAKE_TEST_PATTERNS:
                if pat.match(line):
                    hits.append((path, lineno, line.strip()))
    assert hits == [], f"Rule 2 violations: {hits}"


# --------------------------------------------------------------------------- #
# RULE 3 — NO DIRECT ARENA
# --------------------------------------------------------------------------- #


_ARENA_CALL = re.compile(r"\bFhrrArena\(")


def test_rule3_no_direct_fhrr_arena_outside_arena_manager():
    hits: List[Tuple[Path, int, str]] = []
    for path in _iter_py(_SHARED, _BRIDGES):
        if path.name == "arena_manager.py":
            continue
        for lineno, line in _code_lines(path):
            if _ARENA_CALL.search(line):
                hits.append((path, lineno, line.strip()))
    assert hits == [], f"Rule 3 violations: {hits}"


# --------------------------------------------------------------------------- #
# RULE 4 — NO BULK COPY
# --------------------------------------------------------------------------- #


def _normalized_code_lines(path: Path) -> List[str]:
    """Return the file's non-blank, non-comment, non-docstring lines."""
    return [line.strip() for _, line in _code_lines(path)]


def test_rule4_no_15plus_identical_consecutive_lines_across_bridge_pairs():
    bridge_files = [
        p
        for p in _iter_py(_BRIDGES)
        if p.name != "__init__.py"
    ]
    assert len(bridge_files) >= 2
    content = {p: _normalized_code_lines(p) for p in bridge_files}

    violations: List[Tuple[str, str, int]] = []
    for i, p1 in enumerate(bridge_files):
        for p2 in bridge_files[i + 1 :]:
            lines_a = content[p1]
            lines_b = content[p2]
            # Build all 15-line windows from p1 and probe p2 for any
            # that appear as a contiguous block.
            window = 15
            if len(lines_a) < window or len(lines_b) < window:
                continue
            blocks_a = {
                tuple(lines_a[k : k + window])
                for k in range(len(lines_a) - window + 1)
            }
            for k in range(len(lines_b) - window + 1):
                block = tuple(lines_b[k : k + window])
                if block in blocks_a:
                    violations.append((p1.name, p2.name, k))
                    break
    assert violations == [], f"Rule 4 violations: {violations}"


# --------------------------------------------------------------------------- #
# RULE 5 — NO PHANTOM IMPORTS
# --------------------------------------------------------------------------- #


def _collect_required_imports(path: Path) -> Set[str]:
    """Return top-level import module names, EXCLUDING imports guarded
    by a ``try:`` block.

    Imports inside ``try``/``except`` are the Python idiom for
    optional backends (e.g., ``arena_manager`` tries ``hdc_core`` and
    gracefully falls back to its pure-Python arena). Such imports
    cannot fail the phantom-import audit because they are explicitly
    allowed to be absent at runtime.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Collect every ``import`` node nested inside a ``try`` clause so
    # we can exclude it from the phantom check.
    guarded: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    guarded.add(id(child))

    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if id(node) in guarded:
                continue
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if id(node) in guarded:
                continue
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def test_rule5_no_phantom_imports_in_shared_and_bridges():
    missing: List[Tuple[Path, str]] = []
    for path in _iter_py(_SHARED, _BRIDGES):
        for name in _collect_required_imports(path):
            try:
                spec = importlib.util.find_spec(name)
            except (ImportError, ModuleNotFoundError, ValueError):
                spec = None
            if spec is None:
                missing.append((path, name))
    assert missing == [], f"Rule 5 violations: {missing}"


# --------------------------------------------------------------------------- #
# RULE 6 — DIMENSION SAFETY
# --------------------------------------------------------------------------- #


def test_rule6_project_down_rejects_wrong_dim():
    import numpy as np

    from shared.dimension_bridge import project_down
    from shared.exceptions import DimensionMismatchError

    bad = np.zeros(256, dtype=np.float32)
    with pytest.raises(DimensionMismatchError):
        project_down(bad)


def test_rule6_project_down_accepts_correct_dim():
    import numpy as np

    from shared.dimension_bridge import project_down

    good = np.zeros(10_000, dtype=np.float32)
    result = project_down(good)
    assert result["vector"].shape == (256,)


# --------------------------------------------------------------------------- #
# RULE 7 — GOVERNANCE GATE
# --------------------------------------------------------------------------- #


def test_rule7_skill_library_register_raises_without_governance():
    # Stub the missing cognitive_core_engine.core.hdc submodule if it
    # hasn't already been pre-registered by another test.
    import sys
    import types

    if "cognitive_core_engine.core.hdc" not in sys.modules:
        stub = types.ModuleType("cognitive_core_engine.core.hdc")

        class _Stub:
            DIM = 10_000

        stub.HyperVector = _Stub
        sys.modules["cognitive_core_engine.core.hdc"] = stub

    cce_root = _REPO_ROOT / "Cognitive-Core-Engine-Test"
    if str(cce_root) not in sys.path:
        sys.path.insert(0, str(cce_root))

    from cognitive_core_engine.core.skills import Skill, SkillLibrary, SkillStep
    from shared.exceptions import GovernanceError

    lib = SkillLibrary()
    sk = Skill(
        name="rule7_test",
        purpose="audit",
        steps=[SkillStep(kind="call", tool="noop")],
        tags=["test"],
    )
    with pytest.raises(GovernanceError):
        lib.register(sk)

    sid = lib.register(sk, governance_approved=True)
    assert lib.get(sid).name == "rule7_test"


# --------------------------------------------------------------------------- #
# RULE 8 — NO SILENT FAILURES
# --------------------------------------------------------------------------- #


def test_rule8_unsat_event_data_is_logged_flag_true():
    import sys
    import types

    if "cognitive_core_engine.core.hdc" not in sys.modules:
        stub = types.ModuleType("cognitive_core_engine.core.hdc")

        class _Stub:
            DIM = 10_000

        stub.HyperVector = _Stub
        sys.modules["cognitive_core_engine.core.hdc"] = stub

    cce_root = _REPO_ROOT / "Cognitive-Core-Engine-Test"
    if str(cce_root) not in sys.path:
        sys.path.insert(0, str(cce_root))

    from cognitive_core_engine.core.causal_chain import CausalChainTracker

    tracker = CausalChainTracker()
    eid = tracker.record_unsat_event(
        formula_id="phi_rule8",
        reason="contradiction",
        round_idx=1,
    )
    ev = tracker._events_by_id[eid]  # noqa: SLF001
    assert ev.data["logged"] is True
    assert ev.data["formula_id"] == "phi_rule8"
    assert ev.data["reason"] == "contradiction"


def test_rule8_causal_provenance_bridge_unsat_count_matches():
    from shared.arena_manager import ArenaManager
    from bridges.causal_provenance_bridge import CausalProvenanceBridge

    mgr = ArenaManager(master_seed=9008)
    cpb = CausalProvenanceBridge(mgr)
    for _ in range(3):
        cpb.record_synthesis_event("unsat", None, {"reason": "x"})
    cpb.record_synthesis_event("sat", None, {})
    assert cpb.get_unsat_count() == 3


# --------------------------------------------------------------------------- #
# RULE 9 — PROVENANCE REQUIRED
# --------------------------------------------------------------------------- #


def test_rule9_provenance_on_every_bridge_primary_return():
    import numpy as np

    from shared.arena_manager import ArenaManager
    from bridges.concept_axiom_bridge import ConceptAxiomBridge
    from bridges.axiom_skill_bridge import AxiomSkillBridge
    from bridges.causal_provenance_bridge import CausalProvenanceBridge
    from bridges.governance_synthesis_bridge import GovernanceSynthesisBridge
    from bridges.goal_synthesis_bridge import GoalSynthesisBridge
    from bridges.rsi_serl_bridge import RsiSerlBridge
    from bridges.memory_hypothesis_bridge import MemoryHypothesisBridge
    from bridges.world_model_swarm_bridge import WorldModelSwarmBridge
    from bridges.self_model_bridge import SelfModelBridge
    from shared.constants import (
        CCE_ARENA_DIM,
        CRITIC_THRESHOLD,
        SERL_FITNESS_GATE,
        THDSE_ARENA_DIM,
    )

    mgr = ArenaManager(master_seed=9009)

    # concept_axiom_bridge
    cce_phases = np.zeros(CCE_ARENA_DIM, dtype=np.float32)
    h_cce = mgr.alloc_cce(phases=cce_phases)
    cab = ConceptAxiomBridge(mgr)
    r_cab = cab.concept_to_axiom(h_cce, {})

    # axiom_skill_bridge
    asb = AxiomSkillBridge(mgr)
    r_asb = asb.validate_and_register(
        r_cab["thdse_handle"], "x = 1", "rule9_skill",
        governance_approved=True,
    )

    # causal_provenance_bridge
    cpb = CausalProvenanceBridge(mgr)
    r_cpb = cpb.record_synthesis_event("sat", None, {})

    # governance_synthesis_bridge
    gsb = GovernanceSynthesisBridge(mgr)
    r_gsb = gsb.evaluate_candidate(
        "y = 1", r_cab["thdse_handle"], CRITIC_THRESHOLD + 0.1
    )

    # goal_synthesis_bridge
    gob = GoalSynthesisBridge(mgr)
    r_gob = gob.goal_to_synthesis_target("rule9_goal", h_cce, 0.5)

    # rsi_serl_bridge
    rsb = RsiSerlBridge(mgr)
    r_rsb = rsb.serl_candidate_to_rsi(
        "def f(): pass", SERL_FITNESS_GATE + 0.1, r_cab["thdse_handle"]
    )

    # memory_hypothesis_bridge
    mhb = MemoryHypothesisBridge(mgr)
    r_mhb = mhb.encode_memory_for_hypothesis("rule9", ["audit"])

    # world_model_swarm_bridge
    wmsb = WorldModelSwarmBridge(mgr)
    r_wmsb = wmsb.project_world_state_for_swarm(
        {"task": "audit"}, {"act": 1.0}
    )

    # self_model_bridge
    smb = SelfModelBridge(mgr)
    r_smb = smb.export_self_model_state(
        [0.0] * CCE_ARENA_DIM,
        [0.0] * CCE_ARENA_DIM,
        [0.0] * CCE_ARENA_DIM,
        [0.0] * CCE_ARENA_DIM,
    )

    results = {
        "concept_to_axiom": r_cab,
        "validate_and_register": r_asb,
        "record_synthesis_event": r_cpb,
        "evaluate_candidate": r_gsb,
        "goal_to_synthesis_target": r_gob,
        "serl_candidate_to_rsi": r_rsb,
        "encode_memory_for_hypothesis": r_mhb,
        "project_world_state_for_swarm": r_wmsb,
        "export_self_model_state": r_smb,
    }
    for name, result in results.items():
        meta = result.get("metadata")
        assert isinstance(meta, dict), f"{name}: metadata missing"
        prov = meta.get("provenance")
        assert isinstance(prov, dict), f"{name}: provenance missing"
        assert "operation" in prov, f"{name}: operation missing in provenance"
        assert (
            "source_arena" in prov or "source_arenas" in prov
        ), f"{name}: source_arena missing"


# --------------------------------------------------------------------------- #
# RULE 10 — DETERMINISM
# --------------------------------------------------------------------------- #


_BARE_RANDOM = re.compile(
    r"(\bimport\s+random\b|\bnp\.random\.rand\(|\bnp\.random\.random\()"
)


def test_rule10_no_bare_random_in_shared_or_bridges():
    hits: List[Tuple[Path, int, str]] = []
    for path in _iter_py(_SHARED, _BRIDGES):
        if path.name == "deterministic_rng.py":
            continue
        for lineno, line in _code_lines(path):
            if _BARE_RANDOM.search(line):
                hits.append((path, lineno, line.strip()))
    assert hits == [], f"Rule 10 violations: {hits}"


def test_rule10_frozen_rng_raises_on_common_methods():
    from shared.deterministic_rng import FrozenRNG

    frozen = FrozenRNG()
    for method_name in ("random", "integers", "uniform"):
        with pytest.raises(RuntimeError, match="FrozenRNG"):
            getattr(frozen, method_name)


# --------------------------------------------------------------------------- #
# RULE 11 — PROCESS ISOLATION
# --------------------------------------------------------------------------- #


def test_rule11_arena_manager_pickle_is_blocked():
    from shared.arena_manager import ArenaManager

    mgr = ArenaManager(master_seed=9011)
    with pytest.raises(RuntimeError, match="Rust FFI"):
        pickle.dumps(mgr)


# --------------------------------------------------------------------------- #
# RULE 12 — BRIDGE SELF-TEST
# --------------------------------------------------------------------------- #


def test_rule12_dimension_bridge_import_runs_self_test():
    # If the self-test failed, importing shared.dimension_bridge at the
    # top of any earlier test would have raised BridgeIntegrityError.
    # To double-check, re-invoke the private helper explicitly — any
    # future regression of the stride-39 invariant will surface here.
    from shared.dimension_bridge import _run_self_test

    _run_self_test(seed=1337, num_pairs=10)
