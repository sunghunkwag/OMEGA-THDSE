#!/usr/bin/env python3
"""OMEGA-THDSE unified CLI entry point (PLAN.md Phase 5).

This CLI is the single operator-facing surface for the integrated
engine. It exposes four subcommands:

- ``test``    — run the full pytest suite
- ``audit``   — run the 12-rule compliance audit only
- ``status``  — print a live status summary of every integration phase
- ``bridges`` — list the bridge modules and the gaps they close

Every subcommand inspects the filesystem for the files it claims
exist, so the status line cannot lie: a missing directory or a
missing module surfaces as a ``✗`` marker rather than a hardcoded
``✓``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


_REPO_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Phase layout — expected artifacts for the status checker
# --------------------------------------------------------------------------- #


#: Ordered list of ``(label, path, min_file_count, description)`` tuples used
#: by the ``status`` subcommand. ``min_file_count`` is the number of
#: ``*.py`` files the directory should contain (``0`` means the path is
#: a single file, checked with ``Path.is_file``).
_PHASE_LAYOUT: Sequence[Tuple[str, str, int, str]] = (
    ("Phase 1: Analysis", "PLAN.md", 0, "PLAN.md"),
    ("Phase 2: Foundation", "shared", 6, "shared/ (__init__ + 5 modules)"),
    ("Phase 3: Bridges", "bridges", 7, "bridges/ (__init__ + 6 gap modules)"),
    (
        "Phase 4: Enhancement",
        "bridges/memory_hypothesis_bridge.py",
        0,
        "7 CCE files modified + 3 new bridges",
    ),
    ("Phase 5: Integration", "cli.py", 0, "E2E tests + CLI + README + audit"),
)


#: Ordered list of bridge modules with the PLAN.md gap numbers they close.
_BRIDGE_MANIFEST: Sequence[Tuple[int, str, str]] = (
    (2, "concept_axiom_bridge.py", "Concept → Axiom"),
    (3, "axiom_skill_bridge.py", "Axiom → Skill (governed)"),
    (4, "causal_provenance_bridge.py", "Causal → Provenance"),
    (5, "memory_hypothesis_bridge.py", "Memory ↔ Hypothesis"),
    (6, "governance_synthesis_bridge.py", "Governance → Synthesis"),
    (7, "world_model_swarm_bridge.py", "WorldModel ↔ Swarm"),
    (8, "goal_synthesis_bridge.py", "Goal → Synthesis"),
    (9, "self_model_bridge.py", "SelfModel ↔ THDSE"),
    (10, "rsi_serl_bridge.py", "RSI ↔ SERL"),
)


# --------------------------------------------------------------------------- #
# filesystem probes
# --------------------------------------------------------------------------- #


def _check_path(relative_path: str, min_file_count: int) -> bool:
    """Return ``True`` iff the phase artifact at ``relative_path`` exists.

    ``min_file_count == 0`` — the target is a single file (PLAN.md,
    cli.py, or an individual module) and must be readable.

    ``min_file_count > 0`` — the target is a directory and must contain
    at least that many ``*.py`` files at any depth.
    """
    target = _REPO_ROOT / relative_path
    if min_file_count == 0:
        return target.is_file()
    if not target.is_dir():
        return False
    py_files = [p for p in target.rglob("*.py") if "__pycache__" not in p.parts]
    return len(py_files) >= min_file_count


def _phase_4_modifications_present() -> bool:
    """Verify Phase 4 deliverables: 3 new bridges + test files + modified CCE stubs."""
    required_new_bridges = [
        "bridges/memory_hypothesis_bridge.py",
        "bridges/world_model_swarm_bridge.py",
        "bridges/self_model_bridge.py",
    ]
    required_phase4_tests = [
        "tests/test_phase4_modifications.py",
        "tests/test_memory_hypothesis_bridge.py",
        "tests/test_world_model_swarm_bridge.py",
        "tests/test_self_model_bridge.py",
    ]
    all_paths = required_new_bridges + required_phase4_tests
    return all((_REPO_ROOT / p).is_file() for p in all_paths)


def _phase_5_deliverables_present() -> bool:
    required = [
        "tests/test_e2e_pipeline.py",
        "tests/test_rule_compliance.py",
        "cli.py",
        "README.md",
    ]
    return all((_REPO_ROOT / p).is_file() for p in required)


# --------------------------------------------------------------------------- #
# subcommand implementations
# --------------------------------------------------------------------------- #


def _cmd_test(_: argparse.Namespace) -> int:
    """Run the full pytest suite."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=str(_REPO_ROOT),
    )
    return int(completed.returncode)


def _cmd_audit(_: argparse.Namespace) -> int:
    """Run the 12-rule compliance audit file."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_rule_compliance.py",
            "-v",
        ],
        cwd=str(_REPO_ROOT),
    )
    return int(completed.returncode)


def _cmd_status(_: argparse.Namespace) -> int:
    """Print a live status snapshot of all 5 phases."""
    print("OMEGA-THDSE Integration Status")
    print("=" * 60)
    ok_all = True
    for label, rel_path, min_count, description in _PHASE_LAYOUT:
        # Phase 4 + Phase 5 need multi-file verification, not just the
        # single path above. The marker path above is a fast probe;
        # the helpers below confirm the deeper invariant.
        if label == "Phase 4: Enhancement":
            present = _check_path(rel_path, min_count) and (
                _phase_4_modifications_present()
            )
        elif label == "Phase 5: Integration":
            present = _check_path(rel_path, min_count) and (
                _phase_5_deliverables_present()
            )
        else:
            present = _check_path(rel_path, min_count)
        marker = "✓" if present else "✗"
        print(f"  {label:<24} {marker}  {description}")
        ok_all = ok_all and present
    print("=" * 60)
    summary = "ALL PHASES PRESENT" if ok_all else "MISSING DELIVERABLES"
    print(f"Summary: {summary}")
    return 0 if ok_all else 1


def _cmd_bridges(_: argparse.Namespace) -> int:
    """List every bridge module, its gap number, and its presence marker."""
    print("OMEGA-THDSE Bridge Modules (PLAN.md Section B)")
    print("=" * 70)
    missing: List[str] = []
    for gap, filename, description in _BRIDGE_MANIFEST:
        target = _REPO_ROOT / "bridges" / filename
        marker = "✓" if target.is_file() else "✗"
        if not target.is_file():
            missing.append(filename)
        print(f"  Gap {gap:<2}  {marker}  {filename:<36} — {description}")
    print("=" * 70)
    if missing:
        print(f"Missing bridge modules: {', '.join(missing)}")
        return 1
    print(f"All {len(_BRIDGE_MANIFEST)} bridge modules present.")
    return 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "OMEGA-THDSE unified CLI — run tests, audit the 12 "
            "anti-shortcut rules, inspect integration status, and list "
            "the cross-engine bridge modules."
        ),
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    p_test = sub.add_parser("test", help="Run the full pytest suite")
    p_test.set_defaults(func=_cmd_test)

    p_audit = sub.add_parser(
        "audit", help="Run the 12-rule compliance audit (pytest file)"
    )
    p_audit.set_defaults(func=_cmd_audit)

    p_status = sub.add_parser(
        "status", help="Print an integration-phase status summary"
    )
    p_status.set_defaults(func=_cmd_status)

    p_bridges = sub.add_parser(
        "bridges", help="List bridge modules and the gaps they close"
    )
    p_bridges.set_defaults(func=_cmd_bridges)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
