"""Phase 8C.4 — Comparison report.

Loads ``benchmarks/results/baseline.json`` and
``benchmarks/results/phase8.json`` and prints a side-by-side table
of the per-problem F_eff and solve / partial rates.

PLAN.md Rule 20: ``solve_rate`` is the fraction of problems where a
synthesized program passes EVERY io_example. ``partial_rate`` is the
softer "best_pass_rate > 0.5" metric. The report prints both with
explicit labels — never reports partial_rate as solve_rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS = _REPO_ROOT / "benchmarks" / "results"


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _problem_table_row(
    name: str,
    baseline: Optional[Dict[str, Any]],
    phase8: Optional[Dict[str, Any]],
) -> str:
    def _fmt(d: Optional[Dict[str, Any]], key: str) -> str:
        if d is None:
            return "n/a"
        value = d.get(key)
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    return (
        f"  {name:<22} | "
        f"f_eff base={_fmt(baseline, 'f_eff'):>6} | "
        f"f_eff p8={_fmt(phase8, 'f_eff'):>6} | "
        f"pass base={_fmt(baseline, 'best_pass_rate'):>6} | "
        f"pass p8={_fmt(phase8, 'best_pass_rate'):>6}"
    )


def _index_problems(report: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not report:
        return {}
    return {p["name"]: p for p in report.get("problems", [])}


def main() -> int:
    baseline_path = _RESULTS / "baseline.json"
    phase8_path = _RESULTS / "phase8.json"
    baseline = _load(baseline_path)
    phase8 = _load(phase8_path)

    if baseline is None and phase8 is None:
        print(
            "Neither baseline.json nor phase8.json found. "
            "Run benchmarks/run_baseline.py and benchmarks/run_phase8.py first."
        )
        return 1

    print("=" * 78)
    print(" OMEGA-THDSE Phase 8 — Empirical Comparison")
    print("=" * 78)
    if baseline is None:
        print("  baseline.json: MISSING")
    if phase8 is None:
        print("  phase8.json: MISSING")
    print()

    base_idx = _index_problems(baseline)
    phase_idx = _index_problems(phase8)
    all_names = sorted(set(base_idx.keys()) | set(phase_idx.keys()))

    print(
        "  Problem                | Baseline F_eff | Phase8 F_eff "
        "| Baseline Pass | Phase8 Pass"
    )
    print(
        "  " + "-" * 76
    )
    for name in all_names:
        print(_problem_table_row(name, base_idx.get(name), phase_idx.get(name)))

    def _aggregate(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not report:
            return {
                "solve_rate": 0.0,
                "partial_rate": 0.0,
                "solved_count": 0,
                "problem_count": 0,
                "f_eff_aggregate": 0.0,
            }
        return report.get("totals", {})

    base_totals = _aggregate(baseline)
    phase_totals = _aggregate(phase8)

    print()
    print("  --- AGGREGATE METRICS ---")
    print(
        f"  solve_rate (ALL io_examples pass): "
        f"baseline={base_totals.get('solve_rate', 0):.3f}  "
        f"phase8={phase_totals.get('solve_rate', 0):.3f}"
    )
    print(
        f"  partial_rate (>50% io_examples):   "
        f"baseline={base_totals.get('partial_rate', 0):.3f}  "
        f"phase8={phase_totals.get('partial_rate', 0):.3f}"
    )
    print(
        f"  solved_count:                       "
        f"baseline={base_totals.get('solved_count', 0)}/"
        f"{base_totals.get('problem_count', 0)}  "
        f"phase8={phase_totals.get('solved_count', 0)}/"
        f"{phase_totals.get('problem_count', 0)}"
    )
    print(
        f"  f_eff_aggregate:                    "
        f"baseline={base_totals.get('f_eff_aggregate', 0):.4f}  "
        f"phase8={phase_totals.get('f_eff_aggregate', 0):.4f}"
    )
    print(
        f"  total syntheses attempted:          "
        f"baseline={base_totals.get('total_syntheses_attempted', 0)}  "
        f"phase8={phase_totals.get('total_syntheses_attempted', 0)}"
    )
    print()
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
