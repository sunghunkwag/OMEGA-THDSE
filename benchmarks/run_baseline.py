"""Phase 8C.2 — Baseline measurement (structural-only resonance, no goals).

Runs the existing system on the benchmark suite WITHOUT behavioural
encoding, WITHOUT goal-directed clique selection, and WITHOUT CEGR.
The result is the structural-only floor that Phase 8 must beat.

Output: ``benchmarks/results/baseline.json`` containing per-cycle
data so the comparison report can verify progressions externally.
PLAN.md Rule 23 forbids fabricated metrics — if Z3 is missing the
script prints the standard message and exits with code 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "thdse"))

from benchmarks.runner import Z3Unavailable, run_suite, write_json  # noqa: E402
from benchmarks.sorting_synthesis import SEED_CORPUS, all_problems  # noqa: E402


def main() -> int:
    try:
        report = run_suite(
            problems=all_problems(),
            seed_corpus=SEED_CORPUS,
            use_behavioural=False,
            use_goal_direction=False,
            use_cegr=False,
            max_cycles=3,
        )
    except Z3Unavailable as exc:
        print(str(exc))
        return 1

    out_path = _REPO_ROOT / "benchmarks" / "results" / "baseline.json"
    write_json(out_path, report)

    totals = report["totals"]
    print("Baseline complete.")
    print(
        f"  solved {totals['solved_count']}/{totals['problem_count']} "
        f"(solve_rate={totals['solve_rate']:.3f}, "
        f"partial_rate={totals['partial_rate']:.3f})"
    )
    print(
        f"  total syntheses attempted: {totals['total_syntheses_attempted']}, "
        f"f_eff={totals['f_eff_aggregate']:.4f}"
    )
    print(f"  results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
