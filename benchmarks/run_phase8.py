"""Phase 8C.3 — Phase 8 measurement (behavioural + goal-direction + CEGR).

Same benchmark suite as :mod:`benchmarks.run_baseline`, but the
synthesizer is wired with the Phase 8 enhancements:

- Behavioural encoder attached so axioms carry an io profile and
  the dual-axis resonance blends structural + behavioural similarity.
- Goal-directed clique selection via :meth:`AxiomaticSynthesizer.
  synthesize_for_problem`.
- Counterexample-guided refinement on partially passing candidates.

Output: ``benchmarks/results/phase8.json``.
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
            use_behavioural=True,
            use_goal_direction=True,
            use_cegr=True,
            max_cycles=3,
        )
    except Z3Unavailable as exc:
        print(str(exc))
        return 1

    out_path = _REPO_ROOT / "benchmarks" / "results" / "phase8.json"
    write_json(out_path, report)

    totals = report["totals"]
    print("Phase 8 complete.")
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
