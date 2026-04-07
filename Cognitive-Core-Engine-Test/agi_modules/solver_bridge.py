"""
Solver Bridge (BN-07)

Wraps the ARC-AGI and HumanEval solvers into callables matching the
signatures expected by ExternalBenchmarkHarness.run_full_benchmark().

    arc_solve_fn:       fn(task_context) -> output_grid | None
    humaneval_solve_fn: fn(prompt) -> code_body | None
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple


def create_arc_solve_fn() -> Callable:
    """Create the ARC solver callable for run_full_benchmark().

    The harness passes a task_context dict with "train" pairs and
    "test_input" grid.  The solver infers a rule from train pairs
    and applies it to the test input.
    """
    from agi_modules.arc_solver import solve_arc_task

    def arc_fn(task_context: dict):
        return solve_arc_task(task_context)

    return arc_fn


def create_humaneval_solve_fn() -> Callable:
    """Create the HumanEval solver callable for run_full_benchmark().

    The harness passes the prompt string; the solver returns the
    function body or None.
    """
    from agi_modules.humaneval_solver import solve_humaneval
    return solve_humaneval


def create_solver_pair() -> Tuple[Callable, Callable]:
    """Return (arc_solve_fn, humaneval_solve_fn) for run_full_benchmark()."""
    return create_arc_solve_fn(), create_humaneval_solve_fn()
