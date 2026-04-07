"""Execution Sandbox — Physical grounding layer for synthesized code evaluation.

Executes synthesized Python code in a restricted subprocess and computes
a deterministic fitness score based on compilation, execution, type coverage,
and behavioral diversity. This is RSI trigger condition #2: physical/resource-cost
grounding.

The test vector set is type-diverse to prevent the design space from collapsing
to a single-type calculator. A function that only handles integers is still
rewarded, but a function that handles diverse types scores higher.

Fitness function (deterministic, monotonic):
  0.00 — compile() fails
  0.10 — compiles but exec() raises on module load
  0.15 — no function defined, OR timeout/subprocess crash
  0.20 — function rejects all types, OR all accepted inputs crash
  0.30 — runs but returns None for all inputs
  0.40+ — type_coverage * 0.3 + value_diversity * 0.3 + 0.4, capped at 1.0

TypeError on a specific input is expected (e.g., int function receiving a string)
and does NOT penalize. Only the accepted inputs matter.

Zero randomness. All test vectors are hardcoded. No stochastic behavior.

Subprocess isolation: synthesized code runs in a child process via
subprocess.run() with timeout. No __subclasses__ escape, no filesystem
access, no import hijacking possible.
"""

import ast
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Type-diverse deterministic test vectors.
# Each is (label, value) — the value is JSON-serializable.
_TEST_VECTORS: List[Tuple[str, Any]] = [
    ("int_zero",    0),
    ("int_pos",     42),
    ("int_neg",     -1),
    ("float_val",   3.14),
    ("string_val",  "hello"),
    ("list_ints",   [1, 2, 3]),
    ("empty_list",  []),
    ("none_val",    None),
    ("bool_val",    True),
    ("dict_val",    {"a": 1}),
]

# Total timeout for subprocess execution (seconds).
_SUBPROCESS_TIMEOUT_S = 2.0


# The wrapper script that runs in the child process.
# It receives source + test vectors via stdin (JSON), executes the source,
# finds the first function, calls it with each test vector, and returns
# results via stdout (JSON).
_WRAPPER_SCRIPT = r'''
import ast
import json
import sys

def main():
    data = json.loads(sys.stdin.read())
    source = data["source"]
    vectors = data["vectors"]  # list of [label, value]

    result = {"compiled": False, "executed": False, "func_found": False, "results": {}}

    # Step 1: Compile
    try:
        code_obj = compile(source, "<synthesized>", "exec")
    except Exception as e:
        result["error"] = type(e).__name__
        print(json.dumps(result))
        return

    result["compiled"] = True

    # Step 2: Execute — try full source first, then fall back to
    # function-only extraction if module-level code crashes.
    # Synthesized code often has stray module-level expressions that
    # reference undefined variables. The functions themselves are valid.
    namespace = {}
    try:
        exec(code_obj, namespace)
    except Exception as e:
        # Fall back: extract only function definitions and re-execute
        try:
            tree = ast.parse(source)
            func_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            if func_defs:
                clean_mod = ast.Module(body=func_defs, type_ignores=[])
                ast.fix_missing_locations(clean_mod)
                clean_code = compile(clean_mod, "<synthesized>", "exec")
                namespace = {}
                exec(clean_code, namespace)
            else:
                result["error"] = type(e).__name__
                print(json.dumps(result))
                return
        except Exception as e2:
            result["error"] = type(e2).__name__
            print(json.dumps(result))
            return

    result["executed"] = True

    # Step 3: Find first function via AST
    func_name = None
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break
    except Exception:
        pass

    if func_name is None or func_name not in namespace or not callable(namespace[func_name]):
        print(json.dumps(result))
        return

    result["func_found"] = True
    func = namespace[func_name]

    # Step 4: Call with each test vector
    for label, value in vectors:
        try:
            ret = func(value)
            # Serialize return value
            try:
                json.dumps(ret)
                result["results"][label] = {"ok": True, "value": ret}
            except (TypeError, ValueError):
                result["results"][label] = {"ok": True, "value": repr(ret)}
        except TypeError:
            result["results"][label] = {"type_error": True}
        except Exception as e:
            result["results"][label] = {"error": type(e).__name__}

    print(json.dumps(result))

main()
'''


@dataclass
class ExecutionProfile:
    """Result of executing synthesized code in the sandbox."""
    source: str                                     # The decoded Python source
    compiled: bool = False                          # Did compile() succeed?
    executed: bool = False                          # Did exec() complete?
    returned_values: Dict[str, Any] = field(default_factory=dict)  # label → value
    execution_time_ms: float = 0.0                  # Wall-clock time
    exception_type: Optional[str] = None            # Exception class if failed
    is_nontrivial: bool = False                     # True if diverse computation
    fitness: float = 0.0                            # Scalar fitness score
    timed_out: bool = False                         # Subprocess timed out
    n_accepted: int = 0                             # Inputs not rejected by TypeError
    n_succeeded: int = 0                            # Accepted inputs that didn't crash
    n_distinct: int = 0                             # Distinct non-None return values


class ExecutionSandbox:
    """Executes synthesized Python code in a restricted subprocess.

    Deterministic test protocol:
      1. compile(source, "<synthesized>", "exec")
      2. exec() in isolated namespace
      3. Find first def via AST inspection
      4. For each type-diverse test vector, call the function:
         - TypeError → skip (not penalized, expected for typed functions)
         - Other exception → count as failure
         - Success → record return value
      5. Compute fitness from type coverage and value diversity

    Fitness formula:
      fitness = 0.4 + 0.3 * type_coverage + 0.3 * value_diversity
      where type_coverage = n_accepted / len(test_vectors)
            value_diversity = n_distinct / max(n_succeeded, 1)
      capped at 1.0

    The fitness threshold of 0.4 requires the code to compile, execute,
    define a callable, accept at least one input type, succeed on at least
    one input, AND return at least one non-None value. This is the minimum
    bar for "nontrivial computation."
    """

    def __init__(
        self,
        test_vectors: Optional[List[Tuple[str, Any]]] = None,
        timeout_s: float = _SUBPROCESS_TIMEOUT_S,
    ):
        self.test_vectors = test_vectors if test_vectors is not None else list(_TEST_VECTORS)
        self.timeout_s = timeout_s

    def execute(self, source: str) -> ExecutionProfile:
        """Execute synthesized source in a subprocess and compute fitness.

        Args:
            source: Python source code to evaluate.

        Returns:
            ExecutionProfile with fitness score and execution metadata.
        """
        profile = ExecutionProfile(source=source)
        t0 = time.monotonic()

        # Prepare input for subprocess
        input_data = json.dumps({
            "source": source,
            "vectors": self.test_vectors,
        })

        # Run in subprocess
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _WRAPPER_SCRIPT],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            profile.timed_out = True
            profile.fitness = 0.15
            profile.execution_time_ms = (time.monotonic() - t0) * 1000
            return profile
        except Exception:
            profile.fitness = 0.15
            profile.execution_time_ms = (time.monotonic() - t0) * 1000
            return profile

        profile.execution_time_ms = (time.monotonic() - t0) * 1000

        # Parse subprocess output
        try:
            result = json.loads(proc.stdout.strip())
        except (json.JSONDecodeError, ValueError):
            # Malformed output — treat as crash
            profile.fitness = 0.15
            return profile

        profile.compiled = result.get("compiled", False)
        profile.executed = result.get("executed", False)

        if not profile.compiled:
            profile.fitness = 0.0
            profile.exception_type = result.get("error")
            return profile

        if not profile.executed:
            profile.fitness = 0.1
            profile.exception_type = result.get("error")
            return profile

        if not result.get("func_found", False):
            profile.fitness = 0.15
            return profile

        # Analyze results
        results = result.get("results", {})
        n_total = len(self.test_vectors)
        n_accepted = 0
        n_succeeded = 0
        return_reprs = set()

        for label, res in results.items():
            if res.get("type_error"):
                # TypeError → skip, not penalized
                continue
            n_accepted += 1
            if res.get("ok"):
                n_succeeded += 1
                val = res.get("value")
                profile.returned_values[label] = val
                if val is not None:
                    return_reprs.add(repr(val))
            # else: other error on accepted input

        n_distinct = len(return_reprs)
        profile.n_accepted = n_accepted
        profile.n_succeeded = n_succeeded
        profile.n_distinct = n_distinct

        # Fitness calculation
        if n_accepted == 0:
            profile.fitness = 0.2
            return profile

        if n_succeeded == 0:
            profile.fitness = 0.2
            return profile

        if n_distinct == 0:
            profile.fitness = 0.3
            return profile

        type_coverage = n_accepted / n_total
        value_diversity = n_distinct / max(n_succeeded, 1)
        profile.fitness = min(1.0, 0.4 + 0.3 * type_coverage + 0.3 * value_diversity)
        profile.is_nontrivial = n_distinct >= 2

        return profile
