"""Tests for the Execution Sandbox (SERL Step 1).

9 deterministic tests covering the full fitness gradient with type-diverse
test vectors: compile failure, exec exception, no callable, all-None returns,
identity (diverse types), arithmetic (typed), len (collection-only),
timeout, and str (universal).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.sandbox import ExecutionSandbox, ExecutionProfile


@pytest.fixture
def sandbox():
    return ExecutionSandbox(timeout_s=3.0)


# Test 1: compile-fail source → fitness = 0.0
def test_compile_fail(sandbox):
    """Source that cannot compile gets fitness 0.0."""
    profile = sandbox.execute("def def def")
    assert profile.fitness == 0.0
    assert profile.compiled is False


# Test 2: "def f(x): raise ValueError" → fitness ≤ 0.2
def test_exec_raises_exception(sandbox):
    """Function that raises on every call gets fitness ≤ 0.2."""
    profile = sandbox.execute("def f(x): raise ValueError('boom')")
    assert profile.compiled is True
    assert profile.executed is True
    assert profile.fitness <= 0.2


# Test 3: "def f(x): pass" → fitness = 0.3 (returns None for all)
def test_all_none_returns(sandbox):
    """Function returning None for all inputs gets fitness 0.3."""
    profile = sandbox.execute("def f(x): pass")
    assert profile.compiled is True
    assert profile.executed is True
    assert profile.fitness == 0.3


# Test 4: "x = 42" (no function) → fitness = 0.15
def test_no_function_defined(sandbox):
    """Code with no function definition gets fitness 0.15."""
    profile = sandbox.execute("x = 42")
    assert profile.compiled is True
    assert profile.executed is True
    assert profile.fitness == 0.15


# Test 5: "def f(x): return x" → fitness > 0.5
#   Accepts most types, returns distinct values for each
def test_identity_function(sandbox):
    """Identity function accepts all types, returns distinct values."""
    profile = sandbox.execute("def f(x): return x")
    assert profile.compiled is True
    assert profile.executed is True
    assert profile.fitness > 0.5
    assert profile.n_accepted >= 8  # Most types accepted


# Test 6: "def f(x): return x * 2 + 1" → fitness > 0.4
#   Rejects string/list/dict via TypeError, succeeds on int/float/bool
def test_arithmetic_function(sandbox):
    """Arithmetic function: rejects non-numeric types (not penalized)."""
    profile = sandbox.execute("def f(x): return x * 2 + 1")
    assert profile.compiled is True
    assert profile.executed is True
    assert profile.fitness > 0.4


# Test 7: "def f(x): return len(x)" → fitness > 0.4
#   Rejects int/float/None/bool via TypeError, succeeds on string/list/dict
def test_len_function(sandbox):
    """len() function: rejects scalars (not penalized), succeeds on collections."""
    profile = sandbox.execute("def f(x): return len(x)")
    assert profile.compiled is True
    assert profile.executed is True
    assert profile.fitness > 0.4


# Test 8: "def f(x):\n    while True: pass" → fitness = 0.15 (timeout)
def test_timeout(sandbox):
    """Infinite loop in function triggers subprocess timeout → fitness = 0.15."""
    source = "def f(x):\n    while True: pass"
    profile = sandbox.execute(source)
    assert profile.fitness == 0.15
    assert profile.timed_out is True


# Test 9: "def f(x): return str(x)" → fitness > 0.6
#   Accepts all types, returns distinct strings for each
def test_universal_str_function(sandbox):
    """str() accepts all types, returns distinct strings → high fitness."""
    profile = sandbox.execute("def f(x): return str(x)")
    assert profile.compiled is True
    assert profile.executed is True
    assert profile.fitness > 0.6
    assert profile.n_accepted == 10  # All types accepted
