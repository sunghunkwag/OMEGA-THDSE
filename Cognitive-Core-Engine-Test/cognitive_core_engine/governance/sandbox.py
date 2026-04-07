from __future__ import annotations

import ast
import math
import random
import re
import copy
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from cognitive_core_engine.governance.utils import clamp

# ---------------------------------------------------------------------------
# SAFE_FUNCS / GRAMMAR_PROBS  (needed by validators & exec helpers below)
# ---------------------------------------------------------------------------

SAFE_FUNCS: Dict[str, Callable] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "tanh": math.tanh,
    "abs": abs,
    "sqrt": lambda x: math.sqrt(abs(x) + 1e-12),
    "log": lambda x: math.log(abs(x) + 1e-12),
    "pow2": lambda x: x * x,
    "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-clamp(x, -500, 500))),
    "gamma": lambda x: math.gamma(abs(x) + 1e-09) if abs(x) < 170 else float("inf"),
    "erf": math.erf,
    "ceil": math.ceil,
    "floor": math.floor,
    "sign": lambda x: math.copysign(1.0, x),
    # list helpers (legacy)
    "sorted": sorted,
    "reversed": reversed,
    "max": max,
    "min": min,
    "sum": sum,
    "len": len,
    "list": list,
}

GRAMMAR_PROBS: Dict[str, float] = {k: 1.0 for k in SAFE_FUNCS}
GRAMMAR_PROBS.update({"binop": 2.0, "call": 15.0, "const": 1.0, "var": 2.0})

SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "float": float,
    "int": int,
    "len": len,
    "range": range,
    "list": list,
    "sorted": sorted,
    "reversed": reversed,
    "sum": sum,
}

# ---------------------------
# Algo-mode safe primitives
# ---------------------------

def make_list(size: int = 0, fill: Any = 0) -> List[Any]:
    size = int(clamp(size, 0, 256))
    return [fill for _ in range(size)]

def list_len(xs: Any) -> int:
    return len(xs) if isinstance(xs, list) else 0

def list_get(xs: Any, idx: int, default: Any = 0) -> Any:
    if not isinstance(xs, list) or not xs:
        return default
    i = int(idx)
    if i < 0:
        i = 0
    if i >= len(xs):
        i = len(xs) - 1
    return xs[i]

def list_set(xs: Any, idx: int, val: Any) -> List[Any]:
    if not isinstance(xs, list):
        return make_list()
    if not xs:
        return [val]
    i = int(idx)
    if i < 0:
        i = 0
    if i >= len(xs):
        i = len(xs) - 1
    ys = list(xs)
    ys[i] = val
    return ys

def list_push(xs: Any, val: Any) -> List[Any]:
    ys = list(xs) if isinstance(xs, list) else []
    if len(ys) >= 256:
        return ys
    ys.append(val)
    return ys

def list_pop(xs: Any, default: Any = 0) -> Tuple[List[Any], Any]:
    ys = list(xs) if isinstance(xs, list) else []
    if not ys:
        return (ys, default)
    val = ys.pop()
    return (ys, val)

def list_swap(xs: Any, i: int, j: int) -> List[Any]:
    ys = list(xs) if isinstance(xs, list) else []
    if not ys:
        return ys
    a = int(clamp(i, 0, len(ys) - 1))
    b = int(clamp(j, 0, len(ys) - 1))
    ys[a], ys[b] = ys[b], ys[a]
    return ys

def list_copy(xs: Any) -> List[Any]:
    return list(xs) if isinstance(xs, list) else []

def make_map() -> Dict[Any, Any]:
    return {}

def map_get(m: Any, key: Any, default: Any = 0) -> Any:
    if not isinstance(m, dict):
        return default
    return m.get(key, default)

def map_set(m: Any, key: Any, val: Any) -> Dict[Any, Any]:
    d = dict(m) if isinstance(m, dict) else {}
    if len(d) >= 256 and key not in d:
        return d
    d[key] = val
    return d

def map_has(m: Any, key: Any) -> bool:
    return isinstance(m, dict) and key in m

def safe_range(n: int, limit: int = 256) -> List[int]:
    n = int(clamp(n, 0, limit))
    return list(range(n))

def safe_irange(a: int, b: int, limit: int = 256) -> List[int]:
    a = int(clamp(a, -limit, limit))
    b = int(clamp(b, -limit, limit))
    if a <= b:
        return list(range(a, b))
    return list(range(a, b, -1))

SAFE_ALGO_FUNCS: Dict[str, Callable] = {
    "make_list": make_list,
    "list_len": list_len,
    "list_get": list_get,
    "list_set": list_set,
    "list_push": list_push,
    "list_pop": list_pop,
    "list_swap": list_swap,
    "list_copy": list_copy,
    "make_map": make_map,
    "map_get": map_get,
    "map_set": map_set,
    "map_has": map_has,
    "safe_range": safe_range,
    "safe_irange": safe_irange,
    "clamp": clamp,
    "abs": abs,
    "min": min,
    "max": max,
    "int": int,
}

SAFE_VARS = {"x"} | {f"v{i}" for i in range(10)}


# grid helpers (ARC-like)
def _g_rot90(g):
    return [list(r) for r in zip(*g[::-1])]

def _g_flip(g):
    return g[::-1]

def _g_inv(g):
    return [[1 - c if c in (0, 1) else c for c in r] for r in g]

def _g_get(g, r, c):
    return g[r % len(g)][c % len(g[0])] if g and g[0] else 0

SAFE_FUNCS.update({"rot90": _g_rot90, "flip": _g_flip, "inv": _g_inv, "get": _g_get})
for k in ["rot90", "flip", "inv", "get"]:
    GRAMMAR_PROBS[k] = 1.0


# ---------------------------
# Safety: step limit + validators
# ---------------------------

class StepLimitExceeded(Exception):
    pass

class StepLimitTransformer(ast.NodeTransformer):
    """Inject step counting into loops and function bodies to prevent non-termination."""

    def __init__(self, limit: int = 5000):
        self.limit = limit

    def _inject_steps(self, node: ast.FunctionDef) -> None:
        glob = ast.Global(names=["_steps"])
        reset = ast.parse("_steps = 0").body[0]
        inc = ast.parse("_steps += 1").body[0]
        check = ast.parse(f"if _steps > {self.limit}: raise StepLimitExceeded()").body[0]
        node.body.insert(0, glob)
        node.body.insert(1, reset)
        node.body.insert(2, inc)
        node.body.insert(3, check)

    def visit_FunctionDef(self, node):
        self._inject_steps(node)
        self.generic_visit(node)
        return node

    def visit_While(self, node):
        inc = ast.parse("_steps += 1").body[0]
        check = ast.parse(f"if _steps > {self.limit}: raise StepLimitExceeded()").body[0]
        node.body.insert(0, inc)
        node.body.insert(1, check)
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        inc = ast.parse("_steps += 1").body[0]
        check = ast.parse(f"if _steps > {self.limit}: raise StepLimitExceeded()").body[0]
        node.body.insert(0, inc)
        node.body.insert(1, check)
        self.generic_visit(node)
        return node


class CodeValidator(ast.NodeVisitor):
    """
    Allow a safe subset of Python: assignments, flow control, simple expressions, calls to safe names.
    Forbid imports, attribute access, comprehensions, lambdas, etc.
    """

    _allowed = [
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.Name,
        ast.Constant,
        ast.Expr,
        ast.If,
        ast.While,
        ast.For,
        ast.Break,
        ast.Continue,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.List,
        ast.Tuple,  # critical for tuple-assign (swap)
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.Load,
        ast.Store,
        ast.IfExp,
        ast.operator,
        ast.boolop,
        ast.unaryop,
        ast.cmpop,
    ]
    if hasattr(ast, "Index"):
        _allowed.append(ast.Index)

    ALLOWED = tuple(_allowed)

    def __init__(self):
        self.ok, self.err = (True, None)

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden: {type(node).__name__}")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
                return
        if isinstance(node, ast.Call):
            # forbid attribute calls (e.g., os.system)
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
                return
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in SAFE_BUILTINS:
                self.ok, self.err = (False, "Forbidden subscript on builtin")
                return
        super().generic_visit(node)

def validate_code(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = CodeValidator()
        v.visit(tree)
        return (v.ok, v.err or "")
    except Exception as e:
        return (False, str(e))


class ProgramValidator(ast.NodeVisitor):
    """Strict program-mode validator: Assign/If/Return only, no loops or attributes."""

    _allowed = [
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.Name,
        ast.Constant,
        ast.Expr,
        ast.If,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.Load,
        ast.Store,
        ast.IfExp,
        ast.operator,
        ast.boolop,
        ast.unaryop,
        ast.cmpop,
    ]
    if hasattr(ast, "Index"):
        _allowed.append(ast.Index)

    ALLOWED = tuple(_allowed)

    def __init__(self):
        self.ok, self.err = (True, None)

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden program node: {type(node).__name__}")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
                return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
                return
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in SAFE_BUILTINS:
                self.ok, self.err = (False, "Forbidden subscript on builtin")
                return
        super().generic_visit(node)


def validate_program(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = ProgramValidator()
        v.visit(tree)
        return (v.ok, v.err or "")
    except Exception as e:
        return (False, str(e))


class AlgoProgramValidator(ast.NodeVisitor):
    """Algo-mode validator with bounded structure and constrained attribute access."""

    _allowed = [
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.Name,
        ast.Constant,
        ast.Expr,
        ast.If,
        ast.For,
        ast.While,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.BoolOp,
        ast.IfExp,
        ast.Call,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Attribute,
        ast.Subscript,
        ast.Load,
        ast.Store,
        ast.operator,
        ast.boolop,
        ast.unaryop,
        ast.cmpop,
    ]
    if hasattr(ast, "Index"):
        _allowed.append(ast.Index)

    ALLOWED = tuple(_allowed)

    def __init__(self):
        self.ok, self.err = (True, None)

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden: {type(node).__name__}")
            return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
                return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
                return
        super().generic_visit(node)


def algo_program_limits_ok(
    code: str,
    max_nodes: int = 420,
    max_depth: int = 32,
    max_funcs: int = 8,
    max_locals: int = 48,
    max_consts: int = 128,
    max_subscripts: int = 64,
) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    nodes = sum(1 for _ in ast.walk(tree))
    depth = ast_depth(code)
    funcs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    locals_set = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    consts = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant))
    subs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Subscript))
    return (
        nodes <= max_nodes
        and depth <= max_depth
        and funcs <= max_funcs
        and len(locals_set) <= max_locals
        and consts <= max_consts
        and subs <= max_subscripts
    )


def validate_algo_program(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = AlgoProgramValidator()
        v.visit(tree)
        if not v.ok:
            return (False, v.err or "")
        if not algo_program_limits_ok(code):
            return (False, "algo_program_limits")
        return (True, "")
    except Exception as e:
        return (False, str(e))


class ExprValidator(ast.NodeVisitor):
    """Validate a single expression (mode='eval') allowing only safe names and safe call forms."""
    ALLOWED = (
        ast.Expression,
        ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
        ast.Call,
        ast.Attribute,
        ast.Name, ast.Load,
        ast.Constant,
        ast.List, ast.Tuple, ast.Dict,
        ast.Set,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        ast.Subscript, ast.Slice,
        ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
    )

    def __init__(self, allowed_names: Set[str]):
        self.allowed_names = allowed_names
        self.ok = True
        self.err: Optional[str] = None

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden expr node: {type(node).__name__}")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
            if node.id not in self.allowed_names:
                self.ok, self.err = (False, f"Unknown name: {node.id}")
                return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
                return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
                return
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in SAFE_BUILTINS:
                self.ok, self.err = (False, "Forbidden subscript on builtin")
                return
        super().generic_visit(node)

def validate_expr(expr: str, extra: Optional[Set[str]] = None) -> Tuple[bool, str]:
    """PHASE A: validate expression with safe names only."""
    try:
        extra = extra or set()
        allowed = set(SAFE_FUNCS.keys()) | set(SAFE_BUILTINS.keys()) | set(SAFE_VARS) | set(extra)
        tree = ast.parse(expr, mode="eval")
        v = ExprValidator(allowed)
        v.visit(tree)
        return (v.ok, v.err or "")
    except Exception as e:
        return (False, str(e))

def safe_eval(expr: str, x: Any, extra_funcs: Optional[Dict[str, Callable]] = None) -> Any:
    """PHASE A: safe evaluation of expressions with optional helper functions."""
    ok, _ = validate_expr(expr, extra=set(extra_funcs or {}))
    if not ok:
        return float("nan")
    try:
        env: Dict[str, Any] = {}
        env.update(SAFE_FUNCS)
        env.update(SAFE_BUILTINS)
        if extra_funcs:
            env.update(extra_funcs)
        env["x"] = x
        for i in range(10):
            env[f"v{i}"] = x
        return eval(compile(ast.parse(expr, mode="eval"), "<expr>", "eval"), {"__builtins__": {}}, env)
    except Exception:
        return float("nan")


def node_count(code: str) -> int:
    try:
        return sum(1 for _ in ast.walk(ast.parse(code)))
    except Exception:
        return 999

def ast_depth(code: str) -> int:
    try:
        tree = ast.parse(code)
    except Exception:
        return 0
    max_depth = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1))
    return max_depth


def program_limits_ok(code: str, max_nodes: int = 200, max_depth: int = 20, max_locals: int = 16) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    nodes = sum(1 for _ in ast.walk(tree))
    depth = ast_depth(code)
    locals_set = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return nodes <= max_nodes and depth <= max_depth and len(locals_set) <= max_locals


def safe_exec(code: str, x: Any, timeout_steps: int = 1000, extra_env: Optional[Dict[str, Any]] = None) -> Any:
    """Execute candidate code with step limit. Code must define run(x). Returns Any (float/list/grid)."""
    try:
        tree = ast.parse(code)
        transformer = StepLimitTransformer(timeout_steps)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)

        env: Dict[str, Any] = {"_steps": 0, "StepLimitExceeded": StepLimitExceeded}
        env.update(SAFE_FUNCS)
        env.update(SAFE_BUILTINS)
        if extra_env:
            env.update(extra_env)

        exec(compile(tree, "<lgp>", "exec"), {"__builtins__": {}}, env)
        if "run" not in env:
            return float("nan")
        return env["run"](x)
    except StepLimitExceeded:
        return float("nan")
    except Exception:
        return float("nan")


def safe_exec_algo(
    code: str,
    inp: Any,
    timeout_steps: int = 2000,
    max_runtime_ms: int = 50,
    extra_env: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, int, bool]:
    """Execute algo candidate code with strict step/time limits."""
    start = time.time()
    try:
        tree = ast.parse(code)
        transformer = StepLimitTransformer(timeout_steps)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)

        env: Dict[str, Any] = {"_steps": 0, "StepLimitExceeded": StepLimitExceeded}
        env.update(SAFE_ALGO_FUNCS)
        if extra_env:
            env.update(extra_env)

        exec(compile(tree, "<algo>", "exec"), {"__builtins__": {}}, env)
        if "run" not in env:
            return (None, env.get("_steps", 0), True)
        out = env["run"](inp)
        elapsed_ms = int((time.time() - start) * 1000)
        timed_out = elapsed_ms > max_runtime_ms
        return (out, int(env.get("_steps", 0)), timed_out)
    except StepLimitExceeded:
        return (None, int(env.get("_steps", 0) if "env" in locals() else 0), True)
    except Exception:
        return (None, int(env.get("_steps", 0) if "env" in locals() else 0), True)


def safe_exec_engine(code: str, context: Dict[str, Any], timeout_steps: int = 5000) -> Any:
    """Execute meta-engine code (selection/crossover) with safety limits."""
    try:
        tree = ast.parse(str(code))
        transformer = StepLimitTransformer(timeout_steps)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)

        env: Dict[str, Any] = {"_steps": 0, "StepLimitExceeded": StepLimitExceeded}
        env.update({
            "random": random,
            "math": math,
            "max": max,
            "min": min,
            "len": len,
            "sum": sum,
            "sorted": sorted,
            "int": int,
            "float": float,
            "list": list,
        })
        env.update(context)

        exec(compile(tree, "<engine>", "exec"), env)
        if "run" in env:
            return env["run"]()
        return None
    except Exception:
        return None


def safe_load_module(code: str, timeout_steps: int = 5000) -> Optional[Dict[str, Any]]:
    """PHASE B: safely load a learner module with a restricted environment."""
    ok, err = validate_code(code)
    if not ok:
        return None
    try:
        tree = ast.parse(code)
        transformer = StepLimitTransformer(timeout_steps)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        env: Dict[str, Any] = {"_steps": 0, "StepLimitExceeded": StepLimitExceeded}
        env.update(SAFE_FUNCS)
        env.update(SAFE_BUILTINS)
        exec(compile(tree, "<learner>", "exec"), {"__builtins__": {}}, env)
        return env
    except Exception:
        return None
