from __future__ import annotations

import ast
import json
import random
import re
import textwrap
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from cognitive_core_engine.governance.utils import sha256
from cognitive_core_engine.governance.engine_types import Genome, LearnerGenome
from cognitive_core_engine.governance.sandbox import validate_code, safe_eval, SAFE_FUNCS, GRAMMAR_PROBS


def _pick_node(rng: random.Random, body: ast.AST) -> ast.AST:
    nodes = list(ast.walk(body))
    return rng.choice(nodes[1:]) if len(nodes) > 1 else body

def _to_src(body: ast.AST) -> str:
    try:
        return ast.unparse(body)
    except Exception:
        return "x"

def _random_expr(rng: random.Random, depth: int = 0) -> str:
    if depth > 2:
        return rng.choice(["x", "v0", str(rng.randint(0, 9))])
    options = ["binop", "call", "const", "var"]
    weights = [GRAMMAR_PROBS.get(k, 1.0) for k in options]
    mtype = rng.choices(options, weights=weights, k=1)[0]
    if mtype == "binop":
        op = rng.choice(["+", "-", "*", "/", "**", "%"])
        return f"({_random_expr(rng, depth + 1)} {op} {_random_expr(rng, depth + 1)})"
    if mtype == "call":
        funcs = list(SAFE_FUNCS.keys())
        f_weights = [GRAMMAR_PROBS.get(f, 0.5) for f in funcs]
        fname = rng.choices(funcs, weights=f_weights, k=1)[0]
        return f"{fname}({_random_expr(rng, depth + 1)})"
    if mtype == "const":
        return f"{rng.uniform(-2, 2):.2f}"
    return rng.choice(["x", "v0"])


def op_insert_assign(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    var = f"v{rng.randint(0, 3)}"
    expr = _random_expr(rng)
    new_stmts.insert(idx, f"{var} = {expr}")
    return new_stmts

def op_insert_if(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)
    cond = f"v{rng.randint(0, 3)} < {rng.randint(0, 10)}"
    block = [f"    {s}" for s in new_stmts[idx: idx + 2]]
    new_stmts[idx: idx + 2] = [f"if {cond}:"] + block
    return new_stmts

def op_insert_while(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)
    cond = f"v{rng.randint(0, 3)} < {rng.randint(0, 10)}"
    block = [f"    {s}" for s in new_stmts[idx: idx + 2]]
    new_stmts[idx: idx + 2] = [f"while {cond}:"] + block
    return new_stmts

def op_delete_stmt(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    new_stmts.pop(rng.randint(0, len(new_stmts) - 1))
    return new_stmts

def op_modify_line(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)
    if "=" in new_stmts[idx]:
        var = new_stmts[idx].split("=")[0].strip()
        new_stmts[idx] = f"{var} = {_random_expr(rng)}"
    return new_stmts

def op_tweak_const(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)

    class TweakTransformer(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                val = float(node.value)
                new_val = val + rng.gauss(0, 0.1 * abs(val) + 0.01)
                if rng.random() < 0.05:
                    new_val = -val
                if rng.random() < 0.05:
                    new_val = 0.0
                return ast.Constant(value=new_val)
            return node

    try:
        tree = ast.parse(new_stmts[idx], mode="exec")
        new_tree = TweakTransformer().visit(tree)
        ast.fix_missing_locations(new_tree)
        new_stmts[idx] = ast.unparse(new_tree).strip()
    except Exception:
        pass
    return new_stmts

def op_change_binary(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)
    pops = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod]

    class OpTransformer(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if rng.random() < 0.5:
                node.op = rng.choice(pops)()
            return node

    try:
        tree = ast.parse(new_stmts[idx], mode="exec")
        new_tree = OpTransformer().visit(tree)
        ast.fix_missing_locations(new_tree)
        new_stmts[idx] = ast.unparse(new_tree).strip()
    except Exception:
        pass
    return new_stmts

def op_list_manipulation(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    ops = [
        f"v{rng.randint(0,3)} = x[{rng.randint(0,2)}]",
        f"if len(x) > {rng.randint(1,5)}: v{rng.randint(0,3)} = x[0]",
        "v0, v1 = v1, v0",  # requires Tuple allowed
        f"v{rng.randint(0,3)} = sorted(x)",
    ]
    new_stmts.insert(idx, rng.choice(ops))
    return new_stmts

def op_modify_return(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    active_vars = ["x"] + [f"v{i}" for i in range(4)]
    for i in range(len(new_stmts) - 1, -1, -1):
        if new_stmts[i].strip().startswith("return "):
            new_stmts[i] = f"return {rng.choice(active_vars)}"
            return new_stmts
    new_stmts.append(f"return {rng.choice(active_vars)}")
    return new_stmts


def op_learner_update_step(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    ops = [
        "mem['w'] = mem['w'] + lr * (y_true - y_pred) * x",
        "mem['b'] = mem['b'] + lr * (y_true - y_pred)",
        "mem['t'] = mem['t'] + 1",
        "return mem",
    ]
    new_stmts.insert(idx, rng.choice(ops))
    return new_stmts


def op_learner_objective_tweak(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    expr = rng.choice([
        "return hold + 0.5*stress + 0.01*nodes",
        "return 0.6*hold + 0.3*stress + 0.1*train",
        "return hold + stress + 0.001*nodes",
    ])
    new_stmts.insert(idx, expr)
    return new_stmts


OPERATORS: Dict[str, Callable[[random.Random, List[str]], List[str]]] = {
    "insert_assign": op_insert_assign,
    "insert_if": op_insert_if,
    "insert_while": op_insert_while,
    "delete_stmt": op_delete_stmt,
    "modify_line": op_modify_line,
    "tweak_const": op_tweak_const,
    "change_binary": op_change_binary,
    "list_manip": op_list_manipulation,
    "modify_return": op_modify_return,
    "learner_update": op_learner_update_step,
    "learner_objective": op_learner_objective_tweak,
}
PRIMITIVE_OPS = list(OPERATORS.keys())

# @@OPERATORS_LIB_START@@
OPERATORS_LIB: Dict[str, Dict] = {}
# @@OPERATORS_LIB_END@@


def apply_synthesized_op(rng: random.Random, stmts: List[str], steps: List[str]) -> List[str]:
    result = stmts
    for step in steps:
        if step in OPERATORS:
            result = OPERATORS[step](rng, result)
    return result

def synthesize_new_operator(rng: random.Random) -> Tuple[str, Dict]:
    n_steps = rng.randint(2, 4)
    steps = [rng.choice(PRIMITIVE_OPS) for _ in range(n_steps)]
    name = f"synth_{sha256(''.join(steps) + str(time.time()))[:8]}"
    return (name, {"steps": steps, "score": 0.0})


def mutate_learner(rng: random.Random, learner: LearnerGenome, meta: "MetaState") -> LearnerGenome:
    """PHASE B: mutate a learner genome by selecting a block."""
    blocks = ["encode", "predict", "update", "objective"]
    block = rng.choice(blocks)
    op = meta.sample_op(rng)

    def apply_block(stmts: List[str]) -> List[str]:
        if op in OPERATORS:
            return OPERATORS[op](rng, stmts)
        return stmts

    if block == "encode":
        new_encode = apply_block(learner.encode_stmts)
        return LearnerGenome(new_encode, learner.predict_stmts, learner.update_stmts, learner.objective_stmts, parents=[learner.gid], op_tag=f"mut:{block}:{op}")
    if block == "predict":
        new_predict = apply_block(learner.predict_stmts)
        return LearnerGenome(learner.encode_stmts, new_predict, learner.update_stmts, learner.objective_stmts, parents=[learner.gid], op_tag=f"mut:{block}:{op}")
    if block == "update":
        new_update = apply_block(learner.update_stmts)
        return LearnerGenome(learner.encode_stmts, learner.predict_stmts, new_update, learner.objective_stmts, parents=[learner.gid], op_tag=f"mut:{block}:{op}")
    new_objective = apply_block(learner.objective_stmts)
    return LearnerGenome(learner.encode_stmts, learner.predict_stmts, learner.update_stmts, new_objective, parents=[learner.gid], op_tag=f"mut:{block}:{op}")


# ---------------------------
# Surrogate + MAP-Elites
# ---------------------------

class SurrogateModel:
    def __init__(self, k: int = 5):
        self.k = k
        self.memory: List[Tuple[List[float], float]] = []

    def _extract_features(self, code: str) -> List[float]:
        return [
            len(code),
            code.count("\n"),
            code.count("if "),
            code.count("while "),
            code.count("="),
            code.count("return "),
            code.count("("),
        ]

    def train(self, history: List[Dict]):
        self.memory = []
        for h in history[-200:]:
            src = h.get("code") or h.get("expr")
            if src and "score" in h and isinstance(h["score"], (int, float)):
                feat = self._extract_features(src)
                self.memory.append((feat, float(h["score"])))

    def predict(self, code: str) -> float:
        if not self.memory:
            return 0.0
        target = self._extract_features(code)
        dists = []
        for feat, score in self.memory:
            d = sum((f1 - f2) ** 2 for f1, f2 in zip(target, feat)) ** 0.5
            dists.append((d, score))
        dists.sort(key=lambda x: x[0])
        nearest = dists[: self.k]
        total_w = 0.0
        weighted = 0.0
        for d, s in nearest:
            w = 1.0 / (d + 1e-6)
            weighted += s * w
            total_w += w
        return weighted / total_w if total_w > 0 else 0.0


SURROGATE = SurrogateModel()


class MAPElitesArchive:
    def __init__(self, genome_cls: type = Genome):
        self.grid: Dict[Tuple[int, int], Tuple[float, Any]] = {}
        self.genome_cls = genome_cls

    def _features(self, code: str) -> Tuple[int, int]:
        l_bin = min(20, len(code) // 20)
        d_bin = min(10, code.count("\n") // 2)
        return (l_bin, d_bin)

    def add(self, genome: Any, score: float):
        feat = self._features(genome.code)
        if feat not in self.grid or score < self.grid[feat][0]:
            self.grid[feat] = (score, genome)

    def sample(self, rng: random.Random) -> Optional[Any]:
        if not self.grid:
            return None
        return rng.choice(list(self.grid.values()))[1]

    def snapshot(self) -> Dict:
        return {
            "grid_size": len(self.grid),
            "entries": [(list(k), v[0], asdict(v[1])) for k, v in self.grid.items()],
        }

    def from_snapshot(self, s: Dict) -> "MAPElitesArchive":
        ma = MAPElitesArchive(self.genome_cls)
        for k, score, g_dict in s.get("entries", []):
            ma.grid[tuple(k)] = (score, self.genome_cls(**g_dict))
        return ma


MAP_ELITES = MAPElitesArchive(Genome)
MAP_ELITES_LEARNER = MAPElitesArchive(LearnerGenome)

def map_elites_filename(mode: str) -> str:
    return "map_elites_learner.json" if mode == "learner" else "map_elites.json"

def save_map_elites(path: Path, archive: MAPElitesArchive):
    path.write_text(json.dumps(archive.snapshot(), indent=2), encoding="utf-8")

def load_map_elites(path: Path, archive: MAPElitesArchive):
    if path.exists():
        try:
            loaded = archive.from_snapshot(json.loads(path.read_text(encoding="utf-8")))
            archive.grid = loaded.grid
        except Exception:
            pass


# ---------------------------
# Operator library evolution
# ---------------------------

def evolve_operator_meta(rng: random.Random) -> Tuple[str, Dict]:
    candidates = [v for _, v in OPERATORS_LIB.items() if v.get("score", 0) > -5.0]
    if len(candidates) < 2:
        return synthesize_new_operator(rng)
    p1 = rng.choice(candidates)["steps"]
    p2 = rng.choice(candidates)["steps"]
    cut = rng.randint(0, min(len(p1), len(p2)))
    child_steps = p1[:cut] + p2[cut:]
    if rng.random() < 0.5:
        mut_type = rng.choice(["mod", "add", "del"])
        if mut_type == "mod" and child_steps:
            child_steps[rng.randint(0, len(child_steps) - 1)] = rng.choice(PRIMITIVE_OPS)
        elif mut_type == "add":
            child_steps.insert(rng.randint(0, len(child_steps)), rng.choice(PRIMITIVE_OPS))
        elif mut_type == "del" and len(child_steps) > 1:
            child_steps.pop(rng.randint(0, len(child_steps) - 1))
    child_steps = child_steps[:6] or [rng.choice(PRIMITIVE_OPS)]
    name = f"evo_{sha256(''.join(child_steps) + str(time.time()))[:8]}"
    return (name, {"steps": child_steps, "score": 0.0})

def maybe_evolve_operators_lib(rng: random.Random, threshold: int = 10) -> Optional[str]:
    # remove worst if very bad
    if len(OPERATORS_LIB) > 3:
        sorted_ops = sorted(OPERATORS_LIB.items(), key=lambda x: x[1].get("score", 0))
        worst_name, worst_spec = sorted_ops[0]
        if worst_spec.get("score", 0) < -threshold:
            del OPERATORS_LIB[worst_name]

    # add new until size
    if len(OPERATORS_LIB) < 8:
        if rng.random() < 0.7 and len(OPERATORS_LIB) >= 2:
            name, spec = evolve_operator_meta(rng)
        else:
            name, spec = synthesize_new_operator(rng)
        OPERATORS_LIB[name] = spec
        return name
    return None


# ---------------------------
# Curriculum generator (simple)
# ---------------------------
