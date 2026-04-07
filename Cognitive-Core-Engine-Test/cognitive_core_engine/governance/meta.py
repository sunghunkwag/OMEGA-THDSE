"""Meta-learning, universe simulation, and global state management."""
from __future__ import annotations

import ast
import copy
import json
import math
import os
import random
import re
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from cognitive_core_engine.governance.utils import now_ms, sha256, safe_mkdir, read_json, write_json
from cognitive_core_engine.governance.sandbox import (
    safe_exec, safe_exec_algo, validate_code, validate_program,
    safe_exec_engine, safe_load_module, SAFE_BUILTINS,
)


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

class ProblemGeneratorV2:
    def __init__(self):
        self.archive: List[Dict] = []

    def evolve_task(self, rng: random.Random, current_elites: List[Genome]) -> TaskSpec:
        arc_tasks = get_arc_tasks()
        base_options = ["sort", "reverse", "max", "filter"]
        arc_options = [f"arc_{tid}" for tid in arc_tasks] if arc_tasks else []
        options = base_options + arc_options
        base_name = rng.choice(options) if options else "sort"
        level = rng.randint(1, 3)
        mn = 3 + level
        mx = 5 + level
        if base_name.startswith("arc_"):
            mn, mx = (3, 5)
        return TaskSpec(name=base_name, n_train=64, n_hold=32, x_min=float(mn), x_max=float(mx), noise=0.0)


# ---------------------------
# Task detective (seeding hints)
# ---------------------------

class TaskDetective:
    @staticmethod
    def detect_pattern(batch: Optional[Batch]) -> Optional[str]:
        if not batch or not batch.x_tr:
            return None
        check_set = list(zip(batch.x_tr[:5], batch.y_tr[:5]))
        is_sort = is_rev = is_max = is_min = is_len = True
        for x, y in check_set:
            if not isinstance(x, list) or not isinstance(y, (list, int, float)):
                return None
            if isinstance(y, list):
                if y != sorted(x):
                    is_sort = False
                if y != list(reversed(x)):
                    is_rev = False
            else:
                is_sort = is_rev = False
            if isinstance(y, (int, float)):
                if not x:
                    if y != 0:
                        is_len = False
                else:
                    if y != len(x):
                        is_len = False
                    if y != max(x):
                        is_max = False
                    if y != min(x):
                        is_min = False
            else:
                is_max = is_min = is_len = False
        if is_sort:
            return "HINT_SORT"
        if is_rev:
            return "HINT_REVERSE"
        if is_max:
            return "HINT_MAX"
        if is_min:
            return "HINT_MIN"
        if is_len:
            return "HINT_LEN"
        return None


def seed_genome(rng: random.Random, hint: Optional[str] = None) -> Genome:
    seeds = [
        ["return x"],
        ["return sorted(x)"],
        ["return list(reversed(x))"],
        ["v0 = sorted(x)", "return v0"],
        [f"return {_random_expr(rng, depth=0)}"],
    ]
    if hint == "HINT_SORT":
        seeds.extend([["return sorted(x)"]] * 5)
    elif hint == "HINT_REVERSE":
        seeds.extend([["return list(reversed(x))"]] * 5)
    elif hint == "HINT_MAX":
        seeds.extend([["return max(x)"]] * 5)
    elif hint == "HINT_MIN":
        seeds.extend([["return min(x)"]] * 5)
    elif hint == "HINT_LEN":
        seeds.extend([["return len(x)"]] * 5)
    return Genome(statements=rng.choice(seeds))


def seed_learner_genome(rng: random.Random, hint: Optional[str] = None) -> LearnerGenome:
    """PHASE B: learner seed set with simple predictors and objectives."""
    base_encode = ["return x"]
    base_predict = ["return z"]
    base_update = ["return mem"]
    base_obj = ["return hold + 0.5*stress + 0.01*nodes"]

    linear_predict = ["return mem['w'] * z + mem['b']"]
    linear_update = [
        "mem['w'] = mem['w'] + lr * (y_true - y_pred) * z",
        "mem['b'] = mem['b'] + lr * (y_true - y_pred)",
        "return mem",
    ]

    list_sort_predict = ["return sorted(z)"]
    list_reverse_predict = ["return list(reversed(z))"]
    list_max_predict = ["return max(z) if z else 0"]

    seeds = [
        LearnerGenome(base_encode, base_predict, base_update, base_obj),
        LearnerGenome(base_encode, linear_predict, linear_update, base_obj),
    ]

    if hint == "HINT_SORT":
        seeds.append(LearnerGenome(base_encode, list_sort_predict, base_update, base_obj))
    elif hint == "HINT_REVERSE":
        seeds.append(LearnerGenome(base_encode, list_reverse_predict, base_update, base_obj))
    elif hint == "HINT_MAX":
        seeds.append(LearnerGenome(base_encode, list_max_predict, base_update, base_obj))

    return rng.choice(seeds)


# ---------------------------
# Function library (learned helpers)
# ---------------------------

@dataclass
class LearnedFunc:
    name: str
    expr: str
    trust: float = 1.0
    uses: int = 0


class FunctionLibrary:
    def __init__(self, max_size: int = 16):
        self.funcs: Dict[str, LearnedFunc] = {}
        self.max_size = max_size

    def maybe_adopt(self, rng: random.Random, expr: str, threshold: float = 0.1) -> Optional[str]:
        if len(self.funcs) >= self.max_size or rng.random() > threshold:
            return None
        try:
            tree = ast.parse(expr, mode="eval").body
            nodes = list(ast.walk(tree))
            if len(nodes) < 4:
                return None
            sub = _pick_node(rng, tree)
            sub_expr = _to_src(sub)
            if node_count(sub_expr) < 3:
                return None
            ok, _ = validate_expr(sub_expr, extra=set(self.funcs.keys()))
            if not ok:
                return None
            name = f"h{len(self.funcs) + 1}"
            self.funcs[name] = LearnedFunc(name=name, expr=sub_expr)
            return name
        except Exception:
            return None

    def maybe_inject(self, rng: random.Random, expr: str) -> Tuple[str, Optional[str]]:
        if not self.funcs or rng.random() > 0.2:
            return (expr, None)
        fn = rng.choice(list(self.funcs.values()))
        fn.uses += 1
        try:
            call = f"{fn.name}(x)"
            new = expr.replace("x", call, 1) if rng.random() < 0.5 else f"({expr}+{call})"
            ok, _ = validate_expr(new, extra=set(self.funcs.keys()))
            return (new, fn.name) if ok else (expr, None)
        except Exception:
            return (expr, None)

    def update_trust(self, name: str, improved: bool):
        if name in self.funcs:
            self.funcs[name].trust *= 1.1 if improved else 0.9
            self.funcs[name].trust = clamp(self.funcs[name].trust, 0.1, 10.0)

    def get_helpers(self) -> Dict[str, Callable]:
        # helper functions callable from evolved programs
        helpers: Dict[str, Callable] = {}

        def make_helper(expr: str):
            return lambda x: safe_eval(expr, x, extra_funcs=helpers)

        for n, f in self.funcs.items():
            helpers[n] = make_helper(f.expr)
        return helpers

    def snapshot(self) -> Dict:
        return {"funcs": [asdict(f) for f in self.funcs.values()]}

    def merge(self, other: "FunctionLibrary"):
        for name, func in other.funcs.items():
            if name not in self.funcs:
                self.funcs[name] = func
            else:
                new_name = f"{name}_{len(self.funcs) + 1}"
                self.funcs[new_name] = LearnedFunc(name=new_name, expr=func.expr, trust=func.trust, uses=func.uses)

    @staticmethod
    def from_snapshot(s: Dict) -> "FunctionLibrary":
        lib = FunctionLibrary()
        for fd in s.get("funcs", []):
            lib.funcs[fd["name"]] = LearnedFunc(**fd)
        return lib


@dataclass
class LibraryRecord:
    descriptor: TaskDescriptor
    score_hold: float
    snapshot: Dict[str, Any]


class LibraryArchive:
    def __init__(self, k: int = 2):
        self.k = k
        self.records: List[LibraryRecord] = []

    def add(self, descriptor: TaskDescriptor, score_hold: float, lib: FunctionLibrary):
        self.records.append(LibraryRecord(descriptor=descriptor, score_hold=score_hold, snapshot=lib.snapshot()))

    def _distance(self, a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def select(self, descriptor: TaskDescriptor) -> List[FunctionLibrary]:
        if not self.records:
            return []
        vec = descriptor.vector()
        ranked = sorted(self.records, key=lambda r: (self._distance(vec, r.descriptor.vector()), r.score_hold))
        libs = []
        for rec in ranked[: self.k]:
            libs.append(FunctionLibrary.from_snapshot(rec.snapshot))
        return libs


# ---------------------------
# Grammar induction (single definition)
# ---------------------------

def induce_grammar(pool: List[Genome]):
    if not pool:
        return
    elites = pool[: max(10, len(pool) // 5)]
    counts = {k: 0.1 for k in GRAMMAR_PROBS}
    for g in elites:
        try:
            tree = ast.parse(g.code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in counts:
                        counts[node.func.id] += 1.0
                    counts["call"] += 1.0
                elif isinstance(node, ast.BinOp):
                    counts["binop"] += 1.0
                elif isinstance(node, ast.Name) and node.id == "x":
                    counts["var"] += 1.0
                elif isinstance(node, ast.Constant):
                    counts["const"] += 1.0
        except Exception:
            pass
    total = sum(counts.values())
    if total > 0:
        for k in counts:
            old = GRAMMAR_PROBS.get(k, 1.0)
            target = counts[k] / total * 100.0
            GRAMMAR_PROBS[k] = 0.8 * old + 0.2 * target


def extract_return_expr(stmts: List[str]) -> Optional[str]:
    for stmt in reversed(stmts):
        s = stmt.strip()
        if s.startswith("return "):
            return s[len("return ") :].strip()
    return None


def inject_helpers_into_statements(rng: random.Random, stmts: List[str], library: FunctionLibrary) -> List[str]:
    if not library.funcs:
        return stmts
    new_stmts = []
    injected = False
    for stmt in stmts:
        if not injected and stmt.strip().startswith("return "):
            expr = stmt.strip()[len("return ") :].strip()
            new_expr, helper_name = library.maybe_inject(rng, expr)
            if helper_name:
                stmt = f"return {new_expr}"
                injected = True
        new_stmts.append(stmt)
    return new_stmts


# ---------------------------
# MetaState (L0/L1 source-patchable)
# ---------------------------

OP_WEIGHT_INIT: Dict[str, float] = {
    k: (5.0 if k in ("modify_return", "insert_assign", "list_manip") else 1.0)
    for k in OPERATORS
}


@dataclass
class UpdateRuleGenome:
    """Representation for update rule parameters (meta-level learning algorithm)."""

    learning_rate: float
    momentum: float
    rejection_penalty: float
    reward_scale: float
    uid: str = ""

    def __post_init__(self):
        if not self.uid:
            self.uid = sha256(f"{self.learning_rate}:{self.momentum}:{self.rejection_penalty}:{self.reward_scale}")[:10]

    def apply(self, meta: "MetaState", op: str, delta: float, accepted: bool) -> None:
        reward = (max(0.0, -delta) * self.reward_scale) if accepted else -self.rejection_penalty
        velocity = meta.op_velocity.get(op, 0.0)
        velocity = self.momentum * velocity + reward
        meta.op_velocity[op] = velocity
        meta.op_weights[op] = clamp(meta.op_weights.get(op, 1.0) + self.learning_rate * velocity, 0.1, 8.0)

    def mutate(self, rng: random.Random) -> "UpdateRuleGenome":
        return UpdateRuleGenome(
            learning_rate=clamp(self.learning_rate + rng.uniform(-0.05, 0.05), 0.01, 0.5),
            momentum=clamp(self.momentum + rng.uniform(-0.1, 0.1), 0.0, 0.95),
            rejection_penalty=clamp(self.rejection_penalty + rng.uniform(-0.05, 0.05), 0.01, 0.5),
            reward_scale=clamp(self.reward_scale + rng.uniform(-0.1, 0.1), 0.2, 2.0),
        )

    @staticmethod
    def default() -> "UpdateRuleGenome":
        return UpdateRuleGenome(learning_rate=0.12, momentum=0.3, rejection_penalty=0.12, reward_scale=1.0)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "UpdateRuleGenome":
        return UpdateRuleGenome(
            learning_rate=float(data.get("learning_rate", 0.12)),
            momentum=float(data.get("momentum", 0.3)),
            rejection_penalty=float(data.get("rejection_penalty", 0.12)),
            reward_scale=float(data.get("reward_scale", 1.0)),
            uid=data.get("uid", ""),
        )

@dataclass
class MetaState:
    op_weights: Dict[str, float] = field(default_factory=lambda: dict(OP_WEIGHT_INIT))
    op_velocity: Dict[str, float] = field(default_factory=dict)
    mutation_rate: float = 0.8863
    crossover_rate: float = 0.1971
    complexity_lambda: float = 0.0001
    epsilon_explore: float = 0.4213
    adapt_steps: int = 8
    stuck_counter: int = 0
    update_rule: UpdateRuleGenome = field(default_factory=UpdateRuleGenome.default)
    strategy: EngineStrategy = field(default_factory=lambda: EngineStrategy(
        selection_code=DEFAULT_SELECTION_CODE,
        crossover_code=DEFAULT_CROSSOVER_CODE,
        mutation_policy_code=DEFAULT_MUTATION_CODE
    ))

    def sample_op(self, rng: random.Random) -> str:
        if rng.random() < self.epsilon_explore:
            return rng.choice(list(OPERATORS.keys()))
        total = sum(max(0.01, w) for w in self.op_weights.values())
        r = rng.random() * total
        acc = 0.0
        for k, w in self.op_weights.items():
            acc += max(0.01, w)
            if r <= acc:
                return k
        return rng.choice(list(OPERATORS.keys()))

    def update(self, op: str, delta: float, accepted: bool):
        if op in self.op_weights:
            self.update_rule.apply(self, op, delta, accepted)
        if not accepted:
            self.stuck_counter += 1
            if self.stuck_counter > 20:
                self.epsilon_explore = clamp(self.epsilon_explore + 0.02, 0.1, 0.4)
                self.mutation_rate = clamp(self.mutation_rate + 0.03, 0.4, 0.95)
        else:
            self.stuck_counter = 0
            self.epsilon_explore = clamp(self.epsilon_explore - 0.01, 0.05, 0.3)


class MetaCognitiveEngine:
    @staticmethod
    def analyze_execution(results: List[Tuple[Any, EvalResult]], meta: MetaState):
        errors = [r.err.split(":")[0] for _, r in results if (not r.ok and r.err)]
        if not errors:
            return
        counts = collections.Counter(errors)
        total_err = len(errors)
        if counts.get("TypeError", 0) > total_err * 0.3:
            if "binop" in GRAMMAR_PROBS:
                GRAMMAR_PROBS["binop"] *= 0.5
            GRAMMAR_PROBS["var"] = GRAMMAR_PROBS.get("var", 1.0) * 1.5
        if counts.get("IndexError", 0) > total_err * 0.3:
            if "list_manip" in meta.op_weights:
                meta.op_weights["list_manip"] *= 0.7
        if counts.get("StepLimitExceeded", 0) > total_err * 0.3:
            meta.complexity_lambda *= 2.0


# ---------------------------
# L1 Meta-optimizer policy
# ---------------------------

@dataclass
class MetaPolicy:
    weights: List[List[float]]
    bias: List[float]
    pid: str = ""

    @staticmethod
    def seed(rng: random.Random, n_outputs: int, n_inputs: int) -> "MetaPolicy":
        weights = [[rng.uniform(-0.2, 0.2) for _ in range(n_inputs)] for _ in range(n_outputs)]
        bias = [rng.uniform(-0.1, 0.1) for _ in range(n_outputs)]
        pid = sha256(json.dumps(weights) + json.dumps(bias))[:10]
        return MetaPolicy(weights=weights, bias=bias, pid=pid)

    def _linear(self, features: List[float], idx: int) -> float:
        w = self.weights[idx]
        return sum(fi * wi for fi, wi in zip(features, w)) + self.bias[idx]

    def act(self, descriptor: TaskDescriptor, stats: Dict[str, float]) -> Dict[str, Any]:
        features = descriptor.vector() + [
            stats.get("delta_best", 0.0),
            stats.get("auc_window", 0.0),
            stats.get("timeout_rate", 0.0),
            stats.get("avg_nodes", 0.0),
        ]
        outputs = [self._linear(features, i) for i in range(len(self.weights))]
        mutation_rate = clamp(0.5 + outputs[0], 0.05, 0.98)
        crossover_rate = clamp(0.2 + outputs[1], 0.0, 0.9)
        novelty_weight = clamp(0.2 + outputs[2], 0.0, 1.0)
        branch_insert_rate = clamp(0.1 + outputs[3], 0.0, 0.6)
        op_scale = clamp(1.0 + outputs[4], 0.2, 3.0)
        op_weights = {
            "modify_return": clamp(OP_WEIGHT_INIT.get("modify_return", 1.0) * op_scale, 0.1, 8.0),
            "insert_assign": clamp(OP_WEIGHT_INIT.get("insert_assign", 1.0) * (op_scale + 0.2), 0.1, 8.0),
            "list_manip": clamp(OP_WEIGHT_INIT.get("list_manip", 1.0) * (op_scale - 0.1), 0.1, 8.0),
        }
        return {
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate,
            "novelty_weight": novelty_weight,
            "branch_insert_rate": branch_insert_rate,
            "op_weights": op_weights,
        }

    def mutate(self, rng: random.Random, scale: float = 0.1) -> "MetaPolicy":
        weights = [row[:] for row in self.weights]
        bias = self.bias[:]
        for i in range(len(weights)):
            if rng.random() < 0.7:
                j = rng.randrange(len(weights[i]))
                weights[i][j] += rng.uniform(-scale, scale)
        for i in range(len(bias)):
            if rng.random() < 0.5:
                bias[i] += rng.uniform(-scale, scale)
        pid = sha256(json.dumps(weights) + json.dumps(bias))[:10]
        return MetaPolicy(weights=weights, bias=bias, pid=pid)


# ---------------------------
# Duo-loop (Creator/Critic)
# ---------------------------

@dataclass
class AgentPolicy:
    generator_mode: str
    search_bias: Dict[str, float]
    gate_target: float
    slice_seconds: float


CREATOR_POLICY = AgentPolicy(
    generator_mode="synthesize",
    search_bias={
        "novelty": 1.2,
        "simplicity": 0.4,
        "robustness": 0.3,
        "generalization": 0.2,
        "perf": 0.2,
    },
    gate_target=0.35,
    slice_seconds=6.0,
)

CRITIC_POLICY = AgentPolicy(
    generator_mode="mutate",
    search_bias={
        "novelty": 0.2,
        "simplicity": 0.9,
        "robustness": 1.1,
        "generalization": 1.0,
        "perf": 0.6,
    },
    gate_target=0.7,
    slice_seconds=6.0,
)

# ---------------------------
# Universe / Multiverse
# ---------------------------

@dataclass
class Universe:
    uid: int
    seed: int
    meta: MetaState
    pool: List[Genome]
    library: FunctionLibrary
    discriminator: ProblemGeneratorV2 = field(default_factory=ProblemGeneratorV2)
    eval_mode: str = "solver"
    best: Optional[Genome] = None
    best_score: float = float("inf")
    best_train: float = float("inf")
    best_hold: float = float("inf")
    best_stress: float = float("inf")
    best_test: float = float("inf")
    history: List[Dict] = field(default_factory=list)

    def step(
        self,
        gen: int,
        task: TaskSpec,
        pop_size: int,
        batch: Batch,
        policy_controls: Optional[Union[Dict[str, float], ControlPacket]] = None,
    ) -> Dict:
        rng = random.Random(self.seed + gen * 1009)
        if batch is None:
            self.pool = [seed_genome(rng) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "no_batch"}

        helper_env = self.library.get_helpers()
        if policy_controls:
            self.meta.mutation_rate = clamp(policy_controls.get("mutation_rate", self.meta.mutation_rate), 0.05, 0.98)
            self.meta.crossover_rate = clamp(policy_controls.get("crossover_rate", self.meta.crossover_rate), 0.0, 0.95)
            novelty_weight = clamp(policy_controls.get("novelty_weight", 0.0), 0.0, 1.0)
            branch_rate = clamp(policy_controls.get("branch_insert_rate", 0.0), 0.0, 0.6)
            if isinstance(policy_controls.get("op_weights"), dict):
                for k, v in policy_controls["op_weights"].items():
                    if k in self.meta.op_weights:
                        self.meta.op_weights[k] = clamp(float(v), 0.1, 8.0)
        else:
            novelty_weight = 0.0
            branch_rate = 0.0

        scored: List[Tuple[Genome, EvalResult]] = []
        all_results: List[Tuple[Genome, EvalResult]] = []
        for g in self.pool:
            # Hard gate: enforce input dependence before any scoring/selection.
            gate_ok, gate_reason = _hard_gate_ok(
                g.code,
                batch,
                self.eval_mode if self.eval_mode != "program" else "solver",
                task.name,
                extra_env=helper_env,
            )
            if not gate_ok:
                res = EvalResult(
                    False,
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    node_count(g.code),
                    float("inf"),
                    f"hard_gate:{gate_reason}",
                )
                all_results.append((g, res))
                continue
            if self.eval_mode == "algo":
                res = evaluate_algo(g, batch, task.name, self.meta.complexity_lambda)
            else:
                validator = validate_program if self.eval_mode == "program" else validate_code
                res = evaluate(g, batch, task.name, self.meta.complexity_lambda, extra_env=helper_env, validator=validator)
            all_results.append((g, res))
            if res.ok:
                scored.append((g, res))

        MetaCognitiveEngine.analyze_execution(all_results, self.meta)

        if not scored:
            hint = TaskDetective.detect_pattern(batch)
            self.pool = [seed_genome(rng, hint) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "reseed"}

        scored.sort(key=lambda t: t[1].score)
        timeout_rate = 1.0 - (len(scored) / max(1, len(all_results)))
        avg_nodes = sum(r.nodes for _, r in scored) / max(1, len(scored))

        # MAP-Elites add
        best_g0, best_res0 = scored[0]
        MAP_ELITES.add(best_g0, best_res0.score)

        for g, _ in scored[:3]:
            expr = extract_return_expr(g.statements)
            if expr:
                adopted = self.library.maybe_adopt(rng, expr, threshold=0.3)
                if adopted:
                    break

        # selection via strategy
        sel_ctx = {
            "pool": [g for g, _ in scored],
            "scores": [res.score for _, res in scored],
            "pop_size": pop_size,
            "map_elites": MAP_ELITES,
            "rng": rng,
        }
        sel_res = safe_exec_engine(self.meta.strategy.selection_code, sel_ctx)
        if sel_res and isinstance(sel_res, (tuple, list)) and len(sel_res) == 2:
            elites, parenting_pool = sel_res
        else:
            elites = [g for g, _ in scored[: max(4, pop_size // 10)]]
            parenting_pool = [rng.choice(elites) for _ in range(pop_size - len(elites))]

        candidates: List[Genome] = []
        needed = pop_size - len(elites)
        attempts_needed = max(needed * 2, needed + 8)
        mate_pool = list(elites) + list(parenting_pool)

        while len(candidates) < attempts_needed:
            parent = rng.choice(parenting_pool) if parenting_pool else rng.choice(elites)
            new_stmts = None
            op_tag = "copy"

            # crossover
            if rng.random() < self.meta.crossover_rate and len(mate_pool) > 1:
                p2 = rng.choice(mate_pool)
                cross_ctx = {"p1": parent.statements, "p2": p2.statements, "rng": rng}
                new_stmts = safe_exec_engine(self.meta.strategy.crossover_code, cross_ctx)
                if new_stmts and isinstance(new_stmts, list):
                    op_tag = "crossover"
                else:
                    new_stmts = None

            if not new_stmts:
                new_stmts = parent.statements[:]

            # mutation
            if op_tag in ("copy", "crossover") and rng.random() < self.meta.mutation_rate:
                use_synth = rng.random() < 0.3 and bool(OPERATORS_LIB)
                if use_synth:
                    synth_name = rng.choice(list(OPERATORS_LIB.keys()))
                    steps = OPERATORS_LIB[synth_name].get("steps", [])
                    new_stmts = apply_synthesized_op(rng, new_stmts, steps)
                    op_tag = f"synth:{synth_name}"
                else:
                    op = self.meta.sample_op(rng)
                    if op in OPERATORS:
                        new_stmts = OPERATORS[op](rng, new_stmts)
                    op_tag = f"mut:{op}"

            if rng.random() < branch_rate:
                extra = rng.choice(seed_genome(rng).statements)
                new_stmts = list(new_stmts) + [extra]
                op_tag = f"{op_tag}|branch"

            new_stmts = inject_helpers_into_statements(rng, list(new_stmts), self.library)
            candidates.append(Genome(statements=new_stmts, parents=[parent.gid], op_tag=op_tag))

        # surrogate ranking
        with_pred = [(c, SURROGATE.predict(c.code) + novelty_weight * rng.random()) for c in candidates]
        with_pred.sort(key=lambda x: x[1])
        selected_children = [c for c, _ in with_pred[:needed]]

        self.pool = list(elites) + selected_children

        # occasionally evolve operator library
        if rng.random() < 0.02:
            maybe_evolve_operators_lib(rng)

        # grammar induction
        if gen % 5 == 0:
            induce_grammar(list(elites))

        # acceptance update
        best_g, best_res = scored[0]
        old_score = self.best_score
        accept_margin = 1e-9
        if isinstance(policy_controls, ControlPacket):
            accept_margin = max(accept_margin, policy_controls.acceptance_margin)
        accepted = best_res.score < self.best_score - accept_margin
        if accepted:
            self.best = best_g
            self.best_score = best_res.score
            self.best_train = best_res.train
            self.best_hold = best_res.hold
            self.best_stress = best_res.stress
            self.best_test = best_res.test

        op_used = best_g.op_tag.split(":")[1].split("|")[0] if ":" in best_g.op_tag else "unknown"
        self.meta.update(op_used, self.best_score - old_score, accepted)
        if isinstance(policy_controls, ControlPacket) and self.meta.stuck_counter > policy_controls.patience:
            self.meta.epsilon_explore = clamp(self.meta.epsilon_explore + 0.05, 0.05, 0.5)

        log = {
            "gen": gen,
            "accepted": accepted,
            "score": self.best_score,
            "train": self.best_train,
            "hold": self.best_hold,
            "stress": self.best_stress,
            "test": self.best_test,
            "code": self.best.code if self.best else "none",
            "novelty_weight": novelty_weight,
            "timeout_rate": timeout_rate,
            "avg_nodes": avg_nodes,
        }
        self.history.append(log)
        if gen % 5 == 0:
            SURROGATE.train(self.history)
        return log

    def snapshot(self) -> Dict:
        return {
            "uid": self.uid,
            "seed": self.seed,
            "meta": asdict(self.meta),
            "best": asdict(self.best) if self.best else None,
            "best_score": self.best_score,
            "best_train": self.best_train,
            "best_hold": self.best_hold,
            "best_stress": self.best_stress,
            "best_test": self.best_test,
            "pool": [asdict(g) for g in self.pool[:20]],
            "library": self.library.snapshot(),
            "history": self.history[-50:],
            "eval_mode": self.eval_mode,
        }

    @staticmethod
    def from_snapshot(s: Dict) -> "Universe":
        meta_data = s.get("meta", {})
        if "strategy" in meta_data and isinstance(meta_data["strategy"], dict):
            meta_data["strategy"] = EngineStrategy(**meta_data["strategy"])
        if "update_rule" in meta_data and isinstance(meta_data["update_rule"], dict):
            meta_data["update_rule"] = UpdateRuleGenome.from_dict(meta_data["update_rule"])
        meta = MetaState(**{k: v for k, v in meta_data.items() if k != "op_weights"})
        meta.op_weights = meta_data.get("op_weights", dict(OP_WEIGHT_INIT))
        pool = [Genome(**g) for g in s.get("pool", [])]
        lib = FunctionLibrary.from_snapshot(s.get("library", {}))
        u = Universe(uid=s.get("uid", 0), seed=s.get("seed", 0), meta=meta, pool=pool, library=lib)
        if s.get("best"):
            u.best = Genome(**s["best"])
        u.best_score = s.get("best_score", float("inf"))
        u.best_train = s.get("best_train", float("inf"))
        u.best_hold = s.get("best_hold", float("inf"))
        u.best_stress = s.get("best_stress", float("inf"))
        u.best_test = s.get("best_test", float("inf"))
        u.history = s.get("history", [])
        u.eval_mode = s.get("eval_mode", "solver")
        return u


@dataclass
class UniverseLearner:
    """PHASE C: learner multiverse wrapper."""
    uid: int
    seed: int
    meta: MetaState
    pool: List[LearnerGenome]
    library: FunctionLibrary
    discriminator: ProblemGeneratorV2 = field(default_factory=ProblemGeneratorV2)
    best: Optional[LearnerGenome] = None
    best_score: float = float("inf")
    best_hold: float = float("inf")
    best_stress: float = float("inf")
    best_test: float = float("inf")
    history: List[Dict] = field(default_factory=list)

    def step(self, gen: int, task: TaskSpec, pop_size: int, batch: Batch) -> Dict:
        rng = random.Random(self.seed + gen * 1009)
        if batch is None:
            hint = TaskDetective.detect_pattern(batch)
            self.pool = [seed_learner_genome(rng, hint) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "no_batch"}

        scored: List[Tuple[LearnerGenome, EvalResult]] = []
        all_results: List[Tuple[LearnerGenome, EvalResult]] = []
        for g in self.pool:
            # Hard gate: enforce input dependence before any scoring/selection.
            gate_ok, gate_reason = _hard_gate_ok(g.code, batch, "learner", task.name)
            if not gate_ok:
                res = EvalResult(
                    False,
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    node_count(g.code),
                    float("inf"),
                    f"hard_gate:{gate_reason}",
                )
                all_results.append((g, res))
                continue
            res = evaluate_learner(g, batch, task.name, self.meta.adapt_steps, self.meta.complexity_lambda)
            all_results.append((g, res))
            if res.ok:
                scored.append((g, res))

        MetaCognitiveEngine.analyze_execution(all_results, self.meta)

        if not scored:
            hint = TaskDetective.detect_pattern(batch)
            self.pool = [seed_learner_genome(rng, hint) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "reseed"}

        scored.sort(key=lambda t: t[1].score)
        best_g0, best_res0 = scored[0]
        MAP_ELITES_LEARNER.add(best_g0, best_res0.score)

        sel_ctx = {
            "pool": [g for g, _ in scored],
            "scores": [res.score for _, res in scored],
            "pop_size": pop_size,
            "map_elites": MAP_ELITES_LEARNER,
            "rng": rng,
        }
        sel_res = safe_exec_engine(self.meta.strategy.selection_code, sel_ctx)
        if sel_res and isinstance(sel_res, (tuple, list)) and len(sel_res) == 2:
            elites, parenting_pool = sel_res
        else:
            elites = [g for g, _ in scored[: max(4, pop_size // 10)]]
            parenting_pool = [rng.choice(elites) for _ in range(pop_size - len(elites))]

        candidates: List[LearnerGenome] = []
        needed = pop_size - len(elites)
        attempts_needed = max(needed * 2, needed + 8)
        mate_pool = list(elites) + list(parenting_pool)

        while len(candidates) < attempts_needed:
            parent = rng.choice(parenting_pool) if parenting_pool else rng.choice(elites)
            child = parent
            op_tag = "copy"

            if rng.random() < self.meta.crossover_rate and len(mate_pool) > 1:
                p2 = rng.choice(mate_pool)
                new_encode = safe_exec_engine(self.meta.strategy.crossover_code, {"p1": parent.encode_stmts, "p2": p2.encode_stmts, "rng": rng})
                new_predict = safe_exec_engine(self.meta.strategy.crossover_code, {"p1": parent.predict_stmts, "p2": p2.predict_stmts, "rng": rng})
                new_update = safe_exec_engine(self.meta.strategy.crossover_code, {"p1": parent.update_stmts, "p2": p2.update_stmts, "rng": rng})
                new_objective = safe_exec_engine(self.meta.strategy.crossover_code, {"p1": parent.objective_stmts, "p2": p2.objective_stmts, "rng": rng})
                if all(isinstance(v, list) for v in (new_encode, new_predict, new_update, new_objective)):
                    child = LearnerGenome(new_encode, new_predict, new_update, new_objective, parents=[parent.gid], op_tag="crossover")
                    op_tag = "crossover"

            if op_tag in ("copy", "crossover") and rng.random() < self.meta.mutation_rate:
                child = mutate_learner(rng, child, self.meta)
                op_tag = child.op_tag

            candidates.append(child)

        with_pred = [(c, SURROGATE.predict(c.code)) for c in candidates]
        with_pred.sort(key=lambda x: x[1])
        selected_children = [c for c, _ in with_pred[:needed]]

        self.pool = list(elites) + selected_children

        if rng.random() < 0.02:
            maybe_evolve_operators_lib(rng)

        if gen % 5 == 0:
            induce_grammar([Genome(statements=["return x"])])

        best_g, best_res = scored[0]
        old_score = self.best_score
        accepted = best_res.score < self.best_score - 1e-9
        if accepted:
            self.best = best_g
            self.best_score = best_res.score
            self.best_hold = best_res.hold
            self.best_stress = best_res.stress
            self.best_test = best_res.test

        op_used = best_g.op_tag.split(":")[1].split("|")[0] if ":" in best_g.op_tag else "unknown"
        self.meta.update(op_used, self.best_score - old_score, accepted)

        log = {
            "gen": gen,
            "accepted": accepted,
            "score": self.best_score,
            "hold": self.best_hold,
            "stress": self.best_stress,
            "test": self.best_test,
            "code": self.best.code if self.best else "none",
        }
        self.history.append(log)
        if gen % 5 == 0:
            SURROGATE.train(self.history)
        return log

    def snapshot(self) -> Dict:
        return {
            "uid": self.uid,
            "seed": self.seed,
            "meta": asdict(self.meta),
            "best": asdict(self.best) if self.best else None,
            "best_score": self.best_score,
            "best_hold": self.best_hold,
            "best_stress": self.best_stress,
            "best_test": self.best_test,
            "pool": [asdict(g) for g in self.pool[:20]],
            "library": self.library.snapshot(),
            "history": self.history[-50:],
        }

    @staticmethod
    def from_snapshot(s: Dict) -> "UniverseLearner":
        meta_data = s.get("meta", {})
        if "strategy" in meta_data and isinstance(meta_data["strategy"], dict):
            meta_data["strategy"] = EngineStrategy(**meta_data["strategy"])
        if "update_rule" in meta_data and isinstance(meta_data["update_rule"], dict):
            meta_data["update_rule"] = UpdateRuleGenome.from_dict(meta_data["update_rule"])
        meta = MetaState(**{k: v for k, v in meta_data.items() if k != "op_weights"})
        meta.op_weights = meta_data.get("op_weights", dict(OP_WEIGHT_INIT))
        pool = [LearnerGenome(**g) for g in s.get("pool", [])]
        lib = FunctionLibrary.from_snapshot(s.get("library", {}))
        u = UniverseLearner(uid=s.get("uid", 0), seed=s.get("seed", 0), meta=meta, pool=pool, library=lib)
        if s.get("best"):
            u.best = LearnerGenome(**s["best"])
        u.best_score = s.get("best_score", float("inf"))
        u.best_hold = s.get("best_hold", float("inf"))
        u.best_stress = s.get("best_stress", float("inf"))
        u.best_test = s.get("best_test", float("inf"))
        u.history = s.get("history", [])
        return u


# ---------------------------
# State persistence
# ---------------------------

@dataclass
class GlobalState:
    version: str
    created_ms: int
    updated_ms: int
    base_seed: int
    task: Dict
    universes: List[Dict]
    selected_uid: int = 0
    generations_done: int = 0
    mode: str = "solver"
    rule_dsl: Optional[Dict[str, Any]] = None

STATE_DIR = Path(".rsi_state")
UPDATE_RULE_FILE = STATE_DIR / "update_rule.json"

def save_update_rule(path: Path, rule: UpdateRuleGenome):
    write_json(path, asdict(rule))

def load_update_rule(path: Path) -> UpdateRuleGenome:
    if path.exists():
        try:
            return UpdateRuleGenome.from_dict(read_json(path))
        except Exception:
            return UpdateRuleGenome.default()
    return UpdateRuleGenome.default()

def save_operators_lib(path: Path):
    path.write_text(json.dumps(OPERATORS_LIB, indent=2), encoding="utf-8")

def load_operators_lib(path: Path):
    global OPERATORS_LIB
    if path.exists():
        try:
            OPERATORS_LIB.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass

def save_state(gs: GlobalState):
    gs.updated_ms = now_ms()
    write_json(STATE_DIR / "state.json", asdict(gs))
    save_operators_lib(STATE_DIR / "operators_lib.json")
    if gs.universes:
        meta_snapshot = gs.universes[0].get("meta", {})
        if isinstance(meta_snapshot, dict) and "update_rule" in meta_snapshot:
            try:
                save_update_rule(STATE_DIR / "update_rule.json", UpdateRuleGenome.from_dict(meta_snapshot["update_rule"]))
            except Exception:
                pass
    if gs.mode == "learner":
        save_map_elites(STATE_DIR / map_elites_filename("learner"), MAP_ELITES_LEARNER)
    else:
        save_map_elites(STATE_DIR / map_elites_filename("solver"), MAP_ELITES)

def load_state() -> Optional[GlobalState]:
    p = STATE_DIR / "state.json"
    if not p.exists():
        return None
    try:
        data = read_json(p)
        mode = data.get("mode", "solver")
        load_operators_lib(STATE_DIR / "operators_lib.json")
        if mode == "learner":
            load_map_elites(STATE_DIR / map_elites_filename("learner"), MAP_ELITES_LEARNER)
        else:
            load_map_elites(STATE_DIR / map_elites_filename("solver"), MAP_ELITES)
        data["mode"] = mode
        return GlobalState(**data)
    except Exception:
        return None


def run_multiverse(
    seed: int,
    task: TaskSpec,
    gens: int,
    pop: int,
    n_univ: int,
    resume: bool = False,
    save_every: int = 5,
    mode: str = "solver",
    freeze_eval: bool = True,
) -> GlobalState:
    safe_mkdir(STATE_DIR)
    logger = RunLogger(STATE_DIR / "run_log.jsonl", append=resume)
    task.ensure_descriptor()
    update_rule = load_update_rule(STATE_DIR / "update_rule.json")

    if resume and (gs0 := load_state()):
        mode = gs0.mode
        if mode == "learner":
            us = [UniverseLearner.from_snapshot(s) for s in gs0.universes]
        else:
            us = [Universe.from_snapshot(s) for s in gs0.universes]
        for u in us:
            u.meta.update_rule = update_rule
        start = gs0.generations_done
    else:
        b0 = get_task_batch(task, seed, freeze_eval=freeze_eval)
        hint = TaskDetective.detect_pattern(b0)
        if hint:
            print(f"[Detective] Detected pattern: {hint}. Injecting smart seeds.")
        if mode == "learner":
            us = [
                UniverseLearner(
                    uid=i,
                    seed=seed + i * 9973,
                    meta=MetaState(update_rule=update_rule),
                    pool=[seed_learner_genome(random.Random(seed + i), hint) for _ in range(pop)],
                    library=FunctionLibrary(),
                )
                for i in range(n_univ)
            ]
        else:
            eval_mode = "program" if mode == "program" else ("algo" if mode == "algo" else "solver")
            us = [
                Universe(
                    uid=i,
                    seed=seed + i * 9973,
                    meta=MetaState(update_rule=update_rule),
                    pool=[seed_genome(random.Random(seed + i), hint) for _ in range(pop)],
                    library=FunctionLibrary(),
                    eval_mode=eval_mode,
                )
                for i in range(n_univ)
            ]
        start = 0

    for gen in range(start, start + gens):
        start_ms = now_ms()
        batch = get_task_batch(task, seed, freeze_eval=freeze_eval)
        for u in us:
            if mode == "learner":
                u.step(gen, task, pop, batch)
            else:
                u.step(gen, task, pop, batch)

        us.sort(key=lambda u: u.best_score)
        best = us[0]
        runtime_ms = now_ms() - start_ms
        best_code = best.best.code if best.best else "none"
        code_hash = sha256(best_code)
        novelty = 1.0 if code_hash not in logger.seen_hashes else 0.0
        logger.seen_hashes.add(code_hash)
        accepted = bool(best.history[-1]["accepted"]) if best.history else False
        last_log = best.history[-1] if best.history else {}
        control_packet = {
            "mutation_rate": best.meta.mutation_rate,
            "crossover_rate": best.meta.crossover_rate,
            "epsilon_explore": best.meta.epsilon_explore,
            "acceptance_margin": 1e-9,
            "patience": getattr(best.meta, "patience", 5),
        }
        counterexample_count = len(ALGO_COUNTEREXAMPLES.get(task.name, [])) if mode == "algo" else 0
        logger.log(
            gen=gen,
            task_id=task.name,
            mode=mode,
            score_hold=best.best_hold,
            score_stress=best.best_stress,
            score_test=getattr(best, "best_test", float("inf")),
            runtime_ms=runtime_ms,
            nodes=node_count(best_code),
            code_hash=code_hash,
            accepted=accepted,
            novelty=novelty,
            meta_policy_params={},
            solver_hash=code_hash,
            p1_hash="default",
            err_hold=best.best_hold,
            err_stress=best.best_stress,
            err_test=getattr(best, "best_test", float("inf")),
            steps=last_log.get("avg_nodes"),
            timeout_rate=last_log.get("timeout_rate"),
            counterexample_count=counterexample_count,
            library_size=len(OPERATORS_LIB),
            control_packet=control_packet,
            task_descriptor=task.descriptor.snapshot() if task.descriptor else None,
        )
        print(
            f"[Gen {gen + 1:4d}] Score: {best.best_score:.4f} | Hold: {best.best_hold:.4f} | Stress: {best.best_stress:.4f} | Test: {best.best_test:.4f} | "
            f"{(best.best.code if best.best else 'none')}"
        )

        if save_every > 0 and (gen + 1) % save_every == 0:
            gs = GlobalState(
                "RSI_EXTENDED_v2",
                now_ms(),
                now_ms(),
                seed,
                asdict(task),
                [u.snapshot() for u in us],
                us[0].uid,
                gen + 1,
                mode=mode,
            )
            save_state(gs)

    gs = GlobalState(
        "RSI_EXTENDED_v2",
        now_ms(),
        now_ms(),
        seed,
        asdict(task),
        [u.snapshot() for u in us],
        us[0].uid,
        start + gens,
        mode=mode,
    )
    save_state(gs)
    return gs


