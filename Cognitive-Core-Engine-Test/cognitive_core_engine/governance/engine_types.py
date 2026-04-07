"""Engine types, task descriptors, genome structures, and evaluation functions."""
from __future__ import annotations

import ast
import collections
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

from cognitive_core_engine.governance.sandbox import (
    safe_exec, safe_exec_algo, safe_eval, validate_code, validate_program,
    validate_algo_program, safe_exec_engine, safe_load_module,
    SAFE_BUILTINS, SAFE_VARS,
)
from cognitive_core_engine.governance.utils import clamp


class EngineStrategy:
    selection_code: str
    crossover_code: str
    mutation_policy_code: str
    gid: str = "default"


DEFAULT_SELECTION_CODE = """
def run():
    # Context injected: pool, scores, pop_size, rng, map_elites
    # Returns: (elites, breeding_parents)
    scored = sorted(zip(pool, scores), key=lambda x: x[1])
    elite_k = max(4, pop_size // 10)
    elites = [g for g, s in scored[:elite_k]]

    parents = []
    n_needed = pop_size - len(elites)
    for _ in range(n_needed):
        # 10% chance to pick from MAP-Elites
        if rng.random() < 0.1 and map_elites and map_elites.grid:
            p = map_elites.sample(rng) or rng.choice(elites)
        else:
            p = rng.choice(elites)
        parents.append(p)
    return elites, parents
"""

DEFAULT_CROSSOVER_CODE = """
def run():
    # Context: p1 (stmts), p2 (stmts), rng
    if len(p1) < 2 or len(p2) < 2:
        return p1
    idx_a = rng.randint(0, len(p1))
    idx_b = rng.randint(0, len(p2))
    return p1[:idx_a] + p2[idx_b:]
"""

DEFAULT_MUTATION_CODE = """
def run():
    return "default"
"""


# ---------------------------
# Tasks / Datasets
# ---------------------------

@dataclass
class TaskDescriptor:
    name: str
    family: str
    input_kind: str
    output_kind: str
    n_train: int
    n_hold: int
    n_test: int
    noise: float
    stress_mult: float
    has_switch: bool
    nonlinear: bool

    def vector(self) -> List[float]:
        family_map = {
            "poly": 0.1,
            "piecewise": 0.3,
            "rational": 0.5,
            "switching": 0.7,
            "classification": 0.9,
            "list": 0.2,
            "arc": 0.4,
            "other": 0.6,
        }
        return [
            family_map.get(self.family, 0.0),
            1.0 if self.input_kind == "list" else 0.0,
            1.0 if self.input_kind == "grid" else 0.0,
            1.0 if self.output_kind == "class" else 0.0,
            float(self.n_train) / 100.0,
            float(self.n_hold) / 100.0,
            float(self.n_test) / 100.0,
            clamp(self.noise, 0.0, 1.0),
            clamp(self.stress_mult / 5.0, 0.0, 2.0),
            1.0 if self.has_switch else 0.0,
            1.0 if self.nonlinear else 0.0,
        ]

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskSpec:
    name: str = "poly2"
    x_min: float = -3.0
    x_max: float = 3.0
    n_train: int = 96
    n_hold: int = 96
    n_test: int = 96
    noise: float = 0.01
    stress_mult: float = 3.0
    target_code: Optional[str] = None
    descriptor: Optional[TaskDescriptor] = None

    def ensure_descriptor(self) -> TaskDescriptor:
        if self.descriptor:
            return self.descriptor
        family = "other"
        if self.name in ("poly2", "poly3"):
            family = "poly"
        elif self.name == "piecewise":
            family = "piecewise"
        elif self.name == "rational":
            family = "rational"
        elif self.name == "switching":
            family = "switching"
        elif self.name == "classification":
            family = "classification"
        elif self.name in ("sort", "reverse", "filter", "max", "even_reverse_sort"):
            family = "list"
        elif self.name == "self_audit":
            family = "self_audit"
        elif self.name in ALGO_TASK_NAMES:
            family = "algo"
        elif self.name.startswith("arc_"):
            family = "arc"
        self.descriptor = TaskDescriptor(
            name=self.name,
            family=family,
            input_kind="vector"
            if family == "self_audit"
            else ("list" if family in ("list", "algo") else ("grid" if family == "arc" else "scalar")),
            output_kind="class" if family == "classification" else "scalar",
            n_train=self.n_train,
            n_hold=self.n_hold,
            n_test=self.n_test,
            noise=self.noise,
            stress_mult=self.stress_mult,
            has_switch=self.name == "switching",
            nonlinear=family in ("poly", "piecewise", "rational", "switching"),
        )
        return self.descriptor


# ---------------------------
# Algorithmic task suite (algo mode)
# ---------------------------

ALGO_TASK_NAMES = {
    "sort_int_list",
    "topk",
    "two_sum",
    "balanced_parens",
    "gcd_list",
    "rpn_eval",
    "bfs_shortest_path",
    "coin_change_min",
    "substring_find",
    "unique_count",
    "lis_length",
    "min_path_sum",
    "edit_distance",
}

ALGO_COUNTEREXAMPLES: Dict[str, List[Tuple[Any, Any]]] = {name: [] for name in ALGO_TASK_NAMES}

def _gen_int_list(rng: random.Random, min_len: int, max_len: int, lo: int = -9, hi: int = 9) -> List[int]:
    ln = rng.randint(min_len, max_len)
    return [rng.randint(lo, hi) for _ in range(ln)]

def _gen_parens(rng: random.Random, min_len: int, max_len: int) -> List[int]:
    ln = rng.randint(min_len, max_len)
    return [0 if rng.random() < 0.5 else 1 for _ in range(ln)]

def _gen_graph(rng: random.Random, n_min: int, n_max: int) -> List[List[int]]:
    n = rng.randint(n_min, n_max)
    g = []
    for i in range(n):
        neigh = []
        for j in range(n):
            if i != j and rng.random() < 0.25:
                neigh.append(j)
        g.append(neigh)
    return g

def _algo_descriptor(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "family": "algo",
        "input_kind": "list",
        "output_kind": "scalar",
        "n_train": 0,
        "n_hold": 0,
        "n_test": 0,
        "noise": 0.0,
        "stress_mult": 2.0,
        "has_switch": False,
        "nonlinear": True,
    }

def _algo_task_data(name: str, rng: random.Random, n: int, stress: bool = False) -> Tuple[List[Any], List[Any]]:
    xs: List[Any] = []
    ys: List[Any] = []
    for _ in range(n):
        if name == "sort_int_list":
            x = _gen_int_list(rng, 2, 8 if not stress else 12)
            y = sorted(x)
        elif name == "topk":
            arr = _gen_int_list(rng, 2, 10 if not stress else 14)
            k = rng.randint(1, max(1, len(arr) // 2))
            x = [arr, k]
            y = sorted(arr, reverse=True)[:k]
        elif name == "two_sum":
            arr = _gen_int_list(rng, 2, 10 if not stress else 14)
            i, j = rng.sample(range(len(arr)), 2)
            target = arr[i] + arr[j]
            x = [arr, target]
            y = [i, j]
        elif name == "balanced_parens":
            seq = _gen_parens(rng, 2, 12 if not stress else 18)
            bal = 0
            ok = 1
            for t in seq:
                bal += 1 if t == 0 else -1
                if bal < 0:
                    ok = 0
                    break
            if bal != 0:
                ok = 0
            x = seq
            y = ok
        elif name == "gcd_list":
            arr = [abs(v) + 1 for v in _gen_int_list(rng, 2, 8 if not stress else 12, 1, 9)]
            g = arr[0]
            for v in arr[1:]:
                g = math.gcd(g, v)
            x = arr
            y = g
        elif name == "rpn_eval":
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            op = rng.choice([-1, -2, -3, -4])
            if op == -1:
                y = a + b
            elif op == -2:
                y = a - b
            elif op == -3:
                y = a * b
            else:
                y = a // b if b else 0
            x = [a, b, op]
        elif name == "bfs_shortest_path":
            g = _gen_graph(rng, 4, 7 if not stress else 9)
            s, t = rng.sample(range(len(g)), 2)
            dist = [-1] * len(g)
            dist[s] = 0
            q = [s]
            while q:
                cur = q.pop(0)
                for nxt in g[cur]:
                    if dist[nxt] == -1:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            x = [g, s, t]
            y = dist[t]
        elif name == "coin_change_min":
            coins = [c for c in _gen_int_list(rng, 2, 5 if not stress else 7, 1, 8) if c > 0]
            amount = rng.randint(1, 12 if not stress else 18)
            dp = [float("inf")] * (amount + 1)
            dp[0] = 0
            for c in coins:
                for a in range(c, amount + 1):
                    dp[a] = min(dp[a], dp[a - c] + 1)
            y = -1 if dp[amount] == float("inf") else int(dp[amount])
            x = [coins, amount]
        elif name == "substring_find":
            hay = _gen_int_list(rng, 4, 10 if not stress else 14, 1, 4)
            needle = hay[1:3] if len(hay) > 3 and rng.random() < 0.7 else _gen_int_list(rng, 2, 3, 1, 4)
            idx = -1
            for i in range(len(hay) - len(needle) + 1):
                if hay[i:i + len(needle)] == needle:
                    idx = i
                    break
            x = [hay, needle]
            y = idx
        elif name == "unique_count":
            arr = _gen_int_list(rng, 3, 10 if not stress else 14, 1, 6)
            x = arr
            y = len(set(arr))
        elif name == "lis_length":
            arr = _gen_int_list(rng, 3, 10 if not stress else 14, -5, 9)
            dp = [1 for _ in arr]
            for i in range(len(arr)):
                for j in range(i):
                    if arr[j] < arr[i]:
                        dp[i] = max(dp[i], dp[j] + 1)
            x = arr
            y = max(dp) if dp else 0
        elif name == "min_path_sum":
            rows = rng.randint(2, 5 if not stress else 7)
            cols = rng.randint(2, 5 if not stress else 7)
            grid = [[rng.randint(0, 9) for _ in range(cols)] for _ in range(rows)]
            dp = [[0 for _ in range(cols)] for _ in range(rows)]
            dp[0][0] = grid[0][0]
            for r in range(1, rows):
                dp[r][0] = dp[r - 1][0] + grid[r][0]
            for c in range(1, cols):
                dp[0][c] = dp[0][c - 1] + grid[0][c]
            for r in range(1, rows):
                for c in range(1, cols):
                    dp[r][c] = min(dp[r - 1][c], dp[r][c - 1]) + grid[r][c]
            x = grid
            y = dp[-1][-1]
        elif name == "edit_distance":
            a = _gen_int_list(rng, 2, 6 if not stress else 8, 0, 4)
            b = _gen_int_list(rng, 2, 6 if not stress else 8, 0, 4)
            dp = [[0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
            for i in range(len(a) + 1):
                dp[i][0] = i
            for j in range(len(b) + 1):
                dp[0][j] = j
            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + cost,
                    )
            x = [a, b]
            y = dp[-1][-1]
        else:
            x = []
            y = 0
        xs.append(x)
        ys.append(y)
    return xs, ys

def algo_batch(name: str, seed: int, freeze_eval: bool = True, train_resample_every: int = 1, gen: int = 0) -> Optional[Batch]:
    if name not in ALGO_TASK_NAMES:
        return None
    rng = random.Random(seed)
    hold_rng = random.Random(seed + 11)
    stress_rng = random.Random(seed + 29)
    test_rng = random.Random(seed + 47)
    if not freeze_eval:
        hold_rng = random.Random(seed + 11 + gen)
        stress_rng = random.Random(seed + 29 + gen)
        test_rng = random.Random(seed + 47 + gen)
    train_rng = rng if train_resample_every <= 1 else random.Random(seed + gen // max(1, train_resample_every))
    x_tr, y_tr = _algo_task_data(name, train_rng, 40, stress=False)
    x_ho, y_ho = _algo_task_data(name, hold_rng, 24, stress=False)
    x_st, y_st = _algo_task_data(name, stress_rng, 24, stress=True)
    x_te, y_te = _algo_task_data(name, test_rng, 24, stress=True)
    return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)


@dataclass
class ControlPacket:
    mutation_rate: Optional[float] = None
    crossover_rate: Optional[float] = None
    novelty_weight: float = 0.0
    branch_insert_rate: float = 0.0
    op_weights: Optional[Dict[str, float]] = None
    acceptance_margin: float = 1e-9
    patience: int = 5

    def get(self, key: str, default: Any = None) -> Any:
        val = getattr(self, key, default)
        if val is None:
            return default
        return val


TARGET_FNS = {
    "sort": lambda x: sorted(x),
    "reverse": lambda x: list(reversed(x)),
    "max": lambda x: max(x) if x else 0,
    "filter": lambda x: [v for v in x if v > 0],
    "arc_ident": lambda x: x,
    "arc_rot90": lambda x: [list(r) for r in zip(*x[::-1])],
    "arc_inv": lambda x: [[1 - c if c in (0, 1) else c for c in r] for r in x],
    "poly2": lambda x: 0.7 * x * x - 0.2 * x + 0.3,
    "poly3": lambda x: 0.3 * x ** 3 - 0.5 * x + 0.1,
    "piecewise": lambda x: (-0.5 * x + 1.0) if x < 0 else (0.3 * x * x + 0.1),
    "rational": lambda x: (x * x + 1.0) / (1.0 + 0.5 * abs(x)),
    "sinmix": lambda x: math.sin(x) + 0.3 * math.cos(2 * x),
    "absline": lambda x: abs(x) + 0.2 * x,
    "classification": lambda x: 1.0 if (x + 0.25 * math.sin(3 * x)) > 0 else 0.0,
}


ARC_GYM_PATH = os.path.join(os.path.dirname(__file__), "ARC_GYM")

def load_arc_task(task_id: str) -> Dict:
    fname = task_id
    if not fname.endswith(".json"):
        fname += ".json"
    path = os.path.join(ARC_GYM_PATH, fname)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_arc_tasks() -> List[str]:
    if not os.path.exists(ARC_GYM_PATH):
        return []
    return [f[:-5] for f in os.listdir(ARC_GYM_PATH) if f.endswith(".json")]

@dataclass
class Batch:
    x_tr: List[Any]
    y_tr: List[Any]
    x_ho: List[Any]
    y_ho: List[Any]
    x_st: List[Any]
    y_st: List[Any]
    x_te: List[Any]
    y_te: List[Any]


def _best_code_snapshot() -> str:
    try:
        state = load_state()
    except Exception:
        state = None
    if not state or not state.universes:
        return "def run(x):\n    return x\n"
    target = next((u for u in state.universes if u.get("uid") == state.selected_uid), None)
    if not target:
        target = state.universes[0]
    best = target.get("best")
    if not best:
        return "def run(x):\n    return x\n"
    if state.mode == "learner":
        return LearnerGenome(**best).code
    return Genome(**best).code


def _code_features(code: str) -> List[float]:
    return [
        float(len(code)),
        float(node_count(code)),
        float(code.count("if ")),
        float(code.count("while ")),
        float(code.count("return ")),
    ]

def sample_batch(rng: random.Random, t: TaskSpec) -> Optional[Batch]:
    # function target
    if t.target_code:
        f = lambda x: safe_exec(t.target_code, x)
    elif t.name in ("sort", "reverse", "filter", "max"):
        f = TARGET_FNS.get(t.name) or (lambda x: sorted(x))
    else:
        f = TARGET_FNS.get(t.name, lambda x: x)

    if t.name == "self_audit":
        base_code = _best_code_snapshot()
        base_features = _code_features(base_code)

        def synth_features(k: int, jitter: float) -> List[List[float]]:
            samples = []
            for _ in range(k):
                sample = [max(0.0, f + rng.gauss(0, jitter)) for f in base_features]
                samples.append(sample)
            return samples

        def target_score(vec: List[float]) -> float:
            length, nodes, ifs, whiles, returns = vec
            raw = 0.004 * length + 0.02 * nodes + 0.1 * ifs + 0.15 * whiles + 0.02 * returns
            return math.tanh(raw / 10.0)

        x_tr = synth_features(t.n_train, 3.0)
        x_ho = synth_features(t.n_hold, 4.0)
        x_st = synth_features(t.n_hold, 6.0)
        x_te = synth_features(t.n_test, 4.0)
        y_tr = [target_score(x) for x in x_tr]
        y_ho = [target_score(x) for x in x_ho]
        y_st = [target_score(x) for x in x_st]
        y_te = [target_score(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    # ARC tasks from local json
    json_data = load_arc_task(t.name.replace("arc_", ""))
    if json_data:
        pairs = json_data.get("train", []) + json_data.get("test", [])
        x_all, y_all = [], []
        for p in pairs:
            x_all.append(p["input"])
            y_all.append(p["output"])
            if len(x_all) >= 30:
                break
        if not x_all:
            return None
        return Batch(
            x_all[:20], y_all[:20],
            x_all[:10], y_all[:10],
            x_all[:5],  y_all[:5],
            x_all[5:10], y_all[5:10],
        )

    # list tasks
    def gen_lists(k, min_len, max_len):
        data = []
        for _ in range(k):
            a = max(1, int(min_len))
            b = max(a, int(max_len))
            l = rng.randint(a, b)
            data.append([rng.randint(-100, 100) for _ in range(l)])
        return data

    if t.name == "even_reverse_sort":
        f = lambda x: sorted([n for n in x if n % 2 == 0], reverse=True)
        x_tr = gen_lists(t.n_train, t.x_min, t.x_max)
        x_ho = gen_lists(t.n_hold, t.x_min + 2, t.x_max + 2)
        x_st = gen_lists(t.n_hold, t.x_max + 5, t.x_max + 10)
        y_tr = [f(x) for x in x_tr]
        y_ho = [f(x) for x in x_ho]
        y_st = [f(x) for x in x_st]
        x_te = gen_lists(max(1, t.n_test), t.x_min + 1, t.x_max + 1)
        y_te = [f(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    if t.name in ("sort", "reverse", "filter", "max"):
        x_tr = gen_lists(t.n_train, t.x_min, t.x_max)
        x_ho = gen_lists(t.n_hold, t.x_min + 2, t.x_max + 2)
        x_st = gen_lists(t.n_hold, t.x_max + 5, t.x_max + 10)
        y_tr = [f(x) for x in x_tr]
        y_ho = [f(x) for x in x_ho]
        y_st = [f(x) for x in x_st]
        x_te = gen_lists(max(1, t.n_test), t.x_min + 1, t.x_max + 1)
        y_te = [f(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    # synthetic ARC-like generators if name starts with arc_
    if t.name.startswith("arc_"):
        def gen_grids(k, dim):
            data = []
            for _ in range(k):
                g = [[rng.randint(0, 1) for _ in range(dim)] for _ in range(dim)]
                data.append(g)
            return data
        dim = int(t.x_min) if t.x_min > 0 else 3
        x_tr = gen_grids(20, dim)
        x_ho = gen_grids(10, dim)
        x_st = gen_grids(10, dim + 1)
        y_tr = [f(x) for x in x_tr]
        y_ho = [f(x) for x in x_ho]
        y_st = [f(x) for x in x_st]
        x_te = gen_grids(10, dim)
        y_te = [f(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    if t.name == "switching":
        def target_switch(pair):
            x, s = pair
            return TARGET_FNS["poly2"](x) if s < 0.5 else TARGET_FNS["sinmix"](x)

        def gen_pairs(k, a, b):
            data = []
            for _ in range(k):
                x = a + (b - a) * rng.random()
                s = 1.0 if rng.random() > 0.5 else 0.0
                data.append([x, s])
            return data

        x_tr = gen_pairs(t.n_train, t.x_min, t.x_max)
        x_ho = gen_pairs(t.n_hold, t.x_min, t.x_max)
        x_st = gen_pairs(t.n_hold, t.x_min * t.stress_mult, t.x_max * t.stress_mult)
        x_te = gen_pairs(t.n_test, t.x_min, t.x_max)
        y_tr = [target_switch(x) for x in x_tr]
        y_ho = [target_switch(x) for x in x_ho]
        y_st = [target_switch(x) for x in x_st]
        y_te = [target_switch(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    # numeric regression tasks
    xs = lambda n, a, b: [a + (b - a) * rng.random() for _ in range(n)]
    ys = lambda xv, n: [f(x) + rng.gauss(0, n) if n > 0 else f(x) for x in xv]
    half = 0.5 * (t.x_max - t.x_min)
    mid = 0.5 * (t.x_min + t.x_max)
    x_tr = xs(t.n_train, t.x_min, t.x_max)
    x_ho = xs(t.n_hold, t.x_min, t.x_max)
    x_st = xs(t.n_hold, mid - half * t.stress_mult, mid + half * t.stress_mult)
    x_te = xs(t.n_test, t.x_min, t.x_max)
    return Batch(
        x_tr, ys(x_tr, t.noise),
        x_ho, ys(x_ho, t.noise),
        x_st, ys(x_st, t.noise * t.stress_mult),
        x_te, ys(x_te, t.noise),
    )


def task_suite(seed: int) -> List[TaskSpec]:
    base = [
        TaskSpec(name="poly2", x_min=-3.0, x_max=3.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="poly3", x_min=-4.0, x_max=4.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="piecewise", x_min=-4.0, x_max=4.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="rational", x_min=-5.0, x_max=5.0, n_train=96, n_hold=64, n_test=64, noise=0.02),
        TaskSpec(name="switching", x_min=-3.0, x_max=3.0, n_train=96, n_hold=64, n_test=64, noise=0.0),
        TaskSpec(name="classification", x_min=-4.0, x_max=4.0, n_train=96, n_hold=64, n_test=64, noise=0.0),
        TaskSpec(name="sinmix", x_min=-6.0, x_max=6.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="absline", x_min=-6.0, x_max=6.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="self_audit", x_min=0.0, x_max=1.0, n_train=64, n_hold=48, n_test=48, noise=0.0),
    ]
    rng = random.Random(seed)
    rng.shuffle(base)
    return base


def split_meta_tasks(seed: int, meta_train_ratio: float = 0.6) -> Tuple[List[TaskSpec], List[TaskSpec]]:
    suite = task_suite(seed)
    cut = max(1, int(len(suite) * meta_train_ratio))
    return suite[:cut], suite[cut:]


FROZEN_BATCH_CACHE: Dict[str, Batch] = {}


def _task_cache_key(task: TaskSpec, seed: int) -> str:
    return f"{task.name}:{seed}:{task.x_min}:{task.x_max}:{task.n_train}:{task.n_hold}:{task.n_test}:{task.noise}:{task.stress_mult}:{task.target_code}"


def get_task_batch(
    task: TaskSpec,
    seed: int,
    freeze_eval: bool = True,
    train_resample_every: int = 1,
    gen: int = 0,
) -> Optional[Batch]:
    if task.name in ALGO_TASK_NAMES:
        return algo_batch(task.name, seed, freeze_eval=freeze_eval, train_resample_every=train_resample_every, gen=gen)
    key = _task_cache_key(task, seed)
    if freeze_eval and key in FROZEN_BATCH_CACHE:
        return FROZEN_BATCH_CACHE[key]
    h = int(sha256(key)[:8], 16)
    rng = random.Random(h if freeze_eval else seed)
    batch = sample_batch(rng, task)
    if freeze_eval and batch is not None:
        FROZEN_BATCH_CACHE[key] = batch
    return batch


# ---------------------------
# Genome / Evaluation
# ---------------------------

@dataclass
class Genome:
    statements: List[str]
    gid: str = ""
    parents: List[str] = field(default_factory=list)
    op_tag: str = "init"
    birth_ms: int = 0

    @property
    def code(self) -> str:
        body = "\n    ".join(self.statements) if self.statements else "return x"
        return f"def run(x):\n    # {self.gid}\n    v0=x\n    {body}"

    def __post_init__(self):
        if not self.gid:
            self.gid = sha256("".join(self.statements) + str(time.time()))[:12]
        if not self.birth_ms:
            self.birth_ms = now_ms()


@dataclass
class LearnerGenome:
    """PHASE B: learner genome with encode/predict/update/objective blocks."""
    encode_stmts: List[str]
    predict_stmts: List[str]
    update_stmts: List[str]
    objective_stmts: List[str]
    gid: str = ""
    parents: List[str] = field(default_factory=list)
    op_tag: str = "init"
    birth_ms: int = 0

    @property
    def code(self) -> str:
        def ensure_return(stmts: List[str], fallback: str) -> List[str]:
            for s in stmts:
                if s.strip().startswith("return "):
                    return stmts
            return stmts + [fallback]

        enc = ensure_return(self.encode_stmts or [], "return x")
        pred = ensure_return(self.predict_stmts or [], "return z")
        upd = ensure_return(self.update_stmts or [], "return mem")
        obj = ensure_return(self.objective_stmts or [], "return hold + 0.5*stress + 0.01*nodes")

        enc_body = "\n    ".join(enc) if enc else "return x"
        pred_body = "\n    ".join(pred) if pred else "return z"
        upd_body = "\n    ".join(upd) if upd else "return mem"
        obj_body = "\n    ".join(obj) if obj else "return hold + 0.5*stress + 0.01*nodes"

        return (
            "def init_mem():\n"
            "    return {\"w\": 0.0, \"b\": 0.0, \"t\": 0}\n\n"
            "def encode(x, mem):\n"
            f"    # {self.gid}\n    {enc_body}\n\n"
            "def predict(z, mem):\n"
            f"    {pred_body}\n\n"
            "def update(mem, x, y_pred, y_true, lr=0.05):\n"
            f"    {upd_body}\n\n"
            "def objective(train, hold, stress, nodes):\n"
            f"    {obj_body}\n"
        )

    def __post_init__(self):
        if not self.gid:
            self.gid = sha256("".join(self.encode_stmts + self.predict_stmts + self.update_stmts + self.objective_stmts) + str(time.time()))[:12]
        if not self.birth_ms:
            self.birth_ms = now_ms()


@dataclass
class EvalResult:
    ok: bool
    train: float
    hold: float
    stress: float
    test: float
    nodes: int
    score: float
    err: Optional[str] = None


SCORE_W_HOLD = 0.452390
SCORE_W_STRESS = 0.4
SCORE_W_TRAIN = 0.0


def calc_error(p: Any, t: Any) -> float:
    if isinstance(t, (int, float)):
        if isinstance(p, (int, float)):
            return (p - t) ** 2
        return 1_000_000.0
    if isinstance(t, list):
        if not isinstance(p, list):
            return 1_000_000.0
        if len(p) != len(t):
            return 1000.0 * abs(len(p) - len(t))
        return sum(calc_error(pv, tv) for pv, tv in zip(p, t))
    return 1_000_000.0


def _list_invariance_penalty(x: Any, p: Any, task_name: str) -> float:
    if not isinstance(x, list):
        return 0.0
    if task_name in ("sort", "reverse"):
        if not isinstance(p, list):
            return 5_000.0
        if len(p) != len(x):
            return 2_000.0 + 10.0 * abs(len(p) - len(x))
        try:
            if collections.Counter(p) != collections.Counter(x):
                return 2_000.0
        except TypeError:
            pass
    if task_name == "filter":
        if not isinstance(p, list):
            return 5_000.0
        try:
            x_counts = collections.Counter(x)
            p_counts = collections.Counter(p)
            for k, v in p_counts.items():
                if x_counts.get(k, 0) < v:
                    return 2_000.0
        except TypeError:
            pass
    if task_name == "max":
        if not isinstance(p, (int, float)):
            return 5_000.0
    return 0.0


def calc_loss_sort(p: List[Any], t: List[Any]) -> float:
    if not isinstance(p, list):
        return 1_000_000.0
    if len(p) != len(t):
        return 1000.0 * abs(len(p) - len(t))
    p_sorted = sorted(p) if all(isinstance(x, (int, float)) for x in p) else p
    t_sorted = sorted(t)
    content_loss = sum((a - b) ** 2 for a, b in zip(p_sorted, t_sorted))
    if content_loss > 0.1:
        return 1000.0 + content_loss
    inversions = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                inversions += 1
    return float(inversions)


def calc_heuristic_loss(p: Any, t: Any, task_name: str, x: Any = None) -> float:
    penalty = _list_invariance_penalty(x, p, task_name)
    if task_name == "sort":
        return calc_loss_sort(p, t) + penalty
    if isinstance(t, list):
        if not isinstance(p, list):
            return 1_000_000.0 + penalty
        if len(p) != len(t):
            return 500.0 * abs(len(p) - len(t)) + penalty
        if task_name in ("reverse", "filter"):
            return sum(calc_error(pv, tv) for pv, tv in zip(p, t)) + penalty
    if task_name.startswith("arc_"):
        if not isinstance(p, list) or not p or not isinstance(p[0], list):
            return 1000.0 + penalty
        if len(p) != len(t) or len(p[0]) != len(t[0]):
            return 500.0 + abs(len(p) - len(t)) + abs(len(p[0]) - len(t[0])) + penalty
        err = 0
        for r in range(len(t)):
            for c in range(len(t[0])):
                if p[r][c] != t[r][c]:
                    err += 1
        return float(err) + penalty
    return calc_error(p, t) + penalty


def mse_exec(
    code: str,
    xs: List[Any],
    ys: List[Any],
    task_name: str = "",
    extra_env: Optional[Dict[str, Any]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_code,
) -> Tuple[bool, float, str]:
    ok, err = validator(code)
    if not ok:
        return (False, float("inf"), err)
    if validator == validate_program and not program_limits_ok(code):
        return (False, float("inf"), "program_limits")
    try:
        total_err = 0.0
        for x, y in zip(xs, ys):
            pred = safe_exec(code, x, extra_env=extra_env)
            if pred is None:
                return (False, float("inf"), "No return")
            if task_name in ("sort", "reverse", "max", "filter") or task_name.startswith("arc_"):
                total_err += calc_heuristic_loss(pred, y, task_name, x=x)
            else:
                total_err += calc_error(pred, y)
        return (True, total_err / max(1, len(xs)), "")
    except Exception as e:
        return (False, float("inf"), f"{type(e).__name__}: {str(e)}")


def _algo_equal(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_algo_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_algo_equal(a[k], b[k]) for k in a.keys())
    return a == b


def algo_exec(
    code: str,
    xs: List[Any],
    ys: List[Any],
    task_name: str,
    counterexamples: Optional[List[Tuple[Any, Any]]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_algo_program,
) -> Tuple[bool, float, int, float, int, str]:
    ok, err = validator(code)
    if not ok:
        return (False, 1.0, 0, 1.0, 0, err)
    total = 0
    timeouts = 0
    steps = 0
    failures = 0
    extra = counterexamples[:] if counterexamples else []
    xs_all = list(xs) + [x for x, _ in extra]
    ys_all = list(ys) + [y for _, y in extra]
    for x, y in zip(xs_all, ys_all):
        out, used, timeout = safe_exec_algo(code, x)
        steps += used
        if timeout:
            timeouts += 1
        if not _algo_equal(out, y):
            failures += 1
            if counterexamples is not None and len(counterexamples) < 64:
                counterexamples.append((x, y))
        total += 1
    err_rate = failures / max(1, total)
    timeout_rate = timeouts / max(1, total)
    avg_steps = steps // max(1, total)
    return (True, err_rate, avg_steps, timeout_rate, total, "")


def evaluate_algo(
    g: Genome,
    b: Batch,
    task_name: str,
    lam: float = 0.0001,
) -> EvalResult:
    code = g.code
    counterexamples = ALGO_COUNTEREXAMPLES.get(task_name, [])
    ok1, tr_err, tr_steps, tr_timeout, _, e1 = algo_exec(code, b.x_tr, b.y_tr, task_name, counterexamples)
    ok2, ho_err, ho_steps, ho_timeout, _, e2 = algo_exec(code, b.x_ho, b.y_ho, task_name, counterexamples)
    ok3, st_err, st_steps, st_timeout, _, e3 = algo_exec(code, b.x_st, b.y_st, task_name, counterexamples)
    ok4, te_err, te_steps, te_timeout, _, e4 = algo_exec(code, b.x_te, b.y_te, task_name, counterexamples)
    ok = ok1 and ok2 and ok3 and ok4 and all(math.isfinite(v) for v in (tr_err, ho_err, st_err, te_err))
    nodes = node_count(code)
    step_penalty = 0.0001 * (tr_steps + ho_steps + st_steps + te_steps)
    timeout_penalty = 0.5 * (tr_timeout + ho_timeout + st_timeout + te_timeout)
    if not ok:
        return EvalResult(False, tr_err, ho_err, st_err, te_err, nodes, float("inf"), e1 or e2 or e3 or e4 or "nan")
    # Hard cutoff: stress overflows are rejected before any score aggregation.
    if st_err > STRESS_MAX:
        return EvalResult(False, tr_err, ho_err, st_err, te_err, nodes, float("inf"), "stress_overflow")
    score = SCORE_W_HOLD * ho_err + SCORE_W_STRESS * st_err + SCORE_W_TRAIN * tr_err + lam * nodes + step_penalty + timeout_penalty
    err = e1 or e2 or e3 or e4
    return EvalResult(ok, tr_err, ho_err, st_err, te_err, nodes, score, err or None)


def evaluate(
    g: Genome,
    b: Batch,
    task_name: str,
    lam: float = 0.0001,
    extra_env: Optional[Dict[str, Any]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_code,
) -> EvalResult:
    code = g.code
    ok1, tr, e1 = mse_exec(code, b.x_tr, b.y_tr, task_name, extra_env=extra_env)
    ok2, ho, e2 = mse_exec(code, b.x_ho, b.y_ho, task_name, extra_env=extra_env)
    ok3, st, e3 = mse_exec(code, b.x_st, b.y_st, task_name, extra_env=extra_env)
    ok4, te, e4 = mse_exec(code, b.x_te, b.y_te, task_name, extra_env=extra_env)
    ok = ok1 and ok2 and ok3 and ok4 and all(math.isfinite(v) for v in (tr, ho, st, te))
    nodes = node_count(code)
    if not ok:
        return EvalResult(False, tr, ho, st, te, nodes, float("inf"), e1 or e2 or e3 or e4 or "nan")
    # Hard cutoff: stress overflows are rejected before any score aggregation.
    if st > STRESS_MAX:
        return EvalResult(False, tr, ho, st, te, nodes, float("inf"), "stress_overflow")
    score = SCORE_W_HOLD * ho + SCORE_W_STRESS * st + SCORE_W_TRAIN * tr + lam * nodes
    err = e1 or e2 or e3 or e4
    return EvalResult(ok, tr, ho, st, te, nodes, score, err or None)


def evaluate_learner(
    learner: LearnerGenome,
    b: Batch,
    task_name: str,
    adapt_steps: int = 8,
    lam: float = 0.0001,
) -> EvalResult:
    """PHASE B: evaluate learner with adaptation on training only."""
    env = safe_load_module(learner.code)
    if not env:
        return EvalResult(False, float("inf"), float("inf"), float("inf"), float("inf"), 0, float("inf"), "load_failed")
    required = ["init_mem", "encode", "predict", "update", "objective"]
    if not all(name in env and callable(env[name]) for name in required):
        return EvalResult(False, float("inf"), float("inf"), float("inf"), float("inf"), 0, float("inf"), "missing_funcs")

    init_mem = env["init_mem"]
    encode = env["encode"]
    predict = env["predict"]
    update = env["update"]
    objective = env["objective"]

    try:
        mem = init_mem()
    except Exception:
        mem = {"w": 0.0, "b": 0.0, "t": 0}

    def run_eval(xs: List[Any], ys: List[Any], do_update: bool) -> float:
        nonlocal mem
        total = 0.0
        for i, (x, y) in enumerate(zip(xs, ys)):
            try:
                z = encode(x, mem)
                y_pred = predict(z, mem)
            except Exception:
                y_pred = None
            if task_name in ("sort", "reverse", "max", "filter") or task_name.startswith("arc_"):
                total += calc_heuristic_loss(y_pred, y, task_name, x=x)
            else:
                total += calc_error(y_pred, y)
            if do_update and i < adapt_steps:
                try:
                    mem = update(mem, x, y_pred, y, 0.05)
                except Exception:
                    pass
        return total / max(1, len(xs))

    try:
        train = run_eval(b.x_tr, b.y_tr, do_update=True)
        hold = run_eval(b.x_ho, b.y_ho, do_update=False)
        stress = run_eval(b.x_st, b.y_st, do_update=False)
        test = run_eval(b.x_te, b.y_te, do_update=False)
        nodes = node_count(learner.code)
        ok = all(math.isfinite(v) for v in (train, hold, stress, test))
        if not ok:
            return EvalResult(False, train, hold, stress, test, nodes, float("inf"), "nan")
        # Hard cutoff: stress overflows are rejected before any score aggregation.
        if stress > STRESS_MAX:
            return EvalResult(False, train, hold, stress, test, nodes, float("inf"), "stress_overflow")
        obj = objective(train, hold, stress, nodes)
        if not isinstance(obj, (int, float)) or not math.isfinite(obj):
            obj = SCORE_W_HOLD * hold + SCORE_W_STRESS * stress
        score = float(obj) + lam * nodes
        ok = all(math.isfinite(v) for v in (train, hold, stress, test, score))
        return EvalResult(ok, train, hold, stress, test, nodes, score, None if ok else "nan")
    except Exception as exc:
        return EvalResult(False, float("inf"), float("inf"), float("inf"), float("inf"), 0, float("inf"), str(exc))


# ---------------------------
# Mutation operators
# ---------------------------

