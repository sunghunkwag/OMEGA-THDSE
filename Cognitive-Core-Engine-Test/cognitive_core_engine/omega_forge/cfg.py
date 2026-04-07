"""
Control flow graph analysis for the OMEGA_FORGE engine.

TODO: ControlFlowGraph also exists in unified_rsi. Keep both separate for now,
but consider consolidating in the future to reduce duplication.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import FrozenSet, List, Set, Tuple


class ControlFlowGraph:
    def __init__(self) -> None:
        self.edges: Set[Tuple[int, int, str]] = set()
        self.nodes: Set[int] = set()

    def add_edge(self, f: int, t: int, ty: str) -> None:
        self.edges.add((int(f), int(t), str(ty)))
        self.nodes.add(int(f))
        self.nodes.add(int(t))

    @staticmethod
    def from_trace(trace: List[int], code_len: int) -> "ControlFlowGraph":
        cfg = ControlFlowGraph()
        if not trace:
            return cfg
        for i in range(len(trace) - 1):
            a = trace[i]
            b = trace[i + 1]
            ty = "SEQ"
            if b <= a:
                ty = "BACK"
            cfg.add_edge(a, b, ty)
        # Add terminal edge for out-of-range halt
        last = trace[-1]
        cfg.nodes.add(last)
        cfg.nodes.add(max(0, min(code_len, last + 1)))
        return cfg

    def canonical_hash(self) -> str:
        # canonical: sorted edges + SCC size multiset
        h = hashlib.sha256()
        for f, t, ty in sorted(self.edges):
            h.update(f"{f}->{t}:{ty};".encode("utf-8"))
        scc_sizes = sorted([len(s) for s in self.sccs()])
        h.update(("SCC:" + ",".join(map(str, scc_sizes))).encode("utf-8"))
        return h.hexdigest()[:16]

    def sccs(self) -> List[FrozenSet[int]]:
        # Kosaraju
        if not self.nodes:
            return []
        adj = defaultdict(list)
        radj = defaultdict(list)
        for f, t, _ in self.edges:
            adj[f].append(t)
            radj[t].append(f)

        visited: Set[int] = set()
        order: List[int] = []

        def dfs1(u: int) -> None:
            if u in visited:
                return
            visited.add(u)
            for v in adj[u]:
                dfs1(v)
            order.append(u)

        for n in list(self.nodes):
            dfs1(n)

        visited.clear()
        comps: List[FrozenSet[int]] = []

        def dfs2(u: int, comp: Set[int]) -> None:
            if u in visited:
                return
            visited.add(u)
            comp.add(u)
            for v in radj[u]:
                dfs2(v, comp)

        for u in reversed(order):
            if u not in visited:
                comp: Set[int] = set()
                dfs2(u, comp)
                # SCC is meaningful if size>1 or has a self-loop
                if len(comp) > 1:
                    comps.append(frozenset(comp))
                else:
                    x = next(iter(comp)) if comp else None
                    if x is not None and any((x, x, ty) in self.edges for ty in ("SEQ", "BACK")):
                        comps.append(frozenset(comp))
        return comps

    def edit_distance_to(self, other: "ControlFlowGraph") -> int:
        # symmetric difference on typed edges
        return len(self.edges ^ other.edges)
