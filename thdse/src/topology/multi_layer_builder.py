"""
MultiLayerGraphBuilder — Pure preprocessing tool that decomposes source code
into a multi-layer directed graph with three edge types:

  1. AST (structural parent-child)
  2. CFG (intra-procedural control flow)
  3. Data-Dep (lightweight def-use chains)

No encoding, no swarm coupling. Deterministic graph construction only.
"""

import ast
import networkx as nx
from typing import Dict, List

from src.topology.ast_graph_ca import TopologicalASTGraphCA


class MultiLayerGraphBuilder:
    """Builds a DiGraph with three edge types: ast, cfg, data_dep."""

    EDGE_AST = "ast"
    EDGE_CFG = "cfg"
    EDGE_DATA = "data_dep"

    def __init__(self, steps: int = 5):
        self.ca = TopologicalASTGraphCA(steps=steps)

    def build(self, code: str) -> nx.DiGraph:
        tree = ast.parse(code)

        # Layer 1: AST edges (reuse existing CA builder)
        G = self.ca.code_to_graph(tree=tree)
        for u, v in G.edges:
            G.edges[u, v]["layer"] = self.EDGE_AST

        # Layer 2: intra-procedural CFG edges
        self._add_cfg_edges(G, tree)

        # Layer 3: def-use data dependency edges
        self._add_data_dep_edges(G, tree)

        return G

    # ── CFG edges ────────────────────────────────────────────────

    def _add_cfg_edges(self, G: nx.DiGraph, tree: ast.Module):
        """Sequential control flow between top-level statements
        and within function bodies.  Branch targets for If/While."""
        for node in ast.walk(tree):
            stmts = self._get_body(node)
            if stmts is None:
                continue
            for i in range(len(stmts) - 1):
                src, dst = id(stmts[i]), id(stmts[i + 1])
                if src in G.nodes and dst in G.nodes:
                    G.add_edge(src, dst, layer=self.EDGE_CFG)

            # branch edges
            for stmt in stmts:
                if isinstance(stmt, ast.If):
                    self._add_branch_cfg(G, stmt)
                elif isinstance(stmt, (ast.While, ast.For)):
                    self._add_loop_cfg(G, stmt)

    def _add_branch_cfg(self, G: nx.DiGraph, node: ast.If):
        if_id = id(node)
        if node.body:
            G.add_edge(if_id, id(node.body[0]), layer=self.EDGE_CFG)
        if node.orelse:
            G.add_edge(if_id, id(node.orelse[0]), layer=self.EDGE_CFG)

    def _add_loop_cfg(self, G: nx.DiGraph, node):
        loop_id = id(node)
        if node.body:
            G.add_edge(loop_id, id(node.body[0]), layer=self.EDGE_CFG)
            G.add_edge(id(node.body[-1]), loop_id, layer=self.EDGE_CFG)

    @staticmethod
    def _get_body(node) -> list | None:
        if hasattr(node, "body") and isinstance(node.body, list):
            return node.body
        return None

    # ── Data-dependency edges ────────────────────────────────────

    def _add_data_dep_edges(self, G: nx.DiGraph, tree: ast.Module):
        """Lightweight def-use chains: for each Name(Load), draw an
        edge from the most recent Name(Store) of the same identifier
        within the same scope (function body or module)."""
        for node in ast.walk(tree):
            stmts = self._get_body(node)
            if stmts is None:
                continue
            self._scan_defuse(G, stmts)

    def _scan_defuse(self, G: nx.DiGraph, stmts: list):
        last_def: Dict[str, int] = {}
        for stmt in stmts:
            uses = set()
            defs = set()
            for child in ast.walk(stmt):
                if isinstance(child, ast.Name):
                    if isinstance(child.ctx, ast.Load):
                        uses.add(child.id)
                    elif isinstance(child.ctx, (ast.Store, ast.Del)):
                        defs.add(child.id)

            stmt_id = id(stmt)
            for var in uses:
                if var in last_def and last_def[var] in G.nodes and stmt_id in G.nodes:
                    G.add_edge(last_def[var], stmt_id, layer=self.EDGE_DATA)

            for var in defs:
                last_def[var] = stmt_id
