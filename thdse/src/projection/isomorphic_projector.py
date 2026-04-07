"""
Isomorphic Projector — Maps a multi-layer directed graph (AST + CFG + Data-Dep)
into a single holographic FHRR hypervector via the hdc_core VSA arena.

Mathematical contract:
  - Control-flow sequences  →  bind (⊗) chain  (preserves ordering)
  - Data-dependency sets    →  bundle (⊕)       (preserves membership, order-invariant)
  - AST structural identity →  bind (type ⊗ position ⊗ CA-state)

The output is a LayeredProjection containing per-layer handles AND phase arrays,
enabling algebraic unbinding (conjugate) for layer-aware decoding.
"""

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from src.topology.multi_layer_builder import MultiLayerGraphBuilder
from src.utils.arena_ops import bind_phases, bundle_phases, negate_phases, bind_bundle_fusion_phases


@dataclass
class LayeredProjection:
    """Result of projecting code into the VSA arena, with per-layer decomposition.

    Stores both arena handles and the corresponding phase arrays so that
    the decoder can reconstruct conjugates for algebraic unbinding.
    """
    final_handle: int                           # ast ⊗ cfg ⊗ data (backward compatible)
    ast_handle: int                             # ast_bundle before cross-layer bind
    cfg_handle: Optional[int]                   # cfg_bundle (None if no CFG edges)
    data_handle: Optional[int]                  # data_bundle (None if no data-dep edges)
    ast_phases: List[float] = field(default_factory=list)    # phase array of ast_bundle
    cfg_phases: Optional[List[float]] = None    # phase array of cfg_bundle
    data_phases: Optional[List[float]] = None   # phase array of data_bundle


class IsomorphicProjector:
    """Projects a multi-layer code graph into a LayeredProjection."""

    def __init__(self, arena: Any, dimension: int, ca_steps: int = 5):
        self.arena = arena
        self.dimension = dimension
        self.builder = MultiLayerGraphBuilder(steps=ca_steps)
        self._seed_cache: Dict[str, int] = {}
        self._phase_cache: Dict[str, List[float]] = {}

    # ── Deterministic phase generation ───────────────────────────

    def _deterministic_phases(self, seed: int) -> List[float]:
        """Generate a repeatable phase vector from a 32-bit seed.
        Uses md5-seeded linear congruential expansion — no PRNG state."""
        phases = []
        state = seed & 0xFFFFFFFF
        for _ in range(self.dimension):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            phase = ((state >> 33) / (2**31)) * 2.0 * math.pi - math.pi
            phases.append(phase)
        return phases

    def _get_or_create_handle(self, cache_key: str) -> int:
        """Allocate and inject a deterministic hypervector, cached by key."""
        if cache_key not in self._seed_cache:
            seed = int(hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:8], 16)
            phases = self._deterministic_phases(seed)
            handle = self.arena.allocate()
            self.arena.inject_phases(handle, phases)
            self._seed_cache[cache_key] = handle
            self._phase_cache[cache_key] = phases
        return self._seed_cache[cache_key]

    def _get_phases(self, cache_key: str) -> List[float]:
        """Retrieve the phase array for a cached key."""
        return self._phase_cache[cache_key]

    # ── Node-level projection ────────────────────────────────────

    def _project_node(self, node_id: int, data: dict) -> int:
        """Project a single graph node → handle = type ⊗ pos ⊗ ca_state."""
        node_type = data["type"]
        depth = data.get("depth", 0)
        sibling_idx = data.get("sibling_index", 0)

        type_h = self._get_or_create_handle(f"type:{node_type}")
        pos_h = self._get_or_create_handle(f"pos:{depth}:{sibling_idx}")

        ca_phase = float(np.angle(data["state"]))
        state_h = self.arena.allocate()
        self.arena.inject_phases(state_h, [ca_phase] * self.dimension)

        tmp = self.arena.allocate()
        self.arena.bind(type_h, pos_h, tmp)

        node_h = self.arena.allocate()
        self.arena.bind(tmp, state_h, node_h)
        return node_h

    def _project_node_phases(self, data: dict) -> List[float]:
        """Compute the phase array for a projected node (parallel to arena ops).

        node_phases = type_phases + pos_phases + state_phases  (element-wise)
        """
        node_type = data["type"]
        depth = data.get("depth", 0)
        sibling_idx = data.get("sibling_index", 0)

        # Ensure handles exist (populates phase cache)
        self._get_or_create_handle(f"type:{node_type}")
        self._get_or_create_handle(f"pos:{depth}:{sibling_idx}")

        type_phases = self._get_phases(f"type:{node_type}")
        pos_phases = self._get_phases(f"pos:{depth}:{sibling_idx}")

        ca_phase = float(np.angle(data["state"]))
        state_phases = [ca_phase] * self.dimension

        # bind is phase addition: type ⊗ pos ⊗ state
        return bind_phases(bind_phases(type_phases, pos_phases), state_phases)

    # ── Layer-specific edge projections ──────────────────────────

    def _project_cfg_chains(self, G: nx.DiGraph, node_handles: Dict[int, int]) -> List[int]:
        """CFG edges encode ordering → chain-bind along each maximal CFG path."""
        role_cfg = self._get_or_create_handle("role:cfg")

        cfg_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("layer") == MultiLayerGraphBuilder.EDGE_CFG
        ]
        cfg_sub = nx.DiGraph(cfg_edges)

        chain_handles = []
        sources = [n for n in cfg_sub.nodes() if cfg_sub.in_degree(n) == 0]
        if not sources:
            sources = list(cfg_sub.nodes())[:1]

        visited_edges = set()
        for src in sources:
            path = [src]
            current = src
            while True:
                succs = [s for s in cfg_sub.successors(current)
                         if (current, s) not in visited_edges]
                if not succs:
                    break
                nxt = succs[0]
                visited_edges.add((current, nxt))
                path.append(nxt)
                current = nxt

            if len(path) < 2:
                continue

            chain_h = self.arena.allocate()
            first_node = path[0]
            if first_node in node_handles:
                self.arena.bind(role_cfg, node_handles[first_node], chain_h)
            else:
                continue

            for step_node in path[1:]:
                if step_node not in node_handles:
                    continue
                next_chain = self.arena.allocate()
                self.arena.bind(chain_h, node_handles[step_node], next_chain)
                chain_h = next_chain

            chain_handles.append(chain_h)

        return chain_handles

    def _project_cfg_chain_phases(
        self, G: nx.DiGraph, node_phase_map: Dict[int, List[float]]
    ) -> List[List[float]]:
        """Compute phase arrays for CFG chains (parallel to arena ops)."""
        self._get_or_create_handle("role:cfg")
        role_phases = self._get_phases("role:cfg")

        cfg_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("layer") == MultiLayerGraphBuilder.EDGE_CFG
        ]
        cfg_sub = nx.DiGraph(cfg_edges)

        chain_phase_arrays = []
        sources = [n for n in cfg_sub.nodes() if cfg_sub.in_degree(n) == 0]
        if not sources:
            sources = list(cfg_sub.nodes())[:1]

        visited_edges = set()
        for src in sources:
            path = [src]
            current = src
            while True:
                succs = [s for s in cfg_sub.successors(current)
                         if (current, s) not in visited_edges]
                if not succs:
                    break
                nxt = succs[0]
                visited_edges.add((current, nxt))
                path.append(nxt)
                current = nxt

            if len(path) < 2:
                continue

            first_node = path[0]
            if first_node not in node_phase_map:
                continue

            chain_p = bind_phases(role_phases, node_phase_map[first_node])
            for step_node in path[1:]:
                if step_node not in node_phase_map:
                    continue
                chain_p = bind_phases(chain_p, node_phase_map[step_node])

            chain_phase_arrays.append(chain_p)

        return chain_phase_arrays

    def _project_data_dep_sets(self, G: nx.DiGraph, node_handles: Dict[int, int]) -> List[int]:
        """Data-dep edges → bundle per node, then bind with role."""
        role_data = self._get_or_create_handle("role:data_dep")

        dep_handles = []
        for node in G.nodes():
            neighbors = set()
            for _, v, d in G.out_edges(node, data=True):
                if d.get("layer") == MultiLayerGraphBuilder.EDGE_DATA:
                    neighbors.add(v)
            for u, _, d in G.in_edges(node, data=True):
                if d.get("layer") == MultiLayerGraphBuilder.EDGE_DATA:
                    neighbors.add(u)

            valid = [node_handles[n] for n in neighbors if n in node_handles]
            if not valid:
                continue

            dep_bundle = self.arena.allocate()
            self.arena.bundle(valid, dep_bundle)

            bound = self.arena.allocate()
            self.arena.bind(role_data, dep_bundle, bound)
            dep_handles.append(bound)

        return dep_handles

    def _project_data_dep_phases(
        self, G: nx.DiGraph, node_phase_map: Dict[int, List[float]]
    ) -> List[List[float]]:
        """Compute phase arrays for data-dep projections (parallel to arena ops)."""
        self._get_or_create_handle("role:data_dep")
        role_phases = self._get_phases("role:data_dep")

        dep_phase_arrays = []
        for node in G.nodes():
            neighbors = set()
            for _, v, d in G.out_edges(node, data=True):
                if d.get("layer") == MultiLayerGraphBuilder.EDGE_DATA:
                    neighbors.add(v)
            for u, _, d in G.in_edges(node, data=True):
                if d.get("layer") == MultiLayerGraphBuilder.EDGE_DATA:
                    neighbors.add(u)

            valid_phases = [node_phase_map[n] for n in neighbors if n in node_phase_map]
            if not valid_phases:
                continue

            dep_bundle_p = bundle_phases(valid_phases)
            dep_p = bind_phases(role_phases, dep_bundle_p)
            dep_phase_arrays.append(dep_p)

        return dep_phase_arrays

    # ── Full projection pipeline ─────────────────────────────────

    def project(self, code: str) -> 'LayeredProjection':
        """Map source code → LayeredProjection with per-layer handles and phases.

        The final_handle is the cross-layer bind: ast ⊗ cfg ⊗ data.
        Layer handles and phase arrays enable algebraic unbinding for decoding.
        """
        G = self.builder.build(code)
        G = self.builder.ca.evolve(G)
        return self._project_from_graph(G)

    def project_handle(self, code: str) -> int:
        """Backward-compatible: returns only the final bound handle."""
        return self.project(code).final_handle

    def _project_from_graph(self, G: nx.DiGraph) -> 'LayeredProjection':
        """Core projection logic on a pre-built, evolved graph."""
        # Phase 1: project every node (arena handles + phase arrays)
        node_handles: Dict[int, int] = {}
        node_phase_map: Dict[int, List[float]] = {}
        for node in G.nodes():
            node_handles[node] = self._project_node(node, G.nodes[node])
            node_phase_map[node] = self._project_node_phases(G.nodes[node])

        # Phase 2: AST structural layer — bundle all node vectors
        ast_bundle_h = self.arena.allocate()
        all_node_h = list(node_handles.values())
        all_node_phases = [node_phase_map[n] for n in G.nodes()]
        if all_node_h:
            self.arena.bundle(all_node_h, ast_bundle_h)
        ast_phases = bundle_phases(all_node_phases) if all_node_phases else [0.0] * self.dimension

        # Phase 3: CFG sequential layer
        cfg_chains = self._project_cfg_chains(G, node_handles)
        cfg_chain_phases = self._project_cfg_chain_phases(G, node_phase_map)
        if cfg_chains:
            cfg_layer_h = self.arena.allocate()
            self.arena.bundle(cfg_chains, cfg_layer_h)
            cfg_phases = bundle_phases(cfg_chain_phases)
        else:
            cfg_layer_h = None
            cfg_phases = None

        # Phase 4: Data-dep set layer
        dep_handles = self._project_data_dep_sets(G, node_handles)
        dep_phase_arrays = self._project_data_dep_phases(G, node_phase_map)
        if dep_handles:
            data_layer_h = self.arena.allocate()
            self.arena.bundle(dep_handles, data_layer_h)
            data_phases = bundle_phases(dep_phase_arrays)
        else:
            data_layer_h = None
            data_phases = None

        # Phase 5: Cross-layer bind → final hologram
        result_h = ast_bundle_h
        if cfg_layer_h is not None:
            merged = self.arena.allocate()
            self.arena.bind(result_h, cfg_layer_h, merged)
            result_h = merged
        if data_layer_h is not None:
            merged = self.arena.allocate()
            self.arena.bind(result_h, data_layer_h, merged)
            result_h = merged

        return LayeredProjection(
            final_handle=result_h,
            ast_handle=ast_bundle_h,
            cfg_handle=cfg_layer_h,
            data_handle=data_layer_h,
            ast_phases=ast_phases,
            cfg_phases=cfg_phases,
            data_phases=data_phases,
        )

    def project_graph(self, G: nx.DiGraph) -> 'LayeredProjection':
        """Project a pre-built, pre-evolved graph."""
        return self._project_from_graph(G)

    # ── Contradiction Vector Synthesis ────────────────────────────

    def synthesize_contradiction_vector(
        self, atom_handles: List[int],
    ) -> int:
        """Synthesize a single Contradiction Vector (V_error) from UNSAT core atoms.

        Given the arena handles of the conflicting structural atoms isolated
        from the Z3 UNSAT core, algebraically combine them:

          V_error = bind(atom₁, bind(atom₂, ... bundle([remaining atoms]) ...))

        For a single atom:    V_error = atom
        For two atoms:        V_error = bind(atom₁, atom₂)
        For three+ atoms:     V_error = bind(atom₁, bundle([atom₂, ..., atomₖ]))

        This encodes the exact topological defect as a single hypervector
        whose direction in the FHRR space represents the axis of contradiction.

        Args:
            atom_handles: Arena handles of conflicting structural atoms from
                         the UNSAT core (reverse-mapped from Z3 constraint labels).

        Returns:
            Arena handle of the synthesized V_error.
        """
        if not atom_handles:
            raise ValueError("Cannot synthesize contradiction vector from empty handle list")

        if len(atom_handles) == 1:
            return atom_handles[0]

        if len(atom_handles) == 2:
            v_error = self.arena.allocate()
            self.arena.bind(atom_handles[0], atom_handles[1], v_error)
            return v_error

        # For 3+ atoms: V_error = bind(atom₁, bundle([atom₂, ..., atomₖ]))
        # This uses the fusion operator: fuse(A, {B₁,...,Bₖ}) = A ⊗ (B₁ ⊕ ... ⊕ Bₖ)
        v_error = self.arena.allocate()
        self.arena.bind_bundle_fusion(
            atom_handles[0], atom_handles[1:], v_error,
        )
        return v_error
