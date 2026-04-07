"""Resonance Clique Consensus — Algebraic consensus from candidate phase arrays.

Given N candidate vectors from M agents, find the subset that mutually resonate
above threshold, bundled into a weighted centroid. Uses correlate_matrix (single
FFI call per layer) and Bron-Kerbosch clique extraction.

Adjacency rules (logical OR):
  1. Full structural match: correlate(final_i, final_j) > consensus_threshold
  2. Algorithmic isomorphism: correlate(cfg_i, cfg_j) > layer_threshold AND
     correlate(data_i, data_j) > layer_threshold
  3. Structural isomorphism: correlate(ast_i, ast_j) > layer_threshold AND
     correlate(cfg_i, cfg_j) > layer_threshold

Clique scoring and member weighting use rule-aware effective correlation:
edges formed via algo_iso use mean(cfg_corr, data_corr), edges formed via
struct_iso use mean(ast_corr, cfg_corr), and edges formed via full match
use final_corr. This prevents geometric discrimination against cliques
formed through partial-layer consensus.

No voting. No debate. Pure algebraic composition.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.arena_ops import weighted_bundle_phases


@dataclass
class ConsensusResult:
    """Result of swarm consensus computation."""
    centroid_phases: List[float]          # Weighted centroid of winning clique
    clique_member_indices: List[int]      # Which candidates are in the clique
    clique_agent_ids: List[int]           # Which agents contributed
    mean_resonance: float                 # Mean pairwise effective rho within clique
    clique_size: int                      # Number of members
    # Which adjacency rule triggered each edge
    adjacency_rule_counts: Dict[str, int] = field(
        default_factory=lambda: {"full": 0, "algo_iso": 0, "struct_iso": 0}
    )
    # Per-layer centroid phases
    centroid_ast_phases: Optional[List[float]] = None
    centroid_cfg_phases: Optional[List[float]] = None
    centroid_data_phases: Optional[List[float]] = None


def _bron_kerbosch(
    R: Set[int],
    P: Set[int],
    X: Set[int],
    adj: Dict[int, Set[int]],
    results: List[List[int]],
) -> None:
    """Bron-Kerbosch maximal clique enumeration (deterministic)."""
    if not P and not X:
        results.append(sorted(R))
        return
    pivot_candidates = sorted(P | X)
    if not pivot_candidates:
        return
    pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
    candidates = sorted(P - adj[pivot])
    for v in candidates:
        _bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, results)
        P = P - {v}
        X = X | {v}


def _inject_and_correlate(
    arena: Any,
    phase_arrays: List[Optional[List[float]]],
) -> Tuple[Dict[tuple, float], List[Optional[int]]]:
    """Inject phase arrays into arena and compute the correlation matrix.

    Returns (corr_matrix, handles). Entries with None phases are skipped
    and will have None handles.
    """
    n = len(phase_arrays)
    handles: List[Optional[int]] = [None] * n
    valid_indices = []

    for i, phases in enumerate(phase_arrays):
        if phases is not None:
            h = arena.allocate()
            arena.inject_phases(h, phases)
            handles[i] = h
            valid_indices.append(i)

    corr_matrix: Dict[tuple, float] = {}

    if len(valid_indices) >= 2:
        valid_handles = [handles[i] for i in valid_indices]
        flat = arena.correlate_matrix(valid_handles)

        k = 0
        for ii in range(len(valid_indices)):
            idx_i = valid_indices[ii]
            corr_matrix[(idx_i, idx_i)] = arena.compute_correlation(
                handles[idx_i], handles[idx_i],
            )
            for jj in range(ii + 1, len(valid_indices)):
                idx_j = valid_indices[jj]
                corr = flat[k]
                k += 1
                corr_matrix[(idx_i, idx_j)] = corr
                corr_matrix[(idx_j, idx_i)] = corr
    elif len(valid_indices) == 1:
        idx = valid_indices[0]
        corr_matrix[(idx, idx)] = arena.compute_correlation(
            handles[idx], handles[idx],
        )

    return corr_matrix, handles


def _effective_corr(
    i: int, j: int,
    edge_rules: Dict[Tuple[int, int], str],
    final_corr: Dict[tuple, float],
    cfg_corr: Dict[tuple, float],
    data_corr: Dict[tuple, float],
    ast_corr: Dict[tuple, float],
) -> float:
    """Compute the effective correlation for a pair based on the adjacency rule.

    For full match: use final_corr (cross-layer bound vector).
    For algo_iso: use mean(cfg_corr, data_corr) — the layers that matched.
    For struct_iso: use mean(ast_corr, cfg_corr) — the layers that matched.

    This prevents geometric discrimination: an algo_iso clique with strong
    CFG+Data correlation is scored by its actual strength, not penalized
    by its inherently low final_corr (which is dominated by the differing AST).
    """
    key = (min(i, j), max(i, j))
    rule = edge_rules.get(key, "full")

    if rule == "algo_iso":
        c = cfg_corr.get((i, j), 0.0)
        d = data_corr.get((i, j), 0.0)
        return (c + d) / 2.0
    elif rule == "struct_iso":
        a = ast_corr.get((i, j), 0.0)
        c = cfg_corr.get((i, j), 0.0)
        return (a + c) / 2.0
    else:
        return final_corr.get((i, j), 0.0)


def compute_swarm_consensus(
    arena: Any,
    candidate_phases: List[List[float]],
    candidate_metadata: List[Dict[str, Any]],
    consensus_threshold: float = 0.85,
    min_clique_size: int = 2,
    # Layered consensus parameters
    candidate_ast_phases: Optional[List[Optional[List[float]]]] = None,
    candidate_cfg_phases: Optional[List[Optional[List[float]]]] = None,
    candidate_data_phases: Optional[List[Optional[List[float]]]] = None,
    layer_threshold: float = 0.70,
) -> Optional[ConsensusResult]:
    """Compute the algebraic consensus from candidate phase arrays.

    Algorithm:
      1. Inject all candidate phases into a temporary arena
      2. Call arena.correlate_matrix() per layer (up to 4 calls: final + 3 layers)
      3. Build adjacency graph using layered partial match rules (logical OR)
      4. Run Bron-Kerbosch to find maximal cliques
      5. Select the largest clique (ties broken by mean effective resonance)
      6. Weighted bundle clique members into centroid (rule-aware weights)
      7. Return centroid phases + clique membership + rule statistics

    Args:
        arena: A temporary FhrrArena for consensus computation.
               (NOT any agent's arena -- create a fresh one)
        candidate_phases: List of final-handle phase arrays from all agents.
        candidate_metadata: Parallel list of metadata dicts.
        consensus_threshold: Minimum rho for full-match adjacency.
        min_clique_size: Minimum clique size for valid consensus.
        candidate_ast_phases: Optional per-candidate AST layer phases.
        candidate_cfg_phases: Optional per-candidate CFG layer phases.
        candidate_data_phases: Optional per-candidate Data layer phases.
        layer_threshold: Minimum rho for per-layer adjacency rules.

    Returns:
        ConsensusResult or None if no clique meets the threshold.
    """
    n = len(candidate_phases)
    if n < min_clique_size:
        return None

    # Step 1: Compute final-handle correlation matrix
    final_corr, final_handles = _inject_and_correlate(arena, candidate_phases)

    # Step 2: Compute per-layer correlation matrices (if provided)
    ast_corr: Dict[tuple, float] = {}
    cfg_corr: Dict[tuple, float] = {}
    data_corr: Dict[tuple, float] = {}

    has_layers = (
        candidate_ast_phases is not None
        or candidate_cfg_phases is not None
        or candidate_data_phases is not None
    )

    if candidate_ast_phases is not None:
        ast_corr, _ = _inject_and_correlate(arena, candidate_ast_phases)
    if candidate_cfg_phases is not None:
        cfg_corr, _ = _inject_and_correlate(arena, candidate_cfg_phases)
    if candidate_data_phases is not None:
        data_corr, _ = _inject_and_correlate(arena, candidate_data_phases)

    # Step 3: Build adjacency graph with layered partial match
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    rule_counts = {"full": 0, "algo_iso": 0, "struct_iso": 0}
    # Track which rule connected each edge (canonical key: min, max)
    edge_rules: Dict[Tuple[int, int], str] = {}

    for i in range(n):
        for j in range(i + 1, n):
            matched = False

            # Rule 1: Full structural match
            final_rho = final_corr.get((i, j), 0.0)
            if final_rho > consensus_threshold:
                rule_counts["full"] += 1
                edge_rules[(i, j)] = "full"
                matched = True

            # Rule 2: Algorithmic isomorphism (CFG + Data)
            if not matched and has_layers:
                cfg_ok = (
                    candidate_cfg_phases is not None
                    and candidate_cfg_phases[i] is not None
                    and candidate_cfg_phases[j] is not None
                    and cfg_corr.get((i, j), 0.0) > layer_threshold
                )
                data_ok = (
                    candidate_data_phases is not None
                    and candidate_data_phases[i] is not None
                    and candidate_data_phases[j] is not None
                    and data_corr.get((i, j), 0.0) > layer_threshold
                )
                if cfg_ok and data_ok:
                    rule_counts["algo_iso"] += 1
                    edge_rules[(i, j)] = "algo_iso"
                    matched = True

            # Rule 3: Structural isomorphism (AST + CFG)
            if not matched and has_layers:
                ast_ok = (
                    candidate_ast_phases is not None
                    and candidate_ast_phases[i] is not None
                    and candidate_ast_phases[j] is not None
                    and ast_corr.get((i, j), 0.0) > layer_threshold
                )
                cfg_ok2 = (
                    candidate_cfg_phases is not None
                    and candidate_cfg_phases[i] is not None
                    and candidate_cfg_phases[j] is not None
                    and cfg_corr.get((i, j), 0.0) > layer_threshold
                )
                if ast_ok and cfg_ok2:
                    rule_counts["struct_iso"] += 1
                    edge_rules[(i, j)] = "struct_iso"
                    matched = True

            if matched:
                adj[i].add(j)
                adj[j].add(i)

    # Step 4: Bron-Kerbosch clique extraction
    cliques: List[List[int]] = []
    _bron_kerbosch(set(), set(range(n)), set(), adj, cliques)

    # Filter by minimum size
    valid_cliques = [c for c in cliques if len(c) >= min_clique_size]
    if not valid_cliques:
        return None

    # Step 5: Select best clique using rule-aware effective correlation
    def clique_score(clique: List[int]) -> tuple:
        pairs = [(clique[a], clique[b])
                 for a in range(len(clique)) for b in range(a + 1, len(clique))]
        mean_eff = sum(
            _effective_corr(x, y, edge_rules, final_corr, cfg_corr, data_corr, ast_corr)
            for x, y in pairs
        ) / max(len(pairs), 1)
        return (len(clique), mean_eff)

    best_clique = max(valid_cliques, key=clique_score)
    _, mean_resonance = clique_score(best_clique)

    # Step 6: Weighted bundle using rule-aware effective correlation
    member_weights = []
    for idx in best_clique:
        others = [j for j in best_clique if j != idx]
        if others:
            mean_eff = sum(
                _effective_corr(idx, j, edge_rules, final_corr, cfg_corr, data_corr, ast_corr)
                for j in others
            ) / len(others)
        else:
            mean_eff = 1.0
        # Clamp to [0.1, 1.0] to prevent zero-weight members
        member_weights.append(max(0.1, mean_eff))

    # Weighted bundle in phase domain
    clique_phase_arrays = [candidate_phases[i] for i in best_clique]
    centroid_phases = weighted_bundle_phases(clique_phase_arrays, member_weights)

    # Inject result into arena for downstream use
    centroid_h = arena.allocate()
    arena.inject_phases(centroid_h, centroid_phases)

    # Per-layer weighted centroids
    centroid_ast_phases = None
    centroid_cfg_phases = None
    centroid_data_phases = None

    if candidate_ast_phases is not None:
        ast_arrays = [
            candidate_ast_phases[i] for i in best_clique
            if candidate_ast_phases[i] is not None
        ]
        ast_wts = [
            member_weights[k] for k, i in enumerate(best_clique)
            if candidate_ast_phases[i] is not None
        ]
        if ast_arrays:
            centroid_ast_phases = weighted_bundle_phases(ast_arrays, ast_wts)

    if candidate_cfg_phases is not None:
        cfg_arrays = [
            candidate_cfg_phases[i] for i in best_clique
            if candidate_cfg_phases[i] is not None
        ]
        cfg_wts = [
            member_weights[k] for k, i in enumerate(best_clique)
            if candidate_cfg_phases[i] is not None
        ]
        if cfg_arrays:
            centroid_cfg_phases = weighted_bundle_phases(cfg_arrays, cfg_wts)

    if candidate_data_phases is not None:
        data_arrays = [
            candidate_data_phases[i] for i in best_clique
            if candidate_data_phases[i] is not None
        ]
        data_wts = [
            member_weights[k] for k, i in enumerate(best_clique)
            if candidate_data_phases[i] is not None
        ]
        if data_arrays:
            centroid_data_phases = weighted_bundle_phases(data_arrays, data_wts)

    # Extract agent IDs from metadata
    clique_agent_ids = []
    for idx in best_clique:
        if idx < len(candidate_metadata):
            agent_id = candidate_metadata[idx].get("agent_id", -1)
            clique_agent_ids.append(agent_id)
        else:
            clique_agent_ids.append(-1)

    return ConsensusResult(
        centroid_phases=centroid_phases,
        clique_member_indices=best_clique,
        clique_agent_ids=clique_agent_ids,
        mean_resonance=mean_resonance,
        clique_size=len(best_clique),
        adjacency_rule_counts=rule_counts,
        centroid_ast_phases=centroid_ast_phases,
        centroid_cfg_phases=centroid_cfg_phases,
        centroid_data_phases=centroid_data_phases,
    )
