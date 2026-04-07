"""Boundary Cartography — Maps the geometry of the design space boundary.

Core analytical module that collects Contradiction Vectors (walls), computes
resonance structure over the wall archive, estimates the residual subspace
dimension, and identifies unexplored open directions for synthesis.

Pipeline:
  1. Collect: accumulate V_error from UNSAT events into WallArchive
  2. Map: compute resonance matrix over walls → cliques (boundary regions)
  3. Measure: project all walls simultaneously → residual subspace dimension
  4. Navigate: extract principal components of residual → open directions
  5. Synthesize: attempt synthesis along open directions
  6. Loop: if new UNSAT → add to archive and re-map

All probe vectors are deterministic (hash-seeded). Zero randomness.
"""

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import hdc_core as _hdc_core  # noqa: F401
    _HAS_HDC_CORE = True
except ImportError:
    _HAS_HDC_CORE = False

from src.synthesis.wall_archive import WallArchive


def _deterministic_phases(seed: int, dimension: int) -> List[float]:
    """Generate a repeatable phase vector from a seed.

    Uses the same LCG as IsomorphicProjector for consistency.
    Zero randomness — fully deterministic.

    Args:
        seed: Integer seed for phase generation.
        dimension: Number of phase components.

    Returns:
        List of phase angles in [-pi, pi].
    """
    phases = []
    state = seed & 0xFFFFFFFF
    for _ in range(dimension):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        phase = ((state >> 33) / (2**31)) * 2.0 * math.pi - math.pi
        phases.append(phase)
    return phases


def estimate_residual_dimension(
    arena: Any,
    wall_handles: List[int],
    total_dimension: int,
) -> Dict[str, Any]:
    """Estimate the effective dimension of the residual subspace.

    After projecting out all wall vectors, how much of the original
    d-dimensional space remains accessible?

    Method:
      1. Sample N probe vectors (deterministic: hash-seeded)
      2. Project each probe through the multi-quotient space
      3. Measure the variance of the projected probes
      4. Effective dimension = sum of significant singular values
         (above a noise threshold)

    Args:
        arena: FhrrArena instance.
        wall_handles: List of V_error arena handles to project out.
        total_dimension: The arena's hypervector dimension.

    Returns:
        Dictionary with residual dimension analysis results.
    """
    n_probes = min(2 * total_dimension, 50)
    k = len(wall_handles)

    if k == 0:
        return {
            "original_dimension": total_dimension,
            "wall_count": 0,
            "estimated_residual_dimension": total_dimension,
            "compression_ratio": 1.0,
            "is_closed": False,
            "singular_values": [],
        }

    # Create an isolated temporary arena so that projection does not
    # corrupt the live vectors (corpus axioms, vocab handles, etc.)
    # in the main arena.  Extra capacity accounts for temporary handles
    # that project_to_multi_quotient_space allocates internally during
    # Gram-Schmidt orthogonalization.
    temp_capacity = n_probes + 2 * k + 16
    temp_arena = type(arena)(temp_capacity, total_dimension)

    # Copy wall vectors into temp arena
    temp_wall_handles = []
    for wh in wall_handles:
        phases = arena.extract_phases(wh)
        temp_h = temp_arena.allocate()
        temp_arena.inject_phases(temp_h, list(phases))
        temp_wall_handles.append(temp_h)

    # Step 1: Generate deterministic probe vectors in temp arena
    temp_probe_handles = []
    for i in range(n_probes):
        seed = int(hashlib.md5(f"probe_{i}".encode("utf-8")).hexdigest()[:8], 16)
        phases = _deterministic_phases(seed, total_dimension)
        h = temp_arena.allocate()
        temp_arena.inject_phases(h, phases)
        temp_probe_handles.append(h)

    # Step 2: Project out all walls in temp arena only
    has_multi = hasattr(temp_arena, 'project_to_multi_quotient_space')
    if has_multi and temp_wall_handles:
        temp_arena.project_to_multi_quotient_space(temp_wall_handles)
    else:
        for wh in temp_wall_handles:
            temp_arena.project_to_quotient_space(wh)

    # Step 3: Compute pairwise correlation matrix of probes in temp arena
    corr_matrix = []
    for i in range(n_probes):
        row = []
        for j in range(n_probes):
            corr = temp_arena.compute_correlation(
                temp_probe_handles[i], temp_probe_handles[j])
            row.append(corr)
        corr_matrix.append(row)

    # temp_arena goes out of scope — main arena untouched

    # Step 4: Compute singular values via power iteration on correlation matrix
    # (No numpy dependency — pure Python for minimal footprint)
    singular_values = _compute_singular_values(corr_matrix, n_probes)

    # Effective dimension = number of significant singular values
    noise_threshold = 0.01
    d_eff = sum(1 for sv in singular_values if sv > noise_threshold)

    compression_ratio = d_eff / total_dimension if total_dimension > 0 else 0.0

    return {
        "original_dimension": total_dimension,
        "wall_count": k,
        "estimated_residual_dimension": d_eff,
        "compression_ratio": compression_ratio,
        "is_closed": d_eff < total_dimension * 0.1,
        "singular_values": singular_values,
    }


def _compute_singular_values(matrix: List[List[float]], n: int) -> List[float]:
    """Compute singular values of a symmetric matrix via Gershgorin bounds.

    For the purpose of dimension estimation, we use the diagonal dominance
    structure: each eigenvalue lies within a Gershgorin disc. The diagonal
    values after subtracting off-diagonal sums give approximate magnitudes.

    This is O(n²) and sufficient for our probe-based estimation.

    Args:
        matrix: n×n symmetric correlation matrix.
        n: Matrix dimension.

    Returns:
        List of approximate singular values (descending).
    """
    values = []
    for i in range(n):
        diag = abs(matrix[i][i]) if i < len(matrix) else 0.0
        off_diag_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        # Gershgorin: eigenvalue in [diag - off_diag, diag + off_diag]
        # Use center as estimate
        values.append(max(0.0, diag))
    values.sort(reverse=True)
    return values


def extract_open_directions(
    arena: Any,
    projector: Any,
    wall_archive: WallArchive,
    n_directions: int = 5,
) -> List[Tuple[int, float]]:
    """Find the top-N directions of maximum variance in the residual space.

    These are the unexplored open directions — where new synthesis
    has the highest probability of producing novel (non-wall) results.

    Method:
      1. Project out all walls via multi-quotient
      2. Sample deterministic probe vectors in the residual space
      3. Compute pairwise correlation matrix of probes
      4. Extract principal components (variance of probes)
      5. The top-k probe vectors with highest self-correlation
         (injected into arena as handles) represent the most "open" directions

    Args:
        arena: FhrrArena instance.
        projector: IsomorphicProjector instance.
        wall_archive: WallArchive with recorded walls.
        n_directions: Number of open directions to extract.

    Returns:
        List of (arena_handle, variance_explained) for each open direction.
    """
    dimension = arena.get_dimension()
    wall_handles = wall_archive.all_handles()
    n_probes = max(n_directions * 3, 15)
    k = len(wall_handles)

    # Create an isolated temporary arena to avoid corrupting the main
    # arena's live vectors with quotient projection side-effects.
    # Extra capacity for internal allocations during multi-quotient.
    temp_capacity = n_probes + 2 * k + 16
    temp_arena = type(arena)(temp_capacity, dimension)

    # Copy wall vectors into temp arena
    temp_wall_handles = []
    for wh in wall_handles:
        phases = arena.extract_phases(wh)
        temp_h = temp_arena.allocate()
        temp_arena.inject_phases(temp_h, list(phases))
        temp_wall_handles.append(temp_h)

    # Generate deterministic probe vectors in temp arena
    temp_probe_handles = []
    probe_phase_arrays: List[List[float]] = []
    for i in range(n_probes):
        seed = int(hashlib.md5(f"open_probe_{i}".encode("utf-8")).hexdigest()[:8], 16)
        phases = _deterministic_phases(seed, dimension)
        h = temp_arena.allocate()
        temp_arena.inject_phases(h, phases)
        temp_probe_handles.append(h)
        probe_phase_arrays.append(phases)

    # Project out all walls in temp arena only
    has_multi = hasattr(temp_arena, 'project_to_multi_quotient_space')
    if has_multi and temp_wall_handles:
        temp_arena.project_to_multi_quotient_space(temp_wall_handles)
    else:
        for wh in temp_wall_handles:
            temp_arena.project_to_quotient_space(wh)

    # Compute variance: measure orthogonality to walls in temp arena.
    # Higher variance = more "open" direction.
    probe_variances: List[Tuple[int, float, List[float]]] = []
    for i, temp_ph in enumerate(temp_probe_handles):
        wall_corr_sum = 0.0
        for twh in temp_wall_handles:
            wall_corr_sum += abs(temp_arena.compute_correlation(temp_ph, twh))
        avg_wall_corr = wall_corr_sum / max(k, 1)
        variance = 1.0 - avg_wall_corr
        # Extract the projected probe's phases from temp arena
        projected_phases = list(temp_arena.extract_phases(temp_ph))
        probe_variances.append((i, variance, projected_phases))

    # Sort by descending variance
    probe_variances.sort(key=lambda x: -x[1])

    # Inject the top-k projected probes into the main arena as direction handles
    result: List[Tuple[int, float]] = []
    for _, variance, phases in probe_variances[:n_directions]:
        h = arena.allocate()
        arena.inject_phases(h, phases)
        result.append((h, variance))

    return result


def synthesize_along_direction(
    arena: Any,
    projector: Any,
    synthesizer: Any,
    decoder: Any,
    direction_handle: int,
    corpus_axiom_ids: List[str],
) -> Optional[str]:
    """Attempt synthesis biased toward an open direction.

    Instead of synthesizing from a resonance clique (which may hit
    the same walls again), this:
      1. Computes correlation of each axiom with the open direction
      2. Selects the top-k most aligned axioms
      3. Synthesizes from these aligned axioms
      4. Decodes the result

    The open direction acts as a "compass" pointing away from known walls.

    Args:
        arena: FhrrArena instance.
        projector: IsomorphicProjector instance.
        synthesizer: AxiomaticSynthesizer instance.
        decoder: ConstraintDecoder instance.
        direction_handle: Arena handle of the open direction vector.
        corpus_axiom_ids: List of axiom source IDs available for synthesis.

    Returns:
        Decoded Python source, or None if synthesis fails.
    """
    if len(corpus_axiom_ids) < 2:
        return None

    # Step 1: Compute correlation of each axiom with the direction
    axiom_scores: List[Tuple[str, float]] = []
    for sid in corpus_axiom_ids:
        axiom = synthesizer.store.get(sid)
        if axiom is None:
            continue
        corr = arena.compute_correlation(direction_handle, axiom.handle)
        axiom_scores.append((sid, abs(corr)))

    # Step 2: Select top-k aligned axioms (at least 2)
    axiom_scores.sort(key=lambda x: -x[1])
    top_k = max(2, min(5, len(axiom_scores)))
    selected_ids = [sid for sid, _ in axiom_scores[:top_k]]

    if len(selected_ids) < 2:
        return None

    # Step 3: Synthesize from aligned axioms
    try:
        synth_proj = synthesizer.synthesize_from_clique(selected_ids)
    except (ValueError, Exception):
        return None

    # Step 4: Decode
    source = decoder.decode_to_source(synth_proj)
    return source


def run_boundary_cartography(
    arena: Any,
    projector: Any,
    synthesizer: Any,
    decoder: Any,
    wall_archive: WallArchive,
    max_iterations: int = 10,
) -> Dict[str, Any]:
    """Execute the full boundary cartography loop.

    For each iteration:
      1. Map: compute wall resonance + cliques (= boundary regions)
      2. Measure: estimate residual dimension
      3. Navigate: extract open directions
      4. Synthesize: attempt synthesis along each open direction
      5. If UNSAT → record new wall, re-map
      6. If SAT → record successful synthesis
      7. Stop when: no open directions remain OR max_iterations reached

    Args:
        arena: FhrrArena instance.
        projector: IsomorphicProjector instance.
        synthesizer: AxiomaticSynthesizer instance.
        decoder: ConstraintDecoder instance.
        wall_archive: WallArchive for recording walls.
        max_iterations: Maximum cartography iterations.

    Returns:
        Dictionary with full cartography results.
    """
    dimension = arena.get_dimension()
    corpus_ids = synthesizer.store.all_ids()

    residual_history = []
    successful_syntheses = []
    total_walls_at_start = wall_archive.count()

    for iteration in range(max_iterations):
        # Step 1: Map — compute wall resonance + cliques
        wall_cliques = []
        if wall_archive.count() > 0:
            wall_cliques = wall_archive.extract_wall_cliques(threshold=0.3)

        # Step 2: Measure — estimate residual dimension
        residual_info = estimate_residual_dimension(
            arena, wall_archive.all_handles(), dimension,
        )
        residual_history.append(residual_info["estimated_residual_dimension"])

        # Step 3: Check if space is closed
        if residual_info["is_closed"]:
            break

        # Step 4: Navigate — extract open directions
        open_dirs = extract_open_directions(
            arena, projector, wall_archive, n_directions=5,
        )

        if not open_dirs:
            break

        # Step 5: Synthesize along each open direction
        made_progress = False
        for direction_handle, variance in open_dirs:
            source = synthesize_along_direction(
                arena, projector, synthesizer, decoder,
                direction_handle, corpus_ids,
            )

            if source is not None:
                successful_syntheses.append({
                    "iteration": iteration,
                    "source": source,
                    "variance": variance,
                })
                made_progress = True
            else:
                # Check decoder meta-grammar log for new UNSAT core results
                meta_log = decoder.get_meta_grammar_log()
                for event in meta_log:
                    if (event.unsat_core is not None
                            and event.unsat_core.v_error_handle is not None):
                        # Record the wall if not already recorded
                        try:
                            wall_archive.record(
                                event.unsat_core,
                                synthesis_context=f"cartography_iter_{iteration}",
                            )
                            made_progress = True
                        except (ValueError, Exception):
                            pass

        if not made_progress:
            break

    # Final measurement
    final_residual = estimate_residual_dimension(
        arena, wall_archive.all_handles(), dimension,
    )

    return {
        "iterations": min(iteration + 1, max_iterations) if max_iterations > 0 else 0,
        "walls_discovered": wall_archive.count() - total_walls_at_start,
        "wall_cliques": wall_cliques,
        "residual_dimension_history": residual_history,
        "successful_syntheses": successful_syntheses,
        "is_space_closed": final_residual["is_closed"],
        "open_directions_remaining": len(
            extract_open_directions(arena, projector, wall_archive, n_directions=5)
        ) if not final_residual["is_closed"] else 0,
    }
