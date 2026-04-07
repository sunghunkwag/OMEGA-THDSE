"""
Refactoring Detector — Scans axiom pairs in the corpus to identify
structural duplicates and interface unification candidates.

Detection rules:
  1. Structural Duplicate:
     sim_ast > 0.7 AND sim_cfg > 0.7 AND files in different directories
     → "Consider extracting common code"

  2. Interface Unification Candidate:
     sim_data > 0.8 AND sim_ast < 0.3
     → "Same computation, different expression — unify interface"

Output: structured JSON compatible with the existing report system.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import hdc_core as _hdc_core
except ImportError:
    _hdc_core = None

from src.projection.isomorphic_projector import IsomorphicProjector, LayeredProjection
from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer
from src.analysis.structural_diff import StructuralDiffEngine, LayerSimilarity


@dataclass
class RefactoringCandidate:
    """A detected refactoring opportunity.

    Attributes:
        kind: Type of refactoring ("structural_duplicate" or "interface_unification").
        file_a: Source ID of the first file.
        file_b: Source ID of the second file.
        sim_ast: AST layer similarity.
        sim_cfg: CFG layer similarity.
        sim_data: Data-dep layer similarity.
        recommendation: Human-readable refactoring recommendation.
    """
    kind: str
    file_a: str
    file_b: str
    sim_ast: float
    sim_cfg: Optional[float]
    sim_data: Optional[float]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "kind": self.kind,
            "file_a": self.file_a,
            "file_b": self.file_b,
            "sim_ast": round(self.sim_ast, 6),
            "sim_cfg": round(self.sim_cfg, 6) if self.sim_cfg is not None else None,
            "sim_data": round(self.sim_data, 6) if self.sim_data is not None else None,
            "recommendation": self.recommendation,
        }


def _get_directory(source_id: str) -> str:
    """Extract the directory component from a source ID.

    Args:
        source_id: Source identifier (e.g., "project/subdir/file.py").

    Returns:
        Directory portion (e.g., "project/subdir"), or "" if no separator.
    """
    parts = source_id.rsplit("/", 1)
    return parts[0] if len(parts) > 1 else ""


class RefactoringDetector:
    """Scans axiom pairs to detect refactoring opportunities.

    Usage:
        detector = RefactoringDetector(arena, projector, synthesizer)
        candidates = detector.scan()
        report = detector.to_json(candidates)
    """

    def __init__(
        self,
        arena: Any,
        projector: IsomorphicProjector,
        synthesizer: AxiomaticSynthesizer,
        dup_ast_threshold: float = 0.7,
        dup_cfg_threshold: float = 0.7,
        unify_data_threshold: float = 0.8,
        unify_ast_ceiling: float = 0.3,
    ) -> None:
        """Initialize the refactoring detector.

        Args:
            arena: FHRR arena instance.
            projector: IsomorphicProjector.
            synthesizer: AxiomaticSynthesizer with ingested axioms.
            dup_ast_threshold: Min AST similarity for duplicate detection.
            dup_cfg_threshold: Min CFG similarity for duplicate detection.
            unify_data_threshold: Min data-dep similarity for unification.
            unify_ast_ceiling: Max AST similarity for unification candidate.
        """
        self.arena = arena
        self.projector = projector
        self.synthesizer = synthesizer
        self.diff_engine = StructuralDiffEngine(arena, projector)
        self.dup_ast_threshold = dup_ast_threshold
        self.dup_cfg_threshold = dup_cfg_threshold
        self.unify_data_threshold = unify_data_threshold
        self.unify_ast_ceiling = unify_ast_ceiling

    def scan(self) -> List[RefactoringCandidate]:
        """Scan all axiom pairs for refactoring opportunities.

        Uses per-layer batch correlation to avoid O(N²) Python→Rust round
        trips.  Each layer's full upper-triangle correlation matrix is
        computed in a single Rust FFI call, then thresholded via list
        comprehension.  Only surviving candidate pairs are evaluated
        against the secondary rule conditions.

        Call-count reduction:
          Old: 4 × N*(N-1)/2 individual compute_correlation calls
          New: 2-3 correlate_matrix calls + a small number of individual
               calls for secondary checks on surviving pairs

        Returns:
            List of RefactoringCandidate instances, sorted by relevance.
        """
        candidates: List[RefactoringCandidate] = []
        ids = self.synthesizer.store.all_ids()
        n = len(ids)
        if n < 2:
            return candidates

        has_batch = hasattr(self.arena, 'correlate_matrix')
        if not has_batch:
            return self._scan_fallback(ids)

        # ── Collect per-layer handles with None-layer tracking ──────
        axioms = []
        for sid in ids:
            ax = self.synthesizer.store.get(sid)
            if ax is not None:
                axioms.append((sid, ax))
        n = len(axioms)
        if n < 2:
            return candidates

        # AST handles (always present)
        ast_handles = [ax.projection.ast_handle for _, ax in axioms]
        ast_flat = self.arena.correlate_matrix(ast_handles)

        # CFG handles — track which axioms have CFG
        cfg_global_idx = []  # positions in axioms[] that have cfg
        cfg_handles = []
        for idx, (_, ax) in enumerate(axioms):
            if ax.projection.cfg_handle is not None:
                cfg_global_idx.append(idx)
                cfg_handles.append(ax.projection.cfg_handle)

        # Data handles — track which axioms have data
        data_global_idx = []
        data_handles = []
        for idx, (_, ax) in enumerate(axioms):
            if ax.projection.data_handle is not None:
                data_global_idx.append(idx)
                data_handles.append(ax.projection.data_handle)

        # Pre-compute directory for cross-dir check
        dirs = [_get_directory(sid) for sid, _ in axioms]

        # ── Helper: unpack flat upper-triangle index ────────────────
        def _flat_idx(i: int, j: int, size: int) -> int:
            """Index into flat upper-triangle array for pair (i, j), i < j."""
            return i * size - i * (i + 1) // 2 + j - i - 1

        # ── Rule 1: Structural Duplicate ────────────────────────────
        # Requires: |sim_ast| > 0.7 AND |sim_cfg| > 0.7 AND cross-dir
        #
        # Step 4b-1: filter AST matrix for pairs above dup_ast_threshold
        ast_candidates = []
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(ast_flat[k]) > self.dup_ast_threshold:
                    ast_candidates.append((i, j, ast_flat[k]))
                k += 1

        # Step 4b-2: for surviving pairs, check CFG
        # Build lookup: global_idx → cfg_local_idx
        cfg_local_map = {g: loc for loc, g in enumerate(cfg_global_idx)}

        # Compute CFG matrix only if there are CFG handles and candidates
        cfg_flat = None
        if len(cfg_handles) >= 2:
            cfg_flat = self.arena.correlate_matrix(cfg_handles)

        for i, j, sim_ast in ast_candidates:
            # Both must have CFG handles
            if i not in cfg_local_map or j not in cfg_local_map:
                continue
            # Cross-directory check
            if not (dirs[i] and dirs[j] and dirs[i] != dirs[j]):
                continue
            # Look up CFG correlation
            ci = cfg_local_map[i]
            cj = cfg_local_map[j]
            if ci > cj:
                ci, cj = cj, ci
            cfg_corr = cfg_flat[_flat_idx(ci, cj, len(cfg_handles))]
            if abs(cfg_corr) <= self.dup_cfg_threshold:
                continue

            # Get data similarity for the candidate record (optional)
            sim_data = None
            data_local_map_i = None
            data_local_map_j = None
            for loc, g in enumerate(data_global_idx):
                if g == i:
                    data_local_map_i = loc
                if g == j:
                    data_local_map_j = loc

            if data_local_map_i is not None and data_local_map_j is not None:
                di, dj = data_local_map_i, data_local_map_j
                if di > dj:
                    di, dj = dj, di
                # Compute individual data correlation for this pair
                sim_data = self.arena.compute_correlation(
                    data_handles[di], data_handles[dj])

            id_a = axioms[i][0]
            id_b = axioms[j][0]
            candidates.append(RefactoringCandidate(
                kind="structural_duplicate",
                file_a=id_a,
                file_b=id_b,
                sim_ast=sim_ast,
                sim_cfg=cfg_corr,
                sim_data=sim_data,
                recommendation=(
                    f"Structural duplicate detected: AST={sim_ast:.2f}, "
                    f"CFG={cfg_corr:.2f}. "
                    f"Consider extracting common code into a shared module."
                ),
            ))

        # ── Rule 2: Interface Unification ───────────────────────────
        # Requires: |sim_data| > 0.8 AND |sim_ast| < 0.3
        #
        # Step 4c: compute data matrix, filter, check AST ceiling
        data_local_map = {g: loc for loc, g in enumerate(data_global_idx)}

        if len(data_handles) >= 2:
            data_flat = self.arena.correlate_matrix(data_handles)
            dk = 0
            nd = len(data_handles)
            for di in range(nd):
                for dj in range(di + 1, nd):
                    data_corr = data_flat[dk]
                    dk += 1
                    if abs(data_corr) <= self.unify_data_threshold:
                        continue
                    # Map back to global indices
                    gi = data_global_idx[di]
                    gj = data_global_idx[dj]
                    # Check AST ceiling (reuse ast_flat)
                    ai, aj = gi, gj
                    if ai > aj:
                        ai, aj = aj, ai
                    ast_corr = ast_flat[_flat_idx(ai, aj, n)]
                    if abs(ast_corr) >= self.unify_ast_ceiling:
                        continue

                    # Get CFG similarity for the candidate record (optional)
                    sim_cfg = None
                    if gi in cfg_local_map and gj in cfg_local_map:
                        ci = cfg_local_map[gi]
                        cj = cfg_local_map[gj]
                        if ci > cj:
                            ci, cj = cj, ci
                        if cfg_flat is not None:
                            sim_cfg = cfg_flat[_flat_idx(ci, cj, len(cfg_handles))]

                    id_a = axioms[gi][0]
                    id_b = axioms[gj][0]
                    candidates.append(RefactoringCandidate(
                        kind="interface_unification",
                        file_a=id_a,
                        file_b=id_b,
                        sim_ast=ast_corr,
                        sim_cfg=sim_cfg,
                        sim_data=data_corr,
                        recommendation=(
                            f"Same computation, different expression: "
                            f"Data={data_corr:.2f}, AST={ast_corr:.2f}. "
                            f"Candidate for interface unification."
                        ),
                    ))

        # Sort: structural duplicates first (higher combined sim), then unification
        candidates.sort(key=lambda c: (
            0 if c.kind == "structural_duplicate" else 1,
            -(abs(c.sim_ast) + abs(c.sim_cfg or 0) + abs(c.sim_data or 0)),
        ))

        return candidates

    def _scan_fallback(self, ids: List[str]) -> List[RefactoringCandidate]:
        """Original O(N²) scan for arenas without correlate_matrix."""
        candidates: List[RefactoringCandidate] = []
        for i, id_a in enumerate(ids):
            axiom_a = self.synthesizer.store.get(id_a)
            if axiom_a is None:
                continue
            for id_b in ids[i + 1:]:
                axiom_b = self.synthesizer.store.get(id_b)
                if axiom_b is None:
                    continue
                layer_sim = self.diff_engine.compare_layers(
                    axiom_a.projection, axiom_b.projection,
                )
                dir_a = _get_directory(id_a)
                dir_b = _get_directory(id_b)
                is_cross_dir = dir_a != dir_b and dir_a and dir_b
                if (is_cross_dir
                    and abs(layer_sim.sim_ast) > self.dup_ast_threshold
                    and layer_sim.sim_cfg is not None
                    and abs(layer_sim.sim_cfg) > self.dup_cfg_threshold):
                    candidates.append(RefactoringCandidate(
                        kind="structural_duplicate",
                        file_a=id_a,
                        file_b=id_b,
                        sim_ast=layer_sim.sim_ast,
                        sim_cfg=layer_sim.sim_cfg,
                        sim_data=layer_sim.sim_data,
                        recommendation=(
                            f"Structural duplicate detected: AST={layer_sim.sim_ast:.2f}, "
                            f"CFG={layer_sim.sim_cfg:.2f}. "
                            f"Consider extracting common code into a shared module."
                        ),
                    ))
                if (layer_sim.sim_data is not None
                    and abs(layer_sim.sim_data) > self.unify_data_threshold
                    and abs(layer_sim.sim_ast) < self.unify_ast_ceiling):
                    candidates.append(RefactoringCandidate(
                        kind="interface_unification",
                        file_a=id_a,
                        file_b=id_b,
                        sim_ast=layer_sim.sim_ast,
                        sim_cfg=layer_sim.sim_cfg,
                        sim_data=layer_sim.sim_data,
                        recommendation=(
                            f"Same computation, different expression: "
                            f"Data={layer_sim.sim_data:.2f}, AST={layer_sim.sim_ast:.2f}. "
                            f"Candidate for interface unification."
                        ),
                    ))
        candidates.sort(key=lambda c: (
            0 if c.kind == "structural_duplicate" else 1,
            -(abs(c.sim_ast) + abs(c.sim_cfg or 0) + abs(c.sim_data or 0)),
        ))
        return candidates

    def to_json(self, candidates: List[RefactoringCandidate]) -> Dict[str, Any]:
        """Serialize candidates to a JSON-compatible report structure.

        Args:
            candidates: List of detected refactoring candidates.

        Returns:
            Dictionary with structured report data.
        """
        duplicates = [c.to_dict() for c in candidates if c.kind == "structural_duplicate"]
        unifications = [c.to_dict() for c in candidates if c.kind == "interface_unification"]

        return {
            "refactoring_candidates": {
                "structural_duplicates": duplicates,
                "interface_unification_candidates": unifications,
                "total_candidates": len(candidates),
                "duplicate_count": len(duplicates),
                "unification_count": len(unifications),
            }
        }
