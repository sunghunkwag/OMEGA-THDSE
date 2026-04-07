"""Resonance analysis and structural pattern extraction.

Computes pairwise resonance matrices, hierarchical clustering,
outlier detection, clique interpretation, and cross-project bridging.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer
from src.decoder.constraint_decoder import (
    ConstraintDecoder,
    DecodedConstraints,
    AtomKind,
)
from src.projection.isomorphic_projector import LayeredProjection


class CliqueFingerpint:
    """Human-readable structural fingerprint for a clique."""

    def __init__(
        self,
        clique: List[str],
        node_types: List[str],
        cfg_summary: str,
        data_dep_summary: str,
        description: str,
    ) -> None:
        self.clique = clique
        self.node_types = node_types
        self.cfg_summary = cfg_summary
        self.data_dep_summary = data_dep_summary
        self.description = description

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "clique": self.clique,
            "node_types": self.node_types,
            "cfg_summary": self.cfg_summary,
            "data_dep_summary": self.data_dep_summary,
            "description": self.description,
        }


class ResonanceAnalyzer:
    """Performs structural analysis on a populated AxiomaticSynthesizer.

    Assumes axioms have already been ingested and resonance computed.
    """

    def __init__(
        self,
        synthesizer: AxiomaticSynthesizer,
        decoder: Optional[ConstraintDecoder] = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            synthesizer: AxiomaticSynthesizer with ingested axioms.
            decoder: Optional ConstraintDecoder for structural fingerprinting.
                     If None, fingerprinting will use simplified heuristics.
        """
        self.synth = synthesizer
        self.decoder = decoder
        self._resonance_dict: Dict[Tuple[str, str], float] = {}
        self._ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None

    def compute_resonance(self) -> np.ndarray:
        """Compute the full pairwise resonance matrix.

        Returns:
            2D numpy array of shape (n, n) with resonance scores.
        """
        self._resonance_dict = self.synth.compute_resonance()
        self._ids = self.synth.store.all_ids()
        n = len(self._ids)
        mat = np.zeros((n, n), dtype=np.float64)
        id_to_idx = {sid: i for i, sid in enumerate(self._ids)}

        for (a, b), val in self._resonance_dict.items():
            i, j = id_to_idx.get(a), id_to_idx.get(b)
            if i is not None and j is not None:
                mat[i, j] = val
                mat[j, i] = val

        # Self-correlation = 1.0
        for i in range(n):
            mat[i, i] = 1.0

        self._matrix = mat
        return mat

    def get_ids(self) -> List[str]:
        """Return the list of axiom IDs in matrix order."""
        return list(self._ids)

    def resonance_stats(self) -> Dict[str, float]:
        """Compute basic statistics over the resonance matrix.

        Returns:
            Dict with 'mean', 'std', 'min', 'max' of off-diagonal entries.
        """
        if self._matrix is None:
            self.compute_resonance()
        assert self._matrix is not None
        n = len(self._ids)
        if n < 2:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        # Extract upper triangle (off-diagonal)
        triu_idx = np.triu_indices(n, k=1)
        vals = self._matrix[triu_idx]

        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    def hierarchical_clustering(self) -> np.ndarray:
        """Perform agglomerative hierarchical clustering (Ward linkage).

        Converts the resonance matrix to a distance matrix (1 - resonance)
        and applies Ward's method.

        Returns:
            Linkage matrix suitable for scipy.cluster.hierarchy.dendrogram().
        """
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        if self._matrix is None:
            self.compute_resonance()
        assert self._matrix is not None

        n = len(self._ids)
        if n < 2:
            return np.array([])

        # Convert resonance to distance: d = 1 - resonance
        # Clip to [0, 2] to handle numerical noise
        dist_mat = np.clip(1.0 - self._matrix, 0.0, 2.0)
        # Zero out diagonal
        np.fill_diagonal(dist_mat, 0.0)
        # Make symmetric
        dist_mat = (dist_mat + dist_mat.T) / 2.0

        condensed = squareform(dist_mat)
        return linkage(condensed, method="ward")

    def find_outliers(self, threshold_sigmas: float = 2.0) -> List[Tuple[str, float]]:
        """Identify structurally unique axioms (outliers).

        An outlier has mean resonance with all others falling below
        (global_mean - threshold_sigmas * global_std).

        Args:
            threshold_sigmas: Number of standard deviations below mean.

        Returns:
            List of (source_id, mean_resonance) sorted by ascending resonance.
        """
        if self._matrix is None:
            self.compute_resonance()
        assert self._matrix is not None

        n = len(self._ids)
        if n < 2:
            return []

        # Mean resonance per axiom (excluding self)
        per_axiom_mean = []
        for i in range(n):
            others = [self._matrix[i, j] for j in range(n) if j != i]
            per_axiom_mean.append(float(np.mean(others)))

        global_mean = float(np.mean(per_axiom_mean))
        global_std = float(np.std(per_axiom_mean))
        cutoff = global_mean - threshold_sigmas * global_std

        outliers = [
            (self._ids[i], per_axiom_mean[i])
            for i in range(n)
            if per_axiom_mean[i] < cutoff
        ]

        return sorted(outliers, key=lambda x: x[1])

    def top_resonant_pairs(self, k: int = 10) -> List[Tuple[str, str, float]]:
        """Find the top-k most resonant pairs.

        Args:
            k: Number of pairs to return.

        Returns:
            List of (id_a, id_b, resonance) sorted by descending resonance.
        """
        if self._matrix is None:
            self.compute_resonance()
        assert self._matrix is not None

        n = len(self._ids)
        pairs: List[Tuple[str, str, float]] = []

        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((self._ids[i], self._ids[j], float(self._matrix[i, j])))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:k]

    def bottom_resonant_axioms(self, k: int = 10) -> List[Tuple[str, float]]:
        """Find the k axioms with lowest mean resonance.

        Args:
            k: Number of axioms to return.

        Returns:
            List of (source_id, mean_resonance) sorted ascending.
        """
        if self._matrix is None:
            self.compute_resonance()
        assert self._matrix is not None

        n = len(self._ids)
        if n < 2:
            return []

        means = []
        for i in range(n):
            others = [self._matrix[i, j] for j in range(n) if j != i]
            means.append((self._ids[i], float(np.mean(others))))

        means.sort(key=lambda x: x[1])
        return means[:k]

    def _describe_constraints(self, constraints: DecodedConstraints) -> str:
        """Generate a human-readable description from decoded constraints."""
        parts: List[str] = []

        node_set = set(constraints.active_node_types)

        # Branching
        branch_types = {"If", "IfExp"}
        if branch_types & node_set:
            n_branches = len(constraints.cfg_branches)
            if n_branches > 2:
                parts.append("heavy branching")
            elif n_branches > 0:
                parts.append("light branching")

        # Loops
        loop_types = {"For", "While", "AsyncFor"}
        if loop_types & node_set:
            n_loops = len(constraints.cfg_loops)
            if n_loops > 1:
                parts.append("multiple loops")
            else:
                parts.append("single loop")

        # Function definitions
        if "FunctionDef" in node_set or "AsyncFunctionDef" in node_set:
            parts.append("function definitions")

        # Class definitions
        if "ClassDef" in node_set:
            parts.append("class definitions")

        # Data dependencies
        n_deps = len(constraints.data_deps)
        if n_deps > 5:
            parts.append("tight def-use chains")
        elif n_deps > 0:
            parts.append("sparse data flow")
        else:
            parts.append("no data dependencies")

        # CFG structure
        n_seq = len(constraints.cfg_sequences)
        if n_seq > 5:
            parts.append("linear pipeline")
        elif n_seq == 0:
            parts.append("minimal control flow")

        # Exception handling
        if "Try" in node_set or "ExceptHandler" in node_set:
            parts.append("exception handling")

        # Comprehensions
        comp_types = {"ListComp", "SetComp", "DictComp", "GeneratorExp"}
        if comp_types & node_set:
            parts.append("comprehensions")

        if not parts:
            parts.append("minimal structure")

        return " + ".join(parts)

    def interpret_cliques(
        self,
        min_size: int = 3,
    ) -> List[CliqueFingerpint]:
        """Extract cliques and generate structural fingerprints.

        For each clique, computes the centroid vector (bundle of members)
        and probes it to identify activated structural atoms.

        Args:
            min_size: Minimum clique size to include.

        Returns:
            List of CliqueFingerpint objects.
        """
        cliques = self.synth.extract_cliques(min_size=min_size)
        fingerprints: List[CliqueFingerpint] = []

        for clique in cliques:
            # Compute centroid: bundle all member handles
            member_handles = []
            for sid in clique:
                axiom = self.synth.store.get(sid)
                if axiom is not None:
                    member_handles.append(axiom.handle)

            if len(member_handles) < 2:
                continue

            # Bundle into centroid
            arena = self.synth.arena
            centroid_h = arena.allocate()
            arena.bundle(member_handles, centroid_h)

            # Probe centroid for structural atoms
            if self.decoder is not None:
                try:
                    constraints = self.decoder.probe(centroid_h)
                    description = self._describe_constraints(constraints)
                    node_types = list(constraints.active_node_types)
                    cfg_summary = (
                        f"{len(constraints.cfg_sequences)} sequences, "
                        f"{len(constraints.cfg_branches)} branches, "
                        f"{len(constraints.cfg_loops)} loops"
                    )
                    data_summary = f"{len(constraints.data_deps)} def-use chains"
                except Exception:
                    node_types = []
                    cfg_summary = "probe failed"
                    data_summary = "probe failed"
                    description = f"clique of {len(clique)} members (probe failed)"
            else:
                node_types = []
                cfg_summary = "no decoder"
                data_summary = "no decoder"
                description = f"clique of {len(clique)} members"

            fingerprints.append(CliqueFingerpint(
                clique=clique,
                node_types=node_types,
                cfg_summary=cfg_summary,
                data_dep_summary=data_summary,
                description=description,
            ))

        return fingerprints

    def cross_project_bridges(
        self,
        threshold: float = 0.8,
    ) -> List[Tuple[str, str, float]]:
        """Find cross-project structural twins.

        Identifies pairs of files from different projects with high resonance.
        Project is inferred from the first path component of the source ID.

        Args:
            threshold: Minimum resonance for a pair to be considered a bridge.

        Returns:
            List of (id_a, id_b, resonance) sorted by descending resonance.
        """
        if self._matrix is None:
            self.compute_resonance()
        assert self._matrix is not None

        def get_project(sid: str) -> str:
            """Extract project name from source ID (first path component)."""
            parts = sid.split("/")
            return parts[0] if len(parts) > 1 else ""

        n = len(self._ids)
        bridges: List[Tuple[str, str, float]] = []

        for i in range(n):
            proj_i = get_project(self._ids[i])
            for j in range(i + 1, n):
                proj_j = get_project(self._ids[j])
                if proj_i == proj_j:
                    continue
                if not proj_i or not proj_j:
                    continue
                val = float(self._matrix[i, j])
                if val >= threshold:
                    bridges.append((self._ids[i], self._ids[j], val))

        bridges.sort(key=lambda x: x[2], reverse=True)
        return bridges
