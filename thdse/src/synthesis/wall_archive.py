"""Wall Archive — Persistent store of Contradiction Vectors from UNSAT events.

Collects every V_error produced by the ConstraintDecoder on UNSAT, indexes
them for resonance analysis, and supports clique extraction over boundary
walls (identical to the Bron-Kerbosch algorithm used by AxiomaticSynthesizer).

Design decisions:
  - Stores BOTH arena handles AND phase arrays. Arena handles are ephemeral
    (lost on arena reset), but phase arrays can be re-injected into a fresh
    arena. This enables persistence across sessions via serialize/deserialize.
  - Walls are never deleted. They transition between statuses:
    "active" (proven UNSAT boundary), "resolved" (previously UNSAT, now SAT
    after design space change), or "orphaned" (atoms no longer in vocabulary).
  - Revalidation is deterministic (Z3 re-solve) and triggered by design
    space changes (dimension expansion, vocabulary update), not by time.
"""

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import hdc_core as _hdc_core  # noqa: F401
    _HAS_HDC_CORE = True
except ImportError:
    _HAS_HDC_CORE = False


@dataclass
class WallRecord:
    """A single recorded design space boundary."""

    v_error_handle: int
    v_error_phases: List[float]
    unsat_core_labels: List[str]
    core_atom_handles: List[int]
    synthesis_context: str
    timestamp_index: int
    status: str = "active"  # "active" | "resolved" | "orphaned"


class WallArchive:
    """Persistent store of Contradiction Vectors from UNSAT events."""

    def __init__(self, arena: Any, dimension: int):
        """Initialize the wall archive.

        Args:
            arena: FhrrArena instance for vector operations.
            dimension: Hypervector dimension.
        """
        self.arena = arena
        self.dimension = dimension
        self._walls: List[WallRecord] = []
        self._next_index: int = 0

    def record(self, unsat_core_result: Any, synthesis_context: str) -> WallRecord:
        """Record a new wall. Extract phases from arena for persistence.

        Args:
            unsat_core_result: UnsatCoreResult from the constraint decoder.
            synthesis_context: Which clique/synthesis attempt triggered this.

        Returns:
            The newly created WallRecord.
        """
        v_handle = unsat_core_result.v_error_handle
        if v_handle is None:
            raise ValueError("UnsatCoreResult has no v_error_handle")

        # Extract phases for persistence
        v_phases = list(self.arena.extract_phases(v_handle))

        wall = WallRecord(
            v_error_handle=v_handle,
            v_error_phases=v_phases,
            unsat_core_labels=list(unsat_core_result.core_labels),
            core_atom_handles=list(unsat_core_result.core_atom_handles),
            synthesis_context=synthesis_context,
            timestamp_index=self._next_index,
            status="active",
        )
        self._walls.append(wall)
        self._next_index += 1
        return wall

    def count(self) -> int:
        """Number of walls recorded (all statuses)."""
        return len(self._walls)

    def all_handles(self) -> List[int]:
        """All V_error arena handles for active walls only."""
        return [w.v_error_handle for w in self._walls if w.status == "active"]

    def all_phase_arrays(self) -> List[List[float]]:
        """All V_error phase arrays (for re-injection or analysis)."""
        return [w.v_error_phases for w in self._walls]

    def get_walls(self) -> List[WallRecord]:
        """Return all wall records (all statuses)."""
        return list(self._walls)

    def compute_wall_resonance(self) -> Dict[Tuple[int, int], float]:
        """Pairwise correlation between active V_errors.

        Same math as AxiomStore resonance, but over walls not code.
        Only operates on walls with status == "active".

        Uses batch correlate_matrix when available to reduce Python→Rust
        round trips from n*(n+1)/2 to 1 call + n self-correlation calls.

        Returns:
            Dictionary mapping (active_wall_idx_i, active_wall_idx_j)
            to correlation. Indices are positions within the active subset.
        """
        active = [w for w in self._walls if w.status == "active"]
        matrix: Dict[Tuple[int, int], float] = {}
        n = len(active)
        if n == 0:
            return matrix

        handles = [w.v_error_handle for w in active]
        has_batch = hasattr(self.arena, 'correlate_matrix')

        if has_batch and n >= 2:
            # Self-correlations
            for i in range(n):
                corr = self.arena.compute_correlation(handles[i], handles[i])
                matrix[(i, i)] = corr
            # Upper triangle via single batch call
            flat = self.arena.correlate_matrix(handles)
            k = 0
            for i in range(n):
                for j in range(i + 1, n):
                    matrix[(i, j)] = flat[k]
                    matrix[(j, i)] = flat[k]
                    k += 1
        else:
            # Fallback: individual calls
            for i in range(n):
                for j in range(i, n):
                    corr = self.arena.compute_correlation(
                        active[i].v_error_handle,
                        active[j].v_error_handle,
                    )
                    matrix[(i, j)] = corr
                    matrix[(j, i)] = corr

        return matrix

    def extract_wall_cliques(self, threshold: float = 0.3) -> List[List[int]]:
        """Find cliques of similar walls (= same boundary region).

        Uses Bron-Kerbosch identical to AxiomaticSynthesizer.
        Only operates on active walls.

        Args:
            threshold: Minimum correlation for wall adjacency.

        Returns:
            List of cliques, where each clique is a list of active wall indices.
        """
        active = [w for w in self._walls if w.status == "active"]
        n = len(active)
        if n == 0:
            return []

        resonance = self.compute_wall_resonance()

        # Build adjacency
        adj: Dict[int, set] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i != j and resonance.get((i, j), 0.0) > threshold:
                    adj[i].add(j)

        cliques: List[List[int]] = []
        self._bron_kerbosch(set(), set(range(n)), set(), adj, cliques)
        return sorted(
            [c for c in cliques if len(c) >= 1],
            key=lambda c: (-len(c), c),
        )

    def _bron_kerbosch(
        self,
        R: set,
        P: set,
        X: set,
        adj: Dict[int, set],
        results: List[List[int]],
    ) -> None:
        """Bron-Kerbosch maximal clique enumeration."""
        if not P and not X:
            results.append(sorted(R))
            return
        pivot_candidates = sorted(P | X)
        if not pivot_candidates:
            return
        pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
        candidates = sorted(P - adj[pivot])
        for v in candidates:
            self._bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, results)
            P = P - {v}
            X = X | {v}

    def revalidate_walls(
        self,
        decoder: Any,
        projector: Any,
    ) -> Dict[str, Any]:
        """Re-check all archived walls against the current design space.

        Called after any event that changes the design space:
          - dimension expansion (d → 2d)
          - sub-tree vocabulary update (new corpus ingested)
          - arena reset and re-injection

        For each wall:
          1. Re-probe the original UNSAT core atoms against current vocabulary
          2. Re-encode as Z3 constraints
          3. Re-solve:
             - Still UNSAT → wall is valid, keep it
             - Now SAT → wall is resolved, mark as 'resolved'
             - Atoms no longer in vocabulary → wall is orphaned, mark as 'orphaned'

        Resolved and orphaned walls are moved to inactive status
        (not deleted — they are historical records) and excluded from
        the active boundary map.

        Args:
            decoder: ConstraintDecoder instance for re-probing/re-solving.
            projector: IsomorphicProjector instance.

        Returns:
            Dictionary with revalidation results.
        """
        import z3

        total_checked = 0
        still_valid = 0
        resolved = 0
        orphaned = 0

        for wall in self._walls:
            if wall.status != "active":
                continue
            total_checked += 1

            # Check if the core atom labels still exist in the decoder vocabulary
            vocab = decoder._atom_vocab
            labels_found = [
                label for label in wall.unsat_core_labels
                if label in vocab
            ]

            if not labels_found:
                # Atoms no longer in vocabulary → orphaned
                wall.status = "orphaned"
                orphaned += 1
                continue

            # Re-probe: use the decoder to probe and re-encode
            # Build constraints from the existing core labels
            from src.decoder.constraint_decoder import DecodedConstraints

            constraints = DecodedConstraints()
            for label in labels_found:
                atom = vocab[label]
                # Reconstruct constraint type from label
                if label.startswith("atom:type:"):
                    ntype = label.split(":")[-1]
                    constraints.active_node_types.append(ntype)
                elif label.startswith("atom:cfg_seq:"):
                    pair = label.split(":")[-1]
                    parts = pair.split("->")
                    if len(parts) == 2:
                        constraints.cfg_sequences.append((parts[0], parts[1]))
                elif label.startswith("atom:cfg_branch:"):
                    pair = label.split(":")[-1]
                    parts = pair.split("->")
                    if len(parts) == 2:
                        constraints.cfg_branches.append((parts[0], parts[1], "Pass"))
                elif label.startswith("atom:cfg_loop:"):
                    pair = label.split(":")[-1]
                    parts = pair.split("->")
                    if len(parts) == 2:
                        constraints.cfg_loops.append((parts[0], parts[1]))
                elif label.startswith("atom:data_dep:"):
                    pair = label.split(":")[-1]
                    parts = pair.split("->")
                    if len(parts) == 2:
                        constraints.data_deps.append((parts[0], parts[1]))

            if not constraints.active_node_types:
                # No node type constraints recoverable → orphaned
                wall.status = "orphaned"
                orphaned += 1
                continue

            # Re-encode and re-solve
            try:
                solver, vars_map = decoder.encode_smt(constraints)
                result = solver.check()

                if result == z3.unsat:
                    # Still UNSAT → wall remains valid
                    still_valid += 1
                elif result == z3.sat:
                    # Now SAT → wall is resolved
                    wall.status = "resolved"
                    resolved += 1
                else:
                    # Unknown → keep as active (conservative)
                    still_valid += 1
            except Exception:
                # If re-solve fails, keep wall as active (conservative)
                still_valid += 1

        return {
            "total_checked": total_checked,
            "still_valid": still_valid,
            "resolved": resolved,
            "orphaned": orphaned,
        }

    def serialize(self, filepath: str) -> None:
        """Save wall archive to disk (pickle phase arrays + metadata).

        Args:
            filepath: Path to write the serialized archive.
        """
        data = {
            "dimension": self.dimension,
            "next_index": self._next_index,
            "walls": [
                {
                    "v_error_phases": w.v_error_phases,
                    "unsat_core_labels": w.unsat_core_labels,
                    "core_atom_handles": w.core_atom_handles,
                    "synthesis_context": w.synthesis_context,
                    "timestamp_index": w.timestamp_index,
                    "status": w.status,
                }
                for w in self._walls
            ],
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def deserialize(cls, filepath: str, arena: Any) -> 'WallArchive':
        """Load wall archive from disk and re-inject into arena.

        Args:
            filepath: Path to the serialized archive.
            arena: FhrrArena to re-inject phase arrays into.

        Returns:
            Restored WallArchive instance.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        dimension = data["dimension"]
        archive = cls(arena, dimension)
        archive._next_index = data["next_index"]

        for wall_data in data["walls"]:
            # Re-inject phases into arena
            handle = arena.allocate()
            arena.inject_phases(handle, wall_data["v_error_phases"])

            wall = WallRecord(
                v_error_handle=handle,
                v_error_phases=wall_data["v_error_phases"],
                unsat_core_labels=wall_data["unsat_core_labels"],
                core_atom_handles=wall_data["core_atom_handles"],
                synthesis_context=wall_data["synthesis_context"],
                timestamp_index=wall_data["timestamp_index"],
                status=wall_data.get("status", "active"),
            )
            archive._walls.append(wall)

        return archive
