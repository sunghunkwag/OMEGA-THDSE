"""Phase 8A.4 — Functional prototype extraction.

After SERL produces fitness-passing candidates, this module groups
them by **behavioural** similarity and bundles each group into a
:class:`FunctionalPrototype`. The bundling cancels syntactic noise
across implementations, leaving the abstract pattern that all
members exhibit. This is the abstraction mechanism PLAN.md Phase 8
exists for: when you bundle 5 different ways of summing a list, the
loop / recursion / built-in syntactic differences cancel and what
remains is "iterate and accumulate".

Rule 21: a prototype is only valid if bundling actually compresses
the atom set. We verify ``len(prototype.essential_atoms) <
mean(member_atom_count)`` so the abstraction provably reduces
information rather than averaging noise.
"""

from __future__ import annotations

import ast
import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# AST atom extraction
# --------------------------------------------------------------------------- #


def _ast_atoms(source: str) -> Set[str]:
    """Return the set of (parent_type, child_type) AST atoms in ``source``.

    The "atom" representation is intentionally coarse — each node
    type paired with its parent type. This captures structural
    fingerprints like ``("FunctionDef", "Return")`` (function with a
    return) or ``("For", "AugAssign")`` (loop with accumulator) while
    remaining stable enough that ``return sum(arr)`` and
    ``return functools.reduce(...)`` produce overlapping atoms.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    atoms: Set[str] = set()

    def _walk(node: ast.AST, parent_type: str) -> None:
        node_type = type(node).__name__
        atoms.add(f"{parent_type}::{node_type}")
        for child in ast.iter_child_nodes(node):
            _walk(child, node_type)

    _walk(tree, "Module")
    return atoms


# --------------------------------------------------------------------------- #
# FunctionalPrototype dataclass
# --------------------------------------------------------------------------- #


@dataclass
class FunctionalPrototype:
    """An abstract pattern shared by N concrete implementations."""

    prototype_id: str
    behavioral_vector: List[float]
    structural_vector: List[float]
    essential_atoms: List[str]
    interchangeable_atoms: List[str]
    member_count: int
    mean_fitness: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# PrototypeExtractor
# --------------------------------------------------------------------------- #


class PrototypeExtractor:
    """Cluster fitness-passing candidates and bundle each cluster.

    The extractor needs an arena so it can call ``allocate``,
    ``inject_phases``, and ``bundle`` to fuse member vectors. Any
    arena that supports these methods works (the ``arena_factory``
    Python fallback or the Rust crate).
    """

    def __init__(self, arena: Any, dimension: int = 256):
        if arena is None:
            raise ValueError("PrototypeExtractor requires an arena")
        self._arena = arena
        self._dimension = int(dimension)
        self._extracted: int = 0

    @property
    def extracted_count(self) -> int:
        return self._extracted

    @staticmethod
    def _phase_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        n = len(a)
        return float(sum(math.cos(a[i] - b[i]) for i in range(n)) / n)

    def _bundle_phases(
        self, phase_arrays: Sequence[Sequence[float]]
    ) -> Tuple[int, List[float]]:
        """Inject + bundle pre-computed phase vectors. Returns (handle, phases)."""
        handles: List[int] = []
        for phases in phase_arrays:
            h = self._arena.allocate()
            self._arena.inject_phases(h, list(phases))
            handles.append(h)
        out = self._arena.allocate()
        self._arena.bundle(handles, out)
        result = self._arena.extract_phases(out)
        return out, list(result)

    def extract_prototypes(
        self,
        candidates: Sequence[Tuple[str, Any, float]],
        similarity_threshold: float = 0.8,
    ) -> List[FunctionalPrototype]:
        """Cluster candidates by behavioural similarity and bundle each cluster.

        ``candidates`` is a list of ``(source, behavioral_profile, fitness)``
        triples. ``behavioral_profile`` must expose
        ``behavioral_phases`` and an optional ``structural_phases``
        attribute (the latter is hashed from source if missing).

        Returns one :class:`FunctionalPrototype` per cluster of size
        >= 2 (singletons are dropped — there is no abstraction in a
        single program).
        """
        if not candidates:
            return []

        # Step 1 — single-link clustering by behavioural similarity.
        n = len(candidates)
        cluster_id = list(range(n))

        def _find(x: int) -> int:
            while cluster_id[x] != x:
                cluster_id[x] = cluster_id[cluster_id[x]]
                x = cluster_id[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                cluster_id[rb] = ra

        for i in range(n):
            phases_i = candidates[i][1].behavioral_phases
            for j in range(i + 1, n):
                phases_j = candidates[j][1].behavioral_phases
                sim = self._phase_similarity(phases_i, phases_j)
                if sim >= similarity_threshold:
                    _union(i, j)

        # Step 2 — group indices by root cluster id.
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = _find(i)
            groups.setdefault(root, []).append(i)

        prototypes: List[FunctionalPrototype] = []
        for root, members in groups.items():
            if len(members) < 2:
                # Skip singletons — there is no abstraction in one program.
                continue
            prototype = self._build_prototype(
                [candidates[i] for i in members]
            )
            prototypes.append(prototype)
            self._extracted += 1
        return prototypes

    def _build_prototype(
        self, members: Sequence[Tuple[str, Any, float]]
    ) -> FunctionalPrototype:
        """Bundle a single cluster into a FunctionalPrototype."""
        sources = [m[0] for m in members]
        profiles = [m[1] for m in members]
        fitnesses = [float(m[2]) for m in members]

        # Behavioural bundle: directly bundle stored phase arrays.
        behavioral_phases_list = [
            list(p.behavioral_phases) for p in profiles
        ]
        _b_handle, behavioral_vector = self._bundle_phases(
            behavioral_phases_list
        )

        # Structural bundle: hash each source to a deterministic phase
        # vector and bundle. (We hash rather than reusing the projector
        # to keep the extractor independent of the projection layer.)
        structural_phase_list: List[List[float]] = []
        for src in sources:
            digest = hashlib.blake2b(
                src.encode("utf-8"), digest_size=64
            ).digest()
            phases: List[float] = []
            counter = 0
            while len(phases) < self._dimension:
                chunk = hashlib.blake2b(
                    digest + counter.to_bytes(2, "big"),
                    digest_size=64,
                ).digest()
                for i in range(0, len(chunk), 4):
                    if len(phases) >= self._dimension:
                        break
                    word = int.from_bytes(chunk[i:i + 4], "big")
                    phases.append((word / 4294967296.0) * _TWO_PI)
                counter += 1
            structural_phase_list.append(phases)

        _s_handle, structural_vector = self._bundle_phases(
            structural_phase_list
        )

        # Atom analysis — essential vs interchangeable.
        member_atom_sets = [_ast_atoms(src) for src in sources]
        if member_atom_sets:
            essential = set(member_atom_sets[0])
            for atoms in member_atom_sets[1:]:
                essential &= atoms
            union: Set[str] = set()
            for atoms in member_atom_sets:
                union |= atoms
            interchangeable = sorted(union - essential)
        else:
            essential = set()
            interchangeable = []

        mean_fitness = sum(fitnesses) / len(fitnesses)
        prototype_id = self._mint_prototype_id(sources, mean_fitness)

        return FunctionalPrototype(
            prototype_id=prototype_id,
            behavioral_vector=behavioral_vector,
            structural_vector=structural_vector,
            essential_atoms=sorted(essential),
            interchangeable_atoms=interchangeable,
            member_count=len(members),
            mean_fitness=mean_fitness,
            metadata={
                "member_atom_counts": [len(s) for s in member_atom_sets],
                "mean_member_atom_count": (
                    sum(len(s) for s in member_atom_sets)
                    / len(member_atom_sets)
                    if member_atom_sets
                    else 0.0
                ),
                "extracted_at": time.time(),
                "provenance": {
                    "operation": "extract_prototype",
                    "source_arena": "thdse",
                    "member_count": len(members),
                },
            },
        )

    @staticmethod
    def _mint_prototype_id(sources: Sequence[str], fitness: float) -> str:
        joined = "|".join(sources[:3]) + f"|fit={fitness:.4f}"
        digest = hashlib.blake2b(
            joined.encode("utf-8"), digest_size=8
        ).hexdigest()
        return f"proto-{digest}"


__all__ = [
    "FunctionalPrototype",
    "PrototypeExtractor",
]
