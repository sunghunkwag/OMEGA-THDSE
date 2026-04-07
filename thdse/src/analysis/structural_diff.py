"""
Structural Diff Engine — Layer-decomposed similarity analysis and
structural delta reporting between source files.

Currently, analysis collapses three topology layers (AST, CFG, Data-dep)
into a single scalar resonance. This module unlocks per-layer information:

  - Per-layer cosine similarity: sim_ast, sim_cfg, sim_data
  - Layer-driven similarity/difference diagnosis
  - Sub-tree atom set difference for structural delta reports

All operations derive from existing LayeredProjection data.
No new projections required — just exposing information that was
already computed but never surfaced.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import hdc_core as _hdc_core
except ImportError:
    _hdc_core = None

from src.projection.isomorphic_projector import IsomorphicProjector, LayeredProjection
from src.decoder.subtree_vocab import SubTreeVocabulary, SubTreeAtom


@dataclass
class LayerSimilarity:
    """Per-layer similarity decomposition between two projections.

    Attributes:
        sim_ast: Cosine similarity on the AST structural layer.
        sim_cfg: Cosine similarity on the CFG sequential layer (None if missing).
        sim_data: Cosine similarity on the data-dep layer (None if missing).
        sim_final: Cosine similarity on the cross-layer-bound final handle.
        dominant_layer: Which layer drives the overall similarity.
        diagnosis: Human-readable explanation of the similarity pattern.
    """
    sim_ast: float
    sim_cfg: Optional[float]
    sim_data: Optional[float]
    sim_final: float
    dominant_layer: str = ""
    diagnosis: str = ""


@dataclass
class StructuralDelta:
    """Structural difference report between two projections.

    Attributes:
        layer_sim: Per-layer similarity scores.
        atoms_only_a: Sub-tree atoms activated only in projection A.
        atoms_only_b: Sub-tree atoms activated only in projection B.
        atoms_shared: Sub-tree atoms activated in both projections.
        summary: Human-readable summary of structural differences.
    """
    layer_sim: LayerSimilarity
    atoms_only_a: List[str] = field(default_factory=list)
    atoms_only_b: List[str] = field(default_factory=list)
    atoms_shared: List[str] = field(default_factory=list)
    summary: str = ""


class StructuralDiffEngine:
    """Computes layer-decomposed structural diffs between source files.

    Usage:
        engine = StructuralDiffEngine(arena, projector)
        sim = engine.compare_layers(proj_a, proj_b)
        delta = engine.compute_delta(proj_a, proj_b, vocab=subtree_vocab)
    """

    def __init__(
        self,
        arena: Any,
        projector: IsomorphicProjector,
        activation_threshold: float = 0.04,
    ) -> None:
        """Initialize the structural diff engine.

        Args:
            arena: FHRR arena instance.
            projector: IsomorphicProjector for projecting source files.
            activation_threshold: Minimum correlation for sub-tree activation.
        """
        self.arena = arena
        self.projector = projector
        self.activation_threshold = activation_threshold

    def compare_layers(
        self,
        proj_a: LayeredProjection,
        proj_b: LayeredProjection,
    ) -> LayerSimilarity:
        """Compute per-layer cosine similarity between two projections.

        Args:
            proj_a: First LayeredProjection.
            proj_b: Second LayeredProjection.

        Returns:
            LayerSimilarity with per-layer scores and diagnosis.
        """
        # AST layer (always present)
        sim_ast = self.arena.compute_correlation(proj_a.ast_handle, proj_b.ast_handle)

        # CFG layer
        sim_cfg = None
        if proj_a.cfg_handle is not None and proj_b.cfg_handle is not None:
            sim_cfg = self.arena.compute_correlation(proj_a.cfg_handle, proj_b.cfg_handle)

        # Data-dep layer
        sim_data = None
        if proj_a.data_handle is not None and proj_b.data_handle is not None:
            sim_data = self.arena.compute_correlation(proj_a.data_handle, proj_b.data_handle)

        # Final cross-layer-bound
        sim_final = self.arena.compute_correlation(proj_a.final_handle, proj_b.final_handle)

        # Determine dominant layer and diagnosis
        scores = {"ast": sim_ast}
        if sim_cfg is not None:
            scores["cfg"] = sim_cfg
        if sim_data is not None:
            scores["data"] = sim_data

        dominant = max(scores, key=lambda k: abs(scores[k]))
        diagnosis = self._diagnose(sim_ast, sim_cfg, sim_data)

        return LayerSimilarity(
            sim_ast=sim_ast,
            sim_cfg=sim_cfg,
            sim_data=sim_data,
            sim_final=sim_final,
            dominant_layer=dominant,
            diagnosis=diagnosis,
        )

    def compare_files(
        self,
        source_a: str,
        source_b: str,
    ) -> LayerSimilarity:
        """Compute per-layer similarity between two source code strings.

        Args:
            source_a: First Python source code.
            source_b: Second Python source code.

        Returns:
            LayerSimilarity with per-layer scores and diagnosis.
        """
        proj_a = self.projector.project(source_a)
        proj_b = self.projector.project(source_b)
        return self.compare_layers(proj_a, proj_b)

    def compare_file_paths(
        self,
        path_a: str,
        path_b: str,
    ) -> LayerSimilarity:
        """Compute per-layer similarity between two source files on disk.

        Args:
            path_a: Path to first Python source file.
            path_b: Path to second Python source file.

        Returns:
            LayerSimilarity with per-layer scores and diagnosis.
        """
        with open(path_a, "r", encoding="utf-8") as f:
            source_a = f.read()
        with open(path_b, "r", encoding="utf-8") as f:
            source_b = f.read()
        return self.compare_files(source_a, source_b)

    def compute_delta(
        self,
        proj_a: LayeredProjection,
        proj_b: LayeredProjection,
        vocab: Optional[SubTreeVocabulary] = None,
    ) -> StructuralDelta:
        """Compute full structural delta between two projections.

        Includes per-layer similarity and sub-tree atom set differences
        (if a SubTreeVocabulary is provided).

        Args:
            proj_a: First LayeredProjection.
            proj_b: Second LayeredProjection.
            vocab: Optional sub-tree vocabulary for atom-level delta.

        Returns:
            StructuralDelta with layer similarities and atom differences.
        """
        layer_sim = self.compare_layers(proj_a, proj_b)

        atoms_only_a: List[str] = []
        atoms_only_b: List[str] = []
        atoms_shared: List[str] = []

        if vocab is not None:
            activated_a = self._probe_atoms(proj_a, vocab)
            activated_b = self._probe_atoms(proj_b, vocab)

            set_a = set(activated_a.keys())
            set_b = set(activated_b.keys())

            atoms_only_a = sorted(
                activated_a[h] for h in (set_a - set_b)
            )
            atoms_only_b = sorted(
                activated_b[h] for h in (set_b - set_a)
            )
            atoms_shared = sorted(
                activated_a[h] for h in (set_a & set_b)
            )

        summary = self._build_delta_summary(layer_sim, atoms_only_a, atoms_only_b)

        return StructuralDelta(
            layer_sim=layer_sim,
            atoms_only_a=atoms_only_a,
            atoms_only_b=atoms_only_b,
            atoms_shared=atoms_shared,
            summary=summary,
        )

    def _probe_atoms(
        self,
        proj: LayeredProjection,
        vocab: SubTreeVocabulary,
    ) -> Dict[str, str]:
        """Probe a projection against sub-tree vocabulary.

        Returns:
            Dict mapping tree_hash -> canonical_source for activated atoms.
        """
        result: Dict[str, str] = {}
        for atom in vocab.get_projected_atoms():
            if atom.handle is None:
                continue
            corr = self.arena.compute_correlation(proj.ast_handle, atom.handle)
            if abs(corr) >= self.activation_threshold:
                result[atom.tree_hash] = atom.canonical_source
        return result

    def _diagnose(
        self,
        sim_ast: float,
        sim_cfg: Optional[float],
        sim_data: Optional[float],
    ) -> str:
        """Generate a human-readable diagnosis of the similarity pattern.

        Args:
            sim_ast: AST layer similarity.
            sim_cfg: CFG layer similarity (or None).
            sim_data: Data-dep layer similarity (or None).

        Returns:
            Diagnostic string.
        """
        high = 0.7
        low = 0.3
        parts: List[str] = []

        ast_high = abs(sim_ast) >= high
        ast_low = abs(sim_ast) < low
        cfg_high = sim_cfg is not None and abs(sim_cfg) >= high
        cfg_low = sim_cfg is not None and abs(sim_cfg) < low
        data_high = sim_data is not None and abs(sim_data) >= high
        data_low = sim_data is not None and abs(sim_data) < low

        if ast_high and cfg_high and data_high:
            return "Near-identical structure across all layers"

        if ast_low and cfg_low and data_low:
            return "Completely different structure across all layers"

        if ast_high and cfg_low:
            parts.append(
                f"Same AST shape ({sim_ast:.2f}) but different control flow"
                f" ({sim_cfg:.2f}) — same structure, different logic"
            )
        elif ast_low and cfg_high:
            parts.append(
                f"Different AST shape ({sim_ast:.2f}) but same control flow"
                f" ({sim_cfg:.2f}) — different syntax, same algorithm"
            )

        if data_high and ast_low:
            parts.append(
                "Same data flow pattern despite different syntax"
                " — candidate for abstraction into shared utility"
            )
        elif data_low and ast_high:
            parts.append(
                "Same structure but different data flow"
                " — variable rewiring or different data dependencies"
            )

        if not parts:
            scores = [f"AST={sim_ast:.2f}"]
            if sim_cfg is not None:
                scores.append(f"CFG={sim_cfg:.2f}")
            if sim_data is not None:
                scores.append(f"Data={sim_data:.2f}")
            return f"Mixed similarity: {', '.join(scores)}"

        return "; ".join(parts)

    def _build_delta_summary(
        self,
        layer_sim: LayerSimilarity,
        atoms_only_a: List[str],
        atoms_only_b: List[str],
    ) -> str:
        """Build a summary string for a structural delta.

        Args:
            layer_sim: Layer similarity scores.
            atoms_only_a: Atoms unique to file A.
            atoms_only_b: Atoms unique to file B.

        Returns:
            Summary string.
        """
        parts = [layer_sim.diagnosis]

        if atoms_only_a:
            parts.append(f"File A has {len(atoms_only_a)} unique sub-tree patterns: "
                        f"{'; '.join(atoms_only_a[:3])}")
        if atoms_only_b:
            parts.append(f"File B has {len(atoms_only_b)} unique sub-tree patterns: "
                        f"{'; '.join(atoms_only_b[:3])}")

        return " | ".join(parts)

    def to_dict(self, delta: StructuralDelta) -> Dict[str, Any]:
        """Serialize a StructuralDelta to a JSON-compatible dictionary.

        Args:
            delta: The delta to serialize.

        Returns:
            Dictionary representation.
        """
        ls = delta.layer_sim
        return {
            "layer_similarity": {
                "ast": round(ls.sim_ast, 6),
                "cfg": round(ls.sim_cfg, 6) if ls.sim_cfg is not None else None,
                "data": round(ls.sim_data, 6) if ls.sim_data is not None else None,
                "final": round(ls.sim_final, 6),
                "dominant_layer": ls.dominant_layer,
                "diagnosis": ls.diagnosis,
            },
            "atoms_only_a": delta.atoms_only_a,
            "atoms_only_b": delta.atoms_only_b,
            "atoms_shared": delta.atoms_shared,
            "summary": delta.summary,
        }
