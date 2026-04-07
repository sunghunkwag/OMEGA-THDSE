"""
Temporal Structural Analysis — Classifies code changes between versions
using layer-decomposed topology rather than textual diffs.

Given two versions of the same file, this module:
  1. Projects both through the FHRR pipeline
  2. Computes layer-decomposed delta (AST, CFG, Data-dep)
  3. Classifies the change type:
     - "Structural refactor" (AST changed, data flow preserved)
     - "Logic change" (CFG changed, AST preserved)
     - "Variable rewiring" (data-dep changed, others preserved)
     - "Complete rewrite" (all layers changed)
     - "Cosmetic change" (all layers preserved — whitespace/comments only)

This provides semantic classification of code changes that git diff
cannot — git sees text changes, THDSE sees topological changes.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    import hdc_core as _hdc_core
except ImportError:
    _hdc_core = None

from src.projection.isomorphic_projector import IsomorphicProjector, LayeredProjection
from src.analysis.structural_diff import StructuralDiffEngine, LayerSimilarity


class ChangeType(Enum):
    """Classification of structural change between two versions."""
    COSMETIC = "cosmetic_change"
    STRUCTURAL_REFACTOR = "structural_refactor"
    LOGIC_CHANGE = "logic_change"
    VARIABLE_REWIRING = "variable_rewiring"
    DATA_FLOW_CHANGE = "data_flow_change"
    INTERFACE_CHANGE = "interface_change"
    COMPLETE_REWRITE = "complete_rewrite"
    MIXED_CHANGE = "mixed_change"


@dataclass
class TemporalDelta:
    """Result of comparing two versions of the same file.

    Attributes:
        file_id: Identifier for the file being compared.
        layer_sim: Per-layer similarity scores between versions.
        change_type: Classified type of structural change.
        change_description: Human-readable description of the change.
        preserved_layers: Which layers remained stable across versions.
        changed_layers: Which layers diverged across versions.
    """
    file_id: str
    layer_sim: LayerSimilarity
    change_type: ChangeType
    change_description: str
    preserved_layers: List[str] = field(default_factory=list)
    changed_layers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "file_id": self.file_id,
            "change_type": self.change_type.value,
            "change_description": self.change_description,
            "preserved_layers": self.preserved_layers,
            "changed_layers": self.changed_layers,
            "layer_similarity": {
                "ast": round(self.layer_sim.sim_ast, 6),
                "cfg": round(self.layer_sim.sim_cfg, 6) if self.layer_sim.sim_cfg is not None else None,
                "data": round(self.layer_sim.sim_data, 6) if self.layer_sim.sim_data is not None else None,
                "final": round(self.layer_sim.sim_final, 6),
            },
        }


class TemporalAnalyzer:
    """Analyzes structural changes between versions of source files.

    Usage:
        analyzer = TemporalAnalyzer(arena, projector)
        delta = analyzer.compare_versions(source_v1, source_v2, "myfile.py")
        print(delta.change_type, delta.change_description)
    """

    # Thresholds for classifying layer preservation vs. change
    PRESERVED_THRESHOLD = 0.7   # similarity >= this → layer is preserved
    CHANGED_THRESHOLD = 0.3     # similarity < this → layer has changed

    def __init__(
        self,
        arena: Any,
        projector: IsomorphicProjector,
        preserved_threshold: float = 0.7,
        changed_threshold: float = 0.3,
    ) -> None:
        """Initialize the temporal analyzer.

        Args:
            arena: FHRR arena instance.
            projector: IsomorphicProjector for projecting source files.
            preserved_threshold: Similarity above this means layer is preserved.
            changed_threshold: Similarity below this means layer has changed.
        """
        self.arena = arena
        self.projector = projector
        self.diff_engine = StructuralDiffEngine(arena, projector)
        self.preserved_threshold = preserved_threshold
        self.changed_threshold = changed_threshold

    def compare_versions(
        self,
        source_v1: str,
        source_v2: str,
        file_id: str = "unknown",
    ) -> TemporalDelta:
        """Compare two versions of a source file.

        Args:
            source_v1: Source code of the first (older) version.
            source_v2: Source code of the second (newer) version.
            file_id: Identifier for the file being compared.

        Returns:
            TemporalDelta with classified change type and description.
        """
        proj_v1 = self.projector.project(source_v1)
        proj_v2 = self.projector.project(source_v2)
        return self.compare_projections(proj_v1, proj_v2, file_id)

    def compare_file_paths(
        self,
        path_v1: str,
        path_v2: str,
        file_id: Optional[str] = None,
    ) -> TemporalDelta:
        """Compare two version files on disk.

        Args:
            path_v1: Path to the first (older) version.
            path_v2: Path to the second (newer) version.
            file_id: Optional file identifier (defaults to path_v2).

        Returns:
            TemporalDelta with classified change type.
        """
        with open(path_v1, "r", encoding="utf-8") as f:
            source_v1 = f.read()
        with open(path_v2, "r", encoding="utf-8") as f:
            source_v2 = f.read()

        if file_id is None:
            file_id = path_v2

        return self.compare_versions(source_v1, source_v2, file_id)

    def compare_projections(
        self,
        proj_v1: LayeredProjection,
        proj_v2: LayeredProjection,
        file_id: str = "unknown",
    ) -> TemporalDelta:
        """Compare two pre-computed projections.

        Args:
            proj_v1: Projection of the first version.
            proj_v2: Projection of the second version.
            file_id: Identifier for the file.

        Returns:
            TemporalDelta with classified change type.
        """
        layer_sim = self.diff_engine.compare_layers(proj_v1, proj_v2)

        # Classify each layer
        preserved: List[str] = []
        changed: List[str] = []

        ast_preserved = abs(layer_sim.sim_ast) >= self.preserved_threshold
        ast_changed = abs(layer_sim.sim_ast) < self.changed_threshold

        if ast_preserved:
            preserved.append("ast")
        elif ast_changed:
            changed.append("ast")

        if layer_sim.sim_cfg is not None:
            if abs(layer_sim.sim_cfg) >= self.preserved_threshold:
                preserved.append("cfg")
            elif abs(layer_sim.sim_cfg) < self.changed_threshold:
                changed.append("cfg")

        if layer_sim.sim_data is not None:
            if abs(layer_sim.sim_data) >= self.preserved_threshold:
                preserved.append("data")
            elif abs(layer_sim.sim_data) < self.changed_threshold:
                changed.append("data")

        # Classify change type based on layer patterns
        change_type, description = self._classify(
            layer_sim, preserved, changed,
        )

        return TemporalDelta(
            file_id=file_id,
            layer_sim=layer_sim,
            change_type=change_type,
            change_description=description,
            preserved_layers=preserved,
            changed_layers=changed,
        )

    def _classify(
        self,
        layer_sim: LayerSimilarity,
        preserved: List[str],
        changed: List[str],
    ) -> Tuple[ChangeType, str]:
        """Classify the change type based on layer preservation patterns.

        Args:
            layer_sim: Per-layer similarity scores.
            preserved: List of preserved layer names.
            changed: List of changed layer names.

        Returns:
            Tuple of (ChangeType, description string).
        """
        pset = set(preserved)
        cset = set(changed)

        # All layers preserved → cosmetic change
        if len(pset) >= 2 and not cset:
            return (
                ChangeType.COSMETIC,
                f"Cosmetic change: all topology layers preserved "
                f"(AST={layer_sim.sim_ast:.2f}). "
                f"Likely whitespace, comments, or formatting only.",
            )

        # All layers changed → complete rewrite
        if len(cset) >= 2 and not pset:
            return (
                ChangeType.COMPLETE_REWRITE,
                f"Complete rewrite: all topology layers diverged "
                f"(AST={layer_sim.sim_ast:.2f}, "
                f"CFG={layer_sim.sim_cfg:.2f if layer_sim.sim_cfg is not None else 'N/A'}, "
                f"Data={layer_sim.sim_data:.2f if layer_sim.sim_data is not None else 'N/A'}).",
            )

        # AST changed, data flow preserved → structural refactor
        if "ast" in cset and "data" in pset:
            return (
                ChangeType.STRUCTURAL_REFACTOR,
                f"Structural refactor: AST changed ({layer_sim.sim_ast:.2f}) "
                f"but data flow preserved ({layer_sim.sim_data:.2f}). "
                f"Syntax reorganization without semantic change.",
            )

        # CFG changed, AST preserved → logic change
        if "cfg" in cset and "ast" in pset:
            return (
                ChangeType.LOGIC_CHANGE,
                f"Logic change: CFG diverged ({layer_sim.sim_cfg:.2f}) "
                f"while AST preserved ({layer_sim.sim_ast:.2f}). "
                f"Same structure, different control flow logic.",
            )

        # Data-dep changed, others preserved → variable rewiring
        if "data" in cset and "ast" in pset:
            return (
                ChangeType.VARIABLE_REWIRING,
                f"Variable rewiring: data dependencies changed "
                f"({layer_sim.sim_data:.2f}) while AST preserved "
                f"({layer_sim.sim_ast:.2f}). "
                f"Same structure, different variable relationships.",
            )

        # Data-dep changed, AST changed, CFG preserved
        if "data" in cset and "ast" in cset and "cfg" in pset:
            return (
                ChangeType.INTERFACE_CHANGE,
                f"Interface change: structure and data flow changed but "
                f"control flow preserved ({layer_sim.sim_cfg:.2f}).",
            )

        # Mixed changes
        desc_parts = []
        if "ast" in cset:
            desc_parts.append(f"AST changed ({layer_sim.sim_ast:.2f})")
        elif "ast" in pset:
            desc_parts.append(f"AST preserved ({layer_sim.sim_ast:.2f})")

        if "cfg" in cset:
            desc_parts.append(f"CFG changed ({layer_sim.sim_cfg:.2f})")
        elif "cfg" in pset:
            desc_parts.append(f"CFG preserved ({layer_sim.sim_cfg:.2f})")

        if "data" in cset:
            desc_parts.append(f"Data changed ({layer_sim.sim_data:.2f})")
        elif "data" in pset:
            desc_parts.append(f"Data preserved ({layer_sim.sim_data:.2f})")

        return (
            ChangeType.MIXED_CHANGE,
            f"Mixed change: {'; '.join(desc_parts)}.",
        )

    def batch_compare(
        self,
        version_pairs: List[Tuple[str, str, str]],
    ) -> List[TemporalDelta]:
        """Compare multiple file version pairs.

        Args:
            version_pairs: List of (source_v1, source_v2, file_id) tuples.

        Returns:
            List of TemporalDelta results.
        """
        return [
            self.compare_versions(v1, v2, fid)
            for v1, v2, fid in version_pairs
        ]
