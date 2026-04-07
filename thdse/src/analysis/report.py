"""Report generator for structural analysis results.

Produces structured JSON and human-readable Markdown reports from
resonance analysis, clustering, clique interpretation, structural diffs,
refactoring candidates, and temporal analysis.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.analysis.resonance_analyzer import ResonanceAnalyzer, CliqueFingerpint


class ReportGenerator:
    """Generates JSON and Markdown reports from analysis results.

    Supports extended report sections:
      - Structural diff results (layer-decomposed similarity)
      - Refactoring candidates (structural duplicates + interface unification)
      - Temporal analysis (version-to-version change classification)

    Usage:
        analyzer = ResonanceAnalyzer(synth, decoder)
        analyzer.compute_resonance()
        report = ReportGenerator(analyzer, dimension=256)
        report.add_structural_diffs(diff_results)
        report.add_refactoring_candidates(candidates)
        report.add_temporal_analysis(temporal_results)
        report.generate("/path/to/output/")
    """

    def __init__(
        self,
        analyzer: ResonanceAnalyzer,
        dimension: int,
        corpus_name: str = "unnamed",
    ) -> None:
        """Initialize report generator.

        Args:
            analyzer: Populated ResonanceAnalyzer with computed resonance.
            dimension: Hypervector dimension used.
            corpus_name: Human-readable name for the corpus.
        """
        self.analyzer = analyzer
        self.dimension = dimension
        self.corpus_name = corpus_name
        self._structural_diffs: List[Dict[str, Any]] = []
        self._refactoring_candidates: Optional[Dict[str, Any]] = None
        self._temporal_analysis: List[Dict[str, Any]] = []

    def add_structural_diffs(self, diffs: List[Dict[str, Any]]) -> None:
        """Add structural diff results to the report.

        Args:
            diffs: List of serialized StructuralDelta dictionaries.
        """
        self._structural_diffs = diffs

    def add_refactoring_candidates(self, candidates: Dict[str, Any]) -> None:
        """Add refactoring candidates to the report.

        Args:
            candidates: Serialized RefactoringDetector output.
        """
        self._refactoring_candidates = candidates

    def add_temporal_analysis(self, results: List[Dict[str, Any]]) -> None:
        """Add temporal analysis results to the report.

        Args:
            results: List of serialized TemporalDelta dictionaries.
        """
        self._temporal_analysis = results

    def build_report_data(self) -> Dict[str, Any]:
        """Collect all analysis data into a structured dictionary.

        Returns:
            Dictionary containing all report sections.
        """
        ids = self.analyzer.get_ids()
        stats = self.analyzer.resonance_stats()

        # Section 1: Corpus summary
        summary = {
            "corpus_name": self.corpus_name,
            "total_files": len(ids),
            "dimension": self.dimension,
            "resonance_mean": round(stats["mean"], 6),
            "resonance_std": round(stats["std"], 6),
            "resonance_min": round(stats["min"], 6),
            "resonance_max": round(stats["max"], 6),
        }

        # Section 2: Top resonant pairs
        top_pairs = self.analyzer.top_resonant_pairs(k=10)
        pairs_data = [
            {"file_a": a, "file_b": b, "resonance": round(r, 6)}
            for a, b, r in top_pairs
        ]

        # Section 3: Outliers
        outliers = self.analyzer.bottom_resonant_axioms(k=10)
        outlier_data = [
            {"file": sid, "mean_resonance": round(mr, 6)}
            for sid, mr in outliers
        ]

        # Statistical outliers
        stat_outliers = self.analyzer.find_outliers(threshold_sigmas=2.0)
        stat_outlier_data = [
            {"file": sid, "mean_resonance": round(mr, 6)}
            for sid, mr in stat_outliers
        ]

        # Section 4: Cliques with fingerprints
        fingerprints = self.analyzer.interpret_cliques(min_size=3)
        clique_data = [fp.to_dict() for fp in fingerprints]

        # Also include size-2 cliques for completeness
        small_fingerprints = self.analyzer.interpret_cliques(min_size=2)
        small_clique_data = [
            fp.to_dict() for fp in small_fingerprints
            if len(fp.clique) == 2
        ]

        # Section 5: Cross-project bridges
        bridges = self.analyzer.cross_project_bridges(threshold=0.8)
        bridge_data = [
            {"file_a": a, "file_b": b, "resonance": round(r, 6)}
            for a, b, r in bridges
        ]

        # Also try lower threshold if no bridges found at 0.8
        if not bridge_data:
            bridges_relaxed = self.analyzer.cross_project_bridges(threshold=0.5)
            if bridges_relaxed:
                bridge_data = [
                    {
                        "file_a": a, "file_b": b,
                        "resonance": round(r, 6),
                        "note": "relaxed threshold (0.5)",
                    }
                    for a, b, r in bridges_relaxed[:10]
                ]

        # Section 6: Linkage matrix for dendrogram
        try:
            linkage = self.analyzer.hierarchical_clustering()
            linkage_data = linkage.tolist() if linkage.size > 0 else []
        except Exception:
            linkage_data = []

        result = {
            "summary": summary,
            "top_resonant_pairs": pairs_data,
            "lowest_resonance_outliers": outlier_data,
            "statistical_outliers": stat_outlier_data,
            "cliques_size_3_plus": clique_data,
            "cliques_size_2": small_clique_data,
            "cross_project_bridges": bridge_data,
            "dendrogram_linkage": linkage_data,
            "axiom_ids": ids,
        }

        # Extended sections (LEAP 2+)
        if self._structural_diffs:
            result["structural_diffs"] = self._structural_diffs
        if self._refactoring_candidates:
            result["refactoring_candidates"] = self._refactoring_candidates
        if self._temporal_analysis:
            result["temporal_analysis"] = self._temporal_analysis

        return result

    def generate_json(self, output_path: str) -> str:
        """Write the full report as JSON.

        Args:
            output_path: Directory to write report.json into.

        Returns:
            Path to the written JSON file.
        """
        data = self.build_report_data()
        filepath = os.path.join(output_path, "report.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath

    def generate_markdown(self, output_path: str) -> str:
        """Write a human-readable Markdown report.

        Args:
            output_path: Directory to write report.md into.

        Returns:
            Path to the written Markdown file.
        """
        data = self.build_report_data()
        lines: List[str] = []

        # Header
        lines.append(f"# Structural Analysis Report: {data['summary']['corpus_name']}")
        lines.append("")

        # Summary
        s = data["summary"]
        lines.append("## Corpus Summary")
        lines.append("")
        lines.append(f"- **Total files analyzed:** {s['total_files']}")
        lines.append(f"- **Hypervector dimension:** {s['dimension']}")
        lines.append(f"- **Mean resonance:** {s['resonance_mean']:.6f}")
        lines.append(f"- **Std resonance:** {s['resonance_std']:.6f}")
        lines.append(f"- **Min resonance:** {s['resonance_min']:.6f}")
        lines.append(f"- **Max resonance:** {s['resonance_max']:.6f}")
        lines.append("")
        lines.append("> Resonance measures structural similarity between files in VSA space.")
        lines.append("> Values near 1.0 indicate structurally identical patterns;")
        lines.append("> values near 0.0 indicate orthogonal (completely different) structure.")
        lines.append("")

        # Top pairs
        lines.append("## Top 10 Most Resonant Pairs")
        lines.append("")
        if data["top_resonant_pairs"]:
            lines.append("| Rank | File A | File B | Resonance |")
            lines.append("|------|--------|--------|-----------|")
            for i, p in enumerate(data["top_resonant_pairs"], 1):
                lines.append(f"| {i} | `{p['file_a']}` | `{p['file_b']}` | {p['resonance']:.6f} |")
        else:
            lines.append("*No pairs found.*")
        lines.append("")

        # Outliers
        lines.append("## Top 10 Lowest-Resonance Files (Structural Outliers)")
        lines.append("")
        lines.append("> These files have the lowest average resonance with all other files,")
        lines.append("> meaning their structure is most unique in the corpus.")
        lines.append("")
        if data["lowest_resonance_outliers"]:
            lines.append("| Rank | File | Mean Resonance |")
            lines.append("|------|------|----------------|")
            for i, o in enumerate(data["lowest_resonance_outliers"], 1):
                lines.append(f"| {i} | `{o['file']}` | {o['mean_resonance']:.6f} |")
        else:
            lines.append("*No outliers found.*")
        lines.append("")

        # Statistical outliers
        if data["statistical_outliers"]:
            lines.append("### Statistical Outliers (>2 sigma below mean)")
            lines.append("")
            for o in data["statistical_outliers"]:
                lines.append(f"- `{o['file']}` (mean resonance: {o['mean_resonance']:.6f})")
            lines.append("")

        # Cliques
        lines.append("## Structural Cliques (size >= 3)")
        lines.append("")
        lines.append("> A clique is a group of files that are all mutually structurally similar.")
        lines.append("> The 'structural fingerprint' describes what patterns they share.")
        lines.append("")
        if data["cliques_size_3_plus"]:
            for i, c in enumerate(data["cliques_size_3_plus"], 1):
                lines.append(f"### Clique {i} ({len(c['clique'])} members)")
                lines.append("")
                lines.append(f"**Fingerprint:** {c['description']}")
                lines.append("")
                if c.get("node_types"):
                    lines.append(f"- Node types: {', '.join(c['node_types'])}")
                if c.get("cfg_summary"):
                    lines.append(f"- CFG: {c['cfg_summary']}")
                if c.get("data_dep_summary"):
                    lines.append(f"- Data flow: {c['data_dep_summary']}")
                lines.append("")
                lines.append("Members:")
                for m in c["clique"]:
                    lines.append(f"- `{m}`")
                lines.append("")
        else:
            lines.append("*No cliques of size >= 3 found.*")
        lines.append("")

        # Cross-project bridges
        lines.append("## Cross-Project Structural Twins")
        lines.append("")
        lines.append("> Files from different projects that share nearly identical structure.")
        lines.append("> These are the most interesting discoveries — structural patterns")
        lines.append("> that recur across completely different codebases.")
        lines.append("")
        if data["cross_project_bridges"]:
            lines.append("| File A | File B | Resonance |")
            lines.append("|--------|--------|-----------|")
            for b in data["cross_project_bridges"]:
                note = f" ({b['note']})" if "note" in b else ""
                lines.append(f"| `{b['file_a']}` | `{b['file_b']}` | {b['resonance']:.6f}{note} |")
        else:
            lines.append("*No cross-project bridges found (single project or no high-resonance pairs).*")
        lines.append("")

        # Structural Diffs (LEAP 2)
        if data.get("structural_diffs"):
            lines.append("## Structural Diffs (Layer-Decomposed)")
            lines.append("")
            lines.append("> Per-layer similarity between file pairs, showing which")
            lines.append("> topology layer drives the similarity or difference.")
            lines.append("")
            lines.append("| File A | File B | AST | CFG | Data | Diagnosis |")
            lines.append("|--------|--------|-----|-----|------|-----------|")
            for d in data["structural_diffs"][:20]:
                ls = d.get("layer_similarity", d)
                ast_s = f"{ls.get('ast', 'N/A'):.4f}" if isinstance(ls.get('ast'), (int, float)) else "N/A"
                cfg_s = f"{ls.get('cfg', 'N/A'):.4f}" if isinstance(ls.get('cfg'), (int, float)) else "N/A"
                data_s = f"{ls.get('data', 'N/A'):.4f}" if isinstance(ls.get('data'), (int, float)) else "N/A"
                diag = ls.get("diagnosis", d.get("summary", ""))[:60]
                fa = d.get("file_a", "?")
                fb = d.get("file_b", "?")
                lines.append(f"| `{fa}` | `{fb}` | {ast_s} | {cfg_s} | {data_s} | {diag} |")
            lines.append("")

        # Refactoring Candidates (LEAP 2C)
        if data.get("refactoring_candidates"):
            rc = data["refactoring_candidates"]
            lines.append("## Refactoring Candidates")
            lines.append("")

            dupes = rc.get("structural_duplicates", [])
            if dupes:
                lines.append("### Structural Duplicates")
                lines.append("")
                lines.append("> Files in different directories with highly similar topology.")
                lines.append("> Consider extracting common code into shared modules.")
                lines.append("")
                for d in dupes[:10]:
                    lines.append(f"- `{d['file_a']}` <-> `{d['file_b']}` — {d['recommendation']}")
                lines.append("")

            unifs = rc.get("interface_unification_candidates", [])
            if unifs:
                lines.append("### Interface Unification Candidates")
                lines.append("")
                lines.append("> Files with the same data flow but different syntax.")
                lines.append("> Candidate for unification behind a common interface.")
                lines.append("")
                for u in unifs[:10]:
                    lines.append(f"- `{u['file_a']}` <-> `{u['file_b']}` — {u['recommendation']}")
                lines.append("")

        # Temporal Analysis (LEAP 2D)
        if data.get("temporal_analysis"):
            lines.append("## Temporal Structural Analysis")
            lines.append("")
            lines.append("> Semantic classification of code changes between versions.")
            lines.append("> Git sees text changes; THDSE sees topological changes.")
            lines.append("")
            lines.append("| File | Change Type | Description |")
            lines.append("|------|------------|-------------|")
            for t in data["temporal_analysis"]:
                lines.append(
                    f"| `{t.get('file_id', '?')}` | {t.get('change_type', '?')} "
                    f"| {t.get('change_description', '')[:80]} |"
                )
            lines.append("")

        # Dendrogram info
        lines.append("## Hierarchical Clustering")
        lines.append("")
        if data["dendrogram_linkage"]:
            lines.append(f"Linkage matrix computed ({len(data['dendrogram_linkage'])} merges).")
            lines.append("See `report.json` for the full linkage matrix data,")
            lines.append("which can be visualized with:")
            lines.append("")
            lines.append("```python")
            lines.append("import json")
            lines.append("import numpy as np")
            lines.append("from scipy.cluster.hierarchy import dendrogram")
            lines.append("import matplotlib.pyplot as plt")
            lines.append("")
            lines.append("with open('report.json') as f:")
            lines.append("    data = json.load(f)")
            lines.append("Z = np.array(data['dendrogram_linkage'])")
            lines.append("labels = data['axiom_ids']")
            lines.append("dendrogram(Z, labels=labels, leaf_rotation=90)")
            lines.append("plt.tight_layout()")
            lines.append("plt.savefig('dendrogram.png', dpi=150)")
            lines.append("```")
        else:
            lines.append("*Insufficient data for clustering.*")
        lines.append("")

        filepath = os.path.join(output_path, "report.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath

    def generate(self, output_path: str) -> Tuple[str, str]:
        """Generate both JSON and Markdown reports.

        Args:
            output_path: Directory to write reports into (created if needed).

        Returns:
            Tuple of (json_path, markdown_path).
        """
        os.makedirs(output_path, exist_ok=True)
        json_path = self.generate_json(output_path)
        md_path = self.generate_markdown(output_path)
        return json_path, md_path
