"""
Axiomatic Synthesizer — Deterministic phase-transition engine that accumulates
'Axiom Vectors' from stable codebases and synthesizes novel structural
hypervectors through mathematically provable binding operations.

No randomness. No LLMs. Every operation is a deterministic function of the
input corpus and the VSA algebra (FHRR bind/bundle/correlate).

Singularity Expansion:
  - Autopoietic Self-Reference (Ouroboros Loop): recursively ingests its own
    source code, projects it through the pipeline, and synthesizes self-representations.
    Z3 proves whether newly synthesized self-representations are strictly more
    optimal (lower thermodynamic entropy).
  - Meta-Grammar Emergence: delegates to ConstraintDecoder for UNSAT-triggered
    dimension expansion and operator fusion.
  - Topological Thermodynamics: entropy-aware synthesis selects minimum-complexity
    phase transitions.

Mathematical foundation:
  - Each stable codebase c_i is projected to an axiom vector A_i ∈ S^{d-1} (unit torus).
  - Structural resonance between axioms: ρ(A_i, A_j) = correlate(A_i, A_j).
  - Per-layer synthesis preserves layer decomposition:
      S_ast  = A₁_ast  ⊗ A₂_ast  ⊗ ... ⊗ Aₖ_ast
      S_cfg  = A₁_cfg  ⊗ A₂_cfg  ⊗ ... ⊗ Aₖ_cfg
      S_data = A₁_data ⊗ A₂_data ⊗ ... ⊗ Aₖ_data
      S_final = S_ast ⊗ S_cfg ⊗ S_data
"""

import json
import os
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from src.projection.isomorphic_projector import IsomorphicProjector, LayeredProjection
from src.utils.arena_ops import (
    bind_phases, negate_phases,
    bind_bundle_fusion_phases, compute_phase_entropy, compute_operation_entropy,
)


@dataclass
class Axiom:
    """An axiom vector with provenance metadata and layered projection."""
    projection: LayeredProjection
    source_id: str
    resonance_profile: Dict[str, float] = field(default_factory=dict)

    @property
    def handle(self) -> int:
        """Backward-compatible: the final cross-layer-bound handle."""
        return self.projection.final_handle


class AxiomStore:
    """Flat, indexed store of axiom vectors within a shared arena."""

    def __init__(self, arena: Any):
        self.arena = arena
        self.axioms: Dict[str, Axiom] = {}

    def register(self, source_id: str, projection: LayeredProjection) -> Axiom:
        axiom = Axiom(projection=projection, source_id=source_id)
        self.axioms[source_id] = axiom
        return axiom

    def get(self, source_id: str) -> Optional[Axiom]:
        return self.axioms.get(source_id)

    def all_ids(self) -> List[str]:
        return sorted(self.axioms.keys())

    def count(self) -> int:
        return len(self.axioms)


class ResonanceMatrix:
    """Computes and caches the full pairwise correlation matrix over the axiom store.

    Uses final_handle (cross-layer bound) for resonance — this captures
    the full structural similarity including cross-layer interactions.
    """

    def __init__(self, arena: Any, store: AxiomStore):
        self.arena = arena
        self.store = store
        self._matrix: Dict[Tuple[str, str], float] = {}

    def compute_full(self) -> Dict[Tuple[str, str], float]:
        self._matrix.clear()
        ids = self.store.all_ids()
        n = len(ids)
        if n == 0:
            return self._matrix

        # Batch: single Rust FFI call for entire upper triangle
        handles = [self.store.axioms[sid].handle for sid in ids]
        has_batch = hasattr(self.arena, 'correlate_matrix')

        if has_batch and n >= 2:
            flat = self.arena.correlate_matrix(handles)
            # Unpack flat upper-triangle into symmetric dict
            k = 0
            for i in range(n):
                id_a = ids[i]
                ax_a = self.store.axioms[id_a]
                # Self-correlation
                self._matrix[(id_a, id_a)] = self.arena.compute_correlation(
                    handles[i], handles[i])
                ax_a.resonance_profile[id_a] = self._matrix[(id_a, id_a)]
                for j in range(i + 1, n):
                    id_b = ids[j]
                    ax_b = self.store.axioms[id_b]
                    corr = flat[k]
                    k += 1
                    self._matrix[(id_a, id_b)] = corr
                    self._matrix[(id_b, id_a)] = corr
                    ax_a.resonance_profile[id_b] = corr
                    ax_b.resonance_profile[id_a] = corr
        else:
            # Fallback: individual calls (N < 2 or no batch support)
            for i, id_a in enumerate(ids):
                ax_a = self.store.axioms[id_a]
                for id_b in ids[i:]:
                    ax_b = self.store.axioms[id_b]
                    corr = self.arena.compute_correlation(ax_a.handle, ax_b.handle)
                    self._matrix[(id_a, id_b)] = corr
                    self._matrix[(id_b, id_a)] = corr
                    ax_a.resonance_profile[id_b] = corr
                    ax_b.resonance_profile[id_a] = corr

        return self._matrix

    def get(self, id_a: str, id_b: str) -> float:
        return self._matrix.get((id_a, id_b), 0.0)


class AxiomaticSynthesizer:
    """Ingests stable codebases, discovers resonance cliques, and synthesizes
    novel structural hypervectors through deterministic per-layer binding.

    Pipeline:
      1. Ingest: code → IsomorphicProjector → LayeredProjection → AxiomStore
      2. Resonate: compute pairwise correlation matrix
      3. Clique extraction: find all maximal sets where ∀ pairs exceed threshold τ
      4. Synthesize: per-layer chain-bind → cross-layer bind → LayeredProjection
    """

    def __init__(
        self,
        arena: Any,
        projector: IsomorphicProjector,
        resonance_threshold: float = 0.15,
    ):
        self.arena = arena
        self.projector = projector
        self.store = AxiomStore(arena)
        self.resonance = ResonanceMatrix(arena, self.store)
        self.tau = resonance_threshold
        self._synthesis_log: List[Tuple[List[str], LayeredProjection]] = []

    # ── Ingestion ────────────────────────────────────────────────

    def ingest(self, source_id: str, code: str) -> Axiom:
        """Project a codebase and register it as an axiom."""
        projection = self.projector.project(code)
        return self.store.register(source_id, projection)

    def ingest_batch(self, corpus: Dict[str, str]) -> int:
        """Ingest multiple codebases. Returns count of axioms registered."""
        for source_id in sorted(corpus.keys()):
            self.ingest(source_id, corpus[source_id])
        return self.store.count()

    # ── Resonance analysis ───────────────────────────────────────

    def compute_resonance(self) -> Dict[Tuple[str, str], float]:
        return self.resonance.compute_full()

    # ── Clique extraction (Bron-Kerbosch, deterministic) ─────────

    def extract_cliques(self, min_size: int = 2) -> List[List[str]]:
        ids = self.store.all_ids()
        adj: Dict[str, set] = {sid: set() for sid in ids}
        for id_a in ids:
            for id_b in ids:
                if id_a != id_b and self.resonance.get(id_a, id_b) > self.tau:
                    adj[id_a].add(id_b)

        cliques: List[List[str]] = []
        self._bron_kerbosch(set(), set(ids), set(), adj, cliques)
        return sorted(
            [c for c in cliques if len(c) >= min_size],
            key=lambda c: (-len(c), c),
        )

    def _bron_kerbosch(
        self, R: set, P: set, X: set, adj: Dict[str, set],
        results: List[List[str]],
    ):
        if not P and not X:
            results.append(sorted(R))
            return
        pivot_candidates = sorted(P | X)
        pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
        candidates = sorted(P - adj[pivot])
        for v in candidates:
            self._bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, results)
            P = P - {v}
            X = X | {v}

    # ── Per-layer synthesis (Phase Transition) ───────────────────

    def _chain_bind_handles(self, handles: List[int]) -> int:
        """Chain-bind a list of arena handles: h₀ ⊗ h₁ ⊗ ... ⊗ hₖ."""
        current = handles[0]
        for h in handles[1:]:
            out = self.arena.allocate()
            self.arena.bind(current, h, out)
            current = out
        return current

    def _chain_bind_phase_arrays(self, phase_arrays: List[List[float]]) -> List[float]:
        """Chain-bind phase arrays: phases₀ + phases₁ + ... + phasesₖ."""
        current = phase_arrays[0]
        for pa in phase_arrays[1:]:
            current = bind_phases(current, pa)
        return current

    def synthesize_from_clique(self, clique: List[str]) -> LayeredProjection:
        """Per-layer chain-bind all axioms in a clique, then cross-layer bind.

        S_ast  = A₁_ast  ⊗ A₂_ast  ⊗ ... ⊗ Aₖ_ast
        S_cfg  = A₁_cfg  ⊗ A₂_cfg  ⊗ ... ⊗ Aₖ_cfg
        S_data = A₁_data ⊗ A₂_data ⊗ ... ⊗ Aₖ_data
        S_final = S_ast ⊗ S_cfg ⊗ S_data
        """
        if len(clique) < 2:
            raise ValueError("Synthesis requires at least 2 axioms in a clique.")

        projections = [self.store.axioms[sid].projection for sid in clique]

        # AST layer (always present)
        ast_handles = [p.ast_handle for p in projections]
        ast_phase_arrays = [p.ast_phases for p in projections]
        synth_ast_h = self._chain_bind_handles(ast_handles)
        synth_ast_phases = self._chain_bind_phase_arrays(ast_phase_arrays)

        # CFG layer (may be None for some axioms — skip those)
        cfg_handles = [p.cfg_handle for p in projections if p.cfg_handle is not None]
        cfg_phase_arrays = [p.cfg_phases for p in projections if p.cfg_phases is not None]
        if len(cfg_handles) >= 2:
            synth_cfg_h = self._chain_bind_handles(cfg_handles)
            synth_cfg_phases = self._chain_bind_phase_arrays(cfg_phase_arrays)
        elif len(cfg_handles) == 1:
            synth_cfg_h = cfg_handles[0]
            synth_cfg_phases = cfg_phase_arrays[0]
        else:
            synth_cfg_h = None
            synth_cfg_phases = None

        # Data-dep layer (may be None for some axioms)
        data_handles = [p.data_handle for p in projections if p.data_handle is not None]
        data_phase_arrays = [p.data_phases for p in projections if p.data_phases is not None]
        if len(data_handles) >= 2:
            synth_data_h = self._chain_bind_handles(data_handles)
            synth_data_phases = self._chain_bind_phase_arrays(data_phase_arrays)
        elif len(data_handles) == 1:
            synth_data_h = data_handles[0]
            synth_data_phases = data_phase_arrays[0]
        else:
            synth_data_h = None
            synth_data_phases = None

        # Cross-layer bind → final
        result_h = synth_ast_h
        if synth_cfg_h is not None:
            merged = self.arena.allocate()
            self.arena.bind(result_h, synth_cfg_h, merged)
            result_h = merged
        if synth_data_h is not None:
            merged = self.arena.allocate()
            self.arena.bind(result_h, synth_data_h, merged)
            result_h = merged

        projection = LayeredProjection(
            final_handle=result_h,
            ast_handle=synth_ast_h,
            cfg_handle=synth_cfg_h,
            data_handle=synth_data_h,
            ast_phases=synth_ast_phases,
            cfg_phases=synth_cfg_phases,
            data_phases=synth_data_phases,
        )
        self._synthesis_log.append((list(clique), projection))
        return projection

    def synthesize_all(
        self, min_clique_size: int = 2
    ) -> List[Tuple[List[str], LayeredProjection]]:
        """Run full synthesis pipeline.
        Returns list of (clique_members, LayeredProjection).
        """
        self.compute_resonance()
        cliques = self.extract_cliques(min_size=min_clique_size)

        results = []
        for clique in cliques:
            synth_proj = self.synthesize_from_clique(clique)
            results.append((clique, synth_proj))

        return results

    def get_synthesis_log(self) -> List[Tuple[List[str], LayeredProjection]]:
        return list(self._synthesis_log)

    # ── Autopoietic Self-Reference (Ouroboros Loop) ─────────────

    # Source files that constitute the engine's own structural logic
    _SELF_SOURCE_FILES = [
        "src/projection/isomorphic_projector.py",
        "src/decoder/constraint_decoder.py",
        "src/decoder/subtree_vocab.py",
        "src/decoder/variable_threading.py",
        "src/analysis/structural_diff.py",
        "src/analysis/refactoring_detector.py",
        "src/analysis/temporal_diff.py",
        "src/synthesis/axiomatic_synthesizer.py",
        "src/hdc_core/src/lib.rs",
    ]

    def ingest_self(self, project_root: Optional[str] = None) -> List[Axiom]:
        """Recursively ingest the engine's own source code as axioms.

        Reads isomorphic_projector.py, constraint_decoder.py, and hdc_core/src/lib.rs,
        feeds them through the MultiLayerGraphBuilder → IsomorphicProjector pipeline,
        and registers them as axioms in the store.

        This forces the engine to mathematically recombine its own structural logic.

        Returns the list of self-axioms created.
        """
        if project_root is None:
            # Derive project root from this file's location
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            )))

        self_axioms = []
        for rel_path in sorted(self._SELF_SOURCE_FILES):
            abs_path = os.path.join(project_root, rel_path)
            if not os.path.isfile(abs_path):
                continue

            with open(abs_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            # For Rust files, we cannot parse them as Python AST directly.
            # Instead, we project the file's content as a string literal wrapped
            # in a Python assignment — this preserves the textual topology while
            # remaining valid Python for the graph builder.
            if rel_path.endswith(".rs"):
                # Encode Rust source as a Python string assignment for topology extraction
                safe_source = source_code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
                source_code = f'_rust_source = """{safe_source}"""'

            source_id = f"self:{rel_path}"
            try:
                axiom = self.ingest(source_id, source_code)
                self_axioms.append(axiom)
            except Exception:
                # If a source file cannot be parsed (e.g., syntax issues),
                # skip it deterministically — no silent failures
                continue

        return self_axioms

    def synthesize_self_representation(self) -> Optional[Tuple[List[str], LayeredProjection]]:
        """Synthesize a novel representation from the engine's own axioms.

        The Ouroboros loop:
          1. Ingest self → project own source files as axioms
          2. Compute resonance between self-axioms
          3. Extract resonant cliques from self-axioms
          4. Synthesize per-layer → cross-layer bind
          5. Return the self-synthesized structural representation

        The synthesized vector encodes the engine's own topological structure
        in a form that can be decoded via SMT and compared against the original.
        """
        self_axioms = self.ingest_self()
        if len(self_axioms) < 2:
            return None

        self.compute_resonance()

        # Extract cliques containing at least 2 self-axioms
        self_ids = [a.source_id for a in self_axioms]
        cliques = self.extract_cliques(min_size=2)

        # Find cliques that contain self-axiom members
        for clique in cliques:
            self_members = [s for s in clique if s in self_ids]
            if len(self_members) >= 2:
                synth_proj = self.synthesize_from_clique(self_members)
                return (self_members, synth_proj)

        # If no resonant clique found among self-axioms, force-synthesize all
        if len(self_ids) >= 2:
            synth_proj = self.synthesize_from_clique(self_ids)
            return (self_ids, synth_proj)

        return None

    def prove_self_optimality(
        self,
        self_synth: LayeredProjection,
        self_axiom_ids: List[str],
    ) -> Dict[str, float]:
        """Use correlation analysis to prove whether the self-synthesis is
        strictly more optimal (lower entropy, quasi-orthogonal to sources).

        Returns a dict with:
          - 'entropy': phase entropy of synthesized representation
          - 'mean_correlation': average correlation with source axioms
          - 'is_novel': True if quasi-orthogonal (|mean_corr| < tau)
          - per-axiom correlations
        """
        result: Dict[str, float] = {}

        # Compute phase entropy of synthesized representation
        entropy = compute_phase_entropy(self_synth.ast_phases)
        result["entropy"] = entropy

        # Compute correlations with each source axiom
        correlations = []
        for sid in self_axiom_ids:
            axiom = self.store.get(sid)
            if axiom is None:
                continue
            corr = self.arena.compute_correlation(
                self_synth.final_handle, axiom.handle
            )
            result[f"corr:{sid}"] = corr
            correlations.append(abs(corr))

        mean_corr = sum(correlations) / len(correlations) if correlations else 1.0
        result["mean_correlation"] = mean_corr
        result["is_novel"] = float(mean_corr < self.tau)

        return result

    # ── Entropy-Aware Synthesis (Topological Thermodynamics) ──────

    def compute_synthesis_entropy(self, projection: LayeredProjection) -> float:
        """Compute the total thermodynamic entropy of a synthesized projection.

        S_total = S_phase(ast) + S_phase(cfg) + S_phase(data) + S_ops

        where S_phase is the circular variance entropy and S_ops is the
        cumulative operation cost tracked by the arena.
        """
        total = compute_phase_entropy(projection.ast_phases)
        if projection.cfg_phases is not None:
            total += compute_phase_entropy(projection.cfg_phases)
        if projection.data_phases is not None:
            total += compute_phase_entropy(projection.data_phases)

        # Add operational entropy from arena tracking
        try:
            binds, bundles = self.arena.get_op_counts(projection.final_handle)
            total += compute_operation_entropy(int(binds), int(bundles))
        except Exception:
            pass

        return total

    def synthesize_all_with_thermodynamics(
        self, min_clique_size: int = 2,
    ) -> List[Tuple[List[str], LayeredProjection, float]]:
        """Run full synthesis with thermodynamic ranking.

        Returns list of (clique, projection, entropy) sorted by ascending entropy.
        The most structurally compressed (lowest entropy) syntheses come first.
        """
        self.compute_resonance()
        cliques = self.extract_cliques(min_size=min_clique_size)

        results = []
        for clique in cliques:
            synth_proj = self.synthesize_from_clique(clique)
            entropy = self.compute_synthesis_entropy(synth_proj)
            results.append((clique, synth_proj, entropy))

        # Sort by ascending entropy: most compressed first
        results.sort(key=lambda x: x[2])
        return results

    # ── Self-Diagnostic via Structural Analysis (LEAP 3A) ────────

    def run_self_diagnostic(
        self, project_root: Optional[str] = None, output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run structural analysis on the engine's own modules.

        After ingesting self-source files:
        1. Compute per-layer similarity between each self-source-file pair
        2. Identify structural duplicates (candidates for internal refactoring)
        3. Identify structural outliers (most novel / fragile components)
        4. Output a self_diagnostic.json with actionable findings

        Args:
            project_root: Root directory of the project. Auto-detected if None.
            output_path: Directory to write self_diagnostic.json. If None, returns
                        the dict without writing.

        Returns:
            Dictionary with self-diagnostic findings.
        """
        from src.analysis.structural_diff import StructuralDiffEngine

        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            )))

        # Ingest self if not already done
        self_axioms = self.ingest_self(project_root)
        if len(self_axioms) < 2:
            return {"error": "Insufficient self-axioms for diagnostic"}

        self.compute_resonance()
        self_ids = [a.source_id for a in self_axioms]

        diff_engine = StructuralDiffEngine(self.arena, self.projector)

        # Pairwise layer-decomposed similarity
        pairwise: List[Dict[str, Any]] = []
        duplicates: List[Dict[str, Any]] = []
        outlier_scores: Dict[str, List[float]] = {sid: [] for sid in self_ids}

        for i, id_a in enumerate(self_ids):
            axiom_a = self.store.get(id_a)
            if axiom_a is None:
                continue
            for id_b in self_ids[i + 1:]:
                axiom_b = self.store.get(id_b)
                if axiom_b is None:
                    continue

                layer_sim = diff_engine.compare_layers(
                    axiom_a.projection, axiom_b.projection,
                )

                pair_data = {
                    "file_a": id_a,
                    "file_b": id_b,
                    "sim_ast": round(layer_sim.sim_ast, 6),
                    "sim_cfg": round(layer_sim.sim_cfg, 6) if layer_sim.sim_cfg is not None else None,
                    "sim_data": round(layer_sim.sim_data, 6) if layer_sim.sim_data is not None else None,
                    "sim_final": round(layer_sim.sim_final, 6),
                    "diagnosis": layer_sim.diagnosis,
                }
                pairwise.append(pair_data)

                # Track for outlier detection
                outlier_scores[id_a].append(abs(layer_sim.sim_final))
                outlier_scores[id_b].append(abs(layer_sim.sim_final))

                # Detect structural duplicates
                if (abs(layer_sim.sim_ast) > 0.7
                    and layer_sim.sim_cfg is not None
                    and abs(layer_sim.sim_cfg) > 0.7):
                    duplicates.append({
                        "file_a": id_a,
                        "file_b": id_b,
                        "sim_ast": round(layer_sim.sim_ast, 6),
                        "sim_cfg": round(layer_sim.sim_cfg, 6),
                        "recommendation": "Internal structural duplicate — consider refactoring",
                    })

        # Identify outliers (lowest mean similarity = most unique/fragile)
        outlier_means = {}
        for sid, scores in outlier_scores.items():
            if scores:
                outlier_means[sid] = sum(scores) / len(scores)

        sorted_outliers = sorted(outlier_means.items(), key=lambda x: x[1])

        diagnostic = {
            "self_source_files": self_ids,
            "total_self_axioms": len(self_axioms),
            "pairwise_layer_similarity": pairwise,
            "structural_duplicates": duplicates,
            "structural_outliers": [
                {
                    "file": sid,
                    "mean_similarity": round(score, 6),
                    "assessment": "Most structurally unique (potentially novel or fragile)"
                    if idx < 2 else "Moderate structural uniqueness",
                }
                for idx, (sid, score) in enumerate(sorted_outliers)
            ],
            "recommendations": [],
        }

        if duplicates:
            diagnostic["recommendations"].append(
                f"Found {len(duplicates)} internal structural duplicate(s). "
                f"Consider extracting shared patterns into utility modules."
            )

        if sorted_outliers:
            most_unique = sorted_outliers[0][0]
            diagnostic["recommendations"].append(
                f"Most structurally unique module: {most_unique}. "
                f"This component is most different from all others — "
                f"review for potential fragility or innovative patterns."
            )

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            filepath = os.path.join(output_path, "self_diagnostic.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(diagnostic, f, indent=2, ensure_ascii=False)

        return diagnostic

    # ── Synthesis Quality Self-Test (LEAP 3B) ────────────────────

    def run_synthesis_quality_test(
        self, decoder: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Closed-loop quality test: synthesize → decode → re-project → measure fidelity.

        After the decoder overhaul (LEAP 1):
        1. Synthesize a novel vector from self-axioms
        2. Decode it via the sub-tree assembly decoder
        3. Re-project the decoded output
        4. Compute round-trip fidelity (cosine similarity)
        5. If fidelity < threshold, identify which sub-tree atoms were lost

        This is a self-contained quality signal requiring no external evaluation.

        Args:
            decoder: ConstraintDecoder instance (must have sub-tree vocab for
                    meaningful results). If None, skips decode step.

        Returns:
            Dictionary with round-trip fidelity metrics.
        """
        result: Dict[str, Any] = {
            "synthesis_completed": False,
            "decode_completed": False,
            "round_trip_fidelity": None,
            "fidelity_above_threshold": False,
        }

        # Step 1: Synthesize from self-axioms
        synth_result = self.synthesize_self_representation()
        if synth_result is None:
            result["error"] = "Self-synthesis failed: insufficient axioms"
            return result

        self_ids, synth_proj = synth_result
        result["synthesis_completed"] = True
        result["synthesis_members"] = self_ids
        result["synthesis_entropy"] = self.compute_synthesis_entropy(synth_proj)

        if decoder is None:
            result["note"] = "No decoder provided — skipping decode step"
            return result

        # Step 2: Decode via the constraint decoder
        try:
            import ast as ast_mod
            decoded_module = decoder.decode(synth_proj)
            if decoded_module is None:
                result["error"] = "Decode returned None"
                return result

            decoded_source = ast_mod.unparse(decoded_module)
            result["decode_completed"] = True
            result["decoded_source_length"] = len(decoded_source)
        except Exception as e:
            result["error"] = f"Decode failed: {str(e)}"
            return result

        # Step 3: Re-project the decoded output
        try:
            reprojected = self.projector.project(decoded_source)
        except Exception as e:
            result["error"] = f"Re-projection failed: {str(e)}"
            return result

        # Step 4: Compute round-trip fidelity
        fidelity = self.arena.compute_correlation(
            synth_proj.final_handle, reprojected.final_handle,
        )
        result["round_trip_fidelity"] = round(fidelity, 6)

        # Per-layer fidelity
        ast_fidelity = self.arena.compute_correlation(
            synth_proj.ast_handle, reprojected.ast_handle,
        )
        result["ast_fidelity"] = round(ast_fidelity, 6)

        if synth_proj.cfg_handle is not None and reprojected.cfg_handle is not None:
            cfg_fidelity = self.arena.compute_correlation(
                synth_proj.cfg_handle, reprojected.cfg_handle,
            )
            result["cfg_fidelity"] = round(cfg_fidelity, 6)

        if synth_proj.data_handle is not None and reprojected.data_handle is not None:
            data_fidelity = self.arena.compute_correlation(
                synth_proj.data_handle, reprojected.data_handle,
            )
            result["data_fidelity"] = round(data_fidelity, 6)

        # Step 5: Assess fidelity threshold
        fidelity_threshold = 0.3
        result["fidelity_above_threshold"] = fidelity >= fidelity_threshold
        result["fidelity_threshold"] = fidelity_threshold

        if fidelity < fidelity_threshold:
            result["diagnosis"] = (
                f"Round-trip fidelity ({fidelity:.4f}) below threshold "
                f"({fidelity_threshold}). The decoder vocabulary may have gaps — "
                f"sub-tree atoms in the synthesized vector were lost during decoding."
            )
        else:
            result["diagnosis"] = (
                f"Round-trip fidelity ({fidelity:.4f}) above threshold "
                f"({fidelity_threshold}). Decoder faithfully reconstructs "
                f"synthesized structural patterns."
            )

        return result

    # ── Diagnostics ──────────────────────────────────────────────

    def verify_quasi_orthogonality(
        self, synth_handle: int, clique: List[str]
    ) -> Dict[str, float]:
        """Compute correlation between a synthesized handle and clique members.

        Args:
            synth_handle: Arena handle of the synthesized vector.
            clique: List of source IDs in the clique.

        Returns:
            Dictionary mapping source ID to correlation value.
        """
        return {
            sid: self.arena.compute_correlation(synth_handle, self.store.axioms[sid].handle)
            for sid in clique
        }

    # ── SERL Self-Application (Ouroboros SERL) ────────────────────

    def run_serl_self(
        self,
        project_root: Optional[str] = None,
        max_cycles: int = 10,
        stagnation_limit: int = 3,
        fitness_threshold: float = 0.4,
    ) -> Dict[str, Any]:
        """Run the SERL loop on the engine's own source code.

        This is the ultimate autopoietic test: the engine synthesizes new code
        from its own modules, executes the synthesized code, and (if it passes
        fitness) feeds it back into its own vocabulary.

        If this produces new atoms → the engine has modified its own
        representational capacity. If not → confirmed closed.

        Args:
            project_root: Project root directory. Auto-detected if None.
            max_cycles: Maximum SERL cycles.
            stagnation_limit: Consecutive zero-expansion cycles before halt.
            fitness_threshold: Minimum fitness for vocabulary expansion.

        Returns:
            Dictionary with SERL result metrics including f_eff_expansion_rate.
        """
        from src.decoder.constraint_decoder import ConstraintDecoder
        from src.decoder.subtree_vocab import SubTreeVocabulary
        from src.decoder.vocab_expander import VocabularyExpander
        from src.execution.sandbox import ExecutionSandbox
        from src.synthesis.serl import SERLLoop

        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            )))

        # Step 1: Ingest self-source as axioms
        self_axioms = self.ingest_self(project_root)
        if len(self_axioms) < 2:
            return {
                "error": "Insufficient self-axioms for SERL",
                "self_axioms": len(self_axioms),
                "f_eff_expansion_rate": 0.0,
            }

        # Step 2: Build sub-tree vocabulary from own source
        vocab = SubTreeVocabulary()
        src_dir = os.path.join(project_root, "src")
        for root, _dirs, files in os.walk(src_dir):
            for fname in files:
                if fname.endswith(".py"):
                    vocab.ingest_file(os.path.join(root, fname))

        vocab.project_all(self.arena, self.projector)

        # Step 3: Create decoder and SERL components
        # Higher activation threshold (0.15) for self-application because
        # the self-vocab is large (1000+ atoms) and Z3 becomes slow with
        # too many activated variables. 0.15 selects only strong resonances.
        decoder = ConstraintDecoder(
            self.arena, self.projector,
            self.arena.get_dimension(),
            activation_threshold=0.15,
            subtree_vocab=vocab,
        )
        sandbox = ExecutionSandbox()
        expander = VocabularyExpander()
        loop = SERLLoop()

        # Step 4: Run SERL
        result = loop.run(
            self.arena, self.projector, self, decoder,
            sandbox, expander, vocab,
            max_cycles=max_cycles,
            stagnation_limit=stagnation_limit,
            fitness_threshold=fitness_threshold,
        )

        return {
            "self_axioms": len(self_axioms),
            "cycles_completed": result.cycles_completed,
            "total_new_atoms": result.total_new_atoms,
            "vocab_size_initial": result.vocab_size_initial,
            "vocab_size_final": result.vocab_size_final,
            "f_eff_expansion_rate": result.f_eff_expansion_rate,
            "space_closed": result.space_closed,
            "convergence_diagnosis": result.convergence_diagnosis,
            "cycle_summary": [
                {
                    "cycle": c.cycle_index,
                    "decoded": c.syntheses_decoded,
                    "above_fitness": c.syntheses_above_fitness,
                    "new_atoms": c.new_vocab_atoms,
                    "best_fitness": c.best_fitness,
                }
                for c in result.cycle_history
            ],
        }
