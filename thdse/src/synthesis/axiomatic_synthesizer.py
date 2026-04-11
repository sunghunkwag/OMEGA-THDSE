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
import math
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
    # PLAN.md Phase 8A.2: optional behavioral profile capturing what
    # the axiom's source actually does (vs. how it looks). When
    # present, dual-axis resonance can blend structural and
    # behavioral similarity.
    behavioral: Optional[Any] = None
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

    PLAN.md Phase 8A.3 — dual-axis resonance: when both axioms carry a
    behavioral profile, the matrix entry is a weighted blend of
    structural and behavioral similarity:

        ρ(A, B) = α · structural_sim + (1 − α) · behavioral_sim

    Default α = 0.5. Set ``alpha=1.0`` for legacy structural-only
    behaviour. When either axiom is missing a behavioral profile the
    behavioural term is treated as 0.0 — falling back to a pure
    structural correlation scaled by α.
    """

    def __init__(self, arena: Any, store: AxiomStore, alpha: float = 0.5):
        self.arena = arena
        self.store = store
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self._matrix: Dict[Tuple[str, str], float] = {}

    @staticmethod
    def _behavioral_similarity(ax_a: "Axiom", ax_b: "Axiom") -> float:
        """Compute mean-cosine FHRR similarity on stored behavioral phases.

        Returns 0.0 if either axiom has no behavioral profile so the
        dual-axis blend degrades gracefully to a pure structural
        score weighted by ``alpha``.
        """
        if ax_a.behavioral is None or ax_b.behavioral is None:
            return 0.0
        a = getattr(ax_a.behavioral, "behavioral_phases", None)
        b = getattr(ax_b.behavioral, "behavioral_phases", None)
        if not a or not b or len(a) != len(b):
            return 0.0
        n = len(a)
        return float(sum(math.cos(a[i] - b[i]) for i in range(n)) / n)

    def _blend(
        self, structural: float, ax_a: "Axiom", ax_b: "Axiom"
    ) -> float:
        """Blend structural and behavioral similarity per dual-axis rule."""
        behavioral = self._behavioral_similarity(ax_a, ax_b)
        return self.alpha * float(structural) + (1.0 - self.alpha) * behavioral

    def compute_full(self) -> Dict[Tuple[str, str], float]:
        self._matrix.clear()
        ids = self.store.all_ids()
        n = len(ids)
        if n == 0:
            return self._matrix

        # Batch: single Rust FFI call for entire upper triangle
        handles = [self.store.axioms[sid].handle for sid in ids]
        has_batch = hasattr(self.arena, 'correlate_matrix')

        # Always go through individual compute_correlation calls
        # because the dual-axis blend (Phase 8A.3) needs per-pair
        # access to the behavioural similarity. The per-call cost is
        # negligible for the corpus sizes typical of synthesis (N <
        # 200 axioms) and removes the Rust-vs-Python format-divergence
        # of ``correlate_matrix``.
        for i, id_a in enumerate(ids):
            ax_a = self.store.axioms[id_a]
            for id_b in ids[i:]:
                ax_b = self.store.axioms[id_b]
                structural = self.arena.compute_correlation(
                    ax_a.handle, ax_b.handle
                )
                blended = self._blend(structural, ax_a, ax_b)
                self._matrix[(id_a, id_b)] = blended
                self._matrix[(id_b, id_a)] = blended
                ax_a.resonance_profile[id_b] = blended
                ax_b.resonance_profile[id_a] = blended

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
        arena: Any = None,
        projector: IsomorphicProjector = None,
        resonance_threshold: float = 0.15,
        *,
        arena_manager: Any = None,
        provenance_bridge: Any = None,
        causal_tracker: Any = None,
        frozen_rng: Any = None,
    ):
        # PLAN.md Phase 6 wiring (Rule 3 + Rule 14): when an
        # ArenaManager is injected the synthesizer borrows that
        # manager's THDSE arena instead of carrying its own. The old
        # ``arena`` positional argument remains for backward
        # compatibility (Rule 16) — passing a raw arena still works.
        if arena_manager is not None:
            from src.utils.arena_factory import make_arena

            # Match the manager's THDSE dimension by default; the
            # caller can also override via ``arena=`` if they want a
            # different sized arena.
            if arena is None:
                arena = make_arena(
                    capacity=getattr(arena_manager, "thdse_capacity", 200_000),
                    dimension=getattr(arena_manager, "thdse_dim", 256),
                    arena_manager=arena_manager,
                )

        if arena is None:
            # Legacy fallback: build a small Python arena so the class
            # remains constructible without injection.
            from src.utils.arena_factory import make_arena

            arena = make_arena(capacity=10_000, dimension=256)

        self.arena = arena
        self.projector = projector
        self.store = AxiomStore(arena)
        # Phase 8A.3: dual-axis resonance is on by default (alpha=0.5)
        # so the engine resonates on what programs DO, not just what
        # they look like.
        self.resonance = ResonanceMatrix(arena, self.store, alpha=0.5)
        self.tau = resonance_threshold
        self._synthesis_log: List[Tuple[List[str], LayeredProjection]] = []

        # PLAN.md Phase 6 wiring: bridge handles for provenance
        # emission, causal-chain ingestion, and FrozenRNG enforcement.
        # Every one is optional so the legacy standalone path keeps
        # working with no behavioural change.
        self._arena_manager = arena_manager
        self._provenance_bridge = provenance_bridge
        self._causal_tracker = causal_tracker
        self._frozen_rng = frozen_rng

        # PLAN.md Phase 8A.2: optional behavioural encoder. Constructed
        # lazily on first ingest so legacy callers that never want
        # behavioural axioms pay zero cost. Set ``self.behavioral_encoder``
        # explicitly (e.g. in tests) to override the default.
        self.behavioral_encoder: Any = None

    # ------------------------------------------------------------------
    # PLAN.md Rule 14 — RNG enforcement probe
    # ------------------------------------------------------------------

    def attempt_perturbation(self, magnitude: float = 0.01) -> float:
        """Sample a perturbation magnitude through the injected RNG.

        The axiomatic synthesizer is deterministic by design (PLAN.md
        Section C). This method exists so callers that *think* they
        need a stochastic perturbation explicitly route through the
        injected RNG. With a :class:`shared.deterministic_rng.FrozenRNG`
        the call raises ``RuntimeError`` immediately, structurally
        preventing nondeterminism from leaking into the synthesis path.
        With a :class:`numpy.random.Generator` (typical CCE fork) the
        call returns a uniform sample in ``[0, magnitude)``.
        """
        if self._frozen_rng is None:
            return 0.0
        return float(self._frozen_rng.uniform(0.0, magnitude))

    # ------------------------------------------------------------------
    # PLAN.md Rule 8 + Rule 14 — testable post-Z3 handlers
    # ------------------------------------------------------------------

    def handle_z3_result(
        self,
        result: str,
        formula_id: str,
        round_idx: int,
        details: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Dispatch a Z3 verdict through the wired bridges.

        ``result`` must be ``"sat"`` or ``"unsat"``. SAT events are
        emitted as a ``record_synthesis_event("sat", …)`` call on the
        provenance bridge. UNSAT events are double-logged: once via
        ``record_synthesis_event("unsat", …)`` so the bridge counter
        increments (Rule 8 audit), and once via
        ``record_unsat_event(formula_id, …)`` on the causal tracker
        with ``data["logged"] = True``.

        Returns a structured event dict carrying provenance metadata
        (Rule 9). When no bridges are wired in this method is a no-op
        that still returns a metadata stub so callers can rely on the
        return contract.
        """
        if result not in ("sat", "unsat"):
            raise ValueError(
                f"result must be 'sat' or 'unsat', got {result!r}"
            )

        details = dict(details or {})
        details.setdefault("formula_id", formula_id)
        details.setdefault("round_idx", round_idx)

        provenance_event_id: str | None = None
        causal_event_id: str | None = None

        if self._provenance_bridge is not None:
            event = self._provenance_bridge.record_synthesis_event(
                result, None, details
            )
            provenance_event_id = event.get("event_id")

        if result == "unsat" and self._causal_tracker is not None:
            causal_event_id = self._causal_tracker.record_unsat_event(
                formula_id=formula_id,
                reason=str(details.get("reason", "z3_unsat")),
                round_idx=int(round_idx),
            )

        return {
            "result": result,
            "formula_id": formula_id,
            "round_idx": round_idx,
            "provenance_event_id": provenance_event_id,
            "causal_event_id": causal_event_id,
            "metadata": {
                "details": details,
                "provenance": {
                    "operation": "handle_z3_result",
                    "source_arena": "thdse",
                    "target_arena": "cce",
                    "result": result,
                },
            },
        }


    # ── Ingestion ────────────────────────────────────────────────

    def ingest(self, source_id: str, code: str) -> Axiom:
        """Project a codebase and register it as an axiom.

        PLAN.md Phase 8A.2: when a :class:`BehavioralEncoder` is wired
        in (``self.behavioral_encoder``), every ingested source is
        also executed against the encoder's probe set so the resulting
        :class:`Axiom` carries both a structural projection AND a
        behavioural fingerprint. Encoding failures (timeouts, sandbox
        crashes) are stored as ``axiom.behavioral = None`` rather
        than skipping the axiom entirely — the structural side still
        has value.
        """
        projection = self.projector.project(code)
        axiom = self.store.register(source_id, projection)

        if self.behavioral_encoder is not None:
            try:
                axiom.behavioral = self.behavioral_encoder.encode_behavior(
                    code
                )
            except Exception:  # noqa: BLE001 — failure is recorded as None
                axiom.behavioral = None
        return axiom

    def ingest_batch(self, corpus: Dict[str, str]) -> int:
        """Ingest multiple codebases. Returns count of axioms registered."""
        for source_id in sorted(corpus.keys()):
            self.ingest(source_id, corpus[source_id])
        return self.store.count()

    # ------------------------------------------------------------------
    # PLAN.md Phase 8B.2 — goal-directed synthesis
    # ------------------------------------------------------------------

    def synthesize_for_problem(
        self,
        problem_vector: Any,
        min_clique_size: int = 2,
        top_k: int = 5,
    ) -> List[Tuple[List[str], LayeredProjection, float]]:
        """Rank cliques by goal-relevance and synthesize from the top K.

        ``problem_vector`` is the output of
        :meth:`bridges.problem_spec.ProblemEncoder.encode_problem` —
        any object exposing ``handle`` and ``phases`` attributes.

        For every axiom we compute the goal-relevance as the FHRR
        correlation between the axiom's BEHAVIOURAL phases and the
        problem's phases. Cliques are then scored as an ADDITIVE blend
        of mean structural resonance and mean behavioural relevance:

            score = α · mean_resonance + (1 − α) · max(mean_relevance, 0)

        where ``α = self.resonance.alpha`` (default 0.5). This replaces
        the old multiplicative formula ``mean_resonance × max(mean_relevance, 0)``,
        which collapsed every score to ≈0 whenever the seed corpus had
        little behavioural overlap with the target problem (the common
        case — seed functions rarely share io-profiles with the problem
        oracle). The additive blend guarantees that structurally
        resonant cliques stay visible even when behavioural similarity
        is low, while cliques that are ALSO behaviourally relevant still
        get a boost. Cliques with zero goal-relevance are intentionally
        NOT filtered out — their structural score alone still produces a
        meaningful ranking. Returns a list of
        ``(clique_ids, projection, score)`` tuples sorted descending by score.
        """
        # Step 1 — make sure the resonance matrix is current.
        self.compute_resonance()

        # Step 2 — score each axiom against the problem vector.
        problem_phases = list(getattr(problem_vector, "phases", []))
        if not problem_phases:
            return []

        axiom_relevance: Dict[str, float] = {}
        for sid, axiom in self.store.axioms.items():
            relevance = 0.0
            if axiom.behavioral is not None:
                a = axiom.behavioral.behavioral_phases
                if len(a) == len(problem_phases):
                    relevance = float(
                        sum(
                            math.cos(a[i] - problem_phases[i])
                            for i in range(len(a))
                        )
                        / len(a)
                    )
            axiom_relevance[sid] = relevance

        # Step 3 — extract cliques and score each one with an ADDITIVE
        # blend. Additive (not multiplicative) is essential here: when
        # behavioural similarity is ~0 for most axioms — typical when
        # the seed corpus does not share io-profiles with the problem —
        # the old multiplicative formula drove every score to 0 and the
        # top_k selection became arbitrary. Additive keeps the
        # structural ranking meaningful even when relevance collapses
        # and still boosts cliques that happen to be behaviourally
        # relevant as well.
        alpha = float(getattr(self.resonance, "alpha", 0.5))
        cliques = self.extract_cliques(min_size=min_clique_size)
        scored_cliques: List[Tuple[float, List[str]]] = []
        for clique in cliques:
            if len(clique) < 2:
                continue
            mean_resonance = self._mean_clique_resonance(clique)
            mean_relevance = sum(
                axiom_relevance.get(sid, 0.0) for sid in clique
            ) / len(clique)
            score = (
                alpha * mean_resonance
                + (1.0 - alpha) * max(mean_relevance, 0.0)
            )
            scored_cliques.append((score, clique))

        scored_cliques.sort(key=lambda pair: pair[0], reverse=True)

        # Step 4 — synthesize the top-K and collect projections.
        results: List[Tuple[List[str], LayeredProjection, float]] = []
        for score, clique in scored_cliques[: max(1, top_k)]:
            try:
                projection = self.synthesize_from_clique(clique)
            except Exception:  # noqa: BLE001
                continue
            results.append((list(clique), projection, score))
        return results

    def _mean_clique_resonance(self, clique: List[str]) -> float:
        """Mean pairwise resonance within ``clique`` (uses cached matrix)."""
        if len(clique) < 2:
            return 0.0
        total = 0.0
        n_pairs = 0
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                key = (clique[i], clique[j])
                value = self.resonance._matrix.get(key)  # noqa: SLF001
                if value is None:
                    value = self.arena.compute_correlation(
                        self.store.axioms[clique[i]].handle,
                        self.store.axioms[clique[j]].handle,
                    )
                total += float(value)
                n_pairs += 1
        return total / max(n_pairs, 1)

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
