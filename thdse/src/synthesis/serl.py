"""SERL — Synthesis-Execution-Reingestion Loop.

Closed-loop orchestrator that connects synthesis → decode → execute → evaluate
→ vocabulary expansion → next cycle. This is the mechanism by which the system
achieves (or fails to achieve) design space self-expansion.

Each cycle:
  1. Synthesize from resonance cliques (existing pipeline)
  2. Decode each synthesis to Python source (existing pipeline)
  3. Execute each decoded source in the sandbox (Step 1)
  4. For sources with fitness > threshold:
     a. Expand the sub-tree vocabulary (Step 2)
  5. Measure: did vocabulary grow? (F_eff expansion check)
  6. If vocabulary grew → next cycle with expanded design space
     If no growth for K consecutive cycles → space is closed, halt

The F_eff measurement is the scientific output:
  f_eff_expansion_rate = total_new_atoms / total_syntheses_attempted
  > 0 → system is demonstrably open (at least partially)
  = 0 → system is empirically closed (valid scientific result)

Zero randomness. Deterministic at every step.

PLAN.md Phase 6 wiring: SERL accepts injected bridge handles
(``arena_manager``, ``rsi_serl_bridge``, ``governance_bridge``,
``axiom_skill_bridge``, ``provenance_bridge``, ``causal_tracker``,
``deterministic_rng``). When wired, every fitness-gated candidate
flows through ``RsiSerlBridge.serl_candidate_to_rsi`` → optional
``GovernanceSynthesisBridge.evaluate_candidate`` → optional
``AxiomSkillBridge.validate_and_register``. All decode results — SAT
or UNSAT — generate causal-chain events through the provenance and
causal tracker handles. Tier-1 tests exercise the post-decode handler
(:meth:`SERLLoop.handle_decode_result`) directly with synthetic inputs.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# PLAN.md Phase 6 (Rule 13 revised): the heavy decoder dependencies
# transitively pull in z3, which is not always installed in Tier-1
# environments (bare Python + numpy). Defer those imports to module
# load time only when type-checking; at runtime they are imported
# inside ``run()``. This lets the SERLLoop class — and especially
# its testable post-decode handler — be exercised without Z3.
if TYPE_CHECKING:  # pragma: no cover — typing only
    from src.execution.sandbox import ExecutionSandbox
    from src.decoder.vocab_expander import VocabularyExpander
    from src.decoder.subtree_vocab import SubTreeVocabulary


@dataclass
class SERLCycleResult:
    """Result of a single SERL cycle."""
    cycle_index: int
    syntheses_attempted: int
    syntheses_decoded: int
    syntheses_executed: int
    syntheses_above_fitness: int    # passed fitness threshold
    new_vocab_atoms: int            # added to vocabulary this cycle
    vocab_size_before: int
    vocab_size_after: int
    f_eff_expanded: bool            # True if new_vocab_atoms > 0
    best_fitness: float
    best_source: Optional[str]
    residual_dimension: Optional[int] = None


@dataclass
class SERLResult:
    """Aggregate result of the full SERL loop."""
    cycles_completed: int
    total_new_atoms: int
    vocab_size_initial: int
    vocab_size_final: int
    f_eff_expansion_rate: float     # total_new_atoms / total syntheses attempted
    space_closed: bool              # True if K consecutive cycles produced 0 new atoms
    cycle_history: List[SERLCycleResult] = field(default_factory=list)
    convergence_diagnosis: str = ""


class SERLLoop:
    """Synthesis-Execution-Reingestion Loop.

    Each cycle:
      1. Synthesize from resonance cliques (existing pipeline)
      2. Decode each synthesis to Python source (existing pipeline)
      3. Execute each decoded source in the sandbox (Step 1)
      4. For sources with fitness > threshold:
         a. Expand the sub-tree vocabulary (Step 2)
      5. Measure: did vocabulary grow? (F_eff expansion check)
      6. If vocabulary grew → next cycle with expanded design space
         If no growth for K consecutive cycles → space is closed, halt

    The loop terminates when:
      - max_cycles reached, OR
      - K consecutive cycles with zero new vocab atoms (space closed), OR
      - no cliques found (nothing to synthesize)
    """

    def __init__(
        self,
        *,
        arena_manager: Any = None,
        rsi_serl_bridge: Any = None,
        governance_bridge: Any = None,
        axiom_skill_bridge: Any = None,
        provenance_bridge: Any = None,
        causal_tracker: Any = None,
        deterministic_rng: Any = None,
    ) -> None:
        """Phase 6 dependency injection (all parameters optional).

        Backward-compat (Rule 16): legacy callers that ``SERLLoop()`` and
        invoke ``run(...)`` keep working unchanged — every bridge handle
        defaults to ``None`` and the loop short-circuits the wired paths
        in that case.
        """
        self._arena_manager = arena_manager
        self._rsi_serl_bridge = rsi_serl_bridge
        self._governance_bridge = governance_bridge
        self._axiom_skill_bridge = axiom_skill_bridge
        self._provenance_bridge = provenance_bridge
        self._causal_tracker = causal_tracker
        self._deterministic_rng = deterministic_rng
        self._handler_invocations = 0
        self._registration_attempts = 0
        self._registrations_completed = 0

    # ------------------------------------------------------------------
    # PLAN.md Rule 14 — testable post-decode handler
    # ------------------------------------------------------------------

    def handle_decode_result(
        self,
        source: Optional[str],
        fitness: float,
        thdse_handle: int,
        round_idx: int,
        formula_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatch a single decode outcome through the wired bridges.

        ``source is None`` is the SERL convention for "the constraint
        solver returned UNSAT for this clique"; any non-empty source
        is the SAT branch. The handler is intentionally split out from
        :meth:`run` so Tier-1 tests can call it with synthetic decode
        results without needing to drive the entire synthesis stack.

        Returns a dict carrying:

        - ``result``: ``"sat"`` | ``"unsat"``
        - ``eligible``: bool — True if the candidate cleared the SERL
          fitness gate AND :class:`bridges.rsi_serl_bridge.RsiSerlBridge`
          accepted it.
        - ``approved``: bool — True if governance approved.
        - ``registered``: bool — True if axiom_skill_bridge registered.
        - ``metadata``: dict carrying provenance.
        """
        self._handler_invocations += 1
        formula_id = formula_id or f"serl_round_{round_idx}_h{thdse_handle}"
        result = "sat" if (source and source.strip()) else "unsat"

        # PLAN.md Rule 8: every UNSAT must hit both bridges if wired.
        if result == "unsat":
            if self._provenance_bridge is not None:
                self._provenance_bridge.record_synthesis_event(
                    "unsat",
                    int(thdse_handle),
                    {"formula_id": formula_id, "round_idx": round_idx},
                )
            if self._causal_tracker is not None:
                self._causal_tracker.record_unsat_event(
                    formula_id=formula_id,
                    reason="serl_decode_unsat",
                    round_idx=int(round_idx),
                )
            return {
                "result": "unsat",
                "eligible": False,
                "approved": False,
                "registered": False,
                "metadata": {
                    "formula_id": formula_id,
                    "round_idx": round_idx,
                    "provenance": {
                        "operation": "handle_decode_result",
                        "source_arena": "thdse",
                        "target_arena": "cce",
                        "result": "unsat",
                    },
                },
            }

        # SAT branch — emit provenance, then run RSI / governance / skill.
        if self._provenance_bridge is not None:
            self._provenance_bridge.record_synthesis_event(
                "sat",
                int(thdse_handle),
                {"formula_id": formula_id, "round_idx": round_idx},
            )

        eligible = False
        approved = False
        registered = False
        rsi_result: Dict[str, Any] = {}
        gov_result: Dict[str, Any] = {}
        registration: Dict[str, Any] = {}

        if self._rsi_serl_bridge is not None:
            rsi_result = self._rsi_serl_bridge.serl_candidate_to_rsi(
                source, float(fitness), int(thdse_handle)
            )
            eligible = bool(rsi_result.get("eligible"))

        if eligible and self._governance_bridge is not None:
            gov_result = self._governance_bridge.evaluate_candidate(
                source, int(thdse_handle), float(fitness)
            )
            approved = self._governance_bridge.gate_registration(gov_result)

        if approved and self._axiom_skill_bridge is not None:
            self._registration_attempts += 1
            skill_name = f"serl_skill_{thdse_handle}_{round_idx}"
            try:
                registration = (
                    self._axiom_skill_bridge.validate_and_register(
                        axiom_handle=int(thdse_handle),
                        program_source=source,
                        skill_name=skill_name,
                        governance_approved=True,
                    )
                )
                registered = bool(registration.get("registered"))
                if registered:
                    self._registrations_completed += 1
            except Exception as exc:  # noqa: BLE001
                # Registration failures are recorded as causal events
                # so the integration layer can audit them.
                if self._causal_tracker is not None:
                    self._causal_tracker.record_thdse_provenance(
                        source_arena="thdse",
                        operation="serl_registration_failed",
                        result=str(exc)[:120],
                        round_idx=int(round_idx),
                    )

        return {
            "result": "sat",
            "eligible": eligible,
            "approved": approved,
            "registered": registered,
            "metadata": {
                "formula_id": formula_id,
                "round_idx": round_idx,
                "rsi_result": rsi_result,
                "governance_result": gov_result,
                "registration": registration,
                "handler_invocations": self._handler_invocations,
                "provenance": {
                    "operation": "handle_decode_result",
                    "source_arena": "thdse",
                    "target_arena": "cce",
                    "result": "sat",
                },
            },
        }

    def feedback_skill_performance(
        self, skill_id: str, performance_scores: List[float]
    ) -> Dict[str, Any]:
        """Forward execution feedback to ``RsiSerlBridge`` for fitness shaping."""
        if self._rsi_serl_bridge is None:
            return {
                "feedback_applied": False,
                "metadata": {
                    "reason": "no_rsi_serl_bridge_wired",
                    "provenance": {
                        "operation": "feedback_skill_performance",
                        "source_arena": "cce",
                        "target_arena": "thdse",
                    },
                },
            }
        result = self._rsi_serl_bridge.rsi_skill_to_serl_feedback(
            skill_id, performance_scores
        )
        result["feedback_applied"] = True
        return result

    @property
    def handler_invocations(self) -> int:
        return self._handler_invocations

    @property
    def registration_attempts(self) -> int:
        return self._registration_attempts

    @property
    def registrations_completed(self) -> int:
        return self._registrations_completed

    def run(
        self,
        arena: Any,
        projector: Any,
        synthesizer: Any,
        decoder: Any,
        sandbox: "Any",
        expander: "Any",
        subtree_vocab: "Any",
        max_cycles: int = 20,
        stagnation_limit: int = 5,
        fitness_threshold: float = 0.4,
        min_clique_size: int = 2,
        max_cliques_per_cycle: int = 10,
    ) -> SERLResult:
        """Run the SERL loop.

        Args:
            arena: FHRR arena.
            projector: IsomorphicProjector.
            synthesizer: AxiomaticSynthesizer with ingested corpus.
            decoder: ConstraintDecoder with subtree vocabulary.
            sandbox: ExecutionSandbox for fitness evaluation.
            expander: VocabularyExpander for vocabulary growth.
            subtree_vocab: The SubTreeVocabulary being expanded.
            max_cycles: Maximum number of SERL cycles.
            stagnation_limit: K — halt after this many zero-expansion cycles.
            fitness_threshold: Minimum fitness for vocabulary expansion.
            min_clique_size: Minimum clique size for synthesis.

        Returns:
            SERLResult with full cycle history and F_eff measurement.
        """
        vocab_size_initial = subtree_vocab.size()
        total_new_atoms = 0
        total_syntheses = 0
        consecutive_zero = 0
        cycle_history: List[SERLCycleResult] = []

        for cycle_idx in range(max_cycles):
            vocab_size_before = subtree_vocab.size()

            # Step 1: Compute resonance and extract cliques
            synthesizer.compute_resonance()
            cliques = synthesizer.extract_cliques(min_size=min_clique_size)

            if not cliques:
                # No cliques → nothing to synthesize
                cycle_result = SERLCycleResult(
                    cycle_index=cycle_idx,
                    syntheses_attempted=0,
                    syntheses_decoded=0,
                    syntheses_executed=0,
                    syntheses_above_fitness=0,
                    new_vocab_atoms=0,
                    vocab_size_before=vocab_size_before,
                    vocab_size_after=vocab_size_before,
                    f_eff_expanded=False,
                    best_fitness=0.0,
                    best_source=None,
                )
                cycle_history.append(cycle_result)
                consecutive_zero += 1
                if consecutive_zero >= stagnation_limit:
                    break
                continue

            # Step 2: Synthesize from each clique (limit per cycle)
            syntheses_attempted = 0
            syntheses_decoded = 0
            syntheses_executed = 0
            syntheses_above_fitness = 0
            cycle_new_atoms = 0
            best_fitness = 0.0
            best_source = None

            for clique in cliques[:max_cliques_per_cycle]:
                syntheses_attempted += 1
                total_syntheses += 1

                # Synthesize
                try:
                    synth_proj = synthesizer.synthesize_from_clique(clique)
                except (ValueError, Exception):
                    continue

                # Decode
                source = decoder.decode_to_source(synth_proj)
                # PLAN.md Phase 6 wiring: every decode result — even
                # the ``None`` (UNSAT-proxy) ones — must be passed
                # through ``handle_decode_result`` so the wired
                # provenance bridge sees the event. We use the synthesis
                # projection's final_handle as the THDSE arena handle.
                bridge_handle = getattr(synth_proj, "final_handle", -1)
                self.handle_decode_result(
                    source=source,
                    fitness=0.0,
                    thdse_handle=int(bridge_handle),
                    round_idx=cycle_idx,
                )
                if source is None or source.strip() == "" or source.strip() == "pass":
                    continue
                syntheses_decoded += 1

                # Execute in sandbox
                profile = sandbox.execute(source)
                syntheses_executed += 1

                if profile.fitness > best_fitness:
                    best_fitness = profile.fitness
                    best_source = source

                # Step 4: If fitness above threshold, expand vocabulary
                if profile.fitness >= fitness_threshold:
                    syntheses_above_fitness += 1
                    # PLAN.md Phase 6 wiring: a fitness-passing
                    # candidate is the trigger for the RSI/governance/
                    # skill pipeline. Re-invoke the post-decode handler
                    # with the *real* fitness so the bridges can act on
                    # it.
                    self.handle_decode_result(
                        source=source,
                        fitness=float(profile.fitness),
                        thdse_handle=int(
                            getattr(synth_proj, "final_handle", -1)
                        ),
                        round_idx=cycle_idx,
                        formula_id=f"serl_fit_{cycle_idx}_{syntheses_attempted}",
                    )
                    added = expander.expand(
                        source, profile.fitness,
                        subtree_vocab, arena, projector,
                        fitness_threshold=fitness_threshold,
                    )
                    cycle_new_atoms += added

            vocab_size_after = subtree_vocab.size()
            total_new_atoms += cycle_new_atoms
            f_eff_expanded = cycle_new_atoms > 0

            cycle_result = SERLCycleResult(
                cycle_index=cycle_idx,
                syntheses_attempted=syntheses_attempted,
                syntheses_decoded=syntheses_decoded,
                syntheses_executed=syntheses_executed,
                syntheses_above_fitness=syntheses_above_fitness,
                new_vocab_atoms=cycle_new_atoms,
                vocab_size_before=vocab_size_before,
                vocab_size_after=vocab_size_after,
                f_eff_expanded=f_eff_expanded,
                best_fitness=best_fitness,
                best_source=best_source,
            )
            cycle_history.append(cycle_result)

            if f_eff_expanded:
                consecutive_zero = 0
            else:
                consecutive_zero += 1
                if consecutive_zero >= stagnation_limit:
                    break

        # Compute final metrics
        vocab_size_final = subtree_vocab.size()
        f_eff_rate = total_new_atoms / max(total_syntheses, 1)
        space_closed = consecutive_zero >= stagnation_limit

        # Diagnosis
        expansion_cycles = [c for c in cycle_history if c.f_eff_expanded]
        n_cycles = len(cycle_history)

        if total_new_atoms == 0:
            diagnosis = (
                f"F_eff expansion: 0. Space is closed after {n_cycles} cycles. "
                f"No new vocabulary atoms were produced."
            )
        elif space_closed:
            diagnosis = (
                f"Partial expansion detected — design space grew but converged. "
                f"{total_new_atoms} new atoms across {len(expansion_cycles)}/{n_cycles} cycles, "
                f"then {stagnation_limit} consecutive zero-expansion cycles."
            )
        else:
            diagnosis = (
                f"Active expansion — {total_new_atoms} new atoms across "
                f"{len(expansion_cycles)}/{n_cycles} cycles "
                f"(expansion rate: {f_eff_rate:.4f})."
            )

        return SERLResult(
            cycles_completed=n_cycles,
            total_new_atoms=total_new_atoms,
            vocab_size_initial=vocab_size_initial,
            vocab_size_final=vocab_size_final,
            f_eff_expansion_rate=f_eff_rate,
            space_closed=space_closed,
            cycle_history=cycle_history,
            convergence_diagnosis=diagnosis,
        )
