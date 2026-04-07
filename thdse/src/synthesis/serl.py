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
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

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

    def run(
        self,
        arena: Any,
        projector: Any,
        synthesizer: Any,
        decoder: Any,
        sandbox: ExecutionSandbox,
        expander: VocabularyExpander,
        subtree_vocab: SubTreeVocabulary,
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
