"""Phase 8C — Shared benchmark runner used by both baseline and Phase 8.

The runner builds a real :class:`AxiomaticSynthesizer` over the seed
corpus + each problem's reference solutions, computes resonance,
extracts cliques, synthesizes a candidate from each clique, decodes
each synthesis to source via :class:`ConstraintDecoder` (when Z3 is
available), executes the source against the problem's io_examples,
and records every per-cycle statistic.

PLAN.md Rule 23: this runner does NOT fabricate results. If Z3 is
not installed it raises a ``Z3Unavailable`` error which the
benchmark scripts catch to print the standardised "Z3 missing"
message and exit non-zero. Per-cycle data is recorded into the JSON
output so progressions can be verified externally.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Always run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_THDSE = _REPO_ROOT / "thdse"
for p in (str(_REPO_ROOT), str(_THDSE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.arena_factory import _PyFhrrArenaExtended  # noqa: E402

from src.synthesis.problem_spec import (  # noqa: E402
    ProblemEncoder,
    ProblemSpec,
    score_against_problem,
)


class Z3Unavailable(RuntimeError):
    """Raised when the runner cannot proceed without z3-solver."""


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class CycleResult:
    cycle_index: int
    syntheses_attempted: int
    decoded: int
    above_fitness: int
    new_atoms: int
    best_fitness: float
    best_pass_rate: float
    best_source: str = ""


@dataclass
class ProblemResult:
    name: str
    solved: bool
    best_pass_rate: float
    best_source: str
    total_syntheses_attempted: int
    total_decoded: int
    total_above_fitness: int
    cycles: List[CycleResult] = field(default_factory=list)
    total_atoms_added: int = 0
    f_eff: float = 0.0
    elapsed_seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ensure_z3() -> None:
    try:
        import z3  # noqa: F401
    except ImportError as exc:  # pragma: no cover — handled in main
        raise Z3Unavailable(
            "Z3 not installed. Cannot run empirical validation. "
            "Install z3-solver and re-run."
        ) from exc


def _eval_source_for_problem(
    source: str, spec: ProblemSpec
) -> Tuple[float, Optional[Callable[[Any], Any]]]:
    """Try to compile + extract a callable from ``source`` and score it.

    Returns ``(pass_rate, callable_or_none)``. If the source fails to
    compile or no callable can be extracted, the pass rate is 0.0.
    """
    try:
        namespace: Dict[str, Any] = {}
        exec(source, namespace)
    except Exception:  # noqa: BLE001
        return 0.0, None

    candidate: Optional[Callable[[Any], Any]] = None
    for value in namespace.values():
        if callable(value) and not isinstance(value, type):
            candidate = value
            break

    if candidate is None:
        return 0.0, None

    score = score_against_problem(candidate, spec)
    return score["pass_rate"], candidate


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _build_synthesizer(
    seed_corpus: Dict[str, str],
    problem_corpus: Dict[str, str],
    use_behavioural: bool,
) -> Tuple[Any, Any, Any, Any]:
    """Construct synthesizer + projector + arena + (optional) encoder.

    Returns ``(arena, projector, synthesizer, behavioural_encoder)``.
    The behavioural encoder is ``None`` in baseline mode (so all
    axioms have ``axiom.behavioral = None`` and the dual-axis
    resonance degrades to a structural-only score scaled by alpha).
    """
    _ensure_z3()

    from src.projection.isomorphic_projector import IsomorphicProjector
    from src.projection.behavioral_encoder import BehavioralEncoder
    from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer

    arena = _PyFhrrArenaExtended(capacity=200_000, dimension=256)
    projector = IsomorphicProjector(arena, dimension=256)
    synthesizer = AxiomaticSynthesizer(
        arena=arena, projector=projector, resonance_threshold=0.10
    )
    if use_behavioural:
        behavioural_encoder = BehavioralEncoder(
            arena=arena, dimension=256, n_probes=20
        )
        synthesizer.behavioral_encoder = behavioural_encoder
    else:
        behavioural_encoder = None

    # Ingest the merged corpus (seed library + problem reference solutions
    # if any). The reference solutions are NOT the canonical answer for
    # the problem — they're additional building blocks that may or may
    # not appear verbatim in the synthesized output.
    merged = dict(seed_corpus)
    merged.update(problem_corpus)
    for name in sorted(merged.keys()):
        try:
            synthesizer.ingest(name, merged[name])
        except Exception:  # noqa: BLE001
            continue

    return arena, projector, synthesizer, behavioural_encoder


def _decode_clique(
    synthesizer: Any,
    decoder: Any,
    clique: List[str],
) -> Optional[str]:
    try:
        projection = synthesizer.synthesize_from_clique(clique)
    except Exception:  # noqa: BLE001
        return None
    try:
        return decoder.decode_to_source(projection)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Solver loop
# --------------------------------------------------------------------------- #


def run_problem(
    spec: ProblemSpec,
    seed_corpus: Dict[str, str],
    *,
    use_behavioural: bool,
    use_goal_direction: bool,
    use_cegr: bool,
    max_cycles: int = 3,
    max_cliques_per_cycle: int = 8,
) -> ProblemResult:
    """Run synthesis for a single problem and return the per-cycle result."""
    t0 = time.monotonic()
    arena, projector, synthesizer, behavioural_encoder = _build_synthesizer(
        seed_corpus=seed_corpus,
        problem_corpus={},
        use_behavioural=use_behavioural,
    )

    from src.decoder.constraint_decoder import ConstraintDecoder
    from src.decoder.subtree_vocab import SubTreeVocabulary
    from src.decoder.vocab_expander import VocabularyExpander
    from src.synthesis.problem_spec import ProblemEncoder

    # Build subtree vocab from the corpus.
    vocab = SubTreeVocabulary()
    for src in seed_corpus.values():
        try:
            vocab.ingest_source(src)
        except Exception:  # noqa: BLE001
            continue
    vocab.project_all(arena, projector)

    decoder = ConstraintDecoder(
        arena, projector, dimension=256,
        activation_threshold=0.04,
        subtree_vocab=vocab,
    )
    expander = VocabularyExpander()

    # Optional goal direction setup.
    problem_vector = None
    if use_goal_direction:
        encoder = ProblemEncoder(arena, dimension=256)
        problem_vector = encoder.encode_problem(spec)

    # Optional CEGR setup.
    refiner = None
    if use_cegr:
        from src.synthesis.counterexample_refiner import CounterexampleRefiner

        refiner = CounterexampleRefiner(arena, dimension=256)

    cycles: List[CycleResult] = []
    best_pass_rate = 0.0
    best_source = ""
    total_attempts = 0
    total_decoded = 0
    total_above_fitness = 0
    total_atoms_added = 0
    initial_vocab_size = vocab.size()

    for cycle_idx in range(max_cycles):
        synthesizer.compute_resonance()

        # Pick the cliques to synthesize this cycle.
        if use_goal_direction and problem_vector is not None:
            ranked = synthesizer.synthesize_for_problem(
                problem_vector, min_clique_size=2, top_k=max_cliques_per_cycle
            )
            clique_sources: List[Tuple[List[str], Any]] = [
                (clique, projection) for clique, projection, _score in ranked
            ]
        else:
            cliques = synthesizer.extract_cliques(min_size=2)
            cliques = cliques[:max_cliques_per_cycle]
            clique_sources = []
            for clique in cliques:
                try:
                    projection = synthesizer.synthesize_from_clique(clique)
                except Exception:  # noqa: BLE001
                    continue
                clique_sources.append((clique, projection))

        attempted = 0
        decoded = 0
        above_fitness = 0
        cycle_new_atoms = 0
        cycle_best_pass = 0.0
        cycle_best_source = ""
        cycle_best_fitness = 0.0

        for clique, projection in clique_sources:
            attempted += 1
            try:
                source = decoder.decode_to_source(projection)
            except Exception:  # noqa: BLE001
                source = None
            if not source or not source.strip():
                continue
            decoded += 1

            pass_rate, _func = _eval_source_for_problem(source, spec)
            if pass_rate >= 0.4:
                above_fitness += 1

            if pass_rate > cycle_best_pass:
                cycle_best_pass = pass_rate
                cycle_best_source = source
                cycle_best_fitness = pass_rate

            # Vocab expansion when fitness clears the SERL gate.
            if pass_rate >= 0.4:
                try:
                    added = expander.expand(
                        source, pass_rate, vocab, arena, projector,
                        fitness_threshold=0.4,
                    )
                    cycle_new_atoms += int(added)
                except Exception:  # noqa: BLE001
                    pass

            # CEGR — if a candidate failed but partially passed, fold the
            # arena away from its error vector before the next cycle.
            if (
                use_cegr
                and refiner is not None
                and 0.0 < pass_rate < 1.0
            ):
                # Build failed_ios from the io_examples.
                failed_ios = []
                passed_ios = []
                for input_val, expected in spec.io_examples:
                    try:
                        actual = _eval_source_for_problem(source, spec)[1](
                            input_val
                        )
                    except Exception:  # noqa: BLE001
                        actual = "<crash>"
                    if actual == expected:
                        passed_ios.append((input_val, expected))
                    else:
                        failed_ios.append((input_val, expected, actual))
                if failed_ios:
                    try:
                        refiner.refine(
                            failed_source=source,
                            problem_io=spec.io_examples,
                            passed_ios=passed_ios,
                            failed_ios=failed_ios,
                        )
                    except Exception:  # noqa: BLE001
                        pass

        if cycle_best_pass > best_pass_rate:
            best_pass_rate = cycle_best_pass
            best_source = cycle_best_source

        total_attempts += attempted
        total_decoded += decoded
        total_above_fitness += above_fitness
        total_atoms_added += cycle_new_atoms

        cycles.append(
            CycleResult(
                cycle_index=cycle_idx,
                syntheses_attempted=attempted,
                decoded=decoded,
                above_fitness=above_fitness,
                new_atoms=cycle_new_atoms,
                best_fitness=cycle_best_fitness,
                best_pass_rate=cycle_best_pass,
                best_source=cycle_best_source[:200],
            )
        )

        if best_pass_rate >= 1.0:
            break  # Solved — stop cycling.

    elapsed = time.monotonic() - t0
    f_eff = (
        total_atoms_added / max(total_attempts, 1)
        if total_attempts > 0
        else 0.0
    )

    return ProblemResult(
        name=spec.name,
        solved=best_pass_rate >= 1.0,
        best_pass_rate=best_pass_rate,
        best_source=best_source,
        total_syntheses_attempted=total_attempts,
        total_decoded=total_decoded,
        total_above_fitness=total_above_fitness,
        cycles=cycles,
        total_atoms_added=total_atoms_added,
        f_eff=f_eff,
        elapsed_seconds=elapsed,
    )


def run_suite(
    problems: List[ProblemSpec],
    seed_corpus: Dict[str, str],
    *,
    use_behavioural: bool,
    use_goal_direction: bool,
    use_cegr: bool,
    max_cycles: int = 3,
) -> Dict[str, Any]:
    """Run every problem in ``problems`` and aggregate the metrics."""
    _ensure_z3()
    results: List[ProblemResult] = []
    for spec in problems:
        result = run_problem(
            spec,
            seed_corpus,
            use_behavioural=use_behavioural,
            use_goal_direction=use_goal_direction,
            use_cegr=use_cegr,
            max_cycles=max_cycles,
        )
        results.append(result)

    solved = sum(1 for r in results if r.solved)
    partial = sum(1 for r in results if r.best_pass_rate > 0.5)
    total_attempts = sum(r.total_syntheses_attempted for r in results)
    total_atoms = sum(r.total_atoms_added for r in results)
    f_eff_aggregate = (
        total_atoms / max(total_attempts, 1) if total_attempts > 0 else 0.0
    )

    return {
        "config": {
            "use_behavioural": use_behavioural,
            "use_goal_direction": use_goal_direction,
            "use_cegr": use_cegr,
            "max_cycles": max_cycles,
        },
        "totals": {
            "solve_rate": solved / max(len(results), 1),
            "partial_rate": partial / max(len(results), 1),
            "solved_count": solved,
            "partial_count": partial,
            "problem_count": len(results),
            "total_syntheses_attempted": total_attempts,
            "total_atoms_added": total_atoms,
            "f_eff_aggregate": f_eff_aggregate,
        },
        "problems": [
            {
                "name": r.name,
                "solved": r.solved,
                "best_pass_rate": r.best_pass_rate,
                "best_source": r.best_source,
                "syntheses_attempted": r.total_syntheses_attempted,
                "decoded": r.total_decoded,
                "above_fitness": r.total_above_fitness,
                "atoms_added": r.total_atoms_added,
                "f_eff": r.f_eff,
                "elapsed_seconds": r.elapsed_seconds,
                "cycles": [
                    {
                        "cycle_index": c.cycle_index,
                        "syntheses_attempted": c.syntheses_attempted,
                        "decoded": c.decoded,
                        "above_fitness": c.above_fitness,
                        "new_atoms": c.new_atoms,
                        "best_fitness": c.best_fitness,
                        "best_pass_rate": c.best_pass_rate,
                    }
                    for c in r.cycles
                ],
            }
            for r in results
        ],
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


__all__ = [
    "CycleResult",
    "ProblemResult",
    "Z3Unavailable",
    "run_problem",
    "run_suite",
    "write_json",
]
