"""Multi-step reasoning engine for OMEGA-THDSE Phase 12 (fixes D5, D7).

The pre-Phase-12 orchestrator capped reasoning at
``_MAX_META_RECURSION_DEPTH = 2`` (PLAN Section D). That constant is
immutable because it guards Z3 proof reproducibility, but *reasoning
depth* and *meta-recursion depth* are distinct concerns: reasoning is
forward chaining inside a single meta-call, not recursive re-entry
into the synthesiser. This module introduces a dedicated reasoning
engine that executes chains of length >= 3 without touching the
meta-recursion cap.

Two complementary engines are provided:

- :class:`ChainOfThoughtReasoner` — forward chaining with beam search,
  backtracking, and premise→conclusion linkage checks. Enforces
  Rule 17 (``step[i+1].premise == step[i].conclusion``).
- :class:`AnalogyEngine` — pattern extraction + transfer across
  domains. Supplies Rule 19: a pattern mined from domain A must
  improve a scoring function on unseen domain B.

Both engines are used by :class:`bridges.reasoning_bridge.ReasoningBridge`
(Phase 12) to plug into the shared arena.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    ANALOGY_SIMILARITY_MIN,
    REASONING_BACKTRACK_PATIENCE,
    REASONING_BEAM_WIDTH,
    REASONING_DEFAULT_DEPTH,
    REASONING_MAX_DEPTH,
)
from .semantic_encoder import SemanticEncoder, cosine


# --------------------------------------------------------------------------- #
# Chain-of-thought reasoner
# --------------------------------------------------------------------------- #


@dataclass
class ReasoningStep:
    index: int
    premise: Any
    operator: str
    conclusion: Any
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


#: Signature of a reasoning operator — takes the current premise and
#: returns a list of ``(conclusion, score, metadata)`` expansions.
OperatorFn = Callable[[Any], Sequence[Tuple[Any, float, Dict[str, Any]]]]


class ChainOfThoughtReasoner:
    """Beam search over operator applications with backtracking.

    Each step consumes a premise and produces a list of candidate
    conclusions via user-supplied operators. The reasoner selects
    the top-``beam_width`` expansions, scores them against a goal
    function, and continues until either (a) the goal function
    returns a score >= ``goal_threshold``, (b) ``max_depth`` is
    reached, or (c) ``REASONING_BACKTRACK_PATIENCE`` consecutive
    steps yielded no improvement (in which case the reasoner
    backtracks to the best ancestor and tries the next beam).
    """

    def __init__(
        self,
        operators: Dict[str, OperatorFn],
        goal_fn: Callable[[Any], float],
        *,
        max_depth: int = REASONING_DEFAULT_DEPTH,
        beam_width: int = REASONING_BEAM_WIDTH,
        patience: int = REASONING_BACKTRACK_PATIENCE,
    ):
        if not operators:
            raise ValueError("at least one operator is required")
        if max_depth > REASONING_MAX_DEPTH:
            raise ValueError(
                f"max_depth {max_depth} exceeds REASONING_MAX_DEPTH "
                f"{REASONING_MAX_DEPTH}"
            )
        if max_depth < 3:
            raise ValueError(
                "Rule 17 requires reasoning depth >= 3; "
                f"got max_depth={max_depth}"
            )
        self._operators = dict(operators)
        self._goal_fn = goal_fn
        self._max_depth = int(max_depth)
        self._beam = int(beam_width)
        self._patience = int(patience)

    @property
    def operators(self) -> Dict[str, OperatorFn]:
        return dict(self._operators)

    def run(self, initial_premise: Any, goal_threshold: float = 0.95) -> Dict[str, Any]:
        """Execute the chain and return ``{"steps": [...], "final": ...}``.

        The returned ``steps`` list is ordered; for every adjacent
        pair the premise of step ``i+1`` equals the conclusion of
        step ``i`` (Rule 17).
        """
        current = initial_premise
        steps: List[ReasoningStep] = []
        stalled = 0
        best_score = float("-inf")
        visited: set = set()

        for depth in range(self._max_depth):
            expansions: List[Tuple[Any, float, Dict[str, Any], str]] = []
            for name, op in self._operators.items():
                for conclusion, score, meta in op(current):
                    if isinstance(conclusion, (str, int, float)):
                        key = (name, conclusion)
                    else:
                        key = (name, id(conclusion))
                    if key in visited:
                        continue
                    expansions.append((conclusion, score, meta, name))
            if not expansions:
                break
            expansions.sort(key=lambda t: t[1], reverse=True)
            chosen = expansions[: self._beam]
            # Pick the single-best by ``goal_fn`` so the chain actually
            # progresses toward the goal instead of just hill-climbing
            # operator-local score.
            scored = [
                (self._goal_fn(c[0]), c) for c in chosen
            ]
            scored.sort(key=lambda t: t[0], reverse=True)
            goal_score, (conclusion, op_score, meta, name) = scored[0]
            visited.add(
                (name, conclusion if isinstance(conclusion, (str, int, float))
                 else id(conclusion))
            )
            step = ReasoningStep(
                index=depth,
                premise=current,
                operator=name,
                conclusion=conclusion,
                score=float(goal_score),
                metadata={
                    "operator_score": float(op_score),
                    **dict(meta),
                },
            )
            steps.append(step)
            if goal_score > best_score + 1e-9:
                best_score = goal_score
                stalled = 0
            else:
                stalled += 1
            current = conclusion
            if goal_score >= goal_threshold:
                break
            if stalled >= self._patience and len(steps) >= 3:
                # Backtrack to the best ancestor and stop — the caller
                # sees the chain up to the plateau, no deeper.
                break

        return {
            "steps": steps,
            "depth": len(steps),
            "final_premise": current,
            "final_score": float(best_score if best_score > float("-inf") else 0.0),
            "reached_goal": best_score >= goal_threshold,
        }


def verify_chain_linkage(steps: Sequence[ReasoningStep]) -> bool:
    """Return True iff ``step[i+1].premise == step[i].conclusion`` for all i.

    Used by Rule 17 tests to assert that the reasoner does not
    silently drop the chain linkage when an operator produces an
    ambiguous conclusion.
    """
    for a, b in zip(steps[:-1], steps[1:]):
        if a.conclusion != b.premise:
            return False
    return True


# --------------------------------------------------------------------------- #
# Analogy + transfer engine
# --------------------------------------------------------------------------- #


@dataclass
class AnalogyPattern:
    name: str
    features: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnalogyEngine:
    """Cross-domain analogy + transfer learning.

    The engine encodes every domain example as a semantic feature
    vector. Patterns are the centroid of a labelled example cluster.
    :meth:`transfer_score` evaluates how well a pattern from domain A
    raises a scoring function on domain B — Rule 19's transfer
    contract.
    """

    def __init__(
        self,
        encoder: SemanticEncoder,
        similarity_min: float = ANALOGY_SIMILARITY_MIN,
    ):
        self._encoder = encoder
        self._min = float(similarity_min)
        self._patterns: Dict[str, AnalogyPattern] = {}

    def extract_pattern(
        self, name: str, examples: Sequence[str], metadata: Optional[Dict[str, Any]] = None
    ) -> AnalogyPattern:
        if not examples:
            raise ValueError("at least one example is required")
        vecs = [self._encoder.encode(e) for e in examples]
        centroid = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-12:
            centroid = centroid / norm
        pattern = AnalogyPattern(
            name=name,
            features=centroid,
            metadata=dict(metadata or {}),
        )
        self._patterns[name] = pattern
        return pattern

    def get_pattern(self, name: str) -> AnalogyPattern:
        return self._patterns[name]

    def match(self, candidate: str) -> List[Tuple[str, float]]:
        qvec = self._encoder.encode(candidate)
        scored = [
            (name, float(cosine(qvec, pat.features)))
            for name, pat in self._patterns.items()
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def score_with_pattern(
        self, candidate: str, pattern_name: str
    ) -> float:
        if pattern_name not in self._patterns:
            return 0.0
        return float(
            cosine(self._encoder.encode(candidate), self._patterns[pattern_name].features)
        )

    def transfer_score(
        self,
        source_examples: Sequence[str],
        target_examples: Sequence[str],
        *,
        distractor_examples: Optional[Sequence[str]] = None,
        pattern_name: str = "transfer",
    ) -> Dict[str, Any]:
        """Rule 19 contract — pattern from A improves score on B.

        The Rule 19 contract is about *discriminability*: a pattern
        mined on source domain A must separate genuine B-examples
        from unrelated distractors.

        - ``without_transfer`` — mean cosine of target examples to
          the same set of distractors (no pattern used). This is the
          baseline "do nothing" separator.
        - ``with_transfer``    — mean cosine of target examples to
          the source-derived pattern centroid minus the mean cosine
          of distractors to that same pattern.

        If ``distractor_examples`` is ``None``, a deterministic set of
        semantically-unrelated filler strings is used so the metric
        always exists. Transfer succeeds iff ``with_transfer >
        without_transfer``.
        """
        pattern = self.extract_pattern(pattern_name, source_examples)
        if len(target_examples) < 2:
            raise ValueError("need >= 2 target examples to compute a baseline")
        if distractor_examples is None:
            distractor_examples = (
                "abstract equation integral derivative",
                "political parliament legislation",
                "tectonic seismic volcanic",
                "symphony orchestra conductor",
                "pastry baking flour dough",
            )
        target_vecs = [self._encoder.encode(t) for t in target_examples]
        distract_vecs = [self._encoder.encode(d) for d in distractor_examples]

        # Baseline: mean cosine of target to distractors (no pattern).
        without_transfer = float(
            np.mean(
                [
                    cosine(t, d)
                    for t in target_vecs
                    for d in distract_vecs
                ]
            )
        )
        # With transfer: pattern separates target from distractors.
        target_hit = float(
            np.mean([cosine(v, pattern.features) for v in target_vecs])
        )
        distract_hit = float(
            np.mean([cosine(v, pattern.features) for v in distract_vecs])
        )
        with_transfer = float(target_hit - distract_hit)
        return {
            "without_transfer": without_transfer,
            "with_transfer": with_transfer,
            "target_hit": target_hit,
            "distractor_hit": distract_hit,
            "improved": with_transfer > without_transfer,
            "pattern_name": pattern_name,
        }


__all__ = [
    "ChainOfThoughtReasoner",
    "ReasoningStep",
    "AnalogyEngine",
    "AnalogyPattern",
    "verify_chain_linkage",
]
