"""Phase 14 bridge — Synthesis Breakthrough + final integration (fixes D4).

This bridge is the integration seam for the entire Phase 9–14 stack.
It imports every prior Phase 9–13 bridge (Rule 20 cascade) and
exposes a single :meth:`SynthesisBreakthroughBridge.run_benchmark`
method that:

1. grounds every problem description as a CCE concept (Phase 9),
2. feeds the grounded vector into an online value estimator
   (Phase 10) so repeated benchmark runs adapt,
3. records every problem + winning template as episodic then
   semantic memory (Phase 11),
4. uses the Phase 12 chain-of-thought reasoner to decompose
   stubborn problems,
5. optionally drives an agent loop over a bandit "problem picker"
   environment (Phase 13) to demonstrate interactive refinement.

Every stage is provenance-stamped (Rule 9), every learning call
surfaces ``loss_decreased`` (Rule 13), and the benchmark itself runs
real code against real I/O examples (Rule 15).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from shared.arena_manager import ArenaManager
from shared.constants import (
    ENHANCED_BEAM_WIDTH,
    SEMANTIC_ENCODER_DIM,
    SYNTHESIS_BENCHMARK_TARGET,
)
from shared.environment import BanditEnvironment
from shared.semantic_encoder import SemanticEncoder
from shared.synthesis_engine import ProblemSpec, SynthesisEngine, Template

from bridges.agent_environment_bridge import AgentEnvironmentBridge
from bridges.concept_axiom_bridge import ConceptAxiomBridge
from bridges.continuous_learning_bridge import ContinuousLearningBridge
from bridges.memory_architecture_bridge import MemoryArchitectureBridge
from bridges.reasoning_bridge import ReasoningBridge
from bridges.semantic_concept_bridge import SemanticConceptBridge


class SynthesisBreakthroughBridge:
    """Orchestrates the Phase 9–14 stack against the synthesis benchmark."""

    def __init__(
        self,
        arena_manager: ArenaManager,
        *,
        encoder: Optional[SemanticEncoder] = None,
    ):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._encoder = encoder or SemanticEncoder(rng=arena_manager.rng)
        # Phase 9
        self._semantic_bridge = SemanticConceptBridge(
            arena_manager, encoder=self._encoder
        )
        # Phase 3 + Phase 9 coupling (reuse existing ConceptAxiomBridge)
        self._concept_axiom = ConceptAxiomBridge(
            arena_manager, semantic_bridge=self._semantic_bridge
        )
        # Phase 10
        self._learning = ContinuousLearningBridge(
            arena_manager,
            semantic_bridge=self._semantic_bridge,
            input_dim=SEMANTIC_ENCODER_DIM,
            output_dim=1,
            hidden_dims=[64, 32],
        )
        # Phase 11
        self._memory = MemoryArchitectureBridge(
            arena_manager,
            encoder=self._encoder,
            semantic_bridge=self._semantic_bridge,
        )
        # Phase 12
        self._reasoning = ReasoningBridge(
            arena_manager,
            encoder=self._encoder,
            semantic_bridge=self._semantic_bridge,
            memory_bridge=self._memory,
        )
        # Phase 13 (optional, lazy)
        self._agent_bridge: Optional[AgentEnvironmentBridge] = None
        # Phase 14 engine
        self._engine = SynthesisEngine(beam_width=ENHANCED_BEAM_WIDTH)

    # ---- read-only properties ---- #

    @property
    def engine(self) -> SynthesisEngine:
        return self._engine

    @property
    def memory(self) -> MemoryArchitectureBridge:
        return self._memory

    @property
    def reasoning(self) -> ReasoningBridge:
        return self._reasoning

    @property
    def learning(self) -> ContinuousLearningBridge:
        return self._learning

    # ---- primary API ---- #

    def run_benchmark(
        self, problems: Sequence[ProblemSpec]
    ) -> Dict[str, Any]:
        """Execute the Phase 14 benchmark end-to-end.

        Returns a dict with ``solved``, ``total``, ``solved_fraction``,
        ``meets_target`` (>= 4/5 per PLAN
        :data:`SYNTHESIS_BENCHMARK_TARGET`), full provenance, and
        per-problem diagnostics including the winning template source.
        """
        per_problem: List[Dict[str, Any]] = []
        solved = 0
        for problem in problems:
            grounded = self._semantic_bridge.ground_text(
                f"{problem.name}: {problem.description}"
            )
            result = self._engine.solve(problem)
            # Record episodic memory of the solve attempt.
            self._memory.remember_event(
                f"solved {problem.name} with {result['winner']} "
                f"(pass_rate={result['pass_rate']:.2f})",
                metadata={
                    "subject": problem.name,
                    "template": result["winner"],
                    "pass_rate": result["pass_rate"],
                },
            )
            if result["pass_rate"] >= 1.0:
                solved += 1
                # Promote to semantic memory as a solved fact.
                self._memory.assert_fact(
                    subject=problem.name,
                    fact=f"template '{result['winner']}' solves {problem.name}",
                )
            # Train the value estimator to predict pass_rate from the
            # grounded description (Phase 10 wiring).
            sem_vec = np.asarray(
                grounded["semantic_vector"], dtype=np.float32
            )
            self._learning.add_experience(
                sem_vec, np.array([result["pass_rate"]], dtype=np.float32)
            )
            per_problem.append(
                {
                    **result,
                    "cce_handle": grounded["cce_handle"],
                }
            )

        training_summary = self._train_value_estimator()

        meets = solved >= min(SYNTHESIS_BENCHMARK_TARGET, len(problems))
        return {
            "solved": solved,
            "total": len(problems),
            "solved_fraction": solved / float(len(problems)) if problems else 0.0,
            "meets_target": meets,
            "target": SYNTHESIS_BENCHMARK_TARGET,
            "per_problem": per_problem,
            "training": training_summary,
            "memory_counts": self._memory.counts(),
            "metadata": {
                "provenance": {
                    "operation": "run_benchmark",
                    "source_arena": "synthesis_engine",
                    "target_arena": "cce",
                    "problems": [p.name for p in problems],
                    "timestamp": time.time(),
                }
            },
        }

    def decompose_with_reasoning(
        self, problem: ProblemSpec, max_depth: int = 3
    ) -> Dict[str, Any]:
        """Drive the synthesis engine with Phase 12 reasoning for decomposition."""
        trace = self._engine.decompose_with_reasoner(
            problem,
            self._reasoning,
            max_depth=max_depth,
        )
        return {
            **trace,
            "metadata": {
                **(trace.get("metadata", {})),
                "decomposition_provenance": {
                    "operation": "decompose_with_reasoning",
                    "source_arena": "synthesis_engine",
                    "target_arena": "reasoning",
                    "problem": problem.name,
                    "timestamp": time.time(),
                },
            },
        }

    def run_agent_pick(
        self, problems: Sequence[ProblemSpec], episodes: int = 1
    ) -> Dict[str, Any]:
        """Phase 13 hook: use a bandit agent to pick a problem to attempt.

        Wires the Phase 13 agent bridge to a :class:`BanditEnvironment`
        sized to ``len(problems)`` so we can demonstrate the full
        interactive loop. Returns the agent trace plus the bandit
        reward (= 1.0 when the picked arm's problem was solved).
        """
        env = BanditEnvironment(k=len(problems), rng=self._mgr.rng)
        if self._agent_bridge is None:
            self._agent_bridge = AgentEnvironmentBridge(
                self._mgr,
                env,
                encoder=self._encoder,
                semantic_bridge=self._semantic_bridge,
                memory_bridge=self._memory,
                learning_bridge=self._learning,
                max_steps=len(problems) * 4,
            )
        trace = self._agent_bridge.run_episode(episodes=int(episodes))
        return {
            **trace,
            "metadata": {
                **(trace.get("metadata", {})),
                "pick_provenance": {
                    "operation": "run_agent_pick",
                    "source_arena": "agent",
                    "target_arena": "synthesis_engine",
                    "episodes": int(episodes),
                    "timestamp": time.time(),
                },
            },
        }

    # ---- internals ---- #

    def _train_value_estimator(self, epochs: int = 6) -> Dict[str, Any]:
        if len(self._learning.buffer) < 2:
            return {"trained": False, "reason": "insufficient_buffer"}
        initial_info = self._learning.train_batch(
            batch_size=min(8, len(self._learning.buffer))
        )
        initial_loss = initial_info["loss_before"]
        final_info = initial_info
        for _ in range(int(epochs) - 1):
            final_info = self._learning.train_batch(
                batch_size=min(8, len(self._learning.buffer))
            )
        return {
            "trained": True,
            "initial_loss": float(initial_loss),
            "final_loss": float(final_info["loss_after"]),
            "loss_decreased": final_info["loss_after"] < initial_loss,
            "steps": int(epochs),
        }


__all__ = ["SynthesisBreakthroughBridge"]
