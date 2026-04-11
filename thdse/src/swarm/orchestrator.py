"""SwarmOrchestrator — Central coordinator for the THDSE-Swarm.

Manages agent lifecycle, collects candidates, computes algebraic consensus
via correlate_matrix + Bron-Kerbosch, handles UNSAT broadcasting (Localized
Quotient Isolation with Drift Reconciliation), and tracks swarm-level metrics.

Each round:
  1. Each agent runs local synthesis -> emits candidate PhaseMessages
  2. Orchestrator collects all candidates
  3. Compute consensus via correlate_matrix + Bron-Kerbosch (layered partial match)
  4. Decode consensus centroid -> Z3 -> Python AST
  5. If SAT: execute in sandbox -> evaluate fitness
     - If fitness > threshold: broadcast consensus to all agents
  6. If UNSAT: extract actual V_error -> broadcast ONLY to participating agents
     - Record wall in the global wall ledger for drift reconciliation
  7. Periodically reconcile drift: propagate aged walls to non-participating
     agents to prevent topological fragmentation (Tower of Babel effect)
  8. Track metrics
  9. Halt when: max_rounds reached OR stagnation_limit consecutive
     rounds with zero progress
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# PLAN.md Phase 6 wiring (Rule 3): arena allocation now flows through
# arena_factory.make_arena rather than ``hdc_core.FhrrArena`` directly.
from src.utils.arena_factory import make_arena

from src.decoder.constraint_decoder import ConstraintDecoder
from src.decoder.subtree_vocab import SubTreeVocabulary
from src.execution.sandbox import ExecutionSandbox
from src.projection.isomorphic_projector import IsomorphicProjector, LayeredProjection

from src.swarm.protocol import PhaseMessage, SwarmConfig
from src.swarm.agent import ThdseAgent
from src.swarm.consensus import ConsensusResult, compute_swarm_consensus

logger = logging.getLogger(__name__)


def _agent_synthesis_worker(
    agent_id: int,
    config: SwarmConfig,
    corpus_dict: Dict[str, str],
    wall_phases_list: List[List[float]],
    max_cliques: int,
) -> List[PhaseMessage]:
    """Top-level worker for ProcessPoolExecutor.

    Instantiates a fresh ThdseAgent in the child process (with its own
    FhrrArena and Z3 solver), ingests the corpus, replays accumulated
    walls, runs local synthesis, and returns only the picklable
    PhaseMessage list through the IPC queue.

    Args:
        agent_id: Agent index for sender_id tagging.
        config: SwarmConfig (pure dataclass, fully picklable).
        corpus_dict: {source_id: source_code} for this agent's corpus.
        wall_phases_list: Accumulated wall phase arrays to replay.
        max_cliques: Max cliques for synthesis.

    Returns:
        List of PhaseMessage candidates (float arrays + basic metadata).
    """
    agent = ThdseAgent(agent_id, config, [])
    agent.ingest_corpus_dict(corpus_dict)

    for wall_phases in wall_phases_list:
        agent.receive_wall_broadcast(wall_phases)

    return agent.run_local_synthesis(max_cliques=max_cliques)


@dataclass
class WallLedgerEntry:
    """A record in the orchestrator's global wall ledger.

    Tracks which agents have received each wall and when the wall was
    created, enabling delayed propagation for drift reconciliation.
    """
    wall_phases: List[float]          # Phase array of V_error
    round_created: int                # Round index when wall was discovered
    source_agent_ids: List[int]       # Agents that contributed to the contradiction
    applied_to: Set[int] = field(default_factory=set)  # Agents that have received this wall


@dataclass
class SwarmRoundResult:
    """Result of a single swarm round."""
    round_index: int
    candidates_collected: int
    candidates_per_agent: List[int]
    consensus_reached: bool
    consensus_clique_size: int
    consensus_mean_resonance: float
    decoded_source: Optional[str]
    fitness: float
    walls_broadcast: int
    vocab_growth_per_agent: List[int]
    wall_target_agents: List[int] = field(default_factory=list)
    walls_reconciled: int = 0


@dataclass
class SwarmResult:
    """Aggregate result of the full swarm run."""
    rounds_completed: int
    total_candidates: int
    total_consensus_reached: int
    total_walls_broadcast: int
    total_vocab_growth: int
    best_fitness: float
    best_source: Optional[str]
    round_history: List[SwarmRoundResult] = field(default_factory=list)
    per_agent_stats: List[Dict[str, Any]] = field(default_factory=list)
    convergence_diagnosis: str = ""
    f_eff_expansion_rate: float = 0.0
    total_walls_reconciled: int = 0


class SwarmOrchestrator:
    """Orchestrates the THDSE-Swarm collective intelligence loop."""

    def __init__(
        self,
        config: SwarmConfig,
        *,
        arena_manager: Any = None,
        goal_synthesis_bridge: Any = None,
        world_model_swarm_bridge: Any = None,
        governance_bridge: Any = None,
        axiom_skill_bridge: Any = None,
        provenance_bridge: Any = None,
    ):
        """Initialize the orchestrator and all agents.

        Creates N ThdseAgent instances, each with its assigned corpus.
        Also creates the orchestrator's own arena/decoder for consensus
        decoding (separate from agent arenas).

        PLAN.md Phase 6 wiring: every cross-engine bridge handle is
        accepted via keyword. The bridges are owned by the **main**
        orchestrator process only — Rule 11 forbids passing them to
        worker processes spawned via :class:`ProcessPoolExecutor`. The
        worker function (:func:`_agent_synthesis_worker`) receives only
        plain serializable primitives (config dataclass + dicts +
        lists), never the orchestrator instance or its bridges.
        """
        self.config = config
        self.agents: List[ThdseAgent] = []

        # PLAN.md Phase 6 wiring: store the injected bridges so the
        # orchestrator's pre-round and post-consensus hooks can call
        # them. Each handle defaults to None for backward compat.
        self._arena_manager = arena_manager
        self._goal_synthesis_bridge = goal_synthesis_bridge
        self._world_model_swarm_bridge = world_model_swarm_bridge
        self._governance_bridge = governance_bridge
        self._axiom_skill_bridge = axiom_skill_bridge
        self._provenance_bridge = provenance_bridge
        # Counters for Tier-1 verification.
        self._goal_inject_calls = 0
        self._consensus_check_calls = 0
        self._registration_attempts = 0

        # Validate corpus_paths length
        if len(config.corpus_paths) != config.n_agents:
            raise ValueError(
                f"corpus_paths has {len(config.corpus_paths)} entries, "
                f"expected {config.n_agents} (one per agent)"
            )

        # Check heterogeneity: warn if corpora overlap
        corpus_sets = [set(paths) for paths in config.corpus_paths]
        for i in range(len(corpus_sets)):
            for j in range(i + 1, len(corpus_sets)):
                if corpus_sets[i] and corpus_sets[j] and corpus_sets[i] == corpus_sets[j]:
                    logger.warning(
                        "Agents %d and %d have identical corpora. "
                        "Identical agents produce identical resonance landscapes "
                        "and degrade collective intelligence.", i, j,
                    )

        # Create agents — pass the manager so each agent can use the
        # Rust-backed arena when available.
        for i in range(config.n_agents):
            agent = ThdseAgent(
                i,
                config,
                config.corpus_paths[i],
                arena_manager=arena_manager,
            )
            self.agents.append(agent)

        # PLAN.md Phase 6 wiring: orchestrator-level consensus arena
        # comes from arena_factory rather than a direct hdc_core call
        # (Rule 3).
        self._consensus_arena = make_arena(
            config.arena_capacity,
            config.dimension,
            arena_manager=arena_manager,
        )
        self._consensus_projector = IsomorphicProjector(
            self._consensus_arena, config.dimension,
        )
        self._consensus_sandbox = ExecutionSandbox()

        # Orchestrator's merged vocabulary (built after agent ingestion)
        self._merged_vocab: Optional[SubTreeVocabulary] = None
        self._consensus_decoder: Optional[ConstraintDecoder] = None

        # Global wall ledger for drift reconciliation
        self._wall_ledger: List[WallLedgerEntry] = []

        # Per-agent corpus dicts for process-based parallel synthesis.
        # Populated by ingest_agent_corpora_dicts(); file-based corpora
        # are handled via corpus_paths in config.
        self._agent_corpus_dicts: List[Dict[str, str]] = [
            {} for _ in range(config.n_agents)
        ]

        # Per-agent accumulated wall phases for replay in worker processes
        self._agent_wall_history: List[List[List[float]]] = [
            [] for _ in range(config.n_agents)
        ]

    # ------------------------------------------------------------------
    # PLAN.md Phase 6 — testable pre/post hooks (Rule 14)
    # ------------------------------------------------------------------

    def pre_round_goal_injection(
        self, candidate_goals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank ``candidate_goals`` through :class:`GoalSynthesisBridge`.

        ``candidate_goals`` is a list of dicts that already carry
        ``priority`` and ``projected_similarity`` fields (the typical
        output of :meth:`GoalSynthesisBridge.goal_to_synthesis_target`).
        Returns the ranked list with rank metadata attached. When no
        bridge is wired this is a no-op that returns the input list
        unchanged with a ``rank`` field assigned by insertion order.
        """
        self._goal_inject_calls += 1
        if not candidate_goals:
            return []
        if self._goal_synthesis_bridge is None:
            ranked = []
            for idx, goal in enumerate(candidate_goals, start=1):
                item = dict(goal)
                item["rank"] = idx
                ranked.append(item)
            return ranked
        return self._goal_synthesis_bridge.rank_goals(candidate_goals)

    def post_consensus_handle(
        self,
        consensus_phases: List[float],
        consensus_source: Optional[str],
        fitness: float,
        round_idx: int,
    ) -> Dict[str, Any]:
        """Drive the consensus result through the post-Z3 governance path.

        Steps:
        1. Ask :class:`WorldModelSwarmBridge.incorporate_swarm_consensus`
           whether CCE should adopt the consensus.
        2. If adopted AND a consensus source decoded, run governance
           via :class:`GovernanceSynthesisBridge.evaluate_candidate`.
        3. If governance approves, register through
           :class:`AxiomSkillBridge.validate_and_register`.
        4. Always emit a provenance event when ``provenance_bridge`` is
           wired so the causal chain stays complete.
        """
        self._consensus_check_calls += 1
        adopt_decision: Dict[str, Any] = {}
        approved = False
        registered = False

        if self._world_model_swarm_bridge is not None and consensus_phases:
            adopt_decision = (
                self._world_model_swarm_bridge.incorporate_swarm_consensus(
                    list(consensus_phases)
                )
            )

        should_adopt = bool(adopt_decision.get("should_adopt", True))

        if (
            should_adopt
            and consensus_source
            and consensus_source.strip()
            and self._governance_bridge is not None
        ):
            verdict = self._governance_bridge.evaluate_candidate(
                consensus_source, 0, float(fitness)
            )
            approved = self._governance_bridge.gate_registration(verdict)

            if approved and self._axiom_skill_bridge is not None:
                self._registration_attempts += 1
                skill_name = f"swarm_consensus_round_{round_idx}"
                try:
                    reg = self._axiom_skill_bridge.validate_and_register(
                        axiom_handle=0,
                        program_source=consensus_source,
                        skill_name=skill_name,
                        governance_approved=True,
                    )
                    registered = bool(reg.get("registered"))
                except Exception:  # noqa: BLE001 — recorded below
                    registered = False

        if self._provenance_bridge is not None:
            self._provenance_bridge.record_synthesis_event(
                "swarm_consensus",
                None,
                {
                    "round_idx": round_idx,
                    "should_adopt": should_adopt,
                    "approved": approved,
                    "registered": registered,
                },
            )

        return {
            "should_adopt": should_adopt,
            "approved": approved,
            "registered": registered,
            "metadata": {
                "round_idx": round_idx,
                "adopt_decision": adopt_decision,
                "consensus_check_calls": self._consensus_check_calls,
                "registration_attempts": self._registration_attempts,
                "provenance": {
                    "operation": "post_consensus_handle",
                    "source_arena": "thdse",
                    "target_arena": "cce",
                },
            },
        }

    @property
    def goal_inject_calls(self) -> int:
        return self._goal_inject_calls

    @property
    def consensus_check_calls(self) -> int:
        return self._consensus_check_calls

    @property
    def registration_attempts(self) -> int:
        return self._registration_attempts

    def _build_merged_vocab(self) -> None:
        """Merge SubTreeVocabulary from ALL agents into the orchestrator's decoder.

        Each agent's unique vocab atoms become part of the orchestrator's decoder.
        This means the orchestrator can decode structures that no single agent
        could decode alone.
        """
        self._merged_vocab = SubTreeVocabulary()

        for agent in self.agents:
            # Merge each agent's vocab atoms by ingesting their canonical sources
            for atom in agent.vocab.get_atoms().values():
                self._merged_vocab.ingest_source(atom.canonical_source)

        # Project merged vocab in orchestrator's arena
        self._merged_vocab.project_all(
            self._consensus_arena, self._consensus_projector,
        )

        # Create orchestrator decoder
        self._consensus_decoder = ConstraintDecoder(
            self._consensus_arena, self._consensus_projector,
            self.config.dimension,
            activation_threshold=0.04,
            subtree_vocab=self._merged_vocab,
        )

    def run(self) -> SwarmResult:
        """Execute the full swarm loop.

        Returns SwarmResult with complete round history and metrics.
        """
        # Phase 1: Ingest corpora for all agents
        for agent in self.agents:
            if agent.corpus_paths:
                agent.ingest_corpus()

        # Build merged vocabulary for consensus decoding
        self._build_merged_vocab()

        # Phase 2: Run swarm loop
        round_history: List[SwarmRoundResult] = []
        total_candidates = 0
        total_consensus = 0
        total_walls = 0
        total_reconciled = 0
        best_fitness = 0.0
        best_source: Optional[str] = None
        consecutive_zero_progress = 0

        for round_idx in range(self.config.max_rounds):
            # Track vocab sizes before this round
            vocab_before = [agent.vocab.size() for agent in self.agents]

            # Step 1: Collect candidates from all agents
            all_candidates = self._collect_candidates()
            candidates_per_agent = [0] * self.config.n_agents
            for msg in all_candidates:
                candidates_per_agent[msg.sender_id] += 1

            # Split agents into Baseline (80%) and Heretics (20%)
            baseline_cutoff = int(self.config.n_agents * 0.8)
            baseline_candidates = [m for m in all_candidates if m.sender_id < baseline_cutoff]
            heretic_candidates = [m for m in all_candidates if m.sender_id >= baseline_cutoff]

            total_candidates += len(all_candidates)

            # Step 2: Compute consensus on Baseline Swarm
            consensus_reached = False
            clique_size = 0
            mean_resonance = 0.0
            decoded_source = None
            fitness = 0.0
            walls_this_round = 0
            wall_target_agents: List[int] = []

            if len(baseline_candidates) >= 2:
                # Create temporary arena for consensus via the
                # arena_factory (Rule 3): no direct hdc_core call.
                tmp_arena = make_arena(
                    max(len(baseline_candidates) * 5 + 50, 200),
                    self.config.dimension,
                    arena_manager=self._arena_manager,
                )

                candidate_phases = [msg.phases for msg in baseline_candidates]
                candidate_metadata = [msg.metadata for msg in baseline_candidates]

                # Collect per-layer phases for layered consensus
                ast_phases_list = [msg.ast_phases for msg in baseline_candidates]
                cfg_phases_list = [msg.cfg_phases for msg in baseline_candidates]
                data_phases_list = [msg.data_phases for msg in baseline_candidates]

                # Only pass layer phases if at least one candidate has them
                has_ast = any(p is not None for p in ast_phases_list)
                has_cfg = any(p is not None for p in cfg_phases_list)
                has_data = any(p is not None for p in data_phases_list)

                consensus = compute_swarm_consensus(
                    tmp_arena,
                    candidate_phases,
                    candidate_metadata,
                    consensus_threshold=self.config.consensus_threshold,
                    min_clique_size=2,
                    candidate_ast_phases=ast_phases_list if has_ast else None,
                    candidate_cfg_phases=cfg_phases_list if has_cfg else None,
                    candidate_data_phases=data_phases_list if has_data else None,
                )

                if consensus is not None:
                    consensus_reached = True
                    clique_size = consensus.clique_size
                    mean_resonance = consensus.mean_resonance
                    total_consensus += 1

                    # Step 3: Decode consensus centroid
                    decoded_source, v_error_phases = self._decode_consensus(
                        consensus.centroid_phases,
                    )

                    if decoded_source is not None:
                        # Step 4: Execute and evaluate
                        profile = self._consensus_sandbox.execute(decoded_source)
                        fitness = profile.fitness

                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_source = decoded_source

                        # Step 5: Broadcast consensus to baseline agents if above threshold
                        if fitness >= self.config.fitness_threshold:
                            self._broadcast_consensus(consensus.centroid_phases)
                    else:
                        # UNSAT: broadcast wall ONLY to participating agents
                        actual_v_error = (
                            v_error_phases
                            if v_error_phases is not None
                            else consensus.centroid_phases
                        )
                        wall_target_agents = consensus.clique_agent_ids
                        self._broadcast_wall_targeted(
                            actual_v_error, wall_target_agents, round_idx,
                        )
                        walls_this_round += 1
                        total_walls += 1

            # Process Heretics: bypass consensus, infinite divergence via OmegaForge
            if heretic_candidates:
                import sys
                import os
                # Add CCE-Test to path so we can import omega_forge if not already there
                cce_path = os.path.join(os.path.dirname(__file__), "../../../Cognitive-Core-Engine-Test")
                if cce_path not in sys.path:
                    sys.path.append(cce_path)
                    
                from cognitive_core_engine.omega_forge.engine import OmegaForgeV13
                from src.topology.ast_graph_ca import TopologicalASTGraphCA
                
                forge = OmegaForgeV13()
                for msg in heretic_candidates:
                    h_source, h_error_phases = self._decode_consensus(msg.phases)
                    if h_source is None and h_error_phases is not None:
                        # UNSAT: Trigger OmegaForge mutation and perturb CA
                        pressure = forge.evolutionary_perturbation(h_error_phases)
                        ca = TopologicalASTGraphCA()
                        g = ca.code_to_graph("pass", evolutionary_pressure=pressure)
                        ca.evolve(g)
                        logger.info("Heretic Agent %d diverged with evolutionary pressure %.4f", msg.sender_id, pressure)

            # Drift reconciliation: propagate aged walls to drifting agents
            walls_reconciled = 0
            reconciliation_interval = self.config.drift_reconciliation_interval
            if reconciliation_interval > 0 and (round_idx + 1) % reconciliation_interval == 0:
                walls_reconciled = self._reconcile_drift(round_idx)
                total_reconciled += walls_reconciled

            # Track vocab growth
            vocab_after = [agent.vocab.size() for agent in self.agents]
            vocab_growth = [
                vocab_after[i] - vocab_before[i]
                for i in range(self.config.n_agents)
            ]

            round_result = SwarmRoundResult(
                round_index=round_idx,
                candidates_collected=len(all_candidates),
                candidates_per_agent=candidates_per_agent,
                consensus_reached=consensus_reached,
                consensus_clique_size=clique_size,
                consensus_mean_resonance=mean_resonance,
                decoded_source=decoded_source,
                fitness=fitness,
                walls_broadcast=walls_this_round,
                vocab_growth_per_agent=vocab_growth,
                wall_target_agents=wall_target_agents,
                walls_reconciled=walls_reconciled,
            )
            round_history.append(round_result)

            # Check stagnation
            round_had_progress = (
                sum(vocab_growth) > 0
                or (consensus_reached and fitness >= self.config.fitness_threshold)
            )
            if round_had_progress:
                consecutive_zero_progress = 0
            else:
                consecutive_zero_progress += 1
                if consecutive_zero_progress >= self.config.stagnation_limit:
                    break

        # Compute final metrics
        per_agent_stats = [agent.get_local_stats() for agent in self.agents]
        total_vocab_growth = sum(
            sum(r.vocab_growth_per_agent) for r in round_history
        )

        # F_eff expansion rate
        f_eff = total_vocab_growth / max(total_candidates, 1)

        # Convergence diagnosis
        n_rounds = len(round_history)
        consensus_rounds = sum(1 for r in round_history if r.consensus_reached)
        stagnated = consecutive_zero_progress >= self.config.stagnation_limit

        if total_vocab_growth == 0 and total_consensus == 0:
            diagnosis = (
                f"No consensus reached in {n_rounds} rounds. "
                f"Agents produced {total_candidates} candidates total but none "
                f"resonated above threshold ({self.config.consensus_threshold}). "
                f"Consider lowering consensus_threshold or using more diverse corpora."
            )
        elif total_vocab_growth == 0:
            diagnosis = (
                f"Consensus reached {consensus_rounds}/{n_rounds} rounds but "
                f"no vocabulary growth. Decoded syntheses did not pass fitness "
                f"threshold ({self.config.fitness_threshold})."
            )
        elif stagnated:
            diagnosis = (
                f"Partial expansion -- {total_vocab_growth} new atoms across "
                f"{consensus_rounds}/{n_rounds} consensus rounds, "
                f"then {self.config.stagnation_limit} consecutive zero-progress rounds."
            )
        else:
            diagnosis = (
                f"Active expansion -- {total_vocab_growth} new atoms across "
                f"{consensus_rounds}/{n_rounds} consensus rounds "
                f"(expansion rate: {f_eff:.4f})."
            )

        return SwarmResult(
            rounds_completed=n_rounds,
            total_candidates=total_candidates,
            total_consensus_reached=total_consensus,
            total_walls_broadcast=total_walls,
            total_vocab_growth=total_vocab_growth,
            best_fitness=best_fitness,
            best_source=best_source,
            round_history=round_history,
            per_agent_stats=per_agent_stats,
            convergence_diagnosis=diagnosis,
            f_eff_expansion_rate=f_eff,
            total_walls_reconciled=total_reconciled,
        )

    def _collect_candidates(self) -> List[PhaseMessage]:
        """Run local synthesis on all agents in parallel via ProcessPoolExecutor.

        Each worker process instantiates a fresh ThdseAgent with its own
        FhrrArena (Rust FFI) and Z3 solver, avoiding any IPC serialization
        of unpicklable objects. Only primitive arguments (config, corpus dicts,
        wall phase arrays) cross the process boundary. PhaseMessage results
        contain only float arrays and basic metadata — fully picklable.

        Falls back to sequential execution if only one agent is present.
        """
        if len(self.agents) <= 1:
            all_candidates: List[PhaseMessage] = []
            for agent in self.agents:
                all_candidates.extend(agent.run_local_synthesis(max_cliques=10))
            return all_candidates

        all_candidates: List[PhaseMessage] = []
        with ProcessPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(
                    _agent_synthesis_worker,
                    agent_id=i,
                    config=self.config,
                    corpus_dict=self._agent_corpus_dicts[i],
                    wall_phases_list=self._agent_wall_history[i],
                    max_cliques=10,
                ): i
                for i in range(len(self.agents))
            }
            for future in as_completed(futures):
                candidates = future.result()
                all_candidates.extend(candidates)
        return all_candidates

    def _broadcast_wall_targeted(
        self,
        wall_phases: List[float],
        target_agent_ids: List[int],
        round_index: int = 0,
    ) -> None:
        """Broadcast V_error ONLY to agents that participated in the consensus.

        Agents not in target_agent_ids are not immediately affected. Their
        design space remains intact. The wall is recorded in the global
        ledger for later drift reconciliation.

        Args:
            wall_phases: Phase array of the contradiction vector.
            target_agent_ids: Agent IDs from ConsensusResult.clique_agent_ids.
            round_index: Current round (for drift reconciliation timing).
        """
        targeted = set(target_agent_ids)
        for agent in self.agents:
            if agent.agent_id in targeted:
                agent.receive_wall_broadcast(wall_phases)
                self._agent_wall_history[agent.agent_id].append(wall_phases)

        # Record in wall ledger for drift reconciliation
        entry = WallLedgerEntry(
            wall_phases=wall_phases,
            round_created=round_index,
            source_agent_ids=list(target_agent_ids),
            applied_to=set(targeted),
        )
        self._wall_ledger.append(entry)

    def _broadcast_wall(self, wall_phases: List[float]) -> None:
        """Broadcast V_error to all agents (deprecated).

        Use _broadcast_wall_targeted() instead for localized quotient isolation.
        Kept for backward compatibility.
        """
        self._broadcast_wall_targeted(
            wall_phases, [agent.agent_id for agent in self.agents],
        )

    def _reconcile_drift(self, current_round: int) -> int:
        """Propagate aged walls to agents that haven't received them.

        Only walls older than drift_grace_period are eligible. This gives
        non-participating agents time to exploit structures along the
        contradiction axis before the wall is applied.

        The grace period balances two competing concerns:
          - Too short: equivalent to global broadcast (destroys valid structures)
          - Too long: coordinate systems drift until agents can't communicate

        Returns the number of wall applications performed.
        """
        grace = self.config.drift_grace_period
        all_agent_ids = {a.agent_id for a in self.agents}
        agent_map = {a.agent_id: a for a in self.agents}
        walls_applied = 0

        for entry in self._wall_ledger:
            # Skip walls that haven't aged past the grace period
            age = current_round - entry.round_created
            if age < grace:
                continue

            # Find agents that haven't received this wall yet
            missing = all_agent_ids - entry.applied_to
            for agent_id in sorted(missing):
                agent = agent_map[agent_id]
                agent.receive_wall_broadcast(entry.wall_phases)
                self._agent_wall_history[agent_id].append(entry.wall_phases)
                entry.applied_to.add(agent_id)
                walls_applied += 1

        return walls_applied

    def _broadcast_consensus(self, consensus_phases: List[float]) -> None:
        """Broadcast successful consensus to all agents.

        Each agent attempts to decode and expand vocab from
        the consensus vector.
        """
        for agent in self.agents:
            agent.receive_consensus(consensus_phases)

    def _decode_consensus(
        self, consensus_phases: List[float],
    ) -> Tuple[Optional[str], Optional[List[float]]]:
        """Decode the consensus centroid to Python source.

        Uses the orchestrator's own decoder (not any agent's).

        Returns:
            (decoded_source_or_None, v_error_phases_or_None)
            On SAT success: (source_string, None)
            On UNSAT: (None, v_error_phase_array)
        """
        if self._consensus_decoder is None:
            return None, None

        # Inject consensus into orchestrator's arena
        consensus_h = self._consensus_arena.allocate()
        self._consensus_arena.inject_phases(consensus_h, consensus_phases)

        # Create a LayeredProjection for decoding
        lp = LayeredProjection(
            final_handle=consensus_h,
            ast_handle=consensus_h,
            cfg_handle=None,
            data_handle=None,
            ast_phases=consensus_phases,
            cfg_phases=None,
            data_phases=None,
        )

        source = self._consensus_decoder.decode_to_source(lp)
        if source and source.strip() and source.strip() != "pass":
            return source, None

        # UNSAT: try to extract actual V_error from meta-grammar log
        meta_log = self._consensus_decoder.get_meta_grammar_log()
        for event in reversed(meta_log):
            if (event.unsat_core is not None
                    and event.unsat_core.v_error_handle is not None):
                v_error_phases = list(self._consensus_arena.extract_phases(
                    event.unsat_core.v_error_handle,
                ))
                return None, v_error_phases

        # Fallback: no UNSAT core available
        return None, None

    def ingest_agent_corpora_dicts(
        self, corpora: List[Dict[str, str]],
    ) -> None:
        """Ingest corpora from dictionaries (for testing).

        Args:
            corpora: List of dicts, one per agent (source_id -> source_code).
        """
        if len(corpora) != self.config.n_agents:
            raise ValueError(
                f"Expected {self.config.n_agents} corpora, got {len(corpora)}"
            )

        for i, (agent, corpus) in enumerate(zip(self.agents, corpora)):
            agent.ingest_corpus_dict(corpus)
            self._agent_corpus_dicts[i] = corpus

        # Build merged vocabulary
        self._build_merged_vocab()
