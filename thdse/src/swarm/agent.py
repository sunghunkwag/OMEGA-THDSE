"""ThdseAgent — A single agent in the THDSE-Swarm.

Each agent is a complete, independent THDSE pipeline:
  - FhrrArena (own memory, not shared)
  - IsomorphicProjector
  - AxiomaticSynthesizer
  - ConstraintDecoder (with SubTreeVocabulary)
  - WallArchive (local walls)
  - ExecutionSandbox
  - VocabularyExpander

Communication is via phase arrays only:
  - Receives: V_target (goal injection), V_error (global wall broadcast)
  - Sends: V_cand (candidate solutions)
"""

import os
from typing import Any, Dict, List, Optional

import hdc_core

from src.projection.isomorphic_projector import IsomorphicProjector
from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer
from src.decoder.constraint_decoder import ConstraintDecoder
from src.decoder.subtree_vocab import SubTreeVocabulary
from src.decoder.vocab_expander import VocabularyExpander
from src.synthesis.wall_archive import WallArchive
from src.execution.sandbox import ExecutionSandbox
from src.corpus.ingester import BatchIngester

from src.swarm.protocol import PhaseMessage, SwarmConfig


class ThdseAgent:
    """A single agent in the THDSE-Swarm.

    Each agent is a complete, independent THDSE pipeline.
    Agents do NOT share memory — each has its own FhrrArena.

    Communication is via phase arrays only:
      - Receives: V_target (goal injection), V_error (global wall broadcast)
      - Sends: V_cand (candidate solutions)
    """

    def __init__(
        self,
        agent_id: int,
        config: SwarmConfig,
        corpus_paths: List[str],
    ):
        """Initialize a complete THDSE pipeline for this agent.

        Args:
            agent_id: Unique agent index (0..N-1).
            config: Shared swarm configuration.
            corpus_paths: Directories to ingest as this agent's corpus.
                         MUST be different from other agents' corpora.
        """
        self.agent_id = agent_id
        self.config = config
        self.corpus_paths = corpus_paths

        # Each agent owns its own arena — NOT shared
        self.arena = hdc_core.FhrrArena(config.arena_capacity, config.dimension)
        self.projector = IsomorphicProjector(self.arena, config.dimension)
        self.synthesizer = AxiomaticSynthesizer(
            self.arena, self.projector, resonance_threshold=0.15,
        )
        self.vocab = SubTreeVocabulary()
        self.wall_archive = WallArchive(self.arena, config.dimension)
        self.sandbox = ExecutionSandbox()
        self.expander = VocabularyExpander()

        # Decoder is created after corpus ingestion (needs vocab)
        self.decoder: Optional[ConstraintDecoder] = None

        # Metrics
        self._axiom_count = 0
        self._wall_count = 0
        self._best_fitness = 0.0
        self._timestamp = 0

    def ingest_corpus(self) -> int:
        """Ingest this agent's assigned corpus directories.

        Returns number of axioms ingested.
        Also builds SubTreeVocabulary from ingested sources.
        """
        ingester = BatchIngester(self.synthesizer, verbose=False)

        for corpus_dir in self.corpus_paths:
            if os.path.isdir(corpus_dir):
                ingester.ingest_directory(
                    corpus_dir,
                    project_prefix=os.path.basename(corpus_dir),
                )
                # Also build vocab from the same sources
                for root, _dirs, files in os.walk(corpus_dir):
                    for fname in sorted(files):
                        if fname.endswith(".py"):
                            self.vocab.ingest_file(os.path.join(root, fname))

        # Project vocab atoms
        self.vocab.project_all(self.arena, self.projector)

        # Create decoder with vocabulary and wall archive
        self.decoder = ConstraintDecoder(
            self.arena, self.projector, self.config.dimension,
            activation_threshold=0.04,
            subtree_vocab=self.vocab,
            wall_archive=self.wall_archive,
        )

        self._axiom_count = self.synthesizer.store.count()
        return self._axiom_count

    def ingest_corpus_dict(self, corpus: Dict[str, str]) -> int:
        """Ingest a corpus from a dictionary (source_id -> source_code).

        Useful for testing with inline corpora instead of filesystem directories.

        Returns number of axioms ingested.
        """
        self.synthesizer.ingest_batch(corpus)

        # Build vocab from the same sources
        for source in corpus.values():
            self.vocab.ingest_source(source)

        # Project vocab atoms
        self.vocab.project_all(self.arena, self.projector)

        # Create decoder
        self.decoder = ConstraintDecoder(
            self.arena, self.projector, self.config.dimension,
            activation_threshold=0.04,
            subtree_vocab=self.vocab,
            wall_archive=self.wall_archive,
        )

        self._axiom_count = self.synthesizer.store.count()
        return self._axiom_count

    def inject_target(self, target_phases: List[float]) -> None:
        """Inject a goal vector into this agent's arena.

        The target vector biases this agent's synthesis toward
        a specific region of the design space.
        """
        h = self.arena.allocate()
        self.arena.inject_phases(h, target_phases)

    def run_local_synthesis(self, max_cliques: int = 10) -> List[PhaseMessage]:
        """Run one round of local synthesis.

        Pipeline:
          1. compute_resonance()
          2. extract_cliques(min_size=2)
          3. For each clique: synthesize_from_clique()
          4. For each synthesis: decode_to_source()
          5. For each decoded source: sandbox.execute()
          6. For each fitness > threshold: vocab_expander.expand()
          7. Collect all candidate phase arrays that passed SERL gating

        SERL Gating: Only candidates with fitness >= fitness_threshold
        are emitted. This prevents noise from entering the swarm
        communication channel. Low-quality candidates never leave the agent.

        Returns:
            List of PhaseMessage with message_type="candidate",
            containing the phase arrays of passing candidates.
        """
        if self.decoder is None:
            return []

        if self.synthesizer.store.count() < 2:
            return []

        self.synthesizer.compute_resonance()
        cliques = self.synthesizer.extract_cliques(min_size=2)

        if not cliques:
            return []

        candidates: List[PhaseMessage] = []

        for clique in cliques[:max_cliques]:
            try:
                synth_proj = self.synthesizer.synthesize_from_clique(clique)
            except (ValueError, Exception):
                continue

            # Decode to source
            source = self.decoder.decode_to_source(synth_proj)
            if source is None or source.strip() == "" or source.strip() == "pass":
                continue

            # Execute in sandbox
            profile = self.sandbox.execute(source)

            if profile.fitness > self._best_fitness:
                self._best_fitness = profile.fitness

            # SERL gating: only emit candidates above fitness threshold
            if profile.fitness >= self.config.fitness_threshold:
                # Expand local vocabulary
                self.expander.expand(
                    source, profile.fitness,
                    self.vocab, self.arena, self.projector,
                    fitness_threshold=self.config.fitness_threshold,
                )

                # Extract phase array of the synthesized vector
                phases = list(self.arena.extract_phases(synth_proj.final_handle))
                self._timestamp += 1

                # Extract per-layer phases for layered consensus
                ast_ph = synth_proj.ast_phases if (synth_proj.ast_phases is not None and len(synth_proj.ast_phases) > 0) else None
                cfg_ph = synth_proj.cfg_phases if (synth_proj.cfg_phases is not None and len(synth_proj.cfg_phases) > 0) else None
                data_ph = synth_proj.data_phases if (synth_proj.data_phases is not None and len(synth_proj.data_phases) > 0) else None

                candidates.append(PhaseMessage(
                    sender_id=self.agent_id,
                    message_type="candidate",
                    phases=phases,
                    metadata={
                        "agent_id": self.agent_id,
                        "fitness": profile.fitness,
                        "source_preview": source[:200] if source else None,
                        "clique_size": len(clique),
                    },
                    timestamp=self._timestamp,
                    ast_phases=ast_ph,
                    cfg_phases=cfg_ph,
                    data_phases=data_ph,
                ))

        return candidates

    def receive_wall_broadcast(self, wall_phases: List[float]) -> None:
        """Receive a V_error broadcast from the orchestrator.

        Immediately:
          1. Inject wall_phases into local arena
          2. project_to_quotient_space(v_error_handle)
          3. Record in local WallArchive (as a synthetic wall record)

        This collapses the contradiction axis from this agent's
        entire memory space. The agent's next synthesis round
        will automatically avoid the collapsed region.
        """
        v_error_h = self.arena.allocate()
        self.arena.inject_phases(v_error_h, wall_phases)

        # Project entire arena to quotient space H / <V_error>
        self.arena.project_to_quotient_space(v_error_h)

        self._wall_count += 1

    def receive_consensus(self, consensus_phases: List[float]) -> None:
        """Receive the swarm consensus vector.

        Inject into local arena and attempt to expand local vocabulary
        from it (if it decodes to valid code above fitness threshold).
        """
        if self.decoder is None:
            return

        # Inject consensus into arena
        consensus_h = self.arena.allocate()
        self.arena.inject_phases(consensus_h, consensus_phases)

        # Attempt to decode and expand vocab
        from src.projection.isomorphic_projector import LayeredProjection
        # Create a minimal LayeredProjection for decoding
        lp = LayeredProjection(
            final_handle=consensus_h,
            ast_handle=consensus_h,
            cfg_handle=None,
            data_handle=None,
            ast_phases=consensus_phases,
            cfg_phases=None,
            data_phases=None,
        )

        source = self.decoder.decode_to_source(lp)
        if source and source.strip() and source.strip() != "pass":
            profile = self.sandbox.execute(source)
            if profile.fitness >= self.config.fitness_threshold:
                self.expander.expand(
                    source, profile.fitness,
                    self.vocab, self.arena, self.projector,
                    fitness_threshold=self.config.fitness_threshold,
                )

    def get_local_stats(self) -> Dict[str, Any]:
        """Return agent-level metrics."""
        return {
            "agent_id": self.agent_id,
            "corpus_paths": self.corpus_paths,
            "axiom_count": self.synthesizer.store.count(),
            "wall_count": self._wall_count,
            "vocab_size": self.vocab.size(),
            "best_fitness": self._best_fitness,
        }
