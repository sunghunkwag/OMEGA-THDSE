"""Vocabulary Self-Expansion — Feeds successful synthesis results back into
the sub-tree vocabulary, enabling design space growth across SERL cycles.

This is RSI trigger condition #1: design space self-expansion.

When a synthesized+decoded source passes the fitness threshold:
  1. Parse the source into an AST
  2. Extract sub-trees (same depth 2-4 range as SubTreeVocabulary)
  3. Canonicalize each sub-tree (same alpha-renaming + type markers)
  4. Check if the canonical form already exists in the vocabulary
  5. If NEW (hash not in vocab): register as a new SubTreeAtom
  6. Project the new atom through the FHRR pipeline
  7. The next synthesis cycle can now use this atom in decoding

This is the mechanism by which F_eff grows:
  - Cycle 1: vocabulary has atoms from the original corpus
  - Cycle 1 synthesis produces new code that passes fitness
  - Cycle 2: vocabulary has original atoms + new atoms from cycle 1
  - Cycle 2 can now decode structures that were unreachable in cycle 1

The expansion is CONDITIONAL on fitness: garbage code (fitness < threshold)
does not pollute the vocabulary. Only code that compiles, executes, and
produces nontrivial computation enters the vocabulary.

Critical invariant: the expand method is idempotent. Calling it twice with
the same source does not add duplicate atoms. This is enforced by
SubTreeVocabulary's existing SHA-256 deduplication.
"""

from typing import Any

from src.decoder.subtree_vocab import SubTreeVocabulary


class VocabularyExpander:
    """Feeds successful synthesis results back into the sub-tree vocabulary."""

    def expand(
        self,
        source: str,
        fitness: float,
        vocab: SubTreeVocabulary,
        arena: Any,
        projector: Any,
        fitness_threshold: float = 0.4,
    ) -> int:
        """Attempt to expand vocabulary from a successful synthesis.

        The fitness threshold of 0.4 requires the code to compile, execute
        without error, AND return at least 2 distinct values for 5 test
        inputs. This is the minimum bar for "nontrivial computation."

        Args:
            source: Decoded Python source code from synthesis.
            fitness: Fitness score from ExecutionSandbox.
            vocab: The SubTreeVocabulary to expand.
            arena: FHRR arena for projection.
            projector: IsomorphicProjector for projection.
            fitness_threshold: Minimum fitness to allow vocabulary expansion.

        Returns:
            Number of NEW atoms added to the vocabulary (0 if all duplicates
            or fitness below threshold).
        """
        if fitness < fitness_threshold:
            return 0

        # Step 1-4: ingest_source handles parse → extract → canonicalize → dedup
        new_count = vocab.ingest_source(source)

        if new_count == 0:
            return 0

        # Step 5-6: Project newly added atoms
        # project_all skips atoms that already have handles, so this is safe
        vocab.project_all(arena, projector)

        return new_count
