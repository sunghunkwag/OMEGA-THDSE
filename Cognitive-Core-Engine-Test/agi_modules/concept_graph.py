"""
ConceptGraph — Hierarchical abstraction and concept formation.

Serves AGI capability: enables the system to form multi-level abstractions
from raw actions to meta-strategies, supporting transfer and generalization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

try:
    from cognitive_core_engine.core.fhrr import FhrrVector
    _HDC_AVAILABLE = True
except ImportError:
    _HDC_AVAILABLE = False
    FhrrVector = None  # type: ignore

# --- Named constants (Rule 6) ---

# Promotion thresholds (calibrated for env rewards ~0.02-0.15)
PROMOTE_MIN_USAGE = 3       # Minimum uses before promotion considered
PROMOTE_MIN_REWARD = 0.03   # Minimum avg reward for promotion (env baseline)
# Co-occurrence window: 1 for L0→L1 (easy first rung), 2 for higher levels
CO_OCCUR_WINDOW_L0 = 1      # Low bar for first abstraction layer
CO_OCCUR_WINDOW_DEFAULT = 2  # Standard bar for higher layers

# Solo-promotion: a concept with high usage+reward can promote by itself
SOLO_PROMOTE_MIN_USAGE = 6   # Must be well-used to self-promote
SOLO_PROMOTE_MIN_REWARD = 0.04  # Must be reliably successful
SOLO_PROMOTE_MIN_CONTEXTS = 2   # Must have succeeded in multiple contexts

# Pruning thresholds
PRUNE_MIN_USAGE = 3         # Below this usage count, eligible for pruning
PRUNE_MIN_REWARD = 0.2      # Below this avg reward, eligible for pruning

# Target abstraction depth for AGI scoring
TARGET_DEPTH = 5

# Maximum concept levels
MAX_LEVEL = 5

# Level-injection guard: concepts above this level MUST come from promote()
MAX_EXTERNAL_LEVEL = 0      # add_concept() from outside only creates L0


def _concept_hash(name: str, children: List[str]) -> str:
    """Create deterministic concept ID from name and children.

    Why: deduplication — avoid creating the same concept twice.
    Fallback: uses name-only hash if no children.
    """
    data = json.dumps({"name": name, "children": sorted(children)}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:12]


@dataclass
class ConceptNode:
    """A node in the concept hierarchy.

    Level 0 = raw action, 1 = skill, 2 = strategy, 3+ = meta-strategy.
    """
    concept_id: str
    name: str
    level: int
    children: List[str] = field(default_factory=list)
    success_contexts: List[Dict[str, Any]] = field(default_factory=list)
    failure_contexts: List[Dict[str, Any]] = field(default_factory=list)
    creation_round: int = 0
    usage_count: int = 0
    avg_reward: float = 0.0
    _total_reward: float = 0.0
    _promoted_via: str = ""  # tracks how this node was created


class ConceptGraph:
    """Stores and manages hierarchical concept abstractions.

    Why it exists: without concept formation, the agent operates on flat
    action->reward mappings with no ability to generalize or compose strategies.

    Fallback: empty graph returns sensible defaults (depth=0, no promotions).
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ConceptNode] = {}
        self._co_occurrences: Dict[str, Dict[str, int]] = {}

    def add_concept(self, name: str, level: int, children: List[str],
                    context: Dict[str, Any],
                    creation_round: int = 0,
                    _internal_promote: bool = False) -> str:
        """Create a new ConceptNode, deduplicating by children combo.

        Why: builds the concept hierarchy from observed action patterns.
        Fallback: returns existing concept_id if duplicate detected.
        Raises ValueError if level > MAX_EXTERNAL_LEVEL without _internal_promote flag.
        """
        # Level-injection guard: external callers cannot create level > 0
        if level > MAX_EXTERNAL_LEVEL and not _internal_promote:
            raise ValueError(
                f"Level {level} concepts must be created via promote(), not add_concept(). "
                f"External add_concept() is limited to level <= {MAX_EXTERNAL_LEVEL}."
            )

        concept_id = _concept_hash(name, children)

        # Deduplicate
        if concept_id in self._nodes:
            return concept_id

        node = ConceptNode(
            concept_id=concept_id,
            name=name,
            level=min(level, MAX_LEVEL),
            children=list(children),
            success_contexts=[context] if context else [],
            creation_round=creation_round,
        )
        if _internal_promote:
            node._promoted_via = "promote_chain"
        self._nodes[concept_id] = node
        return concept_id

    def record_usage(self, concept_id: str, reward: float,
                     context: Dict[str, Any], success: bool) -> None:
        """Record usage of a concept with outcome.

        Why: tracks statistics needed for promotion and pruning decisions.
        Fallback: no-op if concept_id not found.
        """
        node = self._nodes.get(concept_id)
        if node is None:
            return

        node.usage_count += 1
        node._total_reward += reward
        node.avg_reward = node._total_reward / node.usage_count

        if success:
            node.success_contexts.append(context)
            if len(node.success_contexts) > 50:
                node.success_contexts = node.success_contexts[-50:]
        else:
            node.failure_contexts.append(context)
            if len(node.failure_contexts) > 50:
                node.failure_contexts = node.failure_contexts[-50:]

    def record_co_occurrence(self, concept_a: str, concept_b: str) -> None:
        """Track co-occurrence of concepts for promotion bundling.

        Why: concepts that succeed together may form higher-level abstractions.
        Fallback: no-op if either concept not found.
        """
        if concept_a not in self._nodes or concept_b not in self._nodes:
            return
        if concept_a not in self._co_occurrences:
            self._co_occurrences[concept_a] = {}
        self._co_occurrences[concept_a][concept_b] = (
            self._co_occurrences[concept_a].get(concept_b, 0) + 1
        )

    def _co_occur_threshold(self, level: int) -> int:
        """Return co-occurrence threshold for a given concept level.

        Why: L0->L1 promotion is easy (window=1) to bootstrap the hierarchy;
        higher levels need more evidence.
        """
        if level <= 0:
            return CO_OCCUR_WINDOW_L0
        return CO_OCCUR_WINDOW_DEFAULT

    def promote(self, concept_id: str, current_round: int = 0) -> Optional[str]:
        """Promote a concept to a higher level by bundling with co-occurring peers,
        or via solo-promotion if the concept is strong enough on its own.

        Why: builds abstraction hierarchy — successful concept combos become strategies.
        Fallback: returns None if promotion criteria not met.
        """
        node = self._nodes.get(concept_id)
        if node is None:
            return None

        if node.usage_count < PROMOTE_MIN_USAGE:
            return None
        if node.avg_reward < PROMOTE_MIN_REWARD:
            return None

        threshold = self._co_occur_threshold(node.level)

        # Find co-occurring concepts at the same level
        co_map = self._co_occurrences.get(concept_id, {})
        partners = [
            cid for cid, count in co_map.items()
            if count >= threshold
            and cid in self._nodes
            and self._nodes[cid].level == node.level
            and self._nodes[cid].avg_reward >= PROMOTE_MIN_REWARD
        ]

        if partners:
            # Bundle with top co-occurring partner
            partner_id = max(partners, key=lambda c: co_map[c])
            partner = self._nodes[partner_id]

            new_level = node.level + 1
            new_name = f"L{new_level}:{node.name}+{partner.name}"
            children = [concept_id, partner_id]

            merged_context = {
                "source_concepts": [node.name, partner.name],
                "promotion_round": current_round,
                "promotion_type": "co_occurrence",
            }
            # Propagate parent contexts to child
            for ctx in node.success_contexts[:5]:
                merged_context.setdefault("domains", [])
                if ctx.get("domain"):
                    merged_context["domains"].append(ctx["domain"])

            new_id = self.add_concept(new_name, new_level, children,
                                      merged_context, current_round,
                                      _internal_promote=True)
            # Seed the new concept with usage from parents
            new_node = self._nodes.get(new_id)
            if new_node and new_node.usage_count == 0:
                avg_parent_reward = (node.avg_reward + partner.avg_reward) / 2
                new_node.usage_count = min(node.usage_count, partner.usage_count)
                new_node.avg_reward = avg_parent_reward
                new_node._total_reward = avg_parent_reward * new_node.usage_count
                # Inherit success contexts from parents
                new_node.success_contexts.extend(node.success_contexts[:10])
                new_node.success_contexts.extend(partner.success_contexts[:10])
            return new_id

        # Solo-promotion fallback: a well-used concept with diverse contexts
        if (node.usage_count >= SOLO_PROMOTE_MIN_USAGE
                and node.avg_reward >= SOLO_PROMOTE_MIN_REWARD
                and len(node.success_contexts) >= SOLO_PROMOTE_MIN_CONTEXTS):
            new_level = node.level + 1
            new_name = f"L{new_level}:solo:{node.name}"
            children = [concept_id]

            solo_context = {
                "source_concepts": [node.name],
                "promotion_round": current_round,
                "promotion_type": "solo",
            }
            new_id = self.add_concept(new_name, new_level, children,
                                      solo_context, current_round,
                                      _internal_promote=True)
            new_node = self._nodes.get(new_id)
            if new_node and new_node.usage_count == 0:
                new_node.usage_count = node.usage_count // 2
                new_node.avg_reward = node.avg_reward
                new_node._total_reward = node.avg_reward * new_node.usage_count
                new_node.success_contexts.extend(node.success_contexts[:15])
            return new_id

        return None

    def promote_cascade(self, current_round: int = 0) -> int:
        """Attempt promotion on ALL eligible concepts at every level, bottom-up.

        Why: a single promote() call only promotes one concept. This sweeps the
        entire graph so that L0 promotions create L1 nodes which can then be
        promoted to L2 in the same sweep, etc.
        Fallback: returns 0 if nothing promotes.
        """
        total_promoted = 0
        for level in range(MAX_LEVEL):
            candidates = [n for n in self._nodes.values() if n.level == level]
            for node in candidates:
                result = self.promote(node.concept_id, current_round)
                if result:
                    total_promoted += 1
        return total_promoted

    def sweep_promote_all(self, current_round: int = 0, max_sweeps: int = 3) -> int:
        """Run promote_cascade repeatedly until no new promotions occur.

        Why: a single cascade may create L1 nodes that become promotable to L2
        only after co-occurrence data propagates. Multiple sweeps ensure the
        hierarchy builds to its natural depth.
        Fallback: returns 0 if nothing promotes. Bounded by max_sweeps.
        """
        total = 0
        for _ in range(max_sweeps):
            promoted = self.promote_cascade(current_round)
            if promoted == 0:
                break
            total += promoted
            # Propagate co-occurrences upward: newly created higher-level
            # concepts inherit co-occurrences from their children
            self._propagate_co_occurrences()
        return total

    def _propagate_co_occurrences(self) -> None:
        """Propagate co-occurrence data from children to parent concepts.

        Why: when L0 concepts A and B co-occur, and both are children of L1
        concept X, then X should co-occur with any other L1 concept whose
        children also co-occurred. This enables multi-level promotion.
        """
        for node in list(self._nodes.values()):
            if not node.children:
                continue
            # For each child, find what the child co-occurred with
            for child_id in node.children:
                child_co = self._co_occurrences.get(child_id, {})
                for partner_child_id, count in child_co.items():
                    # Find the parent of this partner child
                    for other_node in self._nodes.values():
                        if (other_node.concept_id != node.concept_id
                                and other_node.level == node.level
                                and partner_child_id in other_node.children):
                            self.record_co_occurrence(
                                node.concept_id, other_node.concept_id)

    def abstract(self, concept_id: str) -> Dict[str, Any]:
        """Return abstract representation of a concept's transfer radius.

        Why: determines which domains/difficulties a concept generalizes across.
        Fallback: returns empty dict if concept not found.
        """
        node = self._nodes.get(concept_id)
        if node is None:
            return {}

        domains: Set[str] = set()
        difficulties: Set[int] = set()

        for ctx in node.success_contexts:
            if "domain" in ctx:
                domains.add(str(ctx["domain"]))
            if "difficulty" in ctx:
                difficulties.add(int(ctx["difficulty"]))

        return {
            "concept_id": concept_id,
            "name": node.name,
            "level": node.level,
            "domains": sorted(domains),
            "difficulties": sorted(difficulties),
            "transfer_radius": len(domains) * len(difficulties),
            "usage_count": node.usage_count,
            "avg_reward": node.avg_reward,
        }

    def analogize(self, source_concept_id: str,
                  target_domain: str) -> Optional[str]:
        """Find if source concept structure could apply in target domain.

        Why: enables transfer learning by reusing proven abstractions.
        Fallback: returns None if source concept doesn't exist.
        Self-loop guard: if source already has success_contexts in target_domain,
        returns source_concept_id directly instead of creating a duplicate node.
        This prevents concept graph pollution and score inflation.
        """
        source = self._nodes.get(source_concept_id)
        if source is None:
            return None

        if not source.success_contexts:
            return None

        # Self-loop guard: skip if source already covers target_domain
        source_domains = {
            str(ctx.get("domain", ""))
            for ctx in source.success_contexts
        }
        if target_domain in source_domains:
            return source_concept_id  # already covers this domain — no new concept needed

        adapted_name = f"analogy:{source.name}->{target_domain}"
        context = {
            "domain": target_domain,
            "source_concept": source_concept_id,
            "analogy": True,
        }
        return self.add_concept(
            adapted_name, 0, [],  # analogized concepts start at L0
            context, source.creation_round
        )

    def prune(self, min_usage: int = PRUNE_MIN_USAGE,
              min_reward: float = PRUNE_MIN_REWARD) -> int:
        """Remove underperforming concepts. Never deletes high-usage concepts.

        Why: prevents concept bloat and focuses on useful abstractions.
        Fallback: returns 0 if nothing to prune.
        """
        to_remove = [
            cid for cid, node in self._nodes.items()
            if node.usage_count < min_usage and node.avg_reward < min_reward
            and node.usage_count <= 10
        ]

        for cid in to_remove:
            del self._nodes[cid]
            self._co_occurrences.pop(cid, None)

        return len(to_remove)

    def depth(self) -> int:
        """Return maximum concept level in the graph.

        Why: measures abstraction capability — AGI target is depth >= 5.
        Fallback: returns 0 if graph is empty.
        """
        if not self._nodes:
            return 0
        return max(node.level for node in self._nodes.values())

    def concepts_at_level(self, level: int) -> List[ConceptNode]:
        """Return all concepts at a given level.

        Why: used for hierarchical planning to select level-appropriate concepts.
        Fallback: returns empty list if none at that level.
        """
        return [n for n in self._nodes.values() if n.level == level]

    def get(self, concept_id: str) -> Optional[ConceptNode]:
        """Retrieve a concept by ID.

        Why: direct access needed by planner and transfer engine.
        Fallback: returns None if not found.
        """
        return self._nodes.get(concept_id)

    def all_concepts(self) -> List[ConceptNode]:
        """Return all concepts.

        Why: needed for iteration by tracker and diagnostics.
        Fallback: returns empty list.
        """
        return list(self._nodes.values())

    def get_vector(self, domain: str) -> Optional[Any]:
        """Generate HDC vector encoding a domain's concept structure.

        Why: enables HDC-based transfer similarity (BN-04) to use real
        concept structure rather than falling back to SequenceMatcher.

        Encodes: for each concept node whose success_contexts contain
        the domain, create a seed vector from name+level+usage, permute
        by level*10+index, then bundle all vectors.

        Returns None if no concepts match the domain or HDC not available.
        """
        if not _HDC_AVAILABLE or HyperVector is None:
            return None

        matching_nodes: List[Any] = []
        for node in self._nodes.values():
            domains_in_ctx = set()
            for ctx in node.success_contexts:
                if isinstance(ctx, dict):
                    d = ctx.get("domain", "")
                    if d:
                        domains_in_ctx.add(d)
            if domain in domains_in_ctx:
                matching_nodes.append(node)

        if not matching_nodes:
            return None

        vectors = []
        for idx, node in enumerate(matching_nodes):
            # Encode concept properties into the seed string
            seed_str = f"{node.name}:L{node.level}:usage{node.usage_count}:r{node.avg_reward:.3f}"
            vec = FhrrVector.from_seed(seed_str)
            # Permute by level * 10 + index to encode structural position
            shift = node.level * 10 + idx
            vec = vec.permute(shift)
            vectors.append(vec)

        if len(vectors) == 1:
            return vectors[0]
        return FhrrVector.bundle(vectors)

    def remove_concept(self, concept_id: str) -> bool:
        """Remove a concept node by ID.

        Why: supports TransferEngine rollback — concepts created during
        a failed transfer need to be cleaned up.

        Returns True if the concept was found and removed, False otherwise.
        """
        if concept_id in self._nodes:
            del self._nodes[concept_id]
            # Clean up co-occurrence references
            for key in list(self._co_occurrences.keys()):
                if concept_id in key:
                    del self._co_occurrences[key]
            return True
        return False

    def size(self) -> int:
        """Return total concept count.

        Why: used by AGI tracker for abstraction scoring.
        Fallback: returns 0.
        """
        return len(self._nodes)

    # --- Integrity checks (anti-cheat) ---

    def assert_no_level_injection(self) -> bool:
        """Verify that all concepts above level 0 were created via promote().

        Why: prevents gaming depth by directly injecting high-level concepts.
        Returns True if integrity holds.
        """
        for node in self._nodes.values():
            if node.level > 0 and node._promoted_via != "promote_chain":
                return False
        return True
