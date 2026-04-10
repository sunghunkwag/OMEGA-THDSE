"""Shared Memory / Knowledge Base (Neuro-Symbolic) using HDC for associative retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cognitive_core_engine.core.fhrr import FhrrVector
from cognitive_core_engine.core.utils import stable_hash, now_ms, tokenize


@dataclass
class MemoryItem:
    ts_ms: int
    kind: str               # "episode" | "note" | "artifact" | "principle"
    title: str
    content: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = stable_hash(
            {"ts": self.ts_ms, "k": self.kind, "t": self.title, "c": self.content, "tags": self.tags}
        )


class SharedMemory:
    """
    Shared KB using HDC for associative retrieval.
    """

    def __init__(self, max_items: int = 20000, rng: Any = None) -> None:
        self.max_items = max_items
        self._items: List[MemoryItem] = []
        # HDC Memory Index
        self._item_vectors: Dict[str, FhrrVector] = {}
        # Cache common vectors to speed up encoding
        self._token_cache: Dict[str, FhrrVector] = {}
        # Optional DeterministicRNG fork (Phase 4 / PLAN.md Rule 10).
        # FhrrVector.from_seed is already deterministic, so this is a
        # wiring point for any future random sampling inside memory
        # encoding paths.
        self._rng = rng

    def _get_token_hv(self, token: str) -> FhrrVector:
        if token not in self._token_cache:
            self._token_cache[token] = FhrrVector.from_seed(f"token:{token}")
        return self._token_cache[token]

    def _encode_text_bag(self, text: str) -> FhrrVector:
        """Position-bound HDC encoding for improved similarity separation.

        Uses permute(token_hv, position) to create ORDER-SENSITIVE vectors
        plus multi-resolution bundling for better discrimination.
        """
        tokens = tokenize(text)
        if not tokens:
            # Bug fix (Phase 4): the legacy code referenced HyperVector,
            # which has not existed since the FHRR migration. Use
            # FhrrVector.zero() instead.
            return FhrrVector.zero()

        # Position-bound encoding: hv = Σ(permute(token_hv, position))
        position_vecs = [self._get_token_hv(t).permute(i + 1) for i, t in enumerate(tokens)]

        # Multi-resolution bundling based on text length
        n = len(tokens)
        if n < 5:
            # Short: character-level + token-level dual encoding
            char_tokens = list(text.lower().replace(" ", ""))[:20]
            char_vecs = [self._get_token_hv(f"chr:{c}").permute(i + 1)
                         for i, c in enumerate(char_tokens)]
            all_vecs = position_vecs + (char_vecs if char_vecs else [])
        elif n <= 20:
            # Medium: token-level + bigram-level
            bigram_vecs = []
            for i in range(len(tokens) - 1):
                bg = f"{tokens[i]}_{tokens[i+1]}"
                bigram_vecs.append(self._get_token_hv(f"bg:{bg}").permute(i + 1))
            all_vecs = position_vecs + bigram_vecs
        else:
            # Long: token-level + trigram-level
            trigram_vecs = []
            for i in range(len(tokens) - 2):
                tg = f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}"
                trigram_vecs.append(self._get_token_hv(f"tg:{tg}").permute(i + 1))
            all_vecs = position_vecs + trigram_vecs

        return FhrrVector.bundle(all_vecs)

    def _encode_item(self, item: MemoryItem) -> FhrrVector:
        """Encode a memory item with title-weighted bundling.

        Title gets 3x weight (appears 3 times in bundle) because search queries
        are primarily text that should match the title. Kind and tags are
        supplementary signals. This gives title ~60% of bundle bits.
        """
        # 1. Kind
        kind_hv = self._get_token_hv(f"kind:{item.kind}")

        # 2. Title (primary signal — weighted 3x)
        title_hv = self._encode_text_bag(item.title)

        # 3. Tags
        if item.tags:
            tag_vecs = [self._get_token_hv(f"tag:{t}") for t in item.tags]
            tags_hv = FhrrVector.bundle(tag_vecs)
        else:
            tags_hv = FhrrVector.zero()

        # Title-weighted bundle: title appears 3 times to dominate the encoding
        # This ensures search("algorithm research") strongly matches items
        # titled "algorithm task variant N research"
        return FhrrVector.bundle([title_hv, title_hv, title_hv, kind_hv, tags_hv])

    def add(self, kind: str, title: str, content: Dict[str, Any],
            tags: Optional[List[str]] = None) -> str:
        tags = tags or []
        item = MemoryItem(ts_ms=now_ms(), kind=kind, title=title,
                          content=content, tags=tags)
        self._items.append(item)

        # Generate and store HV
        item_hv = self._encode_item(item)
        self._item_vectors[item.id] = item_hv

        if len(self._items) > self.max_items:
            removed = self._items.pop(0)
            self._item_vectors.pop(removed.id, None)

        return item.id

    def accept_thdse_provenance(self, provenance_event: Dict[str, Any]) -> str:
        """Store a THDSE provenance event as a memory item (Phase 4 Gap 5 wiring).

        ``provenance_event`` should carry at minimum ``source_arena``,
        ``operation``, ``result_similarity``, and ``timestamp``. The
        event is persisted as a ``thdse_provenance``-kind memory item
        whose content includes a ``metadata.provenance`` sub-dict
        (PLAN.md Rule 9).
        """
        if not isinstance(provenance_event, dict):
            raise TypeError(
                f"provenance_event must be a dict, got "
                f"{type(provenance_event).__name__}"
            )
        source_arena = provenance_event.get("source_arena", "unknown")
        operation = provenance_event.get("operation", "unknown")
        result_similarity = provenance_event.get("result_similarity")
        timestamp = provenance_event.get("timestamp", now_ms())
        title = f"thdse_provenance:{operation}:{source_arena}"
        content: Dict[str, Any] = {
            "source_arena": source_arena,
            "operation": operation,
            "result_similarity": result_similarity,
            "timestamp": timestamp,
            "raw_event": dict(provenance_event),
            "metadata": {
                "provenance": {
                    "operation": "accept_thdse_provenance",
                    "source_arena": source_arena,
                    "target_arena": "cce_memory",
                    "timestamp": timestamp,
                }
            },
        }
        return self.add(
            kind="thdse_provenance",
            title=title,
            content=content,
            tags=["thdse", "provenance", source_arena],
        )

    def search(self, query: str, k: int = 10,
               kinds: Optional[List[str]] = None,
               tags: Optional[List[str]] = None) -> List[MemoryItem]:

        # 1. Encode Query
        query_parts = []

        # Text query
        if query:
            query_parts.append(self._encode_text_bag(query))

        # Tags query
        if tags:
            tag_vecs = [self._get_token_hv(f"tag:{t}") for t in tags]
            query_parts.append(FhrrVector.bundle(tag_vecs))

        # Kinds (act as filter, but also can be part of query vector)
        if kinds:
             # We typically don't bundle all kinds, we use kinds as a hard filter.
             pass

        if not query_parts:
            return self._items[-k:]

        query_hv = FhrrVector.bundle(query_parts)

        # 2. Score all items
        t_now = now_ms()
        scored: List[Tuple[float, MemoryItem]] = []

        # Optimization: Pre-filter by kind to reduce HDC checks?
        # Or just check all. 8000 checks is fine.

        for it in self._items:
            if kinds is not None and it.kind not in kinds:
                continue

            # HDC Similarity
            it_vec = self._item_vectors.get(it.id)
            if not it_vec:
                continue

            sim = query_hv.similarity(it_vec)

            # Recency & Reward boost
            recency = 1.0 / (1.0 + (t_now - it.ts_ms) / (1000.0 * 60.0 * 30.0))
            reward = float(it.content.get("reward", 0.0)) if isinstance(it.content, dict) else 0.0
            reward_boost = max(0.0, min(0.5, reward))

            # Composite Score
            # With position-bound encoding, text self-similarity > 0.65.
            # After multi-component bundling (kind+title+tags), matching items
            # score ~0.52-0.55. Threshold: 0.51 (above 0.50 random baseline).
            if sim < 0.51:
                continue  # Below structured-encoding relevance threshold

            # Normalize sim to 0..1 range roughly (0.5 -> 0, 1.0 -> 1)
            norm_sim = max(0.0, (sim - 0.5) * 2.0)

            final_score = norm_sim + 0.35 * recency + reward_boost

            if final_score > 0.1:
                scored.append((final_score, it))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Relevance filter: check domain/kind coherence among top results
        top_candidates = scored[:k * 2]  # Over-fetch for filtering
        if len(top_candidates) > k:
            # Count dominant domain in results
            domain_counts: Dict[str, int] = {}
            for _, it in top_candidates:
                d = it.content.get("obs", {}).get("domain", "") if isinstance(it.content, dict) else ""
                if d:
                    domain_counts[d] = domain_counts.get(d, 0) + 1

            if domain_counts:
                dominant_domain = max(domain_counts, key=lambda d: domain_counts[d])
                dominant_count = domain_counts[dominant_domain]
                # Demote outliers if strong domain coherence
                if dominant_count > len(top_candidates) * 0.4:
                    def _relevance_key(pair: Tuple[float, MemoryItem]) -> float:
                        score, item = pair
                        d = item.content.get("obs", {}).get("domain", "") if isinstance(item.content, dict) else ""
                        if d and d == dominant_domain:
                            return score + 0.1
                        return score
                    top_candidates.sort(key=_relevance_key, reverse=True)

        return [it for _, it in top_candidates[:k]]

    def extract_principles(self, k: int = 6) -> List[str]:
        episodes = [it for it in self._items if it.kind == "episode"]
        if not episodes:
            return []
        episodes.sort(key=lambda it: float(it.content.get("reward", 0.0)), reverse=True)
        selected = episodes[:k]
        created: List[str] = []
        for it in selected:
            obs = it.content.get("obs", {})
            action = it.content.get("action", "")
            reward = float(it.content.get("reward", 0.0))
            conditions = {
                "task": obs.get("task"),
                "domain": obs.get("domain"),
                "difficulty": obs.get("difficulty"),
                "phase": obs.get("phase"),
                "action": action,
            }
            pid = self.add(
                "principle",
                f"pattern:{obs.get('task','task')}:{action}",
                {
                    "conditions": conditions,
                    "reward": reward,
                    "source_episode": it.id,
                },
                tags=["principle", "derived"],
            )
            created.append(pid)
        return created

    def dump_summary(self, k: int = 15) -> List[Dict[str, Any]]:
        tail = self._items[-k:]
        return [
            {
                "id": it.id,
                "ts_ms": it.ts_ms,
                "kind": it.kind,
                "title": it.title,
                "tags": it.tags,
            }
            for it in tail
        ]
