"""Gap 4 — CCE CausalChain ↔ THDSE Provenance bridge.

This module translates THDSE synthesis events (Z3 SAT/UNSAT verdicts,
SERL evolution cycles, swarm consensus checkpoints) into structured
records that CCE's :class:`CausalChainTracker` can ingest in Phase 4.

PLAN.md Rule 8 is the critical constraint here: **every UNSAT result
must be logged**. A silent UNSAT would break causal chain completeness
— the agent would never learn that a proposed axiom was disproved.
:meth:`record_synthesis_event` therefore accepts UNSAT events as a
first-class event type and :meth:`get_unsat_count` exposes a direct
audit counter so tests (and future regression checks) can verify that
no UNSAT was ever dropped on the floor.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

from shared.arena_manager import ArenaManager
from shared.constants import CCE_ARENA_DIM, THDSE_ARENA_DIM


_ALLOWED_EVENT_TYPES: frozenset[str] = frozenset(
    {"sat", "unsat", "serl_cycle", "swarm_consensus"}
)


class CausalProvenanceBridge:
    """Record THDSE synthesis events in a chronological provenance log."""

    def __init__(self, arena_manager: ArenaManager):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._chain: list[dict[str, Any]] = []
        self._per_type_counts: dict[str, int] = {
            t: 0 for t in _ALLOWED_EVENT_TYPES
        }

    # ---- event recording ---- #

    def record_synthesis_event(
        self,
        event_type: str,
        thdse_handle: int | None,
        details: dict | None = None,
    ) -> dict[str, Any]:
        """Append a synthesis event to the causal chain.

        ``event_type`` must be one of ``sat``, ``unsat``, ``serl_cycle``,
        or ``swarm_consensus``. ``thdse_handle`` may be ``None`` for
        events (like ``serl_cycle``) that don't reference a single
        handle. ``details`` is an arbitrary free-form dict carried into
        the event record.

        Raises :class:`ValueError` for any event_type outside the
        allowed set — this prevents silent typos from landing in the
        chain as distinct-looking event categories.
        """
        if event_type not in _ALLOWED_EVENT_TYPES:
            raise ValueError(
                f"event_type {event_type!r} is not allowed; "
                f"expected one of {sorted(_ALLOWED_EVENT_TYPES)}"
            )

        if thdse_handle is not None:
            if not isinstance(thdse_handle, int) or thdse_handle < 0:
                raise TypeError(
                    f"thdse_handle must be a non-negative int or None, "
                    f"got {thdse_handle!r}"
                )

        if details is not None and not isinstance(details, dict):
            raise TypeError(
                f"details must be a dict or None, got "
                f"{type(details).__name__}"
            )

        event_id = self._mint_event_id(event_type, thdse_handle)
        timestamp = time.time()
        sequence_index = len(self._chain)

        event = {
            "event_id": event_id,
            "event_type": event_type,
            "thdse_handle": thdse_handle,
            "sequence_index": sequence_index,
            "timestamp": timestamp,
            "details": dict(details) if details else {},
            "metadata": {
                "arena_backend": self._mgr.backend,
                "provenance": {
                    "operation": "record_synthesis_event",
                    "source_arena": "thdse",
                    "target_arena": "cce",
                    "event_type": event_type,
                    "timestamp": timestamp,
                    "sequence_index": sequence_index,
                },
            },
        }
        self._chain.append(event)
        self._per_type_counts[event_type] += 1

        # Rule 8 guard: the fact that we reached this line means the
        # event was persisted. We explicitly return a shallow copy so
        # callers cannot mutate the stored record by accident.
        return {
            "event_id": event_id,
            "event_type": event_type,
            "sequence_index": sequence_index,
            "metadata": dict(event["metadata"]),
        }

    # ---- querying ---- #

    def get_chain(self) -> list[dict[str, Any]]:
        """Return a chronological copy of every recorded event."""
        return [dict(event) for event in self._chain]

    def get_unsat_count(self) -> int:
        """Return the number of UNSAT events logged (Rule 8 auditor)."""
        return self._per_type_counts["unsat"]

    def get_sat_count(self) -> int:
        return self._per_type_counts["sat"]

    def get_serl_cycle_count(self) -> int:
        return self._per_type_counts["serl_cycle"]

    def get_swarm_consensus_count(self) -> int:
        return self._per_type_counts["swarm_consensus"]

    def total_events(self) -> int:
        return len(self._chain)

    def filter_by_type(self, event_type: str) -> list[dict[str, Any]]:
        if event_type not in _ALLOWED_EVENT_TYPES:
            raise ValueError(
                f"event_type {event_type!r} is not allowed; "
                f"expected one of {sorted(_ALLOWED_EVENT_TYPES)}"
            )
        return [
            dict(e) for e in self._chain if e["event_type"] == event_type
        ]

    def describe_dimensions(self) -> dict[str, int]:
        """Return a sanity snapshot of the arenas the bridge is anchored to."""
        return {
            "cce_dim": CCE_ARENA_DIM,
            "thdse_dim": THDSE_ARENA_DIM,
            "cce_dim_on_manager": self._mgr.cce_dim,
            "thdse_dim_on_manager": self._mgr.thdse_dim,
        }

    # ---- internals ---- #

    @staticmethod
    def _mint_event_id(event_type: str, thdse_handle: int | None) -> str:
        token = uuid.uuid4().hex[:8]
        payload = (
            f"{event_type}|{thdse_handle if thdse_handle is not None else '-'}|"
            f"{token}"
        ).encode("utf-8")
        return f"cpb-{hashlib.blake2b(payload, digest_size=6).hexdigest()}"


__all__ = ["CausalProvenanceBridge"]
