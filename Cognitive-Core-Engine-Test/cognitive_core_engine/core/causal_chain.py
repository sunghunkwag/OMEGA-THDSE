"""
CausalChainTracker — Records recursive emergence chains (BN-08 Phase 4).

Tracks the causal relationships between skill births, goal creations,
goal achievements, and capability expansions.  Each event has an optional
cause_event_id linking it to the event that triggered it.

Anti-cheat:
  E7: verify_chain() validates temporal ordering and referential integrity
  E8: starts EMPTY, no pre-seeded events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class CausalEvent:
    """A single event in the emergence chain."""
    event_id: str
    event_type: str  # "skill_born" | "goal_created" | "goal_achieved" | "skill_used" | "capability_expanded"
    timestamp_round: int
    cause_event_id: Optional[str]
    data: Dict[str, Any] = field(default_factory=dict)


class CausalChainTracker:
    """Tracks recursive emergence chains: skill → goal → achievement → skill.

    Anti-cheat E8: starts EMPTY. No events added during __init__().
    All events must have timestamp_round >= 0 and must be temporally ordered
    within chains (cause.timestamp_round <= effect.timestamp_round).
    """

    def __init__(self) -> None:
        self._events: List[CausalEvent] = []
        self._events_by_id: Dict[str, CausalEvent] = {}
        self._chains: Dict[str, List[str]] = {}
        self._next_event_counter: int = 0

    def _make_event_id(self, prefix: str) -> str:
        """Generate a unique event ID."""
        self._next_event_counter += 1
        return f"{prefix}_{self._next_event_counter:06d}"

    def _append_event(self, event: CausalEvent) -> str:
        """Append an event to the log and index it. Returns event_id."""
        if event.timestamp_round < 0:
            raise ValueError(f"Event timestamp_round must be >= 0, got {event.timestamp_round}")
        self._events.append(event)
        self._events_by_id[event.event_id] = event

        # Build or extend chain
        if event.cause_event_id and event.cause_event_id in self._events_by_id:
            # Find which chain the cause belongs to, or start a new one
            found_chain = None
            for chain_id, chain_events in self._chains.items():
                if event.cause_event_id in chain_events:
                    found_chain = chain_id
                    break
            if found_chain is not None:
                self._chains[found_chain].append(event.event_id)
            else:
                chain_id = f"chain_{len(self._chains):04d}"
                self._chains[chain_id] = [event.cause_event_id, event.event_id]
        elif event.cause_event_id is None:
            # Root event — start a new chain
            chain_id = f"chain_{len(self._chains):04d}"
            self._chains[chain_id] = [event.event_id]

        return event.event_id

    def record_skill_birth(
        self,
        skill_id: str,
        genome_fitness: float,
        round_idx: int,
        cause_event_id: Optional[str] = None,
    ) -> str:
        """Record that a new RSI skill was born from OmegaForge evolution."""
        event = CausalEvent(
            event_id=self._make_event_id("sb"),
            event_type="skill_born",
            timestamp_round=round_idx,
            cause_event_id=cause_event_id,
            data={
                "skill_id": skill_id,
                "genome_fitness": genome_fitness,
            },
        )
        return self._append_event(event)

    def record_goal_from_skill(
        self,
        goal_name: str,
        trigger_skill_id: str,
        trigger_event_id: str,
        round_idx: int,
    ) -> str:
        """Record that a new goal was created because a skill enabled it."""
        event = CausalEvent(
            event_id=self._make_event_id("gc"),
            event_type="goal_created",
            timestamp_round=round_idx,
            cause_event_id=trigger_event_id,
            data={
                "goal_name": goal_name,
                "trigger_skill_id": trigger_skill_id,
            },
        )
        return self._append_event(event)

    def record_goal_achieved(
        self,
        goal_name: str,
        reward: float,
        round_idx: int,
        contributing_skill_ids: List[str],
        cause_event_id: Optional[str] = None,
    ) -> str:
        """Record that a skill-derived goal was achieved with positive reward."""
        event = CausalEvent(
            event_id=self._make_event_id("ga"),
            event_type="goal_achieved",
            timestamp_round=round_idx,
            cause_event_id=cause_event_id,
            data={
                "goal_name": goal_name,
                "reward": reward,
                "contributing_skill_ids": contributing_skill_ids,
            },
        )
        return self._append_event(event)

    def record_skill_used(
        self,
        skill_id: str,
        was_accepted: bool,
        reward: float,
        round_idx: int,
        cause_event_id: Optional[str] = None,
    ) -> str:
        """Record that an RSI skill was consulted during action selection."""
        event = CausalEvent(
            event_id=self._make_event_id("su"),
            event_type="skill_used",
            timestamp_round=round_idx,
            cause_event_id=cause_event_id,
            data={
                "skill_id": skill_id,
                "was_accepted": was_accepted,
                "reward": reward,
            },
        )
        return self._append_event(event)

    def record_capability_expansion(
        self,
        new_domain: str,
        enabled_by_chain: str,
        round_idx: int,
        cause_event_id: Optional[str] = None,
    ) -> str:
        """Record that a new domain was opened by the recursive chain."""
        event = CausalEvent(
            event_id=self._make_event_id("ce"),
            event_type="capability_expanded",
            timestamp_round=round_idx,
            cause_event_id=cause_event_id,
            data={
                "new_domain": new_domain,
                "enabled_by_chain": enabled_by_chain,
            },
        )
        return self._append_event(event)

    def max_chain_depth(self) -> int:
        """Return the length of the longest causal chain (number of events)."""
        if not self._chains:
            return 0
        return max(len(chain) for chain in self._chains.values())

    def chains_of_depth(self, min_depth: int) -> List[List[CausalEvent]]:
        """Return all chains with at least min_depth events."""
        result: List[List[CausalEvent]] = []
        for chain_id, event_ids in self._chains.items():
            if len(event_ids) >= min_depth:
                events = [self._events_by_id[eid] for eid in event_ids
                          if eid in self._events_by_id]
                if len(events) >= min_depth:
                    result.append(events)
        return result

    def verify_chain(self, chain_id: str) -> bool:
        """Verify temporal causality and referential integrity of a chain.

        Anti-cheat E7:
        - Every cause must precede its effect (timestamp_round)
        - Every referenced event must exist in the log
        - No self-references
        - Temporal monotonicity within the chain
        """
        event_ids = self._chains.get(chain_id)
        if not event_ids or len(event_ids) < 2:
            return len(event_ids) == 1 if event_ids else False

        prev_round = -1
        for i, eid in enumerate(event_ids):
            # Check event exists
            event = self._events_by_id.get(eid)
            if event is None:
                return False

            # Check temporal monotonicity
            if event.timestamp_round < prev_round:
                return False
            prev_round = event.timestamp_round

            # Check cause reference validity (for non-root events)
            if i > 0:
                if event.cause_event_id is None:
                    return False
                cause = self._events_by_id.get(event.cause_event_id)
                if cause is None:
                    return False
                # Cause must not be a future event
                if cause.timestamp_round > event.timestamp_round:
                    return False
                # No self-reference
                if event.cause_event_id == event.event_id:
                    return False

            # Semantic validation per link type
            if i > 0:
                prev_event = self._events_by_id.get(event_ids[i - 1])
                if prev_event is None:
                    return False
                # skill_born → goal_created: goal data must reference skill
                if prev_event.event_type == "skill_born" and event.event_type == "goal_created":
                    if not event.data.get("trigger_skill_id"):
                        return False
                # goal_created → goal_achieved: reward must be > 0
                if prev_event.event_type == "goal_created" and event.event_type == "goal_achieved":
                    if event.data.get("reward", 0) <= 0:
                        return False
                # goal_achieved → skill_born: temporal ordering already checked
                # (new skill born after goal achieved)

        return True

    @property
    def events(self) -> List[CausalEvent]:
        """Return a copy of all events (read-only access)."""
        return list(self._events)

    @property
    def chains(self) -> Dict[str, List[str]]:
        """Return a copy of all chains (read-only access)."""
        return dict(self._chains)

    def skill_birth_count(self) -> int:
        """Count of skill_born events."""
        return sum(1 for e in self._events if e.event_type == "skill_born")

    def goal_created_count(self) -> int:
        """Count of goal_created events from skills."""
        return sum(1 for e in self._events if e.event_type == "goal_created")

    # ------------------------------------------------------------------
    # Phase 4: New event types for algorithm synthesis integration
    # ------------------------------------------------------------------

    def record_level_unlock(
        self, level: int, round_idx: int,
        cause_event_id: Optional[str] = None,
    ) -> str:
        """Record when curriculum gate opens a new level."""
        event = CausalEvent(
            event_id=self._make_event_id("lu"),
            event_type="level_unlocked",
            timestamp_round=round_idx,
            cause_event_id=cause_event_id,
            data={"level": level},
        )
        return self._append_event(event)

    def record_challenge_created(
        self, challenge_name: str, creator_agent: str, round_idx: int,
        cause_event_id: Optional[str] = None,
    ) -> str:
        """Record when a challenger agent creates a new task."""
        event = CausalEvent(
            event_id=self._make_event_id("cc"),
            event_type="challenge_created",
            timestamp_round=round_idx,
            cause_event_id=cause_event_id,
            data={"challenge_name": challenge_name, "creator_agent": creator_agent},
        )
        return self._append_event(event)

    def record_sr_task_attempted(
        self, task_name: str, reward: float, round_idx: int,
        cause_event_id: Optional[str] = None,
    ) -> str:
        """Record a self-referential task attempt."""
        event = CausalEvent(
            event_id=self._make_event_id("sr"),
            event_type="sr_task_attempted",
            timestamp_round=round_idx,
            cause_event_id=cause_event_id,
            data={"task_name": task_name, "reward": reward},
        )
        return self._append_event(event)

    def record_program_submitted(
        self, task_name: str, reward: float, agent_name: str, round_idx: int,
        cause_event_id: Optional[str] = None,
    ) -> str:
        """Record any program submission to AlgorithmSynthesisEnvironment."""
        event = CausalEvent(
            event_id=self._make_event_id("ps"),
            event_type="program_submitted",
            timestamp_round=round_idx,
            cause_event_id=cause_event_id,
            data={"task_name": task_name, "reward": reward, "agent_name": agent_name},
        )
        return self._append_event(event)
