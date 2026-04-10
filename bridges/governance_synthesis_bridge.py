"""Gap 6 — CCE Governance → THDSE Synthesis bridge.

This is the governance chokepoint through which every THDSE-synthesized
candidate passes before it can be registered as a CCE skill. The
bridge enforces three simultaneous invariants, any of which can veto
a candidate:

1. **Parse-ability**: ``program_source`` must compile under
   :func:`ast.parse` — no dead or half-written code.
2. **Critic threshold**: ``fitness_score`` must meet or exceed
   ``CRITIC_THRESHOLD`` (from PLAN.md Section D).
3. **Credible leap**: if a previous fitness has been recorded, the
   delta between the new and previous score must not exceed
   ``MAX_CREDIBLE_LEAP``. A sudden +0.9 jump is treated as a sign of
   fitness-function exploitation (wireheading).

PLAN.md Rule 7 is threaded through here: downstream bridges (notably
:mod:`bridges.axiom_skill_bridge`) should only register candidates for
which :meth:`gate_registration` returns ``True``.
"""

from __future__ import annotations

import ast
import time
from typing import Any

from shared.arena_manager import ArenaManager
from shared.constants import (
    CRITIC_THRESHOLD,
    MAX_CREDIBLE_LEAP,
    THDSE_ARENA_DIM,
)


class GovernanceSynthesisBridge:
    """Validate THDSE candidate programs against CCE governance rules."""

    def __init__(self, arena_manager: ArenaManager):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._previous_fitness: float | None = None
        self._history: list[dict[str, Any]] = []

    # ---- evaluation ---- #

    def evaluate_candidate(
        self,
        program_source: str,
        axiom_handle: int,
        fitness_score: float,
    ) -> dict[str, Any]:
        """Run all three governance checks and return a structured verdict.

        The returned dict always contains ``approved`` (bool),
        ``fitness`` (float, clamped to input), ``reason`` (human-readable
        explanation), and a ``metadata.provenance`` sub-dict. A
        side-effect of calling this method is that the candidate is
        appended to the bridge's history log, so subsequent calls can
        enforce the ``MAX_CREDIBLE_LEAP`` bound against the most recent
        prior fitness.
        """
        if not isinstance(program_source, str):
            raise TypeError(
                f"program_source must be str, got "
                f"{type(program_source).__name__}"
            )
        if not isinstance(fitness_score, (int, float)):
            raise TypeError(
                f"fitness_score must be numeric, got "
                f"{type(fitness_score).__name__}"
            )
        fitness = float(fitness_score)

        # (1) parse check
        parse_ok = True
        parse_reason = ""
        if not program_source.strip():
            parse_ok = False
            parse_reason = "empty program source"
        else:
            try:
                ast.parse(program_source)
            except SyntaxError as exc:
                parse_ok = False
                parse_reason = f"SyntaxError: {exc.msg}"

        # (2) critic threshold
        critic_ok = fitness >= CRITIC_THRESHOLD
        critic_reason = (
            ""
            if critic_ok
            else f"fitness {fitness:.3f} < CRITIC_THRESHOLD {CRITIC_THRESHOLD}"
        )

        # (3) credible leap
        leap_ok = True
        leap_reason = ""
        leap_delta: float | None = None
        if self._previous_fitness is not None:
            leap_delta = fitness - self._previous_fitness
            if abs(leap_delta) > MAX_CREDIBLE_LEAP:
                leap_ok = False
                leap_reason = (
                    f"fitness delta |{leap_delta:+.3f}| exceeds "
                    f"MAX_CREDIBLE_LEAP {MAX_CREDIBLE_LEAP}"
                )

        # (4) anchor the handle exists in the THDSE arena
        try:
            thdse_phases = self._mgr.get_thdse_phases(axiom_handle)
            handle_ok = thdse_phases.shape == (THDSE_ARENA_DIM,)
            handle_reason = (
                ""
                if handle_ok
                else f"THDSE handle shape {tuple(thdse_phases.shape)} invalid"
            )
        except (IndexError, TypeError) as exc:
            handle_ok = False
            handle_reason = f"invalid axiom handle: {exc}"

        approved = parse_ok and critic_ok and leap_ok and handle_ok
        reason = (
            "approved"
            if approved
            else "; ".join(
                r
                for r in (
                    parse_reason,
                    critic_reason,
                    leap_reason,
                    handle_reason,
                )
                if r
            )
        )

        timestamp = time.time()
        record = {
            "approved": approved,
            "fitness": fitness,
            "reason": reason,
            "axiom_handle": int(axiom_handle),
            "parse_ok": parse_ok,
            "critic_ok": critic_ok,
            "leap_ok": leap_ok,
            "handle_ok": handle_ok,
            "leap_delta": leap_delta,
            "timestamp": timestamp,
        }
        self._history.append(record)
        # The "previous" fitness advances even when the candidate is
        # rejected — future candidates must still be compared against
        # the last observed score, not the last *approved* one.
        self._previous_fitness = fitness

        return {
            "approved": approved,
            "fitness": fitness,
            "reason": reason,
            "metadata": {
                "axiom_handle": int(axiom_handle),
                "checks": {
                    "parse_ok": parse_ok,
                    "critic_ok": critic_ok,
                    "leap_ok": leap_ok,
                    "handle_ok": handle_ok,
                },
                "leap_delta": leap_delta,
                "critic_threshold": CRITIC_THRESHOLD,
                "max_credible_leap": MAX_CREDIBLE_LEAP,
                "history_length": len(self._history),
                "provenance": {
                    "operation": "evaluate_candidate",
                    "source_arena": "cce",
                    "target_arena": "thdse",
                    "timestamp": timestamp,
                },
            },
        }

    def gate_registration(self, candidate_result: dict) -> bool:
        """Return ``True`` iff the candidate result is approved.

        Central chokepoint used by :mod:`bridges.axiom_skill_bridge` to
        decide whether to call ``validate_and_register``. The function
        deliberately returns a plain bool (not a dict) so it can be
        used inline as a guard.
        """
        if not isinstance(candidate_result, dict):
            raise TypeError(
                f"candidate_result must be a dict, got "
                f"{type(candidate_result).__name__}"
            )
        return bool(candidate_result.get("approved") is True)

    # ---- introspection ---- #

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def previous_fitness(self) -> float | None:
        return self._previous_fitness

    def approval_rate(self) -> float:
        if not self._history:
            return 0.0
        approved = sum(1 for r in self._history if r["approved"])
        return approved / len(self._history)


__all__ = ["GovernanceSynthesisBridge"]
