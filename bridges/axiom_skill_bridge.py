"""Gap 3 — THDSE Axiom → CCE Skill bridge, with governance gate.

When THDSE's SERL loop produces a candidate program, the program
cannot simply be dropped into CCE's skill library: that would let
unproven code execute inside the agent. This bridge is the single
chokepoint where a synthesized axiom becomes a registered skill, and
it enforces PLAN.md Rule 7 — ``validate_and_register`` raises
:class:`GovernanceError` unless the caller explicitly passes
``governance_approved=True``.

Approval alone is not enough: the program source must also parse as
valid Python (``ast.parse``) and must bind to a non-empty skill name.
If any check fails the registration is rejected with a structured
dict explaining why.
"""

from __future__ import annotations

import ast
import hashlib
import time
from typing import Any

from shared.arena_manager import ArenaManager
from shared.constants import THDSE_ARENA_DIM
from shared.dimension_bridge import cross_arena_similarity, project_down
from shared.exceptions import GovernanceError


class AxiomSkillBridge:
    """Register THDSE axioms as CCE skills after governance approval."""

    def __init__(self, arena_manager: ArenaManager):
        if not isinstance(arena_manager, ArenaManager):
            raise TypeError(
                f"arena_manager must be ArenaManager, got "
                f"{type(arena_manager).__name__}"
            )
        self._mgr = arena_manager
        self._registry: dict[str, dict[str, Any]] = {}
        self._rejections: list[dict[str, Any]] = []

    # ---- approval path ---- #

    def validate_and_register(
        self,
        axiom_handle: int,
        program_source: str,
        skill_name: str,
        governance_approved: bool = False,
    ) -> dict[str, Any]:
        """Register an axiom as a skill, gated by governance approval.

        Raises :class:`GovernanceError` if ``governance_approved`` is
        not ``True`` (PLAN.md Rule 7). Raises :class:`SyntaxError` if
        ``program_source`` does not parse. Raises :class:`ValueError`
        if ``skill_name`` is empty or duplicates an existing entry.
        """
        if governance_approved is not True:
            self._rejections.append(
                {
                    "axiom_handle": int(axiom_handle),
                    "skill_name": skill_name,
                    "reason": "governance_approved must be True",
                    "timestamp": time.time(),
                }
            )
            raise GovernanceError(
                "AxiomSkillBridge.validate_and_register blocked: "
                "governance_approved flag is not True",
                subject=skill_name,
                reason="unapproved",
            )

        if not isinstance(program_source, str) or not program_source.strip():
            raise ValueError("program_source must be a non-empty string")
        ast.parse(program_source)  # raises SyntaxError on bad code

        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ValueError("skill_name must be a non-empty string")
        if skill_name in self._registry:
            raise ValueError(f"skill_name {skill_name!r} already registered")

        thdse_phases = self._mgr.get_thdse_phases(axiom_handle)
        if thdse_phases.shape != (THDSE_ARENA_DIM,):
            raise ValueError(
                f"THDSE handle {axiom_handle} returned wrong-shape phases: "
                f"{tuple(thdse_phases.shape)}"
            )

        # Allocate a CCE "evidence" concept for the axiom so we can
        # measure cross-arena similarity without inventing a 10k vector
        # out of thin air: we use a zero-seeded CCE handle as the
        # ambient reference (standard practice for "empty" concepts).
        zero_cce: list[float] = [0.0] * self._mgr.cce_dim
        cce_handle = self._mgr.alloc_cce(phases=zero_cce)
        sim = cross_arena_similarity(zero_cce, thdse_phases)
        axiom_similarity = float(sim["similarity"])

        skill_id = self._compute_skill_id(
            skill_name, axiom_handle, program_source
        )
        entry = {
            "skill_id": skill_id,
            "skill_name": skill_name,
            "axiom_handle": int(axiom_handle),
            "cce_reference_handle": int(cce_handle),
            "program_source": program_source,
            "axiom_similarity": axiom_similarity,
            "registered_at": time.time(),
        }
        self._registry[skill_name] = entry

        return {
            "registered": True,
            "skill_id": skill_id,
            "skill_name": skill_name,
            "axiom_similarity": axiom_similarity,
            "metadata": {
                "registry_size": len(self._registry),
                "similarity_provenance": sim["metadata"]["provenance"],
                "provenance": {
                    "operation": "validate_and_register",
                    "source_arena": "thdse",
                    "target_arena": "cce",
                    "governance_approved": True,
                    "timestamp": entry["registered_at"],
                },
            },
        }

    # ---- rejection path ---- #

    def reject_unapproved(
        self, axiom_handle: int, reason: str
    ) -> dict[str, Any]:
        """Record an axiom rejection without registration."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        entry = {
            "axiom_handle": int(axiom_handle),
            "reason": reason,
            "timestamp": time.time(),
        }
        self._rejections.append(entry)
        return {
            "registered": False,
            "reason": reason,
            "metadata": {
                "axiom_handle": int(axiom_handle),
                "rejection_count": len(self._rejections),
                "provenance": {
                    "operation": "reject_unapproved",
                    "source_arena": "thdse",
                    "timestamp": entry["timestamp"],
                },
            },
        }

    # ---- introspection ---- #

    @property
    def registered_count(self) -> int:
        return len(self._registry)

    @property
    def rejection_count(self) -> int:
        return len(self._rejections)

    def get_registration(self, skill_name: str) -> dict[str, Any]:
        if skill_name not in self._registry:
            raise KeyError(f"skill_name {skill_name!r} not registered")
        return dict(self._registry[skill_name])

    def list_skills(self) -> list[str]:
        return sorted(self._registry.keys())

    # ---- internals ---- #

    @staticmethod
    def _compute_skill_id(
        skill_name: str, axiom_handle: int, program_source: str
    ) -> str:
        payload = (
            f"{skill_name}|{int(axiom_handle)}|{len(program_source)}|"
            f"{program_source[:64]}"
        ).encode("utf-8")
        return f"skill-{hashlib.blake2b(payload, digest_size=8).hexdigest()}"


__all__ = ["AxiomSkillBridge"]
