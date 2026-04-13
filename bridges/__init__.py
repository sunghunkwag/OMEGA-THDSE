"""OMEGA-THDSE cross-engine bridge modules (PLAN.md Phase 3).

These bridges close five of the ten integration gaps identified in
PLAN.md Section B:

- Gap 2  → :mod:`bridges.concept_axiom_bridge` — CCE concept → THDSE axiom
- Gap 3  → :mod:`bridges.axiom_skill_bridge`   — THDSE axiom → CCE skill
- Gap 4  → :mod:`bridges.causal_provenance_bridge` — Causal ↔ Provenance
- Gap 6  → :mod:`bridges.governance_synthesis_bridge` — Governance → Synthesis
- Gap 8  → :mod:`bridges.goal_synthesis_bridge`  — Goal → Synthesis
- Gap 10 → :mod:`bridges.rsi_serl_bridge`        — RSI ↔ SERL

All bridges depend ONLY on :mod:`shared` (Phase 2) and the Python
standard library. They never import from ``cognitive_core_engine``,
``thdse``, ``governance``, or ``omega_forge`` — those wire-ups happen
in Phase 4. Every public method returns a dict whose ``metadata``
field carries a ``provenance`` sub-dict naming the operation, source
arena(s), and a monotonic timestamp (PLAN.md Rule 9).
"""

__all__ = [
    "concept_axiom_bridge",
    "axiom_skill_bridge",
    "causal_provenance_bridge",
    "governance_synthesis_bridge",
    "goal_synthesis_bridge",
    "rsi_serl_bridge",
    # Phase 9
    "semantic_concept_bridge",
    # Phase 10
    "continuous_learning_bridge",
    # Phase 11
    "memory_architecture_bridge",
    # Phase 12
    "reasoning_bridge",
    # Phase 13
    "agent_environment_bridge",
    # Phase 14
    "synthesis_breakthrough_bridge",
]

__version__ = "0.14.0"
