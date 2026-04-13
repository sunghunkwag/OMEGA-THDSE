"""OMEGA-THDSE shared infrastructure.

This package contains the foundation layer for the unified AI core:
invariant constants, typed exceptions, deterministic RNG, the central
arena manager, and the asymmetric dimension bridge connecting CCE
(10k-dim) and THDSE (256-dim) FHRR spaces.

All arena access MUST go through :class:`shared.arena_manager.ArenaManager`.
Any cross-arena vector operation MUST go through
:mod:`shared.dimension_bridge`.
"""

__all__ = [
    "constants",
    "exceptions",
    "deterministic_rng",
    "arena_manager",
    "dimension_bridge",
    # Phase 9 — semantic grounding
    "semantic_encoder",
    "perceptual_grounding",
    # Phase 10 — continuous learning
    "online_learner",
    # Phase 11 — deep memory
    "deep_memory",
    # Phase 12 — reasoning
    "reasoning_engine",
    # Phase 13 — environment + agent loop
    "environment",
    "agent_loop",
    # Phase 14 — synthesis breakthrough
    "synthesis_engine",
]

__version__ = "0.14.0"
