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
]

__version__ = "0.2.0"
