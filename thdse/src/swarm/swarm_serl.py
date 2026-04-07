"""Swarm-Level SERL — Wraps SwarmOrchestrator in a SERL-compatible measurement
framework. Tracks F_eff across rounds, detects stagnation, produces
SwarmResult output compatible with the existing SERL reporting infrastructure.
"""

from src.swarm.protocol import SwarmConfig
from src.swarm.orchestrator import SwarmOrchestrator, SwarmResult


def run_swarm_serl(
    config: SwarmConfig,
    max_rounds: int = 20,
    stagnation_limit: int = 5,
) -> SwarmResult:
    """Top-level entry point for THDSE-Swarm.

    1. Create SwarmOrchestrator with config
    2. Run orchestrator.run()
    3. Compute aggregate F_eff:
       f_eff = total_new_vocab_atoms_across_all_agents / total_syntheses_attempted
    4. Return SwarmResult

    Args:
        config: SwarmConfig with agent count, dimension, corpora, etc.
        max_rounds: Override config.max_rounds if desired.
        stagnation_limit: Override config.stagnation_limit if desired.

    Returns:
        SwarmResult with complete round history and metrics.
    """
    # Apply overrides
    config.max_rounds = max_rounds
    config.stagnation_limit = stagnation_limit

    orchestrator = SwarmOrchestrator(config)
    result = orchestrator.run()
    return result
