"""THDSE-Swarm — Deterministic multi-agent collective intelligence via FHRR phase arrays.

Agents communicate exclusively via raw float phase arrays (no text, no LLMs).
Consensus is computed algebraically via resonance clique extraction.
"""

# Lazy imports — modules are loaded on first access to avoid import errors
# during incremental development. Use:
#   from src.swarm.protocol import PhaseMessage, SwarmConfig
#   from src.swarm.consensus import ConsensusResult, compute_swarm_consensus
#   from src.swarm.agent import ThdseAgent
#   from src.swarm.orchestrator import SwarmOrchestrator, SwarmRoundResult, SwarmResult
#   from src.swarm.swarm_serl import run_swarm_serl
