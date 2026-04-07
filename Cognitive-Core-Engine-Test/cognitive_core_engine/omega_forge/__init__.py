"""
Omega Forge — Structural-transition discovery via CFG analysis.

Dependency order:
  instructions -> cfg -> vm -> concepts -> benchmark -> evidence -> engine -> stage1 -> stage2 -> cli

Re-exports key public symbols used by the orchestrator.
Lazy imports to avoid circular dependency chains.
"""
from cognitive_core_engine.omega_forge.instructions import Instruction, ProgramGenome
