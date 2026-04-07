"""
Governance — Evaluation gate, invention system, and meta-control.

Dependency order:
  utils -> sandbox -> critic -> invention -> engine_types -> evolution -> meta -> autopatch -> loops -> cli
"""
from cognitive_core_engine.governance.critic import critic_evaluate_candidate_packet
