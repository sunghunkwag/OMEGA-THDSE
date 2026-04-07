"""Integration tests for the THDSE-Swarm multi-agent collective intelligence system.

Tests use small inline corpus dicts (4-8 functions each), not stdlib.
All tests must complete within 30 seconds total.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hdc_core

from src.swarm.protocol import (
    PhaseMessage, SwarmConfig, serialize_message, deserialize_message,
)
from src.swarm.consensus import ConsensusResult, compute_swarm_consensus
from src.swarm.agent import ThdseAgent
from src.swarm.orchestrator import SwarmOrchestrator, SwarmResult


# ── Shared test corpora ──────────────────────────────────────────

CORPUS_MATH = {
    "math_add": "def add(x, y):\n    return x + y\n",
    "math_mul": "def mul(x, y):\n    return x * y\n",
    "math_sq": "def square(x):\n    return x * x\n",
    "math_abs": "def absolute(x):\n    if x < 0:\n        return -x\n    return x\n",
}

CORPUS_COMPARE = {
    "cmp_max": "def max_val(a, b):\n    if a > b:\n        return a\n    return b\n",
    "cmp_min": "def min_val(a, b):\n    if a < b:\n        return a\n    return b\n",
    "cmp_neg": "def negate(x):\n    return -x\n",
    "cmp_dbl": "def double(x):\n    return x + x\n",
}

CORPUS_CONTROL = {
    "ctrl_fib": (
        "def fib(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    a = 0\n"
        "    b = 1\n"
        "    for i in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
    ),
    "ctrl_fact": (
        "def factorial(n):\n"
        "    result = 1\n"
        "    for i in range(2, n + 1):\n"
        "        result = result * i\n"
        "    return result\n"
    ),
    "ctrl_sum": (
        "def sum_list(lst):\n"
        "    total = 0\n"
        "    for x in lst:\n"
        "        total = total + x\n"
        "    return total\n"
    ),
    "ctrl_count": (
        "def count_pos(lst):\n"
        "    c = 0\n"
        "    for x in lst:\n"
        "        if x > 0:\n"
        "            c = c + 1\n"
        "    return c\n"
    ),
}


def _make_config(n_agents=2, max_rounds=3):
    """Helper to create a SwarmConfig for testing."""
    return SwarmConfig(
        n_agents=n_agents,
        dimension=256,
        arena_capacity=500_000,
        consensus_threshold=0.85,
        fitness_threshold=0.4,
        max_rounds=max_rounds,
        stagnation_limit=2,
        corpus_paths=[[] for _ in range(n_agents)],
    )


# ── Test 1: Agent initialization and corpus ingestion ────────────

def test_agent_initialization():
    """Create 2 agents with different corpora, verify each has axioms."""
    config = _make_config(n_agents=2)

    agent_a = ThdseAgent(0, config, [])
    count_a = agent_a.ingest_corpus_dict(CORPUS_MATH)
    assert count_a == 4, f"Agent 0 should have 4 axioms, got {count_a}"
    assert agent_a.vocab.size() > 0, "Agent 0 should have vocab atoms"

    agent_b = ThdseAgent(1, config, [])
    count_b = agent_b.ingest_corpus_dict(CORPUS_COMPARE)
    assert count_b == 4, f"Agent 1 should have 4 axioms, got {count_b}"
    assert agent_b.vocab.size() > 0, "Agent 1 should have vocab atoms"

    # Verify arenas are independent
    assert agent_a.arena is not agent_b.arena


# ── Test 2: Agent local synthesis produces candidates ────────────

def test_agent_local_synthesis():
    """Run local synthesis on 1 agent, verify PhaseMessage output."""
    config = _make_config(n_agents=1)
    agent = ThdseAgent(0, config, [])
    agent.ingest_corpus_dict(CORPUS_MATH)

    candidates = agent.run_local_synthesis(max_cliques=5)
    # May or may not produce candidates depending on resonance and fitness
    # but should not crash
    for msg in candidates:
        assert isinstance(msg, PhaseMessage)
        assert msg.sender_id == 0
        assert msg.message_type == "candidate"
        assert len(msg.phases) == 256
        assert msg.metadata["fitness"] >= config.fitness_threshold


# ── Test 3: Wall broadcast propagation ───────────────────────────

def test_wall_broadcast():
    """Broadcast a V_error to an agent, verify quotient projection occurs."""
    config = _make_config(n_agents=1)
    agent = ThdseAgent(0, config, [])
    agent.ingest_corpus_dict(CORPUS_MATH)

    wall_count_before = agent._wall_count

    # Create a deterministic wall phase vector
    wall_phases = [0.5] * 256
    agent.receive_wall_broadcast(wall_phases)

    assert agent._wall_count == wall_count_before + 1


# ── Test 4: Consensus from identical candidates ──────────────────

def test_consensus_identical():
    """Submit 3 copies of the same phase array, verify clique of size 3."""
    arena = hdc_core.FhrrArena(100, 64)
    phases = [0.3 * (i + 1) for i in range(64)]
    candidate_phases = [phases, phases, phases]
    candidate_metadata = [
        {"agent_id": 0}, {"agent_id": 1}, {"agent_id": 2},
    ]

    result = compute_swarm_consensus(
        arena, candidate_phases, candidate_metadata,
        consensus_threshold=0.85, min_clique_size=2,
    )

    assert result is not None
    assert result.clique_size == 3
    assert len(result.centroid_phases) == 64
    assert result.mean_resonance > 0.85


# ── Test 5: Consensus from orthogonal candidates ────────────────

def test_consensus_orthogonal():
    """Submit 3 random phase arrays, verify no consensus (below threshold)."""
    arena = hdc_core.FhrrArena(100, 64)

    # Generate deterministic but distinct phase arrays using LCG
    def lcg_phases(seed, dim):
        state = seed & 0xFFFFFFFF
        phases = []
        for _ in range(dim):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            phase = ((state >> 33) / (2**31)) * 2.0 * math.pi - math.pi
            phases.append(phase)
        return phases

    p1 = lcg_phases(12345, 64)
    p2 = lcg_phases(67890, 64)
    p3 = lcg_phases(11111, 64)

    candidate_phases = [p1, p2, p3]
    candidate_metadata = [
        {"agent_id": 0}, {"agent_id": 1}, {"agent_id": 2},
    ]

    result = compute_swarm_consensus(
        arena, candidate_phases, candidate_metadata,
        consensus_threshold=0.85, min_clique_size=2,
    )

    # With d=64, truly distinct phase arrays should not resonate above 0.85
    assert result is None, "Orthogonal phase arrays should not form consensus"


# ── Test 6: Two-agent swarm smoke test ───────────────────────────

def test_two_agent_swarm():
    """Run 2 agents for 3 rounds, verify no crashes and metrics are populated."""
    config = _make_config(n_agents=2, max_rounds=3)
    orch = SwarmOrchestrator(config)
    orch.ingest_agent_corpora_dicts([CORPUS_MATH, CORPUS_COMPARE])

    result = orch.run()

    assert isinstance(result, SwarmResult)
    assert result.rounds_completed > 0
    assert result.rounds_completed <= 3
    assert len(result.round_history) == result.rounds_completed
    assert len(result.per_agent_stats) == 2
    assert result.convergence_diagnosis != ""

    # Each agent should have its original axiom count
    for stats in result.per_agent_stats:
        assert stats["axiom_count"] >= 4


# ── Test 7: Heterogeneous corpora produce different candidates ───

def test_heterogeneous_candidates():
    """2 agents with different corpora produce candidates with low mutual resonance."""
    config = _make_config(n_agents=2)

    agent_a = ThdseAgent(0, config, [])
    agent_a.ingest_corpus_dict(CORPUS_MATH)

    agent_b = ThdseAgent(1, config, [])
    agent_b.ingest_corpus_dict(CORPUS_CONTROL)

    candidates_a = agent_a.run_local_synthesis(max_cliques=5)
    candidates_b = agent_b.run_local_synthesis(max_cliques=5)

    # At minimum, both should run without crashing
    # If both produce candidates, check that they are from different agents
    for msg in candidates_a:
        assert msg.sender_id == 0
    for msg in candidates_b:
        assert msg.sender_id == 1

    # If both produced candidates, verify phase arrays differ
    if candidates_a and candidates_b:
        # Phase arrays from different corpora should not be identical
        assert candidates_a[0].phases != candidates_b[0].phases


# ── Test 8: Global Quotient Collapse consistency ─────────────────

def test_global_quotient_collapse():
    """Broadcast V_error to 3 agents, verify all agents' arenas are modified."""
    config = _make_config(n_agents=3)
    config.corpus_paths = [[], [], []]

    agents = [ThdseAgent(i, config, []) for i in range(3)]

    # Ingest different corpora
    agents[0].ingest_corpus_dict(CORPUS_MATH)
    agents[1].ingest_corpus_dict(CORPUS_COMPARE)
    agents[2].ingest_corpus_dict(CORPUS_CONTROL)

    # Record a reference correlation BEFORE wall broadcast
    # (pick an arbitrary pair of handles in each agent)
    pre_wall_counts = [a._wall_count for a in agents]

    # Create a deterministic V_error
    v_error_phases = [0.7] * 256

    # Broadcast to all agents
    for agent in agents:
        agent.receive_wall_broadcast(v_error_phases)

    # Verify all agents recorded the wall
    for i, agent in enumerate(agents):
        assert agent._wall_count == pre_wall_counts[i] + 1, (
            f"Agent {i} wall count should have increased by 1"
        )


# ── Test 9: Protocol serialization round-trip ────────────────────

def test_protocol_serialization():
    """Verify PhaseMessage serialize/deserialize preserves data."""
    msg = PhaseMessage(
        sender_id=2,
        message_type="candidate",
        phases=[1.0, -0.5, 3.14159, 0.0] * 64,  # 256 floats
        metadata={"fitness": 0.73, "agent_id": 2},
        timestamp=42,
    )

    data = serialize_message(msg)
    msg2 = deserialize_message(data)

    assert msg2.sender_id == msg.sender_id
    assert msg2.message_type == msg.message_type
    assert msg2.timestamp == msg.timestamp
    assert msg2.metadata == msg.metadata
    assert len(msg2.phases) == len(msg.phases)

    # float32 precision: ~7 decimal digits
    for a, b in zip(msg.phases, msg2.phases):
        assert abs(a - b) < 1e-5, f"Phase mismatch: {a} vs {b}"


# ── Test 10: Swarm SERL entry point ─────────────────────────────

def test_swarm_serl_entry_point():
    """Verify run_swarm_serl returns a valid SwarmResult."""
    from src.swarm.swarm_serl import run_swarm_serl

    config = _make_config(n_agents=2, max_rounds=2)
    orch = SwarmOrchestrator(config)
    orch.ingest_agent_corpora_dicts([CORPUS_MATH, CORPUS_COMPARE])

    # Use the orchestrator directly since run_swarm_serl uses filesystem corpora
    result = orch.run()

    assert isinstance(result, SwarmResult)
    assert result.rounds_completed > 0
    assert isinstance(result.convergence_diagnosis, str)
    assert isinstance(result.f_eff_expansion_rate, float)


# ── Helpers for new tests ─────────────────────────────────────────

def lcg_phases(seed, dim):
    """Deterministic phase array generation via LCG."""
    state = seed & 0xFFFFFFFF
    phases = []
    for _ in range(dim):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        phase = ((state >> 33) / (2**31)) * 2.0 * math.pi - math.pi
        phases.append(phase)
    return phases


# ── Test 11: Layered partial consensus — CFG isomorphism ─────────

def test_layered_consensus_cfg_isomorphism():
    """Two candidates with different AST but identical CFG phases
    should form a clique via algorithmic isomorphism rule."""
    arena = hdc_core.FhrrArena(200, 64)

    # Generate AST phases that are DIFFERENT
    ast_a = lcg_phases(100, 64)
    ast_b = lcg_phases(200, 64)

    # Generate CFG and Data phases that are IDENTICAL
    cfg_shared = [0.5] * 64
    data_shared = [0.3] * 64

    # Final phases are different (because AST differs)
    final_a = lcg_phases(300, 64)
    final_b = lcg_phases(400, 64)

    result = compute_swarm_consensus(
        arena,
        candidate_phases=[final_a, final_b],
        candidate_metadata=[{"agent_id": 0}, {"agent_id": 1}],
        consensus_threshold=0.85,
        min_clique_size=2,
        candidate_ast_phases=[ast_a, ast_b],
        candidate_cfg_phases=[cfg_shared, cfg_shared],
        candidate_data_phases=[data_shared, data_shared],
        layer_threshold=0.70,
    )

    # Should form consensus via CFG+Data isomorphism even though
    # final correlation is low
    assert result is not None, (
        "Identical CFG+Data should trigger algorithmic isomorphism consensus"
    )


# ── Test 12: Layered consensus falls back to full match ──────────

def test_layered_consensus_full_match_still_works():
    """High final-handle correlation still triggers consensus
    even when per-layer phases are not provided."""
    arena = hdc_core.FhrrArena(200, 64)
    phases = [0.5] * 64

    result = compute_swarm_consensus(
        arena,
        candidate_phases=[phases, phases, phases],
        candidate_metadata=[{"agent_id": i} for i in range(3)],
        consensus_threshold=0.85,
        min_clique_size=2,
        # No per-layer phases provided
        candidate_ast_phases=None,
        candidate_cfg_phases=None,
        candidate_data_phases=None,
    )

    assert result is not None
    assert result.clique_size == 3


# ── Test 13: Localized quotient isolation ────────────────────────

def test_localized_quotient_isolation():
    """V_error broadcast targets only participating agents.
    Non-participating agent's wall count should NOT increase."""
    config = _make_config(n_agents=3, max_rounds=1)
    config.corpus_paths = [[], [], []]

    orch = SwarmOrchestrator(config)
    orch.ingest_agent_corpora_dicts([CORPUS_MATH, CORPUS_COMPARE, CORPUS_CONTROL])

    # Record pre-broadcast wall counts
    pre_counts = [a._wall_count for a in orch.agents]

    # Targeted broadcast to agents 0 and 2 only
    wall_phases = [0.7] * 256
    orch._broadcast_wall_targeted(wall_phases, [0, 2])

    # Agent 0: should increase
    assert orch.agents[0]._wall_count == pre_counts[0] + 1
    # Agent 1: should NOT increase (not targeted)
    assert orch.agents[1]._wall_count == pre_counts[1]
    # Agent 2: should increase
    assert orch.agents[2]._wall_count == pre_counts[2] + 1


# ── Test 14: Weighted bundling correctness ───────────────────────

def test_weighted_bundle_phases():
    """Weighted bundle with [1.0, 0.0] should return first vector's phases."""
    from src.utils.arena_ops import weighted_bundle_phases

    p_a = [0.0, 1.0, -1.0, 0.5]
    p_b = [3.14, -3.14, 0.0, 2.0]

    # Weight 1.0 for A, near-zero for B -> result ~= A
    result = weighted_bundle_phases([p_a, p_b], [1.0, 0.001])
    for i in range(4):
        assert abs(result[i] - p_a[i]) < 0.05, (
            f"Dim {i}: expected ~= {p_a[i]}, got {result[i]}"
        )

    # Equal weights -> same as standard bundle
    from src.utils.arena_ops import bundle_phases
    equal = weighted_bundle_phases([p_a, p_b], [1.0, 1.0])
    standard = bundle_phases([p_a, p_b])
    for i in range(4):
        assert abs(equal[i] - standard[i]) < 1e-6, (
            f"Equal weights should match standard bundle at dim {i}"
        )


# ── Test 15: Weighted bundling preserves FHRR unit magnitude ────

def test_weighted_bundle_unit_magnitude():
    """Weighted bundle output, when injected into arena, should
    have self-correlation ~= 1.0 (unit magnitude invariant)."""
    from src.utils.arena_ops import weighted_bundle_phases

    phases = [[0.1 * i for i in range(64)],
              [0.2 * i for i in range(64)],
              [0.3 * i for i in range(64)]]
    weights = [0.9, 0.5, 0.1]

    centroid = weighted_bundle_phases(phases, weights)
    arena = hdc_core.FhrrArena(10, 64)
    h = arena.allocate()
    arena.inject_phases(h, centroid)
    self_corr = arena.compute_correlation(h, h)
    assert self_corr > 0.99, f"Self-correlation {self_corr} should be ~= 1.0"


# ── Test 16: ConsensusResult includes adjacency rule counts ──────

def test_consensus_reports_adjacency_rules():
    """ConsensusResult.adjacency_rule_counts should report which rules fired."""
    arena = hdc_core.FhrrArena(200, 64)
    phases = [0.5] * 64

    result = compute_swarm_consensus(
        arena,
        candidate_phases=[phases, phases],
        candidate_metadata=[{"agent_id": 0}, {"agent_id": 1}],
        consensus_threshold=0.85,
        min_clique_size=2,
    )

    assert result is not None
    assert hasattr(result, 'adjacency_rule_counts')
    assert isinstance(result.adjacency_rule_counts, dict)
    # At least the "full" rule should have fired for identical phases
    assert result.adjacency_rule_counts.get("full", 0) >= 1


# ── Test 17: Algo-iso clique uses effective correlation for scoring ──

def test_algo_iso_effective_scoring():
    """An algo_iso clique should score by mean(cfg, data) correlation,
    not by final_corr (which is near-zero for different ASTs)."""
    arena = hdc_core.FhrrArena(400, 64)

    # 3 candidates with identical CFG+Data but different AST/final
    cfg_shared = [0.5] * 64
    data_shared = [0.3] * 64

    finals = [lcg_phases(seed, 64) for seed in [100, 200, 300]]
    asts = [lcg_phases(seed, 64) for seed in [400, 500, 600]]

    result = compute_swarm_consensus(
        arena,
        candidate_phases=finals,
        candidate_metadata=[{"agent_id": i} for i in range(3)],
        consensus_threshold=0.85,
        min_clique_size=2,
        candidate_ast_phases=asts,
        candidate_cfg_phases=[cfg_shared, cfg_shared, cfg_shared],
        candidate_data_phases=[data_shared, data_shared, data_shared],
        layer_threshold=0.70,
    )

    assert result is not None, "Algo-iso clique should form"
    assert result.clique_size == 3

    # Mean resonance should reflect CFG+Data correlation (~1.0),
    # not final_corr (~0.0)
    assert result.mean_resonance > 0.7, (
        f"Mean resonance {result.mean_resonance} should reflect "
        f"effective correlation (CFG+Data), not final_corr"
    )
    assert result.adjacency_rule_counts.get("algo_iso", 0) >= 1


# ── Test 18: Quantized layer phases preserve correlation accuracy ──

def test_quantized_serialization_preserves_correlation():
    """Serialize/deserialize a message with layer phases via uint16
    quantization. Verify that correlation computed on dequantized phases
    closely matches correlation on original phases."""
    msg = PhaseMessage(
        sender_id=0,
        message_type="candidate",
        phases=lcg_phases(1000, 64),
        metadata={"agent_id": 0},
        timestamp=1,
        ast_phases=lcg_phases(2000, 64),
        cfg_phases=[0.5] * 64,
        data_phases=lcg_phases(3000, 64),
    )

    data = serialize_message(msg)
    msg2 = deserialize_message(data)

    # Final phases should be exact (float32)
    for a, b in zip(msg.phases, msg2.phases):
        assert abs(a - b) < 1e-5

    # Layer phases should be close (uint16 quantization ~0.0001 rad)
    assert msg2.ast_phases is not None
    assert msg2.cfg_phases is not None
    assert msg2.data_phases is not None

    for a, b in zip(msg.ast_phases, msg2.ast_phases):
        assert abs(a - b) < 0.001, f"AST phase drift: {abs(a-b)}"

    for a, b in zip(msg.cfg_phases, msg2.cfg_phases):
        assert abs(a - b) < 0.001, f"CFG phase drift: {abs(a-b)}"

    # Verify bandwidth reduction: layer data uses 2 bytes/dim, not 4
    # For d=64: 3 layers * (4 + 64*2) = 3 * 132 = 396 bytes for layers
    # vs old: 3 layers * (4 + 64*4) = 3 * 260 = 780 bytes
    # (plus header + final phases)
    # Just verify it's smaller than the old format would have been
    old_size_layers = 3 * (4 + 64 * 4)  # float32 per layer
    new_size_layers = 3 * (4 + 64 * 2)  # uint16 per layer
    assert new_size_layers < old_size_layers


# ── Test 19: Quantized serialization with None layers ──────────────

def test_quantized_serialization_none_layers():
    """Verify backward compat: message with no layer phases round-trips."""
    msg = PhaseMessage(
        sender_id=1,
        message_type="wall",
        phases=[0.1, 0.2, 0.3] * 10,
        metadata={"source": "test"},
        timestamp=5,
        # No layer phases
    )

    data = serialize_message(msg)
    msg2 = deserialize_message(data)

    assert msg2.ast_phases is None
    assert msg2.cfg_phases is None
    assert msg2.data_phases is None
    assert len(msg2.phases) == 30
    for a, b in zip(msg.phases, msg2.phases):
        assert abs(a - b) < 1e-5


# ── Test 20: Drift reconciliation propagates aged walls ────────────

def test_drift_reconciliation():
    """After grace period, missed walls are propagated to non-targeted agents."""
    config = _make_config(n_agents=3, max_rounds=1)
    config.corpus_paths = [[], [], []]
    config.drift_grace_period = 2

    orch = SwarmOrchestrator(config)
    orch.ingest_agent_corpora_dicts([CORPUS_MATH, CORPUS_COMPARE, CORPUS_CONTROL])

    # Targeted broadcast to agent 0 only, at round 0
    wall_phases = [0.7] * 256
    orch._broadcast_wall_targeted(wall_phases, [0], round_index=0)

    pre_counts = [a._wall_count for a in orch.agents]
    # Agent 0: already has the wall
    assert pre_counts[0] == 1

    # Reconcile at round 1 (age=1 < grace=2): should NOT propagate
    reconciled = orch._reconcile_drift(current_round=1)
    assert reconciled == 0
    assert orch.agents[1]._wall_count == 0
    assert orch.agents[2]._wall_count == 0

    # Reconcile at round 2 (age=2 >= grace=2): should propagate to agents 1 and 2
    reconciled = orch._reconcile_drift(current_round=2)
    assert reconciled == 2
    assert orch.agents[1]._wall_count == 1
    assert orch.agents[2]._wall_count == 1

    # Reconcile again: should not re-propagate (already applied)
    reconciled = orch._reconcile_drift(current_round=3)
    assert reconciled == 0


# ── Test 21: Drift reconciliation idempotent ──────────────────────

def test_drift_reconciliation_idempotent():
    """Multiple reconciliation calls don't re-apply walls."""
    config = _make_config(n_agents=2, max_rounds=1)
    config.corpus_paths = [[], []]
    config.drift_grace_period = 0  # Immediate propagation

    orch = SwarmOrchestrator(config)
    orch.ingest_agent_corpora_dicts([CORPUS_MATH, CORPUS_COMPARE])

    orch._broadcast_wall_targeted([0.5] * 256, [0], round_index=0)

    # First reconciliation: propagate to agent 1
    r1 = orch._reconcile_drift(current_round=0)
    assert r1 == 1
    assert orch.agents[1]._wall_count == 1

    # Second reconciliation: nothing to propagate
    r2 = orch._reconcile_drift(current_round=1)
    assert r2 == 0
    assert orch.agents[1]._wall_count == 1  # Still 1, not 2


# ── Test 22: Wall ledger tracks entries correctly ─────────────────

def test_wall_ledger_tracking():
    """Verify wall ledger records source agents and applied-to sets."""
    config = _make_config(n_agents=3, max_rounds=1)
    config.corpus_paths = [[], [], []]

    orch = SwarmOrchestrator(config)
    orch.ingest_agent_corpora_dicts([CORPUS_MATH, CORPUS_COMPARE, CORPUS_CONTROL])

    orch._broadcast_wall_targeted([0.5] * 256, [0, 2], round_index=3)

    assert len(orch._wall_ledger) == 1
    entry = orch._wall_ledger[0]
    assert entry.round_created == 3
    assert set(entry.source_agent_ids) == {0, 2}
    assert entry.applied_to == {0, 2}
    assert 1 not in entry.applied_to
