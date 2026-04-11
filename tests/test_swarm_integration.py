"""Phase 6 — Tier-1 wiring tests for the THDSE Swarm Orchestrator.

These tests construct a real :class:`SwarmOrchestrator` with real
bridge instances and exercise the testable pre-round and post-consensus
hooks (``pre_round_goal_injection`` and ``post_consensus_handle``)
without launching the actual ``ProcessPoolExecutor`` worker pool. The
worker path is exercised by Tier-2 once a real corpus + Z3 are
available.

They also enforce PLAN.md Rule 11: the orchestrator's worker function
must accept only serializable primitives. We import
:func:`_agent_synthesis_worker` directly and inspect its signature.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_THDSE = _REPO_ROOT / "thdse"
_CCE = _REPO_ROOT / "Cognitive-Core-Engine-Test"
for p in (str(_REPO_ROOT), str(_THDSE), str(_CCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

if "cognitive_core_engine.core.hdc" not in sys.modules:
    _stub = types.ModuleType("cognitive_core_engine.core.hdc")

    class _StubHV:
        DIM = 10_000

    _stub.HyperVector = _StubHV
    sys.modules["cognitive_core_engine.core.hdc"] = _stub


from shared.arena_manager import ArenaManager  # noqa: E402
from shared.constants import (  # noqa: E402
    CRITIC_THRESHOLD,
    THDSE_ARENA_DIM,
)

from bridges.axiom_skill_bridge import AxiomSkillBridge  # noqa: E402
from bridges.causal_provenance_bridge import CausalProvenanceBridge  # noqa: E402
from bridges.goal_synthesis_bridge import GoalSynthesisBridge  # noqa: E402
from bridges.governance_synthesis_bridge import (  # noqa: E402
    GovernanceSynthesisBridge,
)
from bridges.world_model_swarm_bridge import WorldModelSwarmBridge  # noqa: E402

from src.swarm.orchestrator import (  # noqa: E402
    SwarmOrchestrator,
    _agent_synthesis_worker,
)
from src.swarm.protocol import SwarmConfig  # noqa: E402


def _make_config(n_agents: int = 2) -> SwarmConfig:
    return SwarmConfig(
        n_agents=n_agents,
        dimension=THDSE_ARENA_DIM,
        arena_capacity=200,
        consensus_threshold=0.85,
        fitness_threshold=0.4,
        max_rounds=4,
        stagnation_limit=2,
        corpus_paths=[[] for _ in range(n_agents)],
    )


def _seed_cce_handle(mgr: ArenaManager, namespace: str) -> int:
    rng = mgr.rng.fork(namespace)
    phases = rng.uniform(0.0, 2.0 * math.pi, mgr.cce_dim).astype(np.float32)
    return mgr.alloc_cce(phases=phases)


@pytest.fixture
def manager() -> ArenaManager:
    return ArenaManager(master_seed=6201)


@pytest.fixture
def fully_wired_orchestrator(manager: ArenaManager) -> SwarmOrchestrator:
    return SwarmOrchestrator(
        config=_make_config(n_agents=2),
        arena_manager=manager,
        goal_synthesis_bridge=GoalSynthesisBridge(manager),
        world_model_swarm_bridge=WorldModelSwarmBridge(manager),
        governance_bridge=GovernanceSynthesisBridge(manager),
        axiom_skill_bridge=AxiomSkillBridge(manager),
        provenance_bridge=CausalProvenanceBridge(manager),
    )


# --------------------------------------------------------------------------- #
# Constructor + bridge storage
# --------------------------------------------------------------------------- #


def test_legacy_constructor_keeps_working():
    config = _make_config(2)
    orch = SwarmOrchestrator(config)
    assert orch._arena_manager is None  # noqa: SLF001
    assert orch._goal_synthesis_bridge is None  # noqa: SLF001
    assert orch._world_model_swarm_bridge is None  # noqa: SLF001
    assert len(orch.agents) == 2


def test_constructor_stores_all_bridges(
    fully_wired_orchestrator: SwarmOrchestrator,
):
    assert fully_wired_orchestrator._goal_synthesis_bridge is not None  # noqa: SLF001
    assert fully_wired_orchestrator._world_model_swarm_bridge is not None  # noqa: SLF001
    assert fully_wired_orchestrator._governance_bridge is not None  # noqa: SLF001
    assert fully_wired_orchestrator._axiom_skill_bridge is not None  # noqa: SLF001
    assert fully_wired_orchestrator._provenance_bridge is not None  # noqa: SLF001


# --------------------------------------------------------------------------- #
# pre_round_goal_injection
# --------------------------------------------------------------------------- #


def test_pre_round_goal_injection_ranks_by_priority_similarity(
    manager: ArenaManager, fully_wired_orchestrator: SwarmOrchestrator
):
    gsb = fully_wired_orchestrator._goal_synthesis_bridge  # noqa: SLF001
    goals = []
    for desc, prio in (("low", 0.1), ("high", 0.9), ("mid", 0.5)):
        h = _seed_cce_handle(manager, f"goal_{desc}")
        goals.append(gsb.goal_to_synthesis_target(desc, h, prio))
    ranked = fully_wired_orchestrator.pre_round_goal_injection(goals)
    assert [g["metadata"]["goal_description"] for g in ranked] == [
        "high",
        "mid",
        "low",
    ]
    assert ranked[0]["rank"] == 1
    assert fully_wired_orchestrator.goal_inject_calls == 1


def test_pre_round_goal_injection_no_op_with_empty_input(
    fully_wired_orchestrator: SwarmOrchestrator,
):
    ranked = fully_wired_orchestrator.pre_round_goal_injection([])
    assert ranked == []
    assert fully_wired_orchestrator.goal_inject_calls == 1


def test_pre_round_goal_injection_passthrough_when_no_bridge(
    manager: ArenaManager,
):
    orch = SwarmOrchestrator(_make_config(2), arena_manager=manager)
    candidates = [
        {"id": "a", "priority": 0.1, "projected_similarity": 1.0},
        {"id": "b", "priority": 0.9, "projected_similarity": 1.0},
    ]
    ranked = orch.pre_round_goal_injection(candidates)
    # Without a goal bridge wired we just preserve insertion order
    # and stamp 1-indexed ranks.
    assert [g["id"] for g in ranked] == ["a", "b"]
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2


# --------------------------------------------------------------------------- #
# post_consensus_handle
# --------------------------------------------------------------------------- #


def test_post_consensus_handle_full_pipeline_registers(
    manager: ArenaManager,
    fully_wired_orchestrator: SwarmOrchestrator,
):
    # Allocate a real THDSE handle so the governance bridge can read
    # back its phases when validating the candidate. The orchestrator
    # uses handle 0 by convention in post_consensus_handle.
    manager.alloc_thdse(
        phases=np.zeros(THDSE_ARENA_DIM, dtype=np.float32)
    )
    # Zero-phase consensus → similarity 1.0 → adoption true.
    consensus = [0.0] * THDSE_ARENA_DIM
    fitness = CRITIC_THRESHOLD + 0.1
    result = fully_wired_orchestrator.post_consensus_handle(
        consensus_phases=consensus,
        consensus_source="def evolved():\n    return 99\n",
        fitness=fitness,
        round_idx=3,
    )
    assert result["should_adopt"] is True
    assert result["approved"] is True
    assert result["registered"] is True
    asb = fully_wired_orchestrator._axiom_skill_bridge  # noqa: SLF001
    assert asb.registered_count == 1
    pb = fully_wired_orchestrator._provenance_bridge  # noqa: SLF001
    assert pb.get_swarm_consensus_count() == 1


def test_post_consensus_rejects_when_consensus_below_threshold(
    fully_wired_orchestrator: SwarmOrchestrator,
):
    pi_consensus = [math.pi] * THDSE_ARENA_DIM  # Anti-parallel → sim ~ -1.
    result = fully_wired_orchestrator.post_consensus_handle(
        consensus_phases=pi_consensus,
        consensus_source="def x(): return 1\n",
        fitness=CRITIC_THRESHOLD + 0.1,
        round_idx=4,
    )
    assert result["should_adopt"] is False
    assert result["registered"] is False


def test_post_consensus_skips_registration_without_source(
    fully_wired_orchestrator: SwarmOrchestrator,
):
    result = fully_wired_orchestrator.post_consensus_handle(
        consensus_phases=[0.0] * THDSE_ARENA_DIM,
        consensus_source=None,
        fitness=CRITIC_THRESHOLD + 0.1,
        round_idx=5,
    )
    assert result["should_adopt"] is True
    assert result["approved"] is False
    assert result["registered"] is False


def test_post_consensus_emits_provenance_even_when_unregistered(
    fully_wired_orchestrator: SwarmOrchestrator,
):
    fully_wired_orchestrator.post_consensus_handle(
        consensus_phases=[0.0] * THDSE_ARENA_DIM,
        consensus_source=None,
        fitness=0.0,
        round_idx=6,
    )
    pb = fully_wired_orchestrator._provenance_bridge  # noqa: SLF001
    assert pb.get_swarm_consensus_count() == 1


def test_consensus_check_counter_increments(
    fully_wired_orchestrator: SwarmOrchestrator,
):
    for round_idx in range(3):
        fully_wired_orchestrator.post_consensus_handle(
            consensus_phases=[0.0] * THDSE_ARENA_DIM,
            consensus_source=None,
            fitness=0.0,
            round_idx=round_idx,
        )
    assert fully_wired_orchestrator.consensus_check_calls == 3


# --------------------------------------------------------------------------- #
# PLAN.md Rule 11 — process isolation
# --------------------------------------------------------------------------- #


def test_agent_synthesis_worker_signature_excludes_arena_manager():
    """The worker MUST NOT accept arena_manager — Rule 11 forbids
    passing Rust-allocated objects across process boundaries."""
    sig = inspect.signature(_agent_synthesis_worker)
    forbidden = {"arena_manager", "manager", "provenance_bridge"}
    leaked = set(sig.parameters.keys()) & forbidden
    assert leaked == set(), (
        f"_agent_synthesis_worker must not accept manager or bridges; "
        f"found: {leaked}"
    )


def test_agent_synthesis_worker_only_takes_picklable_types():
    """All worker parameters must annotate plain JSON-friendly types."""
    sig = inspect.signature(_agent_synthesis_worker)
    allowed = {"agent_id", "config", "corpus_dict", "wall_phases_list", "max_cliques"}
    actual = set(sig.parameters.keys())
    assert actual == allowed, f"Worker signature changed: {actual}"
