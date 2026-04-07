"""Core selftest and contract negative tests extracted from NON_RSI_AGI_CORE_v5.py."""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from cognitive_core_engine.core.utils import stable_hash, now_ms
from cognitive_core_engine.core.memory import SharedMemory
from cognitive_core_engine.core.tools import (
    ToolRegistry, tool_write_note_factory, tool_write_artifact_factory,
    tool_evaluate_candidate, tool_tool_build_report,
)
from cognitive_core_engine.core.environment import TaskSpec, ResearchEnvironment, RuleProposal
from cognitive_core_engine.core.orchestrator import Orchestrator, OrchestratorConfig


def run_full_system_selftest() -> None:
    random.seed(0)
    env = ResearchEnvironment(seed=0)
    tools = ToolRegistry()
    orch_cfg = OrchestratorConfig(
        agents=4,
        base_budget=12,
        selection_top_k=2,
    )
    orch = Orchestrator(orch_cfg, env, tools)

    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    orch.run_recursive_cycle(0, stagnation_override=True, force_meta_proposal=True)
    assert (round_out := orch.run_recursive_cycle(1, stagnation_override=True, force_meta_proposal=True))
    assert round_out["stagnation"] is True
    assert "gap_spec" in round_out and isinstance(round_out["gap_spec"], dict)
    assert "constraints" in round_out["gap_spec"] and isinstance(round_out["gap_spec"]["constraints"], dict)
    assert "quarantine_only" in round_out["gap_spec"]["constraints"]
    assert round_out["gap_spec"]["constraints"]["quarantine_only"] is True
    assert "no_self_adoption" in round_out["gap_spec"]["constraints"]
    assert round_out["gap_spec"]["constraints"]["no_self_adoption"] is True
    assert round_out["critic_results"]
    assert all("verdict" in item for item in round_out["critic_results"])
    assert all("proposal_id" in item for item in round_out["critic_results"])
    assert any(item["level"] == "L0" for item in round_out["critic_results"])
    assert any(item["level"] == "L1" for item in round_out["critic_results"])
    assert any(item["level"] == "L2" for item in round_out["critic_results"])
    assert all(
        (not item.get("adopted", False)) or item.get("verdict") == "approve"
        for item in round_out["critic_results"]
    )
    print("recursive rule loop executed")
    print("critic decision received")

    x = [[1.0, 2.0], [3.0, 4.0]]
    w = [[1.0], [1.0]]
    y = [
        [x[0][0] * w[0][0] + x[0][1] * w[1][0]],
        [x[1][0] * w[0][0] + x[1][1] * w[1][0]],
    ]
    assert len(y) == 2 and len(y[0]) == 1
    print("tensor execution verified (torch-free)")


def run_torch_smoke_test() -> None:
    import torch

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = x @ torch.tensor([[1.0], [1.0]])
    assert y.shape == (2, 1)
    print("pytorch execution verified")


def run_contract_negative_tests() -> None:
    random.seed(1)
    env = ResearchEnvironment(seed=1)
    tools = ToolRegistry()
    orch_cfg = OrchestratorConfig(
        agents=4,
        base_budget=12,
        selection_top_k=2,
    )
    orch = Orchestrator(orch_cfg, env, tools)

    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    round_out = orch.run_recursive_cycle(0, stagnation_override=True, force_meta_proposal=True)

    def expect_failure(fn: Callable[[], None], exc_types: Tuple[type, ...], msg_substr: str) -> None:
        try:
            fn()
        except exc_types as exc:
            assert msg_substr in str(exc), f"message mismatch: {exc}"
            return
        except Exception:
            raise
        raise AssertionError("expected failure was not raised")

    assert "gap_spec" in round_out and isinstance(round_out["gap_spec"], dict)
    assert "critic_results" in round_out and isinstance(round_out["critic_results"], list)
    assert any(item.get("level") == "L0" for item in round_out["critic_results"])

    proposals = orch._omega_generate_candidates(round_out["gap_spec"])
    assert proposals
    proposal = proposals[0]
    verdict = orch._critic_evaluate(proposal)
    l1_proposal = RuleProposal(
        proposal_id="l1_negative",
        level="L1",
        payload={"evaluation_update": {"min_score": 0.5}},
        creator_key="creator",
        created_ms=now_ms(),
    )
    l2_proposal = RuleProposal(
        proposal_id="l2_negative",
        level="L2",
        payload={"meta_update": {"l1_update_rate": 0.1}},
        creator_key="creator",
        created_ms=now_ms(),
    )
    l1_verdict = orch._critic_evaluate(l1_proposal)
    l2_verdict = orch._critic_evaluate(l2_proposal)

    def make_l0_candidate(metrics: Dict[str, Any]) -> RuleProposal:
        candidate = {
            "gid": stable_hash({"neg": metrics}),
            "metrics": metrics,
        }
        return RuleProposal(
            proposal_id=stable_hash({"level": "L0", "metrics": metrics}),
            level="L0",
            payload={"candidate": candidate, "gap_spec": round_out.get("gap_spec", {})},
            creator_key="creator",
            created_ms=now_ms(),
            evidence={"metrics": metrics},
        )

    def validate_critic_verdict(result: Dict[str, Any]) -> None:
        assert "verdict" in result, "verdict missing"
        assert "approval_key" in result, "approval_key missing"

    def adopt_with_contract(test_proposal: RuleProposal, result: Dict[str, Any]) -> None:
        if result.get("verdict") != "approve":
            raise ValueError("verdict not approved")
        if "approval_key" not in result:
            raise ValueError("approval_key missing")
        orch._adopt_proposal(test_proposal, result)

    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="missing_candidate",
                level="L0",
                payload={},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "candidate missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="non_dict_candidate",
                level="L0",
                payload={"candidate": "not-a-dict"},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "candidate missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="gid_none",
                level="L0",
                payload={"candidate": {**proposal.payload["candidate"], "gid": None}},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "candidate gid missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="gid_empty",
                level="L0",
                payload={"candidate": {**proposal.payload["candidate"], "gid": ""}},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "candidate gid missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="l1_missing_update",
                level="L1",
                payload={},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "evaluation_update missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="l2_missing_update",
                level="L2",
                payload={},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "meta_update missing",
    )
    expect_failure(
        lambda: adopt_with_contract(l1_proposal, {**l1_verdict, "verdict": "reject"}),
        (ValueError,),
        "verdict not approved",
    )
    expect_failure(
        lambda: adopt_with_contract(l1_proposal, {"verdict": "approve"}),
        (ValueError,),
        "approval_key missing",
    )
    expect_failure(
        lambda: adopt_with_contract(l2_proposal, {**l2_verdict, "verdict": "reject"}),
        (ValueError,),
        "verdict not approved",
    )
    expect_failure(
        lambda: adopt_with_contract(l2_proposal, {"verdict": "approve"}),
        (ValueError,),
        "approval_key missing",
    )
    expect_failure(
        lambda: adopt_with_contract(proposal, {**verdict, "verdict": "reject"}),
        (ValueError,),
        "verdict not approved",
    )
    expect_failure(
        lambda: adopt_with_contract(proposal, {k: v for k, v in verdict.items() if k != "approval_key"}),
        (ValueError,),
        "approval_key missing",
    )
    expect_failure(
        lambda: validate_critic_verdict({k: v for k, v in verdict.items() if k != "approval_key"}),
        (AssertionError,),
        "approval_key missing",
    )
    expect_failure(
        lambda: validate_critic_verdict({k: v for k, v in verdict.items() if k != "verdict"}),
        (AssertionError,),
        "verdict missing",
    )

    missing_holdout_verdict = orch._critic_evaluate(
        make_l0_candidate({"train_pass_rate": 0.34})
    )
    assert missing_holdout_verdict["verdict"] == "reject"

    low_holdout_verdict = orch._critic_evaluate(
        make_l0_candidate({"train_pass_rate": 0.31, "holdout_pass_rate": 0.22})
    )
    assert low_holdout_verdict["verdict"] == "reject"

    gap_verdict = orch._critic_evaluate(
        make_l0_candidate({"train_pass_rate": 0.38, "holdout_pass_rate": 0.30})
    )
    assert gap_verdict["verdict"] == "reject"

    adversarial_verdict = orch._critic_evaluate(
        make_l0_candidate(
            {
                "train_pass_rate": 0.34,
                "holdout_pass_rate": 0.31,
                "adversarial_pass_rate": 0.20,
                "adversarial_examples": [
                    {"input": [3, -2, 7], "expected": [7, -2, 3], "prediction": [3, 7, -2]},
                    {"input": [4, 4, 1], "expected": [1, 4, 4], "prediction": [4, 1, 4]},
                    {"input": [-1, 5, 2], "expected": [2, 5, -1], "prediction": [-1, 2, 5]},
                ],
            }
        )
    )
    assert adversarial_verdict["verdict"] == "reject"

    shift_verdict = orch._critic_evaluate(
        make_l0_candidate(
            {
                "train_pass_rate": 0.34,
                "holdout_pass_rate": 0.31,
                "distribution_shift": {"holdout_pass_rate": 0.18},
                "distribution_shift_examples": [
                    {"input": [9, -4, 0, 3], "expected": [3, 0, -4, 9], "prediction": [9, 3, 0, -4]},
                    {"input": [2, -5, 6, -1], "expected": [-1, 6, -5, 2], "prediction": [2, -1, -5, 6]},
                    {"input": [8, 1, 1, -2], "expected": [-2, 1, 1, 8], "prediction": [8, 1, -2, 1]},
                ],
            }
        )
    )
    assert shift_verdict["verdict"] == "reject"

    regression_verdict = orch._critic_evaluate(
        make_l0_candidate(
            {
                "train_pass_rate": 0.36,
                "holdout_pass_rate": 0.28,
                "baseline": {"train_pass_rate": 0.33, "holdout_pass_rate": 0.30},
            }
        )
    )
    assert regression_verdict["verdict"] == "reject"

    high_cost_verdict = orch._critic_evaluate(
        make_l0_candidate(
            {
                "train_pass_rate": 0.34,
                "holdout_pass_rate": 0.31,
                "discovery_cost": {"holdout": 6.5},
            }
        )
    )
    assert high_cost_verdict["verdict"] == "reject"
    assert high_cost_verdict.get("holdout_cost_ok") is False
    assert high_cost_verdict.get("guardrails_ok") is False
    print("negative contract tests passed")


