"""
BN-08: Emergence Mechanism Tests

Verifies the mechanism integrity of the recursive self-improvement loop,
not that specific emergence happens (emergence is stochastic).

10 tests covering:
  - CausalChainTracker: empty init, record & verify, reject fakes
  - EnvironmentCoupledFitness: task changes per round
  - Quarantine: rejects constant-output skills
  - GoalGenerator: skill-derived goal uniqueness
  - Integration: closed loop, performance log, emergence depth
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognitive_core_engine.core.causal_chain import CausalChainTracker, CausalEvent
from cognitive_core_engine.omega_forge.benchmark import EnvironmentCoupledFitness
from cognitive_core_engine.core.skills import SkillLibrary


class TestCausalChainEmptyInit(unittest.TestCase):
    """Anti-cheat E8: tracker starts empty with depth 0."""

    def test_causal_chain_empty_init(self):
        tracker = CausalChainTracker()
        self.assertEqual(tracker.max_chain_depth(), 0)
        self.assertEqual(len(tracker.events), 0)
        self.assertEqual(len(tracker.chains), 0)
        self.assertEqual(tracker.skill_birth_count(), 0)
        self.assertEqual(tracker.goal_created_count(), 0)


class TestCausalChainRecordsAndVerifies(unittest.TestCase):
    """Create a depth-2+ chain and verify it passes verification."""

    def test_causal_chain_records_and_verifies(self):
        tracker = CausalChainTracker()

        # Skill born at round 1
        sb_id = tracker.record_skill_birth(
            skill_id="sk_001", genome_fitness=0.6, round_idx=1)

        # Goal created from that skill at round 1
        gc_id = tracker.record_goal_from_skill(
            goal_name="skill_exploit_predict_reward_d4",
            trigger_skill_id="sk_001",
            trigger_event_id=sb_id,
            round_idx=1)

        # Goal achieved at round 3
        ga_id = tracker.record_goal_achieved(
            goal_name="skill_exploit_predict_reward_d4",
            reward=0.5,
            round_idx=3,
            contributing_skill_ids=["sk_001"],
            cause_event_id=gc_id)

        # Verify chain depth >= 2
        self.assertGreaterEqual(tracker.max_chain_depth(), 2)

        # Verify chain integrity
        chains = tracker.chains
        for chain_id in chains:
            if len(chains[chain_id]) >= 2:
                self.assertTrue(tracker.verify_chain(chain_id),
                                f"Chain {chain_id} failed verification")


class TestCausalChainRejectsFake(unittest.TestCase):
    """Create chain with temporal violation — verify_chain returns False."""

    def test_causal_chain_rejects_fake(self):
        tracker = CausalChainTracker()

        # Manually craft events with temporal violation
        # Effect happens at round 1, cause at round 5
        effect_event = CausalEvent(
            event_id="effect_001",
            event_type="goal_created",
            timestamp_round=1,
            cause_event_id="cause_001",
            data={"goal_name": "fake_goal", "trigger_skill_id": "sk_fake"},
        )
        cause_event = CausalEvent(
            event_id="cause_001",
            event_type="skill_born",
            timestamp_round=5,  # AFTER effect — violation
            cause_event_id=None,
            data={"skill_id": "sk_fake", "genome_fitness": 0.5},
        )

        # Add events to tracker manually
        tracker._events.append(cause_event)
        tracker._events_by_id[cause_event.event_id] = cause_event
        tracker._events.append(effect_event)
        tracker._events_by_id[effect_event.event_id] = effect_event

        # Create chain with temporal violation
        tracker._chains["fake_chain"] = ["cause_001", "effect_001"]

        # The chain should fail because effect.cause (round 5) > effect (round 1)
        self.assertFalse(tracker.verify_chain("fake_chain"))


class TestEnvFitnessChangesPerRound(unittest.TestCase):
    """Anti-cheat E1: tasks must differ across consecutive update_tasks() calls."""

    def test_env_fitness_changes_per_round(self):
        ef = EnvironmentCoupledFitness()

        state_1 = {"recent_rewards": [0.3, 0.2, 0.4], "task_count": 6,
                    "round_idx": 1, "stagnation": False}
        ef.update_tasks(state_1)
        tasks_1 = list(ef.task_names)

        state_2 = {"recent_rewards": [0.5, 0.6, 0.3], "task_count": 8,
                    "round_idx": 5, "stagnation": True}
        ef.update_tasks(state_2)
        tasks_2 = list(ef.task_names)

        # Task names must differ between calls
        self.assertNotEqual(tasks_1, tasks_2,
                            "Task sets must differ across consecutive calls")

        # Must generate at least 3 tasks per call
        self.assertGreaterEqual(len(tasks_1), 3)
        self.assertGreaterEqual(len(tasks_2), 3)


class TestQuarantineRejectsConstantSkill(unittest.TestCase):
    """Anti-cheat E3: constant-output genomes are rejected."""

    def test_quarantine_rejects_constant_skill(self):
        from cognitive_core_engine.omega_forge.instructions import Instruction, ProgramGenome
        from cognitive_core_engine.omega_forge.rsi_pipeline import _quarantine_genome

        # Create a genome that always outputs 42.0 in reg0
        # SET r0 42; HALT
        genome = ProgramGenome(
            gid="constant_test",
            instructions=[
                Instruction("SET", 0, 42, 0),
                Instruction("HALT", 0, 0, 0),
            ],
        )
        report = _quarantine_genome(genome)
        # Should fail because output is constant across all inputs
        self.assertFalse(report.passed,
                         "Constant-output genome should be rejected by quarantine")


class TestSkillDerivedGoalUniqueness(unittest.TestCase):
    """Anti-cheat E5/E6: on_skill_registered produces unique goal names."""

    def test_skill_derived_goal_uniqueness(self):
        from agi_modules.competence_map import CompetenceMap
        from agi_modules.goal_generator import GoalGenerator, HARDCODED_TASK_NAMES

        cm = CompetenceMap()
        cm.update("algorithm", 3, 0.5)
        gg = GoalGenerator(cm, None, random.Random(42))

        event1 = {
            "skill_id": "sk_aaa",
            "capabilities": ["predict_reward"],
            "genome_fitness": 0.5,
        }
        goals_1 = gg.on_skill_registered(event1)
        self.assertGreater(len(goals_1), 0)

        # Same skill ID + capability → no duplicate link
        goals_2 = gg.on_skill_registered(event1)
        # Second call should not produce duplicate goals (link already exists)
        for g in goals_2:
            self.assertNotEqual(g.name, goals_1[0].name,
                                "Duplicate goal name produced")

        # Verify no goal name clashes with hardcoded tasks
        for g in goals_1 + goals_2:
            self.assertNotIn(g.name, HARDCODED_TASK_NAMES)


class TestChainIntegrityNoPreseeding(unittest.TestCase):
    """Anti-cheat E8: CausalChainTracker starts completely empty."""

    def test_chain_integrity_no_preseeding(self):
        tracker = CausalChainTracker()
        self.assertEqual(len(tracker.events), 0)
        self.assertEqual(tracker.max_chain_depth(), 0)
        self.assertEqual(tracker.skill_birth_count(), 0)
        self.assertEqual(tracker.goal_created_count(), 0)

        # Verify no chains exist
        self.assertEqual(len(tracker.chains_of_depth(1)), 0)
        self.assertEqual(len(tracker.chains_of_depth(2)), 0)


class TestClosedLoopIntegration(unittest.TestCase):
    """Run 3 rounds of run_recursive_cycle → verify causal tracker exists."""

    def test_closed_loop_integration(self):
        from cognitive_core_engine.core.environment import ResearchEnvironment
        from cognitive_core_engine.core.tools import (
            ToolRegistry, tool_write_note_factory, tool_write_artifact_factory,
            tool_evaluate_candidate, tool_tool_build_report,
        )
        from cognitive_core_engine.core.orchestrator import Orchestrator, OrchestratorConfig

        random.seed(42)
        env = ResearchEnvironment(seed=42)
        tools = ToolRegistry()
        cfg = OrchestratorConfig(agents=4, base_budget=15, selection_top_k=2)
        orch = Orchestrator(cfg, env, tools)
        tools.register("write_note", tool_write_note_factory(orch.mem))
        tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
        tools.register("evaluate_candidate", tool_evaluate_candidate)
        tools.register("tool_build_report", tool_tool_build_report)

        # Run 3 rounds with stagnation override to trigger OmegaForge
        for r in range(3):
            out = orch.run_recursive_cycle(
                r, stagnation_override=(r > 0))

            # Verify emergence keys exist in output
            self.assertIn("emergence_depth", out)
            self.assertIn("emergence_chains", out)
            self.assertIn("total_skill_births", out)
            self.assertIn("skill_derived_goals", out)

        # Causal tracker should exist and be operational
        self.assertIsNotNone(orch.causal_tracker)
        self.assertGreaterEqual(orch.causal_tracker.max_chain_depth(), 0)


class TestSkillPerformanceLogAppendOnly(unittest.TestCase):
    """Anti-cheat E4: performance log is append-only after first entry."""

    def test_skill_performance_log_append_only(self):
        lib = SkillLibrary()
        lib.log_skill_performance("sk_001", 0.5)
        lib.log_skill_performance("sk_001", 0.7)

        self.assertEqual(len(lib.skill_performance_log["sk_001"]), 2)

        # Attempt to clear should raise
        with self.assertRaises(RuntimeError):
            lib.clear_performance_log()

        # Log should still be intact
        self.assertEqual(len(lib.skill_performance_log["sk_001"]), 2)


class TestEmergenceDepthZeroWithoutLoop(unittest.TestCase):
    """AC8: With OmegaForge disabled, emergence_depth must be 0."""

    def test_emergence_depth_zero_without_loop(self):
        from cognitive_core_engine.core.environment import ResearchEnvironment
        from cognitive_core_engine.core.tools import (
            ToolRegistry, tool_write_note_factory, tool_write_artifact_factory,
            tool_evaluate_candidate, tool_tool_build_report,
        )
        from cognitive_core_engine.core.orchestrator import Orchestrator, OrchestratorConfig

        random.seed(42)
        env = ResearchEnvironment(seed=42)
        tools = ToolRegistry()
        cfg = OrchestratorConfig(agents=4, base_budget=15, selection_top_k=2)
        orch = Orchestrator(cfg, env, tools)
        tools.register("write_note", tool_write_note_factory(orch.mem))
        tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
        tools.register("evaluate_candidate", tool_evaluate_candidate)
        tools.register("tool_build_report", tool_tool_build_report)

        # Run without stagnation override (no OmegaForge candidates)
        for r in range(3):
            out = orch.run_recursive_cycle(r, stagnation_override=False)

        # Without OmegaForge, emergence depth must be 0
        self.assertEqual(out["emergence_depth"], 0)
        self.assertEqual(out["total_skill_births"], 0)
        self.assertEqual(out["skill_derived_goals"], 0)


if __name__ == "__main__":
    unittest.main()
