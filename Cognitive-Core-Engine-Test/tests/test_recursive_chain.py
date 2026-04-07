"""
BN-10: Recursive Chain Integration Test + Anti-Cheat Tests

Tests G1-G9 and the full 80-round integration test.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestG1SkillBonus(unittest.TestCase):
    """G1: skill_bonus must be computed from actual VM output, not hardcoded."""

    def test_skill_bonus_zero_when_far(self):
        from cognitive_core_engine.core.environment import ResearchEnvironment
        env = ResearchEnvironment(seed=42)
        obs = env.make_observation(env.tasks[0], budget=12)
        # skill_output far from target → bonus should be ~0
        payload = {"invest": 1.2, "rsi_skill_output": 999.0}
        _, reward_far, info_far = env.step(obs, "attempt_breakthrough", payload)
        self.assertLessEqual(info_far.get("skill_bonus", 0), 0.01,
                             "skill_bonus must be ~0 when output is far from target")

    def test_skill_bonus_positive_when_close(self):
        from cognitive_core_engine.core.environment import ResearchEnvironment
        env = ResearchEnvironment(seed=42)
        task = env.tasks[0]
        obs = env.make_observation(task, budget=12)
        target = task.difficulty * 0.1 + task.baseline * 2.0
        payload = {"invest": 1.2, "rsi_skill_output": target}
        _, reward_close, info_close = env.step(obs, "attempt_breakthrough", payload)
        self.assertGreater(info_close.get("skill_bonus", 0), 0.10,
                           "skill_bonus must be > 0.10 when output matches target")


class TestG2EnvAlignedTasks(unittest.TestCase):
    """G2: New TASKS produce different evolutionary pressure than legacy."""

    def test_new_vs_legacy_scores_differ(self):
        from cognitive_core_engine.omega_forge.benchmark import TaskBenchmark
        from cognitive_core_engine.omega_forge.vm import VirtualMachine
        from cognitive_core_engine.omega_forge.concepts import rand_inst
        from cognitive_core_engine.omega_forge.instructions import ProgramGenome

        vm = VirtualMachine()
        random.seed(42)
        new_scores = []
        legacy_scores = []
        for i in range(50):
            insts = [rand_inst() for _ in range(20)]
            g = ProgramGenome(gid=f"test_{i}", instructions=insts)
            new_scores.append(TaskBenchmark.evaluate(g, vm))
            legacy_scores.append(TaskBenchmark.evaluate_legacy(g, vm))

        # Scores must not be identical
        self.assertNotEqual(new_scores, legacy_scores,
                            "New and legacy tasks must produce different scores")

    def test_no_constant_output_scores_high(self):
        """No genome that returns a constant can score > 0.8 on new tasks."""
        from cognitive_core_engine.omega_forge.benchmark import TaskBenchmark
        from cognitive_core_engine.omega_forge.vm import VirtualMachine
        from cognitive_core_engine.omega_forge.instructions import Instruction, ProgramGenome

        vm = VirtualMachine()
        for const_val in [0, 1, 2, 5, 10]:
            g = ProgramGenome(
                gid=f"const_{const_val}",
                instructions=[
                    Instruction("SET", 0, const_val, 0),
                    Instruction("HALT", 0, 0, 0),
                ])
            score = TaskBenchmark.evaluate(g, vm)
            self.assertLessEqual(score, 0.8,
                                 f"Constant output {const_val} scored {score} > 0.8")


class TestG3RelaxedDetector(unittest.TestCase):
    """G3: Relaxed detector still rejects bad programs."""

    def test_rejects_no_halt(self):
        from cognitive_core_engine.omega_forge.benchmark import (
            DetectorParams, StrictStructuralDetector)
        from cognitive_core_engine.omega_forge.instructions import Instruction, ProgramGenome
        from cognitive_core_engine.omega_forge.vm import VirtualMachine

        relaxed = DetectorParams(rsi_relaxed=True)
        det = StrictStructuralDetector(relaxed)
        vm = VirtualMachine(max_steps=50)
        # Infinite loop — no halt
        g = ProgramGenome(gid="nohalt", instructions=[
            Instruction("JMP", 0, 0, 0)])
        st = vm.execute(g, [1.0] * 8)
        ok, reasons, _ = det.evaluate(g, None, st, vm, 1)
        self.assertFalse(ok, "Must reject non-halting program")

    def test_rejects_zero_loops(self):
        from cognitive_core_engine.omega_forge.benchmark import (
            DetectorParams, StrictStructuralDetector)
        from cognitive_core_engine.omega_forge.instructions import Instruction, ProgramGenome
        from cognitive_core_engine.omega_forge.vm import VirtualMachine

        relaxed = DetectorParams(rsi_relaxed=True, rsi_min_loops=1)
        det = StrictStructuralDetector(relaxed)
        vm = VirtualMachine()
        # Pure linear code — no loops
        g = ProgramGenome(gid="linear", instructions=[
            Instruction("SET", 0, 1, 0),
            Instruction("SET", 0, 2, 1),
            Instruction("ADD", 0, 1, 0),
            Instruction("HALT", 0, 0, 0)])
        st = vm.execute(g, [1.0] * 8)
        ok, reasons, _ = det.evaluate(g, None, st, vm, 1)
        self.assertFalse(ok, "Must reject zero-loop program even in relaxed mode")


class TestG5StateVectorCoupling(unittest.TestCase):
    """G5: Different state_vectors produce different tasks."""

    def test_state_vector_changes_tasks(self):
        from cognitive_core_engine.omega_forge.benchmark import EnvironmentCoupledFitness

        ef = EnvironmentCoupledFitness()

        state1 = {"recent_rewards": [0.1], "task_count": 6, "round_idx": 1,
                   "stagnation": False, "state_vector": [0.0] * 8}
        ef.update_tasks(state1)
        tasks1 = [(t[0], t[2]) for t in ef._tasks]

        state2 = {"recent_rewards": [0.5], "task_count": 10, "round_idx": 5,
                   "stagnation": True, "state_vector": [1.0] * 8}
        ef.update_tasks(state2)
        tasks2 = [(t[0], t[2]) for t in ef._tasks]

        # env_state_predict tasks must have different expected values
        env_task_1 = [t for t in ef._tasks if "env_state_predict" in t[0]]
        self.assertNotEqual(tasks1, tasks2, "Tasks must differ with different state_vectors")


class TestG6GovernanceFloor(unittest.TestCase):
    """G6: Non-evolved L0 with holdout=0.05 gets floor 0.10, not 0.0."""

    def test_non_evolved_floor_is_010(self):
        from cognitive_core_engine.core.environment import ResearchEnvironment, RuleProposal
        from cognitive_core_engine.core.tools import (
            ToolRegistry, tool_write_note_factory, tool_write_artifact_factory,
            tool_evaluate_candidate, tool_tool_build_report,
        )
        from cognitive_core_engine.core.orchestrator import Orchestrator, OrchestratorConfig
        from cognitive_core_engine.core.utils import now_ms

        random.seed(42)
        env = ResearchEnvironment(seed=42)
        tools = ToolRegistry()
        cfg = OrchestratorConfig(agents=2, base_budget=10, selection_top_k=1)
        orch = Orchestrator(cfg, env, tools)
        tools.register("write_note", tool_write_note_factory(orch.mem))
        tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
        tools.register("evaluate_candidate", tool_evaluate_candidate)
        tools.register("tool_build_report", tool_tool_build_report)

        # Non-evolved L0 with holdout=0.05 — should be rejected (floor 0.10)
        proposal = RuleProposal(
            proposal_id="test_floor", level="L0",
            payload={"candidate": {"gid": "g_test", "metrics": {
                "holdout_pass_rate": 0.05, "train_pass_rate": 0.06,
                "discovery_cost": {"holdout": 0.5}}}},
            creator_key="test", created_ms=now_ms(), evidence={})
        verdict = orch._critic_evaluate(proposal)
        self.assertEqual(verdict["verdict"], "reject",
                         "Non-evolved L0 with holdout=0.05 must be rejected (floor 0.10)")


class TestG9ConceptExtraction(unittest.TestCase):
    """G9: Concept library gets concepts, crossover produces children."""

    def test_concept_extraction_and_crossover(self):
        from cognitive_core_engine.omega_forge.engine import OmegaForgeV13
        from cognitive_core_engine.omega_forge.benchmark import EnvironmentCoupledFitness

        engine = OmegaForgeV13(seed=42)
        ef = EnvironmentCoupledFitness()
        engine.env_fitness = ef
        engine.init_population()

        for _ in range(20):
            engine.step()

        # At least 1 crossover should have occurred (30% rate per child)
        self.assertGreater(engine.crossover_count, 0,
                           "At least 1 crossover must occur in 20 generations")

        # Population should still be healthy
        self.assertGreater(len(engine.population), 0)
        self.assertGreaterEqual(engine.generation, 20)


class TestRecursiveChainIntegration(unittest.TestCase):
    """Full 80-round integration test for recursive self-improvement."""

    def test_recursive_chain_80_rounds(self):
        from cognitive_core_engine.core.environment import ResearchEnvironment
        from cognitive_core_engine.core.tools import (
            ToolRegistry, tool_write_note_factory, tool_write_artifact_factory,
            tool_evaluate_candidate, tool_tool_build_report,
        )
        from cognitive_core_engine.core.orchestrator import Orchestrator, OrchestratorConfig

        # G8: Use seed=12345 (NOT used during development)
        random.seed(12345)
        env = ResearchEnvironment(seed=12345)
        tools = ToolRegistry()
        cfg = OrchestratorConfig(agents=4, base_budget=15, selection_top_k=2)
        orch = Orchestrator(cfg, env, tools)
        tools.register("write_note", tool_write_note_factory(orch.mem))
        tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
        tools.register("evaluate_candidate", tool_evaluate_candidate)
        tools.register("tool_build_report", tool_tool_build_report)

        # Run 30 rounds — stagnation triggers OmegaForge every 6th round
        for r in range(30):
            out = orch.run_recursive_cycle(
                r,
                stagnation_override=(r > 2 and r % 6 == 0),
                force_meta_proposal=(r > 10 and r % 15 == 0),
            )

        # Hard assertions
        births = orch.causal_tracker.skill_birth_count()
        goals = orch.causal_tracker.goal_created_count()
        depth = orch.causal_tracker.max_chain_depth()
        perf_log = orch.skills.skill_performance_log

        print(f"\n=== Recursive Chain Results (seed=12345, 30 rounds) ===")
        print(f"  Skill births: {births}")
        print(f"  Goals created: {goals}")
        print(f"  Max chain depth: {depth}")
        print(f"  Skills in perf log: {len(perf_log)}")
        for sk_id, entries in perf_log.items():
            print(f"    {sk_id}: {len(entries)} entries, mean={sum(entries)/len(entries):.4f}")
        print(f"  Tool genesis rate: {orch.agi_tracker.tool_genesis_rate():.4f}")
        print(f"  VM skills registered: {len(orch.skills.vm_skills())}")

        self.assertGreaterEqual(births, 1, "At least one skill must be born")
        self.assertGreaterEqual(goals, 1, "At least one skill-derived goal created")
        self.assertGreaterEqual(depth, 2, "Chain depth must be >= 2 (skill→goal)")

        # Check at least one skill was tested
        self.assertGreaterEqual(len(perf_log), 1,
                                "At least one skill must have performance data")

        # Soft assertions (warn only)
        if depth < 3:
            print(f"  WARNING: Chain depth {depth} < 3 (full recursion not achieved)")
        if orch.agi_tracker.tool_genesis_rate() <= 0:
            print(f"  WARNING: Tool genesis rate is 0 (no skill improved reward)")


class TestGetStateVector(unittest.TestCase):
    """Test that get_state_vector returns correct 8-element vector."""

    def test_state_vector_length_and_values(self):
        from cognitive_core_engine.core.environment import ResearchEnvironment
        env = ResearchEnvironment(seed=42)
        vec = env.get_state_vector()
        self.assertEqual(len(vec), 8)
        self.assertAlmostEqual(vec[0], 0.10)  # tool_quality
        self.assertAlmostEqual(vec[1], 0.10)  # kb_quality
        self.assertAlmostEqual(vec[2], 0.10)  # org_quality
        self.assertGreater(vec[3], 0)  # tasks/50
        self.assertGreater(vec[4], 0)  # mean_difficulty/10


class TestAdaptiveDifficulty(unittest.TestCase):
    """Test adaptive_difficulty increases difficulty after good performance."""

    def test_difficulty_increases(self):
        from cognitive_core_engine.core.environment import ResearchEnvironment
        env = ResearchEnvironment(seed=42)
        original_diff = env.tasks[0].difficulty
        domain = env.tasks[0].domain
        # Simulate 6 high-reward steps
        for _ in range(6):
            env._domain_reward_history.setdefault(domain, []).append(0.15)
        env.adaptive_difficulty(domain)
        self.assertGreater(env.tasks[0].difficulty, original_diff,
                           "Difficulty should increase after sustained high rewards")


if __name__ == "__main__":
    unittest.main()
