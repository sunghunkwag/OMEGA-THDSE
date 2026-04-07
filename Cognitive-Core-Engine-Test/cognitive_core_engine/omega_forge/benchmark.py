"""
Task benchmark and strict structural detector for the OMEGA_FORGE engine.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from cognitive_core_engine.omega_forge.instructions import (
    ExecutionState,
    ProgramGenome,
)
from cognitive_core_engine.omega_forge.cfg import ControlFlowGraph
from cognitive_core_engine.omega_forge.vm import VirtualMachine


class TaskBenchmark:
    """Evaluates genomes against environment-aligned tasks.

    BN-10 Fix 2: Tasks use the same [tq, kq, oq, difficulty, budget] input
    format as agent skill_input, aligning evolutionary pressure with the
    actual research environment.
    """

    # BN-10: Environment-aligned tasks — inputs match agent skill_input format
    TASKS = [
        # (name, inputs=[tq, kq, oq, difficulty, budget], output_loc, expected)
        ("OPT_ACTION_LOW_INFRA", [0.1, 0.1, 0.1, 3.0, 12.0], "reg0", 1.0),
        ("OPT_ACTION_HIGH_INFRA", [0.8, 0.7, 0.6, 3.0, 12.0], "reg0", 0.0),
        ("OPT_ACTION_HARD_TASK", [0.5, 0.5, 0.5, 8.0, 20.0], "reg0", 0.0),
        ("PREDICT_REWARD_EASY", [0.3, 0.3, 0.3, 2.0, 10.0], "reg0", 0.5),
        ("PREDICT_REWARD_HARD", [0.1, 0.1, 0.1, 8.0, 5.0], "reg0", 0.1),
    ]

    # Legacy tasks kept for backward compatibility
    LEGACY_TASKS = [
        ("SUM_SIMPLE", [1.0, 2.0, 3.0, 4.0, 5.0], "reg0", 15.0),
        ("SUM_SMALL", [2.0, 3.0, 5.0], "reg0", 10.0),
        ("MAX_FIND", [3.0, 7.0, 2.0, 9.0, 1.0], "reg0", 9.0),
        ("COUNT", [1.0, 1.0, 1.0, 1.0], "reg0", 4.0),
        ("DOUBLE_FIRST", [5.0, 0.0, 0.0, 0.0], "mem0", 10.0),
    ]

    @staticmethod
    def evaluate(genome: "ProgramGenome", vm: "VirtualMachine") -> float:
        """Returns task score 0.0-1.0 based on practical task performance."""
        passed = 0
        total = len(TaskBenchmark.TASKS)

        for name, inputs, out_loc, expected in TaskBenchmark.TASKS:
            try:
                st = vm.execute(genome, inputs)
                if out_loc == "reg0":
                    result = st.regs[0]
                elif out_loc == "mem0":
                    result = st.memory.get(0, 0.0)
                else:
                    result = 0.0

                # Check if result matches expected (with tolerance)
                if abs(result - expected) < 0.01:
                    passed += 1
                # Partial credit for being close
                elif abs(result - expected) < expected * 0.1:
                    passed += 0.5
            except:
                pass

        return passed / total

    @staticmethod
    def evaluate_legacy(genome: "ProgramGenome", vm: "VirtualMachine") -> float:
        """Evaluate against legacy math tasks (backward compat)."""
        passed = 0
        total = len(TaskBenchmark.LEGACY_TASKS)
        for name, inputs, out_loc, expected in TaskBenchmark.LEGACY_TASKS:
            try:
                st = vm.execute(genome, inputs)
                if out_loc == "reg0":
                    result = st.regs[0]
                elif out_loc == "mem0":
                    result = st.memory.get(0, 0.0)
                else:
                    result = 0.0
                if abs(result - expected) < 0.01:
                    passed += 1
                elif expected != 0 and abs(result - expected) < abs(expected) * 0.1:
                    passed += 0.5
            except Exception:
                pass
        return passed / max(1, total)


class EnvironmentCoupledFitness:
    """Evaluates genomes against dynamically generated environment-derived tasks.

    BN-08 Phase 1: Tasks are generated FROM the current environment state rather
    than being a static list.  This creates evolutionary pressure that co-evolves
    with the research environment.

    Anti-cheat E1: update_tasks() generates >= 3 tasks per call, tasks must
    differ across consecutive calls.
    Anti-cheat E2: fitness is evaluated AFTER VM execution, not callable during it.
    """

    def __init__(self) -> None:
        self._tasks: List[Tuple[str, List[float], float]] = []
        self._prev_task_hash: int = 0
        self._update_count: int = 0
        # Initialize with baseline tasks derived from default environment
        self._generate_baseline_tasks()

    def _generate_baseline_tasks(self) -> None:
        """Create initial task set before any environment state is available."""
        self._tasks = [
            ("predict_reward_baseline", [0.5, 0.3, 0.2, 3.0, 12.0], 0.5),
            ("action_selection_low", [0.1, 0.1, 0.1, 1.0, 8.0], 0.0),
            ("action_selection_high", [0.9, 0.8, 0.7, 5.0, 20.0], 2.0),
        ]

    def update_tasks(self, env_state: Dict[str, Any]) -> None:
        """Refresh tasks based on current environment state.

        Generates at least 3 tasks from environment observations:
        1. Predict-reward: given [task_quality, knowledge_quality, org_quality,
           difficulty, budget], predict expected reward
        2. Optimal-action: given past rewards, predict best action index
        3. Difficulty-scaling: predict performance at scaled difficulty

        Anti-cheat E1: tasks must differ across consecutive calls.
        """
        self._update_count += 1
        new_tasks: List[Tuple[str, List[float], float]] = []

        # Extract environment features
        recent_rewards = env_state.get("recent_rewards", [0.3, 0.2, 0.4])
        mean_reward = sum(recent_rewards) / max(1, len(recent_rewards))
        task_count = float(env_state.get("task_count", 6))
        round_idx = float(env_state.get("round_idx", 0))
        stagnation = 1.0 if env_state.get("stagnation", False) else 0.0

        # Task 1: Predict mean reward from environment features
        features_1 = [mean_reward, task_count / 50.0, round_idx / 100.0, stagnation, float(self._update_count) / 20.0]
        expected_1 = mean_reward * (1.0 + 0.05 * min(round_idx, 50))
        new_tasks.append((f"predict_reward_r{int(round_idx)}", features_1, expected_1))

        # Task 2: Identify best action index from reward history
        if len(recent_rewards) >= 3:
            last_3 = recent_rewards[-3:]
        else:
            last_3 = recent_rewards + [0.0] * (3 - len(recent_rewards))
        features_2 = last_3 + [round_idx / 100.0, stagnation]
        best_idx = float(last_3.index(max(last_3)))
        new_tasks.append((f"action_select_r{int(round_idx)}", features_2, best_idx))

        # Task 3: Predict performance at scaled difficulty
        difficulty = float(env_state.get("difficulty", 3))
        features_3 = [mean_reward, difficulty / 10.0, task_count / 50.0, stagnation, round_idx / 100.0]
        expected_3 = max(0.0, mean_reward * (1.0 - difficulty / 20.0))
        new_tasks.append((f"difficulty_scale_r{int(round_idx)}", features_3, expected_3))

        # Task 4: Trend prediction (if enough history)
        if len(recent_rewards) >= 4:
            trend = recent_rewards[-1] - recent_rewards[-4]
            features_4 = recent_rewards[-4:] + [round_idx / 100.0]
            new_tasks.append((f"trend_predict_r{int(round_idx)}", features_4, trend))

        # BN-10 Fix 5: Use state_vector directly as VM input for env-state prediction
        state_vector = env_state.get("state_vector", [0.0] * 8)
        if len(state_vector) >= 8:
            expected_mean = mean_reward * (1.0 + state_vector[0] + state_vector[1] + state_vector[2])
            new_tasks.append((f"env_state_predict_r{int(round_idx)}", list(state_vector), expected_mean))

        # Anti-cheat E1: verify tasks differ from previous set
        new_hash = hash(str([(t[0], tuple(t[1]), t[2]) for t in new_tasks]))
        if new_hash != self._prev_task_hash:
            self._tasks = new_tasks
            self._prev_task_hash = new_hash
        else:
            # Force at least one task change by incorporating update_count
            features_extra = [float(self._update_count), mean_reward, stagnation, round_idx / 50.0, task_count / 25.0]
            new_tasks.append((f"forced_change_r{self._update_count}", features_extra, float(self._update_count % 5)))
            self._tasks = new_tasks
            self._prev_task_hash = hash(str([(t[0], tuple(t[1]), t[2]) for t in new_tasks]))

    def evaluate(self, genome: "ProgramGenome", vm: "VirtualMachine") -> float:
        """Evaluate genome against environment-coupled tasks.

        Anti-cheat E2: this runs AFTER VM execution, not during.
        Returns 0.0-1.0 based on task pass rate.
        """
        if not self._tasks:
            return 0.0

        passed = 0.0
        total = len(self._tasks)

        for name, inputs, expected in self._tasks:
            try:
                st = vm.execute(genome, inputs)
                result = st.regs[0]
                # Check result with tolerance
                if abs(result - expected) < 0.01:
                    passed += 1.0
                elif expected != 0 and abs(result - expected) < abs(expected) * 0.15:
                    passed += 0.5
                elif abs(result - expected) < 1.0:
                    passed += 0.25
            except Exception:
                pass

        return passed / max(1, total)

    @property
    def task_names(self) -> List[str]:
        """Return current task names (for capability labeling)."""
        return [t[0] for t in self._tasks]

    @property
    def task_count(self) -> int:
        return len(self._tasks)


# ==============================================================================
# 6) Detector + evidence writer
# ==============================================================================

@dataclass
class DetectorParams:
    # Target: 0.5-5% successes. Use a curriculum so the search has a gradient early,
    # then harden constraints to avoid "linear cheats".
    K_initial: int = 6               # strict CFG edit distance (post-warmup)
    L_initial: int = 10              # strict active subseq length (post-warmup)
    C_coverage: float = 0.55         # min coverage (post-warmup)
    f_rarity: float = 0.001          # rarity threshold (post-warmup)
    N_repro: int = 4                 # reproducibility trials

    require_both: bool = True        # strict mode requires CFG + subseq
    min_loops: int = 2               # STRICT: Require at least 2 loops (Multi-Stage)
    min_scc: int = 2                 # STRICT: Require at least 2 SCCs (Complex Topology)

    allow_cfg_variants: int = 2      # reproducibility CFG variants
    max_cov_span: float = 0.30       # reproducibility coverage stability
    max_loop_span: int = 5           # reproducibility loop stability

    # BN-10 Fix 3: Relaxed mode for RSI candidates
    rsi_relaxed: bool = False
    rsi_min_loops: int = 1
    rsi_min_scc: int = 1
    rsi_coverage: float = 0.35
    rsi_require_both: bool = False  # CFG OR subseq, not both

    # Warmup curriculum (first warmup_gens generations)
    warmup_gens: int = 100
    warmup_K: int = 3
    warmup_L: int = 8
    warmup_cov: float = 0.45
    warmup_require_both: bool = True # Strict warmup
    warmup_min_loops: int = 1        # Ban linear code even in warmup
    warmup_min_scc: int = 1          # Ban acyclic graphs even in warmup


class StrictStructuralDetector:
    def __init__(self, params: Optional[DetectorParams] = None) -> None:
        self.p = params or DetectorParams()
        self.parent_cfgs: Dict[str, ControlFlowGraph] = {}
        self.subseq_counts: Counter = Counter()
        self.subseq_total: int = 0
        self.seen_success_hashes: Set[str] = set()

    def _in_warmup(self, gen: int) -> bool:
        return gen <= self.p.warmup_gens

    def _K(self, gen: int) -> int:
        # Curriculum: easier early, strict later.
        if self._in_warmup(gen):
            return max(1, int(self.p.warmup_K))
        return max(3, int(self.p.K_initial))

    def _L(self, gen: int) -> int:
        if self._in_warmup(gen):
            return max(4, int(self.p.warmup_L))
        return max(6, int(self.p.L_initial))

    def _anti_cheat(self, st: ExecutionState, code_len: int, gen: int) -> Tuple[bool, str]:
        if st.error:
            return False, f"ERR:{st.error}"
        if not st.halted_cleanly:
            return False, "DIRTY_HALT"
        cov = st.coverage(code_len)
        # BN-10 Fix 3: Use relaxed thresholds for RSI mode
        if self.p.rsi_relaxed:
            min_cov = self.p.rsi_coverage
            min_loops = self.p.rsi_min_loops
        elif self._in_warmup(gen):
            min_cov = self.p.warmup_cov
            min_loops = self.p.warmup_min_loops
        else:
            min_cov = self.p.C_coverage
            min_loops = self.p.min_loops
        if cov < min_cov:
            return False, f"LOW_COVERAGE:{cov:.3f}"
        if st.loops_count < min_loops:
            return False, "NO_LOOPS"
        return True, f"ANTI_OK cov={cov:.3f} loops={st.loops_count}"

    def _repro(self, genome: ProgramGenome, vm: VirtualMachine) -> Tuple[bool, str]:
        cfgs: List[str] = []
        covs: List[float] = []
        loops: List[int] = []
        fixed_inputs = [
            [0.0]*8,
            [1.0]*8,
            [2.0]*8,
            [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0],
        ]
        for i in range(self.p.N_repro):
            inputs = fixed_inputs[i % len(fixed_inputs)]
            st = vm.execute(genome, inputs)
            cfgs.append(ControlFlowGraph.from_trace(st.trace, len(genome.instructions)).canonical_hash())
            covs.append(st.coverage(len(genome.instructions)))
            loops.append(st.loops_count)

        if len(set(cfgs)) > self.p.allow_cfg_variants:
            return False, "CFG_UNSTABLE"
        if max(covs) - min(covs) > self.p.max_cov_span:
            return False, "COV_UNSTABLE"
        if max(loops) - min(loops) > self.p.max_loop_span:
            return False, "LOOP_UNSTABLE"
        return True, f"REPRO_OK N={self.p.N_repro}"

    def evaluate(
        self,
        genome: ProgramGenome,
        parent: Optional[ProgramGenome],
        st: ExecutionState,
        vm: VirtualMachine,
        generation: int,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        reasons: List[str] = []
        diag: Dict[str, Any] = {}

        ok, msg = self._anti_cheat(st, len(genome.instructions), generation)
        if not ok:
            return False, [f"ANTI_FAIL:{msg}"], diag
        reasons.append(msg)

        cfg = ControlFlowGraph.from_trace(st.trace, len(genome.instructions))
        diag["cfg_hash"] = cfg.canonical_hash()
        # Track CFG for every genome so children can be compared against their parents (prevents deadlock)
        self.parent_cfgs[genome.gid] = cfg
        p_cfg = self.parent_cfgs.get(parent.gid) if parent else None
        if p_cfg is None and parent is not None:
            # Fallback: compute parent's CFG directly (robust even if parent was never a success)
            pst = vm.execute(parent, [1.0] * 8)
            p_cfg = ControlFlowGraph.from_trace(pst.trace, len(parent.instructions))

        cfg_ok = False
        cfg_msg = "CFG_NO_PARENT"
        if p_cfg is not None:
            dist = cfg.edit_distance_to(p_cfg)
            K = self._K(generation)
            cfg_ok = dist >= K
            cfg_msg = f"CFG dist={dist} K={K}"
            diag["cfg_dist"] = dist
        else:
            diag["cfg_dist"] = None

        scc_n = len(cfg.sccs())
        diag["scc_n"] = scc_n
        if self.p.rsi_relaxed:
            min_scc = self.p.rsi_min_scc
        elif self._in_warmup(generation):
            min_scc = self.p.warmup_min_scc
        else:
            min_scc = self.p.min_scc
        if scc_n < min_scc:
            cfg_ok = False
            cfg_msg = "CFG_NO_SCC"

        # subsequence novelty (only executed pcs, contiguous window in instruction index-space)
        L = self._L(generation)
        ops = genome.op_sequence()
        active: List[Tuple[str, ...]] = []
        visited = st.visited_pcs
        for i in range(0, max(0, len(ops) - L + 1)):
            window_pcs = set(range(i, i + L))
            if window_pcs.issubset(visited):
                active.append(tuple(ops[i : i + L]))

        subseq_ok = False
        subseq_msg = "SUBSEQ_NONE"
        if active:
            # rarity by empirical frequency in archive
            for seq in active:
                freq = (self.subseq_counts.get(seq, 0) / max(1, self.subseq_total))
                if freq < self.p.f_rarity:
                    subseq_ok = True
                    subseq_msg = f"SUBSEQ rarity={freq:.6f} L={L}"
                    # Defer archive updates until AFTER full success (CFG+SUBSEQ+REPRO+UNIQUENESS),
                    # otherwise near-misses rapidly poison rarity and can suppress discovery.
                    diag["_candidate_subseq"] = list(seq)
                    diag["subseq"] = list(seq)
                    diag["subseq_freq"] = freq
                    break
        diag["active_subseq_windows"] = len(active)

        # require both or at least one
        if self.p.rsi_relaxed:
            require_both = self.p.rsi_require_both
        elif self._in_warmup(generation):
            require_both = self.p.warmup_require_both
        else:
            require_both = self.p.require_both
        if require_both:
            if not (cfg_ok and subseq_ok and parent is not None):
                return False, [f"REQUIRE_BOTH_FAIL cfg={cfg_ok}({cfg_msg}) subseq={subseq_ok}({subseq_msg})"], diag
        else:
            if not (cfg_ok or subseq_ok):
                return False, [f"NO_STRUCT_CHANGE {cfg_msg}; {subseq_msg}"], diag

        reasons.append(cfg_msg if cfg_ok else cfg_msg)
        reasons.append(subseq_msg if subseq_ok else subseq_msg)

        # reproducibility
        r_ok, r_msg = self._repro(genome, vm)
        if not r_ok:
            return False, [f"REPRO_FAIL:{r_msg}"], diag
        reasons.append(r_msg)

        # global uniqueness on successes (prevents repeated printing of same "success")
        succ_hash = cfg.canonical_hash() + "|" + genome.code_hash()
        if succ_hash in self.seen_success_hashes:
            return False, ["DUP_SUCCESS_HASH"], diag

        self.seen_success_hashes.add(succ_hash)
        diag["success_hash"] = succ_hash

        # Commit subsequence rarity archive ONLY on confirmed success
        if "subseq" in diag:
            key = tuple(diag["subseq"])
            self.subseq_counts[key] = self.subseq_counts.get(key, 0) + 1
            self.subseq_total += 1
        elif "_candidate_subseq" in diag:
            key = tuple(diag["_candidate_subseq"])
            self.subseq_counts[key] = self.subseq_counts.get(key, 0) + 1
            self.subseq_total += 1

        # store cfg for parent tracking
        self.parent_cfgs[genome.gid] = cfg
        return True, reasons, diag
