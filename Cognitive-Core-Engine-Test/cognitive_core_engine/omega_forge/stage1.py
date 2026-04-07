"""Stage 1 genetic search engine for structural transition discovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import random as global_random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from cognitive_core_engine.omega_forge.instructions import (
    Instruction, ProgramGenome, ExecutionState, OPS, CONTROL_OPS, MEMORY_OPS,
)
from cognitive_core_engine.omega_forge.cfg import ControlFlowGraph
from cognitive_core_engine.omega_forge.vm import VirtualMachine, MacroLibrary
from cognitive_core_engine.omega_forge.concepts import (
    Concept, ConceptLibrary, set_op_bias, set_concept_bias, _sample_op,
    _inst_tuple_list, _concept_hash_from_insts, rand_inst,
)
from cognitive_core_engine.omega_forge.benchmark import TaskBenchmark, DetectorParams, StrictStructuralDetector
from cognitive_core_engine.omega_forge.evidence import EvidenceWriter, EngineConfig


AGG_MODE = "gmean"  # Options: "gmean", "min"
CURRICULUM_SWITCH_GEN = 250  # PATCH 4: Extended from 150 to 250
SUM_GATE_AFTER_SWITCH = 0.2  # PATCH 3: Penalty multiplier for SUM-failing genomes

# ==============================================================================
# PATCH 1: Diverse SUM Cases Generator
# ==============================================================================

def build_sum_cases(seed: int, n_cases: int) -> List[Tuple[List[float], float]]:
    """
    PATCH 1: Generate diverse SUM test cases deterministically.
    Uses local Random to avoid affecting global state.
    """
    rng = global_random.Random(seed)
    cases = set()
    
    # Include empty array
    cases.add(())
    
    # Generate diverse cases
    attempts = 0
    while len(cases) < n_cases and attempts < n_cases * 10:
        attempts += 1
        length = rng.randint(0, 16)
        if length == 0:
            arr = ()
        else:
            arr = tuple(rng.randint(0, 9) for _ in range(length))
        cases.add(arr)
    
    # Convert to required format: (inputs, expected_sum)
    result = []
    for arr in cases:
        inputs = [float(x) for x in arr]
        expected = sum(inputs)
        result.append((inputs, expected))
    
    # Sort for reproducibility
    result.sort(key=lambda x: (len(x[0]), x[1]))
    return result[:n_cases]

# ==============================================================================
# HALF-SKELETON MACROS (unchanged)
# ==============================================================================

class TaskMacroLibrary:
    @staticmethod
    def sum_skeleton() -> List[Instruction]:
        return [
            Instruction("SET", 0, 0, 0),      # r0 = 0 (accumulator)
            Instruction("SET", 0, 0, 2),      # r2 = 0 (index i)
            Instruction("JLT", 2, 1, 2),      # if r2 < r1, continue
            Instruction("JMP", 5, 0, 0),      # else exit
            Instruction("LOAD", 2, 0, 3),     # r3 = memory[r2]
            Instruction("ADD", 0, 3, 0),      # r0 += r3
            Instruction("INC", 0, 0, 2),      # i++
            Instruction("JMP", -5, 0, 0),     # loop back
        ]
    
    @staticmethod
    def max_skeleton() -> List[Instruction]:
        return [
            Instruction("LOAD", 2, 0, 0),
            Instruction("SET", 1, 0, 2),
            Instruction("JLT", 2, 1, 2),
            Instruction("JMP", 6, 0, 0),
            Instruction("LOAD", 2, 0, 3),
            Instruction("JGT", 3, 0, 2),
            Instruction("JMP", 2, 0, 0),
            Instruction("MOV", 3, 0, 0),
            Instruction("INC", 0, 0, 2),
            Instruction("JMP", -7, 0, 0),
        ]
    
    @staticmethod
    def double_skeleton() -> List[Instruction]:
        return [
            Instruction("SET", 0, 0, 2),
            Instruction("JLT", 2, 1, 2),
            Instruction("JMP", 6, 0, 0),
            Instruction("LOAD", 2, 0, 3),
            Instruction("ADD", 3, 3, 3),
            Instruction("STORE", 2, 0, 3),
            Instruction("INC", 0, 0, 2),
            Instruction("JMP", -6, 0, 0),
        ]

# ==============================================================================
# TASK BENCHMARK V4 (All Patches Applied)
# ==============================================================================

class TaskBenchmarkV4:
    """
    Patches implemented:
    1. Diverse SUM cases (24 cases from deterministic generator)
    2. Full-sum dominant scoring
    3. Strict-pass for per-genome counting
    """
    
    # PATCH 1: Generate 24 diverse SUM cases
    SUM_CASES = build_sum_cases(seed=123, n_cases=24)
    
    # MAX and DOUBLE unchanged
    MAX_CASES = [
        ([3.0, 7.0, 2.0, 9.0, 1.0], 9.0),
        ([5.0, 2.0, 8.0], 8.0),
        ([1.0], 1.0),
        ([10.0, 5.0, 7.0, 3.0, 9.0, 2.0], 10.0),
    ]
    
    DOUBLE_CASES = [
        ([3.0, 4.0, 5.0], 6.0),
        ([2.0, 6.0], 4.0),
        ([5.0], 10.0),
    ]
    
    @staticmethod
    def _sum_score(genome, vm, inputs: List[float], expected: float) -> float:
        """
        PATCH 2: Full-sum dominant scoring.
        Prefix bonus is capped at 0.10 as tie-breaker only.
        """
        try:
            st = vm.execute(genome, inputs)
            if st.error or not st.halted_cleanly:
                return 0.0
            result = st.regs[0]
        except:
            return 0.0
        
        # Base score: full-sum error ratio (dominant)
        err = abs(result - expected)
        den = max(1.0, abs(expected))
        ratio = err / den
        
        if ratio < 1e-6:
            base = 1.0
        elif ratio < 0.02:
            base = 0.8
        elif ratio < 0.10:
            base = 0.5
        elif ratio < 0.30:
            base = 0.2
        else:
            base = 0.0
        
        # Prefix bonus: small tie-breaker (capped at 0.10)
        bonus = 0.0
        if len(inputs) > 0:
            cumsum = 0.0
            for i, val in enumerate(inputs):
                cumsum += val
                if abs(result - cumsum) < 1e-6:
                    bonus = max(bonus, 0.05 + 0.05 * (i + 1) / max(1, len(inputs)))
        
        return min(1.0, base + min(0.10, bonus))
    
    @staticmethod
    def _case_score(genome, vm, inputs: List[float], expected: float, out_loc: str) -> float:
        """Standard partial scoring for MAX/DOUBLE."""
        try:
            st = vm.execute(genome, inputs)
            if st.error or not st.halted_cleanly:
                return 0.0
            if out_loc == "reg0":
                result = st.regs[0]
            elif out_loc == "mem0":
                result = st.memory.get(0, 0.0)
            else:
                result = 0.0
        except:
            return 0.0
        
        if abs(expected) < 1e-9:
            return 1.0 if abs(result) < 0.01 else 0.0
        
        error_ratio = abs(result - expected) / abs(expected)
        if error_ratio < 0.001:
            return 1.0
        elif error_ratio < 0.1:
            return 0.8
        elif error_ratio < 0.5:
            return 0.5
        elif error_ratio < 1.0:
            return 0.2
        return 0.0
    
    @staticmethod
    def evaluate(genome, vm) -> Dict[str, float]:
        """Returns per-task-type average scores."""
        scores = {"SUM": 0.0, "MAX": 0.0, "DOUBLE": 0.0}
        
        # SUM
        sum_scores = []
        for inputs, expected in TaskBenchmarkV4.SUM_CASES:
            s = TaskBenchmarkV4._sum_score(genome, vm, inputs, expected)
            sum_scores.append(s)
        scores["SUM"] = sum(sum_scores) / len(sum_scores) if sum_scores else 0.0
        
        # MAX
        max_scores = []
        for inputs, expected in TaskBenchmarkV4.MAX_CASES:
            s = TaskBenchmarkV4._case_score(genome, vm, inputs, expected, "reg0")
            max_scores.append(s)
        scores["MAX"] = sum(max_scores) / len(max_scores) if max_scores else 0.0
        
        # DOUBLE
        dbl_scores = []
        for inputs, expected in TaskBenchmarkV4.DOUBLE_CASES:
            s = TaskBenchmarkV4._case_score(genome, vm, inputs, expected, "mem0")
            dbl_scores.append(s)
        scores["DOUBLE"] = sum(dbl_scores) / len(dbl_scores) if dbl_scores else 0.0
        
        return scores
    
    @staticmethod
    def evaluate_strict_pass(genome, vm) -> Dict[str, bool]:
        """
        PATCH 5: Returns per-task-type strict-pass (ALL cases must pass exactly).
        """
        results = {}
        
        # SUM: all cases must pass
        all_pass = True
        for inputs, expected in TaskBenchmarkV4.SUM_CASES:
            try:
                st = vm.execute(genome, inputs)
                if st.error or not st.halted_cleanly:
                    all_pass = False
                    break
                if abs(st.regs[0] - expected) > 0.01:
                    all_pass = False
                    break
            except:
                all_pass = False
                break
        results["SUM"] = all_pass
        
        # MAX
        all_pass = True
        for inputs, expected in TaskBenchmarkV4.MAX_CASES:
            try:
                st = vm.execute(genome, inputs)
                if st.error or not st.halted_cleanly:
                    all_pass = False
                    break
                if abs(st.regs[0] - expected) > 0.01:
                    all_pass = False
                    break
            except:
                all_pass = False
                break
        results["MAX"] = all_pass
        
        # DOUBLE
        all_pass = True
        for inputs, expected in TaskBenchmarkV4.DOUBLE_CASES:
            try:
                st = vm.execute(genome, inputs)
                if st.error or not st.halted_cleanly:
                    all_pass = False
                    break
                if abs(st.memory.get(0, 0.0) - expected) > 0.01:
                    all_pass = False
                    break
            except:
                all_pass = False
                break
        results["DOUBLE"] = all_pass
        
        return results
    
    @staticmethod
    def debug_sum_outputs(genome, vm, label: str):
        """
        PATCH 6: Debug output for first 3 SUM cases.
        """
        print(f"    {label}:")
        for i, (inputs, expected) in enumerate(TaskBenchmarkV4.SUM_CASES[:3]):
            try:
                st = vm.execute(genome, inputs)
                got = st.regs[0] if st.halted_cleanly else "ERROR"
            except:
                got = "EXCEPTION"
            print(f"      case {i}: input={inputs[:5]}{'...' if len(inputs)>5 else ''} expected={expected} got={got}")

# ==============================================================================
# 5.6) Concept discovery micro-domains
# ==============================================================================

class ConceptDiscoveryBenchmark:
    DOMAINS: List[Dict[str, Any]] = [
        {
            "name": "COPY_FIRST",
            "out_loc": "reg0",
            "train": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0, 7.0]],
            "holdout": [[9.0, 1.0], [2.0, 8.0, 3.0]],
            "adversarial": [[0.0], [-1.0, 2.0]],
            "shift": [[10.0, 11.0, 12.0, 13.0]],
            "oracle": lambda xs: float(xs[0]) if xs else 0.0,
        },
        {
            "name": "PREFIX_SUM_2",
            "out_loc": "reg0",
            "train": [[1.0, 2.0], [2.0, 2.0], [3.0, 1.0]],
            "holdout": [[4.0, 5.0], [0.0, 3.0]],
            "adversarial": [[-1.0, 4.0], [7.0, -2.0]],
            "shift": [[10.0, 20.0]],
            "oracle": lambda xs: float(xs[0] + xs[1]) if len(xs) >= 2 else float(sum(xs)),
        },
        {
            "name": "ARGMAX_TIE_FIRST",
            "out_loc": "reg0",
            "train": [[1.0, 3.0, 2.0], [5.0, 5.0, 4.0], [2.0, 1.0, 2.0]],
            "holdout": [[0.0, 0.0, 1.0], [9.0, 7.0, 9.0]],
            "adversarial": [[-1.0, -1.0, -2.0]],
            "shift": [[100.0, 50.0, 75.0, 100.0]],
            "oracle": lambda xs: float(next((i for i, v in enumerate(xs) if v == max(xs)), 0)) if xs else 0.0,
        },
    ]

    @staticmethod
    def _eval_case(genome: ProgramGenome, vm: VirtualMachine, inputs: List[float], out_loc: str, expected: float) -> bool:
        st = vm.execute(genome, inputs)
        if st.error or not st.halted_cleanly:
            return False
        if out_loc == "reg0":
            got = st.regs[0]
        elif out_loc == "mem0":
            got = st.memory.get(0, 0.0)
        else:
            got = 0.0
        return abs(got - expected) < 0.01

    @staticmethod
    def evaluate(genome: ProgramGenome, vm: VirtualMachine) -> Dict[str, Any]:
        splits = {"train": [], "holdout": [], "adversarial": [], "shift": []}
        for domain in ConceptDiscoveryBenchmark.DOMAINS:
            oracle = domain["oracle"]
            out_loc = domain["out_loc"]
            for split_name in ("train", "holdout", "adversarial", "shift"):
                dataset = domain.get(split_name, [])
                passes = 0
                for inputs in dataset:
                    expected = oracle(inputs)
                    if ConceptDiscoveryBenchmark._eval_case(genome, vm, inputs, out_loc, expected):
                        passes += 1
                total = max(1, len(dataset))
                splits[split_name].append(passes / total)

        train_rate = float(sum(splits["train"]) / max(1, len(splits["train"])))
        holdout_rate = float(sum(splits["holdout"]) / max(1, len(splits["holdout"])))
        adv_rate = float(sum(splits["adversarial"]) / max(1, len(splits["adversarial"])))
        shift_rate = float(sum(splits["shift"]) / max(1, len(splits["shift"])))

        train_count = sum(len(d.get("train", [])) for d in ConceptDiscoveryBenchmark.DOMAINS)
        holdout_count = sum(len(d.get("holdout", [])) for d in ConceptDiscoveryBenchmark.DOMAINS)
        holdout_cost = min(4.0, float(holdout_count) / max(1.0, float(train_count)))

        return {
            "train_pass_rate": train_rate,
            "holdout_pass_rate": holdout_rate,
            "adversarial_pass_rate": adv_rate,
            "distribution_shift": {"holdout_pass_rate": shift_rate},
            "discovery_cost": {"train": float(train_count), "holdout": holdout_cost},
        }

def detect_memorization(genome: ProgramGenome) -> bool:
    large_set = sum(1 for inst in genome.instructions if inst.op == "SET" and abs(inst.a) > 20)
    set_total = sum(1 for inst in genome.instructions if inst.op == "SET")
    if large_set >= 3:
        return True
    if set_total >= max(6, len(genome.instructions) // 2):
        return True
    return False

def find_repeated_subsequence(instructions: List[Instruction],
                              min_len: int = 2,
                              max_len: int = 5) -> Optional[List[Instruction]]:
    if len(instructions) < min_len * 2:
        return None
    best_seq = None
    best_count = 1
    tuples = _inst_tuple_list(instructions)
    for L in range(min_len, min(max_len, len(instructions)) + 1):
        counts: Dict[Tuple[Any, ...], int] = {}
        for i in range(0, len(tuples) - L + 1):
            key = tuple(tuples[i:i + L])
            counts[key] = counts.get(key, 0) + 1
        for key, count in counts.items():
            if count > best_count:
                best_count = count
                best_seq = [Instruction(*t) for t in key]
    if best_count > 1 and best_seq:
        return best_seq
    return None

def detect_concepts_in_genome(genome: ProgramGenome, concepts: ConceptLibrary) -> List[str]:
    if not concepts or not genome.instructions:
        return []
    hits = []
    inst_tuples = _inst_tuple_list(genome.instructions)
    for concept in concepts.all_concepts():
        insts = concepts.compile(concept)
        if not insts:
            continue
        c_tuples = tuple(_inst_tuple_list(insts))
        if len(c_tuples) == 0:
            continue
        for i in range(0, len(inst_tuples) - len(c_tuples) + 1):
            if tuple(inst_tuples[i:i + len(c_tuples)]) == c_tuples:
                hits.append(concept.cid)
                break
    return hits



# ==============================================================================
# Stage 1: Structural Discovery (unchanged)
# ==============================================================================

class Stage1Engine:
    def __init__(self,
                 seed: int = 42,
                 concepts_on: bool = False,
                 concept_budget: int = 80,
                 concept_library_path: str = ""):
        global_random.seed(seed)
        self.vm = VirtualMachine()
        self.detector = StrictStructuralDetector()
        self.cfg = EngineConfig(pop_size=30)
        self.population: List[ProgramGenome] = []
        self.generation: int = 0
        self.candidates: List[Dict[str, Any]] = []
        self.concepts_on = concepts_on
        self.concept_library = ConceptLibrary(max_size=concept_budget)
        self.concept_library_path = concept_library_path
        if self.concepts_on and concept_library_path:
            self.concept_library.load(concept_library_path)
        
    def init_population(self):
        self.population = []
        for i in range(self.cfg.pop_size):
            L = global_random.randint(18, 28)
            insts = [rand_inst() for _ in range(L)]
            g = ProgramGenome(gid=f"init_{i}", instructions=insts, parents=[], generation=0)
            self.population.append(g)
        self.parents_index = {g.gid: g for g in self.population}
    
    def mutate(self, parent: ProgramGenome) -> ProgramGenome:
        child = parent.clone()
        child.generation = self.generation
        child.parents = [parent.gid]
        child.gid = f"g{self.generation}_{global_random.randint(0, 999999)}"

        roll = global_random.random()
        if self.concepts_on and roll < 0.12:
            concept = self._sample_concept()
            if concept:
                insts = self.concept_library.compile(concept)
                if insts and len(child.instructions) + len(insts) < self.cfg.max_code_len:
                    pos = global_random.randint(0, len(child.instructions))
                    child.instructions[pos:pos] = [i.clone() for i in insts]
                    child.concept_trace.append(concept.cid)
                    return child
        if self.concepts_on and roll < 0.22:
            seq = find_repeated_subsequence(parent.instructions, min_len=2, max_len=5)
            if seq:
                cid = f"c{self.generation}_{global_random.randint(0, 999999)}"
                payload = {"instructions": [i.to_tuple() for i in seq]}
                concept = Concept(
                    cid=cid,
                    name=f"macro_len{len(seq)}",
                    kind="macro",
                    payload=payload,
                    compile_fn_id="macro_v1",
                    discovered_gen=self.generation,
                    parents=[parent.gid],
                )
                added_cid = self.concept_library.add_concept(concept, dedup=True)
                if added_cid:
                    child.concept_proposals.append(added_cid)
            return child
        if self.concepts_on and roll < 0.28:
            c1 = self._sample_concept()
            c2 = self._sample_concept()
            if c1 and c2 and c1.cid != c2.cid:
                insts = self.concept_library.compile(c1) + self.concept_library.compile(c2)
                if 0 < len(insts) <= 6:
                    cid = f"c{self.generation}_{global_random.randint(0, 999999)}"
                    payload = {"instructions": [i.to_tuple() for i in insts]}
                    concept = Concept(
                        cid=cid,
                        name=f"compose_{c1.cid}_{c2.cid}",
                        kind="macro",
                        payload=payload,
                        compile_fn_id="macro_v1",
                        discovered_gen=self.generation,
                        parents=[c1.cid, c2.cid],
                    )
                    added_cid = self.concept_library.add_concept(concept, dedup=True)
                    if added_cid:
                        child.concept_proposals.append(added_cid)
            return child
        if roll < 0.15 and len(child.instructions) + 10 < self.cfg.max_code_len:
            skeleton = global_random.choice([
                TaskMacroLibrary.sum_skeleton,
                TaskMacroLibrary.max_skeleton,
                TaskMacroLibrary.double_skeleton,
            ])()
            pos = global_random.randint(0, len(child.instructions))
            child.instructions[pos:pos] = [Instruction(i.op, i.a, i.b, i.c) for i in skeleton]
        elif roll < 0.35 and child.instructions:
            pos = global_random.randint(0, len(child.instructions) - 1)
            op = global_random.choice(["JMP", "JZ", "JNZ", "JGT", "JLT", "CALL", "RET"])
            child.instructions[pos] = Instruction(op, global_random.randint(-8, 8), global_random.randint(0, 7), global_random.randint(0, 7))
        elif roll < 0.60 and len(child.instructions) < self.cfg.max_code_len:
            pos = global_random.randint(0, len(child.instructions))
            child.instructions.insert(pos, rand_inst())
        elif roll < 0.80 and child.instructions:
            pos = global_random.randint(0, len(child.instructions) - 1)
            child.instructions[pos].a = max(-8, min(31, child.instructions[pos].a + global_random.randint(-2, 2)))
        else:
            if len(child.instructions) > 8:
                pos = global_random.randint(0, len(child.instructions) - 1)
                child.instructions.pop(pos)

        return child

    def _sample_concept(self) -> Optional[Concept]:
        concepts = self.concept_library.all_concepts()
        if not concepts:
            return None
        weights = []
        for c in concepts:
            if c.cid in CONCEPT_ANTI_BIAS:
                base = 0.1
            else:
                base = CONCEPT_BIAS.get(c.cid, 1.0)
            length = int(c.stats.get("length", len(self.concept_library.compile(c)) or 1))
            if MACRO_LENGTH_BIAS > 0.0:
                base *= 1.0 / (1.0 + MACRO_LENGTH_BIAS * max(0, length - 1))
            weights.append(max(0.05, base))
        return global_random.choices(concepts, weights=weights, k=1)[0]
    
    def step(self) -> int:
        self.generation += 1
        successes = 0
        
        for g in self.population:
            parent = self.parents_index.get(g.parents[0]) if g.parents else None
            st = self.vm.execute(g, [1.0] * 8)
            passed, reasons, diag = self.detector.evaluate(g, parent, st, self.vm, self.generation)
            
            if passed:
                successes += 1
                if detect_memorization(g):
                    continue
                scores = TaskBenchmarkV4.evaluate(g, self.vm)
                concept_metrics = ConceptDiscoveryBenchmark.evaluate(g, self.vm)
                hints: List[str] = []
                for cid in g.concept_trace:
                    c = self.concept_library.get(cid)
                    if c:
                        hints.append(f"use:{cid}:{c.name}")
                    else:
                        hints.append(f"use:{cid}")
                for cid in g.concept_proposals:
                    c = self.concept_library.get(cid)
                    if c:
                        hints.append(f"propose:{cid}:{c.name}")
                    else:
                        hints.append(f"propose:{cid}")
                candidate = {
                    "gid": g.gid,
                    "generation": self.generation,
                    "code": [(i.op, i.a, i.b, i.c) for i in g.instructions],
                    "metrics": {
                        **concept_metrics,
                        "structural": {"loops": st.loops_count, "scc_n": diag.get("scc_n", 0)},
                        "memorization_suspected": False,
                    },
                    "task_scores": scores,
                    "hints": hints,
                }
                self.candidates.append(candidate)
        
        # Selection
        for g in self.population:
            st2 = self.vm.execute(g, [1.0] * 8)
            cfg2 = ControlFlowGraph.from_trace(st2.trace, len(g.instructions))
            cov = st2.coverage(len(g.instructions))
            scc_n = len(cfg2.sccs())
            score = cov + 0.02 * min(st2.loops_count, 50) + 0.08 * min(scc_n, 6)
            if st2.error or not st2.halted_cleanly:
                score -= 0.5
            g.last_score = score
            g.last_cfg_hash = cfg2.canonical_hash()
        
        ranked = sorted(self.population, key=lambda x: x.last_score, reverse=True)
        elites = ranked[:self.cfg.elite_keep]
        
        next_pop = []
        for e in elites:
            next_pop.append(e.clone())
            for _ in range(self.cfg.children_per_elite):
                next_pop.append(self.mutate(e))
        
        self.population = next_pop[:self.cfg.pop_size]
        self.parents_index = {g.gid: g for g in self.population}
        
        return successes
    
    def run(self, generations: int, out_file: str):
        self.init_population()
        print(f"[Stage 1] Collecting candidates for {generations} generations...")
        
        for gen in range(1, generations + 1):
            self.step()
            if gen % 50 == 0:
                print(f"  [gen {gen}] candidates={len(self.candidates)}")
        
        with open(out_file, 'w') as f:
            for c in self.candidates:
                f.write(json.dumps(c) + "\n")

        if self.concepts_on and self.concept_library_path:
            self.concept_library.save(self.concept_library_path)
        
        print(f"[Stage 1] Done. Saved {len(self.candidates)} candidates to {out_file}")
        return self.candidates

# ==============================================================================
# Stage 2: Task-Aware Evolution (PATCHES 3, 4, 5, 6)
# ==============================================================================

