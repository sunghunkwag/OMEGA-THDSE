"""Stage 2 feedback refinement engine."""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from cognitive_core_engine.omega_forge.instructions import (
    Instruction, ProgramGenome, OPS,
)
from cognitive_core_engine.omega_forge.cfg import ControlFlowGraph
from cognitive_core_engine.omega_forge.vm import VirtualMachine
from cognitive_core_engine.omega_forge.concepts import (
    Concept, ConceptLibrary, _inst_tuple_list, _concept_hash_from_insts,
    set_op_bias, set_concept_bias, rand_inst,
)
from cognitive_core_engine.omega_forge.benchmark import TaskBenchmark, StrictStructuralDetector, DetectorParams
from cognitive_core_engine.omega_forge.evidence import EvidenceWriter, EngineConfig


class Stage2Engine:
    def __init__(self, candidates: List[Dict[str, Any]], seed: int = 42):
        global_random.seed(seed)
        self.vm = VirtualMachine()
        self.candidates = candidates
        self.population: List[ProgramGenome] = []
        self.generation: int = 0
        
    def load_population(self, sample_size: int = 50):
        sorted_cands = sorted(
            self.candidates, 
            key=lambda x: x.get("task_scores", {}).get("SUM", 0), 
            reverse=True
        )
        
        self.population = []
        for i, c in enumerate(sorted_cands[:sample_size]):
            insts = [Instruction(op, a, b, c_) for op, a, b, c_ in c["code"]]
            g = ProgramGenome(gid=f"s2_init_{i}", instructions=insts, generation=0)
            self.population.append(g)
        
        print(f"[Stage 2] Loaded {len(self.population)} genomes (sorted by SUM potential)")
    
    def mutate(self, parent: ProgramGenome) -> ProgramGenome:
        child = parent.clone()
        child.generation = self.generation
        child.parents = [parent.gid]
        child.gid = f"s2_g{self.generation}_{global_random.randint(0, 999999)}"
        
        roll = global_random.random()
        if roll < 0.4 and child.instructions:
            pos = global_random.randint(0, len(child.instructions) - 1)
            inst = child.instructions[pos]
            field = global_random.choice(["a", "b", "c"])
            delta = global_random.randint(-2, 2)
            if field == "a":
                inst.a = max(-8, min(31, inst.a + delta))
            elif field == "b":
                inst.b = max(0, min(7, inst.b + delta))
            else:
                inst.c = max(0, min(7, inst.c + delta))
        elif roll < 0.6 and child.instructions:
            pos = global_random.randint(0, len(child.instructions) - 1)
            useful_ops = ["LOAD", "ADD", "STORE", "JGT", "JLT", "MOV", "INC"]
            child.instructions[pos] = Instruction(
                global_random.choice(useful_ops),
                global_random.randint(0, 7),
                global_random.randint(0, 7),
                global_random.randint(0, 7)
            )
        elif roll < 0.8 and len(child.instructions) >= 2:
            i, j = global_random.sample(range(len(child.instructions)), 2)
            child.instructions[i], child.instructions[j] = child.instructions[j], child.instructions[i]
        else:
            if len(child.instructions) < 60:
                pos = global_random.randint(0, len(child.instructions))
                useful_ops = ["LOAD", "ADD", "STORE", "INC"]
                child.instructions.insert(pos, Instruction(
                    global_random.choice(useful_ops),
                    global_random.randint(0, 7),
                    global_random.randint(0, 7),
                    global_random.randint(0, 7)
                ))
        
        return child
    
    def _compute_fitness(self, scores: Dict[str, float], strict_pass: Dict[str, bool], gen: int) -> float:
        """
        PATCH 3 & 4: Curriculum + SUM strict-pass gate
        """
        sum_s = scores.get("SUM", 0.0)
        max_s = scores.get("MAX", 0.0)
        dbl_s = scores.get("DOUBLE", 0.0)
        
        # Before curriculum switch: SUM-only
        if gen < CURRICULUM_SWITCH_GEN:
            return sum_s
        
        # After switch: gmean aggregation
        eps = 1e-9
        if AGG_MODE == "gmean":
            fitness = (max(sum_s, eps) * max(max_s, eps) * max(dbl_s, eps)) ** (1.0/3.0)
        elif AGG_MODE == "min":
            fitness = min(sum_s, max_s, dbl_s)
        else:
            fitness = (sum_s + max_s + dbl_s) / 3.0
        
        # PATCH 3: SUM gate multiplier
        if not strict_pass.get("SUM", False):
            fitness *= SUM_GATE_AFTER_SWITCH
        
        return fitness
    
    def step(self) -> Dict[str, Any]:
        self.generation += 1
        
        # Log curriculum switch
        if self.generation == CURRICULUM_SWITCH_GEN:
            print(f"\n  *** CURRICULUM SWITCH at gen {self.generation}: SUM-only → {AGG_MODE} + SUM gate ({SUM_GATE_AFTER_SWITCH}x) ***\n")
        
        scores_list = []
        pass_list = []
        for g in self.population:
            scores = TaskBenchmarkV4.evaluate(g, self.vm)
            strict_pass = TaskBenchmarkV4.evaluate_strict_pass(g, self.vm)
            fitness = self._compute_fitness(scores, strict_pass, self.generation)
            g.last_score = fitness
            scores_list.append(scores)
            pass_list.append(strict_pass)
        
        # PATCH 6: Debug at gen 1
        if self.generation == 1:
            print("  [gen 1] DEBUG: Top 3 genomes by SUM score:")
            ranked_by_sum = sorted(zip(self.population, scores_list), key=lambda x: x[1]["SUM"], reverse=True)
            for i, (g, sc) in enumerate(ranked_by_sum[:3]):
                print(f"    Genome {i} (SUM={sc['SUM']:.3f}):")
                TaskBenchmarkV4.debug_sum_outputs(g, self.vm, f"outputs")
        
        avg_sum = sum(s["SUM"] for s in scores_list) / len(scores_list)
        avg_max = sum(s["MAX"] for s in scores_list) / len(scores_list)
        avg_dbl = sum(s["DOUBLE"] for s in scores_list) / len(scores_list)
        sum_pass = sum(1 for p in pass_list if p["SUM"]) / len(pass_list)
        
        ranked = sorted(self.population, key=lambda x: x.last_score, reverse=True)
        elite_count = max(10, len(self.population) // 3)
        elites = ranked[:elite_count]
        
        next_pop = []
        for e in elites:
            next_pop.append(e.clone())
            for _ in range(2):
                next_pop.append(self.mutate(e))
        
        self.population = next_pop[:50]
        
        return {"avg_sum": avg_sum, "avg_max": avg_max, "avg_dbl": avg_dbl, "sum_pass": sum_pass}
    
    def run(self, generations: int):
        print(f"[Stage 2] Task evolution for {generations} generations")
        print(f"  Curriculum: SUM-only until gen {CURRICULUM_SWITCH_GEN}, then {AGG_MODE} + SUM gate")
        print(f"  SUM cases: {len(TaskBenchmarkV4.SUM_CASES)} diverse cases")
        
        for gen in range(1, generations + 1):
            stats = self.step()
            if gen % 50 == 0:
                print(f"  [gen {gen}] SUM={stats['avg_sum']:.3f} (pass:{stats['sum_pass']*100:.1f}%) MAX={stats['avg_max']:.3f} DOUBLE={stats['avg_dbl']:.3f}")
        
        # PATCH 5: Final Benchmark with strict-pass
        print("\n[Stage 2] Final Benchmark (per-genome strict-pass):")
        results = {"SUM": 0, "MAX": 0, "DOUBLE": 0}
        
        for g in self.population:
            passed = TaskBenchmarkV4.evaluate_strict_pass(g, self.vm)
            for task_type, p in passed.items():
                if p:
                    results[task_type] += 1
        
        n = len(self.population)
        for task, count in results.items():
            pct = count / n * 100
            status = "✅" if count > 0 else "❌"
            print(f"  {status} {task}: {count}/{n} ({pct:.1f}%)")
        
        return results

# ==============================================================================
# CLI
# ==============================================================================

# ==============================================================================
# FEEDBACK: Stage 2 -> Stage 1 (Two-Stage + Feedback Bias)
# ==============================================================================
def extract_stage2_feedback(population: List[ProgramGenome],
                            vm: VirtualMachine,
                            n_top: int = 20,
                            require_sum_pass: bool = True,
                            concept_library: Optional[ConceptLibrary] = None) -> Dict[str, Any]:
    """
    Compute simple sampling biases from the best Stage2 genomes.
    Biases are intended to steer Stage1's rand_inst() opcode sampling.

    Returns dict:
      {
        "op_bias": {"LOAD":1.3, ...},
        "concept_bias": {"c1":1.2, ...},
        "macro_length_bias": 0.2,
        "concept_anti_bias": [...],
        "meta": {"n_used":..., "n_top":..., "require_sum_pass":...}
      }
    """
    scored: List[Tuple[float, ProgramGenome, Dict[str, bool]]] = []
    for g in population:
        scores = TaskBenchmarkV4.evaluate(g, vm)
        strict_pass = TaskBenchmarkV4.evaluate_per_genome_pass(g, vm)
        if require_sum_pass and not strict_pass.get("SUM", False):
            continue
        s = float(scores.get("SUM", 0.0))
        # prefer multi-task competence if available
        s = (max(1e-9, s) * max(1e-9, float(scores.get("MAX", 0.0))) * max(1e-9, float(scores.get("DOUBLE", 0.0)))) ** (1.0/3.0)
        scored.append((s, g, strict_pass))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [g for _, g, _ in scored[:max(1, n_top)]]
    if not picked:
        # fall back: use top by SUM score, even if SUM strict-pass is absent
        tmp = []
        for g in population:
            scores = TaskBenchmarkV4.evaluate(g, vm)
            tmp.append((float(scores.get("SUM", 0.0)), g))
        tmp.sort(key=lambda x: x[0], reverse=True)
        picked = [g for _, g in tmp[:max(1, n_top)]]

    op_counts: Dict[str, int] = {op: 0 for op in OPS}
    total = 0
    for g in picked:
        for inst in g.instructions:
            if inst.op in op_counts:
                op_counts[inst.op] += 1
                total += 1

    # Convert counts -> weights with smoothing, emphasize above-average ops
    op_bias: Dict[str, float] = {}
    if total > 0:
        avg = total / max(1, len(OPS))
        for op, c in op_counts.items():
            # weight = 1.0 at avg, >1 if above avg, with mild exponent
            w = ( (c + 1.0) / (avg + 1.0) ) ** 0.7
            op_bias[op] = float(max(0.05, min(5.0, w)))

    concept_bias: Dict[str, float] = {}
    concept_anti_bias: List[str] = []
    macro_length_bias = 0.0
    if concept_library:
        concept_scores: Counter = Counter()
        concept_counts: Counter = Counter()
        concept_lengths: List[int] = []
        for g in picked:
            metrics = ConceptDiscoveryBenchmark.evaluate(g, vm)
            holdout = metrics.get("holdout_pass_rate", 0.0)
            train = metrics.get("train_pass_rate", 0.0)
            gap = max(0.0, train - holdout)
            used = detect_concepts_in_genome(g, concept_library)
            for cid in used:
                concept_counts[cid] += 1
                concept_scores[cid] += holdout
                if gap > 0.25:
                    concept_anti_bias.append(cid)
            for cid in used:
                c = concept_library.get(cid)
                if c:
                    concept_lengths.append(int(c.stats.get("length", 1)))
        for cid, cnt in concept_counts.items():
            score = concept_scores.get(cid, 0.0) / max(1, cnt)
            concept_bias[cid] = float(max(0.05, min(5.0, 0.5 + score)))
        if concept_lengths:
            avg_len = sum(concept_lengths) / max(1, len(concept_lengths))
            macro_length_bias = max(0.0, min(1.5, (avg_len - 1.0) / 4.0))

    return {
        "op_bias": op_bias,
        "concept_bias": concept_bias,
        "macro_length_bias": macro_length_bias,
        "concept_anti_bias": list(sorted(set(concept_anti_bias))),
        "meta": {
            "n_used": len(picked),
            "n_top": n_top,
            "require_sum_pass": bool(require_sum_pass),
        }
    }

def save_feedback_json(feedback: Dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(feedback, indent=2, sort_keys=True), encoding="utf-8")

def load_feedback_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def apply_feedback_to_stage1(feedback: Dict[str, Any]) -> None:
    """
    Applies feedback biases to Stage1 by calling set_op_bias().
    """
    op_bias = (feedback or {}).get("op_bias", {}) if isinstance(feedback, dict) else {}
    set_op_bias(op_bias)
    concept_bias = (feedback or {}).get("concept_bias", {}) if isinstance(feedback, dict) else {}
    anti_bias = (feedback or {}).get("concept_anti_bias", []) if isinstance(feedback, dict) else []
    macro_length_bias = (feedback or {}).get("macro_length_bias", None) if isinstance(feedback, dict) else None
    set_concept_bias(concept_bias, anti_bias=anti_bias, macro_length_bias=macro_length_bias)

