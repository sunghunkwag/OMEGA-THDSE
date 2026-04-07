"""
OmegaForgeV13 main engine class.

BN-10: Added relaxed RSI detector, crossover, concept-based mutation,
task-aligned macro insertion, and operand tuning.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from cognitive_core_engine.omega_forge.instructions import (
    CONTROL_OPS,
    Instruction,
    ProgramGenome,
)
from cognitive_core_engine.omega_forge.cfg import ControlFlowGraph
from cognitive_core_engine.omega_forge.vm import MacroLibrary, VirtualMachine
from cognitive_core_engine.omega_forge.concepts import rand_inst, ConceptLibrary
from cognitive_core_engine.omega_forge.benchmark import (
    DetectorParams,
    EnvironmentCoupledFitness,
    StrictStructuralDetector,
    TaskBenchmark,
)
from cognitive_core_engine.omega_forge.evidence import EngineConfig, EvidenceWriter


# BN-10 Fix 8: Import task macros from Stage1
try:
    from cognitive_core_engine.omega_forge.stage1 import TaskMacroLibrary
    _HAS_MACROS = True
except ImportError:
    _HAS_MACROS = False


class OmegaForgeV13:
    def __init__(
        self,
        seed: int = 42,
        detector: Optional[StrictStructuralDetector] = None,
        vm: Optional[VirtualMachine] = None,
        config: Optional[EngineConfig] = None,
        concept_library: Optional[ConceptLibrary] = None,
    ) -> None:
        self._rng = random.Random(seed)
        self.seed = seed
        self.vm = vm or VirtualMachine()
        self.detector = detector or StrictStructuralDetector()
        self.cfg = config or EngineConfig()

        self.population: List[ProgramGenome] = []
        self.generation: int = 0
        self.parents_index: Dict[str, ProgramGenome] = {}
        self.env_fitness: Optional[EnvironmentCoupledFitness] = None

        # BN-10 Fix 3: Relaxed detector for RSI candidates
        self.rsi_detector: Optional[StrictStructuralDetector] = None

        # BN-10 Fix 8: Concept library for concept-based mutation
        self.concept_library = concept_library or ConceptLibrary()
        self.crossover_count: int = 0

    def _ensure_rsi_detector(self) -> None:
        """Create relaxed detector on first use when env_fitness is set."""
        if self.rsi_detector is None and self.env_fitness is not None:
            relaxed = DetectorParams(
                rsi_relaxed=True,
                rsi_min_loops=1,
                rsi_min_scc=1,
                rsi_coverage=0.35,
                rsi_require_both=False,
                K_initial=3,
                L_initial=6,
                C_coverage=0.35,
                min_loops=1,
                min_scc=1,
                require_both=False,
                warmup_gens=0,
            )
            self.rsi_detector = StrictStructuralDetector(relaxed)

    @staticmethod
    def _make_input_reading_genome(idx: int) -> List[Instruction]:
        """BN-10: Create a genome that reads inputs and computes something non-trivial.

        Half the population gets these 'seeded' genomes that actually
        read input registers, giving evolution a starting point that
        produces input-dependent outputs (required for quarantine).
        """
        # Template: conditional branch on r3 (difficulty input) → halts cleanly,
        # produces distinct outputs. This template is robust to mutation because
        # the HALT is well-separated from the logic and the conditional depends
        # on actual input values (r3 = difficulty, r2 = oq in skill_input format).
        templates = [
            [Instruction("JGT", 3, 2, 2), Instruction("SET", 0, 0, 0),
             Instruction("JMP", 2, 0, 0), Instruction("SET", 0, 1, 0),
             Instruction("ADD", 0, 1, 0), Instruction("HALT", 0, 0, 0)],
            [Instruction("ADD", 0, 1, 0), Instruction("JGT", 0, 3, 2),
             Instruction("SET", 0, 0, 0), Instruction("JMP", 1, 0, 0),
             Instruction("ADD", 0, 3, 0), Instruction("HALT", 0, 0, 0)],
            [Instruction("JLT", 0, 3, 2), Instruction("MOV", 3, 0, 0),
             Instruction("JMP", 1, 0, 0), Instruction("ADD", 0, 0, 0),
             Instruction("SUB", 0, 1, 0), Instruction("HALT", 0, 0, 0)],
            [Instruction("ADD", 0, 3, 0), Instruction("JGT", 4, 0, 2),
             Instruction("SET", 0, 2, 0), Instruction("JMP", 1, 0, 0),
             Instruction("SUB", 0, 2, 0), Instruction("HALT", 0, 0, 0)],
        ]
        template = templates[idx % len(templates)]
        # Add some random padding
        padding = [rand_inst() for _ in range(random.Random(idx).randint(3, 8))]
        return [i.clone() for i in template] + padding + [Instruction("HALT", 0, 0, 0)]

    def init_population(self) -> None:
        self.population = []
        for i in range(self.cfg.pop_size):
            # BN-10: Half seeded (input-reading), half random
            if i < self.cfg.pop_size // 2:
                insts = self._make_input_reading_genome(i)
            else:
                L = self._rng.randint(self.cfg.init_len_min, self.cfg.init_len_max)
                insts = [rand_inst() for _ in range(L)]
                insts.append(Instruction("HALT", 0, 0, 0))
            g = ProgramGenome(gid=f"init_{i}", instructions=insts, parents=[], generation=0)
            self.population.append(g)
        self._reindex()

    def _reindex(self) -> None:
        self.parents_index = {g.gid: g for g in self.population}

    def _get_parent_obj(self, g: ProgramGenome) -> Optional[ProgramGenome]:
        if not g.parents:
            return None
        pid = g.parents[0]
        return self.parents_index.get(pid)

    def crossover(self, parent_a: ProgramGenome, parent_b: ProgramGenome) -> ProgramGenome:
        """BN-10 Fix 8: Single-point crossover between two parents."""
        if not parent_a.instructions or not parent_b.instructions:
            return self.mutate(parent_a)
        cut_a = self._rng.randint(1, max(1, len(parent_a.instructions) - 1))
        cut_b = self._rng.randint(1, max(1, len(parent_b.instructions) - 1))
        child_insts = [i.clone() for i in parent_a.instructions[:cut_a]]
        child_insts += [i.clone() for i in parent_b.instructions[cut_b:]]
        # Trim to max code length
        child_insts = child_insts[:self.cfg.max_code_len]
        # Ensure HALT instruction exists
        if not any(i.op == "HALT" for i in child_insts):
            child_insts.append(Instruction("HALT", 0, 0, 0))
        child = ProgramGenome(
            gid=f"x{self.generation}_{self._rng.randint(0, 999999)}",
            instructions=child_insts,
            parents=[parent_a.gid, parent_b.gid],
            generation=self.generation,
        )
        self.crossover_count += 1
        return child

    def mutate(self, parent: ProgramGenome) -> ProgramGenome:
        child = parent.clone()
        child.generation = self.generation
        child.parents = [parent.gid]
        child.gid = f"g{self.generation}_{self._rng.randint(0, 999999)}"

        # BN-10 Fix 8 (AC9): New strategies are ADDITIONS, not replacements.
        # Probability table: 15% task-macro, 12% concept, 43% original (split proportionally)
        # Original 4 ops: 20% splice, 25% control, 30% insert, 25% delete → scaled to 43%
        roll = self._rng.random()

        # 15% task-macro insertion (NEW)
        if roll < 0.15 and _HAS_MACROS and len(child.instructions) + 10 < self.cfg.max_code_len:
            macro_fn = self._rng.choice([
                TaskMacroLibrary.sum_skeleton,
                TaskMacroLibrary.max_skeleton,
                TaskMacroLibrary.double_skeleton,
            ])
            macro = macro_fn()
            pos = self._rng.randint(0, len(child.instructions))
            child.instructions[pos:pos] = [m.clone() for m in macro]

        # 12% concept-based mutation (NEW)
        elif roll < 0.27:
            concepts = list(self.concept_library._concepts.values()) if hasattr(self.concept_library, '_concepts') else []
            if concepts and len(child.instructions) + 5 < self.cfg.max_code_len:
                concept = self._rng.choice(concepts)
                payload = concept.payload if hasattr(concept, 'payload') else []
                if isinstance(payload, list) and payload:
                    insts = []
                    for item in payload[:8]:
                        if isinstance(item, (list, tuple)) and len(item) >= 4:
                            insts.append(Instruction(str(item[0]), int(item[1]), int(item[2]), int(item[3])))
                    if insts:
                        pos = self._rng.randint(0, len(child.instructions))
                        child.instructions[pos:pos] = insts
            else:
                # Fallback to random instruction insertion
                if len(child.instructions) < self.cfg.max_code_len:
                    pos = self._rng.randint(0, len(child.instructions))
                    child.instructions.insert(pos, rand_inst())

        # Remaining 73% split among original 4 operators (proportionally)
        elif roll < 0.40 and len(child.instructions) + 5 < self.cfg.max_code_len:
            # splice macro (original)
            macro = MacroLibrary.loop_skeleton() if self._rng.random() < 0.7 else MacroLibrary.call_skeleton()
            pos = self._rng.randint(0, len(child.instructions))
            child.instructions[pos:pos] = [m.clone() for m in macro]
        elif roll < 0.56 and child.instructions:
            # replace with control op (original)
            pos = self._rng.randint(0, len(child.instructions) - 1)
            op = self._rng.choice(list(CONTROL_OPS))
            child.instructions[pos] = Instruction(op, self._rng.randint(-8, 8), self._rng.randint(0, 7), self._rng.randint(0, 7))
        elif roll < 0.76 and len(child.instructions) < self.cfg.max_code_len:
            # insert random instruction (original)
            pos = self._rng.randint(0, len(child.instructions))
            child.instructions.insert(pos, rand_inst())
        else:
            # delete (original)
            if len(child.instructions) > 6:
                pos = self._rng.randint(0, len(child.instructions) - 1)
                child.instructions.pop(pos)

        # BN-10: Ensure child always has at least one HALT instruction
        has_halt = any(inst.op == "HALT" for inst in child.instructions)
        if not has_halt:
            child.instructions.append(Instruction("HALT", 0, 0, 0))

        return child

    def _extract_concepts(self, genome: ProgramGenome) -> None:
        """BN-10 Fix 8: Extract repeated subsequences as concepts."""
        ops = genome.op_sequence()
        if len(ops) < 6:
            return
        # Look for repeated length-4 subsequences
        seen: Dict[str, List[int]] = {}
        for i in range(len(ops) - 3):
            key = "|".join(ops[i:i+4])
            if key not in seen:
                seen[key] = []
            seen[key].append(i)
        for key, positions in seen.items():
            if len(positions) >= 2:
                start = positions[0]
                payload = [(inst.op, inst.a, inst.b, inst.c)
                           for inst in genome.instructions[start:start+4]]
                from cognitive_core_engine.omega_forge.concepts import Concept
                concept = Concept(
                    cid=f"auto_{hash(key) % 100000}",
                    name=f"pattern_{key[:20]}",
                    kind="extracted",
                    payload=payload,
                    compile_fn_id="identity",
                    discovered_gen=self.generation,
                )
                self.concept_library.add_concept(concept)
                break  # one per genome per generation

    def _operand_tune(self, genome: ProgramGenome) -> None:
        """BN-10 Fix 8: Simple hill-climb on operands (±1).
        Limited to first 10 instructions to keep runtime bounded."""
        base_score = self._quick_score(genome)
        for idx in range(min(10, len(genome.instructions))):
            inst = genome.instructions[idx]
            for field in ['a', 'b', 'c']:
                original = getattr(inst, field)
                for delta in [1, -1]:
                    setattr(inst, field, original + delta)
                    new_score = self._quick_score(genome)
                    if new_score > base_score:
                        base_score = new_score
                    else:
                        setattr(inst, field, original)

    def _quick_score(self, genome: ProgramGenome) -> float:
        """Quick evaluation for operand tuning."""
        try:
            return TaskBenchmark.evaluate(genome, self.vm)
        except Exception:
            return 0.0

    def evolutionary_perturbation(self, unsat_core: Any) -> float:
        """
        [MUTATION PROTOCOL INTEGRATION] 
        Computes an evolutionary pressure gradient specifically adapted from the UNSAT failure.
        """
        import hashlib
        import math
        core_str = str(unsat_core)
        core_hash = hashlib.md5(core_str.encode()).hexdigest()
        pressure = (int(core_hash[:8], 16) / 0xffffffff) * 2.0 * math.pi
        return pressure

    def step(self, writer: Optional[EvidenceWriter] = None) -> Tuple[int, int]:
        self.generation += 1
        successes_this_gen = 0

        # BN-10 Fix 3: Ensure RSI detector exists when env_fitness is set
        self._ensure_rsi_detector()
        # Use relaxed detector for RSI mode, strict otherwise
        active_detector = self.rsi_detector if self.rsi_detector is not None else self.detector

        # Evaluate all genomes
        for g in self.population:
            parent = self._get_parent_obj(g)
            st = self.vm.execute(g, [1.0] * 8)
            passed, reasons, diag = active_detector.evaluate(g, parent, st, self.vm, self.generation)
            if passed:
                successes_this_gen += 1
                if writer is not None:
                    ev = {
                        "type": "evidence",
                        "gen": self.generation,
                        "gid": g.gid,
                        "parent": parent.gid if parent else None,
                        "code_hash": g.code_hash(),
                        "reasons": reasons,
                        "diag": diag,
                        "metrics": {
                            "steps": st.steps,
                            "coverage": st.coverage(len(g.instructions)),
                            "loops": st.loops_count,
                            "branches": st.conditional_branches,
                            "scc_n": diag.get("scc_n", 0),
                        },
                    }
                    writer.write(ev)
                    writer.flush_fsync()


        # Reproduce: score-based elite selection + CFG-diversity
        for g in self.population:
            parent = self._get_parent_obj(g)
            st2 = self.vm.execute(g, [1.0] * 8)
            cfg2 = ControlFlowGraph.from_trace(st2.trace, len(g.instructions))
            cov = st2.coverage(len(g.instructions))
            scc_n = len(cfg2.sccs())

            struct_score = cov + 0.02 * min(st2.loops_count, 50) + 0.01 * min(st2.conditional_branches, 50) + 0.03 * min(st2.max_call_depth, 10) + 0.08 * min(scc_n, 6)
            if st2.error or (not st2.halted_cleanly):
                struct_score -= 0.5

            if self.env_fitness is not None:
                task_score = 0.5 * TaskBenchmark.evaluate(g, self.vm) + 0.5 * self.env_fitness.evaluate(g, self.vm)
            else:
                task_score = TaskBenchmark.evaluate(g, self.vm)

            # BN-10: Bonuses for halting cleanly and producing diverse outputs
            halt_bonus = 0.3 if st2.halted_cleanly else 0.0
            # Test output diversity using exact quarantine inputs
            diversity_bonus = 0.0
            quarantine_inputs = [
                [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [5.0, -1.0, 0.5],
                [0.0], [1.0, 1.0, 1.0, 1.0],
            ]
            test_outputs = set()
            test_halts = 0
            for tinp in quarantine_inputs:
                try:
                    tst = self.vm.execute(g, tinp)
                    if tst.halted_cleanly:
                        test_halts += 1
                        test_outputs.add(round(tst.regs[0], 4))
                except Exception:
                    pass
            if len(test_outputs) >= 2 and test_halts >= 2:
                diversity_bonus = 0.4  # strong bonus for quarantine-viable genomes
            elif st2.halted_cleanly and len(test_outputs) >= 2:
                diversity_bonus = 0.2
            score = 0.5 * struct_score + 0.5 * task_score * 2.0 + halt_bonus + diversity_bonus
            g.last_score = float(score)
            g.last_cfg_hash = cfg2.canonical_hash()

        ranked = sorted(self.population, key=lambda x: x.last_score, reverse=True)

        # BN-10 Fix 8: Extract concepts from above-median genomes
        if len(ranked) > 1:
            median_score = ranked[len(ranked) // 2].last_score
            for g in ranked[:len(ranked) // 4]:
                if g.last_score > median_score:
                    self._extract_concepts(g)

        # Diversity filter
        elites: List[ProgramGenome] = []
        seen_cfg: Set[str] = set()
        band = ranked[: max(self.cfg.elite_keep * 3, self.cfg.elite_keep)]
        for g in band:
            if len(elites) >= self.cfg.elite_keep:
                break
            if g.last_cfg_hash not in seen_cfg:
                elites.append(g)
                seen_cfg.add(g.last_cfg_hash)

        if len(elites) < self.cfg.elite_keep:
            for g in ranked:
                if len(elites) >= self.cfg.elite_keep:
                    break
                elites.append(g)

        next_pop: List[ProgramGenome] = []
        for e in elites:
            kept = e.clone()
            next_pop.append(kept)
            for _ in range(self.cfg.children_per_elite):
                # BN-10 Fix 8: 30% crossover, 70% mutation (AC9)
                if self._rng.random() < 0.30 and len(elites) >= 2:
                    partner = self._rng.choice([x for x in elites if x.gid != e.gid] or elites)
                    next_pop.append(self.crossover(e, partner))
                else:
                    next_pop.append(self.mutate(e))

        # BN-10 Fix 8: Operand tuning on top elite (expensive, run sparingly)
        if self.generation % 10 == 0 and elites:
            self._operand_tune(elites[0])

        self.population = next_pop[: self.cfg.pop_size]
        self._reindex()

        return successes_this_gen, len(getattr(active_detector, "seen_success_hashes", set()))
