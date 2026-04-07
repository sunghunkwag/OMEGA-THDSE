"""
Virtual machine and macro library for the OMEGA_FORGE engine.
"""
from __future__ import annotations

import math
from typing import Dict, List

from cognitive_core_engine.omega_forge.instructions import (
    ExecutionState,
    Instruction,
    ProgramGenome,
)


class VirtualMachine:
    def __init__(self, max_steps: int = 400, memory_size: int = 64, stack_limit: int = 16) -> None:
        self.max_steps = max_steps
        self.memory_size = memory_size
        self.stack_limit = stack_limit

    def reset(self, inputs: List[float]) -> ExecutionState:
        regs = [0.0] * 8
        mem: Dict[int, float] = {}
        for i, v in enumerate(inputs):
            if i < self.memory_size:
                mem[i] = float(v)
        regs[1] = float(len(inputs))
        return ExecutionState(regs=regs, memory=mem)

    def execute(self, genome: ProgramGenome, inputs: List[float]) -> ExecutionState:
        st = self.reset(inputs)
        code = genome.instructions
        L = len(code)

        recent_hashes: List[int] = []
        while not st.halted and st.steps < self.max_steps:
            if st.pc < 0 or st.pc >= L:
                st.halted = True
                st.halted_cleanly = True
                break

            st.visited_pcs.add(st.pc)
            st.trace.append(st.pc)
            prev_pc = st.pc
            inst = code[st.pc]
            st.steps += 1

            # Degenerate loop detection: if state hashes collapse, stop with error
            state_sig = hash((st.pc, tuple(int(x) for x in st.regs[:4]), len(st.stack)))
            recent_hashes.append(state_sig)
            if len(recent_hashes) > 25:
                recent_hashes.pop(0)
                if len(set(recent_hashes)) < 3:
                    st.error = "DEGENERATE_LOOP"
                    st.halted = True
                    break

            try:
                self._step(st, inst)
            except Exception as e:
                st.error = f"VM_ERR:{e.__class__.__name__}"
                st.halted = True
                break

            # Loop + branch stats
            if st.pc <= prev_pc and not st.halted:
                st.loops_count += 1
            if inst.op in {"JZ", "JNZ", "JGT", "JLT"}:
                st.conditional_branches += 1
            st.max_call_depth = max(st.max_call_depth, len(st.stack))

        return st

    def _step(self, st: ExecutionState, inst: Instruction) -> None:
        op, a, b, c = inst.op, inst.a, inst.b, inst.c
        r = st.regs

        def clamp(x: float) -> float:
            if not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x):
                return 0.0
            return float(max(-1e9, min(1e9, x)))

        def addr(x: float) -> int:
            return int(max(0, min(self.memory_size - 1, int(x))))

        jump = False

        if op == "HALT":
            st.halted = True
            st.halted_cleanly = True
            return

        if op == "SET":
            r[c % 8] = float(a)
        elif op == "MOV":
            r[c % 8] = float(r[a % 8])
        elif op == "SWAP":
            ra, rb = a % 8, b % 8
            r[ra], r[rb] = r[rb], r[ra]
        elif op == "ADD":
            r[c % 8] = clamp(r[a % 8] + r[b % 8])
        elif op == "SUB":
            r[c % 8] = clamp(r[a % 8] - r[b % 8])
        elif op == "MUL":
            r[c % 8] = clamp(r[a % 8] * r[b % 8])
        elif op == "DIV":
            den = r[b % 8]
            r[c % 8] = clamp(r[a % 8] / den) if abs(den) > 1e-9 else 0.0
        elif op == "INC":
            r[c % 8] = clamp(r[c % 8] + 1.0)
        elif op == "DEC":
            r[c % 8] = clamp(r[c % 8] - 1.0)
        elif op == "LOAD":
            idx = addr(r[a % 8])
            st.memory_reads += 1
            r[c % 8] = float(st.memory.get(idx, 0.0))
        elif op == "STORE":
            idx = addr(r[a % 8])
            st.memory_writes += 1
            st.memory[idx] = clamp(r[c % 8])
        elif op == "LDI":
            base = addr(r[a % 8])
            off = addr(r[b % 8])
            st.memory_reads += 1
            r[c % 8] = float(st.memory.get(addr(base + off), 0.0))
        elif op == "STI":
            base = addr(r[a % 8])
            off = addr(r[b % 8])
            st.memory_writes += 1
            st.memory[addr(base + off)] = clamp(r[c % 8])
        elif op == "JMP":
            st.pc += int(a)
            jump = True
        elif op == "JZ":
            if abs(r[a % 8]) < 1e-9:
                st.pc += int(b)
                jump = True
        elif op == "JNZ":
            if abs(r[a % 8]) >= 1e-9:
                st.pc += int(b)
                jump = True
        elif op == "JGT":
            if r[a % 8] > r[b % 8]:
                st.pc += int(c)
                jump = True
        elif op == "JLT":
            if r[a % 8] < r[b % 8]:
                st.pc += int(c)
                jump = True
        elif op == "CALL":
            if len(st.stack) >= self.stack_limit:
                st.error = "STACK_OVERFLOW"
                st.halted = True
                return
            st.stack.append(st.pc + 1)
            st.pc += int(a)
            jump = True
        elif op == "RET":
            if not st.stack:
                st.halted = True
                st.halted_cleanly = True
                jump = True
            else:
                st.pc = st.stack.pop()
                jump = True
        else:
            # Unknown op => halt
            st.error = "UNKNOWN_OP"
            st.halted = True
            return

        if not jump:
            st.pc += 1


class MacroLibrary:
    @staticmethod
    def loop_skeleton(idx_reg: int = 2, limit_reg: int = 1) -> List[Instruction]:
        # i=0 ; if i<limit: body ; i++ ; jump back ; halt path outside
        return [
            Instruction("SET", 0, 0, idx_reg),
            Instruction("JLT", idx_reg, limit_reg, 4),   # jump into body if i < limit
            Instruction("JMP", 6, 0, 0),                # skip body (exit)
            Instruction("INC", 0, 0, idx_reg),          # body: i++
            Instruction("JMP", -3, 0, 0),               # loop back to JLT
        ]

    @staticmethod
    def call_skeleton() -> List[Instruction]:
        # CALL forward to a mini-routine and RET
        return [
            Instruction("CALL", 2, 0, 0),
            Instruction("JMP", 3, 0, 0),
            Instruction("INC", 0, 0, 0),
            Instruction("RET", 0, 0, 0),
        ]
