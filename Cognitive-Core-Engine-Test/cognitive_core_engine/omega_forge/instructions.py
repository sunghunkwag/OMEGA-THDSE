"""
Instruction set, ProgramGenome, and ExecutionState for the OMEGA_FORGE engine.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ==============================================================================
# 1) Instruction set
# ==============================================================================

OPS = [
    "MOV", "SET", "SWAP",
    "ADD", "SUB", "MUL", "DIV", "INC", "DEC",
    "LOAD", "STORE", "LDI", "STI",
    "JMP", "JZ", "JNZ", "JGT", "JLT",
    "CALL", "RET", "HALT"
]
CONTROL_OPS = {"JMP", "JZ", "JNZ", "JGT", "JLT", "CALL", "RET"}
MEMORY_OPS = {"LOAD", "STORE", "LDI", "STI"}


@dataclass
class Instruction:
    op: str
    a: int = 0
    b: int = 0
    c: int = 0

    def clone(self) -> "Instruction":
        return Instruction(self.op, self.a, self.b, self.c)

    def to_tuple(self) -> Tuple[Any, ...]:
        return (self.op, int(self.a), int(self.b), int(self.c))


# ==============================================================================
# 2) Program genome
# ==============================================================================

@dataclass
class ProgramGenome:
    gid: str
    instructions: List[Instruction]
    parents: List[str] = field(default_factory=list)
    generation: int = 0
    last_score: float = 0.0
    last_cfg_hash: str = ""
    concept_trace: List[str] = field(default_factory=list)
    concept_proposals: List[str] = field(default_factory=list)

    def clone(self) -> "ProgramGenome":
        return ProgramGenome(
            gid=self.gid,
            instructions=[i.clone() for i in self.instructions],
            parents=list(self.parents),
            generation=self.generation,
            concept_trace=list(self.concept_trace),
            concept_proposals=list(self.concept_proposals),
        )

    def code_hash(self) -> str:
        h = hashlib.sha256()
        for inst in self.instructions:
            h.update(repr(inst.to_tuple()).encode("utf-8"))
        return h.hexdigest()[:16]

    def op_sequence(self) -> List[str]:
        return [i.op for i in self.instructions]


# ==============================================================================
# 3) Execution state
# ==============================================================================

@dataclass
class ExecutionState:
    regs: List[float]
    memory: Dict[int, float]
    pc: int = 0
    stack: List[int] = field(default_factory=list)
    steps: int = 0
    halted: bool = False
    halted_cleanly: bool = False
    error: Optional[str] = None

    trace: List[int] = field(default_factory=list)
    visited_pcs: Set[int] = field(default_factory=set)

    loops_count: int = 0
    conditional_branches: int = 0
    max_call_depth: int = 0
    memory_reads: int = 0
    memory_writes: int = 0

    def coverage(self, code_len: int) -> float:
        if code_len <= 0:
            return 0.0
        return len(self.visited_pcs) / float(code_len)

    def fingerprint(self) -> Tuple[int, int, int, int, int]:
        return (
            min(self.loops_count, 20),
            min(self.conditional_branches, 20),
            min(self.memory_writes, 50),
            min(self.memory_reads, 50),
            min(self.max_call_depth, 10),
        )
