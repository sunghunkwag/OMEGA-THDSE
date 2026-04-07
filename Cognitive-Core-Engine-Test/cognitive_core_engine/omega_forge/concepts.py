"""
Concept library, opcode bias sampling, and random instruction generation
for the OMEGA_FORGE engine.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from cognitive_core_engine.omega_forge.instructions import (
    OPS,
    Instruction,
)


@dataclass
class Concept:
    cid: str
    name: str
    kind: str
    payload: Dict[str, Any]
    compile_fn_id: str
    stats: Dict[str, Any] = field(default_factory=dict)
    discovered_gen: int = 0
    parents: List[str] = field(default_factory=list)


def _inst_tuple_list(instructions: List[Instruction]) -> List[Tuple[Any, ...]]:
    return [inst.to_tuple() for inst in instructions]


def _concept_hash_from_insts(instructions: List[Instruction]) -> str:
    h = hashlib.sha256()
    for inst in instructions:
        h.update(repr(inst.to_tuple()).encode("utf-8"))
    return h.hexdigest()[:16]


def _compile_macro_v1(payload: Dict[str, Any]) -> List[Instruction]:
    insts = []
    for op, a, b, c in payload.get("instructions", []):
        insts.append(Instruction(str(op), int(a), int(b), int(c)))
    return insts


CONCEPT_COMPILE_FNS: Dict[str, Callable[[Dict[str, Any]], List[Instruction]]] = {
    "macro_v1": _compile_macro_v1,
}


class ConceptLibrary:
    def __init__(self, max_size: int = 200) -> None:
        self.max_size = max_size
        self._concepts: Dict[str, Concept] = {}
        self._hash_index: Dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._concepts)

    def all_concepts(self) -> List[Concept]:
        return list(self._concepts.values())

    def get(self, cid: str) -> Optional[Concept]:
        return self._concepts.get(cid)

    def add_concept(self, concept: Concept, dedup: bool = True) -> Optional[str]:
        instructions = self.compile(concept)
        if not instructions:
            return None
        digest = _concept_hash_from_insts(instructions)
        if dedup and digest in self._hash_index:
            return self._hash_index[digest]
        if len(self._concepts) >= self.max_size:
            return None
        self._concepts[concept.cid] = concept
        self._hash_index[digest] = concept.cid
        concept.stats.setdefault("digest", digest)
        concept.stats.setdefault("length", len(instructions))
        return concept.cid

    def compile(self, concept: Concept) -> List[Instruction]:
        fn = CONCEPT_COMPILE_FNS.get(concept.compile_fn_id)
        if not fn:
            return []
        return fn(concept.payload)

    def save(self, path: str) -> None:
        data = []
        for c in self._concepts.values():
            data.append({
                "cid": c.cid,
                "name": c.name,
                "kind": c.kind,
                "payload": c.payload,
                "compile_fn_id": c.compile_fn_id,
                "stats": c.stats,
                "discovered_gen": c.discovered_gen,
                "parents": c.parents,
            })
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        for entry in raw or []:
            concept = Concept(
                cid=str(entry.get("cid", "")),
                name=str(entry.get("name", "concept")),
                kind=str(entry.get("kind", "macro")),
                payload=dict(entry.get("payload", {})),
                compile_fn_id=str(entry.get("compile_fn_id", "macro_v1")),
                stats=dict(entry.get("stats", {})),
                discovered_gen=int(entry.get("discovered_gen", 0)),
                parents=list(entry.get("parents", [])),
            )
            self.add_concept(concept, dedup=True)


# ------------------------------------------------------------------------------
# Feedback-biased opcode sampling (Stage2 -> Stage1)
# ------------------------------------------------------------------------------
OP_BIAS: Dict[str, float] = {}  # e.g., {"LOAD":1.4,"ADD":1.3,...}
CONCEPT_BIAS: Dict[str, float] = {}
CONCEPT_ANTI_BIAS: Set[str] = set()
MACRO_LENGTH_BIAS: float = 0.0


def set_op_bias(op_bias: Dict[str, float]) -> None:
    """
    Install opcode sampling bias used by rand_inst() in Stage 1.
    Values are nonnegative weights; missing ops default to 1.0.
    """
    global OP_BIAS
    OP_BIAS = {k: float(v) for k, v in (op_bias or {}).items() if float(v) > 0.0}


def set_concept_bias(concept_bias: Dict[str, float],
                     anti_bias: Optional[List[str]] = None,
                     macro_length_bias: Optional[float] = None) -> None:
    global CONCEPT_BIAS, CONCEPT_ANTI_BIAS, MACRO_LENGTH_BIAS
    CONCEPT_BIAS = {k: float(v) for k, v in (concept_bias or {}).items() if float(v) > 0.0}
    CONCEPT_ANTI_BIAS = set(anti_bias or [])
    if macro_length_bias is not None:
        MACRO_LENGTH_BIAS = float(macro_length_bias)


def _sample_op(rng: random.Random) -> str:
    if not OP_BIAS:
        return rng.choice(OPS)
    weights = [OP_BIAS.get(op, 1.0) for op in OPS]
    # Avoid all-zero
    if not any(w > 0.0 for w in weights):
        return rng.choice(OPS)
    return rng.choices(OPS, weights=weights, k=1)[0]


def rand_inst(rng: Optional[random.Random] = None) -> Instruction:
    """
    Random instruction generator. If OP_BIAS is set (via Stage2 feedback),
    opcode selection is weighted accordingly.
    """
    rng = rng or random
    op = _sample_op(rng)
    return Instruction(op, rng.randint(-8, 31), rng.randint(0, 7), rng.randint(0, 7))
