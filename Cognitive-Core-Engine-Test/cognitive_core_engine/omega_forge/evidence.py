"""
Evidence writer and engine configuration for the OMEGA_FORGE engine.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict


class EvidenceWriter:
    def __init__(self, out_path: str) -> None:
        self.out_path = out_path
        # Always write a header marker so "empty file" is never ambiguous
        self.f = open(out_path, "a", encoding="utf-8", buffering=1)
        self.write({"type": "header", "version": "V13_CLEAN", "note": "jsonl; each line is crash-safe"})
        self.flush_fsync()

    def write(self, obj: Dict[str, Any]) -> None:
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def flush_fsync(self) -> None:
        self.f.flush()
        try:
            os.fsync(self.f.fileno())
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.flush_fsync()
        finally:
            self.f.close()


@dataclass
class EngineConfig:
    pop_size: int = 30
    init_len_min: int = 18
    init_len_max: int = 28
    elite_keep: int = 12
    children_per_elite: int = 2
    max_code_len: int = 80
