"""FHRR Vectors via Rust PyO3 FhrrArena."""
from __future__ import annotations
import hdc_core
import zlib
import numpy as np
from typing import List, Optional, Any

# Global Arena for CCE-Test
GLOBAL_ARENA_CAPACITY = 100000
GLOBAL_ARENA_DIMENSION = 10000
ARENA = hdc_core.FhrrArena(GLOBAL_ARENA_CAPACITY, GLOBAL_ARENA_DIMENSION)

class FhrrVector:
    """Wrapper that mimics the old HyperVector interface but natively routes to Rust's FhrrArena."""
    DIM = GLOBAL_ARENA_DIMENSION

    def __init__(self, handle: Optional[int] = None) -> None:
        if handle is None:
            self.handle = ARENA.allocate()
            # Random phases
            phases = (np.random.rand(self.DIM) * 2 * np.pi).tolist()
            ARENA.inject_phases(self.handle, phases)
        else:
            self.handle = handle

    @classmethod
    def from_seed(cls, seed_obj: Any) -> FhrrVector:
        s = str(seed_obj)
        h = zlib.crc32(s.encode('utf-8')) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        phases = (rng.random(cls.DIM) * 2 * np.pi).tolist()
        handle = ARENA.allocate()
        ARENA.inject_phases(handle, phases)
        return cls(handle)

    @classmethod
    def zero(cls) -> FhrrVector:
        handle = ARENA.allocate()
        ARENA.inject_phases(handle, [0.0] * cls.DIM)
        return cls(handle)

    def bind(self, other: FhrrVector) -> FhrrVector:
        out_handle = ARENA.allocate()
        ARENA.bind(self.handle, other.handle, out_handle)
        return FhrrVector(out_handle)

    def fractional_bind(self, other: FhrrVector, role_index: int) -> FhrrVector:
        # FHRR fractional binding is usually multiplying phase by fraction,
        # but to keep it simple, we just apply a standard bind as the old code used permute-then-XOR
        shifted = other.permute(role_index * 7 + 1)
        return self.bind(shifted)

    def permute(self, shifts: int = 1) -> FhrrVector:
        # We can implement permutation by extracting phases, rolling, and injecting
        phases = ARENA.extract_phases(self.handle)
        rolled = np.roll(phases, shifts).tolist()
        out_handle = ARENA.allocate()
        ARENA.inject_phases(out_handle, rolled)
        return FhrrVector(out_handle)

    def similarity(self, other: FhrrVector) -> float:
        # FhrrArena compute_correlation returns FHRR cosine similarity
        raw_corr = ARENA.compute_correlation(self.handle, other.handle)
        # Map roughly to [0,1]
        return (raw_corr + 1.0) / 2.0

    def cosine_similarity(self, other: FhrrVector) -> float:
        return ARENA.compute_correlation(self.handle, other.handle)

    @staticmethod
    def bundle(vectors: List[FhrrVector]) -> FhrrVector:
        if not vectors:
            return FhrrVector.zero()
        out_handle = ARENA.allocate()
        handles = [v.handle for v in vectors]
        ARENA.bundle(handles, out_handle)
        return FhrrVector(out_handle)
