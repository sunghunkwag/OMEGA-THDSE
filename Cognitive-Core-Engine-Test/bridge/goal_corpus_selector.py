import hdc_core
import numpy as np
from typing import List, Dict

class GoalCorpusSelector:
    """
    Direct bridge between CCE-Test goals and THDSE's Rust FHRR Arena.
    Bypasses Python string manipulation, explicitly manipulating floats.
    """
    def __init__(self, capacity: int = 20000, dimension: int = 10000):
        self.arena = hdc_core.FhrrArena(capacity, dimension)
        self.goal_handles: Dict[str, int] = {}
        
    def register_goal(self, goal_id: str, phases: List[float]) -> int:
        """Injects directly into Rust memory."""
        handle = self.arena.allocate()
        self.arena.inject_phases(handle, phases)
        self.goal_handles[goal_id] = handle
        return handle

    def compute_goal_alignment(self, candidate_handle: int) -> Dict[str, float]:
        """Uses Rust SIMD backend to correlate a candidate against all goals."""
        handles = list(self.goal_handles.values())
        keys = list(self.goal_handles.keys())
        
        # Batch correlation computed natively in Rust
        if not handles:
            return {}
        correlations = self.arena.correlate_matrix_subset([candidate_handle], handles)
        return {keys[i]: correlations[i] for i in range(len(keys))}
        
    def swap_fhrr_state(self, h1: int, h2: int) -> int:
        """Executes bind operation natively."""
        out_handle = self.arena.allocate()
        self.arena.bind(h1, h2, out_handle)
        return out_handle
