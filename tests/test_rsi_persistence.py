import pytest
from pathlib import Path
import json
from shared.atom_bank import clear_atoms, load_atoms, save_atoms
from benchmarks.runner import run_suite
from benchmarks.sorting_synthesis import all_problems, SEED_CORPUS

def test_bank_io():
    """Verify basic save/load/dedup."""
    clear_atoms()
    save_atoms(["return x+1", "return x+1"])
    assert load_atoms() == ["return x+1"]
    save_atoms(["return abs(x)"])
    assert load_atoms() == ["return abs(x)", "return x+1"]
    clear_atoms()

def test_extraction_logic():
    """Verify the cleanup filter in runner.py."""
    from benchmarks.runner import _extract_clean_atoms
    src = "def test(x):\n    return x\n    import os\n    return long_string_over_80_chars_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    atoms = _extract_clean_atoms(src)
    assert atoms == ["return x"]

def test_persistence_loop():
    """Verify that solved problems actually populate the bank."""
    clear_atoms()
    # Run a tiny subset (just one problem)
    probs = [p for p in all_problems() if p.name == "sum_list"]
    result = run_suite(
        probs,
        SEED_CORPUS,
        use_behavioural=False,
        use_goal_direction=False,
        use_cegr=False,
        max_cycles=2,
    )
    
    atoms = load_atoms()
    assert len(atoms) > 0
    assert any("sum(" in a for a in atoms)

def test_solve_rate_stability():
    """Verify that solve rate is at least 3/5 with any bank state."""
    # This ensures we didn't break baseline performance
    result = run_suite(
        all_problems(),
        SEED_CORPUS,
        use_behavioural=False,
        use_goal_direction=False,
        use_cegr=False,
        max_cycles=3,
    )
    assert result["totals"]["solved_count"] >= 3
