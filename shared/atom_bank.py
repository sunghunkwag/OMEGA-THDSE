import json
import logging
from pathlib import Path

BANK_PATH = Path(__file__).parent / "atom_bank.json"

def load_atoms() -> list[str]:
    """
    Returns the stored atom list from BANK_PATH.
    If the file does not exist, returns [].
    If the file is malformed JSON, logs a warning and returns [].
    """
    if not BANK_PATH.exists():
        return []
    try:
        with BANK_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Failed to load atom bank from {BANK_PATH}: {e}")
        return []

def save_atoms(new_atoms: list[str]) -> None:
    """
    Loads existing atoms, merges with new_atoms (set union),
    sorts the result, writes back to BANK_PATH as JSON.
    Never writes duplicates.
    """
    existing = set(load_atoms())
    merged = existing.union(set(new_atoms))
    sorted_atoms = sorted(list(merged))
    
    with BANK_PATH.open("w", encoding="utf-8") as f:
        json.dump(sorted_atoms, f, indent=2)

def clear_atoms() -> None:
    """
    Deletes BANK_PATH if it exists. Used only in tests.
    """
    if BANK_PATH.exists():
        BANK_PATH.unlink()
