"""
ARC-AGI Exhaustive Rule-Based Solver (BN-07)

Implements a library of general-purpose grid transformations, then for each
task tries every single and composite (two-step) transformation against ALL
train pairs.  A transformation is accepted only if it produces exact output
for every train pair.

Transformation library:
  Geometric: identity, hflip, vflip, rotate 90/180/270, transpose,
             anti-diagonal flip
  Value:     value_swap (A↔B), value_invert (0↔nonzero),
             value_invert_preserve_center, center_border_swap,
             bitwise_or_fill
  Data-driven: color_map (derived from train pairs)
  Composite: all ordered pairs of the above

Anti-cheat: no hardcoded outputs, no task-index lookups, no memorization.
Rules inferred ONLY from train pairs (train-test firewall).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

Grid = List[List[int]]
Transform = Callable[[Grid], Grid]


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------

def identity(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def horizontal_flip(grid: Grid) -> Grid:
    """Reverse each row."""
    return [row[::-1] for row in grid]


def vertical_flip(grid: Grid) -> Grid:
    """Reverse row order."""
    return grid[::-1]


def rotate_90_cw(grid: Grid) -> Grid:
    """Rotate 90 degrees clockwise."""
    if not grid or not grid[0]:
        return grid
    R, C = len(grid), len(grid[0])
    result = [[0] * R for _ in range(C)]
    for r in range(R):
        for c in range(C):
            result[c][R - 1 - r] = grid[r][c]
    return result


def rotate_180(grid: Grid) -> Grid:
    """Rotate 180 degrees."""
    return [row[::-1] for row in grid[::-1]]


def rotate_270_cw(grid: Grid) -> Grid:
    """Rotate 270 degrees clockwise (= 90 CCW)."""
    if not grid or not grid[0]:
        return grid
    R, C = len(grid), len(grid[0])
    result = [[0] * R for _ in range(C)]
    for r in range(R):
        for c in range(C):
            result[C - 1 - c][r] = grid[r][c]
    return result


def transpose(grid: Grid) -> Grid:
    """Swap rows and columns."""
    if not grid or not grid[0]:
        return grid
    R, C = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(R)] for c in range(C)]


def anti_diagonal_flip(grid: Grid) -> Grid:
    """Flip across the anti-diagonal: (i,j) -> (C-1-j, R-1-i)."""
    if not grid or not grid[0]:
        return grid
    R, C = len(grid), len(grid[0])
    return [[grid[R - 1 - c][C - 1 - r] for c in range(R)] for r in range(C)]


# ---------------------------------------------------------------------------
# Value transforms
# ---------------------------------------------------------------------------

def _grid_values(grid: Grid) -> Set[int]:
    return {v for row in grid for v in row}


def value_swap(grid: Grid) -> Grid:
    """Swap the two distinct values in the grid (exactly 2 values)."""
    vals = sorted(_grid_values(grid))
    if len(vals) != 2:
        return [row[:] for row in grid]
    a, b = vals
    return [[b if v == a else a for v in row] for row in grid]


def value_invert(grid: Grid) -> Grid:
    """Swap 0 with the max nonzero value."""
    vals = _grid_values(grid)
    nonzero = {v for v in vals if v != 0}
    if not nonzero:
        return [row[:] for row in grid]
    nz = max(nonzero)
    return [[nz if v == 0 else (0 if v == nz else v) for v in row] for row in grid]


def value_invert_preserve_center(grid: Grid) -> Grid:
    """Swap 0 with nonzero everywhere EXCEPT the center cell of odd-sized grids."""
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    R, C = len(grid), len(grid[0])
    vals = _grid_values(grid)
    if len(vals) != 2 or 0 not in vals:
        return [row[:] for row in grid]
    nz = max(vals - {0})
    cr, cc = R // 2, C // 2
    is_odd = (R % 2 == 1 and C % 2 == 1)
    result: Grid = []
    for r in range(R):
        row: List[int] = []
        for c in range(C):
            if is_odd and r == cr and c == cc:
                row.append(grid[r][c])
            elif grid[r][c] == 0:
                row.append(nz)
            else:
                row.append(0)
        result.append(row)
    return result


def center_border_swap(grid: Grid) -> Grid:
    """Swap center-cell value with the other value (0) in odd-sized grids."""
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    R, C = len(grid), len(grid[0])
    if R < 3 or C < 3:
        return [row[:] for row in grid]
    cr, cc = R // 2, C // 2
    center_val = grid[cr][cc]
    if center_val == 0:
        return [row[:] for row in grid]
    return [[center_val if v == 0 else (0 if v == center_val else v)
             for v in row] for row in grid]


def bitwise_or_fill(grid: Grid) -> Grid:
    """Replace all 0s with the max nonzero value found in the grid."""
    vals = _grid_values(grid)
    nonzero = {v for v in vals if v != 0}
    if not nonzero:
        return [row[:] for row in grid]
    fill = max(nonzero)
    return [[fill if v == 0 else v for v in row] for row in grid]


# ---------------------------------------------------------------------------
# Color-map based transform (derived from train pairs)
# ---------------------------------------------------------------------------

def _derive_color_map(train_pairs: List[Dict]) -> Optional[Dict[int, int]]:
    """Derive a consistent per-value substitution map from train pairs."""
    mapping: Dict[int, int] = {}
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        if len(inp) != len(out):
            return None
        for r in range(len(inp)):
            if len(inp[r]) != len(out[r]):
                return None
            for c in range(len(inp[r])):
                v_in, v_out = inp[r][c], out[r][c]
                if v_in in mapping:
                    if mapping[v_in] != v_out:
                        return None
                else:
                    mapping[v_in] = v_out
    if all(k == v for k, v in mapping.items()):
        return None
    return mapping


# ---------------------------------------------------------------------------
# Candidate list (ORDER MATTERS: more common/general transforms first)
# ---------------------------------------------------------------------------

_SINGLE_TRANSFORMS: List[Tuple[str, Transform]] = [
    ("identity", identity),
    ("horizontal_flip", horizontal_flip),
    ("vertical_flip", vertical_flip),
    ("rotate_90_cw", rotate_90_cw),
    ("rotate_180", rotate_180),
    ("rotate_270_cw", rotate_270_cw),
    ("transpose", transpose),
    ("anti_diagonal_flip", anti_diagonal_flip),
    ("value_swap", value_swap),
    ("value_invert", value_invert),
    ("value_invert_preserve_center", value_invert_preserve_center),
    ("center_border_swap", center_border_swap),
    ("bitwise_or_fill", bitwise_or_fill),
]


def _check_all_train(fn: Transform, train_pairs: List[Dict]) -> bool:
    """Return True if fn maps every train input to its output exactly."""
    for pair in train_pairs:
        try:
            predicted = fn(pair["input"])
            if predicted != pair["output"]:
                return False
        except Exception:
            return False
    return True


def solve_arc_task(task: dict) -> Optional[Grid]:
    """Attempt to solve an ARC task.  Returns predicted test output or None.

    Args:
        task: dict with "train" (list of {input, output} pairs) and
              "test_input" (the 2D grid to transform).

    The solver infers rules ONLY from train pairs (train-test firewall).
    Returns None if no rule matches all train examples.
    """
    train_pairs: List[Dict] = task.get("train", [])
    test_input: Grid = task.get("test_input", [])

    if not train_pairs or not test_input:
        return None

    # 1. Try single transforms (skip identity to prevent trivial bypass)
    for name, fn in _SINGLE_TRANSFORMS:
        if name == "identity":
            continue
        if _check_all_train(fn, train_pairs):
            return fn(test_input)

    # 2. Try color-map derived from training data
    cmap = _derive_color_map(train_pairs)
    if cmap is not None:
        cm_fn = lambda g, _m=cmap: [[_m.get(v, v) for v in row] for row in g]
        if _check_all_train(cm_fn, train_pairs):
            return cm_fn(test_input)

    # 3. Try compositions of two transforms
    transforms = [(n, fn) for n, fn in _SINGLE_TRANSFORMS]
    for n1, t1 in transforms:
        for n2, t2 in transforms:
            if n1 == "identity" and n2 == "identity":
                continue
            def composed(g: Grid, _a: Transform = t1, _b: Transform = t2) -> Grid:
                return _b(_a(g))
            if _check_all_train(composed, train_pairs):
                result = composed(test_input)
                if result != test_input:
                    return result

    # No rule found — honest failure
    return None
