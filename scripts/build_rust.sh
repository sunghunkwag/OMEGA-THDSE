#!/usr/bin/env bash
# PLAN.md Phase 7.1 — build the Rust hdc_core crate via maturin.
#
# Requires:
#   - cargo / rustc (https://rustup.rs)
#   - maturin (`pip install maturin`)
#   - Python with the same interpreter pytest will use later
#
# After this script completes successfully, ``import hdc_core`` works
# in the active Python environment and ``ArenaManager().backend``
# reports ``"rust"``.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HDC_CORE_DIR="${REPO_ROOT}/thdse/src/hdc_core"

if [ ! -d "${HDC_CORE_DIR}" ]; then
    echo "ERROR: hdc_core source directory missing at ${HDC_CORE_DIR}" >&2
    exit 1
fi

if ! command -v maturin >/dev/null 2>&1; then
    echo "ERROR: maturin not found in PATH. Install via 'pip install maturin'." >&2
    exit 2
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: cargo not found in PATH. Install Rust via https://rustup.rs/." >&2
    exit 3
fi

echo "[build_rust] building hdc_core in release mode..."
cd "${HDC_CORE_DIR}"
maturin develop --release

echo "[build_rust] verifying import..."
python3 -c "import hdc_core; print('hdc_core OK at', hdc_core.__file__)"

echo "[build_rust] verifying ArenaManager backend..."
python3 -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from shared.arena_manager import ArenaManager
mgr = ArenaManager(master_seed=0)
print('ArenaManager backend:', mgr.backend)
assert mgr.backend == 'rust', f'expected rust backend, got {mgr.backend}'
"

echo "[build_rust] done."
