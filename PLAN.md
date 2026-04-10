# PLAN.md — OMEGA-THDSE Unified AI Core Integration Plan

> **Status**: Phase 1 Complete (Human Analysis)  
> **Next**: Phase 2 → Claude Code  
> **Author**: Pre-integration architectural analysis  
> **Date**: 2025-04-10  

---

## Section A: Arena Fragmentation Analysis

### Current State: 3 Separate FHRR Arenas (CRITICAL PROBLEM)

The codebase contains **three independent FhrrArena instances** that never share handles:

| Arena | Location | DIM | CAP | Purpose |
|-------|----------|-----|-----|---------|
| **CCE Arena** | `cognitive_core_engine/core/fhrr.py` | 10,000 | 100,000 | Concept encoding, memory, skills |
| **THDSE Arena** | `thdse/src/hdc_core/src/lib.rs` (Rust) | 256 | 2,000,000 | Axiomatic synthesis, SERL, swarm |
| **Bridge Arena** | `bridge/goal_corpus_selector.py` | 10,000 | 50,000 | Goal selection (ORPHANED — never called) |

### Why This Is Broken

1. **No handle can cross arena boundaries.** A handle allocated in CCE Arena (dim=10000) is meaningless in THDSE Arena (dim=256). They are different vector spaces.
2. **The Bridge Arena in `goal_corpus_selector.py` creates a THIRD independent arena** with its own handles. It imports `hdc_core.FhrrArena` directly — bypassing any centralization.
3. **CCE and THDSE cannot communicate.** There is currently ZERO pathway for a concept encoded in CCE (10k-dim) to influence axiomatic synthesis in THDSE (256-dim), or vice versa.

### Integration Target

Replace all 3 arenas with a **single ArenaManager** that:
- Owns exactly 2 arenas: `cce_arena (DIM=10000, CAP=100000)` and `thdse_arena (DIM=256, CAP=2000000)`
- Provides a `bridge_arena (DIM=10000, CAP=50000)` for cross-engine correlation
- Enforces that NO other code calls `hdc_core.FhrrArena()` directly
- Routes all handle allocation through named methods: `alloc_cce()`, `alloc_thdse()`, `alloc_bridge()`

---

## Section B: Integration Gap Inventory

### 10 Identified Gaps Between CCE and THDSE

| # | Gap | CCE Side | THDSE Side | Severity |
|---|-----|----------|------------|----------|
| 1 | **Arena Isolation** | `fhrr.py` creates own arena | Rust `lib.rs` creates own arena | CRITICAL |
| 2 | **No Concept→Axiom Bridge** | Concepts encoded as 10k-dim FHRR | Axioms encoded as 256-dim FHRR | CRITICAL |
| 3 | **No Axiom→Skill Bridge** | SkillLibrary.register() has no governance gate | SERL produces programs, no registration path | HIGH |
| 4 | **No Causal→Provenance Bridge** | CausalChainTracker tracks events | THDSE has no provenance tracking | HIGH |
| 5 | **No Memory→Hypothesis Bridge** | Memory uses associative retrieval (10k) | Wall archive stores fitness scores only | MEDIUM |
| 6 | **No Governance→Synthesis Bridge** | Critic/Sandbox validate code | Axiomatic synthesizer has no governance hook | CRITICAL |
| 7 | **No World Model→Swarm Bridge** | WorldModel does state prediction | Swarm explores parameter space blindly | MEDIUM |
| 8 | **Goal Generator Isolation** | GoalGenerator creates goals | THDSE has no goal input mechanism | HIGH |
| 9 | **Self-Referential Model Isolation** | SelfReferentialModel encodes 4 HDC states | THDSE cannot read self-model | MEDIUM |
| 10 | **RSI Pipeline Isolation** | RSI quarantines/compiles/registers | THDSE SERL evolves programs separately | HIGH |

### Priority Order for Implementation

**Phase 2 (Foundation):** Gaps 1, 2 — ArenaManager + DimensionBridge
**Phase 3 (Bridges):** Gaps 3, 4, 6, 8, 10 — Five critical bridge modules
**Phase 4 (Enhancement):** Gaps 5, 7, 9 — Three enhancement bridges + core file modifications
**Phase 5 (Integration):** E2E testing + CLI + documentation

---

## Section C: Dimension Bridge Mathematics

### The Problem

CCE uses DIM=10,000. THDSE uses DIM=256. FHRR vectors are **phase vectors** where each component is an angle in [0, 2π). The algebraic operations are:

- **Bind**: element-wise phase addition mod 2π
- **Bundle**: circular mean of phase vectors
- **Similarity**: mean cosine of phase differences

Any dimension bridge must **preserve bind/bundle algebraic properties** or the downstream HDC operations become meaningless.

### Rejected Approaches

#### 1. Fourier Interpolation (DFT → zero-pad → IDFT)
- **Why rejected**: FHRR phases are NOT frequency-domain signals. They are independent circular random variables. Zero-padding a DFT of phase vectors produces artifacts that destroy bind orthogonality. Specifically: if `bind(A, B)` is quasi-orthogonal to `A` in 256-dim, the interpolated 10k-dim versions have NO guarantee of quasi-orthogonality.
- **Mathematical proof**: For FHRR, `E[cos(A_i - B_i)] ≈ 0` when A, B are independent. After DFT interpolation, the 10k components become correlated linear combinations of the 256 originals, violating the independence assumption.

#### 2. Random Projection (256 → 10k via random matrix)
- **Why rejected**: Random projection preserves Euclidean distances (Johnson-Lindenstrauss), but FHRR similarity is **circular** (cosine of phase difference), not Euclidean. The projected vectors would not be valid phase vectors in [0, 2π).

#### 3. Symmetric Algebraic Bridge
- **Why rejected**: There is no algebraic isomorphism between FHRR(256) and FHRR(10000). The spaces have fundamentally different capacity and orthogonality properties.

### Approved Architecture: ASYMMETRIC BRIDGE

The bridge is intentionally asymmetric because the two directions serve different purposes:

#### Direction 1: CCE → THDSE (10k → 256) — Phase Subsampling

```
def project_down(vec_10k: np.ndarray) -> np.ndarray:
    """Project 10,000-dim FHRR vector to 256-dim via phase subsampling."""
    indices = np.arange(0, 10000, 39)[:256]  # Every 39th component
    return vec_10k[indices]
```

**Why this works**: Subsampling preserves bind exactly:
- `subsample(bind(A, B)) = bind(subsample(A), subsample(B))`
- Because bind is element-wise, selecting a subset of components preserves the operation.

Bundle is preserved approximately (mean of subset ≈ mean of full, by LLN for large enough subset).

**Verification test**: For 1000 random vector pairs, `similarity(subsample(bind(A,B)), bind(subsample(A), subsample(B)))` must equal **exactly 1.0** (not approximately — EXACTLY, since subsampling commutes with element-wise addition).

#### Direction 2: THDSE → CCE (256 → 10k) — Semantic Correlation (NOT Reconstruction)

**Critical insight**: We do NOT need to reconstruct a 10k vector from a 256-dim vector. That is mathematically impossible without information loss. Instead, we need to **answer a question**: "How similar is this 256-dim THDSE result to this 10k-dim CCE concept?"

```
def cross_arena_similarity(vec_10k: np.ndarray, vec_256: np.ndarray) -> float:
    """Compute similarity between vectors from different arenas.
    
    Projects the 10k vector DOWN to 256, then computes FHRR similarity
    in the 256-dim space where both vectors are valid.
    """
    projected = project_down(vec_10k)
    # FHRR similarity: mean cosine of phase differences
    return float(np.mean(np.cos(projected - vec_256)))
```

**Why this works**: Instead of lifting 256→10k (impossible without information fabrication), we project 10k→256 (lossless for the selected components) and compare in the shared 256-dim space.

**Verification test**:
- `cross_arena_similarity(A_10k, project_down(A_10k))` must return **> 0.95** (self-similarity through projection)
- `cross_arena_similarity(A_10k, random_256)` must return **≈ 0.0 ± 0.1** (orthogonality with random)

### Bridge Algebra Summary Table

| Operation | Method | Preserves Bind? | Preserves Bundle? | Preserves Similarity? |
|-----------|--------|-----------------|-------------------|-----------------------|
| 10k→256 | Phase subsampling (stride 39) | EXACT | Approximate (LLN) | Approximate |
| 256↔10k similarity | Project down + compare in 256 | N/A | N/A | YES (in 256-space) |
| 256→10k reconstruction | **NOT IMPLEMENTED** | — | — | — |

---

## Section D: Invariant Constants Registry

### IMMUTABLE VALUES — DO NOT MODIFY

These constants are extracted from the existing codebase. Changing ANY of them will break existing tests, alter system behavior, or violate architectural invariants.

```python
# === Arena Dimensions ===
CCE_ARENA_DIM = 10_000
CCE_ARENA_CAP = 100_000
THDSE_ARENA_DIM = 256
THDSE_ARENA_CAP = 2_000_000
BRIDGE_ARENA_DIM = 10_000
BRIDGE_ARENA_CAP = 50_000

# === Orchestrator (C-layer) ===
_MAX_META_RECURSION_DEPTH = 2          # IMMUTABLE — hardcoded safety limit
_MIN_CLEAN_HALTS = 2                    # Required clean halts before meta-recursion exit

# === Agent (B-type) ===
BN06_INITIAL_DEPTH = 1                  # BN-06 adaptive meta-depth initial value
BN06_MAX_DEPTH = 3                      # BN-06 maximum meta-depth
BN06_DEPTH_INCREMENT_THRESHOLD = 2      # Consecutive failures before depth increase
DRIFT_CRITICAL_THRESHOLD = 0.35         # Orchestrator drift detection threshold
DRIFT_WARNING_THRESHOLD = 0.2           # Orchestrator drift warning threshold

# === Skills ===
SKILL_SIMILARITY_THRESHOLD = 0.6        # Minimum similarity for skill retrieval
MAX_SKILLS_RETURNED = 5                 # Maximum skills returned per query

# === Memory ===
MEMORY_SIMILARITY_THRESHOLD = 0.3       # Minimum similarity for memory recall
MAX_MEMORIES_RETURNED = 5               # Maximum memories returned per query
MEMORY_TITLE_WEIGHT = 2                 # Title encoding weight multiplier

# === Self-Referential Model ===
SELF_MODEL_COMPONENTS = 4               # belief, goal, capability, emotion
WIREHEADING_THRESHOLD = 0.92            # Anti-wireheading detection threshold
CONTINUITY_THRESHOLD = 0.85             # Identity continuity threshold

# === SERL (Synthesis Evolution) ===
SERL_FITNESS_GATE = 0.4                 # Minimum fitness for SERL program survival
SERL_POPULATION_SIZE = 50               # SERL population size
SERL_MAX_GENERATIONS = 100              # SERL maximum generations
SERL_MUTATION_RATE = 0.1                # SERL mutation probability
SERL_CROSSOVER_RATE = 0.7               # SERL crossover probability
SERL_TOURNAMENT_SIZE = 5                # SERL tournament selection size

# === Swarm ===
SWARM_CONSENSUS_THRESHOLD = 0.85        # Swarm consensus agreement threshold
SWARM_NUM_AGENTS = 10                   # Number of swarm agents
SWARM_MAX_ITERATIONS = 50               # Swarm maximum iterations

# === Axiomatic Synthesizer ===
Z3_TIMEOUT_MS = 5000                    # Z3 solver timeout in milliseconds

# === RSI Pipeline ===
RSI_QUARANTINE_TIMEOUT = 30             # Seconds before quarantine timeout
RSI_MAX_RETRIES = 3                     # Maximum compilation retries

# === Governance ===
SANDBOX_TIMEOUT = 10                    # Sandbox execution timeout (seconds)
CRITIC_THRESHOLD = 0.7                  # Critic approval threshold

# === Dimension Bridge ===
BRIDGE_SUBSAMPLE_STRIDE = 39            # 10000 / 256 ≈ 39.06, take every 39th
BRIDGE_SELF_SIMILARITY_MIN = 0.95       # Minimum self-similarity through projection
BRIDGE_RANDOM_SIMILARITY_MAX = 0.1      # Maximum similarity with random vectors
MAX_CREDIBLE_LEAP = 0.25                # Maximum credible fitness leap per generation
```

---

## Section E: DeterministicRNG Design

### The Problem

THDSE **must** be deterministic for axiomatic verification (Z3 proofs require reproducibility). CCE uses controlled randomness via seed chains for exploration. Currently, there is no structural guarantee — only convention.

### Structural Solution: DeterministicRNG Interface

```python
# File: shared/deterministic_rng.py

import numpy as np
from typing import Optional

class DeterministicRNG:
    """Structural determinism enforcement for the unified engine.
    
    THDSE operations get FROZEN RNG (no randomness allowed).
    CCE operations get SEEDED RNG (reproducible randomness).
    
    Usage:
        rng = DeterministicRNG(master_seed=42)
        
        # For THDSE (deterministic):
        thdse_rng = rng.fork("thdse")  # Always returns same sequence
        
        # For CCE (reproducible random):
        cce_rng = rng.fork("cce")  # Seeded, reproducible
    """
    
    def __init__(self, master_seed: int = 42):
        self._master_seed = master_seed
        self._forks: dict[str, np.random.Generator] = {}
    
    def fork(self, namespace: str) -> np.random.Generator:
        """Create a deterministic fork for a subsystem."""
        if namespace not in self._forks:
            # Derive child seed deterministically from master + namespace
            child_seed = hash((self._master_seed, namespace)) % (2**32)
            self._forks[namespace] = np.random.default_rng(child_seed)
        return self._forks[namespace]
    
    def reset(self, namespace: Optional[str] = None):
        """Reset a fork (or all forks) to initial state."""
        if namespace:
            if namespace in self._forks:
                child_seed = hash((self._master_seed, namespace)) % (2**32)
                self._forks[namespace] = np.random.default_rng(child_seed)
        else:
            self._forks.clear()
    
    @property
    def master_seed(self) -> int:
        return self._master_seed


class FrozenRNG:
    """RNG that raises on any attempt to generate random numbers.
    
    Used for THDSE synthesis paths where ANY randomness would
    invalidate Z3 axiomatic proofs.
    """
    
    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattribute__(name)
        raise RuntimeError(
            f"FrozenRNG: Attempted to call '{name}()' in a deterministic context. "
            f"THDSE synthesis operations must not use randomness. "
            f"If you need randomness, use DeterministicRNG.fork('cce') instead."
        )
```

### Integration Points

1. **ArenaManager** holds a single `DeterministicRNG(master_seed=42)` instance
2. **THDSE Axiomatic Synthesizer** receives `FrozenRNG` — any call to random methods raises immediately
3. **CCE Agent/Orchestrator** receives `rng.fork("cce")` — reproducible but allows randomness
4. **SERL** receives `rng.fork("serl")` — reproducible evolution
5. **Swarm** receives `rng.fork("swarm")` — reproducible exploration

---

## Section F: Risk Assessment & Mitigation

### Risk 1: Rust FFI Boundary Violations
- **Risk**: Passing Rust-allocated objects (FhrrArena handles, Z3 solver refs) across Python IPC boundaries
- **Mitigation**: ArenaManager wraps all Rust calls; handles are integer IDs, never raw pointers
- **Test**: Pickle/unpickle an ArenaManager → must raise `RuntimeError("Cannot serialize Rust FFI objects")`

### Risk 2: THDSE Determinism Corruption
- **Risk**: Someone adds `random.random()` or `np.random.rand()` in a THDSE synthesis path
- **Mitigation**: FrozenRNG structurally prevents this; grep audit as secondary check
- **Test**: `FrozenRNG().random()` must raise `RuntimeError`
- **Audit**: `grep -rn "import random\|np.random.rand\|np.random.random" thdse/src/synthesis/` must return 0 hits (excluding deterministic_rng.py itself)

### Risk 3: Silent Governance Bypass
- **Risk**: THDSE-synthesized programs registered as skills without governance approval
- **Mitigation**: `SkillLibrary.register()` requires `governance_approved=True` parameter
- **Test**: `skill_lib.register("test", lambda: None)` without `governance_approved=True` must raise `GovernanceError`

### Risk 4: Dimension Confusion
- **Risk**: A 256-dim handle accidentally used in a 10k-dim operation (silent wrong results)
- **Mitigation**: ArenaManager tags every handle with its arena origin; DimensionBridge validates dimensions
- **Test**: `bridge.project_down(vec_256)` must raise `DimensionMismatchError` (input must be 10k)

### Risk 5: Causal Chain Corruption
- **Risk**: THDSE UNSAT results not logged, breaking causal chain completeness
- **Mitigation**: Bridge module catches UNSAT and logs to CausalChainTracker
- **Test**: After a failed synthesis, `tracker.get_chain()` must contain an UNSAT event

### Risk 6: Meta-Recursion Depth Escape
- **Risk**: Integration accidentally allows recursion beyond `_MAX_META_RECURSION_DEPTH = 2`
- **Mitigation**: Constant is hardcoded (not configurable); orchestrator checks on every cycle
- **Test**: Attempting depth=3 must raise `RecursionLimitError`

---

## Section G: Anti-Shortcut Rules (12 Rules)

These rules MUST be enforced in every phase. Claude Code must not violate any of them.

```
RULE 1 (NO STUBS): No function may contain only `pass`, `return None`, 
    `raise NotImplementedError`, or `...` (ellipsis). Every function must 
    have a real implementation.

RULE 2 (NO FAKE TESTS): No test may contain only `assert True`, 
    `assert x is not None`, or assertions without computed values. 
    Every test must assert a specific numerical or behavioral property.

RULE 3 (NO DIRECT ARENA): No file except `arena_manager.py` may call 
    `hdc_core.FhrrArena()` directly. All arena access goes through 
    ArenaManager.

RULE 4 (NO BULK COPY): No more than 15 consecutive lines may be 
    copy-pasted from an existing file. Refactor, don't duplicate.

RULE 5 (NO PHANTOM IMPORTS): Every `import` statement must resolve to 
    an existing module. No importing from files that don't exist yet.

RULE 6 (DIMENSION SAFETY): Any operation mixing handles from different 
    arenas must go through DimensionBridge. Direct cross-arena operations 
    must raise DimensionMismatchError.

RULE 7 (GOVERNANCE GATE): SkillLibrary.register() must require 
    governance_approved=True. Calls without this parameter must raise 
    GovernanceError.

RULE 8 (NO SILENT FAILURES): THDSE UNSAT results must be logged to 
    CausalChainTracker. No synthesis failure may be silently swallowed.

RULE 9 (PROVENANCE REQUIRED): Every bridge operation result must include 
    metadata["provenance"] identifying the source arena and operation.

RULE 10 (DETERMINISM): THDSE synthesis paths must use FrozenRNG. 
    CCE paths must use seeded DeterministicRNG forks. No bare 
    random.random() or np.random.rand() calls.

RULE 11 (PROCESS ISOLATION): No Rust-allocated object (FhrrArena, 
    Z3 solver) may be passed across process boundaries. Use integer 
    handle IDs only.

RULE 12 (BRIDGE SELF-TEST): DimensionBridge must run a self-test on 
    import that verifies: (a) subsample(bind(A,B)) == bind(subsample(A), 
    subsample(B)) for 10 random pairs, (b) self-similarity > 0.95, 
    (c) random similarity < 0.1. If any test fails, raise 
    BridgeIntegrityError on import.
```

---

## Section H: File Structure After Integration

```
OMEGA-THDSE/
├── PLAN.md                          ← THIS FILE
├── .gitignore
├── shared/                          ← NEW: shared infrastructure
│   ├── __init__.py
│   ├── arena_manager.py             ← Phase 2: Central arena ownership
│   ├── dimension_bridge.py          ← Phase 2: Asymmetric 10k↔256 bridge
│   ├── deterministic_rng.py         ← Phase 2: DeterministicRNG + FrozenRNG
│   ├── exceptions.py                ← Phase 2: DimensionMismatchError, GovernanceError, etc.
│   └── constants.py                 ← Phase 2: All invariant constants from Section D
├── bridges/                         ← NEW: cross-engine bridge modules
│   ├── __init__.py
│   ├── concept_axiom_bridge.py      ← Phase 3: Gap 2 — Concept → Axiom
│   ├── axiom_skill_bridge.py        ← Phase 3: Gap 3 — Axiom → Skill (with governance)
│   ├── causal_provenance_bridge.py  ← Phase 3: Gap 4 — Causal → Provenance
│   ├── governance_synthesis_bridge.py ← Phase 3: Gap 6 — Governance → Synthesis
│   ├── goal_synthesis_bridge.py     ← Phase 3: Gap 8 — Goal → Synthesis
│   └── rsi_serl_bridge.py           ← Phase 3: Gap 10 — RSI ↔ SERL
├── Cognitive-Core-Engine-Test/      ← EXISTING (modified in Phase 4)
│   └── cognitive_core_engine/
│       ├── core/
│       │   ├── fhrr.py              ← Phase 4: Remove direct arena, use ArenaManager
│       │   ├── skills.py            ← Phase 4: Add governance_approved gate
│       │   ├── memory.py            ← Phase 4: Wire to DeterministicRNG
│       │   ├── agent.py             ← Phase 4: Wire to DeterministicRNG fork
│       │   ├── orchestrator.py      ← Phase 4: Wire to ArenaManager + bridges
│       │   ├── causal_chain.py      ← Phase 4: Accept THDSE provenance events
│       │   └── ...
│       ├── governance/
│       │   └── ...                  ← Phase 4: Wire critic to synthesis bridge
│       └── agi_modules/
│           ├── self_referential_model.py ← Phase 4: Expose to THDSE read
│           ├── goal_generator.py    ← Phase 4: Wire to goal_synthesis_bridge
│           └── ...
├── thdse/                           ← EXISTING (modified in Phase 4)
│   └── src/
│       ├── hdc_core/                ← Phase 4: Use ArenaManager (no direct arena)
│       ├── synthesis/
│       │   ├── axiomatic_synthesizer.py ← Phase 4: Accept FrozenRNG
│       │   └── serl.py              ← Phase 4: Accept DeterministicRNG fork
│       ├── swarm/                   ← Phase 4: Wire to world model bridge
│       └── ...
├── bridge/                          ← EXISTING (deprecated in Phase 4)
│   └── goal_corpus_selector.py      ← Phase 4: Deprecated, replaced by bridges/
├── tests/                           ← NEW: integration tests
│   ├── __init__.py
│   ├── test_arena_manager.py        ← Phase 2
│   ├── test_dimension_bridge.py     ← Phase 2
│   ├── test_deterministic_rng.py    ← Phase 2
│   ├── test_concept_axiom_bridge.py ← Phase 3
│   ├── test_axiom_skill_bridge.py   ← Phase 3
│   ├── test_governance_synthesis.py ← Phase 3
│   ├── test_goal_synthesis.py       ← Phase 3
│   ├── test_rsi_serl_bridge.py      ← Phase 3
│   ├── test_e2e_pipeline.py         ← Phase 5
│   └── test_rule_compliance.py      ← Phase 5: Automated 12-rule audit
└── cli.py                           ← Phase 5: Unified CLI entry point
```

---

## Section I: Phase Execution Order

| Phase | Scope | Deliverables | Gate Condition |
|-------|-------|-------------|----------------|
| **Phase 1** | Analysis (THIS DOCUMENT) | PLAN.md committed to repo | Human review ✓ |
| **Phase 2** | Foundation Layer | `shared/` (5 files) + `tests/` (3 files) | All 3 test files pass |
| **Phase 3** | Bridge Modules | `bridges/` (6 files) + `tests/` (5 files) | All 5 test files pass + Phase 2 tests still pass |
| **Phase 4** | Core Enhancement | 7+ existing file modifications + regression | All Phase 2+3 tests still pass + modified file tests pass |
| **Phase 5** | Final Integration | E2E tests + CLI + README + audit script | Full test suite passes + 12-rule audit passes |

**CRITICAL**: Each phase MUST pass its gate condition before proceeding to the next phase. Claude Code must output "Phase N complete. All tests pass." before receiving the next prompt.

---

*End of PLAN.md — Phase 1 Analysis Complete*
