# OMEGA-THDSE

**Unified AI core integrating the Cognitive Core Engine (CCE) with the
Topological Hyperdimensional Symbolic Engine (THDSE) under a single,
verifiable, anti-cheat-enforced architecture.**

This repository is the product of a five-phase integration plan that
takes two previously disjoint research engines — a 10,000-dim FHRR
agent-side stack (CCE) and a 256-dim Rust/Z3 axiomatic synthesizer
(THDSE) — and welds them into one coherent system without sacrificing
either side's invariants.

---

## Why this project exists

Before this integration the repository shipped **three independent FHRR
arenas** that could not share handles with one another:

| Arena         | Where it lived                                             | Dim   |
|---------------|------------------------------------------------------------|-------|
| CCE arena     | `cognitive_core_engine/core/fhrr.py` module-level global   | 10000 |
| THDSE arena   | Rust crate (`thdse/src/hdc_core/src/lib.rs`)               | 256   |
| Bridge arena  | `bridge/goal_corpus_selector.py` (orphaned, never called)  | 10000 |

A handle allocated in the CCE arena was meaningless in the THDSE arena
because they live in different vector spaces, and the bridge arena
duplicated CCE's dimensions while bypassing every other subsystem.
Worse still, **there was no pathway at all** through which a CCE concept
could influence a THDSE axiomatic proof — or vice versa.

The fix is a centrally-owned `ArenaManager` that holds exactly the two
working arenas (CCE @ 10k, THDSE @ 256) plus a `bridge_arena` for
cross-engine correlation, paired with an **asymmetric
`DimensionBridge`** whose mathematics were designed specifically to
preserve FHRR bind/bundle invariants across the 10k↔256 boundary.

---

## Architecture at a glance

```
┌──────────────────────────────────────────────────────────────┐
│                    shared/  (Phase 2 foundation)             │
│  ArenaManager  •  DimensionBridge  •  DeterministicRNG       │
│  constants     •  exceptions                                  │
└──────────────────────────────────────────────────────────────┘
            ↑                ↑                    ↑
            │                │                    │
┌───────────┴────────┐ ┌────┴─────────┐ ┌────────┴──────────┐
│ bridges/ (Phase 3) │ │ Phase 4 wire │ │   tests/ (all 5)  │
│ 6 gap modules      │ │ into CCE     │ │   205+ tests      │
│ + Phase 4          │ │ (fhrr,skills,│ │   + E2E + audit   │
│   3 gap modules    │ │  agent, …)   │ │                   │
└────────────────────┘ └──────────────┘ └───────────────────┘
```

**Key principle:** `shared/arena_manager.py` is the **one and only** file
permitted to instantiate `hdc_core.FhrrArena`. Every other file must
request handles through `ArenaManager.alloc_cce()`, `alloc_thdse()`, or
`alloc_bridge()`. This is enforced mechanically by the rule-compliance
audit (see Rule 3 below).

### The asymmetric dimension bridge

FHRR vectors are **phase vectors**: every component is an angle in
`[0, 2π)`, and `bind` is element-wise phase addition mod 2π. Because
bind is element-wise, **stride-39 phase subsampling commutes with bind
exactly**:

```
subsample(bind(A, B)) == bind(subsample(A), subsample(B))
```

That identity is the reason `DimensionBridge.project_down` uses
`np.arange(0, 10000, 39)[:256]` and never anything fancier. The reverse
direction (256 → 10k) is *not implemented at all* — instead,
cross-arena similarity is computed by projecting the 10k vector **down**
to 256 and comparing in the shared 256-dim space. The bridge runs a
bind-commutation self-test on import; any regression of the stride
invariant raises `BridgeIntegrityError` before downstream bridges load.

---

## Directory layout

```
OMEGA-THDSE/
├── PLAN.md                          # Single source of truth (Phase 1)
├── README.md                        # You are here (Phase 5)
├── cli.py                           # Unified CLI entry point (Phase 5)
├── shared/                          # Phase 2 — frozen after Phase 2
│   ├── __init__.py
│   ├── constants.py                 # Invariant constants (PLAN Section D)
│   ├── exceptions.py                # Typed boundary errors
│   ├── deterministic_rng.py         # DeterministicRNG + FrozenRNG
│   ├── arena_manager.py             # Single arena owner (Rule 3)
│   └── dimension_bridge.py          # 10k↔256 asymmetric bridge (Rule 12)
├── bridges/                         # Phases 3 + 4
│   ├── __init__.py
│   ├── concept_axiom_bridge.py      # Gap 2  (Phase 3)
│   ├── axiom_skill_bridge.py        # Gap 3  (Phase 3)
│   ├── causal_provenance_bridge.py  # Gap 4  (Phase 3)
│   ├── governance_synthesis_bridge.py  # Gap 6 (Phase 3)
│   ├── goal_synthesis_bridge.py     # Gap 8  (Phase 3)
│   ├── rsi_serl_bridge.py           # Gap 10 (Phase 3)
│   ├── memory_hypothesis_bridge.py  # Gap 5  (Phase 4)
│   ├── world_model_swarm_bridge.py  # Gap 7  (Phase 4)
│   └── self_model_bridge.py         # Gap 9  (Phase 4)
├── Cognitive-Core-Engine-Test/      # CCE — modified in Phase 4
│   └── cognitive_core_engine/core/  # fhrr, skills, memory, agent,
│       └── …                         # orchestrator, causal_chain, …
├── thdse/                           # THDSE Rust side (unchanged)
└── tests/                           # All five phases' tests
    ├── test_arena_manager.py
    ├── test_dimension_bridge.py
    ├── test_deterministic_rng.py
    ├── test_concept_axiom_bridge.py
    ├── test_axiom_skill_bridge.py
    ├── test_governance_synthesis.py   # includes Rule 8 audit
    ├── test_goal_synthesis.py
    ├── test_rsi_serl_bridge.py
    ├── test_memory_hypothesis_bridge.py
    ├── test_world_model_swarm_bridge.py
    ├── test_self_model_bridge.py
    ├── test_phase4_modifications.py
    ├── test_e2e_pipeline.py           # Phase 5 E2E integration
    └── test_rule_compliance.py        # Phase 5 automated 12-rule audit
```

---

## Quick start

```bash
# Install runtime deps (pytest + numpy).
pip install numpy pytest

# Run the complete test suite (all five phases).
python cli.py test

# Run the 12-rule compliance audit only.
python cli.py audit

# Inspect live integration status (verifies filesystem, no hardcoded ✓).
python cli.py status

# List every bridge module and the PLAN.md gap it closes.
python cli.py bridges
```

`cli.py status` walks the repository, checks that each phase's
artifacts exist, and prints a `✓` or `✗` marker per phase. If anything
is missing the summary line flips to `MISSING DELIVERABLES` and the
exit status is `1`.

---

## Integration gap inventory

PLAN.md Section B identified **ten integration gaps**. All ten are now
closed by the modules in `bridges/`:

| Gap | Description                              | Bridge module                     | Phase |
|-----|------------------------------------------|-----------------------------------|-------|
| 1   | Arena isolation (3 disjoint arenas)      | `shared/arena_manager.py`         | 2     |
| 2   | No concept → axiom pathway               | `concept_axiom_bridge.py`         | 3     |
| 3   | No axiom → skill pathway                 | `axiom_skill_bridge.py`           | 3     |
| 4   | No causal ↔ provenance linkage           | `causal_provenance_bridge.py`     | 3     |
| 5   | Memory ↔ Wall archive hypothesis scoring | `memory_hypothesis_bridge.py`     | 4     |
| 6   | No governance gate on synthesis          | `governance_synthesis_bridge.py`  | 3     |
| 7   | World model ↔ Swarm blind exploration    | `world_model_swarm_bridge.py`     | 4     |
| 8   | Goal generator isolation                 | `goal_synthesis_bridge.py`        | 3     |
| 9   | Self-referential model isolation         | `self_model_bridge.py`            | 4     |
| 10  | RSI pipeline ↔ SERL isolation            | `rsi_serl_bridge.py`              | 3     |

---

## The twelve anti-shortcut rules

PLAN.md Section G defines twelve rules that guard the integration from
the "just ship something" failure mode. They are enforced
automatically by `tests/test_rule_compliance.py` — the suite fails
loudly if any of them are violated.

| #   | Rule                 | One-line statement                                                               |
|-----|----------------------|-----------------------------------------------------------------------------------|
| 1   | NO STUBS             | No function body may be only `pass` / `return None` / `NotImplementedError`.     |
| 2   | NO FAKE TESTS        | No `assert True` / `assert x is not None`. Every assert checks a computed value. |
| 3   | NO DIRECT ARENA      | Only `arena_manager.py` may call `hdc_core.FhrrArena()`.                          |
| 4   | NO BULK COPY         | No 15-line block copy-pasted across bridge files.                                 |
| 5   | NO PHANTOM IMPORTS   | Every required import must resolve. Optional backends live inside `try:`.        |
| 6   | DIMENSION SAFETY     | Cross-arena ops go through `DimensionBridge`; wrong shapes raise.                 |
| 7   | GOVERNANCE GATE      | `SkillLibrary.register()` requires `governance_approved=True` exactly.            |
| 8   | NO SILENT FAILURES   | Every UNSAT event is logged (`data["logged"] = True`).                            |
| 9   | PROVENANCE REQUIRED  | Every bridge return dict carries `metadata["provenance"]`.                        |
| 10  | DETERMINISM          | THDSE paths use `FrozenRNG`; CCE paths use `DeterministicRNG` forks.              |
| 11  | PROCESS ISOLATION    | Rust-backed arenas cannot cross process boundaries (`pickle` raises).             |
| 12  | BRIDGE SELF-TEST     | `DimensionBridge` runs its invariant self-test at import time.                    |

---

## Phase completion checklist

| Phase | Scope                        | Deliverables                                                                     | Status |
|-------|------------------------------|----------------------------------------------------------------------------------|--------|
| 1     | Analysis                     | `PLAN.md` committed                                                              | ✓     |
| 2     | Foundation layer             | `shared/` with 6 files + 3 test modules (~70 tests)                              | ✓     |
| 3     | Bridge modules               | `bridges/` with 6 gap modules + 5 test modules (~70 tests)                       | ✓     |
| 4     | Core enhancement             | 7 CCE files modified + 3 new bridges + 4 test modules (~65 tests)                | ✓     |
| 5     | Final integration            | E2E + 12-rule audit + `cli.py` + this README + full pipeline verification       | ✓     |

Every green check-mark above is produced by `cli.py status` after
walking the real filesystem — there is no hardcoded state.

---

## How the test suite is organised

Running `python cli.py test` executes the tests in three conceptual
layers:

1. **Foundation (Phase 2):** isolate the primitives —
   `test_arena_manager`, `test_dimension_bridge`, `test_deterministic_rng`.
2. **Bridges (Phases 3 + 4):** one file per bridge, plus
   `test_phase4_modifications` for the hand-wired CCE surface changes.
3. **Integration (Phase 5):** `test_e2e_pipeline` exercises real
   cross-module paths end-to-end; `test_rule_compliance` is the
   programmatic 12-rule audit.

The expected steady-state count is **240+ tests** across all phases.
If any test fails, the failing phase's rule-set is the first thing to
inspect — every test traces back to one of the twelve anti-shortcut
rules above.

---

## License & authorship

Integration plan and Phase 2–5 implementation authored as part of the
OMEGA-THDSE unification project. See `PLAN.md` for the complete
architectural analysis and rationale.
