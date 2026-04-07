# Cognitive-Core-Engine-Test

Multi-module architecture integrating fixed cognitive core with invention, governance, and capability layers — featuring a self-referential meta-simulation model with anti-wireheading defense.

> **Scope Notice:** This is a research prototype demonstrating cognitive mechanisms — autonomous goal generation, hierarchical abstraction, intrinsic motivation, recursive self-modeling, and governance-gated self-modification — within a simulated environment.

## Project Structure

```
cognitive_core_engine/
  core/                     # Fixed cognitive orchestrator
    utils.py                  stable_hash, now_ms, tokenize
    hdc.py                    HyperVector (10,000-bit HDC)
    memory.py                 MemoryItem, SharedMemory
    tools.py                  ToolRegistry, tool factories
    skills.py                 SkillStep, Skill, SkillLibrary (vm_skills, perf log)
    algorithm_env.py          AlgorithmSynthesisEnvironment, task hierarchy L0-L4
    world_model.py            TransitionSummary, WorldModel (TD learning)
    planner.py                PlanCandidate, Planner (beam search)
    project_graph.py          ProjectNode, ProjectGraph
    environment.py            RuleProposal, TaskSpec, ResearchEnvironment
    causal_chain.py           CausalChainTracker, CausalEvent (BN-08)
    agent.py                  AgentConfig, Agent (B-type + RSI skill consultation)
    orchestrator.py           OrchestratorConfig, Orchestrator (C-layer + emergence loop)
  omega_forge/              # Invention plugin (invoked on stagnation)
    instructions.py           OPS, Instruction, ProgramGenome, ExecutionState
    cfg.py                    ControlFlowGraph
    vm.py                     VirtualMachine, MacroLibrary
    concepts.py               Concept, ConceptLibrary, rand_inst
    benchmark.py              TaskBenchmark, EnvironmentCoupledFitness, StrictStructuralDetector
    evidence.py               EvidenceWriter, EngineConfig
    engine.py                 OmegaForgeV13
    stage1.py                 Stage1Engine, TaskBenchmarkV4, ConceptDiscoveryBenchmark
    stage2.py                 Stage2Engine, feedback functions
    rsi_pipeline.py           RSISkillRegistrar — OmegaForge → SkillLibrary pipeline (BN-02)
    cli.py                    CLI commands and entry points
  governance/               # Evaluation gate and meta-control
    utils.py                  now_ms, sha256, clamp, safe_mkdir, etc.
    critic.py                 critic_evaluate_candidate_packet, RunLogger
    invention.py              Invention system (18 classes)
    sandbox.py                SAFE_BUILTINS, validators, safe_exec, safe_eval
    engine_types.py           EngineStrategy, TaskSpec, Genome, Batch, evaluate()
    evolution.py              Mutation operators, OPERATORS dict, mutate_learner
    meta.py                   SurrogateModel, MAPElitesArchive, Universe, GlobalState
    autopatch.py              AutoPatch functions, scoring, filtering
    loops.py                  run_duo_loop, run_rsi_loop
    cli.py                    build_parser, cmd_* functions, main()
agi_modules/                # Capability extension modules
  competence_map.py           Zone-of-proximal-development tracking
  goal_generator.py           Autonomous goal creation (frontier/gap/creative/skill-derived)
  intrinsic_motivation.py     Curiosity, novelty, learning progress rewards
  concept_graph.py            Hierarchical abstraction (L0-L5)
  hierarchical_planner.py     Multi-level planning via concept graph
  transfer_engine.py          Cross-domain transfer with HDC structural matching (BN-04)
  self_model.py               Legacy capability prediction (backward compat)
  self_referential_model.py   Advanced self-referential meta-simulation
  difficulty_scheduler.py     Curriculum learning with chaos injection
  self_improvement.py         Empirical env-rollout parameter tuning
  agi_tracker.py              5-axis capability proxy + 3 emergence metrics (BN-08)
  external_benchmark.py       ARC-AGI + HumanEval held-out validation (BN-03)
  arc_solver.py               Rule-based ARC-AGI grid solver (BN-07)
  humaneval_solver.py          Template-matching HumanEval solver (BN-07)
  solver_bridge.py             Solver API bridge for ExternalBenchmarkHarness (BN-07)
data/
  arc_agi_sample.json         20 bundled ARC-AGI tasks (BN-03)
  humaneval_sample.json       10 bundled HumanEval problems (BN-03)
tests/
  test_selftest.py            Core selftest + contract negative tests
  test_benchmarks.py          ADB, ARC, program synthesis benchmarks
  test_agi_integration.py     11 integration tests + anti-cheat audit
  test_solvers.py             21 solver tests with anti-cheat checks (BN-07)
  test_algorithm_env.py       59 algorithm synthesis tests (A1-A7, B1-B12, C1-C8, D1-D8, F1-F10)
  test_emergence.py           10 emergence mechanism tests (BN-08)
  test_flow.py                10 recursive loop plumbing tests (BN-09)
scripts/
  run_results.py              Reproduce baseline evidence logs
  run_agi_evidence.py         50-round evidence with 3-way ablation
  verify_self_improvement.py  Self-improvement verification suite
  run_multi_seed_evidence.py  20-seed statistical evidence (A+ Task 5)
  run_algo_evidence.py        L0 task solve evidence (A+ Task 6)
  dump_causal_chains.py       Causal chain dump (A+ Task 7)
main.py                     # Entry point
```

## Architecture

### Core Call Chain

`Orchestrator -> Omega (on stagnation) -> Governance (critic) -> RSISkillRegistrar -> SkillLibrary -> Agent -> Reward -> GoalGenerator -> Omega (recursive loop)`

### Self-Referential Meta-Simulation Loop

```
Orchestrator.run_round()
  for each agent:
    |-- SelfReferentialModel.encode_self_referential_state()
    |     Binds env observation + ZPD frontier + concept graph + active skills
    |     into unified 10,000-bit hypervector via XOR binding
    |-- SelfReferentialModel.detect_architectural_drift()
    |     Cosine distance on HDC state history; critical drift -> governance rollback
    |-- Agent.act_on_project()
    |     Meta-rollout predicts BOTH next env state AND agent's own policy shift
    |     Depth ceiling adapts to calibration error (BN-06)
    |
  Orchestrator.run_recursive_cycle()
    |-- validate_metric_integrity()
    |     Anti-wireheading gate: rejects self-improvement claims that lack
    |     structural correlation or external benchmark confirmation
    |     Hash-fallback path REMOVED; holdout_rate is mandatory (BN-05)
    |-- Immutable objective anchor checked (read-only HDC vector)
```

### Capability Extension

```
Orchestrator.run_recursive_cycle()
  |-- GoalGenerator.generate()           [autonomous task creation]
  |-- CompetenceMap.update()             [competence tracking]
  |-- ConceptGraph.sweep_promote_all()   [hierarchical abstraction]
  |-- TransferEngine.transfer()          [HDC structural similarity — BN-04]
  |-- SelfImprovementEngine.introspect() [empirical parameter tuning]
  |-- DifficultyScheduler.schedule()     [curriculum adjustment]
  |-- AGIProgressTracker.tick_round()    [5+3 axis measurement — BN-08]
  |-- ExternalBenchmark.run_full_benchmark() [ARC-AGI + HumanEval — BN-03/07]
  `-- CausalChainTracker.record_*()      [recursive emergence tracking — BN-08]
```

### Algorithm Synthesis Environment

```
Agent → submit_program(ProgramGenome) → VM execute on test cases → compare to oracle → reward

Level 0: Single-pass (SUM, MAX, MIN, COUNT)
Level 1: Conditional (COUNT_POSITIVE, SUM_ABOVE_THRESHOLD, CLAMP, FILTER_SUM)
Level 2: Nested loops (BUBBLE_SORT, REVERSE, UNIQUE_COUNT, INNER_PRODUCT)
Level 3: Subroutines (SORT_SUM_TOP_K, MAX_ADJACENT_SUMS, NORMALIZE)
Level 4: Meta-programs (COMPOSE_SUM_MAX, EVAL_AND_COMPARE)
Level 5: Self-referential (IMPROVE_EVOLUTION_YIELD, IMPROVE_FITNESS_DISCRIMINATION, SELF_TEST_IMPROVEMENT)

Curriculum gate: Level N requires >= 60% holdout accuracy on 2+ Level N-1 tasks.
Rewards: ONLY from holdout test case correctness (no formulas, no shaping).

Adversarial Roles:
- Challenger: generates new tasks with oracle validation
- Meta-optimizer: tunes OmegaForge hyperparameters
```

### Recursive Self-Improvement Loop (BN-08 + BN-09)

```
OmegaForge (env-coupled fitness)
  → StrictStructuralDetector → Governance critic (L0 threshold relaxation)
    → RSISkillRegistrar (quarantine + non-trivial output check)
      → SkillLibrary.register() + skill_birth event
        → GoalGenerator.on_skill_registered() → skill-derived goals
          → Agent.choose_action() consults RSI skills (30% override)
            → env.step() → actual reward logged to skill_performance_log
              → CausalChainTracker: skill→goal→achievement chain
                → (loop) new stagnation → OmegaForge evolves again
```

## Technical Details

**Self-Referential Model** (`self_referential_model.py`):
- HDC state encoding: env hash XOR competence profile XOR concept structure XOR skill set
- Recursive meta-rollout: dual simulation predicting env state AND internal policy shift
- Architectural drift detection: cosine distance on unified state history (threshold 0.35)
- Anti-wireheading: immutable objective anchor + metric integrity validation + decoupled evaluation

**HDC Memory**: Title-weighted position-bound encoding, deterministic tie-breaking, similarity threshold 0.51, max 20,000 items.

**World Model**: TD-learning with non-linear features, experience replay (200 samples), gamma 0.9, combined reward: extrinsic (0.6) + intrinsic (0.4).

**Self-Improvement**: Empirical env.step() rollouts (5 baseline + 5 modified episodes). Anti-wireheading gate rejects modifications exceeding MAX_CREDIBLE_LEAP (0.25) or lacking external benchmark correlation. Hash-fallback scoring path fully removed (BN-05).

**Adaptive Meta-Depth** (BN-06): `_meta_depth_ceiling()` computes allowable rollout depth based on calibration error over the last 8 transitions. Calibration error < 0.05 → depth 4; ≥ 0.30 → depth 1. Theorist/strategist roles receive +1 bonus.

**RSI Skill Registration** (BN-02): `RSISkillRegistrar` compiles approved OmegaForge candidates into `ProgramGenome`, quarantines via 5-input smoke-test (min 3 clean halts), wraps in `_VMSkillCallable`, and registers into `SkillLibrary`.

**Transfer Engine** (BN-04): Replaced name-similarity heuristic with ConceptGraph HDC structural vector similarity. Cross-domain transfer now operates on 10,000-bit binding vectors rather than string edit distance.

**External Benchmarks** (BN-03): `run_full_benchmark()` loads `data/arc_agi_sample.json` (20 tasks, ARC canonical format) and `data/humaneval_sample.json` (10 problems, OpenAI HumanEval format). Weighted combined score: ARC-AGI × 0.60 + HumanEval × 0.40. Replaces legacy trivial list-reversal ADB tasks.

**Recursive Emergence** (BN-08): `CausalChainTracker` records skill→goal→achievement chains. `EnvironmentCoupledFitness` generates dynamic tasks from live environment state. RSI skills consulted by agents during action selection with actual `env.step()` reward feedback. Quarantine rejects constant-output genomes.

**Emergence Metrics** (BN-08 + BN-09):
- Tool Genesis Rate: evolved skills that improved reward / total rounds
- Capability Horizon: skill-derived domains solvable (excludes initial + NOVEL_DOMAINS)
- Recursive Depth: longest causal chain in CausalChainTracker

**Capability Scoring** (geometric mean of 5 axes):
- Generalization, Autonomy, Self-Improvement, Abstraction, Open-Endedness

> **Caveat:** These are internal proxy metrics. See RESULTS.md for honest failure reporting.

## Usage

```bash
# Run all tests (core + contract + 11 integration tests)
python main.py selftest

# Run anti-cheat audit
python main.py audit

# Run cognitive engine
python main.py --rounds 40 --agents 8

# Run capability evidence with 3-way ablation
python scripts/run_agi_evidence.py

# Self-improvement verification suite
python scripts/verify_self_improvement.py

# Benchmarks
python main.py benchmark --suite ADB_v1 --seed 0 --trials 20
```

## Evidence Summary

50-round evidence run (seed=42, with anti-wireheading active, post BN-01~09 fixes):

| Configuration | Composite | Domains |
|--------------|----------|---------|
| **Full system** | **0.386** | 38 |
| Ablation A (no capability modules) | 0.004 | 6 |
| Ablation B (no GoalGenerator) | 0.028 | 6 |
| Ablation C (no TransferEngine) | 0.177 | 38 |

### External Benchmark Scores (BN-03 + BN-07)

| Benchmark | Tasks | Solved | Accuracy | Notes |
|-----------|-------|--------|----------|-------|
| ARC-AGI sample | 20 | 20 | **1.000** | Rule-based exhaustive solver (13 transforms + compositions) |
| HumanEval sample | 10 | 10 | **1.000** | Template dispatch + keyword fallback |
| **Combined (60/40)** | 30 | 30 | **1.000** | See methodology note in RESULTS.md |

> **Methodology Note:** These scores reflect simple bundled tasks, NOT the full ARC-AGI-Pub (400+ tasks) or HumanEval (164 problems) benchmarks. The ARC solver uses exhaustive rule search on 13 geometric/value transforms. The HumanEval solver uses function-name dispatch. See RESULTS.md §7 for full details.

Self-improvement score reflects governance-gated scoring with mandatory holdout metrics (hash-fallback removed). See [RESULTS.md](RESULTS.md) for full report.

## Bottleneck Fixes (BN-01 ~ BN-09)

| ID | Fix | Status |
|----|-----|--------|
| BN-01 | WorldModel tiny-transformer rewrite | ✅ Complete |
| BN-02 | OmegaForge → SkillLibrary RSI pipeline | ✅ Complete |
| BN-03 | ARC-AGI + HumanEval real benchmark datasets | ✅ Complete |
| BN-04 | TransferEngine HDC structural similarity | ✅ Complete |
| BN-05 | Governance hash-fallback removed | ✅ Complete |
| BN-06 | Adaptive meta-depth ceiling (calibration-based) | ✅ Complete |
| BN-07 | Wire real ARC + HumanEval solvers to benchmark harness | ✅ Complete |
| BN-08 | Recursive emergent self-improvement loop (CausalChainTracker, EnvironmentCoupledFitness, skill→goal feedback) | ✅ Complete |
| BN-09 | Complete recursive loop plumbing (env fitness wiring, reward feedback, governance flow, goal tracking) | ✅ Complete |
| BN-10 | Real recursive self-improvement (env alignment, detector relaxation, skill-env coupling, Stage1 strategies) | ✅ Complete |

## Scope & Limitations

- External benchmark scores of 1.000 reflect 20 simple bundled ARC tasks and 10 basic HumanEval problems — NOT the full public benchmarks
- Level 4 meta-programs are extremely unlikely to emerge in short runs — the curriculum gate requires solving Level 0-3 first
- L0 algorithmic tasks (SUM/MAX/MIN/COUNT) remain unsolved by evolved VM programs in most seeds — the gap between random mutation search and exact holdout correctness (1e-6 tolerance) is substantial
- Recursive emergence (BN-08/09/10) is stochastic — skill births depend on OmegaForge producing structurally valid, non-constant-output programs
- Concept graph depth (5) partially driven by threshold calibration
- All environments simulated; no real-world grounding
- ~~TransferEngine stubs~~ (FIXED: measure_transfer_success returns real deltas, rollback removes concepts)
- ~~ConceptGraph.get_vector() missing~~ (FIXED: generates HDC vectors from actual concept nodes)
- ~~Hardcoded solve_fn mappings~~ (FIXED: deprecated, separated from AGI scoring)

## License

MIT License
