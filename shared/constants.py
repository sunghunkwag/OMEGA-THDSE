"""Invariant constants for OMEGA-THDSE (PLAN.md Section D).

These values are extracted from the existing codebase and are IMMUTABLE.
Changing any of them will break existing tests, alter system behavior,
or violate architectural invariants. Do NOT round, simplify, or modify
any value in this file.
"""

# === Arena Dimensions ===
CCE_ARENA_DIM = 10_000
CCE_ARENA_CAP = 100_000
CCE_ARENA_CAP_SMALL = 1_000    # safe for memory-limited envs

THDSE_ARENA_DIM = 256
THDSE_ARENA_CAP = 2_000_000
THDSE_ARENA_CAP_SMALL = 10_000

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

# === Phase 9: Semantic Grounding (D1, D2) ===
SEMANTIC_SIMILAR_THRESHOLD = 0.6        # Minimum cosine for a "similar" pair (Rule 14)
SEMANTIC_DISSIMILAR_THRESHOLD = 0.3     # Maximum cosine for a "dissimilar" pair (Rule 14)
SEMANTIC_ENCODER_DIM = 384              # Semantic embedding dimension (matches MiniLM)

# === Phase 10: Continuous Learning (D3) ===
ONLINE_LEARNER_HIDDEN_DIMS = [256, 128, 64]
ONLINE_LEARNER_DEFAULT_LR = 0.001       # Default Adam-ish learning rate
EXPERIENCE_REPLAY_CAPACITY = 10_000     # Replay buffer size
MIN_LOSS_DECREASE_RATIO = 0.8           # loss_after must be <= this * loss_before (Rule 13)

# === Phase 11: Deep Memory (D6) ===
EPISODIC_MEMORY_CAPACITY = 10_000
EPISODIC_CONSOLIDATION_THRESHOLD = 5    # Rehearsals before episodic → semantic consolidation
SEMANTIC_MEMORY_CAPACITY = 50_000
PROCEDURAL_MEMORY_CAPACITY = 5_000
MEMORY_TOP1_ACCURACY_MIN = 0.8          # Top-1 retrieval accuracy (Rule 18)

# === Phase 12: Multi-Step Reasoning (D5, D7) ===
REASONING_DEFAULT_DEPTH = 5             # Beyond the Phase 4 depth-2 cap
REASONING_MAX_DEPTH = 10
REASONING_BEAM_WIDTH = 4
REASONING_BACKTRACK_PATIENCE = 2        # Plateau steps before backtrack
ANALOGY_SIMILARITY_MIN = 0.5            # Analogy transfer threshold

# === Phase 13: Environment Interaction (D8) ===
AGENT_LOOP_MAX_STEPS = 1000
AGENT_CONSOLIDATION_INTERVAL = 25       # Memory consolidation cadence
ENV_ACTION_DIVERSITY_MIN = 0.4          # Non-dummy environment diversity (Rule 16)

# === Phase 14: Synthesis Breakthrough (D4) ===
ENHANCED_BEAM_WIDTH = 20
DECOMPOSITION_MAX_SUBTASKS = 8
SYNTHESIS_BENCHMARK_TARGET = 4          # Target: solve >= 4/5 benchmark problems
