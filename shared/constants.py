"""Invariant constants for OMEGA-THDSE (PLAN.md Section D).

These values are extracted from the existing codebase and are IMMUTABLE.
Changing any of them will break existing tests, alter system behavior,
or violate architectural invariants. Do NOT round, simplify, or modify
any value in this file.
"""

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
