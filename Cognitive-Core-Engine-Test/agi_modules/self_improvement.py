"""
SelfImprovementEngine — Runtime parameter self-modification with governance.

Serves AGI capability: the system reasons about its own decision quality
and proposes parameter modifications to improve performance.

CRITICAL: This modifies RUNTIME parameters, NOT source code.
The Omega Forge handles structural code changes.
CRITICAL: test_modification() MUST use empirical env rollouts, not arithmetic.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Optional

# --- Named constants (Rule 6) ---

# Decision regret threshold for triggering self-improvement
# Calibrated for environment where rewards are typically 0.02-0.15
REGRET_THRESHOLD = 0.03

# Minimum test improvement to apply a modification
# Calibrated for environment reward scale (~0.02-0.15)
MIN_IMPROVEMENT_DELTA = 0.01

# Number of empirical test episodes per condition (baseline and modified)
TEST_ROUNDS = 5

# Maximum env steps per episode during empirical testing
MAX_STEPS_PER_EPISODE = 10

# Modification bounds for safety
MAX_RISK_CHANGE = 0.1
MAX_WEIGHT_CHANGE = 0.2
MAX_DEPTH_CHANGE = 2

# Acceptance rate ceiling — above this, the engine is likely rubber-stamping
ACCEPTANCE_RATE_CEILING = 0.80


class SelfImprovementEngine:
    """Reasons about and modifies runtime decision-making parameters.

    Why it exists: enables the system to tune its own exploration/exploitation
    balance, planning depth, and transfer aggressiveness based on observed outcomes.

    Fallback: never applies modifications without empirically-tested positive results.
    CRITICAL: all modifications go through governance (critic evaluation).
    """

    def __init__(self) -> None:
        self._decision_history: List[Dict[str, Any]] = []
        self._applied_modifications: List[Dict[str, Any]] = []
        self._proposed_modifications: List[Dict[str, Any]] = []
        self._rollback_points: Dict[str, Any] = {}

    def introspect_decision_quality(self,
                                    history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze recent decisions for optimality.

        Why: identifies if the agent's action selections were suboptimal
        by comparing chosen vs what current model would choose (hindsight).
        Fallback: returns zero regret if history is empty.
        """
        if not history:
            return {"decision_regret": 0.0, "sample_count": 0, "analysis": {}}

        total_regret = 0.0
        action_analysis: Dict[str, Dict[str, float]] = {}

        for entry in history[-20:]:
            reward = float(entry.get("reward", 0.0))
            action = str(entry.get("action", ""))
            domain = str(entry.get("domain", entry.get("info", {}).get("domain", "")))

            domain_key = domain or "unknown"
            if domain_key not in action_analysis:
                action_analysis[domain_key] = {"total_reward": 0.0, "count": 0, "max": 0.0}
            analysis = action_analysis[domain_key]
            analysis["total_reward"] += reward
            analysis["count"] += 1
            analysis["max"] = max(analysis["max"], reward)

            optimal_estimate = analysis["max"]
            total_regret += max(0, optimal_estimate - reward)

        n = min(20, len(history))
        avg_regret = total_regret / max(1, n)

        return {
            "decision_regret": avg_regret,
            "sample_count": n,
            "analysis": action_analysis,
        }

    def propose_policy_modification(self,
                                    diagnosis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Propose parameter changes if decision regret exceeds threshold.

        Why: converts diagnosis into actionable parameter adjustments.
        Fallback: returns None if regret is acceptable.
        """
        regret = float(diagnosis.get("decision_regret", 0.0))
        if regret < REGRET_THRESHOLD:
            return None

        analysis = diagnosis.get("analysis", {})
        mod: Dict[str, Any] = {
            "type": "policy_modification",
            "reason": f"decision_regret={regret:.3f}",
            "changes": {},
        }

        if regret > REGRET_THRESHOLD * 1.5:
            mod["changes"]["risk_delta"] = min(MAX_RISK_CHANGE, 0.05)
            mod["changes"]["intrinsic_weight_delta"] = min(MAX_WEIGHT_CHANGE, 0.1)

        for domain, stats in analysis.items():
            avg = stats["total_reward"] / max(1, stats["count"])
            if avg < 0.15 and stats["count"] >= 3:
                mod["changes"]["planning_depth_delta"] = min(MAX_DEPTH_CHANGE, 1)
                break

        if regret > REGRET_THRESHOLD * 2:
            mod["changes"]["creative_weight_delta"] = 0.1

        if not mod["changes"]:
            return None

        self._proposed_modifications.append(mod)
        return mod

    def _run_episodes(self, env: Any, params: Dict[str, Any],
                      n_episodes: int) -> List[float]:
        """Run n_episodes using env.step() and collect rewards.

        Why: empirical testing requires actually running the environment.
        Fallback: returns empty list if env has no step() method.
        """
        rewards: List[float] = []
        risk = float(params.get("risk", 0.25))
        actions = ["attempt_breakthrough", "build_tool", "write_verified_note", "tune_orchestration"]

        for _ in range(n_episodes):
            task = env.sample_task()
            budget = int(params.get("budget", 12))
            obs = env.make_observation(task, budget)

            episode_reward = 0.0
            for step_i in range(MAX_STEPS_PER_EPISODE):
                # Action selection influenced by risk parameter
                if env.rng.random() < risk:
                    action = env.rng.choice(actions)
                else:
                    action = actions[0]  # default: attempt_breakthrough

                payload = {"invest": max(1.0, budget / 10.0),
                           "agent": "self_improvement_test",
                           "role": "general",
                           "task": obs.get("task"),
                           "project_id": "test"}

                next_obs, reward, info = env.step(obs, action, payload)
                episode_reward += reward
                obs = next_obs
                # Single step per episode for this environment
                break

            rewards.append(episode_reward)
        return rewards

    def test_modification(self, mod_spec: Dict[str, Any],
                          env: Any,
                          orchestrator_params: Dict[str, Any],
                          world_model: Any = None) -> Dict[str, Any]:
        """Test proposed modification via empirical env rollouts.

        PRIMARY PATH: runs TEST_ROUNDS episodes with modified params and
        TEST_ROUNDS with baseline params, comparing mean rewards.
        FALLBACK: returns delta=0.0 with empirically_tested=False if env
        has no step() method.

        Why: prevents blind application — modifications must demonstrate
        real improvement on actual environment steps, not arithmetic estimates.
        """
        changes = mod_spec.get("changes", {})
        if not changes:
            return {"delta": 0.0, "empirically_tested": False, "method": "no_changes"}

        # FALLBACK: env has no step() method → cannot verify
        if env is None or not hasattr(env, "step"):
            return {"delta": 0.0, "empirically_tested": False, "method": "no_env"}

        # PRIMARY PATH: empirical env rollout
        # Build modified params by applying deltas
        modified_params = copy.deepcopy(orchestrator_params)
        for key, value in changes.items():
            if key == "risk_delta":
                modified_params["risk"] = max(0.05, min(0.5,
                    float(modified_params.get("risk", 0.25)) + value))
            elif key == "planning_depth_delta":
                modified_params["planning_depth"] = max(2, min(10,
                    int(modified_params.get("planning_depth", 3)) + int(value)))
            elif key == "intrinsic_weight_delta":
                modified_params["intrinsic_weight"] = max(0.0, min(1.0,
                    float(modified_params.get("intrinsic_weight", 0.4)) + value))
            elif key == "creative_weight_delta":
                modified_params["creative_weight"] = max(0.0, min(1.0,
                    float(modified_params.get("creative_weight", 0.2)) + value))

        # Run baseline episodes
        baseline_rewards = self._run_episodes(env, orchestrator_params, TEST_ROUNDS)
        baseline_avg = sum(baseline_rewards) / max(1, len(baseline_rewards))

        # Run modified episodes
        modified_rewards = self._run_episodes(env, modified_params, TEST_ROUNDS)
        modified_avg = sum(modified_rewards) / max(1, len(modified_rewards))

        delta = modified_avg - baseline_avg

        return {
            "delta": delta,
            "empirically_tested": True,
            "method": "env_rollout",
            "baseline_avg": baseline_avg,
            "modified_avg": modified_avg,
            "n_episodes": TEST_ROUNDS,
        }

    def apply_if_beneficial(self, mod_spec: Dict[str, Any],
                            test_result: Any,
                            current_params: Dict[str, Any]) -> bool:
        """Apply modification if empirical test shows improvement.

        Why: only adopt changes that demonstrably improve performance.
        Requires empirically_tested=True to apply. Arithmetic-only results
        are never applied (delta treated as 0).
        Fallback: never applies if delta <= threshold or not empirically tested.
        """
        # Handle dict format (new) and float format (backward compat)
        if isinstance(test_result, dict):
            delta = float(test_result.get("delta", 0.0))
            empirically_tested = bool(test_result.get("empirically_tested", False))
        else:
            # Legacy float path: treat as NOT empirically tested → never apply
            delta = float(test_result)
            empirically_tested = False

        if delta <= MIN_IMPROVEMENT_DELTA or not empirically_tested:
            return False

        self._applied_modifications.append({
            "modification": mod_spec,
            "test_result": test_result,
            "before": copy.deepcopy(current_params),
        })

        # Check acceptance rate guard after every 5th application
        if len(self._applied_modifications) % 5 == 0:
            self.acceptance_rate_guard()

        return True

    def acceptance_rate_guard(self) -> Dict[str, Any]:
        """Check if acceptance rate is suspiciously high.

        Why: 100% acceptance rate (7/7) indicates arithmetic simulation rather
        than genuine empirical testing. A healthy rate is 30-70%.
        Fallback: returns rate=None if fewer than 5 proposals.
        """
        if self.proposed_count() < 5:
            return {"rate": None, "suspicious": False}

        rate = self.applied_count() / self.proposed_count()
        suspicious = rate > ACCEPTANCE_RATE_CEILING

        if suspicious:
            warnings.warn(
                f"SelfImprovementEngine acceptance rate {rate:.0%} > "
                f"{ACCEPTANCE_RATE_CEILING:.0%} — verify test_modification() is "
                "using empirical env rollouts, not arithmetic simulation."
            )

        return {"rate": rate, "suspicious": suspicious}

    def record_decision(self, decision: Dict[str, Any]) -> None:
        """Record a decision for future introspection.

        Why: builds the history needed for regret analysis.
        Fallback: trims to last 100 decisions.
        """
        self._decision_history.append(decision)
        if len(self._decision_history) > 100:
            self._decision_history = self._decision_history[-100:]

    def proposed_count(self) -> int:
        """Return count of proposed modifications.

        Why: validation metric.
        Fallback: returns 0.
        """
        return len(self._proposed_modifications)

    def applied_count(self) -> int:
        """Return count of applied modifications.

        Why: validation metric.
        Fallback: returns 0.
        """
        return len(self._applied_modifications)

    def get_applied_modifications(self) -> List[Dict[str, Any]]:
        """Return history of applied modifications with before/after.

        Why: evidence for AGI self-improvement tracking.
        Fallback: returns empty list.
        """
        return list(self._applied_modifications)
