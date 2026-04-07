"""
SelfModel — Agent self-awareness through capability tracking and failure diagnosis.

Serves AGI capability: enables meta-cognitive reasoning about own capabilities,
allowing the agent to predict performance and diagnose failures.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# --- Named constants (Rule 6) ---

# Calibration window: number of recent tasks for calibration error
CALIBRATION_WINDOW = 20

# Confidence growth rate: how many samples needed for high confidence
CONFIDENCE_GROWTH_RATE = 10

# Minimum predicted performance to attempt a task
MIN_PREDICTED_PERFORMANCE = 0.1

# Learning rate plateau threshold
LEARNING_RATE_PLATEAU_THRESHOLD = 0.01


class FailureType(Enum):
    """Classification of failure modes for diagnostic analysis."""
    EXPLORATION = "exploration"      # Tried wrong actions
    KNOWLEDGE = "knowledge"          # Lacked right concepts
    PLANNING = "planning"            # Had concepts but wrong plan
    TRANSFER = "transfer"            # Applied wrong analogy
    DIFFICULTY = "difficulty"        # Task too hard for current level
    UNKNOWN = "unknown"


class SelfModel:
    """Tracks agent capabilities and provides meta-cognitive functions.

    Why it exists: without self-awareness, the agent cannot predict its own
    performance, skip impossible tasks, or diagnose failure patterns.

    Fallback: returns conservative defaults when insufficient data.
    """

    def __init__(self) -> None:
        self._performance_history: Dict[str, List[float]] = {}
        self._predictions: List[float] = []
        self._actuals: List[float] = []
        self._failure_patterns: Dict[str, Dict[str, int]] = {}
        self._learning_rates: Dict[str, float] = {}
        self._resource_usage: List[Dict[str, Any]] = []
        self._tasks_skipped: int = 0

    def update(self, round_result: Dict[str, Any]) -> None:
        """Update all tracked metrics from a round result.

        Why: maintains running statistics for prediction and diagnosis.
        Fallback: silently skips missing fields.
        """
        domain = str(round_result.get("domain", round_result.get("info", {}).get("domain", "")))
        reward = float(round_result.get("reward", 0.0))
        action = str(round_result.get("action", ""))

        if domain:
            if domain not in self._performance_history:
                self._performance_history[domain] = []
            history = self._performance_history[domain]
            history.append(reward)

            # Track learning rate
            if len(history) >= 5:
                recent = history[-5:]
                older = history[-10:-5] if len(history) >= 10 else history[:5]
                self._learning_rates[domain] = (
                    sum(recent) / len(recent) - sum(older) / len(older)
                )

            # Trim history to prevent unbounded growth
            if len(history) > 100:
                self._performance_history[domain] = history[-100:]

    def predict_performance(self, task: Any) -> Tuple[float, float]:
        """Predict reward and confidence for a task.

        Why: enables informed task selection and skip decisions.
        Fallback: returns (0.3, 0.1) for unknown domains.
        """
        domain = str(getattr(task, 'domain', ''))
        difficulty = int(getattr(task, 'difficulty', 3))

        history = self._performance_history.get(domain, [])
        if not history:
            return 0.3, 0.1  # Low confidence default

        avg = sum(history) / len(history)

        # Adjust for difficulty
        difficulty_factor = 1.0 - (difficulty - 3) * 0.05
        predicted = max(0.0, min(1.0, avg * difficulty_factor))

        # Confidence increases with sample count
        confidence = 1.0 - (1.0 / (1.0 + len(history) / CONFIDENCE_GROWTH_RATE))
        confidence = min(1.0, max(0.0, confidence))

        # Record for calibration tracking
        self._predictions.append(predicted)

        return predicted, confidence

    def record_actual(self, reward: float) -> None:
        """Record actual reward for calibration tracking.

        Why: enables calibration error measurement.
        Fallback: no-op.
        """
        self._actuals.append(reward)
        # Trim
        if len(self._actuals) > 200:
            self._actuals = self._actuals[-200:]
            self._predictions = self._predictions[-200:]

    def should_attempt(self, task: Any) -> Tuple[bool, str]:
        """Decide whether to attempt a task.

        Why: self-aware agents skip impossible tasks to focus resources.
        Fallback: always attempts if insufficient data.
        """
        predicted, confidence = self.predict_performance(task)
        domain = str(getattr(task, 'domain', ''))
        baseline = float(getattr(task, 'baseline', 0.3))

        # Not enough data — attempt anyway
        if confidence < 0.3:
            return True, "insufficient_data_for_skip"

        # Too hard: predicted performance well below baseline
        if predicted < MIN_PREDICTED_PERFORMANCE and confidence > 0.5:
            self._tasks_skipped += 1
            return False, f"predicted_reward={predicted:.2f}_below_threshold"

        # Check learning rate plateau
        lr = self._learning_rates.get(domain, 0.0)
        history = self._performance_history.get(domain, [])
        if (abs(lr) < LEARNING_RATE_PLATEAU_THRESHOLD and
                len(history) > 20 and predicted < baseline):
            self._tasks_skipped += 1
            return False, f"plateaued_in_{domain}_lr={lr:.4f}"

        return True, "within_capabilities"

    def diagnose_failure(self, task: Any,
                         result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze why a task failed.

        Why: failure diagnosis drives targeted goal generation for remediation.
        Fallback: returns UNKNOWN failure type if diagnosis inconclusive.
        """
        domain = str(getattr(task, 'domain', ''))
        reward = float(result.get("reward", 0.0))
        action = str(result.get("action", ""))

        diagnosis: Dict[str, Any] = {
            "failure_type": FailureType.UNKNOWN.value,
            "domain": domain,
            "reward": reward,
            "suggested_remedy": "",
        }

        history = self._performance_history.get(domain, [])

        # Exploration failure: tried wrong actions (low action diversity)
        if domain not in self._failure_patterns:
            self._failure_patterns[domain] = {}
        pattern = self._failure_patterns[domain]
        pattern[action] = pattern.get(action, 0) + 1

        total_failures = sum(pattern.values())
        unique_actions = len(pattern)

        if total_failures > 3 and unique_actions <= 2:
            diagnosis["failure_type"] = FailureType.EXPLORATION.value
            diagnosis["suggested_remedy"] = "increase_exploration_rate"
        elif len(history) > 10 and sum(history[-10:]) / 10 < 0.15:
            diagnosis["failure_type"] = FailureType.KNOWLEDGE.value
            diagnosis["suggested_remedy"] = "acquire_new_concepts"
        elif len(history) > 5 and max(history[-5:]) > 0.4 and reward < 0.15:
            diagnosis["failure_type"] = FailureType.PLANNING.value
            diagnosis["suggested_remedy"] = "improve_planning_depth"
        elif result.get("info", {}).get("transfer_attempted"):
            diagnosis["failure_type"] = FailureType.TRANSFER.value
            diagnosis["suggested_remedy"] = "rollback_transfer"
        else:
            difficulty = int(getattr(task, 'difficulty', 3))
            if difficulty > 6 and reward < 0.1:
                diagnosis["failure_type"] = FailureType.DIFFICULTY.value
                diagnosis["suggested_remedy"] = "reduce_difficulty"
            else:
                diagnosis["suggested_remedy"] = "general_exploration"

        return diagnosis

    def calibration_error(self) -> float:
        """Measure prediction accuracy: |mean(predicted) - mean(actual)|.

        Why: good self-model has calibration_error < 0.1.
        Fallback: returns 1.0 if insufficient data.
        """
        n = min(len(self._predictions), len(self._actuals), CALIBRATION_WINDOW)
        if n < 2:
            return 1.0

        pred = self._predictions[-n:]
        actual = self._actuals[-n:]

        mean_pred = sum(pred) / n
        mean_actual = sum(actual) / n

        return abs(mean_pred - mean_actual)

    def tasks_skipped(self) -> int:
        """Return count of tasks skipped due to self-awareness.

        Why: evidence of meta-cognitive capability.
        Fallback: returns 0.
        """
        return self._tasks_skipped

    def get_learning_rates(self) -> Dict[str, float]:
        """Return learning rate estimates per domain.

        Why: needed by difficulty scheduler and diagnostics.
        Fallback: returns empty dict.
        """
        return dict(self._learning_rates)
