"""Phase 8B.1 — Problem specification + FHRR encoding.

A :class:`ProblemSpec` defines a synthesis target through concrete
``(input, expected_output)`` examples — the same format the
behavioural encoder uses for runtime profiling. :class:`ProblemEncoder`
converts a spec into a 256-dim FHRR vector via the EXACT SAME
hash-bind-bundle pipeline as :class:`BehavioralEncoder.encode_io_pairs`,
so that a program whose behaviour matches the problem oracle produces
a behavioural vector with high correlation to the problem vector.

PLAN.md Rule 19 (NO FAKE BENCHMARKS): every spec must have at least
10 io_examples and every example must be a deterministic
(input, expected_output) pair where the oracle is unambiguous.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Allow importing the encoder primitives without depending on cwd.
_THDSE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _THDSE_ROOT not in sys.path:
    sys.path.insert(0, _THDSE_ROOT)

from src.projection.behavioral_encoder import BehavioralEncoder  # noqa: E402


# --------------------------------------------------------------------------- #
# ProblemSpec
# --------------------------------------------------------------------------- #


@dataclass
class ProblemSpec:
    """A synthesis problem defined by input → expected output examples.

    Rule 19: ``io_examples`` must be a non-empty list of pairs whose
    second element is a deterministic, comparable expected value. The
    oracle is the equality check ``actual == expected`` — programs
    are only counted as solving the problem when every example
    passes.
    """

    name: str
    io_examples: List[Tuple[Any, Any]]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ProblemSpec.name must be a non-empty string")
        if not isinstance(self.io_examples, list) or not self.io_examples:
            raise ValueError(
                "ProblemSpec.io_examples must be a non-empty list of pairs"
            )
        for idx, pair in enumerate(self.io_examples):
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(
                    f"io_examples[{idx}] must be a (input, output) tuple, "
                    f"got {pair!r}"
                )

    @property
    def example_count(self) -> int:
        return len(self.io_examples)


# --------------------------------------------------------------------------- #
# ProblemVector — output of ProblemEncoder.encode_problem
# --------------------------------------------------------------------------- #


@dataclass
class ProblemVector:
    """Encoded FHRR view of a :class:`ProblemSpec`."""

    handle: int
    phases: List[float]
    spec_name: str
    example_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# ProblemEncoder
# --------------------------------------------------------------------------- #


class ProblemEncoder:
    """Encode a :class:`ProblemSpec` into a 256-dim FHRR vector.

    Internally constructs a :class:`BehavioralEncoder` and routes
    through its ``encode_io_pairs`` method, guaranteeing that
    behavioural profiles produced by ``encode_behavior`` (with
    matching probe inputs) are directly correlatable against the
    problem vector.
    """

    def __init__(self, arena: Any, dimension: int = 256):
        if arena is None:
            raise ValueError("ProblemEncoder requires an arena")
        self._arena = arena
        self._dimension = int(dimension)
        # The encoder created here is reused across encode_problem
        # calls so all problems hash through the same parameters.
        self._encoder = BehavioralEncoder(
            arena=arena,
            dimension=int(dimension),
            n_probes=1,  # not used by encode_io_pairs
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_problem(self, spec: ProblemSpec) -> ProblemVector:
        """Hash-bind-bundle the problem's io_examples into an FHRR vector."""
        if not isinstance(spec, ProblemSpec):
            raise TypeError(
                f"spec must be a ProblemSpec, got {type(spec).__name__}"
            )
        # Wrap the expected outputs the same way encode_behavior wraps
        # actual outputs so the encoded representations are directly
        # comparable.
        wrapped_pairs = [
            (input_val, ("__return__", expected))
            for input_val, expected in spec.io_examples
        ]
        profile = self._encoder.encode_io_pairs(wrapped_pairs)
        return ProblemVector(
            handle=profile.behavioral_handle,
            phases=list(profile.behavioral_phases),
            spec_name=spec.name,
            example_count=spec.example_count,
            metadata={
                "encoded_at": time.time(),
                "provenance": {
                    "operation": "encode_problem",
                    "source_arena": "thdse",
                    "spec_name": spec.name,
                    "example_count": spec.example_count,
                },
            },
        )


# --------------------------------------------------------------------------- #
# Oracle scoring — used by benchmark scripts
# --------------------------------------------------------------------------- #


def score_against_problem(
    func: Callable[[Any], Any], spec: ProblemSpec
) -> Dict[str, Any]:
    """Run ``func`` against every io_example and return a score dict.

    The score is the fraction of io_examples that produce
    ``func(input) == expected``. Programs that crash on an input get
    that example marked as failed (a crash is wrong-output by
    definition).
    """
    passed: List[Tuple[Any, Any]] = []
    failed: List[Tuple[Any, Any, Any]] = []
    for input_val, expected in spec.io_examples:
        try:
            actual = func(input_val)
        except Exception as exc:  # noqa: BLE001
            failed.append((input_val, expected, f"<crash:{type(exc).__name__}>"))
            continue
        if actual == expected:
            passed.append((input_val, expected))
        else:
            failed.append((input_val, expected, actual))
    total = max(spec.example_count, 1)
    return {
        "spec_name": spec.name,
        "passed": len(passed),
        "failed": len(failed),
        "total": total,
        "pass_rate": len(passed) / total,
        "passed_ios": passed,
        "failed_ios": failed,
        "metadata": {
            "provenance": {
                "operation": "score_against_problem",
                "source_arena": "thdse",
                "spec_name": spec.name,
                "pass_rate": len(passed) / total,
            },
        },
    }


__all__ = [
    "ProblemEncoder",
    "ProblemSpec",
    "ProblemVector",
    "score_against_problem",
]
