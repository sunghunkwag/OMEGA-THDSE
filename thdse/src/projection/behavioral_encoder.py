"""Phase 8A.1 — Behavioral encoder for input/output FHRR vectors.

The structural projector encodes a program by its AST shape. Two
programs that DO the same thing but LOOK different (e.g.,
``for i in range(n): total += arr[i]`` vs ``return sum(arr)``) end
up with totally unrelated structural vectors and never resonate. The
behavioral encoder fixes this by encoding what a program actually
DOES — its observable input → output mapping — as an FHRR vector.

Pipeline (PLAN.md Phase 8A.1):

1. Generate ``n_probes`` deterministic test inputs via
   :class:`shared.deterministic_rng.DeterministicRNG` (Rule 10 — no
   bare ``random``).
2. Execute the program in :class:`src.execution.sandbox.ExecutionSandbox`
   against every probe. Capture the return value, the type-error
   rejection signal, or the raised exception type.
3. Hash each ``(input, output)`` pair into a deterministic phase
   vector via BLAKE2b. Allocate two arena handles per pair, bind
   them, then bundle every bound handle into a single behavioral
   handle.
4. Return a :class:`BehavioralProfile` carrying the handle, the raw
   phases, the io pairs, an execution summary, and a Rule-9
   provenance metadata dict.

Rule 18 (NO FAKE BEHAVIOR): the encoder MUST execute the program in a
real sandbox. AST-based shortcuts are forbidden — the whole point is
that behavior is defined by execution, not syntax. Tests verify that
``def f(x): return x+1`` and ``def f(x): return 1+x`` produce
behavioral vectors with similarity > 0.95, while ``x+1`` vs ``x*2``
produce similarity < 0.3.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allow ``from shared.deterministic_rng import …`` regardless of cwd.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.deterministic_rng import DeterministicRNG  # noqa: E402

from src.execution.sandbox import ExecutionSandbox  # noqa: E402


_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Default probe inputs — deterministic, type-diverse
# --------------------------------------------------------------------------- #


def _build_default_probes(n_probes: int) -> List[Tuple[str, Any]]:
    """Construct ``n_probes`` deterministic input probes via DRNG.

    The probes cycle through the most useful Python data shapes:
    ints, floats, lists of ints, strings, nested lists, booleans,
    dicts, and tuples. The cycling guarantees broad coverage even
    for low ``n_probes`` (the default of 20 covers each type at
    least twice).
    """
    rng = DeterministicRNG(master_seed=42).fork("behavioral_probes")
    probes: List[Tuple[str, Any]] = [
        ("int_zero", 0),
        ("int_one", 1),
        ("int_neg", -3),
    ]
    if n_probes <= len(probes):
        return probes[:n_probes]

    type_cycle = (
        "int", "float", "list", "string", "nested",
        "bool", "dict", "tuple",
    )
    while len(probes) < n_probes:
        idx = len(probes) - 3
        kind = type_cycle[idx % len(type_cycle)]
        if kind == "int":
            value = int(rng.integers(-100, 100))
            probes.append((f"probe_int_{idx}", value))
        elif kind == "float":
            value = float(rng.uniform(-10.0, 10.0))
            probes.append((f"probe_float_{idx}", round(value, 4)))
        elif kind == "list":
            length = int(rng.integers(0, 8))
            value = [int(rng.integers(-50, 50)) for _ in range(length)]
            probes.append((f"probe_list_{idx}", value))
        elif kind == "string":
            length = int(rng.integers(0, 6))
            chars = "abcdefghij"
            value = "".join(
                chars[int(rng.integers(0, len(chars)))] for _ in range(length)
            )
            probes.append((f"probe_str_{idx}", value))
        elif kind == "nested":
            outer = int(rng.integers(0, 4))
            value = [
                [int(rng.integers(0, 10)) for _ in range(2)]
                for _ in range(outer)
            ]
            probes.append((f"probe_nested_{idx}", value))
        elif kind == "bool":
            value = bool(int(rng.integers(0, 2)))
            probes.append((f"probe_bool_{idx}", value))
        elif kind == "dict":
            value = {
                "k": int(rng.integers(0, 100)),
                "v": int(rng.integers(0, 100)),
            }
            probes.append((f"probe_dict_{idx}", value))
        elif kind == "tuple":
            length = int(rng.integers(2, 4))
            value = [int(rng.integers(0, 50)) for _ in range(length)]
            probes.append((f"probe_tuple_{idx}", value))
    return probes


DEFAULT_PROBE_INPUTS: List[Tuple[str, Any]] = _build_default_probes(20)


# --------------------------------------------------------------------------- #
# Hashing primitives
# --------------------------------------------------------------------------- #


def _serialize_value(value: Any) -> str:
    """Stable string serialization for probe values + outputs.

    The encoding must produce identical bytes for identical values
    across runs and across different programs that return equal
    objects. We avoid Python's built-in ``hash()`` because it is
    randomized for strings under the default PYTHONHASHSEED.
    """
    try:
        import json

        return json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return repr(value)


def hash_to_phases(value: Any, dimension: int, *, tag: str = "") -> List[float]:
    """Deterministic ``dimension``-length phase vector derived from ``value``.

    BLAKE2b chunks (4 bytes each) are stretched to fill the requested
    dimension. The optional ``tag`` lets callers separate "input"
    encodings from "output" encodings even when they hash the same
    underlying value.
    """
    serialized = f"{tag}:{_serialize_value(value)}"
    phases: List[float] = []
    counter = 0
    while len(phases) < dimension:
        chunk = hashlib.blake2b(
            f"{serialized}|{counter}".encode("utf-8"),
            digest_size=64,
        ).digest()
        for i in range(0, len(chunk), 4):
            if len(phases) >= dimension:
                break
            word = int.from_bytes(chunk[i:i + 4], "big")
            phases.append((word / 4294967296.0) * _TWO_PI)
        counter += 1
    return phases


# --------------------------------------------------------------------------- #
# BehavioralProfile dataclass
# --------------------------------------------------------------------------- #


@dataclass
class BehavioralProfile:
    """Result of encoding a program's input/output behavior.

    Attributes mirror the FHRR layer (``handle``, ``phases``) plus a
    debugging-friendly view of the io pairs and the execution summary.
    The :attr:`metadata` dict carries the Rule 9 provenance entry so
    downstream bridges can trace where a profile came from.
    """

    behavioral_handle: int
    behavioral_phases: List[float]
    io_pairs: List[Tuple[Any, Any]]
    execution_profile: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def handle(self) -> int:
        """Alias for :attr:`behavioral_handle` for parity with structural code."""
        return self.behavioral_handle


# --------------------------------------------------------------------------- #
# BehavioralEncoder
# --------------------------------------------------------------------------- #


class BehavioralEncoder:
    """Encode programs as input→output FHRR vectors.

    Construct one encoder per arena. The constructor freezes the probe
    set so every program ingested through this encoder is exercised on
    the same inputs (this is what makes behavioral similarity
    meaningful). When ``probe_inputs`` is omitted the default
    type-diverse probe set is used.
    """

    def __init__(
        self,
        arena: Any,
        dimension: int = 256,
        n_probes: int = 20,
        probe_inputs: Optional[Sequence[Tuple[str, Any]]] = None,
        sandbox: Optional[ExecutionSandbox] = None,
    ):
        if arena is None:
            raise ValueError("BehavioralEncoder requires an arena")
        self._arena = arena
        self._dimension = int(dimension)
        if probe_inputs is None:
            probe_inputs = _build_default_probes(int(n_probes))
        self._probe_inputs = list(probe_inputs)
        # The sandbox is created lazily — every encode_behavior call
        # passes the encoder's frozen probe set as the sandbox test
        # vectors so executions are reproducible.
        self._sandbox = sandbox or ExecutionSandbox(
            test_vectors=list(self._probe_inputs)
        )
        self._encode_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def probe_inputs(self) -> List[Tuple[str, Any]]:
        return list(self._probe_inputs)

    @property
    def encode_count(self) -> int:
        return self._encode_count

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def _bind_io_handles(
        self, io_pairs: Sequence[Tuple[Any, Any]]
    ) -> List[int]:
        """Allocate one bound handle per (input, output) pair."""
        handles: List[int] = []
        for input_val, output_val in io_pairs:
            input_phases = hash_to_phases(
                input_val, self._dimension, tag="input"
            )
            output_phases = hash_to_phases(
                output_val, self._dimension, tag="output"
            )
            h_in = self._arena.allocate()
            self._arena.inject_phases(h_in, input_phases)
            h_out = self._arena.allocate()
            self._arena.inject_phases(h_out, output_phases)
            h_io = self._arena.allocate()
            self._arena.bind(h_in, h_out, h_io)
            handles.append(h_io)
        return handles

    def encode_io_pairs(
        self, io_pairs: Sequence[Tuple[Any, Any]]
    ) -> BehavioralProfile:
        """Bundle pre-computed (input, output) pairs into a behavioral vector.

        This is the path used by :class:`ProblemEncoder` (Phase 8B.1)
        for encoding problem specifications without executing any
        program: the io_pairs are the problem's expected oracle.
        """
        if not io_pairs:
            # An empty io list still produces a deterministic
            # zero-phase handle so callers always have something to
            # correlate against.
            h_zero = self._arena.allocate()
            self._arena.inject_phases(h_zero, [0.0] * self._dimension)
            phases = self._arena.extract_phases(h_zero)
            return BehavioralProfile(
                behavioral_handle=h_zero,
                behavioral_phases=list(phases),
                io_pairs=[],
                execution_profile={"empty": True},
                metadata={
                    "io_pair_count": 0,
                    "provenance": {
                        "operation": "encode_io_pairs",
                        "source_arena": "thdse",
                        "io_pair_count": 0,
                    },
                },
            )

        io_handles = self._bind_io_handles(io_pairs)
        h_behavioral = self._arena.allocate()
        self._arena.bundle(io_handles, h_behavioral)
        phases = self._arena.extract_phases(h_behavioral)
        self._encode_count += 1
        return BehavioralProfile(
            behavioral_handle=h_behavioral,
            behavioral_phases=list(phases),
            io_pairs=list(io_pairs),
            execution_profile={
                "io_pair_count": len(io_pairs),
                "encoded_via": "encode_io_pairs",
            },
            metadata={
                "io_pair_count": len(io_pairs),
                "encode_index": self._encode_count,
                "provenance": {
                    "operation": "encode_io_pairs",
                    "source_arena": "thdse",
                    "io_pair_count": len(io_pairs),
                },
            },
        )

    def encode_behavior(self, source: str) -> BehavioralProfile:
        """Execute ``source`` against the encoder's probes, then bundle the io.

        Programs that crash on a probe still contribute an io pair: the
        output side carries a sentinel like
        ``("__rejected__", label)`` so the failure is recorded as data,
        not silently dropped. This is what allows the encoder to
        distinguish "raises on lists" from "succeeds on lists" — both
        are valid behaviors and both belong in the FHRR signature.
        """
        if not isinstance(source, str):
            raise TypeError(
                f"source must be a string, got {type(source).__name__}"
            )

        profile = self._sandbox.execute(source)
        io_pairs: List[Tuple[Any, Any]] = []

        # Walk the encoder's probes in order so the encoded io_pairs
        # always line up positionally with the probe set — this is
        # what guarantees that two programs with identical observable
        # behavior produce identical bundled vectors.
        for label, input_value in self._probe_inputs:
            stored = profile.returned_values.get(label)
            if stored is not None:
                output_value = ("__return__", stored)
            else:
                exception_type = profile.exception_type
                if not profile.compiled:
                    output_value = ("__compile_error__", exception_type or "")
                elif not profile.executed:
                    output_value = ("__exec_error__", exception_type or "")
                elif profile.timed_out:
                    output_value = ("__timeout__", "")
                else:
                    output_value = ("__rejected__", label)
            io_pairs.append((input_value, output_value))

        io_handles = self._bind_io_handles(io_pairs)
        h_behavioral = self._arena.allocate()
        self._arena.bundle(io_handles, h_behavioral)
        phases = self._arena.extract_phases(h_behavioral)
        self._encode_count += 1

        execution_summary = {
            "compiled": profile.compiled,
            "executed": profile.executed,
            "fitness": float(profile.fitness),
            "timed_out": profile.timed_out,
            "n_accepted": profile.n_accepted,
            "n_succeeded": profile.n_succeeded,
            "n_distinct": profile.n_distinct,
            "exception_type": profile.exception_type,
            "execution_time_ms": profile.execution_time_ms,
        }

        return BehavioralProfile(
            behavioral_handle=h_behavioral,
            behavioral_phases=list(phases),
            io_pairs=io_pairs,
            execution_profile=execution_summary,
            metadata={
                "encode_index": self._encode_count,
                "probe_count": len(self._probe_inputs),
                "provenance": {
                    "operation": "encode_behavior",
                    "source_arena": "thdse",
                    "target_arena": "cce",
                    "probe_count": len(self._probe_inputs),
                },
            },
        )

    # ------------------------------------------------------------------
    # Convenience: similarity helper
    # ------------------------------------------------------------------

    def similarity(
        self, profile_a: BehavioralProfile, profile_b: BehavioralProfile
    ) -> float:
        """Mean-cosine similarity in raw phase space.

        We compute the FHRR correlation directly on the stored phase
        arrays rather than going through the arena: that lets two
        profiles that live in different arenas be compared.
        """
        if self._dimension != len(profile_b.behavioral_phases):
            raise ValueError(
                f"profile_b dimension mismatch: expected {self._dimension}, "
                f"got {len(profile_b.behavioral_phases)}"
            )
        a = profile_a.behavioral_phases
        b = profile_b.behavioral_phases
        if len(a) != len(b):
            raise ValueError("profile dimensions disagree")
        return float(
            sum(math.cos(a[i] - b[i]) for i in range(len(a))) / len(a)
        )


__all__ = [
    "BehavioralEncoder",
    "BehavioralProfile",
    "DEFAULT_PROBE_INPUTS",
    "hash_to_phases",
]
