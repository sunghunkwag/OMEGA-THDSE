"""Typed exceptions for OMEGA-THDSE integration boundaries.

These exceptions guard the architectural invariants defined in PLAN.md:
dimension safety (Rule 6), governance gating (Rule 7), bridge integrity
(Rule 12), and meta-recursion depth (PLAN.md Section F Risk 6).

Raising a typed exception at a boundary is ALWAYS preferred to silent
fallback behavior — every failure of an invariant must be visible.
"""


class OmegaThdseError(Exception):
    """Base class for all OMEGA-THDSE integration errors.

    Catching this base class lets higher-level orchestration code
    distinguish integration-layer failures from generic Python errors
    (ValueError, TypeError, etc.).
    """

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = dict(context) if context else {}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} [{ctx}]"


class DimensionMismatchError(OmegaThdseError):
    """Raised when a vector's dimension does not match the arena it is used in.

    PLAN.md Rule 6: "Any operation mixing handles from different arenas must
    go through DimensionBridge. Direct cross-arena operations must raise
    DimensionMismatchError."

    Typical trigger: passing a 256-dim THDSE vector to :func:`project_down`,
    which requires a 10,000-dim CCE vector.
    """

    def __init__(
        self,
        message: str,
        *,
        expected: int | tuple | None = None,
        actual: int | tuple | None = None,
        operation: str | None = None,
    ):
        context: dict = {}
        if expected is not None:
            context["expected"] = expected
        if actual is not None:
            context["actual"] = actual
        if operation is not None:
            context["operation"] = operation
        super().__init__(message, context=context)
        self.expected = expected
        self.actual = actual
        self.operation = operation


class GovernanceError(OmegaThdseError):
    """Raised when an operation bypasses the governance gate.

    PLAN.md Rule 7: "SkillLibrary.register() must require
    governance_approved=True. Calls without this parameter must raise
    GovernanceError."

    Also raised by the axiom→skill bridge when a synthesized program
    fails Critic/Sandbox validation but registration is still attempted.
    """

    def __init__(
        self,
        message: str,
        *,
        subject: str | None = None,
        reason: str | None = None,
    ):
        context: dict = {}
        if subject is not None:
            context["subject"] = subject
        if reason is not None:
            context["reason"] = reason
        super().__init__(message, context=context)
        self.subject = subject
        self.reason = reason


class BridgeIntegrityError(OmegaThdseError):
    """Raised when the DimensionBridge self-test fails on import.

    PLAN.md Rule 12: "DimensionBridge must run a self-test on import
    that verifies: (a) subsample(bind(A,B)) == bind(subsample(A),
    subsample(B)) for 10 random pairs, (b) self-similarity > 0.95,
    (c) random similarity < 0.1. If any test fails, raise
    BridgeIntegrityError on import."

    A BridgeIntegrityError at import time means the mathematical
    invariants of the asymmetric bridge have been violated — downstream
    code cannot be trusted to operate correctly until the bridge is
    repaired.
    """

    def __init__(
        self,
        message: str,
        *,
        check: str | None = None,
        trial: int | None = None,
        observed: float | None = None,
        threshold: float | None = None,
    ):
        context: dict = {}
        if check is not None:
            context["check"] = check
        if trial is not None:
            context["trial"] = trial
        if observed is not None:
            context["observed"] = observed
        if threshold is not None:
            context["threshold"] = threshold
        super().__init__(message, context=context)
        self.check = check
        self.trial = trial
        self.observed = observed
        self.threshold = threshold


class RecursionLimitError(OmegaThdseError):
    """Raised when meta-recursion exceeds the hardcoded depth limit.

    PLAN.md Section D sets ``_MAX_META_RECURSION_DEPTH = 2`` as an
    immutable safety invariant. Section F Risk 6 requires the
    orchestrator to raise this exception on any attempt to exceed
    depth 2.
    """

    def __init__(
        self,
        message: str,
        *,
        attempted_depth: int | None = None,
        limit: int | None = None,
    ):
        context: dict = {}
        if attempted_depth is not None:
            context["attempted_depth"] = attempted_depth
        if limit is not None:
            context["limit"] = limit
        super().__init__(message, context=context)
        self.attempted_depth = attempted_depth
        self.limit = limit


__all__ = [
    "OmegaThdseError",
    "DimensionMismatchError",
    "GovernanceError",
    "BridgeIntegrityError",
    "RecursionLimitError",
]
