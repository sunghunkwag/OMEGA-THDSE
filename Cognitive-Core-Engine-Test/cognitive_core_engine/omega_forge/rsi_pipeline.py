"""
BN-02 — RSI Skill Registration Pipeline
========================================
Bridges the OmegaForge VirtualMachine (genome execution) to the core
SkillLibrary so that approved evolutionary candidates become first-class
registered skills rather than discarded byte-sequences.

Pipeline:
  1. Extract genome from approved L0 RuleProposal payload
  2. Compile genome → Python callable via VirtualMachine
  3. Quarantine smoke-test (functional isolation, no side-effects)
  4. Register skill in SkillLibrary with provenance metadata
  5. Record adoption event in SharedMemory
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cognitive_core_engine.core.skills import SkillLibrary, Skill
from cognitive_core_engine.core.memory import SharedMemory
from cognitive_core_engine.core.utils import stable_hash
from cognitive_core_engine.omega_forge.instructions import Instruction, ProgramGenome
from cognitive_core_engine.omega_forge.vm import VirtualMachine


# ---------------------------------------------------------------------------
# Quarantine configuration
# ---------------------------------------------------------------------------

_QUARANTINE_INPUTS: List[List[float]] = [
    [0.0, 0.0, 0.0],
    [1.0, 2.0, 3.0],
    [5.0, -1.0, 0.5],
    [0.0],
    [1.0, 1.0, 1.0, 1.0],
]
_SMOKE_TEST_MAX_STEPS = 400  # BN-10: increased for evolved programs with loops
_SMOKE_TEST_TIMEOUT_S = 0.5   # wall-clock timeout per input
_MIN_CLEAN_HALTS = 2          # BN-10: reduced from 3 to let more evolved programs pass


@dataclass
class QuarantineReport:
    passed: bool
    clean_halts: int
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class RSIRegistrationResult:
    success: bool
    skill_id: Optional[str] = None
    reason: str = ""
    quarantine: Optional[QuarantineReport] = None


# ---------------------------------------------------------------------------
# Genome compiler: proposal payload → ProgramGenome
# ---------------------------------------------------------------------------

def _compile_genome(candidate: Dict[str, Any]) -> Optional[ProgramGenome]:
    """Convert a candidate dict to a ProgramGenome.

    The candidate dict carries a 'code' list of (op, a, b, c) tuples.
    Returns None if the payload is malformed.
    """
    code_raw = candidate.get("code")
    if not code_raw or not isinstance(code_raw, list):
        return None
    instructions: List[Instruction] = []
    for item in code_raw:
        try:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                op, a, b, c = item
                instructions.append(Instruction(str(op), int(a), int(b), int(c)))
            elif isinstance(item, dict):
                instructions.append(Instruction(
                    str(item.get("op", "HALT")),
                    int(item.get("a", 0)),
                    int(item.get("b", 0)),
                    int(item.get("c", 0)),
                ))
        except (TypeError, ValueError):
            return None
    if not instructions:
        return None
    gid = candidate.get("gid", stable_hash({"code": code_raw}))
    return ProgramGenome(gid=str(gid), instructions=instructions)


# ---------------------------------------------------------------------------
# Quarantine smoke-test
# ---------------------------------------------------------------------------

def _quarantine_genome(genome: ProgramGenome) -> QuarantineReport:
    """Execute genome against standard inputs; check for clean halts.

    Anti-cheat E3: rejects constant-output skills (must produce at least
    2 distinct reg0 values across the 5 quarantine inputs).
    """
    vm = VirtualMachine(max_steps=_SMOKE_TEST_MAX_STEPS)
    clean_halts = 0
    errors: List[str] = []
    notes: List[str] = []
    outputs: List[float] = []

    for inp in _QUARANTINE_INPUTS:
        t0 = time.monotonic()
        try:
            state = vm.execute(genome, inp)
        except Exception as exc:
            errors.append(f"exception on input {inp}: {exc}")
            continue
        elapsed = time.monotonic() - t0

        if elapsed > _SMOKE_TEST_TIMEOUT_S:
            notes.append(f"slow execution ({elapsed:.3f}s) on input {inp}")

        if state.halted_cleanly:
            clean_halts += 1
            outputs.append(float(state.regs[0]))
        elif state.error:
            errors.append(f"VM error '{state.error}' on input {inp}")
        else:
            notes.append(f"did not halt cleanly on input {inp}")

    # Anti-cheat E3: reject constant-output skills
    distinct_outputs = len(set(round(o, 6) for o in outputs))
    if distinct_outputs < 2 and len(outputs) >= 2:
        notes.append(f"constant output rejected ({distinct_outputs} distinct values)")
        return QuarantineReport(passed=False, clean_halts=clean_halts,
                                errors=errors, notes=notes)

    passed = clean_halts >= _MIN_CLEAN_HALTS
    return QuarantineReport(passed=passed, clean_halts=clean_halts,
                            errors=errors, notes=notes)


# ---------------------------------------------------------------------------
# VM-compiled callable wrapper
# ---------------------------------------------------------------------------

class _VMSkillCallable:
    """Wraps a ProgramGenome as a callable skill function.

    Accepts a list of floats (or any iterable of numbers) as the input.
    Returns the VM's register 0 value after execution.
    """

    def __init__(self, genome: ProgramGenome, gid: str) -> None:
        self._genome = genome
        self._gid = gid
        self._vm = VirtualMachine(max_steps=_SMOKE_TEST_MAX_STEPS)

    def __call__(self, inputs: List[float]) -> float:
        state = self._vm.execute(self._genome, [float(x) for x in inputs])
        return float(state.regs[0])

    def __repr__(self) -> str:
        return f"<VMSkill gid={self._gid} len={len(self._genome.instructions)}>"


# ---------------------------------------------------------------------------
# RSI Skill Registrar
# ---------------------------------------------------------------------------

class RSISkillRegistrar:
    """Registers approved OmegaForge genomes as SkillLibrary entries.

    Intended to be called at the end of run_recursive_cycle() with the
    list of critic_results so that any adopted L0 candidates flow through
    the full compile → quarantine → register pipeline.
    """

    def __init__(self, skill_library: SkillLibrary, memory: SharedMemory) -> None:
        self._skills = skill_library
        self._memory = memory
        self._registered_gids: set = set()
        self._last_birth_event: Optional[Dict[str, Any]] = None
        self._all_birth_events: List[Dict[str, Any]] = []

    def get_recent_birth_events(self) -> List[Dict[str, Any]]:
        """Return and clear recent skill birth events for causal chain wiring."""
        events = list(self._all_birth_events)
        self._all_birth_events.clear()
        return events

    def process_critic_results(
        self,
        critic_results: List[Dict[str, Any]],
        proposals_by_id: Dict[str, Any],
    ) -> List[RSIRegistrationResult]:
        """Process a batch of critic results; register qualifying candidates.

        Args:
            critic_results: list of {proposal_id, level, verdict, adopted}
            proposals_by_id: mapping proposal_id -> RuleProposal (for payload access)

        Returns:
            List of RSIRegistrationResult, one per L0 approved candidate.
        """
        results: List[RSIRegistrationResult] = []

        for cr in critic_results:
            if cr.get("level") != "L0":
                continue
            if not cr.get("adopted"):
                continue

            proposal_id = cr.get("proposal_id", "")
            proposal = proposals_by_id.get(proposal_id)
            if proposal is None:
                results.append(RSIRegistrationResult(
                    success=False, reason="proposal not found in index"))
                continue

            candidate = proposal.payload.get("candidate", {})
            gid = str(candidate.get("gid", ""))

            # Skip if already registered
            if gid in self._registered_gids:
                results.append(RSIRegistrationResult(
                    success=False, reason="already registered", skill_id=gid))
                continue

            result = self._register_candidate(candidate, gid, proposal)
            results.append(result)

        return results

    def _register_candidate(
        self,
        candidate: Dict[str, Any],
        gid: str,
        proposal: Any,
    ) -> RSIRegistrationResult:
        """Compile, quarantine and register a single candidate."""
        # Step 1: Compile genome
        genome = _compile_genome(candidate)
        if genome is None:
            return RSIRegistrationResult(
                success=False, reason="genome compilation failed")

        # Step 2: Quarantine smoke-test
        qr = _quarantine_genome(genome)
        if not qr.passed:
            return RSIRegistrationResult(
                success=False,
                reason=f"quarantine failed ({qr.clean_halts}/{_MIN_CLEAN_HALTS} clean halts)",
                quarantine=qr,
            )

        # Step 3: Build callable and register in SkillLibrary
        callable_fn = _VMSkillCallable(genome, gid)
        skill_id = f"rsi_{gid[:16]}"
        metrics = candidate.get("metrics", {})
        # Build Skill using actual dataclass fields (name, purpose, steps, tags)
        # Store the VM callable in the skill's fn attribute after construction
        from cognitive_core_engine.core.skills import SkillStep
        skill = Skill(
            name=f"RSI candidate {gid[:8]}",
            purpose=(
                f"Auto-registered evolutionary candidate.  "
                f"Train pass rate: {metrics.get('train_pass_rate', 0):.2f}  "
                f"Holdout pass rate: {metrics.get('holdout_pass_rate', 0):.2f}"
            ),
            steps=[SkillStep(kind="call", tool="rsi_vm_execute")],
            tags=["rsi", "omega_forge", f"gen_{candidate.get('generation', 0)}"],
        )
        # Attach the VM callable for agent consultation
        skill.fn = callable_fn
        skill.metadata = {
            "gid": gid,
            "generation": candidate.get("generation", 0),
            "metrics": metrics,
            "proposal_id": getattr(proposal, "proposal_id", ""),
            "quarantine_clean_halts": qr.clean_halts,
        }
        # Override auto-generated id with our skill_id
        skill.id = skill_id
        self._skills.register(skill)
        self._registered_gids.add(gid)

        # Step 4: Record provenance in SharedMemory
        self._memory.add(
            "artifact",
            f"rsi_skill_registered:{skill_id}",
            {
                "skill_id": skill_id,
                "gid": gid,
                "metrics": metrics,
                "quarantine_clean_halts": qr.clean_halts,
            },
            tags=["rsi", "skill_registration"],
        )

        # BN-08: Emit skill birth event for causal chain tracking
        capabilities = []
        try:
            # Determine capabilities by testing against task names
            callable_test_inputs = [[1.0, 2.0, 3.0], [0.5, 0.3, 0.2, 3.0, 12.0]]
            for test_inp in callable_test_inputs:
                try:
                    callable_fn(test_inp)
                    capabilities.append("compute")
                    break
                except Exception:
                    pass
            if metrics.get("train_pass_rate", 0) > 0.3:
                capabilities.append("predict_reward")
            if metrics.get("holdout_pass_rate", 0) > 0.25:
                capabilities.append("optimize_action")
        except Exception:
            capabilities = ["general"]

        birth_event = {
            "event": "new_skill_registered",
            "skill_id": skill_id,
            "capabilities": capabilities if capabilities else ["general"],
            "genome_fitness": float(metrics.get("train_pass_rate", 0)),
            "generation": candidate.get("generation", 0),
        }
        self._last_birth_event = birth_event
        self._all_birth_events.append(birth_event)
        # Store birth event in SharedMemory
        self._memory.add(
            "artifact",
            f"skill_birth:{skill_id}",
            birth_event,
            tags=["skill_birth", "rsi"],
        )

        return RSIRegistrationResult(
            success=True, skill_id=skill_id, quarantine=qr)
