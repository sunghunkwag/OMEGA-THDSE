"""Skill DSL (data-level programs) extracted from NON_RSI_AGI_CORE_v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from cognitive_core_engine.core.utils import stable_hash

if TYPE_CHECKING:
    from cognitive_core_engine.core.tools import ToolRegistry


@dataclass
class SkillStep:
    kind: str
    tool: Optional[str] = None
    args_template: Optional[Dict[str, Any]] = None
    condition: Optional[Dict[str, Any]] = None
    steps: Optional[List["SkillStep"]] = None
    else_steps: Optional[List["SkillStep"]] = None
    list_key: Optional[str] = None
    item_key: Optional[str] = None


@dataclass
class Skill:
    """
    Interpreted skill program:
    - steps are data structures with explicit control-flow
    - supports: call, if, foreach
    - arguments can reference context via ${key}
    """
    name: str
    purpose: str
    steps: List[SkillStep]
    tags: List[str] = field(default_factory=list)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = stable_hash(
            {
                "name": self.name,
                "purpose": self.purpose,
                "steps": [self._serialize_step(s) for s in self.steps],
            }
        )

    def _serialize_step(self, step: SkillStep) -> Dict[str, Any]:
        return {
            "kind": step.kind,
            "tool": step.tool,
            "args_template": step.args_template,
            "condition": step.condition,
            "list_key": step.list_key,
            "item_key": step.item_key,
            "steps": [self._serialize_step(s) for s in step.steps] if step.steps else None,
            "else_steps": [self._serialize_step(s) for s in step.else_steps] if step.else_steps else None,
        }

    def run(self, tools: ToolRegistry, context: Dict[str, Any]) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        ctx = dict(context)

        def subst(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                key = value[2:-1]
                return ctx.get(key)
            if isinstance(value, dict):
                return {k: subst(v) for k, v in value.items()}
            if isinstance(value, list):
                return [subst(v) for v in value]
            return value

        def eval_condition(cond: Dict[str, Any]) -> bool:
            key = cond.get("key")
            op = cond.get("op", "truthy")
            val = cond.get("value")
            cur = ctx.get(key)
            if op == "eq":
                return cur == val
            if op == "neq":
                return cur != val
            if op == "contains":
                return isinstance(cur, (list, str)) and val in cur
            if op == "gt":
                return isinstance(cur, (int, float)) and cur > val
            if op == "lt":
                return isinstance(cur, (int, float)) and cur < val
            if op == "gte":
                return isinstance(cur, (int, float)) and cur >= val
            if op == "lte":
                return isinstance(cur, (int, float)) and cur <= val
            return bool(cur)

        def run_steps(steps: Iterable[SkillStep], depth: int = 0) -> bool:
            if depth > 12:
                return False
            for i, st in enumerate(steps):
                if st.kind == "call" and st.tool:
                    args = subst(st.args_template or {})
                    if not isinstance(args, dict):
                        args = {"value": args}
                    res = tools.call(st.tool, args)
                    trace.append({"i": len(trace), "tool": st.tool, "args": args, "res": res})
                    ctx["last"] = res
                    if isinstance(res, dict):
                        ctx["last_verdict"] = res.get("verdict")
                    ctx[f"step_{len(trace) - 1}"] = res
                    if not res.get("ok", False):
                        return False
                elif st.kind == "if" and st.condition:
                    branch = st.steps if eval_condition(st.condition) else st.else_steps
                    if branch:
                        if not run_steps(branch, depth + 1):
                            return False
                elif st.kind == "foreach" and st.list_key:
                    items = ctx.get(st.list_key, [])
                    if isinstance(items, list) and st.steps:
                        for idx, item in enumerate(items):
                            ctx[st.item_key or "item"] = item
                            ctx["index"] = idx
                            if not run_steps(st.steps, depth + 1):
                                return False
                else:
                    return False
            return True

        ok = run_steps(self.steps)
        return {
            "ok": ok,
            "trace": trace,
            "final": ctx.get("last"),
        }


class SkillLibrary:
    def __init__(self, max_skills: int = 3000) -> None:
        self.max_skills = max_skills
        self._skills: Dict[str, Skill] = {}
        self.skill_performance_log: Dict[str, List[float]] = {}
        self._perf_log_active: bool = False

    def add(self, sk: Skill) -> str:
        self._skills[sk.id] = sk
        if len(self._skills) > self.max_skills:
            for sid in list(self._skills.keys())[: len(self._skills) - self.max_skills]:
                self._skills.pop(sid, None)
        return sk.id

    def register(self, sk: Skill) -> str:
        """Register a skill (alias for add, used by RSI pipeline)."""
        return self.add(sk)

    def vm_skills(self) -> List[Skill]:
        """Return only RSI-originated skills (tagged 'rsi')."""
        return [s for s in self._skills.values() if "rsi" in s.tags]

    def log_skill_performance(self, skill_id: str, reward: float) -> None:
        """Record a reward observation for a skill (append-only).

        Anti-cheat E4: once the first entry is logged, the log
        cannot be cleared or rewritten.
        """
        self._perf_log_active = True
        if skill_id not in self.skill_performance_log:
            self.skill_performance_log[skill_id] = []
        self.skill_performance_log[skill_id].append(float(reward))

    def clear_performance_log(self) -> None:
        """Attempt to clear — raises if log is active (anti-cheat E4)."""
        if self._perf_log_active:
            raise RuntimeError("Performance log is append-only after first entry")
        self.skill_performance_log.clear()

    def list(self, tag: Optional[str] = None) -> List[Skill]:
        vals = list(self._skills.values())
        if tag is None:
            return vals
        return [s for s in vals if tag in s.tags]

    def get(self, sid: str) -> Optional[Skill]:
        return self._skills.get(sid)
