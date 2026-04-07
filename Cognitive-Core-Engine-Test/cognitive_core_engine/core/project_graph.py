"""Project / Goal Graph (C-layer long-horizon structure)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cognitive_core_engine.core.utils import stable_hash


@dataclass
class ProjectNode:
    id: str
    name: str
    task: str
    status: str = "open"      # "open" | "active" | "done"
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    value_estimate: float = 0.0
    history: List[str] = field(default_factory=list)  # memory ids
    value_history: List[float] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)


class ProjectGraph:
    """
    Long-horizon project DAG:
    - orchestrator attaches agent runs to nodes
    - nodes accumulate evidence and value estimates
    - spawn subprojects based on value thresholds
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ProjectNode] = {}

    def create_root(self, name: str, task: str) -> str:
        nid = stable_hash({"name": name, "task": task, "root": True})
        self._nodes[nid] = ProjectNode(id=nid, name=name, task=task, status="open")
        return nid

    def add_child(self, parent_id: str, name: str,
                  task: Optional[str] = None) -> str:
        parent = self._nodes[parent_id]
        nid = stable_hash({"name": name, "task": task or parent.task, "parent": parent_id})
        node = ProjectNode(id=nid, name=name, task=task or parent.task,
                           status="open", parent_id=parent_id)
        self._nodes[nid] = node
        parent.children.append(nid)
        return nid

    def nodes_for_task(self, task: str) -> List[ProjectNode]:
        return [n for n in self._nodes.values() if n.task == task]

    def pick_node_for_round(self, task: str) -> ProjectNode:
        candidates = [n for n in self._nodes.values()
                      if n.task == task and n.status != "done"]
        if not candidates:
            nid = self.create_root(name=f"{task}_root", task=task)
            return self._nodes[nid]
        candidates.sort(key=lambda n: n.value_estimate, reverse=True)
        return candidates[0]

    def update_node(self, nid: str, reward: float,
                    memory_id: Optional[str]) -> None:
        node = self._nodes[nid]
        alpha = 0.25
        node.value_estimate = (1 - alpha) * node.value_estimate + alpha * reward
        node.value_history.append(node.value_estimate)
        if memory_id:
            node.history.append(memory_id)
            node.evidence_refs.append(memory_id)
        if node.value_estimate > 0.18 and len(node.children) < 3:
            self.add_child(parent_id=nid, name=f"{node.name}_infra_focus")
            self.add_child(parent_id=nid, name=f"{node.name}_breakthrough_focus")
        if node.value_estimate > 0.35:
            node.status = "active"
