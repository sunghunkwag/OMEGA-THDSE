"""
Variable Threading Engine — Ensures consistent variable naming across
composed AST sub-trees using a union-find (disjoint set) algorithm.

When the decoder assembles multiple canonical sub-trees into a single
program, placeholder variables (x0, x1, ...) from different sub-trees
must be unified into consistent concrete names based on def-use relationships.

Algorithm:
  1. Build a def-use graph across the ordered sub-trees
  2. For each placeholder slot, determine if it's a definition or use site
  3. Use union-find to merge placeholders that share data flow:
     - All definitions of x0 within the same sub-tree map to one name
     - If sub-tree B's x0 use-site connects to sub-tree A's x0 def-site
       (via data-dep edges), they share the same concrete name
     - Unconnected placeholders get fresh names
  4. Apply the unified naming to all sub-trees

Complexity: O(n * alpha(n)) where alpha is the inverse Ackermann function.
"""

import ast
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


# Deterministic concrete variable names assigned in order
_CONCRETE_NAMES = [
    "x", "y", "z", "n", "m", "k", "i", "j",
    "a", "b", "c", "d", "result", "acc", "val", "item",
    "data", "count", "total", "curr", "prev", "temp",
]


@dataclass
class VarSlot:
    """A variable placeholder slot identified by its sub-tree index and name.

    Attributes:
        subtree_idx: Index of the sub-tree in the ordered sequence.
        placeholder: The placeholder name (e.g., "x0", "x1").
        is_def: Whether this slot is a definition site (Store context).
        is_use: Whether this slot is a use site (Load context).
    """
    subtree_idx: int
    placeholder: str
    is_def: bool = False
    is_use: bool = False


class UnionFind:
    """Disjoint set data structure with path compression and union by rank.

    Supports O(alpha(n)) amortized find and union operations.
    """

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def make_set(self, key: str) -> None:
        """Create a new singleton set.

        Args:
            key: Unique identifier for the element.
        """
        if key not in self._parent:
            self._parent[key] = key
            self._rank[key] = 0

    def find(self, key: str) -> str:
        """Find the representative of the set containing key.

        Uses path compression for amortized O(alpha(n)).

        Args:
            key: Element to find the representative of.

        Returns:
            The representative element of the set.
        """
        if key not in self._parent:
            self.make_set(key)
        if self._parent[key] != key:
            self._parent[key] = self.find(self._parent[key])
        return self._parent[key]

    def union(self, key_a: str, key_b: str) -> None:
        """Merge the sets containing key_a and key_b.

        Uses union by rank for balanced trees.

        Args:
            key_a: First element.
            key_b: Second element.
        """
        root_a = self.find(key_a)
        root_b = self.find(key_b)
        if root_a == root_b:
            return

        if self._rank[root_a] < self._rank[root_b]:
            self._parent[root_a] = root_b
        elif self._rank[root_a] > self._rank[root_b]:
            self._parent[root_b] = root_a
        else:
            self._parent[root_b] = root_a
            self._rank[root_a] += 1

    def groups(self) -> Dict[str, List[str]]:
        """Return all equivalence classes.

        Returns:
            Dictionary mapping representative to list of members.
        """
        result: Dict[str, List[str]] = {}
        for key in self._parent:
            root = self.find(key)
            if root not in result:
                result[root] = []
            result[root].append(key)
        return result


def _extract_var_slots(
    subtree_idx: int,
    node: ast.AST,
) -> List[VarSlot]:
    """Extract all variable placeholder slots from an AST node.

    Walks the AST and finds all Name nodes with placeholder identifiers
    (matching x0, x1, ... pattern), recording whether they are def or use sites.

    Args:
        subtree_idx: Index of this sub-tree in the ordered sequence.
        node: AST node to extract from.

    Returns:
        List of VarSlot instances found in this sub-tree.
    """
    slots: List[VarSlot] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id.startswith("x"):
            # Check if it's a placeholder (x followed by digits)
            suffix = child.id[1:]
            if suffix.isdigit():
                is_def = isinstance(child.ctx, (ast.Store, ast.Del))
                is_use = isinstance(child.ctx, ast.Load)
                slots.append(VarSlot(
                    subtree_idx=subtree_idx,
                    placeholder=child.id,
                    is_def=is_def,
                    is_use=is_use,
                ))
        elif isinstance(child, ast.arg) and child.arg.startswith("x"):
            suffix = child.arg[1:]
            if suffix.isdigit():
                slots.append(VarSlot(
                    subtree_idx=subtree_idx,
                    placeholder=child.arg,
                    is_def=True,
                    is_use=False,
                ))
    return slots


def _make_slot_key(subtree_idx: int, placeholder: str) -> str:
    """Create a unique key for a variable slot.

    Args:
        subtree_idx: Sub-tree index.
        placeholder: Placeholder name.

    Returns:
        String key like "st0:x0".
    """
    return f"st{subtree_idx}:{placeholder}"


class VariableThreader:
    """Threads variables across composed sub-trees using union-find.

    Given an ordered sequence of canonical AST sub-trees with placeholder
    variable names, determines which placeholders should share the same
    concrete variable name based on def-use relationships.

    Usage:
        threader = VariableThreader()
        threader.add_subtrees(ordered_subtrees)
        threader.unify_adjacent()  # connect across sub-tree boundaries
        name_map = threader.resolve()
        threader.apply(ordered_subtrees, name_map)
    """

    def __init__(self) -> None:
        self._uf = UnionFind()
        self._slots: List[VarSlot] = []
        self._subtree_slots: Dict[int, List[VarSlot]] = {}  # idx -> slots
        self._subtree_count: int = 0

    def add_subtrees(self, subtrees: List[ast.AST]) -> None:
        """Register all sub-trees and extract their variable slots.

        Args:
            subtrees: Ordered list of canonical AST sub-tree nodes.
        """
        self._subtree_count = len(subtrees)
        for idx, node in enumerate(subtrees):
            slots = _extract_var_slots(idx, node)
            self._subtree_slots[idx] = slots
            for slot in slots:
                key = _make_slot_key(slot.subtree_idx, slot.placeholder)
                self._uf.make_set(key)
            self._slots.extend(slots)

    def unify_within_subtrees(self) -> None:
        """Unify all occurrences of the same placeholder within each sub-tree.

        Within a single sub-tree, all references to x0 (whether def or use)
        must map to the same concrete name.
        """
        for idx, slots in self._subtree_slots.items():
            # Group by placeholder name within this sub-tree
            by_name: Dict[str, List[VarSlot]] = {}
            for slot in slots:
                if slot.placeholder not in by_name:
                    by_name[slot.placeholder] = []
                by_name[slot.placeholder].append(slot)

            # Unify all occurrences of same placeholder in same sub-tree
            for name, group in by_name.items():
                if len(group) > 1:
                    first_key = _make_slot_key(group[0].subtree_idx, group[0].placeholder)
                    for slot in group[1:]:
                        key = _make_slot_key(slot.subtree_idx, slot.placeholder)
                        self._uf.union(first_key, key)

    def unify_adjacent(self) -> None:
        """Unify variables across adjacent sub-trees based on def-use chains.

        If sub-tree A defines x0 and sub-tree B (immediately following) uses x0,
        they likely refer to the same variable and should share a concrete name.

        This heuristic connects:
        - Definitions in sub-tree i with uses in sub-tree i+1 that share the
          same placeholder index
        - Definitions with the same placeholder name across consecutive sub-trees
        """
        self.unify_within_subtrees()

        for idx in range(self._subtree_count - 1):
            current_slots = self._subtree_slots.get(idx, [])
            next_slots = self._subtree_slots.get(idx + 1, [])

            # Collect defs in current sub-tree
            current_defs: Dict[str, VarSlot] = {}
            for slot in current_slots:
                if slot.is_def:
                    current_defs[slot.placeholder] = slot

            # Connect uses in next sub-tree to defs in current
            for slot in next_slots:
                if slot.is_use and slot.placeholder in current_defs:
                    def_slot = current_defs[slot.placeholder]
                    def_key = _make_slot_key(def_slot.subtree_idx, def_slot.placeholder)
                    use_key = _make_slot_key(slot.subtree_idx, slot.placeholder)
                    self._uf.union(def_key, use_key)

    def unify_by_data_deps(
        self,
        data_deps: List[Tuple[int, str, int, str]],
    ) -> None:
        """Unify variables based on explicit data dependency edges.

        Args:
            data_deps: List of (def_subtree_idx, def_placeholder,
                       use_subtree_idx, use_placeholder) tuples representing
                       data flow edges from the VSA data-dep layer.
        """
        for def_idx, def_var, use_idx, use_var in data_deps:
            def_key = _make_slot_key(def_idx, def_var)
            use_key = _make_slot_key(use_idx, use_var)
            self._uf.make_set(def_key)
            self._uf.make_set(use_key)
            self._uf.union(def_key, use_key)

    def resolve(self) -> Dict[str, str]:
        """Resolve all placeholder slots to concrete variable names.

        Returns:
            Dictionary mapping slot keys ("st0:x0") to concrete names ("x").
        """
        groups = self._uf.groups()

        # Assign concrete names to each equivalence class deterministically
        name_map: Dict[str, str] = {}
        name_idx = 0

        # Sort groups by their representative for determinism
        for rep in sorted(groups.keys()):
            members = sorted(groups[rep])
            if name_idx < len(_CONCRETE_NAMES):
                concrete = _CONCRETE_NAMES[name_idx]
            else:
                concrete = f"v{name_idx}"
            for member in members:
                name_map[member] = concrete
            name_idx += 1

        return name_map

    def apply(
        self,
        subtrees: List[ast.AST],
        name_map: Dict[str, str],
    ) -> List[ast.AST]:
        """Apply the resolved naming to all sub-trees in-place.

        Replaces placeholder names (x0, x1, ...) with concrete names
        according to the name_map.

        Args:
            subtrees: Ordered list of AST sub-tree nodes to rename.
            name_map: Mapping from slot keys to concrete variable names.

        Returns:
            The same list of sub-trees, modified in-place.
        """
        for idx, node in enumerate(subtrees):
            renamer = _ConcreteRenamer(idx, name_map)
            renamer.visit(node)
            ast.fix_missing_locations(node)
        return subtrees


class _ConcreteRenamer(ast.NodeTransformer):
    """Renames placeholder variables to concrete names using the resolved map."""

    def __init__(self, subtree_idx: int, name_map: Dict[str, str]) -> None:
        self._idx = subtree_idx
        self._name_map = name_map

    def _resolve(self, placeholder: str) -> str:
        """Look up concrete name for a placeholder in this sub-tree."""
        key = _make_slot_key(self._idx, placeholder)
        return self._name_map.get(key, placeholder)

    def _is_placeholder(self, name: str) -> bool:
        """Check if a name is a placeholder (x followed by digits)."""
        if not name.startswith("x"):
            return False
        suffix = name[1:]
        return suffix.isdigit()

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename placeholder Name nodes."""
        if self._is_placeholder(node.id):
            node.id = self._resolve(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Rename placeholder function arguments."""
        if self._is_placeholder(node.arg):
            node.arg = self._resolve(node.arg)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Rename placeholder function names."""
        if self._is_placeholder(node.name):
            node.name = self._resolve(node.name)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Rename placeholder async function names."""
        if self._is_placeholder(node.name):
            node.name = self._resolve(node.name)
        self.generic_visit(node)
        return node


def thread_variables(
    subtrees: List[ast.AST],
    data_deps: Optional[List[Tuple[int, str, int, str]]] = None,
) -> List[ast.AST]:
    """Convenience function: thread variables across ordered sub-trees.

    Performs the full threading pipeline:
    1. Extract variable slots from each sub-tree
    2. Unify within sub-trees (same placeholder = same variable)
    3. Unify across adjacent sub-trees (def-use chains)
    4. Optionally unify by explicit data-dep edges
    5. Resolve to concrete names
    6. Apply renaming

    Args:
        subtrees: Ordered list of canonical AST sub-tree nodes.
        data_deps: Optional explicit data dependency edges.

    Returns:
        The sub-trees with placeholders replaced by concrete variable names.
    """
    if not subtrees:
        return subtrees

    threader = VariableThreader()
    threader.add_subtrees(subtrees)
    threader.unify_adjacent()

    if data_deps:
        threader.unify_by_data_deps(data_deps)

    name_map = threader.resolve()
    return threader.apply(subtrees, name_map)
