"""
Sub-Tree Vocabulary — Extracts concrete AST sub-trees from a source corpus,
canonicalizes them, and projects each through the FHRR pipeline to produce
hypervector handles for compositional decoding.

This replaces the abstract atom vocabulary (e.g., "If", "Return") with
concrete AST fragments harvested from real code, enabling the decoder to
assemble real structural patterns instead of placeholder templates.

Canonicalization:
  - Variable names → positional placeholders (x0, x1, ...)
  - Literal values → type markers (INT, STR, FLOAT, BOOL, NONE)
  - Deterministic hashing for deduplication

The vocabulary is built ONCE per corpus ingestion, then reused for all decoding.
"""

import ast
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class SubTreeAtom:
    """A canonicalized AST sub-tree with its FHRR handle and metadata.

    Attributes:
        canonical_ast: The canonicalized AST node (with placeholder vars/literals).
        canonical_source: Unparsed source of the canonical AST for display.
        tree_hash: SHA-256 hex digest of the canonical source (dedup key).
        handle: Arena handle for the projected hypervector (None before projection).
        frequency: Number of times this canonical form appeared in the corpus.
        depth: Depth of the sub-tree (2-4).
        root_type: AST node type name of the root (e.g., "If", "Assign").
        slot_count: Number of distinct placeholder variables in this sub-tree.
        is_definition: Whether this sub-tree defines a variable (has Store ctx).
        is_use: Whether this sub-tree uses a variable (has Load ctx).
    """
    canonical_ast: ast.AST
    canonical_source: str
    tree_hash: str
    handle: Optional[int] = None
    frequency: int = 1
    depth: int = 0
    root_type: str = ""
    slot_count: int = 0
    is_definition: bool = False
    is_use: bool = False


class _NameCanonicalizer(ast.NodeTransformer):
    """Replaces variable names with positional placeholders and literal values
    with type markers, producing a canonical form for structural comparison.

    Variable assignment order determines placeholder index:
    first variable encountered gets x0, second gets x1, etc.
    """

    # Type marker constants used as canonical literal values
    _TYPE_MARKERS = {
        int: 0,       # INT marker
        float: 0.0,   # FLOAT marker
        str: "",      # STR marker
        bool: True,   # BOOL marker
        type(None): None,  # NONE marker
    }

    def __init__(self) -> None:
        self._var_map: Dict[str, str] = {}
        self._var_counter: int = 0
        self._has_store: bool = False
        self._has_load: bool = False

    def _get_placeholder(self, name: str) -> str:
        """Map an original variable name to a positional placeholder."""
        if name in self._var_map:
            return self._var_map[name]
        placeholder = f"x{self._var_counter}"
        self._var_map[name] = placeholder
        self._var_counter += 1
        return placeholder

    @property
    def slot_count(self) -> int:
        """Number of distinct variable slots allocated."""
        return self._var_counter

    @property
    def has_store(self) -> bool:
        """Whether any Store context was encountered."""
        return self._has_store

    @property
    def has_load(self) -> bool:
        """Whether any Load context was encountered."""
        return self._has_load

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Replace variable name with positional placeholder."""
        # Preserve built-in names that are structural (range, len, print, etc.)
        builtins = {
            "range", "len", "print", "int", "float", "str", "bool",
            "list", "dict", "set", "tuple", "enumerate", "zip", "map",
            "filter", "sorted", "reversed", "isinstance", "type",
            "True", "False", "None", "ValueError", "TypeError",
            "KeyError", "IndexError", "StopIteration", "Exception",
        }
        if node.id in builtins:
            return node

        if isinstance(node.ctx, ast.Store):
            self._has_store = True
        elif isinstance(node.ctx, ast.Load):
            self._has_load = True

        new_name = self._get_placeholder(node.id)
        return ast.Name(id=new_name, ctx=node.ctx)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Replace function argument name with placeholder."""
        new_name = self._get_placeholder(node.arg)
        return ast.arg(arg=new_name, annotation=node.annotation)

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        """Replace literal value with type marker."""
        val_type = type(node.value)
        if val_type in self._TYPE_MARKERS:
            return ast.Constant(value=self._TYPE_MARKERS[val_type])
        # bytes, complex, etc. — keep as-is (rare)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Replace function name with placeholder."""
        new_name = self._get_placeholder(node.name)
        node.name = new_name
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Replace async function name with placeholder."""
        new_name = self._get_placeholder(node.name)
        node.name = new_name
        self.generic_visit(node)
        return node


def _compute_depth(node: ast.AST) -> int:
    """Compute the depth of an AST node (leaves = 1)."""
    max_child_depth = 0
    for child in ast.iter_child_nodes(node):
        max_child_depth = max(max_child_depth, _compute_depth(child))
    return max_child_depth + 1


def _extract_subtrees(
    tree: ast.AST,
    min_depth: int = 2,
    max_depth: int = 4,
) -> List[ast.AST]:
    """Extract all sub-trees of depth between min_depth and max_depth.

    Walks the AST and collects nodes whose sub-tree depth falls within range.
    Only collects statement and expression nodes (not auxiliary nodes like
    arguments, keywords, etc.).

    Args:
        tree: Root AST node to walk.
        min_depth: Minimum sub-tree depth (inclusive).
        max_depth: Maximum sub-tree depth (inclusive).

    Returns:
        List of AST nodes whose depth is in [min_depth, max_depth].
    """
    results: List[ast.AST] = []

    # Types worth extracting as sub-trees
    _EXTRACTABLE = (ast.stmt, ast.expr)

    for node in ast.walk(tree):
        if not isinstance(node, _EXTRACTABLE):
            continue
        # Skip bare Name/Constant — too trivial
        if isinstance(node, (ast.Name, ast.Constant)):
            continue
        depth = _compute_depth(node)
        if min_depth <= depth <= max_depth:
            results.append(node)

    return results


def _canonicalize(node: ast.AST) -> Tuple[ast.AST, int, bool, bool]:
    """Canonicalize an AST sub-tree by replacing names and literals.

    Returns:
        Tuple of (canonical_node, slot_count, is_definition, is_use).
    """
    # Deep copy to avoid mutating original
    import copy
    node_copy = copy.deepcopy(node)

    canonicalizer = _NameCanonicalizer()
    canonical = canonicalizer.visit(node_copy)
    ast.fix_missing_locations(canonical)

    return (
        canonical,
        canonicalizer.slot_count,
        canonicalizer.has_store,
        canonicalizer.has_load,
    )


def _hash_ast(node: ast.AST) -> str:
    """Compute a deterministic hash of an AST node's canonical form.

    Uses ast.dump() for structural representation, then SHA-256.
    """
    dump = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def _safe_unparse(node: ast.AST) -> str:
    """Unparse an AST node to source, handling edge cases."""
    try:
        # Wrap bare expressions/statements for unparsing
        if isinstance(node, ast.expr):
            wrapper = ast.Expression(body=node)
            ast.fix_missing_locations(wrapper)
            return ast.unparse(node)
        return ast.unparse(node)
    except Exception:
        return ast.dump(node)


class SubTreeVocabulary:
    """Builds and stores a vocabulary of canonical AST sub-trees from a corpus.

    The vocabulary maps each unique canonical sub-tree to a SubTreeAtom
    with an FHRR hypervector handle for resonance probing.

    Usage:
        vocab = SubTreeVocabulary()
        vocab.ingest_source(source_code_1)
        vocab.ingest_source(source_code_2)
        vocab.project_all(arena, projector)  # assigns handles
        atoms = vocab.get_atoms()
    """

    def __init__(
        self,
        min_depth: int = 2,
        max_depth: int = 4,
    ) -> None:
        """Initialize the sub-tree vocabulary.

        Args:
            min_depth: Minimum sub-tree depth to extract.
            max_depth: Maximum sub-tree depth to extract.
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self._atoms: Dict[str, SubTreeAtom] = {}  # hash -> atom
        self._projected: bool = False

    def ingest_source(self, source: str) -> int:
        """Extract and register sub-trees from a source string.

        Args:
            source: Python source code to parse and extract from.

        Returns:
            Number of new unique sub-trees added.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0

        subtrees = _extract_subtrees(tree, self.min_depth, self.max_depth)
        new_count = 0

        for node in subtrees:
            canonical, slot_count, is_def, is_use = _canonicalize(node)
            tree_hash = _hash_ast(canonical)

            if tree_hash in self._atoms:
                self._atoms[tree_hash].frequency += 1
            else:
                canonical_source = _safe_unparse(canonical)
                root_type = type(canonical).__name__
                depth = _compute_depth(canonical)

                self._atoms[tree_hash] = SubTreeAtom(
                    canonical_ast=canonical,
                    canonical_source=canonical_source,
                    tree_hash=tree_hash,
                    depth=depth,
                    root_type=root_type,
                    slot_count=slot_count,
                    is_definition=is_def,
                    is_use=is_use,
                )
                new_count += 1

        return new_count

    def ingest_file(self, filepath: str) -> int:
        """Extract sub-trees from a file on disk.

        Args:
            filepath: Path to a Python source file.

        Returns:
            Number of new unique sub-trees added.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            return self.ingest_source(source)
        except (OSError, UnicodeDecodeError):
            return 0

    def project_all(self, arena: Any, projector: Any) -> int:
        """Project all sub-tree atoms through the FHRR pipeline.

        Each canonical sub-tree is unparsed to source, projected via the
        IsomorphicProjector, and the resulting handle is stored.

        Args:
            arena: The FHRR arena.
            projector: The IsomorphicProjector instance.

        Returns:
            Number of atoms successfully projected.
        """
        projected = 0
        for tree_hash, atom in self._atoms.items():
            if atom.handle is not None:
                projected += 1
                continue

            # Project the canonical source through the full pipeline
            source = atom.canonical_source
            try:
                # Wrap statement-level nodes in a module for valid projection
                if isinstance(atom.canonical_ast, ast.stmt):
                    projection = projector.project(source)
                else:
                    # Expression nodes: wrap in Expr statement
                    projection = projector.project(source)
                atom.handle = projection.final_handle
                projected += 1
            except Exception:
                # If projection fails (e.g., unparseable canonical form),
                # use a deterministic fallback: hash-seeded phases
                try:
                    seed = int(tree_hash[:8], 16)
                    handle = arena.allocate()
                    phases = projector._deterministic_phases(seed)
                    arena.inject_phases(handle, phases)
                    atom.handle = handle
                    projected += 1
                except Exception:
                    pass

        self._projected = projected > 0
        return projected

    def get_atoms(self) -> Dict[str, SubTreeAtom]:
        """Return all registered atoms keyed by tree hash.

        Returns:
            Dictionary mapping tree hash to SubTreeAtom.
        """
        return dict(self._atoms)

    def get_atoms_by_type(self, root_type: str) -> List[SubTreeAtom]:
        """Return all atoms with a specific root AST node type.

        Args:
            root_type: AST node type name (e.g., "If", "For", "Assign").

        Returns:
            List of matching SubTreeAtom instances.
        """
        return [
            atom for atom in self._atoms.values()
            if atom.root_type == root_type
        ]

    def get_projected_atoms(self) -> List[SubTreeAtom]:
        """Return only atoms that have been projected (have handles).

        Returns:
            List of SubTreeAtom instances with non-None handles.
        """
        return [atom for atom in self._atoms.values() if atom.handle is not None]

    def size(self) -> int:
        """Return the number of unique sub-trees in the vocabulary."""
        return len(self._atoms)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the vocabulary contents.

        Returns:
            Dictionary with counts by root type, depth distribution, etc.
        """
        by_type: Dict[str, int] = {}
        by_depth: Dict[int, int] = {}
        total_freq = 0

        for atom in self._atoms.values():
            by_type[atom.root_type] = by_type.get(atom.root_type, 0) + 1
            by_depth[atom.depth] = by_depth.get(atom.depth, 0) + 1
            total_freq += atom.frequency

        return {
            "unique_subtrees": len(self._atoms),
            "total_occurrences": total_freq,
            "projected": sum(1 for a in self._atoms.values() if a.handle is not None),
            "by_root_type": by_type,
            "by_depth": by_depth,
        }
