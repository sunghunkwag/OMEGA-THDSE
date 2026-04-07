"""Batch ingestion engine for processing directories of Python projects.

Walks project directories, filters Python files, projects each through the
multi-layer topology → FHRR pipeline, and registers results as axioms in
a persistent AxiomStore.
"""

import os
import sys
import time
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from src.projection.isomorphic_projector import IsomorphicProjector
from src.synthesis.axiomatic_synthesizer import AxiomaticSynthesizer, Axiom

logger = logging.getLogger(__name__)


# Filenames/directories to skip during ingestion
_SKIP_DIRS = {
    "__pycache__", ".git", ".svn", ".hg", "node_modules",
    ".tox", ".eggs", "*.egg-info", "dist", "build",
    ".mypy_cache", ".pytest_cache",
}

_SKIP_PATTERNS = {
    "__init__.py",
    "setup.py",
    "conftest.py",
}

# Max file size to attempt parsing (64 KB).
# Larger files consume disproportionate arena handles (~50 handles per KB),
# so capping here keeps memory bounded on 8GB laptops.
_MAX_FILE_BYTES = 64 * 1024


class IngestionStats:
    """Tracks ingestion progress and outcomes."""

    def __init__(self) -> None:
        self.total_found: int = 0
        self.skipped_filter: int = 0
        self.skipped_too_large: int = 0
        self.skipped_syntax_error: int = 0
        self.skipped_projection_error: int = 0
        self.ingested: int = 0
        self.elapsed_seconds: float = 0.0

    def summary(self) -> str:
        """Return human-readable summary string."""
        return (
            f"Ingested {self.ingested}/{self.total_found} files "
            f"({self.skipped_filter} filtered, {self.skipped_too_large} too large, "
            f"{self.skipped_syntax_error} syntax errors, "
            f"{self.skipped_projection_error} projection errors) "
            f"in {self.elapsed_seconds:.1f}s"
        )


class BatchIngester:
    """Ingests directories of Python projects into an AxiomaticSynthesizer.

    Usage:
        import hdc_core
        arena = hdc_core.FhrrArena(1_000_000, 256)
        proj = IsomorphicProjector(arena, 256)
        synth = AxiomaticSynthesizer(arena, proj)
        ingester = BatchIngester(synth)
        stats = ingester.ingest_directory("/path/to/projects")
        ingester.save("corpus.pkl")
    """

    def __init__(
        self,
        synthesizer: AxiomaticSynthesizer,
        skip_tests: bool = True,
        max_file_bytes: int = _MAX_FILE_BYTES,
        verbose: bool = False,
    ) -> None:
        """Initialize the batch ingester.

        Args:
            synthesizer: The AxiomaticSynthesizer to register axioms into.
            skip_tests: Whether to skip test files and directories.
            max_file_bytes: Maximum file size in bytes to attempt parsing.
            verbose: Whether to print progress to stdout.
        """
        self.synthesizer = synthesizer
        self.skip_tests = skip_tests
        self.max_file_bytes = max_file_bytes
        self.verbose = verbose

    def _should_skip_dir(self, dirname: str) -> bool:
        """Check if a directory should be skipped."""
        if dirname in _SKIP_DIRS:
            return True
        if dirname.endswith(".egg-info"):
            return True
        if self.skip_tests and dirname in ("tests", "test", "testing"):
            return True
        return False

    def _should_skip_file(self, filepath: Path) -> bool:
        """Check if a file should be skipped based on name/size."""
        name = filepath.name
        if name in _SKIP_PATTERNS:
            return True
        if self.skip_tests and ("test_" in name or "_test.py" in name):
            return True
        # Skip generated code markers
        if name.startswith("_") and name != "__main__.py":
            return True
        return False

    def _collect_python_files(self, root_dir: str) -> List[Path]:
        """Walk a directory tree and collect eligible Python files."""
        root = Path(root_dir)
        results: List[Path] = []

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skipped directories in-place
            dirnames[:] = [
                d for d in dirnames if not self._should_skip_dir(d)
            ]
            for fname in sorted(filenames):
                if not fname.endswith(".py"):
                    continue
                fpath = Path(dirpath) / fname
                if not self._should_skip_file(fpath):
                    results.append(fpath)

        return results

    def _make_source_id(self, filepath: Path, root_dir: str) -> str:
        """Create a readable source ID from the file path relative to root."""
        try:
            rel = filepath.relative_to(root_dir)
        except ValueError:
            rel = filepath
        return str(rel)

    def ingest_directory(
        self,
        root_dir: str,
        project_prefix: str = "",
    ) -> IngestionStats:
        """Ingest all eligible Python files from a directory tree.

        Args:
            root_dir: Root directory to walk.
            project_prefix: Optional prefix for source IDs (e.g., project name).

        Returns:
            IngestionStats with counts and timing.
        """
        stats = IngestionStats()
        t0 = time.time()

        all_files = self._collect_python_files(root_dir)
        stats.total_found = len(all_files)

        for i, fpath in enumerate(all_files):
            # Check file size
            try:
                fsize = fpath.stat().st_size
            except OSError:
                stats.skipped_filter += 1
                continue

            if fsize > self.max_file_bytes:
                stats.skipped_too_large += 1
                continue

            if fsize == 0:
                stats.skipped_filter += 1
                continue

            # Read source
            try:
                source = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                stats.skipped_filter += 1
                continue

            # Build source ID
            sid = self._make_source_id(fpath, root_dir)
            if project_prefix:
                sid = f"{project_prefix}/{sid}"

            # Attempt ingestion
            try:
                self.synthesizer.ingest(sid, source)
                stats.ingested += 1
            except SyntaxError:
                stats.skipped_syntax_error += 1
                logger.debug("Syntax error in %s", fpath)
            except ValueError as exc:
                # Arena capacity exhausted — stop ingesting
                if "capacity" in str(exc).lower() or "exhausted" in str(exc).lower():
                    logger.warning("Arena capacity exhausted after %d files", stats.ingested)
                    if self.verbose:
                        print(f"  Arena capacity exhausted after {stats.ingested} files, stopping.")
                    break
                stats.skipped_projection_error += 1
                logger.debug("Projection error in %s: %s", fpath, exc)
            except Exception as exc:
                stats.skipped_projection_error += 1
                logger.debug("Projection error in %s: %s", fpath, exc)

            if self.verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{i + 1}/{stats.total_found}] "
                    f"{stats.ingested} ingested, {elapsed:.1f}s elapsed"
                )

        stats.elapsed_seconds = time.time() - t0
        return stats

    def ingest_multiple_projects(
        self,
        project_dirs: List[str],
    ) -> IngestionStats:
        """Ingest multiple project directories with project-name prefixes.

        Args:
            project_dirs: List of paths to project root directories.

        Returns:
            Aggregated IngestionStats across all projects.
        """
        combined = IngestionStats()

        for pdir in project_dirs:
            pname = Path(pdir).name
            if self.verbose:
                print(f"Ingesting project: {pname}")
            sub_stats = self.ingest_directory(pdir, project_prefix=pname)

            combined.total_found += sub_stats.total_found
            combined.skipped_filter += sub_stats.skipped_filter
            combined.skipped_too_large += sub_stats.skipped_too_large
            combined.skipped_syntax_error += sub_stats.skipped_syntax_error
            combined.skipped_projection_error += sub_stats.skipped_projection_error
            combined.ingested += sub_stats.ingested
            combined.elapsed_seconds += sub_stats.elapsed_seconds

            if self.verbose:
                print(f"  {pname}: {sub_stats.summary()}")

        return combined

    def save(self, filepath: str) -> None:
        """Serialize the current AxiomStore to disk.

        Saves axiom source IDs and their phase arrays (numpy-compatible).
        The arena handles are NOT serialized — only the phase data needed
        to reconstruct them.

        Args:
            filepath: Output path (recommended: .pkl extension).
        """
        store = self.synthesizer.store
        data: Dict[str, dict] = {}

        for sid in store.all_ids():
            axiom = store.get(sid)
            if axiom is None:
                continue
            proj = axiom.projection
            entry: dict = {
                "ast_phases": proj.ast_phases,
            }
            if proj.cfg_phases is not None:
                entry["cfg_phases"] = proj.cfg_phases
            if proj.data_phases is not None:
                entry["data_phases"] = proj.data_phases
            data[sid] = entry

        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.verbose:
            print(f"Saved {len(data)} axioms to {filepath}")

    @staticmethod
    def load(
        filepath: str,
        synthesizer: AxiomaticSynthesizer,
    ) -> int:
        """Load a serialized corpus and re-register axioms in the synthesizer.

        Re-injects phase arrays into the arena and registers as axioms.

        Args:
            filepath: Path to the .pkl file saved by save().
            synthesizer: The AxiomaticSynthesizer to load into.

        Returns:
            Number of axioms loaded.
        """
        from src.projection.isomorphic_projector import LayeredProjection

        with open(filepath, "rb") as f:
            data: Dict[str, dict] = pickle.load(f)

        arena = synthesizer.arena
        count = 0

        for sid, entry in data.items():
            ast_phases = entry["ast_phases"]
            cfg_phases = entry.get("cfg_phases")
            data_phases = entry.get("data_phases")
            dim = len(ast_phases)

            # Allocate and inject handles
            ast_h = arena.allocate()
            arena.inject_phases(ast_h, ast_phases)

            cfg_h = None
            if cfg_phases is not None:
                cfg_h = arena.allocate()
                arena.inject_phases(cfg_h, cfg_phases)

            data_h = None
            if data_phases is not None:
                data_h = arena.allocate()
                arena.inject_phases(data_h, data_phases)

            # Compute final handle: ast ⊗ cfg ⊗ data
            final_h = ast_h
            if cfg_h is not None:
                bound = arena.allocate()
                arena.bind(final_h, cfg_h, bound)
                final_h = bound
            if data_h is not None:
                bound = arena.allocate()
                arena.bind(final_h, data_h, bound)
                final_h = bound

            proj = LayeredProjection(
                final_handle=final_h,
                ast_handle=ast_h,
                cfg_handle=cfg_h,
                data_handle=data_h,
                ast_phases=ast_phases,
                cfg_phases=cfg_phases,
                data_phases=data_phases,
            )

            synthesizer.store.register(sid, proj)
            count += 1

        return count
