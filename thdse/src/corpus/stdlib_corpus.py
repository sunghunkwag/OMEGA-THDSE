"""Auto-discovers and ingests the Python standard library as a corpus.

Uses the current Python installation's standard library path. This corpus
is guaranteed available on any system without external downloads.
"""

import os
import sys
import sysconfig
from pathlib import Path
from typing import List, Optional

from src.corpus.ingester import BatchIngester, IngestionStats


class StdlibCorpus:
    """Discovers and ingests Python standard library modules.

    Usage:
        stdlib = StdlibCorpus(synthesizer)
        stats = stdlib.ingest(max_files=500)
    """

    def __init__(
        self,
        ingester: BatchIngester,
    ) -> None:
        """Initialize stdlib corpus handler.

        Args:
            ingester: BatchIngester instance configured with a synthesizer.
        """
        self.ingester = ingester

    @staticmethod
    def find_stdlib_path() -> Optional[str]:
        """Locate the Python standard library directory.

        Returns:
            Path to stdlib directory, or None if not found.
        """
        # Try sysconfig first (most reliable)
        stdlib_path = sysconfig.get_path("stdlib")
        if stdlib_path and os.path.isdir(stdlib_path):
            return stdlib_path

        # Fallback: look relative to sys.executable
        prefix = sys.prefix
        for candidate in [
            os.path.join(prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}"),
            os.path.join(prefix, "Lib"),  # Windows
        ]:
            if os.path.isdir(candidate):
                return candidate

        return None

    @staticmethod
    def list_stdlib_files(
        stdlib_path: str,
        max_files: int = 0,
    ) -> List[Path]:
        """List eligible stdlib Python files.

        Skips test directories, __init__.py, and known problematic modules.

        Args:
            stdlib_path: Path to stdlib root.
            max_files: Maximum number of files to return (0 = unlimited).

        Returns:
            Sorted list of Path objects.
        """
        skip_dirs = {
            "__pycache__", "test", "tests", "tkinter", "turtledemo",
            "idlelib", "lib2to3", "ensurepip", "distutils",
            "unittest", "site-packages", "config",
        }
        skip_files = {
            "__init__.py", "setup.py", "__main__.py",
        }

        results: List[Path] = []
        root = Path(stdlib_path)

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skipped directories
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in skip_dirs and not d.startswith("_")
            )

            for fname in sorted(filenames):
                if not fname.endswith(".py"):
                    continue
                if fname in skip_files:
                    continue
                if fname.startswith("_"):
                    continue
                fpath = Path(dirpath) / fname
                results.append(fpath)
                if max_files > 0 and len(results) >= max_files:
                    return results

        return results

    def ingest(
        self,
        max_files: int = 0,
    ) -> IngestionStats:
        """Ingest Python standard library modules.

        Args:
            max_files: Maximum files to ingest (0 = all available).

        Returns:
            IngestionStats from ingestion.

        Raises:
            RuntimeError: If stdlib path cannot be found.
        """
        stdlib_path = self.find_stdlib_path()
        if stdlib_path is None:
            raise RuntimeError("Could not locate Python standard library")

        if self.ingester.verbose:
            print(f"Stdlib path: {stdlib_path}")

        return self.ingester.ingest_directory(
            stdlib_path,
            project_prefix="stdlib",
        )
