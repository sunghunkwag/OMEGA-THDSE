"""Tests for the empirical validation report generator.

Tests:
  1. test_empirical_report_smoke — Smoke test with max_files=50
  2. test_empirical_report_pair_decomposition — Verify per-layer keys in top_20_pairs
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_empirical_report_smoke():
    """Run generate_empirical_report with max_files=50 and verify structure."""
    from src.analysis.empirical_report import generate_empirical_report

    with tempfile.TemporaryDirectory() as tmpdir:
        report = generate_empirical_report(
            output_path=tmpdir,
            max_files=50,
            dimension=256,
        )

        # Verify required top-level keys
        assert "ingestion" in report
        assert "resonance_stats" in report
        assert "cliques" in report
        assert "top_20_pairs" in report
        assert "outliers" in report
        assert "system_info" in report

        # Verify ingestion
        assert report["ingestion"]["ingested"] >= 30, (
            f"Expected >= 30 modules ingested, got {report['ingestion']['ingested']}"
        )

        # Verify cliques
        assert report["cliques"]["total_size_3_plus"] >= 1, (
            f"Expected >= 1 clique, got {report['cliques']['total_size_3_plus']}"
        )

        # Verify output files exist
        assert os.path.isfile(os.path.join(tmpdir, "empirical_validation.json"))
        assert os.path.isfile(os.path.join(tmpdir, "empirical_validation.md"))


def test_empirical_report_pair_decomposition():
    """Verify each top_20_pairs entry has per-layer keys with valid floats."""
    from src.analysis.empirical_report import generate_empirical_report

    with tempfile.TemporaryDirectory() as tmpdir:
        report = generate_empirical_report(
            output_path=tmpdir,
            max_files=50,
            dimension=256,
        )

        pairs = report["top_20_pairs"]
        assert len(pairs) > 0, "Must have at least one pair"

        for entry in pairs:
            for key in ("sim_ast", "sim_cfg", "sim_data", "sim_final"):
                assert key in entry, f"Missing key {key} in pair entry"
                val = entry[key]
                assert isinstance(val, float), f"{key} must be float, got {type(val)}"
                assert -1.1 <= val <= 1.1, f"{key} out of range: {val}"
