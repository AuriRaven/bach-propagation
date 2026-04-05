# tests/etl/test_execute_rename.py
"""Tests for execute_rename.py — dry run, copy, skip, failure handling."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.baroque_corpus_etl.transform.execute_rename import (
    ExecutionStats,
    execute_plan,
    verify_silver,
)


def _make_plan_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a minimal rename_plan.csv to a temp file."""
    plan_path = tmp_path / "rename_plan.csv"
    pd.DataFrame(rows).to_csv(plan_path, index=False)
    return plan_path


def _make_source_file(tmp_path: Path, name: str = "test.mid") -> Path:
    """Create a minimal fake MIDI source file."""
    p = tmp_path / name
    p.write_bytes(b"MThd" + b"\x00" * 10)
    return p


class TestDryRun:
    def test_dry_run_does_not_copy(self, tmp_path):
        src = _make_source_file(tmp_path)
        dst = tmp_path / "silver" / "bach" / "test.mid"
        plan = _make_plan_csv(tmp_path, [{
            "source_path": str(src),
            "silver_path": str(dst),
            "action": "RENAME",
        }])

        stats = execute_plan(plan, dry_run=True)

        assert not dst.exists()
        assert stats.dry_run == 1
        assert stats.copied == 0

    def test_dry_run_counts_correctly(self, tmp_path):
        src1 = _make_source_file(tmp_path, "a.mid")
        src2 = _make_source_file(tmp_path, "b.mid")
        plan = _make_plan_csv(tmp_path, [
            {"source_path": str(src1), "silver_path": str(tmp_path / "s/a.mid"), "action": "RENAME"},
            {"source_path": str(src2), "silver_path": str(tmp_path / "s/b.mid"), "action": "COLLISION_RESOLVED"},
        ])
        stats = execute_plan(plan, dry_run=True)
        assert stats.dry_run == 2


class TestExecute:
    def test_copies_file(self, tmp_path):
        src = _make_source_file(tmp_path)
        dst = tmp_path / "silver" / "bach" / "test.mid"
        plan = _make_plan_csv(tmp_path, [{
            "source_path": str(src),
            "silver_path": str(dst),
            "action": "RENAME",
        }])

        stats = execute_plan(plan, dry_run=False)

        assert dst.exists()
        assert stats.copied == 1
        assert stats.failed == 0

    def test_creates_parent_directories(self, tmp_path):
        src = _make_source_file(tmp_path)
        dst = tmp_path / "silver" / "deep" / "nested" / "dir" / "test.mid"
        plan = _make_plan_csv(tmp_path, [{
            "source_path": str(src),
            "silver_path": str(dst),
            "action": "RENAME",
        }])

        execute_plan(plan, dry_run=False)
        assert dst.exists()

    def test_bronze_source_unchanged(self, tmp_path):
        src = _make_source_file(tmp_path)
        original_content = src.read_bytes()
        dst = tmp_path / "silver" / "test.mid"
        plan = _make_plan_csv(tmp_path, [{
            "source_path": str(src),
            "silver_path": str(dst),
            "action": "RENAME",
        }])

        execute_plan(plan, dry_run=False)

        assert src.exists()
        assert src.read_bytes() == original_content

    def test_skip_existing_by_default(self, tmp_path):
        src = _make_source_file(tmp_path)
        dst = tmp_path / "silver" / "test.mid"
        dst.parent.mkdir(parents=True)
        dst.write_bytes(b"existing")
        plan = _make_plan_csv(tmp_path, [{
            "source_path": str(src),
            "silver_path": str(dst),
            "action": "RENAME",
        }])

        stats = execute_plan(plan, dry_run=False, overwrite=False)

        assert dst.read_bytes() == b"existing"
        assert stats.skipped_exists == 1

    def test_overwrite_replaces_existing(self, tmp_path):
        src = _make_source_file(tmp_path)
        dst = tmp_path / "silver" / "test.mid"
        dst.parent.mkdir(parents=True)
        dst.write_bytes(b"old content")
        plan = _make_plan_csv(tmp_path, [{
            "source_path": str(src),
            "silver_path": str(dst),
            "action": "RENAME",
        }])

        execute_plan(plan, dry_run=False, overwrite=True)

        assert dst.read_bytes() != b"old content"


class TestSkipActions:
    @pytest.mark.parametrize("action", ["DUPLICATE_DROP", "QUARANTINE", "NEEDS_REVIEW"])
    def test_skip_actions_not_copied(self, tmp_path, action):
        src = _make_source_file(tmp_path)
        dst = tmp_path / "silver" / "test.mid"
        plan = _make_plan_csv(tmp_path, [{
            "source_path": str(src),
            "silver_path": str(dst),
            "action": action,
        }])

        stats = execute_plan(plan, dry_run=False)

        assert not dst.exists()
        assert stats.skipped_action == 1
        assert stats.copied == 0


class TestMissingSource:
    def test_missing_source_counted_as_failure(self, tmp_path):
        plan = _make_plan_csv(tmp_path, [{
            "source_path": str(tmp_path / "nonexistent.mid"),
            "silver_path": str(tmp_path / "silver" / "test.mid"),
            "action": "RENAME",
        }])

        stats = execute_plan(plan, dry_run=False)

        assert stats.failed == 1
        assert stats.copied == 0


class TestVerify:
    def test_verify_passes_when_all_exist(self, tmp_path):
        src = _make_source_file(tmp_path)
        dst = tmp_path / "silver" / "test.mid"
        dst.parent.mkdir(parents=True)
        shutil.copy2(src, dst)
        plan = _make_plan_csv(tmp_path, [{
            "source_path": str(src),
            "silver_path": str(dst),
            "action": "RENAME",
        }])
        # Should not raise
        verify_silver(plan)

    def test_verify_detects_missing(self, tmp_path, caplog):
        plan = _make_plan_csv(tmp_path, [{
            "source_path": str(tmp_path / "src.mid"),
            "silver_path": str(tmp_path / "silver" / "missing.mid"),
            "action": "RENAME",
        }])
        import logging
        with caplog.at_level(logging.WARNING):
            verify_silver(plan)
        assert "VERIFICATION FAILED" in caplog.text