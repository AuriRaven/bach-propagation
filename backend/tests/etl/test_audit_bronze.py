# tests/etl/test_audit_bronze.py
"""Unit tests for the Bronze auditor — no real music21 parsing needed."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.baroque_corpus_etl.transform.audit_bronze import (
    AuditRow,
    compute_content_hash,
    crawl_raw_dir,
    extract_bwv_from_filename,
    extract_bwv_from_metadata,
)


# ---------------------------------------------------------------------------
# AuditRow
# ---------------------------------------------------------------------------

class TestAuditRow:
    def test_initial_status_is_ok(self):
        row = AuditRow()
        assert row.audit_status == "OK"

    def test_add_warning_changes_status(self):
        row = AuditRow()
        row.add_warning("no_metadata")
        assert row.audit_status == "WARNING"
        assert "no_metadata" in row.audit_warnings

    def test_multiple_warnings_pipe_separated(self):
        row = AuditRow()
        row.add_warning("no_metadata")
        row.add_warning("no_bwv")
        assert row.audit_warnings == "no_metadata|no_bwv"

    def test_set_error_overrides_status(self):
        row = AuditRow()
        row.add_warning("no_metadata")
        row.set_error("parse_failed")
        assert row.audit_status == "ERROR"

    def test_to_csv_dict_excludes_private_fields(self):
        row = AuditRow()
        d = row.to_csv_dict()
        assert "_warnings" not in d

    def test_error_msg_truncated_at_512(self):
        row = AuditRow()
        row.set_error("x" * 1000)
        assert len(row.parse_error_msg) == 512


# ---------------------------------------------------------------------------
# BWV extraction
# ---------------------------------------------------------------------------

class TestExtractBWV:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("bwv772.krn", "BWV772"),
            ("BWV_772.mxl", "BWV772"),
            ("bach_bwv-772a.mid", "BWV772A"),
            ("invention_no_bwv.krn", ""),
            ("BWV1080.xml", "BWV1080"),
        ],
    )
    def test_extract_from_filename(self, filename: str, expected: str):
        assert extract_bwv_from_filename(filename) == expected

    def test_extract_from_metadata_title(self):
        mock_score = MagicMock()
        mock_score.metadata.movementNumber = None
        mock_score.metadata.number = None
        mock_score.metadata.title = "Invention BWV772"
        mock_score.metadata.composer = ""
        assert extract_bwv_from_metadata(mock_score) == "BWV772"

    def test_extract_from_metadata_none(self):
        mock_score = MagicMock()
        mock_score.metadata = None
        assert extract_bwv_from_metadata(mock_score) == ""


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

class TestCrawlRawDir:
    def test_yields_supported_extensions(self, tmp_path: Path):
        (tmp_path / "piece.krn").touch()
        (tmp_path / "piece.mxl").touch()
        (tmp_path / "readme.txt").touch()  # Should be excluded

        results = list(crawl_raw_dir(tmp_path))
        assert len(results) == 2
        assert all(p.suffix in {".krn", ".mxl"} for p in results)

    def test_raises_on_missing_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(crawl_raw_dir(tmp_path / "nonexistent"))

    def test_recursive_crawl(self, tmp_path: Path):
        subdir = tmp_path / "bach" / "cantatas"
        subdir.mkdir(parents=True)
        (subdir / "bwv1.mid").touch()
        (tmp_path / "bwv2.krn").touch()

        results = list(crawl_raw_dir(tmp_path))
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_same_content_same_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.krn"
        f2 = tmp_path / "b.krn"
        f1.write_bytes(b"abc" * 100)
        f2.write_bytes(b"abc" * 100)
        assert compute_content_hash(f1) == compute_content_hash(f2)

    def test_different_content_different_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.krn"
        f2 = tmp_path / "b.krn"
        f1.write_bytes(b"abc")
        f2.write_bytes(b"xyz")
        assert compute_content_hash(f1) != compute_content_hash(f2)