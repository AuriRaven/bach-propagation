"""Tests for upload_to_storage.py.

Run from backend/ project root:
    uv run pytest tests/etl/test_upload_to_storage.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Adjust import path so pytest can find the module without an editable install.
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.baroque_corpus_etl.load.upload_to_storage import (
    _collection_from_path,
    _subcollection_from_path,
    iter_silver_files,
    md5_checksum,
    upload_file_atomic,
    VALID_COLLECTIONS,
)


# =============================================================================
# _collection_from_path
# =============================================================================

class TestCollectionFromPath:
    def test_chorales(self):
        assert _collection_from_path("bach/chorales/r001.krn") == "chorales"

    def test_keyboard_subcollection(self):
        assert _collection_from_path("bach/keyboard/english_suites/bwv806.mid") == "keyboard"

    def test_solo_instruments_deep(self):
        path = "bach/solo_instruments/cello/cello_suites/bwv1007_prelude.mid"
        assert _collection_from_path(path) == "solo_instruments"

    def test_orchestral(self):
        assert _collection_from_path("bach/orchestral/concertos/bwv1041.mid") == "orchestral"


# =============================================================================
# _subcollection_from_path
# =============================================================================

class TestSubcollectionFromPath:
    def test_flat_chorales_returns_none(self):
        # bach / chorales / file.krn  → only 3 parts, no subcollection
        assert _subcollection_from_path("bach/chorales/r001.krn") is None

    def test_keyboard_english_suites(self):
        path = "bach/keyboard/english_suites/bwv806.mid"
        assert _subcollection_from_path(path) == "english_suites"

    def test_orchestral_concertos(self):
        assert _subcollection_from_path("bach/orchestral/concertos/bwv1041.mid") == "concertos"

    def test_solo_cello(self):
        path = "bach/solo_instruments/cello/cello_suites/bwv1007_prelude.mid"
        assert _subcollection_from_path(path) == "cello"


# =============================================================================
# iter_silver_files
# =============================================================================

class TestIterSilverFiles:
    def test_raises_if_root_missing(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="Silver root not found"):
            list(iter_silver_files(missing))

    def test_yields_only_krn_and_mid(self, tmp_path):
        # Build a minimal fake silver tree
        chorales = tmp_path / "bach" / "chorales"
        chorales.mkdir(parents=True)
        (chorales / "r001.krn").write_bytes(b"fake krn")
        (chorales / "r001.mid").write_bytes(b"fake mid")
        (chorales / "r001.txt").write_bytes(b"ignored")
        (chorales / ".DS_Store").write_bytes(b"ignored")

        results = list(iter_silver_files(tmp_path))
        filenames = [p.name for p, _ in results]

        assert "r001.krn" in filenames
        assert "r001.mid" in filenames
        assert "r001.txt" not in filenames
        assert ".DS_Store" not in filenames

    def test_excludes_invalid_collections(self, tmp_path):
        # contrast/ should be excluded
        contrast = tmp_path / "bach" / "contrast" / "vivaldi"
        contrast.mkdir(parents=True)
        (contrast / "vivaldi_op3.mid").write_bytes(b"fake")

        results = list(iter_silver_files(tmp_path))
        assert len(results) == 0

    def test_filter_collection(self, tmp_path):
        chorales = tmp_path / "bach" / "chorales"
        chorales.mkdir(parents=True)
        (chorales / "r001.krn").write_bytes(b"fake")

        organ = tmp_path / "bach" / "organ"
        organ.mkdir(parents=True)
        (organ / "bwv525.mid").write_bytes(b"fake")

        # Only chorales
        results = list(iter_silver_files(tmp_path, filter_collection="chorales"))
        assert all(_collection_from_path(rel) == "chorales" for _, rel in results)
        assert len(results) == 1

    def test_silver_paths_are_relative(self, tmp_path):
        chorales = tmp_path / "bach" / "chorales"
        chorales.mkdir(parents=True)
        (chorales / "r001.krn").write_bytes(b"fake")

        _, silver_path = next(iter_silver_files(tmp_path))
        # Must not be an absolute path
        assert not Path(silver_path).is_absolute()
        assert silver_path == "bach/chorales/r001.krn"


# =============================================================================
# md5_checksum
# =============================================================================

class TestMd5Checksum:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.mid"
        f.write_bytes(b"hello bach")
        assert md5_checksum(f) == md5_checksum(f)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.mid"
        f2 = tmp_path / "b.mid"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        assert md5_checksum(f1) != md5_checksum(f2)

    def test_returns_hex_string(self, tmp_path):
        f = tmp_path / "test.krn"
        f.write_bytes(b"data")
        result = md5_checksum(f)
        assert isinstance(result, str)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)


# =============================================================================
# upload_file_atomic — dry run (no real API calls)
# =============================================================================

class TestUploadFileAtomicDryRun:
    def test_dry_run_returns_none(self, tmp_path):
        f = tmp_path / "test.krn"
        f.write_bytes(b"krn content")

        mock_client = MagicMock()
        result = upload_file_atomic(
            client=mock_client,
            abs_path=f,
            silver_path="bach/chorales/test.krn",
            execute=False,
        )
        assert result is None

    def test_dry_run_makes_no_api_calls(self, tmp_path):
        f = tmp_path / "test.krn"
        f.write_bytes(b"krn content")

        mock_client = MagicMock()
        upload_file_atomic(
            client=mock_client,
            abs_path=f,
            silver_path="bach/chorales/test.krn",
            execute=False,
        )
        mock_client.storage.from_.assert_not_called()
        mock_client.table.assert_not_called()


# =============================================================================
# upload_file_atomic — rollback on DB failure
# =============================================================================

class TestUploadFileAtomicRollback:
    def test_rolls_back_storage_on_db_failure(self, tmp_path):
        """If the DB insert fails, the storage object must be deleted."""
        f = tmp_path / "test.mid"
        f.write_bytes(b"midi content")

        mock_client = MagicMock()

        # Storage upload succeeds
        mock_client.storage.from_.return_value.upload.return_value = MagicMock()

        # DB insert raises
        mock_client.table.return_value.upsert.return_value.execute.side_effect = (
            RuntimeError("DB down")
        )

        with pytest.raises(RuntimeError, match="Atomic upload failed"):
            upload_file_atomic(
                client=mock_client,
                abs_path=f,
                silver_path="bach/chorales/test.mid",
                execute=True,
            )

        # Verify rollback was attempted
        mock_client.storage.from_.return_value.remove.assert_called_once_with(
            ["bach/chorales/test.mid"]
        )


# =============================================================================
# VALID_COLLECTIONS sanity check
# =============================================================================

class TestValidCollections:
    def test_expected_collections_present(self):
        expected = {
            "chorales", "inventions", "keyboard", "organ",
            "orchestral", "sacred", "solo_instruments",
        }
        assert expected == VALID_COLLECTIONS

    def test_vivaldi_excluded(self):
        assert "contrast" not in VALID_COLLECTIONS
        assert "vivaldi" not in VALID_COLLECTIONS