"""Tests for generate_embeddings.py.

All tests are pure unit tests — no OpenAI calls, no Supabase, no network.
API interactions are fully mocked.

Run from backend/ project root:
    uv run pytest tests/etl/test_generate_embeddings.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.baroque_corpus_etl.load.generate_embeddings import (
    build_embedding_text,
    _fetch_existing_embedding_ids,
    _upsert_embeddings,
    EMBEDDING_MODEL,
    EMBEDDING_DIMS,
)


# =============================================================================
# build_embedding_text
# =============================================================================

class TestBuildEmbeddingText:
    def _full_row(self, **overrides) -> dict:
        base = {
            "collection": "chorales",
            "bwv": "227",
            "movement_name": None,
            "key_signature": "G major",
            "time_signature": "4/4",
            "form_tag": "chorale",
            "instrument_family": "choir",
        }
        base.update(overrides)
        return base

    def test_full_row_produces_correct_string(self):
        row = self._full_row()
        result = build_embedding_text(row)
        assert result == "Bach | chorales | BWV227 | unknown | G major | 4/4 | chorale | choir"

    def test_null_bwv_becomes_unknown(self):
        row = self._full_row(bwv=None)
        result = build_embedding_text(row)
        assert "BWVunknown" not in result  # prefix should not appear on null
        assert "unknown" in result

    def test_null_key_becomes_unknown(self):
        row = self._full_row(key_signature=None)
        result = build_embedding_text(row)
        assert "unknown" in result

    def test_all_nulls_produces_valid_string(self):
        row = {k: None for k in [
            "collection", "bww", "movement_name", "key_signature",
            "time_signature", "form_tag", "instrument_family", "bwv",
        ]}
        result = build_embedding_text(row)
        # Should still have 8 pipe-separated parts
        parts = result.split(" | ")
        assert len(parts) == 8
        assert parts[0] == "Bach"

    def test_movement_name_included_when_present(self):
        row = self._full_row(movement_name="Prelude")
        result = build_embedding_text(row)
        assert "Prelude" in result

    def test_bwv_prefix_applied(self):
        row = self._full_row(bwv="1007")
        result = build_embedding_text(row)
        assert "BWV1007" in result

    def test_solo_cello_row(self):
        row = {
            "collection": "solo_instruments",
            "bwv": "1007",
            "movement_name": "Prelude",
            "key_signature": "G major",
            "time_signature": "3/4",
            "form_tag": "prelude",
            "instrument_family": "strings",
        }
        result = build_embedding_text(row)
        expected = "Bach | solo_instruments | BWV1007 | Prelude | G major | 3/4 | prelude | strings"
        assert result == expected

    def test_orchestral_row_no_key(self):
        row = {
            "collection": "orchestral",
            "bwv": "1080",
            "movement_name": "Fugue",
            "key_signature": None,
            "time_signature": "4/4",
            "form_tag": "fugue",
            "instrument_family": "mixed",
        }
        result = build_embedding_text(row)
        assert "unknown" in result   # null key → unknown
        assert "BWV1080" in result
        assert "Fugue" in result

    def test_output_is_pipe_separated(self):
        row = self._full_row()
        parts = build_embedding_text(row).split(" | ")
        assert len(parts) == 8

    def test_output_always_starts_with_bach(self):
        row = self._full_row()
        assert build_embedding_text(row).startswith("Bach")

    def test_empty_string_fields_become_unknown(self):
        row = self._full_row(key_signature="", form_tag="  ")
        result = build_embedding_text(row)
        parts = result.split(" | ")
        # key_signature is index 4, form_tag is index 6
        assert parts[4] == "unknown"
        assert parts[6] == "unknown"


# =============================================================================
# _fetch_existing_embedding_ids
# =============================================================================

class TestFetchExistingEmbeddingIds:
    def test_returns_set_of_ids(self):
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.execute.return_value.data = [
            {"id": "uuid-a"},
            {"id": "uuid-b"},
        ]
        result = _fetch_existing_embedding_ids(mock_client)
        assert result == {"uuid-a", "uuid-b"}

    def test_empty_table_returns_empty_set(self):
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.execute.return_value.data = []
        result = _fetch_existing_embedding_ids(mock_client)
        assert result == set()


# =============================================================================
# _upsert_embeddings
# =============================================================================

class TestUpsertEmbeddings:
    def _make_batch(self, n: int = 3) -> list[dict]:
        return [
            {
                "id": f"uuid-{i}",
                "embedding_text": f"Bach | chorales | BWV{i} | unknown | G major | 4/4 | chorale | choir",
                "embedding": [0.1] * 1536,
                "model": EMBEDDING_MODEL,
            }
            for i in range(n)
        ]

    def test_calls_upsert_on_corpus_embeddings(self):
        mock_client = MagicMock()
        batch = self._make_batch(2)
        _upsert_embeddings(mock_client, batch)

        mock_client.table.assert_any_call("corpus_embeddings")

    def test_calls_update_on_corpus_files(self):
        mock_client = MagicMock()
        batch = self._make_batch(2)
        _upsert_embeddings(mock_client, batch)

        mock_client.table.assert_any_call("corpus_files")

    def test_status_updated_to_embedded(self):
        mock_client = MagicMock()
        batch = self._make_batch(2)
        _upsert_embeddings(mock_client, batch)

        # Find the update call on corpus_files
        update_calls = [
            c for c in mock_client.table.return_value.update.call_args_list
        ]
        assert any(
            c == call({"load_status": "embedded"})
            for c in update_calls
        )

    def test_batch_ids_passed_to_in_filter(self):
        mock_client = MagicMock()
        batch = self._make_batch(3)
        _upsert_embeddings(mock_client, batch)

        in_calls = mock_client.table.return_value.update.return_value.in_.call_args_list
        assert len(in_calls) > 0
        # The IDs passed to .in_() must match the batch
        ids_passed = in_calls[0][0][1]  # positional arg 1 of .in_("id", ids)
        assert set(ids_passed) == {"uuid-0", "uuid-1", "uuid-2"}


# =============================================================================
# Constants sanity checks
# =============================================================================

class TestConstants:
    def test_embedding_model_name(self):
        assert EMBEDDING_MODEL == "text-embedding-3-small"

    def test_embedding_dims(self):
        assert EMBEDDING_DIMS == 1536