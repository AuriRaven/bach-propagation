"""Tests for load_corpus.py orchestrator.

All tests are pure unit tests — no Supabase, no network, no file I/O.
Stage functions and Supabase client are fully mocked.

Run from backend/ project root:
    uv run pytest tests/load/test_load_corpus.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.baroque_corpus_etl.load.load_corpus import (
    health_check,
    print_health,
    run_pipeline,
    STAGES,
)


# =============================================================================
# STAGES constant
# =============================================================================

class TestStagesConstant:
    def test_stages_order(self):
        assert STAGES == ("upload", "metadata", "embed")

    def test_stages_count(self):
        assert len(STAGES) == 3


# =============================================================================
# health_check
# =============================================================================

class TestHealthCheck:
    def _mock_client(
        self,
        files: int = 670,
        metadata: int = 670,
        embeddings: int = 670,
        embedded: int = 670,
        failed: int = 0,
    ) -> MagicMock:
        """Build a mock Supabase client that returns controlled counts."""
        client = MagicMock()

        def _select_mock(*args, **kwargs):
            """Return a chain that yields .count based on which table was queried."""
            return MagicMock()

        # We'll track calls by table name via side_effect on .table()
        call_counts = {
            "corpus_files":      files,
            "corpus_metadata":   metadata,
            "corpus_embeddings": embeddings,
        }

        def table_side_effect(name: str):
            mock_table = MagicMock()
            base_count = call_counts.get(name, 0)

            def select_side_effect(*a, **kw):
                mock_select = MagicMock()

                def eq_side_effect(col, val):
                    mock_eq = MagicMock()
                    if val == "embedded":
                        mock_eq.execute.return_value.count = embedded
                    else:
                        mock_eq.execute.return_value.count = base_count
                    return mock_eq

                def like_side_effect(col, pattern):
                    mock_like = MagicMock()
                    mock_like.execute.return_value.count = failed
                    return mock_like

                mock_select.eq.side_effect = eq_side_effect
                mock_select.like.side_effect = like_side_effect
                mock_select.execute.return_value.count = base_count
                return mock_select

            mock_table.select.side_effect = select_side_effect
            return mock_table

        client.table.side_effect = table_side_effect
        return client

    def test_returns_all_expected_keys(self):
        client = self._mock_client()
        result = health_check(client)
        assert set(result.keys()) == {
            "corpus_files", "corpus_metadata", "corpus_embeddings",
            "embedded", "failed",
        }

    def test_counts_are_integers(self):
        client = self._mock_client()
        result = health_check(client)
        for v in result.values():
            assert isinstance(v, int)

    def test_zero_failed(self):
        client = self._mock_client(failed=0)
        result = health_check(client)
        assert result["failed"] == 0

    def test_nonzero_failed(self):
        client = self._mock_client(failed=3)
        result = health_check(client)
        assert result["failed"] == 3


# =============================================================================
# print_health (smoke — just verify it doesn't raise)
# =============================================================================

class TestPrintHealth:
    def test_no_failures_does_not_raise(self, caplog):
        stats = {
            "corpus_files": 670, "corpus_metadata": 670,
            "corpus_embeddings": 670, "embedded": 670, "failed": 0,
        }
        print_health(stats)  # should not raise

    def test_failures_logged_as_warning(self, caplog):
        import logging
        stats = {
            "corpus_files": 670, "corpus_metadata": 670,
            "corpus_embeddings": 670, "embedded": 667, "failed": 3,
        }
        with caplog.at_level(logging.WARNING):
            print_health(stats)
        assert any("failed" in r.message.lower() for r in caplog.records)

    def test_label_prefix_used(self, caplog):
        import logging
        stats = {
            "corpus_files": 10, "corpus_metadata": 10,
            "corpus_embeddings": 10, "embedded": 10, "failed": 0,
        }
        with caplog.at_level(logging.INFO):
            print_health(stats, label="before")
        assert any("[before]" in r.message for r in caplog.records)


# =============================================================================
# run_pipeline — stage routing
# =============================================================================

class TestRunPipelineStageRouting:
    """Verify that the orchestrator calls the correct stage functions."""

    def _run(self, stage=None, execute=False, collection=None, limit=None):
        with (
            patch("scripts.baroque_corpus_etl.load.load_corpus.create_client") as mock_create,
            patch("scripts.baroque_corpus_etl.load.load_corpus.run_upload") as mock_upload,
            patch("scripts.baroque_corpus_etl.load.load_corpus.run_extraction") as mock_extract,
            patch("scripts.baroque_corpus_etl.load.load_corpus.run_embedding") as mock_embed,
            patch("scripts.baroque_corpus_etl.load.load_corpus.health_check") as mock_health,
        ):
            mock_health.return_value = {
                "corpus_files": 10, "corpus_metadata": 10,
                "corpus_embeddings": 10, "embedded": 10, "failed": 0,
            }
            run_pipeline(
                execute=execute,
                stage=stage,
                filter_collection=collection,
                limit=limit,
                silver_root=Path("data/silver"),
            )
            return mock_upload, mock_extract, mock_embed

    def test_all_stages_called_by_default(self):
        upload, extract, embed = self._run(stage=None)
        upload.assert_called_once()
        extract.assert_called_once()
        embed.assert_called_once()

    def test_upload_only_stage(self):
        upload, extract, embed = self._run(stage="upload")
        upload.assert_called_once()
        extract.assert_not_called()
        embed.assert_not_called()

    def test_metadata_only_stage(self):
        upload, extract, embed = self._run(stage="metadata")
        upload.assert_not_called()
        extract.assert_called_once()
        embed.assert_not_called()

    def test_embed_only_stage(self):
        upload, extract, embed = self._run(stage="embed")
        upload.assert_not_called()
        extract.assert_not_called()
        embed.assert_called_once()

    def test_collection_filter_passed_to_upload(self):
        upload, _, _ = self._run(stage="upload", collection="chorales")
        _, kwargs = upload.call_args
        assert kwargs.get("filter_collection") == "chorales" or \
               upload.call_args[0][2] == "chorales"  # positional fallback

    def test_limit_passed_to_all_stages(self):
        upload, extract, embed = self._run(limit=5)
        for fn in (upload, extract, embed):
            args, kwargs = fn.call_args
            limit_val = kwargs.get("limit") or (args[3] if len(args) > 3 else None)
            assert limit_val == 5

    def test_dry_run_calls_stages_with_execute_false(self):
        upload, extract, embed = self._run(execute=False)
        for fn in (upload, extract, embed):
            args, kwargs = fn.call_args
            # Use explicit key presence check — "or" short-circuits on False
            if "execute" in kwargs:
                execute_val = kwargs["execute"]
            elif len(args) > 1:
                execute_val = args[1]
            else:
                execute_val = None
            assert execute_val is False

    def test_health_check_called_twice(self):
        with (
            patch("scripts.baroque_corpus_etl.load.load_corpus.create_client"),
            patch("scripts.baroque_corpus_etl.load.load_corpus.run_upload"),
            patch("scripts.baroque_corpus_etl.load.load_corpus.run_extraction"),
            patch("scripts.baroque_corpus_etl.load.load_corpus.run_embedding"),
            patch("scripts.baroque_corpus_etl.load.load_corpus.health_check") as mock_health,
        ):
            mock_health.return_value = {
                "corpus_files": 0, "corpus_metadata": 0,
                "corpus_embeddings": 0, "embedded": 0, "failed": 0,
            }
            run_pipeline(
                execute=False, stage=None, filter_collection=None,
                limit=None, silver_root=Path("data/silver"),
            )
            assert mock_health.call_count == 2  # before + after


# =============================================================================
# run_pipeline — failure exit
# =============================================================================

class TestRunPipelineFailureHandling:
    def test_exits_1_when_failed_rows_remain(self):
        with (
            patch("scripts.baroque_corpus_etl.load.load_corpus.create_client"),
            patch("scripts.baroque_corpus_etl.load.load_corpus.run_upload"),
            patch("scripts.baroque_corpus_etl.load.load_corpus.run_extraction"),
            patch("scripts.baroque_corpus_etl.load.load_corpus.run_embedding"),
            patch("scripts.baroque_corpus_etl.load.load_corpus.health_check") as mock_health,
        ):
            mock_health.return_value = {
                "corpus_files": 670, "corpus_metadata": 670,
                "corpus_embeddings": 670, "embedded": 668, "failed": 2,
            }
            with pytest.raises(SystemExit) as exc:
                run_pipeline(
                    execute=True, stage=None, filter_collection=None,
                    limit=None, silver_root=Path("data/silver"),
                )
            assert exc.value.code == 1