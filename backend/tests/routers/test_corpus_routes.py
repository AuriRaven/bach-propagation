"""
backend/tests/routers/test_corpus_routes.py

Run: uv run pytest tests/routers/ -v

All Supabase and OpenAI calls are mocked — no live credentials needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.corpus import corpus_router, get_supabase


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_ROW = {
    "id": "abc-123",
    "bwv": "BWV 772",
    "movement_name": "Invention No. 1 in C major",
    "collection": "inventions",
    "storage_object_path": "inventions/bwv772.mid",
    "storage_url": "https://example.com/bwv772.mid",
    "load_status": "embedded",
    "key_signature": "C major",
    "time_signature": "4/4",
    "num_measures": 22,
    "num_voices": 2,
    "duration_seconds": 65.0,
    "form_tag": None,
    "instrument_family": "keyboard",
    "raw_metadata": {},
}


def make_mock_supabase(rows=None, single_row=None, count=None):
    rows = rows or [SAMPLE_ROW]
    single_row = single_row or SAMPLE_ROW
    sb = MagicMock()

    # Chainable query builder
    chain = MagicMock()
    chain.execute.return_value = MagicMock(data=rows, count=count or len(rows))
    chain.eq.return_value      = chain
    chain.ilike.return_value   = chain
    chain.range.return_value   = chain
    chain.order.return_value   = chain
    chain.single.return_value.execute.return_value = MagicMock(data=single_row)

    sb.from_.return_value.select.return_value = chain
    sb.from_.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()

    # Storage signed URL
    sb.storage.from_.return_value.create_signed_url.return_value = {
        "signedURL": "https://signed.example.com/bwv772.mid?token=x"
    }

    # RPC
    sb.rpc.return_value.execute.return_value = MagicMock(data=rows)

    return sb


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(corpus_router, prefix="/api")
    mock_sb = make_mock_supabase()
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    with TestClient(app) as c:
        yield c


# ─── /api/corpus/stats ────────────────────────────────────────────────────────

class TestCorpusStats:
    def test_returns_correct_shape(self, client):
        resp = client.get("/api/corpus/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "total" in body
        assert "by_collection" in body
        assert "by_key_mode" in body

    def test_major_key_bucketed(self, client):
        resp = client.get("/api/corpus/stats")
        body = resp.json()
        # SAMPLE_ROW has "C major"
        assert body["by_key_mode"].get("major", 0) >= 1

    def test_collection_counted(self, client):
        resp = client.get("/api/corpus/stats")
        body = resp.json()
        assert body["by_collection"].get("inventions", 0) >= 1


# ─── /api/corpus (list) ───────────────────────────────────────────────────────

class TestListCorpus:
    def test_paginated_shape(self, client):
        resp = client.get("/api/corpus?page=1&page_size=20")
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body and "total" in body
        assert body["page"] == 1

    def test_collection_filter_passes(self, client):
        resp = client.get("/api/corpus?collection=inventions")
        assert resp.status_code == 200

    def test_key_mode_filter_passes(self, client):
        resp = client.get("/api/corpus?key_mode=minor")
        assert resp.status_code == 200


# ─── /api/corpus/:id ──────────────────────────────────────────────────────────

class TestGetCorpusFile:
    def test_returns_signed_url(self, client):
        resp = client.get("/api/corpus/abc-123")
        assert resp.status_code == 200
        body = resp.json()
        assert body["signed_url"] is not None
        assert "signed.example.com" in body["signed_url"]

    def test_404_when_not_found(self):
        app = FastAPI()
        app.include_router(corpus_router, prefix="/api")
        mock_sb = make_mock_supabase()
        mock_sb.from_.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(data=None)
        app.dependency_overrides[get_supabase] = lambda: mock_sb
        with TestClient(app) as c:
            resp = c.get("/api/corpus/nonexistent")
        assert resp.status_code == 404


# ─── /api/corpus/:id/notation ────────────────────────────────────────────────

class TestGetNotation:
    def test_returns_cached_vexflow(self):
        """If raw_metadata already has vexflow key, no music21 parse needed."""
        cached_payload = {
            "measures": [{"index": 0, "start_beat": 0.0, 
                        "end_beat": 4.0, "notes": []}],
            "time_signature": "4/4",
            "key_signature": "C major", 
            "total_beats": 4.0,
        }
        row_with_cache = {**SAMPLE_ROW, "raw_metadata": {"vexflow": cached_payload}}

        app = FastAPI()
        app.include_router(corpus_router, prefix="/api")
        mock_sb = make_mock_supabase(single_row=row_with_cache)
        # corpus_files lookup uses a separate chain
        cf_chain = MagicMock()
        cf_chain.execute.return_value = MagicMock(data=row_with_cache)
        mock_sb.from_.return_value.select.return_value.eq.return_value.single.return_value = cf_chain
        app.dependency_overrides[get_supabase] = lambda: mock_sb

        with TestClient(app) as c:
            resp = c.get("/api/corpus/abc-123/notation")

        assert resp.status_code == 200
        body = resp.json()
        assert body["time_signature"] == "4/4"
        assert body["key_signature"] == "C major"