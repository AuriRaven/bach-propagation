"""
backend/tests/routers/test_generation.py

FastAPI endpoint tests for POST /api/generate.
All tests use a mock generator injected via module-level patching.
No real checkpoint loaded.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

@dataclass
class _FakeViolation:
    position: int = 0
    from_rn: str = "V7"
    to_rn: str = "IV"
    violation_type: str = "forbidden"
    severity: float = 1.0


@dataclass
class _FakeGrammarReport:
    is_valid: bool = True
    violations: list = None
    tonal_score: float = 0.91
    avg_transition_prob: float = 0.12
    cadence_coverage: float = 0.78

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


@dataclass
class _FakeGeneratedSequence:
    chord_tokens:   list
    rn_sequence:    list
    cadence_tokens: list
    onsets:         list
    grammar_report: _FakeGrammarReport
    metadata:       dict


def _make_seq(n=16):
    return _FakeGeneratedSequence(
        chord_tokens=list(range(n)),
        rn_sequence=(["i", "iv", "V", "i"] * (n // 4)),
        cadence_tokens=[4] * n,
        onsets=[float(i) for i in range(n)],
        grammar_report=_FakeGrammarReport(),
        metadata={"key_mode": "minor", "n_tokens": n},
    )


# Minimal valid MusicXML (base64-encoded for the mock)
_FAKE_XML = b'<?xml version="1.0"?><score-partwise version="3.1"><part id="P1"><measure number="1"></measure></part></score-partwise>'
_FAKE_XML_B64 = base64.b64encode(_FAKE_XML).decode("utf-8")


def _make_mock_generator():
    gen = MagicMock()
    gen.generate.return_value = _make_seq()
    gen.generate_musicxml.return_value = _FAKE_XML_B64
    return gen


# ---------------------------------------------------------------------------
# App fixture — injects mock generator
# ---------------------------------------------------------------------------

@pytest.fixture()
def client_with_generator():
    """TestClient with a mock generator loaded."""
    import routers.generation as gen_module

    original = gen_module._generator
    gen_module._generator = _make_mock_generator()
    try:
        # Import app after patching so lifespan doesn't try to load real checkpoint
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from routers.generation import generation_router

        test_app = FastAPI()
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )
        test_app.include_router(generation_router, prefix="/api")

        with TestClient(test_app) as c:
            yield c
    finally:
        gen_module._generator = original


@pytest.fixture()
def client_no_generator():
    """TestClient with _generator = None (checkpoint missing scenario)."""
    import routers.generation as gen_module

    original = gen_module._generator
    gen_module._generator = None
    try:
        from fastapi import FastAPI
        from routers.generation import generation_router

        test_app = FastAPI()
        test_app.include_router(generation_router, prefix="/api")

        with TestClient(test_app) as c:
            yield c
    finally:
        gen_module._generator = original


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------

class TestGenerateEndpointSuccess:

    def test_returns_200(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={})
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={})
        body = resp.json()
        required = {
            "chord_tokens", "rn_sequence", "tonal_score",
            "is_valid", "forbidden_rate", "musicxml_b64", "generation_time_ms",
        }
        assert required.issubset(body.keys())

    def test_tonal_score_in_range(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={})
        body = resp.json()
        assert 0.0 <= body["tonal_score"] <= 1.0

    def test_forbidden_rate_in_range(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={})
        body = resp.json()
        assert 0.0 <= body["forbidden_rate"] <= 1.0

    def test_musicxml_b64_decodes_to_valid_xml(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={})
        body = resp.json()
        xml_bytes = base64.b64decode(body["musicxml_b64"])
        xml_str = xml_bytes.decode("utf-8")
        assert "<score-partwise" in xml_str

    def test_is_valid_is_bool(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={})
        body = resp.json()
        assert isinstance(body["is_valid"], bool)

    def test_generation_time_ms_positive(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={})
        body = resp.json()
        assert body["generation_time_ms"] >= 0.0

    def test_chord_tokens_non_empty(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={"n_tokens": 16})
        body = resp.json()
        assert len(body["chord_tokens"]) > 0

    def test_rn_sequence_non_empty(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={})
        body = resp.json()
        assert len(body["rn_sequence"]) > 0


# ---------------------------------------------------------------------------
# Tests: 503 when generator not loaded
# ---------------------------------------------------------------------------

class TestGenerateEndpoint503:

    def test_returns_503_when_no_generator(self, client_no_generator):
        resp = client_no_generator.post("/api/generate", json={})
        assert resp.status_code == 503

    def test_503_detail_mentions_checkpoint(self, client_no_generator):
        resp = client_no_generator.post("/api/generate", json={})
        body = resp.json()
        assert "checkpoint" in body["detail"].lower()


# ---------------------------------------------------------------------------
# Tests: request validation
# ---------------------------------------------------------------------------

class TestGenerateRequestValidation:

    def test_n_tokens_too_small_returns_422(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={"n_tokens": 5})
        assert resp.status_code == 422

    def test_n_tokens_too_large_returns_422(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={"n_tokens": 1000})
        assert resp.status_code == 422

    def test_temperature_too_low_returns_422(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={"temperature": 0.0})
        assert resp.status_code == 422

    def test_temperature_too_high_returns_422(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={"temperature": 5.0})
        assert resp.status_code == 422

    def test_invalid_key_mode_returns_422(self, client_with_generator):
        resp = client_with_generator.post("/api/generate",
                                          json={"key_mode": "dorian"})
        assert resp.status_code == 422

    def test_valid_major_key_mode(self, client_with_generator):
        resp = client_with_generator.post("/api/generate",
                                          json={"key_mode": "major"})
        assert resp.status_code == 200

    def test_defaults_accepted(self, client_with_generator):
        """Empty body uses all defaults — must return 200."""
        resp = client_with_generator.post("/api/generate", json={})
        assert resp.status_code == 200

    def test_top_k_too_small_returns_422(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={"top_k": 0})
        assert resp.status_code == 422

    def test_top_k_too_large_returns_422(self, client_with_generator):
        resp = client_with_generator.post("/api/generate", json={"top_k": 100})
        assert resp.status_code == 422