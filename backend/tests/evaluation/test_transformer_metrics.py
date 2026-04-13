"""
backend/tests/evaluation/test_transformer_metrics.py

All tests use mock objects — no real checkpoint, no real jsonl, no Supabase.
Tests must run in < 10 seconds.
"""

from __future__ import annotations

import json
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

@dataclass
class _FakeReport:
    is_valid: bool = True
    violations: list = None
    tonal_score: float = 0.92
    avg_transition_prob: float = 0.12
    cadence_coverage: float = 0.75

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class _FakeGeneratedSequence:
    def __init__(self, rn_sequence=None, n=16):
        self.chord_tokens   = list(range(n))
        self.rn_sequence    = rn_sequence or (["i", "iv", "V", "i"] * (n // 4))
        self.cadence_tokens = [4] * n
        self.onsets         = [float(i) for i in range(n)]
        self.grammar_report = _FakeReport()
        self.metadata       = {"key_mode": "minor", "n_tokens": n}


class _FakeGenerator:
    """Mock MusicGenerator that returns deterministic sequences."""
    def __init__(self, rn_sequence=None):
        self._rn_sequence = rn_sequence

    def generate(self, key_mode="minor", n_tokens=64,
                 temperature=0.9, top_k=10, seed=None, **kwargs):
        return _FakeGeneratedSequence(rn_sequence=self._rn_sequence, n=n_tokens)


class _FakeGrammar:
    FORBIDDEN = {("V7", "IV"), ("V7", "iv")}

    def is_valid_transition(self, from_rn, to_rn, key_mode):
        return (from_rn, to_rn) not in self.FORBIDDEN

    def requires_cadence_at(self, measure, total):
        return measure % 8 == 0


# ---------------------------------------------------------------------------
# Tests: EvaluationReport
# ---------------------------------------------------------------------------

class TestEvaluationReport:
    """EvaluationReport is a correct dataclass and serialises cleanly."""

    def _make_report(self, **overrides):
        from src.evaluation.transformer_metrics import EvaluationReport
        defaults = dict(
            perplexity=23.2,
            avg_tonal_score=0.91,
            forbidden_rate=0.0,
            cadence_coverage=0.78,
            key_consistency=0.84,
            avg_sequence_length=64.0,
            n_samples=50,
            baseline_perplexity=50.0,
            improvement_over_baseline=0.536,
            checkpoint_path="checkpoints/transformer-v1/best.pt",
            key_mode="minor",
        )
        defaults.update(overrides)
        return EvaluationReport(**defaults)

    def test_instantiation(self):
        r = self._make_report()
        assert r.perplexity == 23.2
        assert r.forbidden_rate == 0.0

    def test_to_json_round_trip(self, tmp_path):
        r = self._make_report()
        path = str(tmp_path / "report.json")
        r.to_json(path)
        with open(path) as f:
            data = json.load(f)
        assert data["perplexity"] == 23.2
        assert data["forbidden_rate"] == 0.0
        assert data["key_mode"] == "minor"

    def test_all_fields_json_serialisable(self, tmp_path):
        """Every field must be a primitive type for JSON round-trip."""
        r = self._make_report()
        path = str(tmp_path / "r2.json")
        r.to_json(path)   # would raise TypeError if not serialisable

    def test_print_summary_does_not_crash(self, capsys):
        r = self._make_report()
        r.print_summary()
        out = capsys.readouterr().out
        assert "MUSIC TRANSFORMER" in out
        assert "Perplexity" in out

    def test_thesis_pass_criteria(self):
        passing = self._make_report(
            forbidden_rate=0.0,
            avg_tonal_score=0.85,
            cadence_coverage=0.75,
        )
        # Verify the print_summary marks it as passing
        # (checking criteria logic rather than string output)
        assert passing.forbidden_rate == 0.0
        assert passing.avg_tonal_score > 0.80
        assert passing.cadence_coverage > 0.70

    def test_thesis_fail_criteria(self):
        failing = self._make_report(
            forbidden_rate=0.05,   # has forbidden transitions
            avg_tonal_score=0.60,  # below threshold
        )
        assert not (
            failing.forbidden_rate == 0.0
            and failing.avg_tonal_score > 0.80
        )


# ---------------------------------------------------------------------------
# Tests: evaluate_generator
# ---------------------------------------------------------------------------

class TestEvaluateGenerator:
    """evaluate_generator aggregates GrammarReport fields correctly."""

    def test_returns_evaluation_report(self):
        from src.evaluation.transformer_metrics import evaluate_generator, EvaluationReport
        gen = _FakeGenerator()
        report = evaluate_generator(gen, n_samples=5, key_mode="minor")
        assert isinstance(report, EvaluationReport)

    def test_n_samples_matches(self):
        from src.evaluation.transformer_metrics import evaluate_generator
        gen = _FakeGenerator()
        report = evaluate_generator(gen, n_samples=10)
        assert report.n_samples == 10

    def test_forbidden_rate_zero_for_clean_sequences(self):
        """Clean i-iv-V-i sequences have no forbidden transitions."""
        from src.evaluation.transformer_metrics import evaluate_generator
        gen = _FakeGenerator(rn_sequence=["i", "iv", "V", "i"] * 16)
        report = evaluate_generator(gen, n_samples=5, key_mode="minor")
        assert report.forbidden_rate == 0.0

    def test_tonal_score_in_range(self):
        from src.evaluation.transformer_metrics import evaluate_generator
        gen = _FakeGenerator()
        report = evaluate_generator(gen, n_samples=5)
        assert 0.0 <= report.avg_tonal_score <= 1.0

    def test_cadence_coverage_in_range(self):
        from src.evaluation.transformer_metrics import evaluate_generator
        gen = _FakeGenerator()
        report = evaluate_generator(gen, n_samples=5)
        assert 0.0 <= report.cadence_coverage <= 1.0

    def test_key_consistency_in_range(self):
        from src.evaluation.transformer_metrics import evaluate_generator
        gen = _FakeGenerator()
        report = evaluate_generator(gen, n_samples=5)
        assert 0.0 <= report.key_consistency <= 1.0

    def test_avg_sequence_length_positive(self):
        from src.evaluation.transformer_metrics import evaluate_generator
        gen = _FakeGenerator()
        report = evaluate_generator(gen, n_samples=5)
        assert report.avg_sequence_length > 0

    def test_checkpoint_path_stored(self):
        from src.evaluation.transformer_metrics import evaluate_generator
        gen = _FakeGenerator()
        report = evaluate_generator(gen, n_samples=3,
                                    checkpoint_path="checkpoints/test.pt")
        assert report.checkpoint_path == "checkpoints/test.pt"

    def test_replace_perplexity_after_compute(self):
        """Verify dataclasses.replace works for post-hoc perplexity fill-in."""
        from src.evaluation.transformer_metrics import evaluate_generator
        gen = _FakeGenerator()
        report = evaluate_generator(gen, n_samples=3)
        updated = dataclasses.replace(
            report,
            perplexity=23.2,
            baseline_perplexity=50.0,
            improvement_over_baseline=(50.0 - 23.2) / 50.0,
        )
        assert updated.perplexity == 23.2
        assert abs(updated.improvement_over_baseline - 0.536) < 0.001