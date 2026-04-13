"""
Tests for :mod:`src.grammar.validator` — post-generation scoring.

Real :class:`TransitionMatrix` and :class:`BaroqueGrammar` instances are
used so the validator's integration with those modules stays honest.
The matrix is built from a small synthetic corpus that produces
reasonable probabilities for a clean I–IV–V–I cadence.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from src.corpus.transition_matrix import TransitionMatrix
from src.grammar.rules import BaroqueGrammar
from src.grammar.validator import (
    GrammarReport,
    GrammarValidator,
    GrammarViolation,
    LOW_PROBABILITY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _analysis(rns: List[str], mode: str = "major") -> Dict[str, object]:
    return {
        "score_id": f"t-{mode}",
        "global_key": {"tonic": 0, "mode": mode, "confidence": 1.0, "label": mode},
        "chords": [],
        "roman_numerals": [
            {
                "label": rn,
                "degree": 1,
                "quality": "major",
                "inversion": 0,
                "local_key_tonic": 0,
                "local_key_mode": mode,
                "is_secondary": False,
                "secondary_target": None,
                "function": "TONIC",
            }
            for rn in rns
        ],
        "cadences": [],
    }


@pytest.fixture
def trained_matrix() -> TransitionMatrix:
    """Matrix trained on a dozen I–IV–V–I–style chorales.

    The transitions I→IV, IV→V, V→I dominate so the geometric-mean
    probability on a Bach-like sequence is healthy.
    """
    m = TransitionMatrix()
    m.build_from_analyses([
        _analysis(["I", "IV", "V", "I"]),
        _analysis(["I", "IV", "V", "I"]),
        _analysis(["I", "vi", "IV", "V", "I"]),
        _analysis(["I", "ii", "V", "I"]),
        _analysis(["I", "V", "I", "IV", "V", "I"]),
        _analysis(["I", "IV", "ii", "V", "I"]),
        _analysis(["i", "iv", "V", "i"], mode="minor"),
        _analysis(["i", "iv", "V", "i"], mode="minor"),
        _analysis(["i", "VI", "iv", "V", "i"], mode="minor"),
    ])
    return m


@pytest.fixture
def grammar() -> BaroqueGrammar:
    return BaroqueGrammar()


@pytest.fixture
def validator() -> GrammarValidator:
    return GrammarValidator()


# ---------------------------------------------------------------------------
# TestCleanSequence
# ---------------------------------------------------------------------------


class TestCleanSequence:
    def test_clean_major_sequence_is_valid(
        self, validator, grammar, trained_matrix
    ):
        report = validator.validate(
            ["I", "IV", "V", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
        )
        assert report.is_valid
        assert report.violations == []
        # Every transition was observed → score should be near the top.
        assert report.tonal_score > 0.8

    def test_clean_minor_sequence_is_valid(
        self, validator, grammar, trained_matrix
    ):
        report = validator.validate(
            ["i", "iv", "V", "i"],
            key_mode="minor",
            matrix=trained_matrix,
            grammar=grammar,
        )
        assert report.is_valid
        assert report.tonal_score > 0.8

    def test_fully_valid_report_shape(
        self, validator, grammar, trained_matrix
    ):
        report = validator.validate(
            ["I", "IV", "V", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
        )
        assert isinstance(report, GrammarReport)
        assert 0.0 <= report.tonal_score <= 1.0
        assert 0.0 <= report.avg_transition_prob <= 1.0
        assert report.cadence_coverage == 1.0  # no cadence data supplied


# ---------------------------------------------------------------------------
# TestForbiddenProgressions
# ---------------------------------------------------------------------------


class TestForbiddenProgressions:
    def test_single_forbidden_progression_fails(
        self, validator, grammar, trained_matrix
    ):
        # V7 → IV is in FORBIDDEN_PROGRESSIONS.
        report = validator.validate(
            ["I", "V7", "IV", "V", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
        )
        assert report.is_valid is False
        assert report.tonal_score < 1.0
        forbidden = [v for v in report.violations if v.violation_type == "forbidden"]
        assert len(forbidden) == 1
        assert forbidden[0].from_rn == "V7"
        assert forbidden[0].to_rn == "IV"
        assert forbidden[0].severity == 1.0
        assert forbidden[0].position == 1  # V7 is at index 1

    def test_multiple_forbidden_progressions_all_reported(
        self, validator, grammar, trained_matrix
    ):
        report = validator.validate(
            ["I", "V7", "IV", "V7", "iv", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
        )
        forbidden = [v for v in report.violations if v.violation_type == "forbidden"]
        assert len(forbidden) == 2

    def test_clean_sequence_has_no_forbidden_violations(
        self, validator, grammar, trained_matrix
    ):
        report = validator.validate(
            ["I", "IV", "V", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
        )
        assert not any(v.violation_type == "forbidden" for v in report.violations)


# ---------------------------------------------------------------------------
# TestCadenceCoverage
# ---------------------------------------------------------------------------


class TestCadenceCoverage:
    def test_missing_cadence_reported(
        self, validator, grammar, trained_matrix
    ):
        # 16-bar piece → required cadences at m=8, m=15, m=16. We
        # supply only the ones at 8 and 16, leaving m=15 missing.
        report = validator.validate(
            ["I", "IV", "V", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
            cadence_measures=[8, 16],
            total_measures=16,
        )
        missing = [v for v in report.violations if v.violation_type == "missing_cadence"]
        assert len(missing) == 1
        assert missing[0].position == 15
        assert report.is_valid is False
        assert 0.0 < report.cadence_coverage < 1.0

    def test_all_cadences_present_coverage_is_one(
        self, validator, grammar, trained_matrix
    ):
        report = validator.validate(
            ["I", "IV", "V", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
            cadence_measures=[8, 15, 16],
            total_measures=16,
        )
        assert report.cadence_coverage == 1.0
        assert not any(v.violation_type == "missing_cadence" for v in report.violations)

    def test_no_metric_info_defaults_to_full_coverage(
        self, validator, grammar, trained_matrix
    ):
        # When total_measures and cadence_measures are both None (the
        # common "score a raw progression" case), the validator skips
        # cadence checks entirely.
        report = validator.validate(
            ["I", "IV", "V", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
        )
        assert report.cadence_coverage == 1.0


# ---------------------------------------------------------------------------
# TestKeyFrameViolations
# ---------------------------------------------------------------------------


class TestKeyFrameViolations:
    def test_opening_off_tonic_is_hard_violation(
        self, validator, grammar, trained_matrix
    ):
        report = validator.validate(
            ["V", "I", "IV", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
        )
        key = [v for v in report.violations if v.violation_type == "key_frame"]
        assert len(key) == 1
        assert key[0].severity == 1.0
        assert report.is_valid is False

    def test_both_endpoints_off_tonic(
        self, validator, grammar, trained_matrix
    ):
        report = validator.validate(
            ["V", "I", "IV", "V"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
        )
        key = [v for v in report.violations if v.violation_type == "key_frame"]
        assert len(key) == 2


# ---------------------------------------------------------------------------
# TestLowProbabilityViolations
# ---------------------------------------------------------------------------


class TestLowProbabilityViolations:
    def test_unusual_transition_flagged_without_invalidating(
        self, validator, grammar, trained_matrix
    ):
        # Insert a plausible but rare transition: I → ii6 was never
        # observed in the fixture corpus.
        report = validator.validate(
            ["I", "ii6", "V", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
        )
        low = [v for v in report.violations if v.violation_type == "low_probability"]
        # Each such violation is soft (severity < 1.0).
        for v in low:
            assert 0.0 < v.severity < 1.0

    def test_severity_scales_with_probability(self, validator):
        from src.grammar.validator import _soft_severity
        # Very low probability → severity near the ceiling.
        assert _soft_severity(1e-6) > _soft_severity(LOW_PROBABILITY_THRESHOLD * 0.9)


# ---------------------------------------------------------------------------
# TestReportAggregation
# ---------------------------------------------------------------------------


class TestReportAggregation:
    def test_forbidden_drags_score_down(
        self, validator, grammar, trained_matrix
    ):
        clean = validator.validate(
            ["I", "IV", "V", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
        )
        dirty = validator.validate(
            ["I", "V7", "IV", "V", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
        )
        assert dirty.tonal_score < clean.tonal_score

    def test_summary_is_formattable(self, validator, grammar, trained_matrix):
        report = validator.validate(
            ["I", "IV", "V", "I"],
            key_mode="major",
            matrix=trained_matrix,
            grammar=grammar,
        )
        s = report.summary()
        assert "valid=" in s
        assert "score=" in s
