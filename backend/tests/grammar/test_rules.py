"""
Tests for :mod:`src.grammar.rules` — the Baroque grammar constraints.

No cache required. We build a tiny fake :class:`TransitionMatrix` where
every probability query returns a known constant, which makes the
weight arithmetic verifiable by hand.
"""

from __future__ import annotations

from typing import Dict

import pytest

from src.grammar.rules import (
    CADENCE_INTERVAL_MEASURES,
    DISCOURAGED_PROGRESSIONS,
    FINAL_CADENCE_WINDOW,
    FORBIDDEN_PROGRESSIONS,
    BaroqueGrammar,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeMatrix:
    """Stand-in for :class:`TransitionMatrix` with a constant probability.

    Keeps tests fast and deterministic: we want to assert weight
    *arithmetic*, not corpus-specific probabilities.
    """

    def __init__(self, constant: float = 0.25) -> None:
        self.constant = constant

    def probability(self, mode: str, from_rn: str, to_rn: str) -> float:
        return self.constant


@pytest.fixture
def grammar() -> BaroqueGrammar:
    return BaroqueGrammar()


@pytest.fixture
def matrix() -> _FakeMatrix:
    return _FakeMatrix(0.25)


# ---------------------------------------------------------------------------
# TestIsValidTransition
# ---------------------------------------------------------------------------


class TestIsValidTransition:
    @pytest.mark.parametrize("pair", list(FORBIDDEN_PROGRESSIONS))
    def test_forbidden_pair_rejected(self, grammar: BaroqueGrammar, pair):
        assert grammar.is_valid_transition(pair[0], pair[1], "major") is False

    def test_valid_pair_accepted(self, grammar: BaroqueGrammar):
        assert grammar.is_valid_transition("V", "I", "major") is True
        assert grammar.is_valid_transition("IV", "V", "major") is True

    def test_discouraged_is_still_valid(self, grammar: BaroqueGrammar):
        # Discouraged is different from forbidden — the rule is soft.
        for pair in DISCOURAGED_PROGRESSIONS:
            assert grammar.is_valid_transition(pair[0], pair[1], "major") is True


# ---------------------------------------------------------------------------
# TestTransitionWeight
# ---------------------------------------------------------------------------


class TestTransitionWeight:
    def test_forbidden_returns_zero(self, grammar: BaroqueGrammar, matrix):
        # V7→IV is forbidden.
        assert grammar.transition_weight("V7", "IV", "major", matrix) == 0.0

    def test_valid_returns_matrix_probability(self, grammar: BaroqueGrammar, matrix):
        assert grammar.transition_weight("V", "I", "major", matrix) == matrix.constant

    def test_discouraged_returns_penalised_probability(
        self, grammar: BaroqueGrammar, matrix
    ):
        w = grammar.transition_weight("V", "IV", "major", matrix)
        assert w == pytest.approx(matrix.constant * 0.01)
        assert 0.0 < w < matrix.constant

    def test_discouraged_less_than_valid(self, grammar: BaroqueGrammar, matrix):
        valid = grammar.transition_weight("V", "I", "major", matrix)
        discouraged = grammar.transition_weight("V", "IV", "major", matrix)
        assert discouraged < valid

    def test_different_key_modes_accepted(
        self, grammar: BaroqueGrammar, matrix
    ):
        # key_mode is passed through; currently the forbidden rule set
        # isn't mode-specific, so both modes should behave the same.
        assert grammar.transition_weight("V7", "IV", "major", matrix) == 0.0
        assert grammar.transition_weight("V7", "iv", "minor", matrix) == 0.0


# ---------------------------------------------------------------------------
# TestRequiresCadenceAt
# ---------------------------------------------------------------------------


class TestRequiresCadenceAt:
    def test_periodic_cadences_at_multiples_of_eight(self, grammar: BaroqueGrammar):
        assert grammar.requires_cadence_at(8, 32) is True
        assert grammar.requires_cadence_at(16, 32) is True
        assert grammar.requires_cadence_at(24, 32) is True

    def test_non_cadence_measures_are_free(self, grammar: BaroqueGrammar):
        for m in (1, 2, 3, 4, 5, 6, 7, 9, 10, 15):
            assert grammar.requires_cadence_at(m, 32) is False

    def test_final_window_always_required(self, grammar: BaroqueGrammar):
        # total = 33: final 2 measures are 32 and 33.
        assert grammar.requires_cadence_at(32, 33) is True
        assert grammar.requires_cadence_at(33, 33) is True

    def test_out_of_range_is_false(self, grammar: BaroqueGrammar):
        assert grammar.requires_cadence_at(0, 16) is False
        assert grammar.requires_cadence_at(-1, 16) is False
        assert grammar.requires_cadence_at(20, 16) is False
        assert grammar.requires_cadence_at(1, 0) is False

    def test_required_measures_lists_every_position(self, grammar: BaroqueGrammar):
        required = grammar.required_cadence_measures(16)
        # At total=16, both the periodic rule (m=8, 16) and the final
        # window (m=15, 16) fire. Measure 15 + 16 + 8 = {8, 15, 16}.
        assert 8 in required
        assert 15 in required
        assert 16 in required
        # Measure 1-7 and 9-14 are free.
        for m in list(range(1, 8)) + list(range(9, 15)):
            assert m not in required

    def test_cadence_interval_constant_is_eight(self):
        # Pin the constant — if someone changes it, a bunch of musical
        # assumptions go with it and tests should notice.
        assert CADENCE_INTERVAL_MEASURES == 8
        assert FINAL_CADENCE_WINDOW == 2


# ---------------------------------------------------------------------------
# TestValidateKeyFrame
# ---------------------------------------------------------------------------


class TestValidateKeyFrame:
    def test_well_formed_major_sequence_has_no_violations(self, grammar: BaroqueGrammar):
        violations = grammar.validate_key_frame(
            ["I", "IV", "V", "I"], key_mode="major"
        )
        assert violations == []

    def test_well_formed_minor_sequence_has_no_violations(self, grammar: BaroqueGrammar):
        violations = grammar.validate_key_frame(
            ["i", "iv", "V", "i"], key_mode="minor"
        )
        assert violations == []

    def test_tonic_inversion_counts_as_tonic(self, grammar: BaroqueGrammar):
        # I6, i64, etc. should satisfy the key frame.
        violations = grammar.validate_key_frame(
            ["I6", "IV", "V", "I"], key_mode="major"
        )
        assert violations == []

    def test_opening_on_dominant_flagged(self, grammar: BaroqueGrammar):
        violations = grammar.validate_key_frame(
            ["V", "I", "IV", "I"], key_mode="major"
        )
        assert len(violations) == 1
        assert "opening" in violations[0]

    def test_closing_off_tonic_flagged(self, grammar: BaroqueGrammar):
        violations = grammar.validate_key_frame(
            ["I", "IV", "V", "IV"], key_mode="major"
        )
        assert len(violations) == 1
        assert "closing" in violations[0]

    def test_both_ends_broken(self, grammar: BaroqueGrammar):
        violations = grammar.validate_key_frame(
            ["V", "IV", "V", "ii"], key_mode="major"
        )
        assert len(violations) == 2

    def test_empty_sequence_has_explicit_violation(self, grammar: BaroqueGrammar):
        violations = grammar.validate_key_frame([], key_mode="major")
        assert len(violations) == 1
        assert "empty" in violations[0]

    def test_unknown_mode_reported(self, grammar: BaroqueGrammar):
        violations = grammar.validate_key_frame(["I", "V", "I"], key_mode="dorian")
        assert len(violations) == 1
        assert "unknown" in violations[0].lower()
