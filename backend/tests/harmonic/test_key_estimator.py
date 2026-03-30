"""
Tests for src/harmonic/key_estimator.py

All tests use synthetic MusicalEvent lists — no file I/O required.
"""

import math
import pytest
from fractions import Fraction
from typing import List

from src.core.data_structures import KeyEstimate, MusicalEvent, Score, TimeSignature
from src.harmonic.key_estimator import KeyEstimator, _KS_MAJOR, _KS_MINOR


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _note(pitch: int, onset: float = 0, dur: float = 1, beat_strength: float = 1.0) -> MusicalEvent:
    return MusicalEvent(
        pitch=pitch,
        onset=Fraction(onset),
        duration=Fraction(dur),
        beat_strength=beat_strength,
    )


def _rest(onset: float = 0, dur: float = 1) -> MusicalEvent:
    return MusicalEvent(
        pitch=None,
        onset=Fraction(onset),
        duration=Fraction(dur),
        is_rest=True,
    )


def _c_major_events() -> List[MusicalEvent]:
    """C, E, G, C — strong C-major triad."""
    return [
        _note(60, 0, 2),  # C4, 2 quarters
        _note(64, 2, 2),  # E4
        _note(67, 4, 2),  # G4
        _note(72, 6, 2),  # C5
    ]


def _g_major_events() -> List[MusicalEvent]:
    """G, B, D, G — G major."""
    return [
        _note(67, 0, 2),  # G4
        _note(71, 2, 2),  # B4
        _note(74, 4, 2),  # D5
        _note(79, 6, 2),  # G5
    ]


def _a_minor_events() -> List[MusicalEvent]:
    """A, C, E, A — A minor (pitch-class distribution biased toward A minor)."""
    return [
        _note(69, 0, 3),  # A4, 3 quarters (heavier)
        _note(72, 3, 1),  # C5
        _note(76, 4, 1),  # E5
        _note(57, 5, 3),  # A3, 3 quarters (heavier)
    ]


def _chromatic_events() -> List[MusicalEvent]:
    """All 12 pitch classes with equal duration — ambiguous, low confidence."""
    return [_note(60 + i, i, 1) for i in range(12)]


def _make_score(events: List[MusicalEvent]) -> Score:
    return Score(
        events=events,
        time_signatures=[(Fraction(0), TimeSignature(4, 4))],
    )


# ---------------------------------------------------------------------------
# TestBuildHistogram
# ---------------------------------------------------------------------------

class TestBuildHistogram:
    def setup_method(self):
        self.ke = KeyEstimator()

    def test_returns_twelve_elements(self):
        hist = self.ke._build_histogram([])
        assert len(hist) == 12

    def test_empty_events_returns_all_zeros(self):
        hist = self.ke._build_histogram([])
        assert all(v == 0.0 for v in hist)

    def test_rests_are_skipped(self):
        events = [_rest(0, 4)]
        hist = self.ke._build_histogram(events)
        assert sum(hist) == 0.0

    def test_none_pitch_is_skipped(self):
        ev = MusicalEvent(pitch=None, onset=Fraction(0), duration=Fraction(1), is_rest=True)
        hist = self.ke._build_histogram([ev])
        assert sum(hist) == 0.0

    def test_single_note_all_weight_in_one_bin(self):
        hist = self.ke._build_histogram([_note(60, 0, 2)])  # C4, dur=2
        assert hist[0] == pytest.approx(2.0)
        assert sum(hist) == pytest.approx(2.0)

    def test_duration_weighting(self):
        # C dur=2, G dur=1 → bin[0]=2.0, bin[7]=1.0
        hist = self.ke._build_histogram([_note(60, 0, 2), _note(67, 2, 1)])
        assert hist[0] == pytest.approx(2.0)
        assert hist[7] == pytest.approx(1.0)

    def test_octave_equivalence(self):
        # C4 (60) and C5 (72) both map to pitch class 0
        hist = self.ke._build_histogram([_note(60, 0, 1), _note(72, 1, 1)])
        assert hist[0] == pytest.approx(2.0)
        assert sum(hist) == pytest.approx(2.0)

    def test_multiple_pitch_classes(self):
        events = _c_major_events()  # C, E, G, C — pitch classes 0, 4, 7
        hist = self.ke._build_histogram(events)
        assert hist[0] == pytest.approx(4.0)  # 2+2 quarters
        assert hist[4] == pytest.approx(2.0)
        assert hist[7] == pytest.approx(2.0)
        assert all(hist[i] == 0.0 for i in range(12) if i not in (0, 4, 7))


# ---------------------------------------------------------------------------
# TestCorrelate
# ---------------------------------------------------------------------------

class TestCorrelate:
    def setup_method(self):
        self.ke = KeyEstimator()

    def test_identical_vectors_return_one(self):
        r = self.ke._correlate(_KS_MAJOR, _KS_MAJOR)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_result_between_minus_one_and_one(self):
        import random
        random.seed(42)
        h = [random.random() for _ in range(12)]
        r = self.ke._correlate(h, _KS_MAJOR)
        assert -1.0 <= r <= 1.0

    def test_zero_variance_histogram_returns_zero(self):
        constant = [5.0] * 12  # no variance
        r = self.ke._correlate(constant, _KS_MAJOR)
        assert r == pytest.approx(0.0)

    def test_negated_profile_returns_negative(self):
        # Negate each value of a profile about its mean
        p_mean = sum(_KS_MAJOR) / 12
        negated = [2 * p_mean - v for v in _KS_MAJOR]
        r = self.ke._correlate(_KS_MAJOR, negated)
        assert r < 0.0


# ---------------------------------------------------------------------------
# TestEstimate
# ---------------------------------------------------------------------------

class TestEstimate:
    def setup_method(self):
        self.ke = KeyEstimator()

    def test_c_major_events_returns_c_major(self):
        result = self.ke.estimate(_c_major_events(), Fraction(0))
        assert result.tonic == 0
        assert result.mode == "major"

    def test_g_major_events_returns_g_major(self):
        result = self.ke.estimate(_g_major_events(), Fraction(0))
        assert result.tonic == 7
        assert result.mode == "major"

    def test_a_minor_events_returns_a_minor(self):
        result = self.ke.estimate(_a_minor_events(), Fraction(0))
        assert result.tonic == 9
        assert result.mode == "minor"

    def test_onset_stored_in_result(self):
        onset = Fraction(8)
        result = self.ke.estimate(_c_major_events(), onset)
        assert result.onset == onset

    def test_confidence_between_zero_and_one(self):
        result = self.ke.estimate(_c_major_events(), Fraction(0))
        assert 0.0 <= result.confidence <= 1.0

    def test_chromatic_events_low_confidence(self):
        result = self.ke.estimate(_chromatic_events(), Fraction(0))
        assert result.confidence < 0.7

    def test_empty_events_returns_valid_key_estimate(self):
        result = self.ke.estimate([], Fraction(0))
        assert isinstance(result, KeyEstimate)
        assert result.confidence == pytest.approx(0.0)

    def test_rest_only_returns_low_confidence(self):
        result = self.ke.estimate([_rest(0, 4)], Fraction(0))
        assert result.confidence == pytest.approx(0.0)

    def test_strong_key_has_high_confidence(self):
        # Many notes heavily biased toward C major
        events = _c_major_events() * 4
        result = self.ke.estimate(events, Fraction(0))
        assert result.confidence > 0.7

    def test_result_is_key_estimate_instance(self):
        result = self.ke.estimate(_c_major_events(), Fraction(0))
        assert isinstance(result, KeyEstimate)


# ---------------------------------------------------------------------------
# TestEstimateGlobal
# ---------------------------------------------------------------------------

class TestEstimateGlobal:
    def setup_method(self):
        self.ke = KeyEstimator()

    def test_onset_is_zero(self):
        score = _make_score(_c_major_events())
        result = self.ke.estimate_global(score)
        assert result.onset == Fraction(0)

    def test_c_major_score_identifies_correctly(self):
        score = _make_score(_c_major_events())
        result = self.ke.estimate_global(score)
        assert result.tonic == 0
        assert result.mode == "major"

    def test_g_major_score_identifies_correctly(self):
        score = _make_score(_g_major_events())
        result = self.ke.estimate_global(score)
        assert result.tonic == 7
        assert result.mode == "major"


# ---------------------------------------------------------------------------
# TestEstimateLocal
# ---------------------------------------------------------------------------

class TestEstimateLocal:
    def setup_method(self):
        self.ke = KeyEstimator()

    def _long_score(self) -> Score:
        """16-quarter score of C major events."""
        events = [_note(60 + i % 7, i, 1) for i in range(16)]
        return _make_score(events)

    def test_returns_list_of_key_estimates(self):
        score = self._long_score()
        results = self.ke.estimate_local(score, Fraction(4), Fraction(4))
        assert isinstance(results, list)
        assert all(isinstance(r, KeyEstimate) for r in results)

    def test_count_matches_sliding_windows(self):
        from src.temporal.windower import Windower
        score = self._long_score()
        size = Fraction(4)
        hop = Fraction(4)
        windows = Windower().sliding(score, size, hop)
        results = self.ke.estimate_local(score, size, hop)
        assert len(results) == len(windows)

    def test_each_estimate_onset_matches_window_start(self):
        from src.temporal.windower import Windower
        score = self._long_score()
        size = Fraction(4)
        hop = Fraction(4)
        windows = Windower().sliding(score, size, hop)
        results = self.ke.estimate_local(score, size, hop)
        for w, r in zip(windows, results):
            assert r.onset == w.start

    def test_empty_score_returns_empty_list(self):
        score = _make_score([])
        results = self.ke.estimate_local(score, Fraction(4), Fraction(4))
        assert results == []
