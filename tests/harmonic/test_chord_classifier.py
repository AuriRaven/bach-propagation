"""
Tests for src/harmonic/chord_classifier.py

All tests use synthetic MusicalEvent lists — no file I/O required.
"""

import pytest
from fractions import Fraction
from typing import List

from src.core.data_structures import Chord, MusicalEvent, Score, TimeSignature
from src.harmonic.chord_classifier import (
    ChordClassificationConfig,
    ChordClassifier,
    _CHORD_TEMPLATES,
)
from src.temporal.windower import AnalysisWindow, WindowType


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
    return MusicalEvent(pitch=None, onset=Fraction(onset), duration=Fraction(dur), is_rest=True)


def _window(events: List[MusicalEvent], start: float = 0, end: float = 4) -> AnalysisWindow:
    return AnalysisWindow(start=Fraction(start), end=Fraction(end), events=events)


def _make_score(events: List[MusicalEvent], dur_quarters: int = 8) -> Score:
    return Score(
        events=events,
        time_signatures=[(Fraction(0), TimeSignature(4, 4))],
    )


# C major window: C4, E4, G4 — all beat_strength=1.0
def _c_major_window() -> AnalysisWindow:
    return _window([_note(60), _note(64), _note(67)])


# G major window: G4, B4, D5
def _g_major_window() -> AnalysisWindow:
    return _window([_note(67), _note(71), _note(74)])


# G dominant 7th: G4, B4, D5, F4
def _g7_window() -> AnalysisWindow:
    return _window([_note(67), _note(71), _note(74), _note(65)])


# Rest-only window
def _rest_window() -> AnalysisWindow:
    return _window([_rest(0, 4)])


# First inversion C major: E2 (bass), G4, C5
def _c_major_first_inv_window() -> AnalysisWindow:
    return _window([_note(52, beat_strength=1.0), _note(67), _note(72)])


# ---------------------------------------------------------------------------
# TestBuildPitchClassProfile
# ---------------------------------------------------------------------------

class TestBuildPitchClassProfile:
    def setup_method(self):
        self.cc = ChordClassifier()
        self.cfg = ChordClassificationConfig()

    def test_returns_twelve_elements(self):
        profile = self.cc._build_pitch_class_profile(_c_major_window(), self.cfg)
        assert len(profile) == 12

    def test_rests_contribute_nothing(self):
        profile = self.cc._build_pitch_class_profile(_rest_window(), self.cfg)
        assert sum(profile) == pytest.approx(0.0)

    def test_beat_strength_weighting(self):
        # Strong beat (1.0) vs weak beat (0.5) on different pitch classes
        w = _window([_note(60, beat_strength=1.0), _note(62, beat_strength=0.5)])
        cfg = ChordClassificationConfig(beat_weight_exponent=1.0)
        profile = self.cc._build_pitch_class_profile(w, cfg)
        assert profile[0] == pytest.approx(1.0)   # C, weight = 1.0 * dur(1)
        assert profile[2] == pytest.approx(0.5)   # D, weight = 0.5 * dur(1)

    def test_exponent_amplifies_strong_beats(self):
        w = _window([_note(60, beat_strength=1.0), _note(62, beat_strength=0.5)])
        cfg_sq = ChordClassificationConfig(beat_weight_exponent=2.0)
        profile = self.cc._build_pitch_class_profile(w, cfg_sq)
        # beat_strength=1.0 squared → 1.0; beat_strength=0.5 squared → 0.25
        assert profile[0] == pytest.approx(1.0)
        assert profile[2] == pytest.approx(0.25)

    def test_empty_window_returns_all_zeros(self):
        w = _window([])
        profile = self.cc._build_pitch_class_profile(w, self.cfg)
        assert all(v == pytest.approx(0.0) for v in profile)

    def test_octave_equivalence(self):
        # C4 and C5 both map to pitch class 0
        w = _window([_note(60), _note(72)])
        profile = self.cc._build_pitch_class_profile(w, self.cfg)
        assert profile[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# TestScoreChord
# ---------------------------------------------------------------------------

class TestScoreChord:
    def setup_method(self):
        self.cc = ChordClassifier()

    def test_perfect_match_returns_raw_sum(self):
        # Profile has all weight on chord tones {0, 4, 7} (C major)
        profile = [0.0] * 12
        profile[0] = 3.0
        profile[4] = 3.0
        profile[7] = 3.0
        score = self.cc._score_chord(profile, 0, _CHORD_TEMPLATES["major"])
        assert score == pytest.approx(9.0)  # raw sum: 3+3+3

    def test_no_overlap_returns_zero(self):
        profile = [0.0] * 12
        profile[1] = 5.0  # C# — not in C major triad
        score = self.cc._score_chord(profile, 0, _CHORD_TEMPLATES["major"])
        assert score == pytest.approx(0.0)

    def test_partial_overlap_returns_partial_sum(self):
        profile = [0.0] * 12
        profile[0] = 2.0  # C — in C major
        profile[1] = 2.0  # C# — not in C major
        score = self.cc._score_chord(profile, 0, _CHORD_TEMPLATES["major"])
        assert score == pytest.approx(2.0)  # only C contributes

    def test_7th_chord_beats_triad_after_total_normalisation(self):
        # Window: G, B, D, F — exactly G dominant7
        profile = [0.0] * 12
        for pc in [7, 11, 2, 5]:  # G, B, D, F
            profile[pc] = 1.0
        total = sum(profile)  # 4.0
        triad_raw = self.cc._score_chord(profile, 7, _CHORD_TEMPLATES["major"])     # 3
        dom7_raw  = self.cc._score_chord(profile, 7, _CHORD_TEMPLATES["dominant7"]) # 4
        assert dom7_raw / total > triad_raw / total


# ---------------------------------------------------------------------------
# TestGetInversion
# ---------------------------------------------------------------------------

class TestGetInversion:
    def setup_method(self):
        self.cc = ChordClassifier()

    def test_root_position_c_major(self):
        # C in bass (MIDI 60 = pitch class 0)
        events = [_note(60), _note(64), _note(67)]
        bass_pc, inv = self.cc._get_inversion(events, 0, _CHORD_TEMPLATES["major"])
        assert bass_pc == 0
        assert inv == 0

    def test_first_inversion_c_major(self):
        # E2 (MIDI 52, pc 4) is lowest — first inversion
        events = [_note(52), _note(60), _note(67)]
        bass_pc, inv = self.cc._get_inversion(events, 0, _CHORD_TEMPLATES["major"])
        assert bass_pc == 4
        assert inv == 1

    def test_second_inversion_c_major(self):
        # G2 (MIDI 43, pc 7) is lowest — second inversion
        events = [_note(43), _note(60), _note(64)]
        bass_pc, inv = self.cc._get_inversion(events, 0, _CHORD_TEMPLATES["major"])
        assert bass_pc == 7
        assert inv == 2

    def test_non_chord_tone_bass_returns_zero_inversion(self):
        # D (pc 2) is not in C major triad → inversion=0
        events = [_note(50), _note(60), _note(64)]  # D3 (50), C4, E4
        bass_pc, inv = self.cc._get_inversion(events, 0, _CHORD_TEMPLATES["major"])
        assert inv == 0

    def test_empty_events_returns_root_position(self):
        bass_pc, inv = self.cc._get_inversion([], 5, _CHORD_TEMPLATES["major"])
        assert bass_pc == 5
        assert inv == 0


# ---------------------------------------------------------------------------
# TestClassifyWindow
# ---------------------------------------------------------------------------

class TestClassifyWindow:
    def setup_method(self):
        self.cc = ChordClassifier()
        self.cfg = ChordClassificationConfig()

    def test_c_major_window_identifies_correctly(self):
        result = self.cc.classify_window(_c_major_window(), self.cfg)
        assert result is not None
        assert result.root == 0
        assert result.quality == "major"

    def test_g_major_window_identifies_correctly(self):
        result = self.cc.classify_window(_g_major_window(), self.cfg)
        assert result is not None
        assert result.root == 7
        assert result.quality == "major"

    def test_g_dominant7_window_identifies_correctly(self):
        result = self.cc.classify_window(_g7_window(), self.cfg)
        assert result is not None
        assert result.root == 7
        assert result.quality == "dominant7"

    def test_rest_only_window_returns_none(self):
        result = self.cc.classify_window(_rest_window(), self.cfg)
        assert result is None

    def test_empty_window_returns_none(self):
        result = self.cc.classify_window(_window([]), self.cfg)
        assert result is None

    def test_below_min_confidence_returns_none(self):
        # Single note window: best score ≈ 1/sqrt(3) ≈ 0.58 for a triad.
        # Setting min_confidence=0.9 causes None to be returned.
        single_note_window = _window([_note(60)])
        cfg = ChordClassificationConfig(min_confidence=0.9)
        result = self.cc.classify_window(single_note_window, cfg)
        assert result is None

    def test_chord_onset_matches_window_start(self):
        w = AnalysisWindow(start=Fraction(8), end=Fraction(12), events=[_note(60), _note(64), _note(67)])
        result = self.cc.classify_window(w, self.cfg)
        assert result is not None
        assert result.onset == Fraction(8)

    def test_chord_duration_matches_window_duration(self):
        w = AnalysisWindow(start=Fraction(0), end=Fraction(3), events=[_note(60), _note(64), _note(67)])
        result = self.cc.classify_window(w, self.cfg)
        assert result is not None
        assert result.duration == Fraction(3)

    def test_seventh_chords_disabled_returns_triad(self):
        cfg = ChordClassificationConfig(include_seventh_chords=False)
        result = self.cc.classify_window(_g7_window(), cfg)
        # With seventh chords disabled, should still find a chord — probably G major
        if result is not None:
            assert result.quality in {"major", "minor", "diminished", "augmented"}

    def test_result_is_chord_instance(self):
        result = self.cc.classify_window(_c_major_window(), self.cfg)
        assert isinstance(result, Chord)

    def test_first_inversion_detected(self):
        result = self.cc.classify_window(_c_major_first_inv_window(), self.cfg)
        assert result is not None
        assert result.root == 0
        assert result.inversion == 1


# ---------------------------------------------------------------------------
# TestClassifyScore
# ---------------------------------------------------------------------------

class TestClassifyScore:
    def setup_method(self):
        self.cc = ChordClassifier()
        self.cfg = ChordClassificationConfig()

    def _c_major_score(self) -> Score:
        """4-measure score, each measure a C major chord."""
        events = []
        for m in range(4):
            offset = m * 4
            events += [_note(60, offset, 4), _note(64, offset, 4), _note(67, offset, 4)]
        return _make_score(events)

    def test_returns_list_of_chords(self):
        score = self._c_major_score()
        chords = self.cc.classify_score(score, WindowType.MEASURE, self.cfg)
        assert isinstance(chords, list)
        assert all(isinstance(c, Chord) for c in chords)

    def test_no_nones_in_result(self):
        score = self._c_major_score()
        chords = self.cc.classify_score(score, WindowType.MEASURE, self.cfg)
        assert None not in chords

    def test_chronological_order(self):
        score = self._c_major_score()
        chords = self.cc.classify_score(score, WindowType.MEASURE, self.cfg)
        onsets = [c.onset for c in chords]
        assert onsets == sorted(onsets)

    def test_measure_windowing_produces_one_chord_per_measure(self):
        score = self._c_major_score()
        chords = self.cc.classify_score(score, WindowType.MEASURE, self.cfg)
        # 4 measures, each containing a clear C major chord → 4 chords
        assert len(chords) == 4
