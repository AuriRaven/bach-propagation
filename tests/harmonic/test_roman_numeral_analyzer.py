"""
Tests for src/harmonic/roman_numeral_analyzer.py

All tests use synthetic Chord and KeyEstimate objects — no file I/O required.
"""

import pytest
from fractions import Fraction
from typing import Optional

from src.core.data_structures import (
    Chord,
    HarmonicFunction,
    KeyEstimate,
    RomanNumeral,
)
from src.harmonic.roman_numeral_analyzer import (
    RomanNumeralAnalyzer,
    _MAJOR_INTERVAL_TO_DEGREE,
    _MINOR_INTERVAL_TO_DEGREE,
)
from src.harmonic.chord_classifier import _CHORD_TEMPLATES


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _key(tonic: int, mode: str = "major", confidence: float = 0.9) -> KeyEstimate:
    return KeyEstimate(tonic=tonic, mode=mode, confidence=confidence, onset=Fraction(0))


def _chord(
    root: int,
    quality: str,
    inversion: int = 0,
    onset: float = 0,
    dur: float = 1,
) -> Chord:
    intervals = _CHORD_TEMPLATES[quality]
    pitches = frozenset((root + i) % 12 for i in intervals)
    bass = root  # simplified: use root as bass (tests inversion separately)
    return Chord(
        onset=Fraction(onset),
        duration=Fraction(dur),
        root=root,
        quality=quality,
        pitches=pitches,
        bass=bass,
        inversion=inversion,
    )


C_MAJOR = _key(0, "major")
A_MINOR = _key(9, "minor")
G_MAJOR = _key(7, "major")
D_MINOR = _key(2, "minor")

RNA = RomanNumeralAnalyzer()


# ---------------------------------------------------------------------------
# TestIntervalToDegree
# ---------------------------------------------------------------------------

class TestIntervalToDegree:
    def test_major_all_diatonic(self):
        expected = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
        for interval, degree in expected.items():
            assert RNA._interval_to_degree(interval, "major") == degree

    def test_major_chromatic_returns_none(self):
        chromatic = [1, 3, 6, 8, 10]
        for interval in chromatic:
            assert RNA._interval_to_degree(interval, "major") is None

    def test_minor_all_diatonic(self):
        expected = {0: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6, 10: 7, 11: 7}
        for interval, degree in expected.items():
            assert RNA._interval_to_degree(interval, "minor") == degree

    def test_minor_harmonic_leading_tone(self):
        # interval 11 (maj7) → degree 7 (harmonic leading tone)
        assert RNA._interval_to_degree(11, "minor") == 7

    def test_minor_chromatic_returns_none(self):
        # Intervals not in the minor table
        chromatic = [1, 4, 6, 9]
        for interval in chromatic:
            assert RNA._interval_to_degree(interval, "minor") is None


# ---------------------------------------------------------------------------
# TestAnalyzeBasicDiatonic
# ---------------------------------------------------------------------------

class TestAnalyzeBasicDiatonic:
    def test_I_in_c_major(self):
        rn = RNA.analyze(_chord(0, "major"), C_MAJOR)
        assert rn.degree == 1
        assert rn.local_key_mode == "major"

    def test_ii_in_c_major(self):
        rn = RNA.analyze(_chord(2, "minor"), C_MAJOR)
        assert rn.degree == 2

    def test_IV_in_c_major(self):
        rn = RNA.analyze(_chord(5, "major"), C_MAJOR)
        assert rn.degree == 4

    def test_V_in_c_major(self):
        rn = RNA.analyze(_chord(7, "major"), C_MAJOR)
        assert rn.degree == 5
        assert rn.function == HarmonicFunction.DOMINANT

    def test_vi_in_c_major(self):
        rn = RNA.analyze(_chord(9, "minor"), C_MAJOR)
        assert rn.degree == 6

    def test_viio_in_c_major(self):
        rn = RNA.analyze(_chord(11, "diminished"), C_MAJOR)
        assert rn.degree == 7
        assert rn.function == HarmonicFunction.DOMINANT

    def test_i_in_a_minor(self):
        rn = RNA.analyze(_chord(9, "minor"), A_MINOR)
        assert rn.degree == 1
        assert rn.function == HarmonicFunction.TONIC

    def test_V_in_a_minor(self):
        rn = RNA.analyze(_chord(4, "major"), A_MINOR)  # E major
        assert rn.degree == 5
        assert rn.function == HarmonicFunction.DOMINANT

    def test_function_is_set(self):
        rn = RNA.analyze(_chord(0, "major"), C_MAJOR)
        assert rn.function is not None

    def test_quality_preserved(self):
        rn = RNA.analyze(_chord(0, "major"), C_MAJOR)
        assert rn.quality == "major"

    def test_inversion_preserved(self):
        rn = RNA.analyze(_chord(0, "major", inversion=1), C_MAJOR)
        assert rn.inversion == 1

    def test_local_key_tonic_stored(self):
        rn = RNA.analyze(_chord(0, "major"), C_MAJOR)
        assert rn.local_key_tonic == 0

    def test_tonic_function_for_degree_1(self):
        rn = RNA.analyze(_chord(0, "major"), C_MAJOR)
        assert rn.function == HarmonicFunction.TONIC

    def test_predominant_function_for_degree_4(self):
        rn = RNA.analyze(_chord(5, "major"), C_MAJOR)
        assert rn.function == HarmonicFunction.PREDOMINANT


# ---------------------------------------------------------------------------
# TestIsSecondaryDominant
# ---------------------------------------------------------------------------

class TestIsSecondaryDominant:
    def test_D_major_in_c_major_is_V_of_V(self):
        # D major resolves to G (degree 5 in C major) → V/V
        is_sec, target = RNA._is_secondary_dominant(_chord(2, "major"), C_MAJOR)
        assert is_sec is True
        assert target == 5

    def test_primary_V_is_not_secondary(self):
        # G major in C major is the primary dominant (V)
        is_sec, target = RNA._is_secondary_dominant(_chord(7, "major"), C_MAJOR)
        assert is_sec is False

    def test_minor_chord_is_not_secondary_dominant(self):
        is_sec, _ = RNA._is_secondary_dominant(_chord(2, "minor"), C_MAJOR)
        assert is_sec is False

    def test_A_major_in_c_major_is_V_of_vi(self):
        # A major resolves to D (pc 2 = degree 2 in C), so V/ii
        is_sec, target = RNA._is_secondary_dominant(_chord(9, "major"), C_MAJOR)
        assert is_sec is True
        assert target == 2

    def test_diminished_secondary_leading_tone(self):
        # F# diminished in C major: (6+1)%12 = 7 = G (degree 5) → viio/V
        is_sec, target = RNA._is_secondary_dominant(_chord(6, "diminished"), C_MAJOR)
        assert is_sec is True
        assert target == 5

    def test_dominant7_secondary(self):
        # D dominant7 in C major: V7/V
        is_sec, target = RNA._is_secondary_dominant(_chord(2, "dominant7"), C_MAJOR)
        assert is_sec is True
        assert target == 5


# ---------------------------------------------------------------------------
# TestIsBorrowed
# ---------------------------------------------------------------------------

class TestIsBorrowed:
    def test_f_minor_in_c_major_is_borrowed(self):
        # iv in C major (parallel minor has iv = F minor)
        assert RNA._is_borrowed(_chord(5, "minor"), C_MAJOR) is True

    def test_eb_major_in_c_major_is_borrowed(self):
        # bIII in C major (parallel minor has III = Eb major)
        assert RNA._is_borrowed(_chord(3, "major"), C_MAJOR) is True

    def test_f_major_in_c_major_is_not_borrowed(self):
        # IV in C major is diatonic — not borrowed
        assert RNA._is_borrowed(_chord(5, "major"), C_MAJOR) is False

    def test_c_major_in_c_major_is_not_borrowed(self):
        assert RNA._is_borrowed(_chord(0, "major"), C_MAJOR) is False


# ---------------------------------------------------------------------------
# TestAnalyzeSecondaryDominant
# ---------------------------------------------------------------------------

class TestAnalyzeSecondaryDominant:
    def test_V7_of_V_is_secondary(self):
        # D dominant7 in C major → V7/V
        rn = RNA.analyze(_chord(2, "dominant7"), C_MAJOR)
        assert rn.is_secondary is True

    def test_V7_of_V_target_is_degree_5(self):
        rn = RNA.analyze(_chord(2, "dominant7"), C_MAJOR)
        assert rn.secondary_target == 5

    def test_secondary_function_is_dominant(self):
        rn = RNA.analyze(_chord(2, "dominant7"), C_MAJOR)
        assert rn.function == HarmonicFunction.DOMINANT

    def test_label_contains_slash(self):
        rn = RNA.analyze(_chord(2, "dominant7"), C_MAJOR)
        assert "/" in rn.label


# ---------------------------------------------------------------------------
# TestAnalyzeSequence
# ---------------------------------------------------------------------------

class TestAnalyzeSequence:
    def test_empty_chords_returns_empty(self):
        result = RNA.analyze_sequence([], [C_MAJOR])
        assert result == []

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError):
            RNA.analyze_sequence([_chord(0, "major")], [])

    def test_result_length_matches_chords(self):
        chords = [_chord(0, "major", onset=i) for i in range(5)]
        result = RNA.analyze_sequence(chords, [C_MAJOR])
        assert len(result) == 5

    def test_all_results_are_roman_numerals(self):
        chords = [_chord(0, "major"), _chord(7, "major")]
        result = RNA.analyze_sequence(chords, [C_MAJOR])
        assert all(isinstance(rn, RomanNumeral) for rn in result)

    def test_key_matching_uses_most_recent_key(self):
        # First 4 quarters in C major, then from offset 4 in G major
        c_key = KeyEstimate(tonic=0, mode="major", confidence=0.9, onset=Fraction(0))
        g_key = KeyEstimate(tonic=7, mode="major", confidence=0.9, onset=Fraction(4))

        chords = [
            _chord(0, "major", onset=0),   # C major at offset 0 → C major key
            _chord(7, "major", onset=4),   # G major at offset 4 → G major key
        ]
        result = RNA.analyze_sequence(chords, [c_key, g_key])
        assert result[0].local_key_tonic == 0
        assert result[1].local_key_tonic == 7

    def test_chord_before_first_key_uses_earliest_key(self):
        # Key starts at onset=4; chord at onset=0 — should use the earliest key
        late_key = KeyEstimate(tonic=0, mode="major", confidence=0.9, onset=Fraction(4))
        chords = [_chord(0, "major", onset=0)]
        result = RNA.analyze_sequence(chords, [late_key])
        assert len(result) == 1
        assert result[0].local_key_tonic == 0
