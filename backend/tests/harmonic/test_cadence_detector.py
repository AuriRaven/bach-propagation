"""
Tests for src/harmonic/cadence_detector.py

All tests use synthetic Chord + RomanNumeral fixtures — no score I/O.

Structure:
    TestPerfectAuthenticCadence    – V(7) → I, both root position
    TestImperfectAuthenticCadence  – V(7) → I with inversion
    TestHalfCadence                – ... → V
    TestDeceptiveCadence           – V(7) → vi / VI
    TestPhrygianHalfCadence        – iv⁶ → V (minor only)
    TestPlagalCadence              – IV → I
    TestPriorityOrdering           – stronger cadences win
    TestSecondaryDominantRejected  – V/V → V is tonicization, not HC
    TestKeyChange                  – require_same_key gating
    TestConfidenceThresholds       – min_confidence filter
    TestDownweightWeakFinal        – short arrivals reduce confidence
    TestIncludePlagal              – toggle plagal reporting
    TestDetectWalksSequence        – end-to-end sequence walk
    TestCadenceLabel               – pretty label rendering
"""

from fractions import Fraction
from typing import Optional

import pytest

from src.core.data_structures import Chord, HarmonicEvent, RomanNumeral
from src.harmonic.cadence_detector import (
    Cadence,
    CadenceDetectionConfig,
    CadenceDetector,
    CadenceType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chord(
    onset: float,
    duration: float,
    root: int,
    quality: str = "major",
    inversion: int = 0,
) -> Chord:
    """Build a synthetic Chord with a plausible bass and pitch-class set."""
    # The detector only reads onset/duration/inversion, so the pitch-class
    # content can be a bare stub.
    pitches = frozenset({root, (root + 4) % 12, (root + 7) % 12})
    return Chord(
        onset=Fraction(onset),
        duration=Fraction(duration),
        root=root,
        quality=quality,
        pitches=pitches,
        bass=root,
        inversion=inversion,
    )


def _rn(
    degree: int,
    quality: str = "major",
    inversion: int = 0,
    local_key_tonic: int = 0,
    local_key_mode: str = "major",
    is_secondary: bool = False,
    secondary_target: Optional[int] = None,
) -> RomanNumeral:
    return RomanNumeral(
        degree=degree,
        quality=quality,
        inversion=inversion,
        local_key_tonic=local_key_tonic,
        local_key_mode=local_key_mode,
        is_secondary=is_secondary,
        secondary_target=secondary_target,
    )


def _event(
    onset: float,
    duration: float,
    degree: int,
    quality: str = "major",
    inversion: int = 0,
    root: int = 0,
    local_key_tonic: int = 0,
    local_key_mode: str = "major",
    is_secondary: bool = False,
    secondary_target: Optional[int] = None,
) -> HarmonicEvent:
    """Bundle a synthetic Chord + RomanNumeral into a HarmonicEvent."""
    return HarmonicEvent(
        chord=_chord(onset, duration, root=root, quality=quality, inversion=inversion),
        roman_numeral=_rn(
            degree=degree,
            quality=quality,
            inversion=inversion,
            local_key_tonic=local_key_tonic,
            local_key_mode=local_key_mode,
            is_secondary=is_secondary,
            secondary_target=secondary_target,
        ),
    )


DETECTOR = CadenceDetector()


# ---------------------------------------------------------------------------
# TestPerfectAuthenticCadence
# ---------------------------------------------------------------------------


class TestPerfectAuthenticCadence:
    def test_v_to_i_root_position_is_pac(self):
        pen = _event(0, 1, degree=5, quality="major", inversion=0, root=7)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.AUTHENTIC_PERFECT

    def test_v7_to_i_is_pac(self):
        pen = _event(0, 1, degree=5, quality="dominant7", inversion=0, root=7)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.AUTHENTIC_PERFECT

    def test_pac_confidence_high(self):
        pen = _event(0, 1, degree=5, quality="major", inversion=0, root=7)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.confidence >= 0.85

    def test_v7_pac_stronger_than_triad_pac(self):
        pen_triad = _event(0, 1, degree=5, quality="major", inversion=0, root=7)
        fin_triad = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        pen_7 = _event(0, 1, degree=5, quality="dominant7", inversion=0, root=7)
        fin_7 = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        triad = DETECTOR.classify_pair(pen_triad, fin_triad)
        seventh = DETECTOR.classify_pair(pen_7, fin_7)
        assert triad is not None and seventh is not None
        assert seventh.confidence >= triad.confidence

    def test_onset_is_final_chord_onset(self):
        pen = _event(3, 1, degree=5, quality="major", inversion=0, root=7)
        fin = _event(4, 2, degree=1, quality="major", inversion=0, root=0)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.onset == Fraction(4)


# ---------------------------------------------------------------------------
# TestImperfectAuthenticCadence
# ---------------------------------------------------------------------------


class TestImperfectAuthenticCadence:
    def test_v_to_i_first_inversion_is_iac(self):
        pen = _event(0, 1, degree=5, quality="major", inversion=0, root=7)
        fin = _event(1, 1, degree=1, quality="major", inversion=1, root=0)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.AUTHENTIC_IMPERFECT

    def test_v6_to_i_is_iac(self):
        pen = _event(0, 1, degree=5, quality="major", inversion=1, root=7)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.AUTHENTIC_IMPERFECT

    def test_iac_confidence_below_pac(self):
        pen_pac = _event(0, 1, degree=5, quality="major", inversion=0, root=7)
        fin_pac = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        pen_iac = _event(0, 1, degree=5, quality="major", inversion=1, root=7)
        fin_iac = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        pac = DETECTOR.classify_pair(pen_pac, fin_pac)
        iac = DETECTOR.classify_pair(pen_iac, fin_iac)
        assert pac is not None and iac is not None
        assert iac.confidence < pac.confidence


# ---------------------------------------------------------------------------
# TestHalfCadence
# ---------------------------------------------------------------------------


class TestHalfCadence:
    def test_i_to_v_is_half(self):
        pen = _event(0, 1, degree=1, quality="major", inversion=0, root=0)
        fin = _event(1, 1, degree=5, quality="major", inversion=0, root=7)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.HALF

    def test_ii_to_v_is_half(self):
        pen = _event(0, 1, degree=2, quality="minor", inversion=0, root=2)
        fin = _event(1, 1, degree=5, quality="major", inversion=0, root=7)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.HALF

    def test_iv_to_v_is_half(self):
        pen = _event(0, 1, degree=4, quality="major", inversion=0, root=5)
        fin = _event(1, 1, degree=5, quality="major", inversion=0, root=7)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.HALF

    def test_v_to_v_not_cadence(self):
        # Two successive V chords (e.g. different voicings) are not a cadence.
        pen = _event(0, 1, degree=5, quality="major", inversion=0, root=7)
        fin = _event(1, 1, degree=5, quality="major", inversion=1, root=7)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is None


# ---------------------------------------------------------------------------
# TestDeceptiveCadence
# ---------------------------------------------------------------------------


class TestDeceptiveCadence:
    def test_v_to_vi_in_major_is_deceptive(self):
        pen = _event(0, 1, degree=5, quality="major", inversion=0, root=7)
        fin = _event(1, 1, degree=6, quality="minor", inversion=0, root=9)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.DECEPTIVE

    def test_v7_to_VI_in_minor_is_deceptive(self):
        pen = _event(
            0, 1, degree=5, quality="dominant7", inversion=0,
            root=7, local_key_mode="minor",
        )
        fin = _event(
            1, 1, degree=6, quality="major", inversion=0,
            root=8, local_key_mode="minor",
        )
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.DECEPTIVE

    def test_v_to_iii_not_deceptive(self):
        pen = _event(0, 1, degree=5, quality="major", inversion=0, root=7)
        fin = _event(1, 1, degree=3, quality="minor", inversion=0, root=4)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is None


# ---------------------------------------------------------------------------
# TestPhrygianHalfCadence
# ---------------------------------------------------------------------------


class TestPhrygianHalfCadence:
    def test_iv6_to_v_in_minor_is_phrygian(self):
        pen = _event(
            0, 1, degree=4, quality="minor", inversion=1,
            root=5, local_key_mode="minor",
        )
        fin = _event(
            1, 1, degree=5, quality="major", inversion=0,
            root=7, local_key_mode="minor",
        )
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.PHRYGIAN_HALF

    def test_iv6_to_v_in_major_not_phrygian(self):
        # In major, iv⁶ → V is just a half cadence — PHC is minor-only.
        pen = _event(
            0, 1, degree=4, quality="major", inversion=1,
            root=5, local_key_mode="major",
        )
        fin = _event(
            1, 1, degree=5, quality="major", inversion=0,
            root=7, local_key_mode="major",
        )
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.HALF

    def test_iv_root_position_in_minor_is_just_half(self):
        # Root-position iv → V is a half cadence, not Phrygian.
        pen = _event(
            0, 1, degree=4, quality="minor", inversion=0,
            root=5, local_key_mode="minor",
        )
        fin = _event(
            1, 1, degree=5, quality="major", inversion=0,
            root=7, local_key_mode="minor",
        )
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.HALF


# ---------------------------------------------------------------------------
# TestPlagalCadence
# ---------------------------------------------------------------------------


class TestPlagalCadence:
    def test_IV_to_I_is_plagal(self):
        pen = _event(0, 1, degree=4, quality="major", inversion=0, root=5)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.PLAGAL

    def test_iv_to_i_in_minor_is_plagal(self):
        pen = _event(
            0, 1, degree=4, quality="minor", inversion=0,
            root=5, local_key_mode="minor",
        )
        fin = _event(
            1, 1, degree=1, quality="minor", inversion=0,
            root=0, local_key_mode="minor",
        )
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.PLAGAL

    def test_iv_diminished_not_plagal(self):
        # Only major/minor IV qualifies as subdominant for plagal.
        pen = _event(0, 1, degree=4, quality="diminished", inversion=0, root=5)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is None


# ---------------------------------------------------------------------------
# TestPriorityOrdering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_pac_beats_iac(self):
        # V → I both root-position: should classify as PAC, not IAC.
        pen = _event(0, 1, degree=5, quality="major", inversion=0, root=7)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.AUTHENTIC_PERFECT

    def test_phrygian_beats_half_in_minor(self):
        # iv⁶ → V in minor matches both PHC and the generic HC pattern.
        # PHC is more specific, so it must be tried first and win.
        pen = _event(
            0, 1, degree=4, quality="minor", inversion=1,
            root=5, local_key_mode="minor",
        )
        fin = _event(
            1, 1, degree=5, quality="major", inversion=0,
            root=7, local_key_mode="minor",
        )
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert cad.cadence_type is CadenceType.PHRYGIAN_HALF


# ---------------------------------------------------------------------------
# TestSecondaryDominantRejected
# ---------------------------------------------------------------------------


class TestSecondaryDominantRejected:
    def test_v_of_v_to_v_not_cadence(self):
        # V/V → V is tonicization, not a half cadence in the outer key.
        pen = _event(
            0, 1, degree=5, quality="major", inversion=0,
            root=2, is_secondary=True, secondary_target=5,
        )
        fin = _event(1, 1, degree=5, quality="major", inversion=0, root=7)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is None

    def test_v_of_iv_to_iv_not_cadence(self):
        pen = _event(
            0, 1, degree=5, quality="dominant7", inversion=0,
            root=0, is_secondary=True, secondary_target=4,
        )
        fin = _event(1, 1, degree=4, quality="major", inversion=0, root=5)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is None


# ---------------------------------------------------------------------------
# TestKeyChange
# ---------------------------------------------------------------------------


class TestKeyChange:
    def test_cross_key_rejected_by_default(self):
        # V → I but penultimate thinks it's in C and final thinks it's in G.
        # With require_same_key=True (default), this is not a cadence.
        pen = _event(
            0, 1, degree=5, quality="major", inversion=0,
            root=7, local_key_tonic=0, local_key_mode="major",
        )
        fin = _event(
            1, 1, degree=1, quality="major", inversion=0,
            root=7, local_key_tonic=7, local_key_mode="major",
        )
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is None

    def test_cross_key_allowed_when_disabled(self):
        pen = _event(
            0, 1, degree=5, quality="major", inversion=0,
            root=7, local_key_tonic=0, local_key_mode="major",
        )
        fin = _event(
            1, 1, degree=1, quality="major", inversion=0,
            root=7, local_key_tonic=7, local_key_mode="major",
        )
        cfg = CadenceDetectionConfig(require_same_key=False)
        cad = DETECTOR.classify_pair(pen, fin, cfg)
        assert cad is not None
        assert cad.cadence_type is CadenceType.AUTHENTIC_PERFECT


# ---------------------------------------------------------------------------
# TestConfidenceThresholds
# ---------------------------------------------------------------------------


class TestConfidenceThresholds:
    def test_low_confidence_dropped_from_detect(self):
        # Plagal base confidence is 0.75; set threshold above that.
        pen = _event(0, 1, degree=4, quality="major", inversion=0, root=5)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        cfg = CadenceDetectionConfig(min_confidence=0.95)
        out = DETECTOR.detect([pen, fin], cfg)
        assert out == []

    def test_high_confidence_retained(self):
        pen = _event(0, 1, degree=5, quality="dominant7", inversion=0, root=7)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        cfg = CadenceDetectionConfig(min_confidence=0.6)
        out = DETECTOR.detect([pen, fin], cfg)
        assert len(out) == 1
        assert out[0].cadence_type is CadenceType.AUTHENTIC_PERFECT


# ---------------------------------------------------------------------------
# TestDownweightWeakFinal
# ---------------------------------------------------------------------------


class TestDownweightWeakFinal:
    def test_short_final_reduces_confidence(self):
        # Penultimate is a whole note; final is a sixteenth → ratio ≈ 1/16.
        pen = _event(0, 4, degree=5, quality="major", inversion=0, root=7)
        fin = _event(4, 0.25, degree=1, quality="major", inversion=0, root=0)
        with_downweight = DETECTOR.classify_pair(
            pen, fin, CadenceDetectionConfig(downweight_weak_final=True)
        )
        without_downweight = DETECTOR.classify_pair(
            pen, fin, CadenceDetectionConfig(downweight_weak_final=False)
        )
        assert with_downweight is not None
        assert without_downweight is not None
        assert with_downweight.confidence < without_downweight.confidence

    def test_equal_duration_no_downweight(self):
        pen = _event(0, 1, degree=5, quality="major", inversion=0, root=7)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        downweighted = DETECTOR.classify_pair(
            pen, fin, CadenceDetectionConfig(downweight_weak_final=True)
        )
        plain = DETECTOR.classify_pair(
            pen, fin, CadenceDetectionConfig(downweight_weak_final=False)
        )
        assert downweighted is not None and plain is not None
        # Equal durations → ratio=1 → scale factor = (0.5 + 0.5*1) = 1.0.
        assert downweighted.confidence == pytest.approx(plain.confidence)


# ---------------------------------------------------------------------------
# TestIncludePlagal
# ---------------------------------------------------------------------------


class TestIncludePlagal:
    def test_plagal_included_by_default(self):
        pen = _event(0, 1, degree=4, quality="major", inversion=0, root=5)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        out = DETECTOR.detect([pen, fin])
        assert len(out) == 1
        assert out[0].cadence_type is CadenceType.PLAGAL

    def test_plagal_excluded_when_disabled(self):
        pen = _event(0, 1, degree=4, quality="major", inversion=0, root=5)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        cfg = CadenceDetectionConfig(include_plagal=False)
        out = DETECTOR.detect([pen, fin], cfg)
        assert out == []


# ---------------------------------------------------------------------------
# TestDetectWalksSequence
# ---------------------------------------------------------------------------


class TestDetectWalksSequence:
    def test_empty_input_returns_empty(self):
        assert DETECTOR.detect([]) == []

    def test_single_event_returns_empty(self):
        e = _event(0, 1, degree=1, quality="major", inversion=0, root=0)
        assert DETECTOR.detect([e]) == []

    def test_multiple_cadences_in_sequence(self):
        # I — V — I — IV — I: expect PAC (V→I) then Plagal (IV→I).
        events = [
            _event(0, 1, degree=1, quality="major", inversion=0, root=0),
            _event(1, 1, degree=5, quality="major", inversion=0, root=7),
            _event(2, 1, degree=1, quality="major", inversion=0, root=0),
            _event(3, 1, degree=4, quality="major", inversion=0, root=5),
            _event(4, 1, degree=1, quality="major", inversion=0, root=0),
        ]
        out = DETECTOR.detect(events)
        types = [c.cadence_type for c in out]
        assert CadenceType.AUTHENTIC_PERFECT in types
        assert CadenceType.PLAGAL in types

    def test_results_ordered_by_onset(self):
        events = [
            _event(0, 1, degree=1, quality="major", inversion=0, root=0),
            _event(1, 1, degree=5, quality="major", inversion=0, root=7),
            _event(2, 1, degree=1, quality="major", inversion=0, root=0),
            _event(3, 1, degree=4, quality="major", inversion=0, root=5),
            _event(4, 1, degree=1, quality="major", inversion=0, root=0),
        ]
        out = DETECTOR.detect(events)
        onsets = [c.onset for c in out]
        assert onsets == sorted(onsets)


# ---------------------------------------------------------------------------
# TestCadenceLabel
# ---------------------------------------------------------------------------


class TestCadenceLabel:
    def test_label_renders_roman_arrow_roman(self):
        pen = _event(0, 1, degree=5, quality="dominant7", inversion=0, root=7)
        fin = _event(1, 1, degree=1, quality="major", inversion=0, root=0)
        cad = DETECTOR.classify_pair(pen, fin)
        assert cad is not None
        assert "→" in cad.label
        # The penultimate should render with a "7" somewhere.
        pen_side = cad.label.split("→")[0]
        assert "7" in pen_side
