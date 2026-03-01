"""
Pure unit tests for src/core/data_structures.py — no file I/O required.

Coverage:
  TimeSignature      – construction, properties, validation
  MusicalEvent       – construction, pitch_class, offset, transpose, validation
  Chord              – construction, offset, validation
  HarmonicSegment    – construction, duration, new optional fields
  RomanNumeral.label – representative notation cases
  KeyEstimate        – construction, label, validation
  Score              – event sorting, time-sig lookup, range queries, properties
"""

import pytest
from fractions import Fraction

from src.core.data_structures import (
    Chord,
    HarmonicFunction,
    HarmonicSegment,
    KeyEstimate,
    MusicalEvent,
    ProlongationLevel,
    RomanNumeral,
    Score,
    TimeSignature,
)


# ===========================================================================
# TimeSignature
# ===========================================================================

class TestTimeSignature:

    def test_simple_4_4(self):
        ts = TimeSignature(4, 4)
        assert not ts.is_compound
        assert ts.beats_per_measure == 4
        assert ts.beat_duration == Fraction(1)

    def test_simple_3_4(self):
        ts = TimeSignature(3, 4)
        assert not ts.is_compound
        assert ts.beats_per_measure == 3
        assert ts.beat_duration == Fraction(1)

    def test_simple_2_4(self):
        ts = TimeSignature(2, 4)
        assert not ts.is_compound
        assert ts.beats_per_measure == 2
        assert ts.beat_duration == Fraction(1)

    def test_compound_6_8(self):
        ts = TimeSignature(6, 8)
        assert ts.is_compound
        assert ts.beats_per_measure == 2
        assert ts.beat_duration == Fraction(3, 2)

    def test_compound_9_8(self):
        ts = TimeSignature(9, 8)
        assert ts.is_compound
        assert ts.beats_per_measure == 3
        assert ts.beat_duration == Fraction(3, 2)

    def test_compound_12_8(self):
        ts = TimeSignature(12, 8)
        assert ts.is_compound
        assert ts.beats_per_measure == 4
        assert ts.beat_duration == Fraction(3, 2)

    def test_str(self):
        assert str(TimeSignature(4, 4)) == "4/4"
        assert str(TimeSignature(6, 8)) == "6/8"
        assert str(TimeSignature(3, 4)) == "3/4"

    def test_invalid_zero_numerator(self):
        with pytest.raises(ValueError):
            TimeSignature(0, 4)

    def test_invalid_denominator(self):
        with pytest.raises(ValueError):
            TimeSignature(4, 3)   # 3 not in [1, 2, 4, 8, 16, 32]

    def test_is_frozen(self):
        ts = TimeSignature(4, 4)
        with pytest.raises(Exception):
            ts.numerator = 3   # type: ignore[misc]


# ===========================================================================
# MusicalEvent
# ===========================================================================

class TestMusicalEvent:

    def test_valid_note(self):
        ev = MusicalEvent(pitch=60, onset=Fraction(0), duration=Fraction(1))
        assert ev.pitch == 60
        assert not ev.is_rest
        assert ev.pitch_class == 0

    def test_valid_rest(self):
        ev = MusicalEvent(
            pitch=None, onset=Fraction(0), duration=Fraction(1),
            velocity=0, is_rest=True,
        )
        assert ev.is_rest
        assert ev.pitch is None
        assert ev.pitch_class is None

    def test_offset(self):
        ev = MusicalEvent(pitch=60, onset=Fraction(1), duration=Fraction(2))
        assert ev.offset == Fraction(3)

    def test_pitch_class_c(self):
        assert MusicalEvent(pitch=60, onset=Fraction(0), duration=Fraction(1)).pitch_class == 0

    def test_pitch_class_g(self):
        assert MusicalEvent(pitch=67, onset=Fraction(0), duration=Fraction(1)).pitch_class == 7

    def test_pitch_class_wraps(self):
        assert MusicalEvent(pitch=72, onset=Fraction(0), duration=Fraction(1)).pitch_class == 0

    def test_transpose_up(self):
        ev = MusicalEvent(pitch=60, onset=Fraction(0), duration=Fraction(1))
        up = ev.transpose(12)
        assert up.pitch == 72
        assert up.onset == ev.onset
        assert up.duration == ev.duration
        assert up is not ev

    def test_transpose_clamps_high(self):
        ev = MusicalEvent(pitch=120, onset=Fraction(0), duration=Fraction(1))
        assert ev.transpose(20).pitch == 127

    def test_transpose_clamps_low(self):
        ev = MusicalEvent(pitch=5, onset=Fraction(0), duration=Fraction(1))
        assert ev.transpose(-20).pitch == 0

    def test_transpose_rest_returns_self(self):
        rest = MusicalEvent(
            pitch=None, onset=Fraction(0), duration=Fraction(1),
            velocity=0, is_rest=True,
        )
        assert rest.transpose(12) is rest

    def test_invalid_none_pitch_on_note(self):
        with pytest.raises(ValueError):
            MusicalEvent(pitch=None, onset=Fraction(0), duration=Fraction(1))

    def test_invalid_pitch_set_on_rest(self):
        with pytest.raises(ValueError):
            MusicalEvent(pitch=60, onset=Fraction(0), duration=Fraction(1),
                         velocity=0, is_rest=True)

    def test_invalid_pitch_out_of_range(self):
        with pytest.raises(ValueError):
            MusicalEvent(pitch=200, onset=Fraction(0), duration=Fraction(1))

    def test_invalid_velocity(self):
        with pytest.raises(ValueError):
            MusicalEvent(pitch=60, onset=Fraction(0), duration=Fraction(1), velocity=200)

    def test_invalid_zero_duration(self):
        with pytest.raises(ValueError):
            MusicalEvent(pitch=60, onset=Fraction(0), duration=Fraction(0))

    def test_invalid_beat_strength_too_high(self):
        with pytest.raises(ValueError):
            MusicalEvent(pitch=60, onset=Fraction(0), duration=Fraction(1), beat_strength=1.5)

    def test_invalid_beat_strength_negative(self):
        with pytest.raises(ValueError):
            MusicalEvent(pitch=60, onset=Fraction(0), duration=Fraction(1), beat_strength=-0.1)


# ===========================================================================
# Chord
# ===========================================================================

class TestChord:

    def test_valid_construction(self):
        c = Chord(
            onset=Fraction(0), duration=Fraction(1),
            root=0, quality='major',
            pitches=frozenset([0, 4, 7]), bass=0,
        )
        assert c.root == 0
        assert c.offset == Fraction(1)
        assert c.inversion == 0

    def test_first_inversion(self):
        c = Chord(
            onset=Fraction(0), duration=Fraction(1),
            root=0, quality='major',
            pitches=frozenset([0, 4, 7]), bass=4, inversion=1,
        )
        assert c.bass == 4
        assert c.inversion == 1

    def test_invalid_root(self):
        with pytest.raises(ValueError):
            Chord(onset=Fraction(0), duration=Fraction(1),
                  root=12, quality='major', pitches=frozenset([0, 4, 7]), bass=0)

    def test_invalid_bass(self):
        with pytest.raises(ValueError):
            Chord(onset=Fraction(0), duration=Fraction(1),
                  root=0, quality='major', pitches=frozenset([0, 4, 7]), bass=-1)

    def test_invalid_duration(self):
        with pytest.raises(ValueError):
            Chord(onset=Fraction(0), duration=Fraction(0),
                  root=0, quality='major', pitches=frozenset([0, 4, 7]), bass=0)


# ===========================================================================
# HarmonicSegment
# ===========================================================================

class TestHarmonicSegment:

    def test_valid_segment(self):
        seg = HarmonicSegment(start=Fraction(0), end=Fraction(4),
                              key_tonic=0, key_mode='major')
        assert seg.duration == Fraction(4)
        assert seg.roman_numerals == []
        assert seg.prolongation_level is None
        assert seg.chords == []

    def test_duration_property(self):
        seg = HarmonicSegment(start=Fraction(2), end=Fraction(7),
                              key_tonic=7, key_mode='minor')
        assert seg.duration == Fraction(5)

    def test_with_prolongation_level(self):
        seg = HarmonicSegment(
            start=Fraction(0), end=Fraction(4),
            key_tonic=0, key_mode='major',
            prolongation_level=ProlongationLevel.FOREGROUND,
        )
        assert seg.prolongation_level == ProlongationLevel.FOREGROUND

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError):
            HarmonicSegment(start=Fraction(4), end=Fraction(0),
                            key_tonic=0, key_mode='major')

    def test_invalid_key_tonic(self):
        with pytest.raises(ValueError):
            HarmonicSegment(start=Fraction(0), end=Fraction(4),
                            key_tonic=12, key_mode='major')

    def test_invalid_key_mode(self):
        with pytest.raises(ValueError):
            HarmonicSegment(start=Fraction(0), end=Fraction(4),
                            key_tonic=0, key_mode='dorian')


# ===========================================================================
# RomanNumeral.label
# ===========================================================================

class TestRomanNumeralLabel:
    """
    Tests are grouped by chord family. Cases are chosen for the most common
    patterns; secondary dominants and inversions get dedicated tests.

    Note: for seventh chords, the label is built as base + inversion_figure
    (e.g. V + 7 → V7). Quality markers for dim7 / half-dim7 / major7 are
    *not* added in the current implementation — see TODO in data_structures.py.
    """

    # -- helpers --
    @staticmethod
    def _rn(degree, quality, inversion, mode='major',
            is_secondary=False, secondary_target=None):
        return RomanNumeral(
            degree=degree, quality=quality, inversion=inversion,
            local_key_tonic=0, local_key_mode=mode,
            is_secondary=is_secondary, secondary_target=secondary_target,
        )

    # Triads – major mode
    def test_I(self):
        assert self._rn(1, 'major', 0).label == 'I'

    def test_I6(self):
        assert self._rn(1, 'major', 1).label == 'I6'

    def test_I64(self):
        assert self._rn(1, 'major', 2).label == 'I64'

    def test_ii(self):
        assert self._rn(2, 'minor', 0).label == 'ii'

    def test_IV(self):
        assert self._rn(4, 'major', 0).label == 'IV'

    def test_V(self):
        assert self._rn(5, 'major', 0).label == 'V'

    def test_V6(self):
        assert self._rn(5, 'major', 1).label == 'V6'

    def test_viio(self):
        assert self._rn(7, 'diminished', 0).label == 'viio'

    def test_viio6(self):
        assert self._rn(7, 'diminished', 1).label == 'viio6'

    def test_iio6(self):
        assert self._rn(2, 'diminished', 1).label == 'iio6'

    # Seventh chords – dominant7 (dominant7 label is correct out of the box)
    def test_V7(self):
        assert self._rn(5, 'dominant7', 0).label == 'V7'

    def test_V65(self):
        assert self._rn(5, 'dominant7', 1).label == 'V65'

    def test_V43(self):
        assert self._rn(5, 'dominant7', 2).label == 'V43'

    def test_V42(self):
        assert self._rn(5, 'dominant7', 3).label == 'V42'

    # Secondary dominants
    def test_V7_of_V(self):
        assert self._rn(5, 'dominant7', 0,
                        is_secondary=True, secondary_target=5).label == 'V7/V'

    def test_V7_of_iv_in_minor(self):
        rn = RomanNumeral(
            degree=5, quality='dominant7', inversion=0,
            local_key_tonic=0, local_key_mode='minor',
            is_secondary=True, secondary_target=4,
        )
        assert rn.label == 'V7/iv'

    # Minor mode triads
    def test_i_minor(self):
        rn = RomanNumeral(degree=1, quality='minor', inversion=0,
                          local_key_tonic=0, local_key_mode='minor')
        assert rn.label == 'i'

    def test_iv_minor(self):
        rn = RomanNumeral(degree=4, quality='minor', inversion=0,
                          local_key_tonic=0, local_key_mode='minor')
        assert rn.label == 'iv'

    def test_V_in_minor(self):
        rn = RomanNumeral(degree=5, quality='major', inversion=0,
                          local_key_tonic=0, local_key_mode='minor')
        assert rn.label == 'V'


# ===========================================================================
# KeyEstimate
# ===========================================================================

class TestKeyEstimate:

    def test_c_major(self):
        ke = KeyEstimate(tonic=0, mode='major', confidence=0.9, onset=Fraction(0))
        assert ke.label == 'C major'

    def test_g_major(self):
        ke = KeyEstimate(tonic=7, mode='major', confidence=0.95, onset=Fraction(0))
        assert ke.label == 'G major'

    def test_d_minor(self):
        ke = KeyEstimate(tonic=2, mode='minor', confidence=0.8, onset=Fraction(4))
        assert ke.label == 'D minor'

    def test_eb_major(self):
        ke = KeyEstimate(tonic=3, mode='major', confidence=0.7, onset=Fraction(0))
        assert ke.label == 'Eb major'

    def test_b_minor(self):
        ke = KeyEstimate(tonic=11, mode='minor', confidence=0.85, onset=Fraction(0))
        assert ke.label == 'B minor'

    def test_invalid_tonic_12(self):
        with pytest.raises(ValueError):
            KeyEstimate(tonic=12, mode='major', confidence=0.9, onset=Fraction(0))

    def test_invalid_tonic_negative(self):
        with pytest.raises(ValueError):
            KeyEstimate(tonic=-1, mode='major', confidence=0.9, onset=Fraction(0))

    def test_invalid_confidence_high(self):
        with pytest.raises(ValueError):
            KeyEstimate(tonic=0, mode='major', confidence=1.5, onset=Fraction(0))

    def test_invalid_confidence_negative(self):
        with pytest.raises(ValueError):
            KeyEstimate(tonic=0, mode='major', confidence=-0.1, onset=Fraction(0))

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            KeyEstimate(tonic=0, mode='lydian', confidence=0.9, onset=Fraction(0))

    def test_is_frozen(self):
        ke = KeyEstimate(tonic=0, mode='major', confidence=0.9, onset=Fraction(0))
        with pytest.raises(Exception):
            ke.tonic = 5  # type: ignore[misc]


# ===========================================================================
# Score
# ===========================================================================

def _ev(pitch, onset, duration, voice=0):
    return MusicalEvent(
        pitch=pitch, onset=Fraction(onset), duration=Fraction(duration),
        velocity=64, measure=1, beat=Fraction(0), beat_strength=1.0,
        voice=voice, is_rest=False,
    )


class TestScore:

    def test_events_sorted_by_onset(self):
        events = [_ev(64, 2, 1), _ev(60, 0, 1), _ev(62, 1, 1)]
        score = Score(events=events)
        onsets = [e.onset for e in score.events]
        assert onsets == sorted(onsets)

    def test_default_time_signature_added(self):
        score = Score(events=[_ev(60, 0, 1)])
        assert len(score.time_signatures) == 1
        assert score.time_signatures[0][1] == TimeSignature(4, 4)

    def test_custom_time_signature_preserved(self):
        ts = TimeSignature(3, 4)
        score = Score(events=[_ev(60, 0, 1)],
                      time_signatures=[(Fraction(0), ts)])
        assert score.get_time_signature_at(Fraction(0)) == ts

    def test_get_time_signature_at_changes(self):
        ts1, ts2 = TimeSignature(4, 4), TimeSignature(3, 4)
        score = Score(
            events=[_ev(60, 0, 1)],
            time_signatures=[(Fraction(0), ts1), (Fraction(8), ts2)],
        )
        assert score.get_time_signature_at(Fraction(0)) == ts1
        assert score.get_time_signature_at(Fraction(4)) == ts1
        assert score.get_time_signature_at(Fraction(8)) == ts2
        assert score.get_time_signature_at(Fraction(16)) == ts2

    def test_duration(self):
        # two events: onset=0 dur=1 (offset=1), onset=1 dur=2 (offset=3)
        score = Score(events=[_ev(60, 0, 1), _ev(62, 1, 2)])
        assert score.duration == Fraction(3)

    def test_duration_empty(self):
        assert Score(events=[]).duration == Fraction(0)

    def test_num_voices(self):
        events = [_ev(60, 0, 1, voice=0), _ev(64, 0, 1, voice=1), _ev(67, 0, 1, voice=2)]
        score = Score(events=events)
        assert score.num_voices == 3

    def test_num_voices_empty(self):
        assert Score(events=[]).num_voices == 0

    def test_pitch_range(self):
        rest = MusicalEvent(pitch=None, onset=Fraction(2), duration=Fraction(1),
                            velocity=0, is_rest=True)
        score = Score(events=[_ev(60, 0, 1), _ev(72, 1, 1), rest])
        assert score.pitch_range == (60, 72)

    def test_pitch_range_all_rests(self):
        rest = MusicalEvent(pitch=None, onset=Fraction(0), duration=Fraction(1),
                            velocity=0, is_rest=True)
        assert Score(events=[rest]).pitch_range == (None, None)

    def test_get_events_in_range_exact_boundaries(self):
        # event [0, 1) should NOT appear in window [1, 2)
        events = [_ev(60, 0, 1), _ev(62, 1, 1), _ev(64, 2, 1)]
        score = Score(events=events)
        result = score.get_events_in_range(Fraction(1), Fraction(2))
        assert len(result) == 1
        assert result[0].pitch == 62

    def test_get_events_in_range_overlap(self):
        # event spanning [0, 3) should appear in window [2, 4)
        events = [_ev(60, 0, 3)]
        score = Score(events=events)
        result = score.get_events_in_range(Fraction(2), Fraction(4))
        assert len(result) == 1

    def test_get_events_in_range_empty_window(self):
        score = Score(events=[_ev(60, 0, 1)])
        assert score.get_events_in_range(Fraction(5), Fraction(10)) == []

    def test_get_events_by_voice(self):
        events = [_ev(60, 0, 1, voice=0), _ev(64, 0, 1, voice=1), _ev(67, 1, 1, voice=0)]
        score = Score(events=events)
        v0 = score.get_events_by_voice(0)
        v1 = score.get_events_by_voice(1)
        assert all(e.voice == 0 for e in v0)
        assert all(e.voice == 1 for e in v1)
        assert len(v0) == 2
        assert len(v1) == 1
