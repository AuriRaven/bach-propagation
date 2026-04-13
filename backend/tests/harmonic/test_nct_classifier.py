"""
Tests for src/harmonic/nct_classifier.py

All tests use synthetic MusicalEvent / Chord objects — no file I/O required.

Structure:
    TestSignedInterval             – numeric helpers
    TestIsChordTone                – chord-tone membership check
    TestActiveChordLookup          – chord alignment
    TestClassifyEventChordTone     – positive CT path
    TestClassifyEventPassingTone   – PT detection
    TestClassifyEventNeighborTone  – NT detection
    TestClassifyEventSuspension    – SUS detection
    TestClassifyEventAnticipation  – ANT detection
    TestClassifyEventEscapeTone    – ET detection
    TestClassifyEventAppoggiatura  – APP detection
    TestClassifyEventPedalPoint    – pedal detection
    TestClassifySequence           – end-to-end over an event list
    TestFilterEventsForProfile     – chord-free heuristic filter
"""

import pytest
from fractions import Fraction
from typing import List

from src.core.data_structures import Chord, MusicalEvent
from src.harmonic.nct_classifier import (
    NCTAnnotation,
    NCTClassificationConfig,
    NCTClassifier,
    NCTType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _note(
    pitch: int,
    onset: float = 0,
    dur: float = 1,
    beat_strength: float = 1.0,
    voice: int = 0,
) -> MusicalEvent:
    return MusicalEvent(
        pitch=pitch,
        onset=Fraction(onset),
        duration=Fraction(dur),
        beat_strength=beat_strength,
        voice=voice,
    )


def _rest(onset: float = 0, dur: float = 1, voice: int = 0) -> MusicalEvent:
    return MusicalEvent(
        pitch=None,
        onset=Fraction(onset),
        duration=Fraction(dur),
        is_rest=True,
        velocity=0,
        voice=voice,
    )


def _c_major_chord(onset: float = 0, dur: float = 4) -> Chord:
    """C-major triad as a :class:`Chord` that covers the requested span."""
    return Chord(
        onset=Fraction(onset),
        duration=Fraction(dur),
        root=0,
        quality="major",
        pitches=frozenset({0, 4, 7}),
        bass=0,
        inversion=0,
    )


def _g_major_chord(onset: float = 0, dur: float = 4) -> Chord:
    """G-major triad."""
    return Chord(
        onset=Fraction(onset),
        duration=Fraction(dur),
        root=7,
        quality="major",
        pitches=frozenset({2, 7, 11}),
        bass=7,
        inversion=0,
    )


# ---------------------------------------------------------------------------
# TestSignedInterval
# ---------------------------------------------------------------------------


class TestSignedInterval:
    def test_ascending(self):
        iv = NCTClassifier._signed_interval(_note(60), _note(62))
        assert iv == 2

    def test_descending(self):
        iv = NCTClassifier._signed_interval(_note(64), _note(62))
        assert iv == -2

    def test_none_for_missing_sides(self):
        assert NCTClassifier._signed_interval(None, _note(60)) is None
        assert NCTClassifier._signed_interval(_note(60), None) is None

    def test_none_for_rest(self):
        assert NCTClassifier._signed_interval(_rest(), _note(60)) is None
        assert NCTClassifier._signed_interval(_note(60), _rest()) is None


# ---------------------------------------------------------------------------
# TestIsChordTone
# ---------------------------------------------------------------------------


class TestIsChordTone:
    def test_chord_tone_pitch_class(self):
        assert NCTClassifier._is_chord_tone(60, _c_major_chord()) is True  # C
        assert NCTClassifier._is_chord_tone(64, _c_major_chord()) is True  # E
        assert NCTClassifier._is_chord_tone(67, _c_major_chord()) is True  # G

    def test_non_chord_tone(self):
        assert NCTClassifier._is_chord_tone(62, _c_major_chord()) is False  # D
        assert NCTClassifier._is_chord_tone(65, _c_major_chord()) is False  # F

    def test_none_pitch_is_not_chord_tone(self):
        assert NCTClassifier._is_chord_tone(None, _c_major_chord()) is False

    def test_none_chord_is_not_chord_tone(self):
        assert NCTClassifier._is_chord_tone(60, None) is False

    def test_octave_equivalence(self):
        # C5 (72) is pitch class 0, still in C major
        assert NCTClassifier._is_chord_tone(72, _c_major_chord()) is True


# ---------------------------------------------------------------------------
# TestActiveChordLookup
# ---------------------------------------------------------------------------


class TestActiveChordLookup:
    def test_lookup_inside_chord_span(self):
        chords = [_c_major_chord(0, 4), _g_major_chord(4, 4)]
        active = NCTClassifier._active_chord(Fraction(2), chords)
        assert active is chords[0]

    def test_lookup_on_chord_boundary(self):
        chords = [_c_major_chord(0, 4), _g_major_chord(4, 4)]
        active = NCTClassifier._active_chord(Fraction(4), chords)
        assert active is chords[1]

    def test_next_chord_after(self):
        chords = [_c_major_chord(0, 4), _g_major_chord(4, 4)]
        nxt = NCTClassifier._active_chord_after(Fraction(2), chords)
        assert nxt is chords[1]
        # Nothing after the last chord
        assert NCTClassifier._active_chord_after(Fraction(100), chords) is None


# ---------------------------------------------------------------------------
# TestClassifyEventChordTone
# ---------------------------------------------------------------------------


class TestClassifyEventChordTone:
    def setup_method(self):
        self.clf = NCTClassifier()

    def test_c_note_in_c_major_is_chord_tone(self):
        ev = _note(60)
        ann = self.clf.classify_event(
            event=ev,
            prev_event=None,
            next_event=None,
            current_chord=_c_major_chord(),
            next_chord=None,
        )
        assert ann.nct_type is NCTType.CHORD_TONE
        assert ann.pitch == 60
        assert ann.onset == Fraction(0)

    def test_rest_raises(self):
        ev = _rest()
        with pytest.raises(ValueError):
            self.clf.classify_event(ev, None, None, _c_major_chord(), None)


# ---------------------------------------------------------------------------
# TestClassifyEventPassingTone
# ---------------------------------------------------------------------------


class TestClassifyEventPassingTone:
    def setup_method(self):
        self.clf = NCTClassifier()

    def test_stepwise_ascending_passing_tone(self):
        # C → D → E in C major; D is a passing tone between C and E.
        prev_ev = _note(60, onset=0, beat_strength=1.0)
        ev = _note(62, onset=0.5, dur=0.5, beat_strength=0.3)  # weak
        next_ev = _note(64, onset=1, beat_strength=0.6)
        ann = self.clf.classify_event(
            event=ev,
            prev_event=prev_ev,
            next_event=next_ev,
            current_chord=_c_major_chord(),
            next_chord=None,
        )
        assert ann.nct_type is NCTType.PASSING_TONE
        assert ann.resolution_pitch == 64

    def test_stepwise_descending_passing_tone(self):
        prev_ev = _note(64)
        ev = _note(62, beat_strength=0.3)
        next_ev = _note(60)
        ann = self.clf.classify_event(
            event=ev, prev_event=prev_ev, next_event=next_ev,
            current_chord=_c_major_chord(), next_chord=None,
        )
        assert ann.nct_type is NCTType.PASSING_TONE


# ---------------------------------------------------------------------------
# TestClassifyEventNeighborTone
# ---------------------------------------------------------------------------


class TestClassifyEventNeighborTone:
    def setup_method(self):
        self.clf = NCTClassifier()

    def test_upper_neighbor(self):
        # E → F → E in C major; F is an upper neighbor of E.
        prev_ev = _note(64)
        ev = _note(65, beat_strength=0.3)
        next_ev = _note(64)
        ann = self.clf.classify_event(
            event=ev, prev_event=prev_ev, next_event=next_ev,
            current_chord=_c_major_chord(), next_chord=None,
        )
        assert ann.nct_type is NCTType.NEIGHBOR_TONE
        assert ann.resolution_pitch == 64

    def test_lower_neighbor(self):
        # G → F# → G (chromatic lower neighbor in C major)
        prev_ev = _note(67)
        ev = _note(66, beat_strength=0.3)
        next_ev = _note(67)
        ann = self.clf.classify_event(
            event=ev, prev_event=prev_ev, next_event=next_ev,
            current_chord=_c_major_chord(), next_chord=None,
        )
        assert ann.nct_type is NCTType.NEIGHBOR_TONE


# ---------------------------------------------------------------------------
# TestClassifyEventSuspension
# ---------------------------------------------------------------------------


class TestClassifyEventSuspension:
    def setup_method(self):
        self.clf = NCTClassifier()

    def test_4_3_suspension(self):
        # F (4) suspended over C major on the downbeat, resolves down to E (3).
        # prev = F from the previous chord; current = F on strong beat; next = E.
        prev_ev = _note(65, onset=-1, beat_strength=0.5)
        ev = _note(65, onset=0, beat_strength=1.0)   # strong beat
        next_ev = _note(64, onset=1, beat_strength=0.5)
        ann = self.clf.classify_event(
            event=ev, prev_event=prev_ev, next_event=next_ev,
            current_chord=_c_major_chord(), next_chord=None,
        )
        assert ann.nct_type is NCTType.SUSPENSION
        assert ann.resolution_pitch == 64


# ---------------------------------------------------------------------------
# TestClassifyEventAnticipation
# ---------------------------------------------------------------------------


class TestClassifyEventAnticipation:
    def setup_method(self):
        self.clf = NCTClassifier()

    def test_anticipation_of_next_chord(self):
        # In a C-major chord, anticipate a D (chord tone of G major next).
        # D is NOT in C major (so it bypasses the chord-tone shortcut) but
        # IS in G major — the defining anticipation condition.
        # Weak beat, same pitch re-struck on next beat into G major.
        prev_ev = _note(64, onset=0, beat_strength=1.0)
        ev = _note(62, onset=1.5, dur=0.5, beat_strength=0.3)  # D, weak
        next_ev = _note(62, onset=2, beat_strength=1.0)        # D on downbeat
        ann = self.clf.classify_event(
            event=ev,
            prev_event=prev_ev,
            next_event=next_ev,
            current_chord=_c_major_chord(0, 2),
            next_chord=_g_major_chord(2, 2),
        )
        assert ann.nct_type is NCTType.ANTICIPATION


# ---------------------------------------------------------------------------
# TestClassifyEventEscapeTone
# ---------------------------------------------------------------------------


class TestClassifyEventEscapeTone:
    def setup_method(self):
        self.clf = NCTClassifier()

    def test_echappee_down_then_leap_up(self):
        # E → D (step down) → G (leap up): D is an escape tone.
        prev_ev = _note(64)
        ev = _note(62, beat_strength=0.3)  # D, weak
        next_ev = _note(67)                # leap up to G
        ann = self.clf.classify_event(
            event=ev, prev_event=prev_ev, next_event=next_ev,
            current_chord=_c_major_chord(), next_chord=None,
        )
        assert ann.nct_type is NCTType.ESCAPE_TONE


# ---------------------------------------------------------------------------
# TestClassifyEventAppoggiatura
# ---------------------------------------------------------------------------


class TestClassifyEventAppoggiatura:
    def setup_method(self):
        self.clf = NCTClassifier()

    def test_leap_up_resolve_down(self):
        # Leap from C (60) up to F (65) on strong beat, then down step to E.
        prev_ev = _note(60, onset=0, beat_strength=0.5)
        ev = _note(65, onset=1, beat_strength=1.0)   # strong beat, dissonant
        next_ev = _note(64, onset=2, beat_strength=0.5)
        ann = self.clf.classify_event(
            event=ev, prev_event=prev_ev, next_event=next_ev,
            current_chord=_c_major_chord(), next_chord=None,
        )
        assert ann.nct_type is NCTType.APPOGGIATURA


# ---------------------------------------------------------------------------
# TestClassifyEventPedalPoint
# ---------------------------------------------------------------------------


class TestClassifyEventPedalPoint:
    def setup_method(self):
        self.clf = NCTClassifier()

    def test_long_bass_note_across_chord_change(self):
        # Bass C sustains for 8 beats, covering both C and G chords.
        # Even though C isn't a chord tone of G major (in this synthetic
        # scenario we purposely mark it as NCT against G), it should be
        # flagged as a pedal point.
        bass_pedal = _note(pitch=36, onset=0, dur=8, beat_strength=1.0)
        current = Chord(
            onset=Fraction(0), duration=Fraction(4),
            root=0, quality="major", pitches=frozenset({0, 4, 7}),
            bass=0, inversion=0,
        )
        # Override: query as if the active chord is G major so that C isn't a CT.
        current_g = Chord(
            onset=Fraction(0), duration=Fraction(4),
            root=7, quality="major", pitches=frozenset({2, 7, 11}),
            bass=7, inversion=0,
        )
        nxt = _g_major_chord(4, 4)
        ann = self.clf.classify_event(
            event=bass_pedal, prev_event=None, next_event=None,
            current_chord=current_g, next_chord=nxt,
        )
        assert ann.nct_type is NCTType.PEDAL_POINT

    def test_short_bass_note_is_not_pedal(self):
        # Same register, but only 1 beat long — fails pedal duration test.
        bass_short = _note(pitch=36, onset=0, dur=1, beat_strength=1.0)
        current_g = Chord(
            onset=Fraction(0), duration=Fraction(4),
            root=7, quality="major", pitches=frozenset({2, 7, 11}),
            bass=7, inversion=0,
        )
        ann = self.clf.classify_event(
            event=bass_short, prev_event=None, next_event=None,
            current_chord=current_g, next_chord=None,
        )
        assert ann.nct_type is not NCTType.PEDAL_POINT


# ---------------------------------------------------------------------------
# TestClassifySequence
# ---------------------------------------------------------------------------


class TestClassifySequence:
    def setup_method(self):
        self.clf = NCTClassifier()

    def test_empty_events_returns_empty(self):
        assert self.clf.classify([], []) == []

    def test_rests_are_skipped(self):
        events = [_rest(0, 1), _note(60, 1, 1), _rest(2, 1)]
        result = self.clf.classify(events, [_c_major_chord(0, 4)])
        assert len(result) == 1
        assert result[0].nct_type is NCTType.CHORD_TONE

    def test_scale_over_c_major(self):
        # C D E F G A B C over a single C-major chord.
        # Under strict "flanked by chord tones" PT rules:
        #   C (CT) - D (PT between CT-CT) - E (CT) - F (PT) - G (CT)
        #   - A (neighbours G & B: B is non-CT, so A is not a strict PT)
        #   - B (neighbours A & C: A is non-CT, so B is not a strict PT)
        #   - C (CT)
        # The A/B pair is a "double passing tone" in classical analysis —
        # we intentionally leave it as UNKNOWN rather than over-claim PT.
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        events = []
        for i, p in enumerate(pitches):
            strength = 1.0 if i % 2 == 0 else 0.3
            events.append(_note(p, onset=i * 0.5, dur=0.5, beat_strength=strength))
        result = self.clf.classify(events, [_c_major_chord(0, 4)])

        roles = [a.nct_type for a in result]
        # Chord tones at indices 0, 2, 4, 7
        for ct_idx in (0, 2, 4, 7):
            assert roles[ct_idx] is NCTType.CHORD_TONE, f"idx {ct_idx}: {roles}"
        # Strict passing tones at indices 1, 3
        for pt_idx in (1, 3):
            assert roles[pt_idx] is NCTType.PASSING_TONE, f"idx {pt_idx}: {roles}"
        # A and B are not CTs in C major and their strict flanks fail the
        # single-PT pattern. We accept UNKNOWN (or a weak NCT match) there
        # rather than force-labelling them.
        assert roles[5] is not NCTType.CHORD_TONE  # A is not a CT of C major
        assert roles[6] is not NCTType.CHORD_TONE  # B is not a CT of C major

    def test_annotations_contain_event_onset_and_pitch(self):
        events = [_note(60, 0), _note(64, 1), _note(67, 2)]
        result = self.clf.classify(events, [_c_major_chord(0, 4)])
        assert [a.pitch for a in result] == [60, 64, 67]
        assert [a.onset for a in result] == [Fraction(0), Fraction(1), Fraction(2)]


# ---------------------------------------------------------------------------
# TestFilterEventsForProfile
# ---------------------------------------------------------------------------


class TestFilterEventsForProfile:
    def setup_method(self):
        self.clf = NCTClassifier()

    def test_no_filter_when_all_strong(self):
        events = [_note(60, 0), _note(64, 1), _note(67, 2)]
        assert self.clf.filter_events_for_profile(events) == events

    def test_filters_weak_stepwise_passing_tone(self):
        # Strong C, weak D between, strong E — the D should be filtered.
        events = [
            _note(60, 0, dur=0.5, beat_strength=1.0),
            _note(62, 0.5, dur=0.5, beat_strength=0.2),
            _note(64, 1, dur=0.5, beat_strength=1.0),
        ]
        filtered = self.clf.filter_events_for_profile(events)
        pitches = [e.pitch for e in filtered]
        assert 62 not in pitches
        assert pitches == [60, 64]

    def test_does_not_filter_when_leap(self):
        # Stepwise in but leap out → keep (not a clean passing tone).
        events = [
            _note(60, 0, beat_strength=1.0),
            _note(62, 1, beat_strength=0.2),
            _note(72, 2, beat_strength=1.0),  # leap up
        ]
        filtered = self.clf.filter_events_for_profile(events)
        assert [e.pitch for e in filtered] == [60, 62, 72]

    def test_does_not_filter_rest_flanked(self):
        events = [
            _note(60, 0, beat_strength=1.0),
            _rest(1, 1),
            _note(62, 2, beat_strength=0.2),
            _rest(3, 1),
            _note(64, 4, beat_strength=1.0),
        ]
        filtered = self.clf.filter_events_for_profile(events)
        # The 62 lacks same-voice non-rest neighbours touching it; rests
        # break the filter's prev/next chain.
        assert 62 in [e.pitch for e in filtered if not e.is_rest]

    def test_preserves_rests(self):
        events = [_note(60, 0), _rest(1, 1), _note(64, 2)]
        filtered = self.clf.filter_events_for_profile(events)
        assert any(e.is_rest for e in filtered)

    def test_short_input_passes_through(self):
        events = [_note(60, 0), _note(62, 1)]
        assert self.clf.filter_events_for_profile(events) == events


# ---------------------------------------------------------------------------
# TestChordClassifierIntegration
# ---------------------------------------------------------------------------


class TestChordClassifierIntegration:
    """Smoke tests ensuring the NCT pre-filter in ChordClassifier behaves."""

    def test_nct_filter_improves_triad_ambiguous_window(self):
        """A beat-weighted window with strong C-major tones + a weak D
        passing tone should cleanly resolve to C major rather than being
        pushed toward a 7th-chord interpretation by the stray D."""
        from src.harmonic.chord_classifier import (
            ChordClassificationConfig,
            ChordClassifier,
        )
        from src.temporal.windower import AnalysisWindow

        events = [
            _note(60, 0, dur=Fraction(1), beat_strength=1.0),   # C (strong)
            _note(62, 1, dur=Fraction(1, 2), beat_strength=0.2),  # D (weak)
            _note(64, Fraction(3, 2), dur=Fraction(1, 2), beat_strength=0.6),  # E
            _note(67, 2, dur=Fraction(1), beat_strength=0.8),   # G (strong)
        ]
        window = AnalysisWindow(
            start=Fraction(0), end=Fraction(3), events=events,
        )
        cc = ChordClassifier()
        cfg_on = ChordClassificationConfig(include_nct_filter=True)
        cfg_off = ChordClassificationConfig(include_nct_filter=False)
        chord_on = cc.classify_window(window, cfg_on)
        chord_off = cc.classify_window(window, cfg_off)
        # Both configurations should still classify something
        assert chord_on is not None
        assert chord_off is not None
        # With the filter on, the root should be C.
        assert chord_on.root == 0
        assert chord_on.quality == "major"

    def test_nct_filter_flag_default_is_true(self):
        from src.harmonic.chord_classifier import ChordClassificationConfig

        cfg = ChordClassificationConfig()
        assert cfg.include_nct_filter is True
