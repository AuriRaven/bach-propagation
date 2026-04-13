"""
Tests for src/core/voice_separator.py

All fixtures are synthetic MusicalEvent lists so the tests do not depend on
any MIDI/MusicXML files being present.

Structure:
    TestMonophonicLineStaysInOneVoice   – stepwise melody shouldn't split
    TestOctaveOscillationSplits         – hi/lo/hi/lo ⇒ two voices
    TestArpeggioSplits                  – 3-register arpeggio ⇒ 2–3 voices
    TestPedalPlusMelody                 – sustained bass + upper melody
    TestMaxVoicesCap                    – respects config.max_voices
    TestRestsPreserved                  – rests carry through
    TestSimultaneityGoesToDifferentVoices – chord stack splits
    TestScoreReassembly                 – to_score() returns usable Score
    TestVoiceProfiles                   – per-voice statistics correct
"""

from fractions import Fraction
from typing import List

import pytest

from src.core.data_structures import MusicalEvent, Score, TimeSignature
from src.core.voice_separator import (
    VoiceProfile,
    VoiceSeparationConfig,
    VoiceSeparationResult,
    VoiceSeparator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _note(
    pitch: int,
    onset: float,
    dur: float = 1,
    voice: int = 0,
    beat_strength: float = 1.0,
) -> MusicalEvent:
    return MusicalEvent(
        pitch=pitch,
        onset=Fraction(onset),
        duration=Fraction(dur),
        beat_strength=beat_strength,
        voice=voice,
    )


def _rest(onset: float, dur: float = 1, voice: int = 0) -> MusicalEvent:
    return MusicalEvent(
        pitch=None,
        onset=Fraction(onset),
        duration=Fraction(dur),
        is_rest=True,
        velocity=0,
        voice=voice,
    )


def _score(events: List[MusicalEvent], num: int = 4, den: int = 4) -> Score:
    return Score(
        events=events,
        time_signatures=[(Fraction(0), TimeSignature(num, den))],
    )


# ---------------------------------------------------------------------------
# TestMonophonicLineStaysInOneVoice
# ---------------------------------------------------------------------------


class TestMonophonicLineStaysInOneVoice:
    def test_ascending_c_major_scale(self):
        # C D E F G A B C — stepwise, should remain a single voice.
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        events = [_note(p, onset=i, dur=1) for i, p in enumerate(pitches)]
        result = VoiceSeparator().separate(_score(events))
        voices = {e.voice for e in result.events}
        assert voices == {0}
        assert result.num_voices == 1

    def test_short_leaps_within_octave_stay_single(self):
        # Small intra-octave leaps (e.g. arpeggiating within one register).
        events = [
            _note(60, 0), _note(64, 1), _note(67, 2), _note(64, 3),
            _note(60, 4), _note(64, 5),
        ]
        result = VoiceSeparator().separate(_score(events))
        assert result.num_voices == 1


# ---------------------------------------------------------------------------
# TestOctaveOscillationSplits
# ---------------------------------------------------------------------------


class TestOctaveOscillationSplits:
    def test_two_octave_oscillation_splits_into_two_voices(self):
        # C2 (36) alternates with C5 (72) — a 3-octave leap each time.
        # No sensible single voice can produce this, so it must split.
        pitches = [36, 72, 36, 72, 36, 72, 36, 72]
        events = [_note(p, onset=i, dur=1) for i, p in enumerate(pitches)]
        result = VoiceSeparator().separate(_score(events))
        assert result.num_voices >= 2
        # Each pitch class should end up in exactly one voice.
        voice_of_low = {e.voice for e in result.events if e.pitch == 36}
        voice_of_high = {e.voice for e in result.events if e.pitch == 72}
        assert len(voice_of_low) == 1
        assert len(voice_of_high) == 1
        assert voice_of_low != voice_of_high


# ---------------------------------------------------------------------------
# TestArpeggioSplits
# ---------------------------------------------------------------------------


class TestArpeggioSplits:
    def test_three_register_arpeggio(self):
        # Classic bass-alto-soprano broken chord pattern:
        # G2 (43) – D4 (62) – B4 (71) repeated. The octave-plus leaps
        # should force register stratification.
        pitches = [43, 62, 71, 43, 62, 71, 43, 62, 71]
        events = [_note(p, onset=i, dur=1) for i, p in enumerate(pitches)]
        cfg = VoiceSeparationConfig(max_voices=3, new_voice_threshold=10.0)
        result = VoiceSeparator().separate(_score(events), cfg)
        # At minimum the bass (43) should split off from the upper register.
        assert result.num_voices >= 2
        bass_voices = {e.voice for e in result.events if e.pitch == 43}
        top_voices = {e.voice for e in result.events if e.pitch == 71}
        assert bass_voices != top_voices


# ---------------------------------------------------------------------------
# TestPedalPlusMelody
# ---------------------------------------------------------------------------


class TestPedalPlusMelody:
    def test_pedal_bass_separates_from_upper_melody(self):
        # A low G2 pedal every other beat while a melody moves in the upper
        # octave. The pedal repeats on the same pitch so it must be a voice
        # of its own — any attempt to pull it into the melody voice would
        # demand a 2-octave leap each beat.
        events = [
            _note(43, 0, 1),   # G2 pedal
            _note(72, 1, 1),   # C5 melody
            _note(43, 2, 1),   # G2 pedal
            _note(74, 3, 1),   # D5 melody
            _note(43, 4, 1),   # G2 pedal
            _note(76, 5, 1),   # E5 melody
        ]
        result = VoiceSeparator().separate(_score(events))
        assert result.num_voices >= 2
        pedal_voices = {e.voice for e in result.events if e.pitch == 43}
        melody_voices = {e.voice for e in result.events if e.pitch != 43}
        # All pedal notes in one voice
        assert len(pedal_voices) == 1
        # Pedal voice and melody voice(s) are distinct
        assert pedal_voices.isdisjoint(melody_voices)


# ---------------------------------------------------------------------------
# TestMaxVoicesCap
# ---------------------------------------------------------------------------


class TestMaxVoicesCap:
    def test_respects_max_voices_cap(self):
        # 4 disjoint registers but max_voices=2: separator must squash two
        # of them into the same voice.
        pitches = [36, 60, 72, 84, 36, 60, 72, 84]
        events = [_note(p, onset=i, dur=1) for i, p in enumerate(pitches)]
        cfg = VoiceSeparationConfig(max_voices=2, new_voice_threshold=5.0)
        result = VoiceSeparator().separate(_score(events), cfg)
        assert result.num_voices <= 2

    def test_single_voice_cap(self):
        # With max_voices=1, everything is forced into voice 0 even if
        # there are massive leaps.
        events = [_note(36, 0), _note(84, 1), _note(36, 2), _note(84, 3)]
        cfg = VoiceSeparationConfig(max_voices=1)
        result = VoiceSeparator().separate(_score(events), cfg)
        assert result.num_voices == 1
        assert {e.voice for e in result.events} == {0}


# ---------------------------------------------------------------------------
# TestRestsPreserved
# ---------------------------------------------------------------------------


class TestRestsPreserved:
    def test_rests_carry_through(self):
        events = [_note(60, 0), _rest(1, 1), _note(62, 2), _rest(3, 1), _note(64, 4)]
        result = VoiceSeparator().separate(_score(events))
        # Rests still present in output
        rest_count = sum(1 for e in result.events if e.is_rest)
        assert rest_count == 2
        # Rests keep their original voice (0)
        for e in result.events:
            if e.is_rest:
                assert e.voice == 0

    def test_event_count_preserved(self):
        events = [_note(60, 0), _rest(1, 1), _note(62, 2), _note(72, 3)]
        result = VoiceSeparator().separate(_score(events))
        assert len(result.events) == len(events)

    def test_input_order_preserved(self):
        events = [_note(60, 0), _note(72, 1), _note(60, 2), _note(72, 3)]
        result = VoiceSeparator().separate(_score(events))
        assert [e.pitch for e in result.events] == [e.pitch for e in events]
        assert [e.onset for e in result.events] == [e.onset for e in events]


# ---------------------------------------------------------------------------
# TestSimultaneityGoesToDifferentVoices
# ---------------------------------------------------------------------------


class TestSimultaneityGoesToDifferentVoices:
    def test_two_simultaneous_notes_split(self):
        # Two notes sounding at onset=0, one high one low.
        events = [
            _note(60, 0, dur=4),
            _note(72, 0, dur=4),
            _note(64, 4, dur=4),
            _note(76, 4, dur=4),
        ]
        result = VoiceSeparator().separate(_score(events))
        # The pair at onset=0 must occupy different voices.
        voices_at_0 = {e.voice for e in result.events if e.onset == 0}
        assert len(voices_at_0) == 2

    def test_sustained_voice_blocks_same_voice_reuse(self):
        # Voice 0 has a whole note; a new event during its span cannot
        # reuse voice 0.
        events = [
            _note(60, 0, dur=4),   # voice 0 sustains from 0–4
            _note(72, 1, dur=1),   # new note mid-sustain → must go elsewhere
        ]
        result = VoiceSeparator().separate(_score(events))
        v_long = [e.voice for e in result.events if e.duration == 4][0]
        v_short = [e.voice for e in result.events if e.duration == 1][0]
        assert v_long != v_short


# ---------------------------------------------------------------------------
# TestScoreReassembly
# ---------------------------------------------------------------------------


class TestScoreReassembly:
    def test_to_score_preserves_metadata(self):
        events = [_note(60, 0), _note(72, 1), _note(60, 2), _note(72, 3)]
        src = Score(
            events=events,
            time_signatures=[(Fraction(0), TimeSignature(3, 4))],
            title="Test Piece",
            composer="J. S. Bach",
            tempo=120,
        )
        sep = VoiceSeparator()
        result = sep.separate(src)
        rebuilt = sep.to_score(src, result)
        assert rebuilt.title == "Test Piece"
        assert rebuilt.composer == "J. S. Bach"
        assert rebuilt.tempo == 120
        assert rebuilt.time_signatures == src.time_signatures

    def test_to_score_has_expected_num_voices(self):
        events = [_note(60, 0), _note(84, 1), _note(60, 2), _note(84, 3)]
        src = _score(events)
        sep = VoiceSeparator()
        result = sep.separate(src)
        rebuilt = sep.to_score(src, result)
        # Score.num_voices returns max(voice)+1
        assert rebuilt.num_voices == result.num_voices


# ---------------------------------------------------------------------------
# TestVoiceProfiles
# ---------------------------------------------------------------------------


class TestVoiceProfiles:
    def test_profiles_present_for_each_voice(self):
        events = [_note(36, 0), _note(72, 1), _note(36, 2), _note(72, 3)]
        result = VoiceSeparator().separate(_score(events))
        assert set(result.voice_profiles.keys()) == set(
            {e.voice for e in result.events if not e.is_rest}
        )

    def test_profile_register_matches_contents(self):
        events = [_note(36, 0), _note(38, 1), _note(36, 2), _note(38, 3)]
        result = VoiceSeparator().separate(_score(events))
        # One voice, pitches 36 and 38
        profile = result.voice_profiles[0]
        assert profile.min_pitch == 36
        assert profile.max_pitch == 38
        assert profile.note_count == 4

    def test_profile_mean_pitch_weighted_by_duration(self):
        # Two notes, equal duration → mean is (60 + 64) / 2 = 62.0
        events = [_note(60, 0, dur=1), _note(64, 1, dur=1)]
        result = VoiceSeparator().separate(_score(events))
        profile = result.voice_profiles[0]
        assert profile.mean_pitch == pytest.approx(62.0)

    def test_empty_events_returns_empty_result(self):
        result = VoiceSeparator().separate(_score([_rest(0, 4)]))
        assert result.num_voices == 0
        assert result.voice_profiles == {}


# ---------------------------------------------------------------------------
# TestSeparateEventsDirect
# ---------------------------------------------------------------------------


class TestSeparateEventsDirect:
    def test_separate_events_works_without_score(self):
        events = [_note(60, 0), _note(72, 1), _note(60, 2), _note(72, 3)]
        result = VoiceSeparator().separate_events(events)
        assert isinstance(result, VoiceSeparationResult)
        assert result.num_voices >= 1
