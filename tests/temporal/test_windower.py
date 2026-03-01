"""
Unit tests for Windower — uses synthetic Score objects, no file I/O.

Synthetic fixture: 2 measures of 4/4, one quarter-note event per beat.
  Onsets: 0, 1, 2, 3 (measure 1) | 4, 5, 6, 7 (measure 2)
  Total events: 8   |   Score duration: 8 quarter notes
"""

import pytest
from fractions import Fraction

from src.core.data_structures import MusicalEvent, Score, TimeSignature
from src.temporal.windower import AnalysisWindow, WindowType, Windower


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _note(pitch, onset, voice=0):
    return MusicalEvent(
        pitch=pitch, onset=Fraction(onset), duration=Fraction(1),
        velocity=64, measure=(int(onset) // 4) + 1,
        beat=Fraction(int(onset) % 4), beat_strength=1.0 if onset % 4 == 0 else 0.6,
        voice=voice, is_rest=False,
    )


def _score_4_4(n_measures: int = 2) -> Score:
    """n_measures × 4 quarter-note events in 4/4, one per beat."""
    events = [_note(60 + i % 12, i) for i in range(n_measures * 4)]
    return Score(
        events=events,
        time_signatures=[(Fraction(0), TimeSignature(4, 4))],
    )


# ===========================================================================
# by_measure
# ===========================================================================

class TestByMeasure:

    def test_window_count_matches_measures(self):
        assert len(Windower().by_measure(_score_4_4(2))) == 2
        assert len(Windower().by_measure(_score_4_4(4))) == 4

    def test_first_window_spans_first_measure(self):
        windows = Windower().by_measure(_score_4_4(2))
        assert windows[0].start == Fraction(0)
        assert windows[0].end   == Fraction(4)

    def test_second_window_spans_second_measure(self):
        windows = Windower().by_measure(_score_4_4(2))
        assert windows[1].start == Fraction(4)
        assert windows[1].end   == Fraction(8)

    def test_windows_are_contiguous(self):
        windows = Windower().by_measure(_score_4_4(3))
        for i in range(len(windows) - 1):
            assert windows[i].end == windows[i + 1].start

    def test_measure_numbers_are_sequential(self):
        windows = Windower().by_measure(_score_4_4(3))
        for i, w in enumerate(windows):
            assert w.measure == i + 1

    def test_events_partitioned_across_windows(self):
        score  = _score_4_4(2)
        windows = Windower().by_measure(score)
        assert len(windows[0].events) == 4
        assert len(windows[1].events) == 4

    def test_no_empty_windows(self):
        for w in Windower().by_measure(_score_4_4(2)):
            assert len(w.events) > 0

    def test_last_window_end_equals_score_duration(self):
        score   = _score_4_4(3)
        windows = Windower().by_measure(score)
        assert windows[-1].end == score.duration


# ===========================================================================
# by_beat
# ===========================================================================

class TestByBeat:

    def test_window_count_is_measures_times_beats(self):
        # 2 measures × 4 beats = 8 beat windows
        windows = Windower().by_beat(_score_4_4(2))
        assert len(windows) == 8

    def test_each_beat_window_has_one_event(self):
        windows = Windower().by_beat(_score_4_4(2))
        for w in windows:
            assert len(w.events) == 1

    def test_first_beat_window_span(self):
        w = Windower().by_beat(_score_4_4(2))[0]
        assert w.start == Fraction(0)
        assert w.end   == Fraction(1)

    def test_windows_are_contiguous(self):
        windows = Windower().by_beat(_score_4_4(2))
        for i in range(len(windows) - 1):
            assert windows[i].end == windows[i + 1].start

    def test_beat_positions_restart_each_measure(self):
        windows = Windower().by_beat(_score_4_4(2))
        # beat_position 0 should appear at start of each measure (windows 0 and 4)
        assert windows[0].beat_position == Fraction(0)
        assert windows[4].beat_position == Fraction(0)


# ===========================================================================
# sliding
# ===========================================================================

class TestSliding:

    def test_window_count_with_half_hop(self):
        # score duration=8, size=2, hop=1 → windows at 0,1,2,3,4,5,6,7 → 8 windows
        windows = Windower().sliding(_score_4_4(2), Fraction(2), Fraction(1))
        assert len(windows) == 8

    def test_non_overlapping_hop_equals_size(self):
        # size=hop=2 → 4 windows for 8-quarter score
        windows = Windower().sliding(_score_4_4(2), Fraction(2), Fraction(2))
        assert len(windows) == 4

    def test_window_size_correct(self):
        windows = Windower().sliding(_score_4_4(2), Fraction(2), Fraction(2))
        for w in windows:
            assert w.end - w.start <= Fraction(2)   # last window may be shorter

    def test_last_window_clipped_to_score_end(self):
        score   = _score_4_4(2)
        windows = Windower().sliding(score, Fraction(3), Fraction(3))
        assert windows[-1].end == score.duration

    def test_windows_start_at_correct_offsets(self):
        windows = Windower().sliding(_score_4_4(2), Fraction(2), Fraction(2))
        starts = [w.start for w in windows]
        assert starts == [Fraction(0), Fraction(2), Fraction(4), Fraction(6)]

    def test_overlapping_windows_share_events(self):
        # hop=1, size=2 → windows[0]=[0,2), windows[1]=[1,3)
        # event at onset=1 is in both
        score   = _score_4_4(2)
        windows = Windower().sliding(score, Fraction(2), Fraction(1))
        events_0 = {ev.onset for ev in windows[0].events}
        events_1 = {ev.onset for ev in windows[1].events}
        assert Fraction(1) in events_0 & events_1


# ===========================================================================
# create_windows dispatch
# ===========================================================================

class TestCreateWindows:

    def test_dispatch_beat(self):
        score = _score_4_4(2)
        w1 = Windower().create_windows(score, WindowType.BEAT)
        w2 = Windower().by_beat(score)
        assert len(w1) == len(w2)

    def test_dispatch_measure(self):
        score = _score_4_4(2)
        w1 = Windower().create_windows(score, WindowType.MEASURE)
        w2 = Windower().by_measure(score)
        assert len(w1) == len(w2)

    def test_dispatch_fixed_duration(self):
        score = _score_4_4(2)
        size  = Fraction(2)
        w1 = Windower().create_windows(score, WindowType.FIXED_DURATION, size=size)
        w2 = Windower().sliding(score, size, size)
        assert len(w1) == len(w2)

    def test_dispatch_sliding(self):
        score = _score_4_4(2)
        size, hop = Fraction(2), Fraction(1)
        w1 = Windower().create_windows(score, WindowType.SLIDING, size=size, hop=hop)
        w2 = Windower().sliding(score, size, hop)
        assert len(w1) == len(w2)

    def test_sliding_default_hop_is_half_size(self):
        # When hop is omitted, default = size / 2
        score = _score_4_4(2)
        size  = Fraction(2)
        w_default = Windower().create_windows(score, WindowType.SLIDING, size=size)
        w_explicit = Windower().sliding(score, size, size / 2)
        assert len(w_default) == len(w_explicit)

    def test_fixed_duration_missing_size_raises(self):
        with pytest.raises(ValueError):
            Windower().create_windows(_score_4_4(2), WindowType.FIXED_DURATION)

    def test_sliding_missing_size_raises(self):
        with pytest.raises(ValueError):
            Windower().create_windows(_score_4_4(2), WindowType.SLIDING)


# ===========================================================================
# AnalysisWindow fields
# ===========================================================================

class TestAnalysisWindowFields:

    def test_window_has_correct_fields(self):
        w = AnalysisWindow(
            start=Fraction(0), end=Fraction(4),
            events=[], measure=1,
            beat_position=Fraction(0),
        )
        assert w.start == Fraction(0)
        assert w.end   == Fraction(4)
        assert w.measure == 1
        assert w.beat_position == Fraction(0)

    def test_empty_events_by_default(self):
        w = AnalysisWindow(start=Fraction(0), end=Fraction(1))
        assert w.events == []
