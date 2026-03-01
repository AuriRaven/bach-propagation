"""
Unit tests for Quantizer — uses synthetic Score objects, no file I/O.

Default QuantizationConfig: grid_resolution=1/4, tolerance=1/8.
Because tolerance = grid/2, every value will snap with the default config.
Tests that verify "out of tolerance" use a deliberately tighter tolerance.
"""

import pytest
from fractions import Fraction

from src.core.data_structures import MusicalEvent, Score, TimeSignature
from src.temporal.quantizer import Quantizer, QuantizationConfig, QuantizationStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _note(pitch, onset, duration, voice=0):
    return MusicalEvent(
        pitch=pitch, onset=Fraction(onset), duration=Fraction(duration),
        velocity=64, measure=1, beat=Fraction(0), beat_strength=1.0,
        voice=voice, is_rest=False,
    )


def _score(events):
    return Score(
        events=events,
        time_signatures=[(Fraction(0), TimeSignature(4, 4))],
    )


_DEFAULT  = QuantizationConfig()                                      # grid=1/4, tol=1/8
_TIGHT    = QuantizationConfig(grid_resolution=Fraction(1, 4),
                               tolerance=Fraction(1, 32))             # barely snaps within 1/32


# ===========================================================================
# QuantizationConfig
# ===========================================================================

class TestQuantizationConfig:

    def test_default_tolerance_is_half_resolution(self):
        cfg = QuantizationConfig()
        assert cfg.tolerance == cfg.grid_resolution / 2

    def test_explicit_tolerance_kept(self):
        cfg = QuantizationConfig(grid_resolution=Fraction(1, 4),
                                 tolerance=Fraction(1, 16))
        assert cfg.tolerance == Fraction(1, 16)


# ===========================================================================
# QuantizationStats
# ===========================================================================

class TestQuantizationStats:

    def test_mean_deviation_zero_when_nothing_quantized(self):
        stats = QuantizationStats()
        assert stats.mean_onset_deviation == Fraction(0)

    def test_mean_deviation_computed(self):
        stats = QuantizationStats(
            num_quantized=2,
            total_onset_deviation=Fraction(1, 8),
        )
        assert stats.mean_onset_deviation == Fraction(1, 16)


# ===========================================================================
# Quantizer._snap (static method tested via quantize_event)
# ===========================================================================

class TestSnap:

    def test_exact_grid_point_unchanged(self):
        ev = _note(60, Fraction(1, 4), Fraction(1, 4))
        result = Quantizer().quantize_event(ev, _DEFAULT)
        assert result.onset == Fraction(1, 4)

    def test_slightly_below_halfway_snaps_to_lower(self):
        # onset = 1/4 + 1/64 (tiny overshoot past 1/4, still closer to 1/4)
        onset = Fraction(1, 4) + Fraction(1, 64)
        ev = _note(60, onset, Fraction(1, 4))
        result = Quantizer().quantize_event(ev, _DEFAULT)
        assert result.onset == Fraction(1, 4)

    def test_exactly_halfway_snaps_to_lower(self):
        # onset = 1/4 + 1/8 — equidistant; code picks lower
        onset = Fraction(1, 4) + Fraction(1, 8)
        ev = _note(60, onset, Fraction(1, 4))
        result = Quantizer().quantize_event(ev, _DEFAULT)
        assert result.onset == Fraction(1, 4)

    def test_past_halfway_snaps_to_upper(self):
        # onset = 1/4 + 1/8 + 1/64 — closer to 2/4
        onset = Fraction(1, 4) + Fraction(1, 8) + Fraction(1, 64)
        ev = _note(60, onset, Fraction(1, 4))
        result = Quantizer().quantize_event(ev, _DEFAULT)
        assert result.onset == Fraction(2, 4)

    def test_out_of_tolerance_not_snapped(self):
        # With tight tolerance (1/32), onset = 1/4 + 1/16 is 2/32 away from nearest
        # grid point (1/4), which is > 1/32 → not snapped
        onset = Fraction(1, 4) + Fraction(1, 16)
        ev = _note(60, onset, Fraction(1, 4))
        result = Quantizer().quantize_event(ev, _TIGHT)
        assert result.onset == onset   # unchanged

    def test_within_tight_tolerance_snapped(self):
        # onset = 1/4 + 1/64 — only 1/64 away, within 1/32 tolerance
        onset = Fraction(1, 4) + Fraction(1, 64)
        ev = _note(60, onset, Fraction(1, 4))
        result = Quantizer().quantize_event(ev, _TIGHT)
        assert result.onset == Fraction(1, 4)


# ===========================================================================
# Duration snapping
# ===========================================================================

class TestSnapDuration:

    def test_exact_duration_unchanged(self):
        ev = _note(60, Fraction(0), Fraction(1, 4))
        result = Quantizer().quantize_event(ev, _DEFAULT)
        assert result.duration == Fraction(1, 4)

    def test_duration_snaps_to_nearest_grid_multiple(self):
        # dur = 3/8: equidistant between 1/4 and 1/2, picks lower (1/4)
        ev = _note(60, Fraction(0), Fraction(3, 8))
        result = Quantizer().quantize_event(ev, _DEFAULT)
        assert result.duration == Fraction(1, 4)

    def test_duration_snaps_to_half(self):
        # dur = 7/16: closer to 1/2 (= 8/16) than 1/4 (= 4/16)
        ev = _note(60, Fraction(0), Fraction(7, 16))
        result = Quantizer().quantize_event(ev, _DEFAULT)
        assert result.duration == Fraction(1, 2)

    def test_tiny_duration_stays_at_minimum_grid(self):
        # dur = 1/64 — floors to 0, which is below grid, so clamped to grid
        ev = _note(60, Fraction(0), Fraction(1, 64))
        result = Quantizer().quantize_event(ev, _DEFAULT)
        assert result.duration >= _DEFAULT.grid_resolution
        assert result.duration > 0


# ===========================================================================
# quantize_event preserves non-temporal fields
# ===========================================================================

class TestQuantizeEventPreservesFields:

    def test_pitch_preserved(self):
        ev = _note(67, Fraction(0), Fraction(1))
        result = Quantizer().quantize_event(ev, _DEFAULT)
        assert result.pitch == 67

    def test_velocity_preserved(self):
        ev = MusicalEvent(pitch=60, onset=Fraction(0), duration=Fraction(1),
                          velocity=90, measure=1, beat=Fraction(0),
                          beat_strength=0.6, voice=2, is_rest=False)
        result = Quantizer().quantize_event(ev, _DEFAULT)
        assert result.velocity == 90
        assert result.voice == 2
        assert result.beat_strength == 0.6

    def test_rest_preserved(self):
        rest = MusicalEvent(pitch=None, onset=Fraction(0), duration=Fraction(1),
                            velocity=0, is_rest=True)
        result = Quantizer().quantize_event(rest, _DEFAULT)
        assert result.is_rest
        assert result.pitch is None


# ===========================================================================
# quantize_score
# ===========================================================================

class TestQuantizeScore:

    def test_all_aligned_zero_quantized(self):
        events = [_note(60 + i, Fraction(i, 4), Fraction(1, 4)) for i in range(8)]
        score = _score(events)
        _, stats = Quantizer().quantize_score(score, _DEFAULT)
        assert stats.num_quantized == 0
        assert stats.max_onset_deviation == Fraction(0)

    def test_off_grid_event_counted_in_stats(self):
        onset = Fraction(1, 4) + Fraction(1, 64)    # 1/64 off-grid
        events = [_note(60, onset, Fraction(1, 4))]
        score = _score(events)
        _, stats = Quantizer().quantize_score(score, _DEFAULT)
        assert stats.num_quantized == 1
        assert stats.max_onset_deviation == Fraction(1, 64)
        assert stats.total_onset_deviation == Fraction(1, 64)

    def test_output_score_same_event_count(self):
        events = [_note(60 + i, Fraction(i), Fraction(1)) for i in range(6)]
        score = _score(events)
        new_score, _ = Quantizer().quantize_score(score, _DEFAULT)
        assert len(new_score.events) == len(score.events)

    def test_output_score_inherits_time_signatures(self):
        ts = TimeSignature(3, 4)
        events = [_note(60, Fraction(0), Fraction(1))]
        score = Score(events=events, time_signatures=[(Fraction(0), ts)])
        new_score, _ = Quantizer().quantize_score(score, _DEFAULT)
        assert new_score.time_signatures[0][1] == ts

    def test_stats_num_events_matches_input(self):
        events = [_note(60, Fraction(i), Fraction(1)) for i in range(5)]
        score = _score(events)
        _, stats = Quantizer().quantize_score(score, _DEFAULT)
        assert stats.num_events == 5

    def test_mean_deviation_zero_when_nothing_quantized(self):
        events = [_note(60, Fraction(i, 4), Fraction(1, 4)) for i in range(4)]
        score = _score(events)
        _, stats = Quantizer().quantize_score(score, _DEFAULT)
        assert stats.mean_onset_deviation == Fraction(0)
