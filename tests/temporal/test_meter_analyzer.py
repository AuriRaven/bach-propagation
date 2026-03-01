"""
Pure unit tests for MeterAnalyzer — no file I/O.

Weight hierarchy produced by build_metric_grid (default subdivisions=4):
  1.0  downbeat          (level 0)
  0.8  secondary strong  (level 1) – beat 3 in 4/4, beat 2 in 6/8
  0.6  main beat         (level 2)
  0.4  half-beat         (level 3)
  0.2  sub-subdivision   (level 4)
  0.1  fallback (weight_at for unknown positions)
"""

import pytest
from fractions import Fraction

from src.core.data_structures import TimeSignature
from src.temporal.meter_analyzer import MeterAnalyzer, MetricGrid, MetricWeight


# ===========================================================================
# MetricGrid.weight_at
# ===========================================================================

class TestMetricGrid:

    def test_returns_weight_for_known_position(self):
        grid = MetricGrid(
            time_signature=TimeSignature(4, 4),
            weights=[
                MetricWeight(position=Fraction(0), level=0, weight=1.0),
                MetricWeight(position=Fraction(1), level=2, weight=0.6),
            ],
        )
        assert grid.weight_at(Fraction(0)) == 1.0
        assert grid.weight_at(Fraction(1)) == 0.6

    def test_fallback_for_unknown_position(self):
        grid = MetricGrid(time_signature=TimeSignature(4, 4), weights=[])
        assert grid.weight_at(Fraction(99)) == 0.1


# ===========================================================================
# 4/4 — the reference meter
# ===========================================================================

class TestMeterAnalyzer44:
    """
    4/4 with subdivisions=4 → sixteenth-note grid.
    Measure length = 4 quarters → 16 grid positions (0, 1/4, 1/2, …, 15/4).
    """

    @pytest.fixture(scope="class")
    def grid(self):
        return MeterAnalyzer().build_metric_grid(TimeSignature(4, 4))

    def test_grid_has_16_positions(self, grid):
        assert len(grid.weights) == 16

    def test_downbeat(self, grid):
        assert grid.weight_at(Fraction(0)) == 1.0

    def test_beat_2(self, grid):
        assert grid.weight_at(Fraction(1)) == 0.6

    def test_beat_3_secondary_strong(self, grid):
        assert grid.weight_at(Fraction(2)) == 0.8

    def test_beat_4(self, grid):
        assert grid.weight_at(Fraction(3)) == 0.6

    def test_half_beat_eighth_note(self, grid):
        # half of beat_duration (1) = 1/2
        assert grid.weight_at(Fraction(1, 2)) == 0.4

    def test_quarter_beat_sixteenth_note(self, grid):
        assert grid.weight_at(Fraction(1, 4)) == 0.2

    def test_all_weights_positive_and_leq_one(self, grid):
        for w in grid.weights:
            assert 0.0 < w.weight <= 1.0

    def test_positions_span_full_measure(self, grid):
        positions = {w.position for w in grid.weights}
        assert Fraction(0) in positions
        assert max(positions) < Fraction(4)   # measure length

    def test_all_positions_ascending(self, grid):
        positions = [w.position for w in grid.weights]
        assert positions == sorted(positions)


# ===========================================================================
# 3/4
# ===========================================================================

class TestMeterAnalyzer34:
    """
    3/4 with subdivisions=4 → 12 positions (0, 1/4, …, 11/4).
    No secondary strong beat (no beat that qualifies in 3/4).
    """

    @pytest.fixture(scope="class")
    def grid(self):
        return MeterAnalyzer().build_metric_grid(TimeSignature(3, 4))

    def test_grid_has_12_positions(self, grid):
        assert len(grid.weights) == 12

    def test_downbeat(self, grid):
        assert grid.weight_at(Fraction(0)) == 1.0

    def test_beat_2_plain_strong(self, grid):
        assert grid.weight_at(Fraction(1)) == 0.6

    def test_beat_3_plain_strong(self, grid):
        # 3/4 has no secondary-strong beat — both inner beats are 0.6
        assert grid.weight_at(Fraction(2)) == 0.6

    def test_half_beat(self, grid):
        assert grid.weight_at(Fraction(1, 2)) == 0.4


# ===========================================================================
# 6/8 — compound meter
# ===========================================================================

class TestMeterAnalyzer68:
    """
    6/8: compound, beats_per_measure=2, beat_duration=3/2.
    With subdivisions=4, sub_dur = (3/2)/4 = 3/8.
    Grid positions: 0, 3/8, 3/4, 9/8, 3/2, 15/8, 9/4, 21/8  (8 total).
    """

    @pytest.fixture(scope="class")
    def grid(self):
        return MeterAnalyzer().build_metric_grid(TimeSignature(6, 8))

    def test_grid_has_8_positions(self, grid):
        assert len(grid.weights) == 8

    def test_downbeat(self, grid):
        assert grid.weight_at(Fraction(0)) == 1.0

    def test_second_main_beat_is_secondary_strong(self, grid):
        # beat 2 at offset = beat_duration = 3/2
        assert grid.weight_at(Fraction(3, 2)) == 0.8

    def test_half_beat_of_first_beat(self, grid):
        # half of beat_duration (3/2) = 3/4
        assert grid.weight_at(Fraction(3, 4)) == 0.4

    def test_sub_subdivision(self, grid):
        # 3/8 is one sub_dur step — not a half-beat position
        assert grid.weight_at(Fraction(3, 8)) == 0.2


# ===========================================================================
# get_beat_strength convenience wrapper
# ===========================================================================

class TestGetBeatStrength:

    def test_consistent_with_grid_for_44(self):
        ma = MeterAnalyzer()
        ts = TimeSignature(4, 4)
        grid = ma.build_metric_grid(ts)
        for w in grid.weights:
            assert ma.get_beat_strength(w.position, ts) == w.weight

    def test_unknown_position_returns_fallback(self):
        ma = MeterAnalyzer()
        ts = TimeSignature(4, 4)
        assert ma.get_beat_strength(Fraction(99), ts) == 0.1
