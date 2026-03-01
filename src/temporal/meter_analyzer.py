"""
Detailed metric hierarchy analysis.

Replaces the basic ``_calculate_beat_strength`` in ``score_loader`` with a
full hierarchical subdivision model that handles simple, compound, and
irregular meters.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List

from src.core.data_structures import TimeSignature


@dataclass
class MetricWeight:
    """A single metric position and its hierarchical weight."""
    position: Fraction
    level: int
    weight: float


@dataclass
class MetricGrid:
    """Full metric hierarchy for one measure under a given time signature."""
    time_signature: TimeSignature
    weights: List[MetricWeight] = field(default_factory=list)

    def weight_at(self, position: Fraction) -> float:
        """Look up the weight for a given position, defaulting to 0.1."""
        for w in self.weights:
            if w.position == position:
                return w.weight
        return 0.1


class MeterAnalyzer:
    """Builds and queries hierarchical metric grids."""

    def build_metric_grid(
        self, time_sig: TimeSignature, subdivisions: int = 4
    ) -> MetricGrid:
        """
        Compute hierarchical beat weights for all positions in a measure.

        Weight hierarchy:
            1.0  — downbeat
            0.8  — mid-bar strong beat (beat 3 in 4/4, beat 4 in 6/8)
            0.6  — other main beats
            0.4  — subdivision
            0.2  — sub-subdivision

        Args:
            time_sig: The time signature to analyse.
            subdivisions: How many sub-beat divisions to generate (default 4).

        Returns:
            A ``MetricGrid`` with weights at every position.
        """
        measure_length = Fraction(time_sig.numerator * 4, time_sig.denominator)
        beat_dur = time_sig.beat_duration
        n_beats = time_sig.beats_per_measure

        # Determine the smallest grid unit
        sub_dur = beat_dur / subdivisions

        weights: List[MetricWeight] = []
        pos = Fraction(0)
        while pos < measure_length:
            beat_index = pos / beat_dur  # which beat are we on?

            if pos == 0:
                # Downbeat
                weights.append(MetricWeight(position=pos, level=0, weight=1.0))
            elif beat_index == int(beat_index):
                # On a main beat
                bi = int(beat_index)
                if self._is_secondary_strong(bi, n_beats, time_sig):
                    weights.append(MetricWeight(position=pos, level=1, weight=0.8))
                else:
                    weights.append(MetricWeight(position=pos, level=2, weight=0.6))
            else:
                # Subdivision — check depth
                sub_index = pos / sub_dur
                if sub_index == int(sub_index) and (pos / (beat_dur / 2)) == int(pos / (beat_dur / 2)):
                    weights.append(MetricWeight(position=pos, level=3, weight=0.4))
                else:
                    weights.append(MetricWeight(position=pos, level=4, weight=0.2))

            pos += sub_dur

        return MetricGrid(time_signature=time_sig, weights=weights)

    def get_beat_strength(
        self, position: Fraction, time_sig: TimeSignature
    ) -> float:
        """
        Query the metric weight at any position within a measure.

        A convenience wrapper that builds the grid on the fly. For repeated
        queries within the same time signature, build the grid once and use
        ``MetricGrid.weight_at`` directly.
        """
        grid = self.build_metric_grid(time_sig)
        return grid.weight_at(position)

    @staticmethod
    def _is_secondary_strong(
        beat_index: int, n_beats: int, time_sig: TimeSignature
    ) -> bool:
        """Determine whether *beat_index* is a secondary strong beat."""
        if time_sig.is_compound:
            # In compound meters (6/8 → 2 beats, 12/8 → 4 beats),
            # the mid-point beat is the secondary strong beat.
            return beat_index == n_beats // 2
        # Simple meters: beat 3 in 4/4 is secondary strong
        if n_beats == 4 and beat_index == 2:
            return True
        if n_beats >= 5 and beat_index == n_beats // 2:
            return True
        return False
