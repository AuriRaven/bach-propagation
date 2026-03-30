"""
Snap musical events to a regular temporal grid.

Quantization is needed before harmonic analysis to ensure that simultaneous
events are aligned to the same onset and that note durations are
representable as exact fractions.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List

from src.core.data_structures import MusicalEvent, Score, TimeSignature


@dataclass
class QuantizationConfig:
    """Configuration for the quantizer.

    Attributes:
        grid_resolution: Size of each grid cell in quarter notes.
                        Default ``Fraction(1, 4)`` = sixteenth-note grid.
        tolerance: Maximum distance from a grid point for snapping.
                  Events further away are left untouched. Default matches
                  ``grid_resolution / 2``.
    """
    grid_resolution: Fraction = field(default_factory=lambda: Fraction(1, 4))
    tolerance: Fraction = field(default=None)

    def __post_init__(self):
        if self.tolerance is None:
            self.tolerance = self.grid_resolution / 2


@dataclass
class QuantizationStats:
    """Statistics about a quantization pass."""
    num_events: int = 0
    num_quantized: int = 0
    max_onset_deviation: Fraction = Fraction(0)
    total_onset_deviation: Fraction = Fraction(0)

    @property
    def mean_onset_deviation(self) -> Fraction:
        if self.num_quantized == 0:
            return Fraction(0)
        return self.total_onset_deviation / self.num_quantized


class Quantizer:
    """Snaps events to the nearest point on a temporal grid."""

    def quantize_event(
        self, event: MusicalEvent, config: QuantizationConfig
    ) -> MusicalEvent:
        """Snap a single event's onset and duration to the grid.

        Returns a *new* MusicalEvent (MusicalEvent is not frozen, but we avoid
        mutating the original).
        """
        new_onset = self._snap(event.onset, config)
        new_duration = self._snap_duration(event.duration, config)
        # Ensure duration stays positive
        if new_duration <= 0:
            new_duration = config.grid_resolution
        return MusicalEvent(
            pitch=event.pitch,
            onset=new_onset,
            duration=new_duration,
            velocity=event.velocity,
            measure=event.measure,
            beat=event.beat,
            beat_strength=event.beat_strength,
            voice=event.voice,
            is_rest=event.is_rest,
        )

    def quantize_score(
        self, score: Score, config: QuantizationConfig
    ) -> tuple[Score, QuantizationStats]:
        """Quantize all events in a score.

        Returns:
            A tuple of (new_score, stats).
        """
        stats = QuantizationStats()
        new_events: List[MusicalEvent] = []
        for ev in score.events:
            stats.num_events += 1
            q_ev = self.quantize_event(ev, config)
            deviation = abs(q_ev.onset - ev.onset)
            if deviation > 0:
                stats.num_quantized += 1
                stats.total_onset_deviation += deviation
                if deviation > stats.max_onset_deviation:
                    stats.max_onset_deviation = deviation
            new_events.append(q_ev)

        new_score = Score(
            events=new_events,
            time_signatures=list(score.time_signatures),
            title=score.title,
            composer=score.composer,
            tempo=score.tempo,
            metadata=dict(score.metadata),
        )
        return new_score, stats

    @staticmethod
    def _snap(value: Fraction, config: QuantizationConfig) -> Fraction:
        """Snap *value* to the nearest grid point within tolerance."""
        grid = config.grid_resolution
        lower = (value // grid) * grid
        upper = lower + grid
        dist_lower = value - lower
        dist_upper = upper - value
        if dist_lower <= dist_upper:
            nearest, dist = lower, dist_lower
        else:
            nearest, dist = upper, dist_upper
        if dist <= config.tolerance:
            return nearest
        return value

    @staticmethod
    def _snap_duration(dur: Fraction, config: QuantizationConfig) -> Fraction:
        """Snap a duration to the nearest grid multiple."""
        grid = config.grid_resolution
        lower = max(grid, (dur // grid) * grid)
        upper = lower + grid
        if (dur - lower) <= (upper - dur):
            return lower
        return upper
