"""
Segment a score into analysis windows for downstream harmonic analysis.

Supports beat-level, measure-level, fixed-duration, and sliding windows.
"""

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import List, Optional

from src.core.data_structures import MusicalEvent, Score


class WindowType(Enum):
    """Supported windowing strategies."""
    BEAT = "beat"
    MEASURE = "measure"
    FIXED_DURATION = "fixed_duration"
    SLIDING = "sliding"


@dataclass
class AnalysisWindow:
    """A single analysis window containing the events that fall within it."""
    start: Fraction
    end: Fraction
    events: List[MusicalEvent] = field(default_factory=list)
    measure: int = 1
    beat_position: Fraction = field(default_factory=lambda: Fraction(0))


class Windower:
    """Creates analysis windows from a Score."""

    def create_windows(
        self,
        score: Score,
        window_type: WindowType,
        size: Optional[Fraction] = None,
        hop: Optional[Fraction] = None,
    ) -> List[AnalysisWindow]:
        """Dispatch to the appropriate windowing strategy."""
        if window_type == WindowType.BEAT:
            return self.by_beat(score)
        elif window_type == WindowType.MEASURE:
            return self.by_measure(score)
        elif window_type == WindowType.FIXED_DURATION:
            if size is None:
                raise ValueError("size is required for FIXED_DURATION windows")
            return self._fixed(score, size)
        elif window_type == WindowType.SLIDING:
            if size is None:
                raise ValueError("size is required for SLIDING windows")
            if hop is None:
                hop = size / 2
            return self.sliding(score, size, hop)
        raise ValueError(f"Unknown window type: {window_type}")

    def by_beat(self, score: Score) -> List[AnalysisWindow]:
        """One window per beat, determined by time signatures."""
        windows: List[AnalysisWindow] = []
        duration = score.duration
        pos = Fraction(0)

        while pos < duration:
            ts = score.get_time_signature_at(pos)
            beat_dur = ts.beat_duration
            end = min(pos + beat_dur, duration)

            # Determine measure number and beat position
            measure_len = Fraction(ts.numerator * 4, ts.denominator)
            measure_num = int(pos // measure_len) + 1
            beat_pos = pos % measure_len

            events = score.get_events_in_range(pos, end)
            windows.append(AnalysisWindow(
                start=pos, end=end, events=events,
                measure=measure_num, beat_position=beat_pos,
            ))
            pos = end

        return windows

    def by_measure(self, score: Score) -> List[AnalysisWindow]:
        """One window per measure."""
        windows: List[AnalysisWindow] = []
        duration = score.duration
        pos = Fraction(0)
        measure_num = 1

        while pos < duration:
            ts = score.get_time_signature_at(pos)
            measure_len = Fraction(ts.numerator * 4, ts.denominator)
            end = min(pos + measure_len, duration)

            events = score.get_events_in_range(pos, end)
            windows.append(AnalysisWindow(
                start=pos, end=end, events=events,
                measure=measure_num, beat_position=Fraction(0),
            ))
            pos = end
            measure_num += 1

        return windows

    def sliding(
        self, score: Score, size: Fraction, hop: Fraction
    ) -> List[AnalysisWindow]:
        """Sliding window with configurable size and hop."""
        windows: List[AnalysisWindow] = []
        duration = score.duration
        pos = Fraction(0)

        while pos < duration:
            end = min(pos + size, duration)
            ts = score.get_time_signature_at(pos)
            measure_len = Fraction(ts.numerator * 4, ts.denominator)
            measure_num = int(pos // measure_len) + 1
            beat_pos = pos % measure_len

            events = score.get_events_in_range(pos, end)
            windows.append(AnalysisWindow(
                start=pos, end=end, events=events,
                measure=measure_num, beat_position=beat_pos,
            ))
            pos += hop

        return windows

    def _fixed(self, score: Score, size: Fraction) -> List[AnalysisWindow]:
        """Non-overlapping fixed-duration windows."""
        return self.sliding(score, size, hop=size)
