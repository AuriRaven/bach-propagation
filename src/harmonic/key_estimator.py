"""
Key estimation using the Krumhansl-Schmuckler algorithm.

Analyses pitch-class distributions in MusicalEvent lists and returns KeyEstimate
objects. Operates entirely on our core data structures — no music21 required.
"""

import math
from fractions import Fraction
from typing import List, Tuple

from src.core.data_structures import KeyEstimate, MusicalEvent, Score
from src.temporal.windower import Windower

# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles (Krumhansl & Schmuckler 1986)
# Index 0 = C, index 1 = C#, …, index 11 = B
# ---------------------------------------------------------------------------
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


class KeyEstimator:
    """
    Krumhansl-Schmuckler key-finding on lists of MusicalEvent.

    All methods are stateless — instantiate once and reuse freely.
    """

    def estimate(
        self,
        events: List[MusicalEvent],
        onset: Fraction,
    ) -> KeyEstimate:
        """
        Estimate the key from a list of events.

        Algorithm:
          1. Build a 12-element pitch-class histogram weighted by event duration.
          2. For each of 24 candidates (12 roots × 2 modes), rotate the KS
             profile to that root and compute Pearson r against the histogram.
          3. Select the candidate with the highest r.
          4. Normalize r to [0, 1] as confidence = (r + 1) / 2.

        Args:
            events: MusicalEvent list; rests and pitched events with pitch=None
                    are silently skipped.
            onset:  Score position this estimate applies to.

        Returns:
            KeyEstimate with tonic (0–11), mode, confidence (0–1), onset.
        """
        histogram = self._build_histogram(events)
        if sum(histogram) == 0.0:
            return KeyEstimate(tonic=0, mode="major", confidence=0.0, onset=onset)

        best_r = -2.0
        best_tonic = 0
        best_mode = "major"

        for root in range(12):
            for mode, profile in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
                rotated = [profile[(i - root) % 12] for i in range(12)]
                r = self._correlate(histogram, rotated)
                if r > best_r:
                    best_r = r
                    best_tonic = root
                    best_mode = mode

        confidence = (best_r + 1.0) / 2.0
        # Guard against floating-point drift outside [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        return KeyEstimate(tonic=best_tonic, mode=best_mode, confidence=confidence, onset=onset)

    def estimate_global(self, score: Score) -> KeyEstimate:
        """
        Whole-score key estimate.

        Uses all events in score.events. onset is fixed to Fraction(0).

        Returns:
            KeyEstimate at onset=Fraction(0).
        """
        return self.estimate(score.events, Fraction(0))

    def estimate_local(
        self,
        score: Score,
        window_size: Fraction,
        hop: Fraction,
    ) -> List[KeyEstimate]:
        """
        Sliding-window local key estimates.

        Uses Windower.sliding to segment the score, then calls estimate on each
        window. Empty windows (no pitched events) produce a KeyEstimate with
        confidence=0.0 as a sentinel.

        Args:
            score: Score to analyse.
            window_size: Width of each window in quarter notes.
            hop: Step size between window starts in quarter notes.

        Returns:
            List of KeyEstimate objects in onset order.
        """
        windows = Windower().sliding(score, window_size, hop)
        return [self.estimate(w.events, w.start) for w in windows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_histogram(self, events: List[MusicalEvent]) -> List[float]:
        """
        Build a 12-element pitch-class histogram weighted by duration.

        Rests (is_rest=True or pitch is None) are skipped.
        Each pitched note contributes float(duration) to its pitch-class bin.

        Returns:
            List[float] of length 12. All zeros if there are no pitched events.
        """
        histogram = [0.0] * 12
        for ev in events:
            if ev.is_rest or ev.pitch is None:
                continue
            histogram[ev.pitch % 12] += float(ev.duration)
        return histogram

    def _correlate(self, histogram: List[float], profile: List[float]) -> float:
        """
        Pearson correlation coefficient between two length-12 vectors.

        Returns 0.0 if the histogram has zero variance (e.g. single pitch class
        or all-zero input after the zero-sum guard in estimate()).

        Formula:
            r = Σ(h_i - h̄)(p_i - p̄) / sqrt(Σ(h_i - h̄)² · Σ(p_i - p̄)²)
        """
        n = len(histogram)
        h_mean = sum(histogram) / n
        p_mean = sum(profile) / n

        num = sum((histogram[i] - h_mean) * (profile[i] - p_mean) for i in range(n))
        h_var = sum((histogram[i] - h_mean) ** 2 for i in range(n))
        p_var = sum((profile[i] - p_mean) ** 2 for i in range(n))

        denom = math.sqrt(h_var * p_var)
        if denom == 0.0:
            return 0.0
        return num / denom
