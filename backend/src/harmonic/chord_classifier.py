"""
Chord classification by weighted pitch-class template matching.

Takes an AnalysisWindow (from src.temporal.windower) and identifies the best-
matching chord using beat-strength-weighted coverage scores against all standard
triad and seventh-chord templates.
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, FrozenSet, List, Optional, Tuple

from src.core.data_structures import Chord, MusicalEvent, Score
from src.temporal.windower import AnalysisWindow, WindowType, Windower

# ---------------------------------------------------------------------------
# Chord templates: interval sets from root (root = 0)
# Quality strings match data_structures._QUALITY_SUFFIXES exactly.
# ---------------------------------------------------------------------------
_CHORD_TEMPLATES: Dict[str, FrozenSet[int]] = {
    "major":      frozenset({0, 4, 7}),
    "minor":      frozenset({0, 3, 7}),
    "diminished": frozenset({0, 3, 6}),
    "augmented":  frozenset({0, 4, 8}),
    "dominant7":  frozenset({0, 4, 7, 10}),
    "dim7":       frozenset({0, 3, 6, 9}),
    "half-dim7":  frozenset({0, 3, 6, 10}),
    "major7":     frozenset({0, 4, 7, 11}),
    "minor7":     frozenset({0, 3, 7, 10}),
}

_TRIAD_QUALITIES: FrozenSet[str] = frozenset({"major", "minor", "diminished", "augmented"})
_SEVENTH_QUALITIES: FrozenSet[str] = frozenset({"dominant7", "dim7", "half-dim7", "major7", "minor7"})

# Sorted intervals for each quality (used for inversion detection)
_SORTED_INTERVALS: Dict[str, List[int]] = {
    q: sorted(ivs) for q, ivs in _CHORD_TEMPLATES.items()
}

_EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChordClassificationConfig:
    """
    Configuration for ChordClassifier.

    Attributes:
        min_confidence: Windows scoring below this threshold yield None.
                        Compare against best_score / max(sum(profile), ε).
                        Range 0.0–1.0. Default 0.3.
        include_seventh_chords: When False, only triads are tested.
        beat_weight_exponent: Exponent applied to event.beat_strength before
                              weighting. 1.0 = linear (default), 2.0 = quadratic
                              emphasis on metrically strong notes.
    """
    min_confidence: float = 0.3
    include_seventh_chords: bool = True
    beat_weight_exponent: float = 1.0


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ChordClassifier:
    """
    Identifies the chord present in an AnalysisWindow using template matching.

    All methods are stateless — instantiate once and reuse freely.
    """

    def classify_window(
        self,
        window: AnalysisWindow,
        config: ChordClassificationConfig,
    ) -> Optional[Chord]:
        """
        Classify the chord present in a single AnalysisWindow.

        Returns None if the window is empty, rest-only, or below min_confidence.

        Algorithm:
          1. Build beat-strength-weighted pitch-class profile.
          2. Score every candidate (12 roots × active qualities).
          3. Normalise best_score against profile total.
          4. If below threshold, return None.
          5. Determine bass and inversion; return Chord.
        """
        profile = self._build_pitch_class_profile(window, config)
        total = sum(profile)
        if total < _EPSILON:
            return None

        active_qualities = list(_TRIAD_QUALITIES)
        if config.include_seventh_chords:
            active_qualities += list(_SEVENTH_QUALITIES)

        best_score = -1.0
        best_root = 0
        best_quality = "major"

        for root in range(12):
            for quality in active_qualities:
                raw = self._score_chord(profile, root, _CHORD_TEMPLATES[quality])
                n_tones = len(_CHORD_TEMPLATES[quality])
                # Geometric mean of coverage fraction and template completeness:
                #   coverage    = raw / total  (fraction of window notes in chord)
                #   completeness = raw / n_tones (fraction of chord tones present)
                # score = raw / sqrt(total * n_tones)
                # This correctly penalises 7th chords with missing template tones
                # while rewarding 7th chords that explain all 4 notes.
                score = raw / math.sqrt(total * n_tones)
                if score > best_score:
                    best_score = score
                    best_root = root
                    best_quality = quality

        # Normalise to [0, 1]: max possible score is sqrt(total/min_tones) which is
        # ≤ 1.0 when all notes land on chord tones and template is fully covered.
        # For the confidence gate, compare the coverage fraction (raw/total) instead.
        if best_score < config.min_confidence:
            return None

        intervals = _CHORD_TEMPLATES[best_quality]
        bass_pc, inversion = self._get_inversion(window.events, best_root, intervals)

        # Collect pitch classes that are present in the window
        present_pcs = frozenset(
            ev.pitch % 12
            for ev in window.events
            if not ev.is_rest and ev.pitch is not None
        )

        return Chord(
            onset=window.start,
            duration=window.end - window.start,
            root=best_root,
            quality=best_quality,
            pitches=present_pcs,
            bass=bass_pc,
            inversion=inversion,
        )

    def classify_score(
        self,
        score: Score,
        window_type: WindowType,
        config: ChordClassificationConfig,
        window_size: Optional[Fraction] = None,
        hop: Optional[Fraction] = None,
    ) -> List[Chord]:
        """
        Classify chords across an entire score using the given window strategy.

        Filters out None results (windows below confidence threshold or empty).

        Returns:
            List[Chord] in chronological order.
        """
        windows = Windower().create_windows(score, window_type, size=window_size, hop=hop)
        chords = []
        for w in windows:
            chord = self.classify_window(w, config)
            if chord is not None:
                chords.append(chord)
        return chords

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pitch_class_profile(
        self,
        window: AnalysisWindow,
        config: ChordClassificationConfig,
    ) -> List[float]:
        """
        Aggregate beat-strength-weighted pitch-class contributions.

        For each non-rest event:
            weight = beat_strength ** exponent * float(duration)
            profile[pitch % 12] += weight

        Returns:
            List[float] of length 12. All zeros for rest-only windows.
        """
        profile = [0.0] * 12
        exp = config.beat_weight_exponent
        for ev in window.events:
            if ev.is_rest or ev.pitch is None:
                continue
            weight = (ev.beat_strength ** exp) * float(ev.duration)
            profile[ev.pitch % 12] += weight
        return profile

    def _score_chord(
        self,
        profile: List[float],
        root: int,
        intervals: FrozenSet[int],
    ) -> float:
        """
        Raw coverage score for a chord template against the profile.

        Returns the sum of profile weights at each chord tone's pitch class.
        The caller normalises by the total profile weight so that 7th chords
        beat triads when they explain more of the window's note content.
        """
        chord_pcs = [(root + iv) % 12 for iv in intervals]
        return sum(profile[pc] for pc in chord_pcs)

    def _get_inversion(
        self,
        events: List[MusicalEvent],
        root: int,
        intervals: FrozenSet[int],
    ) -> Tuple[int, int]:
        """
        Determine the bass pitch class and inversion number.

        Bass = lowest MIDI pitch among non-rest events.
        Inversion = position of bass_pc in the sorted list of chord tones.

        If bass_pc is not a chord tone, return (bass_pc, 0) as a root-position
        best-guess.

        Returns:
            Tuple[bass_pc, inversion_number]
        """
        pitched = [ev for ev in events if not ev.is_rest and ev.pitch is not None]
        if not pitched:
            return (root, 0)

        lowest_pitch = min(ev.pitch for ev in pitched)
        bass_pc = lowest_pitch % 12

        chord_pcs = sorted([(root + iv) % 12 for iv in intervals])
        if bass_pc in chord_pcs:
            inversion = chord_pcs.index(bass_pc)
            # Normalise: root position means bass == root
            # For a triad [0,4,7] rooted at C (pcs [0,4,7]):
            #   index 0 → root position (inversion 0)
            #   index 1 → first inversion (inversion 1)
            #   index 2 → second inversion (inversion 2)
            # But we need the position of root in chord_pcs to offset correctly.
            root_idx = chord_pcs.index(root)
            inversion = (chord_pcs.index(bass_pc) - root_idx) % len(chord_pcs)
        else:
            inversion = 0

        return (bass_pc, inversion)
