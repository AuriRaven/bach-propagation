"""
Roman numeral analysis: map (Chord, KeyEstimate) pairs to RomanNumeral objects.

All logic is arithmetic on pitch-class integers plus lookup tables. No music21
required.
"""

from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from src.core.data_structures import (
    Chord,
    HarmonicFunction,
    KeyEstimate,
    RomanNumeral,
)

# ---------------------------------------------------------------------------
# Interval → scale degree lookup tables
# ---------------------------------------------------------------------------

# Semitones from tonic → scale degree (1-based) for each mode
_MAJOR_INTERVAL_TO_DEGREE: Dict[int, int] = {
    0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7,
}

_MINOR_INTERVAL_TO_DEGREE: Dict[int, int] = {
    0: 1,
    2: 2,
    3: 3,   # minor third
    5: 4,
    7: 5,
    8: 6,   # minor sixth (natural/melodic)
    10: 7,  # minor seventh (natural)
    11: 7,  # harmonic leading tone (raised 7)
}

# Semitones from tonic for each diatonic degree (for secondary-dominant detection)
_MAJOR_DEGREE_PCS: Dict[int, int] = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
_MINOR_DEGREE_PCS: Dict[int, int] = {1: 0, 2: 2, 3: 3, 4: 5, 5: 7, 6: 8, 7: 10}

# Expected diatonic quality for each degree+mode
_MAJOR_DIATONIC_QUALITY: Dict[int, str] = {
    1: "major", 2: "minor", 3: "minor",
    4: "major", 5: "major", 6: "minor", 7: "diminished",
}
_MINOR_DIATONIC_QUALITY: Dict[int, str] = {
    1: "minor", 2: "diminished", 3: "major",
    4: "minor", 5: "major",     6: "major",  7: "diminished",
}

# Base harmonic function for each scale degree (mode-independent first pass)
_BASE_FUNCTION: Dict[int, HarmonicFunction] = {
    1: HarmonicFunction.TONIC,
    2: HarmonicFunction.PREDOMINANT,
    3: HarmonicFunction.AMBIGUOUS,
    4: HarmonicFunction.PREDOMINANT,
    5: HarmonicFunction.DOMINANT,
    6: HarmonicFunction.TONIC,
    7: HarmonicFunction.DOMINANT,
}

# Qualities that can act as secondary dominants (V/x or V7/x)
_SECONDARY_DOMINANT_QUALITIES = frozenset({"major", "dominant7"})
# Qualities that act as secondary leading-tone chords (viio/x)
_SECONDARY_LEADINGTONE_QUALITIES = frozenset({"diminished", "dim7", "half-dim7"})


class RomanNumeralAnalyzer:
    """
    Convert (Chord, KeyEstimate) pairs into RomanNumeral objects.

    All methods are stateless — instantiate once and reuse freely.
    """

    def analyze(self, chord: Chord, key: KeyEstimate) -> RomanNumeral:
        """
        Produce a RomanNumeral for one chord in one key.

        Steps:
          1. Compute semitone interval from tonic to chord root (mod 12).
          2. Look up diatonic scale degree. If found, optionally check for
             secondary dominant status.
          3. If chromatic (degree not found):
             a. Check secondary dominant pattern.
             b. Check borrowed chord (modal mixture).
             c. Fall back to degree=1 if unresolvable.
          4. Assign HarmonicFunction.
          5. Return frozen RomanNumeral.

        Returns:
            RomanNumeral (frozen dataclass).
        """
        interval = (chord.root - key.tonic) % 12

        # --- try diatonic ---
        degree = self._interval_to_degree(interval, key.mode)
        is_secondary = False
        secondary_target: Optional[int] = None

        if degree is not None:
            # Only flag secondary if the chord's quality is "wrong" for this
            # diatonic degree — e.g. D major (should be D minor in C major)
            # is a secondary dominant; G major (correctly major as degree 5)
            # is the primary dominant, NOT a secondary dominant.
            expected_q = (
                _MAJOR_DIATONIC_QUALITY if key.mode == "major" else _MINOR_DIATONIC_QUALITY
            ).get(degree)
            quality_is_chromatic = (
                chord.quality != expected_q
                and chord.quality in (_SECONDARY_DOMINANT_QUALITIES | _SECONDARY_LEADINGTONE_QUALITIES)
            )
            if quality_is_chromatic:
                sec, target = self._is_secondary_dominant(chord, key)
                if sec:
                    is_secondary = True
                    secondary_target = target
        else:
            # Chromatic chord — check secondary dominant first
            sec, target = self._is_secondary_dominant(chord, key)
            if sec:
                is_secondary = True
                secondary_target = target
                # Compute the degree from a secondary-dominant perspective:
                # the chord resolves to target, so its Roman numeral is V (or viio)
                # relative to that target. Use degree=5 for dominant qualities,
                # degree=7 for leading-tone qualities.
                if chord.quality in _SECONDARY_DOMINANT_QUALITIES:
                    degree = 5
                else:
                    degree = 7
            elif self._is_borrowed(chord, key):
                # Re-examine in the parallel mode
                parallel_mode = "minor" if key.mode == "major" else "major"
                degree = self._interval_to_degree(interval, parallel_mode)
                if degree is None:
                    degree = 1  # safe fallback
            else:
                degree = 1  # last-resort fallback

        function = self._assign_function(degree)
        if is_secondary:
            function = HarmonicFunction.DOMINANT

        return RomanNumeral(
            degree=degree,
            quality=chord.quality,
            inversion=chord.inversion,
            local_key_tonic=key.tonic,
            local_key_mode=key.mode,
            is_secondary=is_secondary,
            secondary_target=secondary_target,
            function=function,
        )

    def analyze_sequence(
        self,
        chords: List[Chord],
        keys: List[KeyEstimate],
    ) -> List[RomanNumeral]:
        """
        Analyze a list of chords against a list of key estimates.

        For each chord, the active key is the KeyEstimate whose onset is the
        largest value ≤ chord.onset. If no key precedes the chord, the
        earliest key is used.

        Args:
            chords: List of Chord objects in onset order.
            keys:   List of KeyEstimate objects (any order is accepted).

        Returns:
            List[RomanNumeral] in the same order as chords.

        Raises:
            ValueError: If keys is empty.
        """
        if not keys:
            raise ValueError("keys must not be empty")
        if not chords:
            return []

        sorted_keys = sorted(keys, key=lambda k: k.onset)

        def _active_key(chord: Chord) -> KeyEstimate:
            active = sorted_keys[0]
            for k in sorted_keys:
                if k.onset <= chord.onset:
                    active = k
                else:
                    break
            return active

        return [self.analyze(chord, _active_key(chord)) for chord in chords]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interval_to_degree(self, interval: int, mode: str) -> Optional[int]:
        """
        Convert a semitone interval from tonic (0–11) to scale degree (1–7).

        Returns None for intervals that are not diatonic in the given mode.
        """
        table = _MAJOR_INTERVAL_TO_DEGREE if mode == "major" else _MINOR_INTERVAL_TO_DEGREE
        return table.get(interval)

    def _is_secondary_dominant(
        self, chord: Chord, key: KeyEstimate
    ) -> Tuple[bool, Optional[int]]:
        """
        Test whether the chord functions as a secondary dominant or secondary
        leading-tone chord.

        For dominant-quality chords (major/dominant7):
          A chord at root R is V of key T when T = (R + 5) % 12.
          Check if that resolution tonic is the tonic of a diatonic degree.

        For leading-tone quality chords (diminished/dim7/half-dim7):
          Check if (chord.root + 1) % 12 is the tonic of a diatonic degree
          (the chord is one semitone below the target's tonic).

        The primary V/viio of the current key are NOT considered secondary.

        Returns:
            Tuple[is_secondary, target_degree]
        """
        degree_pcs = (
            _MAJOR_DEGREE_PCS if key.mode == "major" else _MINOR_DEGREE_PCS
        )

        if chord.quality in _SECONDARY_DOMINANT_QUALITIES:
            # A chord at root R is V of the key with tonic T when R is the
            # dominant of T, i.e. R = (T + 7) % 12  →  T = (R + 5) % 12.
            # Example: D (2) is V of G (7) because (2 + 5) % 12 = 7.
            resolution_pc = (chord.root + 5) % 12
            for deg, deg_pc in degree_pcs.items():
                target_pc = (key.tonic + deg_pc) % 12
                if resolution_pc == target_pc:
                    # Don't flag chords that resolve to the tonic (that would
                    # make every V into "secondary" — skip degree 1).
                    if deg == 1:
                        continue
                    return (True, deg)

        elif chord.quality in _SECONDARY_LEADINGTONE_QUALITIES:
            # Check viio/x: semitone below a diatonic degree's tonic
            resolution_pc = (chord.root + 1) % 12
            for deg, deg_pc in degree_pcs.items():
                target_pc = (key.tonic + deg_pc) % 12
                if resolution_pc == target_pc:
                    if deg == 1:
                        continue
                    return (True, deg)

        return (False, None)

    def _is_borrowed(self, chord: Chord, key: KeyEstimate) -> bool:
        """
        Test whether the chord is borrowed from the parallel mode (modal mixture).

        A chord is borrowed if:
          1. Its root interval is diatonic in the parallel mode.
          2. Its quality matches the diatonic expectation of the parallel mode
             for that degree.

        Returns:
            True if the chord appears borrowed.
        """
        parallel_mode = "minor" if key.mode == "major" else "major"
        interval = (chord.root - key.tonic) % 12
        parallel_degree = self._interval_to_degree(interval, parallel_mode)
        if parallel_degree is None:
            return False

        expected_table = (
            _MAJOR_DIATONIC_QUALITY if parallel_mode == "major" else _MINOR_DIATONIC_QUALITY
        )
        expected_quality = expected_table.get(parallel_degree)
        return chord.quality == expected_quality

    def _assign_function(self, degree: int) -> HarmonicFunction:
        """
        Assign base harmonic function from scale degree.

        Lookup is mode-independent at this stage; context-sensitive
        refinements are applied later by FunctionLabeler.
        """
        return _BASE_FUNCTION.get(degree, HarmonicFunction.AMBIGUOUS)
