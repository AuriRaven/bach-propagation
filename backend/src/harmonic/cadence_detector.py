"""
Cadence detection over Roman-numeral analysis.

Walks consecutive :class:`HarmonicEvent` pairs and classifies each pair
against a fixed priority list of Baroque cadence patterns:

    1. Perfect Authentic Cadence (PAC)   — V(7) → I, both root position
    2. Imperfect Authentic Cadence (IAC) — V(7) → I with any inversion
    3. Deceptive Cadence (DC)            — V → vi (or VI in minor)
    4. Phrygian Half Cadence (PHC)       — iv⁶ → V (minor only)
    5. Half Cadence (HC)                 — anything (≠ V) → V
    6. Plagal Cadence (PC)               — IV → I

Phrygian half cadence is tried before the generic half cadence because
it is the more specific pattern — iv⁶ → V in minor is stylistically
distinctive and should not be flattened into a generic HC label.

The first pattern that matches a pair is returned, so stronger cadences
take precedence over weaker ones. Secondary dominants are rejected —
``V/V → V`` is tonicization, not a cadence in the outer key.

The detector is stateless: one instance may be reused across scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import List, Optional, Sequence

from src.core.data_structures import (
    Chord,
    HarmonicEvent,
    HarmonicFunction,
    RomanNumeral,
)

# Qualities that make a degree-5 chord act as a dominant.
_DOMINANT_QUALITIES: frozenset[str] = frozenset({"major", "dominant7"})
# Qualities that make a degree-4 chord act as a subdominant (Plagal).
_PLAGAL_QUALITIES: frozenset[str] = frozenset({"major", "minor"})


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class CadenceType(Enum):
    """Standard Baroque cadence categories, ranked by priority."""

    AUTHENTIC_PERFECT = "authentic_perfect"
    AUTHENTIC_IMPERFECT = "authentic_imperfect"
    HALF = "half"
    DECEPTIVE = "deceptive"
    PHRYGIAN_HALF = "phrygian_half"
    PLAGAL = "plagal"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cadence:
    """A detected cadence between two :class:`HarmonicEvent` objects.

    Attributes:
        onset: Onset of the *final* chord — the moment of arrival.
        cadence_type: The classification.
        penultimate: Harmonic event acting as the cadential approach.
        final: Harmonic event acting as the cadential arrival.
        confidence: Detector confidence in ``[0.0, 1.0]``.
    """

    onset: Fraction
    cadence_type: CadenceType
    penultimate: HarmonicEvent
    final: HarmonicEvent
    confidence: float

    @property
    def label(self) -> str:
        """Pretty progression label, e.g. ``"V7 → I"``."""
        return (
            f"{self.penultimate.roman_numeral.label} "
            f"→ {self.final.roman_numeral.label}"
        )


@dataclass
class CadenceDetectionConfig:
    """Tunable parameters for :class:`CadenceDetector`.

    Attributes:
        min_confidence: Cadences whose confidence falls below this threshold
            are dropped from the returned list.
        include_plagal: When False, plagal cadences (IV → I) are not
            reported — Bach writes very few in instrumental works, and
            callers targeting chorale authenticity often want them off.
        require_same_key: When True, a pair is only considered cadential
            if penultimate and final agree on ``local_key_tonic`` and
            ``local_key_mode``. Prevents a modulation boundary from being
            mislabelled as a cadence.
        downweight_weak_final: When True, multiplies confidence by the
            final chord's duration normalised against the penultimate's,
            so that a blink-of-an-eye "arrival" is penalised.
    """

    min_confidence: float = 0.5
    include_plagal: bool = True
    require_same_key: bool = True
    downweight_weak_final: bool = True


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CadenceDetector:
    """Detects Baroque cadences in a harmonic-event sequence.

    Usage:
        >>> detector = CadenceDetector()
        >>> cadences = detector.detect(harmonic_events)

    The detector does not mutate its inputs. All methods are stateless.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        harmonic_events: Sequence[HarmonicEvent],
        config: Optional[CadenceDetectionConfig] = None,
    ) -> List[Cadence]:
        """Walk the sequence and return every detected cadence.

        Args:
            harmonic_events: Chord + Roman-numeral pairs in chronological
                order. Pairs whose ``roman_numeral.is_secondary`` is True
                are rejected as cadential approaches (tonicization is not
                a cadence).
            config: Tuning parameters. Default instance is used when
                ``None``.

        Returns:
            List of :class:`Cadence`, ordered by the final chord's onset.
        """
        cfg = config or CadenceDetectionConfig()
        results: List[Cadence] = []
        for i in range(len(harmonic_events) - 1):
            pen = harmonic_events[i]
            fin = harmonic_events[i + 1]
            cadence = self.classify_pair(pen, fin, cfg)
            if cadence is None:
                continue
            if cadence.confidence < cfg.min_confidence:
                continue
            if cadence.cadence_type is CadenceType.PLAGAL and not cfg.include_plagal:
                continue
            results.append(cadence)
        return results

    def classify_pair(
        self,
        penultimate: HarmonicEvent,
        final: HarmonicEvent,
        config: Optional[CadenceDetectionConfig] = None,
    ) -> Optional[Cadence]:
        """Classify a single cadential pair.

        Tries patterns in priority order and returns the first match
        (``None`` if no pattern fires). Exposed separately from
        :meth:`detect` so callers can probe any pair — e.g. at a phrase
        boundary they suspect — without walking a full sequence.

        The dispatch order is:

        1. :meth:`_try_authentic` — handles PAC and IAC together.
        2. :meth:`_try_deceptive`
        3. :meth:`_try_phrygian_half` — more specific than plain HC.
        4. :meth:`_try_half`
        5. :meth:`_try_plagal`
        """
        cfg = config or CadenceDetectionConfig()

        # Secondary dominants are tonicizations, not cadences in the
        # outer key — short-circuit before pattern matching.
        if penultimate.roman_numeral.is_secondary:
            return None

        if cfg.require_same_key and not self._same_key(penultimate, final):
            return None

        for try_fn in (
            self._try_authentic,
            self._try_deceptive,
            self._try_phrygian_half,
            self._try_half,
            self._try_plagal,
        ):
            cadence = try_fn(penultimate, final, cfg)
            if cadence is not None:
                return cadence
        return None

    # ------------------------------------------------------------------
    # Pattern matchers
    # ------------------------------------------------------------------

    def _try_authentic(
        self,
        pen: HarmonicEvent,
        fin: HarmonicEvent,
        cfg: CadenceDetectionConfig,
    ) -> Optional[Cadence]:
        """PAC / IAC: V(7) → I with the PAC variant requiring root position.

        PAC is the stronger variant — the dispatcher tries it first and
        only falls through to IAC when the inversion check fails.
        """
        pen_rn = pen.roman_numeral
        fin_rn = fin.roman_numeral
        if pen_rn.degree != 5:
            return None
        if fin_rn.degree != 1:
            return None
        if pen_rn.quality not in _DOMINANT_QUALITIES:
            return None

        both_root_position = (pen.chord.inversion == 0 and fin.chord.inversion == 0)
        if both_root_position:
            confidence = 0.95
            ctype = CadenceType.AUTHENTIC_PERFECT
        else:
            confidence = 0.8
            ctype = CadenceType.AUTHENTIC_IMPERFECT

        # 7th-chord approach is slightly more decisive than a bare triad.
        if pen_rn.quality == "dominant7":
            confidence = min(1.0, confidence + 0.03)

        confidence = self._apply_final_weight(confidence, pen, fin, cfg)
        return Cadence(
            onset=fin.chord.onset,
            cadence_type=ctype,
            penultimate=pen,
            final=fin,
            confidence=confidence,
        )

    def _try_half(
        self,
        pen: HarmonicEvent,
        fin: HarmonicEvent,
        cfg: CadenceDetectionConfig,
    ) -> Optional[Cadence]:
        """HC: any non-dominant approach → V.

        The pair must land on a dominant-functioning chord on degree 5,
        and the approach chord must not itself be a dominant (V → V is
        not a cadence).
        """
        pen_rn = pen.roman_numeral
        fin_rn = fin.roman_numeral
        if fin_rn.degree != 5:
            return None
        if fin_rn.quality not in _DOMINANT_QUALITIES:
            return None
        if pen_rn.degree == 5 and pen_rn.quality in _DOMINANT_QUALITIES:
            # V → V is not a cadence.
            return None

        confidence = 0.8
        # Conventional approach chords strengthen the reading.
        if pen_rn.degree in (1, 2, 4, 6):
            confidence = 0.85
        confidence = self._apply_final_weight(confidence, pen, fin, cfg)
        return Cadence(
            onset=fin.chord.onset,
            cadence_type=CadenceType.HALF,
            penultimate=pen,
            final=fin,
            confidence=confidence,
        )

    def _try_deceptive(
        self,
        pen: HarmonicEvent,
        fin: HarmonicEvent,
        cfg: CadenceDetectionConfig,
    ) -> Optional[Cadence]:
        """DC: V(7) → vi (major) or V(7) → VI (minor).

        We only check the degree of the arrival — the quality of the
        submediant varies with mode.
        """
        pen_rn = pen.roman_numeral
        fin_rn = fin.roman_numeral
        if pen_rn.degree != 5:
            return None
        if pen_rn.quality not in _DOMINANT_QUALITIES:
            return None
        if fin_rn.degree != 6:
            return None

        confidence = 0.85
        confidence = self._apply_final_weight(confidence, pen, fin, cfg)
        return Cadence(
            onset=fin.chord.onset,
            cadence_type=CadenceType.DECEPTIVE,
            penultimate=pen,
            final=fin,
            confidence=confidence,
        )

    def _try_phrygian_half(
        self,
        pen: HarmonicEvent,
        fin: HarmonicEvent,
        cfg: CadenceDetectionConfig,
    ) -> Optional[Cadence]:
        """PHC: iv⁶ → V in a minor key.

        Distinctive Baroque pattern — the bass descends by semitone from
        ♭6 to 5. We check the mode via the Roman numeral's local key and
        require first-inversion iv.
        """
        pen_rn = pen.roman_numeral
        fin_rn = fin.roman_numeral
        if pen_rn.local_key_mode != "minor":
            return None
        if pen_rn.degree != 4 or fin_rn.degree != 5:
            return None
        if pen.chord.inversion != 1:
            return None
        if fin_rn.quality not in _DOMINANT_QUALITIES:
            return None

        confidence = 0.95
        confidence = self._apply_final_weight(confidence, pen, fin, cfg)
        return Cadence(
            onset=fin.chord.onset,
            cadence_type=CadenceType.PHRYGIAN_HALF,
            penultimate=pen,
            final=fin,
            confidence=confidence,
        )

    def _try_plagal(
        self,
        pen: HarmonicEvent,
        fin: HarmonicEvent,
        cfg: CadenceDetectionConfig,
    ) -> Optional[Cadence]:
        """PC: IV → I (or iv → i in minor)."""
        pen_rn = pen.roman_numeral
        fin_rn = fin.roman_numeral
        if pen_rn.degree != 4 or fin_rn.degree != 1:
            return None
        if pen_rn.quality not in _PLAGAL_QUALITIES:
            return None

        confidence = 0.75
        confidence = self._apply_final_weight(confidence, pen, fin, cfg)
        return Cadence(
            onset=fin.chord.onset,
            cadence_type=CadenceType.PLAGAL,
            penultimate=pen,
            final=fin,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _same_key(a: HarmonicEvent, b: HarmonicEvent) -> bool:
        """True iff both events share their Roman-numeral local key."""
        ra = a.roman_numeral
        rb = b.roman_numeral
        return (
            ra.local_key_tonic == rb.local_key_tonic
            and ra.local_key_mode == rb.local_key_mode
        )

    @staticmethod
    def _apply_final_weight(
        base_confidence: float,
        pen: HarmonicEvent,
        fin: HarmonicEvent,
        cfg: CadenceDetectionConfig,
    ) -> float:
        """Optionally trim confidence when the arrival chord is very short.

        When ``cfg.downweight_weak_final`` is True, we scale the base
        confidence by ``min(1, fin.duration / pen.duration)`` so that a
        cadence whose "arrival" is a fleeting passing chord is not
        rewarded as heavily as one where the final chord settles.
        Clamped to ``[0.0, 1.0]``.
        """
        if not cfg.downweight_weak_final:
            return max(0.0, min(1.0, base_confidence))
        pen_dur = float(pen.duration) or 1.0
        fin_dur = float(fin.duration)
        ratio = min(1.0, fin_dur / pen_dur) if pen_dur > 0 else 1.0
        return max(0.0, min(1.0, base_confidence * (0.5 + 0.5 * ratio)))


__all__ = [
    "Cadence",
    "CadenceDetectionConfig",
    "CadenceDetector",
    "CadenceType",
]
