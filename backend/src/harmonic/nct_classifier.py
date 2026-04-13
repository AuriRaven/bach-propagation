"""
Non-chord tone (NCT) classification.

Given a sequence of :class:`MusicalEvent` objects and the chords that are active
during the piece, this module labels each pitched event with its harmonic role
(chord tone vs. specific NCT type).

Two public entry points:

* :meth:`NCTClassifier.classify` — full classification with chord context.
  Produces one :class:`NCTAnnotation` per pitched event.

* :meth:`NCTClassifier.filter_events_for_profile` — chord-agnostic pre-filter
  used by :class:`ChordClassifier`. Removes only events that are unambiguously
  ornamental (weak-beat, short, stepwise neighbours on both sides) so that the
  chord classifier's pitch-class profile is not skewed by passing/neighbour
  tones. Safe to apply before any chord has been identified.

Conventions:
* All temporal values are :class:`fractions.Fraction`.
* Intervals are measured in signed semitones (``next - prev``) — positive means
  rising motion.
* The NCT taxonomy follows standard Baroque voice-leading terminology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence

from src.core.data_structures import Chord, MusicalEvent

# ---------------------------------------------------------------------------
# Chord-template table (duplicated from chord_classifier to avoid a circular
# import — ChordClassifier imports this module). Keep in sync.
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

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class NCTType(Enum):
    """Harmonic role of a single pitched event.

    ``CHORD_TONE`` is a positive classification — the event belongs to the
    active harmony. The remaining values describe specific non-chord tone
    categories following standard Baroque voice-leading terminology.
    ``UNKNOWN`` is reserved for events that fail every pattern match (e.g.
    chromatic tones with no clear resolution).
    """

    CHORD_TONE = "chord_tone"
    PASSING_TONE = "passing_tone"
    NEIGHBOR_TONE = "neighbor_tone"
    SUSPENSION = "suspension"
    ANTICIPATION = "anticipation"
    ESCAPE_TONE = "escape_tone"
    APPOGGIATURA = "appoggiatura"
    PEDAL_POINT = "pedal_point"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NCTAnnotation:
    """The harmonic-role label attached to a single :class:`MusicalEvent`.

    Attributes:
        onset: Onset of the annotated event (quarter notes from score start).
        pitch: MIDI pitch number of the annotated event.
        voice: Voice index of the annotated event.
        nct_type: Harmonic role classification.
        confidence: Classifier confidence in ``[0.0, 1.0]``.
        resolution_pitch: The pitch an NCT resolves to, when applicable
            (e.g. suspension → chord tone one step below). ``None`` when the
            resolution is not defined for this NCT type or cannot be
            determined from local context.
    """

    onset: Fraction
    pitch: int
    voice: int
    nct_type: NCTType
    confidence: float
    resolution_pitch: Optional[int] = None


@dataclass
class NCTClassificationConfig:
    """Tunable parameters for :class:`NCTClassifier`.

    Attributes:
        step_threshold_semitones: Upper bound on ``|interval|`` for an interval
            to count as "stepwise" (chromatic step or whole step). Anything
            larger is treated as a leap.
        weak_beat_threshold: Beat-strength at-or-below which an event is
            considered metrically weak. Used by pattern matchers that require
            weak-beat placement (passing tones, anticipations, etc.).
        strong_beat_threshold: Beat-strength at-or-above which an event is
            considered metrically strong. Used for suspensions and
            appoggiaturas.
        pedal_min_duration: Minimum duration (quarter notes) for a bass note
            to be considered a pedal point candidate.
        pedal_register_max_midi: Highest MIDI pitch that still qualifies as
            "bass register" for pedal-point detection.
        chord_tone_default_confidence: Confidence assigned to events that
            trivially sit inside the active chord template.
    """

    step_threshold_semitones: int = 2
    weak_beat_threshold: float = 0.5
    strong_beat_threshold: float = 0.75
    pedal_min_duration: Fraction = field(default_factory=lambda: Fraction(4))
    pedal_register_max_midi: int = 55  # ~G3 — covers cello/bass register
    chord_tone_default_confidence: float = 0.95


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class NCTClassifier:
    """Classify non-chord tones against a sequence of chords.

    The classifier is stateless — all context is passed in per-call, so a
    single instance may be reused freely across scores and windows.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        events: Sequence[MusicalEvent],
        chords: Sequence[Chord],
        config: Optional[NCTClassificationConfig] = None,
    ) -> List[NCTAnnotation]:
        """Annotate each pitched event with its harmonic role.

        Rests and ``pitch is None`` events are silently skipped — the
        returned list contains one annotation per *pitched* event, in the
        order the events were supplied.

        Args:
            events: The events to classify. Need not be sorted — the method
                sorts a local copy by ``(onset, voice)`` before walking it.
            chords: Chords that are active somewhere in the score. May be
                unsorted or overlap; the classifier looks up the chord whose
                time span contains each event's onset.
            config: Tuning parameters. A default instance is used when
                ``None``.

        Returns:
            List of :class:`NCTAnnotation`, one per pitched event.
        """
        cfg = config or NCTClassificationConfig()
        if not events:
            return []

        sorted_chords: List[Chord] = sorted(chords, key=lambda c: c.onset)

        # Build per-voice adjacency (including rests so that a rest between
        # two pitched notes *breaks* the intervallic chain, which is what
        # strict NCT definitions require).
        prev_map, next_map = self._build_voice_adjacency(events)

        annotations: List[NCTAnnotation] = []
        for idx, ev in enumerate(events):
            if ev.is_rest or ev.pitch is None:
                continue
            prev_ev = prev_map[idx]
            next_ev = next_map[idx]
            # Rests are adjacency-breakers for NCT purposes.
            if prev_ev is not None and prev_ev.is_rest:
                prev_ev = None
            if next_ev is not None and next_ev.is_rest:
                next_ev = None
            chord_here = self._active_chord(ev.onset, sorted_chords)
            chord_next = self._active_chord_after(ev.onset, sorted_chords)
            annotation = self.classify_event(
                event=ev,
                prev_event=prev_ev,
                next_event=next_ev,
                current_chord=chord_here,
                next_chord=chord_next,
                config=cfg,
            )
            annotations.append(annotation)

        return annotations

    def classify_event(
        self,
        event: MusicalEvent,
        prev_event: Optional[MusicalEvent],
        next_event: Optional[MusicalEvent],
        current_chord: Optional[Chord],
        next_chord: Optional[Chord],
        config: Optional[NCTClassificationConfig] = None,
    ) -> NCTAnnotation:
        """Classify a single pitched event against its immediate context.

        This is the core dispatcher used by :meth:`classify`, exposed publicly
        because it is also useful in isolation (e.g. when testing a single
        ornament without building up a full chord list).

        The dispatch order matters — earlier patterns take precedence:

        1. Chord tone check (fast positive classification).
        2. Pedal point (long sustained bass note through chord change).
        3. Suspension (tied/re-struck pitch on a strong beat, steps down).
        4. Anticipation (weak-beat note belonging to the *next* chord).
        5. Passing tone (stepwise motion in one direction).
        6. Neighbor tone (step away, step back to the same pitch).
        7. Escape tone (step away, leap back in the opposite direction).
        8. Appoggiatura (leap in, resolve by step, strong beat).
        9. Fallback: ``UNKNOWN`` with low confidence.

        Args:
            event: The event to classify. Must be pitched (non-rest).
            prev_event: Previous event in the same voice, or ``None``.
            next_event: Next event in the same voice, or ``None``.
            current_chord: Chord active at ``event.onset``, or ``None``.
            next_chord: Chord active immediately after ``event``, or ``None``.
            config: Tuning parameters. A default instance is used when
                ``None``.

        Returns:
            :class:`NCTAnnotation` describing the event's harmonic role.
        """
        cfg = config or NCTClassificationConfig()

        if event.is_rest or event.pitch is None:
            raise ValueError("classify_event requires a pitched event")

        # --- 1. Chord tone ---------------------------------------------------
        if self._is_chord_tone(event.pitch, current_chord):
            return NCTAnnotation(
                onset=event.onset,
                pitch=event.pitch,
                voice=event.voice,
                nct_type=NCTType.CHORD_TONE,
                confidence=cfg.chord_tone_default_confidence,
            )

        # --- 2. Pedal point --------------------------------------------------
        # A long bass note that is a chord tone of at least one chord but
        # spans a chord change.
        if self._is_pedal_candidate(event, current_chord, next_chord, cfg):
            return NCTAnnotation(
                onset=event.onset,
                pitch=event.pitch,
                voice=event.voice,
                nct_type=NCTType.PEDAL_POINT,
                confidence=0.8,
            )

        # Interval context — signed semitones (positive = ascending)
        iv_in = self._signed_interval(prev_event, event)
        iv_out = self._signed_interval(event, next_event)
        step = cfg.step_threshold_semitones

        prev_is_ct = self._is_chord_tone(
            prev_event.pitch if prev_event else None, current_chord
        )
        next_is_ct = self._is_chord_tone(
            next_event.pitch if next_event else None, current_chord
        )

        # --- 3. Suspension ---------------------------------------------------
        # Characteristic pattern: a chord tone of the *previous* harmony is
        # held (or re-struck on the same pitch) into the current chord's
        # strong beat, then resolves down by step to a chord tone.
        if (
            prev_event is not None
            and prev_event.pitch == event.pitch
            and event.beat_strength >= cfg.strong_beat_threshold
            and iv_out is not None
            and -step <= iv_out < 0
            and next_is_ct
        ):
            return NCTAnnotation(
                onset=event.onset,
                pitch=event.pitch,
                voice=event.voice,
                nct_type=NCTType.SUSPENSION,
                confidence=0.9,
                resolution_pitch=next_event.pitch if next_event else None,
            )

        # --- 4. Anticipation -------------------------------------------------
        # A weak-beat note that already belongs to the *next* chord and is
        # immediately re-struck (same pitch) on the next event.
        if (
            next_event is not None
            and next_event.pitch == event.pitch
            and event.beat_strength <= cfg.weak_beat_threshold
            and self._is_chord_tone(event.pitch, next_chord)
        ):
            return NCTAnnotation(
                onset=event.onset,
                pitch=event.pitch,
                voice=event.voice,
                nct_type=NCTType.ANTICIPATION,
                confidence=0.85,
                resolution_pitch=event.pitch,
            )

        # --- 5. Passing tone -------------------------------------------------
        # Stepwise motion in the same direction, flanked by chord tones.
        if (
            iv_in is not None
            and iv_out is not None
            and prev_is_ct
            and next_is_ct
            and 0 < abs(iv_in) <= step
            and 0 < abs(iv_out) <= step
            and (iv_in > 0) == (iv_out > 0)  # same direction
        ):
            return NCTAnnotation(
                onset=event.onset,
                pitch=event.pitch,
                voice=event.voice,
                nct_type=NCTType.PASSING_TONE,
                confidence=0.9,
                resolution_pitch=next_event.pitch if next_event else None,
            )

        # --- 6. Neighbor tone ------------------------------------------------
        # Step away from a chord tone and back to the same pitch.
        if (
            prev_event is not None
            and next_event is not None
            and prev_event.pitch == next_event.pitch
            and prev_is_ct
            and iv_in is not None
            and iv_out is not None
            and 0 < abs(iv_in) <= step
            and 0 < abs(iv_out) <= step
            and (iv_in > 0) != (iv_out > 0)  # opposite direction
        ):
            return NCTAnnotation(
                onset=event.onset,
                pitch=event.pitch,
                voice=event.voice,
                nct_type=NCTType.NEIGHBOR_TONE,
                confidence=0.9,
                resolution_pitch=next_event.pitch,
            )

        # --- 7. Escape tone --------------------------------------------------
        # Step away from a chord tone, then leap back in the opposite
        # direction to another chord tone.
        if (
            iv_in is not None
            and iv_out is not None
            and prev_is_ct
            and next_is_ct
            and 0 < abs(iv_in) <= step
            and abs(iv_out) > step
            and (iv_in > 0) != (iv_out > 0)
        ):
            return NCTAnnotation(
                onset=event.onset,
                pitch=event.pitch,
                voice=event.voice,
                nct_type=NCTType.ESCAPE_TONE,
                confidence=0.75,
                resolution_pitch=next_event.pitch if next_event else None,
            )

        # --- 8. Appoggiatura -------------------------------------------------
        # Leap to a dissonance on a strong beat, resolve by step to a chord
        # tone. Direction of resolution must oppose the incoming leap.
        if (
            iv_in is not None
            and iv_out is not None
            and abs(iv_in) > step
            and 0 < abs(iv_out) <= step
            and event.beat_strength >= cfg.strong_beat_threshold
            and next_is_ct
            and (iv_in > 0) != (iv_out > 0)
        ):
            return NCTAnnotation(
                onset=event.onset,
                pitch=event.pitch,
                voice=event.voice,
                nct_type=NCTType.APPOGGIATURA,
                confidence=0.8,
                resolution_pitch=next_event.pitch if next_event else None,
            )

        # --- 9. Fallback -----------------------------------------------------
        return NCTAnnotation(
            onset=event.onset,
            pitch=event.pitch,
            voice=event.voice,
            nct_type=NCTType.UNKNOWN,
            confidence=0.2,
        )

    def filter_events_for_profile(
        self,
        events: Sequence[MusicalEvent],
        config: Optional[NCTClassificationConfig] = None,
    ) -> List[MusicalEvent]:
        """Heuristically drop obvious NCTs *without* chord context.

        Used by :class:`ChordClassifier` to debias its pitch-class profile
        before any chord has been identified. Because no chord is available
        yet, this method is deliberately conservative — it only filters
        events whose metric position, duration, and immediate intervallic
        context mark them unambiguously as passing or neighbor tones.

        The filter removes an event E iff *all* of:

        * ``E.beat_strength < cfg.weak_beat_threshold``
        * ``|interval(prev, E)| ≤ step_threshold``
        * ``|interval(E, next)| ≤ step_threshold``
        * ``prev`` or ``next`` (in the same voice) has
          ``beat_strength ≥ cfg.strong_beat_threshold``

        Args:
            events: Candidate events (rests are ignored and always kept).
            config: Tuning parameters. Default instance is used when
                ``None``.

        Returns:
            A new list of events with likely NCTs removed. The input order
            is preserved.
        """
        cfg = config or NCTClassificationConfig()
        if len(events) < 3:
            # Need at least a prev + self + next triplet for any filter.
            return list(events)

        prev_map, next_map = self._build_voice_adjacency(events)

        drop: set[int] = set()
        for i, ev in enumerate(events):
            if ev.is_rest or ev.pitch is None:
                continue
            if ev.beat_strength >= cfg.weak_beat_threshold:
                continue
            prev_ev = prev_map[i]
            next_ev = next_map[i]
            # Rests break the chain — a passing tone must have immediate
            # pitched neighbours on both sides.
            if prev_ev is None or next_ev is None:
                continue
            if prev_ev.is_rest or next_ev.is_rest:
                continue
            iv_in = self._signed_interval(prev_ev, ev)
            iv_out = self._signed_interval(ev, next_ev)
            if iv_in is None or iv_out is None:
                continue
            if abs(iv_in) == 0 or abs(iv_in) > cfg.step_threshold_semitones:
                continue
            if abs(iv_out) == 0 or abs(iv_out) > cfg.step_threshold_semitones:
                continue
            if (
                prev_ev.beat_strength < cfg.strong_beat_threshold
                and next_ev.beat_strength < cfg.strong_beat_threshold
            ):
                continue
            drop.add(i)

        return [e for i, e in enumerate(events) if i not in drop]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_voice_adjacency(
        events: Sequence[MusicalEvent],
    ) -> tuple[List[Optional[MusicalEvent]], List[Optional[MusicalEvent]]]:
        """Build per-voice immediate-neighbour lookup tables.

        For each index ``i`` into ``events``, ``prev[i]`` is the event
        immediately preceding ``events[i]`` in the same voice (by onset
        ordering), and ``next[i]`` the one immediately following. Rests are
        *kept* in the chain so callers can detect chain breaks themselves
        — callers that want strict pitched-only adjacency should treat a
        rest neighbour as ``None``.

        Args:
            events: Event list; any order is accepted — the method sorts a
                local view by ``(voice, onset)`` to establish adjacency.

        Returns:
            Tuple ``(prev_map, next_map)`` indexed by the original position
            of each event in ``events``.
        """
        n = len(events)
        prev_map: List[Optional[MusicalEvent]] = [None] * n
        next_map: List[Optional[MusicalEvent]] = [None] * n

        # Group indices by voice, ordered by onset.
        by_voice: dict[int, List[int]] = {}
        for i, ev in enumerate(events):
            by_voice.setdefault(ev.voice, []).append(i)
        for voice, idx_list in by_voice.items():
            idx_list.sort(key=lambda i: (events[i].onset, events[i].duration))
            for pos, i in enumerate(idx_list):
                if pos > 0:
                    prev_map[i] = events[idx_list[pos - 1]]
                if pos + 1 < len(idx_list):
                    next_map[i] = events[idx_list[pos + 1]]

        return prev_map, next_map

    @staticmethod
    def _signed_interval(
        a: Optional[MusicalEvent],
        b: Optional[MusicalEvent],
    ) -> Optional[int]:
        """Signed MIDI-semitone interval from ``a`` to ``b``.

        Returns ``None`` if either side is missing, a rest, or pitch-less.
        """
        if (
            a is None
            or b is None
            or a.is_rest
            or b.is_rest
            or a.pitch is None
            or b.pitch is None
        ):
            return None
        return b.pitch - a.pitch

    @staticmethod
    def _is_chord_tone(pitch: Optional[int], chord: Optional[Chord]) -> bool:
        """True iff ``pitch``'s pitch class is part of the chord's template.

        The template is derived from ``(chord.root, chord.quality)`` so that
        the check reflects what the chord theoretically contains — *not*
        ``chord.pitches``, which records the pitch classes heard in the
        classification window (and therefore also contains any NCTs sounded
        during that window).

        Falls back to ``chord.pitches`` only when the quality is unknown to
        the template table.
        """
        if pitch is None or chord is None:
            return False
        template = _CHORD_TEMPLATES.get(chord.quality)
        if template is None:
            if not chord.pitches:
                return False
            return (pitch % 12) in chord.pitches
        chord_pcs = {(chord.root + iv) % 12 for iv in template}
        return (pitch % 12) in chord_pcs

    @staticmethod
    def _active_chord(onset: Fraction, chords: Sequence[Chord]) -> Optional[Chord]:
        """Return the chord whose ``[onset, offset)`` contains ``onset``.

        Falls back to the latest chord starting at or before ``onset`` when no
        chord strictly contains it (useful when the query sits in a gap).
        """
        active: Optional[Chord] = None
        for c in chords:
            if c.onset > onset:
                break
            if c.onset <= onset < c.offset:
                return c
            active = c
        return active

    @staticmethod
    def _active_chord_after(
        onset: Fraction, chords: Sequence[Chord]
    ) -> Optional[Chord]:
        """Return the first chord whose onset is strictly greater than ``onset``."""
        for c in chords:
            if c.onset > onset:
                return c
        return None

    @staticmethod
    def _is_pedal_candidate(
        event: MusicalEvent,
        current_chord: Optional[Chord],
        next_chord: Optional[Chord],
        config: NCTClassificationConfig,
    ) -> bool:
        """True iff ``event`` looks like a pedal point.

        A pedal point is a long-held bass note that sounds through a change
        of harmony. We approximate that with:

        * Event sits in the bass register
          (``pitch <= pedal_register_max_midi``).
        * Event's duration reaches the minimum pedal length.
        * Event duration is longer than the current chord's duration
          (so it outlasts the chord it started under).
        """
        if event.pitch is None:
            return False
        if event.pitch > config.pedal_register_max_midi:
            return False
        if event.duration < config.pedal_min_duration:
            return False
        if current_chord is None:
            return False
        # Outlasts current chord → bridges into a different chord region.
        if event.offset > current_chord.offset and next_chord is not None:
            return True
        return False


__all__ = [
    "NCTAnnotation",
    "NCTClassificationConfig",
    "NCTClassifier",
    "NCTType",
]
