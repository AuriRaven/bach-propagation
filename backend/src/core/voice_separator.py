"""
Voice separation from single melodic lines.

Bach's solo-string works (cello suites, violin sonatas/partitas) are notated
as single melodic lines but *imply* polyphony: arpeggiated chords, pedal
tones, and leaping bass/soprano dialogue are standard devices. Downstream
harmonic analysis benefits from making those implicit voices explicit —
Roman-numeral analysis, cadence detection, and prolongation all assume
voice-separated input.

This module implements a proximity-weighted greedy voice assignment
algorithm:

1. Walk events in onset order.
2. For each event, compute a cost against every active voice based on:
   * pitch proximity to the voice's most recent note,
   * temporal gap since that note,
   * register drift from the voice's running mean pitch.
3. Voices with a note still sounding at the event's onset are rejected
   (a voice may carry only one note at a time).
4. If every active voice is too far away *and* we have capacity under
   ``max_voices``, spawn a new voice.
5. Otherwise assign the event to the lowest-cost voice.

The classifier is conservative about spawning voices — it prefers to keep
a stepwise melodic line together and only splits when the register leap
is large enough to be implausible as continuous melody.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from src.core.data_structures import MusicalEvent, Score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class VoiceSeparationConfig:
    """Tunable parameters for :class:`VoiceSeparator`.

    Attributes:
        max_voices: Upper bound on the number of voices the separator will
            spawn. Solo cello suites typically need 2–3.
        pitch_proximity_weight: Cost per semitone between the incoming event
            and the voice's most recent pitch. Dominant term — a 12-semitone
            (octave) leap adds 12 units of cost.
        temporal_gap_weight: Cost per quarter note of silence between the
            voice's previous note offset and the incoming event's onset.
            Encourages new voices for events appearing after long gaps.
        register_bias_weight: Cost per semitone of deviation from the voice's
            running mean pitch. Small, meant only as a tiebreaker between
            otherwise equivalent voices.
        new_voice_threshold: If the lowest assignment cost exceeds this value
            *and* ``max_voices`` has not been reached, the event is assigned
            to a newly-spawned voice instead.
        min_voice_separation_semitones: When all candidate voices are within
            this register width, the separator treats them as a single
            register band and prefers to reuse the closest voice rather
            than spawn.
    """

    max_voices: int = 3
    pitch_proximity_weight: float = 1.0
    temporal_gap_weight: float = 1.0
    register_bias_weight: float = 0.25
    new_voice_threshold: float = 10.0
    min_voice_separation_semitones: int = 7


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceProfile:
    """Summary statistics for a single reconstructed voice.

    Attributes:
        voice: Voice index (0-based).
        mean_pitch: Duration-weighted mean MIDI pitch.
        min_pitch: Lowest MIDI pitch in the voice.
        max_pitch: Highest MIDI pitch in the voice.
        note_count: Number of (non-rest) notes assigned to the voice.
    """

    voice: int
    mean_pitch: float
    min_pitch: int
    max_pitch: int
    note_count: int


@dataclass
class VoiceSeparationResult:
    """Output of :meth:`VoiceSeparator.separate`.

    Attributes:
        events: New :class:`MusicalEvent` list with ``voice`` fields
            re-assigned. Rests carry through with their original voice
            unchanged. Length and ordering match the input events.
        voice_profiles: Per-voice summary keyed by voice index.
    """

    events: List[MusicalEvent]
    voice_profiles: Dict[int, VoiceProfile] = field(default_factory=dict)

    @property
    def num_voices(self) -> int:
        """Number of voices the separator actually used."""
        return len(self.voice_profiles)


# ---------------------------------------------------------------------------
# Separator
# ---------------------------------------------------------------------------


class VoiceSeparator:
    """Assigns events to implied voices using proximity + register heuristics.

    The separator is stateless — a single instance may be reused across
    scores. Both :meth:`separate` and :meth:`separate_events` return a
    :class:`VoiceSeparationResult` so callers can inspect the per-voice
    register bands alongside the re-voiced events.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def separate(
        self,
        score: Score,
        config: Optional[VoiceSeparationConfig] = None,
    ) -> VoiceSeparationResult:
        """Separate the score's events into implied voices.

        Args:
            score: Source score. Typically single-voice (e.g. a cello suite
                loaded with ``separate_voices=False``); multi-voice scores
                are re-analysed from scratch so the output voice count
                reflects *implied* polyphony rather than the input layout.
            config: Tuning parameters; defaults are used when ``None``.

        Returns:
            :class:`VoiceSeparationResult` with new events and per-voice
            statistics.
        """
        return self.separate_events(score.events, config)

    def separate_events(
        self,
        events: Sequence[MusicalEvent],
        config: Optional[VoiceSeparationConfig] = None,
    ) -> VoiceSeparationResult:
        """Separate a raw event list into implied voices.

        Same contract as :meth:`separate` but takes events directly so the
        method can be used without wrapping them in a Score.

        Args:
            events: Events in any order. A local copy is sorted by
                ``(onset, -pitch)`` before assignment so that, when a
                chord is notated as stacked simultaneous notes, the highest
                pitch is processed first and tends to land in voice 0.
            config: Tuning parameters; defaults are used when ``None``.

        Returns:
            :class:`VoiceSeparationResult` with one entry per input event
            (in the original order).
        """
        cfg = config or VoiceSeparationConfig()

        # Remember original order so we can return the result in input order.
        indexed: List[Tuple[int, MusicalEvent]] = list(enumerate(events))
        # Process in onset order, high-pitch-first for simultaneities — this
        # encourages the top line to claim voice 0 consistently.
        indexed.sort(
            key=lambda it: (
                it[1].onset,
                -(it[1].pitch if it[1].pitch is not None else -1),
            )
        )

        voice_state: List[_VoiceState] = []
        assignments: Dict[int, int] = {}

        for orig_idx, ev in indexed:
            if ev.is_rest or ev.pitch is None:
                # Rests keep their original voice assignment — they are not
                # melodically tracked.
                assignments[orig_idx] = ev.voice
                continue

            chosen = self._choose_voice(ev, voice_state, cfg)
            if chosen is None:
                chosen = len(voice_state)
                voice_state.append(_VoiceState())
            assignments[orig_idx] = chosen
            voice_state[chosen].record(ev)

        # Rebuild events in the original input order with updated voice.
        new_events: List[MusicalEvent] = []
        for i, ev in enumerate(events):
            v = assignments.get(i, ev.voice)
            if v == ev.voice:
                new_events.append(ev)
            else:
                new_events.append(replace(ev, voice=v))

        profiles = {
            idx: st.finalize(idx) for idx, st in enumerate(voice_state)
        }
        return VoiceSeparationResult(events=new_events, voice_profiles=profiles)

    def to_score(
        self,
        score: Score,
        result: VoiceSeparationResult,
    ) -> Score:
        """Return a new :class:`Score` with the separator's events applied.

        Time signatures, metadata, and title are carried over from the
        source score. Useful when downstream code wants a drop-in
        replacement that looks like any other :class:`Score`.
        """
        return Score(
            events=list(result.events),
            time_signatures=list(score.time_signatures),
            title=score.title,
            composer=score.composer,
            tempo=score.tempo,
            metadata=dict(score.metadata),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _choose_voice(
        self,
        event: MusicalEvent,
        voice_state: List["_VoiceState"],
        config: VoiceSeparationConfig,
    ) -> Optional[int]:
        """Return the best existing voice for ``event`` or ``None`` to spawn.

        A voice is ineligible if it already has a note sounding at
        ``event.onset``. Among eligible voices we pick the lowest cost; if
        that cost exceeds ``new_voice_threshold`` *and* we still have
        capacity under ``max_voices``, we return ``None`` to signal
        "spawn a new voice".
        """
        best_voice: Optional[int] = None
        best_cost = float("inf")

        for idx, state in enumerate(voice_state):
            if state.is_sounding_at(event.onset):
                continue
            cost = self._assignment_cost(event, state, config)
            if cost < best_cost:
                best_cost = cost
                best_voice = idx

        # Spawn a new voice when every existing voice is far away and we have
        # capacity. If capacity is exhausted, fall back to the closest voice
        # even if its cost is high.
        can_spawn = len(voice_state) < config.max_voices
        if can_spawn and (best_voice is None or best_cost > config.new_voice_threshold):
            return None
        return best_voice

    @staticmethod
    def _assignment_cost(
        event: MusicalEvent,
        state: "_VoiceState",
        config: VoiceSeparationConfig,
    ) -> float:
        """Cost of assigning ``event`` to the voice described by ``state``.

        Sum of three terms:
          * ``pitch_proximity_weight * |event.pitch - last_pitch|``
          * ``temporal_gap_weight    * max(0, onset - last_offset)``
          * ``register_bias_weight   * |event.pitch - voice_mean_pitch|``
        """
        pitch = event.pitch
        assert pitch is not None  # non-rest events only reach here
        pitch_cost = abs(pitch - state.last_pitch) * config.pitch_proximity_weight
        gap = max(Fraction(0), event.onset - state.last_offset)
        time_cost = float(gap) * config.temporal_gap_weight
        reg_cost = abs(pitch - state.mean_pitch) * config.register_bias_weight
        return pitch_cost + time_cost + reg_cost


# ---------------------------------------------------------------------------
# Per-voice running state (private)
# ---------------------------------------------------------------------------


@dataclass
class _VoiceState:
    """Running-state tracker for one voice during separation."""

    last_pitch: int = 0
    last_offset: Fraction = field(default_factory=lambda: Fraction(0))
    last_onset: Fraction = field(default_factory=lambda: Fraction(0))
    sum_pitch_weighted: float = 0.0
    sum_weight: float = 0.0
    min_pitch: int = 127
    max_pitch: int = 0
    note_count: int = 0
    _initialised: bool = False

    @property
    def mean_pitch(self) -> float:
        """Duration-weighted mean pitch; returns ``last_pitch`` before any
        note has been recorded."""
        if self.sum_weight <= 0:
            return float(self.last_pitch)
        return self.sum_pitch_weighted / self.sum_weight

    def is_sounding_at(self, onset: Fraction) -> bool:
        """True iff this voice still has a note sounding at ``onset``.

        Always False before the first note is recorded.
        """
        if not self._initialised:
            return False
        # Strict inequality: a note that ends exactly at ``onset`` is no
        # longer sounding, so the voice is free for the new event.
        return self.last_onset <= onset < self.last_offset

    def record(self, event: MusicalEvent) -> None:
        """Fold an event into the running state."""
        assert event.pitch is not None
        weight = max(float(event.duration), 1e-6)
        self.sum_pitch_weighted += event.pitch * weight
        self.sum_weight += weight
        self.last_pitch = event.pitch
        self.last_onset = event.onset
        self.last_offset = event.offset
        self.min_pitch = min(self.min_pitch, event.pitch)
        self.max_pitch = max(self.max_pitch, event.pitch)
        self.note_count += 1
        self._initialised = True

    def finalize(self, voice_index: int) -> VoiceProfile:
        """Render the immutable summary for this voice."""
        return VoiceProfile(
            voice=voice_index,
            mean_pitch=self.mean_pitch,
            min_pitch=self.min_pitch if self._initialised else 0,
            max_pitch=self.max_pitch if self._initialised else 0,
            note_count=self.note_count,
        )


__all__ = [
    "VoiceProfile",
    "VoiceSeparationConfig",
    "VoiceSeparationResult",
    "VoiceSeparator",
]
