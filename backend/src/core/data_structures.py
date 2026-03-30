"""
Core data structures for musical representation in the Bach-style generator.

This module provides fundamental classes for representing musical elements with
precise temporal and harmonic information. All temporal values use Fraction for
exact representation, avoiding floating-point errors.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, List, Dict, Any
from enum import Enum


class ProlongationLevel(Enum):
    """Which structural level a note belongs to in Schenkerian analysis."""
    FOREGROUND = "foreground"
    MIDDLEGROUND = "middleground"
    BACKGROUND = "background"


class HarmonicFunction(Enum):
    """Harmonic function labels for chord classification."""
    TONIC = "tonic"
    PREDOMINANT = "predominant"
    DOMINANT = "dominant"
    AMBIGUOUS = "ambiguous"


class Accidental(Enum):
    """Musical accidentals."""
    DOUBLE_FLAT = -2
    FLAT = -1
    NATURAL = 0
    SHARP = 1
    DOUBLE_SHARP = 2


@dataclass(frozen=True)
class TimeSignature:
    """
    Represents a musical time signature.
    
    Attributes:
        numerator: Number of beats per measure
        denominator: Note value that represents one beat (4 = quarter note, 8 = eighth note)
    """
    numerator: int
    denominator: int
    
    def __post_init__(self):
        if self.numerator <= 0:
            raise ValueError(f"Time signature numerator must be positive, got {self.numerator}")
        if self.denominator not in [1, 2, 4, 8, 16, 32]:
            raise ValueError(f"Invalid time signature denominator: {self.denominator}")
    
    @property
    def is_compound(self) -> bool:
        """Returns True if this is a compound meter (numerator divisible by 3, > 3)."""
        return self.numerator > 3 and self.numerator % 3 == 0
    
    @property
    def beats_per_measure(self) -> int:
        """Returns the number of beats per measure."""
        if self.is_compound:
            return self.numerator // 3
        return self.numerator
    
    @property
    def beat_duration(self) -> Fraction:
        """Returns the duration of one beat in quarter notes."""
        if self.is_compound:
            # In 6/8, one beat = dotted quarter = 3/8 = 3/2 quarter notes
            return Fraction(3 * 4, self.denominator)
        return Fraction(4, self.denominator)
    
    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass
class MusicalEvent:
    """
    Represents a single musical event (note or rest) with complete temporal context.
    
    Attributes:
        pitch: MIDI pitch number (0-127), or None for rests
        onset: Start time in quarter notes from beginning of score
        duration: Length in quarter notes
        velocity: MIDI velocity (0-127), representing dynamics
        measure: Measure number (1-indexed)
        beat: Beat position within the measure (0-indexed)
        beat_strength: Metric weight (0.0 = weakest, 1.0 = downbeat)
        voice: Voice number for multi-voice scores (0-indexed)
        is_rest: Whether this event is a rest
    """
    pitch: Optional[int]
    onset: Fraction
    duration: Fraction
    velocity: int = 64
    measure: int = 1
    beat: Fraction = Fraction(0)
    beat_strength: float = 1.0
    voice: int = 0
    is_rest: bool = False
    
    def __post_init__(self):
        """Validate event data."""
        if not self.is_rest and (self.pitch is None or not 0 <= self.pitch <= 127):
            raise ValueError(f"Invalid MIDI pitch: {self.pitch}")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"Invalid velocity: {self.velocity}")
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")
        if not 0.0 <= self.beat_strength <= 1.0:
            raise ValueError(f"Beat strength must be in [0.0, 1.0], got {self.beat_strength}")
        if self.is_rest and self.pitch is not None:
            raise ValueError("Rest cannot have a pitch")
    
    @property
    def offset(self) -> Fraction:
        """End time of the event in quarter notes."""
        return self.onset + self.duration
    
    @property
    def pitch_class(self) -> Optional[int]:
        """Returns pitch class (0-11) or None for rests."""
        return self.pitch % 12 if self.pitch is not None else None
    
    def transpose(self, semitones: int) -> 'MusicalEvent':
        """Return a new event transposed by the given number of semitones."""
        if self.is_rest:
            return self
        new_pitch = max(0, min(127, self.pitch + semitones))
        return MusicalEvent(
            pitch=new_pitch,
            onset=self.onset,
            duration=self.duration,
            velocity=self.velocity,
            measure=self.measure,
            beat=self.beat,
            beat_strength=self.beat_strength,
            voice=self.voice,
            is_rest=False
        )


@dataclass
class Chord:
    """
    Represents a detected chord with harmonic analysis.
    
    Attributes:
        onset: Start time in quarter notes
        duration: Length in quarter notes
        root: Root pitch class (0-11)
        quality: Chord quality (e.g., 'major', 'minor', 'diminished')
        pitches: Set of pitch classes in the chord
        bass: Bass note pitch class (may differ from root for inversions)
        inversion: Inversion number (0 = root position, 1 = first inversion, etc.)
        roman_numeral: Roman numeral analysis in current key
    """
    onset: Fraction
    duration: Fraction
    root: int
    quality: str
    pitches: frozenset[int]
    bass: int
    inversion: int = 0
    roman_numeral: Optional[str] = None
    
    def __post_init__(self):
        """Validate chord data."""
        if not 0 <= self.root <= 11:
            raise ValueError(f"Root must be 0-11, got {self.root}")
        if not 0 <= self.bass <= 11:
            raise ValueError(f"Bass must be 0-11, got {self.bass}")
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")
    
    @property
    def offset(self) -> Fraction:
        """End time of the chord."""
        return self.onset + self.duration


@dataclass
class HarmonicEvent:
    """
    Bundles a Chord with its RomanNumeral for prolongation analysis.

    Provides a unified onset/duration interface so prolongation rules can treat
    a (chord, roman_numeral) pair as a single addressable event.
    """
    chord: 'Chord'
    roman_numeral: 'RomanNumeral'

    @property
    def onset(self) -> Fraction:
        """Start time of the underlying chord."""
        return self.chord.onset

    @property
    def duration(self) -> Fraction:
        """Duration of the underlying chord."""
        return self.chord.duration


_DEGREE_LABELS_MAJOR = {1: 'I', 2: 'ii', 3: 'iii', 4: 'IV', 5: 'V', 6: 'vi', 7: 'viio'}
_DEGREE_LABELS_MINOR = {1: 'i', 2: 'iio', 3: 'III', 4: 'iv', 5: 'V', 6: 'VI', 7: 'viio'}
_QUALITY_SUFFIXES = {
    'major': '', 'minor': '', 'diminished': 'o', 'augmented': '+',
    'dominant7': '7', 'dim7': 'o7', 'half-dim7': 'ø7',
    'major7': 'M7', 'minor7': '7',
}
_INVERSION_FIGURES = {
    0: '', 1: '6', 2: '64',  # triads
}
_INVERSION_FIGURES_7TH = {
    0: '7', 1: '65', 2: '43', 3: '42',
}
_SEVENTH_QUALITY_MARKS = {
    'dominant7': '',   # V7   — '7' already in figure
    'dim7':      'o',  # viio7
    'half-dim7': 'ø',  # viiø7
    'major7':    'M',  # IM7
    'minor7':    '',   # ii7
}
_PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


@dataclass(frozen=True)
class RomanNumeral:
    """
    Represents a Roman numeral analysis of a chord within a key.

    Attributes:
        degree: Scale degree (1-7)
        quality: Chord quality string
        inversion: Inversion number
        local_key_tonic: Pitch class of the local key (0-11)
        local_key_mode: 'major' or 'minor'
        is_secondary: Whether this is a secondary dominant/function (e.g., V/V)
        secondary_target: Scale degree of the secondary target
        function: Harmonic function classification
    """
    degree: int
    quality: str
    inversion: int
    local_key_tonic: int
    local_key_mode: str
    is_secondary: bool = False
    secondary_target: Optional[int] = None
    function: Optional[HarmonicFunction] = None

    @property
    def label(self) -> str:
        """Render standard Roman numeral notation (e.g. 'V7/V', 'viio6')."""
        if self.local_key_mode == 'minor':
            base = _DEGREE_LABELS_MINOR.get(self.degree, str(self.degree))
        else:
            base = _DEGREE_LABELS_MAJOR.get(self.degree, str(self.degree))

        # Override case based on actual quality
        if self.quality in ('major', 'augmented', 'dominant7', 'major7'):
            base = base.upper().rstrip('O')
        elif self.quality in ('minor', 'diminished', 'dim7', 'half-dim7', 'minor7'):
            base = base.lower().rstrip('o')

        is_seventh = '7' in self.quality
        if is_seventh:
            mark = _SEVENTH_QUALITY_MARKS.get(self.quality, '')
            figures = _INVERSION_FIGURES_7TH.get(self.inversion, str(self.inversion))
            result = base + mark + figures
        else:
            suffix = _QUALITY_SUFFIXES.get(self.quality, '')
            figures = _INVERSION_FIGURES.get(self.inversion, str(self.inversion))
            result = base + suffix + figures

        if self.is_secondary and self.secondary_target is not None:
            target_labels = (
                _DEGREE_LABELS_MINOR if self.local_key_mode == 'minor'
                else _DEGREE_LABELS_MAJOR
            )
            target = target_labels.get(self.secondary_target, str(self.secondary_target))
            result += f"/{target}"

        return result


@dataclass(frozen=True)
class KeyEstimate:
    """
    Represents an estimated key at a point in the score.

    Attributes:
        tonic: Pitch class of the tonic (0-11)
        mode: 'major' or 'minor'
        confidence: Confidence score (0.0-1.0)
        onset: Position in the score where this key estimate applies
    """
    tonic: int
    mode: str
    confidence: float
    onset: Fraction

    def __post_init__(self):
        if not 0 <= self.tonic <= 11:
            raise ValueError(f"Tonic must be 0-11, got {self.tonic}")
        if self.mode not in ('major', 'minor'):
            raise ValueError(f"Mode must be 'major' or 'minor', got {self.mode}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")

    @property
    def label(self) -> str:
        """Human-readable key label (e.g. 'G major', 'D minor')."""
        return f"{_PITCH_CLASS_NAMES[self.tonic]} {self.mode}"


@dataclass
class HarmonicSegment:
    """
    Represents a tonal section with consistent key and chord progression.
    
    Attributes:
        start: Start time in quarter notes
        end: End time in quarter notes
        key_tonic: Tonic pitch class of the key (0-11)
        key_mode: Mode ('major' or 'minor')
        chords: List of chords in this segment
    """
    start: Fraction
    end: Fraction
    key_tonic: int
    key_mode: str
    chords: List[Chord] = field(default_factory=list)
    roman_numerals: List[RomanNumeral] = field(default_factory=list)
    prolongation_level: Optional[ProlongationLevel] = None

    def __post_init__(self):
        """Validate segment data."""
        if not 0 <= self.key_tonic <= 11:
            raise ValueError(f"Key tonic must be 0-11, got {self.key_tonic}")
        if self.key_mode not in ['major', 'minor']:
            raise ValueError(f"Key mode must be 'major' or 'minor', got {self.key_mode}")
        if self.end <= self.start:
            raise ValueError(f"End time must be after start time")

    @property
    def duration(self) -> Fraction:
        """Duration of the segment."""
        return self.end - self.start


@dataclass
class Score:
    """
    Complete score representation with all musical events and metadata.
    
    Attributes:
        events: List of all musical events in chronological order
        time_signatures: List of (onset, TimeSignature) tuples for time signature changes
        title: Score title
        composer: Composer name
        tempo: Tempo in BPM (beats per minute)
        metadata: Additional metadata dictionary
    """
    events: List[MusicalEvent]
    time_signatures: List[tuple[Fraction, TimeSignature]] = field(default_factory=list)
    title: Optional[str] = None
    composer: Optional[str] = None
    tempo: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and sort score data."""
        # Sort events by onset time
        self.events.sort(key=lambda e: (e.onset, e.voice))
        
        # Sort time signatures by onset
        self.time_signatures.sort(key=lambda ts: ts[0])
        
        # Ensure at least one time signature
        if not self.time_signatures:
            self.time_signatures.append((Fraction(0), TimeSignature(4, 4)))
    
    def get_time_signature_at(self, onset: Fraction) -> TimeSignature:
        """
        Get the time signature in effect at the given onset time.
        
        Args:
            onset: Time position in quarter notes
            
        Returns:
            TimeSignature in effect at that time
        """
        current_ts = self.time_signatures[0][1]
        for ts_onset, ts in self.time_signatures:
            if ts_onset > onset:
                break
            current_ts = ts
        return current_ts
    
    @property
    def duration(self) -> Fraction:
        """Total duration of the score in quarter notes."""
        if not self.events:
            return Fraction(0)
        return max(event.offset for event in self.events)
    
    @property
    def num_voices(self) -> int:
        """Number of distinct voices in the score."""
        if not self.events:
            return 0
        return max(event.voice for event in self.events) + 1
    
    @property
    def pitch_range(self) -> tuple[Optional[int], Optional[int]]:
        """Returns (lowest_pitch, highest_pitch) or (None, None) if no notes."""
        pitches = [e.pitch for e in self.events if not e.is_rest]
        if not pitches:
            return (None, None)
        return (min(pitches), max(pitches))
    
    def get_events_in_range(self, start: Fraction, end: Fraction) -> List[MusicalEvent]:
        """
        Get all events that overlap with the given time range.
        
        Args:
            start: Start time in quarter notes
            end: End time in quarter notes
            
        Returns:
            List of events that overlap with [start, end)
        """
        return [
            event for event in self.events
            if event.onset < end and event.offset > start
        ]
    
    def get_events_by_voice(self, voice: int) -> List[MusicalEvent]:
        """Get all events for a specific voice."""
        return [event for event in self.events if event.voice == voice]