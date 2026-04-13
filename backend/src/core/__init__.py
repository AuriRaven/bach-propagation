"""Core data structures and score loading for bach-propagation."""

from .data_structures import (
    Accidental,
    Chord,
    HarmonicEvent,
    HarmonicFunction,
    HarmonicSegment,
    KeyEstimate,
    MusicalEvent,
    ProlongationLevel,
    RomanNumeral,
    Score,
    TimeSignature,
)
from .score_loader import ScoreLoader, ScoreLoadError, load_score
from .voice_separator import (
    VoiceProfile,
    VoiceSeparationConfig,
    VoiceSeparationResult,
    VoiceSeparator,
)

__all__ = [
    "Accidental",
    "Chord",
    "HarmonicEvent",
    "HarmonicFunction",
    "HarmonicSegment",
    "KeyEstimate",
    "MusicalEvent",
    "ProlongationLevel",
    "RomanNumeral",
    "Score",
    "ScoreLoadError",
    "ScoreLoader",
    "TimeSignature",
    "VoiceProfile",
    "VoiceSeparationConfig",
    "VoiceSeparationResult",
    "VoiceSeparator",
    "load_score",
]
