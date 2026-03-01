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
    "load_score",
]
