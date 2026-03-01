"""
Thin wrappers around music21 to isolate the dependency.

All music21-specific logic should live here so that the rest of the codebase
only depends on our own data structures.
"""

from typing import List, Optional, Tuple

from music21 import converter, corpus, note, stream

_PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
_NAME_TO_PC = {
    'C': 0, 'B#': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4,
    'F': 5, 'E#': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11,
}


def load_m21_score(filepath: str) -> stream.Score:
    """Parse any supported music format and return a music21 Score."""
    return converter.parse(filepath)


def get_pitch_classes(s: stream.Stream) -> List[int]:
    """Extract pitch classes (0-11) from all notes in a stream."""
    return [n.pitch.pitchClass for n in s.flatten().notes if isinstance(n, note.Note)]


def get_notes_in_range(
    s: stream.Stream, start: float, end: float
) -> List[note.Note]:
    """Return notes whose onset falls within [start, end) in quarter-note offsets."""
    results = []
    for n in s.flatten().notes:
        if start <= n.offset < end:
            results.append(n)
    return results


def get_key_profile(s: stream.Stream) -> Tuple[int, str, float]:
    """
    Analyse the stream's key using music21's Krumhansl-Schmuckler algorithm.

    Returns:
        (tonic_pitch_class, mode, confidence) where mode is 'major' or 'minor'
        and confidence is the correlation coefficient (0.0-1.0 range, clamped).
    """
    key = s.analyze('key')
    tonic_pc = key.tonic.pitchClass
    mode = key.mode  # 'major' or 'minor'
    confidence = max(0.0, min(1.0, key.correlationCoefficient))
    return tonic_pc, mode, confidence


def pitch_class_to_name(pc: int) -> str:
    """Convert pitch class (0-11) to note name. E.g. 0 → 'C', 7 → 'G'."""
    return _PITCH_CLASS_NAMES[pc % 12]


def name_to_pitch_class(name: str) -> int:
    """Convert note name to pitch class. E.g. 'C' → 0, 'Bb' → 10."""
    pc = _NAME_TO_PC.get(name)
    if pc is None:
        raise ValueError(f"Unknown note name: {name!r}")
    return pc


def interval_between(pc1: int, pc2: int) -> int:
    """Semitone distance between two pitch classes (mod 12, always 0-11)."""
    return (pc2 - pc1) % 12


def load_bach_chorale(bwv: Optional[int] = None) -> stream.Score:
    """
    Load a Bach chorale from the music21 corpus.

    Args:
        bwv: BWV number. If None, loads the first available chorale.

    Returns:
        music21 Score object.
    """
    if bwv is not None:
        results = corpus.search(f'bwv{bwv}')
        if not results:
            raise ValueError(f"No Bach chorale found for BWV {bwv}")
        return results[0].parse()

    results = corpus.search('bach', 'composer')
    if not results:
        raise ValueError("No Bach works found in the music21 corpus")
    return results[0].parse()
