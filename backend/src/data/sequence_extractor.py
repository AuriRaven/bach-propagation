"""
Sequence extraction: run the full analysis pipeline on a score and emit
a flat list of event dicts ready for tokenisation.

Public API
----------
ExtractionConfig   — parameters controlling the extraction
extract_sequence   — Path → list[dict]
extract_corpus     — directory of MIDI files → list[list[dict]]
augment_sequences  — 12-key transposition augmentation

Each event dict
---------------
{
    'onset':             float,       # quarter notes
    'duration':          float,
    'pitch_classes':     list[int],   # sorted, 0-11
    'chord_root':        int,         # 0-11
    'chord_quality':     str,         # 'major', 'minor', etc.
    'rn_degree':         int,         # 1-7
    'harmonic_function': int,         # 0=T, 1=PD, 2=D
    'local_key_tonic':   int,         # 0-11
    'local_key_mode':    str,         # 'major' or 'minor'
    'metric_weight':     float,
    'inversion':         int,
    # optional (only if config.include_prolongation=True):
    'prolongation_level': int,        # 0=foreground, 1=middleground, 2=background
}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

from src.core.data_structures import HarmonicEvent, HarmonicFunction
from src.core.score_loader import ScoreLoader, ScoreLoadError
from src.harmonic import (
    ChordClassificationConfig,
    ChordClassifier,
    FunctionLabeler,
    KeyEstimator,
    RomanNumeralAnalyzer,
)
from src.temporal.windower import WindowType


# ---------------------------------------------------------------------------
# Function → int mapping
# ---------------------------------------------------------------------------

_FUNCTION_TO_INT: Dict[Optional[HarmonicFunction], int] = {
    HarmonicFunction.TONIC: 0,
    HarmonicFunction.PREDOMINANT: 1,
    HarmonicFunction.DOMINANT: 2,
    HarmonicFunction.AMBIGUOUS: 0,  # collapse AMBIGUOUS → TONIC for 3-class model
    None: 0,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExtractionConfig:
    """Parameters controlling sequence extraction."""
    window_type: WindowType = WindowType.MEASURE
    key_window: Fraction = field(default_factory=lambda: Fraction(16))
    key_hop: Fraction = field(default_factory=lambda: Fraction(8))
    min_chord_confidence: float = 0.25
    include_seventh_chords: bool = True
    beat_weight_exponent: float = 1.0
    include_prolongation: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_sequence(
    path: str | Path,
    config: ExtractionConfig,
) -> List[dict]:
    """
    Run the full analysis pipeline on a single MIDI/MusicXML file.

    Returns an empty list if the file cannot be loaded or yields no chords.
    """
    path = Path(path)
    loader = ScoreLoader(separate_voices=False)
    try:
        score = loader.load(path)
    except ScoreLoadError:
        return []

    return _run_pipeline(score, config)


def extract_corpus(
    corpus_dir: str | Path,
    config: ExtractionConfig,
) -> List[List[dict]]:
    """
    Extract sequences from every MIDI file in *corpus_dir*.

    Returns a list of sequences (one per file). Files that fail are silently
    skipped.
    """
    corpus_dir = Path(corpus_dir)
    sequences: List[List[dict]] = []

    midi_files = sorted(corpus_dir.glob('**/*.mid')) + sorted(corpus_dir.glob('**/*.midi'))
    for midi_path in midi_files:
        seq = extract_sequence(midi_path, config)
        if seq:
            sequences.append(seq)

    return sequences


def augment_sequences(
    sequences: List[List[dict]],
    n_shifts: int = 12,
) -> List[List[dict]]:
    """
    Transpose each sequence by 0, 1, …, n_shifts-1 semitones.

    n_shifts=12 → 12× the input (one copy per key).
    n_shifts=1  → identical copy (shift=0 only).

    Only pitch-related fields are modified:
        chord_root, local_key_tonic, pitch_classes
    All other fields (quality, degree, function, inversion, metric_weight,
    onset, duration) are unchanged.
    """
    augmented: List[List[dict]] = []
    for seq in sequences:
        for shift in range(n_shifts):
            augmented.append(_transpose_sequence(seq, shift))
    return augmented


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_pipeline(score, config: ExtractionConfig) -> List[dict]:
    """Execute the full harmonic analysis pipeline and return event dicts."""
    ke = KeyEstimator()
    global_key = ke.estimate_global(score)
    local_keys = ke.estimate_local(score, config.key_window, config.key_hop)

    cfg = ChordClassificationConfig(
        min_confidence=config.min_chord_confidence,
        include_seventh_chords=config.include_seventh_chords,
        beat_weight_exponent=config.beat_weight_exponent,
    )
    chords = ChordClassifier().classify_score(score, config.window_type, cfg)
    if not chords:
        return []

    key_list = local_keys if local_keys else [global_key]
    rns = RomanNumeralAnalyzer().analyze_sequence(chords, key_list)
    labelled = FunctionLabeler().label_sequence(rns)

    # Optional: prolongation level tagging
    prol_levels: Optional[List[int]] = None
    if config.include_prolongation:
        prol_levels = _extract_prolongation_levels_from_pipeline(
            chords, labelled, score
        )

    events: List[dict] = []
    for i, (chord, rn) in enumerate(zip(chords, labelled)):
        ev: dict = {
            'onset': float(chord.onset),
            'duration': float(chord.duration),
            'pitch_classes': sorted(chord.pitches),
            'chord_root': chord.root,
            'chord_quality': chord.quality,
            'rn_degree': rn.degree,
            'harmonic_function': _FUNCTION_TO_INT[rn.function],
            'local_key_tonic': rn.local_key_tonic,
            'local_key_mode': rn.local_key_mode,
            'metric_weight': _metric_weight_at(score, chord.onset),
            'inversion': chord.inversion,
        }
        if prol_levels is not None:
            ev['prolongation_level'] = prol_levels[i] if i < len(prol_levels) else 0
        events.append(ev)

    return events


def _metric_weight_at(score, onset: Fraction) -> float:
    """Return the metric weight for the given onset position."""
    from src.temporal.meter_analyzer import MeterAnalyzer
    ts = score.get_time_signature_at(onset)
    grid = MeterAnalyzer().build_metric_grid(ts)
    measure_len = Fraction(ts.numerator * 4, ts.denominator)
    beat_pos = onset % measure_len
    return grid.weight_at(beat_pos)


def _extract_prolongation_levels_from_pipeline(
    chords,
    labelled_rns,
    score,
) -> List[int]:
    """
    Build HarmonicEvents from pipeline output, run ProlongationAnalyzer,
    and return a list of structural levels (one per chord index).

    Returns all-zeros on any failure.
    """
    try:
        from src.core.data_structures import HarmonicEvent
        from src.schenkerian import ProlongationAnalyzer

        harmonic_events = [
            HarmonicEvent(chord=chord, roman_numeral=rn)
            for chord, rn in zip(chords, labelled_rns)
        ]
        root = ProlongationAnalyzer().analyze(harmonic_events, score)
        if root is None:
            return [0] * len(chords)

        levels = _collect_levels(root, len(chords), harmonic_events)
        return levels
    except Exception:
        return [0] * len(chords)


def _collect_levels(root, n_events: int, harmonic_events) -> List[int]:
    """
    Walk the ProlongationNode tree and extract the structural level for each
    foreground event (level=0 node). Returns a list indexed by chord position.
    """
    # Map id(HarmonicEvent) → chord index
    id_to_idx = {id(ev): i for i, ev in enumerate(harmonic_events)}
    levels = [0] * n_events

    def _walk(node, inherited_level: int):
        ev_id = id(node.event)
        if ev_id in id_to_idx:
            levels[id_to_idx[ev_id]] = node.level
        for child in node.children:
            _walk(child, node.level)

    _walk(root, root.level)
    return levels


def _transpose_sequence(seq: List[dict], semitones: int) -> List[dict]:
    """
    Return a new sequence with chord_root, local_key_tonic, and pitch_classes
    shifted by *semitones* (mod 12). All other fields are copied unchanged.
    """
    if semitones == 0:
        return [dict(ev) for ev in seq]

    transposed: List[dict] = []
    for ev in seq:
        new_ev = dict(ev)
        new_ev['chord_root'] = (ev['chord_root'] + semitones) % 12
        new_ev['local_key_tonic'] = (ev['local_key_tonic'] + semitones) % 12
        new_ev['pitch_classes'] = sorted(
            (pc + semitones) % 12 for pc in ev['pitch_classes']
        )
        transposed.append(new_ev)
    return transposed
