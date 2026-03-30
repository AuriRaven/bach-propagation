"""
Corpus validation: run the harmonic pipeline on Bach chorales from music21's
built-in corpus and report key estimation and chord classification results.

Uses music21's built-in corpus — no local chorale files required.

Usage:
    uv run python scripts/validate_chorales.py

Output:
    BWV    | Key           | Chords | RNs | Sample analysis
    -------|---------------|--------|-----|----------------
    66.6   | E major       |     18 |  18 | I  IV  V  I  ...
    ...
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

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
# Chorale list — all available in music21's built-in corpus
# ---------------------------------------------------------------------------

_CHORALES = [
    '66.6',    # "Nun komm, der Heiden Heiland" — E major
    '253',     # "Ach Gott, vom Himmel sieh darein" — G minor
    '227',     # "Jesu, Joy of Man's Desiring" — G major (BWV 147)
    '232',     # Mass in B minor excerpt
    '255',     # "Befiehl du deine Wege" — A major
    '371',     # "Wie schön leuchtet der Morgenstern" — F# major
    '245.5',   # St. John Passion chorale — D major
    '245.11',  # St. John Passion chorale
    '245.40',  # St. John Passion chorale
    '248.12',  # Christmas Oratorio chorale — A major
]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def analyse_chorale(bwv: str) -> dict:
    """Run the full pipeline on a chorale and return a summary dict."""
    loader = ScoreLoader(separate_voices=True)
    score = loader.load_corpus(bwv)

    ke = KeyEstimator()
    global_key = ke.estimate_global(score)
    local_keys = ke.estimate_local(score, Fraction(16), Fraction(8))

    cfg = ChordClassificationConfig(
        min_confidence=0.25,
        include_seventh_chords=True,
        beat_weight_exponent=1.0,
    )
    chords = ChordClassifier().classify_score(score, WindowType.MEASURE, cfg)

    key_list = local_keys if local_keys else [global_key]
    rns = RomanNumeralAnalyzer().analyze_sequence(chords, key_list)
    labelled = FunctionLabeler().label_sequence(rns)

    return {
        'bwv':        bwv,
        'key':        global_key.label,
        'confidence': global_key.confidence,
        'n_chords':   len(chords),
        'n_rns':      len(labelled),
        'rn_sample':  ' '.join(rn.label for rn in labelled[:8]),
        'functions':  ' '.join(
            rn.function.name[0] if rn.function else '?' for rn in labelled[:8]
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    header = (
        f"{'BWV':>8} | {'Key':>14} | {'Conf':>5} | "
        f"{'Chords':>6} | {'RNs':>4} | Sample RNs (first 8)"
    )
    separator = "-" * len(header)
    print(header)
    print(separator)

    passed = 0
    failed = 0

    for bwv in _CHORALES:
        try:
            r = analyse_chorale(bwv)
            print(
                f"{r['bwv']:>8} | {r['key']:>14} | {r['confidence']:>5.3f} | "
                f"{r['n_chords']:>6} | {r['n_rns']:>4} | {r['rn_sample']}"
            )
            passed += 1
        except ScoreLoadError as e:
            print(f"{'BWV ' + bwv:>8} | {'SKIPPED':>14} | {'—':>5} | "
                  f"{'—':>6} | {'—':>4} | {e}")
            failed += 1
        except Exception as e:
            print(f"{'BWV ' + bwv:>8} | {'ERROR':>14} | {'—':>5} | "
                  f"{'—':>6} | {'—':>4} | {e}")
            failed += 1

    print(separator)
    print(f"Processed {passed} chorales successfully, {failed} failed/skipped.")


if __name__ == '__main__':
    main()
