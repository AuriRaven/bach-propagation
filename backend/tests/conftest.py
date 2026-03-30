"""
Shared pytest fixtures and helpers for the bach-propagation test suite.

Paths
-----
PRELUDE_PATH  – BWV 1007 Suite No.1 Prelude (cello, monophonic, 3/4)
ADAGIO_PATH   – BWV 1001 Sonata No.1 Adagio  (violin, chordal passages)

Fixtures
--------
prelude_score   scope=module – Score loaded with separate_voices=False
adagio_score    scope=module – Score loaded with separate_voices=True

Helper factories (importable by test modules)
--------------------------------------------
make_note(pitch, onset, duration, ...)
make_rest(onset, duration, ...)
make_score(events, numerator=4, denominator=4)
"""

from fractions import Fraction
from pathlib import Path

import pytest

from src.core.data_structures import MusicalEvent, Score, TimeSignature
from src.core.score_loader import ScoreLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_SOLO = _PROJECT_ROOT / "data" / "samples" / "01_bach" / "solo_instruments"

PRELUDE_PATH = _SOLO / "cello" / "cello_suites" / "bwv1007_suite_no_1_in_g_major" / "01_prelude.mid"
ADAGIO_PATH  = _SOLO / "violin" / "sonatas" / "bwv1001_sonata_no_1_in_g_minor" / "01_adagio.mid"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prelude_score():
    """BWV 1007 Prelude loaded flat (separate_voices=False)."""
    if not PRELUDE_PATH.exists():
        pytest.skip("Sample MIDI files not found — place them under data/samples/")
    return ScoreLoader(separate_voices=False).load(PRELUDE_PATH)


@pytest.fixture(scope="module")
def adagio_score():
    """BWV 1001 Adagio loaded with voice separation enabled."""
    if not ADAGIO_PATH.exists():
        pytest.skip("Sample MIDI files not found — place them under data/samples/")
    return ScoreLoader(separate_voices=True).load(ADAGIO_PATH)


# ---------------------------------------------------------------------------
# Small factory helpers (used directly in test files)
# ---------------------------------------------------------------------------

def make_note(
    pitch: int,
    onset,
    duration,
    voice: int = 0,
    measure: int = 1,
    beat_strength: float = 1.0,
) -> MusicalEvent:
    return MusicalEvent(
        pitch=pitch,
        onset=Fraction(onset),
        duration=Fraction(duration),
        velocity=64,
        measure=measure,
        beat=Fraction(0),
        beat_strength=beat_strength,
        voice=voice,
        is_rest=False,
    )


def make_rest(onset, duration, voice: int = 0, measure: int = 1) -> MusicalEvent:
    return MusicalEvent(
        pitch=None,
        onset=Fraction(onset),
        duration=Fraction(duration),
        velocity=0,
        measure=measure,
        beat=Fraction(0),
        beat_strength=1.0,
        voice=voice,
        is_rest=True,
    )


def make_score(events, numerator: int = 4, denominator: int = 4) -> Score:
    ts = TimeSignature(numerator, denominator)
    return Score(events=events, time_signatures=[(Fraction(0), ts)])
