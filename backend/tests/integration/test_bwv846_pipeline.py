"""
End-to-end integration test: BWV 846 (WTC Book I, C major Prelude).

Runs the full pipeline:
  ScoreLoader.load_corpus → KeyEstimator → ChordClassifier
  → RomanNumeralAnalyzer → FunctionLabeler

BWV 846 is ideal for integration testing: arpeggiated texture means each
measure contains exactly one chord (explicit harmony), the key never changes,
and the expected Roman numeral sequence is well-documented in every harmony
textbook. All assertions are against known analytical facts.

Tests skip gracefully if music21's built-in corpus is unavailable.
"""

import pytest
from fractions import Fraction

from src.core.data_structures import HarmonicFunction
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
# Corpus availability guard
# ---------------------------------------------------------------------------

def _corpus_available() -> bool:
    try:
        from music21 import corpus
        corpus.parse('bwv846')
        return True
    except Exception:
        return False


_CORPUS_OK = _corpus_available()
_SKIP = pytest.mark.skipif(not _CORPUS_OK, reason="music21 corpus not available")


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bwv846_score():
    if not _CORPUS_OK:
        pytest.skip("music21 corpus not available")
    return ScoreLoader(separate_voices=False).load_corpus('846')


@pytest.fixture(scope="module")
def bwv846_analysis(bwv846_score):
    """Run the full pipeline; return (global_key, chords, rns, labelled)."""
    ke = KeyEstimator()
    global_key = ke.estimate_global(bwv846_score)
    local_keys = ke.estimate_local(bwv846_score, Fraction(16), Fraction(8))

    cfg = ChordClassificationConfig(
        min_confidence=0.25,
        include_seventh_chords=True,
        beat_weight_exponent=1.0,
    )
    chords = ChordClassifier().classify_score(bwv846_score, WindowType.MEASURE, cfg)

    key_list = local_keys if local_keys else [global_key]
    rns = RomanNumeralAnalyzer().analyze_sequence(chords, key_list)
    labelled = FunctionLabeler().label_sequence(rns)

    return global_key, chords, rns, labelled


# ---------------------------------------------------------------------------
# TestBWV846GlobalKey
# ---------------------------------------------------------------------------

@_SKIP
class TestBWV846GlobalKey:

    def test_tonic_is_c(self, bwv846_analysis):
        global_key, *_ = bwv846_analysis
        assert global_key.tonic == 0, (
            f"Expected C (0), got {global_key.label}"
        )

    def test_mode_is_major(self, bwv846_analysis):
        global_key, *_ = bwv846_analysis
        assert global_key.mode == 'major'

    def test_confidence_above_threshold(self, bwv846_analysis):
        global_key, *_ = bwv846_analysis
        assert global_key.confidence > 0.6, (
            f"Confidence too low: {global_key.confidence:.3f}"
        )


# ---------------------------------------------------------------------------
# TestBWV846Chords
# ---------------------------------------------------------------------------

@_SKIP
class TestBWV846Chords:

    def test_first_chord_root_is_c(self, bwv846_analysis):
        _, chords, _, _ = bwv846_analysis
        assert chords[0].root == 0, (
            f"First chord root: {chords[0].root} ({chords[0].quality})"
        )

    def test_first_chord_quality_is_major(self, bwv846_analysis):
        _, chords, _, _ = bwv846_analysis
        assert chords[0].quality == 'major', (
            f"First chord quality: {chords[0].quality}"
        )

    def test_chords_cover_all_measures(self, bwv846_analysis):
        # BWV 846 has 35 measures; after filtering rest-only/low-conf windows
        # we expect at least 30 classified chords.
        _, chords, _, _ = bwv846_analysis
        assert len(chords) >= 30, (
            f"Too few chords classified: {len(chords)}"
        )

    def test_g_dominant_chord_present(self, bwv846_analysis):
        # G (root=7) appears as V multiple times in C major.
        _, chords, _, _ = bwv846_analysis
        g_chords = [c for c in chords if c.root == 7]
        assert len(g_chords) >= 1, "No G chord found — V is expected in C major prelude"


# ---------------------------------------------------------------------------
# TestBWV846RomanNumerals
# ---------------------------------------------------------------------------

@_SKIP
class TestBWV846RomanNumerals:

    def test_first_rn_is_degree_one(self, bwv846_analysis):
        _, _, rns, _ = bwv846_analysis
        assert rns[0].degree == 1, f"First RN degree: {rns[0].degree}"

    def test_first_rn_function_is_tonic(self, bwv846_analysis):
        _, _, _, labelled = bwv846_analysis
        assert labelled[0].function == HarmonicFunction.TONIC

    def test_dominant_rns_present(self, bwv846_analysis):
        _, _, _, labelled = bwv846_analysis
        dominant_rns = [rn for rn in labelled
                        if rn.function == HarmonicFunction.DOMINANT]
        assert len(dominant_rns) >= 5, (
            f"Too few DOMINANT RNs: {len(dominant_rns)}"
        )

    def test_tonic_rns_present(self, bwv846_analysis):
        _, _, _, labelled = bwv846_analysis
        tonic_rns = [rn for rn in labelled
                     if rn.function == HarmonicFunction.TONIC]
        assert len(tonic_rns) >= 3

    def test_print_measure_table(self, bwv846_analysis, bwv846_score, capsys):
        """Print the measure-by-measure analysis table for manual verification."""
        global_key, chords, _, labelled = bwv846_analysis
        header = f"\n{'Onset':>6} | {'Root':>4} {'Quality':>10} | {'Key':>10} | {'RN':>8} | Function"
        print(header)
        print("-" * 60)
        for chord, rn in zip(chords[:20], labelled[:20]):
            func_name = rn.function.name if rn.function else '?'
            print(
                f"{float(chord.onset):>6.1f} | {chord.root:>4} {chord.quality:>10} | "
                f"{global_key.label:>10} | {rn.label:>8} | {func_name}"
            )
        # Ensure table was printed without errors (captured by capsys)
        captured = capsys.readouterr()
        assert "Onset" in captured.out
        assert "TONIC" in captured.out or "DOMINANT" in captured.out
