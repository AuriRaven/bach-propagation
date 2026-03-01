"""
Integration tests for ScoreLoader — requires sample MIDI files under data/samples/.

Tests are skipped gracefully when the MIDI files are absent so the suite
still passes in CI environments that do not ship the sample data.

Coverage:
  Basic load        – Score structure, event validity, sort order
  separate_voices   – voice=0 for flat mode; same count for monophonic MIDI
  Time-sig cache    – _get_time_signature_at_offset calls _extract once per score
  Error handling    – missing file, bad extension
  Convenience fn    – load_score() wrapper
"""

import pytest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from src.core.score_loader import ScoreLoader, ScoreLoadError, load_score
from src.core.data_structures import Score

# ---------------------------------------------------------------------------
# Paths (resolved from the project root so tests run from any cwd)
# ---------------------------------------------------------------------------

_SOLO = Path(__file__).parent.parent.parent / "data" / "samples" / "01_bach" / "solo_instruments"
_PRELUDE = _SOLO / "cello" / "cello_suites" / "bwv1007_suite_no_1_in_g_major" / "01_prelude.mid"
_ADAGIO  = _SOLO / "violin" / "sonatas" / "bwv1001_sonata_no_1_in_g_minor" / "01_adagio.mid"

_SAMPLES_PRESENT = _PRELUDE.exists()


# ===========================================================================
# Basic loading — uses the module-scoped fixture from conftest.py
# ===========================================================================

class TestBasicLoad:
    """Structural sanity checks on the loaded Score object."""

    def test_returns_score_instance(self, prelude_score):
        assert isinstance(prelude_score, Score)

    def test_has_events(self, prelude_score):
        assert len(prelude_score.events) > 0

    def test_events_sorted_by_onset(self, prelude_score):
        onsets = [ev.onset for ev in prelude_score.events]
        assert onsets == sorted(onsets)

    def test_positive_duration(self, prelude_score):
        assert prelude_score.duration > 0

    def test_all_events_positive_duration(self, prelude_score):
        for ev in prelude_score.events:
            assert ev.duration > 0, f"Zero/negative duration at onset {ev.onset}"

    def test_all_notes_valid_midi_pitch(self, prelude_score):
        for ev in prelude_score.events:
            if not ev.is_rest:
                assert ev.pitch is not None
                assert 0 <= ev.pitch <= 127, f"Pitch {ev.pitch} out of MIDI range"

    def test_all_events_valid_beat_strength(self, prelude_score):
        for ev in prelude_score.events:
            assert 0.0 <= ev.beat_strength <= 1.0

    def test_at_least_one_time_signature(self, prelude_score):
        assert len(prelude_score.time_signatures) >= 1

    def test_metadata_source_file(self, prelude_score):
        assert 'source_file' in prelude_score.metadata

    def test_title_populated(self, prelude_score):
        # Title may come from MIDI metadata or fall back to the filename stem
        assert prelude_score.title is not None
        assert len(prelude_score.title) > 0


# ===========================================================================
# separate_voices parameter
# ===========================================================================

class TestSeparateVoices:

    @pytest.mark.skipif(not _SAMPLES_PRESENT, reason="Sample MIDI not found")
    def test_flat_all_events_in_voice_zero(self):
        loader = ScoreLoader(separate_voices=False)
        score = loader.load(_PRELUDE)
        assert all(ev.voice == 0 for ev in score.events)
        assert score.num_voices == 1

    @pytest.mark.skipif(not _SAMPLES_PRESENT, reason="Sample MIDI not found")
    def test_separate_voices_reflects_midi_tracks(self):
        """MuseScore exports each notated voice as its own MIDI track, so even
        a cello suite may have 3-4 tracks (some nearly empty). With
        separate_voices=True the loader preserves that structure."""
        sep = ScoreLoader(separate_voices=True).load(_PRELUDE)
        n_parts = sep.metadata.get('num_parts', 1)
        assert sep.num_voices == n_parts

    @pytest.mark.skipif(not _SAMPLES_PRESENT, reason="Sample MIDI not found")
    def test_flat_always_one_voice_separate_always_at_least_one(self):
        flat = ScoreLoader(separate_voices=False).load(_PRELUDE)
        sep  = ScoreLoader(separate_voices=True).load(_PRELUDE)
        assert flat.num_voices == 1
        assert sep.num_voices >= 1


# ===========================================================================
# Time-signature cache
# ===========================================================================

class TestTimeSignatureCache:

    @pytest.mark.skipif(not _SAMPLES_PRESENT, reason="Sample MIDI not found")
    def test_cache_populated_after_first_query(self):
        from music21 import converter
        m21_score = converter.parse(str(_PRELUDE))
        loader = ScoreLoader()
        loader._get_time_signature_at_offset(m21_score, Fraction(0))
        assert loader._cached_time_sigs is not None
        assert loader._cached_time_sigs_score_id == id(m21_score)

    @pytest.mark.skipif(not _SAMPLES_PRESENT, reason="Sample MIDI not found")
    def test_extract_called_once_for_multiple_queries(self):
        from music21 import converter
        m21_score = converter.parse(str(_PRELUDE))
        loader = ScoreLoader()

        with patch.object(
            loader, '_extract_time_signatures',
            wraps=loader._extract_time_signatures,
        ) as mock_extract:
            loader._get_time_signature_at_offset(m21_score, Fraction(0))
            loader._get_time_signature_at_offset(m21_score, Fraction(4))
            loader._get_time_signature_at_offset(m21_score, Fraction(8))

        assert mock_extract.call_count == 1, (
            "_extract_time_signatures should be called only once per score"
        )

    @pytest.mark.skipif(not _SAMPLES_PRESENT, reason="Sample MIDI not found")
    def test_cache_resets_for_different_score(self):
        from music21 import converter
        m21_a = converter.parse(str(_PRELUDE))
        m21_b = converter.parse(str(_PRELUDE))   # same content, different object
        loader = ScoreLoader()

        with patch.object(
            loader, '_extract_time_signatures',
            wraps=loader._extract_time_signatures,
        ) as mock_extract:
            loader._get_time_signature_at_offset(m21_a, Fraction(0))
            loader._get_time_signature_at_offset(m21_b, Fraction(0))

        assert mock_extract.call_count == 2


# ===========================================================================
# Error handling
# ===========================================================================

class TestErrorHandling:

    def test_missing_file_raises_score_load_error(self):
        loader = ScoreLoader()
        with pytest.raises(ScoreLoadError):
            loader.load("definitely_does_not_exist.mid")

    def test_unsupported_extension_raises_score_load_error(self, tmp_path):
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not music")
        loader = ScoreLoader()
        with pytest.raises(ScoreLoadError):
            loader.load(bad_file)

    def test_score_load_error_is_exception(self):
        assert issubclass(ScoreLoadError, Exception)


# ===========================================================================
# Corpus loading
# ===========================================================================

def _corpus_available() -> bool:
    try:
        from music21 import corpus
        corpus.parse('bwv846')
        return True
    except Exception:
        return False


_CORPUS_PRESENT = _corpus_available()


@pytest.mark.skipif(not _CORPUS_PRESENT, reason="music21 corpus not available")
class TestCorpusLoading:

    def test_load_corpus_bwv846_returns_score(self):
        score = ScoreLoader(separate_voices=False).load_corpus('846')
        assert isinstance(score, Score)
        assert len(score.events) > 0

    def test_load_corpus_bwv846_key_is_c_major(self):
        from src.harmonic.key_estimator import KeyEstimator
        score = ScoreLoader(separate_voices=False).load_corpus('846')
        key = KeyEstimator().estimate_global(score)
        assert key.tonic == 0
        assert key.mode == 'major'

    def test_load_from_m21_same_event_count_as_load_corpus(self):
        from music21 import corpus
        m21_score = corpus.parse('bwv846')
        loader = ScoreLoader(separate_voices=False)
        score_a = loader.load_corpus('846')
        score_b = loader.load_from_m21(m21_score)
        assert len(score_a.events) == len(score_b.events)


# ===========================================================================
# Convenience wrapper
# ===========================================================================

@pytest.mark.skipif(not _SAMPLES_PRESENT, reason="Sample MIDI not found")
def test_load_score_convenience_function():
    score = load_score(_PRELUDE, separate_voices=False)
    assert isinstance(score, Score)
    assert len(score.events) > 0
