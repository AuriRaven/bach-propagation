"""
Tests for src/data/sequence_extractor.py

TestExtractSequenceUnit  — unit tests using mocked pipeline components
TestAugmentSequences     — transposition augmentation
TestTransposeSequence    — _transpose_sequence helper
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.core.data_structures import (
    Chord,
    HarmonicFunction,
    KeyEstimate,
    RomanNumeral,
    Score,
    TimeSignature,
)
from src.data.sequence_extractor import (
    ExtractionConfig,
    augment_sequences,
    extract_sequence,
    _transpose_sequence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(n: int = 4, d: int = 4) -> TimeSignature:
    return TimeSignature(n, d)


def _make_score() -> Score:
    from tests.conftest import make_note, make_score
    events = [
        make_note(60, onset=0, duration=4),
        make_note(64, onset=4, duration=4),
        make_note(67, onset=8, duration=4),
    ]
    return make_score(events)


def _make_chord(onset: int, root: int = 0, quality: str = 'major') -> Chord:
    return Chord(
        onset=Fraction(onset),
        duration=Fraction(4),
        root=root,
        quality=quality,
        pitches=frozenset([root, (root + 4) % 12, (root + 7) % 12]),
        bass=root,
        inversion=0,
    )


def _make_rn(
    degree: int = 1,
    quality: str = 'major',
    function: HarmonicFunction = HarmonicFunction.TONIC,
    tonic: int = 0,
) -> RomanNumeral:
    return RomanNumeral(
        degree=degree,
        quality=quality,
        inversion=0,
        local_key_tonic=tonic,
        local_key_mode='major',
        function=function,
    )


def _make_seq(n: int = 3) -> List[dict]:
    """Build a synthetic event dict list of length n."""
    functions = [HarmonicFunction.TONIC, HarmonicFunction.PREDOMINANT, HarmonicFunction.DOMINANT]
    return [
        {
            'onset': float(i * 4),
            'duration': 4.0,
            'pitch_classes': [0, 4, 7],
            'chord_root': i % 12,
            'chord_quality': 'major',
            'rn_degree': i + 1,
            'harmonic_function': i % 3,
            'local_key_tonic': 0,
            'local_key_mode': 'major',
            'metric_weight': 1.0,
            'inversion': 0,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# TestExtractSequenceUnit
# ---------------------------------------------------------------------------

class TestExtractSequenceUnit:

    def _run_extract(self, tmp_path: Path, score: Score, chords, rns, labelled_rns):
        """Patch the pipeline and call extract_sequence."""
        midi_file = tmp_path / "test.mid"
        midi_file.touch()

        global_key = KeyEstimate(tonic=0, mode='major', confidence=0.9, onset=Fraction(0))
        local_keys = []

        with (
            patch('src.data.sequence_extractor.ScoreLoader') as mock_loader_cls,
            patch('src.data.sequence_extractor.KeyEstimator') as mock_ke_cls,
            patch('src.data.sequence_extractor.ChordClassifier') as mock_cc_cls,
            patch('src.data.sequence_extractor.RomanNumeralAnalyzer') as mock_rna_cls,
            patch('src.data.sequence_extractor.FunctionLabeler') as mock_fl_cls,
        ):
            mock_loader = MagicMock()
            mock_loader.load.return_value = score
            mock_loader_cls.return_value = mock_loader

            mock_ke = MagicMock()
            mock_ke.estimate_global.return_value = global_key
            mock_ke.estimate_local.return_value = local_keys
            mock_ke_cls.return_value = mock_ke

            mock_cc = MagicMock()
            mock_cc.classify_score.return_value = chords
            mock_cc_cls.return_value = mock_cc

            mock_rna = MagicMock()
            mock_rna.analyze_sequence.return_value = rns
            mock_rna_cls.return_value = mock_rna

            mock_fl = MagicMock()
            mock_fl.label_sequence.return_value = labelled_rns
            mock_fl_cls.return_value = mock_fl

            cfg = ExtractionConfig()
            return extract_sequence(midi_file, cfg)

    def test_returns_list_of_dicts(self, tmp_path):
        score = _make_score()
        chords = [_make_chord(0), _make_chord(4, root=7), _make_chord(8, root=0)]
        rns = [_make_rn(1), _make_rn(5, 'major', HarmonicFunction.DOMINANT, 0), _make_rn(1)]
        result = self._run_extract(tmp_path, score, chords, rns, rns)
        assert isinstance(result, list)
        assert all(isinstance(ev, dict) for ev in result)

    def test_length_matches_chords(self, tmp_path):
        score = _make_score()
        chords = [_make_chord(0), _make_chord(4)]
        rns = [_make_rn(1), _make_rn(4, 'minor', HarmonicFunction.PREDOMINANT)]
        result = self._run_extract(tmp_path, score, chords, rns, rns)
        assert len(result) == 2

    def test_required_keys_present(self, tmp_path):
        required = {
            'onset', 'duration', 'pitch_classes', 'chord_root', 'chord_quality',
            'rn_degree', 'harmonic_function', 'local_key_tonic', 'local_key_mode',
            'metric_weight', 'inversion',
        }
        score = _make_score()
        chords = [_make_chord(0)]
        rns = [_make_rn(1)]
        result = self._run_extract(tmp_path, score, chords, rns, rns)
        assert len(result) == 1
        assert required.issubset(result[0].keys())

    def test_empty_chord_list_returns_empty(self, tmp_path):
        score = _make_score()
        result = self._run_extract(tmp_path, score, chords=[], rns=[], labelled_rns=[])
        assert result == []

    def test_harmonic_function_tonic_is_zero(self, tmp_path):
        score = _make_score()
        chords = [_make_chord(0)]
        rns = [_make_rn(1, function=HarmonicFunction.TONIC)]
        result = self._run_extract(tmp_path, score, chords, rns, rns)
        assert result[0]['harmonic_function'] == 0

    def test_harmonic_function_predominant_is_one(self, tmp_path):
        score = _make_score()
        chords = [_make_chord(0)]
        rns = [_make_rn(4, function=HarmonicFunction.PREDOMINANT)]
        result = self._run_extract(tmp_path, score, chords, rns, rns)
        assert result[0]['harmonic_function'] == 1

    def test_harmonic_function_dominant_is_two(self, tmp_path):
        score = _make_score()
        chords = [_make_chord(0)]
        rns = [_make_rn(5, function=HarmonicFunction.DOMINANT)]
        result = self._run_extract(tmp_path, score, chords, rns, rns)
        assert result[0]['harmonic_function'] == 2

    def test_metric_weight_is_float(self, tmp_path):
        score = _make_score()
        chords = [_make_chord(0)]
        rns = [_make_rn(1)]
        result = self._run_extract(tmp_path, score, chords, rns, rns)
        assert isinstance(result[0]['metric_weight'], float)

    def test_pitch_classes_are_sorted(self, tmp_path):
        score = _make_score()
        chords = [_make_chord(0)]  # pitches: frozenset {0, 4, 7}
        rns = [_make_rn(1)]
        result = self._run_extract(tmp_path, score, chords, rns, rns)
        pcs = result[0]['pitch_classes']
        assert pcs == sorted(pcs)

    def test_onset_and_duration_are_floats(self, tmp_path):
        score = _make_score()
        chords = [_make_chord(0)]
        rns = [_make_rn(1)]
        result = self._run_extract(tmp_path, score, chords, rns, rns)
        assert isinstance(result[0]['onset'], float)
        assert isinstance(result[0]['duration'], float)


# ---------------------------------------------------------------------------
# TestAugmentSequences
# ---------------------------------------------------------------------------

class TestAugmentSequences:

    def test_n_shifts_1_returns_same_length(self):
        seqs = [_make_seq(3)]
        result = augment_sequences(seqs, n_shifts=1)
        assert len(result) == 1

    def test_n_shifts_12_returns_12x_input(self):
        seqs = [_make_seq(3), _make_seq(2)]
        result = augment_sequences(seqs, n_shifts=12)
        assert len(result) == 24  # 2 seqs × 12 shifts

    def test_root_wraps_0_to_11(self):
        seq = [{'chord_root': 11, 'local_key_tonic': 0, 'pitch_classes': [11]}]
        result = augment_sequences([seq], n_shifts=12)
        roots = [r[0]['chord_root'] for r in result]
        assert set(roots) == set(range(12))

    def test_key_tonic_wraps(self):
        seq = [{'chord_root': 0, 'local_key_tonic': 10, 'pitch_classes': [0]}]
        result = augment_sequences([seq], n_shifts=12)
        tonics = [r[0]['local_key_tonic'] for r in result]
        assert set(tonics) == set(range(12))

    def test_pitch_classes_wrap(self):
        seq = [{'chord_root': 0, 'local_key_tonic': 0, 'pitch_classes': [0, 6, 11]}]
        result = augment_sequences([seq], n_shifts=2)
        # shift=1: [1, 7, 0] sorted → [0, 1, 7]
        pcs_shift1 = result[1][0]['pitch_classes']
        assert sorted(pcs_shift1) == pcs_shift1
        assert all(0 <= pc <= 11 for pc in pcs_shift1)

    def test_quality_unchanged(self):
        seq = [{'chord_root': 0, 'local_key_tonic': 0, 'pitch_classes': [0], 'chord_quality': 'diminished'}]
        result = augment_sequences([seq], n_shifts=6)
        for shifted in result:
            assert shifted[0]['chord_quality'] == 'diminished'

    def test_degree_and_function_unchanged(self):
        seq = [{'chord_root': 0, 'local_key_tonic': 0, 'pitch_classes': [0],
                'rn_degree': 5, 'harmonic_function': 2}]
        result = augment_sequences([seq], n_shifts=6)
        for shifted in result:
            assert shifted[0]['rn_degree'] == 5
            assert shifted[0]['harmonic_function'] == 2


# ---------------------------------------------------------------------------
# TestTransposeSequence
# ---------------------------------------------------------------------------

class TestTransposeSequence:

    def test_zero_shift_returns_identical(self):
        seq = _make_seq(3)
        result = _transpose_sequence(seq, 0)
        assert result == seq

    def test_zero_shift_returns_copy_not_same(self):
        seq = _make_seq(2)
        result = _transpose_sequence(seq, 0)
        result[0]['chord_root'] = 99
        assert seq[0]['chord_root'] != 99

    def test_root_plus_6_wraps_correctly(self):
        seq = [{'chord_root': 9, 'local_key_tonic': 0, 'pitch_classes': [9]}]
        result = _transpose_sequence(seq, 6)
        assert result[0]['chord_root'] == 3  # (9 + 6) % 12

    def test_pitch_classes_shifted(self):
        seq = [{'chord_root': 0, 'local_key_tonic': 0, 'pitch_classes': [0, 4, 7]}]
        result = _transpose_sequence(seq, 3)
        assert result[0]['pitch_classes'] == [3, 7, 10]

    def test_non_pitch_fields_preserved(self):
        seq = [{
            'onset': 0.0,
            'duration': 4.0,
            'chord_root': 0,
            'local_key_tonic': 0,
            'pitch_classes': [0, 4, 7],
            'chord_quality': 'major',
            'rn_degree': 1,
            'harmonic_function': 0,
            'local_key_mode': 'major',
            'metric_weight': 1.0,
            'inversion': 0,
        }]
        result = _transpose_sequence(seq, 5)
        ev = result[0]
        assert ev['onset'] == 0.0
        assert ev['duration'] == 4.0
        assert ev['chord_quality'] == 'major'
        assert ev['rn_degree'] == 1
        assert ev['harmonic_function'] == 0
        assert ev['local_key_mode'] == 'major'
        assert ev['metric_weight'] == 1.0
        assert ev['inversion'] == 0
