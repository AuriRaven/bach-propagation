"""
Tests for :mod:`src.data.harmonic_encoder`.

Structure:
    TestLabelExtractors         — chord / rn / cadence label rendering
    TestBuildVocabulary         — frequency sort, ties, min-count filtering
    TestHarmonicVocabularyIO    — save → load round-trip, schema-version guard
    TestEncoderAlignment        — chord/rn/cadence stream length invariants
    TestEncoderCadencePlacement — cadences land at final_index, NONE elsewhere
    TestEncoderUnk              — unseen labels fall back to UNK
    TestEncoderDecode           — decode renders specials + labels
    TestFromAnalyses            — in-memory construction shortcut
    TestEncodeFile              — load + encode JSON from disk
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from src.data.harmonic_encoder import (
    BOS,
    CAD_NONE,
    EOS,
    EncodedSequence,
    HarmonicEncoder,
    HarmonicEncoderConfig,
    HarmonicEncoderError,
    HarmonicVocabulary,
    PAD,
    SchemaVersionMismatch,
    UNK,
    VOCAB_SCHEMA_VERSION,
    build_vocabulary,
    cadence_label,
    chord_label,
    rn_label,
)


# ---------------------------------------------------------------------------
# Fixtures — in-memory Stage 4 analysis dicts
# ---------------------------------------------------------------------------


def _chord(root: int, quality: str, onset: str = "0") -> Dict[str, object]:
    return {
        "onset": onset,
        "duration": "1",
        "root": root,
        "quality": quality,
        "pitches": [],
        "bass": root,
        "inversion": 0,
    }


def _rn(label: str) -> Dict[str, object]:
    return {
        "degree": 1,
        "quality": "major",
        "inversion": 0,
        "local_key_tonic": 0,
        "local_key_mode": "major",
        "is_secondary": False,
        "secondary_target": None,
        "function": "TONIC",
        "label": label,
    }


def _cad(cadence_type: str, penultimate_index: int, final_index: int) -> Dict[str, object]:
    return {
        "onset": str(final_index),
        "cadence_type": cadence_type,
        "label": cadence_type,
        "confidence": 1.0,
        "penultimate_index": penultimate_index,
        "final_index": final_index,
    }


@pytest.fixture
def c_major_analysis() -> Dict[str, object]:
    """I–IV–V–I in C major with a PAC at the end (V→I, indices 2→3)."""
    return {
        "score_id": "toy",
        "chords": [
            _chord(0, "major", "0"),   # C: I
            _chord(5, "major", "1"),   # F: IV
            _chord(7, "major", "2"),   # G: V
            _chord(0, "major", "3"),   # C: I
        ],
        "roman_numerals": [
            _rn("I"),
            _rn("IV"),
            _rn("V"),
            _rn("I"),
        ],
        "cadences": [
            _cad("authentic_perfect", 2, 3),
        ],
    }


@pytest.fixture
def minor_analysis() -> Dict[str, object]:
    """i–iv–V–i in A minor with a half cadence at index 2."""
    return {
        "score_id": "toy2",
        "chords": [
            _chord(9, "minor", "0"),
            _chord(2, "minor", "1"),
            _chord(4, "major", "2"),
            _chord(9, "minor", "3"),
        ],
        "roman_numerals": [
            _rn("i"),
            _rn("iv"),
            _rn("V"),
            _rn("i"),
        ],
        "cadences": [
            _cad("half", 1, 2),
        ],
    }


# ---------------------------------------------------------------------------
# TestLabelExtractors
# ---------------------------------------------------------------------------


class TestLabelExtractors:
    def test_chord_label_formats_root_and_quality(self):
        assert chord_label(_chord(0, "major")) == "C:major"
        assert chord_label(_chord(7, "minor7")) == "G:minor7"
        # Pitch-class 3 is "Eb" in the project table.
        assert chord_label(_chord(3, "diminished")) == "Eb:diminished"

    def test_rn_label_uses_stored_label_field(self):
        assert rn_label(_rn("V7")) == "V7"
        assert rn_label(_rn("V/V")) == "V/V"
        assert rn_label(_rn("ii\u00f865")) == "ii\u00f865"

    def test_cadence_label_uses_cadence_type(self):
        assert cadence_label(_cad("authentic_perfect", 0, 1)) == "authentic_perfect"
        assert cadence_label(_cad("phrygian_half", 0, 1)) == "phrygian_half"


# ---------------------------------------------------------------------------
# TestBuildVocabulary
# ---------------------------------------------------------------------------


class TestBuildVocabulary:
    def test_specials_get_canonical_ids(self, c_major_analysis):
        vocab = build_vocabulary([c_major_analysis], HarmonicEncoderConfig())
        assert vocab.chord_to_id["<pad>"] == PAD
        assert vocab.chord_to_id["<bos>"] == BOS
        assert vocab.chord_to_id["<eos>"] == EOS
        assert vocab.chord_to_id["<unk>"] == UNK
        # Cadence stream has an extra sentinel at CAD_NONE.
        assert vocab.cadence_to_id["<none>"] == CAD_NONE

    def test_chord_labels_are_sorted_by_frequency(self):
        # C:major appears 3x, F:major 1x — C should get a lower id.
        analysis = {
            "score_id": "freq",
            "chords": [
                _chord(0, "major"), _chord(0, "major"), _chord(0, "major"),
                _chord(5, "major"),
            ],
            "roman_numerals": [_rn("I"), _rn("I"), _rn("I"), _rn("IV")],
            "cadences": [],
        }
        vocab = build_vocabulary([analysis], HarmonicEncoderConfig())
        assert vocab.chord_to_id["C:major"] < vocab.chord_to_id["F:major"]

    def test_alphabetical_tiebreak(self):
        # A:major and B:major each appear once — ids should be alphabetical.
        analysis = {
            "score_id": "tie",
            "chords": [_chord(9, "major"), _chord(11, "major")],
            "roman_numerals": [_rn("VI"), _rn("VII")],
            "cadences": [],
        }
        vocab = build_vocabulary([analysis], HarmonicEncoderConfig())
        assert vocab.chord_to_id["A:major"] < vocab.chord_to_id["B:major"]

    def test_min_chord_count_filters_rare_labels(self):
        analysis = {
            "score_id": "filter",
            "chords": [
                _chord(0, "major"), _chord(0, "major"), _chord(0, "major"),
                _chord(5, "major"),  # appears once — should be filtered
            ],
            "roman_numerals": [_rn("I"), _rn("I"), _rn("I"), _rn("IV")],
            "cadences": [],
        }
        cfg = HarmonicEncoderConfig(min_chord_count=2, min_rn_count=1)
        vocab = build_vocabulary([analysis], cfg)
        assert "C:major" in vocab.chord_to_id
        assert "F:major" not in vocab.chord_to_id

    def test_min_rn_count_filters_rare_labels(self):
        analysis = {
            "score_id": "filter",
            "chords": [
                _chord(0, "major"), _chord(0, "major"), _chord(0, "major"),
            ],
            # "I" appears twice, "V/V" once — with min_rn_count=2, only
            # "I" should survive.
            "roman_numerals": [_rn("I"), _rn("I"), _rn("V/V")],
            "cadences": [],
        }
        cfg = HarmonicEncoderConfig(min_chord_count=1, min_rn_count=2)
        vocab = build_vocabulary([analysis], cfg)
        assert "I" in vocab.rn_to_id
        assert "V/V" not in vocab.rn_to_id

    def test_cadence_vocab_shape(self, c_major_analysis, minor_analysis):
        vocab = build_vocabulary(
            [c_major_analysis, minor_analysis], HarmonicEncoderConfig()
        )
        # 4 shared specials + <none> sentinel + 2 cadence types = 7.
        assert vocab.cadence_vocab_size == 7
        assert "authentic_perfect" in vocab.cadence_to_id
        assert "half" in vocab.cadence_to_id


# ---------------------------------------------------------------------------
# TestHarmonicVocabularyIO
# ---------------------------------------------------------------------------


class TestHarmonicVocabularyIO:
    def test_save_load_roundtrip(self, tmp_path: Path, c_major_analysis):
        vocab = build_vocabulary([c_major_analysis], HarmonicEncoderConfig())
        path = tmp_path / "vocab.json"
        vocab.save(path)
        loaded = HarmonicVocabulary.load(path)
        assert loaded.chord_to_id == vocab.chord_to_id
        assert loaded.rn_to_id == vocab.rn_to_id
        assert loaded.cadence_to_id == vocab.cadence_to_id

    def test_written_file_records_schema_version(self, tmp_path: Path):
        HarmonicVocabulary().save(tmp_path / "v.json")
        data = json.loads((tmp_path / "v.json").read_text())
        assert data["schema_version"] == VOCAB_SCHEMA_VERSION

    def test_load_rejects_schema_mismatch(self, tmp_path: Path):
        path = tmp_path / "v.json"
        path.write_text(json.dumps({
            "schema_version": "0.0.1",
            "chord_to_id": {},
            "rn_to_id": {},
            "cadence_to_id": {},
        }))
        with pytest.raises(SchemaVersionMismatch):
            HarmonicVocabulary.load(path)


# ---------------------------------------------------------------------------
# TestEncoderAlignment
# ---------------------------------------------------------------------------


class TestEncoderAlignment:
    def test_all_streams_same_length(self, c_major_analysis):
        enc = HarmonicEncoder.from_analyses([c_major_analysis])
        seq = enc.encode(c_major_analysis)
        # 4 chords + BOS + EOS = 6
        assert seq.length == 6
        assert len(seq.chord_tokens) == len(seq.rn_tokens) == len(seq.cadence_tokens)

    def test_sequence_wrapped_with_bos_eos(self, c_major_analysis):
        enc = HarmonicEncoder.from_analyses([c_major_analysis])
        seq = enc.encode(c_major_analysis)
        assert seq.chord_tokens[0] == BOS
        assert seq.rn_tokens[0] == BOS
        assert seq.cadence_tokens[0] == BOS
        assert seq.chord_tokens[-1] == EOS
        assert seq.rn_tokens[-1] == EOS
        assert seq.cadence_tokens[-1] == EOS

    def test_onset_strings_align_with_chords(self, c_major_analysis):
        enc = HarmonicEncoder.from_analyses([c_major_analysis])
        seq = enc.encode(c_major_analysis)
        # Specials have empty onset; body follows chord onsets.
        assert seq.onsets[0] == ""
        assert seq.onsets[-1] == ""
        assert seq.onsets[1:-1] == ["0", "1", "2", "3"]

    def test_mismatched_chord_rn_length_raises(self):
        bad = {
            "score_id": "bad",
            "chords": [_chord(0, "major"), _chord(5, "major")],
            "roman_numerals": [_rn("I")],   # only one!
            "cadences": [],
        }
        enc = HarmonicEncoder.from_analyses([bad])  # build_vocabulary tolerates mismatch
        with pytest.raises(HarmonicEncoderError):
            enc.encode(bad)


# ---------------------------------------------------------------------------
# TestEncoderCadencePlacement
# ---------------------------------------------------------------------------


class TestEncoderCadencePlacement:
    def test_cadence_lands_at_final_index(self, c_major_analysis):
        enc = HarmonicEncoder.from_analyses([c_major_analysis])
        seq = enc.encode(c_major_analysis)
        # final_index=3 (last chord) — position in stream is 3 + BOS offset = 4.
        cad_id = seq.cadence_tokens[4]
        assert cad_id == enc.vocabulary.cadence_id("authentic_perfect")

    def test_non_cadence_timesteps_are_none(self, c_major_analysis):
        enc = HarmonicEncoder.from_analyses([c_major_analysis])
        seq = enc.encode(c_major_analysis)
        # Indices 1, 2, 3 (body before PAC) should all be CAD_NONE.
        assert seq.cadence_tokens[1:4] == [CAD_NONE, CAD_NONE, CAD_NONE]

    def test_cadence_out_of_range_is_ignored(self):
        # Mimic a corrupt cadence with final_index pointing past the chord
        # list — encoder must silently fall back to NONE rather than crash.
        analysis = {
            "score_id": "corrupt",
            "chords": [_chord(0, "major")],
            "roman_numerals": [_rn("I")],
            "cadences": [_cad("authentic_perfect", 5, 9)],
        }
        enc = HarmonicEncoder.from_analyses([analysis])
        seq = enc.encode(analysis)
        # The single body timestep should be NONE since the cadence index
        # fell outside [0, 1).
        assert seq.cadence_tokens[1] == CAD_NONE


# ---------------------------------------------------------------------------
# TestEncoderUnk
# ---------------------------------------------------------------------------


class TestEncoderUnk:
    def test_unseen_chord_falls_back_to_unk(self, c_major_analysis):
        enc = HarmonicEncoder.from_analyses([c_major_analysis])
        # Encode a brand-new analysis containing Bb:major, which was not in
        # the training set.
        new = {
            "score_id": "new",
            "chords": [_chord(10, "major")],     # Bb:major — unseen
            "roman_numerals": [_rn("bVII")],     # bVII — unseen
            "cadences": [],
        }
        seq = enc.encode(new)
        # Body timestep = UNK for both streams.
        assert seq.chord_tokens[1] == UNK
        assert seq.rn_tokens[1] == UNK


# ---------------------------------------------------------------------------
# TestEncoderDecode
# ---------------------------------------------------------------------------


class TestEncoderDecode:
    def test_decode_renders_specials_and_body(self, c_major_analysis):
        enc = HarmonicEncoder.from_analyses([c_major_analysis])
        seq = enc.encode(c_major_analysis)
        decoded = enc.decode(seq)
        # Specials at boundaries.
        assert decoded["chord"][0] == "<bos>"
        assert decoded["chord"][-1] == "<eos>"
        # Body chord labels in the right order.
        assert decoded["chord"][1:5] == ["C:major", "F:major", "G:major", "C:major"]
        # Cadence renders CAD_NONE as "<none>" and the final cadence by name.
        assert decoded["cadence"][1] == "<none>"
        assert decoded["cadence"][4] == "authentic_perfect"


# ---------------------------------------------------------------------------
# TestFromAnalyses
# ---------------------------------------------------------------------------


class TestFromAnalyses:
    def test_from_analyses_builds_encoder_in_memory(self, c_major_analysis):
        enc = HarmonicEncoder.from_analyses([c_major_analysis])
        assert enc.vocabulary.chord_vocab_size > 4
        assert enc.vocabulary.rn_vocab_size > 4

    def test_from_analyses_respects_config(self):
        analysis = {
            "score_id": "cfg",
            "chords": [
                _chord(0, "major"), _chord(0, "major"),
                _chord(5, "major"),  # single occurrence — filtered
            ],
            "roman_numerals": [_rn("I"), _rn("I"), _rn("IV")],
            "cadences": [],
        }
        enc = HarmonicEncoder.from_analyses(
            [analysis],
            HarmonicEncoderConfig(min_chord_count=2, min_rn_count=1),
        )
        assert "F:major" not in enc.vocabulary.chord_to_id


# ---------------------------------------------------------------------------
# TestEncodeFile
# ---------------------------------------------------------------------------


class TestEncodeFile:
    def test_encode_file_reads_from_disk(self, tmp_path: Path, c_major_analysis):
        path = tmp_path / "toy.json"
        path.write_text(json.dumps(c_major_analysis))
        enc = HarmonicEncoder.from_cache_dir(tmp_path)
        seq = enc.encode_file(path)
        assert seq.score_id == "toy"
        assert seq.length == 6

    def test_from_cache_dir_ignores_manifest(
        self, tmp_path: Path, c_major_analysis
    ):
        (tmp_path / "manifest.json").write_text(json.dumps({
            "totals": {"ok": 1, "failed": 0, "skipped": 0, "total": 1},
            "entries": [],
        }))
        (tmp_path / "toy.json").write_text(json.dumps(c_major_analysis))
        enc = HarmonicEncoder.from_cache_dir(tmp_path)
        # Should have exactly the chord labels from the toy analysis — not
        # crash on the manifest.
        assert "C:major" in enc.vocabulary.chord_to_id

    def test_from_cache_dir_missing_raises(self, tmp_path: Path):
        missing = tmp_path / "no-such-dir"
        with pytest.raises(FileNotFoundError):
            HarmonicEncoder.from_cache_dir(missing)


# ---------------------------------------------------------------------------
# TestEncodedSequenceValidation
# ---------------------------------------------------------------------------


class TestEncodedSequenceValidation:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            EncodedSequence(
                score_id="x",
                chord_tokens=[BOS, EOS],
                rn_tokens=[BOS],                # wrong length
                cadence_tokens=[BOS, EOS],
                onsets=["", ""],
            )

    def test_to_dict_and_from_dict_roundtrip(self):
        seq = EncodedSequence(
            score_id="x",
            chord_tokens=[BOS, 5, EOS],
            rn_tokens=[BOS, 6, EOS],
            cadence_tokens=[BOS, CAD_NONE, EOS],
            onsets=["", "0", ""],
        )
        d = seq.to_dict()
        seq2 = EncodedSequence.from_dict(d)
        assert seq2 == seq
