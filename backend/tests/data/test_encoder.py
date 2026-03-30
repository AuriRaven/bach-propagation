"""
Tests for src/data/encoder.py

TestChordToken      — chord_to_token / token_to_chord
TestFunctionToken   — function_to_token / token_to_function
TestKeyToken        — key_to_token / token_to_key
TestEncodeSequence  — encode_sequence / encode_functions
TestChordFunctionMap — build_chord_function_map / token_function
"""

from __future__ import annotations

import pytest

from src.data.encoder import (
    CHORD_OFFSET,
    FUNCTION_OFFSET,
    KEY_OFFSET,
    START,
    END,
    TOTAL_VOCAB,
    Encoder,
)


# ---------------------------------------------------------------------------
# TestChordToken
# ---------------------------------------------------------------------------

class TestChordToken:

    def test_c_major_is_chord_offset(self):
        enc = Encoder()
        # root=0, quality='major' → quality_idx=0 → CHORD_OFFSET + 0*5 + 0
        assert enc.chord_to_token(0, 'major') == CHORD_OFFSET

    def test_g_dominant7_correct_token(self):
        enc = Encoder()
        # root=7, quality='dominant7' → quality_idx=4 → CHORD_OFFSET + 7*5 + 4 = 3+39 = 42
        expected = CHORD_OFFSET + 7 * 5 + 4
        assert enc.chord_to_token(7, 'dominant7') == expected

    def test_round_trip_decode(self):
        enc = Encoder()
        for root in range(12):
            for quality in ['major', 'minor', 'diminished', 'augmented', 'dominant7']:
                tok = enc.chord_to_token(root, quality)
                decoded_root, decoded_quality = enc.token_to_chord(tok)
                assert decoded_root == root
                assert decoded_quality == quality

    def test_unknown_quality_maps_to_major_idx(self):
        enc = Encoder()
        # unknown quality → idx 0 (major)
        tok = enc.chord_to_token(3, 'unknown_quality')
        _, decoded_quality = enc.token_to_chord(tok)
        assert decoded_quality == 'major'

    def test_all_60_chord_tokens_unique(self):
        enc = Encoder()
        tokens = set()
        for root in range(12):
            for quality in ['major', 'minor', 'diminished', 'augmented', 'dominant7']:
                tokens.add(enc.chord_to_token(root, quality))
        assert len(tokens) == 60

    def test_non_canonical_quality_maps_to_canonical(self):
        enc = Encoder()
        # dim7 → idx 2 (same as diminished)
        tok_dim7 = enc.chord_to_token(0, 'dim7')
        tok_dim = enc.chord_to_token(0, 'diminished')
        assert tok_dim7 == tok_dim

    def test_major7_maps_to_major_idx(self):
        enc = Encoder()
        tok_maj7 = enc.chord_to_token(0, 'major7')
        tok_maj = enc.chord_to_token(0, 'major')
        assert tok_maj7 == tok_maj

    def test_token_to_chord_invalid_token_raises(self):
        enc = Encoder()
        with pytest.raises(ValueError):
            enc.token_to_chord(0)  # PAD is not a chord token


# ---------------------------------------------------------------------------
# TestFunctionToken
# ---------------------------------------------------------------------------

class TestFunctionToken:

    def test_tonic_maps_to_function_offset(self):
        enc = Encoder()
        assert enc.function_to_token(0) == FUNCTION_OFFSET

    def test_predominant_maps_to_function_offset_plus_1(self):
        enc = Encoder()
        assert enc.function_to_token(1) == FUNCTION_OFFSET + 1

    def test_dominant_maps_to_function_offset_plus_2(self):
        enc = Encoder()
        assert enc.function_to_token(2) == FUNCTION_OFFSET + 2

    def test_round_trip(self):
        enc = Encoder()
        for fn in range(3):
            assert enc.token_to_function(enc.function_to_token(fn)) == fn


# ---------------------------------------------------------------------------
# TestKeyToken
# ---------------------------------------------------------------------------

class TestKeyToken:

    def test_c_major_is_key_offset(self):
        enc = Encoder()
        assert enc.key_to_token(0, 'major') == KEY_OFFSET

    def test_c_minor_is_key_offset_plus_1(self):
        enc = Encoder()
        assert enc.key_to_token(0, 'minor') == KEY_OFFSET + 1

    def test_b_minor_is_last_key_token(self):
        enc = Encoder()
        # tonic=11, mode='minor' → KEY_OFFSET + 11*2 + 1 = 66 + 23 = 89
        assert enc.key_to_token(11, 'minor') == 89

    def test_round_trip(self):
        enc = Encoder()
        for tonic in range(12):
            for mode in ['major', 'minor']:
                tok = enc.key_to_token(tonic, mode)
                decoded_tonic, decoded_mode = enc.token_to_key(tok)
                assert decoded_tonic == tonic
                assert decoded_mode == mode


# ---------------------------------------------------------------------------
# TestEncodeSequence
# ---------------------------------------------------------------------------

def _make_event(root: int, quality: str = 'major', fn: int = 0) -> dict:
    return {
        'chord_root': root,
        'chord_quality': quality,
        'harmonic_function': fn,
    }


class TestEncodeSequence:

    def test_starts_with_start_token(self):
        enc = Encoder()
        tokens = enc.encode_sequence([_make_event(0)])
        assert tokens[0] == START

    def test_ends_with_end_token(self):
        enc = Encoder()
        tokens = enc.encode_sequence([_make_event(0)])
        assert tokens[-1] == END

    def test_length_is_seq_plus_2(self):
        enc = Encoder()
        seq = [_make_event(i) for i in range(5)]
        tokens = enc.encode_sequence(seq)
        assert len(tokens) == 7  # START + 5 chords + END

    def test_empty_sequence_gives_start_end(self):
        enc = Encoder()
        tokens = enc.encode_sequence([])
        assert tokens == [START, END]

    def test_body_tokens_match_chord_tokens(self):
        enc = Encoder()
        seq = [_make_event(0, 'major'), _make_event(7, 'dominant7')]
        tokens = enc.encode_sequence(seq)
        expected_body = [
            enc.chord_to_token(0, 'major'),
            enc.chord_to_token(7, 'dominant7'),
        ]
        assert tokens[1:-1] == expected_body

    def test_encode_functions_no_start_end(self):
        enc = Encoder()
        seq = [_make_event(0, fn=0), _make_event(4, fn=1), _make_event(7, fn=2)]
        fns = enc.encode_functions(seq)
        assert fns == [0, 1, 2]

    def test_encode_functions_length_matches_seq(self):
        enc = Encoder()
        seq = [_make_event(i % 12) for i in range(8)]
        fns = enc.encode_functions(seq)
        assert len(fns) == 8


# ---------------------------------------------------------------------------
# TestChordFunctionMap
# ---------------------------------------------------------------------------

class TestChordFunctionMap:

    def test_majority_vote_correct(self):
        enc = Encoder()
        # C major → function 0 (x3), function 1 (x1) → majority 0
        seq = [
            [_make_event(0, fn=0), _make_event(0, fn=0), _make_event(0, fn=0),
             _make_event(0, fn=1)]
        ]
        mapping = enc.build_chord_function_map(seq)
        tok = enc.chord_to_token(0, 'major')
        assert mapping[tok] == 0

    def test_token_function_uses_map(self):
        enc = Encoder()
        seq = [[_make_event(7, 'dominant7', fn=2)]]
        enc.build_chord_function_map(seq)
        tok = enc.chord_to_token(7, 'dominant7')
        assert enc.token_function(tok) == 2

    def test_unknown_token_falls_back_to_0(self):
        enc = Encoder()
        # no build_chord_function_map called
        assert enc.token_function(enc.chord_to_token(5, 'minor')) == 0
