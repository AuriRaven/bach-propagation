"""
Tests for :class:`src.data.transformer_dataset.HarmonicSequenceDataset`.

All tests run off a tiny synthetic JSONL (5 sequences) written to
``tmp_path`` — no real corpus file is touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest
import torch

from src.data.harmonic_encoder import BOS, CAD_NONE, EOS, PAD
from src.data.transformer_dataset import HarmonicSequenceDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seq(score_id: str, length: int) -> Dict[str, object]:
    """Build a deterministic synthetic sequence record.

    Pattern: [BOS, 5, 6, 7, ..., EOS] so the body has length `length-2`
    with every chord having a distinct id for traceability.
    """
    body_chord = list(range(5, 5 + max(length - 2, 0)))
    body_rn = list(range(10, 10 + max(length - 2, 0)))
    body_cad = [CAD_NONE] * max(length - 2, 0)
    return {
        "score_id": score_id,
        "chord_tokens":   [BOS, *body_chord, EOS],
        "rn_tokens":      [BOS, *body_rn, EOS],
        "cadence_tokens": [BOS, *body_cad, EOS],
        "onsets":         [""] + [str(i) for i in range(len(body_chord))] + [""],
    }


@pytest.fixture
def synthetic_jsonl(tmp_path: Path) -> Path:
    """Five sequences of length 20 at distinct score_ids."""
    path = tmp_path / "encoded_sequences.jsonl"
    records = [_seq(f"bwv{i}", 20) for i in range(5)]
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


@pytest.fixture
def vocab_file(tmp_path: Path) -> Path:
    """Minimal vocab.json — the dataset only reads the PAD id from it."""
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({
        "schema_version": "1.0.0",
        "specials": {"PAD": 0, "BOS": 1, "EOS": 2, "UNK": 3, "CAD_NONE": 4},
        "chord_to_id": {}, "rn_to_id": {}, "cadence_to_id": {},
    }))
    return path


# ---------------------------------------------------------------------------
# TestBasicShape
# ---------------------------------------------------------------------------


class TestBasicShape:
    def test_getitem_returns_expected_keys(self, synthetic_jsonl, vocab_file):
        ds = HarmonicSequenceDataset(
            synthetic_jsonl, vocab_file,
            split="train", window_size=16, stride=8,
        )
        # Skip if train split happens to be empty for this hash set.
        if len(ds) == 0:
            pytest.skip("train split was empty for this synthetic set")
        item = ds[0]
        assert set(item.keys()) == {
            "chord_input", "rn_input", "cadence_input",
            "chord_target", "attention_mask",
        }

    def test_all_tensors_have_correct_shape_and_dtype(
        self, synthetic_jsonl, vocab_file
    ):
        W = 16
        ds = HarmonicSequenceDataset(
            synthetic_jsonl, vocab_file,
            split="train", window_size=W, stride=W,
        )
        if len(ds) == 0:
            pytest.skip("train split empty")
        item = ds[0]
        for k, v in item.items():
            assert isinstance(v, torch.Tensor), k
            assert v.dtype == torch.long, k
            assert v.shape == (W,), f"{k} has shape {v.shape}, expected ({W},)"


# ---------------------------------------------------------------------------
# TestNextTokenAlignment
# ---------------------------------------------------------------------------


class TestNextTokenAlignment:
    def test_target_is_input_shifted_by_one(self, synthetic_jsonl, vocab_file):
        """For any window whose input doesn't end in PAD, target[t] must
        equal the token that *follows* input[t] in the original
        sequence — i.e. target[t] == source[start + t + 1]."""
        ds = HarmonicSequenceDataset(
            synthetic_jsonl, vocab_file,
            split="train", window_size=8, stride=8,
        )
        if len(ds) == 0:
            pytest.skip("train split empty")
        for idx in range(len(ds)):
            item = ds[idx]
            ci = item["chord_input"].tolist()
            ct = item["chord_target"].tolist()
            # Wherever there was a valid "next" token (i.e. we weren't at
            # the tail of the sequence), target should equal the next
            # input-position token.
            for t in range(len(ci) - 1):
                if ci[t + 1] != PAD:
                    assert ct[t] == ci[t + 1], (
                        f"misaligned at idx={idx} t={t}: "
                        f"input={ci[t]}, target={ct[t]}, next_input={ci[t+1]}"
                    )


# ---------------------------------------------------------------------------
# TestAttentionMask
# ---------------------------------------------------------------------------


class TestAttentionMask:
    def test_mask_is_one_for_real_and_zero_for_pad(self, tmp_path, vocab_file):
        # Short sequence (length 6 incl. BOS/EOS) with a long window
        # forces padding we can verify.
        jsonl = tmp_path / "short.jsonl"
        jsonl.write_text(json.dumps(_seq("bwvSHORT", 6)) + "\n")

        ds = HarmonicSequenceDataset(
            jsonl, vocab_file,
            # Override to make sure the single short score lands in this split.
            split=_which_split("bwvSHORT"),
            window_size=16, stride=8,
        )
        assert len(ds) == 1
        item = ds[0]
        ci = item["chord_input"].tolist()
        am = item["attention_mask"].tolist()
        for t, tok in enumerate(ci):
            expected = 1 if tok != PAD else 0
            assert am[t] == expected, (
                f"mask mismatch at t={t}: token={tok}, mask={am[t]}"
            )

    def test_mask_all_ones_when_no_padding_needed(
        self, synthetic_jsonl, vocab_file
    ):
        ds = HarmonicSequenceDataset(
            synthetic_jsonl, vocab_file,
            split="train", window_size=8, stride=8,
        )
        if len(ds) == 0:
            pytest.skip("train split empty")
        for idx in range(len(ds)):
            item = ds[idx]
            ci = item["chord_input"].tolist()
            if all(tok != PAD for tok in ci):
                assert all(m == 1 for m in item["attention_mask"].tolist())


# ---------------------------------------------------------------------------
# TestLeakageAndSplits
# ---------------------------------------------------------------------------


class TestLeakageAndSplits:
    def test_no_score_id_appears_in_two_splits(self, tmp_path, vocab_file):
        # Use 30 distinct score_ids — enough to populate all three buckets.
        jsonl = tmp_path / "many.jsonl"
        with jsonl.open("w", encoding="utf-8") as f:
            for i in range(30):
                f.write(json.dumps(_seq(f"bwv{i}.6", 20)) + "\n")

        train = HarmonicSequenceDataset(jsonl, vocab_file, split="train", window_size=8)
        val = HarmonicSequenceDataset(jsonl, vocab_file, split="val",   window_size=8)
        test = HarmonicSequenceDataset(jsonl, vocab_file, split="test",  window_size=8)

        sets = [set(train.score_ids), set(val.score_ids), set(test.score_ids)]
        for a in range(3):
            for b in range(a + 1, 3):
                assert sets[a].isdisjoint(sets[b]), (
                    f"split leakage: {sets[a] & sets[b]}"
                )

    def test_splits_partition_total_scores(self, tmp_path, vocab_file):
        jsonl = tmp_path / "many.jsonl"
        score_ids = [f"bwv{i}.6" for i in range(30)]
        with jsonl.open("w", encoding="utf-8") as f:
            for s in score_ids:
                f.write(json.dumps(_seq(s, 20)) + "\n")

        seen = set()
        for split in ("train", "val", "test"):
            ds = HarmonicSequenceDataset(jsonl, vocab_file, split=split, window_size=8)
            seen.update(ds.score_ids)
        assert seen == set(score_ids)

    def test_split_is_deterministic(self, synthetic_jsonl, vocab_file):
        a = HarmonicSequenceDataset(
            synthetic_jsonl, vocab_file, split="train", window_size=8
        )
        b = HarmonicSequenceDataset(
            synthetic_jsonl, vocab_file, split="train", window_size=8
        )
        assert a.score_ids == b.score_ids

    def test_invalid_split_rejected(self, synthetic_jsonl, vocab_file):
        with pytest.raises(ValueError, match="split"):
            HarmonicSequenceDataset(
                synthetic_jsonl, vocab_file, split="holdout", window_size=8
            )


# ---------------------------------------------------------------------------
# TestWindowing
# ---------------------------------------------------------------------------


class TestWindowing:
    def test_stride_produces_expected_window_count(self, tmp_path, vocab_file):
        # Sequence length 21 (19 body + BOS + EOS). With W=8 we need 9
        # raw tokens per window, so valid starts are 0 .. 12 step stride.
        jsonl = tmp_path / "long.jsonl"
        jsonl.write_text(json.dumps(_seq("bwvLONG", 21)) + "\n")
        split = _which_split("bwvLONG")

        ds = HarmonicSequenceDataset(
            jsonl, vocab_file, split=split, window_size=8, stride=4,
        )
        # Starts: 0, 4, 8, 12 → 4 windows.
        assert len(ds) == 4

    def test_short_sequence_produces_one_padded_window(
        self, tmp_path, vocab_file
    ):
        jsonl = tmp_path / "tiny.jsonl"
        jsonl.write_text(json.dumps(_seq("bwvTINY", 5)) + "\n")
        split = _which_split("bwvTINY")

        ds = HarmonicSequenceDataset(
            jsonl, vocab_file, split=split, window_size=16, stride=8,
        )
        assert len(ds) == 1
        item = ds[0]
        ci = item["chord_input"].tolist()
        # Head is real (BOS + body + EOS = 5 tokens; input drops the last
        # target alignment so first 4 are real), rest must be PAD.
        assert any(tok == PAD for tok in ci)

    def test_rejects_window_size_of_one(self, synthetic_jsonl, vocab_file):
        with pytest.raises(ValueError, match="window_size"):
            HarmonicSequenceDataset(
                synthetic_jsonl, vocab_file, split="train", window_size=1
            )

    def test_rejects_non_positive_stride(self, synthetic_jsonl, vocab_file):
        with pytest.raises(ValueError, match="stride"):
            HarmonicSequenceDataset(
                synthetic_jsonl, vocab_file,
                split="train", window_size=8, stride=0,
            )

    def test_one_chord_sequence_is_dropped(self, tmp_path, vocab_file):
        """A sequence with fewer than 2 timesteps has nothing to predict
        and must be excluded — otherwise shift-by-one slicing blows up."""
        jsonl = tmp_path / "useless.jsonl"
        jsonl.write_text(json.dumps({
            "score_id": "bwvNULL",
            "chord_tokens":   [BOS],
            "rn_tokens":      [BOS],
            "cadence_tokens": [BOS],
            "onsets":         [""],
        }) + "\n")
        split = _which_split("bwvNULL")
        ds = HarmonicSequenceDataset(
            jsonl, vocab_file, split=split, window_size=8
        )
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# TestBosEosPlacement
# ---------------------------------------------------------------------------


class TestBosEosPlacement:
    def test_first_window_starts_with_bos(self, tmp_path, vocab_file):
        jsonl = tmp_path / "one.jsonl"
        jsonl.write_text(json.dumps(_seq("bwvONLY", 20)) + "\n")
        split = _which_split("bwvONLY")
        ds = HarmonicSequenceDataset(
            jsonl, vocab_file, split=split, window_size=8, stride=8,
        )
        assert len(ds) >= 1
        item = ds[0]
        assert item["chord_input"][0].item() == BOS
        assert item["rn_input"][0].item() == BOS
        assert item["cadence_input"][0].item() == BOS

    def test_eos_appears_in_final_window(self, tmp_path, vocab_file):
        jsonl = tmp_path / "one.jsonl"
        jsonl.write_text(json.dumps(_seq("bwvONLY", 20)) + "\n")
        split = _which_split("bwvONLY")
        ds = HarmonicSequenceDataset(
            jsonl, vocab_file, split=split, window_size=32, stride=16,
        )
        assert len(ds) >= 1
        # Combine the chord_target tokens across all windows (because
        # EOS sits at the tail and targets are shifted by one).
        all_targets = set()
        for i in range(len(ds)):
            all_targets.update(ds[i]["chord_target"].tolist())
        assert EOS in all_targets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _which_split(score_id: str) -> str:
    """Helper: return the split bucket for a score_id so single-score
    fixtures can pick the right split to instantiate."""
    from src.data.transformer_dataset import _bucket_of  # noqa
    return _bucket_of(score_id)
