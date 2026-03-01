"""
Tests for src/data/dataset.py

TestHarmonyDataset  — length, getitem, padding, truncation
TestBuildDatasets   — split sizes, no overlap, reproducibility
"""

from __future__ import annotations

import torch
import pytest

from src.data.dataset import HarmonyDataset, build_datasets
from src.data.encoder import START, END, PAD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seq(n: int = 8) -> list[int]:
    """Simple token sequence: START + body + END."""
    return [START] + list(range(3, 3 + n)) + [END]


# ---------------------------------------------------------------------------
# TestHarmonyDataset
# ---------------------------------------------------------------------------

class TestHarmonyDataset:

    def test_length_matches_n_sequences(self):
        seqs = [_seq(4), _seq(6), _seq(3)]
        ds = HarmonyDataset(seqs, max_len=32)
        assert len(ds) == 3

    def test_getitem_returns_tuple_of_tensors(self):
        ds = HarmonyDataset([_seq(4)], max_len=32)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        inp, tgt = item
        assert isinstance(inp, torch.Tensor)
        assert isinstance(tgt, torch.Tensor)

    def test_input_target_same_shape(self):
        ds = HarmonyDataset([_seq(4)], max_len=32)
        inp, tgt = ds[0]
        assert inp.shape == tgt.shape

    def test_shape_is_max_len_minus_1(self):
        max_len = 16
        ds = HarmonyDataset([_seq(4)], max_len=max_len)
        inp, tgt = ds[0]
        assert inp.shape == (max_len - 1,)
        assert tgt.shape == (max_len - 1,)

    def test_first_input_is_start_token(self):
        ds = HarmonyDataset([_seq(4)], max_len=32)
        inp, _ = ds[0]
        assert inp[0].item() == START

    def test_last_non_pad_target_is_end_token(self):
        seq = _seq(4)  # START + 4 tokens + END
        ds = HarmonyDataset([seq], max_len=32)
        _, tgt = ds[0]
        non_pad = [t for t in tgt.tolist() if t != PAD]
        assert non_pad[-1] == END

    def test_padding_fills_to_max_len_minus_1(self):
        short_seq = [START, 10, END]  # length 3
        ds = HarmonyDataset([short_seq], max_len=32)
        inp, tgt = ds[0]
        # inp = [START, 10] + 29 PADs → total 31
        pad_count_inp = (inp == PAD).sum().item()
        assert pad_count_inp == 31 - 2  # 31 - len([START, 10])

    def test_long_sequence_is_truncated_to_max_len(self):
        long_seq = [START] + list(range(3, 3 + 300)) + [END]
        ds = HarmonyDataset([long_seq], max_len=32)
        inp, tgt = ds[0]
        assert inp.shape == (31,)  # max_len - 1

    def test_dtype_is_long(self):
        ds = HarmonyDataset([_seq(4)], max_len=16)
        inp, tgt = ds[0]
        assert inp.dtype == torch.long
        assert tgt.dtype == torch.long


# ---------------------------------------------------------------------------
# TestBuildDatasets
# ---------------------------------------------------------------------------

class TestBuildDatasets:

    def test_split_sizes_correct(self):
        seqs = [_seq(4) for _ in range(10)]
        train_ds, val_ds = build_datasets(seqs, train_frac=0.8)
        assert len(train_ds) == 8
        assert len(val_ds) == 2

    def test_no_overlap_between_train_and_val(self):
        seqs = [[START, 3 + i, END] for i in range(20)]
        train_ds, val_ds = build_datasets(seqs, train_frac=0.8)
        # Collect second body token (unique per sequence) from each split
        train_tokens = set()
        for i in range(len(train_ds)):
            inp, _ = train_ds[i]
            train_tokens.add(inp[1].item())  # body token after START
        val_tokens = set()
        for i in range(len(val_ds)):
            inp, _ = val_ds[i]
            val_tokens.add(inp[1].item())
        assert train_tokens.isdisjoint(val_tokens)

    def test_seed_reproducible(self):
        seqs = [_seq(4) for _ in range(20)]
        train1, val1 = build_datasets(seqs, seed=42)
        train2, val2 = build_datasets(seqs, seed=42)
        for i in range(len(train1)):
            inp1, _ = train1[i]
            inp2, _ = train2[i]
            assert torch.equal(inp1, inp2)
