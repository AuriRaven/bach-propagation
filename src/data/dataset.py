"""
PyTorch Dataset and DataLoader utilities for chord sequence modelling.

HarmonyDataset  — wraps encoded token sequences for next-token prediction
build_datasets  — shuffle + 80/20 train/val split
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.encoder import PAD


class HarmonyDataset(Dataset):
    """
    PyTorch Dataset for next-token prediction on chord sequences.

    Each item is ``(input_ids, target_ids)`` where:
    - ``input_ids = seq[:-1]``  padded to ``max_len - 1``
    - ``target_ids = seq[1:]``  padded to ``max_len - 1``

    Both tensors are ``torch.LongTensor`` of shape ``(max_len - 1,)``.

    Args:
        encoded_sequences: List of integer token lists produced by Encoder.
        max_len:           Maximum usable sequence length (including START/END).
                           Sequences longer than max_len are truncated to max_len.
        pad_idx:           Token used for padding (default: PAD = 0).
    """

    def __init__(
        self,
        encoded_sequences: List[List[int]],
        max_len: int = 256,
        pad_idx: int = PAD,
    ) -> None:
        self._max_len = max_len
        self._pad_idx = pad_idx
        self._sequences: List[List[int]] = [
            seq[:max_len] for seq in encoded_sequences
        ]

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        seq = self._sequences[idx]
        target_len = self._max_len - 1

        # input = seq[:-1], target = seq[1:]
        inp = seq[:-1]
        tgt = seq[1:]

        # Pad to target_len
        inp_padded = inp + [self._pad_idx] * max(0, target_len - len(inp))
        tgt_padded = tgt + [self._pad_idx] * max(0, target_len - len(tgt))

        return (
            torch.tensor(inp_padded[:target_len], dtype=torch.long),
            torch.tensor(tgt_padded[:target_len], dtype=torch.long),
        )


def build_datasets(
    sequences: List[List[int]],
    train_frac: float = 0.8,
    max_len: int = 256,
    pad_idx: int = PAD,
    seed: int = 42,
) -> Tuple[HarmonyDataset, HarmonyDataset]:
    """
    Shuffle and split encoded sequences into train / validation datasets.

    Args:
        sequences:   List of integer token lists.
        train_frac:  Fraction of sequences for training (default 0.8).
        max_len:     Passed through to HarmonyDataset.
        pad_idx:     Passed through to HarmonyDataset.
        seed:        Random seed for reproducibility.

    Returns:
        (train_dataset, val_dataset)
    """
    rng = random.Random(seed)
    shuffled = list(sequences)
    rng.shuffle(shuffled)

    split = max(1, int(len(shuffled) * train_frac))
    train_seqs = shuffled[:split]
    val_seqs = shuffled[split:]

    return (
        HarmonyDataset(train_seqs, max_len=max_len, pad_idx=pad_idx),
        HarmonyDataset(val_seqs, max_len=max_len, pad_idx=pad_idx),
    )
