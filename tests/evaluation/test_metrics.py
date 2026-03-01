"""
Tests for src/evaluation/metrics.py

TestNextChordAccuracy     — 100%, 0%, PAD masking
TestFunctionAccuracy      — function map usage
TestPerplexity            — float ≥ 1.0, ordering
TestBuildTransitionMatrix — probabilities, known transitions, empty
TestTransitionValidityRate — 100% for valid, 0% for impossible
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List
from unittest.mock import MagicMock

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import pytest

from src.data.encoder import Encoder, PAD, START, END, TOTAL_VOCAB
from src.data.dataset import HarmonyDataset
from src.models.baseline_lstm import BaroqueHarmonyLSTM
from src.evaluation.metrics import (
    build_transition_matrix,
    next_chord_accuracy,
    function_accuracy,
    perplexity,
    transition_validity_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_model() -> BaroqueHarmonyLSTM:
    return BaroqueHarmonyLSTM(vocab_size=TOTAL_VOCAB, embed_dim=8, hidden_dim=16, n_layers=1)


def _loader_from_tensors(inp: Tensor, tgt: Tensor, batch_size: int = 2) -> DataLoader:
    ds = TensorDataset(inp, tgt)
    return DataLoader(ds, batch_size=batch_size)


def _perfect_model(inp: Tensor, correct_tgt: Tensor) -> BaroqueHarmonyLSTM:
    """Return a model whose forward() always predicts correct_tgt (via argmax)."""
    model = _tiny_model()
    # Monkey-patch forward to return one-hot logits matching correct_tgt
    B, T = correct_tgt.shape
    C = TOTAL_VOCAB

    def _forward(x, hidden=None):
        logits = torch.full((x.shape[0], x.shape[1], C), -1e9)
        for b in range(x.shape[0]):
            for t in range(x.shape[1]):
                logits[b, t, correct_tgt[b, t].item()] = 1e9
        return logits, (torch.zeros(1, x.shape[0], model.hidden_dim),) * 2

    model.forward = _forward
    return model


# ---------------------------------------------------------------------------
# TestNextChordAccuracy
# ---------------------------------------------------------------------------

class TestNextChordAccuracy:

    def test_100_percent_when_predictions_equal_targets(self):
        B, T = 2, 8
        # Create targets with non-PAD tokens
        tgt = torch.randint(3, 30, (B, T), dtype=torch.long)
        inp = torch.ones(B, T, dtype=torch.long)  # dummy input
        model = _perfect_model(inp, tgt)
        loader = _loader_from_tensors(inp, tgt)
        acc = next_chord_accuracy(model, loader, pad_idx=PAD)
        assert abs(acc - 1.0) < 1e-6

    def test_0_percent_when_all_wrong(self):
        B, T = 2, 5
        tgt = torch.full((B, T), 5, dtype=torch.long)
        inp = torch.ones(B, T, dtype=torch.long)
        # Model always predicts token 6 (≠ 5)
        model = _tiny_model()
        wrong_tgt = torch.full((B, T), 6, dtype=torch.long)

        def _forward(x, hidden=None):
            logits = torch.full((x.shape[0], x.shape[1], TOTAL_VOCAB), -1e9)
            logits[:, :, 6] = 1e9
            return logits, (torch.zeros(1, x.shape[0], model.hidden_dim),) * 2

        model.forward = _forward
        loader = _loader_from_tensors(inp, tgt)
        acc = next_chord_accuracy(model, loader, pad_idx=PAD)
        assert acc == 0.0

    def test_pad_positions_ignored(self):
        B, T = 1, 4
        # Half the positions are PAD
        tgt = torch.tensor([[5, 6, PAD, PAD]], dtype=torch.long)
        inp = torch.ones(B, T, dtype=torch.long)
        # Perfect predictions for non-PAD; PAD positions return wrong token
        model = _tiny_model()

        def _forward(x, hidden=None):
            logits = torch.full((x.shape[0], x.shape[1], TOTAL_VOCAB), -1e9)
            logits[0, 0, 5] = 1e9   # correct
            logits[0, 1, 6] = 1e9   # correct
            logits[0, 2, 89] = 1e9  # wrong, but PAD → ignored
            logits[0, 3, 89] = 1e9  # wrong, but PAD → ignored
            return logits, (torch.zeros(1, x.shape[0], model.hidden_dim),) * 2

        model.forward = _forward
        loader = _loader_from_tensors(inp, tgt)
        acc = next_chord_accuracy(model, loader, pad_idx=PAD)
        assert abs(acc - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# TestFunctionAccuracy
# ---------------------------------------------------------------------------

class TestFunctionAccuracy:

    def _build_encoder_with_map(self) -> Encoder:
        enc = Encoder()
        # token 3 (C major) → function 0; token 42 (G dom7) → function 2
        seq = [[
            {'chord_root': 0, 'chord_quality': 'major', 'harmonic_function': 0},
            {'chord_root': 7, 'chord_quality': 'dominant7', 'harmonic_function': 2},
        ]]
        enc.build_chord_function_map(seq)
        return enc

    def test_correct_when_function_maps_match(self):
        enc = self._build_encoder_with_map()
        B, T = 1, 2
        # Target: [3 (T=0), 42 (D=2)]
        tgt = torch.tensor([[3, 42]], dtype=torch.long)
        inp = torch.ones(B, T, dtype=torch.long)
        model = _perfect_model(inp, tgt)  # predicts same tokens
        loader = _loader_from_tensors(inp, tgt)
        acc = function_accuracy(model, loader, enc, pad_idx=PAD)
        assert abs(acc - 1.0) < 1e-6

    def test_wrong_when_function_different(self):
        enc = self._build_encoder_with_map()
        B, T = 1, 1
        tgt = torch.tensor([[3]], dtype=torch.long)    # C major → function 0
        inp = torch.ones(B, T, dtype=torch.long)
        # Model predicts 42 (G dom7 → function 2) when target is 3 (function 0)
        model = _tiny_model()

        def _forward(x, hidden=None):
            logits = torch.full((x.shape[0], x.shape[1], TOTAL_VOCAB), -1e9)
            logits[:, :, 42] = 1e9
            return logits, (torch.zeros(1, x.shape[0], model.hidden_dim),) * 2

        model.forward = _forward
        loader = _loader_from_tensors(inp, tgt)
        acc = function_accuracy(model, loader, enc, pad_idx=PAD)
        assert acc == 0.0


# ---------------------------------------------------------------------------
# TestPerplexity
# ---------------------------------------------------------------------------

class TestPerplexity:

    def test_returns_float_ge_1(self):
        model = _tiny_model()
        B, T = 2, 8
        inp = torch.randint(3, 30, (B, T))
        tgt = torch.randint(3, 30, (B, T))
        loader = _loader_from_tensors(inp, tgt)
        ppl = perplexity(model, loader)
        assert isinstance(ppl, float)
        assert ppl >= 1.0

    def test_lower_perplexity_for_perfect_model(self):
        B, T = 2, 6
        tgt = torch.randint(3, 30, (B, T), dtype=torch.long)
        inp = torch.ones(B, T, dtype=torch.long)
        perfect = _perfect_model(inp, tgt)
        random_model = _tiny_model()
        loader = _loader_from_tensors(inp, tgt)
        ppl_perfect = perplexity(perfect, loader)
        ppl_random = perplexity(random_model, loader)
        assert ppl_perfect < ppl_random


# ---------------------------------------------------------------------------
# TestBuildTransitionMatrix
# ---------------------------------------------------------------------------

class TestBuildTransitionMatrix:

    def test_probabilities_sum_to_1_per_row(self):
        seqs = [[3, 10, 20, 3, 5, END]]
        matrix = build_transition_matrix(seqs)
        from_tokens = set(f for f, _ in matrix.keys())
        for tok in from_tokens:
            row_sum = sum(p for (f, t), p in matrix.items() if f == tok)
            assert abs(row_sum - 1.0) < 1e-9

    def test_known_transition_present(self):
        seqs = [[3, 10, 3, 10]]  # 3→10 twice, 10→3 twice
        matrix = build_transition_matrix(seqs)
        assert (3, 10) in matrix
        assert abs(matrix[(3, 10)] - 1.0) < 1e-9

    def test_empty_sequences_return_empty_dict(self):
        matrix = build_transition_matrix([])
        assert matrix == {}


# ---------------------------------------------------------------------------
# TestTransitionValidityRate
# ---------------------------------------------------------------------------

class TestTransitionValidityRate:

    def test_100_percent_for_known_valid_transitions(self):
        # Build a model that always predicts the same sequence 3→10→3→...
        B, T = 1, 4
        tgt = torch.tensor([[3, 10, 3, 10]], dtype=torch.long)
        inp = torch.ones(B, T, dtype=torch.long)

        model = _tiny_model()

        def _forward(x, hidden=None):
            logits = torch.full((x.shape[0], x.shape[1], TOTAL_VOCAB), -1e9)
            # Alternate predictions: 3, 10, 3, 10
            preds = [3, 10, 3, 10]
            for t, p in enumerate(preds):
                logits[0, t, p] = 1e9
            return logits, (torch.zeros(1, x.shape[0], model.hidden_dim),) * 2

        model.forward = _forward

        # Build transition matrix from same sequence
        matrix = build_transition_matrix([[3, 10, 3, 10, 3]])
        loader = _loader_from_tensors(inp, tgt)
        rate = transition_validity_rate(model, loader, matrix, pad_idx=PAD, threshold=0.01)
        assert abs(rate - 1.0) < 1e-6

    def test_0_percent_for_impossible_transitions(self):
        B, T = 1, 4
        tgt = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        inp = torch.ones(B, T, dtype=torch.long)

        model = _tiny_model()

        def _forward(x, hidden=None):
            logits = torch.full((x.shape[0], x.shape[1], TOTAL_VOCAB), -1e9)
            # Predict tokens 50, 51, 52, 53 — not in matrix at all
            for t, p in enumerate([50, 51, 52, 53]):
                logits[0, t, p] = 1e9
            return logits, (torch.zeros(1, x.shape[0], model.hidden_dim),) * 2

        model.forward = _forward

        # Matrix only contains 3→10 transition
        matrix = {(3, 10): 1.0}
        loader = _loader_from_tensors(inp, tgt)
        rate = transition_validity_rate(model, loader, matrix, pad_idx=PAD, threshold=0.01)
        assert rate == 0.0
