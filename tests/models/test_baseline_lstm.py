"""
Tests for src/models/baseline_lstm.py

TestBaroqueHarmonyLSTM — output shapes, hidden state, trainability
"""

from __future__ import annotations

import torch
import pytest

from src.data.encoder import TOTAL_VOCAB, PAD
from src.models.baseline_lstm import BaroqueHarmonyLSTM


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model() -> BaroqueHarmonyLSTM:
    return BaroqueHarmonyLSTM(vocab_size=TOTAL_VOCAB, embed_dim=16, hidden_dim=32, n_layers=2)


# ---------------------------------------------------------------------------
# TestBaroqueHarmonyLSTM
# ---------------------------------------------------------------------------

class TestBaroqueHarmonyLSTM:

    def test_output_shape(self, model):
        B, T = 4, 10
        x = torch.randint(3, TOTAL_VOCAB, (B, T))
        logits, _ = model(x)
        assert logits.shape == (B, T, TOTAL_VOCAB)

    def test_hidden_state_shape(self, model):
        B, T = 3, 8
        x = torch.randint(3, TOTAL_VOCAB, (B, T))
        _, (h, c) = model(x)
        assert h.shape == (model.n_layers, B, model.hidden_dim)
        assert c.shape == (model.n_layers, B, model.hidden_dim)

    def test_batch_size_1(self, model):
        x = torch.randint(3, TOTAL_VOCAB, (1, 5))
        logits, _ = model(x)
        assert logits.shape == (1, 5, TOTAL_VOCAB)

    def test_seq_len_1(self, model):
        x = torch.randint(3, TOTAL_VOCAB, (4, 1))
        logits, _ = model(x)
        assert logits.shape == (4, 1, TOTAL_VOCAB)

    def test_padding_idx_has_zero_embedding(self, model):
        pad_emb = model.embedding(torch.tensor([PAD]))
        assert torch.all(pad_emb == 0)

    def test_init_hidden_shape(self, model):
        B = 5
        h0, c0 = model.init_hidden(B, device='cpu')
        assert h0.shape == (model.n_layers, B, model.hidden_dim)
        assert c0.shape == (model.n_layers, B, model.hidden_dim)

    def test_init_hidden_is_zeros(self, model):
        h0, c0 = model.init_hidden(2, device='cpu')
        assert torch.all(h0 == 0)
        assert torch.all(c0 == 0)

    def test_forward_with_explicit_hidden(self, model):
        B, T = 2, 6
        x = torch.randint(3, TOTAL_VOCAB, (B, T))
        h0, c0 = model.init_hidden(B)
        logits, (h1, c1) = model(x, (h0, c0))
        assert logits.shape == (B, T, TOTAL_VOCAB)
        assert h1.shape == h0.shape

    def test_parameters_are_trainable(self, model):
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
