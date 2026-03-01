"""
Baseline LSTM for next-chord prediction.

Architecture
------------
    Embedding(vocab_size, embed_dim, padding_idx=0)
        → 2-layer LSTM(embed_dim, hidden_dim, dropout)
        → Linear(hidden_dim, vocab_size)

Input:  (batch, seq_len) LongTensor of chord tokens
Output: (batch, seq_len, vocab_size) raw logit tensor

Use ``nn.CrossEntropyLoss(ignore_index=0)`` for training (no LogSoftmax here).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.data.encoder import TOTAL_VOCAB


class BaroqueHarmonyLSTM(nn.Module):
    """
    2-layer LSTM chord sequence model.

    Args:
        vocab_size:  Total vocabulary size (default: TOTAL_VOCAB = 90).
        embed_dim:   Token embedding dimension (default: 64).
        hidden_dim:  LSTM hidden state dimension (default: 128).
        n_layers:    Number of LSTM layers (default: 2).
        dropout:     Dropout probability between LSTM layers (default: 0.2).
    """

    def __init__(
        self,
        vocab_size: int = TOTAL_VOCAB,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass.

        Args:
            x:      (batch, seq_len) token ids.
            hidden: Optional (h0, c0) initial state.

        Returns:
            logits: (batch, seq_len, vocab_size) raw logits.
            hidden: updated LSTM hidden state tuple.
        """
        embedded = self.embedding(x)                    # (B, T, embed_dim)
        output, hidden = self.lstm(embedded, hidden)    # (B, T, hidden_dim)
        logits = self.fc(output)                        # (B, T, vocab_size)
        return logits, hidden

    def init_hidden(
        self,
        batch_size: int,
        device: str = 'cpu',
    ) -> Tuple[Tensor, Tensor]:
        """
        Return zero-initialised (h0, c0) for the given batch size.

        Returns:
            h0: (n_layers, batch_size, hidden_dim)
            c0: (n_layers, batch_size, hidden_dim)
        """
        zeros = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return zeros, zeros.clone()
