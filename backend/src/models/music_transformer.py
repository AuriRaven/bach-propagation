"""
MusicTransformer — autoregressive next-chord predictor over three
aligned harmonic streams (chord / roman-numeral / cadence).

Architecture
------------
::

    (chord_ids, rn_ids, cadence_ids)
             │
             ▼
    MultiStreamEmbedding  ──► token embeddings (chord + rn + cadence)
                             + learned positional encoding
                             fused via concat + Linear → d_model
             │
             ▼
    N × MusicTransformerBlock
             ├── Pre-norm Multi-Head Causal Self-Attention
             └── Pre-norm Feed-Forward (d_ff = 4·d_model)
             │
             ▼
    LayerNorm → Linear(d_model, chord_vocab_size)
             │
             ▼
          logits (B, T, chord_vocab_size)

Design notes
------------
* **Three embedding tables, one fusion.**  Flat concatenation followed
  by a single linear projection back to ``d_model``. The model sees one
  dense vector per timestep; downstream layers are vanilla transformer
  blocks. Cheaper and more stable than multi-head cross-attention over
  the three streams separately, and the learned projection lets the
  model rebalance the streams rather than enforcing equal contribution.
* **Pre-LayerNorm.**  Puts the norm *before* attention / FFN instead of
  after the residual. Trains far more reliably on small M4-Pro-sized
  models without a warmup tuned to the model depth.
* **Causal mask.**  Built on the fly from ``torch.triu`` on each
  forward. Padding is handled via the optional ``attention_mask``:
  columns corresponding to PAD positions are masked out so the model
  never attends to padding.
* **Self-describing checkpoints.**  ``save_checkpoint`` persists both
  the state dict *and* the hyperparameters + vocab sizes used to build
  the module. ``load_checkpoint`` reconstructs the module from those
  fields — inference code never needs to know the original training
  command's hyperparameters.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Schema pin for checkpoints
# ---------------------------------------------------------------------------

CHECKPOINT_SCHEMA_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# MultiStreamEmbedding
# ---------------------------------------------------------------------------


class MultiStreamEmbedding(nn.Module):
    """Learned embeddings for three aligned streams + positional code.

    Each stream has its own ``nn.Embedding``; outputs are concatenated
    along the feature axis and fused through a single linear layer back
    to ``d_model``. A learned positional embedding is added after
    fusion so the transformer has absolute position signal.

    Args:
        chord_vocab_size: Number of distinct chord token ids (including
            specials).
        rn_vocab_size: Number of distinct Roman-numeral token ids.
        cadence_vocab_size: Number of distinct cadence token ids.
        d_model: Output feature dimension.
        max_seq_len: Maximum window length the model will ever see;
            positional embedding rows are pre-allocated to this size.
        dropout: Dropout probability applied to the fused embedding.
    """

    def __init__(
        self,
        chord_vocab_size: int,
        rn_vocab_size: int,
        cadence_vocab_size: int,
        d_model: int = 256,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Each stream gets a third of the fused space (rounded).
        per_stream = d_model // 3
        # Make sure the three per-stream dims sum *exactly* to something
        # we can project to d_model without leftover columns.
        self.chord_embed = nn.Embedding(chord_vocab_size, per_stream, padding_idx=0)
        self.rn_embed = nn.Embedding(rn_vocab_size, per_stream, padding_idx=0)
        self.cadence_embed = nn.Embedding(cadence_vocab_size, per_stream, padding_idx=0)

        fused_dim = per_stream * 3
        self.fuse = nn.Linear(fused_dim, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        chord_tokens: Tensor,
        rn_tokens: Tensor,
        cadence_tokens: Tensor,
    ) -> Tensor:
        """Return fused embeddings of shape ``(B, T, d_model)``."""
        B, T = chord_tokens.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len={self.max_seq_len}"
            )
        # (B, T, per_stream) each
        c = self.chord_embed(chord_tokens)
        r = self.rn_embed(rn_tokens)
        k = self.cadence_embed(cadence_tokens)
        fused = torch.cat([c, r, k], dim=-1)       # (B, T, 3*per_stream)
        fused = self.fuse(fused)                   # (B, T, d_model)

        positions = torch.arange(T, device=chord_tokens.device).unsqueeze(0)  # (1, T)
        fused = fused + self.pos_embed(positions)  # broadcast to (B, T, d_model)
        return self.dropout(fused)


# ---------------------------------------------------------------------------
# MusicTransformerBlock
# ---------------------------------------------------------------------------


class MusicTransformerBlock(nn.Module):
    """Pre-norm causal transformer block.

    Order: LN → self-attention → residual → LN → FFN → residual.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads

        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        causal_mask: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply one pre-norm causal block.

        Args:
            x: Input tensor of shape ``(B, T, d_model)``.
            causal_mask: Additive attention mask of shape ``(T, T)`` —
                ``0`` on the lower triangle (including the diagonal),
                ``-inf`` on the upper triangle.
            key_padding_mask: Optional mask of shape ``(B, T)`` where
                ``True`` marks positions to ignore (padding). Note the
                inversion relative to the dataset's ``attention_mask``,
                which is ``1`` for *real* tokens.
        """
        h = self.norm_attn(x)
        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)

        h = self.norm_ffn(x)
        x = x + self.dropout(self.ffn(h))
        return x


# ---------------------------------------------------------------------------
# MusicTransformer
# ---------------------------------------------------------------------------


class MusicTransformer(nn.Module):
    """Autoregressive multi-stream transformer for next-chord prediction.

    The head is a single ``Linear`` projection to the chord vocabulary —
    we predict chords only. Roman-numeral and cadence streams act as
    additional conditioning signals, not additional targets. Separate
    heads for the other streams are left as a future extension.
    """

    def __init__(
        self,
        chord_vocab_size: int,
        rn_vocab_size: int,
        cadence_vocab_size: int,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1536,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self._hparams: Dict[str, Any] = {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "dropout": dropout,
            "max_seq_len": max_seq_len,
        }
        self._vocab_config: Dict[str, int] = {
            "chord_vocab_size": chord_vocab_size,
            "rn_vocab_size": rn_vocab_size,
            "cadence_vocab_size": cadence_vocab_size,
        }
        self.chord_vocab_size = chord_vocab_size

        self.embedding = MultiStreamEmbedding(
            chord_vocab_size=chord_vocab_size,
            rn_vocab_size=rn_vocab_size,
            cadence_vocab_size=cadence_vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList([
            MusicTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, chord_vocab_size)

        # Share the chord-embedding weight with the output head. Slight
        # parameter saving and a mild inductive bias toward chord-
        # reconstruction coherence. Only valid when per-stream embedding
        # dim × 3 projects down to d_model — we use a separate projection
        # so weight tying is a simple identity on the chord-embed table.
        # Opting *out* of tying keeps the math simple and gives the head
        # full freedom; revisit if checkpoint size becomes a concern.

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        chord_tokens: Tensor,
        rn_tokens: Tensor,
        cadence_tokens: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict logits over ``chord_vocab_size`` at every position.

        Args:
            chord_tokens: (B, T) chord input ids.
            rn_tokens: (B, T) RN input ids.
            cadence_tokens: (B, T) cadence input ids.
            attention_mask: Optional (B, T) tensor; ``1`` at real
                positions, ``0`` at padding. Matches the dataset's
                convention. Internally flipped for PyTorch.

        Returns:
            Tensor of shape (B, T, chord_vocab_size).
        """
        B, T = chord_tokens.shape

        x = self.embedding(chord_tokens, rn_tokens, cadence_tokens)

        # Causal mask — (T, T) additive, -inf above the diagonal.
        causal_mask = _causal_mask(T, device=x.device, dtype=x.dtype)

        # key_padding_mask: -inf at padded positions, 0 elsewhere. We use
        # a float additive form (same dtype as causal_mask) because mixing
        # bool key_padding_mask with float attn_mask is deprecated in
        # recent PyTorch.
        key_padding_mask: Optional[Tensor] = None
        if attention_mask is not None:
            pad_positions = attention_mask == 0
            key_padding_mask = torch.zeros_like(pad_positions, dtype=x.dtype)
            key_padding_mask = key_padding_mask.masked_fill(
                pad_positions, float("-inf")
            )

        for block in self.blocks:
            x = block(x, causal_mask=causal_mask, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)
        return self.head(x)  # (B, T, chord_vocab_size)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Read-only view of constructor hyperparameters (for logging)."""
        return dict(self._hparams)

    @property
    def vocab_config(self) -> Dict[str, int]:
        """Read-only view of the three vocab sizes."""
        return dict(self._vocab_config)

    # ------------------------------------------------------------------
    # Construction from vocab
    # ------------------------------------------------------------------

    @classmethod
    def from_vocab(
        cls,
        vocab_path: str | Path,
        **kwargs: Any,
    ) -> "MusicTransformer":
        """Build a transformer sized to a vocab.json.

        The three vocabulary sizes are read from the file; all other
        hyperparameters fall through to the ``__init__`` defaults, or
        can be overridden via ``**kwargs``.
        """
        vocab = _read_vocab(Path(vocab_path))
        return cls(
            chord_vocab_size=len(vocab["chord_to_id"]),
            rn_vocab_size=len(vocab["rn_to_id"]),
            cadence_vocab_size=len(vocab["cadence_to_id"]),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a self-contained checkpoint.

        Layout::

            {
              "schema_version": "1.0.0",
              "model_state_dict": …,
              "hyperparameters": {d_model, n_heads, n_layers, d_ff,
                                  dropout, max_seq_len},
              "vocab_config":   {chord_vocab_size, rn_vocab_size,
                                 cadence_vocab_size},
              "extra":          {...}  # epoch, val_loss, …
            }
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": self.state_dict(),
            "hyperparameters": dict(self._hparams),
            "vocab_config": dict(self._vocab_config),
            "extra": dict(extra) if extra else {},
        }, out)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> Tuple["MusicTransformer", Dict[str, Any]]:
        """Rebuild a MusicTransformer from disk.

        Returns:
            (model, extra) tuple. ``extra`` is the dict saved alongside
            the weights (e.g. epoch / val_loss).
        """
        ckpt = torch.load(Path(path), map_location=device, weights_only=False)
        version = ckpt.get("schema_version")
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint {path} has schema_version={version!r}, "
                f"expected {CHECKPOINT_SCHEMA_VERSION!r}"
            )
        model = cls(
            **ckpt["vocab_config"],
            **ckpt["hyperparameters"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model, ckpt.get("extra", {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """(T, T) additive mask with -inf on the upper triangle.

    PyTorch's ``MultiheadAttention`` adds this to the pre-softmax
    attention weights, so positions j > i are driven to zero probability.
    """
    mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask


def _read_vocab(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "MultiStreamEmbedding",
    "MusicTransformer",
    "MusicTransformerBlock",
]
