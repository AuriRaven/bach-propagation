"""
Train :class:`src.models.music_transformer.MusicTransformer` on encoded
Bach-chorale sequences.

Usage::

    uv run python scripts/train_transformer.py \\
        --jsonl   data/encoded/encoded_sequences.jsonl \\
        --vocab   data/encoded/vocab.json \\
        --output-dir checkpoints/transformer-v1 \\
        --epochs 60 --batch-size 32 --lr 1e-4

The script:
  1. Builds three :class:`HarmonicSequenceDataset` splits
     (deterministic, by score_id hash) and wraps them in DataLoaders.
  2. Builds a :class:`MusicTransformer` sized to the vocab JSON via
     :meth:`MusicTransformer.from_vocab`.
  3. Hands everything to :class:`TransformerTrainer`, which runs the
     full loop — AdamW + warmup/cosine, CE loss, gradient clipping,
     best/last checkpoints, early stopping — and writes
     ``training_log.jsonl`` alongside the checkpoints.

Device selection follows MPS → CUDA → CPU.  Override with
``--device {mps|cuda|cpu}``.

Exit codes
----------
0    training completed normally (or stopped early)
2    missing / malformed input files
3    transformer failed to construct (e.g. vocab mismatch)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.transformer_dataset import HarmonicSequenceDataset
from src.models.music_transformer import MusicTransformer
from src.models.train_transformer import (
    TrainerConfig,
    TransformerTrainer,
    select_device,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the multi-stream Music Transformer.",
    )
    # Data
    p.add_argument(
        "--jsonl",
        type=Path,
        default=Path("data/encoded/encoded_sequences.jsonl"),
        help="Path to encoded_sequences.jsonl (Stage 5 output).",
    )
    p.add_argument(
        "--vocab",
        type=Path,
        default=Path("data/encoded/vocab.json"),
        help="Path to vocab.json (Stage 5 output).",
    )
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)

    # Model (override the defaults from MusicTransformer.__init__)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--d-ff", type=int, default=1536)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=512)

    # Training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--grammar-weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    # Output / device
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/transformer-v1"),
    )
    p.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        default=None,
        help="Override the MPS→CUDA→CPU auto-select.",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.jsonl.exists():
        print(f"error: --jsonl file not found: {args.jsonl}", file=sys.stderr)
        return 2
    if not args.vocab.exists():
        print(f"error: --vocab file not found: {args.vocab}", file=sys.stderr)
        return 2

    print(f"[train] jsonl={args.jsonl}")
    print(f"[train] vocab={args.vocab}")
    print(f"[train] output_dir={args.output_dir}")

    train_ds = HarmonicSequenceDataset(
        args.jsonl, args.vocab,
        split="train", window_size=args.window_size, stride=args.stride,
    )
    val_ds = HarmonicSequenceDataset(
        args.jsonl, args.vocab,
        split="val", window_size=args.window_size, stride=args.stride,
    )
    print(
        f"[train] train windows={len(train_ds)} "
        f"val windows={len(val_ds)} "
        f"train scores={len(train_ds.score_ids)} "
        f"val scores={len(val_ds.score_ids)}"
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("error: empty train or val split", file=sys.stderr)
        return 2

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    try:
        model = MusicTransformer.from_vocab(
            args.vocab,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
        )
    except Exception as e:  # pragma: no cover - defensive
        print(f"error: could not build MusicTransformer: {e}", file=sys.stderr)
        return 3

    print(f"[train] model params: {model.count_parameters():,}")

    config = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        grammar_weight=args.grammar_weight,
        seed=args.seed,
    )
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        config=config,
        device=args.device,
    )
    results = trainer.fit()

    print()
    print(f"[train] best epoch       = {results['best_epoch']}")
    print(f"[train] best val loss    = {results['best_val_loss']:.4f}")
    print(f"[train] best checkpoint  = {results['best_checkpoint']}")
    print(f"[train] last checkpoint  = {results['last_checkpoint']}")
    print(f"[train] stopped early    = {results['stopped_early']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
