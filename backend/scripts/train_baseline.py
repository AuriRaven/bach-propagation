"""
Train the baseline LSTM on extracted harmonic sequences.

Usage:
    uv run python scripts/train_baseline.py [OPTIONS]

Options:
    --sequences PATH     Input JSON file of sequences (default: data/sequences.json)
    --epochs N           Training epochs (default: 30)
    --lr FLOAT           Learning rate (default: 0.001)
    --batch-size N       Batch size (default: 32)
    --patience N         Early stopping patience (default: 5)
    --checkpoint PATH    Save best model here (default: model_best.pt)
    --device STR         'cpu' or 'cuda' (default: cpu)

Example:
    uv run python scripts/train_baseline.py --epochs 5  # quick smoke test
    uv run python scripts/train_baseline.py --epochs 30 --checkpoint model_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_datasets
from src.data.encoder import Encoder
from src.models.baseline_lstm import BaroqueHarmonyLSTM
from src.models.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline LSTM on harmonic sequences")
    parser.add_argument('--sequences', default='data/sequences.json', type=Path)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--checkpoint', default='model_best.pt', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    args = parser.parse_args()

    # Load sequences
    print(f"Loading sequences from {args.sequences} ...")
    with open(args.sequences) as f:
        raw_sequences = json.load(f)
    print(f"  {len(raw_sequences)} sequences loaded.")

    # Encode
    enc = Encoder()
    enc.build_chord_function_map(raw_sequences)
    encoded = [enc.encode_sequence(seq) for seq in raw_sequences]

    # Build datasets
    train_ds, val_ds = build_datasets(encoded)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = BaroqueHarmonyLSTM()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: BaroqueHarmonyLSTM  ({n_params:,} parameters)")
    print(f"Training for up to {args.epochs} epochs on {args.device} ...")
    print()

    results = train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        checkpoint_path=args.checkpoint,
        verbose=True,
    )

    print()
    print(f"Best epoch: {results['best_epoch']}  |  Best val loss: {results['best_val_loss']:.4f}")
    print(f"Checkpoint saved to: {args.checkpoint}")


if __name__ == '__main__':
    main()
