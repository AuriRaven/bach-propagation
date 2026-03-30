"""
Evaluate the trained LSTM against a random baseline.

Usage:
    uv run python scripts/evaluate_baseline.py [OPTIONS]

Options:
    --checkpoint PATH    Trained model checkpoint (default: model_best.pt)
    --sequences PATH     Sequences JSON file (default: data/sequences.json)
    --device STR         'cpu' or 'cuda' (default: cpu)
    --batch-size N       DataLoader batch size (default: 32)

Output:
    A results table comparing the LSTM vs random baseline on:
    - Next-chord accuracy
    - Harmonic function accuracy
    - Transition validity rate
    - Perplexity

Example:
    uv run python scripts/evaluate_baseline.py --checkpoint model_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import build_datasets
from src.data.encoder import Encoder, TOTAL_VOCAB, PAD
from src.evaluation.metrics import (
    build_transition_matrix,
    evaluate_all,
    random_baseline,
)
from src.models.baseline_lstm import BaroqueHarmonyLSTM


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline LSTM")
    parser.add_argument('--checkpoint', default='model_best.pt', type=str)
    parser.add_argument('--sequences', default='data/sequences.json', type=Path)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--batch-size', default=32, type=int)
    args = parser.parse_args()

    # Load sequences
    with open(args.sequences) as f:
        raw_sequences = json.load(f)

    enc = Encoder()
    enc.build_chord_function_map(raw_sequences)
    encoded = [enc.encode_sequence(seq) for seq in raw_sequences]

    _, val_ds = build_datasets(encoded)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Load model
    model = BaroqueHarmonyLSTM()
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model = model.to(args.device)

    # Transition matrix from the full corpus (same domain as training data)
    print("Building transition matrix from sequences.json ...")
    transition_matrix = build_transition_matrix(encoded, TOTAL_VOCAB)
    print(f"  {len(transition_matrix)} transition pairs found.\n")

    # LSTM metrics
    print("Evaluating LSTM ...")
    lstm_results = evaluate_all(
        model=model,
        data_loader=val_loader,
        encoder=enc,
        transition_matrix=transition_matrix,
        pad_idx=PAD,
        device=args.device,
    )

    # Random baseline
    rand_results = random_baseline(
        vocab_size=TOTAL_VOCAB,
        chord_sequences=encoded,
        pad_idx=PAD,
    )

    # Print results table
    print()
    metrics = [
        ('next_chord_accuracy', 'Next-chord accuracy'),
        ('function_accuracy', 'Function accuracy'),
        ('transition_validity_rate', 'Transition validity'),
        ('perplexity', 'Perplexity'),
    ]
    col_w = 24
    print(f"{'Metric':<{col_w}} {'LSTM':>10} {'Random':>10}")
    print('-' * (col_w + 22))
    for key, label in metrics:
        lstm_val = lstm_results[key]
        rand_val = rand_results[key]
        if key == 'perplexity':
            print(f"{label:<{col_w}} {lstm_val:>10.2f} {rand_val:>10.2f}")
        else:
            print(f"{label:<{col_w}} {lstm_val:>9.1%} {rand_val:>9.1%}")


if __name__ == '__main__':
    main()
