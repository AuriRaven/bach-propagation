#!/usr/bin/env python
"""
backend/scripts/evaluate_transformer.py

CLI evaluation script for the Music Transformer.
Run locally after training with caffeinate.

Usage:
    uv run python scripts/evaluate_transformer.py \\
        --checkpoint checkpoints/transformer-v1/best.pt \\
        --data-dir   data/encoded \\
        --n-samples  50 \\
        --output-report reports/evaluation_v1.json

Options:
    --checkpoint     Path to best.pt checkpoint (required)
    --data-dir       Directory containing vocab.json, encoded_sequences.jsonl,
                     transition_matrix.json (default: data/encoded)
    --n-samples      Number of sequences to generate for grammar evaluation
                     (default: 50)
    --key-mode       "minor" or "major" (default: minor)
    --output-report  Path for JSON report output (default: reports/evaluation.json)
    --device         Override device: mps / cuda / cpu (default: auto)
    --batch-size     Batch size for perplexity computation (default: 32)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure backend/src is importable when run from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Music Transformer")
    p.add_argument("--checkpoint",    required=True,
                   help="Path to best.pt")
    p.add_argument("--data-dir",      default="data/encoded",
                   help="Directory with vocab.json etc.")
    p.add_argument("--n-samples",     type=int, default=50,
                   help="Sequences to generate for grammar evaluation")
    p.add_argument("--key-mode",      default="minor",
                   choices=["minor", "major"],
                   help="Key mode for generation")
    p.add_argument("--output-report", default="reports/evaluation.json",
                   help="Output JSON report path")
    p.add_argument("--device",        default="auto",
                   help="Device override: mps / cuda / cpu / auto")
    p.add_argument("--batch-size",    type=int, default=32,
                   help="Batch size for perplexity computation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir   = Path(args.data_dir)
    vocab_path = str(data_dir / "vocab.json")
    matrix_path = str(data_dir / "transition_matrix.json")
    jsonl_path  = str(data_dir / "encoded_sequences.jsonl")

    # ── Validate inputs ───────────────────────────────────────────────────
    for p in (args.checkpoint, vocab_path, matrix_path, jsonl_path):
        if not Path(p).exists():
            print(f"ERROR: Required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Data directory:     {args.data_dir}")
    print(f"Key mode:           {args.key_mode}")
    print(f"Samples:            {args.n_samples}")
    print()

    # ── Resolve device ────────────────────────────────────────────────────
    import torch
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load generator ────────────────────────────────────────────────────
    from src.models.generator import MusicGenerator
    print("Loading generator from checkpoint…")
    generator = MusicGenerator.from_checkpoint(
        checkpoint_path=args.checkpoint,
        vocab_path=vocab_path,
        transition_matrix_path=matrix_path,
        device=str(device),
    )
    print("Generator loaded.\n")

    # ── Compute transformer perplexity ────────────────────────────────────
    from src.data.transformer_dataset import HarmonicSequenceDataset
    from src.models.music_transformer import MusicTransformer
    from src.evaluation.transformer_metrics import (
        compute_transformer_perplexity,
        compute_baseline_perplexity,
        evaluate_generator,
    )

    print("Computing perplexity on test set…")
    test_dataset = HarmonicSequenceDataset(
        jsonl_path=jsonl_path,
        vocab_path=vocab_path,
        split="test",
        window_size=128,
        stride=64,
    )
    model, _ = MusicTransformer.load_checkpoint(args.checkpoint, device=str(device))
    ppl = compute_transformer_perplexity(model, test_dataset, device, args.batch_size)
    print(f"Transformer perplexity: {ppl:.2f}")

    # ── Baseline perplexity ───────────────────────────────────────────────
    print("Computing baseline perplexity…")
    baseline_ppl = compute_baseline_perplexity(jsonl_path, vocab_path, device)
    print(f"Baseline perplexity:    {baseline_ppl:.2f}")

    # ── Grammar evaluation ────────────────────────────────────────────────
    print(f"\nGenerating {args.n_samples} sequences for grammar evaluation…")
    report = evaluate_generator(
        generator=generator,
        n_samples=args.n_samples,
        key_mode=args.key_mode,
        checkpoint_path=args.checkpoint,
    )

    # Fill in the perplexity values computed above
    import dataclasses
    report = dataclasses.replace(
        report,
        perplexity=ppl,
        baseline_perplexity=baseline_ppl,
        improvement_over_baseline=(
            (baseline_ppl - ppl) / baseline_ppl if ppl > 0 else 0.0
        ),
    )

    # ── Print and save ────────────────────────────────────────────────────
    report.print_summary()
    report.to_json(args.output_report)


if __name__ == "__main__":
    main()