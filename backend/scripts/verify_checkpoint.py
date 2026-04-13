"""
Sanity-check a saved MusicTransformer checkpoint.

Usage::

    uv run python scripts/verify_checkpoint.py PATH/TO/best.pt

Reads the schema version, hyperparameters, vocab config, and any extra
fields (epoch / val loss / …) from the checkpoint, rebuilds the model,
runs a dry forward pass on a tiny random batch, and prints a short
report. The command exits non-zero if anything looks wrong.

Exit codes
----------
0    checkpoint loaded and the forward pass produced finite outputs
2    file missing / unreadable
3    schema mismatch or construction failure
4    forward pass produced NaN / Inf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.models.music_transformer import (
    CHECKPOINT_SCHEMA_VERSION,
    MusicTransformer,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify a saved MusicTransformer checkpoint.",
    )
    p.add_argument("checkpoint", type=Path, help="Path to the .pt file.")
    p.add_argument(
        "--batch-size", type=int, default=2,
        help="Dry-run batch size (default: 2).",
    )
    p.add_argument(
        "--seq-len", type=int, default=16,
        help="Dry-run sequence length (default: 16).",
    )
    p.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Device to load the checkpoint onto (default: cpu).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    ckpt = args.checkpoint
    if not ckpt.exists():
        print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    try:
        model, extra = MusicTransformer.load_checkpoint(ckpt, device=args.device)
    except ValueError as e:
        print(f"error: schema mismatch: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # pragma: no cover - defensive
        print(f"error: failed to load checkpoint: {e}", file=sys.stderr)
        return 3

    print(f"checkpoint:        {ckpt}")
    print(f"schema_version:    {CHECKPOINT_SCHEMA_VERSION}")
    print(f"device:            {args.device}")
    print(f"parameters:        {model.count_parameters():,}")
    print("hyperparameters:")
    for k, v in sorted(model.hyperparameters.items()):
        print(f"  {k:<16}{v}")
    print("vocab_config:")
    for k, v in sorted(model.vocab_config.items()):
        print(f"  {k:<20}{v}")
    if extra:
        print("extra:")
        for k, v in sorted(extra.items()):
            print(f"  {k:<20}{v}")

    # Tiny dry-run forward pass — catches shape / device mismatches that
    # slip past a bare state-dict load.
    vc = model.vocab_config
    seq_len = min(args.seq_len, model.hyperparameters["max_seq_len"])
    dev = torch.device(args.device)
    chord = torch.randint(0, vc["chord_vocab_size"], (args.batch_size, seq_len), device=dev)
    rn = torch.randint(0, vc["rn_vocab_size"], (args.batch_size, seq_len), device=dev)
    cad = torch.randint(0, vc["cadence_vocab_size"], (args.batch_size, seq_len), device=dev)
    model.eval()
    with torch.no_grad():
        out = model(chord, rn, cad)
    if not torch.isfinite(out).all():
        print("error: forward pass produced non-finite outputs", file=sys.stderr)
        return 4
    print(
        f"forward pass:      ok  "
        f"(output shape={tuple(out.shape)}, finite=True)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
