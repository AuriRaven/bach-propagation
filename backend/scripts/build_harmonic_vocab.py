"""
Build a harmonic vocabulary and encode the Stage 4 analysis cache.

Reads every per-score JSON written by :mod:`scripts.batch_harmonic_analysis`,
constructs three parallel vocabularies (chord, roman numeral, cadence), and
writes out:

    vocab.json                — token ↔ id maps for all three streams
    encoded_sequences.jsonl   — one EncodedSequence dict per line
    summary.json              — corpus-level counts + top-N token breakdowns

Usage:
    uv run python scripts/build_harmonic_vocab.py
    uv run python scripts/build_harmonic_vocab.py \
        --cache-dir data/analysis_cache \
        --output-dir data/encoded \
        --min-chord-count 2 \
        --min-rn-count 2

The script is **idempotent and additive**: re-running it just overwrites
the output files. It does not mutate the Stage 4 cache.

Exit codes:
    0   success
    2   CLI/argument error or missing cache directory
    3   cache dir contained no usable analyses
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Ensure backend/src is importable when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.harmonic_encoder import (  # noqa: E402
    HarmonicEncoder,
    HarmonicEncoderConfig,
    HarmonicVocabulary,
    VOCAB_SCHEMA_VERSION,
    _iter_analysis_files,
    build_vocabulary,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_harmonic_vocab",
        description=(
            "Build token vocabulary + encoded sequences from the Stage 4 "
            "harmonic-analysis cache."
        ),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/analysis_cache"),
        help="Directory containing per-score analysis JSONs "
             "(default: data/analysis_cache).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/encoded"),
        help="Where to write vocab.json / encoded_sequences.jsonl / summary.json "
             "(default: data/encoded).",
    )
    p.add_argument(
        "--min-chord-count",
        type=int,
        default=1,
        help="Drop chord labels observed fewer than this many times (default: 1).",
    )
    p.add_argument(
        "--min-rn-count",
        type=int,
        default=1,
        help="Drop Roman-numeral labels observed fewer than this many times "
             "(default: 1).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return p


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def _summarise(
    vocab: HarmonicVocabulary,
    n_sequences: int,
    n_timesteps: int,
    cache_dir: Path,
    output_dir: Path,
    config: HarmonicEncoderConfig,
) -> Dict[str, object]:
    """Assemble the summary.json payload."""

    def _top_labels(mapping: Dict[str, int], k: int = 20) -> List[Dict[str, object]]:
        # Lower ids mean higher frequency (minus the specials). Skip any
        # label that starts with '<' — those are reserved.
        entries = sorted(
            ((label, idx) for label, idx in mapping.items() if not label.startswith("<")),
            key=lambda kv: kv[1],
        )
        return [{"label": label, "id": idx} for label, idx in entries[:k]]

    return {
        "schema_version": VOCAB_SCHEMA_VERSION,
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "config": {
            "min_chord_count": config.min_chord_count,
            "min_rn_count": config.min_rn_count,
        },
        "sizes": {
            "chord_vocab": vocab.chord_vocab_size,
            "rn_vocab": vocab.rn_vocab_size,
            "cadence_vocab": vocab.cadence_vocab_size,
            "n_sequences": n_sequences,
            "n_timesteps": n_timesteps,
        },
        "top_chords": _top_labels(vocab.chord_to_id),
        "top_rns": _top_labels(vocab.rn_to_id),
        "cadence_vocab": _top_labels(vocab.cadence_to_id, k=len(vocab.cadence_to_id)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cache_dir: Path = args.cache_dir
    output_dir: Path = args.output_dir

    if not cache_dir.exists():
        print(f"error: cache dir not found: {cache_dir}", file=sys.stderr)
        return 2

    config = HarmonicEncoderConfig(
        min_chord_count=args.min_chord_count,
        min_rn_count=args.min_rn_count,
    )

    # First pass: build vocabulary from every analysis file.
    if not args.quiet:
        print(f"Scanning {cache_dir} for analyses …")
    analyses = list(_iter_analysis_files(cache_dir))
    if not analyses:
        print(f"error: no parseable analyses found under {cache_dir}", file=sys.stderr)
        return 3

    if not args.quiet:
        print(f"Loaded {len(analyses)} analyses — building vocabulary …")
    vocab = build_vocabulary(analyses, config)
    encoder = HarmonicEncoder(vocab)

    # Second pass: encode each analysis.
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences_path = output_dir / "encoded_sequences.jsonl"
    vocab_path = output_dir / "vocab.json"
    summary_path = output_dir / "summary.json"

    n_timesteps = 0
    with sequences_path.open("w", encoding="utf-8") as out:
        for analysis in analyses:
            seq = encoder.encode(analysis)
            out.write(json.dumps(seq.to_dict(), ensure_ascii=False))
            out.write("\n")
            n_timesteps += seq.length

    vocab.save(vocab_path)
    summary = _summarise(
        vocab=vocab,
        n_sequences=len(analyses),
        n_timesteps=n_timesteps,
        cache_dir=cache_dir,
        output_dir=output_dir,
        config=config,
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print()
        print(f"Wrote {vocab_path}")
        print(f"Wrote {sequences_path}  ({len(analyses)} sequences, "
              f"{n_timesteps} timesteps)")
        print(f"Wrote {summary_path}")
        print()
        print("Vocabulary sizes (including specials):")
        print(f"  chord   : {vocab.chord_vocab_size}")
        print(f"  rn      : {vocab.rn_vocab_size}")
        print(f"  cadence : {vocab.cadence_vocab_size}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
