"""
Extract harmonic sequences from a corpus directory of MIDI files.

Usage:
    uv run python scripts/extract_all.py [OPTIONS]

Options:
    --corpus-dir PATH          Directory to search for MIDI files
                               (default: data/samples)
    --output PATH              Output JSON file
                               (default: data/sequences.json)
    --augment                  Apply 12-key transposition augmentation
    --include-prolongation     Include prolongation level tags (slow)

Output:
    JSON array of sequences; each sequence is a list of event dicts.

Example:
    uv run python scripts/extract_all.py --augment
    uv run python scripts/extract_all.py --corpus-dir data/samples --output data/sequences.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sequence_extractor import ExtractionConfig, augment_sequences, extract_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract harmonic sequences from MIDI corpus")
    parser.add_argument(
        '--corpus-dir', default='data/samples', type=Path,
        help='Root directory to search for MIDI files (default: data/samples)',
    )
    parser.add_argument(
        '--output', default='data/sequences.json', type=Path,
        help='Output JSON file path (default: data/sequences.json)',
    )
    parser.add_argument(
        '--augment', action='store_true',
        help='Apply 12-key transposition augmentation',
    )
    parser.add_argument(
        '--include-prolongation', action='store_true',
        help='Include prolongation level tags (considerably slower)',
    )
    args = parser.parse_args()

    config = ExtractionConfig(include_prolongation=args.include_prolongation)

    print(f"Scanning: {args.corpus_dir}")
    sequences = extract_corpus(args.corpus_dir, config)
    print(f"Files processed: {len(sequences)}")

    if args.augment:
        sequences = augment_sequences(sequences, n_shifts=12)
        print(f"After 12-key augmentation: {len(sequences)} sequences")

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(sequences, f)

    print(f"Saved {len(sequences)} sequences → {args.output}")


if __name__ == '__main__':
    main()
