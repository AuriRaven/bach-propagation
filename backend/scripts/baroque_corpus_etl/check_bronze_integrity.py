"""
Bronze-layer integrity checker for the Baroque Corpus ETL.

Scans ``data/raw/`` and writes ``data/raw/bronze_stats.json`` with:

    total_files     — total corpus file count (excludes JSON / catalog files)
    by_composer     — {composer_slug: file_count}, sorted alphabetically
    by_format       — {".xml": n, ".mid": n, ".krn": n, …}, sorted by count
    zero_byte_files — list of relative paths whose size == 0 (corrupt downloads)
    generated_at    — ISO-8601 UTC timestamp

Usage
-----
::

    # From the backend root:
    uv run python scripts/baroque_corpus_etl/check_bronze_integrity.py

    # Custom root / output:
    uv run python scripts/baroque_corpus_etl/check_bronze_integrity.py \\
        --root data/raw --output reports/bronze_stats.json

Exit codes
----------
0  — scan complete (warnings printed for zero-byte files, not errors)
1  — staging root does not exist
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Files to skip entirely when counting corpus entries.
_SKIP_NAMES: frozenset[str] = frozenset({
    "bronze_stats.json",
    "failed_downloads.json",
    "MASTER_CATALOG.json",
})

#: Extensions that are metadata / tooling, not score files.
_SKIP_EXTS: frozenset[str] = frozenset({
    ".json", ".py", ".pyc", ".txt", ".md", ".log", ".ini", ".cfg",
})

#: Extensions we actively care about (reported even if count is 0).
_SCORE_EXTS: frozenset[str] = frozenset({
    ".xml", ".musicxml", ".mxl", ".krn", ".mid", ".midi",
})


# ---------------------------------------------------------------------------
# Core scan (pure function — no I/O side-effects, easy to unit-test)
# ---------------------------------------------------------------------------


def scan_staging_area(root: Path) -> dict:
    """Walk *root* recursively and return a stats dict.

    The returned dict is JSON-serialisable and matches the schema documented
    in the module docstring.
    """
    total_files:  int = 0
    by_composer:  dict[str, int] = defaultdict(int)
    by_format:    dict[str, int] = defaultdict(int)
    zero_byte:    list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in _SKIP_NAMES or path.suffix.lower() in _SKIP_EXTS:
            continue

        total_files += 1
        ext = path.suffix.lower() or "(no ext)"
        by_format[ext] += 1

        # The first path component under root is the composer slug.
        try:
            parts = path.relative_to(root).parts
            composer = parts[0] if parts else "unknown"
        except ValueError:
            composer = "unknown"
        by_composer[composer] += 1

        if path.stat().st_size == 0:
            zero_byte.append(str(path.relative_to(root)))

    # Sort formats by count descending so the most common formats appear first.
    sorted_format = dict(
        sorted(by_format.items(), key=lambda kv: kv[1], reverse=True)
    )

    return {
        "total_files":     total_files,
        "by_composer":     dict(sorted(by_composer.items())),
        "by_format":       sorted_format,
        "zero_byte_files": zero_byte,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def _print_report(stats: dict, output_path: Path) -> None:
    """Print a human-readable summary of *stats* to stdout."""
    n_zero = len(stats["zero_byte_files"])

    print(f"\n{'═' * 62}")
    print("  BRONZE LAYER  —  Integrity Report")
    print(f"{'═' * 62}")
    print(f"  Total files     : {stats['total_files']:>7,}")

    # --- By Composer ---
    print(f"\n  {'Composer':<24} {'Files':>7}  Distribution")
    print(f"  {'─' * 54}")
    total = stats["total_files"] or 1  # avoid ZeroDivisionError
    for composer, count in stats["by_composer"].items():
        bar_len = max(1, round(count / total * 30))
        bar = "█" * bar_len
        print(f"  {composer:<24} {count:>7,}  {bar}")

    # --- By Format ---
    print(f"\n  {'Format':<14} {'Files':>7}")
    print(f"  {'─' * 24}")
    for fmt, count in stats["by_format"].items():
        marker = "✓" if fmt in _SCORE_EXTS else " "
        print(f"  {marker} {fmt:<12} {count:>7,}")

    # --- Zero-byte files ---
    print()
    if n_zero:
        print(f"  ⚠  Zero-byte files  : {n_zero}  ← likely corrupt downloads")
        for p in stats["zero_byte_files"][:20]:
            print(f"       {p}")
        if n_zero > 20:
            print(f"       … and {n_zero - 20} more  (see {output_path.name})")
    else:
        print("  ✓  No zero-byte files detected")

    print(f"{'═' * 62}")
    print(f"  Report → {output_path}")
    print(f"{'═' * 62}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def check_bronze_integrity(
    staging_root: Path = Path("data/raw"),
    output_path: Path | None = None,
) -> Path:
    """Scan *staging_root*, print a summary, and write the JSON report.

    Returns the path of the written ``bronze_stats.json``.
    Exits with code 1 if *staging_root* does not exist.
    """
    if not staging_root.exists():
        print(f"✗ Staging root not found: {staging_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {staging_root.resolve()} …")
    stats = scan_staging_area(staging_root)

    out = output_path or (staging_root / "bronze_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _print_report(stats, out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan data/raw/ and generate bronze_stats.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default="data/raw",
        help="Path to the bronze staging root.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override the output JSON path (default: {root}/bronze_stats.json).",
    )
    args = parser.parse_args()

    check_bronze_integrity(
        staging_root=Path(args.root),
        output_path=Path(args.output) if args.output else None,
    )
