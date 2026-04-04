"""
Master orchestrator for Phase 1: Extract (Baroque Corpus ETL).

Runs all extractors **sequentially** in Polite Researcher order to stay within
the three-connection semaphore and jitter-delay constraints imposed by
``BaseExtractor``.

Extraction sequence
-------------------
1. KernScores  — J.S. Bach Humdrum corpus via CLI (``humcat`` / ``humsplit``)
2. Mutopia     — MusicXML-first scores via async HTTP (prioritises ``.xml``)
3. KunstDerFuge — MIDI-only corpus via async HTTP (Bach + all Baroque)

Each extractor writes its own ``failed_downloads.json`` to its staging subtree.
After all three complete, ``check_bronze_integrity`` scans ``data/raw/`` and
emits ``data/raw/bronze_stats.json``.

Polite Researcher policy (enforced by BaseExtractor)
-----------------------------------------------------
- ``SEMAPHORE_LIMIT = 3`` concurrent HTTP connections
- Per-request jitter: ``random.uniform(1.0, 3.5)`` seconds, or the site's
  ``Crawl-delay`` from ``/robots.txt`` when higher
- User-Agent: ``BachPropagation-ThesisBot/1.0 (Research Project; ...UNAM)``

Usage
-----
::

    # From the backend root:
    uv run python scripts/baroque_corpus_etl/extract/build_complete_baroque_corpus.py

    # Override staging root:
    STAGING_ROOT=data/staging uv run python ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrapping
#
#   _SCRIPT_DIR  →  extract/
#   _ETL_DIR     →  baroque_corpus_etl/     (contains check_bronze_integrity)
#   _BACKEND_SRC →  backend/src/            (utils.network, etc.)
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent
_ETL_DIR     = _SCRIPT_DIR.parent
_BACKEND_SRC = _SCRIPT_DIR.parents[2] / "src"

for _p in (_BACKEND_SRC, _SCRIPT_DIR, _ETL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Extractor imports  (new class names post-refactor)
# ---------------------------------------------------------------------------
from download_kernscores_bach import KernScoresBachExtractor   # subprocess-based
from download_kunstderfuge_p1 import KunstDerFugeExtractor     # async HTTP, MIDI
from download_mutopia_baroque  import MutopiaExtractor          # async HTTP, XML+MIDI

# ---------------------------------------------------------------------------
# Validation import (one level above in baroque_corpus_etl/)
# ---------------------------------------------------------------------------
from check_bronze_integrity import check_bronze_integrity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Allow overriding via env var for CI / test runs.
STAGING_ROOT = Path(os.environ.get("STAGING_ROOT", "data/raw"))

_LOG_FMT  = "%(asctime)s %(levelname)-8s %(message)s"
_LOG_DATE = "%H:%M:%S"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(title: str, step: str) -> None:
    """Print a prominent section header."""
    print(f"\n{'═' * 70}")
    print(f"  {step}  {title}")
    print(f"{'═' * 70}")


def _divider() -> None:
    print("─" * 70)


# ---------------------------------------------------------------------------
# Async orchestration
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run all Phase 1 extractors sequentially, then validate the bronze layer.

    Extractors are **not** gathered — they run one after the other so that the
    combined HTTP load stays within the polite-researcher limits even if the
    same domain appears in multiple extractors.
    """
    run_start = datetime.now(timezone.utc)

    print("\n" + "═" * 70)
    print("  BAROQUE CORPUS ETL  —  Phase 1 : Extract")
    print("═" * 70)
    print(f"  Staging root : {STAGING_ROOT.resolve()}")
    print(f"  Started      : {run_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("═" * 70)

    # ── [1/3] KernScores ──────────────────────────────────────────────────
    _section("KernScores — J.S. Bach Humdrum corpus (CLI)", "[1/3]")
    _divider()
    # KernScoresBachExtractor uses subprocess (humcat/humsplit/hum2mid),
    # so no HTTP session is involved.  run() is still async for interface
    # consistency with the other extractors.
    await KernScoresBachExtractor(staging_root=STAGING_ROOT).run()

    # ── [2/3] Mutopia ─────────────────────────────────────────────────────
    _section("Mutopia Project — MusicXML-first Baroque scores", "[2/3]")
    _divider()
    # Mutopia is run before KunstDerFuge because it provides higher-quality
    # MusicXML files.  If a piece exists in both sources the Transform phase
    # will prefer .xml over .mid at load time.
    await MutopiaExtractor(staging_root=STAGING_ROOT).run()

    # ── [3/3] KunstDerFuge ────────────────────────────────────────────────
    _section("KunstDerFuge — MIDI corpus (Bach + Baroque)", "[3/3]")
    _divider()
    # KunstDerFugeExtractor defaults to ALL_COMPOSERS (Bach + 8 other
    # Baroque composers) in a single pass, processing each composer's
    # catalogue page sequentially with concurrent file downloads.
    await KunstDerFugeExtractor(staging_root=STAGING_ROOT).run()

    # ── Bronze Layer Validation ───────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  Bronze Layer Validation")
    print("═" * 70)

    stats_path = check_bronze_integrity(staging_root=STAGING_ROOT)
    stats      = json.loads(stats_path.read_text(encoding="utf-8"))

    n_files  = stats["total_files"]
    n_errors = len(stats["zero_byte_files"])
    status   = "✓" if n_errors == 0 else "⚠"

    elapsed = datetime.now(timezone.utc) - run_start

    # Concise one-line verdict (the detailed table was already printed by
    # check_bronze_integrity above).
    print(
        f"\n  {status}  Bronze Layer Verified: "
        f"{n_files:,} files, {n_errors} error(s)."
    )
    print(f"  Elapsed : {elapsed}")
    print(f"  Report  : {stats_path.resolve()}")
    print("\n" + "═" * 70)
    print("  Phase 1 : Extract  —  COMPLETE")
    print("═" * 70 + "\n")

    # Surface zero-byte files as a non-zero exit if there are errors, so CI
    # pipelines can gate the Transform phase on a clean bronze layer.
    if n_errors:
        print(
            f"  ⚠  {n_errors} zero-byte file(s) detected.  Inspect "
            f"bronze_stats.json before running Phase 2 (Transform).\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT, datefmt=_LOG_DATE)
    asyncio.run(main())
