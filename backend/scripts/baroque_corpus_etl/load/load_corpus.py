"""Bach Propagation — Load Corpus Orchestrator.

Runs all three load stages in order:
    1. upload_to_storage    — Silver files → Supabase Storage + corpus_files
    2. extract_metadata     — music21 extraction → corpus_metadata
    3. generate_embeddings  — OpenAI vectors  → corpus_embeddings

Each stage is idempotent: re-running the orchestrator skips already-completed
files and only processes pending or failed rows.

Usage (dry-run, always default):
    uv run scripts/baroque_corpus_etl/load/load_corpus.py

Usage (full pipeline, real operations):
    uv run scripts/baroque_corpus_etl/load/load_corpus.py --execute

Usage (single stage):
    uv run scripts/baroque_corpus_etl/load/load_corpus.py --execute --stage upload
    uv run scripts/baroque_corpus_etl/load/load_corpus.py --execute --stage metadata
    uv run scripts/baroque_corpus_etl/load/load_corpus.py --execute --stage embed

Usage (single collection):
    uv run scripts/baroque_corpus_etl/load/load_corpus.py --execute --collection chorales

Usage (smoke test — 10 files through all stages):
    uv run scripts/baroque_corpus_etl/load/load_corpus.py --execute --limit 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Stage imports
# ---------------------------------------------------------------------------

# Resolve backend/ root so imports work when called from any cwd
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))

from scripts.baroque_corpus_etl.load.upload_to_storage import (
    run_upload,
    SILVER_ROOT,
    VALID_COLLECTIONS,
)
from scripts.baroque_corpus_etl.load.extract_metadata import (
    run_extraction,
)
from scripts.baroque_corpus_etl.load.generate_embeddings import (
    run_embedding,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

STAGES = ("upload", "metadata", "embed")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def health_check(client: Client) -> dict[str, int]:
    """Query row counts across all three tables.

    Args:
        client: Authenticated Supabase client.

    Returns:
        Dict with keys: corpus_files, corpus_metadata, corpus_embeddings,
        embedded (files with load_status = 'embedded'),
        failed (files with any *_failed status).
    """
    def count(table: str, filters: dict | None = None) -> int:
        q = client.table(table).select("id", count="exact")
        if filters:
            for col, val in filters.items():
                q = q.eq(col, val)
        return q.execute().count or 0

    def count_like(table: str, col: str, pattern: str) -> int:
        return (
            client.table(table)
            .select("id", count="exact")
            .like(col, pattern)
            .execute()
            .count or 0
        )

    return {
        "corpus_files":       count("corpus_files"),
        "corpus_metadata":    count("corpus_metadata"),
        "corpus_embeddings":  count("corpus_embeddings"),
        "embedded":           count("corpus_files", {"load_status": "embedded"}),
        "failed":             count_like("corpus_files", "load_status", "%failed%"),
    }


def print_health(stats: dict[str, int], label: str = "") -> None:
    """Print a formatted health summary.

    Args:
        stats: Output of health_check().
        label: Optional label (e.g. "Before" / "After").
    """
    prefix = f"[{label}] " if label else ""
    log.info("%scorpus_files      : %d rows", prefix, stats["corpus_files"])
    log.info("%scorpus_metadata   : %d rows", prefix, stats["corpus_metadata"])
    log.info("%scorpus_embeddings : %d rows", prefix, stats["corpus_embeddings"])
    log.info("%sfully embedded    : %d files", prefix, stats["embedded"])
    if stats["failed"] > 0:
        log.warning("%sfailed rows       : %d  ← re-run to retry", prefix, stats["failed"])
    else:
        log.info("%sfailed rows       : 0", prefix)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    execute: bool,
    stage: str | None,
    filter_collection: str | None,
    limit: int | None,
    silver_root: Path,
) -> None:
    """Run the full load pipeline (or a single stage).

    Args:
        execute: If False, all stages run in dry-run mode.
        stage: If set, run only this stage ('upload' | 'metadata' | 'embed').
        filter_collection: Process only this collection across all stages.
        limit: Stop each stage after this many files.
        silver_root: Path to data/silver/ directory.
    """
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    mode_label = "EXECUTE" if execute else "DRY RUN"

    log.info("=" * 60)
    log.info("Bach Propagation — Load Corpus Orchestrator [%s]", mode_label)
    if stage:
        log.info("Stage  : %s only", stage)
    else:
        log.info("Stages : %s", " → ".join(STAGES))
    if filter_collection:
        log.info("Filter : collection = %s", filter_collection)
    if limit:
        log.info("Limit  : %d files per stage", limit)
    log.info("=" * 60)

    # ── Pre-run health check ─────────────────────────────────────────────────
    log.info("--- Health check (before) ---")
    before = health_check(client)
    print_health(before, label="before")

    run_stages = [stage] if stage else list(STAGES)
    stage_times: dict[str, float] = {}

    # ── Stage 1: Upload ──────────────────────────────────────────────────────
    if "upload" in run_stages:
        log.info("--- Stage 1/3: upload_to_storage ---")
        t0 = time.perf_counter()
        run_upload(
            silver_root=silver_root,
            execute=execute,
            filter_collection=filter_collection,
            limit=limit,
        )
        stage_times["upload"] = time.perf_counter() - t0

    # ── Stage 2: Metadata ────────────────────────────────────────────────────
    if "metadata" in run_stages:
        log.info("--- Stage 2/3: extract_metadata ---")
        t0 = time.perf_counter()
        run_extraction(
            silver_root=silver_root,
            execute=execute,
            filter_collection=filter_collection,
            limit=limit,
        )
        stage_times["metadata"] = time.perf_counter() - t0

    # ── Stage 3: Embed ───────────────────────────────────────────────────────
    if "embed" in run_stages:
        log.info("--- Stage 3/3: generate_embeddings ---")
        t0 = time.perf_counter()
        run_embedding(
            execute=execute,
            filter_collection=filter_collection,
            limit=limit,
        )
        stage_times["embed"] = time.perf_counter() - t0

    # ── Post-run health check ────────────────────────────────────────────────
    log.info("--- Health check (after) ---")
    after = health_check(client)
    print_health(after, label="after")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Pipeline complete.")
    for s, elapsed in stage_times.items():
        log.info("  %-10s : %.1fs", s, elapsed)
    total_elapsed = sum(stage_times.values())
    log.info("  %-10s : %.1fs", "total", total_elapsed)

    if after["failed"] > 0:
        log.warning("%d file(s) still in a failed state.", after["failed"])
        log.warning("Re-run with --execute to retry failed rows only.")
        sys.exit(1)

    if execute and not stage:
        # Full pipeline success — verify all three tables match
        n = after["corpus_files"]
        if after["corpus_metadata"] == n and after["corpus_embeddings"] == n:
            log.info("✓ All three tables aligned at %d rows.", n)
        else:
            log.warning(
                "Table mismatch: files=%d metadata=%d embeddings=%d",
                n, after["corpus_metadata"], after["corpus_embeddings"],
            )
            sys.exit(1)

    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bach Propagation — Load corpus orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute", action="store_true", default=False,
        help="Run real operations. Without this flag: dry run across all stages.",
    )
    parser.add_argument(
        "--stage", type=str, default=None, choices=STAGES,
        help="Run only one stage (default: all three in order).",
    )
    parser.add_argument(
        "--collection", type=str, default=None,
        choices=sorted(VALID_COLLECTIONS),
        help="Process only this collection (optional).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Stop each stage after N files (smoke testing).",
    )
    parser.add_argument(
        "--silver-root", type=Path, default=SILVER_ROOT,
        help=f"Path to data/silver/ directory (default: {SILVER_ROOT}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        execute=args.execute,
        stage=args.stage,
        filter_collection=args.collection,
        limit=args.limit,
        silver_root=args.silver_root,
    )