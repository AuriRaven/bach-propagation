"""Generate OpenAI embeddings for the Bach corpus and store in corpus_embeddings.

Reads rows from corpus_full view (corpus_files JOIN corpus_metadata).
Builds a structured metadata string per file, embeds it with
text-embedding-3-small (1536 dims), and upserts into corpus_embeddings.

Embedding text schema:
    "Bach | {collection} | BWV{bwv} | {movement_name} | {key_signature} | {time_signature} | {form_tag} | {instrument_family}"

Null fields are replaced with "unknown" so the embedding space stays consistent.

Prerequisites:
    - upload_to_storage.py   must have run (corpus_files populated)
    - extract_metadata.py    must have run (corpus_metadata populated)

Usage (dry-run, always default):
    uv run scripts/baroque_corpus_etl/load/generate_embeddings.py

Usage (real embedding):
    uv run scripts/baroque_corpus_etl/load/generate_embeddings.py --execute

Usage (single collection):
    uv run scripts/baroque_corpus_etl/load/generate_embeddings.py --execute --collection chorales

Usage (limit for smoke test):
    uv run scripts/baroque_corpus_etl/load/generate_embeddings.py --execute --limit 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from tqdm import tqdm

try:
    from openai import OpenAI, RateLimitError, APIError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIMS: int = 1536

# OpenAI allows up to 2048 inputs per batch call.
# We use a conservative batch size to stay well within token limits.
BATCH_SIZE: int = 50

# Polite delay between OpenAI batch calls (seconds).
# text-embedding-3-small rate limit: 1M tokens/min on free tier.
# 50 metadata strings × ~30 tokens = ~1500 tokens per batch — very safe.
OPENAI_DELAY_SECONDS: float = 0.2

# Polite delay between Supabase upsert calls (seconds).
DB_DELAY_SECONDS: float = 0.03

VALID_COLLECTIONS = {
    "chorales", "inventions", "keyboard", "organ",
    "orchestral", "sacred", "solo_instruments",
}

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
# Embedding text builder
# ---------------------------------------------------------------------------

def build_embedding_text(row: dict[str, Any]) -> str:
    """Build a structured metadata string for embedding.

    The schema is consistent and descriptive so the embedding space reflects
    musical semantics rather than arbitrary file paths.

    Schema:
        "Bach | {collection} | BWV{bwv} | {movement_name} | {key_signature} |
         {time_signature} | {form_tag} | {instrument_family}"

    Null fields are replaced with "unknown" for embedding consistency.

    Args:
        row: A row from the corpus_full view (joined corpus_files + corpus_metadata).

    Returns:
        Structured string ready to pass to the embeddings API.

    Examples:
        >>> build_embedding_text({
        ...     "collection": "chorales", "bwv": "227", "movement_name": None,
        ...     "key_signature": "G major", "time_signature": "4/4",
        ...     "form_tag": "chorale", "instrument_family": "choir",
        ... })
        'Bach | chorales | BWV227 | unknown | G major | 4/4 | chorale | choir'
    """
    def _val(v: Any, prefix: str = "") -> str:
        if v is None or str(v).strip() == "":
            return "unknown"
        return f"{prefix}{v}"

    parts = [
        "Bach",
        _val(row.get("collection")),
        _val(row.get("bwv"), prefix="BWV"),
        _val(row.get("movement_name")),
        _val(row.get("key_signature")),
        _val(row.get("time_signature")),
        _val(row.get("form_tag")),
        _val(row.get("instrument_family")),
    ]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# OpenAI embedding call with retry
# ---------------------------------------------------------------------------

def _make_embed_fn(client: "OpenAI"):
    """Return a retry-wrapped embedding function bound to the given client.

    Wraps with tenacity so transient RateLimitError and APIError are retried
    with exponential backoff. Defined as a factory so tests can inject a
    mock client cleanly.

    Args:
        client: Authenticated OpenAI client instance.

    Returns:
        Callable that takes a list of strings and returns a list of float vectors.
    """
    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def _embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        # Response data is ordered to match input order (OpenAI guarantee)
        return [item.embedding for item in response.data]

    return _embed


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _fetch_existing_embedding_ids(client: Client) -> set[str]:
    """Return IDs already present in corpus_embeddings (idempotency).

    Args:
        client: Authenticated Supabase client.

    Returns:
        Set of UUID strings.
    """
    result = client.table("corpus_embeddings").select("id").execute()
    return {r["id"] for r in (result.data or [])}


def _fetch_target_rows(
    client: Client,
    filter_collection: str | None,
    limit: int | None,
) -> list[dict]:
    """Fetch corpus_full rows that need embeddings.

    Fetches files with load_status = 'metadata_extracted' that do not yet
    have a corpus_embeddings row.

    Args:
        client: Authenticated Supabase client.
        filter_collection: Optional collection filter.
        limit: Maximum rows to return.

    Returns:
        List of corpus_full view row dicts.
    """
    query = (
        client.table("corpus_full")
        .select(
            "id, collection, subcollection, bwv, movement_name, filename, "
            "key_signature, time_signature, form_tag, instrument_family, load_status"
        )
        .eq("load_status", "metadata_extracted")
    )
    if filter_collection:
        query = query.eq("collection", filter_collection)
    if limit:
        query = query.limit(limit)

    rows = query.execute().data or []

    # Filter out already-embedded rows (idempotency)
    existing_ids = _fetch_existing_embedding_ids(client)
    return [r for r in rows if r["id"] not in existing_ids]


def _upsert_embeddings(
    client: Client,
    batch: list[dict[str, Any]],
) -> None:
    """Upsert a batch of embedding rows and advance load_status.

    Each item in batch must have: id, embedding_text, embedding, model.
    Also updates corpus_files.load_status to 'embedded' for each id.

    Args:
        client: Authenticated Supabase client.
        batch: List of dicts ready for corpus_embeddings upsert.
    """
    client.table("corpus_embeddings").upsert(
        batch, on_conflict="id"
    ).execute()

    ids = [row["id"] for row in batch]
    # Update load_status for all IDs in one call using .in_()
    client.table("corpus_files").update(
        {"load_status": "embedded"}
    ).in_("id", ids).execute()


def _mark_failed(client: Client, file_id: str, error: str) -> None:
    """Mark a corpus_files row as embedding_failed.

    Args:
        client: Authenticated Supabase client.
        file_id: UUID of the row.
        error: Error string to store.
    """
    client.table("corpus_files").update(
        {"load_status": "embedding_failed", "error_message": error}
    ).eq("id", file_id).execute()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_embedding(
    execute: bool,
    filter_collection: str | None,
    limit: int | None,
) -> None:
    """Run embedding generation for all metadata-extracted corpus files.

    Args:
        execute: If False, dry-run only (no API calls).
        filter_collection: Process only this collection if specified.
        limit: Stop after this many files.
    """
    if not OPENAI_AVAILABLE:
        log.error("openai package not installed. Run: uv add openai")
        sys.exit(1)

    db: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    oai = OpenAI(api_key=OPENAI_API_KEY)
    embed = _make_embed_fn(oai)

    mode_label = "EXECUTE" if execute else "DRY RUN"
    log.info("=" * 60)
    log.info("Bach Propagation — Embedding Generator [%s]", mode_label)
    log.info("Model  : %s (%d dims)", EMBEDDING_MODEL, EMBEDDING_DIMS)
    log.info("Batch  : %d files per OpenAI call", BATCH_SIZE)
    if filter_collection:
        log.info("Filter : collection = %s", filter_collection)
    if limit:
        log.info("Limit  : %d files", limit)
    log.info("=" * 60)

    rows = _fetch_target_rows(db, filter_collection, limit)
    total = len(rows)
    log.info("Files to embed: %d", total)

    if not execute:
        log.info("Dry run. Pass --execute to embed for real.")
        for r in rows[:10]:
            text = build_embedding_text(r)
            log.info("  Would embed: %s", text)
        if total > 10:
            log.info("  … and %d more.", total - 10)
        estimated_cost = total * 30 / 1_000_000 * 0.02
        log.info("Estimated cost: ~$%.6f USD", estimated_cost)
        return

    stats = {"embedded": 0, "failed": 0}
    progress = tqdm(total=total, unit="file", desc="Embedding", dynamic_ncols=True)

    # Process in batches to minimise OpenAI API round-trips
    for batch_start in range(0, total, BATCH_SIZE):
        batch_rows = rows[batch_start : batch_start + BATCH_SIZE]
        texts = [build_embedding_text(r) for r in batch_rows]

        try:
            vectors = embed(texts)
        except Exception as exc:
            # Entire batch failed — mark all rows as failed individually
            log.error(
                "OpenAI batch failed (rows %d–%d): %s",
                batch_start, batch_start + len(batch_rows) - 1, exc,
            )
            for row in batch_rows:
                try:
                    _mark_failed(db, row["id"], str(exc))
                except Exception:
                    pass
            stats["failed"] += len(batch_rows)
            progress.update(len(batch_rows))
            continue

        # Build upsert payload
        upsert_batch = [
            {
                "id":             row["id"],
                "embedding_text": text,
                "embedding":      vector,
                "model":          EMBEDDING_MODEL,
            }
            for row, text, vector in zip(batch_rows, texts, vectors)
        ]

        try:
            _upsert_embeddings(db, upsert_batch)
            stats["embedded"] += len(batch_rows)
        except Exception as exc:
            log.error("DB upsert failed for batch at row %d: %s", batch_start, exc)
            for row in batch_rows:
                try:
                    _mark_failed(db, row["id"], str(exc))
                except Exception:
                    pass
            stats["failed"] += len(batch_rows)

        progress.update(len(batch_rows))
        progress.set_postfix(embedded=stats["embedded"], failed=stats["failed"])
        time.sleep(OPENAI_DELAY_SECONDS)

    progress.close()

    log.info("-" * 60)
    log.info("Embedding complete.")
    log.info("  Embedded : %d", stats["embedded"])
    log.info("  Failed   : %d", stats["failed"])
    log.info("-" * 60)

    if stats["failed"] > 0:
        log.warning("%d file(s) failed. Re-run to retry.", stats["failed"])
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OpenAI embeddings for the Bach corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute", action="store_true", default=False,
        help="Actually call OpenAI and write embeddings. Without this: dry run.",
    )
    parser.add_argument(
        "--collection", type=str, default=None,
        choices=sorted(VALID_COLLECTIONS),
        help="Embed only this collection (optional).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Stop after N files (smoke testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_embedding(
        execute=args.execute,
        filter_collection=args.collection,
        limit=args.limit,
    )