"""Upload Silver corpus files to Supabase Storage with atomic rollback.

Each file is processed atomically:
  1. Upload file to Supabase Storage bucket.
  2. Insert a row in corpus_files.
  If either step fails, the storage upload is rolled back (deleted) before
  raising, so no orphaned objects are left in the bucket.

Usage (dry-run, always default):
    uv run scripts/baroque_corpus_etl/load/upload_to_storage.py

Usage (real upload):
    uv run scripts/baroque_corpus_etl/load/upload_to_storage.py --execute

Usage (single collection, dry-run):
    uv run scripts/baroque_corpus_etl/load/upload_to_storage.py --collection chorales

Usage (real, limit 10 files for smoke test):
    uv run scripts/baroque_corpus_etl/load/upload_to_storage.py --execute --limit 10
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from supabase import Client, create_client
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # reads backend/.env

SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

BUCKET_NAME: str = "baroque-corpus"

# Root of the Silver layer, relative to the backend/ project root.
# When running `uv run` from backend/, Path("data/silver") resolves correctly.
SILVER_ROOT: Path = Path("data/silver")

# Sub-directories inside silver/ that are valid for upload.
# Matches the taxonomy in the project brief (excludes contrast/ and midi_unresolved/).
VALID_COLLECTIONS = {
    "chorales",
    "inventions",
    "keyboard",
    "organ",
    "orchestral",
    "sacred",
    "solo_instruments",
}

# Polite delay between Supabase Storage API calls (seconds).
# Free tier is generous but we stay respectful.
UPLOAD_DELAY_SECONDS: float = 0.05

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
# File discovery (generator — never loads full list into RAM)
# ---------------------------------------------------------------------------

def _collection_from_path(silver_path: str) -> str:
    """Extract top-level collection name from a silver-relative path.

    Args:
        silver_path: Relative path string, e.g. "bach/chorales/file.krn".

    Returns:
        Collection name, e.g. "chorales".
    """
    # silver_path format: "bach/<collection>/..."
    parts = Path(silver_path).parts
    return parts[1] if len(parts) > 1 else "unknown"


def _subcollection_from_path(silver_path: str) -> str | None:
    """Extract optional subcollection from a silver-relative path.

    Args:
        silver_path: Relative path string, e.g. "bach/keyboard/english_suites/file.mid".

    Returns:
        Subcollection name or None for flat collections.
    """
    parts = Path(silver_path).parts
    # bach / collection / subcollection / ... / file
    # parts[0] = "bach", parts[1] = collection, parts[2] = subcollection (maybe)
    if len(parts) > 3:
        # There is at least one intermediate directory between collection and file.
        return parts[2]
    return None


def iter_silver_files(
    silver_root: Path,
    filter_collection: str | None = None,
) -> Generator[tuple[Path, str], None, None]:
    """Yield (absolute_path, silver_relative_path) for every uploadable file.

    Only yields files with extensions .krn or .mid that belong to a valid
    collection. Uses a generator so the full file list is never in RAM.

    Args:
        silver_root: Absolute or CWD-relative path to data/silver/.
        filter_collection: If given, only yield files from this collection.

    Yields:
        Tuples of (absolute Path, silver-relative path string).
        Example: (Path("/…/data/silver/bach/chorales/r001.krn"), "bach/chorales/r001.krn")
    """
    if not silver_root.exists():
        raise FileNotFoundError(
            f"Silver root not found: {silver_root.resolve()}. "
            "Run from backend/ project root with `uv run`."
        )

    for path in sorted(silver_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".krn", ".mid"}:
            continue

        # Build the silver-relative path (e.g. "bach/chorales/file.krn")
        rel = path.relative_to(silver_root)
        rel_str = rel.as_posix()

        collection = _collection_from_path(rel_str)

        if collection not in VALID_COLLECTIONS:
            continue  # skip contrast/, midi_unresolved/, etc.

        if filter_collection and collection != filter_collection:
            continue

        yield path, rel_str


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------

def md5_checksum(path: Path) -> str:
    """Compute MD5 hex digest of a file for idempotency checks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex MD5 string.
    """
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Existence check (idempotency)
# ---------------------------------------------------------------------------

def _already_uploaded(client: Client, silver_path: str) -> bool:
    """Return True if a row with this silver_path already exists in corpus_files.

    Args:
        client: Authenticated Supabase client.
        silver_path: Silver-relative path used as the unique key.

    Returns:
        True if the file was previously uploaded successfully.
    """
    result = (
        client.table("corpus_files")
        .select("id, load_status")
        .eq("silver_path", silver_path)
        .execute()
    )
    if not result.data:
        return False
    status = result.data[0].get("load_status", "pending")
    # Re-upload if the previous attempt failed or is still pending.
    return status not in {"pending", "failed"}


# ---------------------------------------------------------------------------
# Atomic upload
# ---------------------------------------------------------------------------

def upload_file_atomic(
    client: Client,
    abs_path: Path,
    silver_path: str,
    execute: bool = False,
) -> dict | None:
    """Upload one file to Storage and insert its corpus_files row atomically.

    If the Storage upload succeeds but the DB insert fails, the Storage object
    is deleted before raising, leaving no orphaned files.

    Args:
        client: Authenticated Supabase client (service role).
        abs_path: Absolute path to the local Silver file.
        silver_path: Silver-relative path (used as storage object path + DB key).
        execute: If False, performs a dry run (no real API calls).

    Returns:
        The inserted corpus_files row dict, or None on dry run.

    Raises:
        RuntimeError: If DB insert fails after successful storage upload
                      (rollback is attempted first).
        Exception: Any unexpected Supabase API error.
    """
    file_format = abs_path.suffix.lstrip(".").lower()  # "krn" or "mid"
    collection = _collection_from_path(silver_path)
    subcollection = _subcollection_from_path(silver_path)
    file_size = abs_path.stat().st_size
    checksum = md5_checksum(abs_path)

    # Storage object path mirrors the silver_path (keeps it navigable in bucket)
    storage_object_path = silver_path

    mime_type, _ = mimetypes.guess_type(abs_path.name)
    mime_type = mime_type or "application/octet-stream"

    if not execute:
        log.debug("[DRY RUN] Would upload: %s (%d bytes)", silver_path, file_size)
        return None

    # ── Step 1: Upload to Storage ────────────────────────────────────────────
    with abs_path.open("rb") as fh:
        file_bytes = fh.read()

    storage_response = (
        client.storage
        .from_(BUCKET_NAME)
        .upload(
            path=storage_object_path,
            file=file_bytes,
            file_options={"content-type": mime_type, "upsert": "true"},
        )
    )

    # Build the storage URL (private bucket → use path; UI will generate signed URLs)
    storage_url = (
        f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{storage_object_path}"
    )

    # ── Step 2: Insert corpus_files row ─────────────────────────────────────
    row = {
        "silver_path": silver_path,
        "filename": abs_path.name,
        "file_format": file_format,
        "file_size_bytes": file_size,
        "checksum_md5": checksum,
        "collection": collection,
        "subcollection": subcollection,
        "storage_object_path": storage_object_path,
        "storage_url": storage_url,
        "load_status": "uploaded",
    }

    try:
        result = (
            client.table("corpus_files")
            .upsert(row, on_conflict="silver_path")
            .execute()
        )
    except Exception as db_err:
        # ── Rollback: delete the Storage object ─────────────────────────────
        log.error(
            "DB insert failed for %s — rolling back storage upload. Error: %s",
            silver_path,
            db_err,
        )
        try:
            client.storage.from_(BUCKET_NAME).remove([storage_object_path])
            log.info("Storage rollback succeeded for: %s", storage_object_path)
        except Exception as rollback_err:
            log.critical(
                "ROLLBACK FAILED for %s — orphaned object may exist in bucket! "
                "Manual cleanup required. Error: %s",
                storage_object_path,
                rollback_err,
            )
        raise RuntimeError(
            f"Atomic upload failed for {silver_path}: {db_err}"
        ) from db_err

    return result.data[0] if result.data else row


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_upload(
    silver_root: Path,
    execute: bool,
    filter_collection: str | None,
    limit: int | None,
) -> None:
    """Run the full upload pass over the Silver corpus.

    Args:
        silver_root: Path to data/silver/ directory.
        execute: If False, dry-run only (no API calls).
        filter_collection: Upload only this collection if specified.
        limit: Stop after this many files (for smoke testing).
    """
    client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    mode_label = "EXECUTE" if execute else "DRY RUN"
    log.info("=" * 60)
    log.info("Bach Propagation — Storage Uploader [%s]", mode_label)
    log.info("Bucket  : %s", BUCKET_NAME)
    log.info("Silver  : %s", silver_root.resolve())
    if filter_collection:
        log.info("Filter  : collection = %s", filter_collection)
    if limit:
        log.info("Limit   : %d files", limit)
    log.info("=" * 60)

    if not execute:
        log.info("No files will be uploaded. Pass --execute to run for real.")

    stats = {"skipped": 0, "uploaded": 0, "failed": 0, "dry_run": 0}
    file_gen = iter_silver_files(silver_root, filter_collection)

    # Wrap generator in tqdm; total is unknown until we walk the tree,
    # so we disable the ETA and show a spinner-style bar.
    progress = tqdm(file_gen, unit="file", desc="Uploading", dynamic_ncols=True)

    processed = 0
    for abs_path, silver_path in progress:
        if limit and processed >= limit:
            log.info("Reached --limit %d. Stopping.", limit)
            break
        processed += 1
        progress.set_postfix(
            uploaded=stats["uploaded"],
            skipped=stats["skipped"],
            failed=stats["failed"],
        )

        if not execute:
            stats["dry_run"] += 1
            log.debug("[DRY RUN] %s", silver_path)
            continue

        # Idempotency: skip files already in a terminal success state.
        if _already_uploaded(client, silver_path):
            stats["skipped"] += 1
            log.debug("Already uploaded, skipping: %s", silver_path)
            continue

        try:
            upload_file_atomic(client, abs_path, silver_path, execute=True)
            stats["uploaded"] += 1
            time.sleep(UPLOAD_DELAY_SECONDS)
        except Exception as exc:
            stats["failed"] += 1
            log.error("FAILED: %s — %s", silver_path, exc)
            # Mark as failed in DB so resume logic can detect it.
            try:
                client.table("corpus_files").upsert(
                    {"silver_path": silver_path, "load_status": "failed",
                     "error_message": str(exc), "filename": abs_path.name,
                     "file_format": abs_path.suffix.lstrip(".").lower(),
                     "collection": _collection_from_path(silver_path)},
                    on_conflict="silver_path",
                ).execute()
            except Exception:
                pass  # Best-effort; don't mask the original error.

    progress.close()

    log.info("-" * 60)
    log.info("Upload complete.")
    if execute:
        log.info("  Uploaded : %d", stats["uploaded"])
        log.info("  Skipped  : %d (already done)", stats["skipped"])
        log.info("  Failed   : %d", stats["failed"])
    else:
        log.info("  Would process : %d files (dry run)", stats["dry_run"])
        log.info("  Run with --execute to upload for real.")
    log.info("-" * 60)

    if stats["failed"] > 0:
        log.warning("%d file(s) failed. Check logs above.", stats["failed"])
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload Bach Propagation Silver corpus to Supabase Storage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually upload files. Without this flag, the script is a dry run.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        choices=sorted(VALID_COLLECTIONS),
        help="Upload only files from this collection (optional).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N files (useful for smoke testing with --execute).",
    )
    parser.add_argument(
        "--silver-root",
        type=Path,
        default=SILVER_ROOT,
        help=f"Path to data/silver/ directory (default: {SILVER_ROOT}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_upload(
        silver_root=args.silver_root,
        execute=args.execute,
        filter_collection=args.collection,
        limit=args.limit,
    )