"""Extract musicological metadata from Silver corpus files into corpus_metadata.

Extraction strategy:
  .krn files  → music21.parse() — full metadata (key, time, measures, voices, duration)
  .mid files  → music21.parse() attempted first; falls back to filename parsing if
                music21 fails or key detection confidence is low.

Populates: corpus_metadata table (1:1 with corpus_files).
Prerequisite: upload_to_storage.py must have run (corpus_files rows must exist).

Usage (dry-run, always default):
    uv run scripts/baroque_corpus_etl/load/extract_metadata.py

Usage (real extraction):
    uv run scripts/baroque_corpus_etl/load/extract_metadata.py --execute

Usage (single collection):
    uv run scripts/baroque_corpus_etl/load/extract_metadata.py --execute --collection chorales

Usage (limit for smoke test):
    uv run scripts/baroque_corpus_etl/load/extract_metadata.py --execute --limit 10
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client
from tqdm import tqdm

# music21 import — optional so tests can mock it cleanly
try:
    from music21 import converter
    from music21 import key as m21key
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

SILVER_ROOT: Path = Path("data/silver")

# Minimum music21 key-detection correlation to trust the result.
# Below this threshold we store NULL rather than a guess.
KEY_CONFIDENCE_THRESHOLD: float = 0.8

# Polite delay between Supabase DB calls (seconds)
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
# Filename parsing helpers
# ---------------------------------------------------------------------------

_BWV_RE = re.compile(r"bwv_?(\d{1,4})", re.IGNORECASE)

_MOVEMENT_KEYWORDS = {
    "prelude", "allemande", "courante", "sarabande", "gigue", "menuet",
    "minuet", "bourree", "gavotte", "aria", "fugue", "invention",
    "sinfonia", "toccata", "fantasia", "chaconne", "passacaglia",
    "overture", "rondo", "march", "polonaise", "andante", "allegro",
    "adagio", "grave", "largo", "presto", "vivace", "variation",
    "canon", "ricercar",
}


def parse_bwv_from_filename(filename: str) -> str | None:
    """Extract BWV number string from a filename.

    Args:
        filename: Bare filename (e.g. "bach_BWV988.mid").

    Returns:
        BWV string (e.g. "988") or None if not found.
    """
    match = _BWV_RE.search(filename)
    return match.group(1) if match else None


def parse_movement_from_filename(filename: str) -> str | None:
    """Extract movement name from a filename by keyword matching.

    Args:
        filename: Filename (with or without extension).

    Returns:
        Capitalised movement name (e.g. "Prelude") or None.
    """
    stem = re.sub(r"[_\-]", " ", Path(filename).stem.lower())
    for keyword in _MOVEMENT_KEYWORDS:
        if keyword in stem.split():
            return keyword.capitalize()
    return None


def infer_form_tag(
    collection: str,
    movement_name: str | None,
    filename: str,
) -> str | None:
    """Infer a form tag from collection and movement context.

    Args:
        collection: Silver collection name (e.g. "chorales").
        movement_name: Parsed movement name or None.
        filename: Bare filename for additional heuristics.

    Returns:
        Form tag string or None.
    """
    # Collection-level fixed tags
    fixed: dict[str, str] = {
        "chorales": "chorale",
        "organ": "organ_work",
        "sacred": "sacred_work",
    }
    if collection in fixed:
        return fixed[collection]

    if collection == "inventions":
        return "sinfonia" if "sinfonia" in filename.lower() else "invention"

    # Movement-name driven tags
    if movement_name:
        mv = movement_name.lower()
        if mv == "fugue":
            return "fugue"
        if mv in {"prelude", "toccata", "fantasia"}:
            return "prelude"
        if mv in {"allemande", "courante", "sarabande", "gigue",
                  "menuet", "minuet", "bourree", "gavotte"}:
            return "suite_movement"
        if mv in {"aria", "variation"}:
            return "aria_variation"
        if mv in {"canon", "ricercar"}:
            return "contrapuntal"

    # Filename heuristics for orchestral
    if collection == "orchestral":
        fn_lower = filename.lower()
        if "fugue" in fn_lower:
            return "fugue"
        if "canon" in fn_lower:
            return "canon"
        return "orchestral_work"

    if collection == "keyboard":
        return "keyboard_work"
    if collection == "solo_instruments":
        return "solo_work"

    return None


def infer_instrument_family(collection: str, subcollection: str | None) -> str:
    """Map collection/subcollection to instrument family string.

    Args:
        collection: Silver collection name.
        subcollection: Optional subcollection name.

    Returns:
        Instrument family string (never None).
    """
    simple_map: dict[str, str] = {
        "chorales": "choir",
        "inventions": "keyboard",
        "keyboard": "keyboard",
        "organ": "organ",
        "sacred": "mixed",
    }
    if collection in simple_map:
        return simple_map[collection]

    if collection == "solo_instruments":
        sub = (subcollection or "").lower()
        if "cello" in sub:
            return "strings"
        if "violin" in sub or "partita" in sub or "sonata" in sub:
            return "strings"
        if "flute" in sub:
            return "woodwinds"
        if "lute" in sub:
            return "plucked"
        return "solo"

    if collection == "orchestral":
        sub = (subcollection or "").lower()
        if "art_of_fugue" in sub:
            return "keyboard"
        return "mixed"

    return "unknown"


# ---------------------------------------------------------------------------
# music21 extraction
# ---------------------------------------------------------------------------

def _extract_with_music21(abs_path: Path) -> dict[str, Any]:
    """Parse a file with music21 and return raw extracted fields.

    Args:
        abs_path: Absolute path to a .krn or .mid file.

    Returns:
        Dict with keys: key_signature, key_mode, time_signature,
        num_measures, num_voices, duration_seconds, raw_metadata,
        extraction_method.

    Raises:
        Exception: If music21 cannot parse the file at all.
    """
    score = converter.parse(str(abs_path))

    # ── Key ──────────────────────────────────────────────────────────────────
    key_sig: str | None = None
    key_mode: str | None = None
    try:
        detected = score.analyze("key")
        if detected.correlationCoefficient >= KEY_CONFIDENCE_THRESHOLD:
            key_sig = f"{detected.tonic.name} {detected.mode}"
            key_mode = detected.mode
        else:
            log.debug(
                "Key confidence %.2f below threshold for %s",
                detected.correlationCoefficient,
                abs_path.name,
            )
    except Exception as exc:
        log.debug("Key detection failed for %s: %s", abs_path.name, exc)

    # ── Time signature ───────────────────────────────────────────────────────
    time_sig: str | None = None
    try:
        ts_list = score.flatten().getElementsByClass("TimeSignature")
        if ts_list:
            ts = ts_list[0]
            time_sig = f"{ts.numerator}/{ts.denominator}"
    except Exception as exc:
        log.debug("Time sig failed for %s: %s", abs_path.name, exc)

    # ── Measure count ────────────────────────────────────────────────────────
    num_measures: int | None = None
    try:
        parts = score.parts
        if parts:
            num_measures = len(parts[0].getElementsByClass("Measure"))
    except Exception as exc:
        log.debug("Measure count failed for %s: %s", abs_path.name, exc)

    # ── Voice count ──────────────────────────────────────────────────────────
    num_voices: int | None = None
    try:
        num_voices = len(score.parts)
    except Exception as exc:
        log.debug("Voice count failed for %s: %s", abs_path.name, exc)

    # ── Duration ─────────────────────────────────────────────────────────────
    duration_seconds: float | None = None
    try:
        tempo_marks = score.flatten().getElementsByClass("MetronomeMark")
        bpm = float(tempo_marks[0].number) if tempo_marks and tempo_marks[0].number else 120.0
        duration_seconds = round(float(score.duration.quarterLength) * (60.0 / bpm), 2)
    except Exception as exc:
        log.debug("Duration failed for %s: %s", abs_path.name, exc)

    # ── Raw metadata dump ────────────────────────────────────────────────────
    raw: dict[str, Any] = {}
    try:
        md = score.metadata
        if md:
            raw = {
                "title": md.title,
                "composer": md.composer,
                "date": str(md.date) if md.date else None,
            }
    except Exception:
        pass

    return {
        "key_signature": key_sig,
        "key_mode": key_mode,
        "time_signature": time_sig,
        "num_measures": num_measures,
        "num_voices": num_voices,
        "duration_seconds": duration_seconds,
        "raw_metadata": raw or None,
        "extraction_method": "music21",
    }


# ---------------------------------------------------------------------------
# Combined extraction entry point
# ---------------------------------------------------------------------------

def extract_metadata(
    abs_path: Path,
    silver_path: str,
    collection: str,
    subcollection: str | None,
) -> dict[str, Any]:
    """Extract metadata for one corpus file (music21 + filename fallback).

    Args:
        abs_path: Absolute path to the local Silver file.
        silver_path: Silver-relative path (for logging).
        collection: Silver collection name.
        subcollection: Optional subcollection name.

    Returns:
        Dict ready for upsert into corpus_metadata (no 'id' key — caller adds it).
    """
    filename = abs_path.name
    file_format = abs_path.suffix.lstrip(".").lower()

    m21_result: dict[str, Any] = {}
    m21_succeeded = False

    if MUSIC21_AVAILABLE:
        try:
            m21_result = _extract_with_music21(abs_path)
            m21_succeeded = True
        except Exception as exc:
            log.warning(
                "music21 failed for %s (%s) — using filename fallback",
                silver_path, exc,
            )
            if file_format == "krn":
                log.error("music21 failed on .krn file — this is unexpected: %s", silver_path)

    movement_name = parse_movement_from_filename(filename)
    form_tag = infer_form_tag(collection, movement_name, filename)
    instrument_family = infer_instrument_family(collection, subcollection)
    extraction_method = "music21" if m21_succeeded else "filename_parsing"

    return {
        "key_signature":     m21_result.get("key_signature"),
        "key_mode":          m21_result.get("key_mode"),
        "time_signature":    m21_result.get("time_signature"),
        "num_measures":      m21_result.get("num_measures"),
        "num_voices":        m21_result.get("num_voices"),
        "duration_seconds":  m21_result.get("duration_seconds"),
        "instrument_family": instrument_family,
        "form_tag":          form_tag,
        "extraction_method": extraction_method,
        "raw_metadata":      m21_result.get("raw_metadata"),
    }


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _fetch_existing_metadata_ids(client: Client) -> set[str]:
    """Return set of IDs already present in corpus_metadata (idempotency).

    Args:
        client: Authenticated Supabase client.

    Returns:
        Set of UUID strings.
    """
    result = client.table("corpus_metadata").select("id").execute()
    return {r["id"] for r in (result.data or [])}


def _fetch_target_rows(
    client: Client,
    filter_collection: str | None,
    limit: int | None,
) -> list[dict]:
    """Fetch corpus_files rows ready for metadata extraction.

    Rows must have load_status IN ('uploaded', 'metadata_failed') and
    must NOT already have a corpus_metadata row.

    Args:
        client: Authenticated Supabase client.
        filter_collection: Optional collection filter.
        limit: Maximum rows to return.

    Returns:
        List of corpus_files row dicts.
    """
    query = (
        client.table("corpus_files")
        .select("id, silver_path, filename, file_format, collection, subcollection")
        .in_("load_status", ["uploaded", "metadata_failed"])
    )
    if filter_collection:
        query = query.eq("collection", filter_collection)
    if limit:
        query = query.limit(limit)

    rows = query.execute().data or []
    existing_ids = _fetch_existing_metadata_ids(client)
    return [r for r in rows if r["id"] not in existing_ids]


def _upsert_metadata(client: Client, file_id: str, metadata: dict) -> None:
    """Write metadata row and advance corpus_files.load_status.

    Args:
        client: Authenticated Supabase client.
        file_id: UUID matching corpus_files.id.
        metadata: Extracted metadata dict (without 'id').
    """
    client.table("corpus_metadata").upsert(
        {"id": file_id, **metadata}, on_conflict="id"
    ).execute()

    client.table("corpus_files").update(
        {"load_status": "metadata_extracted"}
    ).eq("id", file_id).execute()


def _mark_failed(client: Client, file_id: str, error: str) -> None:
    """Mark a corpus_files row as metadata_failed.

    Args:
        client: Authenticated Supabase client.
        file_id: UUID of the row.
        error: Error string to store.
    """
    client.table("corpus_files").update(
        {"load_status": "metadata_failed", "error_message": error}
    ).eq("id", file_id).execute()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_extraction(
    silver_root: Path,
    execute: bool,
    filter_collection: str | None,
    limit: int | None,
) -> None:
    """Run metadata extraction over all uploaded corpus files.

    Args:
        silver_root: Path to data/silver/ directory.
        execute: If False, dry-run only.
        filter_collection: Process only this collection if specified.
        limit: Stop after this many files.
    """
    if not MUSIC21_AVAILABLE:
        log.error("music21 is not installed. Run: uv add music21")
        sys.exit(1)

    client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    mode_label = "EXECUTE" if execute else "DRY RUN"
    log.info("=" * 60)
    log.info("Bach Propagation — Metadata Extractor [%s]", mode_label)
    log.info("Silver : %s", silver_root.resolve())
    if filter_collection:
        log.info("Filter : collection = %s", filter_collection)
    if limit:
        log.info("Limit  : %d files", limit)
    log.info("=" * 60)

    rows = _fetch_target_rows(client, filter_collection, limit)
    log.info("Files to process: %d", len(rows))

    if not execute:
        log.info("Dry run. Pass --execute to run for real.")
        for r in rows[:10]:
            log.info("  Would extract: %s", r["silver_path"])
        if len(rows) > 10:
            log.info("  … and %d more.", len(rows) - 10)
        return

    stats = {"extracted": 0, "failed": 0, "skipped": 0}
    progress = tqdm(rows, unit="file", desc="Extracting", dynamic_ncols=True)

    for row in progress:
        file_id: str = row["id"]
        silver_path: str = row["silver_path"]
        collection: str = row["collection"]
        subcollection: str | None = row.get("subcollection")

        abs_path = silver_root / silver_path
        if not abs_path.exists():
            log.warning("File missing from disk: %s", silver_path)
            stats["skipped"] += 1
            continue

        progress.set_postfix(extracted=stats["extracted"], failed=stats["failed"])

        try:
            metadata = extract_metadata(abs_path, silver_path, collection, subcollection)
            _upsert_metadata(client, file_id, metadata)
            stats["extracted"] += 1
            time.sleep(DB_DELAY_SECONDS)
        except Exception as exc:
            stats["failed"] += 1
            log.error("FAILED: %s — %s", silver_path, exc)
            try:
                _mark_failed(client, file_id, str(exc))
            except Exception:
                pass

    progress.close()

    log.info("-" * 60)
    log.info("Extraction complete.")
    log.info("  Extracted : %d", stats["extracted"])
    log.info("  Failed    : %d", stats["failed"])
    log.info("  Skipped   : %d (file missing from disk)", stats["skipped"])
    log.info("-" * 60)

    if stats["failed"] > 0:
        log.warning("%d file(s) failed. Re-run to retry.", stats["failed"])
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract musicological metadata from Silver corpus files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute", action="store_true", default=False,
        help="Actually extract and write metadata. Without this flag: dry run.",
    )
    parser.add_argument(
        "--collection", type=str, default=None,
        choices=sorted(VALID_COLLECTIONS),
        help="Process only this collection (optional).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Stop after N files (smoke testing).",
    )
    parser.add_argument(
        "--silver-root", type=Path, default=SILVER_ROOT,
        help=f"Path to data/silver/ directory (default: {SILVER_ROOT}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_extraction(
        silver_root=args.silver_root,
        execute=args.execute,
        filter_collection=args.collection,
        limit=args.limit,
    )