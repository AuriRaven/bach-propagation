"""Extract musicological metadata from Silver corpus files into corpus_metadata.

Extraction strategy:
  .krn files  → music21.parse() — full metadata (key, time, measures, voices, duration)
  .mid files  → music21.parse() attempted first; falls back to filename parsing.

Fix: Added robust duration calculation with 120 BPM fallback for files without tempo marks.
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
# Helpers (Parsing & Inference)
# ---------------------------------------------------------------------------

_MOVEMENT_KEYWORDS = {
    "prelude", "allemande", "courante", "sarabande", "gigue", "menuet",
    "minuet", "bourree", "gavotte", "aria", "fugue", "invention",
    "sinfonia", "toccata", "fantasia", "chaconne", "passacaglia",
    "overture", "rondo", "march", "polonaise", "variation", "canon",
}

def parse_movement_from_filename(filename: str) -> str | None:
    stem = re.sub(r"[_\-]", " ", Path(filename).stem.lower())
    for keyword in _MOVEMENT_KEYWORDS:
        if keyword in stem.split():
            return keyword.capitalize()
    return None

def infer_form_tag(collection: str, movement_name: str | None, filename: str) -> str | None:
    fixed = {"chorales": "chorale", "organ": "organ_work", "sacred": "sacred_work"}
    if collection in fixed: return fixed[collection]
    if collection == "inventions": return "sinfonia" if "sinfonia" in filename.lower() else "invention"
    return collection.rstrip('s') + "_work" if collection in ["keyboard", "orchestral", "solo_instruments"] else None

def infer_instrument_family(collection: str, subcollection: str | None) -> str:
    simple_map = {"chorales": "choir", "inventions": "keyboard", "keyboard": "keyboard", "organ": "organ", "sacred": "mixed"}
    if collection in simple_map: return simple_map[collection]
    if "cello" in (subcollection or "").lower(): return "strings"
    return "mixed" if collection == "orchestral" else "solo"

# ---------------------------------------------------------------------------
# music21 extraction (Robust Version)
# ---------------------------------------------------------------------------

def _extract_with_music21(abs_path: Path) -> dict[str, Any]:
    score = converter.parse(str(abs_path))
    
    # ── Key ──────────────────────────────────────────────────────────────────
    key_sig, key_mode = None, None
    try:
        detected = score.analyze("key")
        if detected.correlationCoefficient >= KEY_CONFIDENCE_THRESHOLD:
            key_sig, key_mode = f"{detected.tonic.name} {detected.mode}", detected.mode
    except: pass

    # ── Time Signature ───────────────────────────────────────────────────────
    time_sig = None
    try:
        ts = score.flatten().getElementsByClass("TimeSignature")
        if ts: time_sig = f"{ts[0].numerator}/{ts[0].denominator}"
    except: pass

    # ── Duration (Ultra-Robust Fix) ──────────────────────────────────────────
    duration_seconds = None
    try:
        total_ql = float(score.duration.quarterLength)
        if total_ql > 0:
            # Intento 1: ¿Tiene marcas de tempo reales?
            try:
                secs = score.seconds
                if secs and secs > 0:
                    duration_seconds = round(float(secs), 2)
            except: pass

            # Intento 2 (Fallback): Si falla o no hay tempo, asumimos 120 BPM
            if duration_seconds is None or duration_seconds == 0:
                # Duración = total_ql * (60s / 120bpm) -> total_ql / 2
                duration_seconds = round(total_ql * 0.5, 2)
    except:
        duration_seconds = 0.0

    return {
        "key_signature": key_sig,
        "key_mode": key_mode,
        "time_signature": time_sig,
        "num_measures": len(score.parts[0].getElementsByClass("Measure")) if score.parts else None,
        "num_voices": len(score.parts),
        "duration_seconds": duration_seconds,
        "extraction_method": "music21"
    }

# ---------------------------------------------------------------------------
# Combined extraction entry point
# ---------------------------------------------------------------------------

def extract_metadata(abs_path: Path, silver_path: str, collection: str, subcollection: str | None) -> dict[str, Any]:
    m21_result = {}
    if MUSIC21_AVAILABLE:
        try:
            m21_result = _extract_with_music21(abs_path)
        except Exception as exc:
            log.warning(f"music21 failed for {silver_path}: {exc}")

    mv_name = parse_movement_from_filename(abs_path.name)
    return {
        **m21_result,
        "instrument_family": infer_instrument_family(collection, subcollection),
        "form_tag": infer_form_tag(collection, mv_name, abs_path.name),
        "extraction_method": "music21" if m21_result else "filename_parsing"
    }

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _fetch_target_rows(client: Client, filter_collection: str | None, limit: int | None) -> list[dict]:
    # Quitamos filtros de estado para permitir re-procesar (Update/Upsert)
    query = client.table("corpus_files").select("id, silver_path, filename, file_format, collection, subcollection")
    if filter_collection:
        query = query.eq("collection", filter_collection)
    if limit:
        query = query.limit(limit)
    
    rows = query.execute().data or []
    return rows

def _upsert_metadata(client: Client, file_id: str, metadata: dict) -> None:
    client.table("corpus_metadata").upsert({"id": file_id, **metadata}, on_conflict="id").execute()
    client.table("corpus_files").update({"load_status": "metadata_extracted"}).eq("id", file_id).execute()

def _mark_failed(client: Client, file_id: str, error: str) -> None:
    client.table("corpus_files").update({"load_status": "metadata_failed"}).eq("id", file_id).execute()

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_extraction(silver_root: Path, execute: bool, filter_collection: str | None, limit: int | None) -> None:
    if not MUSIC21_AVAILABLE:
        log.error("music21 is not installed."); sys.exit(1)

    client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    rows = _fetch_target_rows(client, filter_collection, limit)
    log.info(f"Files to process: {len(rows)}")

    if not execute:
        log.info("Dry run mode. Pass --execute to save results."); return

    stats = {"extracted": 0, "failed": 0, "skipped": 0}
    for row in tqdm(rows, unit="file", desc="Extracting"):
        abs_path = silver_root / row["silver_path"]
        if not abs_path.exists():
            stats["skipped"] += 1; continue
        try:
            meta = extract_metadata(abs_path, row["silver_path"], row["collection"], row.get("subcollection"))
            _upsert_metadata(client, row["id"], meta)
            stats["extracted"] += 1
            time.sleep(DB_DELAY_SECONDS)
        except Exception as exc:
            stats["failed"] += 1
            log.error(f"FAILED: {row['silver_path']} — {exc}")
            _mark_failed(client, row["id"], str(exc))

    log.info(f"Complete. Extracted: {stats['extracted']}, Failed: {stats['failed']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--collection", type=str)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    
    run_extraction(SILVER_ROOT, args.execute, args.collection, args.limit)