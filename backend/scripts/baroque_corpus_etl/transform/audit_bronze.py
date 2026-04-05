# scripts/baroque_corpus_etl/transform/audit_bronze.py
"""
Bronze Layer Auditor — Bach Propagation ETL Pipeline
=====================================================
Stage 1: Crawl data/raw/, probe every symbolic/MIDI file with music21,
and emit bronze_audit.csv as the Source of Truth for Silver transforms.

Usage:
    python scripts/baroque_corpus_etl/transform/audit_bronze.py \
        --raw-dir data/raw \
        --output reports/bronze_audit.csv \
        --workers 4

Architecture:
    - Generator-based crawler: O(1) RAM per file path
    - music21 lazy header probe: parses only measure 1 for speed
    - Structured warnings: parse_error | no_metadata | ambiguous_bwv | multiple_keys
    - All rows keyed by content_hash (SHA-256, first 16KB) for dedup detection
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generator, Iterator

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".krn", ".mxl", ".xml", ".musicxml", ".mid", ".midi", ".abc", ".nwc"}
)

BWV_PATTERN: re.Pattern = re.compile(
    r"bwv[_\-\s]?(\d{1,4}[a-z]?)", re.IGNORECASE
)

CSV_FIELDNAMES: list[str] = [
    # Identity
    "file_path",
    "filename",
    "extension",
    "size_bytes",
    "content_hash",
    # Musicological metadata
    "composer",
    "work_title",
    "bwv_number",
    "movement_num",
    "movement_name",
    "catalog_number_raw",
    # Musical features (measure 1 probe)
    "key_signature",
    "time_signature",
    "tempo_bpm",
    "part_count",
    "instrumentation",
    "measure_count",
    "note_count_estimate",
    # Format details
    "format_detected",
    "has_lyrics",
    "is_choral",
    # Audit
    "audit_status",       # OK | WARNING | ERROR
    "audit_warnings",     # pipe-separated warning codes
    "parse_error_msg",
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AuditRow:
    """One row in bronze_audit.csv — represents a single file's probe result."""

    # Identity
    file_path: str = ""
    filename: str = ""
    extension: str = ""
    size_bytes: int = 0
    content_hash: str = ""

    # Musicological metadata
    composer: str = ""
    work_title: str = ""
    bwv_number: str = ""
    movement_num: str = ""
    movement_name: str = ""
    catalog_number_raw: str = ""

    # Musical features
    key_signature: str = ""
    time_signature: str = ""
    tempo_bpm: str = ""
    part_count: int = 0
    instrumentation: str = ""
    measure_count: int = 0
    note_count_estimate: int = 0

    # Format details
    format_detected: str = ""
    has_lyrics: bool = False
    is_choral: bool = False

    # Audit
    audit_status: str = "OK"
    audit_warnings: str = ""
    parse_error_msg: str = ""

    # Internal (not written to CSV)
    _warnings: list[str] = field(default_factory=list, repr=False)

    def add_warning(self, code: str) -> None:
        self._warnings.append(code)
        self.audit_warnings = "|".join(self._warnings)
        if self.audit_status == "OK":
            self.audit_status = "WARNING"

    def set_error(self, msg: str) -> None:
        self.audit_status = "ERROR"
        self.parse_error_msg = msg[:512]  # truncate stack traces

    def to_csv_dict(self) -> dict:
        d = asdict(self)
        d.pop("_warnings", None)
        return d


# ---------------------------------------------------------------------------
# File crawler (generator — O(1) RAM)
# ---------------------------------------------------------------------------

def crawl_raw_dir(raw_dir: Path) -> Generator[Path, None, None]:
    """
    Recursively yield every supported music file under raw_dir.

    Uses os.scandir internally (via Path.rglob) for efficiency.
    Yields one Path at a time — never materializes the full file list.

    Args:
        raw_dir: Root directory of the Bronze layer (data/raw/).

    Yields:
        Path objects for each supported music file.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Bronze directory not found: {raw_dir}")

    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def count_raw_files(raw_dir: Path) -> int:
    """Pre-count files for tqdm progress bar (single fast os.walk pass)."""
    return sum(
        1
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Content hash (deduplication key)
# ---------------------------------------------------------------------------

def compute_content_hash(path: Path, chunk_size: int = 16_384) -> str:
    """
    SHA-256 of the first 16KB of the file.

    Using partial content is intentional: fast enough for 2k+ files
    while catching exact duplicates. Full-file hash would be ~10x slower.

    Args:
        path: File to hash.
        chunk_size: Bytes to read (default 16KB).

    Returns:
        Hex digest string (64 chars).
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(chunk_size))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# BWV extraction helpers
# ---------------------------------------------------------------------------

def extract_bwv_from_filename(filename: str) -> str:
    """
    Regex-extract BWV number from filename stem.

    Handles formats like: bwv772, BWV_772, bwv-772a, 772

    Args:
        filename: File name (with or without extension).

    Returns:
        Normalised BWV string like "BWV772" or "" if not found.
    """
    match = BWV_PATTERN.search(filename)
    if match:
        return f"BWV{match.group(1).upper()}"
    return ""


def extract_bwv_from_metadata(score) -> str:  # type: ignore[return]
    """
    Extract BWV from music21 score metadata fields.

    Checks: movementNumber, number, title, and the raw metadata string.

    Args:
        score: music21 Score object.

    Returns:
        Normalised BWV string or "".
    """
    if score.metadata is None:
        return ""

    candidates = [
        str(score.metadata.movementNumber or ""),
        str(score.metadata.number or ""),
        str(score.metadata.title or ""),
        str(score.metadata.composer or ""),
    ]
    for candidate in candidates:
        match = BWV_PATTERN.search(candidate)
        if match:
            return f"BWV{match.group(1).upper()}"
    return ""


# ---------------------------------------------------------------------------
# music21 probe (the heavy part — isolated for multiprocessing)
# ---------------------------------------------------------------------------

def probe_file(file_path: Path) -> AuditRow:
    """
    Parse a single music file and extract audit metadata.

    Designed to run in a subprocess (ProcessPoolExecutor) to avoid
    music21's global state contamination across files.

    Strategy:
        1. Hash first (cheap, no music21 needed).
        2. Parse score with music21.
        3. Extract metadata from score.metadata.
        4. Probe measure 1 only for key/time sig (fast path).
        5. Estimate note count from flat note stream.
        6. Detect instrumentation from part names.

    Args:
        file_path: Absolute path to the music file.

    Returns:
        Populated AuditRow dataclass.
    """
    # Deferred import: each worker process loads music21 independently
    import music21  # noqa: PLC0415
    from music21 import converter, stream, tempo  # noqa: PLC0415

    row = AuditRow(
        file_path=str(file_path),
        filename=file_path.name,
        extension=file_path.suffix.lower(),
        size_bytes=file_path.stat().st_size,
    )

    # --- Hash ---
    try:
        row.content_hash = compute_content_hash(file_path)
    except OSError as e:
        row.set_error(f"hash_failed: {e}")
        return row

    # --- Parse ---
    try:
        # forceSource=True skips pickle cache — ensures fresh header read
        score = converter.parse(str(file_path), forceSource=True)
    except Exception as e:  # noqa: BLE001
        row.set_error(f"parse_failed: {type(e).__name__}: {e}")
        return row

    row.format_detected = score.__class__.__name__

    # --- Metadata ---
    if score.metadata:
        row.composer = str(score.metadata.composer or "").strip()
        row.work_title = str(score.metadata.title or "").strip()
        row.movement_num = str(score.metadata.movementNumber or "").strip()
        row.movement_name = str(score.metadata.movementName or "").strip()
        row.catalog_number_raw = str(score.metadata.number or "").strip()
    else:
        row.add_warning("no_metadata")

    # --- BWV resolution (filename wins over metadata for Bach corpus) ---
    bwv_from_file = extract_bwv_from_filename(file_path.stem)
    bwv_from_meta = extract_bwv_from_metadata(score)

    if bwv_from_file:
        row.bwv_number = bwv_from_file
        if bwv_from_meta and bwv_from_meta != bwv_from_file:
            row.add_warning(f"ambiguous_bwv:file={bwv_from_file},meta={bwv_from_meta}")
    elif bwv_from_meta:
        row.bwv_number = bwv_from_meta
    else:
        row.add_warning("no_bwv")

    # --- Parts & instrumentation ---
    parts = score.parts if isinstance(score, stream.Score) else []
    row.part_count = len(parts)

    part_names: list[str] = []
    for part in parts:
        name = (part.partName or part.id or "Unknown").strip()
        part_names.append(name)

    row.instrumentation = " | ".join(part_names) if part_names else "unknown"

    # Heuristic: choral if 4 parts with SATB-ish names
    satb_keywords = {"soprano", "alto", "tenor", "bass", "s.", "a.", "t.", "b."}
    lower_names = {n.lower() for n in part_names}
    row.is_choral = len(parts) == 4 and bool(lower_names & satb_keywords)

    # --- Key & time signature (measure 1 fast-path) ---
    keys_found: list[str] = []
    time_sigs_found: list[str] = []

    try:
        # Use flattened stream of first part for speed
        first_part = parts[0] if parts else score
        measures = first_part.getElementsByClass("Measure")
        first_measure = next(iter(measures), None)

        if first_measure:
            ks_list = first_measure.getElementsByClass("KeySignature")
            ts_list = first_measure.getElementsByClass("TimeSignature")

            for ks in ks_list:
                keys_found.append(str(ks.asKey()))
            for ts in ts_list:
                time_sigs_found.append(ts.ratioString)

        # Fallback: analyse if no explicit key signature in m1
        if not keys_found:
            # Cheap: analyse only first 8 measures to infer key
            snippet = first_part.measures(1, 8)
            analysis = snippet.analyze("key")
            if analysis:
                keys_found.append(f"{analysis} (inferred)")
                row.add_warning("key_inferred")

    except Exception as e:  # noqa: BLE001
        row.add_warning(f"key_probe_failed:{type(e).__name__}")

    if len(set(keys_found)) > 1:
        row.add_warning("multiple_keys")

    row.key_signature = keys_found[0] if keys_found else "unknown"
    row.time_signature = time_sigs_found[0] if time_sigs_found else "unknown"

    # --- Tempo ---
    try:
        tempo_marks = score.flat.getElementsByClass(tempo.MetronomeMark)
        first_tempo = next(iter(tempo_marks), None)
        if first_tempo and first_tempo.number:
            row.tempo_bpm = str(round(first_tempo.number, 1))
    except Exception:  # noqa: BLE001
        pass  # Tempo is non-critical; silence the error

    # --- Measure count ---
    try:
        if parts:
            all_measures = parts[0].getElementsByClass("Measure")
            row.measure_count = len(list(all_measures))
    except Exception:  # noqa: BLE001
        row.add_warning("measure_count_failed")

    # --- Note count estimate (flat stream, count Note+Chord) ---
    try:
        from music21 import note as m21_note  # noqa: PLC0415

        flat = score.flat
        notes = flat.getElementsByClass([m21_note.Note, m21_note.Rest])
        row.note_count_estimate = len(list(notes))
    except Exception:  # noqa: BLE001
        row.add_warning("note_count_failed")

    # --- Lyrics detection ---
    try:
        flat_notes = score.flat.notes
        row.has_lyrics = any(
            bool(n.lyrics) for n in flat_notes
        )
    except Exception:  # noqa: BLE001
        pass

    return row


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_audit(
    raw_dir: Path,
    output_csv: Path,
    workers: int = 4,
    limit: int | None = None,
) -> None:
    """
    Orchestrate the Bronze audit: crawl → probe → write CSV.

    Args:
        raw_dir: Root of the Bronze data directory.
        output_csv: Destination path for bronze_audit.csv.
        workers: Number of parallel worker processes for music21 parsing.
        limit: Optional cap on file count (for dev/testing).
    """
    logger = logging.getLogger(__name__)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Counting files in %s ...", raw_dir)
    total = count_raw_files(raw_dir)
    if limit:
        total = min(total, limit)
    logger.info("Found %d supported files to audit.", total)

    stats = {"ok": 0, "warning": 0, "error": 0}

    def _file_iter() -> Iterator[Path]:
        count = 0
        for p in crawl_raw_dir(raw_dir):
            yield p
            count += 1
            if limit and count >= limit:
                return

    with (
        output_csv.open("w", newline="", encoding="utf-8") as csv_file,
        ProcessPoolExecutor(max_workers=workers) as executor,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        futures = {
            executor.submit(probe_file, path): path
            for path in _file_iter()
        }

        with tqdm(
            total=total,
            desc="Auditing Bronze",
            unit="file",
            dynamic_ncols=True,
        ) as pbar:
            for future in as_completed(futures):
                path = futures[future]
                try:
                    row: AuditRow = future.result()
                except Exception as e:  # noqa: BLE001
                    row = AuditRow(
                        file_path=str(path),
                        filename=path.name,
                        extension=path.suffix.lower(),
                        size_bytes=path.stat().st_size,
                    )
                    row.set_error(f"executor_error: {e}")

                stats[row.audit_status.lower()] += 1
                writer.writerow(row.to_csv_dict())
                csv_file.flush()  # Progressive write — safe against crashes

                pbar.set_postfix(
                    ok=stats["ok"],
                    warn=stats["warning"],
                    err=stats["error"],
                )
                pbar.update(1)

    logger.info(
        "Audit complete → %s | OK: %d | WARN: %d | ERR: %d",
        output_csv,
        stats["ok"],
        stats["warning"],
        stats["error"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_bronze",
        description="Bach Propagation — Bronze Layer Auditor (Stage 1 ETL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory of the Bronze layer (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/bronze_audit.csv"),
        help="Output CSV path (default: reports/bronze_audit.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker processes for music21 parsing (default: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap file count for dev/smoke testing (default: no limit)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        run_audit(
            raw_dir=args.raw_dir,
            output_csv=args.output,
            workers=args.workers,
            limit=args.limit,
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())