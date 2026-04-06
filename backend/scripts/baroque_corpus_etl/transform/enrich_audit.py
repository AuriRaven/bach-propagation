# scripts/baroque_corpus_etl/transform/enrich_audit.py
"""
Audit Enricher — Bach Propagation ETL Pipeline
===============================================
Stage 2: Read bronze_audit.csv, enrich every row with:
  - source_id          (from source_detector)
  - composer_resolved  (from source_detector)
  - collection         (human-readable)
  - catalog_number_resolved (from filename_decoder)
  - movement_name_decoded   (from filename_decoder)
  - silver_slug        (proposed Silver filename stem)
  - silver_action      (KEEP_PRIMARY | KEEP_CONTRAST | DUPLICATE_DROP | QUARANTINE | NEEDS_REVIEW)

Output: reports/enriched_audit.csv

Usage:
    python scripts/baroque_corpus_etl/transform/enrich_audit.py \
        --input  reports/bronze_audit.csv \
        --output reports/enriched_audit.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.baroque_corpus_etl.transform.catalog_registry import get_policy
from scripts.baroque_corpus_etl.transform.filename_decoder import decode_filename
from scripts.baroque_corpus_etl.transform.source_detector import detect_source

# ---------------------------------------------------------------------------
# New columns added by this stage
# ---------------------------------------------------------------------------

ENRICHED_COLUMNS: list[str] = [
    "source_id",
    "composer_resolved",
    "collection",
    "catalog_number_resolved",
    "movement_name_decoded",
    "silver_slug",
    "silver_action",
    "enrich_notes",
]


def enrich_row(row: pd.Series) -> dict:
    """
    Enrich a single bronze_audit row with source, catalog, and Silver metadata.

    Args:
        row: A pandas Series representing one bronze_audit.csv row.

    Returns:
        Dict of new column values to merge into the row.
    """
    path = Path(str(row["file_path"]))
    stem = path.stem
    ext  = row.get("extension", path.suffix.lower())

    # --- Source detection ---
    source = detect_source(path)
    policy = get_policy(source.source_id)

    # --- Filename decoding ---
    decoded = decode_filename(stem, source.source_id, source.composer)

    # --- Silver action resolution ---
    if decoded.is_quarantine:
        silver_action = "QUARANTINE"
    elif policy.dedup_action == "DUPLICATE_DROP":
        silver_action = "DUPLICATE_DROP"
    elif row.get("audit_status") == "ERROR" and policy.role == "primary":
        silver_action = "NEEDS_REVIEW"
    else:
        silver_action = policy.base_silver_action

    notes_parts = []
    if decoded.decode_notes:
        notes_parts.append(decoded.decode_notes)
    if source.source_id == "unknown":
        notes_parts.append("source_undetected")
    if row.get("audit_status") == "ERROR":
        notes_parts.append("parse_error_in_bronze")

    # --- Compose final slug: {stem}.{ext} ---
    slug_stem = decoded.silver_slug_stem or f"{source.composer}_unknown_{stem}"
    silver_slug = f"{slug_stem}{ext}"

    return {
        "source_id":               source.source_id,
        "composer_resolved":       source.composer,
        "collection":              source.collection,
        "catalog_number_resolved": decoded.catalog_number_resolved,
        "movement_name_decoded":   decoded.movement_name_decoded,
        "silver_slug":             silver_slug,
        "silver_action":           silver_action,
        "enrich_notes":            "|".join(notes_parts),
    }


def run_enrichment(input_csv: Path, output_csv: Path) -> None:
    """
    Read bronze_audit.csv, enrich all rows, write enriched_audit.csv.

    Args:
        input_csv: Path to bronze_audit.csv.
        output_csv: Destination for enriched_audit.csv.
    """
    logger = logging.getLogger(__name__)

    logger.info("Reading %s ...", input_csv)
    df = pd.read_csv(input_csv, dtype=str).fillna("")
    total = len(df)
    logger.info("Enriching %d rows ...", total)

    enriched_rows: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=total, desc="Enriching", unit="row"):
        enriched_rows.append(enrich_row(row))

    enriched_df = pd.DataFrame(enriched_rows)
    result_df = pd.concat([df, enriched_df], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    # Summary
    action_counts = result_df["silver_action"].value_counts()
    logger.info("Enrichment complete → %s", output_csv)
    for action, count in action_counts.items():
        logger.info("  %-20s %d", action, count)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enrich_audit",
        description="Bach Propagation — Audit Enricher (Stage 2 ETL)",
    )
    parser.add_argument(
        "--input",  type=Path, default=Path("reports/bronze_audit.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/enriched_audit.csv")
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
    run_enrichment(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())