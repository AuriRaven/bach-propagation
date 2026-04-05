# scripts/baroque_corpus_etl/transform/rename_plan.py
"""
Rename Planner — Bach Propagation ETL Pipeline
===============================================
Stage 3: Read enriched_audit.csv and produce rename_plan.csv —
a human-reviewable mapping of every source file to its Silver path.

NEVER moves or copies files. Output is a plan only.
Execute the plan with execute_rename.py after human review.

Output columns:
    source_path      — absolute path to Bronze file
    silver_path      — proposed absolute path in data/silver/
    silver_slug      — final normalized filename
    source_id        — provenance
    catalog_number   — resolved catalog number (BWV, K, HWV...)
    action           — RENAME | COLLISION_RESOLVED | NEEDS_REVIEW | SKIP
    note             — human-readable explanation

Usage:
    python scripts/baroque_corpus_etl/transform/rename_plan.py \
        --enriched  reports/enriched_audit.csv \
        --output    reports/rename_plan.csv \
        --silver-dir data/silver \
        --raw-dir   data/raw
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# BWV normalization
# ---------------------------------------------------------------------------

_LEADING_ZEROS_RE = re.compile(r"^(BWV|HWV|K|RV|TWV)0+(\d)", re.IGNORECASE)
_LETTER_SUFFIX_RE = re.compile(r"^(BWV|HWV|K|RV|TWV)(\d+)([a-z]+)$", re.IGNORECASE)


def normalize_catalog(catalog: str) -> str:
    """
    Normalize catalog number to consistent format.

    Rules:
        - Uppercase prefix
        - No leading zeros on numeric part
        - Lowercase letter suffix preserved (BWV772a not BWV772A)

    Examples:
        BWV0541P  → BWV541p
        BWV810E   → BWV810e
        K001      → K1
        BWV772    → BWV772

    Args:
        catalog: Raw catalog string from enriched_audit.csv.

    Returns:
        Normalized catalog string.
    """
    if not catalog or catalog == "nan":
        return ""

    # Strip leading zeros: BWV0541 → BWV541
    catalog = _LEADING_ZEROS_RE.sub(
        lambda m: m.group(1).upper() + m.group(2), catalog
    )

    # Normalize letter suffix to lowercase: BWV810E → BWV810e
    match = _LETTER_SUFFIX_RE.match(catalog)
    if match:
        catalog = f"{match.group(1).upper()}{match.group(2)}{match.group(3).lower()}"
    else:
        # Just uppercase the prefix
        for prefix in ("BWV", "HWV", "TWV"):
            if catalog.upper().startswith(prefix):
                catalog = prefix + catalog[len(prefix):]
                break

    return catalog


# ---------------------------------------------------------------------------
# Slug normalization
# ---------------------------------------------------------------------------

_UNSAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9_\-\.]")


def normalize_slug(slug: str) -> str:
    """
    Sanitize a slug for use as a filesystem filename.

    - Replace unsafe characters with underscores
    - Collapse multiple underscores
    - Strip leading/trailing underscores from stem

    Args:
        slug: Raw silver_slug from enriched_audit.csv.

    Returns:
        Safe filesystem slug.
    """
    stem, _, ext = slug.rpartition(".")
    clean = _UNSAFE_CHARS_RE.sub("_", stem)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return f"{clean}.{ext}" if ext else clean


# ---------------------------------------------------------------------------
# Silver directory structure
# ---------------------------------------------------------------------------

_SOURCE_TO_SUBDIR: dict[str, str] = {
    "bulk_chorales_371":       "bach/chorales",
    "kernscores_inventions":   "bach/inventions",
    "kernscores_sinfonias":    "bach/sinfonias",
    "kernscores_wtc":          "bach/wtc",
    "kernscores_art_of_fugue": "bach/art_of_fugue",
    "kernscores_musical_offering": "bach/musical_offering",
    "kernscores_brandenburg":  "bach/brandenburg",
    "kunstderfuge_js":         "bach/midi",
    "kunstderfuge_vivaldi":    "contrast/vivaldi",
}


def silver_subdir(source_id: str) -> str:
    """Return the Silver subdirectory for a given source_id."""
    return _SOURCE_TO_SUBDIR.get(source_id, f"misc/{source_id}")


# ---------------------------------------------------------------------------
# Collision resolver
# ---------------------------------------------------------------------------

def resolve_collisions(
    plan_rows: list[dict],
) -> list[dict]:
    """
    Detect slug collisions and resolve them by appending _v{n} suffixes.

    A collision occurs when two different source files would produce
    the same silver_slug. The first file keeps the original slug;
    subsequent files get _v2, _v3 etc. appended to the stem.

    Special case: Art of Fugue c02/c02b — the 'b' variant is detected
    from the source filename and gets a _b suffix instead of _v2.

    Args:
        plan_rows: List of plan row dicts with 'silver_slug' key.

    Returns:
        Updated plan_rows with collisions resolved.
    """
    # Group rows by proposed silver_slug
    slug_groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(plan_rows):
        slug_groups[row["silver_slug"]].append(i)

    for slug, indices in slug_groups.items():
        if len(indices) == 1:
            continue  # no collision

        stem, _, ext = slug.rpartition(".")

        for rank, idx in enumerate(indices):
            row = plan_rows[idx]
            source_filename = Path(row["source_path"]).stem.lower()

            if rank == 0:
                # First file keeps original slug, just annotate it
                row["action"] = "COLLISION_RESOLVED"
                row["note"] = (
                    f"collision with {len(indices)-1} other file(s) — kept as primary"
                )
                continue

            # Detect 'b' variant in Art of Fugue: c02b → _b suffix
            if re.search(r"c\d+b", source_filename):
                new_slug = f"{stem}_b.{ext}"
                variant_note = "Art of Fugue variant (b)"
            else:
                new_slug = f"{stem}_v{rank + 1}.{ext}"
                variant_note = f"collision resolved with _v{rank + 1} suffix"

            row["silver_slug"] = new_slug
            # Update silver_path to match new slug
            silver_dir = Path(row["silver_path"]).parent
            row["silver_path"] = str(silver_dir / new_slug)
            row["action"] = "COLLISION_RESOLVED"
            row["note"] = variant_note

    return plan_rows


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------

def build_rename_plan(
    enriched_csv: Path,
    silver_dir: Path,
    raw_dir: Path,
) -> list[dict]:
    """
    Build the full rename plan from enriched_audit.csv.

    Only processes rows with silver_action == KEEP_PRIMARY or KEEP_CONTRAST.
    Skips DUPLICATE_DROP, QUARANTINE, NEEDS_REVIEW.

    Args:
        enriched_csv: Path to reports/enriched_audit.csv.
        silver_dir:   Root of the Silver data directory.
        raw_dir:      Root of the Bronze data directory (for path resolution).

    Returns:
        List of plan row dicts ready for CSV output.
    """
    df = pd.read_csv(enriched_csv, dtype=str).fillna("")

    keep_actions = {"KEEP_PRIMARY", "KEEP_CONTRAST"}
    keep_df = df[df["silver_action"].isin(keep_actions)].copy()

    plan_rows: list[dict] = []

    for _, row in keep_df.iterrows():
        source_path = Path(str(row["file_path"]))

        # Make absolute if relative
        if not source_path.is_absolute():
            # source_path is relative like 'data/raw/bach_js/...'
            # raw_dir is absolute like '/Users/.../backend/data/raw'
            # Strip the 'data/raw/' prefix and join to raw_dir
            str_path = str(source_path)
            for prefix in ("data/raw/", "data\\raw\\"):
                if str_path.startswith(prefix):
                    str_path = str_path[len(prefix):]
                    break
            source_path = (raw_dir / str_path).resolve()

        source_id    = row.get("source_id", "unknown")
        silver_action = row.get("silver_action", "")
        raw_slug     = row.get("silver_slug", "")
        catalog_raw  = row.get("catalog_number_resolved", "")

        # Normalize catalog and slug
        catalog_norm = normalize_catalog(catalog_raw)
        clean_slug   = normalize_slug(raw_slug)

        # Rebuild slug with normalized catalog if catalog changed
        if catalog_raw and catalog_norm and catalog_raw != catalog_norm:
            clean_slug = clean_slug.replace(catalog_raw, catalog_norm)

        # Determine Silver subdirectory
        subdir  = silver_subdir(source_id)
        silver_path = silver_dir / subdir / clean_slug

        # Determine action
        if silver_action == "KEEP_PRIMARY":
            action = "RENAME"
        elif silver_action == "KEEP_CONTRAST":
            action = "RENAME"
        else:
            action = "NEEDS_REVIEW"

        plan_rows.append({
            "source_path":    str(source_path),
            "silver_path":    str(silver_path),
            "silver_slug":    clean_slug,
            "source_id":      source_id,
            "catalog_number": catalog_norm,
            "movement":       row.get("movement_name_decoded", ""),
            "silver_action":  silver_action,
            "action":         action,
            "note":           row.get("enrich_notes", ""),
        })

    # Resolve collisions
    plan_rows = resolve_collisions(plan_rows)

    return plan_rows


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_plan(
    enriched_csv: Path,
    output_csv: Path,
    silver_dir: Path,
    raw_dir: Path,
) -> None:
    """
    Build and write the rename plan CSV.

    Args:
        enriched_csv: Input enriched_audit.csv.
        output_csv:   Output rename_plan.csv.
        silver_dir:   Root Silver directory.
        raw_dir:      Root Bronze/raw directory.
    """
    logger = logging.getLogger(__name__)

    logger.info("Building rename plan from %s ...", enriched_csv)
    plan_rows = build_rename_plan(enriched_csv, silver_dir, raw_dir)

    df = pd.DataFrame(plan_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Summary
    logger.info("Rename plan → %s", output_csv)
    logger.info("  Total entries     : %d", len(df))

    action_counts = df["action"].value_counts()
    for action, count in action_counts.items():
        logger.info("  %-25s %d", action, count)

    collision_count = (df["action"] == "COLLISION_RESOLVED").sum()
    if collision_count:
        logger.warning(
            "  %d collision(s) resolved — review rename_plan.csv before executing",
            collision_count,
        )

    logger.info("Review %s before running execute_rename.py", output_csv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rename_plan",
        description="Bach Propagation — Silver Rename Planner (Stage 3 ETL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--enriched",
        type=Path,
        default=Path("reports/enriched_audit.csv"),
        help="Input enriched_audit.csv (default: reports/enriched_audit.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rename_plan.csv"),
        help="Output rename_plan.csv (default: reports/rename_plan.csv)",
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=Path("data/silver"),
        help="Silver layer root directory (default: data/silver)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Bronze raw directory (default: data/raw)",
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
        run_plan(
            enriched_csv=args.enriched,
            output_csv=args.output,
            silver_dir=args.silver_dir.resolve(),
            raw_dir=args.raw_dir.resolve(),
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())