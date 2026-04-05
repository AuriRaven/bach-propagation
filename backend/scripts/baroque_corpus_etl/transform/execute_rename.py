# scripts/baroque_corpus_etl/transform/execute_rename.py
"""
Rename Executor — Bach Propagation ETL Pipeline
================================================
Stage 4: Execute the rename_plan.csv — copy files from Bronze (data/raw/)
to Silver (data/silver/) using the proposed silver_path mappings.

SAFETY RULES:
    - COPIES files, never moves. Bronze layer is never touched.
    - Dry-run mode is the default. Pass --execute to perform real copies.
    - Idempotent: re-running skips already-copied files (unless --overwrite).
    - Only processes actions: RENAME | COLLISION_RESOLVED
    - Skips: NEEDS_REVIEW | DUPLICATE_DROP | QUARANTINE

Usage:
    # Dry run (default — safe to run anytime)
    python scripts/baroque_corpus_etl/transform/execute_rename.py \
        --plan reports/rename_plan.csv

    # Real execution
    python scripts/baroque_corpus_etl/transform/execute_rename.py \
        --plan reports/rename_plan.csv \
        --execute

    # Re-run and overwrite existing Silver files
    python scripts/baroque_corpus_etl/transform/execute_rename.py \
        --plan reports/rename_plan.csv \
        --execute \
        --overwrite
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Actions we will act on vs skip
# ---------------------------------------------------------------------------

EXECUTE_ACTIONS: frozenset[str] = frozenset({"RENAME", "COLLISION_RESOLVED"})
SKIP_ACTIONS: frozenset[str] = frozenset({"DUPLICATE_DROP", "QUARANTINE", "NEEDS_REVIEW"})

# ---------------------------------------------------------------------------
# Execution result tracking
# ---------------------------------------------------------------------------

@dataclass
class ExecutionStats:
    """Tracks counts across the execution run."""
    copied: int = 0
    skipped_action: int = 0
    skipped_exists: int = 0
    failed: int = 0
    dry_run: int = 0
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Copied             : {self.copied}",
            f"  Dry-run (pending)  : {self.dry_run}",
            f"  Skipped (action)   : {self.skipped_action}",
            f"  Skipped (exists)   : {self.skipped_exists}",
            f"  Failed             : {self.failed}",
        ]
        if self.failures:
            lines.append("\n  Failed files:")
            for f in self.failures[:10]:
                lines.append(f"    {f}")
            if len(self.failures) > 10:
                lines.append(f"    ... and {len(self.failures) - 10} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core executor
# ---------------------------------------------------------------------------

def execute_plan(
    plan_csv: Path,
    dry_run: bool = True,
    overwrite: bool = False,
) -> ExecutionStats:
    """
    Execute or simulate the rename plan.

    Args:
        plan_csv:  Path to reports/rename_plan.csv.
        dry_run:   If True, log what would happen but don't copy anything.
        overwrite: If True, overwrite existing Silver files.

    Returns:
        ExecutionStats with full accounting of what happened.
    """
    logger = logging.getLogger(__name__)

    if not plan_csv.exists():
        raise FileNotFoundError(f"Rename plan not found: {plan_csv}")

    df = pd.read_csv(plan_csv, dtype=str).fillna("")
    stats = ExecutionStats()

    mode_label = "DRY RUN" if dry_run else "EXECUTING"
    logger.info("[%s] Processing %d plan entries ...", mode_label, len(df))

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=mode_label,
        unit="file",
        dynamic_ncols=True,
    ):
        action      = row.get("action", "").strip()
        source_path = Path(str(row.get("source_path", "")).strip())
        silver_path = Path(str(row.get("silver_path", "")).strip())

        # --- Skip non-execute actions ---
        if action in SKIP_ACTIONS or action not in EXECUTE_ACTIONS:
            stats.skipped_action += 1
            logger.debug("SKIP [%s] %s", action, source_path.name)
            continue

        # --- Validate source exists ---
        if not source_path.exists():
            stats.failed += 1
            msg = f"source missing: {source_path}"
            stats.failures.append(msg)
            logger.warning("MISSING  %s", source_path)
            continue

        # --- Skip if Silver already exists and not overwriting ---
        if silver_path.exists() and not overwrite:
            stats.skipped_exists += 1
            logger.debug("EXISTS   %s", silver_path.name)
            continue

        # --- Dry run: log only ---
        if dry_run:
            stats.dry_run += 1
            logger.debug(
                "WOULD COPY  %s\n           → %s",
                source_path,
                silver_path,
            )
            continue

        # --- Real copy ---
        try:
            silver_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source_path), str(silver_path))
            stats.copied += 1
            logger.debug("COPIED  %s → %s", source_path.name, silver_path)
        except OSError as e:
            stats.failed += 1
            msg = f"{source_path.name}: {e}"
            stats.failures.append(msg)
            logger.error("FAILED  %s — %s", source_path.name, e)

    return stats


# ---------------------------------------------------------------------------
# Post-execution verification
# ---------------------------------------------------------------------------

def verify_silver(plan_csv: Path) -> None:
    """
    After execution, verify every expected Silver file exists.

    Prints a report of any missing files.

    Args:
        plan_csv: Path to rename_plan.csv.
    """
    logger = logging.getLogger(__name__)
    df = pd.read_csv(plan_csv, dtype=str).fillna("")
    execute_df = df[df["action"].isin(EXECUTE_ACTIONS)]

    missing = []
    for _, row in execute_df.iterrows():
        silver_path = Path(str(row["silver_path"]).strip())
        if not silver_path.exists():
            missing.append(silver_path)

    if missing:
        logger.warning("VERIFICATION FAILED — %d missing Silver files:", len(missing))
        for p in missing[:20]:
            logger.warning("  MISSING: %s", p)
        if len(missing) > 20:
            logger.warning("  ... and %d more", len(missing) - 20)
    else:
        logger.info(
            "VERIFICATION PASSED — all %d Silver files present ✅",
            len(execute_df),
        )


# ---------------------------------------------------------------------------
# Silver structure summary
# ---------------------------------------------------------------------------

def print_silver_summary(plan_csv: Path) -> None:
    """Print a tree-style summary of the Silver directory structure."""
    logger = logging.getLogger(__name__)
    df = pd.read_csv(plan_csv, dtype=str).fillna("")
    execute_df = df[df["action"].isin(EXECUTE_ACTIONS)]

    # Extract subdir from silver_path (2 levels above filename)
    def get_subdir(p: str) -> str:
        parts = Path(p).parts
        try:
            silver_idx = next(
                i for i, part in enumerate(parts) if part == "silver"
            )
            return "/".join(parts[silver_idx + 1:-1])
        except StopIteration:
            return "unknown"

    execute_df = execute_df.copy()
    execute_df["subdir"] = execute_df["silver_path"].apply(get_subdir)
    counts = execute_df.groupby("subdir").size()

    logger.info("Silver directory structure:")
    for subdir, count in counts.items():
        logger.info("  data/silver/%-35s %d files", subdir + "/", count)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    plan_csv: Path,
    dry_run: bool,
    overwrite: bool,
    verify: bool,
) -> int:
    """
    Full execution run with optional verification.

    Returns:
        Exit code: 0 = success, 1 = failures occurred.
    """
    logger = logging.getLogger(__name__)

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE — no files will be copied")
        logger.info("Pass --execute to perform real file operations")
        logger.info("=" * 60)

    stats = execute_plan(plan_csv, dry_run=dry_run, overwrite=overwrite)

    logger.info("=" * 60)
    logger.info("Execution summary:")
    logger.info("%s", stats.summary())

    if not dry_run:
        print_silver_summary(plan_csv)

    if verify and not dry_run:
        logger.info("Running verification ...")
        verify_silver(plan_csv)

    return 1 if stats.failed > 0 else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="execute_rename",
        description="Bach Propagation — Silver Rename Executor (Stage 4 ETL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Safe dry run (default)
  python execute_rename.py --plan reports/rename_plan.csv

  # Real execution with verification
  python execute_rename.py --plan reports/rename_plan.csv --execute --verify

  # Re-run and overwrite existing Silver files
  python execute_rename.py --plan reports/rename_plan.csv --execute --overwrite
        """,
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("reports/rename_plan.csv"),
        help="Rename plan CSV (default: reports/rename_plan.csv)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Perform real file copies (default: dry run only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing Silver files (default: skip existing)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="After execution, verify all Silver files exist",
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
        return run(
            plan_csv=args.plan,
            dry_run=not args.execute,
            overwrite=args.overwrite,
            verify=args.verify,
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())