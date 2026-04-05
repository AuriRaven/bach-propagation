# scripts/baroque_corpus_etl/load/reorganize_silver.py
"""
Silver Reorganizer — Bach Propagation ETL Pipeline
===================================================
Load Phase: Reorganize data/silver/bach/midi/ into semantic subcollections
based on BWV number and instrumentation.

Input:  data/silver/bach/midi/    (flat, from execute_rename.py)
Output: data/silver/bach/{subcollection}/  (organized)

SAFETY: Copies files, never moves. Original bach/midi/ preserved until
--cleanup flag is passed after verification.

Final structure:
    bach/
        chorales/               ← untouched (already clean)
        inventions/             ← untouched (already clean)
        keyboard/
            goldberg/           ← BWV 988
            english_suites/     ← BWV 806-811
            french_suites/      ← BWV 812-817
            partitas/           ← BWV 825-830
            lute/               ← BWV 995-1000
            misc/               ← remaining keyboard
        organ/                  ← BWV 525-590, 599-668, 686, 720-740
        solo_instruments/
            violin/sonatas/     ← BWV 1001, 1003, 1005
            violin/partitas/    ← BWV 1002, 1004, 1006
            cello/cello_suites/ ← BWV 1007-1012
            flute/              ← BWV 1013
            lute/               ← BWV 995-1000
        orchestral/
            concertos/          ← BWV 1041-1065
            orchestral_suites/  ← BWV 1066-1071
            musical_offering/   ← BWV 1072-1079
            art_of_fugue/       ← BWV 1080
            chamber/            ← BWV 1027-1040
        sacred/                 ← BWV 1-249 (cantatas, masses, etc.)
        midi_unresolved/        ← bach_unknown_* that cannot be resolved

Usage:
    # Dry run
    python scripts/baroque_corpus_etl/load/reorganize_silver.py \
        --silver-dir data/silver

    # Execute
    python scripts/baroque_corpus_etl/load/reorganize_silver.py \
        --silver-dir data/silver \
        --execute

    # Execute and remove bach/midi/ after verification
    python scripts/baroque_corpus_etl/load/reorganize_silver.py \
        --silver-dir data/silver \
        --execute \
        --cleanup
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# BWV extraction
# ---------------------------------------------------------------------------

_BWV_FROM_SLUG_RE = re.compile(r"BWV(\d{1,4})", re.IGNORECASE)

# Unknown file embedded code patterns
_UNKNOWN_CS_RE  = re.compile(r"_cs([1-6])_(\d+)([a-z]+)")
_UNKNOWN_VS_RE  = re.compile(r"_vs([1-3])_(\d+)([a-z]+)")
_UNKNOWN_VP_RE  = re.compile(r"_vp([1-3])_(\d+)([a-z]+)")
_UNKNOWN_FP_RE  = re.compile(r"_fp_(\d+)([a-z]+)")
_UNKNOWN_PFA_RE = re.compile(r"_pfa_(\d+)([a-z]+)")
_UNKNOWN_BJS_RE = re.compile(r"_bjs(\d{4})([a-z]?)_")
_UNKNOWN_JSB_RE = re.compile(r"_jsb(\w+)_")

_CELLO_BWV   = {1: 1007, 2: 1008, 3: 1009, 4: 1010, 5: 1011, 6: 1012}
_VN_SON_BWV  = {1: 1001, 2: 1003, 3: 1005}
_VN_PAR_BWV  = {1: 1002, 2: 1004, 3: 1006}

_MOVEMENT_ABBREV = {
    "pre": "prelude", "fug": "fugue", "all": "allemande",
    "ald": "allemande_double", "cou": "courante", "cod": "courante_double",
    "sar": "sarabande", "gig": "gigue", "men": "menuet", "bou": "bourree",
    "gav": "gavotte", "lou": "loure", "min": "minuet", "cha": "chaconne",
    "dou": "double", "tb": "tempo_di_borea", "ada": "adagio",
    "and": "andante", "alg": "allegro", "lar": "largo", "gra": "grave",
    "sic": "siciliana", "prs": "presto", "al": "allemande", "co": "courante",
    "sa": "sarabande",
}

def _decode_abbrev(abbrev: str) -> str:
    return _MOVEMENT_ABBREV.get(abbrev.lower(), abbrev.lower())


# ---------------------------------------------------------------------------
# BWV routing
# ---------------------------------------------------------------------------

def route_bwv(bwv_num: int) -> str:
    """
    Map a BWV number to its Silver subcollection path under bach/.

    Args:
        bwv_num: Integer BWV catalog number.

    Returns:
        Relative path string under data/silver/bach/.
    """
    # Violin Sonatas & Partitas
    if bwv_num == 1001: return "solo_instruments/violin/sonatas"
    if bwv_num == 1002: return "solo_instruments/violin/partitas"
    if bwv_num == 1003: return "solo_instruments/violin/sonatas"
    if bwv_num == 1004: return "solo_instruments/violin/partitas"
    if bwv_num == 1005: return "solo_instruments/violin/sonatas"
    if bwv_num == 1006: return "solo_instruments/violin/partitas"
    # Cello Suites
    if 1007 <= bwv_num <= 1012: return "solo_instruments/cello/cello_suites"
    # Flute Partita
    if bwv_num == 1013: return "solo_instruments/flute"
    # Chamber music (gamba/flute sonatas, etc.)
    if 1014 <= bwv_num <= 1040: return "orchestral/chamber"
    # Orchestral Concertos
    if 1041 <= bwv_num <= 1065: return "orchestral/concertos"
    # Orchestral Suites
    if 1066 <= bwv_num <= 1071: return "orchestral/orchestral_suites"
    # Canons + Musical Offering
    if 1072 <= bwv_num <= 1079: return "orchestral/musical_offering"
    # Art of Fugue
    if bwv_num == 1080: return "orchestral/art_of_fugue"
    # Remaining large orchestral
    if 1081 <= bwv_num <= 1120: return "orchestral/misc"
    # Goldberg Variations
    if bwv_num == 988: return "keyboard/goldberg"
    # English Suites
    if 806 <= bwv_num <= 811: return "keyboard/english_suites"
    # French Suites
    if 812 <= bwv_num <= 817: return "keyboard/french_suites"
    # Keyboard Partitas
    if 825 <= bwv_num <= 830: return "keyboard/partitas"
    # Lute / Guitar works
    if bwv_num in (995, 996, 997, 998, 999, 1000): return "solo_instruments/lute"
    # Organ Trio Sonatas
    if 525 <= bwv_num <= 530: return "organ"
    # Organ Preludes, Fugues, Toccatas, Fantasias
    if 531 <= bwv_num <= 598: return "organ"
    # Orgelbüchlein and Chorale Preludes
    if 599 <= bwv_num <= 668: return "organ"
    # Individual Organ Chorale Preludes
    if 669 <= bwv_num <= 740: return "organ"
    # Sacred Cantatas, Masses, Oratorios
    if 1 <= bwv_num <= 249: return "sacred"
    # Remaining keyboard works
    return "keyboard/misc"


# ---------------------------------------------------------------------------
# Unknown filename resolver
# ---------------------------------------------------------------------------

@dataclass
class ResolvedFile:
    """Result of resolving a Silver filename to a subcollection and clean slug."""
    source_path: Path
    target_subdir: str          # e.g. "solo_instruments/cello/cello_suites"
    target_slug: str            # e.g. "bach_BWV1007_01_prelude.mid"
    bwv_num: int | None = None
    resolved: bool = True
    note: str = ""


def resolve_named_file(path: Path) -> ResolvedFile:
    """
    Resolve a properly-named Silver file (e.g. bach_BWV988_01_aria.mid).

    Extracts BWV number from filename and routes via route_bwv().

    Args:
        path: Path to the Silver file.

    Returns:
        ResolvedFile with target location.
    """
    stem = path.stem
    ext  = path.suffix

    bwv_match = _BWV_FROM_SLUG_RE.search(stem)
    if not bwv_match:
        return ResolvedFile(
            source_path=path,
            target_subdir="midi_unresolved",
            target_slug=path.name,
            resolved=False,
            note="no_bwv_in_filename",
        )

    bwv_num = int(re.sub(r"[a-z]", "", bwv_match.group(1)))
    subdir  = route_bwv(bwv_num)

    return ResolvedFile(
        source_path=path,
        target_subdir=subdir,
        target_slug=path.name,
        bwv_num=bwv_num,
    )


def resolve_unknown_file(path: Path) -> ResolvedFile:
    """
    Resolve a bach_unknown_* Silver file by parsing its embedded code.

    Handles: cs, vs, vp, fp, pfa, bjs prefixes.

    Args:
        path: Path to the bach_unknown_* file.

    Returns:
        ResolvedFile with resolved target, or unresolved if code unknown.
    """
    stem = path.stem
    ext  = path.suffix
    lower = stem.lower()

    # --- Cello Suites: _cs{n}_ ---
    cs = _UNKNOWN_CS_RE.search(lower)
    if cs:
        n = int(cs.group(1))
        bwv_num = _CELLO_BWV.get(n)
        mvt_num = cs.group(2).zfill(2)
        mvt_name = _decode_abbrev(cs.group(3))
        if bwv_num:
            return ResolvedFile(
                source_path=path,
                target_subdir="solo_instruments/cello/cello_suites",
                target_slug=f"bach_BWV{bwv_num}_{mvt_num}_{mvt_name}{ext}",
                bwv_num=bwv_num,
                note="resolved_from_unknown",
            )

    # --- Violin Sonatas: _vs{n}_ ---
    vs = _UNKNOWN_VS_RE.search(lower)
    if vs:
        n = int(vs.group(1))
        bwv_num = _VN_SON_BWV.get(n)
        mvt_num = vs.group(2).zfill(2)
        mvt_name = _decode_abbrev(vs.group(3))
        if bwv_num:
            return ResolvedFile(
                source_path=path,
                target_subdir="solo_instruments/violin/sonatas",
                target_slug=f"bach_BWV{bwv_num}_{mvt_num}_{mvt_name}{ext}",
                bwv_num=bwv_num,
                note="resolved_from_unknown",
            )

    # --- Violin Partitas: _vp{n}_ ---
    vp = _UNKNOWN_VP_RE.search(lower)
    if vp:
        n = int(vp.group(1))
        bwv_num = _VN_PAR_BWV.get(n)
        mvt_num = vp.group(2).zfill(2)
        mvt_name = _decode_abbrev(vp.group(3))
        if bwv_num:
            return ResolvedFile(
                source_path=path,
                target_subdir="solo_instruments/violin/partitas",
                target_slug=f"bach_BWV{bwv_num}_{mvt_num}_{mvt_name}{ext}",
                bwv_num=bwv_num,
                note="resolved_from_unknown",
            )

    # --- Flute Partita: _fp_ → BWV1013 ---
    fp = _UNKNOWN_FP_RE.search(lower)
    if fp:
        mvt_num = fp.group(1).zfill(2)
        mvt_name = _decode_abbrev(fp.group(2))
        return ResolvedFile(
            source_path=path,
            target_subdir="solo_instruments/flute",
            target_slug=f"bach_BWV1013_{mvt_num}_{mvt_name}{ext}",
            bwv_num=1013,
            note="resolved_from_unknown",
        )

    # --- Prelude Fugue Allegro: _pfa_ → BWV998 ---
    pfa = _UNKNOWN_PFA_RE.search(lower)
    if pfa:
        mvt_num = pfa.group(1).zfill(2)
        mvt_name = _decode_abbrev(pfa.group(2))
        return ResolvedFile(
            source_path=path,
            target_subdir="solo_instruments/lute",
            target_slug=f"bach_BWV998_{mvt_num}_{mvt_name}{ext}",
            bwv_num=998,
            note="resolved_from_unknown",
        )

    # --- BJS works: _bjs1031a_ → BWV1031 ---
    bjs = _UNKNOWN_BJS_RE.search(lower)
    if bjs:
        bwv_num = int(bjs.group(1))
        variant = bjs.group(2)
        mvt_num = {"a": "01", "b": "02", "c": "03"}.get(variant, "01")
        subdir  = route_bwv(bwv_num)
        return ResolvedFile(
            source_path=path,
            target_subdir=subdir,
            target_slug=f"bach_BWV{bwv_num}_{mvt_num}{ext}",
            bwv_num=bwv_num,
            note="resolved_from_unknown",
        )

    # --- Prelude for solo cello ---
    if "prelude_for_solo_cello" in lower:
        return ResolvedFile(
            source_path=path,
            target_subdir="solo_instruments/cello/cello_suites",
            target_slug=f"bach_cello_prelude_unidentified{ext}",
            resolved=True,
            note="unidentified_cello_prelude",
        )

    # --- Could not resolve ---
    return ResolvedFile(
        source_path=path,
        target_subdir="midi_unresolved",
        target_slug=path.name,
        resolved=False,
        note="embedded_code_not_matched",
    )


# ---------------------------------------------------------------------------
# Main reorganizer
# ---------------------------------------------------------------------------

@dataclass
class ReorgStats:
    copied: int = 0
    dry_run: int = 0
    unresolved: int = 0
    skipped_exists: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)


def reorganize(
    silver_dir: Path,
    dry_run: bool = True,
    overwrite: bool = False,
    cleanup: bool = False,
) -> ReorgStats:
    """
    Scan bach/midi/, resolve every file, copy to target subcollection.

    Args:
        silver_dir: Root of data/silver/.
        dry_run:    If True, log only — no copies.
        overwrite:  If True, overwrite existing targets.
        cleanup:    If True and not dry_run, remove bach/midi/ after copy.

    Returns:
        ReorgStats.
    """
    logger = logging.getLogger(__name__)
    stats  = ReorgStats()
    midi_dir = silver_dir / "bach" / "midi"

    if not midi_dir.exists():
        raise FileNotFoundError(f"bach/midi/ not found: {midi_dir}")

    files = list(midi_dir.rglob("*.mid"))
    logger.info("[%s] Found %d files in bach/midi/",
                "DRY RUN" if dry_run else "EXECUTING", len(files))

    resolved: list[ResolvedFile] = []

    for path in files:
        stem = path.stem
        if stem.startswith("bach_unknown_"):
            r = resolve_unknown_file(path)
        else:
            r = resolve_named_file(path)
        resolved.append(r)

    # Detect slug collisions within each subdir
    seen: dict[str, str] = {}
    for r in resolved:
        key = f"{r.target_subdir}/{r.target_slug}"
        if key in seen:
            # Append _v2, _v3 suffix
            stem, _, ext = r.target_slug.rpartition(".")
            count = sum(1 for k in seen if k.startswith(f"{r.target_subdir}/{stem}"))
            r.target_slug = f"{stem}_v{count + 1}.{ext}"
        seen[f"{r.target_subdir}/{r.target_slug}"] = str(r.source_path)

    for r in tqdm(resolved, desc="DRY RUN" if dry_run else "Reorganizing", unit="file"):
        if not r.resolved:
            stats.unresolved += 1

        target_path = silver_dir / "bach" / r.target_subdir / r.target_slug

        if target_path.exists() and not overwrite:
            stats.skipped_exists += 1
            logger.debug("EXISTS   %s", target_path.name)
            continue

        if dry_run:
            stats.dry_run += 1
            logger.debug(
                "WOULD COPY  %s\n            → bach/%s/%s",
                r.source_path.name,
                r.target_subdir,
                r.target_slug,
            )
            continue

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(r.source_path), str(target_path))
            stats.copied += 1
            logger.debug("COPIED  %s → %s", r.source_path.name, r.target_slug)
        except OSError as e:
            stats.failed += 1
            stats.failures.append(f"{r.source_path.name}: {e}")
            logger.error("FAILED  %s — %s", r.source_path.name, e)

    if cleanup and not dry_run and stats.failed == 0:
        logger.info("Removing bach/midi/ (%d files) ...", len(files))
        shutil.rmtree(str(midi_dir))
        logger.info("bach/midi/ removed.")

    return stats


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(silver_dir: Path) -> None:
    """Print file counts per Silver subcollection."""
    logger = logging.getLogger(__name__)
    bach_dir = silver_dir / "bach"

    logger.info("Silver structure after reorganization:")
    for subdir in sorted(bach_dir.rglob("*")):
        if subdir.is_dir():
            count = sum(1 for _ in subdir.iterdir() if _.is_file())
            if count:
                rel = subdir.relative_to(silver_dir)
                logger.info("  %-50s %d files", str(rel) + "/", count)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reorganize_silver",
        description="Bach Propagation — Silver Reorganizer (Load Phase)",
    )
    parser.add_argument(
        "--silver-dir", type=Path, default=Path("data/silver"),
    )
    parser.add_argument(
        "--execute", action="store_true", default=False,
        help="Perform real copies (default: dry run)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
    )
    parser.add_argument(
        "--cleanup", action="store_true", default=False,
        help="Remove bach/midi/ after successful reorganization",
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
    logger = logging.getLogger(__name__)

    dry_run = not args.execute
    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN — no files will be copied")
        logger.info("Pass --execute to perform real copies")
        logger.info("=" * 60)

    try:
        stats = reorganize(
            silver_dir=args.silver_dir.resolve(),
            dry_run=dry_run,
            overwrite=args.overwrite,
            cleanup=args.cleanup,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info("  Copied        : %d", stats.copied)
    logger.info("  Dry-run       : %d", stats.dry_run)
    logger.info("  Unresolved    : %d", stats.unresolved)
    logger.info("  Skipped       : %d", stats.skipped_exists)
    logger.info("  Failed        : %d", stats.failed)

    if stats.failures:
        for f in stats.failures[:10]:
            logger.error("  FAILED: %s", f)

    if not dry_run:
        print_summary(args.silver_dir.resolve())

    return 1 if stats.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())