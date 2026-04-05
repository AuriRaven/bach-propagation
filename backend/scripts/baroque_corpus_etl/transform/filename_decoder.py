# scripts/baroque_corpus_etl/transform/filename_decoder.py
"""
Filename Decoder — Bach Propagation ETL Pipeline
=================================================
Per-source filename parsing rules. Each source has its own naming grammar;
mixing them into one regex would be fragile.

Returns a DecodedFilename dataclass that feeds directly into enrich_audit.py.

Corruption detection: filenames containing 'midi.asp' are failed HTTP
downloads and are immediately quarantined.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from scripts.baroque_corpus_etl.transform.catalog_registry import (
    CATALOG_PATTERNS,
    KERNSCORES_BWV_LOOKUP,
    KERNSCORES_CELLO_SUITES,
    KERNSCORES_VIOLIN,
    KUNSTDERFUGE_JS_ART_OF_FUGUE_PREFIX,
    KUNSTDERFUGE_JS_BMM_PREFIX,
    KUNSTDERFUGE_JS_GOLDBERG_PREFIX,
    KUNSTDERFUGE_JS_LUTE_997_PREFIX,
    KUNSTDERFUGE_JS_MUSICAL_OFFERING_PREFIX,
    decode_movement_abbrev,
)


@dataclass
class DecodedFilename:
    """
    Structured metadata extracted from a filename.

    All fields default to empty string — callers should check truthiness.
    """
    catalog_number_resolved: str = ""  # e.g. BWV772, HWV426, K001
    movement_num: str = ""             # e.g. "01", "02"
    movement_name_decoded: str = ""    # e.g. "Prelude", "Allemande"
    silver_slug_stem: str = ""         # proposed filename stem (no ext)
    decode_notes: str = ""             # warnings or decode info
    is_quarantine: bool = False        # True if filename is corrupted


# ---------------------------------------------------------------------------
# Corruption check
# ---------------------------------------------------------------------------

_CORRUPTION_MARKERS = ("midi.asp", "midi.asp?", ".asp?file=")

def is_corrupted_filename(filename: str) -> bool:
    """Return True if filename is a failed HTTP download URL."""
    lower = filename.lower()
    return any(marker in lower for marker in _CORRUPTION_MARKERS)


# ---------------------------------------------------------------------------
# KernScores decoder (data/raw/bach/{collection}/)
# ---------------------------------------------------------------------------

_KERN_WORK_CODE_RE = re.compile(
    r"^([a-z]+)(\d{1,2})(?:_(\d*)([a-z]+))?([ab]?)$", re.IGNORECASE
)

def decode_kernscores(stem: str) -> DecodedFilename:
    """
    Decode a KernScores Bach filename stem.

    Handles:
        inven01          → BWV772 (Two-Part Invention 1)
        sinfo03          → BWV789 (Sinfonia 3)
        cs1_1pre         → BWV1007 movement 01 Prelude
        vs2_3and         → BWV1003 movement 03 Andante
        wtcii01a         → BWV870 movement 01 Prelude (a=prelude, b=fugue)
        chor001          → chorale (no BWV without manifest lookup)

    Args:
        stem: Filename without extension.

    Returns:
        DecodedFilename with resolved catalog number and movement info.
    """
    result = DecodedFilename()
    lower = stem.lower()

    # --- Chorale: chor001 → no direct BWV without manifest ---
    if lower.startswith("chor"):
        num = re.search(r"\d+", lower)
        result.catalog_number_resolved = ""
        result.movement_num = num.group() if num else ""
        result.movement_name_decoded = "Chorale"
        result.silver_slug_stem = f"bach_chorale_{result.movement_num.zfill(3)}"
        result.decode_notes = "BWV requires manifest lookup"
        return result

    # --- WTC Book II: wtcii01a, wtcii01b ---
    wtc_match = re.match(r"wtcii(\d{2})([ab])", lower)
    if wtc_match:
        num = int(wtc_match.group(1))
        suffix = wtc_match.group(2)
        bwv = f"BWV{869 + num}"
        mvt_name = "Prelude" if suffix == "a" else "Fugue"
        mvt_num = "01" if suffix == "a" else "02"
        result.catalog_number_resolved = bwv
        result.movement_num = mvt_num
        result.movement_name_decoded = mvt_name
        result.silver_slug_stem = f"bach_wtc2_{bwv}_{mvt_num}_{mvt_name.lower()}"
        return result

    # --- Violin/Cello prefix lookup: cs1, vs1, vp1 etc. ---
    parts = lower.split("_", 1)
    work_prefix = parts[0]

    bwv = KERNSCORES_BWV_LOOKUP.get(work_prefix, "")

    if not bwv:
        stripped = work_prefix.rstrip("ab")
        bwv = KERNSCORES_BWV_LOOKUP.get(stripped, "")

    result.catalog_number_resolved = bwv

    if len(parts) == 2:
        mvt_raw = parts[1]
        mvt_digits = re.match(r"(\d+)", mvt_raw)
        mvt_abbrev = re.sub(r"^\d+", "", mvt_raw)
        result.movement_num = mvt_digits.group(1).zfill(2) if mvt_digits else "01"
        result.movement_name_decoded = decode_movement_abbrev(mvt_abbrev)
    else:
        result.movement_num = "01"
        result.movement_name_decoded = ""

    if bwv:
        result.silver_slug_stem = (
            f"bach_{bwv}_{result.movement_num}_{result.movement_name_decoded.lower()}"
            if result.movement_name_decoded
            else f"bach_{bwv}_{result.movement_num}"
        )
    else:
        result.silver_slug_stem = f"bach_unknown_{stem}"
        result.decode_notes = "BWV not resolved from KernScores lookup"

    return result


# ---------------------------------------------------------------------------
# KunstDerFuge JS decoder (data/raw/bach_js/bach/)
# ---------------------------------------------------------------------------

_KDF_JS_RE = re.compile(
    r"^bach_(.+?)_mid_(.+?)_full$", re.IGNORECASE
)

def decode_kunstderfuge_js(stem: str) -> DecodedFilename:
    """
    Decode a KunstDerFuge JS (bach_js) filename stem.

    Pattern: bach_{work_code}_mid_{bwv_ref}_full

    Handles in order:
        - Goldberg Variations (988_*)
        - B Minor Mass (bjsbmm*)
        - Musical Offering (1079_*)
        - Art of Fugue (1080*)
        - Lute Suite BWV 997 (997_*)
        - Cello Suites (cs1-cs6 + movement)
        - Violin Sonatas (vs1-vs3 + movement)
        - Violin Partitas (vp1-vp3 + movement)
        - Flute Partita (fp_*)
        - Prelude-Fugue-Allegro (pfa_*)
        - BJS numbered works (bjs1031*)
        - Explicit BWV in work code (bwv*)

    Args:
        stem: Filename stem (no extension, no path).

    Returns:
        DecodedFilename.
    """
    result = DecodedFilename()
    lower = stem.lower()
    match = _KDF_JS_RE.match(lower)

    if not match:
        result.silver_slug_stem = f"bach_unknown_{stem}"
        result.decode_notes = "kdf_js_pattern_mismatch"
        return result

    work_code = match.group(1)
    bwv_ref   = match.group(2)

    # --- Try explicit BWV from bwv_ref field ---
    bwv_from_ref = ""
    bwv_match = re.search(r"bwv(\d{1,4}[a-z]?)", bwv_ref)
    if bwv_match:
        bwv_from_ref = f"BWV{bwv_match.group(1).upper()}"

    # --- Goldberg Variations: 988_aria, 988_v01–30 ---
    if work_code.startswith("988"):
        result.catalog_number_resolved = KUNSTDERFUGE_JS_GOLDBERG_PREFIX
        variation_match = re.search(r"v(\d{2})", work_code)
        if "aria" in work_code:
            result.movement_num = "00"
            result.movement_name_decoded = "Aria"
        elif variation_match:
            result.movement_num = variation_match.group(1)
            result.movement_name_decoded = f"Variation_{variation_match.group(1)}"
        result.silver_slug_stem = (
            f"bach_BWV988_{result.movement_num}_{result.movement_name_decoded.lower()}"
        )
        return result

    # --- B Minor Mass: bjsbmm01–23 ---
    if work_code.startswith("bjsbmm"):
        num = re.search(r"\d+", work_code)
        result.catalog_number_resolved = KUNSTDERFUGE_JS_BMM_PREFIX
        result.movement_num = num.group().zfill(2) if num else "00"
        result.silver_slug_stem = f"bach_BWV232_{result.movement_num}"
        return result

    # --- Musical Offering: 1079_XX ---
    if work_code.startswith("1079"):
        num = re.search(r"1079_(\d+)", work_code)
        result.catalog_number_resolved = KUNSTDERFUGE_JS_MUSICAL_OFFERING_PREFIX
        result.movement_num = num.group(1).zfill(2) if num else "00"
        result.silver_slug_stem = f"bach_BWV1079_{result.movement_num}"
        return result

    # --- Art of Fugue: 1080_XX ---
    if work_code.startswith("1080"):
        result.catalog_number_resolved = KUNSTDERFUGE_JS_ART_OF_FUGUE_PREFIX
        c_match = re.search(r"c(\d+)", work_code)
        result.movement_num = c_match.group(1).zfill(2) if c_match else "00"
        result.movement_name_decoded = (
            f"Contrapunctus_{result.movement_num}" if c_match else "Unknown"
        )
        result.silver_slug_stem = (
            f"bach_BWV1080_{result.movement_num}_{result.movement_name_decoded.lower()}"
        )
        return result

    # --- Lute Suite BWV 997: 997_X ---
    if work_code.startswith("997"):
        result.catalog_number_resolved = KUNSTDERFUGE_JS_LUTE_997_PREFIX
        mvt_parts = work_code.split("_", 1)
        if len(mvt_parts) == 2:
            mvt_abbrev = re.sub(r"^\d+", "", mvt_parts[1])
            result.movement_name_decoded = decode_movement_abbrev(mvt_abbrev)
            mvt_num_match = re.match(r"(\d+)", mvt_parts[1])
            result.movement_num = mvt_num_match.group(1).zfill(2) if mvt_num_match else "01"
        result.silver_slug_stem = (
            f"bach_BWV997_{result.movement_num}_{result.movement_name_decoded.lower()}"
        )
        return result

    # --- Cello Suites: cs{n}_{num}{abbrev} → BWV1007-1012 ---
    cs_match = re.match(r"^(cs[1-6])_(.+)$", work_code)
    if cs_match:
        bwv = KERNSCORES_CELLO_SUITES.get(cs_match.group(1), "")
        mvt_raw = cs_match.group(2)
        mvt_num_match = re.match(r"(\d+)", mvt_raw)
        mvt_abbrev = re.sub(r"^\d+", "", mvt_raw)
        result.catalog_number_resolved = bwv
        result.movement_num = mvt_num_match.group(1).zfill(2) if mvt_num_match else "01"
        result.movement_name_decoded = decode_movement_abbrev(mvt_abbrev)
        result.silver_slug_stem = (
            f"bach_{bwv}_{result.movement_num}_{result.movement_name_decoded.lower()}"
            if bwv else f"bach_unknown_{stem}"
        )
        return result

    # --- Violin Sonatas: vs{n}_{num}{abbrev} → BWV1001, 1003, 1005 ---
    vs_match = re.match(r"^(vs[1-3])_(.+)$", work_code)
    if vs_match:
        bwv = KERNSCORES_VIOLIN.get(vs_match.group(1), "")
        mvt_raw = vs_match.group(2)
        mvt_num_match = re.match(r"(\d+)", mvt_raw)
        mvt_abbrev = re.sub(r"^\d+", "", mvt_raw)
        result.catalog_number_resolved = bwv
        result.movement_num = mvt_num_match.group(1).zfill(2) if mvt_num_match else "01"
        result.movement_name_decoded = decode_movement_abbrev(mvt_abbrev)
        result.silver_slug_stem = (
            f"bach_{bwv}_{result.movement_num}_{result.movement_name_decoded.lower()}"
            if bwv else f"bach_unknown_{stem}"
        )
        return result

    # --- Violin Partitas: vp{n}_{num}{abbrev} → BWV1002, 1004, 1006 ---
    vp_match = re.match(r"^(vp[1-3])_(.+)$", work_code)
    if vp_match:
        bwv = KERNSCORES_VIOLIN.get(vp_match.group(1), "")
        mvt_raw = vp_match.group(2)
        mvt_num_match = re.match(r"(\d+)", mvt_raw)
        mvt_abbrev = re.sub(r"^\d+", "", mvt_raw)
        result.catalog_number_resolved = bwv
        result.movement_num = mvt_num_match.group(1).zfill(2) if mvt_num_match else "01"
        result.movement_name_decoded = decode_movement_abbrev(mvt_abbrev)
        result.silver_slug_stem = (
            f"bach_{bwv}_{result.movement_num}_{result.movement_name_decoded.lower()}"
            if bwv else f"bach_unknown_{stem}"
        )
        return result

    # --- Flute Partita: fp_{num}{abbrev} → BWV1013 ---
    fp_match = re.match(r"^fp_(.+)$", work_code)
    if fp_match:
        mvt_raw = fp_match.group(1)
        mvt_num_match = re.match(r"(\d+)", mvt_raw)
        mvt_abbrev = re.sub(r"^\d+", "", mvt_raw)
        result.catalog_number_resolved = "BWV1013"
        result.movement_num = mvt_num_match.group(1).zfill(2) if mvt_num_match else "01"
        result.movement_name_decoded = decode_movement_abbrev(mvt_abbrev)
        result.silver_slug_stem = (
            f"bach_BWV1013_{result.movement_num}_{result.movement_name_decoded.lower()}"
        )
        return result

    # --- Prelude-Fugue-Allegro: pfa_{num}{abbrev} → BWV998 ---
    pfa_match = re.match(r"^pfa_(.+)$", work_code)
    if pfa_match:
        mvt_raw = pfa_match.group(1)
        mvt_num_match = re.match(r"(\d+)", mvt_raw)
        mvt_abbrev = re.sub(r"^\d+", "", mvt_raw)
        result.catalog_number_resolved = "BWV998"
        result.movement_num = mvt_num_match.group(1).zfill(2) if mvt_num_match else "01"
        result.movement_name_decoded = decode_movement_abbrev(mvt_abbrev)
        result.silver_slug_stem = (
            f"bach_BWV998_{result.movement_num}_{result.movement_name_decoded.lower()}"
        )
        return result

    # --- BJS numbered works: bjs1031a/b/c → BWV1031 ---
    bjs_match = re.match(r"^bjs(\d{4})([a-z]?)$", work_code)
    if bjs_match:
        bwv_num = bjs_match.group(1)
        variant = bjs_match.group(2)
        result.catalog_number_resolved = f"BWV{bwv_num}"
        result.movement_num = {"a": "01", "b": "02", "c": "03"}.get(variant, "01")
        result.silver_slug_stem = f"bach_BWV{bwv_num}_{result.movement_num}"
        return result

    # --- Explicit BWV in work_code: bwv772, jsbwv532 etc. ---
    bwv_from_code = ""
    bwv_code_match = re.search(r"bwv(\d{1,4}[a-z]?)", work_code)
    if bwv_code_match:
        bwv_from_code = f"BWV{bwv_code_match.group(1).upper()}"

    resolved_bwv = bwv_from_code or bwv_from_ref

    # Movement suffix: bwv525_1, bwv539_1
    mvt_suffix_match = re.search(r"bwv\d+[a-z]?_(\d+)", work_code)
    if mvt_suffix_match:
        result.movement_num = mvt_suffix_match.group(1).zfill(2)

    result.catalog_number_resolved = resolved_bwv
    if resolved_bwv:
        mvt = f"_{result.movement_num}" if result.movement_num else ""
        result.silver_slug_stem = f"bach_{resolved_bwv}{mvt}"
    else:
        result.silver_slug_stem = f"bach_unknown_{stem}"
        result.decode_notes = "BWV not resolved"

    return result


# ---------------------------------------------------------------------------
# KunstDerFuge contrast composers decoder
# ---------------------------------------------------------------------------

_KDF_CONTRAST_RE = re.compile(
    r"^(\w+)_kunstderfuge_unknown_(.+?)_(?:\(c\)|\(nc\)).+$", re.IGNORECASE
)
_HWV_RE  = re.compile(r"hwv[_\-](\d{1,3}[a-z]?)", re.IGNORECASE)
_KIRK_RE = re.compile(r"k[_\-](\d{1,3})", re.IGNORECASE)

def decode_kunstderfuge_contrast(stem: str, composer: str) -> DecodedFilename:
    """
    Decode a KunstDerFuge contrast composer filename stem.

    Args:
        stem: Filename stem.
        composer: Composer name (handel, scarlatti, vivaldi, etc.)

    Returns:
        DecodedFilename with KEEP_CONTRAST action.
    """
    result = DecodedFilename()
    lower = stem.lower()

    match = _KDF_CONTRAST_RE.match(lower)
    descriptor = match.group(2) if match else lower

    # --- Handel HWV ---
    if composer == "handel":
        hwv_match = _HWV_RE.search(descriptor)
        if hwv_match:
            result.catalog_number_resolved = f"HWV{hwv_match.group(1).upper()}"
            after_hwv = descriptor[hwv_match.end():]
            mvt_match = re.match(r"[_\-](\d+)[_\-]?(\w+)?", after_hwv)
            if mvt_match:
                result.movement_num = mvt_match.group(1).zfill(2)
                result.movement_name_decoded = (
                    mvt_match.group(2).capitalize() if mvt_match.group(2) else ""
                )
            result.silver_slug_stem = (
                f"handel_{result.catalog_number_resolved}"
                f"_{result.movement_num}"
                f"_{result.movement_name_decoded.lower()}"
            ).rstrip("_")
            return result

    # --- Scarlatti K. (Kirkpatrick) ---
    if composer == "scarlatti":
        k_match = _KIRK_RE.search(descriptor)
        if k_match:
            result.catalog_number_resolved = f"K{k_match.group(1).zfill(3)}"
            result.silver_slug_stem = f"scarlatti_{result.catalog_number_resolved}"
            return result

    # --- Vivaldi / Telemann / Couperin / Rameau ---
    clean_descriptor = re.sub(r"[^a-z0-9_]+", "_", descriptor).strip("_")
    result.catalog_number_resolved = ""
    result.silver_slug_stem = f"{composer}_{clean_descriptor}"
    result.decode_notes = "no_catalog_system"
    return result


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def decode_filename(stem: str, source_id: str, composer: str) -> DecodedFilename:
    """
    Route filename decoding to the appropriate per-source parser.

    Args:
        stem: Filename without extension.
        source_id: From source_detector.detect_source().
        composer: Composer name.

    Returns:
        DecodedFilename.
    """
    if is_corrupted_filename(stem):
        result = DecodedFilename()
        result.is_quarantine = True
        result.decode_notes = "corrupted_filename:http_url_as_name"
        result.silver_slug_stem = "QUARANTINE"
        return result

    if source_id.startswith("kernscores"):
        return decode_kernscores(stem)

    if source_id == "kunstderfuge_js":
        return decode_kunstderfuge_js(stem)

    if source_id.startswith("kunstderfuge_") and composer != "bach":
        return decode_kunstderfuge_contrast(stem, composer)

    if source_id.startswith("bulk_") or source_id == "openscore":
        return decode_kernscores(stem)

    result = DecodedFilename()
    result.decode_notes = f"no_decoder_for_source:{source_id}"
    result.silver_slug_stem = f"{composer}_unknown_{stem}"
    return result