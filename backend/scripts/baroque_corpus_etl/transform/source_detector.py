# scripts/baroque_corpus_etl/transform/source_detector.py
"""
Source Detector — Bach Propagation ETL Pipeline
================================================
Determines file provenance purely from folder path structure.
Zero I/O, zero music21. Pure path string operations.

Source taxonomy:
    kernscores_*    → data/raw/bach/{collection}/
    kunstderfuge_js → data/raw/bach_js/bach/
    bulk_chorales_* → data/raw/bulk_xml/bach-chorales-{370|371}/
    openscore       → data/raw/bulk_xml/openscore-bach-chorales/
    kunstderfuge_*  → data/raw/{composer}/kunstderfuge/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceInfo:
    """Provenance result for a single file path."""
    source_id: str
    composer: str
    collection: str   # human-readable collection name
    raw_subdir: str   # immediate parent folder name (useful for debugging)


# Maps data/raw/bach/{subdir} → source_id
_KERNSCORES_BACH_SUBDIRS: dict[str, str] = {
    "inventions":        "kernscores_inventions",
    "chorales_371":      "kernscores_chorales",
    "wtc":               "kernscores_wtc",
    "art_of_the_fugue":  "kernscores_art_of_fugue",
    "musical_offering":  "kernscores_musical_offering",
    "brandenburg":       "kernscores_brandenburg",
}

_COLLECTION_NAMES: dict[str, str] = {
    "kernscores_inventions":       "Two-Part Inventions & Sinfonias",
    "kernscores_chorales":         "371 Bach Chorales",
    "kernscores_wtc":              "Well-Tempered Clavier",
    "kernscores_art_of_fugue":     "Art of the Fugue",
    "kernscores_musical_offering": "Musical Offering",
    "kernscores_brandenburg":      "Brandenburg Concertos",
    "kunstderfuge_js":             "KunstDerFuge Bach MIDI",
    "bulk_chorales_370":           "Bulk Chorales 370 (DUPLICATE)",
    "bulk_chorales_371":           "Bulk Chorales 371",
    "openscore":                   "OpenScore Bach Chorales",
    "kunstderfuge_handel":         "KunstDerFuge Handel",
    "kunstderfuge_scarlatti":      "KunstDerFuge Scarlatti",
    "kunstderfuge_vivaldi":        "KunstDerFuge Vivaldi",
    "kunstderfuge_telemann":       "KunstDerFuge Telemann",
    "kunstderfuge_couperin":       "KunstDerFuge Couperin",
    "kunstderfuge_rameau":         "KunstDerFuge Rameau",
    "unknown":                     "Unknown Source",
}


def detect_source(file_path: Path) -> SourceInfo:
    """
    Determine the provenance of a file from its folder path alone.

    Resolution order (most specific → least specific):
        1. bach_js/bach/              → kunstderfuge_js
        2. bulk_xml/bach-chorales-370 → bulk_chorales_370
        3. bulk_xml/bach-chorales-371 → bulk_chorales_371
        4. bulk_xml/openscore-*       → openscore
        5. bach/{subdir}/             → kernscores_{subdir}
        6. {composer}/kunstderfuge/   → kunstderfuge_{composer}
        7. fallback                   → unknown

    Args:
        file_path: Absolute or relative path to the music file.

    Returns:
        SourceInfo with source_id, composer, collection, raw_subdir.
    """
    parts = file_path.parts
    raw_subdir = file_path.parent.name

    # Normalize: find index of 'raw' in path to make detection path-root agnostic
    try:
        raw_idx = next(i for i, p in enumerate(parts) if p == "raw")
    except StopIteration:
        return SourceInfo("unknown", "unknown", "Unknown Source", raw_subdir)

    relative = parts[raw_idx + 1:]  # everything after 'raw/'

    if not relative:
        return SourceInfo("unknown", "unknown", "Unknown Source", raw_subdir)

    top = relative[0]  # e.g. 'bach', 'bach_js', 'bulk_xml', 'handel'

    # --- bach_js ---
    if top == "bach_js":
        return SourceInfo(
            source_id="kunstderfuge_js",
            composer="bach",
            collection=_COLLECTION_NAMES["kunstderfuge_js"],
            raw_subdir=raw_subdir,
        )

    # --- bulk_xml ---
    if top == "bulk_xml" and len(relative) >= 2:
        subdir = relative[1]
        if "370" in subdir:
            sid = "bulk_chorales_370"
        elif "371" in subdir:
            sid = "bulk_chorales_371"
        elif "openscore" in subdir.lower():
            sid = "openscore"
        else:
            sid = "unknown"
        return SourceInfo(
            source_id=sid,
            composer="bach",
            collection=_COLLECTION_NAMES.get(sid, "Unknown"),
            raw_subdir=raw_subdir,
        )

    # --- data/raw/bach/{collection}/ (KernScores) ---
    if top == "bach" and len(relative) >= 2:
        collection_dir = relative[1]
        # Match collection dir to known KernScores collections
        for key, source_id in _KERNSCORES_BACH_SUBDIRS.items():
            if collection_dir == key or collection_dir.startswith(key):
                return SourceInfo(
                    source_id=source_id,
                    composer="bach",
                    collection=_COLLECTION_NAMES[source_id],
                    raw_subdir=raw_subdir,
                )
        # bach/ folder but unrecognised subcollection
        return SourceInfo(
            source_id="kernscores_unknown",
            composer="bach",
            collection=f"KernScores Unknown ({collection_dir})",
            raw_subdir=raw_subdir,
        )

    # --- {composer}/kunstderfuge/ ---
    known_contrast = {
        "handel", "scarlatti", "vivaldi", "telemann", "couperin", "rameau"
    }
    if top in known_contrast and len(relative) >= 2 and relative[1] == "kunstderfuge":
        sid = f"kunstderfuge_{top}"
        return SourceInfo(
            source_id=sid,
            composer=top,
            collection=_COLLECTION_NAMES.get(sid, f"KunstDerFuge {top.capitalize()}"),
            raw_subdir=raw_subdir,
        )

    return SourceInfo("unknown", "unknown", "Unknown Source", raw_subdir)