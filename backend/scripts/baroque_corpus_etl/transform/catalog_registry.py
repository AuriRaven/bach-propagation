# scripts/baroque_corpus_etl/transform/catalog_registry.py
"""
Catalog Registry — Bach Propagation ETL Pipeline
=================================================
Single source of truth for:
  - Composer catalog number patterns (BWV, HWV, K., RV, TWV)
  - KernScores abbreviation → BWV static lookup tables
  - Movement abbreviation → human-readable name decoder
  - Silver action policies per source

No I/O, no music21 — pure data. Import cost: ~0ms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Catalog number regex patterns (per composer)
# ---------------------------------------------------------------------------

CATALOG_PATTERNS: dict[str, re.Pattern] = {
    "bach":      re.compile(r"bwv[_\-\s]?(\d{1,4}[a-z]?)", re.IGNORECASE),
    "handel":    re.compile(r"hwv[_\-\s]?(\d{1,3}[a-z]?)", re.IGNORECASE),
    "scarlatti": re.compile(r"k[_\-\s]?(\d{1,3})",          re.IGNORECASE),
    "vivaldi":   re.compile(r"rv[_\-\s]?(\d{1,3})",          re.IGNORECASE),
    "telemann":  re.compile(r"twv[_\-\s]?(\d{1,2}:\d{1,4})", re.IGNORECASE),
}

CATALOG_PREFIX: dict[str, str] = {
    "bach":      "BWV",
    "handel":    "HWV",
    "scarlatti": "K",
    "vivaldi":   "RV",
    "telemann":  "TWV",
    "couperin":  "",   # no standard catalog
    "rameau":    "",   # no standard catalog
}

# ---------------------------------------------------------------------------
# KernScores collection codes → BWV lookup
# ---------------------------------------------------------------------------

# Two-Part Inventions: inven01–15 → BWV 772–786
KERNSCORES_INVENTIONS: Final[dict[str, str]] = {
    f"inven{i:02d}": f"BWV{771 + i}" for i in range(1, 16)
}

# Three-Part Sinfonias: sinfo01–15 → BWV 787–801
KERNSCORES_SINFONIAS: Final[dict[str, str]] = {
    f"sinfo{i:02d}": f"BWV{786 + i}" for i in range(1, 16)
}

# Cello Suites: cs1–cs6 → BWV 1007–1012
KERNSCORES_CELLO_SUITES: Final[dict[str, str]] = {
    f"cs{i}": f"BWV{1006 + i}" for i in range(1, 7)
}

# Violin Sonatas (odd: 1,3,5) and Partitas (even: 2,4,6)
# vs1=BWV1001, vp1=BWV1002, vs2=BWV1003, vp2=BWV1004, vs3=BWV1005, vp3=BWV1006
KERNSCORES_VIOLIN: Final[dict[str, str]] = {
    "vs1": "BWV1001", "vp1": "BWV1002",
    "vs2": "BWV1003", "vp2": "BWV1004",
    "vs3": "BWV1005", "vp3": "BWV1006",
}

# Well-Tempered Clavier Book II: wtcii01–24 (Book I is in separate wtc folder)
# BWV 870–893 (preludes=a, fugues=b)
KERNSCORES_WTC_II: Final[dict[str, str]] = {
    f"wtcii{i:02d}": f"BWV{869 + i}" for i in range(1, 25)
}

# Goldberg Variations: BWV 988
# 988_aria, 988_v01–30 all belong to BWV 988
KUNSTDERFUGE_JS_GOLDBERG_PREFIX: Final[str] = "BWV988"

# B Minor Mass movements: bjsbmm01–23 → BWV 232
KUNSTDERFUGE_JS_BMM_PREFIX: Final[str] = "BWV232"

# Musical Offering movements: 1079_XX → BWV 1079
KUNSTDERFUGE_JS_MUSICAL_OFFERING_PREFIX: Final[str] = "BWV1079"

# Art of Fugue: 1080_XX → BWV 1080
KUNSTDERFUGE_JS_ART_OF_FUGUE_PREFIX: Final[str] = "BWV1080"

# Lute Suite BWV 997
KUNSTDERFUGE_JS_LUTE_997_PREFIX: Final[str] = "BWV997"

# French Partita / Flute Partita — fp_ prefix (BWV 1013 or French suite)
# pfa_ prefix = Partita for flute/another instrument — needs NEEDS_REVIEW
KUNSTDERFUGE_JS_AMBIGUOUS_PREFIXES: Final[set[str]] = {"fp", "pfa"}

# Merge all KernScores lookup tables into one dict keyed by stem prefix
KERNSCORES_BWV_LOOKUP: Final[dict[str, str]] = {
    **KERNSCORES_INVENTIONS,
    **KERNSCORES_SINFONIAS,
    **KERNSCORES_CELLO_SUITES,
    **KERNSCORES_VIOLIN,
    **KERNSCORES_WTC_II,
}

# ---------------------------------------------------------------------------
# Movement abbreviation decoder
# ---------------------------------------------------------------------------

MOVEMENT_ABBREV: Final[dict[str, str]] = {
    # Standard dance movements
    "pre":  "Prelude",
    "fug":  "Fugue",
    "all":  "Allemande",
    "ald":  "Allemande_Double",
    "cou":  "Courante",
    "cod":  "Courante_Double",
    "sar":  "Sarabande",
    "sad":  "Sarabande_Double",
    "gig":  "Gigue",
    "men":  "Menuet",
    "bou":  "Bourree",
    "gav":  "Gavotte",
    "lou":  "Loure",
    "min":  "Minuet",
    "cha":  "Chaconne",
    "dou":  "Double",
    "tb":   "Tempo_di_Borea",
    "tbd":  "Tempo_di_Borea_Double",
    # Tempo markings used as movement names
    "ada":  "Adagio",
    "and":  "Andante",
    "alg":  "Allegro",
    "lar":  "Largo",
    "gra":  "Grave",
    "sic":  "Siciliana",
    "prs":  "Presto",
    "al":   "Allemande",   # short form in vp/vs files
    "co":   "Courante",    # short form
    "sa":   "Sarabande",   # short form
    # Goldberg-specific
    "aria": "Aria",
    # Art of Fugue
    "c":    "Contrapunctus",
    # Musical Offering
    "ric":  "Ricercar",
    # Other
    "sin":  "Sinfonia",
    "cho":  "Chorale",
}

def decode_movement_abbrev(abbrev: str) -> str:
    """
    Decode a movement abbreviation to a human-readable name.

    Strips leading digits (e.g. '1pre' → 'pre' → 'Prelude').

    Args:
        abbrev: Raw movement code from filename (e.g. '1pre', '2all', 'aria').

    Returns:
        Human-readable movement name, or the original abbrev if unknown.
    """
    clean = abbrev.lstrip("0123456789").lower()
    return MOVEMENT_ABBREV.get(clean, abbrev)

# ---------------------------------------------------------------------------
# Silver action policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourcePolicy:
    """Defines how a source's files should be treated in the Silver layer."""
    source_id: str
    composer: str
    role: str           # primary | contrast
    catalog_system: str # BWV | HWV | K | RV | TWV | none
    dedup_action: str   # KEEP | DUPLICATE_DROP
    base_silver_action: str  # KEEP_PRIMARY | KEEP_CONTRAST | DUPLICATE_DROP

SOURCE_POLICIES: Final[dict[str, SourcePolicy]] = {
    "kernscores_inventions": SourcePolicy(
        "kernscores_inventions", "bach", "primary", "BWV", "KEEP", "KEEP_PRIMARY"
    ),
    "kernscores_sinfonias": SourcePolicy(
        "kernscores_sinfonias", "bach", "primary", "BWV", "KEEP", "KEEP_PRIMARY"
    ),
    "kernscores_wtc": SourcePolicy(
        "kernscores_wtc", "bach", "primary", "BWV", "KEEP", "KEEP_PRIMARY"
    ),
    "kernscores_chorales": SourcePolicy(
        "kernscores_chorales", "bach", "primary", "BWV", "KEEP", "KEEP_PRIMARY"
    ),
    "kernscores_art_of_fugue": SourcePolicy(
        "kernscores_art_of_fugue", "bach", "primary", "BWV", "KEEP", "KEEP_PRIMARY"
    ),
    "kernscores_musical_offering": SourcePolicy(
        "kernscores_musical_offering", "bach", "primary", "BWV", "KEEP", "KEEP_PRIMARY"
    ),
    "kernscores_brandenburg": SourcePolicy(
        "kernscores_brandenburg", "bach", "primary", "BWV", "KEEP", "KEEP_PRIMARY"
    ),
    "kunstderfuge_js": SourcePolicy(
        "kunstderfuge_js", "bach", "primary", "BWV", "KEEP", "KEEP_PRIMARY"
    ),
    "bulk_chorales_371": SourcePolicy(
        "bulk_chorales_371", "bach", "primary", "BWV", "KEEP", "KEEP_PRIMARY"
    ),
    "bulk_chorales_370": SourcePolicy(
        "bulk_chorales_370", "bach", "primary", "BWV", "DUPLICATE_DROP", "DUPLICATE_DROP"
    ),
    "openscore": SourcePolicy(
        "openscore", "bach", "primary", "BWV", "KEEP", "KEEP_PRIMARY"
    ),
    "kunstderfuge_handel": SourcePolicy(
        "kunstderfuge_handel", "handel", "contrast", "HWV", "KEEP", "KEEP_CONTRAST"
    ),
    "kunstderfuge_scarlatti": SourcePolicy(
        "kunstderfuge_scarlatti", "scarlatti", "contrast", "K", "KEEP", "KEEP_CONTRAST"
    ),
    "kunstderfuge_vivaldi": SourcePolicy(
        "kunstderfuge_vivaldi", "vivaldi", "contrast", "RV", "KEEP", "KEEP_CONTRAST"
    ),
    "kunstderfuge_telemann": SourcePolicy(
        "kunstderfuge_telemann", "telemann", "contrast", "TWV", "KEEP", "KEEP_CONTRAST"
    ),
    "kunstderfuge_couperin": SourcePolicy(
        "kunstderfuge_couperin", "couperin", "contrast", "none", "KEEP", "KEEP_CONTRAST"
    ),
    "kunstderfuge_rameau": SourcePolicy(
        "kunstderfuge_rameau", "rameau", "contrast", "none", "KEEP", "KEEP_CONTRAST"
    ),
    "unknown": SourcePolicy(
        "unknown", "unknown", "unknown", "none", "KEEP", "NEEDS_REVIEW"
    ),
}

def get_policy(source_id: str) -> SourcePolicy:
    """Return the SourcePolicy for a given source_id, defaulting to 'unknown'."""
    return SOURCE_POLICIES.get(source_id, SOURCE_POLICIES["unknown"])