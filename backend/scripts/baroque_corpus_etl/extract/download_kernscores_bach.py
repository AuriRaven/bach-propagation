"""
Extractor for the complete Bach corpus on KernScores.

Downloads Humdrum (.krn) files via the ``humcat``/``humsplit`` CLI tools,
converts them to MIDI with ``hum2mid``, and writes a catalog JSON.

Inherits from ``BaseExtractor`` for:
  - Safe subprocess execution (no ``shell=True``)
  - Idempotency check via ``is_downloaded``
  - Canonical staging path construction
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow running directly from the scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from utils.network import BaseExtractor, is_downloaded

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collection manifest
# ---------------------------------------------------------------------------

#: Maps KernScores collection slug → human-readable description.
BACH_COLLECTIONS: dict[str, str] = {
    "chorales-371":   "371 chorales for 4 voices",
    "wtc":            "Well-Tempered Clavier (48 preludes + fugues)",
    "inventions":     "15 two-part inventions + 15 sinfonias",
    "art-of-the-fugue": "The Art of the Fugue",
    "brandenburg":    "6 Brandenburg Concertos",
    "musical-offering": "Musical Offering",
}

# KernScores URI prefix per collection (chorales live at root, rest under bach/).
_KERN_URI: dict[str, str] = {
    "chorales-371":     "h://chorales-371",
    "wtc":              "h://bach/wtc",
    "inventions":       "h://bach/inventions",
    "art-of-the-fugue": "h://bach/art-of-the-fugue",
    "brandenburg":      "h://bach/brandenburg",
    "musical-offering": "h://bach/musical-offering",
}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class KernScoresBachExtractor(BaseExtractor):
    """Download the complete Bach KernScores corpus.

    Uses ``humcat`` + ``humsplit`` (Humdrum Extras) to bulk-fetch each
    collection, then converts every ``.krn`` to MIDI with ``hum2mid``.

    Prerequisites (must be on PATH)::

        brew install humdrum-tools   # provides humcat, humsplit, hum2mid
    """

    def __init__(
        self,
        staging_root: Path = Path("data/raw"),
        collections: dict[str, str] | None = None,
    ) -> None:
        super().__init__(staging_root=staging_root)
        self.collections = collections or BACH_COLLECTIONS
        # All Bach Humdrum files land under data/raw/bach/
        self._bach_root = self.staging_root / "bach"
        self._bach_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Download and convert every configured Bach collection."""
        for slug, description in self.collections.items():
            print(f"\n{'=' * 60}")
            print(f"Collection : {description}")
            print(f"Slug       : {slug}")
            print("=" * 60)
            self._download_collection(slug)

        self.generate_catalog()

    # ------------------------------------------------------------------
    # Download + conversion
    # ------------------------------------------------------------------

    def _collection_path(self, slug: str) -> Path:
        """Return (and create) the output directory for *slug*."""
        path = self._bach_root / slug.replace("-", "_")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _collection_has_files(self, slug: str) -> bool:
        """Return True if the collection directory already contains .krn files."""
        path = self._collection_path(slug)
        return any(path.glob("*.krn"))

    def _download_collection(self, slug: str) -> None:
        """Bulk-download a collection with ``humcat | humsplit``."""
        output_path = self._collection_path(slug)

        if self._collection_has_files(slug):
            krn_count = sum(1 for _ in output_path.glob("*.krn"))
            print(f"  ✓ Skipped (already downloaded: {krn_count} .krn files)")
            self._batch_convert_to_midi(output_path)
            return

        uri = _KERN_URI[slug]

        rc, _out, err = self.run_pipe(
            ["humcat", "-s", uri],
            ["humsplit"],
            cwd=output_path,
            timeout=300,
        )

        if rc == 0:
            krn_files = list(output_path.glob("*.krn"))
            print(f"  ✓ Downloaded {len(krn_files)} .krn files")
            self._batch_convert_to_midi(output_path)
        else:
            logger.error("humcat/humsplit failed for %s: %s", slug, err.strip())
            print(f"  ✗ Error: {err.strip()}")

    def _batch_convert_to_midi(self, directory: Path) -> None:
        """Convert all ``.krn`` files in *directory* to MIDI using ``hum2mid``."""
        kern_files = list(directory.glob("*.krn"))
        if not kern_files:
            return

        print(f"  Converting {len(kern_files)} .krn → .mid ...")
        converted = 0

        for kern_file in kern_files:
            midi_file = kern_file.with_suffix(".mid")

            if is_downloaded(midi_file):
                converted += 1
                continue

            rc, _out, err = self.run_subprocess(
                ["hum2mid", str(kern_file), "-o", str(midi_file)],
                timeout=30,
            )
            if rc == 0:
                converted += 1
            else:
                logger.warning("hum2mid failed for %s: %s", kern_file.name, err.strip())
                print(f"    ✗ {kern_file.name}: {err.strip()}")

        print(f"  ✓ {converted}/{len(kern_files)} .mid files ready")

    # ------------------------------------------------------------------
    # Catalog generation
    # ------------------------------------------------------------------

    def generate_catalog(self) -> None:
        """Write a JSON catalog of all downloaded files to the bach root."""
        catalog: dict = {
            "source": "KernScores",
            "formats": ["Humdrum (.krn)", "MIDI (.mid)"],
            "composer": "J.S. Bach",
            "collections": {},
        }

        for slug in self.collections:
            coll_path = self._collection_path(slug)
            if not coll_path.exists():
                continue

            midi_files = sorted(coll_path.glob("*.mid"))
            krn_files  = sorted(coll_path.glob("*.krn"))

            catalog["collections"][slug] = {
                "description": self.collections[slug],
                "krn_count":   len(krn_files),
                "midi_count":  len(midi_files),
                # Sample of first 10 MIDI files for quick inspection.
                "midi_sample": [f.name for f in midi_files[:10]],
            }

        catalog_path = self._bach_root / "catalog_kernscores.json"
        catalog_path.write_text(
            json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        total_midi = sum(c["midi_count"] for c in catalog["collections"].values())
        total_krn  = sum(c["krn_count"]  for c in catalog["collections"].values())

        print(f"\n{'=' * 60}")
        print("KERNSCORES — BACH  |  Summary")
        print("=" * 60)
        print(f"  Collections : {len(catalog['collections'])}")
        print(f"  .krn files  : {total_krn}")
        print(f"  .mid files  : {total_midi}")
        print(f"  Catalog     : {catalog_path}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    extractor = KernScoresBachExtractor()
    asyncio.run(extractor.run())
