"""
Async extractor for the Mutopia Project Baroque corpus.

Downloads MusicXML (.xml / .musicxml) and MIDI (.mid) files for a configurable
set of Baroque composers.  Humdrum (.krn) and PDF/LilyPond files are ignored.

Design
------
- Inherits ``BaseExtractor`` for retry, jitter, idempotency, and failure log.
- Composer is parsed *dynamically* from each piece's metadata table, so files
  land under the correct ``data/raw/{parsed_composer}/`` path even when the
  search index uses shorthand codes (e.g. "BachJS" → "J.S. Bach").
- Catalog number is extracted from the "Opus/Catalog" metadata field and
  passed to ``build_destination`` for canonical naming.
- Concurrency: pieces within a composer are fetched sequentially (each piece
  page is one request + per-file downloads).  File downloads within a piece
  use ``asyncio.gather`` so all formats are fetched in parallel subject to the
  semaphore.
- Clean exit: any failure is caught, recorded, and flushed to
  ``data/raw/failed_downloads.json`` at the end of ``run()``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from urllib.parse import urljoin
import aiohttp

from bs4 import BeautifulSoup

# Allow running directly from the scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from utils.network import (
    FORMAT_PRIORITY,
    BaseExtractor,
    catalog_slug,
    slugify,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.mutopiaproject.org"

#: Mutopia composer codes → canonical display name (fallback only; the page
#: metadata is the primary source of the composer name).
BAROQUE_COMPOSERS: dict[str, str] = {
    "BachJS":      "J.S. Bach",
    "VivaldiA":    "Antonio Vivaldi",
    "HandelGF":    "George Frideric Handel",
    "TelemannGP":  "Georg Philipp Telemann",
    "CorelliA":    "Arcangelo Corelli",
    "ScarlattiD":  "Domenico Scarlatti",
    "PurcellH":    "Henry Purcell",
    "CoupFr":      "François Couperin",
    "RameauJP":    "Jean-Philippe Rameau",
    "MonteverdiC": "Claudio Monteverdi",
    "BuxtehudeD":  "Dieterich Buxtehude",
}

#: File extensions we want to download (in priority order).
_WANTED_EXTS: frozenset[str] = frozenset({".xml", ".musicxml", ".mid", ".midi"})


# ---------------------------------------------------------------------------
# HTML parsing helpers (pure functions — no network I/O)
# ---------------------------------------------------------------------------


#: Binary / non-HTML extensions that appear as links in the search results
#: table but are not piece-detail pages.  Following these would cause UTF-8
#: decode errors because aiohttp tries to decode them as text.
_BINARY_EXTS: frozenset[str] = frozenset({
    ".zip", ".gz", ".bz2", ".tar",          # archives
    ".ps", ".eps",                           # PostScript
    ".pdf",                                  # PDF
    ".ly", ".mscz", ".sib",                 # notation source files
    ".png", ".jpg", ".jpeg", ".gif", ".svg", # images
    ".mp3", ".ogg", ".flac",                 # audio
    ".mid", ".midi",                         # MIDI (piece pages, not score links)
})


def _parse_search_results(html: str, base_url: str) -> list[dict[str, str]]:
    """Return a list of ``{title, url}`` dicts from the composer search page.

    Only links whose target does **not** end in a known binary extension are
    included.  This filters out direct-download links to archives, PostScript
    files, PDFs, and other non-HTML resources that appear in the results table
    and would cause UTF-8 decode errors when fetched as text.
    """
    soup = BeautifulSoup(html, "html.parser")
    pieces: list[dict[str, str]] = []

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        link = cells[0].find("a", href=True)
        if not link:
            continue

        href = link["href"]
        # Skip any link whose path ends with a binary extension.
        if Path(href.split("?")[0]).suffix.lower() in _BINARY_EXTS:
            continue

        pieces.append({
            "title": cells[0].get_text(strip=True),
            "url":   urljoin(base_url, href),
        })

    return pieces


def _parse_piece_page(html: str, base_url: str, fallback_composer: str) -> dict:
    """
    Extract structured metadata and downloadable file links from a piece page.

    Returns a dict with keys:
      - ``composer``: parsed from the metadata table, or *fallback_composer*
      - ``title``:    parsed from the metadata table or ``<h2>``
      - ``catalog_no``: from the "Opus/Catalog" row, or ``None``
      - ``files``:    list of ``{url, ext}`` dicts, sorted by ``FORMAT_PRIORITY``
    """
    soup = BeautifulSoup(html, "html.parser")

    # ---- metadata table -----------------------------------------------
    # Mutopia piece pages contain a definition-style table:
    #   <td>Composer</td><td>J.S. Bach</td>
    #   <td>Title</td><td>Prelude in C major</td>
    #   <td>Opus/Catalog</td><td>BWV 846</td>
    meta: dict[str, str] = {}
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 2:
            key = cells[0].get_text(strip=True).lower().rstrip(":")
            val = cells[1].get_text(strip=True)
            if key and val:
                meta[key] = val

    composer  = meta.get("composer")  or meta.get("arranger") or fallback_composer
    title_raw = meta.get("title")
    catalog   = meta.get("opus/catalog") or meta.get("opus") or meta.get("catalog")

    # Fall back to <h2> for title if not in table.
    if not title_raw:
        h2 = soup.find("h2")
        title_raw = h2.get_text(strip=True) if h2 else "unknown"

    # ---- downloadable files -------------------------------------------
    raw_files: list[dict[str, str | int]] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        ext = Path(href).suffix.lower()
        if ext in _WANTED_EXTS:
            raw_files.append({
                "url": urljoin(base_url, href),
                "ext": ext,
                "priority": FORMAT_PRIORITY.get(ext, 99),
            })

    # De-duplicate by URL, then sort: MusicXML first, then MIDI.
    seen: set[str] = set()
    files: list[dict[str, str]] = []
    for f in sorted(raw_files, key=lambda x: x["priority"]):
        url = str(f["url"])
        if url not in seen:
            seen.add(url)
            files.append({"url": url, "ext": str(f["ext"])})

    return {
        "composer":   composer,
        "title":      title_raw,
        "catalog_no": catalog,
        "files":      files,
    }


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class MutopiaExtractor(BaseExtractor):
    """Download Baroque scores from the Mutopia Project.

    Args:
        staging_root: Root of the local staging area (default: ``data/raw``).
        composers:    Override the default ``BAROQUE_COMPOSERS`` dict.
    """

    def __init__(
        self,
        staging_root: Path = Path("data/raw"),
        composers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(staging_root=staging_root)
        self.composers = composers or BAROQUE_COMPOSERS

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Download all configured Baroque composers from Mutopia."""
        connector = aiohttp.TCPConnector(ssl=True)
        async with aiohttp.ClientSession(
            headers=self._session_headers, connector=connector
        ) as session:
            crawl_delay = await self._get_crawl_delay(session, BASE_URL)

            for code, display_name in self.composers.items():
                await self._process_composer(session, code, display_name, crawl_delay)

        failed_path = self.write_failed_log()
        if failed_path:
            print(f"\n  ⚠ {len(self._failed)} failure(s) logged → {failed_path}")

        print(f"\n{'=' * 60}")
        print("MUTOPIA BAROQUE — Complete")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Per-composer logic
    # ------------------------------------------------------------------

    async def _process_composer(
        self,
        session: aiohttp.ClientSession,
        code: str,
        display_name: str,
        crawl_delay: float | None,
    ) -> None:
        search_url = (
            f"{BASE_URL}/cgibin/make-table.cgi?Composer={code}"
        )

        print(f"\n{'=' * 60}")
        print(f"Composer : {display_name}  ({code})")
        print("=" * 60)

        try:
            html = await self._fetch_text(session, search_url)
            await self._polite_delay(crawl_delay)
        except Exception as exc:
            logger.error("Could not fetch catalog for %s: %s", code, exc)
            print(f"  ✗ Catalog fetch failed: {exc}")
            return

        pieces = _parse_search_results(html, BASE_URL)
        print(f"  Found {len(pieces)} piece(s)")

        for i, piece in enumerate(pieces, 1):
            print(f"  [{i:>3}/{len(pieces)}] {piece['title'][:60]}")
            await self._process_piece(session, piece, display_name, crawl_delay)

    # ------------------------------------------------------------------
    # Per-piece logic
    # ------------------------------------------------------------------

    async def _process_piece(
        self,
        session: aiohttp.ClientSession,
        piece: dict[str, str],
        fallback_composer: str,
        crawl_delay: float | None,
    ) -> None:
        try:
            html = await self._fetch_text(session, piece["url"])
            await self._polite_delay(crawl_delay)
        except Exception as exc:
            logger.error("Could not fetch piece page %s: %s", piece["url"], exc)
            print(f"       ✗ Page fetch failed: {exc}")
            return

        meta = _parse_piece_page(html, BASE_URL, fallback_composer)

        if not meta["files"]:
            logger.debug("No downloadable files for '%s'", meta["title"])
            return

        # Extract catalog number from the "Opus/Catalog" metadata field.
        # If it contains something like "BWV 846", catalog_slug will extract it.
        cat_no = (
            catalog_slug(meta["composer"], meta["catalog_no"])
            if meta["catalog_no"]
            else None
        )

        # Download all wanted files for this piece concurrently (semaphore-bounded).
        tasks = [
            self._download_score_file(
                session,
                file_info["url"],
                meta["composer"],
                meta["title"],
                file_info["ext"],
                cat_no,
                crawl_delay,
            )
            for file_info in meta["files"]
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # download_file returns True (success/skip) or False (permanent failure).
        # gather return_exceptions=True wraps unexpected errors as Exception objects.
        n_ok     = sum(1 for r in results if r is True)
        n_failed = len(results) - n_ok

        if n_failed:
            print(f"       ✗ {n_failed} file(s) failed")
        if n_ok:
            print(f"       ✓ {n_ok} file(s) ready")

    # ------------------------------------------------------------------
    # Single-file download
    # ------------------------------------------------------------------

    async def _download_score_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        composer: str,
        work_title: str,
        ext: str,
        catalog_no: str | None,
        crawl_delay: float | None,
    ) -> bool:
        dest = self.destination(
            composer,
            work_title,
            ext,
            catalog_no=catalog_no,
        )
        return await self.download_file(session, url, dest, crawl_delay=crawl_delay)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    extractor = MutopiaExtractor()
    asyncio.run(extractor.run())
