"""
Extractor for J.S. Bach MIDI files from jsbach.net.

jsbach.net is the definitive freely-available MIDI reference for Bach's output.
This extractor covers ALL major categories on the site — cello, keyboard,
violin, organ, orchestral, cantatas, and chamber — to maximise Bach coverage.

Design
------
- Inherits ``BaseExtractor`` for retry, idempotency, encoding-safe fetch, and
  failure log.
- Each page is a simple HTML document with anchor tags pointing to ``.mid``
  files.  No JavaScript rendering or pagination is required.
- BWV numbers are extracted from the link text / surrounding context using the
  same regex pattern as ``catalog_slug`` in the base module.
- Files are saved under a canonical subdirectory of ``data/raw/bach_js/``:
    ``data/raw/bach_js/cello/``       ← Solo Cello Suites (BWV 1007–1012)
    ``data/raw/bach_js/keyboard/``    ← Well-Tempered Clavier, Partitas, etc.
    ``data/raw/bach_js/violin/``      ← Violin Sonatas & Partitas
    ``data/raw/bach_js/organ/``       ← Organ works (Toccatas, Preludes, etc.)
    ``data/raw/bach_js/orchestral/``  ← Brandenburg Concertos, Suites
    ``data/raw/bach_js/cantatas/``    ← Vocal cantatas BWV 1–200+
    ``data/raw/bach_js/chamber/``     ← Sonatas, Suites (mixed ensemble)
- Filename convention: ``bach_<bwv>_<slug>_full.mid``

Discovery strategy
------------------
The site index at ``http://www.jsbach.net/midi/index.html`` lists 26 sub-pages
using a ``midi_*.html`` naming convention (e.g. ``midi_solo_cello.html``,
``midi_organ.html``).  The old hardcoded list (``cello.html``,
``keyboard.html``, …) returned 404 for every URL.

The extractor now discovers all sub-pages **dynamically** from the index so
it is robust to the site adding or renaming pages in future.  Each discovered
sub-page is fetched and all ``.mid`` links are collected.

Confirmed real file counts (2026-04-04 live probe):
    midi_solo_cello.html          36
    midi_solo_violin.html         31
    midi_organ.html               52
    midi_goldbergvariations.html  33
    midi_english_suites.html      48
    midi_wtc2.html                48
    midi_mass_in_b.html           23
    … (26 sub-pages, 363 total)

Usage
-----
Run all pages (default)::

    cd backend
    uv run python scripts/baroque_corpus_etl/extract/download_jsbach_net.py

Run in dry-run mode (lists sub-pages without downloading)::

    uv run python scripts/baroque_corpus_etl/extract/download_jsbach_net.py --dry-run

"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup

# Allow running directly from the scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from utils.network import BaseExtractor, build_destination, slugify

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The MIDI section root.  index.html lists all 26 sub-pages.
BASE_URL  = "http://www.jsbach.net"
INDEX_URL = "http://www.jsbach.net/midi/index.html"

#: Regex to pull a BWV number from a string such as
#: "BWV 1007", "bwv1010", "BWV 543a", "BWV 227/4".
_BWV_RE = re.compile(r"\bBWV\s*(\d+[a-z]?(?:[./]\d+)?)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# HTML parsing helpers (pure functions — no network I/O)
# ---------------------------------------------------------------------------


def _extract_bwv(text: str) -> str | None:
    """Return the first BWV token found in *text*, or ``None``.

    >>> _extract_bwv("Suite No. 1 for Cello BWV 1007")
    'BWV1007'
    >>> _extract_bwv("Nothing here") is None
    True
    """
    m = _BWV_RE.search(text)
    if not m:
        return None
    return f"BWV{m.group(1)}"


def _discover_subpages(html: str, index_url: str) -> list[str]:
    """Return all HTML sub-page URLs found in the jsbach.net MIDI index.

    Only links under the ``/midi/`` directory on the same host are returned.
    The index itself is excluded.  Duplicates are removed preserving order.

    >>> pages = _discover_subpages('<a href="midi_organ.html">Organ</a>', INDEX_URL)
    >>> pages[0].endswith('midi_organ.html')
    True
    """
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    result: list[str] = []

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        abs_url = urljoin(index_url, href)
        # Keep only HTML pages in the same /midi/ directory, not the index itself.
        if (
            abs_url not in seen
            and abs_url != index_url
            and "jsbach.net/midi/" in abs_url
            and abs_url.lower().endswith((".html", ".htm"))
        ):
            seen.add(abs_url)
            result.append(abs_url)

    return result


def _parse_midi_links(html: str, page_url: str) -> list[dict[str, str]]:
    """Return all ``.mid`` links found on a jsbach.net sub-page.

    Each entry is a dict::

        {
            "url":   "http://www.jsbach.net/midi/cs1-1pre.mid",
            "title": "Cello Suite No. 1 Prelude BWV 1007",
            "bwv":   "BWV1007",   # or "unknown"
        }

    Strategy: walk every ``<a href="*.mid">`` tag.  Prefer the link's own
    text; if it's empty, scan the parent element's text.  This copes with
    both ``<a>BWV 1007</a>`` patterns and table-cell layouts where the label
    is in a sibling cell.
    """
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if not href.lower().endswith((".mid", ".midi")):
            continue

        absolute_url = urljoin(page_url, href)
        if absolute_url in seen_urls:
            continue
        seen_urls.add(absolute_url)

        # Prefer link text; fall back to parent-element text.
        link_text = a.get_text(" ", strip=True)
        if not link_text and a.parent:
            link_text = a.parent.get_text(" ", strip=True)

        bwv = _extract_bwv(link_text) or _extract_bwv(href)

        results.append({
            "url":   absolute_url,
            "title": link_text or Path(href).stem,
            "bwv":   bwv or "unknown",
        })

    return results


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class JSBachNetExtractor(BaseExtractor):
    """Download all Bach MIDI files from jsbach.net.

    Crawl strategy (dynamic, not hardcoded):
    1. Fetch ``INDEX_URL`` (``/midi/index.html``).
    2. Collect every ``/midi/*.html`` sub-page linked from the index.
    3. Fetch each sub-page and download all ``.mid`` links found on it.

    All files land flat in ``data/raw/bach_js/`` with canonical
    ``build_destination`` naming.  A single output directory is used because
    the site doesn't cleanly separate by instrument — many pages cross-link
    to the same MIDI files (e.g. Art of Fugue appears on both the Art of
    Fugue page and John Sankey's page).  The ``is_downloaded`` idempotency
    guard in ``download_file`` de-duplicates these naturally.

    Args:
        staging_root: Root of the local staging area (default: ``data/raw``).
        dry_run:      List discovered sub-pages without downloading anything.
    """

    def __init__(
        self,
        staging_root: Path = Path("data/raw"),
        dry_run: bool = False,
    ) -> None:
        super().__init__(staging_root=staging_root)
        self.dry_run = dry_run
        # All jsbach.net files land under data/raw/bach_js/
        self._js_root = self.staging_root / "bach_js"
        self._js_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Discover all jsbach.net sub-pages from the index, then download."""
        # jsbach.net is HTTP-only (no TLS certificate).
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(
            headers=self._session_headers, connector=connector
        ) as session:
            subpages = await self._discover(session)

            if self.dry_run:
                print(f"\n  [dry-run] Would fetch {len(subpages)} sub-pages:")
                for p in subpages:
                    print(f"    {p}")
                return

            total_ok = total_fail = 0
            for i, page_url in enumerate(subpages, 1):
                ok, fail = await self._process_page(session, page_url, i, len(subpages))
                total_ok   += ok
                total_fail += fail

        failed_path = self.write_failed_log()
        if failed_path:
            print(f"\n  ⚠  {len(self._failed)} failure(s) logged → {failed_path}")

        print(f"\n{'=' * 60}")
        print(f"JSBACH.NET — Complete: {total_ok} downloaded, {total_fail} failed")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — discover sub-pages from index
    # ------------------------------------------------------------------

    async def _discover(self, session: aiohttp.ClientSession) -> list[str]:
        """Fetch INDEX_URL and return the list of MIDI sub-page URLs."""
        print(f"\n  Fetching index: {INDEX_URL}")
        try:
            html = await self._fetch_text(session, INDEX_URL)
        except Exception as exc:
            logger.error("Could not fetch jsbach.net index: %s", exc)
            print(f"  ✗ Index fetch failed: {exc}")
            return []

        pages = _discover_subpages(html, INDEX_URL)
        print(f"  Discovered {len(pages)} sub-page(s) from index")
        return pages

    # ------------------------------------------------------------------
    # Step 2 — process one sub-page
    # ------------------------------------------------------------------

    async def _process_page(
        self,
        session: aiohttp.ClientSession,
        page_url: str,
        idx: int,
        total: int,
    ) -> tuple[int, int]:
        """Fetch *page_url*, parse MIDI links, and download each one.

        Returns ``(ok_count, failed_count)``.
        """
        page_name = page_url.rsplit("/", 1)[-1]
        print(f"\n  [{idx:>2}/{total}] {page_name}")

        try:
            html = await self._fetch_text(session, page_url)
            await self._polite_delay(None)      # jitter between page fetches
        except aiohttp.ClientResponseError as exc:
            if exc.status == 404:
                logger.info("Page not found (404): %s", page_url)
                print(f"    ↷ 404 — skipping")
                return 0, 0
            logger.error("HTTP %d fetching %s", exc.status, page_url)
            print(f"    ✗ HTTP {exc.status}")
            return 0, 0
        except Exception as exc:
            logger.error("Could not fetch %s: %s", page_url, exc)
            print(f"    ✗ Fetch failed: {exc}")
            return 0, 0

        links = _parse_midi_links(html, page_url)
        if not links:
            logger.debug("No MIDI links on %s", page_url)
            print(f"    0 MIDI links")
            return 0, 0

        print(f"    {len(links)} MIDI link(s) — downloading …")
        ok = fail = 0
        for link in links:
            dest = build_destination(
                self._js_root,
                "bach",
                link["title"] or link["bwv"],
                ".mid",
                catalog_no=link["bwv"] if link["bwv"] != "unknown" else None,
            )
            success = await self.download_file(session, link["url"], dest, crawl_delay=None)
            if success:
                ok += 1
            else:
                fail += 1

        print(f"    ✓ {ok}  ✗ {fail}")
        return ok, fail

    # ------------------------------------------------------------------
    # Catalog generation
    # ------------------------------------------------------------------

    def write_catalog(self) -> Path:
        """Write a JSON catalog of all downloaded files and return its path."""
        import json

        midi_files = sorted(self._js_root.rglob("*.mid"))
        catalog: dict = {
            "source": "jsbach.net",
            "index_url": INDEX_URL,
            "composer": "J.S. Bach",
            "midi_count": len(midi_files),
            "files": [
                {
                    "name": f.name,
                    "size_bytes": f.stat().st_size,
                    "bwv": (_extract_bwv(f.stem) or "unknown"),
                }
                for f in midi_files
            ],
        }

        catalog_path = self._js_root / "catalog_jsbach_net.json"
        catalog_path.write_text(
            json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"\n  Catalog written: {catalog_path}  ({len(midi_files)} files)")
        return catalog_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Download all Bach MIDI files from jsbach.net (363+ files across 26 pages)"
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory for downloaded files (default: data/raw)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered sub-pages without downloading",
    )
    parser.add_argument(
        "--catalog",
        action="store_true",
        help="Write a JSON catalog after downloading",
    )
    args = parser.parse_args()

    extractor = JSBachNetExtractor(
        staging_root=args.staging_root,
        dry_run=args.dry_run,
    )
    asyncio.run(extractor.run())

    if args.catalog:
        extractor.write_catalog()
