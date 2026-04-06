"""
Async extractor for kunstderfuge.com — Bach + Baroque corpus.

kunstderfuge.com is the largest classical MIDI archive (~19,300 files,
~1,500 from Bach alone).  This extractor covers J.S. Bach and all other
major Baroque composers available on the site.

This module supersedes the original p2 script; ``download_kunstderfuge_p2.py``
now imports ``KunstDerFugeExtractor`` directly and configures it with a
non-Bach composer list.

Output layout
-------------
Files land at::

    data/raw/{composer}/kunstderfuge/{composer}_{source}_{catalog_no}_{stem}.mid

Example::

    data/raw/bach/kunstderfuge/bach_kunstderfuge_bwv846_wtc1_prelude_c.mid

The original URL stem is preserved as the *movement* field so full provenance
survives normalisation.  ``catalog_no`` is extracted via
``utils.network.catalog_slug`` (falls back to ``'unknown'`` when absent).

Concurrency
-----------
All MIDI links found on a composer page are dispatched as a single
``asyncio.gather`` batch, bounded to ``SEMAPHORE_LIMIT = 3`` concurrent
connections.  Each download is followed by a polite jitter delay (or the
site's ``Crawl-delay`` if declared in ``/robots.txt``).
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
    BaseExtractor,
    build_destination,
    catalog_slug,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Composer registry
# ---------------------------------------------------------------------------

#: All composers available on kunstderfuge.com that we want to download.
#: Key = URL slug (``/{slug}.htm``), value = display name.
#:
#: NOTE: "bach" is intentionally excluded here.
#: Live probing (2026-04-04) confirmed that kunstderfuge.com serves 0 parseable
#: MIDI links on bach.htm and midi.htm.  The site requires a paid subscription
#: to access MIDI files; the links are either dynamically injected or gated
#: behind an ASP login endpoint (``/-/subscription.htm``).
#: Bach MIDI is covered by jsbach.net (363+ files) and KernScores instead.
ALL_COMPOSERS: dict[str, str] = {
    "vivaldi":   "Antonio Vivaldi",
    "handel":    "George Frideric Handel",
    "telemann":  "Georg Philipp Telemann",
    "corelli":   "Arcangelo Corelli",
    "scarlatti": "Domenico Scarlatti",
    "purcell":   "Henry Purcell",
    "couperin":  "François Couperin",
    "rameau":    "Jean-Philippe Rameau",
}

#: Convenience subset retained for compatibility — now maps to an empty dict
#: since bach.htm yields 0 MIDI links (subscription paywall).
BACH_ONLY: dict[str, str] = {}

BASE_URL = "https://www.kunstderfuge.com"

# ---------------------------------------------------------------------------
# Pure parsing helpers
# ---------------------------------------------------------------------------


def _categorize_bach(link_text: str, href: str) -> str:
    """Heuristically bucket a Bach MIDI link into one of five work categories.

    Category is carried as metadata only — it does NOT affect the output
    path.  It can be used during the Transform phase to annotate features.
    """
    lt = link_text.lower()
    hr = href.lower()

    # Organ: explicit label OR BWV 5xx / 6xx / 7xx URL fragments.
    if "organ" in lt or "/organ" in hr or "bwv5" in hr or "bwv6" in hr or "bwv7" in hr:
        return "organ"

    # Vocal: cantatas, masses, passions, motets.
    if any(kw in lt for kw in ("cantata", "mass", "passion", "motet", "magnificat")):
        return "vocal"

    # Orchestral: concertos, orchestral suites, Brandenburg.
    if any(kw in lt for kw in ("brandenburg", "concerto", "orchestral", "overture")):
        return "orchestral"

    # Chamber: solo strings / winds, sonatas, suites.
    if any(kw in lt for kw in ("cello", "violin", "flute", "sonata", "suite")):
        return "chamber"

    # Default: keyboard (WTC, Inventions, Goldberg, Partitas, etc.)
    return "keyboard"


def _parse_midi_links(
    html: str,
    base_url: str,
    composer_slug: str,
) -> list[dict[str, str]]:
    """Extract and de-duplicate MIDI download links from a composer page.

    Returns a list of ``{"url": ..., "category": ...}`` dicts where
    *category* is a work-type label (meaningful for Bach; ``"general"`` for
    other composers).
    """
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict[str, str]] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href: str = a["href"]

        # Accept links that point directly to a .mid file or contain a
        # /midi/ path segment (some entries use indirect sub-paths).
        if not (href.lower().endswith(".mid") or "/midi/" in href.lower()):
            continue

        url = urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)

        link_text = a.get_text(strip=True)
        category = (
            _categorize_bach(link_text, href)
            if composer_slug == "bach"
            else "general"
        )
        results.append({"url": url, "category": category})

    return results


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class KunstDerFugeExtractor(BaseExtractor):
    """Download MIDI files from kunstderfuge.com.

    Args:
        staging_root: Root of the local staging area (default: ``data/raw``).
        composers:    Dict ``{url_slug: display_name}``.  Defaults to
                      ``ALL_COMPOSERS`` when *None*.
    """

    def __init__(
        self,
        staging_root: Path = Path("data/raw"),
        composers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(staging_root=staging_root)
        self.composers = composers or ALL_COMPOSERS

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Download all configured composers from kunstderfuge.com."""
        connector = aiohttp.TCPConnector(ssl=True)
        async with aiohttp.ClientSession(
            headers=self._session_headers, connector=connector
        ) as session:
            crawl_delay = await self._get_crawl_delay(session, BASE_URL)

            for slug, display_name in self.composers.items():
                await self._process_composer(session, slug, display_name, crawl_delay)

        failed_path = self.write_failed_log()
        if failed_path:
            print(f"\n  ⚠  {len(self._failed)} failure(s) logged → {failed_path}")

        print(f"\n{'=' * 60}")
        print("KUNSTDERFUGE — Extraction complete")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Per-composer logic
    # ------------------------------------------------------------------

    async def _process_composer(
        self,
        session: aiohttp.ClientSession,
        slug: str,
        display_name: str,
        crawl_delay: float | None,
    ) -> None:
        catalog_url = f"{BASE_URL}/{slug}.htm"

        print(f"\n{'=' * 60}")
        print(f"Composer : {display_name}  →  {slug}.htm")
        print("=" * 60)

        try:
            html = await self._fetch_text(session, catalog_url)
            # Polite delay after fetching the index page itself.
            await self._polite_delay(crawl_delay)
        except Exception as exc:
            logger.error("Failed to fetch catalog for %s: %s", slug, exc)
            print(f"  ✗ Catalog page unavailable: {exc}")
            return

        midi_links = _parse_midi_links(html, BASE_URL, slug)
        print(f"  Found {len(midi_links)} MIDI link(s)")

        if not midi_links:
            if slug == "bach":
                logger.warning(
                    "0 MIDI links on bach.htm — kunstderfuge.com requires a paid "
                    "subscription to access Bach MIDI files. Use jsbach.net instead."
                )
                print(
                    "  ⚠  0 links (paywall detected). "
                    "Bach MIDI is sourced from jsbach.net — skipping."
                )
            else:
                logger.warning("No MIDI links found on %s.htm", slug)
            return

        # Dispatch all downloads concurrently; semaphore limits actual I/O.
        tasks = [
            self.download_file(
                session,
                item["url"],
                self._make_dest(slug, item["url"]),
                crawl_delay=crawl_delay,
            )
            for item in midi_links
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        n_ok     = sum(1 for r in results if r is True)
        n_failed = len(results) - n_ok
        print(f"  ✓ {n_ok:>4} file(s) ready   ✗ {n_failed} failed")

    # ------------------------------------------------------------------
    # Path construction
    # ------------------------------------------------------------------

    def _make_dest(self, composer_slug: str, url: str) -> Path:
        """Return the canonical destination path for one KunstDerFuge file.

        Uses ``build_destination`` from ``utils.network`` to enforce the
        staging layout and naming convention::

            data/raw/{composer}/kunstderfuge/{composer}_{source}_{catalog_no}_{stem}.mid

        The original URL filename stem is preserved as the *movement* field
        so that provenance survives normalisation.  A leading
        ``"{composer}_"`` prefix is stripped from the stem first to prevent
        doubling (e.g. ``"bach_bwv846"`` → stem ``"bwv846"``).
        """
        original_name = url.split("/")[-1]
        stem = Path(original_name).stem

        # Strip composer prefix if the site already baked it in.
        prefix = f"{composer_slug}_"
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]

        cat_no = catalog_slug(composer_slug, stem) or None

        return build_destination(
            self.staging_root,
            composer_slug,
            "kunstderfuge",   # logical source group → directory segment
            ".mid",
            catalog_no=cat_no,
            movement=stem,    # original stem as movement tag → unique filenames
        )


# ---------------------------------------------------------------------------
# CLI entry point (Bach only by default)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    extractor = KunstDerFugeExtractor(composers=BACH_ONLY)
    asyncio.run(extractor.run())
