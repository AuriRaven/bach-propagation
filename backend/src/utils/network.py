"""
Async base extractor for the Baroque corpus ETL pipeline.

Provides
--------
- Pure helper functions for slug/path generation (no side-effects, testable).
- ``BaseExtractor`` ABC:
    - ``aiohttp`` session + per-instance ``asyncio.Semaphore``.
    - Retry/back-off via ``tenacity`` on every network call.
    - Idempotency: skip files already on disk with size > 0.
    - Canonical staging layout: ``data/raw/{composer}/{work}/{filename}``.
    - Naming convention: ``{composer}_{work}_{catalog_no}_{movement}{ext}``.
    - Safe subprocess helpers (no ``shell=True``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import subprocess
import unicodedata
import urllib.robotparser
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Optional
from urllib.parse import urlparse

import aiohttp
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog-number prefix registry
# ---------------------------------------------------------------------------

#: Maps composer slug → ordered list of catalog prefixes to search for.
CATALOG_PREFIXES: dict[str, list[str]] = {
    "bach":       ["BWV"],
    "handel":     ["HWV"],
    "vivaldi":    ["RV"],
    "scarlatti":  ["K"],
    "telemann":   ["TWV"],
    "corelli":    ["Op"],
    "purcell":    ["Z"],
    "couperin":   ["Op"],
    "rameau":     ["RCT"],
    "monteverdi": ["SV"],
    "buxtehude":  ["BuxWV"],
}

# ---------------------------------------------------------------------------
# Format priority (lower = preferred)
# ---------------------------------------------------------------------------

FORMAT_PRIORITY: dict[str, int] = {
    ".xml":      1,
    ".musicxml": 1,
    ".krn":      2,
    ".mid":      3,
    ".midi":     3,
}

# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^\w]+")


def slugify(text: str) -> str:
    """Normalise *text* to a safe ASCII lowercase filename segment.

    Accented characters are decomposed to their ASCII base; everything that is
    not a word character is collapsed to a single underscore.

    >>> slugify("Well-Tempered Clavier")
    'well_tempered_clavier'
    >>> slugify("François Couperin")
    'francois_couperin'
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _SLUG_RE.sub("_", text.lower()).strip("_")
    return text or "unknown"


def catalog_slug(composer: str, raw_title: str) -> str:
    """Extract the catalog number token from *raw_title* for *composer*.

    Returns the canonical token (e.g. ``'BWV846'``, ``'RV269'``) or
    ``'unknown'`` when no match is found.

    >>> catalog_slug("bach", "Prelude No.1 BWV 846")
    'BWV846'
    >>> catalog_slug("vivaldi", "The Four Seasons RV 269")
    'RV269'
    >>> catalog_slug("bach", "Some untitled piece")
    'unknown'
    """
    prefixes = CATALOG_PREFIXES.get(composer.lower(), [])
    for prefix in prefixes:
        pattern = rf"\b{re.escape(prefix)}\s*(\d+[a-z]?(?:[./]\d+)?)"
        m = re.search(pattern, raw_title, re.IGNORECASE)
        if m:
            return f"{prefix.upper()}{m.group(1)}"
    return "unknown"


def build_filename(
    composer: str,
    work_title: str,
    ext: str,
    *,
    catalog_no: Optional[str] = None,
    movement: Optional[str | int] = None,
) -> str:
    """Build the canonical filename for a score file.

    Pattern: ``{composer}_{work_slug}_{catalog_no}_{movement}{ext}``

    Missing *catalog_no* falls back to ``'unknown'``; missing *movement*
    falls back to ``'full'``.  *ext* is normalised to include the leading dot.

    >>> build_filename("bach", "Prelude", ".xml", catalog_no="BWV846", movement=1)
    'bach_prelude_bwv846_1.xml'
    >>> build_filename("vivaldi", "Spring", ".mid")
    'vivaldi_spring_unknown_full.mid'
    """
    composer_s = slugify(composer)
    work_s = slugify(work_title)[:40]
    cat_s = slugify(catalog_no) if catalog_no else "unknown"
    mov_s = str(movement) if movement is not None else "full"
    ext = ext if ext.startswith(".") else f".{ext}"
    return f"{composer_s}_{work_s}_{cat_s}_{mov_s}{ext}"


def build_destination(
    root: Path,
    composer: str,
    work_title: str,
    ext: str,
    *,
    catalog_no: Optional[str] = None,
    movement: Optional[str | int] = None,
) -> Path:
    """Return the full destination ``Path`` under *root*, creating parent dirs.

    Layout: ``{root}/{composer_slug}/{work_slug[:40]}/{filename}``
    """
    work_dir = root / slugify(composer) / slugify(work_title)[:40]
    work_dir.mkdir(parents=True, exist_ok=True)
    filename = build_filename(
        composer, work_title, ext, catalog_no=catalog_no, movement=movement
    )
    return work_dir / filename


def is_downloaded(path: Path) -> bool:
    """Return ``True`` if *path* already exists on disk with size > 0 bytes."""
    return path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# BaseExtractor
# ---------------------------------------------------------------------------


class BaseExtractor(ABC):
    """Abstract base class for all Baroque corpus extractors.

    Subclasses must implement ``run()``.  HTTP-based subclasses should use
    ``download_file`` / ``_fetch_bytes`` / ``_fetch_text``; CLI-based ones
    (e.g. KernScores) should use ``run_subprocess`` / ``run_pipe``.

    Usage (HTTP subclass)::

        class MyExtractor(BaseExtractor):
            async def run(self) -> None:
                async with aiohttp.ClientSession(headers=self._session_headers) as session:
                    dest = self.destination("bach", "Prelude", ".xml",
                                            catalog_no="BWV846", movement=1)
                    await self.download_file(session, url, dest)

        asyncio.run(MyExtractor().run())

    Usage (subprocess subclass)::

        class KernExtractor(BaseExtractor):
            async def run(self) -> None:
                rc, out, err = self.run_pipe(
                    ["humcat", "-s", "h://bach/wtc"],
                    ["humsplit"],
                    cwd=output_path,
                )
    """

    #: Maximum concurrent HTTP connections — kept low to respect volunteer servers.
    SEMAPHORE_LIMIT: ClassVar[int] = 3

    def __init__(
        self,
        staging_root: Path = Path("data/raw"),
        semaphore_limit: Optional[int] = None,
    ) -> None:
        self.staging_root = staging_root
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self._semaphore = asyncio.Semaphore(semaphore_limit or self.SEMAPHORE_LIMIT)
        # Per-host robots.txt crawl-delay cache  (None = not set / not found).
        self._crawl_delays: dict[str, float | None] = {}
        # Accumulated download failures for this run.
        self._failed: list[dict] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self) -> None:
        """Execute the full extraction for this source."""

    # ------------------------------------------------------------------
    # Path / naming helpers
    # ------------------------------------------------------------------

    @property
    def _session_headers(self) -> dict[str, str]:
        return {
            "User-Agent": (
                "BachPropagation-ThesisBot/1.0 "
                "(Research Project; AI + Musicology; "
                "contact: aura6423@ciencias.unam.mx)"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            ),
        }

    def destination(
        self,
        composer: str,
        work_title: str,
        ext: str,
        *,
        catalog_no: Optional[str] = None,
        movement: Optional[str | int] = None,
    ) -> Path:
        """Return the canonical destination path for a score file."""
        return build_destination(
            self.staging_root,
            composer,
            work_title,
            ext,
            catalog_no=catalog_no,
            movement=movement,
        )

    # ------------------------------------------------------------------
    # HTTP helpers (with tenacity retry)
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _fetch_bytes(
        self, session: aiohttp.ClientSession, url: str
    ) -> bytes:
        """Fetch *url* and return raw bytes. Retries up to 5× with backoff."""
        async with self._semaphore:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                resp.raise_for_status()
                return await resp.read()

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _fetch_text(
        self, session: aiohttp.ClientSession, url: str
    ) -> str:
        """Fetch *url* and return decoded text. Retries up to 5× with backoff.

        Encoding resolution order:
        1. Charset declared in the Content-Type response header.
        2. UTF-8 (strict).
        3. windows-1252 (a superset of ISO-8859-1 / Latin-1 — covers most
           legacy European music-archive sites).
        4. UTF-8 with ``errors='replace'`` as a last resort so we never crash.
        """
        async with self._semaphore:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                resp.raise_for_status()
                raw: bytes = await resp.read()

                # 1. Honour the server's declared charset when present.
                content_type = resp.headers.get("Content-Type", "")
                declared_enc: str | None = None
                for part in content_type.split(";"):
                    part = part.strip()
                    if part.lower().startswith("charset="):
                        declared_enc = part.split("=", 1)[1].strip().strip('"')
                        break

                if declared_enc:
                    try:
                        return raw.decode(declared_enc)
                    except (UnicodeDecodeError, LookupError):
                        logger.debug(
                            "Declared encoding %r failed for %s; trying fallbacks",
                            declared_enc, url,
                        )

                # 2–3. Try common encodings in order.
                for enc in ("utf-8", "windows-1252"):
                    try:
                        return raw.decode(enc)
                    except UnicodeDecodeError:
                        continue

                # 4. Give up cleanly — replace undecodable bytes rather than crash.
                logger.warning(
                    "All encodings failed for %s — decoding with replacement chars", url
                )
                return raw.decode("utf-8", errors="replace")

    async def download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        dest: Path,
        *,
        crawl_delay: Optional[float] = None,
    ) -> bool:
        """Download *url* to *dest* with idempotency, retry, and polite delay.

        The delay is applied *after* a successful write so the semaphore is
        already released before we sleep.  Returns ``True`` on success
        (including skip), ``False`` after exhausted retries.
        """
        if is_downloaded(dest):
            logger.debug("Skip (already downloaded): %s", dest.name)
            return True

        try:
            data = await self._fetch_bytes(session, url)
            dest.write_bytes(data)
            logger.info("Downloaded %s → %s", url, dest)
            await self._polite_delay(crawl_delay)
            return True
        except Exception as exc:
            self._record_failure(url, dest, exc)
            logger.error("Permanent failure: %s — %s", url, exc)
            return False

    # ------------------------------------------------------------------
    # Polite-researcher helpers
    # ------------------------------------------------------------------

    async def _get_crawl_delay(
        self, session: aiohttp.ClientSession, base_url: str
    ) -> Optional[float]:
        """Fetch ``/robots.txt`` for *base_url* and return the ``Crawl-delay``
        for ``*``, or ``None`` if not specified / unreachable.

        Results are cached per host so robots.txt is fetched at most once.
        """
        parsed = urlparse(base_url)
        host = f"{parsed.scheme}://{parsed.netloc}"

        if host in self._crawl_delays:
            return self._crawl_delays[host]

        robots_url = f"{host}/robots.txt"
        try:
            text = await self._fetch_text(session, robots_url)
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(text.splitlines())
            delay = parser.crawl_delay("*")
            if delay is not None:
                delay = float(delay)
                logger.info("robots.txt Crawl-delay for %s: %.1fs", host, delay)
        except Exception as exc:
            logger.debug("Could not read robots.txt for %s: %s", host, exc)
            delay = None

        self._crawl_delays[host] = delay
        return delay

    async def _polite_delay(self, crawl_delay: Optional[float] = None) -> None:
        """Sleep politely between requests.

        Uses *crawl_delay* from robots.txt when available; otherwise falls
        back to a uniform random jitter in [1.0, 3.5] seconds.
        """
        delay = crawl_delay if crawl_delay is not None else random.uniform(1.0, 3.5)
        await asyncio.sleep(delay)

    def _record_failure(self, url: str, dest: Path, exc: Exception) -> None:
        """Append a failure entry to the in-memory log."""
        self._failed.append({
            "url": url,
            "dest": str(dest),
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def write_failed_log(self, path: Optional[Path] = None) -> Optional[Path]:
        """Persist accumulated failures to a JSON file and return its path.

        Appends to an existing file if one already exists (so failures from
        multiple runs accumulate).  Returns ``None`` if there are no failures.
        """
        if not self._failed:
            return None

        out = path or (self.staging_root / "failed_downloads.json")
        existing: list = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        existing.extend(self._failed)
        out.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.warning("Wrote %d failure(s) to %s", len(self._failed), out)
        return out

    # ------------------------------------------------------------------
    # Subprocess helpers (no shell=True)
    # ------------------------------------------------------------------

    def run_subprocess(
        self,
        args: list[str],
        cwd: Optional[Path] = None,
        timeout: int = 120,
    ) -> tuple[int, str, str]:
        """Run a command without a shell.

        Returns ``(returncode, stdout, stderr)``.

        Raises ``subprocess.TimeoutExpired`` if the process exceeds *timeout*
        seconds.
        """
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr

    def run_pipe(
        self,
        args1: list[str],
        args2: list[str],
        cwd: Optional[Path] = None,
        timeout: int = 300,
    ) -> tuple[int, str, str]:
        """Run ``args1 | args2`` as a safe two-process pipe (no ``shell=True``).

        Returns ``(returncode_of_args2, stdout, stderr)``.

        Example::

            rc, out, err = self.run_pipe(
                ["humcat", "-s", "h://bach/chorales-371"],
                ["humsplit"],
                cwd=output_path,
            )
        """
        with subprocess.Popen(
            args1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
        ) as proc1:
            with subprocess.Popen(
                args2,
                stdin=proc1.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
            ) as proc2:
                # Allow proc1 to receive SIGPIPE if proc2 exits early.
                assert proc1.stdout is not None
                proc1.stdout.close()

                try:
                    stdout, stderr = proc2.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc2.kill()
                    proc1.kill()
                    raise

        return proc2.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")
