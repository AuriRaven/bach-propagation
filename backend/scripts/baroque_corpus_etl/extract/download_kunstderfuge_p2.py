"""
Baroque (non-Bach) downloader for kunstderfuge.com.

This script is a thin entry point.  All logic lives in
``download_kunstderfuge_p1.KunstDerFugeExtractor``.

Running this script downloads the same composers as before
(Vivaldi, Handel, Telemann, Corelli, Scarlatti, Purcell, Couperin, Rameau)
but now with full async/retry/idempotency support from ``BaseExtractor``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Allow running directly from the scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from download_kunstderfuge_p1 import ALL_COMPOSERS, KunstDerFugeExtractor

#: All composers from the unified registry except Bach, who has his own runs
#: via ``download_kunstderfuge_p1.py`` and ``download_kernscores_bach.py``.
BAROQUE_ONLY: dict[str, str] = {
    slug: name for slug, name in ALL_COMPOSERS.items() if slug != "bach"
}

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    extractor = KunstDerFugeExtractor(composers=BAROQUE_ONLY)
    asyncio.run(extractor.run())
