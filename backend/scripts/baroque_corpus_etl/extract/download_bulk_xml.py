"""
Bulk MusicXML dataset downloader for Baroque pre-training corpus.

Instead of slow 1-by-1 crawling, this script downloads entire dataset
archives in a single operation via ``git clone --depth 1`` or a direct
ZIP/tar download.  Files are filtered for MusicXML formats and moved into
``data/raw/bulk_xml/<dataset>/``.

There is intentionally **no per-file crawl delay** — we are fetching a single
large archive from a public Git hosting service, not crawling a volunteer
website page-by-page.

Supported datasets (selected for open licensing + MusicXML availability)
-------------------------------------------------------------------------
bach-chorales-370
    Craig Sapp's complete digitisation of Bach's 370 chorales in Humdrum +
    MusicXML. GitHub: https://github.com/craigsapp/bach-370-chorales
    Yields ~370 .musicxml files.

openscore-lieder
    OpenScore's Lieder corpus (>1,000 songs, 19th-century focus but useful
    for pre-training).  GitHub: https://github.com/OpenScore/Lieder
    Yields .mxl files (compressed MusicXML).

openscore-bach-chorales
    OpenScore's own Bach choral engravings.
    GitHub: https://github.com/OpenScore/BachChorales
    Yields .mxl files.

music21-corpus
    The music21 built-in corpus subset mirrored on GitHub.
    GitHub: https://github.com/cuthbertLab/music21
    Very large — filtered to bach/ and baroque/ subdirectories only.

Usage
-----
Download all datasets::

    cd backend
    uv run python scripts/baroque_corpus_etl/extract/download_bulk_xml.py

Download one dataset::

    uv run python scripts/baroque_corpus_etl/extract/download_bulk_xml.py \\
        --datasets bach-chorales-370

Dry-run (print what would be done without downloading)::

    uv run python scripts/baroque_corpus_etl/extract/download_bulk_xml.py --dry-run

"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from urllib.error import URLError

# Allow running directly from the scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from utils.network import BaseExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

#: Score file extensions we copy from the cloned repos.
#: Includes Humdrum (.krn) because craigsapp/bach-370-chorales stores files in
#: that format — the music21 score loader can ingest .krn natively.
_WANTED_EXTS: frozenset[str] = frozenset({".xml", ".musicxml", ".mxl", ".krn"})

#: Sub-paths to include from large repos that contain many non-music files.
#: If ``None``, the whole clone is searched.
_NO_FILTER: list[str] | None = None


@dataclass
class DatasetSpec:
    """Specification for a bulk dataset."""

    key: str
    description: str
    #: Public ``git clone`` URL.
    git_url: str
    #: Alternative direct archive URL (used when git is unavailable).
    zip_url: Optional[str] = None
    #: Subdirectory paths *within the clone* to search for MusicXML files.
    #: ``None`` means search the whole repo.  Use to avoid cloning noise from
    #: large multi-purpose repositories.
    include_paths: list[str] | None = None
    #: Expected minimum number of files — used for sanity check warnings.
    min_expected: int = 1
    #: Tags for documentation purposes only.
    tags: list[str] = field(default_factory=list)


DATASETS: dict[str, DatasetSpec] = {
    "bach-chorales-370": DatasetSpec(
        key="bach-chorales-370",
        description="Craig Sapp's 370 Bach Chorales (Humdrum .krn) — verified 2026-04-04",
        # SSH URL — uses the macOS SSH agent (no interactive prompt).
        git_url="git@github.com:craigsapp/bach-370-chorales.git",
        # ZIP fallback (HTTPS) for environments without an SSH key.
        zip_url="https://github.com/craigsapp/bach-370-chorales/archive/refs/heads/master.zip",
        # Score files live in kern/ (.krn); explicitly scope to avoid recursing
        # into .git/, documentation/, etc.
        include_paths=["kern"],
        min_expected=350,
        tags=["bach", "chorales", "ground-truth"],
    ),
    "bach-chorales-371": DatasetSpec(
        key="bach-chorales-371",
        description="Craig Sapp's 371 Bach Chorales (Humdrum .krn) — verified 2026-04-04",
        # A companion repo with an additional chorale not in the 370 set.
        git_url="git@github.com:craigsapp/bach-371-chorales.git",
        zip_url="https://github.com/craigsapp/bach-371-chorales/archive/refs/heads/master.zip",
        include_paths=["kern"],
        min_expected=1,
        tags=["bach", "chorales"],
    ),
    "openscore-lieder": DatasetSpec(
        key="openscore-lieder",
        description="OpenScore Lieder corpus — 1000+ songs in MusicXML — verified 2026-04-04",
        # NOTE: OpenScore/BachChorales does NOT exist (GitHub 404 confirmed).
        # OpenScore/Lieder is the real repo; it contains Lieder engraved as .mxl.
        git_url="git@github.com:OpenScore/Lieder.git",
        zip_url="https://github.com/OpenScore/Lieder/archive/refs/heads/main.zip",
        include_paths=["scores"],   # only the scores/ subfolder
        min_expected=500,
        tags=["lieder", "19th-century", "openscore"],
    ),
}


# ---------------------------------------------------------------------------
# Bulk downloader (not a web crawler — no BaseExtractor HTTP methods used)
# ---------------------------------------------------------------------------


class BulkXMLDownloader(BaseExtractor):
    """Download and extract bulk MusicXML dataset archives.

    This class inherits ``BaseExtractor`` for its subprocess helpers and
    canonical staging-root convention, but does **not** use the HTTP retry
    stack — bulk archives are fetched in a single operation using either
    ``git clone`` (preferred) or ``urllib`` as a fallback.

    Args:
        staging_root: Root of the local staging area (default: ``data/raw``).
        datasets:     Keys from ``DATASETS`` to download.  ``None`` → all.
        dry_run:      Print what would be done without downloading anything.
        force:        Re-download even if output directory already exists.
    """

    def __init__(
        self,
        staging_root: Path = Path("data/raw"),
        datasets: list[str] | None = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> None:
        super().__init__(staging_root=staging_root)

        if datasets is None:
            self._specs = list(DATASETS.values())
        else:
            unknown = set(datasets) - DATASETS.keys()
            if unknown:
                raise ValueError(
                    f"Unknown dataset(s): {unknown!r}. Valid: {list(DATASETS)}"
                )
            self._specs = [DATASETS[k] for k in datasets]

        self.dry_run = dry_run
        self.force = force
        self._bulk_root = self.staging_root / "bulk_xml"
        self._bulk_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:  # noqa: D102  (satisfies ABC)
        """Process every configured dataset."""
        results: list[dict] = []

        for spec in self._specs:
            print(f"\n{'=' * 60}")
            print(f"Dataset : {spec.description}")
            print(f"Key     : {spec.key}")
            print("=" * 60)

            count = self._process_dataset(spec)
            results.append({"key": spec.key, "files_copied": count})

        # Summary
        print(f"\n{'=' * 60}")
        print("BULK XML — Summary")
        print("=" * 60)
        total = 0
        for r in results:
            print(f"  {r['key']:40s} {r['files_copied']:>5} file(s)")
            total += r["files_copied"]
        print(f"  {'TOTAL':40s} {total:>5} file(s)")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Per-dataset logic
    # ------------------------------------------------------------------

    def _process_dataset(self, spec: DatasetSpec) -> int:
        """Download, extract, and copy files for *spec*.  Returns file count."""
        output_dir = self._bulk_root / spec.key
        output_dir.mkdir(parents=True, exist_ok=True)

        # Idempotency: skip if we already have files (unless --force).
        existing = list(output_dir.rglob("*"))
        existing_xml = [f for f in existing if f.suffix.lower() in _WANTED_EXTS]
        if existing_xml and not self.force:
            print(
                f"  ✓ Skipped (already have {len(existing_xml)} file(s) in {output_dir})"
            )
            print(f"    Use --force to re-download.")
            return len(existing_xml)

        if self.dry_run:
            print(f"  [dry-run] Would clone: {spec.git_url}")
            print(f"  [dry-run] Output dir : {output_dir}")
            return 0

        # Work in a temp directory so partial downloads don't pollute output.
        with tempfile.TemporaryDirectory(prefix="bach_bulk_") as tmp_str:
            tmp = Path(tmp_str)
            clone_dir = tmp / "repo"

            fetched = self._git_clone(spec, clone_dir) or self._zip_download(
                spec, clone_dir, tmp
            )

            if not fetched:
                logger.error("All fetch methods failed for %s", spec.key)
                print("  ✗ Could not fetch dataset — see log for details")
                return 0

            # Collect MusicXML files from specified include_paths (or whole repo).
            xml_files = self._collect_xml_files(clone_dir, spec.include_paths)
            print(f"  Found {len(xml_files)} MusicXML file(s) in archive")

            if len(xml_files) < spec.min_expected:
                logger.warning(
                    "Only %d files found for %s (expected ≥ %d) — "
                    "repo structure may have changed",
                    len(xml_files), spec.key, spec.min_expected,
                )

            # Copy files into output_dir, preserving relative sub-paths so
            # directory names (BWV numbers, opus folders, etc.) are kept.
            copied = self._copy_to_output(xml_files, clone_dir, output_dir)

        if copied >= spec.min_expected:
            print(f"  ✓ {copied} file(s) → {output_dir}")
        else:
            print(f"  ⚠  {copied} file(s) copied (expected ≥ {spec.min_expected})")

        return copied

    # ------------------------------------------------------------------
    # Fetch strategies
    # ------------------------------------------------------------------

    def _git_clone(self, spec: DatasetSpec, target: Path) -> bool:
        """Attempt ``git clone --depth 1`` into *target*.

        Returns ``True`` on success.  The ``--depth 1`` flag fetches only the
        latest commit — sufficient for file content and far faster than a full
        history clone for large repos.

        SSH notes
        ---------
        git URLs use ``git@github.com:…`` (SSH).  macOS SSH agent should have
        the key loaded automatically.  ``GIT_TERMINAL_PROMPT=0`` makes git fail
        immediately with a non-zero exit code instead of hanging to ask for a
        password, which would block the ETL pipeline indefinitely.
        ``GIT_SSH_COMMAND`` enables BatchMode so the SSH layer also never
        prompts interactively.
        """
        if not shutil.which("git"):
            logger.debug("git not found on PATH — skipping git clone strategy")
            return False

        print(f"  Cloning {spec.git_url} …")

        # Inherit the full current environment so the SSH agent socket path
        # ($SSH_AUTH_SOCK) is available, then override the two prompt variables.
        clone_env = {
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_SSH_COMMAND": "ssh -oBatchMode=yes -oStrictHostKeyChecking=accept-new",
        }

        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", spec.git_url, str(target)],
            capture_output=True,
            text=True,
            timeout=600,
            env=clone_env,
        )
        if result.returncode == 0:
            print("  ✓ Clone successful")
            return True

        logger.warning("git clone failed (rc=%d): %s", result.returncode, result.stderr.strip())
        print(f"  ✗ git clone failed: {result.stderr.strip()[:300]}")
        return False

    def _zip_download(
        self, spec: DatasetSpec, target: Path, tmp: Path
    ) -> bool:
        """Fallback: download the ZIP archive via urllib and extract it.

        The extracted directory is renamed to *target* so subsequent code
        can treat both strategies identically.
        """
        if not spec.zip_url:
            logger.debug("No zip_url for %s — cannot use fallback", spec.key)
            return False

        zip_path = tmp / f"{spec.key}.zip"
        print(f"  Downloading ZIP from {spec.zip_url} …")

        try:
            urlretrieve(spec.zip_url, zip_path)  # noqa: S310 — URL is hardcoded
        except URLError as exc:
            logger.error("ZIP download failed for %s: %s", spec.key, exc)
            print(f"  ✗ ZIP download failed: {exc}")
            return False

        print(f"  Extracting {zip_path.name} …")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp / "extracted")
        except zipfile.BadZipFile as exc:
            logger.error("Bad ZIP for %s: %s", spec.key, exc)
            print(f"  ✗ Bad ZIP: {exc}")
            return False

        # GitHub ZIPs contain a single top-level directory (repo-branch/).
        extracted_root = tmp / "extracted"
        top_dirs = [d for d in extracted_root.iterdir() if d.is_dir()]
        if len(top_dirs) == 1:
            shutil.move(str(top_dirs[0]), str(target))
            print("  ✓ ZIP extracted")
            return True

        # Multiple top-level dirs — use the extracted root directly.
        shutil.move(str(extracted_root), str(target))
        print("  ✓ ZIP extracted (multiple top-level dirs)")
        return True

    # ------------------------------------------------------------------
    # File collection + copy
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_xml_files(
        repo_dir: Path, include_paths: list[str] | None
    ) -> list[Path]:
        """Return all MusicXML files under *repo_dir*, filtered by *include_paths*."""
        search_roots: list[Path]
        if include_paths:
            search_roots = [repo_dir / p for p in include_paths if (repo_dir / p).exists()]
            if not search_roots:
                logger.warning(
                    "None of include_paths %r exist under %s — searching whole repo",
                    include_paths, repo_dir,
                )
                search_roots = [repo_dir]
        else:
            search_roots = [repo_dir]

        xml_files: list[Path] = []
        for root in search_roots:
            for f in root.rglob("*"):
                if f.is_file() and f.suffix.lower() in _WANTED_EXTS:
                    xml_files.append(f)

        return sorted(xml_files)

    @staticmethod
    def _copy_to_output(
        files: list[Path], repo_dir: Path, output_dir: Path
    ) -> int:
        """Copy *files* into *output_dir*, preserving paths relative to *repo_dir*."""
        copied = 0
        for src in files:
            try:
                rel = src.relative_to(repo_dir)
            except ValueError:
                rel = Path(src.name)

            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists() and dest.stat().st_size == src.stat().st_size:
                copied += 1  # already present and same size — count as success
                continue

            shutil.copy2(src, dest)
            copied += 1

        return copied


#: Canonical alias used by the master orchestrator.
BulkXMLExtractor = BulkXMLDownloader


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
        description="Download bulk MusicXML datasets for Baroque corpus pre-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k:30s} — {v.description}" for k, v in DATASETS.items()
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=None,
        metavar="DATASET",
        help="Datasets to download (default: all). Choices: " + ", ".join(DATASETS),
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
        help="Print what would be done without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output directory already has files",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for key, spec in DATASETS.items():
            print(f"  {key:30s} {spec.description}")
            print(f"    {'git':6s}: {spec.git_url}")
            if spec.zip_url:
                print(f"    {'zip':6s}: {spec.zip_url}")
            print(f"    tags : {', '.join(spec.tags)}")
        sys.exit(0)

    import asyncio

    downloader = BulkXMLDownloader(
        staging_root=args.staging_root,
        datasets=args.datasets,
        dry_run=args.dry_run,
        force=args.force,
    )
    asyncio.run(downloader.run())
