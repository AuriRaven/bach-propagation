"""
Batch pre-computation of harmonic analyses for a Bach corpus.

Runs the full harmonic pipeline (key → chord → Roman numeral → harmonic
function → NCT → cadence) over a corpus of scores and caches the result
as JSON so downstream consumers (the harmonic encoder, the Analysis
panel in the web UI, offline experiments) can avoid re-running the
analyzers on every request.

Pipeline per score:

    ScoreLoader
      ↳ (optional) VoiceSeparator          — implied polyphony
      ↳ KeyEstimator.estimate_global       — one key
      ↳ KeyEstimator.estimate_local        — windowed keys
      ↳ ChordClassifier.classify_score     — chord list
      ↳ RomanNumeralAnalyzer.analyze_sequence
      ↳ FunctionLabeler.label_sequence
      ↳ NCTClassifier.classify             — non-chord-tone roles
      ↳ CadenceDetector.detect             — cadence list

Usage
-----

    # All music21 Bach chorales (default — 371 pieces)
    uv run python scripts/batch_harmonic_analysis.py --chorales

    # Single piece by BWV
    uv run python scripts/batch_harmonic_analysis.py --bwv 66.6

    # Local MusicXML/MIDI tree
    uv run python scripts/batch_harmonic_analysis.py \\
        --source path --path data/silver/chorales

    # Dry-run by limiting to N pieces
    uv run python scripts/batch_harmonic_analysis.py --chorales --limit 10

    # Re-process everything, ignoring existing cache
    uv run python scripts/batch_harmonic_analysis.py --chorales --force

    # Separate implied voices first (for cello / violin solo works)
    uv run python scripts/batch_harmonic_analysis.py \\
        --bwv 1007 --separate-voices

Output layout::

    <output-dir>/
      manifest.json                — index of every run (ok / failed / skipped)
      failures.jsonl               — append-only JSON lines log of failures
      corpus/<score_id>.json       — per-score analysis (one file per piece)
      path/<relative_path>.json    — mirrors the input tree for --source path

Each per-score JSON conforms to :data:`ANALYSIS_SCHEMA_VERSION`; Fractions
are rendered as strings (``"3/8"``) so the round-trip is exact.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Resolve backend/ so the script works regardless of cwd
# ---------------------------------------------------------------------------

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from src.core.data_structures import (  # noqa: E402
    Chord,
    HarmonicEvent,
    HarmonicFunction,
    KeyEstimate,
    MusicalEvent,
    RomanNumeral,
    Score,
    TimeSignature,
)
from src.core.score_loader import ScoreLoader, ScoreLoadError  # noqa: E402
from src.core.voice_separator import (  # noqa: E402
    VoiceSeparationConfig,
    VoiceSeparator,
)
from src.harmonic import (  # noqa: E402
    CadenceDetectionConfig,
    CadenceDetector,
    ChordClassificationConfig,
    ChordClassifier,
    FunctionLabeler,
    KeyEstimator,
    NCTClassificationConfig,
    NCTClassifier,
    RomanNumeralAnalyzer,
)
from src.harmonic.cadence_detector import Cadence  # noqa: E402
from src.harmonic.nct_classifier import NCTAnnotation  # noqa: E402
from src.temporal.windower import WindowType  # noqa: E402

# ---------------------------------------------------------------------------
# Version tag
# ---------------------------------------------------------------------------

#: Bump when the JSON output shape changes. Consumers should refuse to read
#: a cache whose version they don't recognise.
ANALYSIS_SCHEMA_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("batch_harmonic_analysis")


def _configure_logging(level: int = logging.INFO) -> None:
    """Apply a minimal, script-friendly log format to the root logger."""
    root = logging.getLogger()
    if root.handlers:
        # Respect caller-provided handlers (e.g. pytest capture).
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _frac(value: Fraction) -> str:
    """Render a :class:`Fraction` as an exact string (``"3/8"`` / ``"0"``)."""
    return str(value)


def _time_sig_to_json(ts_pair: Tuple[Fraction, TimeSignature]) -> Dict[str, object]:
    onset, ts = ts_pair
    return {
        "onset": _frac(onset),
        "numerator": ts.numerator,
        "denominator": ts.denominator,
    }


def _key_to_json(key: KeyEstimate) -> Dict[str, object]:
    return {
        "onset": _frac(key.onset),
        "tonic": key.tonic,
        "mode": key.mode,
        "confidence": round(float(key.confidence), 6),
        "label": key.label,
    }


def _chord_to_json(chord: Chord) -> Dict[str, object]:
    return {
        "onset": _frac(chord.onset),
        "duration": _frac(chord.duration),
        "root": chord.root,
        "quality": chord.quality,
        "pitches": sorted(int(p) for p in chord.pitches),
        "bass": chord.bass,
        "inversion": chord.inversion,
    }


def _rn_to_json(rn: RomanNumeral) -> Dict[str, object]:
    return {
        "degree": rn.degree,
        "quality": rn.quality,
        "inversion": rn.inversion,
        "local_key_tonic": rn.local_key_tonic,
        "local_key_mode": rn.local_key_mode,
        "is_secondary": rn.is_secondary,
        "secondary_target": rn.secondary_target,
        "function": rn.function.value if rn.function is not None else None,
        "label": rn.label,
    }


def _nct_to_json(ann: NCTAnnotation) -> Dict[str, object]:
    return {
        "onset": _frac(ann.onset),
        "pitch": ann.pitch,
        "voice": ann.voice,
        "nct_type": ann.nct_type.value,
        "confidence": round(float(ann.confidence), 6),
        "resolution_pitch": ann.resolution_pitch,
    }


def _cadence_to_json(
    cad: Cadence,
    index_of: Dict[int, int],
) -> Dict[str, object]:
    """Serialize a cadence; record penultimate/final indices into the
    score's harmonic_event list so consumers can jump back."""
    return {
        "onset": _frac(cad.onset),
        "cadence_type": cad.cadence_type.value,
        "label": cad.label,
        "confidence": round(float(cad.confidence), 6),
        "penultimate_index": index_of.get(id(cad.penultimate), -1),
        "final_index": index_of.get(id(cad.final), -1),
    }


# ---------------------------------------------------------------------------
# Analysis result shape
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """In-memory container for a single score's batch output."""

    score_id: str
    source: str
    input_path: Optional[str]
    title: Optional[str]
    composer: Optional[str]
    time_signatures: List[Dict[str, object]]
    num_events: int
    num_voices: int
    duration_ql: str
    global_key: Dict[str, object]
    local_keys: List[Dict[str, object]]
    chords: List[Dict[str, object]]
    roman_numerals: List[Dict[str, object]]
    nct_annotations: List[Dict[str, object]]
    cadences: List[Dict[str, object]]
    stats: Dict[str, object]
    runtime_ms: int
    pipeline_version: str = ANALYSIS_SCHEMA_VERSION
    separate_voices: bool = False

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Tunable parameters for the batch driver.

    Attributes:
        window_type: ``MEASURE`` is the safe default for chorales; ``BEAT``
            gives finer time resolution on fast-moving works.
        local_key_window: Window size (quarter notes) for local-key tracking.
        local_key_hop:    Hop between local-key windows.
        chord_min_confidence: Below this the chord classifier emits None.
        cadence_min_confidence: Cadences below this are dropped.
        separate_voices: Run :class:`VoiceSeparator` before analysis — enable
            for single-voice works that imply polyphony (cello suites,
            violin partitas).
        max_voices: Cap passed to :class:`VoiceSeparator`.
    """

    window_type: WindowType = WindowType.MEASURE
    local_key_window: Fraction = field(default_factory=lambda: Fraction(16))
    local_key_hop: Fraction = field(default_factory=lambda: Fraction(8))
    chord_min_confidence: float = 0.25
    cadence_min_confidence: float = 0.5
    separate_voices: bool = False
    max_voices: int = 3


def analyse_score(
    score: Score,
    *,
    score_id: str,
    source: str,
    input_path: Optional[Path],
    config: PipelineConfig,
) -> AnalysisResult:
    """Run the full pipeline on one pre-loaded :class:`Score`.

    The function is pure — no file I/O. Callers wrap this in the batch
    driver to handle persistence and failure reporting.
    """
    t0 = time.perf_counter()

    if config.separate_voices:
        vs = VoiceSeparator()
        vs_result = vs.separate(
            score, VoiceSeparationConfig(max_voices=config.max_voices)
        )
        score = vs.to_score(score, vs_result)

    key_est = KeyEstimator()
    global_key = key_est.estimate_global(score)
    local_keys = key_est.estimate_local(
        score, config.local_key_window, config.local_key_hop
    )
    effective_keys: List[KeyEstimate] = local_keys if local_keys else [global_key]

    cc_cfg = ChordClassificationConfig(
        min_confidence=config.chord_min_confidence,
        include_seventh_chords=True,
    )
    chords: List[Chord] = ChordClassifier().classify_score(
        score, config.window_type, cc_cfg
    )

    rns: List[RomanNumeral] = RomanNumeralAnalyzer().analyze_sequence(
        chords, effective_keys
    )
    rns = FunctionLabeler().label_sequence(rns)

    nct_annotations: List[NCTAnnotation] = NCTClassifier().classify(
        score.events, chords, NCTClassificationConfig()
    )

    harmonic_events: List[HarmonicEvent] = [
        HarmonicEvent(chord=ch, roman_numeral=rn)
        for ch, rn in zip(chords, rns)
    ]
    # Remember each event's index so cadences can point back.
    index_of = {id(ev): i for i, ev in enumerate(harmonic_events)}

    cad_cfg = CadenceDetectionConfig(min_confidence=config.cadence_min_confidence)
    cadences: List[Cadence] = CadenceDetector().detect(harmonic_events, cad_cfg)

    stats = _compute_stats(chords, rns, cadences, nct_annotations)

    runtime_ms = int((time.perf_counter() - t0) * 1000)

    return AnalysisResult(
        score_id=score_id,
        source=source,
        input_path=str(input_path) if input_path is not None else None,
        title=score.title,
        composer=score.composer,
        time_signatures=[_time_sig_to_json(ts) for ts in score.time_signatures],
        num_events=len(score.events),
        num_voices=score.num_voices,
        duration_ql=_frac(score.duration),
        global_key=_key_to_json(global_key),
        local_keys=[_key_to_json(k) for k in local_keys],
        chords=[_chord_to_json(c) for c in chords],
        roman_numerals=[_rn_to_json(r) for r in rns],
        nct_annotations=[_nct_to_json(a) for a in nct_annotations],
        cadences=[_cadence_to_json(c, index_of) for c in cadences],
        stats=stats,
        runtime_ms=runtime_ms,
        separate_voices=config.separate_voices,
    )


def _compute_stats(
    chords: List[Chord],
    rns: List[RomanNumeral],
    cadences: List[Cadence],
    ncts: List[NCTAnnotation],
) -> Dict[str, object]:
    """Aggregate counts callers might want without parsing the full list."""
    chord_quality_counts: Dict[str, int] = {}
    for c in chords:
        chord_quality_counts[c.quality] = chord_quality_counts.get(c.quality, 0) + 1

    degree_counts: Dict[str, int] = {}
    function_counts: Dict[str, int] = {}
    secondary_count = 0
    for r in rns:
        key = str(r.degree)
        degree_counts[key] = degree_counts.get(key, 0) + 1
        if r.function is not None:
            function_counts[r.function.value] = (
                function_counts.get(r.function.value, 0) + 1
            )
        if r.is_secondary:
            secondary_count += 1

    cadence_type_counts: Dict[str, int] = {}
    for cad in cadences:
        key = cad.cadence_type.value
        cadence_type_counts[key] = cadence_type_counts.get(key, 0) + 1

    nct_type_counts: Dict[str, int] = {}
    for ann in ncts:
        key = ann.nct_type.value
        nct_type_counts[key] = nct_type_counts.get(key, 0) + 1

    return {
        "n_chords": len(chords),
        "n_roman_numerals": len(rns),
        "n_cadences": len(cadences),
        "n_nct_annotations": len(ncts),
        "chord_quality_counts": chord_quality_counts,
        "degree_counts": degree_counts,
        "function_counts": function_counts,
        "cadence_type_counts": cadence_type_counts,
        "nct_type_counts": nct_type_counts,
        "secondary_dominant_count": secondary_count,
    }


# ---------------------------------------------------------------------------
# Source enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Task:
    """One unit of work for the batch driver."""

    score_id: str          # stable identifier (used in cache path)
    source: str            # "corpus" or "path"
    loader_args: tuple     # passed through to the loader dispatch
    relative_output: Path  # cache filename relative to the output root


def _iter_corpus_chorales() -> Iterator[_Task]:
    """Yield every Bach chorale available in music21's built-in corpus.

    Chorale identifiers come from ``music21.corpus.chorales.Iterator`` — we
    use its filename scheme (``bwv66.6``) to derive a stable score_id.
    """
    # Local import so import-time of the script is fast even if music21
    # isn't configured.
    from music21 import corpus  # noqa: WPS433

    catalog = corpus.chorales.Iterator(
        numberingSystem="bwv",
        returnType="filename",
    )
    # music21 returns filenames like ``bach/bwv66.6`` or ``bach/bwv66.6.mxl``
    # — BWV ids like ``66.6`` contain dots, so ``Path.stem`` misreads them
    # as extensions. Parse manually: drop the directory prefix, then any
    # music21 extension, then the leading ``bwv``.
    _KNOWN_EXT = (".mxl", ".xml", ".musicxml", ".mid", ".midi", ".krn")
    for filename in catalog:
        base = filename.rsplit("/", 1)[-1]
        for ext in _KNOWN_EXT:
            if base.endswith(ext):
                base = base[: -len(ext)]
                break
        bwv = base.removeprefix("bwv")
        yield _Task(
            score_id=f"bwv{bwv}",
            source="corpus",
            loader_args=("corpus", bwv),
            relative_output=Path("corpus") / f"bwv{bwv}.json",
        )


def _iter_single_bwv(bwv: str) -> Iterator[_Task]:
    yield _Task(
        score_id=f"bwv{bwv}",
        source="corpus",
        loader_args=("corpus", bwv),
        relative_output=Path("corpus") / f"bwv{bwv}.json",
    )


def _iter_path(root: Path) -> Iterator[_Task]:
    """Yield every MusicXML/MIDI file under ``root``.

    The cache filename mirrors the input tree — e.g. ``foo/bar/01.mxl``
    becomes ``path/foo/bar/01.json``.
    """
    patterns = ("*.mxl", "*.xml", "*.musicxml", "*.mid", "*.midi")
    seen: set = set()
    for pat in patterns:
        for src in sorted(root.rglob(pat)):
            if src in seen:
                continue
            seen.add(src)
            rel = src.relative_to(root).with_suffix(".json")
            yield _Task(
                score_id=str(src.relative_to(root)),
                source="path",
                loader_args=("path", str(src)),
                relative_output=Path("path") / rel,
            )


def _load_score(loader: ScoreLoader, loader_args: tuple) -> Tuple[Score, Optional[Path]]:
    kind, payload = loader_args
    if kind == "corpus":
        return loader.load_corpus(payload), None
    if kind == "path":
        p = Path(payload)
        return loader.load(p), p
    raise ValueError(f"unknown loader kind: {kind!r}")


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


@dataclass
class _ManifestEntry:
    score_id: str
    source: str
    status: str            # "ok" | "failed" | "skipped"
    output: Optional[str] = None
    runtime_ms: Optional[int] = None
    n_cadences: Optional[int] = None
    error: Optional[str] = None


def run_batch(
    tasks: Iterable[_Task],
    *,
    output_dir: Path,
    config: PipelineConfig,
    force: bool,
    fail_fast: bool,
    progress: bool,
    limit: Optional[int] = None,
) -> List[_ManifestEntry]:
    """Execute ``tasks`` and persist each result to disk.

    Returns the manifest entries so the caller can render a summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_log = output_dir / "failures.jsonl"

    loader = ScoreLoader(separate_voices=True)

    if limit is not None:
        tasks = list(tasks)[:limit]

    entries: List[_ManifestEntry] = []

    if progress:
        try:
            from tqdm import tqdm  # noqa: WPS433
            tasks_iter = tqdm(list(tasks), unit="score", dynamic_ncols=True)
        except ImportError:
            tasks_iter = tasks
    else:
        tasks_iter = tasks

    for task in tasks_iter:
        target = output_dir / task.relative_output
        if target.exists() and not force:
            entries.append(_ManifestEntry(
                score_id=task.score_id,
                source=task.source,
                status="skipped",
                output=str(target.relative_to(output_dir)),
            ))
            continue

        try:
            score, input_path = _load_score(loader, task.loader_args)
            result = analyse_score(
                score,
                score_id=task.score_id,
                source=task.source,
                input_path=input_path,
                config=config,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            entries.append(_ManifestEntry(
                score_id=task.score_id,
                source=task.source,
                status="ok",
                output=str(target.relative_to(output_dir)),
                runtime_ms=result.runtime_ms,
                n_cadences=len(result.cadences),
            ))
            _update_postfix(tasks_iter, entries)
        except ScoreLoadError as exc:
            _record_failure(
                entries, failures_log, task, exc, category="load"
            )
            if fail_fast:
                raise
        except Exception as exc:  # noqa: BLE001
            _record_failure(
                entries, failures_log, task, exc, category="pipeline"
            )
            if fail_fast:
                raise

    return entries


def _update_postfix(tasks_iter, entries: List[_ManifestEntry]) -> None:
    """Refresh the tqdm bar's running-stats label (if tqdm is active)."""
    set_postfix = getattr(tasks_iter, "set_postfix", None)
    if set_postfix is None:
        return
    ok = sum(1 for e in entries if e.status == "ok")
    failed = sum(1 for e in entries if e.status == "failed")
    skipped = sum(1 for e in entries if e.status == "skipped")
    set_postfix(ok=ok, failed=failed, skipped=skipped)


def _record_failure(
    entries: List[_ManifestEntry],
    log_path: Path,
    task: _Task,
    exc: BaseException,
    *,
    category: str,
) -> None:
    """Append a failure to ``failures.jsonl`` and the in-memory manifest."""
    tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    entries.append(_ManifestEntry(
        score_id=task.score_id,
        source=task.source,
        status="failed",
        error=f"[{category}] {tb}",
    ))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "score_id": task.score_id,
            "source": task.source,
            "category": category,
            "error": tb,
        }, ensure_ascii=False) + "\n")
    log.warning("[%s] %s: %s", category, task.score_id, tb)


def write_manifest(
    output_dir: Path,
    entries: List[_ManifestEntry],
    *,
    source_description: str,
    config: PipelineConfig,
) -> Path:
    """Write a top-level index summarising the batch run."""
    totals = {
        "ok": sum(1 for e in entries if e.status == "ok"),
        "failed": sum(1 for e in entries if e.status == "failed"),
        "skipped": sum(1 for e in entries if e.status == "skipped"),
        "total": len(entries),
    }
    manifest = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "pipeline_version": ANALYSIS_SCHEMA_VERSION,
        "source_description": source_description,
        "pipeline_config": {
            "window_type": config.window_type.value,
            "local_key_window": _frac(config.local_key_window),
            "local_key_hop": _frac(config.local_key_hop),
            "chord_min_confidence": config.chord_min_confidence,
            "cadence_min_confidence": config.cadence_min_confidence,
            "separate_voices": config.separate_voices,
            "max_voices": config.max_voices,
        },
        "totals": totals,
        "entries": [dataclasses.asdict(e) for e in entries],
    }
    path = output_dir / "manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="batch_harmonic_analysis",
        description=(
            "Run the full harmonic pipeline over a Bach corpus and write "
            "per-score JSON caches. Safe to re-run — files already present "
            "are skipped unless --force is given."
        ),
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--chorales",
        action="store_true",
        help="Iterate every BWV chorale in music21's built-in corpus.",
    )
    src.add_argument(
        "--bwv",
        metavar="BWV",
        help="Single music21 corpus piece by BWV identifier (e.g. '66.6', '1007').",
    )
    src.add_argument(
        "--path",
        type=Path,
        metavar="DIR",
        help="Process every MusicXML/MIDI file under DIR (recursively).",
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=_BACKEND_ROOT / "data" / "analysis_cache",
        help="Where per-score JSONs, failures.jsonl, and manifest.json land.",
    )
    p.add_argument(
        "--window",
        choices=("measure", "beat"),
        default="measure",
        help="Chord-classification window strategy (default: measure).",
    )
    p.add_argument(
        "--local-key-window",
        type=int,
        default=16,
        help="Local-key window size in quarter notes (default: 16).",
    )
    p.add_argument(
        "--local-key-hop",
        type=int,
        default=8,
        help="Local-key window hop in quarter notes (default: 8).",
    )
    p.add_argument(
        "--chord-min-confidence",
        type=float,
        default=0.25,
        help="Chord classifier drops windows below this threshold (default: 0.25).",
    )
    p.add_argument(
        "--cadence-min-confidence",
        type=float,
        default=0.5,
        help="Cadence detector drops pairs below this threshold (default: 0.5).",
    )
    p.add_argument(
        "--separate-voices",
        action="store_true",
        help="Run voice separation first (for solo cello / violin works).",
    )
    p.add_argument(
        "--max-voices",
        type=int,
        default=3,
        help="Upper bound on voices spawned by --separate-voices (default: 3).",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after processing N scores. Useful for smoke tests.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-process scores that already have a cached JSON.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first failure (default: log and continue).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the tqdm progress bar.",
    )
    return p


def _window_type_from_str(name: str) -> WindowType:
    return {"measure": WindowType.MEASURE, "beat": WindowType.BEAT}[name]


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging()

    cfg = PipelineConfig(
        window_type=_window_type_from_str(args.window),
        local_key_window=Fraction(args.local_key_window),
        local_key_hop=Fraction(args.local_key_hop),
        chord_min_confidence=args.chord_min_confidence,
        cadence_min_confidence=args.cadence_min_confidence,
        separate_voices=args.separate_voices,
        max_voices=args.max_voices,
    )

    if args.chorales:
        tasks_iter: Iterator[_Task] = _iter_corpus_chorales()
        source_description = "music21.corpus.chorales (all BWV chorales)"
    elif args.bwv:
        tasks_iter = _iter_single_bwv(args.bwv)
        source_description = f"music21 corpus — bwv{args.bwv}"
    else:
        if not args.path.exists():
            log.error("--path does not exist: %s", args.path)
            return 2
        tasks_iter = _iter_path(args.path)
        source_description = f"path tree — {args.path}"

    log.info("Source:   %s", source_description)
    log.info("Output:   %s", args.output_dir)
    log.info("Pipeline: window=%s, separate_voices=%s", args.window, args.separate_voices)

    entries = run_batch(
        tasks_iter,
        output_dir=args.output_dir,
        config=cfg,
        force=args.force,
        fail_fast=args.fail_fast,
        progress=not args.quiet,
        limit=args.limit,
    )
    manifest_path = write_manifest(
        args.output_dir,
        entries,
        source_description=source_description,
        config=cfg,
    )

    ok = sum(1 for e in entries if e.status == "ok")
    failed = sum(1 for e in entries if e.status == "failed")
    skipped = sum(1 for e in entries if e.status == "skipped")
    log.info("Done: ok=%d failed=%d skipped=%d total=%d", ok, failed, skipped, len(entries))
    log.info("Manifest: %s", manifest_path)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
