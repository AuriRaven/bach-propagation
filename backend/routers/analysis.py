"""
backend/routers/analysis.py

Two endpoint groups:

1. Corpus-level analytics (fast — queries corpus_metadata, no parsing)
   GET /api/analysis/corpus
       → key mode distribution, collection breakdown, form tags,
         voice counts, duration histogram, num_measures distribution

2. Per-score harmonic analysis (slow first run, cached in raw_metadata)
   GET /api/analysis/score/{corpus_id}
       → global key estimate, chord frequency, roman numeral distribution,
         harmonic function breakdown, secondary dominant count

Register in main.py:
    from routers.analysis import analysis_router
    app.include_router(analysis_router, prefix="/api")
"""

from __future__ import annotations

import asyncio
import logging
import math
import urllib.request
import tempfile
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .corpus import get_supabase, _signed_url

logger = logging.getLogger(__name__)

analysis_router = APIRouter(tags=["analysis"])

ANALYSIS_CACHE_KEY = "harmonic_analysis"


# ─── Response models ──────────────────────────────────────────────────────────

class KeyModeCount(BaseModel):
    mode: str
    count: int


class CollectionCount(BaseModel):
    collection: str
    count: int


class FormTagCount(BaseModel):
    form_tag: str
    count: int


class VoiceCount(BaseModel):
    num_voices: int
    count: int


class DurationBucket(BaseModel):
    label: str          # e.g. "0-60s"
    min_s: float
    max_s: float
    count: int


class CorpusAnalytics(BaseModel):
    total: int
    by_key_mode: list[KeyModeCount]
    by_collection: list[CollectionCount]
    by_form_tag: list[FormTagCount]
    by_voice_count: list[VoiceCount]
    duration_histogram: list[DurationBucket]
    measures_histogram: list[DurationBucket]   # reuse bucket shape


class RomanNumeralFreq(BaseModel):
    numeral: str        # e.g. "I", "V", "ii", "V/V"
    degree: int
    count: int
    percentage: float


class ChordQualityFreq(BaseModel):
    quality: str
    count: int
    percentage: float


class HarmonicFunctionFreq(BaseModel):
    function: str       # TONIC | DOMINANT | PREDOMINANT | AMBIGUOUS
    count: int
    percentage: float


class ScoreHarmonicAnalysis(BaseModel):
    corpus_id: str
    global_key_tonic: int       # 0-11
    global_key_mode: str        # "major" | "minor"
    global_key_confidence: float
    total_chords: int
    roman_numerals: list[RomanNumeralFreq]
    chord_qualities: list[ChordQualityFreq]
    harmonic_functions: list[HarmonicFunctionFreq]
    secondary_dominant_count: int
    borrowed_chord_count: int


# ─── Endpoint 1: Corpus analytics ────────────────────────────────────────────

@analysis_router.get("/analysis/corpus", response_model=CorpusAnalytics)
async def corpus_analytics(supabase=Depends(get_supabase)):
    """
    Aggregate analytics from corpus_metadata — no file parsing needed.
    Runs fast against the existing DB columns.
    """
    resp = supabase.from_("corpus_metadata").select(
        "key_mode, num_voices, duration_seconds, num_measures, form_tag",
        count="exact",
    ).execute()

    if resp.data is None:
        raise HTTPException(status_code=502, detail="corpus_metadata query failed")

    # Also get collection breakdown from corpus_files
    col_resp = supabase.from_("corpus_files").select("collection").execute()
    col_rows = col_resp.data or []

    rows = resp.data
    total = resp.count or len(rows)

    # ── Key mode distribution ─────────────────────────────────────────────
    mode_counts: dict[str, int] = {}
    for r in rows:
        mode = (r.get("key_mode") or "unknown").lower()
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    by_key_mode = [
        KeyModeCount(mode=m, count=c)
        for m, c in sorted(mode_counts.items(), key=lambda x: -x[1])
    ]

    # ── Collection breakdown ──────────────────────────────────────────────
    col_counts: dict[str, int] = {}
    for r in col_rows:
        col = (r.get("collection") or "unknown").lower()
        col_counts[col] = col_counts.get(col, 0) + 1

    by_collection = [
        CollectionCount(collection=c, count=n)
        for c, n in sorted(col_counts.items(), key=lambda x: -x[1])
    ]

    # ── Form tag distribution ─────────────────────────────────────────────
    form_counts: dict[str, int] = {}
    for r in rows:
        tag = r.get("form_tag") or "untagged"
        form_counts[tag] = form_counts.get(tag, 0) + 1

    by_form_tag = [
        FormTagCount(form_tag=t, count=c)
        for t, c in sorted(form_counts.items(), key=lambda x: -x[1])
    ]

    # ── Voice count distribution ──────────────────────────────────────────
    voice_counts: dict[int, int] = {}
    for r in rows:
        v = r.get("num_voices") or 0
        voice_counts[v] = voice_counts.get(v, 0) + 1

    by_voice_count = [
        VoiceCount(num_voices=v, count=c)
        for v, c in sorted(voice_counts.items())
    ]

    # ── Duration histogram (seconds) ──────────────────────────────────────
    DURATION_BUCKETS = [(0, 60), (60, 120), (120, 180), (180, 300), (300, 600), (600, 9999)]
    dur_counts = [0] * len(DURATION_BUCKETS)
    for r in rows:
        d = r.get("duration_seconds") or 0
        for i, (lo, hi) in enumerate(DURATION_BUCKETS):
            if lo <= d < hi:
                dur_counts[i] += 1
                break

    labels = ["0–1m", "1–2m", "2–3m", "3–5m", "5–10m", "10m+"]
    duration_histogram = [
        DurationBucket(label=labels[i], min_s=float(DURATION_BUCKETS[i][0]),
                       max_s=float(DURATION_BUCKETS[i][1]), count=dur_counts[i])
        for i in range(len(DURATION_BUCKETS))
    ]

    # ── Measures histogram ────────────────────────────────────────────────
    MEASURE_BUCKETS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 99999)]
    meas_counts = [0] * len(MEASURE_BUCKETS)
    for r in rows:
        m = r.get("num_measures") or 0
        for i, (lo, hi) in enumerate(MEASURE_BUCKETS):
            if lo <= m < hi:
                meas_counts[i] += 1
                break

    mlabels = ["0–16", "16–32", "32–64", "64–128", "128–256", "256+"]
    measures_histogram = [
        DurationBucket(label=mlabels[i], min_s=float(MEASURE_BUCKETS[i][0]),
                       max_s=float(MEASURE_BUCKETS[i][1]), count=meas_counts[i])
        for i in range(len(MEASURE_BUCKETS))
    ]

    return CorpusAnalytics(
        total=total,
        by_key_mode=by_key_mode,
        by_collection=by_collection,
        by_form_tag=by_form_tag,
        by_voice_count=by_voice_count,
        duration_histogram=duration_histogram,
        measures_histogram=measures_histogram,
    )


# ─── Endpoint 2: Per-score harmonic analysis ─────────────────────────────────

@analysis_router.get("/analysis/score/{corpus_id}", response_model=ScoreHarmonicAnalysis)
async def score_harmonic_analysis(
    corpus_id: str,
    refresh: bool = False,
    supabase=Depends(get_supabase),
):
    """
    Run full harmonic analysis on a corpus file using the music analysis engine.

    Pipeline:
      1. Check raw_metadata cache.
      2. Download MIDI via signed URL.
      3. Parse with music21 → Score data structure.
      4. KeyEstimator.estimate_global() → global key.
      5. ChordClassifier.classify_score() → chord sequence.
      6. RomanNumeralAnalyzer.analyze_sequence() → roman numerals.
      7. Aggregate frequencies → return + cache.

    Uses ?refresh=true to force re-analysis.
    """
    # ── 1. Fetch file row with cache ──────────────────────────────────────
    file_resp = (
        supabase.from_("corpus_files")
        .select("id, storage_object_path, raw_metadata")
        .eq("id", corpus_id)
        .single()
        .execute()
    )
    if not file_resp.data:
        raise HTTPException(status_code=404, detail="Corpus file not found")

    raw_metadata: dict = file_resp.data.get("raw_metadata") or {}
    cached = raw_metadata.get(ANALYSIS_CACHE_KEY)

    if not refresh and cached and cached.get("total_chords", 0) > 0:
        logger.info("Harmonic analysis cache hit for %s", corpus_id)
        return ScoreHarmonicAnalysis(**cached)

    # ── 2. Get signed URL ─────────────────────────────────────────────────
    storage_path = file_resp.data.get("storage_object_path", "")
    if not storage_path:
        raise HTTPException(status_code=422, detail="No storage path for this file")

    logger.info("Running harmonic analysis for %s", corpus_id)
    signed = _signed_url(supabase, storage_path)

    # ── 3. Run analysis in thread ─────────────────────────────────────────
    try:
        result_dict = await asyncio.to_thread(
            _run_harmonic_analysis, corpus_id, signed, storage_path
        )
    except Exception as exc:
        logger.exception("Harmonic analysis failed for %s", corpus_id)
        raise HTTPException(
            status_code=500, detail=f"Harmonic analysis failed: {exc}"
        ) from exc

    # ── 4. Cache ──────────────────────────────────────────────────────────
    try:
        supabase.from_("corpus_files").update(
            {"raw_metadata": {**raw_metadata, ANALYSIS_CACHE_KEY: result_dict}}
        ).eq("id", corpus_id).execute()
    except Exception:
        logger.warning("Failed to cache harmonic analysis for %s", corpus_id)

    return ScoreHarmonicAnalysis(**result_dict)


# ─── Core analysis pipeline (sync — runs in thread pool) ─────────────────────

def _run_harmonic_analysis(
    corpus_id: str,
    signed_url: str,
    storage_path: str,
) -> dict[str, Any]:
    """
    Full harmonic analysis pipeline using the existing music analysis engine.

    Assumption: score_loader.ScoreLoader.load(path) returns a Score object
    with .events: List[MusicalEvent]. If your ScoreLoader uses a different
    signature, adjust the import below.
    """
    from fractions import Fraction

    import music21 as m21

    # ── Download ──────────────────────────────────────────────────────────
    suffix = Path(storage_path).suffix.lower() or ".mid"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        urllib.request.urlretrieve(signed_url, tmp.name)  # noqa: S310
        tmp_path = tmp.name

    # ── Load into Score data structure ────────────────────────────────────
    # score_loader.py converts music21 → Score data structure.
    # Assumption: ScoreLoader().load(path) → Score
    # If your API is different (e.g. load_score(path)), adjust here.
    try:
        from src.core.score_loader import ScoreLoader
        score = ScoreLoader().load(tmp_path)
    except (ImportError, AttributeError):
        # Fallback: build a minimal Score from music21 directly
        logger.warning("ScoreLoader import failed — using music21 fallback for %s", corpus_id)
        score = _score_from_music21(tmp_path)

    if not score or not score.events:
        return _empty_analysis(corpus_id)

    # ── Key estimation ────────────────────────────────────────────────────
    from src.harmonic.key_estimator import KeyEstimator
    global_key = KeyEstimator().estimate_global(score)
    logger.info(
        "Key for %s: %s %s (confidence=%.2f)",
        corpus_id, global_key.tonic, global_key.mode, global_key.confidence,
    )

    # ── Chord classification ──────────────────────────────────────────────
    from src.harmonic.chord_classifier import ChordClassifier, ChordClassificationConfig
    from src.temporal.windower import WindowType

    config = ChordClassificationConfig(
        min_confidence=0.25,
        include_seventh_chords=True,
        beat_weight_exponent=1.0,
    )

    try:
        chords = ChordClassifier().classify_score(
            score,
            window_type=WindowType.BEAT,
            config=config,
        )
    except Exception as exc:
        logger.warning("Chord classification failed for %s: %s", corpus_id, exc)
        chords = []

    if not chords:
        return _empty_analysis(corpus_id)

    logger.info("Found %d chords for %s", len(chords), corpus_id)

    # ── Roman numeral analysis ────────────────────────────────────────────
    from src.harmonic.roman_numeral_analyzer import RomanNumeralAnalyzer

    try:
        roman_numerals = RomanNumeralAnalyzer().analyze_sequence(
            chords, [global_key]
        )
    except Exception as exc:
        logger.warning("Roman numeral analysis failed for %s: %s", corpus_id, exc)
        roman_numerals = []

    # ── Aggregate frequencies ─────────────────────────────────────────────
    total = len(chords)

    # Roman numeral frequency
    rn_counts: dict[str, dict] = {}
    for rn in roman_numerals:
        key_str = _rn_label(rn)
        if key_str not in rn_counts:
            rn_counts[key_str] = {"degree": rn.degree, "count": 0}
        rn_counts[key_str]["count"] += 1

    rn_freq = sorted(
        [
            {
                "numeral": label,
                "degree": data["degree"],
                "count": data["count"],
                "percentage": round(data["count"] / total * 100, 1),
            }
            for label, data in rn_counts.items()
        ],
        key=lambda x: -x["count"],
    )[:12]  # top 12

    # Chord quality frequency
    quality_counts: dict[str, int] = {}
    for ch in chords:
        quality_counts[ch.quality] = quality_counts.get(ch.quality, 0) + 1

    quality_freq = sorted(
        [
            {
                "quality": q,
                "count": c,
                "percentage": round(c / total * 100, 1),
            }
            for q, c in quality_counts.items()
        ],
        key=lambda x: -x["count"],
    )

    # Harmonic function frequency
    func_counts: dict[str, int] = {}
    for rn in roman_numerals:
        fn = rn.function.name if hasattr(rn.function, "name") else str(rn.function)
        func_counts[fn] = func_counts.get(fn, 0) + 1

    func_freq = [
        {
            "function": fn,
            "count": c,
            "percentage": round(c / total * 100, 1),
        }
        for fn, c in sorted(func_counts.items(), key=lambda x: -x[1])
    ]

    secondary_count = sum(1 for rn in roman_numerals if rn.is_secondary)
    borrowed_count  = 0  # RomanNumeralAnalyzer flags borrowed via parallel mode logic

    tonic_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    return {
        "corpus_id":                corpus_id,
        "global_key_tonic":         global_key.tonic,
        "global_key_mode":          global_key.mode,
        "global_key_confidence":    round(global_key.confidence, 3),
        "total_chords":             total,
        "roman_numerals":           rn_freq,
        "chord_qualities":          quality_freq,
        "harmonic_functions":       func_freq,
        "secondary_dominant_count": secondary_count,
        "borrowed_chord_count":     borrowed_count,
    }


def _empty_analysis(corpus_id: str) -> dict[str, Any]:
    return {
        "corpus_id": corpus_id,
        "global_key_tonic": 0,
        "global_key_mode": "major",
        "global_key_confidence": 0.0,
        "total_chords": 0,
        "roman_numerals": [],
        "chord_qualities": [],
        "harmonic_functions": [],
        "secondary_dominant_count": 0,
        "borrowed_chord_count": 0,
    }


def _rn_label(rn) -> str:
    """Convert RomanNumeral dataclass to display string like 'V', 'ii', 'V/V'."""
    degree_to_roman = {1:"I", 2:"II", 3:"III", 4:"IV", 5:"V", 6:"VI", 7:"VII"}
    base = degree_to_roman.get(rn.degree, str(rn.degree))

    # Lowercase for minor/diminished/half-dim
    if rn.quality in ("minor", "diminished", "half-dim7", "dim7", "minor7"):
        base = base.lower()

    if rn.quality == "diminished":
        base += "°"
    elif rn.quality == "dim7":
        base += "°7"
    elif rn.quality == "half-dim7":
        base += "ø7"
    elif rn.quality in ("dominant7", "major7", "minor7"):
        base += "7"

    if rn.is_secondary and rn.secondary_target:
        target_roman = degree_to_roman.get(rn.secondary_target, "?")
        base = f"{base}/{target_roman}"

    return base


def _score_from_music21(path: str):
    """
    Fallback Score builder if ScoreLoader import fails.
    Converts music21 events → MusicalEvent list → Score.
    """
    import music21 as m21
    from fractions import Fraction

    try:
        from src.core.data_structures import MusicalEvent, Score, TimeSignature
    except ImportError:
        return None

    m21_score = m21.converter.parse(path)
    if m21_score.hasPartLikeStreams():
        parts = list(m21_score.parts)
        flat = parts[0].flatten() if parts else m21_score.flatten()
    else:
        flat = m21_score.flatten()

    ts_obj = flat.recurse().getElementsByClass(m21.meter.TimeSignature).first()
    ts = TimeSignature(
        numerator=ts_obj.numerator if ts_obj else 4,
        denominator=ts_obj.denominator if ts_obj else 4,
    )

    events = []
    for el in flat.notesAndRests:
        offset   = Fraction(el.offset).limit_denominator(64)
        duration = Fraction(el.duration.quarterLength).limit_denominator(64)

        if el.isRest:
            events.append(MusicalEvent(
                onset=offset, duration=duration,
                pitch=None, is_rest=True, beat_strength=0.1,
            ))
        else:
            pitches = el.pitches if hasattr(el, "pitches") else [el.pitch]
            for p in pitches:
                events.append(MusicalEvent(
                    onset=offset, duration=duration,
                    pitch=p.midi, is_rest=False,
                    beat_strength=el.beatStrength if hasattr(el, "beatStrength") else 0.5,
                ))

    return Score(events=events, time_signature=ts)