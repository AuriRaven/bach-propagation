"""
backend/routers/corpus.py

All corpus API routes. Registered in main.py with:
    app.include_router(corpus_router, prefix="/api")

Service role Supabase client stays server-side only.
Signed URLs are generated here — never exposed as a raw key to the frontend.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

corpus_router = APIRouter(tags=["corpus"])

SIGNED_URL_EXPIRY = 3600  # 1 hour


# ─── Supabase client ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_supabase():
    """Cached service-role client. Never expose to the frontend."""
    import os
    from supabase import create_client
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )


# ─── Pydantic models ──────────────────────────────────────────────────────────

class CorpusFileRow(BaseModel):
    id: str
    bwv: str | None = None
    movement_name: str | None = None
    collection: str
    storage_object_path: str
    storage_url: str | None = None
    load_status: str
    key_signature: str | None = None
    time_signature: str | None = None
    num_measures: int | None = None
    num_voices: int | None = None
    duration_seconds: float | None = None
    form_tag: str | None = None
    instrument_family: str | None = None
    signed_url: str | None = None


class PaginatedResponse(BaseModel):
    items: list[CorpusFileRow]
    total: int
    page: int
    page_size: int
    has_next: bool


class CorpusStats(BaseModel):
    total: int
    by_collection: dict[str, int]
    by_key_mode: dict[str, int]


class VexFlowNote(BaseModel):
    type: str                       # "note" | "rest"
    pitch: str | None = None        # single pitch e.g. "C4"
    pitches: list[str] | None = None  # chord pitches
    duration: str                   # fraction string e.g. "1/4"
    offset: str                     # beat offset within measure


class VexFlowMeasure(BaseModel):
    index: int
    start_beat: float
    end_beat: float
    notes: list[VexFlowNote]


class VexFlowPayload(BaseModel):
    measures: list[VexFlowMeasure]
    time_signature: str
    key_signature: str
    total_beats: float


class NotationParseRequest(BaseModel):
    """Request body for POST /api/notation/parse."""
    musicxml_b64: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _signed_url(supabase, path: str) -> str:
    resp = supabase.storage.from_("baroque-corpus").create_signed_url(
        path, SIGNED_URL_EXPIRY
    )
    if not resp or "signedURL" not in resp:
        raise HTTPException(status_code=502, detail="Failed to generate signed URL")
    return resp["signedURL"]


def _row_to_model(row: dict) -> CorpusFileRow:
    return CorpusFileRow(
        id=row["id"],
        bwv=row.get("bwv"),
        movement_name=row.get("movement_name"),
        collection=row.get("collection", ""),
        storage_object_path=row.get("storage_object_path", ""),  # may be absent in corpus_full
        storage_url=row.get("storage_url"),
        load_status=row.get("load_status", ""),
        key_signature=row.get("key_signature"),
        time_signature=row.get("time_signature"),
        num_measures=row.get("num_measures"),
        num_voices=row.get("num_voices"),
        duration_seconds=row.get("duration_seconds"),
        form_tag=row.get("form_tag"),
        instrument_family=row.get("instrument_family"),
    )


# ─── Routes — order matters: specific paths before /{id} ─────────────────────

@corpus_router.get("/corpus/stats", response_model=CorpusStats)
async def corpus_stats(supabase=Depends(get_supabase)):
    """Aggregate statistics — total, by_collection, by_key_mode."""
    resp = (
        supabase.from_("corpus_full")
        .select("collection, key_signature", count="exact")
        .execute()
    )
    if resp.data is None:
        raise HTTPException(status_code=502, detail="Database query failed")

    by_collection: dict[str, int] = {}
    by_key_mode: dict[str, int] = {}

    for row in resp.data:
        col = row.get("collection") or "unknown"
        by_collection[col] = by_collection.get(col, 0) + 1

        ks = (row.get("key_signature") or "").lower()
        mode = "minor" if "minor" in ks else "major" if "major" in ks else "unknown"
        by_key_mode[mode] = by_key_mode.get(mode, 0) + 1

    return CorpusStats(
        total=resp.count or len(resp.data),
        by_collection=by_collection,
        by_key_mode=by_key_mode,
    )


@corpus_router.get("/corpus/search", response_model=list[CorpusFileRow])
async def search_corpus(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    supabase=Depends(get_supabase),
):
    """Semantic search via OpenAI text-embedding-3-small + pgvector match_corpus RPC."""
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    emb = await asyncio.to_thread(
        lambda: client.embeddings.create(model="text-embedding-3-small", input=q)
    )
    query_embedding = emb.data[0].embedding

    rpc_resp = supabase.rpc(
        "match_corpus",
        {
            "query_embedding": query_embedding,
            "match_count": limit,
            "filter_collection": None,
            "filter_key_mode": None,
        },
    ).execute()

    if rpc_resp.data is None:
        raise HTTPException(status_code=502, detail="Semantic search failed")

    return [_row_to_model(r) for r in rpc_resp.data]


@corpus_router.get("/corpus", response_model=PaginatedResponse)
async def list_corpus(
    collection: str | None = Query(None),
    key_mode: str | None = Query(None),
    form_tag: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    supabase=Depends(get_supabase),
):
    """Paginated corpus listing with optional filters from the corpus_full view."""
    offset = (page - 1) * page_size

    q = supabase.from_("corpus_full").select("*", count="exact")
    if collection:
        q = q.eq("collection", collection)
    if form_tag:
        q = q.eq("form_tag", form_tag)
    if key_mode:
        q = q.ilike("key_signature", f"%{key_mode}%")

    resp = q.range(offset, offset + page_size - 1).order("bwv").execute()
    if resp.data is None:
        raise HTTPException(status_code=502, detail="Database query failed")

    total = resp.count or 0
    return PaginatedResponse(
        items=[_row_to_model(r) for r in resp.data],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(offset + page_size) < total,
    )


@corpus_router.get("/corpus/{corpus_id}", response_model=CorpusFileRow)
async def get_corpus_file(corpus_id: str, supabase=Depends(get_supabase)):
    """Single corpus file with a fresh 1-hour signed URL."""
    # corpus_full has all the metadata fields
    resp = (
        supabase.from_("corpus_full")
        .select("*")
        .eq("id", corpus_id)
        .single()
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Corpus file not found")

    row = _row_to_model(resp.data)

    # storage_object_path lives on corpus_files, not corpus_full view
    # fetch it separately to generate the signed URL
    path_resp = (
        supabase.from_("corpus_files")
        .select("storage_object_path")
        .eq("id", corpus_id)
        .single()
        .execute()
    )
    storage_path = (path_resp.data or {}).get("storage_object_path", "")
    if storage_path:
        row.signed_url = _signed_url(supabase, storage_path)
    else:
        logger.warning("No storage_object_path for corpus file %s", corpus_id)

    return row


@corpus_router.get("/corpus/{corpus_id}/notation", response_model=VexFlowPayload)
async def get_notation(
    corpus_id: str,
    refresh: bool = False,          # ?refresh=true busts the cache
    supabase=Depends(get_supabase),
):
    """
    Returns notation payload for a corpus file.

    Cache: stored in corpus_files.raw_metadata after first parse.
    Add ?refresh=true to force re-parse (useful when cache has 0 measures).
    """
    # ── 1. Fetch file row ─────────────────────────────────────────────────
    try:
        file_resp = (
            supabase.from_("corpus_files")
            .select("id, storage_object_path, raw_metadata")
            .eq("id", corpus_id)
            .single()
            .execute()
        )
    except Exception:
        logger.warning("raw_metadata column unavailable, falling back for %s", corpus_id)
        file_resp = (
            supabase.from_("corpus_files")
            .select("id, storage_object_path")
            .eq("id", corpus_id)
            .single()
            .execute()
        )

    if not file_resp.data:
        raise HTTPException(status_code=404, detail="Corpus file not found")

    # ── 2. Cache check — skip if refresh=true OR cache has 0 measures ────
    raw_metadata: dict = file_resp.data.get("raw_metadata") or {}
    cached = raw_metadata.get("vexflow")

    use_cache = (
        not refresh
        and cached is not None
        and isinstance(cached.get("measures"), list)
        and len(cached["measures"]) > 0   # never serve a 0-measure cache
    )

    if use_cache:
        logger.info("Notation cache hit for %s (%d measures)", corpus_id, len(cached["measures"]))
        return VexFlowPayload(**cached)

    if cached is not None and not use_cache:
        logger.warning(
            "Stale/empty notation cache for %s (measures=%s) — re-parsing",
            corpus_id,
            len(cached.get("measures", [])) if cached else "none",
        )

    # ── 3. Download + parse ───────────────────────────────────────────────
    storage_path = file_resp.data.get("storage_object_path", "")
    if not storage_path:
        raise HTTPException(status_code=422, detail="No storage path for this file")

    logger.info("Parsing notation for %s (%s)", corpus_id, storage_path)
    signed = _signed_url(supabase, storage_path)

    try:
        payload = await asyncio.to_thread(_parse_score_to_vexflow, signed, storage_path)
    except Exception as exc:
        logger.exception("Notation extraction failed for %s", corpus_id)
        raise HTTPException(
            status_code=500, detail=f"Notation extraction failed: {exc}"
        ) from exc

    if len(payload.measures) == 0:
        logger.warning("Parse produced 0 measures for %s — not caching", corpus_id)
        # Return payload anyway so frontend can show an appropriate message
        return payload

    # ── 4. Cache ──────────────────────────────────────────────────────────
    try:
        supabase.from_("corpus_files").update(
            {"raw_metadata": {**raw_metadata, "vexflow": payload.model_dump()}}
        ).eq("id", corpus_id).execute()
        logger.info("Cached notation for %s (%d measures)", corpus_id, len(payload.measures))
    except Exception:
        logger.warning("Failed to cache notation for %s", corpus_id)

    return payload


@corpus_router.post("/notation/parse", response_model=VexFlowPayload)
async def parse_notation(req: NotationParseRequest):
    """
    Parse base64-encoded MusicXML into a VexFlowPayload.

    Used by the generation pipeline: the model produces MusicXML,
    this endpoint converts it to the same format the piano roll expects.
    No Supabase dependency — purely a conversion utility.
    """
    try:
        payload = await asyncio.to_thread(
            _parse_musicxml_b64_to_vexflow, req.musicxml_b64
        )
    except Exception as exc:
        logger.exception("MusicXML base64 parse failed")
        raise HTTPException(
            status_code=422, detail=f"Failed to parse MusicXML: {exc}"
        ) from exc

    return payload


# ─── music21 → notation payload (synchronous, runs in thread pool) ───────────

def _parse_file_to_vexflow(file_path: str, suffix: str) -> VexFlowPayload:
    """
    Core parser: local file path → VexFlowPayload.

    Shared by both the corpus notation endpoint (downloads from Supabase)
    and the generation notation endpoint (decodes from base64).
    """
    from fractions import Fraction

    import music21 as m21

    score = m21.converter.parse(file_path)

    # ── Extract time + key sig before flattening ──────────────────────────
    ts_obj = score.recurse().getElementsByClass(m21.meter.TimeSignature).first()
    ks_obj = score.recurse().getElementsByClass(m21.key.KeySignature).first()

    time_sig_str = f"{ts_obj.numerator}/{ts_obj.denominator}" if ts_obj else "4/4"
    key_sig_str  = ks_obj.asKey().name if ks_obj else "C major"

    beats_per_measure = int(ts_obj.numerator) if ts_obj else 4

    # ── Collapse to a single flat stream of notes ─────────────────────────
    # For multi-part scores use Part 0 (melody / top voice)
    if score.hasPartLikeStreams():
        parts = list(score.parts)
        working = parts[0] if parts else score
    else:
        working = score

    # MIDI files need explicit measure creation
    if suffix in (".mid", ".midi"):
        working = working.flatten()
        working = working.makeMeasures(inPlace=False)
    else:
        # MusicXML already has measures — just flatten within each measure
        pass

    # ── Iterate measures ──────────────────────────────────────────────────
    measures: list[VexFlowMeasure] = []
    total_beats = Fraction(0)

    measure_stream = list(working.getElementsByClass(m21.stream.Measure))
    if not measure_stream:
        # Last resort: treat the whole piece as one measure
        working = working.flatten().makeMeasures(inPlace=False)
        measure_stream = list(working.getElementsByClass(m21.stream.Measure))

    for i, measure in enumerate(measure_stream):
        start_beat = total_beats
        dur = Fraction(measure.duration.quarterLength).limit_denominator(64)
        if dur == 0:
            dur = Fraction(beats_per_measure)
        end_beat = start_beat + dur

        notes: list[VexFlowNote] = []
        for el in measure.flatten().notesAndRests:
            el_dur = str(Fraction(el.duration.quarterLength).limit_denominator(64))
            # absolute beat offset = measure start + element offset within measure
            abs_offset = float(start_beat + Fraction(el.offset).limit_denominator(64))
            el_offset = str(abs_offset)

            if el.isRest:
                notes.append(VexFlowNote(type="rest", duration=el_dur, offset=el_offset))
            elif hasattr(el, "pitches") and len(el.pitches) > 1:
                # Chord — store all pitches + MIDI numbers for piano roll
                notes.append(VexFlowNote(
                    type="note",
                    pitches=[str(p) for p in el.pitches],
                    duration=el_dur,
                    offset=el_offset,
                ))
            else:
                notes.append(VexFlowNote(
                    type="note",
                    pitch=str(el.pitch),
                    duration=el_dur,
                    offset=el_offset,
                ))

        measures.append(VexFlowMeasure(
            index=i,
            start_beat=float(start_beat),
            end_beat=float(end_beat),
            notes=notes,
        ))
        total_beats = end_beat

    logger.info(
        "Parsed %d measures, %.1f total beats, key=%s, time=%s",
        len(measures), float(total_beats), key_sig_str, time_sig_str,
    )

    return VexFlowPayload(
        measures=measures,
        time_signature=time_sig_str,
        key_signature=key_sig_str,
        total_beats=float(total_beats),
    )


def _parse_score_to_vexflow(signed_url: str, storage_path: str) -> VexFlowPayload:
    """
    Download from signed URL + parse.
    Used by GET /api/corpus/{id}/notation.
    """
    import tempfile
    import urllib.request

    suffix = Path(storage_path).suffix.lower() or ".mid"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        urllib.request.urlretrieve(signed_url, tmp.name)  # noqa: S310
        tmp_path = tmp.name

    return _parse_file_to_vexflow(tmp_path, suffix)


def _parse_musicxml_b64_to_vexflow(musicxml_b64: str) -> VexFlowPayload:
    """
    Decode base64 MusicXML + parse.
    Used by POST /api/notation/parse.
    """
    import base64
    import tempfile

    xml_bytes = base64.b64decode(musicxml_b64)

    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        tmp.write(xml_bytes)
        tmp_path = tmp.name

    return _parse_file_to_vexflow(tmp_path, ".musicxml")