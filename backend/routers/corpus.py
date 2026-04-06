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
        storage_object_path=row.get("storage_object_path", ""),
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
    row.signed_url = _signed_url(supabase, resp.data["storage_object_path"])
    return row


@corpus_router.get("/corpus/{corpus_id}/notation", response_model=VexFlowPayload)
async def get_notation(corpus_id: str, supabase=Depends(get_supabase)):
    """
    Returns VexFlow-ready notation payload for a corpus file.

    Cache strategy: result is stored in corpus_files.raw_metadata JSONB after
    first parse, so subsequent calls are instant.
    """
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
    if "vexflow" in raw_metadata:
        return VexFlowPayload(**raw_metadata["vexflow"])

    storage_path = file_resp.data["storage_object_path"]
    signed = _signed_url(supabase, storage_path)

    try:
        payload = await asyncio.to_thread(_parse_score_to_vexflow, signed, storage_path)
    except Exception as exc:
        logger.exception("Notation extraction failed for %s", corpus_id)
        raise HTTPException(
            status_code=500, detail=f"Notation extraction failed: {exc}"
        ) from exc

    supabase.from_("corpus_files").update(
        {"raw_metadata": {**raw_metadata, "vexflow": payload.model_dump()}}
    ).eq("id", corpus_id).execute()

    return payload


# ─── music21 → VexFlow (synchronous, runs in thread pool) ────────────────────

def _parse_score_to_vexflow(signed_url: str, storage_path: str) -> VexFlowPayload:
    import tempfile
    import urllib.request
    from fractions import Fraction

    import music21 as m21

    suffix = Path(storage_path).suffix or ".mid"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        urllib.request.urlretrieve(signed_url, tmp.name)  # noqa: S310
        tmp_path = tmp.name

    score = m21.converter.parse(tmp_path)

    # Flatten single-part scores (MuseScore quirk — mirrors score_loader.py)
    if score.hasPartLikeStreams():
        parts = list(score.parts)
        if len(parts) == 1:
            score = parts[0].flatten()
    else:
        score = score.flatten()

    ts_obj = score.recurse().getElementsByClass(m21.meter.TimeSignature).first()
    ks_obj = score.recurse().getElementsByClass(m21.key.KeySignature).first()

    time_sig_str = f"{ts_obj.numerator}/{ts_obj.denominator}" if ts_obj else "4/4"
    key_sig_str = ks_obj.asKey().name if ks_obj else "C major"

    measures: list[VexFlowMeasure] = []
    total_beats = Fraction(0)

    for i, measure in enumerate(score.getElementsByClass(m21.stream.Measure)):
        start_beat = total_beats
        dur = Fraction(measure.duration.quarterLength).limit_denominator(64)
        end_beat = start_beat + dur

        notes: list[VexFlowNote] = []
        for el in measure.flatten().notesAndRests:
            el_dur = str(Fraction(el.duration.quarterLength).limit_denominator(64))
            el_offset = str(Fraction(el.offset).limit_denominator(64))

            if el.isRest:
                notes.append(VexFlowNote(type="rest", duration=el_dur, offset=el_offset))
            elif hasattr(el, "pitches") and len(el.pitches) > 1:
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

    return VexFlowPayload(
        measures=measures,
        time_signature=time_sig_str,
        key_signature=key_sig_str,
        total_beats=float(total_beats),
    )