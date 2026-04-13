"""
backend/routers/generation.py

POST /api/generate — Grammar-constrained Bach-style generation.

The MusicGenerator is loaded once at application startup via the
FastAPI lifespan context manager and reused across all requests.
If the checkpoint file is not found at startup, the generator is
set to None and the endpoint returns HTTP 503.

Register in main.py:
    from routers.generation import generation_router, lifespan
    # Pass lifespan to FastAPI(..., lifespan=lifespan)
    app.include_router(generation_router, prefix="/api")
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global generator state
# ---------------------------------------------------------------------------

_generator: Optional["MusicGenerator"] = None   # type: ignore[name-defined]

# Default paths — can be overridden by calling load_generator() directly
_DEFAULT_CHECKPOINT   = "checkpoints/transformer-v1/best.pt"
_DEFAULT_VOCAB        = "data/encoded/vocab.json"
_DEFAULT_MATRIX       = "data/encoded/transition_matrix.json"


async def load_generator(
    checkpoint_path: str = _DEFAULT_CHECKPOINT,
    vocab_path:      str = _DEFAULT_VOCAB,
    transition_matrix_path: str = _DEFAULT_MATRIX,
    device: str = "auto",
) -> None:
    """
    Load the MusicGenerator and store it in the module-level singleton.
    Call this from the FastAPI lifespan or startup event.

    If the checkpoint does not exist, logs a WARNING and leaves _generator
    as None. The endpoint will return 503 in that case.
    """
    global _generator

    for p in (checkpoint_path, vocab_path, transition_matrix_path):
        if not Path(p).exists():
            logger.warning(
                "Generator not loaded — file not found: %s. "
                "POST /api/generate will return 503.", p
            )
            return

    try:
        from src.models.generator import MusicGenerator
        logger.info("Loading MusicGenerator from %s …", checkpoint_path)

        # Run in thread — loading a checkpoint is synchronous I/O
        _generator = await asyncio.to_thread(
            MusicGenerator.from_checkpoint,
            checkpoint_path,
            vocab_path,
            transition_matrix_path,
            device,
        )
        logger.info("MusicGenerator loaded successfully.")
    except Exception as exc:
        logger.warning("Failed to load MusicGenerator: %s. "
                       "POST /api/generate will return 503.", exc)
        _generator = None


# ---------------------------------------------------------------------------
# Lifespan — modern FastAPI pattern (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    """
    FastAPI lifespan context manager.
    Usage in main.py:
        from routers.generation import lifespan
        app = FastAPI(..., lifespan=lifespan)
    """
    await load_generator()
    yield
    # Teardown (nothing required for the generator)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerationRequest(BaseModel):
    key_mode:    str   = Field(default="minor", pattern="^(minor|major)$")
    n_tokens:    int   = Field(default=128, ge=16, le=512)
    temperature: float = Field(default=0.9, ge=0.1, le=2.0)
    top_k:       int   = Field(default=10, ge=1, le=50)
    prompt_bwv:  Optional[str] = Field(default=None,
                                        description="BWV number to use as prompt")


class GenerationResponse(BaseModel):
    chord_tokens:       list[int]
    rn_sequence:        list[str]
    tonal_score:        float
    is_valid:           bool
    forbidden_rate:     float
    musicxml_b64:       str
    generation_time_ms: float


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

generation_router = APIRouter(tags=["generation"])


@generation_router.post("/generate", response_model=GenerationResponse)
async def generate_music(request: GenerationRequest):
    """
    Generate a Bach-style chord sequence constrained by Baroque grammar.

    The generator applies hard constraints at every decoding step:
    - Forbidden V7→IV and V7→iv transitions are blocked (logit = -inf)
    - Cadence tokens are boosted at required metric positions
    - Output is validated by GrammarValidator after generation

    Returns MusicXML as a base64-encoded string (decode and open in MuseScore).

    Raises:
        503: If the model checkpoint was not found at startup.
        500: If generation fails unexpectedly.
    """
    if _generator is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Music generator not available. "
                f"Checkpoint not found at '{_DEFAULT_CHECKPOINT}'. "
                "Train the model first and ensure the checkpoint exists."
            ),
        )

    t_start = time.perf_counter()

    try:
        # Run generation in a thread — avoids blocking the event loop
        # (torch inference is synchronous)
        seq = await asyncio.to_thread(
            _generator.generate,
            request.key_mode,
            request.n_tokens,
            request.temperature,
            request.top_k,
        )

        musicxml_b64 = await asyncio.to_thread(
            _generator.generate_musicxml,
            request.key_mode,
            request.n_tokens,
            request.temperature,
        )
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Compute forbidden_rate for the response
    from src.grammar.rules import BaroqueGrammar
    grammar = BaroqueGrammar()
    rns = seq.rn_sequence
    forbidden = sum(
        1 for i in range(len(rns) - 1)
        if not grammar.is_valid_transition(rns[i], rns[i + 1], request.key_mode)
    )
    forbidden_rate = forbidden / max(len(rns) - 1, 1)

    return GenerationResponse(
        chord_tokens=seq.chord_tokens,
        rn_sequence=seq.rn_sequence,
        tonal_score=seq.grammar_report.tonal_score,
        is_valid=seq.grammar_report.is_valid,
        forbidden_rate=forbidden_rate,
        musicxml_b64=musicxml_b64,
        generation_time_ms=round(elapsed_ms, 1),
    )