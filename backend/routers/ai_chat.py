"""
backend/routers/ai_chat.py

POST /api/ai/chat — GPT-4o powered chat with two tools:
  • searchCorpus  → semantic vector search, returns CorpusFileRow list
  • fetchAnalysis → signals frontend to switch to Analysis view

Response is streamed as Server-Sent Events so tokens appear progressively.
Uses OpenAI only — no Anthropic dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .corpus import get_supabase, _row_to_model

logger = logging.getLogger(__name__)

ai_router = APIRouter(tags=["ai"])


# ─── Request / Response Models ────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class AppContext(BaseModel):
    active_nav: str | None = None
    active_score_id: str | None = None
    active_score_name: str | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    context: AppContext = AppContext()


# ─── Tool definitions (OpenAI function calling format) ────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "searchCorpus",
            "description": (
                "Search the Bach corpus for pieces matching a natural language query. "
                "Use when the user asks to find, browse, or explore Bach pieces by "
                "key, form, collection, or musical character."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query, e.g. 'fugue in G minor'",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 8)",
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetchAnalysis",
            "description": (
                "Navigate to the Analysis view and optionally pre-filter it. "
                "Use when the user asks about statistics, distributions, or "
                "analysis (e.g. 'show key distribution of chorales')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter_collection": {
                        "type": "string",
                        "description": "Collection to pre-filter, e.g. 'chorales'",
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["key_distribution", "form_distribution", "overview"],
                    },
                },
                "required": [],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an expert music theorist and Bach specialist embedded
in the Bach Propagation research workbench. Help users explore the corpus and
understand Baroque music theory.

Available tools:
- searchCorpus: find Bach pieces by natural language query
- fetchAnalysis: navigate to statistical analysis views

For music theory questions, answer directly and concisely. Keep responses
focused — the user is a researcher working toward a thesis deadline."""


# ─── Route ────────────────────────────────────────────────────────────────────

@ai_router.post("/ai/chat")
async def ai_chat(request: ChatRequest, supabase=Depends(get_supabase)):
    """
    Streams a GPT-4o response with tool_use support as Server-Sent Events.

    SSE event shapes:
      {"type": "text_delta",  "text": "..."}
      {"type": "tool_use",    "tool": "searchCorpus",  "result": [...]}
      {"type": "tool_use",    "tool": "fetchAnalysis", "result": {...}}
      {"type": "done"}
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    context_note = ""
    if request.context.active_score_name:
        context_note = f"\n\n[Active workbench score: {request.context.active_score_name}]"

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    if context_note and messages:
        messages[-1]["content"] += context_note

    async def event_stream():
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1024,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
                tools=TOOLS,
                tool_choice="auto",
            )
        )

        choice = response.choices[0]
        message = choice.message

        # Stream any text content
        if message.content:
            yield f"data: {json.dumps({'type': 'text_delta', 'text': message.content})}\n\n"

        # Handle tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if name == "searchCorpus":
                    result = await _run_search(
                        args.get("query", ""),
                        args.get("limit", 8),
                        supabase,
                    )
                    yield f"data: {json.dumps({'type': 'tool_use', 'tool': 'searchCorpus', 'result': result})}\n\n"

                elif name == "fetchAnalysis":
                    result = {
                        "filter_collection": args.get("filter_collection"),
                        "analysis_type": args.get("analysis_type", "overview"),
                    }
                    yield f"data: {json.dumps({'type': 'tool_use', 'tool': 'fetchAnalysis', 'result': result})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _run_search(query: str, limit: int, supabase) -> list[dict]:
    """Embed query and call match_corpus RPC."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    emb = await asyncio.to_thread(
        lambda: client.embeddings.create(model="text-embedding-3-small", input=query)
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

    rows = rpc_resp.data or []
    return [_row_to_model(r).model_dump() for r in rows]