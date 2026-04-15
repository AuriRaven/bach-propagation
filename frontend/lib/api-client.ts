/**
 * frontend/lib/api-client.ts
 *
 * Typed wrappers over fetch() for every /api/* endpoint.
 * All requests go to FastAPI — never directly to Supabase from the client.
 */

import type { CorpusFile, VexFlowPayload } from "./app-state"

// Support both env var names for docker-compose compatibility
const BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "http://localhost:8000"

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = "ApiError"
  }
}

async function fetcher<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = 8_000,
): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...init,
    })
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
      throw new ApiError(res.status, text)
    }
    return res.json() as Promise<T>
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new ApiError(408, `Request timed out after ${timeoutMs / 1000}s`)
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}

// ─── Corpus types ─────────────────────────────────────────────────────────────

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  has_next: boolean
}

export interface CorpusListFilters {
  collection?: string
  key_mode?: string
  form_tag?: string
  page?: number
  page_size?: number
}

export interface CorpusStats {
  total: number
  by_collection: Record<string, number>
  by_key_mode: Record<string, number>
}

// ─── Analysis types ───────────────────────────────────────────────────────────

export interface KeyModeCount    { mode: string; count: number }
export interface CollectionCount { collection: string; count: number }
export interface FormTagCount    { form_tag: string; count: number }
export interface VoiceCount      { num_voices: number; count: number }
export interface Bucket          { label: string; min_s: number; max_s: number; count: number }

export interface CorpusAnalytics {
  total: number
  by_key_mode: KeyModeCount[]
  by_collection: CollectionCount[]
  by_form_tag: FormTagCount[]
  by_voice_count: VoiceCount[]
  duration_histogram: Bucket[]
  measures_histogram: Bucket[]
}

export interface RomanNumeralFreq  { numeral: string; degree: number; count: number; percentage: number }
export interface ChordQualityFreq  { quality: string; count: number; percentage: number }
export interface HarmonicFuncFreq  { function: string; count: number; percentage: number }

export interface ScoreHarmonicAnalysis {
  corpus_id: string
  global_key_tonic: number
  global_key_mode: string
  global_key_confidence: number
  total_chords: number
  roman_numerals: RomanNumeralFreq[]
  chord_qualities: ChordQualityFreq[]
  harmonic_functions: HarmonicFuncFreq[]
  secondary_dominant_count: number
  borrowed_chord_count: number
}

// ─── Generation types ─────────────────────────────────────────────────────────

export interface GenerationRequest {
  key_mode?: "minor" | "major"
  n_tokens?: number          // 16–512
  temperature?: number       // 0.1–2.0
  top_k?: number             // 1–50
  prompt_bwv?: string | null
}

export interface GenerationResponse {
  chord_tokens: number[]
  rn_sequence: string[]
  tonal_score: number
  is_valid: boolean
  forbidden_rate: number
  musicxml_b64: string
  generation_time_ms: number
}

// ─── SSE types ────────────────────────────────────────────────────────────────

export type SseEvent =
  | { type: "text_delta"; text: string }
  | { type: "tool_use"; tool: "searchCorpus"; result: CorpusFile[] }
  | { type: "tool_use"; tool: "fetchAnalysis"; result: { filter_collection?: string; analysis_type?: string } }
  | { type: "done" }

export interface ChatMessage { role: "user" | "assistant"; content: string }
export interface AppContext   { active_nav?: string; active_score_id?: string; active_score_name?: string }

export async function* streamAiChat(
  messages: ChatMessage[],
  context: AppContext = {},
): AsyncGenerator<SseEvent> {
  const res = await fetch(`${BASE_URL}/api/ai/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, context }),
  })
  if (!res.ok || !res.body) throw new ApiError(res.status, await res.text())

  const reader  = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n\n")
    buffer = lines.pop() ?? ""
    for (const line of lines) {
      const trimmed = line.replace(/^data: /, "").trim()
      if (!trimmed) continue
      try { yield JSON.parse(trimmed) as SseEvent } catch { /* skip */ }
    }
  }
}

// ─── API surface ──────────────────────────────────────────────────────────────

export const api = {
  corpus: {
    list(filters: CorpusListFilters = {}): Promise<PaginatedResponse<CorpusFile>> {
      const p = new URLSearchParams()
      if (filters.collection) p.set("collection", filters.collection)
      if (filters.key_mode)   p.set("key_mode",   filters.key_mode)
      if (filters.form_tag)   p.set("form_tag",   filters.form_tag)
      if (filters.page)       p.set("page",       String(filters.page))
      if (filters.page_size)  p.set("page_size",  String(filters.page_size))
      const qs = p.toString() ? `?${p}` : ""
      return fetcher<PaginatedResponse<CorpusFile>>(`/api/corpus${qs}`)
    },

    get(id: string): Promise<CorpusFile> {
      return fetcher<CorpusFile>(`/api/corpus/${id}`)
    },

    notation(id: string, refresh = false): Promise<VexFlowPayload> {
      const qs = refresh ? "?refresh=true" : ""
      return fetcher<VexFlowPayload>(`/api/corpus/${id}/notation${qs}`, undefined, 60_000)
    },

    search(query: string, limit = 10): Promise<CorpusFile[]> {
      const p = new URLSearchParams({ q: query, limit: String(limit) })
      return fetcher<CorpusFile[]>(`/api/corpus/search?${p}`)
    },

    stats(): Promise<CorpusStats> {
      return fetcher<CorpusStats>("/api/corpus/stats")
    },
  },

  analysis: {
    /** Corpus-level aggregate analytics — fast, no MIDI parsing */
    corpus(): Promise<CorpusAnalytics> {
      return fetcher<CorpusAnalytics>("/api/analysis/corpus", undefined, 15_000)
    },

    /** Per-score harmonic analysis — slow first run (~30s), cached after */
    score(id: string, refresh = false): Promise<ScoreHarmonicAnalysis> {
      const qs = refresh ? "?refresh=true" : ""
      return fetcher<ScoreHarmonicAnalysis>(
        `/api/analysis/score/${id}${qs}`,
        undefined,
        90_000,   // 90s — chord classification on large scores can be slow
      )
    },
  },

  notation: {
    /** Parse base64-encoded MusicXML into VexFlowPayload for the piano roll.
     *  Used after generation to display the result. */
    parse(musicxml_b64: string): Promise<VexFlowPayload> {
      return fetcher<VexFlowPayload>(
        "/api/notation/parse",
        { method: "POST", body: JSON.stringify({ musicxml_b64 }) },
        60_000,  // 60s — music21 parsing can be slow
      )
    },
  },

  generation: {
    /** Generate chord sequence via the Music Transformer model.
     *  Returns 503 if the model checkpoint is not loaded. */
    generate(req: GenerationRequest): Promise<GenerationResponse> {
      return fetcher<GenerationResponse>(
        "/api/generate",
        { method: "POST", body: JSON.stringify(req) },
        120_000,  // 120s — generation on CPU can be slow
      )
    },
  },
}