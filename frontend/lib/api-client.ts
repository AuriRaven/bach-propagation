/**
 * frontend/lib/api-client.ts
 *
 * Typed wrappers over fetch() for every /api/* endpoint.
 * All requests go to FastAPI — never directly to Supabase from the client.
 */

import type { CorpusFile, VexFlowPayload } from "./app-state"

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = "ApiError"
  }
}

async function fetcher<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new ApiError(res.status, text)
  }
  return res.json() as Promise<T>
}

// ─── Response types ───────────────────────────────────────────────────────────

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

// ─── SSE types ────────────────────────────────────────────────────────────────

export type SseEvent =
  | { type: "text_delta"; text: string }
  | { type: "tool_use"; tool: "searchCorpus"; result: CorpusFile[] }
  | { type: "tool_use"; tool: "fetchAnalysis"; result: { filter_collection?: string; analysis_type?: string } }
  | { type: "done" }

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

export interface AppContext {
  active_nav?: string
  active_score_id?: string
  active_score_name?: string
}

/**
 * Streams Server-Sent Events from POST /api/ai/chat.
 * Yields typed SseEvent objects — caller owns side-effects.
 */
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

  const reader = res.body.getReader()
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
      try { yield JSON.parse(trimmed) as SseEvent } catch { /* skip malformed */ }
    }
  }
}

// ─── API surface ──────────────────────────────────────────────────────────────

export const api = {
  corpus: {
    list(filters: CorpusListFilters = {}): Promise<PaginatedResponse<CorpusFile>> {
      const p = new URLSearchParams()
      if (filters.collection) p.set("collection", filters.collection)
      if (filters.key_mode)   p.set("key_mode", filters.key_mode)
      if (filters.form_tag)   p.set("form_tag", filters.form_tag)
      if (filters.page)       p.set("page", String(filters.page))
      if (filters.page_size)  p.set("page_size", String(filters.page_size))
      const qs = p.toString() ? `?${p}` : ""
      return fetcher<PaginatedResponse<CorpusFile>>(`/api/corpus${qs}`)
    },

    get(id: string): Promise<CorpusFile> {
      return fetcher<CorpusFile>(`/api/corpus/${id}`)
    },

    notation(id: string): Promise<VexFlowPayload> {
      return fetcher<VexFlowPayload>(`/api/corpus/${id}/notation`)
    },

    search(query: string, limit = 10): Promise<CorpusFile[]> {
      const p = new URLSearchParams({ q: query, limit: String(limit) })
      return fetcher<CorpusFile[]>(`/api/corpus/search?${p}`)
    },

    stats(): Promise<CorpusStats> {
      return fetcher<CorpusStats>("/api/corpus/stats")
    },
  },
}