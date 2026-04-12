/**
 * frontend/hooks/use-ai-chat.ts
 *
 * Copilot chat hook. Streams responses from POST /api/ai/chat via SSE.
 *
 * Tool routing:
 *   searchCorpus  → stores results in state, navigates to Library
 *   fetchAnalysis → navigates to Analysis
 *
 * Version card scaffold: messages that trigger a tool_use get a
 * `versionCard` field attached — ready for Play/Revert once the
 * transformer model exists.
 */

"use client"

import { useState, useCallback, useRef } from "react"
import { useAppState } from "@/lib/app-state"
import { streamAiChat, type ChatMessage } from "@/lib/api-client"
import type { CorpusFile } from "@/lib/app-state"

// ─── Message types ────────────────────────────────────────────────────────────

export type MessageRole = "user" | "assistant" | "system"

export interface VersionCard {
  label: string           // e.g. "Corpus search: fugue in G minor"
  type: "search" | "analysis" | "generation"
  results?: CorpusFile[]  // for search results
  scoreId?: string        // for analysis
}

export interface ChatMsg {
  id:          string
  role:        MessageRole
  content:     string
  isStreaming?: boolean
  versionCard?: VersionCard
  timestamp:   Date
}

// ─── Hook ────────────────────────────────────────────────────────────────────

export interface UseAiChatReturn {
  messages:   ChatMsg[]
  isThinking: boolean
  send:       (text: string) => Promise<void>
  clear:      () => void
}

export function useAiChat(): UseAiChatReturn {
  const { activeNav, setActiveNav, activeScore } = useAppState()
  const [messages,   setMessages]   = useState<ChatMsg[]>([
    {
      id:        "welcome",
      role:      "assistant",
      content:   "Hello! I'm your Bach Propagation Copilot. I can help you explore the corpus, analyse harmonic structure, or navigate the workbench.\n\nTry asking: *\"Find a Bach fugue in D minor\"* or *\"Analyse the harmonic structure of the loaded piece.\"*",
      timestamp: new Date(),
    },
  ])
  const [isThinking, setIsThinking] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  // ── Helpers ────────────────────────────────────────────────────────────────

  const uid = () => Math.random().toString(36).slice(2, 9)

  const appendMsg = (msg: ChatMsg) =>
    setMessages((prev) => [...prev, msg])

  const updateLast = (updater: (msg: ChatMsg) => ChatMsg) =>
    setMessages((prev) => {
      if (!prev.length) return prev
      return [...prev.slice(0, -1), updater(prev[prev.length - 1])]
    })

  // ── Context builder ────────────────────────────────────────────────────────

  const buildContext = useCallback(() => ({
    active_nav:        activeNav,
    active_score_id:   activeScore?.id,
    active_score_name: activeScore?.movement_name
      ?? (activeScore?.bwv ? `BWV ${activeScore.bwv}` : undefined),
  }), [activeNav, activeScore])

  // ── Send ───────────────────────────────────────────────────────────────────

  const send = useCallback(async (text: string) => {
    if (!text.trim() || isThinking) return

    // Abort any in-flight request
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    // Add user message
    const userMsg: ChatMsg = {
      id: uid(), role: "user", content: text.trim(), timestamp: new Date(),
    }
    appendMsg(userMsg)
    setIsThinking(true)

    // Placeholder for streaming assistant response
    const assistantId = uid()
    const assistantPlaceholder: ChatMsg = {
      id: assistantId, role: "assistant",
      content: "", isStreaming: true, timestamp: new Date(),
    }
    appendMsg(assistantPlaceholder)

    // Build history for the API (exclude system welcome, exclude placeholder)
    const history: ChatMessage[] = messages
      .filter((m) => m.role !== "system" && m.id !== "welcome")
      .map((m) => ({ role: m.role === "user" ? "user" : "assistant", content: m.content }))
    history.push({ role: "user", content: text.trim() })

    try {
      let accumulated = ""
      let versionCard: VersionCard | undefined

      for await (const event of streamAiChat(history, buildContext())) {
        if (event.type === "text_delta") {
          accumulated += event.text
          updateLast((m) => ({ ...m, content: accumulated, isStreaming: true }))
        }

        else if (event.type === "tool_use") {

          // ── searchCorpus ─────────────────────────────────────────────────
          if (event.tool === "searchCorpus") {
            const results = event.result as CorpusFile[]
            versionCard = {
              label:   `Corpus search · ${results.length} result${results.length !== 1 ? "s" : ""}`,
              type:    "search",
              results,
            }
            // Navigate to Library so user sees the results in context
            setActiveNav("Library")

            // Append a system-style message listing top results
            const preview = results.slice(0, 5).map((f) => {
              const name = f.movement_name?.trim() || `BWV ${f.bwv}` || "Untitled"
              return `• **${name}** — ${f.collection?.replace(/_/g, " ")} · ${f.key_signature ?? "?"}`
            }).join("\n")

            const suffix = results.length > 5
              ? `\n\n_Showing 5 of ${results.length} results. Switch to Library to see all._`
              : ""

            accumulated += accumulated
              ? `\n\n${preview}${suffix}`
              : `${preview}${suffix}`

            updateLast((m) => ({ ...m, content: accumulated, versionCard, isStreaming: true }))
          }

          // ── fetchAnalysis ────────────────────────────────────────────────
          else if (event.tool === "fetchAnalysis") {
            versionCard = {
              label:   "Navigated to Analysis",
              type:    "analysis",
              scoreId: activeScore?.id,
            }
            setActiveNav("Analysis")

            const note = activeScore
              ? `\n\n_Switched to Analysis view for ${activeScore.movement_name ?? `BWV ${activeScore.bwv}`}._`
              : "\n\n_Switched to Analysis view. Load a piece from the Library to see its harmonic analysis._"

            accumulated += note
            updateLast((m) => ({ ...m, content: accumulated, versionCard, isStreaming: true }))
          }
        }

        else if (event.type === "done") {
          updateLast((m) => ({ ...m, isStreaming: false, versionCard }))
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return
      updateLast((m) => ({
        ...m,
        content: m.content || "Sorry, I encountered an error. Please try again.",
        isStreaming: false,
      }))
    } finally {
      setIsThinking(false)
    }
  }, [messages, isThinking, buildContext, setActiveNav, activeScore])

  // ── Clear ──────────────────────────────────────────────────────────────────

  const clear = useCallback(() => {
    abortRef.current?.abort()
    setMessages([{
      id:        "welcome",
      role:      "assistant",
      content:   "Chat cleared. What would you like to explore?",
      timestamp: new Date(),
    }])
    setIsThinking(false)
  }, [])

  return { messages, isThinking, send, clear }
}