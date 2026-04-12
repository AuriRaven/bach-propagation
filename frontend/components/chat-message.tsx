/**
 * frontend/components/chat-message.tsx
 *
 * Renders a single chat message bubble.
 *
 * Supports:
 *   - User / assistant roles with distinct styling
 *   - Minimal markdown: **bold**, *italic*, bullet lists, _italic_
 *   - Streaming cursor animation
 *   - Version card scaffold (Play / Revert disabled until model exists)
 */

"use client"

import { Music2, Search, BarChart3, Clock, Play, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { ChatMsg, VersionCard } from "@/hooks/use-ai-chat"

// ─── Markdown-lite renderer ───────────────────────────────────────────────────

function renderContent(text: string): React.ReactNode[] {
  const lines = text.split("\n")
  const nodes: React.ReactNode[] = []

  lines.forEach((line, i) => {
    if (line.startsWith("• ") || line.startsWith("- ")) {
      nodes.push(
        <li key={i} className="ml-3 list-disc list-inside text-sm leading-relaxed">
          {renderInline(line.slice(2))}
        </li>
      )
    } else if (line.trim() === "") {
      nodes.push(<div key={i} className="h-2" />)
    } else {
      nodes.push(
        <p key={i} className="text-sm leading-relaxed">
          {renderInline(line)}
        </p>
      )
    }
  })

  return nodes
}

function renderInline(text: string): React.ReactNode {
  // Bold **text**, italic *text* and _text_
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|_[^_]+_)/)
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i} className="font-semibold text-foreground">{part.slice(2, -2)}</strong>
    }
    if ((part.startsWith("*") && part.endsWith("*")) ||
        (part.startsWith("_") && part.endsWith("_"))) {
      return <em key={i} className="italic text-muted-foreground">{part.slice(1, -1)}</em>
    }
    return part
  })
}

// ─── Version card ─────────────────────────────────────────────────────────────

const VERSION_ICONS = {
  search:     Search,
  analysis:   BarChart3,
  generation: Music2,
}

function VersionCardWidget({ card }: { card: VersionCard }) {
  const Icon = VERSION_ICONS[card.type]

  return (
    <div className="mt-3 rounded-lg border border-primary/20 bg-primary/5 p-3">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-3.5 h-3.5 text-primary" />
        <span className="text-xs font-medium text-primary uppercase tracking-wider">
          {card.label}
        </span>
      </div>

      {/* Generation actions — scaffold, disabled until model exists */}
      {card.type === "generation" && (
        <div className="flex gap-2 mt-2">
          <Button variant="outline" size="sm"
            className="h-7 text-xs border-primary/30 text-primary hover:bg-primary/10"
            disabled title="Available after model training">
            <Play className="w-3 h-3 mr-1" />
            Play version
          </Button>
          <Button variant="outline" size="sm"
            className="h-7 text-xs border-muted-foreground/30 text-muted-foreground"
            disabled title="Available after model training">
            <RotateCcw className="w-3 h-3 mr-1" />
            Revert
          </Button>
        </div>
      )}

      {/* Search result count */}
      {card.type === "search" && card.results && (
        <p className="text-xs text-muted-foreground">
          {card.results.length} piece{card.results.length !== 1 ? "s" : ""} found in Library
        </p>
      )}

      {/* Analysis navigation */}
      {card.type === "analysis" && (
        <p className="text-xs text-muted-foreground">
          Switched to Analysis view
        </p>
      )}
    </div>
  )
}

// ─── Streaming cursor ─────────────────────────────────────────────────────────

function StreamingCursor() {
  return (
    <span className="inline-block w-1.5 h-3.5 bg-primary ml-0.5 rounded-sm animate-pulse" />
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export function ChatMessage({ msg }: { msg: ChatMsg }) {
  const isUser = msg.role === "user"

  if (isUser) {
    return (
      <div className="flex justify-end mb-3">
        <div className="max-w-[85%] bg-primary/15 border border-primary/20 rounded-2xl rounded-tr-sm px-4 py-2.5">
          <p className="text-sm text-foreground leading-relaxed">{msg.content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start mb-3">
      <div className="max-w-[92%] space-y-1">
        {/* Bubble */}
        <div className="bg-card border border-border/60 rounded-2xl rounded-tl-sm px-4 py-3">
          <div className="space-y-1">
            {renderContent(msg.content)}
            {msg.isStreaming && <StreamingCursor />}
          </div>

          {/* Version card */}
          {msg.versionCard && !msg.isStreaming && (
            <VersionCardWidget card={msg.versionCard} />
          )}
        </div>

        {/* Timestamp — suppressHydrationWarning prevents server/client timezone mismatch */}
        <div className="flex items-center gap-1 px-1">
          <Clock className="w-2.5 h-2.5 text-muted-foreground/40" />
          <span className="text-[10px] text-muted-foreground/40" suppressHydrationWarning>
            {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
        </div>
      </div>
    </div>
  )
}