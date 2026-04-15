"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import {
  Play, Pause, Square, Undo2, Redo2,
  Library, FilePlus, BarChart3, Settings,
  RefreshCw, Sparkles, Loader2,
  MessageSquare, SlidersHorizontal,
  Send, Trash2,
} from "lucide-react"
import { Slider }  from "@/components/ui/slider"
import { Button }  from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"

import { CompositionView } from "@/components/views/composition-view"
import { AnalysisView }    from "@/components/views/analysis-view"
import { LibraryView }     from "@/components/views/library-view"
import { SettingsView }    from "@/components/views/settings-view"
import { ChatMessage }     from "@/components/chat-message"

import { useAppState, type VexFlowPayload, type CorpusFile } from "@/lib/app-state"
import { useMidiPlayer } from "@/hooks/use-midi-player"
import { useAiChat, type UseAiChatReturn } from "@/hooks/use-ai-chat"
import { useGeneration } from "@/hooks/use-generation"
import { api, type GenerationResponse } from "@/lib/api-client"

type ViewType    = "Library" | "New Composition" | "Analysis" | "Settings"
type PanelTab    = "copilot" | "controls"

// ─── Slider → API mappers ─────────────────────────────────────────────────────

/** Complexity slider (0–100) → n_tokens (16–512) */
function complexityToTokens(v: number): number {
  return Math.round(16 + (v / 100) * (512 - 16))
}

/** Ornamentation slider (0–100) → temperature (0.1–2.0) */
function ornamentationToTemp(v: number): number {
  return Math.round((0.1 + (v / 100) * (2.0 - 0.1)) * 100) / 100
}

/** Counterpoint slider (0–100) → top_k (1–50) */
function counterpointToTopK(v: number): number {
  return Math.round(1 + (v / 100) * (50 - 1))
}

/** Extract "major" | "minor" from select value like "d-minor" */
function keySelectToMode(v: string): "major" | "minor" {
  return v.endsWith("-major") ? "major" : "minor"
}

// ─── Loading overlay ──────────────────────────────────────────────────────────

function WorkbenchLoadingOverlay() {
  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center">
      <div className="flex flex-col items-center gap-3">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <p className="text-sm text-muted-foreground font-mono tracking-widest uppercase">
          Loading Score…
        </p>
      </div>
    </div>
  )
}

// ─── Progress scrubber ────────────────────────────────────────────────────────

function ProgressScrubber({
  position, duration, onSeek, disabled,
}: {
  position: number; duration: number
  onSeek: (beat: number) => void; disabled: boolean
}) {
  const pct = duration > 0 ? Math.min((position / duration) * 100, 100) : 0
  const fmt = (s: number) => {
    const n = Math.round(s)
    return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, "0")}`
  }

  return (
    <div className="flex items-center gap-2 flex-1 max-w-xs">
      <span className="text-xs text-muted-foreground font-mono w-8 text-right tabular-nums">
        {fmt(position * 0.5)}
      </span>
      <div className="flex-1 relative h-1.5 bg-muted rounded-full overflow-hidden">
        <div className="absolute left-0 top-0 h-full bg-primary rounded-full transition-none"
          style={{ width: `${pct}%` }} />
        <input type="range" min={0} max={Math.max(duration, 1)} step={0.25}
          value={position} disabled={disabled}
          onChange={(e) => onSeek(parseFloat(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-default" />
      </div>
      <span className="text-xs text-muted-foreground font-mono w-8 tabular-nums">
        {fmt(duration * 0.5)}
      </span>
    </div>
  )
}

// ─── Copilot panel ────────────────────────────────────────────────────────────

function CopilotPanel({ chat, onPlayVersion, onRevert }: {
  chat: UseAiChatReturn
  onPlayVersion?: () => void
  onRevert?: () => void
}) {
  const { messages, isThinking, send, clear } = chat
  const [input,   setInput]   = useState("")
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || isThinking) return
    setInput("")
    await send(text)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      void handleSend()
    }
  }

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto py-2 pr-1 space-y-1">
        {messages.map((msg) => (
          <ChatMessage key={msg.id} msg={msg}
            onPlayVersion={onPlayVersion}
            onRevert={onRevert}
          />
        ))}

        {/* Thinking indicator */}
        {isThinking && (
          <div className="flex justify-start mb-3">
            <div className="bg-card border border-border/60 rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex items-center gap-1.5">
                {[0, 150, 300].map((delay) => (
                  <span key={delay}
                    className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce"
                    style={{ animationDelay: `${delay}ms` }} />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="pt-3 border-t border-border shrink-0">
        <div className="relative">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the corpus, request analysis, or search for a piece…"
            className="resize-none pr-10 text-sm bg-card border-border min-h-[72px] max-h-32"
            disabled={isThinking}
          />
          <Button
            size="icon"
            variant="ghost"
            className="absolute bottom-2 right-2 h-7 w-7 text-primary hover:bg-primary/10"
            onClick={() => void handleSend()}
            disabled={isThinking || !input.trim()}
          >
            <Send className="w-3.5 h-3.5" />
          </Button>
        </div>
        <div className="flex items-center justify-between mt-2">
          <p className="text-[10px] text-muted-foreground/50">
            Enter to send · Shift+Enter for new line
          </p>
          <Button variant="ghost" size="sm"
            className="h-6 text-[10px] text-muted-foreground/50 hover:text-muted-foreground px-1"
            onClick={clear}>
            <Trash2 className="w-2.5 h-2.5 mr-1" />
            Clear
          </Button>
        </div>
      </div>
    </div>
  )
}

// ─── Controls panel ───────────────────────────────────────────────────────────

interface ControlsPanelProps {
  activeScore:    ReturnType<typeof useAppState>["activeScore"]
  // Controlled slider/select state
  keySelect:      string;       onKeyChange:          (v: string) => void
  complexity:     number[];     onComplexityChange:   (v: number[]) => void
  counterpoint:   number[];     onCounterpointChange: (v: number[]) => void
  ornamentation:  number[];     onOrnamentationChange:(v: number[]) => void
  // Generation
  isGenerating:   boolean
  lastResult:     GenerationResponse | null
  genError:       string | null
  onRegenerate:   () => void
}

function ControlsPanel({
  activeScore,
  keySelect, onKeyChange,
  complexity, onComplexityChange,
  counterpoint, onCounterpointChange,
  ornamentation, onOrnamentationChange,
  isGenerating, lastResult, genError,
  onRegenerate,
}: ControlsPanelProps) {
  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-y-auto">
      {/* Key & Time */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="text-xs text-muted-foreground uppercase tracking-wider mb-2 block">Key</label>
          <Select value={keySelect} onValueChange={onKeyChange}>
            <SelectTrigger className="bg-card border-border"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="c-major">C Major</SelectItem>
              <SelectItem value="d-minor">D Minor</SelectItem>
              <SelectItem value="g-major">G Major</SelectItem>
              <SelectItem value="a-minor">A Minor</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label className="text-xs text-muted-foreground uppercase tracking-wider mb-2 block">Time</label>
          <Select defaultValue="3-8">
            <SelectTrigger className="bg-card border-border"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="4-4">4/4</SelectItem>
              <SelectItem value="3-4">3/4</SelectItem>
              <SelectItem value="3-8">3/8</SelectItem>
              <SelectItem value="6-8">6/8</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Sliders */}
      <div className="space-y-6 mb-6">
        {([
          { label: "Complexity",           value: complexity,    set: onComplexityChange,    hint: `${complexityToTokens(complexity[0])} tokens` },
          { label: "Counterpoint Density", value: counterpoint,  set: onCounterpointChange,  hint: `top-k ${counterpointToTopK(counterpoint[0])}` },
          { label: "Ornamentation",        value: ornamentation, set: onOrnamentationChange, hint: `temp ${ornamentationToTemp(ornamentation[0])}` },
        ] as const).map(({ label, value, set, hint }) => (
          <div key={label}>
            <div className="flex justify-between items-center mb-3">
              <label className="text-sm font-medium">{label}</label>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-muted-foreground font-mono">{hint}</span>
                <span className="text-sm text-primary font-semibold">{value[0]}%</span>
              </div>
            </div>
            <Slider value={value as number[]} onValueChange={set} max={100} step={1}
              className="[&_[role=slider]]:bg-primary [&_[role=slider]]:border-primary" />
          </div>
        ))}
      </div>

      {/* Generation result card */}
      {lastResult && (
        <div className="bg-card rounded-lg p-4 border border-accent/40 mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-4 h-4 text-accent" />
            <span className="text-sm font-semibold text-accent uppercase tracking-wider">
              Generation Result
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-muted-foreground">Tonal score</span>
              <p className="font-semibold text-foreground">{lastResult.tonal_score.toFixed(2)}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Grammar</span>
              <p className={`font-semibold ${lastResult.is_valid ? "text-green-400" : "text-amber-400"}`}>
                {lastResult.is_valid ? "Valid" : "Has violations"}
              </p>
            </div>
            <div>
              <span className="text-muted-foreground">Forbidden rate</span>
              <p className="font-semibold text-foreground">{lastResult.forbidden_rate.toFixed(4)}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Time</span>
              <p className="font-semibold text-foreground">{(lastResult.generation_time_ms / 1000).toFixed(1)}s</p>
            </div>
          </div>
        </div>
      )}

      {/* Error card */}
      {genError && (
        <div className="bg-destructive/10 rounded-lg p-4 border border-destructive/30 mb-6">
          <p className="text-sm text-destructive">{genError}</p>
        </div>
      )}

      {/* Propagator Insight */}
      <div className="bg-card rounded-lg p-4 border border-border mb-6">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="w-4 h-4 text-accent" />
          <span className="text-sm font-semibold text-accent uppercase tracking-wider">
            Propagator Insight
          </span>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed mb-4">
          {activeScore
            ? `Analysing ${activeScore.key_signature ?? "key"} — ${activeScore.num_measures ?? "?"} measures loaded.`
            : "Load a piece from the Library to see contextual insights."
          }
        </p>
        <Button variant="outline" className="w-full border-primary/50 text-primary hover:bg-primary/10">
          Apply Suggestion
        </Button>
      </div>

      <div className="flex-1" />

      {/* Regenerate */}
      <Button
        className="w-full bg-primary hover:bg-primary/90 text-primary-foreground py-6 text-base font-semibold"
        onClick={onRegenerate}
        disabled={isGenerating}
      >
        {isGenerating ? (
          <>
            <Loader2 className="w-5 h-5 mr-2 animate-spin" />
            Generating…
          </>
        ) : (
          <>
            <RefreshCw className="w-5 h-5 mr-2" />
            Regenerate Phrases
          </>
        )}
      </Button>
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function BachWorkbench() {
  const [panelTab, setPanelTab] = useState<PanelTab>("copilot")

  // ── Lifted control state ────────────────────────────────────────────────────
  const [keySelect,     setKeySelect]     = useState("d-minor")
  const [complexity,    setComplexity]    = useState([75])
  const [counterpoint,  setCounterpoint]  = useState([60])
  const [ornamentation, setOrnamentation] = useState([42])

  const {
    activeNav, setActiveNav,
    activeScore, setActiveScore,
    notationData,
    playbackState, playbackPosition,
    isLoadingWorkbench,
    setNotationData,
  } = useAppState()

  const { play, pause, stop, seekTo, loadFromNotation, durationBeats } = useMidiPlayer()
  const chat = useAiChat()
  const { isThinking } = chat

  // ── Revert state — saved before generation loads new notation ──────────────
  const prevNotationRef   = useRef<VexFlowPayload | null>(null)
  const prevActiveScoreRef = useRef<CorpusFile | null>(null)
  const hasGeneratedRef    = useRef(false)

  // ── Generation hook with Copilot message on complete ────────────────────────
  // We keep keySelect in a ref so the onComplete callback always sees the
  // current value without needing to recreate useGeneration.
  const keySelectRef = useRef(keySelect)
  keySelectRef.current = keySelect

  const gen = useGeneration({
    onComplete: (result: GenerationResponse) => {
      // Switch to Copilot tab so the user sees the result message
      setPanelTab("copilot")

      const mode      = keySelectToMode(keySelectRef.current)
      const topChords = result.rn_sequence.slice(0, 5).join(" → ")
      const grammar   = result.is_valid ? "Valid" : "Has violations"
      const content =
        `**Generation complete** — ${result.chord_tokens.length} chord events in ${mode} mode.\n\n` +
        `**Tonal score:** ${result.tonal_score.toFixed(2)} · **Grammar:** ${grammar}\n` +
        `**Forbidden rate:** ${result.forbidden_rate.toFixed(4)} · **Time:** ${(result.generation_time_ms / 1000).toFixed(1)}s\n\n` +
        `**Top chords:** ${topChords}`

      // Inject directly as an assistant message — no AI round-trip
      chat.appendMessage(content, "assistant", {
        label: `Generated ${result.chord_tokens.length} chords`,
        type:  "generation",
      })

      // Parse MusicXML → VexFlowPayload and load into piano roll
      if (result.musicxml_b64) {
        // Save current state for Revert — only on first generation
        if (!hasGeneratedRef.current) {
          prevNotationRef.current    = notationData ?? null
          prevActiveScoreRef.current = activeScore ?? null
        }

        api.notation.parse(result.musicxml_b64)
          .then((notation) => {
            hasGeneratedRef.current = true
            setNotationData(notation)
            setActiveNav("New Composition")
            chat.appendMessage(
              `_Score loaded into the piano roll — ${notation.measures.length} measures, ${notation.key_signature}, ${notation.time_signature}._`,
              "assistant",
            )
          })
          .catch((err) => {
            console.warn("[BachWorkbench] Notation parse failed:", err)
            chat.appendMessage(
              "_Could not parse the generated score for the piano roll. The generation data is still available in the Controls panel._",
              "assistant",
            )
          })
      }
    },
  })

  const handleRegenerate = useCallback(() => {
    void gen.generate({
      key_mode:    keySelectToMode(keySelect),
      n_tokens:    complexityToTokens(complexity[0]),
      temperature: ornamentationToTemp(ornamentation[0]),
      top_k:       counterpointToTopK(counterpoint[0]),
      prompt_bwv:  activeScore?.bwv ?? null,
    })
  }, [gen, keySelect, complexity, ornamentation, counterpoint, activeScore])

  // ── Play version — load generated notation into Tone.js and play ──────────
  const handlePlayVersion = useCallback(async () => {
    if (!notationData) return
    await stop()
    await loadFromNotation(notationData, 120)
    await play()
    setActiveNav("New Composition")
  }, [notationData, stop, loadFromNotation, play, setActiveNav])

  // ── Revert — restore the state from before generation ─────────────────────
  const handleRevert = useCallback(async () => {
    await stop()
    setNotationData(prevNotationRef.current)
    setActiveScore(prevActiveScoreRef.current)
    hasGeneratedRef.current = false
    chat.appendMessage("_Reverted to previous score._", "assistant")
  }, [stop, setNotationData, setActiveScore, chat])

  // ── Badge: pulse for both AI chat and generation ────────────────────────────
  const aiActive = isThinking || gen.isGenerating

  const navItems = [
    { name: "Library"         as const, icon: Library   },
    { name: "New Composition" as const, icon: FilePlus   },
    { name: "Analysis"        as const, icon: BarChart3  },
    { name: "Settings"        as const, icon: Settings   },
  ]

  const getHeaderTitle = () => {
    if (activeNav === "New Composition" && activeScore) {
      const name = activeScore.movement_name?.trim()
        || (activeScore.bwv ? `BWV ${activeScore.bwv}` : null)
        || activeScore.collection?.replace(/_/g, " ")
        || "UNTITLED"
      return name.toUpperCase()
    }
    switch (activeNav) {
      case "New Composition": return "NEW COMPOSITION"
      case "Analysis":        return "HARMONIC ANALYSIS"
      case "Library":         return "BAROQUE CORPUS EXPLORER"
      case "Settings":        return "WORKBENCH SETTINGS"
      default:                return ""
    }
  }

  const renderMainContent = () => {
    switch (activeNav) {
      case "New Composition": return <CompositionView />
      case "Analysis":        return <AnalysisView />
      case "Library":         return <LibraryView />
      case "Settings":        return <SettingsView />
      default:                return null
    }
  }

  const handlePlayPause = async () => {
    if (playbackState === "playing") await pause()
    else await play()
  }

  const isInComposition = activeNav === "New Composition"
  const hasScore        = !!activeScore

  // Synth status label
  const synthStatus = playbackState === "playing"
    ? "PLAYING"
    : playbackState === "paused"
    ? "PAUSED"
    : "READY"

  return (
    <div className="flex h-screen bg-background text-foreground">
      {isLoadingWorkbench && <WorkbenchLoadingOverlay />}

      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <aside className="w-64 bg-sidebar border-r border-sidebar-border flex flex-col">
        <div className="p-4 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg flex items-center justify-center overflow-hidden"
            style={{ border: "1px solid rgba(167, 139, 250, 0.2)" }}>
            <img src="/logo-motif-ai.png" alt="Bach Propagation Logo"
              className="w-10 h-10 object-contain"
              style={{ filter: "brightness(1.2) contrast(1.1) drop-shadow(0 0 8px rgba(168, 130, 255, 0.6))" }} />
          </div>
          <div>
            <h1 className="font-serif text-lg font-semibold tracking-tight">Bach Propagation</h1>
            <p className="text-xs text-muted-foreground">Workbench v2.4</p>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4">
          {navItems.map((item) => (
            <button key={item.name} onClick={() => setActiveNav(item.name)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-colors font-serif ${
                activeNav === item.name
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
              }`}>
              <item.icon className="w-5 h-5" />
              <span>{item.name}</span>
            </button>
          ))}
        </nav>

        <div className="p-4">
          <div className="bg-card rounded-lg p-4 border border-border">
            <div className="h-16 mb-3 relative">
              <svg className="w-full h-full" viewBox="0 0 200 60">
                <path d="M0,30 Q20,10 40,30 T80,30 T120,30 T160,30 T200,30"
                  fill="none" stroke="currentColor" strokeWidth="2" className="text-primary/40" />
                <path d="M0,30 Q25,45 50,30 T100,30 T150,30"
                  fill="none" stroke="currentColor" strokeWidth="2" className="text-accent/60" />
              </svg>
            </div>
            <p className="text-xs text-accent uppercase tracking-wider font-semibold text-center">
              Neural Engine
            </p>
            <div className="mt-2 h-1.5 bg-muted rounded-full overflow-hidden flex">
              <div className="h-full bg-primary w-1/3 rounded-full" />
              <div className="h-full bg-accent/60 w-1/4 rounded-full ml-0.5" />
            </div>
          </div>
        </div>
      </aside>

      {/* ── Main Content ─────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Toolbar */}
        <header className="h-14 border-b border-border flex items-center gap-3 px-4 shrink-0">
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon"
              className="text-muted-foreground hover:text-foreground"
              onClick={handlePlayPause}
              disabled={!isInComposition || !hasScore}
              title={playbackState === "playing" ? "Pause" : "Play"}>
              {playbackState === "playing"
                ? <Pause className="w-4 h-4" />
                : <Play  className="w-4 h-4" />}
            </Button>
            <Button variant="ghost" size="icon"
              className="text-muted-foreground hover:text-foreground"
              onClick={() => void stop()}
              disabled={!isInComposition || playbackState === "stopped"}
              title="Stop">
              <Square className="w-4 h-4" />
            </Button>
          </div>

          {isInComposition && hasScore && (
            <ProgressScrubber
              position={playbackPosition}
              duration={durationBeats}
              onSeek={(b) => void seekTo(b)}
              disabled={playbackState === "stopped" && durationBeats === 0}
            />
          )}

          <div className="w-px h-6 bg-border mx-1" />

          <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
            <Undo2 className="w-4 h-4" />
          </Button>
          <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
            <Redo2 className="w-4 h-4" />
          </Button>

          <div className="flex-1" />

          <h2 className="font-mono text-xs tracking-widest text-muted-foreground truncate max-w-xs">
            {getHeaderTitle()}
          </h2>

          {/* AI ASSISTED badge — pulses when AI is thinking or generating */}
          <span className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-semibold shrink-0 transition-colors ${
            aiActive
              ? "border-accent text-accent bg-accent/10"
              : "border-primary text-primary"
          }`}>
            {aiActive && (
              <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
            )}
            {gen.isGenerating
              ? "GENERATING…"
              : isThinking
              ? "AI THINKING…"
              : "AI ASSISTED"}
          </span>
        </header>

        {/* Content */}
        <div className="flex-1 p-6 overflow-auto">
          {renderMainContent()}
        </div>

        {/* Status bar */}
        <footer className="h-8 border-t border-border flex items-center justify-between px-4 text-xs text-muted-foreground shrink-0">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${
                playbackState === "playing"
                  ? "bg-accent animate-pulse"
                  : playbackState === "paused"
                  ? "bg-primary"
                  : "bg-muted-foreground"
              }`} />
              SYNTH:{" "}
              <span className={`font-semibold ${
                playbackState === "playing" ? "text-accent" : "text-muted-foreground"
              }`}>
                {synthStatus}
              </span>
            </span>
            {activeScore?.num_measures && (
              <span>MEASURES: {activeScore.num_measures}</span>
            )}
          </div>
          <div className="flex items-center gap-4">
            <span>LATENCY: 14MS</span>
            <span className="text-primary font-semibold">PROPAGATOR V1.0</span>
          </div>
        </footer>
      </main>

      {/* ── Right Panel ──────────────────────────────────────────────────── */}
      <aside className="w-80 border-l border-border bg-background flex flex-col">

        {/* Panel header + tab switcher */}
        <div className="p-4 pb-0 shrink-0">
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className={`w-5 h-5 ${aiActive ? "text-accent animate-pulse" : "text-primary"}`} />
            <h2 className="font-serif text-lg font-semibold">AI Control Panel</h2>
          </div>

          {/* Segment control */}
          <div className="flex rounded-lg border border-border bg-card p-0.5 mb-4">
            <button
              onClick={() => setPanelTab("copilot")}
              className={`flex-1 flex items-center justify-center gap-1.5 rounded-md py-1.5 text-xs font-medium transition-colors ${
                panelTab === "copilot"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}>
              <MessageSquare className="w-3.5 h-3.5" />
              Copilot
            </button>
            <button
              onClick={() => setPanelTab("controls")}
              className={`flex-1 flex items-center justify-center gap-1.5 rounded-md py-1.5 text-xs font-medium transition-colors ${
                panelTab === "controls"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}>
              <SlidersHorizontal className="w-3.5 h-3.5" />
              Controls
            </button>
          </div>
        </div>

        {/* Panel content */}
        <div className="flex-1 min-h-0 px-4 pb-4 flex flex-col">
          {panelTab === "copilot"
            ? <CopilotPanel chat={chat}
                onPlayVersion={handlePlayVersion}
                onRevert={handleRevert}
              />
            : <ControlsPanel
                activeScore={activeScore}
                keySelect={keySelect}       onKeyChange={setKeySelect}
                complexity={complexity}      onComplexityChange={setComplexity}
                counterpoint={counterpoint}  onCounterpointChange={setCounterpoint}
                ornamentation={ornamentation} onOrnamentationChange={setOrnamentation}
                isGenerating={gen.isGenerating}
                lastResult={gen.lastResult}
                genError={gen.error}
                onRegenerate={handleRegenerate}
              />
          }
        </div>
      </aside>
    </div>
  )
}