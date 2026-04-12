"use client"

/**
 * frontend/components/views/analysis-view.tsx
 *
 * Score-only harmonic analysis.
 * Corpus-level analytics have moved to library-view.tsx.
 *
 * Shows:
 *   - Empty state when no piece is loaded
 *   - "Analyse" button (manual trigger — avoids surprise 60s wait)
 *   - Radar chart (roman numeral balance)
 *   - Bar chart (chord quality frequency)
 *   - Horizontal bar chart (harmonic function breakdown)
 *   - Pattern cards (secondary dominants, borrowed chords, etc.)
 */

import { useState, useCallback } from "react"
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
} from "recharts"
import {
  Music, Repeat, TrendingUp, Waves,
  Loader2, RefreshCw, BookOpen, Play,
} from "lucide-react"
import { Button } from "@/components/ui/button"

import { api, type ScoreHarmonicAnalysis } from "@/lib/api-client"
import { useAppState } from "@/lib/app-state"

// ─── Palette ──────────────────────────────────────────────────────────────────

const COLORS = [
  "oklch(0.75 0.15 290)", "oklch(0.70 0.12 310)", "oklch(0.78 0.12 280)",
  "oklch(0.65 0.18 300)", "oklch(0.80 0.10 270)", "oklch(0.72 0.14 295)",
  "oklch(0.68 0.16 285)", "oklch(0.76 0.13 305)",
]

const GLOW = "0 0 20px rgba(168,130,255,0.1), 0 0 40px rgba(168,130,255,0.05)"

const TOOLTIP_STYLE = {
  backgroundColor: "oklch(0.16 0.025 280)",
  border: "1px solid oklch(0.28 0.04 280)",
  borderRadius: "8px",
  color: "oklch(0.95 0.02 300)",
}

const NOTE_NAMES = ["C","C♯","D","D♯","E","F","F♯","G","G♯","A","A♯","B"]

function tonicName(n: number) { return NOTE_NAMES[n % 12] ?? "?" }

// ─── Sub-components ───────────────────────────────────────────────────────────

function Card({ title, subtitle, children }: {
  title: string; subtitle?: string; children: React.ReactNode
}) {
  return (
    <div className="bg-card rounded-xl p-6 border border-border" style={{ boxShadow: GLOW }}>
      <h3 className="font-serif text-lg font-semibold mb-1">{title}</h3>
      {subtitle && <p className="text-xs text-muted-foreground mb-5">{subtitle}</p>}
      {children}
    </div>
  )
}

// ─── Empty state — no piece loaded ───────────────────────────────────────────

function NoScoreState() {
  return (
    <div className="flex flex-col items-center justify-center h-96 gap-4 text-muted-foreground">
      <BookOpen className="w-12 h-12 opacity-20" />
      <div className="text-center">
        <p className="font-serif text-lg mb-1">No piece loaded</p>
        <p className="text-sm max-w-xs leading-relaxed">
          Go to the <strong>Library</strong>, select a piece, and click{" "}
          <strong>Load into Workbench</strong>. Then return here to analyse it.
        </p>
      </div>
    </div>
  )
}

// ─── Ready state — piece loaded, analysis not yet run ─────────────────────────

function ReadyState({ title, onAnalyse }: { title: string; onAnalyse: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-96 gap-5 text-muted-foreground">
      <Music className="w-12 h-12 opacity-30 text-primary" />
      <div className="text-center">
        <p className="font-serif text-lg text-foreground mb-1">{title}</p>
        <p className="text-sm max-w-xs leading-relaxed mb-6">
          Run the harmonic analysis engine on this piece. First run takes 30–90 seconds;
          subsequent runs are instant from cache.
        </p>
        <Button onClick={onAnalyse}
          className="bg-primary hover:bg-primary/90 text-primary-foreground gap-2">
          <Play className="w-4 h-4" />
          Analyse this piece
        </Button>
      </div>
    </div>
  )
}

// ─── Loading state ────────────────────────────────────────────────────────────

function AnalysingState({ title }: { title: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-96 gap-4 text-muted-foreground">
      <Loader2 className="w-10 h-10 animate-spin text-primary" />
      <div className="text-center">
        <p className="font-serif text-lg text-foreground mb-1">Analysing {title}</p>
        <p className="text-sm">Running key estimation, chord classification and roman numeral analysis…</p>
        <p className="text-xs mt-2 opacity-60">First run: 30–90s · Subsequent runs: &lt;1s from cache</p>
      </div>
    </div>
  )
}

// ─── Analysis results ─────────────────────────────────────────────────────────

function AnalysisResults({ data, title, onReanalyse, reanalysing }: {
  data: ScoreHarmonicAnalysis
  title: string
  onReanalyse: () => void
  reanalysing: boolean
}) {
  const keyName    = `${tonicName(data.global_key_tonic)} ${data.global_key_mode}`
  const confidence = Math.round(data.global_key_confidence * 100)

  // Radar — top 7 roman numerals, normalised to 100
  const topNumerals = data.roman_numerals.slice(0, 7)
  const maxPct      = Math.max(...topNumerals.map((r) => r.percentage), 1)
  const radarData   = topNumerals.map((rn) => ({
    harmony:  rn.numeral,
    value:    Math.round((rn.percentage / maxPct) * 100),
    fullMark: 100,
  }))

  // Bar — chord quality
  const qualityBars = data.chord_qualities.slice(0, 8).map((q) => ({
    name: q.quality
      .replace("dominant7", "dom7").replace("major7", "M7")
      .replace("minor7", "m7").replace("diminished", "dim")
      .replace("augmented", "aug"),
    frequency: q.count,
  }))

  // Horizontal bars — harmonic function
  const funcBars = data.harmonic_functions.map((f) => ({
    name: f.function, value: f.percentage,
  }))

  // Pattern cards
  const patterns = [
    { id: 1, name: "Secondary Dominants",
      occurrences: data.secondary_dominant_count,
      detail: "chromatic tonicisations", icon: TrendingUp, color: "text-primary" },
    { id: 2, name: "Borrowed Chords",
      occurrences: data.borrowed_chord_count,
      detail: "modal mixture events", icon: Repeat, color: "text-accent" },
    { id: 3, name: "Total Chord Events",
      occurrences: data.total_chords,
      detail: "detected chord events", icon: Music, color: "text-primary" },
    { id: 4, name: "Dominant Function",
      occurrences: data.harmonic_functions.find((f) => f.function === "DOMINANT")?.count ?? 0,
      detail: "dominant-function chords", icon: Waves, color: "text-accent" },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-serif text-xl font-semibold">{title}</h3>
          <p className="text-sm text-muted-foreground mt-0.5">
            Global key:{" "}
            <span className="text-primary font-medium">{keyName}</span>
            <span className="ml-2 text-xs opacity-60">({confidence}% confidence)</span>
            <span className="ml-3 text-xs opacity-50">· {data.total_chords} chord events</span>
          </p>
        </div>
        <Button variant="outline" size="sm" disabled={reanalysing}
          className="border-primary/50 text-primary hover:bg-primary/10 shrink-0"
          onClick={onReanalyse}>
          <RefreshCw className={`w-3 h-3 mr-1.5 ${reanalysing ? "animate-spin" : ""}`} />
          {reanalysing ? "Re-analysing…" : "Re-analyse"}
        </Button>
      </div>

      {/* Harmonic distribution */}
      <Card
        title="Harmonic Distribution"
        subtitle={`Roman numeral and chord quality analysis · ${data.total_chords} chord events`}
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Radar */}
          <div className="bg-background/50 rounded-lg p-4 border border-border/50">
            <h4 className="text-xs font-medium text-muted-foreground mb-4 uppercase tracking-wider">
              Harmonic Balance
            </h4>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid stroke="oklch(0.28 0.04 280)" strokeDasharray="3 3" />
                  <PolarAngleAxis dataKey="harmony"
                    tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 12 }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]}
                    tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 9 }} axisLine={false} />
                  <Radar name="%" dataKey="value"
                    stroke="oklch(0.75 0.15 290)" fill="oklch(0.75 0.15 290)"
                    fillOpacity={0.3} strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Chord quality bar */}
          <div className="bg-background/50 rounded-lg p-4 border border-border/50">
            <h4 className="text-xs font-medium text-muted-foreground mb-4 uppercase tracking-wider">
              Chord Quality Frequency
            </h4>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={qualityBars} barCategoryGap="20%">
                  <XAxis dataKey="name"
                    tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 11 }}
                    axisLine={{ stroke: "oklch(0.28 0.04 280)" }} tickLine={false} />
                  <YAxis tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                    axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={TOOLTIP_STYLE}
                    cursor={{ fill: "oklch(0.25 0.05 290)", opacity: 0.3 }} />
                  <Bar dataKey="frequency" radius={[4,4,0,0]}>
                    {qualityBars.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </Card>

      {/* Harmonic function breakdown */}
      {funcBars.length > 0 && (
        <Card title="Harmonic Function Breakdown"
          subtitle="Tonic / Dominant / Predominant distribution">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={funcBars} layout="vertical" barCategoryGap="30%">
                <XAxis type="number"
                  tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                  axisLine={false} tickLine={false}
                  tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
                <YAxis type="category" dataKey="name" width={110}
                  tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 11 }}
                  axisLine={false} tickLine={false} />
                <Tooltip formatter={(v: number) => [`${v}%`, "Share"]}
                  contentStyle={TOOLTIP_STYLE}
                  cursor={{ fill: "oklch(0.25 0.05 290)", opacity: 0.3 }} />
                <Bar dataKey="value" radius={[0,4,4,0]}>
                  {funcBars.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* Pattern recognition cards */}
      <Card title="Pattern Recognition"
        subtitle="Harmonic events detected by the analysis engine">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {patterns.map((p) => (
            <div key={p.id}
              className="bg-background/50 rounded-lg p-4 border border-border/50 hover:border-primary/50 transition-colors">
              <div className="flex items-start gap-3">
                <div className={`p-2 rounded-lg bg-primary/10 ${p.color}`}>
                  <p.icon className="w-5 h-5" />
                </div>
                <div className="flex-1">
                  <h4 className="font-medium text-foreground">{p.name}</h4>
                  <p className="text-xs text-muted-foreground mt-1">{p.detail}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <span className="text-xs px-2 py-0.5 rounded-full bg-primary/20 text-primary font-medium">
                      {p.occurrences} events
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

// ─── Main export ──────────────────────────────────────────────────────────────

export function AnalysisView() {
  const { activeScore } = useAppState()
  const [data,        setData]        = useState<ScoreHarmonicAnalysis | null>(null)
  const [status,      setStatus]      = useState<"idle" | "loading" | "done" | "error">("idle")
  const [reanalysing, setReanalysing] = useState(false)
  const [errorMsg,    setErrorMsg]    = useState<string | null>(null)

  // Reset when score changes
  const scoreId = activeScore?.id
  const prevId  = useState(scoreId)[0]
  if (scoreId !== prevId && status !== "idle") {
    setData(null)
    setStatus("idle")
  }

  const runAnalysis = useCallback(async (refresh = false) => {
    if (!activeScore) return
    refresh ? setReanalysing(true) : setStatus("loading")
    setErrorMsg(null)
    try {
      const result = await api.analysis.score(activeScore.id, refresh)
      setData(result)
      setStatus("done")
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : "Analysis failed")
      setStatus("error")
    } finally {
      setReanalysing(false)
    }
  }, [activeScore])

  const title = activeScore
    ? (activeScore.movement_name?.trim() || `BWV ${activeScore.bwv}` || "Loaded piece")
    : ""

  return (
    <div className="space-y-4 h-full">
      {/* View header */}
      <div className="flex items-center justify-between mb-2">
        <div>
          <h2 className="font-serif text-xl font-semibold">Score Analysis</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Harmonic analysis powered by the Bach Propagation analysis engine
          </p>
        </div>
        {activeScore && status === "done" && (
          <span className="text-xs px-2.5 py-1 rounded-full border border-primary/40 text-primary font-mono">
            {activeScore.bwv ? `BWV ${activeScore.bwv}` : "Loaded"}
          </span>
        )}
      </div>

      {/* Content states */}
      {!activeScore && <NoScoreState />}

      {activeScore && status === "idle" && (
        <ReadyState title={title} onAnalyse={() => void runAnalysis(false)} />
      )}

      {activeScore && status === "loading" && <AnalysingState title={title} />}

      {activeScore && status === "error" && (
        <div className="flex flex-col items-center justify-center h-64 gap-3">
          <p className="text-sm text-destructive">{errorMsg}</p>
          <Button variant="outline" size="sm" onClick={() => void runAnalysis(true)}>
            Retry
          </Button>
        </div>
      )}

      {activeScore && status === "done" && data && (
        <AnalysisResults
          data={data}
          title={title}
          onReanalyse={() => void runAnalysis(true)}
          reanalysing={reanalysing}
        />
      )}

      <div className="flex justify-between text-sm text-muted-foreground font-serif italic px-2">
        <span>Bach Propagation Engine v2.0</span>
        <span>Harmonic Analysis Report</span>
      </div>
    </div>
  )
}