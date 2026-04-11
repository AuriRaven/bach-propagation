"use client"

/**
 * frontend/components/views/analysis-view.tsx
 *
 * Two tabs:
 *   Corpus — aggregate analytics from corpus_metadata (always available)
 *   Score  — per-score harmonic analysis using the Python analysis engine
 *            (only available when a score is loaded in the workbench)
 *
 * Preserves the original design: radar chart, bar chart, pattern cards,
 * purple glow aesthetic.
 */

import { useEffect, useState, useCallback } from "react"
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
  PieChart, Pie,
} from "recharts"
import {
  Music, Repeat, TrendingUp, Waves, Loader2, RefreshCw,
  BarChart3, BookOpen, AlertCircle,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

import { api, type CorpusAnalytics, type ScoreHarmonicAnalysis } from "@/lib/api-client"
import { useAppState } from "@/lib/app-state"

// ─── Palette (matches library-view purple glow) ───────────────────────────────

const COLORS = [
  "oklch(0.75 0.15 290)", "oklch(0.70 0.12 310)", "oklch(0.78 0.12 280)",
  "oklch(0.65 0.18 300)", "oklch(0.80 0.10 270)", "oklch(0.72 0.14 295)",
  "oklch(0.68 0.16 285)", "oklch(0.76 0.13 305)",
]

const CARD_SHADOW = "0 0 20px rgba(168,130,255,0.1), 0 0 40px rgba(168,130,255,0.05)"

// ─── Tonic number → note name ─────────────────────────────────────────────────

const NOTE_NAMES = ["C","C♯","D","D♯","E","F","F♯","G","G♯","A","A♯","B"]

function tonicName(n: number): string {
  return NOTE_NAMES[n % 12] ?? "?"
}

// ─── Shared card wrapper ──────────────────────────────────────────────────────

function Card({ title, subtitle, children }: {
  title: string; subtitle?: string; children: React.ReactNode
}) {
  return (
    <div className="bg-card rounded-xl p-6 border border-border"
      style={{ boxShadow: CARD_SHADOW }}>
      <h3 className="font-serif text-lg font-semibold mb-1">{title}</h3>
      {subtitle && <p className="text-xs text-muted-foreground mb-4">{subtitle}</p>}
      {children}
    </div>
  )
}

function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center h-48 gap-3 text-muted-foreground">
      <Loader2 className="w-5 h-5 animate-spin" />
      <span className="text-sm">{label}</span>
    </div>
  )
}

function Empty({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-48 gap-2 text-muted-foreground">
      <AlertCircle className="w-6 h-6 opacity-40" />
      <span className="text-sm italic">{message}</span>
    </div>
  )
}

// ─── Corpus Tab ───────────────────────────────────────────────────────────────

function CorpusTab() {
  const [data, setData]       = useState<CorpusAnalytics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    api.analysis.corpus()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Loading label="Loading corpus analytics…" />
  if (error)   return <Empty message={`Failed to load: ${error}`} />
  if (!data)   return null

  // Donut data from by_collection
  const donutData = data.by_collection.slice(0, 6).map((c, i) => ({
    name: c.collection.replace(/_/g, " "),
    value: c.count,
    color: COLORS[i % COLORS.length],
  }))

  // Scatter proxy — num_measures buckets vs duration buckets
  const durationBars = data.duration_histogram.filter((b) => b.count > 0)
  const measureBars  = data.measures_histogram.filter((b) => b.count > 0)

  // Key mode donut
  const modePie = data.by_key_mode.map((k, i) => ({
    name: k.mode.charAt(0).toUpperCase() + k.mode.slice(1),
    value: k.count,
    color: COLORS[(i + 2) % COLORS.length],
  }))

  // Voice count bars
  const voiceBars = data.by_voice_count.filter((v) => v.num_voices > 0)

  return (
    <div className="space-y-6">
      {/* Row 1: Collection donut + Key mode donut */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Collection Distribution" subtitle="Corpus composition by collection">
          <div className="flex items-center gap-6">
            <div className="w-44 h-44 min-w-[176px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={donutData} cx="50%" cy="50%" innerRadius={45}
                    outerRadius={65} paddingAngle={3} dataKey="value" stroke="none">
                    {donutData.map((d, i) => <Cell key={i} fill={d.color} />)}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex-1 space-y-2">
              {donutData.map((d) => (
                <div key={d.name} className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: d.color }} />
                  <span className="text-xs flex-1 capitalize">{d.name}</span>
                  <span className="text-xs font-semibold text-primary">{d.value}</span>
                </div>
              ))}
            </div>
          </div>
        </Card>

        <Card title="Key Mode Distribution" subtitle="Major vs minor across the corpus">
          <div className="flex items-center gap-6">
            <div className="w-44 h-44 min-w-[176px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={modePie} cx="50%" cy="50%" innerRadius={45}
                    outerRadius={65} paddingAngle={3} dataKey="value" stroke="none">
                    {modePie.map((d, i) => <Cell key={i} fill={d.color} />)}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex-1 space-y-3">
              {modePie.map((d) => (
                <div key={d.name} className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: d.color }} />
                  <span className="text-sm flex-1">{d.name}</span>
                  <span className="text-sm font-semibold text-primary">{d.value}</span>
                </div>
              ))}
              <p className="text-xs text-muted-foreground mt-2">
                Total: {data.total} pieces
              </p>
            </div>
          </div>
        </Card>
      </div>

      {/* Row 2: Duration histogram + Voice count */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Duration Distribution" subtitle="Piece lengths across the corpus">
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={durationBars} barCategoryGap="20%">
                <XAxis dataKey="label" tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 11 }}
                  axisLine={{ stroke: "oklch(0.28 0.04 280)" }} tickLine={false} />
                <YAxis tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                  axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{
                  backgroundColor: "oklch(0.16 0.025 280)",
                  border: "1px solid oklch(0.28 0.04 280)",
                  borderRadius: "8px", color: "oklch(0.95 0.02 300)",
                }} cursor={{ fill: "oklch(0.25 0.05 290)", opacity: 0.3 }} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {durationBars.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Voice Count Distribution" subtitle="Number of voices per piece">
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={voiceBars} barCategoryGap="20%">
                <XAxis dataKey="num_voices"
                  tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 11 }}
                  axisLine={{ stroke: "oklch(0.28 0.04 280)" }} tickLine={false}
                  label={{ value: "Voices", position: "insideBottom", offset: -2,
                    fill: "oklch(0.65 0.05 290)", fontSize: 10 }} />
                <YAxis tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                  axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{
                  backgroundColor: "oklch(0.16 0.025 280)",
                  border: "1px solid oklch(0.28 0.04 280)",
                  borderRadius: "8px", color: "oklch(0.95 0.02 300)",
                }} cursor={{ fill: "oklch(0.25 0.05 290)", opacity: 0.3 }} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {voiceBars.map((_, i) => <Cell key={i} fill={COLORS[(i + 3) % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Row 3: Measures histogram */}
      <Card title="Measure Count Distribution" subtitle="Score length in measures">
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={measureBars} barCategoryGap="20%">
              <XAxis dataKey="label" tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 11 }}
                axisLine={{ stroke: "oklch(0.28 0.04 280)" }} tickLine={false} />
              <YAxis tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{
                backgroundColor: "oklch(0.16 0.025 280)",
                border: "1px solid oklch(0.28 0.04 280)",
                borderRadius: "8px", color: "oklch(0.95 0.02 300)",
              }} cursor={{ fill: "oklch(0.25 0.05 290)", opacity: 0.3 }} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {measureBars.map((_, i) => <Cell key={i} fill={COLORS[(i + 1) % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  )
}

// ─── Score Tab ────────────────────────────────────────────────────────────────

function ScoreTab() {
  const { activeScore } = useAppState()
  const [data, setData]         = useState<ScoreHarmonicAnalysis | null>(null)
  const [loading, setLoading]   = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError]       = useState<string | null>(null)

  const load = useCallback(async (refresh = false) => {
    if (!activeScore) return
    refresh ? setRefreshing(true) : setLoading(true)
    setError(null)
    try {
      const result = await api.analysis.score(activeScore.id, refresh)
      setData(result)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Analysis failed")
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [activeScore])

  // Auto-load when score changes
  useEffect(() => {
    if (activeScore) {
      setData(null)
      void load(false)
    }
  }, [activeScore?.id])  // eslint-disable-line react-hooks/exhaustive-deps

  if (!activeScore) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3 text-muted-foreground">
        <BookOpen className="w-10 h-10 opacity-20" />
        <p className="text-sm font-serif italic text-center max-w-xs">
          Load a piece from the Library into the Workbench to see its harmonic analysis
        </p>
      </div>
    )
  }

  if (loading) return <Loading label={`Analysing ${activeScore.movement_name ?? `BWV ${activeScore.bwv}`}… (this may take 30–60s on first run)`} />

  if (error) return (
    <div className="flex flex-col items-center justify-center h-64 gap-3">
      <p className="text-sm text-destructive">{error}</p>
      <Button variant="outline" size="sm" onClick={() => void load(true)}>Retry</Button>
    </div>
  )

  if (!data || data.total_chords === 0) return (
    <Empty message="No harmonic data could be extracted from this file. Try a different piece." />
  )

  // ── Radar data — top 7 roman numerals ────────────────────────────────────
  const topNumerals = data.roman_numerals.slice(0, 7)
  const radarData = topNumerals.map((rn) => ({
    harmony: rn.numeral,
    value: rn.percentage,
    fullMark: 100,
  }))

  // ── Bar data — chord quality frequency ───────────────────────────────────
  const qualityBars = data.chord_qualities.slice(0, 8).map((q) => ({
    name: q.quality.replace("dominant", "dom").replace("major7", "M7")
      .replace("minor7", "m7").replace("diminished", "dim").replace("augmented", "aug"),
    frequency: q.count,
  }))

  // ── Harmonic function bars ────────────────────────────────────────────────
  const funcBars = data.harmonic_functions.map((f) => ({
    name: f.function,
    value: f.percentage,
  }))

  const keyName = `${tonicName(data.global_key_tonic)} ${data.global_key_mode}`
  const confidence = Math.round(data.global_key_confidence * 100)

  const patternItems = [
    {
      id: 1, name: "Secondary Dominants",
      occurrences: data.secondary_dominant_count,
      detail: "chromatic tonicisations",
      icon: TrendingUp, color: "text-primary",
    },
    {
      id: 2, name: "Borrowed Chords",
      occurrences: data.borrowed_chord_count,
      detail: "modal mixture events",
      icon: Repeat, color: "text-accent",
    },
    {
      id: 3, name: "Total Chords",
      occurrences: data.total_chords,
      detail: "detected chord events",
      icon: Music, color: "text-primary",
    },
    {
      id: 4, name: "Dominant Function",
      occurrences: data.harmonic_functions.find((f) => f.function === "DOMINANT")?.count ?? 0,
      detail: "dominant-function events",
      icon: Waves, color: "text-accent",
    },
  ]

  return (
    <div className="space-y-6">
      {/* Score header + refresh */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-serif text-xl font-semibold">
            {activeScore.movement_name ?? `BWV ${activeScore.bwv}`}
          </h3>
          <p className="text-sm text-muted-foreground mt-0.5">
            Global key: <span className="text-primary font-medium">{keyName}</span>
            <span className="ml-2 text-xs opacity-60">({confidence}% confidence)</span>
          </p>
        </div>
        <Button variant="outline" size="sm" disabled={refreshing}
          className="border-primary/50 text-primary hover:bg-primary/10"
          onClick={() => void load(true)}>
          <RefreshCw className={`w-3 h-3 mr-1.5 ${refreshing ? "animate-spin" : ""}`} />
          {refreshing ? "Re-analysing…" : "Re-analyse"}
        </Button>
      </div>

      {/* Harmonic distribution */}
      <div className="bg-card rounded-xl p-6 border border-border" style={{ boxShadow: CARD_SHADOW }}>
        <h3 className="font-serif text-lg font-semibold mb-1">Harmonic Distribution</h3>
        <p className="text-xs text-muted-foreground mb-5">
          Roman numeral and chord quality analysis · {data.total_chords} chord events
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Radar — roman numeral balance */}
          <div className="bg-background/50 rounded-lg p-4 border border-border/50">
            <h4 className="text-xs font-medium text-muted-foreground mb-4 uppercase tracking-wider">
              Harmonic Balance
            </h4>
            <div className="h-60">
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

          {/* Bar — chord quality frequency */}
          <div className="bg-background/50 rounded-lg p-4 border border-border/50">
            <h4 className="text-xs font-medium text-muted-foreground mb-4 uppercase tracking-wider">
              Chord Quality Frequency
            </h4>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={qualityBars} barCategoryGap="20%">
                  <XAxis dataKey="name"
                    tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 11 }}
                    axisLine={{ stroke: "oklch(0.28 0.04 280)" }} tickLine={false} />
                  <YAxis tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                    axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{
                    backgroundColor: "oklch(0.16 0.025 280)",
                    border: "1px solid oklch(0.28 0.04 280)",
                    borderRadius: "8px", color: "oklch(0.95 0.02 300)",
                  }} cursor={{ fill: "oklch(0.25 0.05 290)", opacity: 0.3 }} />
                  <Bar dataKey="frequency" radius={[4, 4, 0, 0]}>
                    {qualityBars.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Harmonic function breakdown */}
      {funcBars.length > 0 && (
        <Card title="Harmonic Function Breakdown"
          subtitle="Tonic / Dominant / Predominant distribution">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={funcBars} layout="vertical" barCategoryGap="30%">
                <XAxis type="number" tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                  axisLine={false} tickLine={false}
                  tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
                <YAxis type="category" dataKey="name" width={110}
                  tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 11 }}
                  axisLine={false} tickLine={false} />
                <Tooltip formatter={(v: number) => [`${v}%`, "Share"]}
                  contentStyle={{
                    backgroundColor: "oklch(0.16 0.025 280)",
                    border: "1px solid oklch(0.28 0.04 280)",
                    borderRadius: "8px", color: "oklch(0.95 0.02 300)",
                  }} cursor={{ fill: "oklch(0.25 0.05 290)", opacity: 0.3 }} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {funcBars.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* Pattern cards */}
      <div className="bg-card rounded-xl p-6 border border-border" style={{ boxShadow: CARD_SHADOW }}>
        <h3 className="font-serif text-lg font-semibold mb-1">Pattern Recognition</h3>
        <p className="text-xs text-muted-foreground mb-5">
          Harmonic events detected by the analysis engine
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {patternItems.map((p) => (
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
      </div>
    </div>
  )
}

// ─── Main export ──────────────────────────────────────────────────────────────

export function AnalysisView() {
  const { activeScore } = useAppState()

  return (
    <div className="space-y-4 h-full">
      <Tabs defaultValue="corpus">
        <div className="flex items-center justify-between mb-4">
          <TabsList className="bg-card border border-border">
            <TabsTrigger value="corpus" className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Corpus Analytics
            </TabsTrigger>
            <TabsTrigger value="score" className="flex items-center gap-2">
              <Music className="w-4 h-4" />
              Score Analysis
              {activeScore && (
                <span className="ml-1 px-1.5 py-0.5 text-xs rounded-full bg-primary/20 text-primary">
                  {activeScore.bwv ?? "loaded"}
                </span>
              )}
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="corpus">
          <CorpusTab />
        </TabsContent>

        <TabsContent value="score">
          <ScoreTab />
        </TabsContent>
      </Tabs>

      <div className="flex justify-between text-sm text-muted-foreground font-serif italic px-2">
        <span>Bach Propagation Engine v2.0</span>
        <span>Analysis Report</span>
      </div>
    </div>
  )
}