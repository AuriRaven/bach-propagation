"use client"

import { useEffect, useState, useCallback, useRef } from "react"
import {
  PieChart, Pie, Cell, ResponsiveContainer,
  ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  Tooltip, BarChart, Bar,
} from "recharts"
import {
  Music, Database, Users, FileMusic, Filter,
  Upload, Search, ChevronLeft, ChevronRight, Loader2,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input }  from "@/components/ui/input"
import {
  Select, SelectContent, SelectItem,
  SelectTrigger, SelectValue,
} from "@/components/ui/select"

import { api, type CorpusStats, type CorpusAnalytics } from "@/lib/api-client"
import type { CorpusFile } from "@/lib/app-state"
import { useLoadIntoWorkbench } from "@/hooks/use-load-into-workbench"

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

const PAGE_SIZE = 20

// ─── Stats cards ──────────────────────────────────────────────────────────────

function StatsCards({ stats, analytics }: {
  stats: CorpusStats | null
  analytics: CorpusAnalytics | null
}) {
  const minor   = analytics?.by_key_mode.find((k) => k.mode === "minor")?.count ?? 0
  const major   = analytics?.by_key_mode.find((k) => k.mode === "major")?.count ?? 0
  const collections = analytics ? Object.keys(
    analytics.by_collection.reduce((a, c) => ({ ...a, [c.collection]: c.count }), {})
  ).length : 0

  const cards = [
    { title: "Total Tracks",  value: stats ? String(stats.total) : "—",
      subtitle: "compositions in corpus", icon: FileMusic },
    { title: "Collections",   value: analytics ? String(collections) : "—",
      subtitle: analytics ? analytics.by_collection.slice(0,3).map(c => c.collection.replace(/_/g," ")).join(", ") : "loading…",
      icon: Users },
    { title: "Minor / Major", value: analytics ? `${minor} / ${major}` : "—",
      subtitle: "by key mode", icon: Music },
    { title: "Corpus Status", value: "Live", subtitle: "embedded + indexed", icon: Database },
  ]

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((c) => (
        <div key={c.title} className="bg-card rounded-xl p-5 border border-border relative overflow-hidden"
          style={{ boxShadow: GLOW }}>
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">{c.title}</p>
              <p className="text-3xl font-serif font-bold text-primary">{c.value}</p>
              <p className="text-xs text-muted-foreground mt-1">{c.subtitle}</p>
            </div>
            <div className="p-2 rounded-lg bg-primary/10"><c.icon className="w-5 h-5 text-primary" /></div>
          </div>
          <div className="absolute -bottom-4 -right-4 w-24 h-24 rounded-full opacity-20"
            style={{ background: "radial-gradient(circle, oklch(0.75 0.15 290) 0%, transparent 70%)" }} />
        </div>
      ))}
    </div>
  )
}

// ─── Corpus analytics charts ──────────────────────────────────────────────────

interface ScatterPoint {
  x: number   // num_measures
  y: number   // duration_seconds
  z: number   // fixed size
  collection: string
  name: string
  color: string
}

function CorpusCharts({ analytics, scatterData }: {
  analytics: CorpusAnalytics | null
  scatterData: ScatterPoint[]
}) {
  if (!analytics) return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {[0,1].map((i) => (
        <div key={i} className="bg-card rounded-xl p-6 border border-border h-64 flex items-center justify-center"
          style={{ boxShadow: GLOW }}>
          <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
        </div>
      ))}
    </div>
  )

  // Collection donut
  const collectionDonut = analytics.by_collection.slice(0, 6).map((c, i) => ({
    name: c.collection.replace(/_/g, " "),
    value: c.count,
    color: COLORS[i % COLORS.length],
  }))

  // Key mode donut
  const modeDonut = analytics.by_key_mode.map((k, i) => ({
    name: k.mode.charAt(0).toUpperCase() + k.mode.slice(1),
    value: k.count,
    color: COLORS[(i + 2) % COLORS.length],
  }))

  // Duration bars
  const durationBars = analytics.duration_histogram.filter((b) => b.count > 0)

  // Voice count bars
  const voiceBars = analytics.by_voice_count.filter((v) => v.num_voices > 0 && v.num_voices <= 8)

  // Collections for scatter legend
  const collectionColors = Object.fromEntries(
    analytics.by_collection.slice(0, 6).map((c, i) => [c.collection, COLORS[i % COLORS.length]])
  )
  const scatterByCollection = analytics.by_collection.slice(0, 6).map((c) => ({
    name: c.collection.replace(/_/g, " "),
    color: collectionColors[c.collection] ?? COLORS[0],
    points: scatterData.filter((p) => p.collection === c.collection),
  }))

  return (
    <div className="space-y-6">
      {/* Row 1: Two donuts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Collection distribution */}
        <div className="bg-card rounded-xl p-6 border border-border"
          style={{ boxShadow: GLOW, backdropFilter: "blur(10px)" }}>
          <h3 className="font-serif text-lg font-semibold mb-1">Collection Distribution</h3>
          <p className="text-xs text-muted-foreground mb-4">Corpus composition by collection</p>
          <div className="flex items-center gap-6">
            <div className="w-44 h-44 min-w-[176px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={collectionDonut} cx="50%" cy="50%" innerRadius={45}
                    outerRadius={65} paddingAngle={3} dataKey="value" stroke="none">
                    {collectionDonut.map((d, i) => <Cell key={i} fill={d.color} />)}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex-1 space-y-2">
              {collectionDonut.map((d) => (
                <div key={d.name} className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: d.color }} />
                  <span className="text-xs flex-1 capitalize">{d.name}</span>
                  <span className="text-xs font-semibold text-primary">{d.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Key mode distribution */}
        <div className="bg-card rounded-xl p-6 border border-border"
          style={{ boxShadow: GLOW, backdropFilter: "blur(10px)" }}>
          <h3 className="font-serif text-lg font-semibold mb-1">Key Mode Distribution</h3>
          <p className="text-xs text-muted-foreground mb-4">Major vs minor across the corpus</p>
          <div className="flex items-center gap-6">
            <div className="w-44 h-44 min-w-[176px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={modeDonut} cx="50%" cy="50%" innerRadius={45}
                    outerRadius={65} paddingAngle={3} dataKey="value" stroke="none">
                    {modeDonut.map((d, i) => <Cell key={i} fill={d.color} />)}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex-1 space-y-3">
              {modeDonut.map((d) => (
                <div key={d.name} className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: d.color }} />
                  <span className="text-sm flex-1">{d.name}</span>
                  <span className="text-sm font-semibold text-primary">{d.value}</span>
                </div>
              ))}
              <p className="text-xs text-muted-foreground mt-1">Total: {analytics.total} pieces</p>
            </div>
          </div>
        </div>
      </div>

      {/* Row 2: Stylistic density scatter — real data */}
      <div className="bg-card rounded-xl p-6 border border-border"
        style={{ boxShadow: GLOW, backdropFilter: "blur(10px)" }}>
        <h3 className="font-serif text-lg font-semibold mb-1">Stylistic Density</h3>
        <p className="text-xs text-muted-foreground mb-4">
          Measures vs Duration — each dot is a piece, coloured by collection
        </p>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <XAxis type="number" dataKey="x" name="Measures" domain={[0, "auto"]}
                tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                axisLine={{ stroke: "oklch(0.28 0.04 280)" }} tickLine={false}
                label={{ value: "Measures", position: "bottom",
                  fill: "oklch(0.65 0.05 290)", fontSize: 10, offset: 0 }} />
              <YAxis type="number" dataKey="y" name="Duration (s)" domain={[0, "auto"]}
                tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                axisLine={{ stroke: "oklch(0.28 0.04 280)" }} tickLine={false}
                label={{ value: "Duration (s)", angle: -90, position: "insideLeft",
                  fill: "oklch(0.65 0.05 290)", fontSize: 10, offset: 10 }} />
              <ZAxis type="number" dataKey="z" range={[20, 60]} />
              <Tooltip contentStyle={TOOLTIP_STYLE}
                formatter={(v: number, name: string) => {
                  if (name === "x") return [v, "Measures"]
                  if (name === "y") return [`${Math.round(v)}s`, "Duration"]
                  return [v, name]
                }} />
              {scatterByCollection.map((col) => (
                <Scatter key={col.name} name={col.name}
                  data={col.points} fill={col.color} opacity={0.7} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="flex flex-wrap gap-4 mt-2 justify-center">
          {scatterByCollection.filter((c) => c.points.length > 0).map((c) => (
            <div key={c.name} className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: c.color }} />
              <span className="text-xs text-muted-foreground capitalize">{c.name}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Row 3: Duration histogram + Voice count */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-card rounded-xl p-6 border border-border"
          style={{ boxShadow: GLOW, backdropFilter: "blur(10px)" }}>
          <h3 className="font-serif text-lg font-semibold mb-1">Duration Distribution</h3>
          <p className="text-xs text-muted-foreground mb-4">Piece lengths across the corpus</p>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={durationBars} barCategoryGap="20%">
                <XAxis dataKey="label" tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 11 }}
                  axisLine={{ stroke: "oklch(0.28 0.04 280)" }} tickLine={false} />
                <YAxis tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                  axisLine={false} tickLine={false} />
                <Tooltip contentStyle={TOOLTIP_STYLE}
                  cursor={{ fill: "oklch(0.25 0.05 290)", opacity: 0.3 }} />
                <Bar dataKey="count" radius={[4,4,0,0]}>
                  {durationBars.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-card rounded-xl p-6 border border-border"
          style={{ boxShadow: GLOW, backdropFilter: "blur(10px)" }}>
          <h3 className="font-serif text-lg font-semibold mb-1">Voice Count Distribution</h3>
          <p className="text-xs text-muted-foreground mb-4">Number of voices per piece</p>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={voiceBars} barCategoryGap="20%">
                <XAxis dataKey="num_voices" tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 11 }}
                  axisLine={{ stroke: "oklch(0.28 0.04 280)" }} tickLine={false}
                  label={{ value: "Voices", position: "insideBottom", offset: -2,
                    fill: "oklch(0.65 0.05 290)", fontSize: 10 }} />
                <YAxis tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                  axisLine={false} tickLine={false} />
                <Tooltip contentStyle={TOOLTIP_STYLE}
                  cursor={{ fill: "oklch(0.25 0.05 290)", opacity: 0.3 }} />
                <Bar dataKey="count" radius={[4,4,0,0]}>
                  {voiceBars.map((_, i) => <Cell key={i} fill={COLORS[(i+3) % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function LibraryView() {
  const [stats,     setStats]     = useState<CorpusStats | null>(null)
  const [analytics, setAnalytics] = useState<CorpusAnalytics | null>(null)
  const [scatterData, setScatterData] = useState<ScatterPoint[]>([])
  const [files,     setFiles]     = useState<CorpusFile[]>([])
  const [total,     setTotal]     = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [error,     setError]     = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState("")
  const [debouncedQuery, setDebouncedQuery] = useState("")
  const [collection, setCollection] = useState("all")
  const [keyMode,   setKeyMode]   = useState("all")
  const [page,      setPage]      = useState(1)
  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const loadIntoWorkbench = useLoadIntoWorkbench()

  // ── Initial data loads ────────────────────────────────────────────────────
  useEffect(() => {
    // Corpus stats (fast)
    api.corpus.stats().then(setStats).catch(console.error)

    // Corpus analytics (slightly slower — aggregation query)
    api.analysis.corpus().then((data) => {
      setAnalytics(data)
    }).catch(console.error)

    // Scatter sample — first 100 pieces with real num_measures + duration_seconds
    api.corpus.list({ page_size: 100, page: 1 }).then((resp) => {
      const collectionIndex: Record<string, number> = {}
      let ci = 0
      const points: ScatterPoint[] = resp.items
        .filter((f) => f.num_measures && f.duration_seconds)
        .map((f) => {
          const col = f.collection ?? "unknown"
          if (!(col in collectionIndex)) collectionIndex[col] = ci++
          return {
            x: f.num_measures ?? 0,
            y: f.duration_seconds ?? 0,
            z: 1,
            collection: col,
            name: f.movement_name ?? `BWV ${f.bwv}` ?? "—",
            color: COLORS[collectionIndex[col] % COLORS.length],
          }
        })
      setScatterData(points)
    }).catch(console.error)
  }, [])

  // ── Search debounce ───────────────────────────────────────────────────────
  useEffect(() => {
    if (searchTimer.current) clearTimeout(searchTimer.current)
    searchTimer.current = setTimeout(() => {
      setDebouncedQuery(searchQuery)
      setPage(1)
    }, 400)
    return () => { if (searchTimer.current) clearTimeout(searchTimer.current) }
  }, [searchQuery])

  // ── Fetch table data ──────────────────────────────────────────────────────
  const fetchFiles = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      if (debouncedQuery.trim()) {
        const results = await api.corpus.search(debouncedQuery, 50)
        setFiles(results)
        setTotal(results.length)
      } else {
        const resp = await api.corpus.list({
          page, page_size: PAGE_SIZE,
          collection: collection !== "all" ? collection : undefined,
          key_mode:   keyMode   !== "all" ? keyMode   : undefined,
        })
        setFiles(resp.items)
        setTotal(resp.total)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load corpus")
    } finally {
      setIsLoading(false)
    }
  }, [debouncedQuery, collection, keyMode, page])

  useEffect(() => { void fetchFiles() }, [fetchFiles])

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="space-y-6 h-full">

      {/* Stats cards */}
      <StatsCards stats={stats} analytics={analytics} />

      {/* Corpus analytics charts */}
      <CorpusCharts analytics={analytics} scatterData={scatterData} />

      {/* Repository table */}
      <div className="bg-card rounded-xl border border-border overflow-hidden"
        style={{ boxShadow: GLOW, backdropFilter: "blur(10px)" }}>

        {/* Filter bar */}
        <div className="p-4 border-b border-border flex flex-wrap items-center gap-3">
          <div className="relative flex-1 min-w-48">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input className="pl-9 bg-background border-border h-9 text-sm"
              placeholder="Search corpus… (AI-assisted)"
              value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} />
          </div>
          <div className="flex items-center gap-3">
            <Filter className="w-4 h-4 text-muted-foreground" />
            <Select value={collection} onValueChange={(v) => { setCollection(v); setPage(1) }}>
              <SelectTrigger className="w-40 bg-background border-border h-9">
                <SelectValue placeholder="Collection" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Collections</SelectItem>
                <SelectItem value="chorales">Chorales</SelectItem>
                <SelectItem value="keyboard">Keyboard</SelectItem>
                <SelectItem value="solo_instruments">Solo Instruments</SelectItem>
                <SelectItem value="organ">Organ</SelectItem>
                <SelectItem value="orchestral">Orchestral</SelectItem>
                <SelectItem value="inventions">Inventions</SelectItem>
              </SelectContent>
            </Select>
            <Select value={keyMode} onValueChange={(v) => { setKeyMode(v); setPage(1) }}>
              <SelectTrigger className="w-32 bg-background border-border h-9">
                <SelectValue placeholder="Key mode" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Modes</SelectItem>
                <SelectItem value="major">Major</SelectItem>
                <SelectItem value="minor">Minor</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <span className="text-xs text-muted-foreground ml-auto">
            {isLoading ? "Loading…" : `Showing ${files.length} of ${total}`}
          </span>
        </div>

        {error && (
          <div className="px-4 py-3 text-sm text-destructive bg-destructive/10 border-b border-border">
            {error}
          </div>
        )}

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border bg-background/50">
                {["Name","BWV","Collection","Key","Measures","Status",""].map((h) => (
                  <th key={h} className={`p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider ${h === "" ? "text-right" : "text-left"}`}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr><td colSpan={7} className="text-center py-12">
                  <Loader2 className="w-5 h-5 animate-spin mx-auto text-muted-foreground" />
                </td></tr>
              ) : files.length === 0 ? (
                <tr><td colSpan={7} className="text-center py-12 text-muted-foreground font-serif italic">
                  No compositions found
                </td></tr>
              ) : (
                files.map((file) => (
                  <CorpusRow key={file.id} file={file} onLoad={loadIntoWorkbench} />
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {!debouncedQuery && totalPages > 1 && (
          <div className="p-4 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
            <span>{(page-1)*PAGE_SIZE+1}–{Math.min(page*PAGE_SIZE, total)} of {total}</span>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" disabled={page===1}
                onClick={() => setPage((p) => p-1)}>
                <ChevronLeft className="w-3 h-3" />
              </Button>
              <Button variant="outline" size="sm" disabled={page>=totalPages}
                onClick={() => setPage((p) => p+1)}>
                <ChevronRight className="w-3 h-3" />
              </Button>
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-between text-sm text-muted-foreground font-serif italic px-2">
        <span>Bach Propagation Engine v2.0</span>
        <span>Baroque Corpus Explorer</span>
      </div>
    </div>
  )
}

// ─── Table row ────────────────────────────────────────────────────────────────

function getDisplayName(file: CorpusFile): string {
  if (file.movement_name?.trim()) return file.movement_name
  const parts: string[] = []
  if (file.bwv) parts.push(`BWV ${file.bwv}`)
  const col = file.collection?.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
  if (col) parts.push(col)
  if (file.key_signature) parts.push(`in ${file.key_signature}`)
  return parts.length > 0 ? parts.join(" · ") : "Untitled"
}

function CorpusRow({ file, onLoad }: { file: CorpusFile; onLoad: (id: string) => Promise<void> }) {
  const [loading, setLoading] = useState(false)

  const handleLoad = async () => {
    setLoading(true)
    try { await onLoad(file.id) }
    finally { setLoading(false) }
  }

  const ks      = (file.key_signature ?? "").toLowerCase()
  const isMinor = ks.includes("minor")

  return (
    <tr className="border-b border-border/50 hover:bg-primary/5 transition-colors group">
      <td className="p-4">
        <span className="font-serif font-medium">{getDisplayName(file)}</span>
      </td>
      <td className="p-4">
        <span className="text-sm text-muted-foreground font-mono">{file.bwv ?? "—"}</span>
      </td>
      <td className="p-4">
        <span className="text-sm text-muted-foreground capitalize">
          {file.collection?.replace(/_/g, " ")}
        </span>
      </td>
      <td className="p-4">
        <span className="text-sm text-muted-foreground">{file.key_signature ?? "—"}</span>
      </td>
      <td className="p-4">
        <span className="text-sm text-muted-foreground">{file.num_measures ?? "—"}</span>
      </td>
      <td className="p-4">
        <span className={`inline-flex px-2.5 py-1 rounded-full text-xs font-semibold ${
          isMinor ? "bg-accent/20 text-accent" : "bg-primary/20 text-primary"
        }`}>
          {isMinor ? "Minor" : "Major"}
        </span>
      </td>
      <td className="p-4 text-right">
        <Button variant="outline" size="sm"
          className="border-primary/50 text-primary hover:bg-primary/10"
          disabled={loading} onClick={handleLoad}>
          {loading
            ? <Loader2 className="w-3 h-3 animate-spin mr-1.5" />
            : <Upload className="w-3 h-3 mr-1.5" />
          }
          Load into Workbench
        </Button>
      </td>
    </tr>
  )
}