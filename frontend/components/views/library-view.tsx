"use client"

import { useEffect, useState, useCallback, useRef } from "react"
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ZAxis,
} from "recharts"
import {
  Music,
  Database,
  Users,
  FileMusic,
  Filter,
  Upload,
  Search,
  ChevronLeft,
  ChevronRight,
  Loader2,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

import { api, type CorpusStats } from "@/lib/api-client"
import type { CorpusFile } from "@/lib/app-state"
import { useLoadIntoWorkbench } from "@/hooks/use-load-into-workbench"

// ─── Chart palette (unchanged from original) ─────────────────────────────────

const CHART_COLORS = [
  "oklch(0.75 0.15 290)",
  "oklch(0.70 0.12 310)",
  "oklch(0.78 0.12 280)",
  "oklch(0.65 0.18 300)",
  "oklch(0.72 0.14 320)",
  "oklch(0.68 0.16 270)",
]

// Scatter plot stays as illustrative mock — requires computing stylistic
// features from audio, which is a separate pipeline. Real data populates
// the stats cards and table.
const SCATTER_MOCK = [
  { x: 85, y: 78, z: 315, label: "Bach"     },
  { x: 72, y: 82, z: 180, label: "Bach"     },
  { x: 90, y: 65, z: 220, label: "Bach"     },
  { x: 55, y: 70, z: 54,  label: "Other"    },
  { x: 65, y: 88, z: 45,  label: "Other"    },
  { x: 40, y: 55, z: 36,  label: "Other"    },
]

const PAGE_SIZE = 20

// ─── Stats cards ──────────────────────────────────────────────────────────────

function StatsCards({ stats }: { stats: CorpusStats | null }) {
  const totalCollections = stats ? Object.keys(stats.by_collection).length : 0
  const minor = stats?.by_key_mode.minor ?? 0
  const major = stats?.by_key_mode.major ?? 0

  const cards = [
    {
      title: "Total Tracks",
      value: stats ? String(stats.total) : "—",
      subtitle: "compositions in corpus",
      icon: FileMusic,
    },
    {
      title: "Collections",
      value: stats ? String(totalCollections) : "—",
      subtitle: stats
        ? Object.keys(stats.by_collection).slice(0, 3).join(", ")
        : "loading…",
      icon: Users,
    },
    {
      title: "Minor / Major",
      value: stats ? `${minor} / ${major}` : "—",
      subtitle: "by key mode",
      icon: Music,
    },
    {
      title: "Corpus Status",
      value: stats ? "Live" : "—",
      subtitle: "embedded + indexed",
      icon: Database,
    },
  ]

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card) => (
        <div
          key={card.title}
          className="bg-card rounded-xl p-5 border border-border relative overflow-hidden"
          style={{
            boxShadow: "0 0 20px rgba(168, 130, 255, 0.1), 0 0 40px rgba(168, 130, 255, 0.05)",
          }}
        >
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
                {card.title}
              </p>
              <p className="text-3xl font-serif font-bold text-primary">{card.value}</p>
              <p className="text-xs text-muted-foreground mt-1">{card.subtitle}</p>
            </div>
            <div className="p-2 rounded-lg bg-primary/10">
              <card.icon className="w-5 h-5 text-primary" />
            </div>
          </div>
          <div
            className="absolute -bottom-4 -right-4 w-24 h-24 rounded-full opacity-20"
            style={{
              background: "radial-gradient(circle, oklch(0.75 0.15 290) 0%, transparent 70%)",
            }}
          />
        </div>
      ))}
    </div>
  )
}

// ─── Charts row ───────────────────────────────────────────────────────────────

function ChartsRow({ stats }: { stats: CorpusStats | null }) {
  const collectionEntries = stats
    ? Object.entries(stats.by_collection)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6)
    : []

  const donutData = collectionEntries.map(([name, value], i) => ({
    name,
    value,
    color: CHART_COLORS[i % CHART_COLORS.length],
  }))

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Collection distribution donut */}
      <div
        className="bg-card rounded-xl p-6 border border-border"
        style={{
          boxShadow: "0 0 20px rgba(168, 130, 255, 0.1), 0 0 40px rgba(168, 130, 255, 0.05)",
          backdropFilter: "blur(10px)",
        }}
      >
        <h3 className="font-serif text-lg font-semibold mb-1">Collection Distribution</h3>
        <p className="text-xs text-muted-foreground mb-4">Corpus composition by collection</p>

        {!stats ? (
          <div className="h-48 flex items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="flex items-center gap-6">
            <div className="w-48 h-48 min-w-[192px] min-h-[192px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={donutData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={70}
                    paddingAngle={3}
                    dataKey="value"
                    stroke="none"
                  >
                    {donutData.map((entry, i) => (
                      <Cell key={`cell-${i}`} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex-1 space-y-3">
              {donutData.map((d) => (
                <div key={d.name} className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: d.color }} />
                  <span className="text-sm flex-1 capitalize">{d.name}</span>
                  <span className="text-sm font-semibold text-primary">{d.value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Stylistic density scatter (illustrative) */}
      <div
        className="bg-card rounded-xl p-6 border border-border"
        style={{
          boxShadow: "0 0 20px rgba(168, 130, 255, 0.1), 0 0 40px rgba(168, 130, 255, 0.05)",
          backdropFilter: "blur(10px)",
        }}
      >
        <h3 className="font-serif text-lg font-semibold mb-1">Stylistic Density</h3>
        <p className="text-xs text-muted-foreground mb-4">Rhythmic Complexity vs Melodic Range</p>

        <div className="h-48 min-h-[192px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
              <XAxis
                type="number" dataKey="x" name="Rhythmic Complexity" domain={[20, 100]}
                tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                axisLine={{ stroke: "oklch(0.28 0.04 280)" }} tickLine={false}
                label={{ value: "Rhythmic Complexity", position: "bottom",
                  fill: "oklch(0.65 0.05 290)", fontSize: 10, offset: 0 }}
              />
              <YAxis
                type="number" dataKey="y" name="Melodic Range" domain={[30, 100]}
                tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                axisLine={{ stroke: "oklch(0.28 0.04 280)" }} tickLine={false}
                label={{ value: "Melodic Range", angle: -90, position: "insideLeft",
                  fill: "oklch(0.65 0.05 290)", fontSize: 10, offset: 10 }}
              />
              <ZAxis type="number" dataKey="z" range={[40, 200]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "oklch(0.16 0.025 280)",
                  border: "1px solid oklch(0.28 0.04 280)",
                  borderRadius: "8px",
                  color: "oklch(0.95 0.02 300)",
                }}
              />
              <Scatter
                data={SCATTER_MOCK.filter((d) => d.label === "Bach")}
                fill="oklch(0.75 0.15 290)"
                name="Bach"
              />
              <Scatter
                data={SCATTER_MOCK.filter((d) => d.label === "Other")}
                fill="oklch(0.70 0.12 310)"
                name="Other"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="flex gap-4 mt-3 justify-center">
          {["Bach", "Other"].map((label, i) => (
            <div key={label} className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: CHART_COLORS[i] }} />
              <span className="text-xs text-muted-foreground">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function LibraryView() {
  const [stats, setStats]       = useState<CorpusStats | null>(null)
  const [files, setFiles]       = useState<CorpusFile[]>([])
  const [total, setTotal]       = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError]       = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState("")
  const [debouncedQuery, setDebouncedQuery] = useState("")
  const [collection, setCollection] = useState("all")
  const [keyMode, setKeyMode]   = useState("all")
  const [page, setPage]         = useState(1)
  const searchTimer             = useRef<ReturnType<typeof setTimeout> | null>(null)
  const loadIntoWorkbench       = useLoadIntoWorkbench()

  // Load stats once
  useEffect(() => {
    api.corpus.stats().then(setStats).catch(console.error)
  }, [])

  // Debounce search input
  useEffect(() => {
    if (searchTimer.current) clearTimeout(searchTimer.current)
    searchTimer.current = setTimeout(() => {
      setDebouncedQuery(searchQuery)
      setPage(1)
    }, 400)
    return () => { if (searchTimer.current) clearTimeout(searchTimer.current) }
  }, [searchQuery])

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
          page,
          page_size: PAGE_SIZE,
          collection: collection !== "all" ? collection : undefined,
          key_mode:   keyMode !== "all"    ? keyMode    : undefined,
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

      {/* Stats cards — real data */}
      <StatsCards stats={stats} />

      {/* Charts row — donut uses real stats, scatter is illustrative */}
      <ChartsRow stats={stats} />

      {/* Repository Table */}
      <div
        className="bg-card rounded-xl border border-border overflow-hidden"
        style={{
          boxShadow: "0 0 20px rgba(168, 130, 255, 0.1), 0 0 40px rgba(168, 130, 255, 0.05)",
          backdropFilter: "blur(10px)",
        }}
      >
        {/* Filter / Search bar */}
        <div className="p-4 border-b border-border flex flex-wrap items-center gap-3">
          {/* AI-assisted search */}
          <div className="relative flex-1 min-w-48">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              className="pl-9 bg-background border-border h-9 text-sm"
              placeholder="Search corpus… (AI-assisted)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
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
                <SelectItem value="inventions">Inventions</SelectItem>
                <SelectItem value="sinfonias">Sinfonias</SelectItem>
                <SelectItem value="preludes">Preludes</SelectItem>
                <SelectItem value="fugues">Fugues</SelectItem>
                <SelectItem value="suites">Suites</SelectItem>
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

        {/* Error */}
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
                {["Name", "BWV", "Collection", "Key", "Measures", "Status", ""].map((h) => (
                  <th
                    key={h}
                    className={`p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider ${
                      h === "" ? "text-right" : "text-left"
                    }`}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr>
                  <td colSpan={7} className="text-center py-12">
                    <Loader2 className="w-5 h-5 animate-spin mx-auto text-muted-foreground" />
                  </td>
                </tr>
              ) : files.length === 0 ? (
                <tr>
                  <td colSpan={7} className="text-center py-12 text-muted-foreground font-serif italic">
                    No compositions found
                  </td>
                </tr>
              ) : (
                files.map((file) => <CorpusRow key={file.id} file={file} onLoad={loadIntoWorkbench} />)
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {!debouncedQuery && totalPages > 1 && (
          <div className="p-4 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
            <span>
              {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, total)} of {total}
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline" size="sm"
                disabled={page === 1}
                onClick={() => setPage((p) => p - 1)}
              >
                <ChevronLeft className="w-3 h-3" />
              </Button>
              <Button
                variant="outline" size="sm"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
              >
                <ChevronRight className="w-3 h-3" />
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex justify-between text-sm text-muted-foreground font-serif italic px-2">
        <span>Bach Propagation Engine v2.0</span>
        <span>Baroque Corpus Explorer</span>
      </div>
    </div>
  )
}

// ─── Table row ────────────────────────────────────────────────────────────────

function CorpusRow({
  file,
  onLoad,
}: {
  file: CorpusFile
  onLoad: (id: string) => Promise<void>
}) {
  const [loading, setLoading] = useState(false)

  const handleLoad = async () => {
    setLoading(true)
    try { await onLoad(file.id) }
    finally { setLoading(false) }
  }

  const ks = (file.key_signature ?? "").toLowerCase()
  const isMinor = ks.includes("minor")

  return (
    <tr className="border-b border-border/50 hover:bg-primary/5 transition-colors group">
      <td className="p-4">
        <span className="font-serif font-medium">{file.movement_name ?? "Untitled"}</span>
      </td>
      <td className="p-4">
        <span className="text-sm text-muted-foreground font-mono">{file.bwv ?? "—"}</span>
      </td>
      <td className="p-4">
        <span className="text-sm text-muted-foreground capitalize">{file.collection}</span>
      </td>
      <td className="p-4">
        <span className="text-sm text-muted-foreground">{file.key_signature ?? "—"}</span>
      </td>
      <td className="p-4">
        <span className="text-sm text-muted-foreground">{file.num_measures ?? "—"}</span>
      </td>
      <td className="p-4">
        <span
          className={`inline-flex px-2.5 py-1 rounded-full text-xs font-semibold ${
            isMinor ? "bg-accent/20 text-accent" : "bg-primary/20 text-primary"
          }`}
        >
          {isMinor ? "Minor" : "Major"}
        </span>
      </td>
      <td className="p-4 text-right">
        <Button
          variant="outline"
          size="sm"
          className="border-primary/50 text-primary hover:bg-primary/10"
          disabled={loading}
          onClick={handleLoad}
        >
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