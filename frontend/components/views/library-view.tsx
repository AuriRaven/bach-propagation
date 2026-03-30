"use client"

import { useState } from "react"
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
import { Music, Database, Users, FileMusic, Filter, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

// Summary cards data
const summaryCards = [
  {
    title: "Total Tracks",
    value: "450",
    subtitle: "compositions",
    icon: FileMusic,
  },
  {
    title: "Composers",
    value: "4",
    subtitle: "Bach, Telemann, Vivaldi, Corelli",
    icon: Users,
  },
  {
    title: "Cello Solo Subset",
    value: "36",
    subtitle: "target focus",
    icon: Music,
  },
  {
    title: "Total Tokens",
    value: "2.4M",
    subtitle: "training data volume",
    icon: Database,
  },
]

// Composer distribution data for donut chart
const composerData = [
  { name: "Bach", value: 70, color: "oklch(0.75 0.15 290)" },
  { name: "Telemann", value: 12, color: "oklch(0.70 0.12 310)" },
  { name: "Vivaldi", value: 10, color: "oklch(0.78 0.12 280)" },
  { name: "Corelli", value: 8, color: "oklch(0.65 0.18 300)" },
]

// Stylistic density scatter plot data
const stylisticData = [
  { composer: "Bach", x: 85, y: 78, z: 315, color: "oklch(0.75 0.15 290)" },
  { composer: "Bach", x: 72, y: 82, z: 180, color: "oklch(0.75 0.15 290)" },
  { composer: "Bach", x: 90, y: 65, z: 220, color: "oklch(0.75 0.15 290)" },
  { composer: "Telemann", x: 55, y: 70, z: 54, color: "oklch(0.70 0.12 310)" },
  { composer: "Telemann", x: 48, y: 62, z: 42, color: "oklch(0.70 0.12 310)" },
  { composer: "Vivaldi", x: 65, y: 88, z: 45, color: "oklch(0.78 0.12 280)" },
  { composer: "Vivaldi", x: 58, y: 92, z: 38, color: "oklch(0.78 0.12 280)" },
  { composer: "Corelli", x: 40, y: 55, z: 36, color: "oklch(0.65 0.18 300)" },
  { composer: "Corelli", x: 35, y: 48, z: 28, color: "oklch(0.65 0.18 300)" },
]

// Repository table data
const repositoryData = [
  { id: 1, name: "Suite No. 1 - Prelude", composer: "Bach", instrument: "Cello Solo", status: "target" },
  { id: 2, name: "Suite No. 1 - Allemande", composer: "Bach", instrument: "Cello Solo", status: "target" },
  { id: 3, name: "Suite No. 2 - Sarabande", composer: "Bach", instrument: "Cello Solo", status: "target" },
  { id: 4, name: "Suite No. 3 - Bourée", composer: "Bach", instrument: "Cello Solo", status: "target" },
  { id: 5, name: "Sonata in D Minor", composer: "Bach", instrument: "Violin", status: "augmentation" },
  { id: 6, name: "Concerto in A Minor", composer: "Vivaldi", instrument: "Cello Solo", status: "augmentation" },
  { id: 7, name: "Trio Sonata No. 2", composer: "Corelli", instrument: "Trio Sonata", status: "augmentation" },
  { id: 8, name: "Fantasia in G", composer: "Telemann", instrument: "Violin", status: "augmentation" },
  { id: 9, name: "Suite No. 4 - Gigue", composer: "Bach", instrument: "Cello Solo", status: "target" },
  { id: 10, name: "Cello Sonata RV 40", composer: "Vivaldi", instrument: "Cello Solo", status: "augmentation" },
]

export function LibraryView() {
  const [filter, setFilter] = useState<string>("all")

  const filteredData = filter === "cello-solo" 
    ? repositoryData.filter(item => item.instrument === "Cello Solo")
    : repositoryData

  return (
    <div className="space-y-6 h-full">
      {/* Global Corpus Analytics - Top Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {summaryCards.map((card) => (
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
            {/* Subtle glow effect */}
            <div 
              className="absolute -bottom-4 -right-4 w-24 h-24 rounded-full opacity-20"
              style={{
                background: "radial-gradient(circle, oklch(0.75 0.15 290) 0%, transparent 70%)",
              }}
            />
          </div>
        ))}
      </div>

      {/* Comparative Visualizations - Middle Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Composer Distribution - Donut Chart */}
        <div 
          className="bg-card rounded-xl p-6 border border-border"
          style={{
            boxShadow: "0 0 20px rgba(168, 130, 255, 0.1), 0 0 40px rgba(168, 130, 255, 0.05)",
            backdropFilter: "blur(10px)",
          }}
        >
          <h3 className="font-serif text-lg font-semibold mb-1">Composer Distribution</h3>
          <p className="text-xs text-muted-foreground mb-4">Dataset composition by composer</p>
          
          <div className="flex items-center gap-6">
            <div className="w-48 h-48 min-w-[192px] min-h-[192px]">
              <ResponsiveContainer width="100%" height="100%" minWidth={150} minHeight={150}>
                <PieChart>
                  <Pie
                    data={composerData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={70}
                    paddingAngle={3}
                    dataKey="value"
                    stroke="none"
                  >
                    {composerData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex-1 space-y-3">
              {composerData.map((composer) => (
                <div key={composer.name} className="flex items-center gap-3">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: composer.color }}
                  />
                  <span className="text-sm flex-1">{composer.name}</span>
                  <span className="text-sm font-semibold text-primary">{composer.value}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Stylistic Density - Scatter Plot */}
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
            <ResponsiveContainer width="100%" height="100%" minWidth={200} minHeight={150}>
              <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                <XAxis 
                  type="number" 
                  dataKey="x" 
                  name="Rhythmic Complexity"
                  domain={[20, 100]}
                  tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                  axisLine={{ stroke: "oklch(0.28 0.04 280)" }}
                  tickLine={false}
                  label={{ 
                    value: "Rhythmic Complexity", 
                    position: "bottom", 
                    fill: "oklch(0.65 0.05 290)",
                    fontSize: 10,
                    offset: 0
                  }}
                />
                <YAxis 
                  type="number" 
                  dataKey="y" 
                  name="Melodic Range"
                  domain={[30, 100]}
                  tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                  axisLine={{ stroke: "oklch(0.28 0.04 280)" }}
                  tickLine={false}
                  label={{ 
                    value: "Melodic Range", 
                    angle: -90, 
                    position: "insideLeft",
                    fill: "oklch(0.65 0.05 290)",
                    fontSize: 10,
                    offset: 10
                  }}
                />
                <ZAxis type="number" dataKey="z" range={[40, 200]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "oklch(0.16 0.025 280)",
                    border: "1px solid oklch(0.28 0.04 280)",
                    borderRadius: "8px",
                    color: "oklch(0.95 0.02 300)",
                  }}
                  labelStyle={{ color: "oklch(0.75 0.15 290)" }}
                  formatter={(value: number, name: string) => {
                    if (name === "x") return [value, "Rhythmic Complexity"]
                    if (name === "y") return [value, "Melodic Range"]
                    return [value, name]
                  }}
                />
                {composerData.map((composer) => (
                  <Scatter
                    key={composer.name}
                    name={composer.name}
                    data={stylisticData.filter(d => d.composer === composer.name)}
                    fill={composer.color}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          
          {/* Legend */}
          <div className="flex flex-wrap gap-4 mt-3 justify-center">
            {composerData.map((composer) => (
              <div key={composer.name} className="flex items-center gap-2">
                <div 
                  className="w-2 h-2 rounded-full" 
                  style={{ backgroundColor: composer.color }}
                />
                <span className="text-xs text-muted-foreground">{composer.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Repository Table - Main Content */}
      <div 
        className="bg-card rounded-xl border border-border overflow-hidden"
        style={{
          boxShadow: "0 0 20px rgba(168, 130, 255, 0.1), 0 0 40px rgba(168, 130, 255, 0.05)",
          backdropFilter: "blur(10px)",
        }}
      >
        {/* Filter Bar */}
        <div className="p-4 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Filter className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm font-medium">Filter:</span>
            <Select value={filter} onValueChange={setFilter}>
              <SelectTrigger className="w-44 bg-background border-border">
                <SelectValue placeholder="Select filter" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Baroque</SelectItem>
                <SelectItem value="cello-solo">Cello Solo Only</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <span className="text-xs text-muted-foreground">
            Showing {filteredData.length} of {repositoryData.length} compositions
          </span>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border bg-background/50">
                <th className="text-left p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Name
                </th>
                <th className="text-left p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Composer
                </th>
                <th className="text-left p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Instrument
                </th>
                <th className="text-left p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Status
                </th>
                <th className="text-right p-4 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredData.map((item) => (
                <tr 
                  key={item.id} 
                  className="border-b border-border/50 hover:bg-primary/5 transition-colors"
                >
                  <td className="p-4">
                    <span className="font-serif font-medium">{item.name}</span>
                  </td>
                  <td className="p-4">
                    <span className="text-sm text-muted-foreground">{item.composer}</span>
                  </td>
                  <td className="p-4">
                    <span className="text-sm text-muted-foreground">{item.instrument}</span>
                  </td>
                  <td className="p-4">
                    <span 
                      className={`inline-flex px-2.5 py-1 rounded-full text-xs font-semibold ${
                        item.status === "target"
                          ? "bg-primary/20 text-primary"
                          : "bg-accent/20 text-accent"
                      }`}
                    >
                      {item.status === "target" ? "Target Style" : "Training Augmentation"}
                    </span>
                  </td>
                  <td className="p-4 text-right">
                    <Button 
                      variant="outline" 
                      size="sm"
                      className="border-primary/50 text-primary hover:bg-primary/10"
                    >
                      <Upload className="w-3 h-3 mr-1.5" />
                      Load into Workbench
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer */}
      <div className="flex justify-between text-sm text-muted-foreground font-serif italic px-2">
        <span>Bach Propagation Engine v2.0</span>
        <span>Baroque Corpus Explorer</span>
      </div>
    </div>
  )
}
