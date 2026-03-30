"use client"

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from "recharts"
import { Music, Repeat, TrendingUp, Waves } from "lucide-react"

// Harmonic distribution data for radar chart
const harmonicData = [
  { harmony: "I", value: 85, fullMark: 100 },
  { harmony: "ii", value: 45, fullMark: 100 },
  { harmony: "iii", value: 30, fullMark: 100 },
  { harmony: "IV", value: 70, fullMark: 100 },
  { harmony: "V", value: 90, fullMark: 100 },
  { harmony: "vi", value: 55, fullMark: 100 },
  { harmony: "vii°", value: 25, fullMark: 100 },
]

// Bar chart data for chord progression frequency
const chordFrequencyData = [
  { name: "i", frequency: 24 },
  { name: "iv", frequency: 18 },
  { name: "V", frequency: 21 },
  { name: "VI", frequency: 12 },
  { name: "III", frequency: 9 },
  { name: "VII", frequency: 8 },
  { name: "ii°", frequency: 5 },
]

// Pattern recognition motifs
const patterns = [
  {
    id: 1,
    name: "Ascending Sequence",
    occurrences: 12,
    bars: "1-4, 9-12, 17-20",
    icon: TrendingUp,
    color: "text-primary",
  },
  {
    id: 2,
    name: "Counterpoint Inversion",
    occurrences: 8,
    bars: "5-6, 13-14, 21-22",
    icon: Repeat,
    color: "text-accent",
  },
  {
    id: 3,
    name: "Mordent Figure",
    occurrences: 15,
    bars: "Throughout",
    icon: Waves,
    color: "text-primary",
  },
  {
    id: 4,
    name: "Cadential Pattern",
    occurrences: 4,
    bars: "8, 16, 24, 32",
    icon: Music,
    color: "text-accent",
  },
]

// Neon purple gradient colors for bars
const barColors = [
  "oklch(0.75 0.15 290)",
  "oklch(0.70 0.12 300)",
  "oklch(0.78 0.12 280)",
  "oklch(0.65 0.18 310)",
  "oklch(0.80 0.10 270)",
  "oklch(0.72 0.14 295)",
  "oklch(0.68 0.16 285)",
]

export function AnalysisView() {
  return (
    <div className="space-y-6 h-full">
      {/* Harmonic Distribution Section */}
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="font-serif text-xl font-semibold mb-1">Harmonic Distribution</h3>
        <p className="text-sm text-muted-foreground mb-6">
          Analysis of chord usage frequency in BWV 775
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Radar Chart */}
          <div className="bg-background/50 rounded-lg p-4 border border-border/50">
            <h4 className="text-sm font-medium text-muted-foreground mb-4 uppercase tracking-wider">
              Harmonic Balance
            </h4>
            <div className="h-64 min-h-[256px]">
              <ResponsiveContainer width="100%" height="100%" minWidth={200} minHeight={200}>
                <RadarChart data={harmonicData}>
                  <PolarGrid 
                    stroke="oklch(0.28 0.04 280)" 
                    strokeDasharray="3 3"
                  />
                  <PolarAngleAxis 
                    dataKey="harmony" 
                    tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 12 }}
                  />
                  <PolarRadiusAxis 
                    angle={90} 
                    domain={[0, 100]}
                    tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                    axisLine={false}
                  />
                  <Radar
                    name="Frequency"
                    dataKey="value"
                    stroke="oklch(0.75 0.15 290)"
                    fill="oklch(0.75 0.15 290)"
                    fillOpacity={0.3}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Bar Chart */}
          <div className="bg-background/50 rounded-lg p-4 border border-border/50">
            <h4 className="text-sm font-medium text-muted-foreground mb-4 uppercase tracking-wider">
              Chord Progression Frequency
            </h4>
            <div className="h-64 min-h-[256px]">
              <ResponsiveContainer width="100%" height="100%" minWidth={200} minHeight={200}>
                <BarChart data={chordFrequencyData} barCategoryGap="20%">
                  <XAxis 
                    dataKey="name" 
                    tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 12 }}
                    axisLine={{ stroke: "oklch(0.28 0.04 280)" }}
                    tickLine={false}
                  />
                  <YAxis 
                    tick={{ fill: "oklch(0.65 0.05 290)", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "oklch(0.16 0.025 280)",
                      border: "1px solid oklch(0.28 0.04 280)",
                      borderRadius: "8px",
                      color: "oklch(0.95 0.02 300)",
                    }}
                    labelStyle={{ color: "oklch(0.75 0.15 290)" }}
                    cursor={{ fill: "oklch(0.25 0.05 290)", opacity: 0.3 }}
                  />
                  <Bar dataKey="frequency" radius={[4, 4, 0, 0]}>
                    {chordFrequencyData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={barColors[index % barColors.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Pattern Recognition Section */}
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="font-serif text-xl font-semibold mb-1">Pattern Recognition</h3>
        <p className="text-sm text-muted-foreground mb-6">
          Recurring musical motifs detected in the composition
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {patterns.map((pattern) => (
            <div
              key={pattern.id}
              className="bg-background/50 rounded-lg p-4 border border-border/50 hover:border-primary/50 transition-colors"
            >
              <div className="flex items-start gap-3">
                <div className={`p-2 rounded-lg bg-primary/10 ${pattern.color}`}>
                  <pattern.icon className="w-5 h-5" />
                </div>
                <div className="flex-1">
                  <h4 className="font-medium text-foreground">{pattern.name}</h4>
                  <p className="text-xs text-muted-foreground mt-1">
                    Bars: {pattern.bars}
                  </p>
                  <div className="flex items-center gap-2 mt-2">
                    <span className="text-xs px-2 py-0.5 rounded-full bg-primary/20 text-primary font-medium">
                      {pattern.occurrences} occurrences
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="flex justify-between text-sm text-muted-foreground font-serif italic px-2">
        <span>Bach Propagation Engine v2.0</span>
        <span>Analysis Report</span>
      </div>
    </div>
  )
}
