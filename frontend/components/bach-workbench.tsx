"use client"

import { useState } from "react"
import { 
  Play, 
  Square, 
  Undo2, 
  Redo2, 
  Library, 
  FilePlus, 
  BarChart3, 
  Settings, 
  RefreshCw,
  Sparkles
} from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { CompositionView } from "@/components/views/composition-view"
import { AnalysisView } from "@/components/views/analysis-view"
import { LibraryView } from "@/components/views/library-view"
import { SettingsView } from "@/components/views/settings-view"

type ViewType = "Library" | "New Composition" | "Analysis" | "Settings"

export default function BachWorkbench() {
  const [complexity, setComplexity] = useState([75])
  const [counterpoint, setCounterpoint] = useState([60])
  const [ornamentation, setOrnamentation] = useState([42])
  const [activeNav, setActiveNav] = useState<ViewType>("Library")

  const navItems = [
    { name: "Library" as const, icon: Library },
    { name: "New Composition" as const, icon: FilePlus },
    { name: "Analysis" as const, icon: BarChart3 },
    { name: "Settings" as const, icon: Settings },
  ]

  // Get the header title based on active view
  const getHeaderTitle = () => {
    switch (activeNav) {
      case "New Composition":
        return "INVENTION NO. 4 IN D MINOR"
      case "Analysis":
        return "HARMONIC ANALYSIS - BWV 775"
      case "Library":
        return "BAROQUE CORPUS EXPLORER"
      case "Settings":
        return "WORKBENCH SETTINGS"
      default:
        return ""
    }
  }

  // Render the main content based on active view
  const renderMainContent = () => {
    switch (activeNav) {
      case "New Composition":
        return <CompositionView />
      case "Analysis":
        return <AnalysisView />
      case "Library":
        return <LibraryView />
      case "Settings":
        return <SettingsView />
      default:
        return null
    }
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      {/* Sidebar */}
      <aside className="w-64 bg-sidebar border-r border-sidebar-border flex flex-col">
        {/* Logo */}
        <div className="p-4 flex items-center gap-3">
          <div 
            className="w-10 h-10 rounded-lg flex items-center justify-center overflow-hidden"
            style={{ border: "1px solid rgba(167, 139, 250, 0.2)" }}
          >
            <img 
              key="logo-motif-ai"
              src={"/logo-motif-ai.png"} 
              alt="Bach Propagation Logo"
              className="w-10 h-10 object-contain"
              style={{
                filter: "brightness(1.2) contrast(1.1) drop-shadow(0 0 8px rgba(168, 130, 255, 0.6)) drop-shadow(0 0 16px rgba(168, 130, 255, 0.4))"
              }}
            />
          </div>
          <div>
            <h1 className="font-serif text-lg font-semibold tracking-tight">Bach Propagation</h1>
            <p className="text-xs text-muted-foreground">Workbench v2.4</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4">
          {navItems.map((item) => (
            <button
              key={item.name}
              onClick={() => setActiveNav(item.name)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-colors font-serif ${
                activeNav === item.name
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
              }`}
            >
              <item.icon className="w-5 h-5" />
              <span>{item.name}</span>
            </button>
          ))}
        </nav>

        {/* Neural Engine */}
        <div className="p-4">
          <div className="bg-card rounded-lg p-4 border border-border">
            <div className="h-16 mb-3 relative">
              {/* Waveform visualization */}
              <svg className="w-full h-full" viewBox="0 0 200 60">
                <path
                  d="M0,30 Q20,10 40,30 T80,30 T120,30 T160,30 T200,30"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-primary/40"
                />
                <path
                  d="M0,30 Q25,45 50,30 T100,30 T150,30"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-accent/60"
                />
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

      {/* Main Content */}
      <main className="flex-1 flex flex-col">
        {/* Toolbar */}
        <header className="h-14 border-b border-border flex items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
              <Play className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
              <Square className="w-4 h-4" />
            </Button>
            <div className="w-px h-6 bg-border mx-2" />
            <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
              <Undo2 className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
              <Redo2 className="w-5 h-5" />
            </Button>
          </div>

          <div className="flex items-center gap-4">
            <h2 className="font-mono text-sm tracking-widest text-muted-foreground">
              {getHeaderTitle()}
            </h2>
            <span className="px-3 py-1 rounded-full border border-primary text-primary text-xs font-semibold">
              AI ASSISTED
            </span>
          </div>
        </header>

        {/* Dynamic Content Area */}
        <div className="flex-1 p-6 overflow-auto">
          {renderMainContent()}
        </div>

        {/* Status Bar */}
        <footer className="h-8 border-t border-border flex items-center justify-between px-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />
              AI STATUS: <span className="text-accent font-semibold">READY</span>
            </span>
            <span>CURRENT MEASURE: 12 / 32</span>
          </div>
          <div className="flex items-center gap-4">
            <span>LATENCY: 14MS</span>
            <span className="text-primary font-semibold">PROPAGATOR V2.4.1</span>
          </div>
        </footer>
      </main>

      {/* Right Panel - AI Control */}
      <aside className="w-80 border-l border-border bg-background p-6 flex flex-col">
        {/* Header */}
        <div className="flex items-center gap-3 mb-2">
          <Sparkles className="w-6 h-6 text-primary" />
          <h2 className="font-serif text-xl font-semibold">AI Control Panel</h2>
        </div>
        <p className="text-sm text-muted-foreground mb-6">Generative Counterpoint Settings</p>

        {/* Key & Time Selectors */}
        <div className="grid grid-cols-2 gap-4 mb-8">
          <div>
            <label className="text-xs text-muted-foreground uppercase tracking-wider mb-2 block">
              Key
            </label>
            <Select defaultValue="d-minor">
              <SelectTrigger className="bg-card border-border">
                <SelectValue placeholder="Select key" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="c-major">C Major</SelectItem>
                <SelectItem value="d-minor">D Minor</SelectItem>
                <SelectItem value="g-major">G Major</SelectItem>
                <SelectItem value="a-minor">A Minor</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="text-xs text-muted-foreground uppercase tracking-wider mb-2 block">
              Time
            </label>
            <Select defaultValue="3-8">
              <SelectTrigger className="bg-card border-border">
                <SelectValue placeholder="Select time" />
              </SelectTrigger>
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
        <div className="space-y-6 mb-8">
          <div>
            <div className="flex justify-between items-center mb-3">
              <label className="text-sm font-medium">Complexity</label>
              <span className="text-sm text-primary font-semibold">{complexity[0]}%</span>
            </div>
            <Slider
              value={complexity}
              onValueChange={setComplexity}
              max={100}
              step={1}
              className="[&_[role=slider]]:bg-primary [&_[role=slider]]:border-primary"
            />
          </div>

          <div>
            <div className="flex justify-between items-center mb-3">
              <label className="text-sm font-medium">Counterpoint Density</label>
              <span className="text-sm text-primary font-semibold">{counterpoint[0]}%</span>
            </div>
            <Slider
              value={counterpoint}
              onValueChange={setCounterpoint}
              max={100}
              step={1}
              className="[&_[role=slider]]:bg-primary [&_[role=slider]]:border-primary"
            />
          </div>

          <div>
            <div className="flex justify-between items-center mb-3">
              <label className="text-sm font-medium">Ornamentation</label>
              <span className="text-sm text-primary font-semibold">{ornamentation[0]}%</span>
            </div>
            <Slider
              value={ornamentation}
              onValueChange={setOrnamentation}
              max={100}
              step={1}
              className="[&_[role=slider]]:bg-primary [&_[role=slider]]:border-primary"
            />
          </div>
        </div>

        {/* Propagator Insight */}
        <div className="bg-card rounded-lg p-4 border border-border mb-8">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-4 h-4 text-accent" />
            <span className="text-sm font-semibold text-accent uppercase tracking-wider">
              Propagator Insight
            </span>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed mb-4">
            The current melodic curve in Bar 12 suggests a higher likelihood of an appoggiatura 
            resolution in the soprano voice.
          </p>
          <Button 
            variant="outline" 
            className="w-full border-primary/50 text-primary hover:bg-primary/10"
          >
            Apply Suggestion
          </Button>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Regenerate Button */}
        <Button className="w-full bg-primary hover:bg-primary/90 text-primary-foreground py-6 text-base font-semibold">
          <RefreshCw className="w-5 h-5 mr-2" />
          Regenerate Phrases
        </Button>
      </aside>
    </div>
  )
}
