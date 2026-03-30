"use client"

import { useState, useEffect } from "react"
import { Globe, Music, Activity, Cpu, Palette, Server, Info, Layers, Brain, Zap } from "lucide-react"
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"

export function SettingsView() {
  const [language, setLanguage] = useState("en")
  const [tuning, setTuning] = useState("standard")
  const [synthVoice, setSynthVoice] = useState("legato")
  const [endpoint, setEndpoint] = useState("http://localhost:8000")
  const [glowIntensity, setGlowIntensity] = useState([70])
  const [isConnected, setIsConnected] = useState(true)
  
  // Simulated real-time metrics
  const [latency, setLatency] = useState(14)
  const [vramUsage, setVramUsage] = useState(68)
  const [trustFactor, setTrustFactor] = useState(87)
  const [sparklineData, setSparklineData] = useState([12, 15, 14, 16, 13, 14, 15, 12, 14, 13, 15, 14])

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setLatency(prev => Math.max(8, Math.min(25, prev + (Math.random() - 0.5) * 4)))
      setVramUsage(prev => Math.max(60, Math.min(85, prev + (Math.random() - 0.5) * 3)))
      setTrustFactor(prev => Math.max(75, Math.min(95, prev + (Math.random() - 0.5) * 2)))
      setSparklineData(prev => [...prev.slice(1), Math.round(8 + Math.random() * 12)])
    }, 1500)
    return () => clearInterval(interval)
  }, [])

  // Calculate circular gauge path
  const gaugeRadius = 36
  const gaugeCircumference = 2 * Math.PI * gaugeRadius
  const gaugeProgress = (trustFactor / 100) * gaugeCircumference

  return (
    <div className="space-y-6 h-full overflow-y-auto pb-8">
      {/* Connection Status Banner */}
      <div 
        className="flex items-center justify-between px-4 py-3 rounded-xl border"
        style={{
          backgroundColor: isConnected ? "rgba(34, 197, 94, 0.05)" : "rgba(239, 68, 68, 0.05)",
          borderColor: isConnected ? "rgba(34, 197, 94, 0.2)" : "rgba(239, 68, 68, 0.2)",
          boxShadow: isConnected 
            ? "0 0 20px rgba(34, 197, 94, 0.1)" 
            : "0 0 20px rgba(239, 68, 68, 0.1)",
        }}
      >
        <div className="flex items-center gap-3">
          <div 
            className="w-3 h-3 rounded-full animate-pulse"
            style={{
              backgroundColor: isConnected ? "#22c55e" : "#ef4444",
              boxShadow: isConnected 
                ? "0 0 8px rgba(34, 197, 94, 0.8), 0 0 16px rgba(34, 197, 94, 0.4)" 
                : "0 0 8px rgba(239, 68, 68, 0.8), 0 0 16px rgba(239, 68, 68, 0.4)",
            }}
          />
          <span className="font-mono text-sm" style={{ color: isConnected ? "#22c55e" : "#ef4444" }}>
            {isConnected ? "Connected to FastAPI" : "Backend Offline"}
          </span>
        </div>
        <button
          onClick={() => setIsConnected(!isConnected)}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors font-mono"
        >
          [Toggle Demo]
        </button>
      </div>

      {/* System Health & AI Observability */}
      <div 
        className="bg-card rounded-xl p-6 border border-border"
        style={{
          boxShadow: "0 0 20px rgba(168, 130, 255, 0.1), 0 0 40px rgba(168, 130, 255, 0.05)",
        }}
      >
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-primary/10">
            <Activity className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-serif text-lg font-semibold">System Health & AI Observability</h3>
            <p className="text-xs text-muted-foreground font-mono">Orchestrator Model Monitoring</p>
          </div>
        </div>

        {/* Model Health Dashboard - 3 Mini Monitors */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          {/* Inference Latency */}
          <div 
            className="bg-background/50 rounded-lg p-4 border border-border/50"
            style={{
              boxShadow: "inset 0 1px 0 rgba(168, 130, 255, 0.1)",
            }}
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-muted-foreground font-mono uppercase tracking-wider">Inference Latency</span>
              <Zap className="w-4 h-4 text-primary" />
            </div>
            
            {/* Sparkline visualization */}
            <div className="h-8 flex items-end gap-0.5 mb-2">
              {sparklineData.map((value, i) => (
                <div
                  key={i}
                  className="flex-1 rounded-t transition-all duration-300"
                  style={{
                    height: `${(value / 25) * 100}%`,
                    backgroundColor: i === sparklineData.length - 1 
                      ? "rgba(168, 130, 255, 1)" 
                      : "rgba(168, 130, 255, 0.4)",
                    boxShadow: i === sparklineData.length - 1 
                      ? "0 0 8px rgba(168, 130, 255, 0.6)" 
                      : "none",
                  }}
                />
              ))}
            </div>
            
            <div className="flex items-baseline gap-1">
              <span 
                className="text-2xl font-bold font-mono text-primary"
                style={{ textShadow: "0 0 10px rgba(168, 130, 255, 0.5)" }}
              >
                {Math.round(latency)}
              </span>
              <span className="text-xs text-muted-foreground font-mono">ms</span>
            </div>
          </div>

          {/* VRAM Usage */}
          <div 
            className="bg-background/50 rounded-lg p-4 border border-border/50"
            style={{
              boxShadow: "inset 0 1px 0 rgba(168, 130, 255, 0.1)",
            }}
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-muted-foreground font-mono uppercase tracking-wider">VRAM Usage</span>
              <Cpu className="w-4 h-4 text-primary" />
            </div>
            
            {/* Progress bar */}
            <div className="h-3 bg-muted rounded-full overflow-hidden mb-3">
              <div 
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${vramUsage}%`,
                  background: "linear-gradient(90deg, rgba(168, 130, 255, 0.6), rgba(168, 130, 255, 1))",
                  boxShadow: "0 0 10px rgba(168, 130, 255, 0.5)",
                }}
              />
            </div>
            
            <div className="flex items-baseline justify-between">
              <div className="flex items-baseline gap-1">
                <span 
                  className="text-2xl font-bold font-mono text-primary"
                  style={{ textShadow: "0 0 10px rgba(168, 130, 255, 0.5)" }}
                >
                  {Math.round(vramUsage)}
                </span>
                <span className="text-xs text-muted-foreground font-mono">%</span>
              </div>
              <span className="text-xs text-muted-foreground font-mono">8.2 / 12 GB</span>
            </div>
          </div>

          {/* Model Trust Factor */}
          <div 
            className="bg-background/50 rounded-lg p-4 border border-border/50"
            style={{
              boxShadow: "inset 0 1px 0 rgba(168, 130, 255, 0.1)",
            }}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-muted-foreground font-mono uppercase tracking-wider">Trust Factor</span>
              <Brain className="w-4 h-4 text-primary" />
            </div>
            
            {/* Circular gauge */}
            <div className="flex items-center justify-center">
              <div className="relative w-20 h-20">
                <svg className="w-full h-full -rotate-90" viewBox="0 0 80 80">
                  {/* Background circle */}
                  <circle
                    cx="40"
                    cy="40"
                    r={gaugeRadius}
                    fill="none"
                    stroke="rgba(168, 130, 255, 0.15)"
                    strokeWidth="6"
                  />
                  {/* Progress circle */}
                  <circle
                    cx="40"
                    cy="40"
                    r={gaugeRadius}
                    fill="none"
                    stroke="rgba(168, 130, 255, 1)"
                    strokeWidth="6"
                    strokeLinecap="round"
                    strokeDasharray={gaugeCircumference}
                    strokeDashoffset={gaugeCircumference - gaugeProgress}
                    style={{
                      filter: "drop-shadow(0 0 6px rgba(168, 130, 255, 0.6))",
                      transition: "stroke-dashoffset 0.5s ease-out",
                    }}
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span 
                    className="text-lg font-bold font-mono text-primary"
                    style={{ textShadow: "0 0 10px rgba(168, 130, 255, 0.5)" }}
                  >
                    {Math.round(trustFactor)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Orchestrator Details */}
        <div className="space-y-4">
          {/* Active Checkpoint */}
          <div>
            <label className="text-xs text-muted-foreground font-mono uppercase tracking-wider mb-2 block">
              Active Checkpoint
            </label>
            <div 
              className="bg-background rounded-lg p-3 border border-border/50 font-mono text-sm"
              style={{
                background: "linear-gradient(135deg, rgba(0,0,0,0.4) 0%, rgba(20,10,30,0.6) 100%)",
                boxShadow: "inset 0 0 20px rgba(0,0,0,0.3), 0 0 1px rgba(168, 130, 255, 0.3)",
              }}
            >
              <span className="text-muted-foreground">$ </span>
              <span className="text-primary" style={{ textShadow: "0 0 8px rgba(168, 130, 255, 0.4)" }}>
                bach-transformer-v2-suite-focus.ckpt
              </span>
            </div>
          </div>

          {/* Architecture Info */}
          <div>
            <label className="text-xs text-muted-foreground font-mono uppercase tracking-wider mb-2 block">
              Architecture Info
            </label>
            <div className="flex gap-3">
              {[
                { label: "Layers", value: "12" },
                { label: "Heads", value: "8" },
                { label: "Context Window", value: "2048" },
              ].map((item) => (
                <div 
                  key={item.label}
                  className="px-3 py-2 rounded-lg border border-border/50 bg-background/30"
                >
                  <span className="text-xs text-muted-foreground font-mono">{item.label}: </span>
                  <span 
                    className="text-sm font-bold font-mono text-primary"
                    style={{ textShadow: "0 0 6px rgba(168, 130, 255, 0.3)" }}
                  >
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Language & Localization */}
      <div 
        className="bg-card rounded-xl p-6 border border-border"
        style={{
          boxShadow: "0 0 20px rgba(168, 130, 255, 0.1), 0 0 40px rgba(168, 130, 255, 0.05)",
        }}
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <Globe className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-serif text-lg font-semibold">Language & Localization</h3>
            <p className="text-xs text-muted-foreground">Set your preferred language</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <label className="text-sm text-muted-foreground min-w-32">Display Language</label>
          <Select value={language} onValueChange={setLanguage}>
            <SelectTrigger className="w-48 bg-background border-border">
              <SelectValue placeholder="Select language" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="en">English</SelectItem>
              <SelectItem value="es">Espa&ntilde;ol</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Audio & Tuning */}
      <div 
        className="bg-card rounded-xl p-6 border border-border"
        style={{
          boxShadow: "0 0 20px rgba(168, 130, 255, 0.1), 0 0 40px rgba(168, 130, 255, 0.05)",
        }}
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <Music className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-serif text-lg font-semibold">Audio & Tuning</h3>
            <p className="text-xs text-muted-foreground">The Baroque Experience</p>
          </div>
        </div>
        
        <div className="space-y-6">
          {/* Master Tuning */}
          <div>
            <div className="flex items-center gap-4 mb-2">
              <label className="text-sm text-muted-foreground min-w-32">Master Tuning</label>
              <div className="flex rounded-lg overflow-hidden border border-border bg-background">
                <button
                  onClick={() => setTuning("standard")}
                  className={`px-4 py-2 text-sm font-medium transition-colors ${
                    tuning === "standard"
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  }`}
                >
                  Standard (A=440Hz)
                </button>
                <button
                  onClick={() => setTuning("baroque")}
                  className={`px-4 py-2 text-sm font-medium transition-colors border-l border-border ${
                    tuning === "baroque"
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  }`}
                >
                  Baroque (A=415Hz)
                </button>
              </div>
            </div>
            <div className="ml-36 flex items-start gap-2">
              <Info className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
              <p className="text-xs text-muted-foreground italic leading-relaxed">
                Baroque tuning sounds approximately a semi-tone lower (e.g., D Minor will sound like C# Minor).
              </p>
            </div>
          </div>

          {/* Synth Voice */}
          <div className="flex items-center gap-4">
            <label className="text-sm text-muted-foreground min-w-32">Synth Voice</label>
            <Select value={synthVoice} onValueChange={setSynthVoice}>
              <SelectTrigger className="w-56 bg-background border-border">
                <SelectValue placeholder="Select synth voice" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="legato">Legato Cello</SelectItem>
                <SelectItem value="staccato">Staccato Ensemble</SelectItem>
                <SelectItem value="sine">Sine Wave Debug</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* AI & Backend Configuration */}
      <div 
        className="bg-card rounded-xl p-6 border border-border"
        style={{
          boxShadow: "0 0 20px rgba(168, 130, 255, 0.1), 0 0 40px rgba(168, 130, 255, 0.05)",
        }}
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <Server className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-serif text-lg font-semibold">AI & Backend Configuration</h3>
            <p className="text-xs text-muted-foreground">Server and model settings</p>
          </div>
        </div>
        
        <div className="space-y-5">
          {/* Endpoint URL */}
          <div className="flex items-center gap-4">
            <label className="text-sm text-muted-foreground min-w-32">Endpoint URL</label>
            <Input
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)}
              className="flex-1 max-w-md bg-background border-border font-mono text-sm"
              placeholder="http://localhost:8000"
            />
          </div>

          {/* Model Info */}
          <div className="flex items-center gap-4">
            <label className="text-sm text-muted-foreground min-w-32">Model Info</label>
            <div 
              className="px-4 py-2 rounded-lg border border-primary/30 bg-primary/5"
              style={{
                boxShadow: "0 0 10px rgba(168, 130, 255, 0.15)",
              }}
            >
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-primary" />
                <span className="text-sm font-semibold text-primary">
                  Bach-Propagation Transformer v2.4
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Interface Preferences */}
      <div 
        className="bg-card rounded-xl p-6 border border-border"
        style={{
          boxShadow: "0 0 20px rgba(168, 130, 255, 0.1), 0 0 40px rgba(168, 130, 255, 0.05)",
        }}
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <Palette className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-serif text-lg font-semibold">Interface Preferences</h3>
            <p className="text-xs text-muted-foreground">Customize your experience</p>
          </div>
        </div>
        
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <label className="text-sm text-muted-foreground min-w-32">Neon Glow Intensity</label>
            <div className="flex-1 max-w-md flex items-center gap-4">
              <Slider
                value={glowIntensity}
                onValueChange={setGlowIntensity}
                max={100}
                step={1}
                className="flex-1 [&_[role=slider]]:bg-primary [&_[role=slider]]:border-primary"
              />
              <span className="text-sm font-semibold text-primary w-12 text-right">
                {glowIntensity[0]}%
              </span>
            </div>
          </div>
          
          {/* Preview of glow effect */}
          <div className="ml-36">
            <div 
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-primary/50 bg-primary/10"
              style={{
                boxShadow: `0 0 ${Math.round(glowIntensity[0] * 0.3)}px rgba(168, 130, 255, ${glowIntensity[0] / 100 * 0.5}), 0 0 ${Math.round(glowIntensity[0] * 0.6)}px rgba(168, 130, 255, ${glowIntensity[0] / 100 * 0.3})`,
              }}
            >
              <span className="text-xs text-muted-foreground">Glow Preview</span>
            </div>
          </div>
        </div>
      </div>

      {/* Professional Branding Footer */}
      <div className="flex items-center justify-center gap-3 py-6 border-t border-border/50">
        <div 
          className="w-8 h-8 rounded-lg flex items-center justify-center overflow-hidden"
          style={{ border: "1px solid rgba(167, 139, 250, 0.2)" }}
        >
          <img 
            src={"/logo-motif-ai.png"} 
            alt="Motif AI Logo"
            className="w-8 h-8 object-contain"
            style={{
              filter: "brightness(1.2) contrast(1.1) drop-shadow(0 0 6px rgba(168, 130, 255, 0.5))"
            }}
          />
        </div>
        <span className="text-sm text-muted-foreground font-serif">
          Powered by <span className="text-primary font-semibold">Motif AI</span>
        </span>
      </div>
    </div>
  )
}
