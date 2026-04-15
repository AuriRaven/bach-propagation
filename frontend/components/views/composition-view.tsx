"use client"

/**
 * frontend/components/views/composition-view.tsx
 *
 * Piano-roll renderer with real-time note highlighting.
 * Progress scrubber lives in the workbench toolbar (bach-workbench.tsx).
 *
 * Rendering uses requestAnimationFrame-driven redraw so the playhead
 * stays smooth even at 60fps. Beat tracking uses transport.seconds
 * (monotonic) converted to quarter beats — no BBT string parsing.
 */

import { useEffect, useRef, useCallback, useState } from "react"
import { RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useAppState, type VexFlowPayload, type VexFlowNote } from "@/lib/app-state"
import { api } from "@/lib/api-client"

// ─── Music helpers ────────────────────────────────────────────────────────────

function pitchToMidi(pitch: string): number {
  const match = pitch.match(/^([A-G])(#{1,2}|b+|-+)?(-?\d+)$/)
  if (!match) return 60
  const [, name, acc = "", octave] = match
  const noteMap: Record<string, number> = { C:0,D:2,E:4,F:5,G:7,A:9,B:11 }
  let semi = noteMap[name] ?? 0
  for (const ch of acc) {
    if (ch === "#") semi++
    else if (ch === "b" || ch === "-") semi--
  }
  return (parseInt(octave) + 1) * 12 + semi
}

function notePitches(note: VexFlowNote): number[] {
  if (note.type === "rest") return []
  if (note.pitches) return note.pitches.map(pitchToMidi)
  if (note.pitch)   return [pitchToMidi(note.pitch)]
  return []
}

function parseBeat(s: string): number {
  if (s.includes("/")) {
    const [n, d] = s.split("/").map(Number)
    return d ? n / d : n
  }
  return parseFloat(s) || 0
}

// ─── Flat note list ───────────────────────────────────────────────────────────

interface FlatNote { midi: number; startBeat: number; endBeat: number }

function buildFlatNotes(payload: VexFlowPayload): FlatNote[] {
  const flat: FlatNote[] = []
  for (const measure of payload.measures) {
    for (const note of measure.notes) {
      if (note.type === "rest") continue
      const startBeat = parseBeat(note.offset)
      const dur       = parseBeat(note.duration)
      const endBeat   = startBeat + Math.max(dur, 0.125)
      for (const midi of notePitches(note)) {
        flat.push({ midi, startBeat, endBeat })
      }
    }
  }
  return flat
}

// ─── Canvas renderer ──────────────────────────────────────────────────────────

function renderPianoRoll(
  canvas: HTMLCanvasElement,
  payload: VexFlowPayload,
  flatNotes: FlatNote[],
  cursorBeat: number,
) {
  const ctx = canvas.getContext("2d")
  if (!ctx) return

  const W = canvas.width
  const H = canvas.height

  const midis   = flatNotes.map((n) => n.midi)
  const minMidi = Math.max(21,  Math.min(...midis, 60) - 3)
  const maxMidi = Math.min(108, Math.max(...midis, 72) + 3)
  const pitchRange = maxMidi - minMidi + 1

  const totalBeats      = Math.max(payload.total_beats, 1)
  const beatsPerMeasure = parseInt(payload.time_signature?.split("/")[0] ?? "4")

  const LABEL_W = 34, PAD_R = 12, PAD_T = 8, PAD_B = 22
  const rollW = W - LABEL_W - PAD_R
  const rollH = H - PAD_T - PAD_B
  const beatW  = rollW / totalBeats
  const pitchH = rollH / pitchRange

  const bx = (beat: number) => LABEL_W + beat * beatW
  const py = (midi: number) => PAD_T + (maxMidi - midi) * pitchH

  ctx.fillStyle = "#0e0d14"
  ctx.fillRect(0, 0, W, H)

  // Pitch rows
  for (let m = minMidi; m <= maxMidi; m++) {
    const note = m % 12
    if ([1,3,6,8,10].includes(note)) {
      ctx.fillStyle = "rgba(0,0,0,0.22)"
      ctx.fillRect(LABEL_W, py(m), rollW, pitchH)
    }
    if (note === 0) {
      ctx.strokeStyle = "rgba(168,130,255,0.20)"
      ctx.lineWidth = 0.8
      ctx.beginPath(); ctx.moveTo(LABEL_W, py(m)); ctx.lineTo(W - PAD_R, py(m)); ctx.stroke()
    }
  }

  // Measure lines + numbers
  const totalMeasures = Math.ceil(totalBeats / beatsPerMeasure)
  const labelEvery    = Math.max(1, Math.floor(totalMeasures / 16))
  ctx.font = "9px monospace"; ctx.textAlign = "center"

  for (let m = 0; m <= totalMeasures; m++) {
    const x = bx(m * beatsPerMeasure)
    ctx.strokeStyle = m === 0 ? "rgba(168,130,255,0.35)" : "rgba(168,130,255,0.14)"
    ctx.lineWidth   = m === 0 ? 1.5 : 0.5
    ctx.beginPath(); ctx.moveTo(x, PAD_T); ctx.lineTo(x, H - PAD_B); ctx.stroke()
    if (m < totalMeasures && m % labelEvery === 0) {
      ctx.fillStyle = "rgba(168,130,255,0.4)"
      ctx.fillText(String(m + 1), x + (beatW * beatsPerMeasure) / 2, H - 5)
    }
  }

  // Beat sub-lines
  ctx.strokeStyle = "rgba(168,130,255,0.06)"; ctx.lineWidth = 0.3
  for (let b = 0; b <= totalBeats; b++) {
    if (b % beatsPerMeasure !== 0) {
      const x = bx(b)
      ctx.beginPath(); ctx.moveTo(x, PAD_T); ctx.lineTo(x, H - PAD_B); ctx.stroke()
    }
  }

  // Notes
  const N_PAD = 1.5, MIN_W = 3, RADIUS = 2
  for (const note of flatNotes) {
    if (note.midi < minMidi || note.midi > maxMidi) continue
    const x = bx(note.startBeat)
    const y = py(note.midi) + N_PAD
    const w = Math.max(MIN_W, (note.endBeat - note.startBeat) * beatW - N_PAD * 2)
    const h = Math.max(3, pitchH - N_PAD * 2)
    const isActive = cursorBeat >= note.startBeat && cursorBeat < note.endBeat
    const isPast   = cursorBeat >= note.endBeat

    ctx.beginPath(); ctx.roundRect(x, y, w, h, RADIUS)
    if (isActive) {
      ctx.shadowColor = "rgba(200,160,255,0.6)"; ctx.shadowBlur = 12
      ctx.fillStyle   = "rgba(185,145,255,1.0)"; ctx.fill()
      ctx.shadowBlur  = 0
      ctx.fillStyle   = "rgba(255,248,255,0.45)"
      ctx.fillRect(x + 1, y + 1, Math.min(w - 2, 8), Math.max(1, h * 0.4))
    } else if (isPast) {
      ctx.fillStyle = "rgba(80,55,140,0.32)"; ctx.fill()
    } else {
      ctx.fillStyle = "rgba(120,88,198,0.58)"; ctx.fill()
    }
  }

  // Pitch labels
  ctx.font = "9px monospace"; ctx.textAlign = "right"
  for (let m = minMidi; m <= maxMidi; m++) {
    if (m % 12 !== 0) continue
    const octave = Math.floor(m / 12) - 1
    const hasActive = flatNotes.some(
      (n) => n.midi === m && cursorBeat >= n.startBeat && cursorBeat < n.endBeat
    )
    ctx.fillStyle = hasActive ? "rgba(220,200,255,0.95)" : "rgba(168,130,255,0.45)"
    ctx.fillText(`C${octave}`, LABEL_W - 4, py(m) + pitchH / 2 + 3)
  }

  // Playhead
  const headX = bx(cursorBeat)
  if (headX >= LABEL_W && headX <= W - PAD_R) {
    ctx.shadowColor = "rgba(220,180,255,0.7)"; ctx.shadowBlur = 8
    ctx.strokeStyle = "rgba(220,180,255,0.9)"; ctx.lineWidth = 1.5
    ctx.setLineDash([4, 3])
    ctx.beginPath(); ctx.moveTo(headX, PAD_T); ctx.lineTo(headX, H - PAD_B); ctx.stroke()
    ctx.setLineDash([]); ctx.shadowBlur = 0
    ctx.fillStyle = "rgba(220,180,255,0.9)"
    ctx.beginPath()
    ctx.moveTo(headX - 5, PAD_T); ctx.lineTo(headX + 5, PAD_T); ctx.lineTo(headX, PAD_T + 9)
    ctx.closePath(); ctx.fill()
  }
}

// ─── Empty state ──────────────────────────────────────────────────────────────

function EmptyScore() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 text-muted-foreground">
      <svg className="w-72 h-28 opacity-20" viewBox="0 0 288 100">
        {[18,32,46,60,74].map((y) => (
          <line key={y} x1="48" y1={y} x2="268" y2={y} stroke="currentColor" strokeWidth="1" />
        ))}
        <text x="12" y="62" fontSize="52" fontFamily="serif" fill="currentColor">𝄞</text>
        {[[80,55],[100,42],[120,48],[140,38],[160,52],[180,44],[200,50],[220,40],[240,46],[260,54]].map(([x,y],i) => (
          <circle key={i} cx={x} cy={y} r="4" fill="currentColor" opacity={0.2 + i * 0.07} />
        ))}
      </svg>
      <p className="text-sm font-serif italic text-center max-w-xs leading-relaxed">
        Select a piece from the <strong>Library</strong> and click{" "}
        <strong>Load into Workbench</strong>
      </p>
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export function CompositionView() {
  const { activeScore, notationData, playbackPosition, setNotationData } = useAppState()
  const canvasRef    = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const flatRef      = useRef<FlatNote[]>([])
  const [refreshing, setRefreshing] = useState(false)

  const handleRefresh = useCallback(async () => {
    if (!activeScore) return
    setRefreshing(true)
    setNotationData(null)
    try {
      const notation = await api.corpus.notation(activeScore.id, true)
      setNotationData(notation)
    } catch (err) {
      console.error("[CompositionView] refresh failed:", err)
    } finally {
      setRefreshing(false)
    }
  }, [activeScore, setNotationData])

  useEffect(() => {
    flatRef.current = notationData ? buildFlatNotes(notationData) : []
  }, [notationData])

  const redraw = useCallback(() => {
    const canvas    = canvasRef.current
    const container = containerRef.current
    if (!canvas || !notationData) return

    const rect = container!.getBoundingClientRect()
    if (rect.width < 10 || rect.height < 10) {
      requestAnimationFrame(redraw)
      return
    }

    const w = Math.floor(rect.width)
    const h = Math.floor(rect.height)
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h
    }

    if (flatRef.current.length === 0) {
      const ctx = canvas.getContext("2d")!
      ctx.fillStyle = "#0e0d14"; ctx.fillRect(0, 0, w, h)
      ctx.fillStyle = "rgba(168,130,255,0.4)"; ctx.font = "14px serif"
      ctx.textAlign = "center"
      ctx.fillText("No note data available for this file", w / 2, h / 2 - 10)
      ctx.font = "11px monospace"; ctx.fillStyle = "rgba(168,130,255,0.25)"
      ctx.fillText("Try clicking 'Refresh score' below", w / 2, h / 2 + 14)
      return
    }

    renderPianoRoll(canvas, notationData, flatRef.current, playbackPosition)
  }, [notationData, playbackPosition])

  useEffect(() => { redraw() }, [redraw])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(() => redraw())
    ro.observe(el)
    return () => ro.disconnect()
  }, [redraw])

  if (!notationData) {
    return (
      <div className="bg-card rounded-xl p-8 min-h-full border border-border flex flex-col">
        <h2 className="text-center font-serif text-2xl italic text-foreground/90 mb-8">
          New Composition
        </h2>
        <div className="flex-1"><EmptyScore /></div>
        <div className="flex justify-between text-sm text-muted-foreground mt-8 font-serif italic">
          <span>Bach Propagation Engine v2.0</span>
          <span>Sheet No. 1</span>
        </div>
      </div>
    )
  }

  const title = activeScore?.movement_name?.trim()
    || (activeScore?.bwv ? `BWV ${activeScore.bwv}` : null)
    || activeScore?.collection?.replace(/_/g, " ")
    || "Generated Composition"

  const subtitle = activeScore
    ? [activeScore.key_signature, activeScore.time_signature].filter(Boolean).join(" · ")
    : [notationData.key_signature, notationData.time_signature].filter(Boolean).join(" · ")

  return (
    <div className="bg-card rounded-xl p-6 min-h-full border border-border flex flex-col gap-4">
      <div className="text-center">
        <h2 className="font-serif text-2xl italic text-foreground/90">{title}</h2>
        {subtitle && (
          <p className="text-xs text-muted-foreground font-mono tracking-widest mt-1">
            {subtitle.toUpperCase()}
          </p>
        )}
      </div>

      <div className="flex items-center justify-center gap-6 text-xs text-muted-foreground font-mono select-none">
        <span className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 rounded-sm" style={{background:"rgba(80,55,140,0.45)"}} />
          Past
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 rounded-sm" style={{background:"rgba(120,88,198,0.65)"}} />
          Upcoming
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 rounded-sm"
            style={{background:"rgba(185,145,255,1)", boxShadow:"0 0 8px rgba(185,145,255,0.9)"}} />
          Playing
        </span>
      </div>

      <div ref={containerRef} className="flex-1 rounded-lg overflow-hidden"
        style={{
          minHeight: 340, background: "#0e0d14",
          boxShadow: "inset 0 0 40px rgba(0,0,0,0.6), 0 0 20px rgba(168,130,255,0.08)",
        }}>
        <canvas ref={canvasRef} className="block w-full h-full" />
      </div>

      <div className="flex justify-between items-center text-sm text-muted-foreground font-serif italic">
        <span>Bach Propagation Engine v2.0</span>
        <div className="flex items-center gap-3">
          <span>
            {notationData.measures.length > 0
              ? `${notationData.measures.length} measures · ${notationData.key_signature}`
              : notationData.key_signature}
          </span>
          {activeScore && (
            <Button variant="ghost" size="sm"
              className="h-7 px-2 text-xs text-muted-foreground hover:text-primary"
              onClick={handleRefresh} disabled={refreshing}
              title="Force re-parse notation from MIDI">
              <RefreshCw className={`w-3 h-3 mr-1 ${refreshing ? "animate-spin" : ""}`} />
              {refreshing ? "Re-parsing…" : "Refresh score"}
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}