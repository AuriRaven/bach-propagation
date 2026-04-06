"use client"

/**
 * frontend/components/views/composition-view.tsx
 *
 * Renders the loaded score via VexFlow 4.x in a <canvas> element.
 * Tracks AppState.playbackPosition to move a red cursor bar in real time.
 *
 * Install: pnpm add vexflow
 *
 * SSR note: VexFlow manipulates the DOM directly — all rendering happens
 * inside useEffect (client-side only). No dynamic import needed since this
 * is already a "use client" component.
 */

import { useEffect, useRef, useCallback } from "react"
import { useAppState, type VexFlowMeasure, type VexFlowNote } from "@/lib/app-state"

// ─── Duration conversion: music21 fraction → VexFlow key ─────────────────────
// music21 quarterLength fractions: 4=whole, 2=half, 1=quarter, 0.5=eighth, etc.
function durationToVex(fraction: string): string {
  const [num, den] = fraction.split("/").map(Number)
  const quarters = den ? num / den : num   // e.g. "1/4" → 0.25, "1" → 1

  if (quarters >= 4)    return "w"
  if (quarters >= 2)    return "h"
  if (quarters >= 1)    return "q"
  if (quarters >= 0.5)  return "8"
  if (quarters >= 0.25) return "16"
  return "32"
}

// music21 pitch "C4" → VexFlow "c/4"
function pitchToVex(pitch: string): string {
  // music21 uses sharps/naturals: C4, D#4, Eb4, B-4 (flat = "-" suffix)
  const match = pitch.match(/^([A-G])(#{1,2}|b+|-+)?(\d)$/)
  if (!match) return "c/4"
  const [, name, acc = "", octave] = match
  const vexAcc = acc.replace(/-/g, "b")  // music21 "B-4" → "Bb" → "b"
  return `${name.toLowerCase()}${vexAcc}/${octave}`
}

// ─── VexFlow renderer ─────────────────────────────────────────────────────────

const STAVE_HEIGHT  = 120   // px per staff system
const STAVE_MARGIN  = 20    // px left margin
const MEASURES_PER_LINE = 4
const MEASURE_WIDTH = 200   // px
const CANVAS_WIDTH  = STAVE_MARGIN + MEASURES_PER_LINE * MEASURE_WIDTH + 20

function renderScore(
  canvas: HTMLCanvasElement,
  measures: VexFlowMeasure[],
  cursorBeat: number,
) {
  // Dynamic import is synchronous after the module is bundled — safe here
  // because this function only runs inside useEffect (client-side).
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const Vex = require("vexflow") as typeof import("vexflow")
  const { Renderer, Stave, StaveNote, Voice, Formatter, Accidental } = Vex.default ?? Vex

  const totalLines = Math.ceil(measures.length / MEASURES_PER_LINE)
  const canvasHeight = totalLines * STAVE_HEIGHT * 2 + 60  // treble + bass * lines

  canvas.width  = CANVAS_WIDTH
  canvas.height = canvasHeight

  const renderer = new Renderer(canvas, Renderer.Backends.CANVAS)
  renderer.resize(CANVAS_WIDTH, canvasHeight)
  const ctx = renderer.getContext()

  // Dark theme colours to match the app
  ctx.setFont("Arial", 10)
  ctx.setFillStyle("oklch(0.85 0.05 290)")
  ctx.setStrokeStyle("oklch(0.85 0.05 290)")

  // Clear background
  const rawCtx = canvas.getContext("2d")!
  rawCtx.fillStyle = "transparent"
  rawCtx.clearRect(0, 0, canvas.width, canvas.height)

  // Find the measure that contains the cursor beat
  const activeMeasureIdx = measures.findIndex(
    (m) => cursorBeat >= m.start_beat && cursorBeat < m.end_beat,
  )

  for (let line = 0; line < totalLines; line++) {
    const lineMeasures = measures.slice(
      line * MEASURES_PER_LINE,
      (line + 1) * MEASURES_PER_LINE,
    )

    const yTreble = line * STAVE_HEIGHT * 2 + 20
    const yBass   = yTreble + STAVE_HEIGHT

    for (let mi = 0; mi < lineMeasures.length; mi++) {
      const measure = lineMeasures[mi]
      const x = STAVE_MARGIN + mi * MEASURE_WIDTH
      const absIdx = line * MEASURES_PER_LINE + mi

      // ── Treble stave ──────────────────────────────────────────────────
      const trebleStave = new Stave(x, yTreble, MEASURE_WIDTH - 10)
      if (mi === 0) trebleStave.addClef("treble")
      trebleStave.setContext(ctx).draw()

      // ── Bass stave ────────────────────────────────────────────────────
      const bassStave = new Stave(x, yBass, MEASURE_WIDTH - 10)
      if (mi === 0) bassStave.addClef("bass")
      bassStave.setContext(ctx).draw()

      // ── Build VexFlow notes from measure data ─────────────────────────
      const vexNotes = _buildVexNotes(measure.notes, Vex)

      if (vexNotes.length > 0) {
        try {
          const voice = new Voice({ numBeats: 4, beatValue: 4 }).setStrict(false)
          voice.addTickables(vexNotes)
          new Formatter().joinVoices([voice]).format([voice], MEASURE_WIDTH - 20)
          voice.draw(ctx, trebleStave)
        } catch {
          // Skip malformed measure — don't crash the renderer
        }
      }

      // ── Cursor highlight ──────────────────────────────────────────────
      if (absIdx === activeMeasureIdx) {
        const progress =
          (cursorBeat - measure.start_beat) /
          Math.max(measure.end_beat - measure.start_beat, 1)
        const cursorX = x + progress * (MEASURE_WIDTH - 10)

        rawCtx.save()
        rawCtx.strokeStyle = "oklch(0.75 0.2 25)"   // amber-red cursor
        rawCtx.lineWidth   = 2
        rawCtx.globalAlpha = 0.85
        rawCtx.setLineDash([4, 2])
        rawCtx.beginPath()
        rawCtx.moveTo(cursorX, yTreble)
        rawCtx.lineTo(cursorX, yBass + 45)
        rawCtx.stroke()
        rawCtx.restore()
      }
    }
  }
}

function _buildVexNotes(
  notes: VexFlowNote[],
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  Vex: any,
): InstanceType<typeof import("vexflow").StaveNote>[] {
  const { StaveNote, Accidental } = Vex.default ?? Vex
  const out: unknown[] = []

  for (const note of notes) {
    try {
      const dur = durationToVex(note.duration)

      if (note.type === "rest") {
        out.push(
          new StaveNote({ keys: ["b/4"], duration: `${dur}r`, align_center: true }),
        )
        continue
      }

      const pitches = note.pitches
        ? note.pitches.map(pitchToVex)
        : [pitchToVex(note.pitch ?? "C4")]

      const sn = new StaveNote({ keys: pitches, duration: dur, clef: "treble" })

      // Add accidentals where needed
      pitches.forEach((p, idx) => {
        if (p.includes("#")) sn.addModifier(new Accidental("#"), idx)
        else if (p.includes("b") && !p.startsWith("b/"))
          sn.addModifier(new Accidental("b"), idx)
      })

      out.push(sn)
    } catch {
      // Skip un-renderable notes silently
    }
  }

  return out as InstanceType<typeof import("vexflow").StaveNote>[]
}

// ─── Empty state ──────────────────────────────────────────────────────────────

function EmptyScore() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-muted-foreground">
      {/* Decorative staff lines */}
      <svg className="w-64 h-24 opacity-30" viewBox="0 0 256 80">
        {[15, 28, 41, 54, 67].map((y) => (
          <line key={y} x1="20" y1={y} x2="236" y2={y} stroke="currentColor" strokeWidth="1" />
        ))}
        <text x="14" y="52" fontSize="48" fontFamily="serif" fill="currentColor">𝄞</text>
        <text x="14" y="76" fontSize="24" fontFamily="serif" fill="currentColor" opacity="0.5">
          No score loaded
        </text>
      </svg>
      <p className="text-sm font-serif italic">
        Select a piece from the Library and click <strong>Load into Workbench</strong>
      </p>
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export function CompositionView() {
  const { activeScore, notationData, playbackPosition } = useAppState()
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const redraw = useCallback(() => {
    if (!canvasRef.current || !notationData) return
    try {
      renderScore(canvasRef.current, notationData.measures, playbackPosition)
    } catch (err) {
      console.error("[CompositionView] VexFlow render error:", err)
    }
  }, [notationData, playbackPosition])

  // Re-render whenever notation data or playback position changes
  useEffect(() => { redraw() }, [redraw])

  if (!activeScore || !notationData) {
    return (
      <div className="bg-card rounded-xl p-8 min-h-full border border-border flex flex-col">
        <h2 className="text-center font-serif text-2xl italic text-foreground/90 mb-8">
          New Composition
        </h2>
        <div className="flex-1">
          <EmptyScore />
        </div>
        <div className="flex justify-between text-sm text-muted-foreground mt-8 font-serif italic">
          <span>Bach Propagation Engine v2.0</span>
          <span>Sheet No. 1</span>
        </div>
      </div>
    )
  }

  const title = activeScore.movement_name ?? `BWV ${activeScore.bwv}` ?? "Score"
  const subtitle = [activeScore.key_signature, activeScore.time_signature]
    .filter(Boolean)
    .join(" · ")

  return (
    <div className="bg-card rounded-xl p-8 min-h-full border border-border flex flex-col">
      {/* Title */}
      <h2 className="text-center font-serif text-2xl italic text-foreground/90 mb-1">
        {title}
      </h2>
      {subtitle && (
        <p className="text-center text-xs text-muted-foreground font-mono tracking-widest mb-8">
          {subtitle.toUpperCase()}
        </p>
      )}

      {/* VexFlow canvas */}
      <div className="flex-1 overflow-auto">
        <canvas
          ref={canvasRef}
          className="mx-auto block"
          style={{ maxWidth: "100%" }}
        />
      </div>

      {/* Footer */}
      <div className="flex justify-between text-sm text-muted-foreground mt-8 font-serif italic">
        <span>Bach Propagation Engine v2.0</span>
        <span>
          {notationData.measures.length} measures · {notationData.key_signature}
        </span>
      </div>
    </div>
  )
}