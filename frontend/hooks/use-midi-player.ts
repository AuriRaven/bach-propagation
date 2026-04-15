/**
 * frontend/hooks/use-midi-player.ts
 *
 * Tone.js MIDI playback with stable beat-position tracking.
 *
 * Two load paths:
 *   1. MIDI file from signed URL (corpus pieces)
 *   2. VexFlowPayload from generation (loadFromNotation)
 *
 * Beat tracking uses transport.seconds (monotonic) × (bpm/60) — never
 * the BBT string position, which resets on loop or reposition.
 *
 * Exposes: play, pause, stop, seekTo, loadFromNotation, playbackState, durationBeats
 */

"use client"

import { useEffect, useRef, useCallback } from "react"
import { useAppState, type VexFlowPayload } from "@/lib/app-state"

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Convert music21 pitch notation to Tone.js compatible format.
 *  "E-4" → "Eb4", "B--3" → "Bbb3", "F#4" stays "F#4" */
function m21PitchToTone(pitch: string): string {
  return pitch.replace(/-/g, "b")
}

/** Parse a fraction string like "1/4" → 0.25, or a plain number "2" → 2.0 */
function parseFraction(s: string): number {
  if (s.includes("/")) {
    const [n, d] = s.split("/").map(Number)
    return d ? n / d : n
  }
  return parseFloat(s) || 0
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useMidiPlayer() {
  const {
    activeScore,
    notationData,
    setPlaybackState,
    setPlaybackPosition,
    playbackState,
  } = useAppState()

  const loadedUrlRef    = useRef<string | null>(null)
  const synthsRef       = useRef<import("tone").PolySynth[]>([])
  const bpmRef          = useRef<number>(120)
  const isReadyRef      = useRef(false)
  const rafRef          = useRef<number | null>(null)
  const sourceRef       = useRef<"midi" | "notation" | null>(null)

  // ── Teardown ──────────────────────────────────────────────────────────────
  const teardown = useCallback(async () => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }

    const Tone = await import("tone")
    const transport = Tone.getTransport()
    transport.stop()
    transport.cancel()
    transport.loop     = false
    transport.position = "0:0:0"

    for (const s of synthsRef.current) {
      try { s.releaseAll(); s.dispose() } catch { /* already disposed */ }
    }
    synthsRef.current = []
    loadedUrlRef.current = null
    isReadyRef.current   = false
    sourceRef.current    = null
  }, [])

  // ── rAF tick — updates playback position every animation frame ───────────
  const startTick = useCallback((Tone: typeof import("tone")) => {
    const tick = () => {
      const transport = Tone.getTransport()
      if (transport.state === "started") {
        const beats = transport.seconds * (bpmRef.current / 60)
        setPlaybackPosition(beats)
        rafRef.current = requestAnimationFrame(tick)
      } else {
        rafRef.current = null
      }
    }
    rafRef.current = requestAnimationFrame(tick)
  }, [setPlaybackPosition])

  // ── Load MIDI when signed_url changes ────────────────────────────────────
  useEffect(() => {
    if (!activeScore?.signed_url) return
    if (loadedUrlRef.current === activeScore.signed_url) return

    void (async () => {
      const Tone  = await import("tone")
      const { Midi } = await import("@tonejs/midi")

      await teardown()

      let midi: InstanceType<typeof Midi>
      try {
        const buf = await fetch(activeScore.signed_url!).then((r) => r.arrayBuffer())
        midi = new Midi(buf)
      } catch (err) {
        console.error("[useMidiPlayer] MIDI fetch failed:", err)
        return
      }

      loadedUrlRef.current = activeScore.signed_url!

      const bpm = midi.header.tempos[0]?.bpm ?? 120
      bpmRef.current = bpm

      const transport = Tone.getTransport()
      transport.bpm.value = bpm
      transport.loop      = false

      // Schedule notes
      for (const track of midi.tracks) {
        if (track.notes.length === 0) continue

        const synth = new Tone.PolySynth(Tone.Synth, {
          oscillator: { type: "triangle" },
          envelope:   { attack: 0.02, decay: 0.1, sustain: 0.5, release: 0.8 },
          volume:     -14,
        }).toDestination()

        synthsRef.current.push(synth)

        for (const note of track.notes) {
          transport.schedule((time) => {
            synth.triggerAttackRelease(note.name, note.duration, time, note.velocity)
          }, note.time)
        }
      }

      isReadyRef.current = true
      sourceRef.current  = "midi"
    })()

    return () => { void teardown() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeScore?.signed_url])

  // ── Load from VexFlowPayload (generated scores) ──────────────────────────

  const loadFromNotation = useCallback(async (data: VexFlowPayload, bpm = 120) => {
    const Tone = await import("tone")
    await teardown()

    bpmRef.current = bpm
    const transport = Tone.getTransport()
    transport.bpm.value = bpm
    transport.loop = false

    const synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "triangle" },
      envelope:   { attack: 0.02, decay: 0.1, sustain: 0.5, release: 0.8 },
      volume:     -14,
    }).toDestination()

    synthsRef.current.push(synth)

    let scheduledCount = 0

    for (const measure of data.measures) {
      for (const note of measure.notes) {
        if (note.type === "rest") continue

        const pitches = note.pitches
          ? note.pitches.map(m21PitchToTone)
          : note.pitch
          ? [m21PitchToTone(note.pitch)]
          : []

        if (pitches.length === 0) continue

        const offsetBeats   = parseFraction(note.offset)
        const durationBeats = parseFraction(note.duration)
        const offsetSec     = offsetBeats * (60 / bpm)
        const durationSec   = Math.max(durationBeats * (60 / bpm), 0.05)

        transport.schedule((time) => {
          synth.triggerAttackRelease(pitches, durationSec, time)
        }, offsetSec)

        scheduledCount++
      }
    }

    console.info(
      `[useMidiPlayer] Loaded ${scheduledCount} notes from notation, ` +
      `${data.measures.length} measures, ${bpm} BPM`
    )

    isReadyRef.current = true
    sourceRef.current  = "notation"
  }, [teardown])

  // ── Controls ──────────────────────────────────────────────────────────────

  const play = useCallback(async () => {
    if (!isReadyRef.current) return
    const Tone = await import("tone")
    await Tone.start()
    Tone.getTransport().start()
    setPlaybackState("playing")
    startTick(Tone)
  }, [setPlaybackState, startTick])

  const pause = useCallback(async () => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    const Tone = await import("tone")
    Tone.getTransport().pause()
    setPlaybackState("paused")
  }, [setPlaybackState])

  const stop = useCallback(async () => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    const Tone = await import("tone")
    const transport = Tone.getTransport()
    transport.stop()
    transport.position = "0:0:0"
    setPlaybackState("stopped")
    setPlaybackPosition(0)
  }, [setPlaybackState, setPlaybackPosition])

  const seekTo = useCallback(async (beat: number) => {
    const Tone = await import("tone")
    Tone.getTransport().seconds = beat / (bpmRef.current / 60)
    setPlaybackPosition(beat)
  }, [setPlaybackPosition])

  // Total duration in beats from notation data
  const durationBeats = notationData?.total_beats ?? 0

  return { play, pause, stop, seekTo, loadFromNotation, playbackState, durationBeats }
}