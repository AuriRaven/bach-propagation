/**
 * frontend/hooks/use-midi-player.ts
 *
 * Tone.js MIDI playback with tempo-map-aware beat tracking.
 *
 * Two load paths:
 *   1. MIDI file from signed URL (corpus pieces)
 *   2. VexFlowPayload from generation (loadFromNotation)
 *
 * Beat tracking builds a tempo map from MIDI header tempos and uses
 * piecewise integration to convert transport.seconds → beats. This
 * handles multi-tempo pieces correctly (e.g. BWV 1004 Chaconne).
 *
 * Auto-stops when playback reaches the end of the piece.
 *
 * Exposes: play, pause, stop, seekTo, loadFromNotation, playbackState, durationBeats
 */

"use client"

import { useEffect, useRef, useCallback } from "react"
import { useAppState, type VexFlowPayload } from "@/lib/app-state"

// ─── Tempo map types and helpers ──────────────────────────────────────────────

/** A segment of constant tempo: from startSec to the next segment. */
interface TempoSegment {
  startSec: number   // wall-clock seconds where this tempo begins
  startBeat: number  // cumulative beats at this point
  bpm: number        // tempo for this segment
}

/** Convert wall-clock seconds to beats using a tempo map. */
function secondsToBeats(seconds: number, tempoMap: TempoSegment[]): number {
  if (tempoMap.length === 0) return seconds * 2 // fallback: 120 BPM

  // Find the active segment (last segment where startSec <= seconds)
  let seg = tempoMap[0]
  for (let i = 1; i < tempoMap.length; i++) {
    if (tempoMap[i].startSec <= seconds) seg = tempoMap[i]
    else break
  }

  const elapsed = seconds - seg.startSec
  return seg.startBeat + elapsed * (seg.bpm / 60)
}

/** Convert beats to wall-clock seconds using a tempo map. */
function beatsToSeconds(beats: number, tempoMap: TempoSegment[]): number {
  if (tempoMap.length === 0) return beats / 2 // fallback: 120 BPM

  let seg = tempoMap[0]
  for (let i = 1; i < tempoMap.length; i++) {
    if (tempoMap[i].startBeat <= beats) seg = tempoMap[i]
    else break
  }

  const elapsedBeats = beats - seg.startBeat
  return seg.startSec + elapsedBeats * (60 / seg.bpm)
}

/** Build a tempo map from @tonejs/midi header tempos. */
function buildTempoMap(
  tempos: Array<{ bpm: number; ticks: number; time: number }>,
): TempoSegment[] {
  if (!tempos || tempos.length === 0) {
    return [{ startSec: 0, startBeat: 0, bpm: 120 }]
  }

  // Sort by time
  const sorted = [...tempos].sort((a, b) => a.time - b.time)
  const segments: TempoSegment[] = []
  let cumulativeBeats = 0

  for (let i = 0; i < sorted.length; i++) {
    const t = sorted[i]
    if (i > 0) {
      // Accumulate beats from previous segment
      const prev = segments[i - 1]
      const elapsed = t.time - prev.startSec
      cumulativeBeats = prev.startBeat + elapsed * (prev.bpm / 60)
    }
    segments.push({
      startSec: t.time,
      startBeat: cumulativeBeats,
      bpm: t.bpm,
    })
  }

  return segments
}

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
  const tempoMapRef     = useRef<TempoSegment[]>([{ startSec: 0, startBeat: 0, bpm: 120 }])
  const durationSecRef  = useRef<number>(Infinity) // total duration in seconds for auto-stop
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
    tempoMapRef.current  = [{ startSec: 0, startBeat: 0, bpm: 120 }]
    durationSecRef.current = Infinity
  }, [])

  // ── rAF tick — uses tempo map for accurate beat position ──────────────────
  // Also auto-stops when transport.seconds exceeds total duration.
  const stopRef = useRef<(() => Promise<void>) | null>(null)

  const startTick = useCallback((Tone: typeof import("tone")) => {
    const tick = () => {
      const transport = Tone.getTransport()
      if (transport.state === "started") {
        const sec = transport.seconds
        const beats = secondsToBeats(sec, tempoMapRef.current)
        setPlaybackPosition(beats)

        // Auto-stop at end of piece
        if (sec >= durationSecRef.current) {
          // Schedule stop on next microtask to avoid calling setState during rAF
          void stopRef.current?.()
          rafRef.current = null
          return
        }

        rafRef.current = requestAnimationFrame(tick)
      } else {
        rafRef.current = null
      }
    }
    rafRef.current = requestAnimationFrame(tick)
  }, [setPlaybackPosition])

  // ── Load MIDI when signed_url changes ────────────────────────────────────
  // Only attempt MIDI parse for .mid/.midi files. .krn and .musicxml files
  // cannot be parsed by @tonejs/midi (they need music21 on the backend).
  useEffect(() => {
    if (!activeScore?.signed_url) return
    if (loadedUrlRef.current === activeScore.signed_url) return

    // Check file format — skip non-MIDI files
    const score = activeScore as Record<string, unknown>
    const format = (score.file_format as string | undefined)
      ?? (score.storage_object_path as string | undefined)?.split(".").pop()?.toLowerCase()
    if (format && !["mid", "midi"].includes(format)) {
      console.info(`[useMidiPlayer] Skipping MIDI load for .${format} file — audio not available`)
      return
    }

    void (async () => {
      const Tone  = await import("tone")
      const { Midi } = await import("@tonejs/midi")

      await teardown()

      let midi: InstanceType<typeof Midi>
      try {
        const buf = await fetch(activeScore.signed_url!).then((r) => r.arrayBuffer())
        midi = new Midi(buf)
      } catch (err) {
        console.warn("[useMidiPlayer] MIDI parse failed — file may not be in MIDI format:", err)
        return
      }

      loadedUrlRef.current = activeScore.signed_url!

      // Build tempo map from all tempo events
      tempoMapRef.current = buildTempoMap(midi.header.tempos as Array<{ bpm: number; ticks: number; time: number }>)

      const firstBpm = tempoMapRef.current[0].bpm
      const transport = Tone.getTransport()
      transport.bpm.value = firstBpm
      transport.loop      = false

      // Compute total duration in seconds from the latest note end
      let maxEndSec = 0
      for (const track of midi.tracks) {
        for (const note of track.notes) {
          const end = note.time + note.duration
          if (end > maxEndSec) maxEndSec = end
        }
      }
      durationSecRef.current = maxEndSec + 0.5 // small buffer

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

    tempoMapRef.current = [{ startSec: 0, startBeat: 0, bpm }]

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
    let maxEndSec = 0

    for (const measure of data.measures) {
      for (const note of measure.notes) {
        if (note.type === "rest") continue

        const pitches = note.pitches
          ? note.pitches.map(m21PitchToTone)
          : note.pitch
          ? [m21PitchToTone(note.pitch)]
          : []

        if (pitches.length === 0) continue

        const offsetBeats    = parseFraction(note.offset)
        const durationBeats  = parseFraction(note.duration)
        const offsetSec      = offsetBeats * (60 / bpm)
        const durationSec    = Math.max(durationBeats * (60 / bpm), 0.05)

        const endSec = offsetSec + durationSec
        if (endSec > maxEndSec) maxEndSec = endSec

        transport.schedule((time) => {
          synth.triggerAttackRelease(pitches, durationSec, time)
        }, offsetSec)

        scheduledCount++
      }
    }

    durationSecRef.current = maxEndSec + 0.5

    console.info(
      `[useMidiPlayer] Loaded ${scheduledCount} notes from notation, ` +
      `${data.measures.length} measures, ${bpm} BPM`
    )

    isReadyRef.current = true
    sourceRef.current  = "notation"
  }, [teardown])

  // ── Controls ──────────────────────────────────────────────────────────────

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

  // Keep stopRef in sync so the rAF tick can call it
  stopRef.current = stop

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

  const seekTo = useCallback(async (beat: number) => {
    const Tone = await import("tone")
    // Use tempo map for accurate beat → seconds conversion
    Tone.getTransport().seconds = beatsToSeconds(beat, tempoMapRef.current)
    setPlaybackPosition(beat)
  }, [setPlaybackPosition])

  // Total duration in beats from notation data
  const durationBeats = notationData?.total_beats ?? 0

  return { play, pause, stop, seekTo, loadFromNotation, playbackState, durationBeats }
}