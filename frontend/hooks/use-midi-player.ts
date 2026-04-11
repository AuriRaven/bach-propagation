/**
 * frontend/hooks/use-midi-player.ts
 *
 * Tone.js MIDI playback. Fixes:
 *   - Playhead calculation uses transport.seconds → beats (not BBT string)
 *   - Proper teardown between piece loads
 *   - No looping (Transport.loop = false)
 */

"use client"

import { useEffect, useRef, useCallback } from "react"
import { useAppState } from "@/lib/app-state"

export function useMidiPlayer() {
  const {
    activeScore,
    setPlaybackState,
    setPlaybackPosition,
    playbackState,
  } = useAppState()

  const loadedUrlRef  = useRef<string | null>(null)
  const synthsRef     = useRef<import("tone").PolySynth[]>([])
  const tickEventRef  = useRef<number | null>(null)   // Tone event ID for the repeat ticker
  const isReadyRef    = useRef(false)

  // ── Teardown — stop transport, dispose synths, cancel all events ─────────
  const teardown = useCallback(async () => {
    const Tone = await import("tone")
    const transport = Tone.getTransport()

    transport.stop()
    transport.cancel()       // removes all scheduled events
    transport.loop = false
    transport.position = "0:0:0"

    for (const s of synthsRef.current) {
      try { s.releaseAll(); s.dispose() } catch { /* already disposed */ }
    }
    synthsRef.current  = []
    tickEventRef.current = null
    loadedUrlRef.current = null
    isReadyRef.current   = false
  }, [])

  // ── Load MIDI when activeScore.signed_url changes ────────────────────────
  useEffect(() => {
    if (!activeScore?.signed_url) return
    if (loadedUrlRef.current === activeScore.signed_url) return

    void (async () => {
      const Tone = await import("tone")
      const { Midi } = await import("@tonejs/midi")

      await teardown()

      // Fetch and parse MIDI
      let midi: InstanceType<typeof Midi>
      try {
        const buf = await fetch(activeScore.signed_url!).then((r) => r.arrayBuffer())
        midi = new Midi(buf)
      } catch (err) {
        console.error("[useMidiPlayer] MIDI fetch failed:", err)
        return
      }

      loadedUrlRef.current = activeScore.signed_url!

      const transport = Tone.getTransport()
      transport.bpm.value = midi.header.tempos[0]?.bpm ?? 120
      transport.loop      = false   // never loop — prevents playhead reset

      // Schedule notes from all tracks
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

      // ── Tick: update playback position every 64ms (~15fps) ──────────────
      // Uses transport.seconds (absolute seconds elapsed) converted to beats.
      // This is monotonically increasing and never resets mid-playback.
      const bpm = transport.bpm.value
      const tickId = transport.scheduleRepeat(() => {
        const seconds    = transport.seconds          // seconds since transport start
        const beatPosition = seconds * (bpm / 60)    // quarter beats elapsed
        setPlaybackPosition(beatPosition)
      }, "16n")

      tickEventRef.current = tickId as unknown as number
      isReadyRef.current   = true
    })()

    return () => { void teardown() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeScore?.signed_url])

  // ── Controls ──────────────────────────────────────────────────────────────

  const play = useCallback(async () => {
    if (!isReadyRef.current) return
    const Tone = await import("tone")
    await Tone.start()                    // resume AudioContext on user gesture
    Tone.getTransport().start()
    setPlaybackState("playing")
  }, [setPlaybackState])

  const pause = useCallback(async () => {
    const Tone = await import("tone")
    Tone.getTransport().pause()
    setPlaybackState("paused")
  }, [setPlaybackState])

  const stop = useCallback(async () => {
    const Tone = await import("tone")
    const transport = Tone.getTransport()
    transport.stop()
    transport.position = "0:0:0"
    setPlaybackState("stopped")
    setPlaybackPosition(0)
  }, [setPlaybackState, setPlaybackPosition])

  const seekTo = useCallback(async (beat: number) => {
    const Tone  = await import("tone")
    const bpm   = Tone.getTransport().bpm.value
    const secs  = beat / (bpm / 60)
    Tone.getTransport().seconds = secs
    setPlaybackPosition(beat)
  }, [setPlaybackPosition])

  return { play, pause, stop, seekTo, playbackState }
}