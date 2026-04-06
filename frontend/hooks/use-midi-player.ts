/**
 * frontend/hooks/use-midi-player.ts
 *
 * Tone.js MIDI playback hook.
 * Loads the MIDI file from activeScore.signed_url, schedules all notes via
 * Tone.Transport, and ticks AppState.playbackPosition every 16th note so
 * VexFlowRenderer can move its cursor in real time.
 *
 * Install: pnpm add tone @tonejs/midi
 */

"use client"

import { useEffect, useRef, useCallback } from "react"
import { useAppState } from "@/lib/app-state"

// Lazy-load Tone.js and @tonejs/midi — both are client-side only
async function getTone() {
  const Tone = await import("tone")
  return Tone
}

async function getMidi(url: string) {
  const { Midi } = await import("@tonejs/midi")
  const arrayBuffer = await fetch(url).then((r) => r.arrayBuffer())
  return new Midi(arrayBuffer)
}

export function useMidiPlayer() {
  const { activeScore, setPlaybackState, setPlaybackPosition, playbackState } =
    useAppState()

  const synthsRef  = useRef<import("tone").PolySynth[]>([])
  const tickerRef  = useRef<import("tone").ToneEventCallback | null>(null)
  const loadedUrl  = useRef<string | null>(null)
  const isSetup    = useRef(false)

  // ── Teardown helper ──────────────────────────────────────────────────────
  const teardown = useCallback(async () => {
    const Tone = await getTone()
    Tone.getTransport().stop()
    Tone.getTransport().cancel()
    for (const s of synthsRef.current) {
      s.releaseAll()
      s.dispose()
    }
    synthsRef.current = []
    loadedUrl.current = null
    isSetup.current = false
  }, [])

  // ── Load + schedule MIDI whenever activeScore.signed_url changes ─────────
  useEffect(() => {
    if (!activeScore?.signed_url) return
    if (loadedUrl.current === activeScore.signed_url) return

    void (async () => {
      const Tone = await getTone()
      await teardown()

      const midi = await getMidi(activeScore.signed_url!)
      loadedUrl.current = activeScore.signed_url!

      const transport = Tone.getTransport()
      transport.bpm.value = midi.header.tempos[0]?.bpm ?? 120

      // Schedule every track
      for (const track of midi.tracks) {
        const synth = new Tone.PolySynth(Tone.Synth, {
          oscillator: { type: "triangle" },
          envelope: { attack: 0.02, decay: 0.1, sustain: 0.5, release: 0.8 },
          volume: -12,
        }).toDestination()

        synthsRef.current.push(synth)

        for (const note of track.notes) {
          transport.schedule((time) => {
            synth.triggerAttackRelease(
              note.name,
              note.duration,
              time,
              note.velocity,
            )
          }, note.time)
        }
      }

      // Tick playback position every 16th note
      transport.scheduleRepeat((time) => {
        // Convert Tone.Transport position to quarter beats
        const [bars, beats, sixteenths] = String(transport.position)
          .split(":")
          .map(Number)
        const beatsPerBar =
          midi.header.timeSignatures[0]?.timeSignature[0] ?? 4
        const quarterBeat =
          bars * beatsPerBar + beats + sixteenths / 4
        setPlaybackPosition(quarterBeat)
      }, "16n")

      isSetup.current = true
    })()

    return () => {
      void teardown()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeScore?.signed_url])

  // ── Controls ──────────────────────────────────────────────────────────────

  const play = useCallback(async () => {
    const Tone = await getTone()
    // AudioContext must be resumed on user gesture
    await Tone.start()
    Tone.getTransport().start()
    setPlaybackState("playing")
  }, [setPlaybackState])

  const pause = useCallback(async () => {
    const Tone = await getTone()
    Tone.getTransport().pause()
    setPlaybackState("paused")
  }, [setPlaybackState])

  const stop = useCallback(async () => {
    const Tone = await getTone()
    Tone.getTransport().stop()
    Tone.getTransport().position = "0:0:0"
    setPlaybackState("stopped")
    setPlaybackPosition(0)
  }, [setPlaybackState, setPlaybackPosition])

  const seekTo = useCallback(async (beat: number) => {
    const Tone = await getTone()
    // Convert quarter beats to Tone.js time string
    Tone.getTransport().position = `${beat}i` // ticks — use seconds instead
    const seconds = Tone.getTransport().toSeconds(`${beat * 60 / Tone.getTransport().bpm.value}`)
    Tone.getTransport().seconds = beat * (60 / Tone.getTransport().bpm.value)
  }, [])

  return { play, pause, stop, seekTo, playbackState }
}