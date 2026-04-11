/**
 * frontend/lib/app-state.tsx
 *
 * Global Zustand store. ActiveNav matches the existing ViewType in
 * bach-workbench.tsx exactly — do not rename these values.
 */

import { create } from "zustand"
import { devtools } from "zustand/middleware"

// ─── Domain types ─────────────────────────────────────────────────────────────

// Must stay in sync with ViewType in bach-workbench.tsx
export type ActiveNav = "Library" | "New Composition" | "Analysis" | "Settings"

export type PlaybackState = "stopped" | "playing" | "paused"

/** Mirrors corpus_full view columns + injected signed_url */
export interface CorpusFile {
  id: string
  bwv: string | null
  movement_name: string | null
  collection: string
  storage_object_path: string
  storage_url: string | null
  signed_url?: string        // injected by GET /api/corpus/:id
  load_status: string
  key_signature?: string | null
  time_signature?: string | null
  num_measures?: number | null
  num_voices?: number | null
  duration_seconds?: number | null
  form_tag?: string | null
  instrument_family?: string | null
}

export interface VexFlowNote {
  type: "note" | "rest"
  pitch?: string             // single pitch e.g. "C4"
  pitches?: string[]         // chord
  duration: string           // fraction string e.g. "1/4"
  offset: string
}

export interface VexFlowMeasure {
  index: number
  start_beat: number
  end_beat: number
  notes: VexFlowNote[]
}

export interface VexFlowPayload {
  measures: VexFlowMeasure[]
  time_signature: string
  key_signature: string
  total_beats: number
}

// ─── Store ────────────────────────────────────────────────────────────────────

interface AppState {
  activeNav: ActiveNav
  activeScore: CorpusFile | null
  notationData: VexFlowPayload | null
  playbackState: PlaybackState
  playbackPosition: number      // absolute quarter-beat position
  isLoadingWorkbench: boolean
  aiContext: "library" | "analysis" | null

  setActiveNav: (nav: ActiveNav) => void
  setActiveScore: (score: CorpusFile | null) => void
  setNotationData: (data: VexFlowPayload | null) => void
  setPlaybackState: (state: PlaybackState) => void
  setPlaybackPosition: (beat: number) => void
  setIsLoadingWorkbench: (v: boolean) => void
  setAiContext: (ctx: "library" | "analysis" | null) => void

  /** Resets score + notation + playback atomically (e.g. on nav away) */
  clearWorkbench: () => void
}

export const useAppState = create<AppState>()(
  devtools(
    (set) => ({
      activeNav: "Library",
      activeScore: null,
      notationData: null,
      playbackState: "stopped",
      playbackPosition: 0,
      isLoadingWorkbench: false,
      aiContext: null,

      setActiveNav: (nav) => set({ activeNav: nav }, false, "setActiveNav"),
      setActiveScore: (score) => set({ activeScore: score }, false, "setActiveScore"),
      setNotationData: (data) => set({ notationData: data }, false, "setNotationData"),
      setPlaybackState: (state) => set({ playbackState: state }, false, "setPlaybackState"),
      setPlaybackPosition: (beat) => set({ playbackPosition: beat }, false, "setPlaybackPosition"),
      setIsLoadingWorkbench: (v) => set({ isLoadingWorkbench: v }, false, "setIsLoadingWorkbench"),
      setAiContext: (ctx) => set({ aiContext: ctx }, false, "setAiContext"),

      clearWorkbench: () =>
        set(
          { activeScore: null, notationData: null, playbackState: "stopped", playbackPosition: 0 },
          false,
          "clearWorkbench",
        ),
    }),
    { name: "BachPropagation" },
  ),
)