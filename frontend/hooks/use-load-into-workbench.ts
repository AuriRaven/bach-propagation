/**
 * frontend/hooks/use-load-into-workbench.ts
 *
 * Atomically fetches file details + notation data in parallel, hydrates
 * AppState, then navigates to "Analysis".
 *
 * Library → Analysis: loading a corpus piece opens the harmonic analysis view.
 * New Composition is reserved for generated content (via the generation pipeline).
 *
 * Invariant: both activeScore and notationData must resolve before navigation.
 */

"use client"

import { useCallback } from "react"
import { useAppState } from "@/lib/app-state"
import { api } from "@/lib/api-client"

export function useLoadIntoWorkbench() {
  const {
    setActiveScore,
    setNotationData,
    setActiveNav,
    setIsLoadingWorkbench,
    setPlaybackState,
    setPlaybackPosition,
  } = useAppState()

  return useCallback(async (corpusFileId: string): Promise<void> => {
    setIsLoadingWorkbench(true)
    setPlaybackState("stopped")
    setPlaybackPosition(0)

    // ── Clear stale data immediately so old piano roll never flashes ──────
    setActiveScore(null)
    setNotationData(null)

    try {
      const [file, notation] = await Promise.all([
        api.corpus.get(corpusFileId),
        api.corpus.notation(corpusFileId),
      ])

      setActiveScore(file)
      setNotationData(notation)
      setActiveNav("Analysis")
    } catch (err) {
      console.error("[useLoadIntoWorkbench]", err)
      throw err
    } finally {
      setIsLoadingWorkbench(false)
    }
  }, [setActiveScore, setNotationData, setActiveNav, setIsLoadingWorkbench, setPlaybackState, setPlaybackPosition])
}