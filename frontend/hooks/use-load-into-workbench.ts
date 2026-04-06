/**
 * frontend/hooks/use-load-into-workbench.ts
 *
 * Atomically fetches file details + notation data in parallel, hydrates
 * AppState, then navigates to "New Composition".
 *
 * Invariant: the Composition view is never shown with null activeScore
 * or null notationData — both must resolve before navigation.
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

    try {
      // Parallel fetch — file details (incl. signed_url) + notation payload
      const [file, notation] = await Promise.all([
        api.corpus.get(corpusFileId),
        api.corpus.notation(corpusFileId),
      ])

      // Atomic hydration — both must succeed before switching views
      setActiveScore(file)
      setNotationData(notation)
      setActiveNav("New Composition")   // matches ViewType in bach-workbench.tsx
    } catch (err) {
      console.error("[useLoadIntoWorkbench]", err)
      throw err
    } finally {
      setIsLoadingWorkbench(false)
    }
  }, [setActiveScore, setNotationData, setActiveNav, setIsLoadingWorkbench, setPlaybackState, setPlaybackPosition])
}