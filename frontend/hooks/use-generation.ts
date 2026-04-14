/**
 * frontend/hooks/use-generation.ts
 *
 * Hook wrapping POST /api/generate.
 *
 * Usage in BachWorkbench:
 *   const gen = useGeneration({ onComplete: (result) => { ... } })
 *   gen.generate({ key_mode: "minor", n_tokens: 64, temperature: 0.8, top_k: 10 })
 *
 * The optional onComplete callback lets the parent post a summary
 * message into the Copilot panel without creating a circular dependency.
 */

"use client"

import { useState, useCallback, useRef } from "react"
import { api, ApiError, type GenerationRequest, type GenerationResponse } from "@/lib/api-client"

export interface UseGenerationOptions {
  onComplete?: (result: GenerationResponse) => void
}

export interface UseGenerationReturn {
  isGenerating: boolean
  lastResult:   GenerationResponse | null
  error:        string | null
  generate:     (req: GenerationRequest) => Promise<void>
}

export function useGeneration(options: UseGenerationOptions = {}): UseGenerationReturn {
  const [isGenerating, setIsGenerating] = useState(false)
  const [lastResult,   setLastResult]   = useState<GenerationResponse | null>(null)
  const [error,        setError]        = useState<string | null>(null)

  // Keep onComplete ref-stable so callers don't need to memoize
  const onCompleteRef = useRef(options.onComplete)
  onCompleteRef.current = options.onComplete

  const generate = useCallback(async (req: GenerationRequest) => {
    if (isGenerating) return

    setIsGenerating(true)
    setError(null)

    try {
      const result = await api.generation.generate(req)
      setLastResult(result)
      onCompleteRef.current?.(result)
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) {
        setError("Model not loaded — the transformer checkpoint was not found on the server.")
      } else if (err instanceof ApiError && err.status === 408) {
        setError("Generation timed out. Try reducing the number of tokens.")
      } else if (err instanceof ApiError) {
        setError(`Generation failed (${err.status}): ${err.message}`)
      } else {
        setError("Generation failed — unexpected error.")
      }
    } finally {
      setIsGenerating(false)
    }
  }, [isGenerating])

  return { isGenerating, lastResult, error, generate }
}