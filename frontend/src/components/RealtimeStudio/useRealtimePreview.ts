import { useCallback, useEffect, useRef } from 'react'
import { apiFetch } from '../../api'
import { useStore, type RealtimePreviewState } from '../../store'
import { documentToPngB64, promptFromLayerNotes, type RealtimeDocument } from './canvasDocument'

function statusLabel(status: string): RealtimePreviewState['status'] {
  if (status === 'queued') return 'queued'
  if (status === 'running') return 'running'
  if (status === 'done') return 'ready'
  if (status === 'error') return 'error'
  return 'idle'
}

function buildPrompt(prompt: string, document: RealtimeDocument): string {
  const notes = promptFromLayerNotes(document)
  return [prompt.trim(), notes.trim()].filter(Boolean).join('\n\nCanvas notes:\n')
}

export function useRealtimePreview() {
  const document = useStore(s => s.realtime.document)
  const prompt = useStore(s => s.realtime.prompt)
  const negativePrompt = useStore(s => s.realtime.negativePrompt)
  const sessionId = useStore(s => s.realtime.preview.sessionId)
  const paused = useStore(s => s.realtime.preview.paused)
  const settings = useStore(s => s.realtime.settings)
  const setRealtimePreview = useStore(s => s.setRealtimePreview)
  const latestJobRef = useRef<string | null>(null)
  const pollTimer = useRef<number | null>(null)
  const debounceTimer = useRef<number | null>(null)

  const clearPoll = () => {
    if (pollTimer.current !== null) window.clearTimeout(pollTimer.current)
    pollTimer.current = null
  }

  const pollJob = useCallback((jobId: string) => {
    clearPoll()
    pollTimer.current = window.setTimeout(async () => {
      try {
        const job = await apiFetch.realtimePreviewStatus(jobId)
        if (latestJobRef.current !== jobId) return
        if (job.status === 'done' && job.image_b64) {
          setRealtimePreview({
            status: 'ready',
            progress: 100,
            image: job.image_b64,
            seed: job.seed ?? null,
            metadata: job.metadata ?? null,
            lastUpdated: Date.now(),
            error: null,
          })
          return
        }
        if (job.status === 'error') {
          setRealtimePreview({ status: 'error', error: job.error ?? 'Preview failed', lastUpdated: Date.now() })
          return
        }
        if (job.status === 'cancelled' || job.status === 'stale') {
          setRealtimePreview({ status: 'idle', lastUpdated: Date.now() })
          return
        }
        setRealtimePreview({ status: statusLabel(job.status), progress: job.progress ?? 0, error: null })
        pollJob(jobId)
      } catch (e) {
        if (latestJobRef.current === jobId) {
          setRealtimePreview({ status: 'error', error: e instanceof Error ? e.message : 'Preview polling failed' })
        }
      }
    }, 900)
  }, [setRealtimePreview])

  const previewNow = useCallback(async () => {
    if (paused) return
    const previous = latestJobRef.current
    if (previous) {
      apiFetch.cancelRealtimePreview(previous).catch(() => undefined)
    }
    setRealtimePreview({ status: 'queued', error: null, progress: 0 })
    try {
      const canvasB64 = await documentToPngB64(document)
      const job = await apiFetch.realtimePreview({
        session_id: sessionId,
        prompt: buildPrompt(prompt, document),
        negative_prompt: negativePrompt,
        canvas_image_b64: canvasB64,
        width: settings.previewSize,
        height: Math.round(settings.previewSize * (document.height / document.width)),
        preview_steps: settings.previewSteps,
        moodboard_strength: settings.canvasInfluence,
        seed: settings.lockSeed ? settings.seed : -1,
      })
      latestJobRef.current = job.job_id
      setRealtimePreview({
        status: statusLabel(job.status),
        jobId: job.job_id,
        revision: job.revision,
        error: null,
      })
      pollJob(job.job_id)
    } catch (e) {
      setRealtimePreview({
        status: 'error',
        error: e instanceof Error ? e.message : 'Could not start preview',
        lastUpdated: Date.now(),
      })
    }
  }, [document, negativePrompt, paused, pollJob, prompt, sessionId, setRealtimePreview, settings.canvasInfluence, settings.lockSeed, settings.previewSize, settings.previewSteps, settings.seed])

  const cancelPreview = useCallback(async () => {
    clearPoll()
    if (debounceTimer.current !== null) window.clearTimeout(debounceTimer.current)
    debounceTimer.current = null
    const jobId = latestJobRef.current
    latestJobRef.current = null
    if (jobId) await apiFetch.cancelRealtimePreview(jobId).catch(() => undefined)
    setRealtimePreview({ status: 'idle', jobId: null, error: null, progress: 0 })
  }, [setRealtimePreview])

  useEffect(() => {
    if (!settings.autoPreview || paused || document.layers.length === 0) return
    if (debounceTimer.current !== null) window.clearTimeout(debounceTimer.current)
    debounceTimer.current = window.setTimeout(() => {
      previewNow()
    }, settings.debounceMs)
    return () => {
      if (debounceTimer.current !== null) window.clearTimeout(debounceTimer.current)
    }
  }, [
    document,
    prompt,
    negativePrompt,
    settings.autoPreview,
    settings.debounceMs,
    settings.previewSize,
    settings.previewSteps,
    settings.canvasInfluence,
    settings.lockSeed,
    settings.seed,
    paused,
    previewNow,
  ])

  useEffect(() => {
    if (!paused) {
      if (settings.autoPreview && document.layers.length > 0) {
        if (debounceTimer.current !== null) window.clearTimeout(debounceTimer.current)
        debounceTimer.current = window.setTimeout(() => previewNow(), settings.debounceMs)
      }
      return
    }
    clearPoll()
    if (debounceTimer.current !== null) window.clearTimeout(debounceTimer.current)
    debounceTimer.current = null
    const jobId = latestJobRef.current
    latestJobRef.current = null
    if (jobId) apiFetch.cancelRealtimePreview(jobId).catch(() => undefined)
    setRealtimePreview({ status: 'idle', jobId: null, error: null, progress: 0 })
  }, [paused])

  useEffect(() => () => {
    clearPoll()
    if (debounceTimer.current !== null) window.clearTimeout(debounceTimer.current)
  }, [])

  return { previewNow, cancelPreview }
}
