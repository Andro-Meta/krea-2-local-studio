import React, { useCallback, useEffect, useLayoutEffect, useReducer, useRef, useState } from 'react'
import { Alert, Box, Button, Checkbox, Chip, CircularProgress, Collapse, FormControlLabel, LinearProgress, MenuItem, Paper, Slider, Snackbar, Stack, TextField, Tooltip, Typography } from '@mui/material'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import TipsAndUpdatesIcon from '@mui/icons-material/TipsAndUpdates'
import { useStore } from '../../store'
import { apiFetch, connectWS, type GpuTaskResponse, type MoodboardSuggestion, type PromptPlan } from '../../api'
import {
  activeGpuTaskStorageKey,
  canDeliverTaskResult,
  reconcileActiveTaskIdentity,
} from '../../lib/activeTaskPersistence'
import { createTaskWatcher, type TaskWatcher } from '../../lib/taskWatcher'
import {
  applyGuardedPromptMutation,
  initialWandTaskState,
  parsePersistedWandTask,
  reduceWandTaskState,
  serializePersistedWandTask,
  wandCancelAriaLabel,
  wandProgressAriaLabel,
  wandStatusAnnouncement,
  type WandResult,
  type WandSubmission,
} from '../../lib/wandTaskState'
import CreatePromptFromImage from '../CreatePromptFromImage'

const ABLITERATED_QWEN = 'huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated'
const WAND_TASK_KIND = 'prompt_expand'

function persistWandTask(storageKey: string | null, submission: WandSubmission): boolean {
  if (!storageKey) return false
  localStorage.setItem(storageKey, serializePersistedWandTask(submission))
  return true
}

function clearPersistedWandTask(storageKey: string | null, expectedJobId: string): boolean {
  if (!storageKey) return false
  const persisted = parsePersistedWandTask(localStorage.getItem(storageKey))
  if (persisted?.jobId !== expectedJobId) return false
  localStorage.removeItem(storageKey)
  return true
}

function wandChoiceFromModel(modelId: string) {
  if (!modelId) return 'default'
  return /Huihui-Qwen3-VL-4B-Instruct-abliterated|qwen3_vl_4b_abliterated/i.test(modelId) ? 'abliterated' : 'custom'
}

function modelFromWandChoice(choice: string, current: string) {
  if (choice === 'default') return ''
  if (choice === 'abliterated') return ABLITERATED_QWEN
  return current && wandChoiceFromModel(current) === 'custom' ? current : 'custom/repo-or-path'
}

export default function PromptSection() {
  const { params, setParam, setParams, setPromptBusy, moodboardSuggestions, setMoodboardSuggestions } = useStore()
  const [wandState, dispatchWand] = useReducer(reduceWandTaskState, initialWandTaskState)
  const [submittingWand, setSubmittingWand] = useState(false)
  const [planning, setPlanning] = useState(false)
  const [plan, setPlan] = useState<PromptPlan | null>(null)
  const [notice, setNotice] = useState<{ message: string; severity: 'success' | 'warning' | 'error' } | null>(null)
  const [preWandPrompt, setPreWandPrompt] = useState('')
  const [wandModel, setWandModel] = useState('')
  const [wandBackend, setWandBackend] = useState<'local' | 'openrouter' | 'ideogram-json'>('local')
  const [localLlmBackend, setLocalLlmBackend] = useState<'comfy' | 'transformers' | 'gguf_server'>('comfy')
  const [wandDevice, setWandDevice] = useState<'auto' | 'cuda' | 'cpu'>('auto')
  const [showWandAdvanced, setShowWandAdvanced] = useState(false)
  const [showPromptTools, setShowPromptTools] = useState(false)
  const [activeTaskKey, setActiveTaskKey] = useState<string | null>(null)
  const watcherRef = useRef<TaskWatcher | null>(null)
  const watchedJobIdRef = useRef<string | null>(null)
  const watchedStorageKeyRef = useRef<string | null>(null)
  const watchedSubmissionRef = useRef<WandSubmission | null>(null)
  const resolvedStorageKeyRef = useRef<string | null>(null)
  const activeTaskKeyRef = useRef<string | null>(null)
  const submittingWandRef = useRef(false)
  const promptLockedRef = useRef(false)
  const promptRef = useRef(params.prompt)
  const promptRevisionRef = useRef(0)
  // Snapshot of server-side wand settings so we only PUT /api/settings (an
  // admin-only endpoint) when the user actually changed something here.
  const [serverWand, setServerWand] = useState<{ model: string; backend: string; llm: string; device: string } | null>(null)

  useEffect(() => {
    apiFetch.settings()
      .then(settings => {
        setWandModel(settings.local_qwen_model_id ?? '')
        setWandBackend(settings.prompt_expander_backend ?? 'local')
        setLocalLlmBackend(settings.local_llm_backend ?? 'comfy')
        setWandDevice(settings.local_qwen_device ?? 'auto')
        setServerWand({
          model: settings.local_qwen_model_id ?? '',
          backend: settings.prompt_expander_backend ?? 'local',
          llm: settings.local_llm_backend ?? 'comfy',
          device: settings.local_qwen_device ?? 'auto',
        })
      })
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    if (promptRef.current === params.prompt) return
    promptRef.current = params.prompt
    promptRevisionRef.current += 1
  }, [params.prompt])

  const writePrompt = useCallback((prompt: string, trackUserRevision = true) => {
    return applyGuardedPromptMutation(promptLockedRef.current, value => {
      if (trackUserRevision) promptRevisionRef.current += 1
      promptRef.current = value
      setParam('prompt', value)
    }, prompt)
  }, [setParam])

  useEffect(() => {
    let disposed = false
    let inFlight = false
    let rerunRequested = false
    let refreshTimer: ReturnType<typeof setTimeout> | null = null
    const resolveAuth = () => {
      if (disposed) return
      if (inFlight) {
        rerunRequested = true
        return
      }
      inFlight = true
      apiFetch.authMe()
        .then(session => {
          if (!disposed) {
            const nextKey = activeGpuTaskStorageKey(session?.username, WAND_TASK_KIND)
            if (
              activeTaskKeyRef.current === null
              && watcherRef.current
              && watchedStorageKeyRef.current === null
              && watchedSubmissionRef.current
            ) {
              persistWandTask(nextKey, watchedSubmissionRef.current)
              watchedStorageKeyRef.current = nextKey
            }
            activeTaskKeyRef.current = nextKey
            setActiveTaskKey(nextKey)
          }
        })
        .catch(() => undefined)
        .finally(() => {
          inFlight = false
          if (disposed || !rerunRequested) return
          rerunRequested = false
          refreshTimer = setTimeout(resolveAuth, 75)
        })
    }
    const scheduleAuthRefresh = () => {
      if (disposed) return
      if (refreshTimer !== null) clearTimeout(refreshTimer)
      refreshTimer = setTimeout(resolveAuth, 75)
    }
    const onVisible = () => {
      if (document.visibilityState === 'visible') scheduleAuthRefresh()
    }
    resolveAuth()
    window.addEventListener('online', scheduleAuthRefresh)
    window.addEventListener('focus', scheduleAuthRefresh)
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      disposed = true
      if (refreshTimer !== null) clearTimeout(refreshTimer)
      window.removeEventListener('online', scheduleAuthRefresh)
      window.removeEventListener('focus', scheduleAuthRefresh)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [])

  const watchWandTask = useCallback((
    submission: WandSubmission,
    storageKey: string | null,
    initialSnapshot?: GpuTaskResponse<WandResult>,
  ) => {
    watcherRef.current?.stop()
    watchedJobIdRef.current = submission.jobId
    watchedStorageKeyRef.current = storageKey
    watchedSubmissionRef.current = submission
    persistWandTask(storageKey, submission)
    dispatchWand({ type: 'enqueued', submission })
    if (initialSnapshot) dispatchWand({ type: 'snapshot', snapshot: initialSnapshot })

    promptLockedRef.current = initialSnapshot?.status === 'running'
      || initialSnapshot?.status === 'finalizing'
    setPromptBusy(true)
    const watcher = createTaskWatcher<WandResult>({
      jobId: submission.jobId,
      fetchSnapshot: () => apiFetch.jobStatus(submission.jobId) as Promise<GpuTaskResponse<WandResult>>,
      openSocket: (onSnapshot, onClose) => connectWS(
        submission.jobId,
        data => onSnapshot(data as Partial<GpuTaskResponse<WandResult>>),
        onClose,
      ),
      onSnapshot: snapshot => {
        if (
          watchedJobIdRef.current !== submission.jobId
          || !canDeliverTaskResult(activeTaskKeyRef.current, watchedStorageKeyRef.current)
        ) return
        promptLockedRef.current = promptLockedRef.current
          || snapshot.status === 'running'
          || snapshot.status === 'finalizing'
        setPromptBusy(true)
        dispatchWand({ type: 'snapshot', snapshot })
      },
      onTerminal: snapshot => {
        if (
          watchedJobIdRef.current !== submission.jobId
          || !canDeliverTaskResult(activeTaskKeyRef.current, watchedStorageKeyRef.current)
        ) return
        clearPersistedWandTask(watchedStorageKeyRef.current, submission.jobId)
        watcherRef.current = null
        watchedJobIdRef.current = null
        watchedStorageKeyRef.current = null
        watchedSubmissionRef.current = null
        promptLockedRef.current = false
        setPromptBusy(false)

        if (snapshot.status === 'done' && snapshot.result) {
          dispatchWand({
            type: 'completed',
            currentPrompt: promptRef.current,
            currentRevision: promptRevisionRef.current,
            result: snapshot.result,
          })
          if (snapshot.result.error) {
            setNotice({ severity: 'warning', message: snapshot.result.error })
          } else if (!snapshot.result.changed || !snapshot.result.expanded) {
            setNotice({ severity: 'warning', message: 'The wand did not return a different prompt.' })
          }
          return
        }

        dispatchWand({ type: 'terminal' })
        if (snapshot.status === 'cancelled') {
          setNotice({ severity: 'warning', message: 'Magic Wand was cancelled.' })
        } else if (snapshot.status === 'blocked') {
          setNotice({ severity: 'warning', message: snapshot.error || 'Magic Wand was blocked by the safety filter.' })
        } else {
          setNotice({ severity: 'error', message: snapshot.error || 'Prompt expansion failed.' })
        }
      },
      onConnectionNote: note => {
        if (
          note
          && canDeliverTaskResult(activeTaskKeyRef.current, watchedStorageKeyRef.current)
        ) setNotice({ severity: 'warning', message: note })
      },
      onError: error => {
        if (
          watchedJobIdRef.current !== submission.jobId
          || !canDeliverTaskResult(activeTaskKeyRef.current, watchedStorageKeyRef.current)
        ) return
        clearPersistedWandTask(watchedStorageKeyRef.current, submission.jobId)
        watcherRef.current = null
        watchedJobIdRef.current = null
        watchedStorageKeyRef.current = null
        watchedSubmissionRef.current = null
        promptLockedRef.current = false
        setPromptBusy(false)
        dispatchWand({ type: 'terminal' })
        setNotice({ severity: 'error', message: error.message })
      },
      acknowledgeAfterDelivery: snapshot =>
        snapshot.status === 'done'
          ? apiFetch.ackJob(submission.jobId).then(() => undefined).catch(() => undefined)
          : undefined,
    })
    watcherRef.current = watcher
    watcher.start()
  }, [setPromptBusy])

  useEffect(() => {
    if (!activeTaskKey) return
    const transition = reconcileActiveTaskIdentity({
      previousResolvedKey: resolvedStorageKeyRef.current,
      nextResolvedKey: activeTaskKey,
      watcherActive: watcherRef.current !== null,
      watchedStorageKey: watchedStorageKeyRef.current,
    })
    resolvedStorageKeyRef.current = activeTaskKey

    if (transition.stopWatcher) {
      watcherRef.current?.stop()
      watcherRef.current = null
      watchedJobIdRef.current = null
      watchedStorageKeyRef.current = null
      watchedSubmissionRef.current = null
      promptLockedRef.current = false
    }
    if (transition.identityChanged) {
      setPromptBusy(false)
      setPreWandPrompt('')
      setMoodboardSuggestions([])
      setNotice(null)
      dispatchWand({ type: 'identity-changed' })
    }

    if (transition.adoptStorageKey) {
      const submission = watchedSubmissionRef.current
      if (submission) {
        persistWandTask(transition.adoptStorageKey, submission)
        watchedStorageKeyRef.current = transition.adoptStorageKey
      }
      return
    }

    if (!transition.consultStorageKey) return
    const raw = localStorage.getItem(transition.consultStorageKey)
    if (!raw) return
    const submission = parsePersistedWandTask(raw)
    if (!submission) {
      localStorage.removeItem(transition.consultStorageKey)
      return
    }
    promptRevisionRef.current = Math.max(promptRevisionRef.current, submission.submittedRevision)
    watchWandTask(submission, transition.consultStorageKey)
  }, [activeTaskKey, setMoodboardSuggestions, setPromptBusy, watchWandTask])

  useEffect(() => () => {
    watcherRef.current?.stop()
    promptLockedRef.current = false
    setPromptBusy(false)
  }, [setPromptBusy])

  useLayoutEffect(() => {
    if (!wandState.autoApplied) return
    const { originalPrompt, result } = wandState.autoApplied
    if (!writePrompt(result.expanded, false)) return
    setPreWandPrompt(originalPrompt)
    setMoodboardSuggestions(result.suggested_moodboards ?? [])
    const label = result.backend === 'openrouter'
      ? 'OpenRouter'
      : result.backend === 'ideogram-json'
        ? 'Ideogram JSON'
        : localLlmBackend === 'gguf_server'
          ? 'local GGUF helper'
          : localLlmBackend === 'comfy'
            ? 'ComfyUI QwenVL'
            : 'Local Qwen3-VL'
    setNotice({ severity: 'success', message: `Prompt expanded with ${label}.` })
    dispatchWand({ type: 'auto-applied' })
  }, [localLlmBackend, setMoodboardSuggestions, wandState.autoApplied, writePrompt])

  const handleExpand = async () => {
    if (!promptRef.current || submittingWandRef.current || watcherRef.current) return
    submittingWandRef.current = true
    setSubmittingWand(true)
    try {
      const wandChanged = !serverWand
        || serverWand.model !== wandModel || serverWand.backend !== wandBackend
        || serverWand.llm !== localLlmBackend || serverWand.device !== wandDevice
      if (wandChanged) {
        // Settings PUT is admin-only in share mode. Non-admins still get the
        // wand: the chosen backend rides along in the expand request instead.
        try {
          await apiFetch.updateSettings({
            prompt_expander_backend: wandBackend,
            local_llm_backend: localLlmBackend,
            local_qwen_model_id: wandModel,
            local_qwen_device: wandDevice,
          })
          setServerWand({ model: wandModel, backend: wandBackend, llm: localLlmBackend, device: wandDevice })
        } catch {
          setNotice({ severity: 'warning', message: 'Wand settings are admin-only; using the server defaults for this expansion.' })
        }
      }
      const submittedPrompt = promptRef.current
      const submittedRevision = promptRevisionRef.current
      const queued = await apiFetch.submitExpandPrompt(submittedPrompt, wandBackend)
      const submission = { jobId: queued.job_id, submittedPrompt, submittedRevision }
      watchWandTask(submission, activeTaskKeyRef.current, {
        job_id: queued.job_id,
        status: queued.status,
        progress: 0,
        images: [],
        task_kind: queued.task_kind,
        queue_position: queued.queue_position,
        queue_length: queued.queue_length,
      })
    } catch (err) {
      setNotice({ severity: 'error', message: err instanceof Error ? err.message : 'Prompt expansion failed.' })
    } finally {
      submittingWandRef.current = false
      setSubmittingWand(false)
    }
  }

  const cancelWand = async () => {
    const active = wandState.active
    if (!active || active.cancelPending || !['queued', 'running'].includes(active.snapshot.status)) return
    dispatchWand({ type: 'cancel-requested' })
    try {
      await apiFetch.cancelJob(active.submission.jobId)
    } catch {
      dispatchWand({ type: 'cancel-failed' })
      setNotice({ severity: 'error', message: 'Could not cancel Magic Wand.' })
    }
  }

  const undoMagicWand = () => {
    if (!preWandPrompt) return
    if (!writePrompt(preWandPrompt)) return
    setPreWandPrompt('')
    setNotice({ severity: 'success', message: 'Restored the prompt from before Magic Wand.' })
  }

  const applyPendingWand = () => {
    const pending = wandState.pending
    if (!pending) return
    const undoSource = promptRef.current
    if (!writePrompt(pending.result.expanded)) return
    setPreWandPrompt(undoSource)
    setMoodboardSuggestions(pending.result.suggested_moodboards ?? [])
    dispatchWand({ type: 'discard-pending' })
    setNotice({ severity: 'success', message: 'Applied the completed Magic Wand result.' })
  }

  const copyPendingWand = async () => {
    const expanded = wandState.pending?.result.expanded
    if (!expanded) return
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable')
      await navigator.clipboard.writeText(expanded)
      setNotice({ severity: 'success', message: 'Magic Wand result copied.' })
    } catch {
      setNotice({ severity: 'error', message: 'Could not copy the Magic Wand result to the clipboard.' })
    }
  }

  const applyMoodboardSuggestion = (board: MoodboardSuggestion) => {
    const ids = Array.from(new Set([...params.selected_moodboard_ids, board.id]))
    const uuids = board.uuid ? Array.from(new Set([...params.moodboard_uuids, board.uuid])) : params.moodboard_uuids
    setParams({
      selected_moodboard_ids: ids,
      moodboard_uuids: uuids,
      moodboard_strength: params.moodboard_strength || 0.35,
    })
    setNotice({ severity: 'success', message: `Applied moodboard "${board.title}".` })
  }

  const removeMoodboard = (id: number, uuid = '') => {
    setParams({
      selected_moodboard_ids: params.selected_moodboard_ids.filter(existing => existing !== id),
      moodboard_uuids: uuid ? params.moodboard_uuids.filter(existing => existing !== uuid) : params.moodboard_uuids,
    })
  }

  const activeMoodboards = params.selected_moodboard_ids.map(id => {
    const suggestion = moodboardSuggestions.find(board => board.id === id)
    return { id, title: suggestion?.title ?? `Moodboard #${id}`, uuid: suggestion?.uuid ?? '' }
  })

  const handlePlan = async () => {
    if (!params.prompt || planning) return
    setPlanning(true)
    try {
      const result = await apiFetch.planPrompt(params.prompt, params.prompt_planner_max_tokens)
      setPlan(result)
      setParam('prompt_planner_show_output', true)
      if (result.error) {
        setNotice({ severity: 'warning', message: result.error })
      } else {
        const label = result.backend === 'local' ? 'Local Qwen3-VL' : 'heuristic fallback'
        setNotice({ severity: 'success', message: `Prompt planned with ${label}.` })
      }
    } catch (err) {
      setNotice({ severity: 'error', message: err instanceof Error ? err.message : 'Prompt planning failed.' })
    } finally {
      setPlanning(false)
    }
  }

  // Xperiment setup lives in QuickPresets (the "Xperiment" chip); the old
  // duplicate handler here was never wired to any UI and has been removed.

  const activeWand = wandState.active
  const wandStatus = activeWand?.snapshot.status
  const wandButtonLabel = submittingWand
    ? 'Submitting Magic Wand…'
    : wandStatus === 'queued'
      ? activeWand?.snapshot.queue_position
        ? `Queued #${activeWand.snapshot.queue_position}`
        : 'Queued'
      : wandStatus === 'cancellation_requested'
        ? 'Cancelling Magic Wand…'
        : activeWand
          ? 'Magic Wand running…'
          : 'Magic wand'
  const canCancelWand = !!activeWand && ['queued', 'running'].includes(activeWand.snapshot.status)
  const promptLocked = !!activeWand?.promptLocked
  const wandAnnouncement = activeWand
    ? wandStatusAnnouncement(activeWand.snapshot, activeWand.cancelPending)
    : ''

  return (
    <Stack spacing={1}>
      <Paper variant="outlined" sx={{ p: 1.25, bgcolor: 'background.default' }}>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ xs: 'stretch', sm: 'center' }} justifyContent="space-between">
          <Box>
            <Typography variant="subtitle2">Create prompt from image</Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Upload an image and the local Qwen3-VL reverse-engineers a prompt to recreate it. Add optional guidance to focus on — or change — specific parts; leave it blank for a full auto prompt.
            </Typography>
          </Box>
          <CreatePromptFromImage
            value={params.prompt}
            onChange={prompt => { writePrompt(prompt) }}
            disabled={promptLocked}
            label="Create from image"
            withGuidance
          />
        </Stack>
      </Paper>
      <Stack spacing={1}>
        <TextField
          label="Prompt"
          multiline
          minRows={3}
          maxRows={8}
          fullWidth
          value={params.prompt}
          onChange={e => { writePrompt(e.target.value) }}
          disabled={promptLocked}
          placeholder="Describe the image you want to create…"
        />
        {activeMoodboards.length > 0 && (
          <Paper variant="outlined" sx={{ p: 1, borderColor: 'rgba(208,188,255,0.35)', bgcolor: 'rgba(208,188,255,0.06)' }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.75 }}>
              Active moodboards are applied as text guidance on generate:
            </Typography>
            <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
              {activeMoodboards.map(board => (
                <Chip
                  key={board.id}
                  size="small"
                  color="primary"
                  label={board.title}
                  onDelete={() => removeMoodboard(board.id, board.uuid)}
                />
              ))}
            </Stack>
          </Paper>
        )}
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={1} alignItems={{ xs: 'stretch', md: 'flex-start' }}>
          <Button
            variant="outlined"
            onClick={handleExpand}
            disabled={submittingWand || !!activeWand || !params.prompt}
            startIcon={submittingWand || !!activeWand ? <CircularProgress size={16} color="inherit" /> : <AutoFixHighIcon />}
            sx={{ alignSelf: { xs: 'stretch', md: 'flex-start' }, minHeight: 40 }}
          >
            {wandButtonLabel}
          </Button>
          {canCancelWand && (
            <Button
              variant="text"
              color="warning"
              onClick={cancelWand}
              disabled={activeWand.cancelPending}
              aria-label={wandCancelAriaLabel(activeWand.cancelPending)}
              sx={{ alignSelf: { xs: 'stretch', md: 'flex-start' }, minHeight: 40 }}
            >
              {activeWand.cancelPending ? 'Cancelling…' : 'Cancel Wand'}
            </Button>
          )}
          <Tooltip
            title={
              <span>
                <b>Krea 2 prompting tips:</b><br />
                Use natural language; describe the scene as to a person.<br />
                Long, detailed prompts work best. Put requested text in quotes.
              </span>
            }
          >
            <Button variant="text" size="small" startIcon={<TipsAndUpdatesIcon fontSize="small" />} sx={{ alignSelf: { xs: 'stretch', md: 'flex-start' }, minHeight: 40 }}>
              Prompt tips
            </Button>
          </Tooltip>
          {preWandPrompt && (
            <Button
              variant="text"
              size="small"
              onClick={undoMagicWand}
              disabled={promptLocked}
              sx={{ alignSelf: { xs: 'stretch', md: 'flex-start' }, minHeight: 40 }}
            >
              Undo Magic Wand
            </Button>
          )}
        </Stack>
        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: -0.25 }}>
          Magic Wand expands your prompt and can suggest matching moodboards. Applying a suggestion makes it visible above.
        </Typography>
        {activeWand && (
          <Paper variant="outlined" sx={{ p: 1, borderColor: 'rgba(208,188,255,0.35)', bgcolor: 'rgba(208,188,255,0.05)' }}>
            <Stack spacing={0.75}>
              <Typography
                variant="caption"
                color="text.secondary"
                role="status"
                aria-live="polite"
                aria-atomic="true"
              >
                {wandAnnouncement}
              </Typography>
              <LinearProgress
                variant={activeWand.snapshot.status === 'running' && activeWand.snapshot.progress > 0 ? 'determinate' : 'indeterminate'}
                value={activeWand.snapshot.progress}
                aria-label={wandProgressAriaLabel(activeWand.snapshot.status)}
                sx={{ height: 3, borderRadius: 2 }}
              />
            </Stack>
          </Paper>
        )}
        {wandState.pending && (
          <Paper variant="outlined" sx={{ p: 1.25, borderColor: 'warning.main', bgcolor: 'rgba(255,183,77,0.06)' }}>
            <Stack spacing={1}>
              <Alert severity="warning" variant="outlined">
                Magic Wand finished, but your prompt changed while it was queued. Review the result before applying it.
              </Alert>
              <TextField label="Original submission" value={wandState.pending.originalPrompt} multiline minRows={2} size="small" fullWidth InputProps={{ readOnly: true }} />
              <TextField label="Current prompt when completed" value={wandState.pending.currentPrompt} multiline minRows={2} size="small" fullWidth InputProps={{ readOnly: true }} />
              <TextField label="Magic Wand result" value={wandState.pending.result.expanded} multiline minRows={3} size="small" fullWidth InputProps={{ readOnly: true }} />
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
                <Button variant="contained" size="small" onClick={applyPendingWand} disabled={promptLocked}>Apply result</Button>
                <Button variant="outlined" size="small" onClick={copyPendingWand}>Copy result</Button>
                <Button variant="text" size="small" onClick={() => dispatchWand({ type: 'discard-pending' })}>Discard</Button>
              </Stack>
            </Stack>
          </Paper>
        )}
        {moodboardSuggestions.length > 0 && (
          <Paper variant="outlined" sx={{ p: 1, borderColor: 'rgba(202,196,208,0.18)', bgcolor: 'rgba(255,255,255,0.03)' }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.75 }}>
              Magic Wand found moodboards that fit this prompt. Apply one to add its text guidance; image influence remains opt-in in Image Prompt.
            </Typography>
            <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
              {moodboardSuggestions.map(board => (
                <Tooltip key={board.id} title={board.reason || 'Apply this moodboard'} arrow>
                  <Chip
                    clickable
                    size="small"
                    label={board.title}
                    color={params.selected_moodboard_ids.includes(board.id) ? 'primary' : 'default'}
                    variant={params.selected_moodboard_ids.includes(board.id) ? 'filled' : 'outlined'}
                    onClick={() => applyMoodboardSuggestion(board)}
                  />
                </Tooltip>
              ))}
            </Stack>
          </Paper>
        )}
      </Stack>
      <Box>
        <Button size="small" variant="text" onClick={() => setShowPromptTools(v => !v)} sx={{ minHeight: 32 }}>
          {showPromptTools ? 'Hide prompt tools' : 'Prompt tools — planner, expression steering & negative prompt'}
        </Button>
        <Collapse in={showPromptTools}>
          <Stack spacing={1} sx={{ pt: 0.5 }}>
      <Paper variant="outlined" sx={{ p: 1.5, bgcolor: 'background.default' }}>
        <Stack spacing={1}>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ sm: 'center' }}>
            <FormControlLabel
              control={<Checkbox checked={params.use_prompt_planner} onChange={e => setParam('use_prompt_planner', e.target.checked)} />}
              label="Enhance for adherence"
            />
            <FormControlLabel
              control={<Checkbox checked={params.use_prompt_expander} onChange={e => setParam('use_prompt_expander', e.target.checked)} />}
              label="Expand on generate"
            />
            <FormControlLabel
              control={<Checkbox checked={params.prompt_planner_lock_original} onChange={e => setParam('prompt_planner_lock_original', e.target.checked)} />}
              label="Lock original prompt"
            />
            <FormControlLabel
              control={<Checkbox checked={params.prompt_planner_use_regions} onChange={e => setParam('prompt_planner_use_regions', e.target.checked)} />}
              label="Use planner for regions"
            />
            <Button size="small" variant="outlined" onClick={handlePlan} disabled={planning || !params.prompt}>
              {planning ? 'Planning…' : 'Show planner output'}
            </Button>
          </Stack>
          <Box sx={{ px: 1 }}>
            <Typography variant="caption" color="text.secondary">Max planner tokens: {params.prompt_planner_max_tokens}</Typography>
            <Slider
              min={128}
              max={1600}
              step={64}
              size="small"
              value={params.prompt_planner_max_tokens}
              onChange={(_, value) => setParam('prompt_planner_max_tokens', value as number)}
            />
          </Box>
          <Collapse in={params.prompt_planner_show_output && !!plan}>
            {plan ? (
              <Stack spacing={1}>
                <TextField label="Planned prompt" value={plan.planned_prompt} multiline minRows={3} size="small" fullWidth InputProps={{ readOnly: true }} />
                {plan.negative_prompt ? <TextField label="Planner negative prompt" value={plan.negative_prompt} size="small" fullWidth InputProps={{ readOnly: true }} /> : null}
                <Typography variant="caption" color="text.secondary">
                  Subject: {plan.subject || 'n/a'} · Composition: {plan.composition || 'n/a'} · Lighting: {plan.lighting || 'n/a'}
                </Typography>
                <Stack direction="row" spacing={1}>
                  <Button size="small" variant="contained" disabled={promptLocked} onClick={() => {
                    if (!writePrompt(plan.planned_prompt)) return
                    if (plan.negative_prompt && !params.negative_prompt) setParam('negative_prompt', plan.negative_prompt)
                  }}>
                    Apply planned prompt
                  </Button>
                  <Button size="small" onClick={() => setParam('prompt_planner_show_output', false)}>Hide</Button>
                </Stack>
              </Stack>
            ) : null}
          </Collapse>
        </Stack>
      </Paper>
      <Paper variant="outlined" sx={{ p: 1.5, bgcolor: 'background.default' }}>
        <Stack spacing={1}>
          <FormControlLabel
            control={<Checkbox checked={params.think_steering_enabled} onChange={e => setParam('think_steering_enabled', e.target.checked)} />}
            label="Expression steering (<think> block)"
          />
          <Typography variant="caption" color="text.secondary" sx={{ px: 1, mt: -0.5 }}>
            Restores emotion/intensity that Turbo's distillation flattens, in-distribution — a gentler alternative to the Emotion rebalance preset. Leave the text blank to use the default expression nudge.
          </Typography>
          <Collapse in={params.think_steering_enabled}>
            <TextField
              label="Think text (optional)"
              value={params.think_text}
              onChange={e => setParam('think_text', e.target.value)}
              multiline
              minRows={2}
              maxRows={4}
              fullWidth
              size="small"
              placeholder="e.g. show genuine fear and tension, dramatic lighting…"
            />
          </Collapse>
        </Stack>
      </Paper>
      <TextField
        label="Negative prompt"
        multiline
        minRows={1}
        maxRows={3}
        fullWidth
        value={params.negative_prompt}
        onChange={e => setParam('negative_prompt', e.target.value)}
        placeholder="What to avoid (optional; Turbo usually leaves this empty)…"
        size="small"
      />
          </Stack>
        </Collapse>
      </Box>
      <Snackbar
        open={!!notice}
        autoHideDuration={5000}
        onClose={() => setNotice(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        {notice ? (
          <Alert severity={notice.severity} variant="filled" onClose={() => setNotice(null)}>
            {notice.message}
          </Alert>
        ) : undefined}
      </Snackbar>
    </Stack>
  )
}
