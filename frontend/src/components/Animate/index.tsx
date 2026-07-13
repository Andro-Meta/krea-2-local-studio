import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Paper,
  Stack,
  Tab,
  Tabs,
  Typography,
} from '@mui/material'
import AutoAwesomeMotionOutlinedIcon from '@mui/icons-material/AutoAwesomeMotionOutlined'
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import {
  apiFetch,
  connectWS,
  type AnimateRequest,
  type AnimationResult,
  type AnimationUploadResponse,
  type AppSettings,
  type AuthSession,
} from '../../api'
import {
  activeGpuTaskStorageKey,
  clearPersistedActiveTask,
  persistActiveTask,
  readPersistedActiveTask,
} from '../../lib/activeTaskPersistence'
import { consumeAnimationResultHandoff } from '../../lib/animationResultHandoff'
import { clearConsumedVideoUpload, submissionFailureKeepsUpload } from '../../lib/animationRuntime'
import {
  initialAnimateTaskState,
  reduceAnimateTaskState,
  type AnimateSubmission,
} from '../../lib/animateTaskState'
import { createTaskWatcher, type TaskWatcher } from '../../lib/taskWatcher'
import { normalizeKreaDeforumStatus } from '../../lib/kreaDeforumStatus'
import { TAB, useStore } from '../../store'
import AdvancedControls from './AdvancedControls'
import AnimationProgress from './AnimationProgress'
import BasicControls from './BasicControls'
import MotionControls from './MotionControls'
import {
  ANIMATE_DEFAULTS,
  ANIMATE_PRESETS,
  applyMotionPreset,
  calculateRenderedFrames,
  validateAnimateRequest,
  type AnimatePresetId,
  type MotionPresetId,
} from './presets'
import TimelineControls from './TimelineControls'
import VideoResult from './VideoResult'

type Section = 'basic' | 'motion' | 'timeline' | 'advanced'

const SECTION_LABELS: Record<Section, string> = {
  basic: 'Basic',
  motion: 'Motion',
  timeline: 'Timeline',
  advanced: 'Advanced',
}

function errorMessage(error: unknown, fallback: string): string {
  const value = error as { response?: { data?: { detail?: string } }; message?: string }
  return value.response?.data?.detail || value.message || fallback
}

function safeStoredResult(raw: string | null): AnimationResult | null {
  if (!raw) return null
  try {
    const value = JSON.parse(raw) as AnimationResult
    return value && typeof value.video_url === 'string' && typeof value.poster_url === 'string'
      ? value
      : null
  } catch {
    return null
  }
}

async function imageFileToPng(file: File, width: number, height: number): Promise<string> {
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
    throw new Error('Choose a PNG, JPEG, or WebP starting image.')
  }
  if (file.size < 1 || file.size > 16 * 1024 * 1024) {
    throw new Error('Starting images must be smaller than 16 MB.')
  }
  const source = URL.createObjectURL(file)
  try {
    const image = new Image()
    image.src = source
    await image.decode()
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext('2d')
    if (!context) throw new Error('This browser cannot prepare the starting image.')
    context.drawImage(image, 0, 0, width, height)
    return canvas.toDataURL('image/png')
  } finally {
    URL.revokeObjectURL(source)
  }
}

export default function AnimatePanel() {
  const [form, setForm] = useState<AnimateRequest>({ ...ANIMATE_DEFAULTS })
  const [preset, setPreset] = useState<AnimatePresetId>('custom')
  const [section, setSection] = useState<Section>('basic')
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [auth, setAuth] = useState<AuthSession | null>(null)
  const [identityReady, setIdentityReady] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [submissionError, setSubmissionError] = useState('')
  const [initPreview, setInitPreview] = useState('')
  const [initError, setInitError] = useState('')
  const [upload, setUpload] = useState<AnimationUploadResponse | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadError, setUploadError] = useState('')
  const [connectionNote, setConnectionNote] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [task, dispatch] = useReducer(reduceAnimateTaskState, initialAnimateTaskState)
  const watcherRef = useRef<TaskWatcher | null>(null)
  const watchedStorageKeyRef = useRef<string | null>(null)
  const formRevisionRef = useRef(0)
  const deliveredJobsRef = useRef(new Set<string>())
  const setTab = useStore(state => state.setTab)

  const username = auth?.authenticated ? auth.username : auth?.share_auth === false ? 'local' : null
  const storageKey = identityReady && username
    ? activeGpuTaskStorageKey(username, 'animation')
    : null
  const resultStorageKey = storageKey ? `${storageKey}:last-result` : null
  const metaStorageKey = storageKey ? `${storageKey}:meta` : null
  const limits = settings?.animation
  const runtimeStatus = useMemo(
    () => normalizeKreaDeforumStatus(settings?.krea_deforum),
    [settings?.krea_deforum],
  )
  const errors = useMemo(
    () => validateAnimateRequest(form, {
      maxFrames: limits?.max_frames,
      maxDimension: limits?.max_dimension,
    }),
    [form, limits],
  )
  const work = calculateRenderedFrames(form, limits?.max_frames)
  const active = !!task.active
  const runtimeReady = runtimeStatus.available
  const invalid = Object.keys(errors).length > 0 || !!initError
  const unavailableReason = loadError
    ? 'Settings are unavailable.'
    : !identityReady
      ? 'Checking your session…'
      : !runtimeReady
        ? 'KreaDeforum setup is required.'
        : form.animation_mode === '3D' && !runtimeStatus.midas_ready
          ? 'MiDaS setup is required for 3D.'
        : uploading
          ? 'Wait for the source upload to finish.'
          : active
            ? 'An animation is already active.'
            : invalid
              ? 'Fix the highlighted fields.'
              : ''
  const canSubmit = !unavailableReason && !submitting

  const stopWatcher = useCallback(() => {
    watcherRef.current?.stop()
    watcherRef.current = null
    watchedStorageKeyRef.current = null
  }, [])

  const clearActivePersistence = useCallback((jobId: string, key: string | null) => {
    if (key) {
      clearPersistedActiveTask(localStorage, key, jobId)
      localStorage.removeItem(`${key}:meta`)
    }
  }, [])

  const clearVideoUploadState = useCallback(() => {
    setForm(current => ({ ...current, source_video_upload_id: '' }))
    setUpload(null)
    setUploadProgress(0)
  }, [])

  const startWatcher = useCallback((submission: AnimateSubmission, key: string | null) => {
    stopWatcher()
    watchedStorageKeyRef.current = key
    const watcher = createTaskWatcher<AnimationResult>({
      jobId: submission.jobId,
      fetchSnapshot: () => apiFetch.jobStatus<AnimationResult>(submission.jobId),
      openSocket: (onSnapshot, onClose) => connectWS(submission.jobId, data => {
        if (data && typeof data === 'object') onSnapshot(data)
      }, onClose),
      onSnapshot: snapshot => dispatch({ type: 'snapshot', snapshot }),
      onConnectionNote: setConnectionNote,
      onError: error => {
        clearVideoUploadState()
        dispatch({ type: 'terminal', jobId: submission.jobId, status: 'error', error: error.message })
        clearActivePersistence(submission.jobId, key)
      },
      onTerminal: snapshot => {
        clearVideoUploadState()
        if (snapshot.status === 'done') {
          const result = snapshot.result
          if (!result?.video_url || !result.poster_url) {
            throw new Error('Animation finished without a usable video result.')
          }
          if (key) localStorage.setItem(`${key}:last-result`, JSON.stringify(result))
          dispatch({
            type: 'delivered',
            jobId: submission.jobId,
            formRevision: submission.formRevision,
            result,
          })
          deliveredJobsRef.current.add(submission.jobId)
        } else if (snapshot.status === 'error' || snapshot.status === 'blocked' || snapshot.status === 'cancelled') {
          dispatch({
            type: 'terminal',
            jobId: submission.jobId,
            status: snapshot.status,
            error: snapshot.error || undefined,
          })
        }
        clearActivePersistence(submission.jobId, key)
      },
      acknowledgeAfterDelivery: snapshot => (
        snapshot.status === 'done' && deliveredJobsRef.current.has(submission.jobId)
          ? apiFetch.ackJob(submission.jobId).then(() => undefined)
          : undefined
      ),
    })
    watcherRef.current = watcher
    watcher.start()
  }, [clearActivePersistence, clearVideoUploadState, stopWatcher])

  useEffect(() => {
    let cancelled = false
    Promise.allSettled([apiFetch.settings(), apiFetch.authMe()]).then(([settingsResult, authResult]) => {
      if (cancelled) return
      if (settingsResult.status === 'fulfilled') setSettings(settingsResult.value)
      else setLoadError('Could not load Animate settings. Restart Krea and try again.')
      if (authResult.status === 'fulfilled') setAuth(authResult.value)
      else setAuth({ authenticated: false, share_auth: false, username: 'local', role: 'admin' })
      setIdentityReady(true)
    })
    return () => {
      cancelled = true
      stopWatcher()
    }
  }, [stopWatcher])

  useEffect(() => {
    if (!identityReady || !storageKey) return
    dispatch({ type: 'identity-changed' })
    const previous = safeStoredResult(resultStorageKey ? localStorage.getItem(resultStorageKey) : null)
    if (previous) dispatch({ type: 'hydrate-result', result: previous })
    const handoff = consumeAnimationResultHandoff(localStorage, username)
    if (handoff) {
      if (resultStorageKey) localStorage.setItem(resultStorageKey, JSON.stringify(handoff.result))
      dispatch({ type: 'hydrate-result', result: handoff.result })
      void apiFetch.ackJob(handoff.job_id).catch(() => undefined)
    }
    const jobId = readPersistedActiveTask(localStorage, storageKey)
    if (!jobId) return
    let revision = 0
    let videoTransferred = false
    try {
      const meta = JSON.parse(localStorage.getItem(metaStorageKey || '') || '{}') as { formRevision?: number; videoTransferred?: boolean }
      if (Number.isSafeInteger(meta.formRevision) && (meta.formRevision ?? -1) >= 0) revision = meta.formRevision!
      videoTransferred = meta.videoTransferred === true
    } catch {
      // The active job id is authoritative; malformed optional metadata falls back to revision zero.
    }
    const submission = { jobId, formRevision: revision, videoTransferred }
    dispatch({ type: 'restored', submission })
    startWatcher(submission, storageKey)
  }, [identityReady, storageKey, resultStorageKey, metaStorageKey, startWatcher, username])

  const update = useCallback(<K extends keyof AnimateRequest>(key: K, value: AnimateRequest[K]) => {
    formRevisionRef.current += 1
    setPreset('custom')
    setForm(current => {
      if (key === 'animation_mode' && value !== 'Video Input') {
        setUpload(null)
        return { ...current, [key]: value, source_video_upload_id: '' }
      }
      return { ...current, [key]: value }
    })
  }, [])

  const applyPreset = (next: AnimatePresetId) => {
    formRevisionRef.current += 1
    setPreset(next)
    if (next === 'custom') return
    setForm(current => ({
      ...ANIMATE_PRESETS[next],
      prompt_schedule: current.prompt_schedule,
      negative_prompt: current.negative_prompt,
      init_image_b64: current.init_image_b64,
      source_video_upload_id: current.animation_mode === 'Video Input' ? current.source_video_upload_id : '',
      animation_mode: current.animation_mode,
    }))
  }

  const applyMotion = (next: MotionPresetId) => {
    if (next === 'slow_orbit' && !runtimeStatus.midas_ready) return
    try {
      const updated = applyMotionPreset(form, next)
      formRevisionRef.current += 1
      setSubmissionError('')
      setForm(updated)
    } catch {
      setSubmissionError('Fix duration, FPS, or rendered-frame errors before applying a motion preset.')
    }
  }

  const chooseInitImage = async (file: File | null) => {
    setInitError('')
    if (!file) {
      setInitPreview('')
      update('init_image_b64', '')
      return
    }
    try {
      const encoded = await imageFileToPng(file, form.width, form.height)
      setInitPreview(encoded)
      update('init_image_b64', encoded)
    } catch (error) {
      setInitError(errorMessage(error, 'Could not prepare the starting image.'))
    }
  }

  const chooseVideo = async (file: File | null) => {
    setUploadError('')
    if (!file) {
      setUpload(null)
      update('source_video_upload_id', '')
      return
    }
    const allowed = limits?.upload_content_types ?? []
    if (form.animation_mode !== 'Video Input') {
      setUploadError('Select Video Input mode before uploading.')
      return
    }
    if (allowed.length && !allowed.includes(file.type)) {
      setUploadError(`Unsupported video type. Use ${allowed.join(', ')}.`)
      return
    }
    if (limits && file.size > limits.max_upload_bytes) {
      setUploadError(`Video exceeds the ${Math.round(limits.max_upload_bytes / 1024 / 1024)} MB limit.`)
      return
    }
    setUploading(true)
    setUploadProgress(0)
    try {
      const response = await apiFetch.uploadAnimationSource(file, setUploadProgress)
      setUpload(response)
      update('source_video_upload_id', response.upload_id)
    } catch (error) {
      setUploadError(errorMessage(error, 'Video upload failed.'))
    } finally {
      setUploading(false)
    }
  }

  const focusFirstError = () => {
    const first = Object.keys(errors)[0]
    const id = first ? `animate-${first.replace(/_/g, '-')}` : ''
    const target = id ? document.getElementById(id) : null
    target?.focus()
  }

  const submit = async () => {
    if (!canSubmit || !storageKey) {
      focusFirstError()
      return
    }
    setSubmitting(true)
    setSubmissionError('')
    dispatch({ type: 'clear-error' })
    if (form.animation_mode === '3D' && !runtimeStatus.midas_ready) {
      setSubmissionError(runtimeStatus.midas_reason)
      setSubmitting(false)
      return
    }
    const submittedUpload = form.source_video_upload_id
    try {
      let initImage = form.init_image_b64
      if (initPreview) {
        const response = await fetch(initPreview)
        const file = new File([await response.blob()], 'start.png', { type: 'image/png' })
        initImage = await imageFileToPng(file, form.width, form.height)
      }
      const request: AnimateRequest = {
        ...form,
        init_image_b64: initImage,
        source_video_upload_id: form.animation_mode === 'Video Input' ? form.source_video_upload_id : '',
      }
      const currentErrors = validateAnimateRequest(request, {
        maxFrames: limits?.max_frames,
        maxDimension: limits?.max_dimension,
      })
      if (Object.keys(currentErrors).length) {
        focusFirstError()
        return
      }
      const queued = await apiFetch.animate(request)
      const consumed = clearConsumedVideoUpload(request)
      setForm(current => ({ ...current, source_video_upload_id: '' }))
      setUpload(null)
      setUploadProgress(0)
      const submission = {
        jobId: queued.job_id,
        formRevision: formRevisionRef.current,
        videoTransferred: consumed.videoTransferred,
      }
      persistActiveTask(localStorage, storageKey, submission.jobId)
      localStorage.setItem(`${storageKey}:meta`, JSON.stringify({
        formRevision: submission.formRevision,
        videoTransferred: submission.videoTransferred,
      }))
      dispatch({ type: 'enqueued', submission })
      startWatcher(submission, storageKey)
    } catch (error) {
      if (submittedUpload && !submissionFailureKeepsUpload(error)) clearVideoUploadState()
      setSubmissionError(errorMessage(error, 'Could not queue the animation.'))
    } finally {
      setSubmitting(false)
    }
  }

  const cancel = async () => {
    const jobId = task.active?.submission.jobId
    if (!jobId || task.active?.cancelPending) return
    dispatch({ type: 'cancel-requested' })
    try {
      await apiFetch.cancelJob(jobId)
      watcherRef.current?.wake()
    } catch (error) {
      dispatch({ type: 'cancel-failed' })
      setConnectionNote(errorMessage(error, 'Could not request cancellation.'))
    }
  }

  const download = async () => {
    if (!task.result) return
    setDownloading(true)
    try {
      const blob = await apiFetch.downloadOwnedMedia(task.result.video_url)
      const href = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = href
      link.download = `krea-animation-${encodeURIComponent(String(task.result.gallery_id))}.mp4`
      link.click()
      URL.revokeObjectURL(href)
    } catch (error) {
      setConnectionNote(errorMessage(error, 'Could not download the video.'))
    } finally {
      setDownloading(false)
    }
  }

  return (
    <Box sx={{ maxWidth: 1120, mx: 'auto', px: { xs: 1.5, sm: 2 }, py: 2, pb: { xs: 12, md: 3 }, overflowX: 'clip' }}>
      <Stack spacing={2}>
        <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ sm: 'center' }} gap={1}>
          <Box>
            <Stack direction="row" spacing={1} alignItems="center">
              <AutoAwesomeMotionOutlinedIcon color="primary" />
              <Typography variant="h4" component="h1">Animate</Typography>
              <Chip size="small" color="warning" variant="outlined" label="Experimental" />
            </Stack>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, maxWidth: 680 }}>
              Build short KreaDeforum animations from prompt keyframes, camera schedules, a starting frame, or source video.
            </Typography>
          </Box>
          <Chip
            label={!settings ? 'Checking readiness…' : runtimeReady ? 'Ready' : 'Setup needed'}
            color={!settings ? 'default' : runtimeReady ? 'success' : 'warning'}
          />
        </Stack>

        {!runtimeReady && settings && (
          <Alert severity="warning">
            KreaDeforum nodes or the Krea 2 chunking patch are unavailable. Run <b>install.bat</b>, then restart ComfyUI.
            {!!runtimeStatus.missing_nodes.length && ` Missing: ${runtimeStatus.missing_nodes.join(', ')}.`}
            {!!runtimeStatus.incompatible_capabilities.length && ` Incompatible: ${runtimeStatus.incompatible_capabilities.join(', ')}.`}
          </Alert>
        )}
        {runtimeReady && !runtimeStatus.midas_ready && (
          <Alert severity="info">
            2D animation is ready. 3D is disabled: {runtimeStatus.midas_reason}
          </Alert>
        )}
        {loadError && <Alert severity="error" role="alert">{loadError}</Alert>}
        {submissionError && <Alert severity="error" role="alert">{submissionError}</Alert>}
        {auth?.role === 'child' && (
          <Alert severity="info">
            Child-account prompts, starting images, and sampled video frames are safety checked. Blocked media stays private and may be available to an admin for review.
          </Alert>
        )}
        {initError && <Alert severity="error" role="alert">{initError}</Alert>}

        <Paper variant="outlined" sx={{ overflow: 'hidden' }}>
          <Tabs
            value={section}
            onChange={(_, value: Section) => setSection(value)}
            variant="scrollable"
            scrollButtons="auto"
            aria-label="Animation settings sections"
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            {(Object.keys(SECTION_LABELS) as Section[]).map(id => (
              <Tab
                key={id}
                value={id}
                label={
                  <Stack alignItems="center" spacing={0.25}>
                    <span>{SECTION_LABELS[id]}</span>
                    <Typography variant="caption" color="text.secondary" noWrap>
                      {id === 'basic' ? `${form.duration_seconds}s · ${form.width}×${form.height}` :
                        id === 'motion' ? `${form.animation_mode} · cadence ${form.diffusion_cadence}` :
                          id === 'timeline' ? 'Raw schedules' : `${form.steps} steps · ${form.sampler_name}`}
                    </Typography>
                  </Stack>
                }
                sx={{ minHeight: 64, minWidth: { xs: 132, sm: 160 } }}
              />
            ))}
          </Tabs>
          <Box sx={{ p: { xs: 1.5, sm: 2.5 } }}>
            {section === 'basic' && (
              <BasicControls
                value={form} errors={errors} update={update} disabled={submitting}
                preset={preset} onPreset={applyPreset}
                initPreview={initPreview} onInitImage={chooseInitImage}
              />
            )}
            {section === 'motion' && (
              <MotionControls
                value={form} errors={errors} update={update} disabled={submitting}
                upload={upload} uploading={uploading} uploadProgress={uploadProgress}
                uploadError={uploadError} allowedVideoTypes={limits?.upload_content_types ?? []}
                onVideo={chooseVideo} runtime={runtimeStatus} onMotionPreset={applyMotion}
              />
            )}
            {section === 'timeline' && <TimelineControls value={form} errors={errors} update={update} disabled={submitting} />}
            {section === 'advanced' && (
              <AdvancedControls value={form} errors={errors} update={update} disabled={submitting} runtime={runtimeStatus} />
            )}
          </Box>
        </Paper>

        <AnimationProgress task={task} connectionNote={connectionNote} onCancel={cancel} />
        {task.result && (
          <VideoResult
            result={task.result}
            downloading={downloading}
            onDownload={download}
            onOpenGallery={() => setTab(TAB.GALLERY)}
          />
        )}

        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          justifyContent="space-between"
          alignItems={{ sm: 'center' }}
          gap={1}
          sx={{
            position: { xs: 'sticky', md: 'static' },
            bottom: { xs: 72, md: 'auto' },
            zIndex: 10,
            bgcolor: { xs: 'background.paper', md: 'transparent' },
            border: { xs: '1px solid', md: 0 },
            borderColor: 'divider',
            borderRadius: 2,
            p: { xs: 1.25, md: 0 },
            boxShadow: { xs: 6, md: 0 },
          }}
        >
          <Typography variant="caption" color={unavailableReason ? 'warning.main' : 'text.secondary'} aria-live="polite">
            {unavailableReason || `${work.frames} frames · about ${work.diffusionFrames} diffusion calls`}
          </Typography>
          <Button
            variant="contained"
            size="large"
            onClick={submit}
            disabled={!canSubmit}
            startIcon={submitting ? <CircularProgress size={18} color="inherit" /> : <AutoAwesomeMotionOutlinedIcon />}
            sx={{ minHeight: 48, minWidth: { sm: 200 } }}
          >
            {submitting ? 'Queueing…' : 'Queue animation'}
          </Button>
        </Stack>
      </Stack>
    </Box>
  )
}
