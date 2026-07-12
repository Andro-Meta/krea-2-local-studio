import { useCallback, useEffect, useRef, useState } from 'react'
import { Alert, Box, Chip, CircularProgress, IconButton, LinearProgress, Paper, Snackbar, Stack, Tooltip, Typography } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import QueueMusicIcon from '@mui/icons-material/PlaylistPlay'
import { apiFetch, type GpuTaskKind, type QueueJob } from '../../api'
import { reconcilePendingCancellations } from '../../lib/queueState'
import { useStore } from '../../store'

const ACTIVE = new Set(['queued', 'running', 'cancellation_requested', 'finalizing'])

const TASK_LABELS: Record<GpuTaskKind, string> = {
  generation: 'Generation',
  prompt_expand: 'Magic Wand',
  prompt_plan: 'Prompt planner',
  image_describe: 'Image description',
  upscale: 'Upscale',
  depth_preview: 'Depth preview',
  moodboard_guidance: 'Moodboard guidance',
  background_enrichment: 'Moodboard guidance',
  model_warmup: 'Generation',
}

function jobLabel(job: QueueJob): string {
  if (!job.mine) return job.summary
  const kind = job.task_kind ?? 'generation'
  const label = TASK_LABELS[kind]
  return kind === 'generation' && job.summary ? `${label} · ${job.summary}` : label
}

/** "12:19 AM" for today, "Jul 10, 12:19 AM" for older — phones coming back
    after hours need to know WHEN a job actually finished. */
function jobTime(unixSeconds?: number | null): string {
  if (!unixSeconds) return ''
  const d = new Date(unixSeconds * 1000)
  const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  return d.toDateString() === new Date().toDateString()
    ? time
    : `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })}, ${time}`
}

function jobTimeline(j: QueueJob): string {
  if (j.status === 'running') return j.started_at ? `started ${jobTime(j.started_at)}` : ''
  if (j.status === 'queued') return j.queued_at ? `queued ${jobTime(j.queued_at)}` : ''
  if (!j.finished_at) return ''
  const label = j.status === 'done' ? 'finished' : j.status
  const duration = j.started_at && j.finished_at > j.started_at
    ? ` · ${Math.round(j.finished_at - j.started_at)}s`
    : ''
  return `${label} ${jobTime(j.finished_at)}${duration}`
}

type Sev = 'default' | 'primary' | 'success' | 'error' | 'warning'
function statusChip(j: QueueJob): { label: string; color: Sev } {
  switch (j.status) {
    case 'queued': return { label: j.queue_position ? `Queued #${j.queue_position}` : 'Queued', color: 'default' }
    // Running at ~0% is the model/encoder load phase (the abliterated text encoder
    // + diffusion model streaming into VRAM) before sampling steps report progress.
    case 'running': return { label: j.progress < 3 ? 'Loading models…' : `Running ${j.progress}%`, color: 'primary' }
    case 'cancellation_requested': return { label: 'Cancelling…', color: 'warning' }
    case 'finalizing': return { label: 'Finalizing…', color: 'primary' }
    case 'done': return { label: 'Done', color: 'success' }
    case 'error': return { label: 'Error', color: 'error' }
    case 'blocked': return { label: 'Blocked', color: 'warning' }
    case 'cancelled': return { label: 'Cancelled', color: 'default' }
    default: return { label: j.status || '—', color: 'default' }
  }
}

export default function GenerationQueue() {
  const setResults = useStore(s => s.setResults)
  const setJobId = useStore(s => s.setJobId)
  const admission = useStore(s => s.admission)
  const setAdmission = useStore(s => s.setAdmission)
  const [jobs, setJobs] = useState<QueueJob[]>([])
  const [toast, setToast] = useState('')
  const [cancelling, setCancelling] = useState<Set<string>>(() => new Set())
  const cancellingRef = useRef<Set<string>>(new Set())
  const timer = useRef<number | undefined>(undefined)
  const mounted = useRef(false)
  const polling = useRef(false)
  const pollAgain = useRef(false)

  const poll = useCallback(async () => {
    if (!mounted.current || polling.current) return
    polling.current = true
    const schedulePoll = (delayMs: number) => {
      if (!mounted.current) return
      if (timer.current) window.clearTimeout(timer.current)
      timer.current = window.setTimeout(poll, delayMs)
    }
    try {
      const response = await apiFetch.jobs(24)
      if (!mounted.current) return
      setJobs(response.jobs)
      setAdmission(response.admission)
      setCancelling(current => {
        const next = reconcilePendingCancellations(current, response.jobs)
        cancellingRef.current = next
        return next
      })
      const active = response.jobs.some(j => ACTIVE.has(j.status))
      schedulePoll(active ? 1800 : 6000)
    } catch {
      schedulePoll(8000)
    } finally {
      polling.current = false
      if (pollAgain.current && mounted.current) {
        pollAgain.current = false
        if (timer.current) window.clearTimeout(timer.current)
        timer.current = window.setTimeout(poll, 0)
      }
    }
  }, [setAdmission])

  useEffect(() => {
    mounted.current = true
    poll()
    // Phones suspend timers when the screen is off; refresh the moment the
    // page is visible again instead of waiting for the stale timer.
    const onWake = () => {
      if (document.visibilityState === 'visible') {
        if (timer.current) window.clearTimeout(timer.current)
        poll()
      }
    }
    document.addEventListener('visibilitychange', onWake)
    window.addEventListener('online', onWake)
    return () => {
      mounted.current = false
      pollAgain.current = false
      if (timer.current) window.clearTimeout(timer.current)
      timer.current = undefined
      document.removeEventListener('visibilitychange', onWake)
      window.removeEventListener('online', onWake)
    }
  }, [poll])

  if (jobs.length === 0 && !admission) return null
  const activeCount = jobs.filter(j => ACTIVE.has(j.status)).length
  // Your next spot in line (projected across all users' queued work).
  const myNextPosition = jobs
    .filter(j => j.mine !== false && j.status === 'queued' && j.queue_position)
    .reduce<number | null>((min, j) => (min === null || (j.queue_position as number) < min ? (j.queue_position as number) : min), null)

  const openJob = async (j: QueueJob) => {
    if (j.mine === false || j.status !== 'done') return
    try {
      const full = await apiFetch.jobStatus(j.job_id)
      if (full.images?.length) {
        setResults(full.images, full.seed ?? undefined, full.metadata)
        setJobId(j.job_id)
      }
    } catch {
      setToast('Could not open this result — it may have been cleaned up on the server.')
    }
  }

  const cancel = async (e: React.MouseEvent, j: QueueJob) => {
    e.stopPropagation()
    if (cancellingRef.current.has(j.job_id)) return
    const pending = new Set(cancellingRef.current).add(j.job_id)
    cancellingRef.current = pending
    setCancelling(pending)
    try {
      await apiFetch.cancelJob(j.job_id)
    } catch {
      setToast('Could not cancel this job.')
      setCancelling(current => {
        const next = new Set(current)
        next.delete(j.job_id)
        cancellingRef.current = next
        return next
      })
      return
    }
    if (timer.current) window.clearTimeout(timer.current)
    if (polling.current) {
      pollAgain.current = true
    } else {
      void poll()
    }
  }

  return (
    <Paper variant="outlined" sx={{ p: 1.25, borderColor: 'rgba(202,196,208,0.18)' }}>
      <Stack direction="row" alignItems="center" spacing={0.75} sx={{ mb: 1 }}>
        <QueueMusicIcon fontSize="small" sx={{ color: 'text.secondary' }} />
        <Typography variant="caption" sx={{ color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 1 }}>
          Queue{activeCount ? ` · ${activeCount} active` : ''}{myNextPosition ? ` · you're #${myNextPosition}` : ''}
        </Typography>
        {admission && (
          <Chip
            size="small"
            color={admission.per_user_active >= admission.per_user_limit ? 'warning' : 'default'}
            variant="outlined"
            label={`${admission.per_user_active}/${admission.per_user_limit} task slots in use`}
            sx={{ ml: 'auto', height: 22 }}
          />
        )}
      </Stack>
      {jobs.some(j => j.mine === false) && (
        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: 0.75 }}>
          Grayed entries are other users' jobs — only their queue position is shown. Turns rotate fairly between users.
        </Typography>
      )}
      <Stack spacing={0.75}>
        {jobs.map(j => {
          const foreign = j.mine === false
          const chip = statusChip(j)
          const canOpen = !foreign && j.status === 'done'
          const canCancel = !foreign && (j.status === 'queued' || j.status === 'running')
          const cancelPending = cancelling.has(j.job_id)
          return (
            <Box
              key={j.job_id}
              onClick={() => openJob(j)}
              sx={{
                display: 'flex', alignItems: 'center', gap: 1, p: 0.75, borderRadius: 1.5,
                border: '1px solid rgba(202,196,208,0.12)', cursor: canOpen ? 'pointer' : 'default',
                '&:hover': canOpen ? { borderColor: 'rgba(202,196,208,0.35)' } : undefined,
              }}
            >
              <Box sx={{ width: 40, height: 40, borderRadius: 1, overflow: 'hidden', flexShrink: 0,
                         bgcolor: 'rgba(0,0,0,0.25)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {j.thumb
                  ? <Box component="img" src={j.thumb} alt="" sx={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  : j.status === 'running'
                    ? <CircularProgress size={16} />
                    : <Typography variant="caption" sx={{ color: 'text.disabled' }}>{j.is_batch ? `×${j.batch_count ?? ''}` : '—'}</Typography>}
              </Box>
              <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                <Stack direction="row" spacing={0.75} alignItems="center">
                  <Typography variant="body2" noWrap sx={{ flexGrow: 1, color: foreign ? 'text.disabled' : 'text.primary', fontStyle: foreign ? 'italic' : 'normal' }}>
                    {jobLabel(j)}{j.is_batch ? ` · batch ${j.batch_count ?? ''}` : ''}
                  </Typography>
                  <Tooltip title={j.error || (j.status === 'running' && j.progress < 3 ? 'Loading the abliterated text encoder + diffusion model into VRAM' : '')} arrow disableHoverListener={!j.error && !(j.status === 'running' && j.progress < 3)}>
                    <Chip size="small" label={chip.label} color={chip.color} variant={chip.color === 'default' ? 'outlined' : 'filled'} sx={{ height: 20 }} />
                  </Tooltip>
                  {canCancel && (
                    <IconButton size="small" onClick={e => cancel(e, j)} disabled={cancelPending} sx={{ p: 0.25, minWidth: 36, minHeight: 36 }} aria-label={cancelPending ? 'Cancelling job' : 'Cancel job'}>
                      {cancelPending ? <CircularProgress size={14} /> : <CloseIcon sx={{ fontSize: 16 }} />}
                    </IconButton>
                  )}
                </Stack>
                {ACTIVE.has(j.status) && (
                  <LinearProgress
                    variant={j.status === 'running' && j.progress >= 3 ? 'determinate' : 'indeterminate'}
                    value={j.progress}
                    sx={{ mt: 0.5, height: 3, borderRadius: 2 }}
                  />
                )}
                {(jobTimeline(j) || (j.seed != null && j.status === 'done')) && (
                  <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                    {[jobTimeline(j), j.seed != null && j.status === 'done' ? `seed ${j.seed}` : ''].filter(Boolean).join(' · ')}
                  </Typography>
                )}
              </Box>
            </Box>
          )
        })}
      </Stack>
      <Snackbar open={!!toast} autoHideDuration={5000} onClose={() => setToast('')}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
        <Alert severity="error" onClose={() => setToast('')} variant="filled">{toast}</Alert>
      </Snackbar>
    </Paper>
  )
}
