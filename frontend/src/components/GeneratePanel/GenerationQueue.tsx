import { useCallback, useEffect, useRef, useState } from 'react'
import { Alert, Box, Chip, CircularProgress, IconButton, LinearProgress, Paper, Snackbar, Stack, Tooltip, Typography } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import QueueMusicIcon from '@mui/icons-material/PlaylistPlay'
import { apiFetch, type QueueJob } from '../../api'
import { useStore } from '../../store'

const ACTIVE = new Set(['queued', 'running'])

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
  const [jobs, setJobs] = useState<QueueJob[]>([])
  const [toast, setToast] = useState('')
  const timer = useRef<number | undefined>(undefined)

  const poll = useCallback(async () => {
    try {
      const list = await apiFetch.jobs(24)
      setJobs(list)
      const active = list.some(j => ACTIVE.has(j.status))
      timer.current = window.setTimeout(poll, active ? 1800 : 6000)
    } catch {
      timer.current = window.setTimeout(poll, 8000)
    }
  }, [])

  useEffect(() => {
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
      if (timer.current) window.clearTimeout(timer.current)
      document.removeEventListener('visibilitychange', onWake)
      window.removeEventListener('online', onWake)
    }
  }, [poll])

  if (jobs.length === 0) return null
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
        setResults(full.images, full.seed, full.metadata)
        setJobId(j.job_id)
      }
    } catch {
      setToast('Could not open this result — it may have been cleaned up on the server.')
    }
  }

  const cancel = async (e: React.MouseEvent, j: QueueJob) => {
    e.stopPropagation()
    try {
      await apiFetch.cancelJob(j.job_id)
    } catch {
      setToast('Could not cancel this job.')
    }
    if (timer.current) window.clearTimeout(timer.current)
    poll()
  }

  return (
    <Paper variant="outlined" sx={{ p: 1.25, borderColor: 'rgba(202,196,208,0.18)' }}>
      <Stack direction="row" alignItems="center" spacing={0.75} sx={{ mb: 1 }}>
        <QueueMusicIcon fontSize="small" sx={{ color: 'text.secondary' }} />
        <Typography variant="caption" sx={{ color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 1 }}>
          Queue{activeCount ? ` · ${activeCount} active` : ''}{myNextPosition ? ` · you're #${myNextPosition}` : ''}
        </Typography>
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
          const canCancel = !foreign && j.status === 'queued'
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
                    {j.summary || 'Generation'}{j.is_batch ? ` · batch ${j.batch_count ?? ''}` : ''}
                  </Typography>
                  <Tooltip title={j.error || (j.status === 'running' && j.progress < 3 ? 'Loading the abliterated text encoder + diffusion model into VRAM' : '')} arrow disableHoverListener={!j.error && !(j.status === 'running' && j.progress < 3)}>
                    <Chip size="small" label={chip.label} color={chip.color} variant={chip.color === 'default' ? 'outlined' : 'filled'} sx={{ height: 20 }} />
                  </Tooltip>
                  {canCancel && (
                    <IconButton size="small" onClick={e => cancel(e, j)} sx={{ p: 0.25, minWidth: 36, minHeight: 36 }} aria-label="Cancel job">
                      <CloseIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  )}
                </Stack>
                {(j.status === 'running' || j.status === 'queued') && (
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
