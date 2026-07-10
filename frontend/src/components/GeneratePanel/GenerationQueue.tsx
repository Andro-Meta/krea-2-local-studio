import { useCallback, useEffect, useRef, useState } from 'react'
import { Box, Chip, CircularProgress, IconButton, LinearProgress, Paper, Stack, Tooltip, Typography } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import QueueMusicIcon from '@mui/icons-material/PlaylistPlay'
import { apiFetch, type QueueJob } from '../../api'
import { useStore } from '../../store'

const ACTIVE = new Set(['queued', 'running'])

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
    return () => { if (timer.current) window.clearTimeout(timer.current) }
  }, [poll])

  if (jobs.length === 0) return null
  const activeCount = jobs.filter(j => ACTIVE.has(j.status)).length

  const openJob = async (j: QueueJob) => {
    if (j.status !== 'done') return
    try {
      const full = await apiFetch.jobStatus(j.job_id)
      if (full.images?.length) {
        setResults(full.images, full.seed, full.metadata)
        setJobId(j.job_id)
      }
    } catch { /* ignore */ }
  }

  const cancel = async (e: React.MouseEvent, j: QueueJob) => {
    e.stopPropagation()
    try { await apiFetch.cancelJob(j.job_id) } catch { /* ignore */ }
    if (timer.current) window.clearTimeout(timer.current)
    poll()
  }

  return (
    <Paper variant="outlined" sx={{ p: 1.25, borderColor: 'rgba(202,196,208,0.18)' }}>
      <Stack direction="row" alignItems="center" spacing={0.75} sx={{ mb: 1 }}>
        <QueueMusicIcon fontSize="small" sx={{ color: 'text.secondary' }} />
        <Typography variant="caption" sx={{ color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 1 }}>
          Queue{activeCount ? ` · ${activeCount} active` : ''}
        </Typography>
      </Stack>
      <Stack spacing={0.75}>
        {jobs.map(j => {
          const chip = statusChip(j)
          const canOpen = j.status === 'done'
          const canCancel = j.status === 'queued'
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
                  <Typography variant="body2" noWrap sx={{ flexGrow: 1 }}>
                    {j.summary || 'Generation'}{j.is_batch ? ` · batch ${j.batch_count ?? ''}` : ''}
                  </Typography>
                  <Tooltip title={j.error || (j.status === 'running' && j.progress < 3 ? 'Loading the abliterated text encoder + diffusion model into VRAM' : '')} arrow disableHoverListener={!j.error && !(j.status === 'running' && j.progress < 3)}>
                    <Chip size="small" label={chip.label} color={chip.color} variant={chip.color === 'default' ? 'outlined' : 'filled'} sx={{ height: 20 }} />
                  </Tooltip>
                  {canCancel && (
                    <IconButton size="small" onClick={e => cancel(e, j)} sx={{ p: 0.25 }}>
                      <CloseIcon sx={{ fontSize: 15 }} />
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
                {j.seed != null && j.status === 'done' && (
                  <Typography variant="caption" sx={{ color: 'text.disabled' }}>seed {j.seed}</Typography>
                )}
              </Box>
            </Box>
          )
        })}
      </Stack>
    </Paper>
  )
}
