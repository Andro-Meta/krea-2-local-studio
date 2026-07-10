import { useEffect, useState } from 'react'
import { Alert, Box, Button, Chip, LinearProgress, Paper, Stack, Typography } from '@mui/material'
import OpenInNewIcon from '@mui/icons-material/OpenInNew'
import { publicUrl } from '../../api'
import type { LabCaseRun } from './labRunner'

function imageSrc(value: string): string {
  if (!value) return ''
  if (value.startsWith('data:') || value.startsWith('http')) return value
  return `data:image/png;base64,${value}`
}

function firstFilename(item: LabCaseRun): string {
  const metadata = item.job?.metadata?.[0] as any
  return metadata?.filename || ''
}

function statusColor(status: LabCaseRun['status']): 'default' | 'primary' | 'success' | 'error' | 'warning' | 'info' {
  if (status === 'done') return 'success'
  if (status === 'error' || status === 'blocked') return 'error'
  if (status === 'skipped' || status === 'cancelled') return 'warning'
  if (status === 'running' || status === 'queued') return 'info'
  return 'default'
}

export default function CompareGrid({ cases }: { cases: LabCaseRun[] }) {
  const active = cases.some(item => item.status === 'running' || item.status === 'queued')
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    if (!active) return
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [active])

  if (!cases.length) return null

  return (
    <Stack spacing={1.5}>
      <Typography variant="subtitle2" sx={{ color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 1 }}>
        Results
      </Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))', xl: 'repeat(4, minmax(0, 1fr))' }, gap: 1.25 }}>
        {cases.map(item => {
          const image = item.job?.images?.[0] ?? ''
          const filename = firstFilename(item)
          const metadata = item.job?.metadata?.[0] as any
          const isActive = item.status === 'running' || item.status === 'queued'
          const liveElapsed = isActive && item.startedAt ? (now - item.startedAt) / 1000 : null
          const pct = Math.max(0, Math.min(100, Math.round(item.job?.progress ?? 0)))
          return (
            <Paper key={item.caseId} variant="outlined" sx={{ overflow: 'hidden', bgcolor: 'rgba(255,255,255,0.02)' }}>
              <Box sx={{ aspectRatio: '1 / 1', bgcolor: 'rgba(0,0,0,0.25)', display: 'grid', placeItems: 'center' }}>
                {image ? (
                  <Box component="img" src={imageSrc(image)} alt={item.label} sx={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : isActive ? (
                  <Stack spacing={1} sx={{ width: '80%', alignItems: 'center' }}>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                      {item.status === 'queued' ? 'Queued…' : `Rendering… ${liveElapsed != null ? liveElapsed.toFixed(0) : 0}s`}
                    </Typography>
                    <Box sx={{ width: '100%' }}>
                      <LinearProgress variant={pct > 0 ? 'determinate' : 'indeterminate'} value={pct} />
                    </Box>
                    {pct > 0 && <Typography variant="caption" sx={{ color: 'text.disabled' }}>{pct}% (per stage)</Typography>}
                  </Stack>
                ) : (
                  <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                    {item.status === 'idle' ? 'Not run yet' : item.status}
                  </Typography>
                )}
              </Box>
              <Stack spacing={0.75} sx={{ p: 1 }}>
                <Stack direction="row" justifyContent="space-between" gap={1} alignItems="center">
                  <Typography variant="body2" sx={{ fontWeight: 700 }} noWrap title={item.label}>{item.label}</Typography>
                  <Chip size="small" color={statusColor(item.status)} label={item.status} />
                </Stack>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  {item.elapsedSec != null
                    ? `${item.elapsedSec.toFixed(1)}s`
                    : liveElapsed != null ? `${liveElapsed.toFixed(0)}s elapsed…` : 'queued timing pending'}
                  {metadata?.seed != null ? ` · seed ${metadata.seed}` : ''}
                </Typography>
                <Typography variant="caption" sx={{ color: 'text.disabled' }} noWrap title={`${item.request.sampler}/${item.request.scheduler}`}>
                  {item.request.checkpoint} · {item.request.sampler}/{item.request.scheduler} · {item.request.steps} steps · CFG {item.request.cfg}
                </Typography>
                {item.error && <Alert severity="error" sx={{ py: 0 }}>{item.error}</Alert>}
                {filename && (
                  <Button size="small" variant="text" endIcon={<OpenInNewIcon sx={{ fontSize: 14 }} />} href={publicUrl(`/api/outputs/${encodeURIComponent(filename)}`)} target="_blank" rel="noreferrer">
                    Open output
                  </Button>
                )}
              </Stack>
            </Paper>
          )
        })}
      </Box>
    </Stack>
  )
}
