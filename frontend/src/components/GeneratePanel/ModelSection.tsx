import React from 'react'
import { Box, Chip, Stack, Tooltip, Typography } from '@mui/material'
import { useStore } from '../../store'

const CHECKPOINTS = [
  {
    id: 'turbo', label: 'Turbo',
    desc: '8 steps · up to 2048px · distilled for speed · CFG=0',
  },
  {
    id: 'raw', label: 'RAW',
    desc: '52 steps · up to 1024px · training-quality detail · CFG=3.5',
  },
]

const QUANTS = [
  { id: 'bf16', label: 'bf16', desc: '~16–20 GB VRAM · full precision' },
  { id: 'fp8',  label: 'fp8',  desc: '~8–13 GB VRAM · quantized' },
]

export default function ModelSection() {
  const { params, setParam, setParams, systemReport } = useStore()
  const loaded = systemReport?.model_status?.loaded
  const loadedCp = systemReport?.model_status?.checkpoint ?? ''

  return (
    <Box>
      <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block', textTransform: 'uppercase', letterSpacing: 1 }}>
        Model
      </Typography>
      <Stack spacing={1.5}>
        <Stack direction="row" spacing={1} flexWrap="wrap">
          {CHECKPOINTS.map(c => (
            <Tooltip key={c.id} title={c.desc} placement="top" arrow>
              <Chip
                label={c.label}
                variant={params.checkpoint === c.id ? 'filled' : 'outlined'}
                color={params.checkpoint === c.id ? 'primary' : 'default'}
                onClick={() => {
                  setParam('checkpoint', c.id as 'turbo' | 'raw')
                  if (c.id === 'turbo') setParams({ steps: 8, cfg: 0.0, mu: 1.15 })
                  else setParams({ steps: 52, cfg: 3.5, mu: 0 })
                }}
                clickable
              />
            </Tooltip>
          ))}
        </Stack>
        <Stack direction="row" spacing={1}>
          {QUANTS.map(q => (
            <Tooltip key={q.id} title={q.desc} placement="top" arrow>
              <Chip
                label={q.label}
                variant={params.quantization === q.id ? 'filled' : 'outlined'}
                color={params.quantization === q.id ? 'secondary' : 'default'}
                onClick={() => setParam('quantization', q.id as 'bf16' | 'fp8')}
                clickable
                size="small"
              />
            </Tooltip>
          ))}
        </Stack>
        {loaded && loadedCp && (
          <Typography variant="caption" sx={{ color: 'success.main', wordBreak: 'break-all' }}>
            Loaded: {loadedCp.split(/[\\/]/).pop()}
          </Typography>
        )}
      </Stack>
    </Box>
  )
}
