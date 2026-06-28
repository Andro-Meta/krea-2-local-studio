import React, { useEffect, useState } from 'react'
import { Alert, Box, Chip, Grid, Slider, Stack, TextField, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { useStore } from '../../store'
import { apiFetch } from '../../api'

// Fallback grid mirrors backend resolution.py (long side = tier; 16-aligned).
const FALLBACK_DIMS: Record<string, Record<string, [number, number]>> = {
  '1k': {
    '1:1': [1024, 1024], '4:3': [1024, 768], '3:4': [768, 1024], '3:2': [1024, 688],
    '2:3': [688, 1024], '16:9': [1024, 576], '9:16': [576, 1024], '21:9': [1024, 432],
  },
  '2k': {
    '1:1': [2048, 2048], '4:3': [2048, 1536], '3:4': [1536, 2048], '3:2': [2048, 1360],
    '2:3': [1360, 2048], '16:9': [2048, 1152], '9:16': [1152, 2048], '21:9': [2048, 880],
  },
}
const ASPECTS = ['1:1', '4:3', '3:4', '3:2', '2:3', '16:9', '9:16', '21:9']

export default function DimensionSection() {
  const { params, setParam, setParams } = useStore()
  const [dims, setDims] = useState(FALLBACK_DIMS)
  const [advice, setAdvice] = useState<{ blocks_to_swap: number; fits: boolean; free_vram_gb: number | null } | null>(null)

  useEffect(() => {
    apiFetch.resolutionOptions()
      .then(r => { if (r?.dimensions) setDims(r.dimensions) })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (params.resolution_tier !== '2k') { setAdvice(null); return }
    let cancelled = false
    apiFetch.runtimeAdvice(params.width, params.height, params.quantization)
      .then(a => { if (!cancelled) setAdvice({ blocks_to_swap: a.blocks_to_swap, fits: a.fits, free_vram_gb: a.free_vram_gb }) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [params.resolution_tier, params.width, params.height, params.quantization])

  const applyTierAspect = (tier: '1k' | '2k', aspect: string) => {
    const pair = dims[tier]?.[aspect] ?? FALLBACK_DIMS[tier][aspect] ?? [1024, 1024]
    setParams({ resolution_tier: tier, aspect_ratio: aspect, width: pair[0], height: pair[1] })
  }

  const isActiveAspect = (aspect: string) => {
    const pair = dims[params.resolution_tier]?.[aspect]
    return params.aspect_ratio === aspect && !!pair && params.width === pair[0] && params.height === pair[1]
  }

  return (
    <Box>
      <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block', textTransform: 'uppercase', letterSpacing: 1 }}>Dimensions</Typography>
      <Stack spacing={1.5}>
        <ToggleButtonGroup
          size="small"
          exclusive
          value={params.resolution_tier}
          onChange={(_, tier) => tier && applyTierAspect(tier, params.aspect_ratio)}
          fullWidth
        >
          <ToggleButton value="1k">1K</ToggleButton>
          <ToggleButton value="2k">2K</ToggleButton>
        </ToggleButtonGroup>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
          {ASPECTS.map(aspect => (
            <Chip
              key={aspect}
              label={aspect}
              size="small"
              variant={isActiveAspect(aspect) ? 'filled' : 'outlined'}
              color={isActiveAspect(aspect) ? 'primary' : 'default'}
              onClick={() => applyTierAspect(params.resolution_tier, aspect)}
              clickable
            />
          ))}
        </Box>
        {params.resolution_tier === '2k' && (
          <Alert severity={advice && !advice.fits ? 'warning' : 'info'} sx={{ py: 0 }}>
            2K is ~4× the pixels of 1K — much slower and more VRAM-hungry (attention scales with token count). Turbo holds a fixed shift at 2K; RAW uses resolution-adaptive shift.
            {advice && (
              advice.fits
                ? ` Estimated to fit${advice.free_vram_gb != null ? ` (~${advice.free_vram_gb}GB free)` : ''}${advice.blocks_to_swap ? `; recommend loading with ~${advice.blocks_to_swap} block-swap.` : '.'}`
                : ` May not fit${advice.free_vram_gb != null ? ` (~${advice.free_vram_gb}GB free)` : ''} — load with ~${advice.blocks_to_swap} block-swap (System tab), use fp8, or lower the resolution.`
            )}
          </Alert>
        )}
        <Grid container spacing={1.5}>
          <Grid item xs={6}>
            <TextField
              label="Width"
              type="number"
              value={params.width}
              onChange={e => setParam('width', Math.max(256, Math.min(2048, Number(e.target.value))))}
              size="small"
              fullWidth
              inputProps={{ step: 16 }}
              helperText="Aligned to 16"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              label="Height"
              type="number"
              value={params.height}
              onChange={e => setParam('height', Math.max(256, Math.min(2048, Number(e.target.value))))}
              size="small"
              fullWidth
              inputProps={{ step: 16 }}
              helperText="Aligned to 16"
            />
          </Grid>
        </Grid>
        <Stack direction="row" spacing={2} alignItems="center">
          <Typography variant="body2" sx={{ minWidth: 60, color: 'text.secondary' }}>Batch</Typography>
          <Slider
            value={params.num_images}
            min={1} max={4} step={1}
            onChange={(_, v) => setParam('num_images', v as number)}
            marks valueLabelDisplay="auto"
            sx={{ flex: 1 }}
          />
          <Typography variant="body2" sx={{ minWidth: 16 }}>{params.num_images}</Typography>
        </Stack>
      </Stack>
    </Box>
  )
}
