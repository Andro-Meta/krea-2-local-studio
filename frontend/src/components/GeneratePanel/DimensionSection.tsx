import React, { useEffect, useState } from 'react'
import { Alert, Box, Grid, Slider, Stack, TextField, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
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
const align16 = (value: number) => Math.max(256, Math.min(2048, Math.round((Number(value) || 256) / 16) * 16))

// Proportional glyph so the chip's shape signals portrait (tall) vs landscape (wide).
function aspectGlyph(aspect: string, maxSide = 26): { w: number; h: number } {
  const [a, b] = aspect.split(':').map(Number)
  if (!a || !b) return { w: maxSide, h: maxSide }
  return {
    w: a >= b ? maxSide : Math.round(maxSide * (a / b)),
    h: b >= a ? maxSide : Math.round(maxSide * (b / a)),
  }
}

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

  const setCustomDimension = (key: 'width' | 'height', value: number) => {
    setParams({ [key]: align16(value), aspect_ratio: 'custom' } as Partial<typeof params>)
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
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1.25 }}>
          {ASPECTS.map(aspect => {
            const active = isActiveAspect(aspect)
            const { w, h } = aspectGlyph(aspect)
            return (
              <Box
                key={aspect}
                onClick={() => applyTierAspect(params.resolution_tier, aspect)}
                title={aspect}
                sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.25, cursor: 'pointer', minWidth: 30 }}
              >
                <Box sx={{ height: 28, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Box
                    sx={{
                      width: w,
                      height: h,
                      borderRadius: 0.5,
                      border: '2px solid',
                      borderColor: active ? 'primary.main' : 'text.disabled',
                      bgcolor: active ? 'primary.main' : 'transparent',
                      opacity: active ? 0.85 : 1,
                      transition: 'all 0.15s',
                    }}
                  />
                </Box>
                <Typography variant="caption" sx={{ fontSize: 10, lineHeight: 1, color: active ? 'primary.main' : 'text.secondary', fontWeight: active ? 700 : 400 }}>
                  {aspect}
                </Typography>
              </Box>
            )
          })}
        </Box>
        <Grid container spacing={1.5}>
          <Grid item xs={6}>
            <TextField
              label="Width"
              type="number"
              value={params.width}
              onChange={e => setCustomDimension('width', Number(e.target.value))}
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
              onChange={e => setCustomDimension('height', Number(e.target.value))}
              size="small"
              fullWidth
              inputProps={{ step: 16 }}
              helperText="Aligned to 16"
            />
          </Grid>
        </Grid>
      </Stack>
    </Box>
  )
}
