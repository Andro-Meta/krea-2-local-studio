import React from 'react'
import { Box, Chip, Grid, Slider, Stack, TextField, Typography } from '@mui/material'
import { useStore } from '../../store'

interface Preset { label: string; w: number; h: number; ratio: string }

const PRESETS: Preset[] = [
  { label: '1:1', w: 1024, h: 1024, ratio: '1:1' },
  { label: '16:9', w: 1280, h: 720, ratio: '16:9' },
  { label: '9:16', w: 720, h: 1280, ratio: '9:16' },
  { label: '4:3', w: 1024, h: 768, ratio: '4:3' },
  { label: '3:2', w: 1024, h: 683, ratio: '3:2' },
  { label: '2:3', w: 683, h: 1024, ratio: '2:3' },
  { label: '21:9', w: 1344, h: 576, ratio: '21:9' },
  { label: '2048²', w: 2048, h: 2048, ratio: '1:1' },
]

export default function DimensionSection() {
  const { params, setParam, setParams } = useStore()
  const isMatch = (p: Preset) => params.width === p.w && params.height === p.h
  return (
    <Box>
      <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block', textTransform: 'uppercase', letterSpacing: 1 }}>Dimensions</Typography>
      <Stack spacing={1.5}>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
          {PRESETS.map(p => (
            <Chip
              key={p.label}
              label={p.label}
              size="small"
              variant={isMatch(p) ? 'filled' : 'outlined'}
              color={isMatch(p) ? 'primary' : 'default'}
              onClick={() => setParams({ width: p.w, height: p.h })}
              clickable
            />
          ))}
        </Box>
        <Grid container spacing={1.5}>
          <Grid item xs={6}>
            <TextField
              label="Width"
              type="number"
              value={params.width}
              onChange={e => setParam('width', Math.max(256, Math.min(2048, Number(e.target.value))))}
              size="small"
              fullWidth
              inputProps={{ step: 64 }}
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
              inputProps={{ step: 64 }}
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
