import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert, Box, Button, Chip, CircularProgress, Paper, Slider, Stack,
  ToggleButton, ToggleButtonGroup, Typography,
} from '@mui/material'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import OpenInFullIcon from '@mui/icons-material/OpenInFull'
import { useStore } from '../../store'
import { buildOutpaintImage, type OutpaintPads } from '../../lib/outpaint'

type Direction = 'left' | 'right' | 'top' | 'bottom' | 'all'

function readFileB64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error('Could not read image'))
    reader.onload = ev => resolve(String(ev.target?.result ?? '').split(',')[1])
    reader.readAsDataURL(file)
  })
}

function imageSize(b64: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight })
    img.onerror = () => reject(new Error('Could not load source image'))
    img.src = `data:image/png;base64,${b64}`
  })
}

export default function OutpaintPanel() {
  const { params, setParam, setParams } = useStore()
  const [sourceB64, setSourceB64] = useState(params.init_image_b64)
  const [directions, setDirections] = useState<Direction[]>(['all'])
  const [amount, setAmount] = useState(0.5)
  const [overlap, setOverlap] = useState(32)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  useEffect(() => { setParams({ mode: 'outpaint' }) }, [])
  useEffect(() => {
    if (params.mode === 'outpaint' && params.init_image_b64 && params.init_image_b64 !== sourceB64) {
      setSourceB64(params.init_image_b64)
    }
  }, [params.init_image_b64, params.mode])

  const helperPrompt = useMemo(() => (
    'continue the existing lighting, perspective, color palette, texture, depth of field, and composition beyond the original frame'
  ), [])

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const b64 = await readFileB64(file)
    setSourceB64(b64)
    setParams({ init_image_b64: b64, mask_b64: '', mode: 'outpaint' })
  }

  const makePads = async (): Promise<OutpaintPads> => {
    const { width, height } = await imageSize(sourceB64)
    const horizontal = Math.round(width * amount)
    const vertical = Math.round(height * amount)
    const all = directions.includes('all')
    return {
      left: all || directions.includes('left') ? horizontal : 0,
      right: all || directions.includes('right') ? horizontal : 0,
      top: all || directions.includes('top') ? vertical : 0,
      bottom: all || directions.includes('bottom') ? vertical : 0,
    }
  }

  const prepare = async () => {
    if (!sourceB64) return
    setBusy(true)
    setMessage(null)
    try {
      const pads = await makePads()
      const result = await buildOutpaintImage(sourceB64, pads, overlap)
      setParams({
        mode: 'outpaint',
        init_image_b64: result.init_image_b64,
        mask_b64: result.mask_b64,
        width: result.width,
        height: result.height,
        denoise: 1.0,
        prompt: params.prompt.trim() ? params.prompt : helperPrompt,
      })
      setMessage(`Prepared ${result.width} x ${result.height} outpaint canvas. Generate when ready.`)
    } catch (e: any) {
      setMessage(e?.message ?? 'Could not prepare outpaint canvas')
    } finally {
      setBusy(false)
    }
  }

  const directionValue = directions.includes('all') ? ['all'] : directions

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2 }, maxWidth: 900, mx: 'auto' }}>
      <Stack spacing={2}>
        <Paper
          variant="outlined"
          sx={{
            borderStyle: sourceB64 ? 'solid' : 'dashed', borderRadius: 2, p: 2,
            textAlign: 'center', cursor: sourceB64 ? 'default' : 'pointer',
          }}
          onClick={() => !sourceB64 && document.getElementById('outpaint-upload')?.click()}
        >
          <input id="outpaint-upload" type="file" accept="image/*" hidden onChange={handleFile} />
          {sourceB64 ? (
            <Stack spacing={1.25} alignItems="center">
              <img src={`data:image/png;base64,${sourceB64}`} alt="Outpaint source" style={{ maxWidth: '100%', maxHeight: 320, borderRadius: 12 }} />
              <Button size="small" variant="outlined" onClick={() => document.getElementById('outpaint-upload')?.click()}>
                Replace source
              </Button>
            </Stack>
          ) : (
            <Stack alignItems="center" spacing={0.75} sx={{ py: 3 }}>
              <UploadFileIcon sx={{ color: 'text.secondary', fontSize: 40 }} />
              <Typography>Upload an image, or load one from the lightbox</Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                Outpaint expands the canvas, masks the new area, then uses Krea inpaint to continue the scene.
              </Typography>
            </Stack>
          )}
        </Paper>

        <Box>
          <Typography variant="body2" sx={{ mb: 1 }}>Expand direction</Typography>
          <ToggleButtonGroup
            value={directionValue}
            onChange={(_, value: Direction[]) => {
              if (!value.length) return
              setDirections(value.includes('all') ? ['all'] : value.filter(v => v !== 'all'))
            }}
            size="small"
            color="primary"
            sx={{ flexWrap: 'wrap', gap: 0.75, '& .MuiToggleButtonGroup-grouped': { borderRadius: 99, border: 1 } }}
          >
            <ToggleButton value="all">All sides</ToggleButton>
            <ToggleButton value="left">Left</ToggleButton>
            <ToggleButton value="right">Right</ToggleButton>
            <ToggleButton value="top">Top</ToggleButton>
            <ToggleButton value="bottom">Bottom</ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <Box>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="body2">Expansion amount</Typography>
            <Chip label={`${Math.round(amount * 100)}%`} size="small" />
          </Stack>
          <Slider
            value={amount}
            min={0.25}
            max={1}
            step={0.25}
            marks={[0.25, 0.5, 1].map(v => ({ value: v, label: `${v * 100}%` }))}
            onChange={(_, value) => setAmount(value as number)}
          />
        </Box>

        <Box>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="body2">Blend overlap</Typography>
            <Chip label={`${overlap}px`} size="small" />
          </Stack>
          <Slider value={overlap} min={0} max={96} step={8} onChange={(_, value) => setOverlap(value as number)} />
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            A small overlap gives Krea room to harmonize seams while preserving most of the original image.
          </Typography>
        </Box>

        <Alert severity="info">
          Prompt tip: describe what should continue outside the frame, plus “{helperPrompt}.”
        </Alert>

        {message && <Alert severity={message.startsWith('Prepared') ? 'success' : 'error'} onClose={() => setMessage(null)}>{message}</Alert>}

        <Button
          variant="contained"
          size="large"
          startIcon={busy ? <CircularProgress size={18} color="inherit" /> : <OpenInFullIcon />}
          disabled={!sourceB64 || busy}
          onClick={prepare}
          fullWidth
        >
          {busy ? 'Preparing...' : 'Prepare outpaint'}
        </Button>
      </Stack>
    </Box>
  )
}
