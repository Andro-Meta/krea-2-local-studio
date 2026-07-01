import React, { useState } from 'react'
import {
  Backdrop, Box, CircularProgress, Grid, IconButton, Menu, MenuItem,
  Paper, Snackbar, Alert, Tooltip, Typography,
} from '@mui/material'
import DownloadIcon from '@mui/icons-material/Download'
import ZoomOutMapIcon from '@mui/icons-material/ZoomOutMap'
import { useStore } from '../../store'
import { apiFetch } from '../../api'
import { downloadImage } from '../../lib/imageActions'

const UPSCALE_METHODS = [
  {
    key: 'refine_2pass',
    label: '2× refine (recommended)',
    sub: 'pre-upscale + 2 low-denoise refine passes · quick',
    opts: { upscale_by: 2, steps: 8, cfg: 1, denoise: 0.25, sampler: 'euler', scheduler: 'beta' },
  },
  {
    key: 'ultimate',
    label: 'Ultimate SD Upscale 2× balanced',
    sub: 'Euler/simple · 8 steps · 1024 tile · band-pass seams',
    opts: { upscale_by: 2, steps: 8, cfg: 1, denoise: 0.28, tile_size: 1024, tile_padding: 96, mask_blur: 12, seam_mode: 'band_pass', tile_mode: 'chess', sampler: 'euler', scheduler: 'simple' },
  },
  {
    key: 'ultimate',
    label: 'Ultimate SD Upscale 2× best',
    sub: 'Euler/simple · 8 steps · half-tile seam pass · tiled decode',
    opts: { upscale_by: 2, steps: 8, cfg: 1, denoise: 0.24, tile_size: 1280, tile_padding: 128, mask_blur: 16, seam_mode: 'half_tile_intersections', tile_mode: 'chess', sampler: 'euler', scheduler: 'simple', tiled_decode: true },
  },
  { key: 'realesrgan', label: 'RealESRGAN 4×', sub: 'fast, no diffusion' },
  { key: 'pid_upscale', label: 'PiD 4× native (experimental)', sub: 'loads native PiD runtime · unloads Krea first · needs ~15GB VRAM', opts: { upscale_by: 4 } },
  { key: 'tiled_vae', label: 'Tiled VAE 2×', sub: 'lossless re-decode' },
  { key: 'model_refine', label: 'Detail refine 1×', sub: 'sharpen, no resize' },
]

interface Props {
  images: string[]
  seed: number | null
  metadata?: Array<Record<string, any>>
}

export default function ResultsView({ images, seed, metadata = [] }: Props) {
  const { openLightbox, params } = useStore()
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null)
  const [activeIdx, setActiveIdx] = useState(0)
  const [busy, setBusy] = useState<string | null>(null)   // method label while running
  const [toast, setToast] = useState<string | null>(null)

  if (!images.length) return null

  const openMenu = (e: React.MouseEvent<HTMLElement>, idx: number) => {
    setActiveIdx(idx)
    setMenuAnchor(e.currentTarget)
  }

  const runUpscale = async (method: string, label: string, opts: Record<string, any> = {}) => {
    setMenuAnchor(null)
    setBusy(label)
    try {
      const { image_b64, metadata: upscaledMetadata } = await apiFetch.upscale(images[activeIdx], method, {
        prompt: params.prompt,
        ...opts,
      })
      openLightbox([{ src: `data:image/png;base64,${image_b64}`, prompt: params.prompt, metadata: upscaledMetadata }])
      setToast(`${label} complete — opened full size`)
    } catch (e: any) {
      setToast(e?.response?.data?.detail ?? e.message ?? 'Upscale failed')
    }
    setBusy(null)
  }

  return (
    <Box>
      {seed != null && (
        <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block' }}>
          Seed: {seed}
        </Typography>
      )}
      <Grid container spacing={1}>
        {images.map((b64, i) => {
          const src = `data:image/png;base64,${b64}`
          return (
            <Grid item xs={12} sm={images.length > 1 ? 6 : 12} key={i}>
              <Paper sx={{ position: 'relative', overflow: 'hidden', borderRadius: 2, cursor: 'pointer' }}>
                <img
                  src={src}
                  alt={`Result ${i + 1}`}
                  style={{ width: '100%', display: 'block', borderRadius: 8 }}
                  onClick={() => openLightbox(images.map((img, idx) => ({
                    src: `data:image/png;base64,${img}`,
                    prompt: params.prompt,
                    filename: `krea2_result_${idx + 1}.png`,
                    metadata: metadata[idx],
                  })), i)}
                />
                <Box sx={{ position: 'absolute', top: 8, right: 8, display: 'flex', gap: 0.5 }}>
                  <Tooltip title="Upscale">
                    <IconButton size="small" sx={{ bgcolor: 'rgba(0,0,0,0.6)' }}
                      onClick={(e) => openMenu(e, i)}>
                      <ZoomOutMapIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Download">
                    <IconButton size="small" sx={{ bgcolor: 'rgba(0,0,0,0.6)' }}
                      onClick={(e) => { e.stopPropagation(); downloadImage(src, `krea2_result_${i + 1}.png`) }}>
                      <DownloadIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Paper>
            </Grid>
          )
        })}
      </Grid>

      <Menu anchorEl={menuAnchor} open={!!menuAnchor} onClose={() => setMenuAnchor(null)}>
        <Typography variant="caption" sx={{ px: 2, py: 0.5, display: 'block', color: 'text.secondary' }}>
          Upscale method
        </Typography>
        {UPSCALE_METHODS.map(m => (
          <MenuItem key={`${m.key}-${m.label}`} onClick={() => runUpscale(m.key, m.label, m.opts)} sx={{ display: 'block', py: 1 }}>
            <Typography variant="body2">{m.label}</Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>{m.sub}</Typography>
          </MenuItem>
        ))}
      </Menu>

      <Backdrop open={!!busy} sx={{ zIndex: t => t.zIndex.modal + 1, flexDirection: 'column', gap: 2, color: '#fff' }}>
        <CircularProgress color="inherit" />
        <Box sx={{ textAlign: 'center' }}>
          <Typography>{busy}…</Typography>
          <Typography variant="caption" sx={{ opacity: 0.7 }}>
            Tiled upscaling can take a few minutes — please wait.
          </Typography>
        </Box>
      </Backdrop>

      <Snackbar open={!!toast} autoHideDuration={5000} onClose={() => setToast(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
        <Alert severity="info" onClose={() => setToast(null)} variant="filled">{toast}</Alert>
      </Snackbar>
    </Box>
  )
}
