import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import {
  Alert, Box, Button, CircularProgress, Paper, Slider, Stack, ToggleButton, ToggleButtonGroup, Typography,
} from '@mui/material'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import { apiFetch } from '../../api'
import { useStore } from '../../store'

// Pull an embedded generation prompt out of a PNG's text chunks (our images store
// krea2_metadata as JSON). Returns '' if none — the caller then auto-describes.
async function promptFromPng(file: File): Promise<string> {
  try {
    if (!file.type.includes('png')) return ''
    const buf = new Uint8Array(await file.arrayBuffer())
    if (buf.length < 8) return ''
    const dec = new TextDecoder('latin1')
    let off = 8
    while (off + 12 <= buf.length) {
      const len = (buf[off] << 24) | (buf[off + 1] << 16) | (buf[off + 2] << 8) | buf[off + 3]
      const type = dec.decode(buf.subarray(off + 4, off + 8))
      if (type === 'IEND') break
      if (type === 'tEXt') {
        const text = dec.decode(buf.subarray(off + 8, off + 8 + len))
        const nul = text.indexOf('\0')
        const keyword = nul >= 0 ? text.slice(0, nul) : ''
        const value = nul >= 0 ? text.slice(nul + 1) : text
        if (keyword === 'krea2_metadata' || keyword === 'parameters') {
          try {
            const j = JSON.parse(value)
            if (j && typeof j.prompt === 'string' && j.prompt.trim()) return j.prompt.trim()
          } catch { /* not json */ }
        }
        if (keyword === 'prompt' && value.trim()) return value.trim()
      }
      off += 12 + len
    }
  } catch { /* ignore */ }
  return ''
}

const TARGETS = [
  { label: '1K', w: 1024, h: 1024, upscaler: 'esrgan_x2' as const },
  { label: '2K', w: 2048, h: 2048, upscaler: 'esrgan_x2' as const },
  { label: '4K', w: 4096, h: 4096, upscaler: 'remacri_x4' as const },
]

export default function UpscalePanel() {
  const { params, setParam, setParams } = useStore()
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [busy, setBusy] = useState<'reading' | 'describing' | null>(null)
  const [msg, setMsg] = useState<{ severity: 'success' | 'info' | 'error'; text: string } | null>(null)

  // Entering the Upscale tab puts Generate into Mr. Flow mode; leaving resets it
  // so the other Create tabs aren't stuck upscaling.
  useEffect(() => {
    setParams({ mrflow: true, god_mode: false, mode: 'txt2img' })
    return () => setParams({ mrflow: false, init_image_b64: '' })
  }, [setParams])

  const readFile = (file: File): Promise<string> => new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onerror = () => reject(new Error('Could not read image'))
    r.onload = () => resolve(String(r.result ?? ''))
    r.readAsDataURL(file)
  })

  const onFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setBusy('reading')
    setMsg(null)
    try {
      const dataUrl = await readFile(file)
      // Set up the Mr. Flow upscale recipe (backend skips the base render when an
      // init image is present, so it upscales this image instead of generating).
      setParams({
        mrflow: true, god_mode: false, mode: 'txt2img',
        init_image_b64: dataUrl,
        moodboard_images: [], ref_image1_b64: '', ref_image2_b64: '',
      })
      // Prompt: use embedded metadata if present, else auto-describe (silent).
      let prompt = await promptFromPng(file)
      if (prompt) {
        setParam('prompt', prompt)
        setMsg({ severity: 'success', text: 'Image loaded — prompt read from its metadata.' })
      } else {
        setBusy('describing')
        setMsg({ severity: 'info', text: 'Image loaded — no metadata, reading a prompt from the image…' })
        try {
          const r = await apiFetch.describeImage(dataUrl.split(',')[1] ?? dataUrl)
          prompt = r.prompt
          setParam('prompt', prompt)
          setMsg({ severity: 'success', text: 'Image loaded — prompt auto-generated from the image.' })
        } catch {
          setMsg({ severity: 'error', text: 'Loaded, but auto-prompt failed (GPU busy?). It will still upscale; add a prompt above if you want.' })
        }
      }
    } finally {
      setBusy(null)
    }
  }

  const setTarget = (t: typeof TARGETS[number]) =>
    setParams({ width: t.w, height: t.h, mrflow_upscaler: t.upscaler })

  const activeTarget = TARGETS.find(t => t.w === params.width && t.upscaler === params.mrflow_upscaler)
  const effDenoise = params.mrflow_refine_denoise > 0 ? params.mrflow_refine_denoise : 0.12
  const effSteps = params.mrflow_refine_steps > 0 ? params.mrflow_refine_steps : 1

  return (
    <Box sx={{ mb: 2 }}>
      <Paper variant="outlined" sx={{ p: 2, borderColor: 'rgba(202,196,208,0.18)' }}>
        <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Mr. Flow Upscale</Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 1.5 }}>
          Upload an image; it's upscaled with RealESRGAN/Remacri then given one prompt-guided Krea-2 refine pass.
          The prompt is taken from the image's metadata, or auto-read from the image when there's none. Then hit Generate below.
        </Typography>

        <input ref={inputRef} type="file" accept="image/*" hidden onChange={onFile} />
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1.5 }}>
          <Button
            variant="outlined" size="small"
            startIcon={busy ? <CircularProgress size={14} /> : <UploadFileIcon fontSize="small" />}
            onClick={() => inputRef.current?.click()}
            disabled={!!busy}
          >
            {busy === 'reading' ? 'Reading…' : busy === 'describing' ? 'Reading prompt…' : 'Upload image'}
          </Button>
          {params.init_image_b64 && (
            <Box component="img" src={params.init_image_b64} alt="source"
              sx={{ height: 44, borderRadius: 1, border: '1px solid rgba(202,196,208,0.25)' }} />
          )}
        </Stack>

        <Stack spacing={1.75}>
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>Target size</Typography>
            <ToggleButtonGroup
              size="small" exclusive fullWidth sx={{ mt: 0.5 }}
              value={activeTarget?.label ?? null}
              onChange={(_, v) => { const t = TARGETS.find(x => x.label === v); if (t) setTarget(t) }}
            >
              {TARGETS.map(t => <ToggleButton key={t.label} value={t.label}>{t.label}</ToggleButton>)}
            </ToggleButtonGroup>
          </Box>

          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>Upscaler</Typography>
            <ToggleButtonGroup
              size="small" exclusive fullWidth sx={{ mt: 0.5 }}
              value={params.mrflow_upscaler}
              onChange={(_, v) => v && setParam('mrflow_upscaler', v)}
            >
              <ToggleButton value="esrgan_x2">RealESRGAN ×2</ToggleButton>
              <ToggleButton value="remacri_x4">Foolhardy Remacri ×4</ToggleButton>
            </ToggleButtonGroup>
          </Box>

          <Box>
            <Stack direction="row" justifyContent="space-between" alignItems="baseline">
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>Refine strength</Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                {params.mrflow_refine_denoise > 0 ? effDenoise.toFixed(2) : `auto (~${effDenoise.toFixed(2)})`}
              </Typography>
            </Stack>
            <Slider size="small" min={0} max={0.30} step={0.01}
              value={params.mrflow_refine_denoise}
              onChange={(_, v) => setParam('mrflow_refine_denoise', v as number)}
              marks={[{ value: 0, label: 'auto' }, { value: 0.12, label: '0.12' }, { value: 0.30, label: '0.30' }]}
            />
            <Typography variant="caption" sx={{ color: 'text.disabled' }}>
              Higher = more Krea-2 rework/detail; lower = closer to the raw upscale.
            </Typography>
          </Box>

          <Box>
            <Stack direction="row" justifyContent="space-between" alignItems="baseline">
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>Refine steps</Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>{effSteps}</Typography>
            </Stack>
            <ToggleButtonGroup
              size="small" exclusive fullWidth sx={{ mt: 0.5 }}
              value={effSteps}
              onChange={(_, v) => v && setParam('mrflow_refine_steps', v as number)}
            >
              <ToggleButton value={1}>1 (fast)</ToggleButton>
              <ToggleButton value={2}>2</ToggleButton>
              <ToggleButton value={3}>3 (max)</ToggleButton>
            </ToggleButtonGroup>
          </Box>
        </Stack>

        {msg && <Alert severity={msg.severity} sx={{ py: 0, mt: 1.5 }} onClose={() => setMsg(null)}>{msg.text}</Alert>}
        {!params.init_image_b64 && (
          <Alert severity="info" sx={{ py: 0, mt: 1.5 }}>Upload an image to upscale, then press Generate below.</Alert>
        )}
      </Paper>
    </Box>
  )
}
