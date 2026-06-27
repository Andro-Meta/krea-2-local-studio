import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  Box, CircularProgress, IconButton, InputAdornment, Slider, Stack,
  TextField, ToggleButton, ToggleButtonGroup, Tooltip, Typography,
} from '@mui/material'
import BrushIcon from '@mui/icons-material/Brush'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import ClearIcon from '@mui/icons-material/Clear'
import { apiFetch } from '../../api'

interface Props { imageB64: string; onMaskChange: (maskB64: string) => void }

export default function MaskCanvas({ imageB64, onMaskChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const maskRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const [brushSize, setBrushSize] = useState(30)
  const [tool, setTool] = useState<'brush' | 'eraser'>('brush')
  const [painting, setPainting] = useState(false)
  const [autoPrompt, setAutoPrompt] = useState('')
  const [autoLoading, setAutoLoading] = useState(false)
  const [autoError, setAutoError] = useState('')
  const [preKind, setPreKind] = useState<'canny' | 'soft_edge' | 'lineart' | 'depth'>('canny')
  const [preLoading, setPreLoading] = useState(false)
  const [prePreview, setPrePreview] = useState('')

  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      const c = canvasRef.current!
      c.width = img.width; c.height = img.height
      const m = maskRef.current!
      m.width = img.width; m.height = img.height
      const ctx = c.getContext('2d')!
      ctx.drawImage(img, 0, 0)
    }
    img.src = `data:image/png;base64,${imageB64}`
  }, [imageB64])

  // Redraw the display canvas = base image + translucent mask overlay.
  const redrawDisplay = useCallback(() => {
    const dc = canvasRef.current!, m = maskRef.current!
    const dctx = dc.getContext('2d')!
    dctx.clearRect(0, 0, dc.width, dc.height)
    if (imgRef.current) dctx.drawImage(imgRef.current, 0, 0)
    dctx.globalAlpha = 0.5
    dctx.fillStyle = '#D0BCFF'
    dctx.drawImage(m, 0, 0)
    dctx.globalAlpha = 1
  }, [])

  const getPos = (e: React.MouseEvent<Element>, canvas: HTMLCanvasElement) => {
    const r = canvas.getBoundingClientRect()
    return [(e.clientX - r.left) * (canvas.width / r.width),
            (e.clientY - r.top) * (canvas.height / r.height)] as [number, number]
  }

  const paint = useCallback((e: React.MouseEvent<Element>) => {
    if (!painting) return
    const m = maskRef.current!
    const ctx = m.getContext('2d')!
    const [x, y] = getPos(e, m)
    ctx.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over'
    ctx.fillStyle = 'white'
    ctx.beginPath()
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2)
    ctx.fill()
    redrawDisplay()
    onMaskChange(m.toDataURL('image/png').split(',')[1])
  }, [painting, tool, brushSize, onMaskChange, redrawDisplay])

  const clearMask = () => {
    const m = maskRef.current!
    m.getContext('2d')!.clearRect(0, 0, m.width, m.height)
    redrawDisplay()
    onMaskChange('')
  }

  // Auto-mask: ask the backend (CLIPSeg) for a mask from a text prompt, paint it in.
  const runAutoMask = async () => {
    if (!autoPrompt.trim()) return
    setAutoLoading(true); setAutoError('')
    try {
      const maskB64 = await apiFetch.autoMask(imageB64, autoPrompt.trim())
      const mimg = new Image()
      mimg.onload = () => {
        const m = maskRef.current!
        const tmp = document.createElement('canvas')
        tmp.width = m.width; tmp.height = m.height
        const tctx = tmp.getContext('2d')!
        tctx.drawImage(mimg, 0, 0, m.width, m.height)
        const md = tctx.getImageData(0, 0, m.width, m.height)
        const mctx = m.getContext('2d')!
        const out = mctx.createImageData(m.width, m.height)
        let any = false
        for (let i = 0; i < md.data.length; i += 4) {
          const on = md.data[i] > 127           // grayscale luminance
          if (on) any = true
          out.data[i] = 255; out.data[i + 1] = 255; out.data[i + 2] = 255
          out.data[i + 3] = on ? 255 : 0
        }
        mctx.putImageData(out, 0, 0)
        redrawDisplay()
        onMaskChange(any ? m.toDataURL('image/png').split(',')[1] : '')
        if (!any) setAutoError(`Nothing matched "${autoPrompt}" — try a different word`)
      }
      mimg.src = `data:image/png;base64,${maskB64}`
    } catch (e: any) {
      setAutoError(e?.response?.data?.detail ?? e.message ?? 'Auto-mask failed')
    }
    setAutoLoading(false)
  }

  const runPreprocessor = async () => {
    setPreLoading(true); setAutoError('')
    try {
      const preview = await apiFetch.preprocessorPreview(imageB64, { kind: preKind, resolution: 768 })
      setPrePreview(preview.image_b64)
    } catch (e: any) {
      setAutoError(e?.response?.data?.detail ?? e.message ?? 'Preprocessor preview failed')
    }
    setPreLoading(false)
  }

  return (
    <Box>
      {/* Auto-mask from text */}
      <TextField
        label="Auto-mask from text (CLIPSeg)"
        placeholder="e.g. the sky, the person — comma-separated"
        value={autoPrompt}
        onChange={e => setAutoPrompt(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && runAutoMask()}
        size="small"
        fullWidth
        error={!!autoError}
        helperText={autoError || 'Describe the region to mask, then ✦. Paint to refine.'}
        sx={{ mb: 1 }}
        InputProps={{
          endAdornment: (
            <InputAdornment position="end">
              <Tooltip title="Generate mask from text">
                <span>
                  <IconButton size="small" onClick={runAutoMask} disabled={autoLoading || !autoPrompt.trim()}>
                    {autoLoading ? <CircularProgress size={16} /> : <AutoAwesomeIcon sx={{ fontSize: 18 }} />}
                  </IconButton>
                </span>
              </Tooltip>
            </InputAdornment>
          ),
        }}
      />

      <Stack direction="row" spacing={1} alignItems="center" mb={1} flexWrap="wrap">
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>Control preview</Typography>
        <ToggleButtonGroup size="small" value={preKind} exclusive onChange={(_, v) => v && setPreKind(v)}>
          <ToggleButton value="canny">Canny</ToggleButton>
          <ToggleButton value="soft_edge">Soft</ToggleButton>
          <ToggleButton value="lineart">Lineart</ToggleButton>
          <ToggleButton value="depth">Depth</ToggleButton>
        </ToggleButtonGroup>
        <Tooltip title="Generate preprocessor preview">
          <span>
            <IconButton size="small" onClick={runPreprocessor} disabled={preLoading}>
              {preLoading ? <CircularProgress size={16} /> : <AutoFixHighIcon sx={{ fontSize: 18 }} />}
            </IconButton>
          </span>
        </Tooltip>
      </Stack>
      {prePreview && (
        <Box sx={{ mb: 1, borderRadius: 1, overflow: 'hidden', border: '1px solid', borderColor: 'divider' }}>
          <img src={`data:image/png;base64,${prePreview}`} alt={`${preKind} preview`} style={{ width: '100%', display: 'block' }} />
        </Box>
      )}

      <Stack direction="row" spacing={2} alignItems="center" mb={1} flexWrap="wrap">
        <ToggleButtonGroup size="small" value={tool} exclusive onChange={(_, v) => v && setTool(v)}>
          <ToggleButton value="brush"><BrushIcon fontSize="small" /></ToggleButton>
          <ToggleButton value="eraser"><AutoFixHighIcon fontSize="small" /></ToggleButton>
        </ToggleButtonGroup>
        <Stack direction="row" spacing={1} alignItems="center" sx={{ flex: 1, minWidth: 120 }}>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>Brush</Typography>
          <Slider value={brushSize} min={4} max={120} step={2} onChange={(_, v) => setBrushSize(v as number)} size="small" />
          <Typography variant="caption" sx={{ minWidth: 28 }}>{brushSize}px</Typography>
        </Stack>
        <Tooltip title="Clear mask">
          <IconButton size="small" onClick={clearMask}><ClearIcon fontSize="small" /></IconButton>
        </Tooltip>
      </Stack>

      <Box sx={{ position: 'relative', overflow: 'hidden', borderRadius: 2, cursor: 'crosshair' }}>
        <canvas ref={canvasRef} style={{ display: 'block', maxWidth: '100%', userSelect: 'none' }} />
        <canvas ref={maskRef} style={{ display: 'none' }} />
        <Box
          sx={{ position: 'absolute', inset: 0 }}
          onMouseDown={() => setPainting(true)}
          onMouseUp={() => setPainting(false)}
          onMouseLeave={() => setPainting(false)}
          onMouseMove={(e: React.MouseEvent<HTMLDivElement>) => paint(e)}
        />
      </Box>
    </Box>
  )
}
