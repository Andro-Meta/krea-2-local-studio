import { useEffect, useRef } from 'react'
import { Box, Button, IconButton, Stack, TextField, Tooltip, Typography } from '@mui/material'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import AddBoxIcon from '@mui/icons-material/AddBox'

export interface EditRegion {
  id: string
  x: number
  y: number
  w: number
  h: number
  prompt: string
  referenceB64: string
  referencePreview: string
}

const REGION_COLORS = ['#4fc3f7', '#ff8a65', '#81c784', '#ba68c8', '#ffd54f', '#f06292']

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

function readFile(file: File): Promise<{ b64: string; preview: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const preview = String(reader.result || '')
      resolve({ b64: preview.split(',')[1] || preview, preview })
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

async function clipboardImageFile(): Promise<File | null> {
  try {
    if (!navigator.clipboard?.read) return null
    const items = await navigator.clipboard.read()
    for (const item of items) {
      const type = item.types.find(t => t.startsWith('image/'))
      if (type) {
        const blob = await item.getType(type)
        return new File([blob], 'pasted.png', { type })
      }
    }
  } catch {
    return null
  }
  return null
}

interface Props {
  background: string
  aspectW: number
  aspectH: number
  regions: EditRegion[]
  onChange: (regions: EditRegion[]) => void
}

export default function RegionCanvas({ background, aspectW, aspectH, regions, onChange }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const regionsRef = useRef<EditRegion[]>(regions)
  const drag = useRef<{ id: string; mode: 'move' | 'resize'; startX: number; startY: number; orig: EditRegion } | null>(null)

  useEffect(() => { regionsRef.current = regions }, [regions])

  const paddingRatio = aspectH / Math.max(1, aspectW)

  const beginDrag = (event: React.PointerEvent, id: string, mode: 'move' | 'resize') => {
    event.preventDefault()
    event.stopPropagation()
    const region = regionsRef.current.find(r => r.id === id)
    if (!region) return
    drag.current = { id, mode, startX: event.clientX, startY: event.clientY, orig: { ...region } }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  const onMove = (event: PointerEvent) => {
    const state = drag.current
    const container = containerRef.current
    if (!state || !container) return
    const rect = container.getBoundingClientRect()
    const dx = (event.clientX - state.startX) / rect.width
    const dy = (event.clientY - state.startY) / rect.height
    onChange(regionsRef.current.map(r => {
      if (r.id !== state.id) return r
      if (state.mode === 'move') {
        return { ...r, x: clamp(state.orig.x + dx, 0, 1 - r.w), y: clamp(state.orig.y + dy, 0, 1 - r.h) }
      }
      return { ...r, w: clamp(state.orig.w + dx, 0.05, 1 - r.x), h: clamp(state.orig.h + dy, 0.05, 1 - r.y) }
    }))
  }

  const onUp = () => {
    drag.current = null
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', onUp)
  }

  useEffect(() => () => {
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', onUp)
  }, [])

  const addRegion = () => {
    if (regions.length >= 6) return
    const index = regions.length
    const region: EditRegion = {
      id: `r_${Date.now()}_${index}`,
      x: index % 2 === 0 ? 0.05 : 0.5,
      y: 0.1,
      w: 0.42,
      h: 0.8,
      prompt: '',
      referenceB64: '',
      referencePreview: '',
    }
    onChange([...regions, region])
  }

  const updateRegion = (id: string, patch: Partial<EditRegion>) => {
    onChange(regions.map(r => (r.id === id ? { ...r, ...patch } : r)))
  }

  const removeRegion = (id: string) => onChange(regions.filter(r => r.id !== id))

  const uploadReference = async (id: string, file?: File) => {
    if (!file) return
    const loaded = await readFile(file)
    updateRegion(id, { referenceB64: loaded.b64, referencePreview: loaded.preview })
  }

  const pasteReference = async (id: string) => {
    const file = await clipboardImageFile()
    if (file) uploadReference(id, file)
  }

  return (
    <Stack spacing={1.25}>
      <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          Placement boxes ({regions.length}/6) — drag to move, pull the corner to resize.
        </Typography>
        <Button size="small" startIcon={<AddBoxIcon fontSize="small" />} onClick={addRegion} disabled={regions.length >= 6}>
          Add box
        </Button>
      </Stack>
      <Typography variant="caption" sx={{ color: 'text.disabled' }}>
        Each box's reference should be a clear face/head crop of that person — anything else in the photo (props, clothing, background) can bleed into that box.
      </Typography>

      <Box
        ref={containerRef}
        sx={{
          position: 'relative',
          width: '100%',
          pt: `${paddingRatio * 100}%`,
          borderRadius: 2,
          overflow: 'hidden',
          bgcolor: 'rgba(0,0,0,0.35)',
          border: '1px solid',
          borderColor: 'divider',
          backgroundImage: background ? `url(${background})` : undefined,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          touchAction: 'none',
        }}
      >
        {!background && (
          <Box sx={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', px: 2, textAlign: 'center', pointerEvents: 'none' }}>
            <Typography variant="caption" sx={{ color: 'text.disabled' }}>
              No scene image — each box places its person on a generated background.
            </Typography>
          </Box>
        )}
        {regions.map((region, index) => {
          const color = REGION_COLORS[index % REGION_COLORS.length]
          return (
            <Box
              key={region.id}
              onPointerDown={event => beginDrag(event, region.id, 'move')}
              sx={{
                position: 'absolute',
                left: `${region.x * 100}%`,
                top: `${region.y * 100}%`,
                width: `${region.w * 100}%`,
                height: `${region.h * 100}%`,
                border: `2px solid ${color}`,
                borderRadius: 1,
                cursor: 'move',
                boxShadow: '0 0 0 9999px rgba(0,0,0,0.04)',
                bgcolor: 'rgba(0,0,0,0.05)',
              }}
            >
              <Box sx={{ position: 'absolute', top: 2, left: 2, px: 0.75, borderRadius: 0.5, bgcolor: color, color: '#000', fontSize: 12, fontWeight: 700, zIndex: 1 }}>
                {index + 1}
              </Box>
              {region.referencePreview ? (
                <Box component="img" src={region.referencePreview} alt="" sx={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.55, pointerEvents: 'none' }} />
              ) : (
                <Box sx={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', pointerEvents: 'none' }}>
                  <Typography variant="caption" sx={{ color, fontWeight: 600, textShadow: '0 1px 2px rgba(0,0,0,0.8)' }}>
                    Box {index + 1}: add a face below
                  </Typography>
                </Box>
              )}
              <Box
                onPointerDown={event => beginDrag(event, region.id, 'resize')}
                sx={{
                  position: 'absolute', right: -1, bottom: -1, width: 16, height: 16,
                  bgcolor: color, borderRadius: '3px 0 3px 0', cursor: 'nwse-resize',
                }}
              />
            </Box>
          )
        })}
      </Box>

      {regions.map((region, index) => {
        const color = REGION_COLORS[index % REGION_COLORS.length]
        const refInputId = `region-ref-${region.id}`
        return (
          <Stack key={region.id} direction="row" spacing={1} alignItems="flex-start" sx={{ p: 1, borderRadius: 1, border: '1px solid', borderColor: 'divider' }}>
            <Box sx={{ width: 22, height: 22, borderRadius: 0.5, bgcolor: color, color: '#000', display: 'grid', placeItems: 'center', fontWeight: 700, flexShrink: 0 }}>{index + 1}</Box>
            <Stack spacing={0.75} sx={{ flex: 1 }}>
              <TextField
                size="small"
                label={`Box ${index + 1} prompt (who / what goes here)`}
                value={region.prompt}
                onChange={event => updateRegion(region.id, { prompt: event.target.value })}
                fullWidth
                multiline
                minRows={1}
              />
              <Stack direction="row" spacing={1} alignItems="center">
                <input id={refInputId} hidden type="file" accept="image/*" onChange={event => uploadReference(region.id, event.target.files?.[0])} />
                <Button size="small" variant="outlined" onClick={() => document.getElementById(refInputId)?.click()}>
                  {region.referenceB64 ? 'Change face' : 'Add face'}
                </Button>
                <Button size="small" variant="outlined" onClick={() => pasteReference(region.id)}>Paste</Button>
                {region.referenceB64 && (
                  <Button size="small" color="inherit" onClick={() => updateRegion(region.id, { referenceB64: '', referencePreview: '' })}>Clear</Button>
                )}
                {region.referencePreview && <Box component="img" src={region.referencePreview} alt="" sx={{ width: 34, height: 34, objectFit: 'cover', borderRadius: 0.5 }} />}
              </Stack>
            </Stack>
            <Tooltip title="Remove box">
              <IconButton size="small" onClick={() => removeRegion(region.id)}><DeleteOutlineIcon fontSize="small" /></IconButton>
            </Tooltip>
          </Stack>
        )
      })}
    </Stack>
  )
}
