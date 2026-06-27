import React, { useRef, useState } from 'react'
import {
  Box, Button, Collapse, IconButton, MenuItem, Slider, Stack, TextField, Tooltip, Typography,
} from '@mui/material'
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate'
import CloseIcon from '@mui/icons-material/Close'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { useStore, type StyleReference } from '../../store'

const MAX_STYLE_REFS = 10
const TOKEN_SIZES: StyleReference['token_size'][] = ['low', 'normal', 'high', 'max']
const ROLES = ['style', 'layout', 'subject', 'mood', 'texture'] as const

function move<T>(items: T[], from: number, to: number) {
  const next = [...items]
  const [item] = next.splice(from, 1)
  next.splice(to, 0, item)
  return next
}

export default function StyleReferenceSection() {
  const { params, setParam } = useStore()
  const fileRef = useRef<HTMLInputElement>(null)
  const [open, setOpen] = useState(false)
  const refs = params.style_references
  const active = refs.length > 0

  const updateRef = (index: number, patch: Partial<StyleReference>) => {
    setParam('style_references', refs.map((ref, i) => i === index ? { ...ref, ...patch } : ref))
  }

  const removeRef = (index: number) => {
    setParam('style_references', refs.filter((_, i) => i !== index))
  }

  const addMask = (index: number, file?: File) => {
    if (!file) return
    const reader = new FileReader()
    reader.onload = ev => {
      const b64 = String(ev.target?.result || '').split(',')[1]
      if (b64) updateRef(index, { mask_b64: b64 })
    }
    reader.readAsDataURL(file)
  }

  const addImages = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []).slice(0, Math.max(0, MAX_STYLE_REFS - refs.length))
    files.forEach(file => {
      const reader = new FileReader()
      reader.onload = ev => {
        const b64 = String(ev.target?.result || '').split(',')[1]
        if (!b64) return
        const current = useStore.getState().params.style_references
        if (current.length >= MAX_STYLE_REFS) return
        setParam('style_references', [
          ...current,
          { image_b64: b64, strength: 1.0, role: 'style', token_size: 'normal', vision_position: 'before_prompt' },
        ])
      }
      reader.readAsDataURL(file)
    })
    e.target.value = ''
  }

  return (
    <Box>
      <Stack direction="row" alignItems="center" justifyContent="space-between"
        sx={{ cursor: 'pointer' }} onClick={() => setOpen(o => !o)}>
        <Typography variant="caption" sx={{ color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 1 }}>
          Style References{active ? ` · ${refs.length}/10` : ''}
        </Typography>
        <ExpandMoreIcon sx={{ color: 'text.secondary', transform: open ? 'rotate(180deg)' : 'none', transition: '0.2s' }} />
      </Stack>

      <Collapse in={open}>
        <Box sx={{ pt: 1 }}>
          <Typography variant="caption" sx={{ color: 'text.disabled', mb: 1, display: 'block' }}>
            Comfy-style references accept up to 10 images. Strength ranges from -2.0 to 2.0; negative values are advanced and push away from a reference.
          </Typography>

          <TextField
            select
            size="small"
            label="Reference fusion"
            value={params.style_fusion_mode}
            onChange={e => setParam('style_fusion_mode', e.target.value as typeof params.style_fusion_mode)}
            fullWidth
            helperText="Style only ignores layout/target refs. Preserve structure keeps source composition in edit modes. Semantic fusion blends prompt and references."
            sx={{ mb: 1 }}
          >
            <MenuItem value="semantic_fusion">Semantic fusion</MenuItem>
            <MenuItem value="style_only">Style transfer only</MenuItem>
            <MenuItem value="preserve_structure">Preserve structure</MenuItem>
          </TextField>

          <Stack spacing={1}>
            {refs.map((ref, index) => (
              <Box key={`${index}-${ref.image_b64.slice(0, 12)}`} sx={{ border: '1px solid rgba(202,196,208,0.16)', borderRadius: 1, p: 1 }}>
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ xs: 'stretch', sm: 'center' }}>
                  <Box sx={{ position: 'relative', width: 64, height: 64, borderRadius: 1, overflow: 'hidden', flex: '0 0 auto' }}>
                    <img src={`data:image/png;base64,${ref.image_b64}`} alt={`style ref ${index + 1}`}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    <IconButton size="small" onClick={() => removeRef(index)}
                      sx={{ position: 'absolute', top: -2, right: -2, p: '1px', bgcolor: 'rgba(0,0,0,0.6)' }}>
                      <CloseIcon sx={{ fontSize: 12 }} />
                    </IconButton>
                  </Box>

                  <Box sx={{ flex: 1, minWidth: 180 }}>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Typography variant="body2" sx={{ color: 'text.secondary' }}>Strength</Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono', fontSize: 12 }}>
                        {ref.strength.toFixed(2)}
                      </Typography>
                    </Stack>
                    <Slider
                      value={ref.strength}
                      min={-2}
                      max={2}
                      step={0.05}
                      size="small"
                      onChange={(_, value) => updateRef(index, { strength: value as number })}
                    />
                  </Box>

                  <TextField
                    select
                    size="small"
                    label="Role"
                    value={ref.role}
                    onChange={e => updateRef(index, { role: e.target.value })}
                    sx={{ minWidth: 120 }}
                  >
                    {ROLES.map(role => <MenuItem key={role} value={role}>{role}</MenuItem>)}
                  </TextField>

                  <Stack direction="row" spacing={0.5}>
                    <Button size="small" disabled={index === 0} onClick={() => setParam('style_references', move(refs, index, index - 1))}>
                      Up
                    </Button>
                    <Button size="small" disabled={index === refs.length - 1} onClick={() => setParam('style_references', move(refs, index, index + 1))}>
                      Down
                    </Button>
                  </Stack>
                </Stack>

                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} sx={{ mt: 1 }}>
                  <TextField
                    select
                    size="small"
                    label="Token budget"
                    value={ref.token_size}
                    onChange={e => updateRef(index, { token_size: e.target.value as StyleReference['token_size'] })}
                    sx={{ minWidth: 130 }}
                  >
                    {TOKEN_SIZES.map(size => <MenuItem key={size} value={size}>{size}</MenuItem>)}
                  </TextField>
                  <TextField
                    size="small"
                    type="number"
                    label="Vision MP"
                    value={ref.vision_megapixels ?? ''}
                    onChange={e => updateRef(index, { vision_megapixels: e.target.value ? Number(e.target.value) : null })}
                    inputProps={{ min: 0.1, max: 4, step: 0.1 }}
                    sx={{ minWidth: 110 }}
                  />
                  <TextField
                    size="small"
                    type="number"
                    label="Mask padding"
                    value={ref.mask_padding ?? 0}
                    onChange={e => updateRef(index, { mask_padding: Math.max(0, Number(e.target.value) || 0) })}
                    inputProps={{ min: 0, max: 512, step: 8 }}
                    sx={{ minWidth: 125 }}
                  />
                  <TextField
                    select
                    size="small"
                    label="Image position"
                    value={ref.vision_position ?? 'before_prompt'}
                    onChange={e => updateRef(index, { vision_position: e.target.value as StyleReference['vision_position'] })}
                    sx={{ minWidth: 150 }}
                  >
                    <MenuItem value="before_prompt">Image before prompt</MenuItem>
                    <MenuItem value="after_prompt">Image after prompt</MenuItem>
                  </TextField>
                  <Button size="small" component="label" variant={ref.mask_b64 ? 'contained' : 'outlined'}>
                    {ref.mask_b64 ? 'Mask set' : 'Add mask'}
                    <input
                      type="file"
                      accept="image/*"
                      hidden
                      onChange={e => {
                        addMask(index, e.target.files?.[0])
                        e.currentTarget.value = ''
                      }}
                    />
                  </Button>
                </Stack>
                <TextField
                  size="small"
                  label="Reference instruction"
                  value={ref.system_prompt ?? ''}
                  onChange={e => updateRef(index, { system_prompt: e.target.value.slice(0, 512) })}
                  fullWidth
                  multiline
                  maxRows={2}
                  sx={{ mt: 1 }}
                  helperText="Optional Qwen system prompt override for this reference, max 512 characters."
                />
              </Box>
            ))}

            <Tooltip title={refs.length >= MAX_STYLE_REFS ? 'Maximum 10 style references' : 'Add style reference image(s)'}>
              <span>
                <Button
                  variant="outlined"
                  startIcon={<AddPhotoAlternateIcon />}
                  onClick={() => fileRef.current?.click()}
                  disabled={refs.length >= MAX_STYLE_REFS}
                  fullWidth
                >
                  Add Style Reference
                </Button>
              </span>
            </Tooltip>
            <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={addImages} />
          </Stack>
        </Box>
      </Collapse>
    </Box>
  )
}
