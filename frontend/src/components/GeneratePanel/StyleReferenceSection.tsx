import React, { useRef, useState } from 'react'
import {
  Box, Button, Collapse, FormControlLabel, IconButton, MenuItem, Slider, Stack, Switch, TextField, Tooltip, Typography,
} from '@mui/material'
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate'
import CloseIcon from '@mui/icons-material/Close'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import { useStore, type StyleReference } from '../../store'

// Opt-in image prompting. Moodboard text guidance remains the default/primary path.
// "Match style" averages separate style-only image encodes; "Copy composition"
// uses the older multi-image Krea2 encode that can intentionally copy/compose refs.
const MAX_REFS = 4

const INFLUENCE: Array<{ value: StyleReference['token_size']; label: string }> = [
  { value: 'low', label: 'Subtle' },
  { value: 'normal', label: 'Balanced' },
  { value: 'high', label: 'Strong' },
  { value: 'max', label: 'Max' },
]

export default function StyleReferenceSection() {
  const { params, setParam } = useStore()
  const fileRef = useRef<HTMLInputElement>(null)
  const refs = params.style_references
  const active = refs.length > 0 || params.image_prompt_enabled
  const [open, setOpen] = useState(active)
  const hasSelectedMoodboard = params.selected_moodboard_ids.length > 0 || params.moodboard_uuids.length > 0
  const mode = params.image_prompt_mode
  const strengthLabel = params.image_prompt_strength <= 0.12
    ? 'Strong'
    : params.image_prompt_strength <= 0.25 ? 'Balanced' : 'Subtle'

  const updateRef = (index: number, patch: Partial<StyleReference>) => {
    setParam('style_references', refs.map((ref, i) => (i === index ? { ...ref, ...patch } : ref)))
  }

  const removeRef = (index: number) => {
    setParam('style_references', refs.filter((_, i) => i !== index))
  }

  const addImages = (e: React.ChangeEvent<HTMLInputElement>) => {
    const remaining = Math.max(0, MAX_REFS - refs.length)
    const files = Array.from(e.target.files ?? []).slice(0, remaining)
    files.forEach(file => {
      const reader = new FileReader()
      reader.onload = ev => {
        const b64 = String(ev.target?.result || '').split(',')[1]
        if (!b64) return
        const current = useStore.getState().params.style_references
        if (current.length >= MAX_REFS) return
        setParam('image_prompt_enabled', true)
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
          Image Prompt{active ? ` · ${params.image_prompt_enabled ? mode === 'match_style' ? 'match style' : 'copy composition' : 'off'}${refs.length ? ` · ${refs.length}/${MAX_REFS} upload` : ''}` : ''}
          <Tooltip title="Moodboard text remains the default. Turn this on only when you also want selected moodboard images or uploaded images to influence generation. Match style transfers feel/texture; Copy composition may copy subjects/layout.">
            <InfoOutlinedIcon sx={{ fontSize: 14, color: 'text.disabled', ml: 0.5, verticalAlign: 'middle' }} />
          </Tooltip>
        </Typography>
        <ExpandMoreIcon sx={{ color: 'text.secondary', transform: open ? 'rotate(180deg)' : 'none', transition: '0.2s' }} />
      </Stack>

      <Collapse in={open}>
        <Box sx={{ pt: 1 }}>
          <Typography variant="caption" sx={{ color: 'text.disabled', mb: 1.25, display: 'block' }}>
            Text guidance from selected moodboards is still the default. Enable images when you want the board's
            pictures or your uploads to affect the generation directly. Use Match style for feel/medium; use Copy
            composition only when you want reference content/layout.
          </Typography>

          <FormControlLabel
            control={<Switch checked={params.image_prompt_enabled} onChange={e => setParam('image_prompt_enabled', e.target.checked)} />}
            label={hasSelectedMoodboard ? 'Use selected moodboard images / uploaded refs' : 'Use uploaded reference images'}
            sx={{ mb: 1 }}
          />

          {params.image_prompt_enabled && (
            <Stack spacing={1.5} sx={{ mb: 1.5 }}>
              <TextField
                select
                size="small"
                label="Image mode"
                value={params.image_prompt_mode}
                onChange={e => setParam('image_prompt_mode', e.target.value as typeof params.image_prompt_mode)}
                fullWidth
                helperText={mode === 'match_style'
                  ? 'Averages images separately to transfer shared feel without collage. Best for neutral subjects.'
                  : 'Uses the old multi-image path. Can copy subjects/layout or collage distinct refs.'}
              >
                <MenuItem value="match_style">Match style (recommended)</MenuItem>
                <MenuItem value="copy_composition">Copy composition / subject</MenuItem>
              </TextField>

              {mode === 'match_style' && (
                <Box>
                  <Stack direction="row" justifyContent="space-between" alignItems="baseline">
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>Style strength</Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono', fontSize: 12 }}>
                      {strengthLabel} · MP {params.image_prompt_strength.toFixed(2)}
                    </Typography>
                  </Stack>
                  <Slider
                    value={params.image_prompt_strength}
                    min={0.1}
                    max={1}
                    step={0.05}
                    size="small"
                    marks={[{ value: 0.1, label: 'Strong' }, { value: 0.2, label: 'Balanced' }, { value: 0.5, label: 'Subtle' }]}
                    onChange={(_, v) => setParam('image_prompt_strength', v as number)}
                  />
                  <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                    Lower values transfer more mood/medium. 0.20 is the tested balanced default.
                  </Typography>
                </Box>
              )}
            </Stack>
          )}

          <Stack spacing={1}>
            {refs.map((ref, index) => (
              <Box key={`${index}-${ref.image_b64.slice(0, 12)}`}
                sx={{ border: '1px solid rgba(202,196,208,0.16)', borderRadius: 1, p: 1 }}>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Box sx={{ position: 'relative', width: 56, height: 56, borderRadius: 1, overflow: 'hidden', flex: '0 0 auto' }}>
                    <img src={`data:image/png;base64,${ref.image_b64}`} alt={`reference ${index + 1}`}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    <IconButton size="small" onClick={() => removeRef(index)}
                      sx={{ position: 'absolute', top: -2, right: -2, p: '1px', bgcolor: 'rgba(0,0,0,0.6)' }}>
                      <CloseIcon sx={{ fontSize: 12 }} />
                    </IconButton>
                  </Box>

                  <TextField
                    select
                    size="small"
                    label={mode === 'copy_composition' ? `Image ${index + 1} influence` : `Image ${index + 1}`}
                    value={ref.token_size}
                    onChange={e => updateRef(index, { token_size: e.target.value as StyleReference['token_size'] })}
                    helperText={mode === 'copy_composition' ? 'Only used by Copy composition.' : 'Included in averaged style.'}
                    fullWidth
                  >
                    {INFLUENCE.map(o => <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>)}
                  </TextField>
                </Stack>
              </Box>
            ))}

            <Tooltip title={refs.length >= MAX_REFS ? `Maximum ${MAX_REFS} reference images` : 'Add reference image(s)'}>
              <span>
                <Button
                  variant="outlined"
                  startIcon={<AddPhotoAlternateIcon />}
                  onClick={() => fileRef.current?.click()}
                  disabled={refs.length >= MAX_REFS}
                  sx={{ borderStyle: 'dashed' }}
                  fullWidth
                >
                  {refs.length ? 'Add another reference' : 'Add reference image'}
                </Button>
              </span>
            </Tooltip>
            <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={addImages} />
          </Stack>

          {active && (
            <>
              {mode === 'copy_composition' && (
              <Box sx={{ mt: 1.75 }}>
                <Stack direction="row" justifyContent="space-between" alignItems="baseline">
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>Overall strength</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono', fontSize: 12 }}>
                    {params.rebalance_multiplier.toFixed(2)}
                  </Typography>
                </Stack>
                <Slider
                  value={params.rebalance_multiplier}
                  min={0.25}
                  max={2}
                  step={0.05}
                  size="small"
                  marks={[{ value: 1, label: '1.0' }]}
                  onChange={(_, v) => setParam('rebalance_multiplier', v as number)}
                />
                <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                  1.0 = default · higher pulls harder toward the references · lower loosens their grip.
                </Typography>
              </Box>
              )}
            </>
          )}
        </Box>
      </Collapse>
    </Box>
  )
}
