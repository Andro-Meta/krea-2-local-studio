import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react'
import {
  Alert, Box, Button, Card, CardContent, Chip, FormControl, InputLabel, MenuItem,
  Select, Stack, TextField, ToggleButton, ToggleButtonGroup, Typography,
} from '@mui/material'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import { useStore } from '../../store'

type RedrawRole = 'scene' | 'person' | 'object' | 'style' | 'mood'

interface RedrawSlot {
  id: string
  label: string
  role: RedrawRole
  image: string
  note: string
}

const roleCopy: Record<RedrawRole, string> = {
  scene: 'scene/location and composition',
  person: 'person or character reference',
  object: 'object/prop reference',
  style: 'visual style reference only',
  mood: 'mood, lighting, color, and atmosphere',
}

const roleGuide: Record<RedrawRole, string> = {
  scene: 'Use for a place, room, landscape, camera angle, or base composition.',
  person: 'Use for identity, clothing, pose, or character design. Exact face identity is not guaranteed.',
  object: 'Use for a prop, product, vehicle, animal, logo-like shape, or item to include.',
  style: 'Use for art direction only: medium, rendering style, palette, texture, lens, era.',
  mood: 'Use for lighting, weather, time of day, color grade, atmosphere, or horror/romance/etc.',
}

const roleNotePlaceholder: Record<RedrawRole, string> = {
  scene: 'e.g. use Niagara Falls as the background, keep this camera angle',
  person: 'e.g. preserve face likeness and red jacket, standing near the falls',
  object: 'e.g. add this backpack near the person, match perspective',
  style: 'e.g. use only the painterly texture and color palette',
  mood: 'e.g. use the foggy blue moonlight, not the subject',
}

const modeGuide = {
  creative: 'Best when the source is rough, flat, or low quality. Krea redraws the whole image into one coherent result.',
  insert: 'Best when you want a person/object from one image placed into a scene from another. Use clear role notes.',
  style: 'Best when the subject/composition should stay conceptually similar but the look should come from style references.',
}

function readFileB64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error('Could not read image'))
    reader.onload = ev => resolve(String(ev.target?.result ?? '').split(',')[1])
    reader.readAsDataURL(file)
  })
}

function align16(value: number): number {
  return Math.max(16, Math.ceil(value / 16) * 16)
}

function roleInstruction(slot: RedrawSlot, pictureNumber: number) {
  const note = slot.note.trim()
  const noteText = note ? ` ${note}` : ''
  return `Picture ${pictureNumber} is the ${roleCopy[slot.role]}.${noteText}`
}

function buildPrompt(slots: RedrawSlot[], instruction: string, mode: 'creative' | 'insert' | 'style') {
  const active = slots.filter(slot => slot.image)
  const roles = active.map((slot, idx) => roleInstruction(slot, idx + 1)).join('\n')
  const userInstruction = instruction.trim()
  const modeInstruction = mode === 'insert'
    ? 'Place the referenced person/object into the scene as a coherent new image, matching lighting, perspective, scale, shadows, and style.'
    : mode === 'style'
      ? 'Redraw the base image using the style/mood references while preserving the main subject and composition.'
      : 'Create one finished coherent image using the references. Do not paste images together; redraw the whole frame so lighting, perspective, and style are unified.'

  return [
    roles,
    modeInstruction,
    userInstruction || 'Generate the final image from these references.',
  ].filter(Boolean).join('\n\n')
}

function UploadCard({
  slot,
  onImage,
  onRole,
  onNote,
  onClear,
}: {
  slot: RedrawSlot
  onImage: (b64: string) => void
  onRole: (role: RedrawRole) => void
  onNote: (note: string) => void
  onClear: () => void
}) {
  const inputId = `redraw-${slot.id}`
  const handleFile = useCallback(async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    onImage(await readFileB64(file))
  }, [onImage])

  return (
    <Card variant="outlined" sx={{ height: '100%', borderRadius: 3 }}>
      <CardContent>
        <Stack spacing={1.25}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" gap={1}>
            <Typography variant="subtitle2">{slot.label}</Typography>
            {slot.image && <Chip label="Loaded" size="small" color="success" variant="outlined" />}
          </Stack>

          <Box
            onClick={() => document.getElementById(inputId)?.click()}
            sx={{
              border: '1px dashed',
              borderColor: slot.image ? 'divider' : 'primary.main',
              borderRadius: 2,
              minHeight: 152,
              display: 'grid',
              placeItems: 'center',
              overflow: 'hidden',
              cursor: 'pointer',
              bgcolor: 'background.default',
            }}
          >
            <input id={inputId} type="file" accept="image/*" hidden onChange={handleFile} />
            {slot.image ? (
              <img
                src={`data:image/png;base64,${slot.image}`}
                alt={slot.label}
                style={{ width: '100%', height: 152, objectFit: 'cover' }}
              />
            ) : (
              <Stack alignItems="center" spacing={0.75} sx={{ color: 'text.secondary', p: 2, textAlign: 'center' }}>
                <UploadFileIcon />
                <Typography variant="body2">Upload reference</Typography>
              </Stack>
            )}
          </Box>

          <FormControl size="small" fullWidth>
            <InputLabel>Role</InputLabel>
            <Select label="Role" value={slot.role} onChange={e => onRole(e.target.value as RedrawRole)}>
              <MenuItem value="scene">Scene / location</MenuItem>
              <MenuItem value="person">Person / subject</MenuItem>
              <MenuItem value="object">Object / prop</MenuItem>
              <MenuItem value="style">Style only</MenuItem>
              <MenuItem value="mood">Mood / lighting</MenuItem>
            </Select>
          </FormControl>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {roleGuide[slot.role]}
          </Typography>

          <TextField
            size="small"
            label="Role note"
            value={slot.note}
            onChange={e => onNote(e.target.value)}
            placeholder={roleNotePlaceholder[slot.role]}
            helperText="Short notes work best: what to preserve, what to ignore, and where it belongs."
            multiline
            minRows={2}
          />

          {slot.image && (
            <Button size="small" variant="text" color="inherit" onClick={onClear}>
              Clear image
            </Button>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}

export default function RedrawPanel() {
  const { params, setParams } = useStore()
  const [mode, setMode] = useState<'creative' | 'insert' | 'style'>('creative')
  const [instruction, setInstruction] = useState('')
  const [slots, setSlots] = useState<RedrawSlot[]>([
    { id: 'scene', label: 'Picture 1', role: 'scene', image: '', note: 'Use as the base scene/location.' },
    { id: 'ref2', label: 'Picture 2', role: 'person', image: '', note: '' },
    { id: 'ref3', label: 'Picture 3', role: 'object', image: '', note: '' },
    { id: 'ref4', label: 'Picture 4', role: 'style', image: '', note: '' },
  ])

  useEffect(() => {
    setParams({ mode: 'redraw' })
  }, [])

  const activeImages = useMemo(() => slots.filter(slot => slot.image), [slots])
  const promptPreview = useMemo(
    () => buildPrompt(slots, instruction, mode),
    [slots, instruction, mode],
  )

  const updateSlot = (id: string, patch: Partial<RedrawSlot>) => {
    setSlots(current => current.map(slot => slot.id === id ? { ...slot, ...patch } : slot))
  }

  const prepare = () => {
    const prompt = buildPrompt(slots, instruction, mode)
    setParams({
      mode: 'redraw',
      prompt,
      negative_prompt: params.negative_prompt || 'pasted collage, mismatched lighting, wrong scale, duplicate person, deformed face, extra limbs, bad hands, text artifacts',
      init_image_b64: '',
      mask_b64: '',
      ref_image1_b64: '',
      ref_image2_b64: '',
      ref_image3_b64: '',
      moodboard_images: activeImages.map(slot => slot.image),
      moodboard_strength: 0.75,
      denoise: 1.0,
      width: align16(params.width || 1024),
      height: align16(params.height || 1024),
      use_rebalance: true,
      rebalance_multiplier: Math.max(params.rebalance_multiplier || 4, 4),
    })
  }

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2 }, maxWidth: 1100, mx: 'auto' }}>
      <Stack spacing={2}>
        <Alert severity="info" icon={<AutoFixHighIcon />}>
          Redraw creates a new coherent image from references. It does not preserve pixels exactly. Use it when you want
          Krea to reinterpret the inputs into one unified result.
        </Alert>

        <Stack direction={{ xs: 'column', md: 'row' }} spacing={1.5}>
          <Card variant="outlined" sx={{ flex: 1, borderRadius: 3 }}>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>How to think about references</Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                The role tells Krea what an image is for. The note tells Krea what matters inside that image.
                If you upload multiple people or props, be explicit: “Picture 2 is the person on the left” or
                “use only the jacket, not the background.”
              </Typography>
            </CardContent>
          </Card>
          <Card variant="outlined" sx={{ flex: 1, borderRadius: 3 }}>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>Boundaries</Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                This is reference-conditioned generation, not a face swap or exact compositing tool. One scene plus
                one or two references is strongest. Four different people from four photos can work as a concept,
                but likeness, count, pose, and clothing may drift.
              </Typography>
            </CardContent>
          </Card>
        </Stack>

        <Box>
          <Typography variant="body2" sx={{ mb: 1 }}>Redraw mode</Typography>
          <ToggleButtonGroup
            value={mode}
            exclusive
            onChange={(_, value) => value && setMode(value)}
            size="small"
            color="primary"
            sx={{ flexWrap: 'wrap', gap: 0.75, '& .MuiToggleButtonGroup-grouped': { borderRadius: 99, border: 1 } }}
          >
            <ToggleButton value="creative">Creative redraw</ToggleButton>
            <ToggleButton value="insert">Guided insert</ToggleButton>
            <ToggleButton value="style">Style redraw</ToggleButton>
          </ToggleButtonGroup>
          <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', mt: 0.75 }}>
            {modeGuide[mode]}
          </Typography>
        </Box>

        <Stack direction={{ xs: 'column', md: 'row' }} spacing={1.5}>
          {slots.map(slot => (
            <Box key={slot.id} sx={{ flex: 1, minWidth: 0 }}>
              <UploadCard
                slot={slot}
                onImage={image => updateSlot(slot.id, { image })}
                onRole={role => updateSlot(slot.id, { role })}
                onNote={note => updateSlot(slot.id, { note })}
                onClear={() => updateSlot(slot.id, { image: '' })}
              />
            </Box>
          ))}
        </Stack>

        <TextField
          label="What should Krea create?"
          value={instruction}
          onChange={e => setInstruction(e.target.value)}
          placeholder="Example: Put Picture 2 standing at Niagara Falls from Picture 1, with the jacket from Picture 3, in the cinematic lighting of Picture 4."
          multiline
          minRows={3}
          fullWidth
        />

        <TextField
          label="Prompt preview"
          value={promptPreview}
          multiline
          minRows={4}
          fullWidth
          InputProps={{ readOnly: true }}
          helperText="This is the exact role-aware prompt that will be sent after you click Prepare."
        />

        <Button variant="contained" size="large" onClick={prepare} disabled={!activeImages.length} fullWidth>
          Prepare redraw prompt from {activeImages.length} image{activeImages.length === 1 ? '' : 's'}
        </Button>

        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          Boundary: this is reference-conditioned generation, not identity-locked compositing. Use fewer references for stronger fidelity.
        </Typography>
      </Stack>
    </Box>
  )
}
