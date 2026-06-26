import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react'
import {
  Alert, Box, Button, Card, CardActionArea, CardContent, Chip, Collapse,
  FormControl, FormControlLabel, InputLabel, MenuItem, Select, Slider, Stack,
  Switch, TextField, ToggleButton, ToggleButtonGroup, Typography,
} from '@mui/material'
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import BrushIcon from '@mui/icons-material/Brush'
import CollectionsIcon from '@mui/icons-material/Collections'
import CompareIcon from '@mui/icons-material/Compare'
import DesignServicesIcon from '@mui/icons-material/DesignServices'
import OpenInFullIcon from '@mui/icons-material/OpenInFull'
import PaletteIcon from '@mui/icons-material/Palette'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import { useStore } from '../../store'
import MaskCanvas from '../Inpaint/MaskCanvas'
import { buildOutpaintImage } from '../../lib/outpaint'

type StudioTaskId = 'recreate' | 'insert' | 'extend' | 'sketch' | 'style' | 'moodboard' | 'preserve'
type ReferenceRole = 'scene' | 'person' | 'object' | 'sketch' | 'style' | 'mood'
type PipelineKind = 'redraw' | 'img2img' | 'inpaint' | 'outpaint'
type ExtendMode = 'redraw' | 'preserve'
type PreserveMode = 'whole' | 'masked'

interface ReferenceSlot {
  id: string
  label: string
  role: ReferenceRole
  image: string
  note: string
}

interface StudioTask {
  id: StudioTaskId
  title: string
  kicker: string
  description: string
  icon: typeof AutoFixHighIcon
  defaultInstruction: string
  pipeline: PipelineKind
  slots: Array<Pick<ReferenceSlot, 'id' | 'label' | 'role' | 'note'>>
}

const roleCopy: Record<ReferenceRole, string> = {
  scene: 'scene/location and composition',
  person: 'person or character reference',
  object: 'object/prop reference',
  sketch: 'sketch/layout reference',
  style: 'visual style reference only',
  mood: 'mood, lighting, color, and atmosphere',
}

const roleGuide: Record<ReferenceRole, string> = {
  scene: 'Use for a place, room, landscape, camera angle, or base composition.',
  person: 'Use for identity, clothing, pose, or character design. Exact identity is not guaranteed.',
  object: 'Use for a prop, product, vehicle, animal, logo-like shape, or item to include.',
  sketch: 'Use for rough layout, drawn shapes, pose, composition, or product silhouette.',
  style: 'Use for medium, rendering style, palette, texture, lens, era, or art direction.',
  mood: 'Use for lighting, weather, time of day, color grade, and atmosphere.',
}

const roleNotePlaceholder: Record<ReferenceRole, string> = {
  scene: 'e.g. keep this camera angle and waterfall background',
  person: 'e.g. preserve face likeness and red jacket, standing near the falls',
  object: 'e.g. add this backpack near the person, match perspective',
  sketch: 'e.g. use the drawing as layout, render as realistic product photography',
  style: 'e.g. use only the painterly texture and color palette',
  mood: 'e.g. use the foggy blue moonlight, not the subject',
}

const tasks: StudioTask[] = [
  {
    id: 'recreate',
    title: 'Recreate / Redraw',
    kicker: 'Best for rough or low-quality sources',
    description: 'Use an image as the idea and generate one finished coherent version.',
    icon: AutoFixHighIcon,
    defaultInstruction: 'Create a finished coherent image based on the reference. Preserve the main subject and composition, but redraw the whole frame so lighting, style, and detail are unified.',
    pipeline: 'redraw',
    slots: [
      { id: 'scene', label: 'Picture 1', role: 'scene', note: 'Use as the base composition.' },
      { id: 'style', label: 'Picture 2', role: 'style', note: '' },
      { id: 'mood', label: 'Picture 3', role: 'mood', note: '' },
      { id: 'extra', label: 'Picture 4', role: 'object', note: '' },
    ],
  },
  {
    id: 'insert',
    title: 'Add or Replace',
    kicker: 'Scene plus person/object references',
    description: 'Place a subject or object from one image into another scene.',
    icon: AddPhotoAlternateIcon,
    defaultInstruction: 'Place the referenced person or object into the scene as a coherent new image. Match lighting, perspective, scale, shadows, and style.',
    pipeline: 'redraw',
    slots: [
      { id: 'scene', label: 'Scene', role: 'scene', note: 'Use as the location/background.' },
      { id: 'subject', label: 'Subject', role: 'person', note: 'Insert this subject into the scene.' },
      { id: 'object', label: 'Object', role: 'object', note: '' },
      { id: 'style', label: 'Style', role: 'style', note: '' },
    ],
  },
  {
    id: 'extend',
    title: 'Extend Image',
    kicker: 'Outpaint or redraw wider/taller',
    description: 'Expand a canvas. Preserve photos exactly, or redraw rough sources into a wide frame.',
    icon: OpenInFullIcon,
    defaultInstruction: 'Extend the image into a wider finished composition with no visible border. Preserve the subject and mood.',
    pipeline: 'outpaint',
    slots: [
      { id: 'scene', label: 'Source', role: 'scene', note: 'Use as the image to extend.' },
      { id: 'style', label: 'Style', role: 'style', note: '' },
      { id: 'mood', label: 'Mood', role: 'mood', note: '' },
      { id: 'extra', label: 'Extra ref', role: 'object', note: '' },
    ],
  },
  {
    id: 'sketch',
    title: 'Sketch to Realistic',
    kicker: 'Drawings to polished images',
    description: 'Turn a sketch, layout, or simple drawing into realism or a selected style.',
    icon: BrushIcon,
    defaultInstruction: 'Use the sketch as layout and composition. Render it as a polished realistic image with coherent lighting, materials, shadows, and detail.',
    pipeline: 'redraw',
    slots: [
      { id: 'sketch', label: 'Sketch', role: 'sketch', note: 'Use as layout and silhouette.' },
      { id: 'style', label: 'Target style', role: 'style', note: 'Use as the final visual style.' },
      { id: 'mood', label: 'Mood', role: 'mood', note: '' },
      { id: 'object', label: 'Object detail', role: 'object', note: '' },
    ],
  },
  {
    id: 'style',
    title: 'Style Transfer',
    kicker: 'Keep concept, change look',
    description: 'Apply the look of one or more images to a subject or scene.',
    icon: PaletteIcon,
    defaultInstruction: 'Keep the main subject and composition conceptually similar, but redraw the image using the style references.',
    pipeline: 'redraw',
    slots: [
      { id: 'scene', label: 'Subject', role: 'scene', note: 'Use as the subject/composition.' },
      { id: 'style1', label: 'Style 1', role: 'style', note: 'Use for art direction only.' },
      { id: 'style2', label: 'Style 2', role: 'style', note: '' },
      { id: 'mood', label: 'Mood', role: 'mood', note: '' },
    ],
  },
  {
    id: 'moodboard',
    title: 'Moodboard Direction',
    kicker: 'Art direct from references',
    description: 'Blend several images into a visual direction for a new generation.',
    icon: CollectionsIcon,
    defaultInstruction: 'Create a new image guided by the moodboard references. Use their shared style, palette, lighting, texture, and atmosphere.',
    pipeline: 'redraw',
    slots: [
      { id: 'mood1', label: 'Mood 1', role: 'mood', note: '' },
      { id: 'mood2', label: 'Mood 2', role: 'mood', note: '' },
      { id: 'style', label: 'Style', role: 'style', note: '' },
      { id: 'scene', label: 'Scene idea', role: 'scene', note: '' },
    ],
  },
  {
    id: 'preserve',
    title: 'Preserve Pixels',
    kicker: 'Strict edit / inpaint / img2img',
    description: 'Keep the source exact where possible. Use this for precise edits and masks.',
    icon: CompareIcon,
    defaultInstruction: 'Edit the source while preserving unmasked pixels and keeping the result coherent.',
    pipeline: 'img2img',
    slots: [
      { id: 'source', label: 'Source', role: 'scene', note: 'Pixels to preserve or edit.' },
      { id: 'reference', label: 'Reference', role: 'object', note: '' },
      { id: 'style', label: 'Style', role: 'style', note: '' },
      { id: 'mood', label: 'Mood', role: 'mood', note: '' },
    ],
  },
]

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
    img.onerror = () => reject(new Error('Could not load image'))
    img.src = `data:image/png;base64,${b64}`
  })
}

function align16(value: number) {
  return Math.max(16, Math.ceil(value / 16) * 16)
}

async function wideFrameForImage(b64: string) {
  const { width, height } = await imageSize(b64)
  const targetRatio = 16 / 9
  if (width / height < targetRatio) {
    return { width: align16(Math.round(height * targetRatio)), height: align16(height) }
  }
  return { width: align16(width), height: align16(Math.round(width / targetRatio)) }
}

function roleInstruction(slot: ReferenceSlot, pictureNumber: number) {
  const note = slot.note.trim()
  return `Picture ${pictureNumber} is the ${roleCopy[slot.role]}.${note ? ` ${note}` : ''}`
}

function buildRolePrompt(slots: ReferenceSlot[], instruction: string, task: StudioTask) {
  const active = slots.filter(slot => slot.image)
  const roleLines = active.map((slot, idx) => roleInstruction(slot, idx + 1)).join('\n')
  return [
    roleLines,
    task.defaultInstruction,
    instruction.trim() || 'Generate the final image from these references.',
  ].filter(Boolean).join('\n\n')
}

function slotsForTask(task: StudioTask, seedImage = ''): ReferenceSlot[] {
  return task.slots.map((slot, index) => ({
    ...slot,
    image: index === 0 ? seedImage : '',
  }))
}

function ReferenceCard({
  slot,
  onImage,
  onRole,
  onNote,
  onClear,
}: {
  slot: ReferenceSlot
  onImage: (b64: string) => void
  onRole: (role: ReferenceRole) => void
  onNote: (note: string) => void
  onClear: () => void
}) {
  const inputId = `studio-${slot.id}`
  const handleFile = useCallback(async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    onImage(await readFileB64(file))
  }, [onImage])

  return (
    <Card variant="outlined" sx={{ borderRadius: 3, height: '100%' }}>
      <CardContent>
        <Stack spacing={1.25}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" gap={1}>
            <Typography variant="subtitle2">{slot.label}</Typography>
            {slot.image && <Chip label="Loaded" size="small" color="success" variant="outlined" />}
          </Stack>
          <Box
            onClick={() => document.getElementById(inputId)?.click()}
            sx={{
              minHeight: 154,
              border: '1px dashed',
              borderColor: slot.image ? 'divider' : 'primary.main',
              borderRadius: 2,
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
                style={{ width: '100%', height: 154, objectFit: 'cover' }}
              />
            ) : (
              <Stack alignItems="center" spacing={0.75} sx={{ p: 2, color: 'text.secondary', textAlign: 'center' }}>
                <UploadFileIcon />
                <Typography variant="body2">Upload image</Typography>
              </Stack>
            )}
          </Box>
          <FormControl size="small" fullWidth>
            <InputLabel>Role</InputLabel>
            <Select label="Role" value={slot.role} onChange={e => onRole(e.target.value as ReferenceRole)}>
              <MenuItem value="scene">Scene / location</MenuItem>
              <MenuItem value="person">Person / subject</MenuItem>
              <MenuItem value="object">Object / prop</MenuItem>
              <MenuItem value="sketch">Sketch / layout</MenuItem>
              <MenuItem value="style">Style only</MenuItem>
              <MenuItem value="mood">Mood / lighting</MenuItem>
            </Select>
          </FormControl>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>{roleGuide[slot.role]}</Typography>
          <TextField
            label="Role note"
            size="small"
            value={slot.note}
            onChange={e => onNote(e.target.value)}
            placeholder={roleNotePlaceholder[slot.role]}
            helperText="Short notes work best: what to preserve, ignore, or place."
            multiline
            minRows={2}
          />
          {slot.image && <Button size="small" color="inherit" onClick={onClear}>Clear image</Button>}
        </Stack>
      </CardContent>
    </Card>
  )
}

export default function RedrawStudio() {
  const { params, setParams } = useStore()
  const initialTaskId: StudioTaskId = params.mode === 'outpaint'
    ? 'extend'
    : params.mode === 'inpaint' || params.mode === 'img2img'
      ? 'preserve'
      : 'recreate'
  const initialTask = tasks.find(task => task.id === initialTaskId) ?? tasks[0]
  const [taskId, setTaskId] = useState<StudioTaskId>(initialTask.id)
  const [slots, setSlots] = useState<ReferenceSlot[]>(() => slotsForTask(initialTask, params.init_image_b64))
  const [instruction, setInstruction] = useState('')
  const [extendMode, setExtendMode] = useState<ExtendMode>('redraw')
  const [preserveMode, setPreserveMode] = useState<PreserveMode>('whole')
  const [outpaintOverlap, setOutpaintOverlap] = useState(128)
  const [denoise, setDenoise] = useState(0.72)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [readyMessage, setReadyMessage] = useState<string | null>(null)

  const task = tasks.find(item => item.id === taskId) ?? initialTask
  const activeImages = useMemo(() => slots.filter(slot => slot.image), [slots])
  const promptPreview = useMemo(() => buildRolePrompt(slots, instruction, task), [slots, instruction, task])
  const sourceImage = slots[0]?.image ?? ''

  useEffect(() => {
    setParams({ mode: params.mode === 'txt2img' ? 'redraw' : params.mode })
  }, [])

  const selectTask = (nextTaskId: StudioTaskId) => {
    const nextTask = tasks.find(item => item.id === nextTaskId) ?? initialTask
    const carryImage = slots[0]?.image || params.init_image_b64
    setTaskId(nextTaskId)
    setSlots(slotsForTask(nextTask, carryImage))
    setInstruction('')
    setReadyMessage(null)
  }

  const updateSlot = (id: string, patch: Partial<ReferenceSlot>) => {
    setSlots(current => current.map(slot => slot.id === id ? { ...slot, ...patch } : slot))
    setReadyMessage(null)
  }

  const prepareReferenceRedraw = (dimensions?: { width: number; height: number }) => {
    setParams({
      mode: 'redraw',
      prompt: promptPreview,
      negative_prompt: params.negative_prompt || 'pasted collage, mismatched lighting, wrong scale, duplicate person, deformed face, extra limbs, bad hands, text artifacts',
      init_image_b64: '',
      mask_b64: '',
      ref_image1_b64: '',
      ref_image2_b64: '',
      ref_image3_b64: '',
      moodboard_images: activeImages.map(slot => slot.image),
      moodboard_strength: task.id === 'moodboard' ? 0.85 : 0.75,
      denoise: 1.0,
      width: dimensions?.width ?? params.width,
      height: dimensions?.height ?? params.height,
      use_rebalance: true,
      rebalance_multiplier: Math.max(params.rebalance_multiplier || 4, 4),
    })
    setReadyMessage('Reference redraw is ready. Use Generate below.')
  }

  const preparePreserve = () => {
    if (!sourceImage) return
    setParams({
      mode: preserveMode === 'masked' ? 'inpaint' : 'img2img',
      prompt: promptPreview,
      init_image_b64: sourceImage,
      mask_b64: preserveMode === 'masked' ? params.mask_b64 : '',
      moodboard_images: activeImages.slice(1).map(slot => slot.image),
      denoise,
      use_rebalance: true,
    })
    setReadyMessage(preserveMode === 'masked'
      ? 'Masked preserve edit is ready. Paint or adjust the mask, then Generate.'
      : 'Whole-image preserve edit is ready. Use Generate below.')
  }

  const prepareExtend = async () => {
    if (!sourceImage) return
    const target = await wideFrameForImage(sourceImage)
    if (extendMode === 'redraw') {
      prepareReferenceRedraw(target)
      return
    }
    const { width, height } = await imageSize(sourceImage)
    const horizontal = Math.max(0, target.width - align16(width))
    const vertical = Math.max(0, target.height - align16(height))
    const result = await buildOutpaintImage(
      sourceImage,
      {
        left: Math.floor(horizontal / 2),
        right: Math.ceil(horizontal / 2),
        top: Math.floor(vertical / 2),
        bottom: Math.ceil(vertical / 2),
      },
      outpaintOverlap,
    )
    setParams({
      mode: 'outpaint',
      prompt: promptPreview,
      init_image_b64: result.init_image_b64,
      mask_b64: result.mask_b64,
      moodboard_images: activeImages.slice(1).map(slot => slot.image),
      width: result.width,
      height: result.height,
      denoise: 1.0,
      use_rebalance: true,
    })
    setReadyMessage('Preserve-source extension is ready. Use Generate below.')
  }

  const prepare = async () => {
    if (!activeImages.length) return
    if (task.id === 'preserve') {
      preparePreserve()
      return
    }
    if (task.id === 'extend') {
      await prepareExtend()
      return
    }
    prepareReferenceRedraw()
  }

  const needsMask = task.id === 'preserve' && preserveMode === 'masked' && sourceImage

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2 }, maxWidth: 1180, mx: 'auto' }}>
      <Stack spacing={2.5}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 700, mb: 0.5 }}>Redraw Studio</Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', maxWidth: 780 }}>
            Choose what you want to do. Redraw tasks reinterpret images into one coherent result; preserve-pixel tasks keep the source exact where possible.
          </Typography>
        </Box>

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }, gap: 1.5 }}>
          {tasks.map(item => {
            const Icon = item.icon
            const selected = item.id === task.id
            return (
              <Card
                key={item.id}
                variant="outlined"
                sx={{
                  borderRadius: 4,
                  borderColor: selected ? 'primary.main' : 'divider',
                  bgcolor: selected ? 'action.selected' : 'background.paper',
                }}
              >
                <CardActionArea onClick={() => selectTask(item.id)} sx={{ minHeight: 162, p: 0.5 }}>
                  <CardContent>
                    <Stack spacing={1}>
                      <Box sx={{ width: 44, height: 44, borderRadius: 3, display: 'grid', placeItems: 'center', bgcolor: selected ? 'primary.main' : 'action.hover', color: selected ? 'primary.contrastText' : 'primary.main' }}>
                        <Icon />
                      </Box>
                      <Box>
                        <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>{item.title}</Typography>
                        <Typography variant="caption" sx={{ color: 'primary.main', fontWeight: 700 }}>{item.kicker}</Typography>
                      </Box>
                      <Typography variant="body2" sx={{ color: 'text.secondary' }}>{item.description}</Typography>
                    </Stack>
                  </CardContent>
                </CardActionArea>
              </Card>
            )
          })}
        </Box>

        <Card variant="outlined" sx={{ borderRadius: 4 }}>
          <CardContent>
            <Stack spacing={2}>
              <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" gap={1}>
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 700 }}>{task.title}</Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>{task.description}</Typography>
                </Box>
                <Chip
                  label={task.id === 'preserve' || (task.id === 'extend' && extendMode === 'preserve') ? 'Preserve pixels' : 'Redraw whole image'}
                  color={task.id === 'preserve' || (task.id === 'extend' && extendMode === 'preserve') ? 'secondary' : 'primary'}
                  variant="outlined"
                />
              </Stack>

              {task.id === 'extend' && (
                <Box>
                  <Typography variant="body2" sx={{ mb: 1 }}>Extend strategy</Typography>
                  <ToggleButtonGroup value={extendMode} exclusive onChange={(_, value) => value && setExtendMode(value)} size="small" color="primary">
                    <ToggleButton value="redraw">Creative redraw</ToggleButton>
                    <ToggleButton value="preserve">Preserve source</ToggleButton>
                  </ToggleButtonGroup>
                  <Typography variant="caption" sx={{ display: 'block', mt: 0.75, color: 'text.secondary' }}>
                    Creative redraw is best for sketches or rough images. Preserve source is best for finished photos.
                  </Typography>
                </Box>
              )}

              {task.id === 'preserve' && (
                <Box>
                  <Typography variant="body2" sx={{ mb: 1 }}>Preserve mode</Typography>
                  <ToggleButtonGroup value={preserveMode} exclusive onChange={(_, value) => value && setPreserveMode(value)} size="small" color="primary">
                    <ToggleButton value="whole">Whole image edit</ToggleButton>
                    <ToggleButton value="masked">Masked edit</ToggleButton>
                  </ToggleButtonGroup>
                </Box>
              )}

              <Stack direction={{ xs: 'column', md: 'row' }} spacing={1.5}>
                {slots.map(slot => (
                  <Box key={slot.id} sx={{ flex: 1, minWidth: 0 }}>
                    <ReferenceCard
                      slot={slot}
                      onImage={image => updateSlot(slot.id, { image })}
                      onRole={role => updateSlot(slot.id, { role })}
                      onNote={note => updateSlot(slot.id, { note })}
                      onClear={() => updateSlot(slot.id, { image: '' })}
                    />
                  </Box>
                ))}
              </Stack>

              {needsMask && (
                <Box>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>Mask placement</Typography>
                  <MaskCanvas imageB64={sourceImage} onMaskChange={mask => setParams({ mask_b64: mask })} />
                </Box>
              )}

              <TextField
                label="What should Krea create?"
                value={instruction}
                onChange={e => { setInstruction(e.target.value); setReadyMessage(null) }}
                placeholder="Example: put Picture 2 standing at Niagara Falls from Picture 1, in the cinematic lighting from Picture 4."
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
                helperText="This is the role-aware prompt that Prepare sends into the shared Generate panel."
              />

              <FormControlLabel
                control={<Switch checked={showAdvanced} onChange={e => setShowAdvanced(e.target.checked)} />}
                label="Show task-specific advanced controls"
              />
              <Collapse in={showAdvanced}>
                <Stack spacing={2}>
                  {(task.id === 'preserve' || (task.id === 'extend' && extendMode === 'preserve')) && (
                    <Box>
                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="body2">Denoise strength</Typography>
                        <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono', fontSize: 12 }}>{denoise.toFixed(2)}</Typography>
                      </Stack>
                      <Slider value={denoise} min={0.05} max={1} step={0.01} onChange={(_, value) => setDenoise(value as number)} />
                    </Box>
                  )}
                  {task.id === 'extend' && extendMode === 'preserve' && (
                    <Box>
                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="body2">Blend overlap</Typography>
                        <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono', fontSize: 12 }}>{outpaintOverlap}px</Typography>
                      </Stack>
                      <Slider value={outpaintOverlap} min={0} max={192} step={8} onChange={(_, value) => setOutpaintOverlap(value as number)} />
                    </Box>
                  )}
                </Stack>
              </Collapse>

              {readyMessage && <Alert severity="success">{readyMessage}</Alert>}

              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
                <Button variant="contained" size="large" onClick={prepare} disabled={!activeImages.length} fullWidth>
                  Prepare {task.title}
                </Button>
                <Button variant="outlined" size="large" onClick={() => selectTask(task.id)} fullWidth>
                  Reset task
                </Button>
              </Stack>

              <Alert severity="info">
                Reference redraw is not face-swap compositing. Use fewer references for stronger fidelity, and choose Preserve Pixels when exact source pixels matter.
              </Alert>
            </Stack>
          </CardContent>
        </Card>
      </Stack>
    </Box>
  )
}
