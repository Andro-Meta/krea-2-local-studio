import {
  useCallback, useEffect, useMemo, useRef, useState,
  type ChangeEvent, type DragEvent,
} from 'react'
import {
  Alert, Box, Button, Card, CardContent, Chip, CircularProgress, Collapse, Divider,
  FormControl, FormControlLabel, IconButton, InputLabel, MenuItem, Select, Slider,
  Stack, Switch, Tab, Tabs, TextField, ToggleButton, ToggleButtonGroup, Tooltip, Typography,
} from '@mui/material'
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import BrushIcon from '@mui/icons-material/Brush'
import CollectionsIcon from '@mui/icons-material/Collections'
import CompareIcon from '@mui/icons-material/Compare'
import OpenInFullIcon from '@mui/icons-material/OpenInFull'
import PaletteIcon from '@mui/icons-material/Palette'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import CloseIcon from '@mui/icons-material/Close'
import SwapHorizIcon from '@mui/icons-material/SwapHoriz'
import AspectRatioIcon from '@mui/icons-material/AspectRatio'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import StyleIcon from '@mui/icons-material/Style'
import { useStore } from '../../store'
import { apiFetch, type LoraInfo } from '../../api'
import MaskCanvas from '../Inpaint/MaskCanvas'
import { buildOutpaintImage } from '../../lib/outpaint'
import {
  loraToGeneration,
  presetFor,
  type RedrawQualityMode,
  type RedrawTaskKind,
} from './qualityPresets'

type StudioTaskId = 'recreate' | 'insert' | 'extend' | 'sketch' | 'style' | 'moodboard' | 'preserve' | 'nk2e_edit' | 'depth'
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

const roleNotePlaceholder: Record<ReferenceRole, string> = {
  scene: 'e.g. keep this camera angle and waterfall background',
  person: 'e.g. preserve face likeness and red jacket',
  object: 'e.g. add this backpack near the person, match perspective',
  sketch: 'e.g. use the drawing as layout, render as realistic photo',
  style: 'e.g. use only the painterly texture and color palette',
  mood: 'e.g. use the foggy blue moonlight, not the subject',
}

const tasks: StudioTask[] = [
  {
    id: 'recreate',
    title: 'Recreate',
    kicker: 'Redraw into one finished image',
    description: 'Use the image as the idea and generate one finished, coherent version. Best for rough or low-quality sources.',
    icon: AutoFixHighIcon,
    defaultInstruction: 'Create a finished coherent image based on the reference. Preserve the main subject and composition, but redraw the whole frame so lighting, style, and detail are unified.',
    pipeline: 'redraw',
    slots: [
      { id: 'scene', label: 'Source', role: 'scene', note: 'Use as the base composition.' },
      { id: 'style', label: 'Reference 2', role: 'style', note: '' },
      { id: 'mood', label: 'Reference 3', role: 'mood', note: '' },
      { id: 'extra', label: 'Reference 4', role: 'object', note: '' },
    ],
  },
  {
    id: 'depth',
    title: 'Depth Control',
    kicker: 'Match a photo\u2019s depth/composition',
    description: 'Extract a depth map from your source (Depth-Anything-3) and generate a brand-new image from your prompt that follows the source\u2019s 3D layout.',
    icon: OpenInFullIcon,
    defaultInstruction: '',
    pipeline: 'redraw',
    slots: [
      { id: 'scene', label: 'Depth source', role: 'scene', note: 'Its depth/composition guides the render.' },
    ],
  },
  {
    id: 'preserve',
    title: 'Edit (preserve)',
    kicker: 'Keep pixels · img2img / inpaint',
    description: 'Keep the source exact where possible. Use this for precise whole-image edits or masked inpainting.',
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
  {
    id: 'insert',
    title: 'Add / Replace',
    kicker: 'Put a subject into a scene',
    description: 'Place a subject or object from one image into another scene, matching lighting and perspective.',
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
    id: 'style',
    title: 'Style Transfer',
    kicker: 'Keep concept, change look',
    description: 'Apply the look of one or more images (or a Krea style LoRA) to your subject.',
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
    id: 'sketch',
    title: 'Sketch → Real',
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
    id: 'extend',
    title: 'Extend',
    kicker: 'Outpaint wider / taller',
    description: 'Expand the canvas. Preserve photos exactly, or redraw rough sources into a wide frame.',
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
    id: 'moodboard',
    title: 'Moodboard',
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
    id: 'nk2e_edit',
    title: 'NK2E Edit',
    kicker: 'Experimental instruction edit',
    description: 'Localized instruction edits (hair, accessories, objects, lighting) while preserving identity and composition.',
    icon: AutoFixHighIcon,
    defaultInstruction: 'Edit the source image according to the instruction while preserving identity, composition, camera angle, and lighting. Keep the change localized and coherent.',
    pipeline: 'img2img',
    slots: [
      { id: 'source', label: 'Source', role: 'scene', note: 'Use as the image to edit.' },
      { id: 'reference', label: 'Reference', role: 'object', note: '' },
      { id: 'style', label: 'Style', role: 'style', note: '' },
      { id: 'mood', label: 'Mood', role: 'mood', note: '' },
    ],
  },
]

// Denoise strength presets for the preserve/extend-preserve pipelines.
const STRENGTH_PRESETS = [
  { id: 'subtle', label: 'Subtle', value: 0.3, hint: 'Small changes, keep almost everything' },
  { id: 'balanced', label: 'Balanced', value: 0.5, hint: 'Noticeable edits, keep composition' },
  { id: 'strong', label: 'Strong', value: 0.72, hint: 'Big changes, looser adherence' },
  { id: 'reimagine', label: 'Reimagine', value: 1.0, hint: 'Full redraw, ignore original detail' },
] as const

const ROLE_OPTIONS: Array<{ value: ReferenceRole; label: string }> = [
  { value: 'scene', label: 'Scene / location' },
  { value: 'person', label: 'Person / subject' },
  { value: 'object', label: 'Object / prop' },
  { value: 'sketch', label: 'Sketch / layout' },
  { value: 'style', label: 'Style only' },
  { value: 'mood', label: 'Mood / lighting' },
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

function aspectLabel(w: number, h: number): string {
  const g = (a: number, b: number): number => (b === 0 ? a : g(b, a % b))
  const d = g(w, h) || 1
  const rw = w / d, rh = h / d
  if (rw <= 32 && rh <= 32) return `${rw}:${rh}`
  return (w / h).toFixed(2)
}

function roleInstruction(slot: ReferenceSlot, pictureNumber: number) {
  const note = slot.note.trim()
  return `Picture ${pictureNumber} is the ${roleCopy[slot.role]}.${note ? ` ${note}` : ''}`
}

function buildRolePrompt(slots: ReferenceSlot[], instruction: string, task: StudioTask, presetHint = '') {
  const active = slots.filter(slot => slot.image)
  const roleLines = active.map((slot, idx) => roleInstruction(slot, idx + 1)).join('\n')
  return [
    roleLines,
    task.defaultInstruction,
    presetHint,
    instruction.trim() || 'Generate the final image from these references.',
  ].filter(Boolean).join('\n\n')
}

function presetTaskFor(task: StudioTask, extendMode: ExtendMode, preserveMode: PreserveMode): RedrawTaskKind {
  if (task.id === 'extend') return extendMode === 'preserve' ? 'extend_preserve' : 'extend_redraw'
  if (task.id === 'preserve') return preserveMode === 'masked' ? 'preserve_masked' : 'preserve_whole'
  // Depth control has its own pipeline and doesn't use the quality preset; map it
  // to a harmless existing kind so the preset lookup stays typed.
  if (task.id === 'depth') return 'recreate'
  return task.id
}

function slotsForTask(task: StudioTask, seedImage = ''): ReferenceSlot[] {
  return task.slots.map((slot, index) => ({
    ...slot,
    image: index === 0 ? seedImage : '',
  }))
}

// ---------------------------------------------------------------------------
// Source dropzone (primary image) — drag/drop + paste + click
// ---------------------------------------------------------------------------

function SourceDropzone({
  image, roleLabel, onImage, onClear,
}: {
  image: string
  roleLabel: string
  onImage: (b64: string) => void
  onClear: () => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)
  const [size, setSize] = useState<{ width: number; height: number } | null>(null)

  useEffect(() => {
    let cancelled = false
    if (!image) { setSize(null); return }
    imageSize(image).then(s => { if (!cancelled) setSize(s) }).catch(() => setSize(null))
    return () => { cancelled = true }
  }, [image])

  const takeFile = useCallback(async (file?: File | null) => {
    if (file && file.type.startsWith('image/')) onImage(await readFileB64(file))
  }, [onImage])

  const onDrop = (e: DragEvent) => {
    e.preventDefault(); setDrag(false)
    takeFile(e.dataTransfer.files?.[0])
  }

  if (image) {
    return (
      <Box sx={{ borderRadius: 3, overflow: 'hidden', border: '1px solid', borderColor: 'divider', bgcolor: 'background.default' }}>
        <Box sx={{ position: 'relative', display: 'grid', placeItems: 'center', bgcolor: '#0c0d11', minHeight: 240, maxHeight: 460 }}>
          <img src={`data:image/png;base64,${image}`} alt="source"
            style={{ maxWidth: '100%', maxHeight: 460, objectFit: 'contain', display: 'block' }} />
          <Stack direction="row" spacing={0.5} sx={{ position: 'absolute', top: 8, right: 8 }}>
            <Tooltip title="Replace image" arrow>
              <IconButton size="small" onClick={() => inputRef.current?.click()}
                sx={{ bgcolor: 'rgba(0,0,0,0.55)', color: '#fff', '&:hover': { bgcolor: 'rgba(0,0,0,0.75)' } }}>
                <SwapHorizIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Remove image" arrow>
              <IconButton size="small" onClick={onClear}
                sx={{ bgcolor: 'rgba(0,0,0,0.55)', color: '#fff', '&:hover': { bgcolor: 'rgba(0,0,0,0.75)' } }}>
                <CloseIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Stack>
          <Chip label={roleLabel} size="small"
            sx={{ position: 'absolute', top: 8, left: 8, bgcolor: 'rgba(0,0,0,0.55)', color: '#fff', fontWeight: 600 }} />
          {size && (
            <Chip label={`${size.width}×${size.height} · ${aspectLabel(size.width, size.height)}`} size="small"
              sx={{ position: 'absolute', bottom: 8, left: 8, bgcolor: 'rgba(0,0,0,0.55)', color: '#fff', fontFamily: 'Roboto Mono', fontSize: 11 }} />
          )}
        </Box>
        <input ref={inputRef} type="file" accept="image/*" hidden
          onChange={(e: ChangeEvent<HTMLInputElement>) => takeFile(e.target.files?.[0])} />
      </Box>
    )
  }

  return (
    <Box
      onClick={() => inputRef.current?.click()}
      onDragOver={e => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
      sx={{
        borderRadius: 3, border: '2px dashed', borderColor: drag ? 'primary.main' : 'rgba(202,196,208,0.28)',
        bgcolor: drag ? 'action.hover' : 'background.default', cursor: 'pointer', minHeight: 220,
        display: 'grid', placeItems: 'center', textAlign: 'center', p: 3,
        transition: 'border-color .15s, background .15s', '&:hover': { borderColor: 'primary.main' },
      }}
    >
      <input ref={inputRef} type="file" accept="image/*" hidden
        onChange={(e: ChangeEvent<HTMLInputElement>) => takeFile(e.target.files?.[0])} />
      <Stack alignItems="center" spacing={1}>
        <Box sx={{ width: 56, height: 56, borderRadius: '50%', display: 'grid', placeItems: 'center', bgcolor: 'action.hover', color: 'primary.main' }}>
          <UploadFileIcon />
        </Box>
        <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>Drop, paste, or click to add your image</Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          Drag a file here · paste from clipboard (Ctrl/⌘+V) · or browse
        </Typography>
      </Stack>
    </Box>
  )
}

// ---------------------------------------------------------------------------
// Before / after compare slider
// ---------------------------------------------------------------------------

function BeforeAfter({ before, after }: { before: string; after: string }) {
  const [pos, setPos] = useState(50)
  return (
    <Box>
      <Box sx={{ position: 'relative', borderRadius: 3, overflow: 'hidden', border: '1px solid', borderColor: 'divider', bgcolor: '#0c0d11', userSelect: 'none' }}>
        <img src={`data:image/png;base64,${after}`} alt="after" style={{ display: 'block', width: '100%', objectFit: 'contain', maxHeight: 460 }} />
        <Box sx={{ position: 'absolute', inset: 0, width: `${pos}%`, overflow: 'hidden', borderRight: '2px solid rgba(255,255,255,0.7)' }}>
          <img src={`data:image/png;base64,${before}`} alt="before"
            style={{ display: 'block', height: '100%', width: 'auto', maxHeight: 460, objectFit: 'cover' }} />
        </Box>
        <Chip label="Before" size="small" sx={{ position: 'absolute', bottom: 8, left: 8, bgcolor: 'rgba(0,0,0,0.6)', color: '#fff' }} />
        <Chip label="After" size="small" sx={{ position: 'absolute', bottom: 8, right: 8, bgcolor: 'rgba(0,0,0,0.6)', color: '#fff' }} />
      </Box>
      <Slider value={pos} min={0} max={100} onChange={(_, v) => setPos(v as number)} size="small" sx={{ mt: 0.5 }} />
    </Box>
  )
}

// ---------------------------------------------------------------------------
// Compact reference card (optional extra references)
// ---------------------------------------------------------------------------

function CompactReference({
  slot, onImage, onRole, onNote, onClear,
}: {
  slot: ReferenceSlot
  onImage: (b64: string) => void
  onRole: (role: ReferenceRole) => void
  onNote: (note: string) => void
  onClear: () => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const takeFile = useCallback(async (file?: File | null) => {
    if (file && file.type.startsWith('image/')) onImage(await readFileB64(file))
  }, [onImage])

  return (
    <Card variant="outlined" sx={{ borderRadius: 2.5, width: 200, flexShrink: 0 }}>
      <CardContent sx={{ p: 1.25, '&:last-child': { pb: 1.25 } }}>
        <Stack spacing={0.75}>
          <Box
            onClick={() => inputRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); takeFile(e.dataTransfer.files?.[0]) }}
            sx={{
              position: 'relative', height: 120, borderRadius: 2, overflow: 'hidden', cursor: 'pointer',
              border: '1px dashed', borderColor: slot.image ? 'divider' : 'primary.main',
              display: 'grid', placeItems: 'center', bgcolor: 'background.default',
            }}
          >
            <input ref={inputRef} type="file" accept="image/*" hidden
              onChange={(e: ChangeEvent<HTMLInputElement>) => takeFile(e.target.files?.[0])} />
            {slot.image ? (
              <>
                <img src={`data:image/png;base64,${slot.image}`} alt={slot.label}
                  style={{ width: '100%', height: 120, objectFit: 'cover' }} />
                <IconButton size="small" onClick={e => { e.stopPropagation(); onClear() }}
                  sx={{ position: 'absolute', top: 4, right: 4, bgcolor: 'rgba(0,0,0,0.55)', color: '#fff', '&:hover': { bgcolor: 'rgba(0,0,0,0.75)' } }}>
                  <CloseIcon sx={{ fontSize: 15 }} />
                </IconButton>
              </>
            ) : (
              <Stack alignItems="center" spacing={0.25} sx={{ color: 'text.secondary' }}>
                <AddPhotoAlternateIcon fontSize="small" />
                <Typography variant="caption">{slot.label}</Typography>
              </Stack>
            )}
          </Box>
          <FormControl size="small" fullWidth>
            <Select value={slot.role} onChange={e => onRole(e.target.value as ReferenceRole)} sx={{ fontSize: 13 }}>
              {ROLE_OPTIONS.map(o => <MenuItem key={o.value} value={o.value} sx={{ fontSize: 13 }}>{o.label}</MenuItem>)}
            </Select>
          </FormControl>
          <TextField size="small" value={slot.note} onChange={e => onNote(e.target.value)}
            placeholder={roleNotePlaceholder[slot.role]} multiline minRows={1} maxRows={3}
            InputProps={{ sx: { fontSize: 12.5 } }} />
        </Stack>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------

export default function RedrawStudio() {
  const { params, setParams, results } = useStore()
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
  const [denoise, setDenoise] = useState(0.5)
  const [qualityMode, setQualityMode] = useState<RedrawQualityMode>('balanced')
  const [selectedStyleLora, setSelectedStyleLora] = useState('')
  const [loras, setLoras] = useState<LoraInfo[]>([])
  const [readyMessage, setReadyMessage] = useState<string | null>(null)
  const [showRefs, setShowRefs] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  // Edit-engine options (Actual-Denoise + Qwen-edit-style in-context vision).
  const [useActualDenoise, setUseActualDenoise] = useState(false)
  const [useIncontext, setUseIncontext] = useState(false)
  const [visionPosition, setVisionPosition] = useState<'before' | 'after'>('before')
  const [visionDetail, setVisionDetail] = useState(1.0)
  const [incontextEncoder, setIncontextEncoder] = useState<'krea2' | 'qwen_edit_plus'>('krea2')
  const [incontextSystemPrompt, setIncontextSystemPrompt] = useState('')
  const [describing, setDescribing] = useState<'' | 'recreate' | 'style'>('')
  const [useStyleTransfer, setUseStyleTransfer] = useState(false)
  // Depth ControlNet: strength of the depth Control LoRA + RAW/Turbo base.
  const [depthStrength, setDepthStrength] = useState(1.2)
  const [depthEngine, setDepthEngine] = useState<'raw' | 'turbo'>('raw')
  const [depthGuidance, setDepthGuidance] = useState('')
  const [depthEstimator, setDepthEstimator] = useState<'da3' | 'depth_anything_v2' | 'zoe' | 'midas'>('da3')
  const [depthResolution, setDepthResolution] = useState(504)
  const [depthInvert, setDepthInvert] = useState(false)
  const [depthPreview, setDepthPreview] = useState('')
  const [depthPreviewing, setDepthPreviewing] = useState(false)
  const [styleMethod, setStyleMethod] = useState<'AdaIN' | 'WCT' | 'WCT2' | 'scattersort'>('AdaIN')
  const [styleWeight, setStyleWeight] = useState(0.8)
  const [useEnhancer, setUseEnhancer] = useState(false)
  const [enhancerStrength, setEnhancerStrength] = useState(1.0)

  const task = tasks.find(item => item.id === taskId) ?? initialTask
  const preset = useMemo(
    () => presetFor(presetTaskFor(task, extendMode, preserveMode), qualityMode),
    [task, extendMode, preserveMode, qualityMode],
  )
  const styleLoras = useMemo(() => loras.filter(lora => lora.is_official), [loras])
  const selectedLoraInfo = useMemo(
    () => styleLoras.find(lora => lora.name === selectedStyleLora) ?? null,
    [styleLoras, selectedStyleLora],
  )
  const taskLoraInfo = useMemo(
    () => task.id === 'nk2e_edit'
      ? (loras.find(lora => lora.name === 'nk2e_v01') ?? null)
      : selectedLoraInfo,
    [loras, selectedLoraInfo, task.id],
  )
  const activeImages = useMemo(() => slots.filter(slot => slot.image), [slots])
  const promptPreview = useMemo(
    () => buildRolePrompt(slots, instruction, task, preset.promptHint),
    [slots, instruction, task, preset.promptHint],
  )
  const sourceImage = slots[0]?.image ?? ''
  const extraSlots = slots.slice(1)
  const attachedRefs = extraSlots.filter(s => s.image).length
  const latestResult = results?.[0] ?? ''
  const activePipelineMode: PipelineKind = task.id === 'extend'
    ? (extendMode === 'preserve' ? 'outpaint' : 'redraw')
    : task.id === 'preserve'
      ? (preserveMode === 'masked' ? 'inpaint' : 'img2img')
      : task.pipeline
  // Enforce the community recipe engine (Turbo Int8) unless the user has
  // deliberately selected GGUF, which we respect.
  const activeEngine = params.diffusion_engine === 'native_gguf' ? 'native_gguf' : preset.diffusionEngine
  const activeQuantization = activeEngine === 'native_gguf' ? 'gguf' : preset.quantization
  const usesDenoise = task.id === 'preserve' || (task.id === 'extend' && extendMode === 'preserve')
  const showStyleLora = task.id === 'style' || task.id === 'moodboard'

  // Engine / sampler / steps / CFG come from the Quick recipe the user picked
  // (Turbo/RAW buttons above) — Redraw only sets its task-specific params so it
  // doesn't override the chosen recipe.
  useEffect(() => {
    setParams({
      mode: activePipelineMode,
      denoise: usesDenoise ? denoise : preset.denoise,
      edit_provider: preset.editProvider,
      style_fusion_mode: activePipelineMode === 'redraw' ? 'semantic_fusion' : 'preserve_structure',
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePipelineMode, denoise, usesDenoise, preset])

  useEffect(() => {
    apiFetch.loras().then(setLoras).catch(() => setLoras([]))
  }, [])

  // Paste-anywhere sets the source image (ignored while typing in a field).
  useEffect(() => {
    const onPaste = async (e: globalThis.ClipboardEvent) => {
      const el = document.activeElement
      const tag = el?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (el as HTMLElement)?.isContentEditable) return
      const item = Array.from(e.clipboardData?.items ?? []).find(i => i.type.startsWith('image/'))
      const file = item?.getAsFile()
      if (file) {
        e.preventDefault()
        const b64 = await readFileB64(file)
        setSlots(cur => cur.map((s, i) => (i === 0 ? { ...s, image: b64 } : s)))
        setReadyMessage(null)
      }
    }
    window.addEventListener('paste', onPaste as unknown as EventListener)
    return () => window.removeEventListener('paste', onPaste as unknown as EventListener)
  }, [])

  const selectTask = (nextTaskId: StudioTaskId) => {
    const nextTask = tasks.find(item => item.id === nextTaskId) ?? initialTask
    const carryImage = slots[0]?.image || params.init_image_b64
    setTaskId(nextTaskId)
    setExtendMode('redraw')
    setPreserveMode('whole')
    setDenoise(presetFor(presetTaskFor(nextTask, 'redraw', 'whole'), qualityMode).denoise)
    setSlots(slotsForTask(nextTask, carryImage))
    setInstruction('')
    setReadyMessage(null)
    setShowRefs(['insert', 'style', 'sketch', 'moodboard'].includes(nextTaskId))
    setUseActualDenoise(false)
    setUseIncontext(false)
    setVisionPosition('before')
    setVisionDetail(1.0)
    setIncontextEncoder('krea2')
    setIncontextSystemPrompt('')
    setUseStyleTransfer(false)
    setStyleMethod('AdaIN')
    setStyleWeight(0.8)
  }

  // Edit-engine params injected into every prepare (no-op values when disabled).
  const editEngineParams = () => ({
    actual_denoise: useActualDenoise,
    incontext_edit: useIncontext,
    incontext_image_b64: useIncontext ? sourceImage : '',
    incontext_vision_position: visionPosition,
    incontext_vision_megapixels: visionDetail,
    incontext_encoder: incontextEncoder,
    incontext_system_prompt: incontextSystemPrompt,
  })

  // RES4LYF style-transfer params (active only on the Style Transfer tab).
  const styleActive = task.id === 'style' && useStyleTransfer && !!sourceImage
  const styleTransferParams = () => ({
    style_transfer_image_b64: styleActive ? sourceImage : '',
    style_transfer_method: styleMethod,
    style_transfer_weight: styleWeight,
  })

  // Krea 2 Enhancer (model patch) — applies to every task.
  const enhancerParams = () => ({
    krea_enhancer_enabled: useEnhancer,
    krea_enhancer_variant: (useEnhancer ? 'current' : 'off') as 'current' | 'off',
    krea_enhancer_strength: enhancerStrength,
  })

  const updateSlot = (id: string, patch: Partial<ReferenceSlot>) => {
    setSlots(current => current.map(slot => slot.id === id ? { ...slot, ...patch } : slot))
    setReadyMessage(null)
  }

  const matchOutputSize = async () => {
    if (!sourceImage) return
    const { width, height } = await imageSize(sourceImage)
    setParams({ width: align16(width), height: align16(height) })
    setReadyMessage(`Output size set to ${align16(width)}×${align16(height)} to match your image.`)
  }

  // Local (abliterated Qwen3-VL) image -> prompt. mode 'style' extracts only the
  // art direction so it can be transferred to a new subject; 'recreate' describes
  // the whole image. Appends into the instruction box.
  const describeSource = async (m: 'recreate' | 'style', guidance = '') => {
    if (!sourceImage || describing) return
    setDescribing(m)
    try {
      const r = await apiFetch.describeImage(sourceImage, m, guidance.trim())
      setInstruction(prev => (prev.trim() ? `${prev.trim()}\n${r.prompt}` : r.prompt))
      setReadyMessage(null)
    } catch {
      setReadyMessage('Local description failed — check that the abliterated Qwen3-VL is set up (Xperiment / Local AI Assets).')
    } finally {
      setDescribing('')
    }
  }

  const runDepthPreview = async () => {
    if (!sourceImage || depthPreviewing) return
    setDepthPreviewing(true)
    try {
      const r = await apiFetch.depthPreview(sourceImage, depthEstimator, depthResolution, depthInvert)
      setDepthPreview(r.image_b64)
      setReadyMessage(null)
    } catch {
      setReadyMessage('Depth preview failed — the estimator model may still be downloading on first use. Try again in a moment.')
    } finally {
      setDepthPreviewing(false)
    }
  }

  const scrollToGenerate = () => {
    window.setTimeout(() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }), 60)
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
      moodboard_images: styleActive ? [] : activeImages.map(slot => slot.image),
      moodboard_strength: preset.moodboardStrength,
      style_fusion_mode: 'semantic_fusion',
      denoise: preset.denoise,
      edit_provider: preset.editProvider,
      use_prompt_expander: preset.usePromptExpander,
      loras: loraToGeneration(taskLoraInfo),
      width: dimensions?.width ?? params.width,
      height: dimensions?.height ?? params.height,
      use_rebalance: true,
      rebalance_multiplier: params.rebalance_multiplier || 1,
      ...editEngineParams(),
      ...styleTransferParams(),
      ...enhancerParams(),
    })
    setReadyMessage(styleActive
      ? 'Style transfer ready — your source image drives the style. Scroll down and Generate.'
      : 'Ready. Scroll down and press Generate.')
    scrollToGenerate()
  }

  const preparePreserve = () => {
    if (!sourceImage) return
    setParams({
      mode: preserveMode === 'masked' ? 'inpaint' : 'img2img',
      prompt: promptPreview,
      init_image_b64: sourceImage,
      mask_b64: preserveMode === 'masked' ? params.mask_b64 : '',
      moodboard_images: activeImages.slice(1).map(slot => slot.image),
      moodboard_strength: preset.moodboardStrength,
      style_fusion_mode: 'preserve_structure',
      denoise,
      edit_provider: preset.editProvider,
      use_prompt_expander: preset.usePromptExpander,
      loras: loraToGeneration(taskLoraInfo),
      use_rebalance: true,
      ...editEngineParams(),
      ...enhancerParams(),
      style_transfer_image_b64: '',
    })
    setReadyMessage(preserveMode === 'masked'
      ? 'Ready. Paint/adjust the mask, then scroll down and press Generate.'
      : 'Ready. Scroll down and press Generate.')
    scrollToGenerate()
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
      moodboard_strength: preset.moodboardStrength,
      style_fusion_mode: 'preserve_structure',
      width: result.width,
      height: result.height,
      denoise: preset.denoise,
      edit_provider: preset.editProvider,
      use_prompt_expander: preset.usePromptExpander,
      loras: loraToGeneration(taskLoraInfo),
      use_rebalance: true,
      ...enhancerParams(),
      style_transfer_image_b64: '',
    })
    setReadyMessage('Ready. Scroll down and press Generate.')
    scrollToGenerate()
  }

  const prepareDepthControl = () => {
    if (!sourceImage) return
    const turbo = depthEngine === 'turbo'
    setParams({
      mode: 'redraw',
      prompt: promptPreview,
      init_image_b64: sourceImage,
      moodboard_images: [],
      ref_image1_b64: '', ref_image2_b64: '', ref_image3_b64: '',
      depth_control: true,
      depth_control_strength: depthStrength,
      depth_estimator: depthEstimator,
      depth_resolution: depthResolution,
      depth_invert: depthInvert,
      // Reference workflow defaults: er_sde/beta @8, RAW CFG 3 / Turbo CFG 1.
      checkpoint: turbo ? 'turbo' : 'raw',
      model_profile: turbo ? 'krea_turbo' : 'krea_raw',
      diffusion_engine: 'native_pytorch',
      quantization: turbo ? 'fp8' : 'fp8',
      sampler: 'er_sde',
      scheduler: turbo ? 'beta57' : 'beta',
      steps: 8,
      cfg: turbo ? 1.0 : 3.0,
      mu: turbo ? 1.15 : null,
      cfg_zero_star: false,
      seed_variance_preset: 'off',
      style_transfer_image_b64: '',
      loras: [],
    })
    setReadyMessage('Depth control ready. Write your prompt above, then scroll down and Generate.')
    scrollToGenerate()
  }

  const prepare = async () => {
    if (!activeImages.length) return
    if (task.id === 'depth') return prepareDepthControl()
    if (task.id === 'preserve') return preparePreserve()
    if (task.id === 'extend') return prepareExtend()
    prepareReferenceRedraw()
  }

  const needsMask = task.id === 'preserve' && preserveMode === 'masked' && !!sourceImage
  const strengthPresetId = STRENGTH_PRESETS.find(p => Math.abs(p.value - denoise) < 0.001)?.id ?? 'custom'

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2 }, maxWidth: 1000, mx: 'auto' }}>
      <Stack spacing={2}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 700 }}>Redraw Studio</Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Pick a workflow, add your image, then Generate.
          </Typography>
        </Box>

        {/* Workflow sub-tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={taskId}
            onChange={(_, v) => selectTask(v as StudioTaskId)}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
            sx={{ minHeight: 44, '& .MuiTab-root': { minHeight: 44, textTransform: 'none', fontWeight: 600 } }}
          >
            {tasks.map(item => {
              const Icon = item.icon
              return (
                <Tab
                  key={item.id}
                  value={item.id}
                  icon={<Icon sx={{ fontSize: 18 }} />}
                  iconPosition="start"
                  label={item.title}
                />
              )
            })}
          </Tabs>
        </Box>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mt: -0.5 }}>
          <Typography variant="body2" sx={{ color: 'text.secondary', flex: 1, minWidth: 200 }}>{task.description}</Typography>
          <Chip size="small" variant="outlined"
            color={activePipelineMode === 'redraw' ? 'primary' : 'secondary'}
            label={`Workflow: ${activePipelineMode.replace('_', ' ')}`} />
        </Stack>

        {/* Step 1 — Source */}
        <Box>
          <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 700 }}>1 · Your image</Typography>
          {sourceImage && latestResult ? (
            <Stack spacing={1}>
              <BeforeAfter before={sourceImage} after={latestResult} />
              <Stack direction="row" spacing={1}>
                <Button size="small" startIcon={<SwapHorizIcon />} onClick={() => updateSlot(slots[0].id, { image: latestResult })}>
                  Use result as source
                </Button>
                <Button size="small" color="inherit" startIcon={<CloseIcon />} onClick={() => updateSlot(slots[0].id, { image: '' })}>
                  Clear
                </Button>
              </Stack>
            </Stack>
          ) : (
            <SourceDropzone
              image={sourceImage}
              roleLabel={slots[0]?.label ?? 'Source'}
              onImage={image => updateSlot(slots[0].id, { image })}
              onClear={() => updateSlot(slots[0].id, { image: '' })}
            />
          )}
          {sourceImage && (
            <Stack direction="row" spacing={1} sx={{ mt: 1 }} flexWrap="wrap" useFlexGap>
              <Button size="small" variant="outlined" startIcon={<AspectRatioIcon />} onClick={matchOutputSize}>
                Match output size
              </Button>
              <Tooltip title="Caption this image into the prompt using the local abliterated Qwen3-VL (no ChatGPT)." arrow>
                <Button size="small" variant="outlined" onClick={() => describeSource('recreate')} disabled={!!describing}
                  startIcon={describing === 'recreate' ? <CircularProgress size={14} /> : <AutoAwesomeIcon />}>
                  Describe
                </Button>
              </Tooltip>
              <Tooltip title="Extract only the style/art-direction of this image into the prompt (local VL) — to transfer its look to a new subject." arrow>
                <Button size="small" variant="outlined" onClick={() => describeSource('style')} disabled={!!describing}
                  startIcon={describing === 'style' ? <CircularProgress size={14} /> : <StyleIcon />}>
                  Extract style
                </Button>
              </Tooltip>
            </Stack>
          )}
        </Box>

        {/* Task-specific mode toggles */}
        {(task.id === 'extend' || task.id === 'preserve') && (
          <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
            {task.id === 'extend' && (
              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>Extend strategy</Typography>
                <ToggleButtonGroup value={extendMode} exclusive size="small" color="primary"
                  onChange={(_, v) => v && setExtendMode(v)}>
                  <ToggleButton value="redraw">Creative redraw</ToggleButton>
                  <ToggleButton value="preserve">Preserve source</ToggleButton>
                </ToggleButtonGroup>
              </Box>
            )}
            {task.id === 'preserve' && (
              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>Edit mode</Typography>
                <ToggleButtonGroup value={preserveMode} exclusive size="small" color="primary"
                  onChange={(_, v) => v && setPreserveMode(v)}>
                  <ToggleButton value="whole">Whole image</ToggleButton>
                  <ToggleButton value="masked">Masked area</ToggleButton>
                </ToggleButtonGroup>
              </Box>
            )}
          </Stack>
        )}

        {/* Mask */}
        {needsMask && (
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>Paint the area to edit</Typography>
            <MaskCanvas imageB64={sourceImage} onMaskChange={mask => setParams({ mask_b64: mask })} />
          </Box>
        )}

        {/* Step 3 — Instruction + strength */}
        <Box>
          <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 700 }}>2 · Describe & tune</Typography>
          <Stack spacing={1.25}>
            <TextField
              label="What should Krea create? (optional)"
              value={instruction}
              onChange={e => { setInstruction(e.target.value); setReadyMessage(null) }}
              placeholder="e.g. cinematic lighting, keep her face, change the background to Niagara Falls"
              multiline minRows={2} fullWidth
            />

            {usesDenoise && (
              <Box>
                <Stack direction="row" justifyContent="space-between" alignItems="baseline">
                  <Typography variant="body2">Edit strength</Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'Roboto Mono', color: 'text.secondary' }}>denoise {denoise.toFixed(2)}</Typography>
                </Stack>
                <ToggleButtonGroup exclusive size="small" value={strengthPresetId} sx={{ flexWrap: 'wrap', mt: 0.5 }}
                  onChange={(_, v) => { if (v) { const p = STRENGTH_PRESETS.find(x => x.id === v); if (p) { setDenoise(p.value); setReadyMessage(null) } } }}>
                  {STRENGTH_PRESETS.map(p => (
                    <ToggleButton key={p.id} value={p.id} sx={{ textTransform: 'none', px: 1.25 }}>
                      <Tooltip title={p.hint} arrow><span>{p.label}</span></Tooltip>
                    </ToggleButton>
                  ))}
                </ToggleButtonGroup>
                <Slider value={denoise} min={0.05} max={1} step={0.01} size="small" sx={{ mt: 0.5 }}
                  onChange={(_, v) => { setDenoise(v as number); setReadyMessage(null) }} />
              </Box>
            )}

            {task.id === 'extend' && extendMode === 'preserve' && (
              <Box>
                <Stack direction="row" justifyContent="space-between">
                  <Typography variant="body2">Blend overlap</Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'Roboto Mono', color: 'text.secondary' }}>{outpaintOverlap}px</Typography>
                </Stack>
                <Slider value={outpaintOverlap} min={0} max={192} step={8} size="small"
                  onChange={(_, v) => setOutpaintOverlap(v as number)} />
              </Box>
            )}

            {/* Depth ControlNet engine — depth task only */}
            {task.id === 'depth' && (
              <Box sx={{ p: 1.25, borderRadius: 2, border: '1px solid', borderColor: 'divider', bgcolor: 'rgba(255,255,255,0.02)' }}>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                  Depth control engine
                </Typography>
                <Stack spacing={1} sx={{ mt: 0.75 }}>
                  {!sourceImage && (
                    <Typography variant="caption" sx={{ color: 'warning.main' }}>Add a depth source image above.</Typography>
                  )}
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 56 }}>Base</Typography>
                    <ToggleButtonGroup exclusive size="small" value={depthEngine} onChange={(_, v) => v && setDepthEngine(v)}>
                      <ToggleButton value="raw" sx={{ textTransform: 'none', px: 1.25 }}>RAW (default)</ToggleButton>
                      <ToggleButton value="turbo" sx={{ textTransform: 'none', px: 1.25 }}>Turbo (faster)</ToggleButton>
                    </ToggleButtonGroup>
                  </Stack>
                  {/* Adherence presets -> strength */}
                  <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                    <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 56 }}>Adherence</Typography>
                    <ToggleButtonGroup exclusive size="small"
                      value={depthStrength <= 0.95 ? 0.8 : depthStrength >= 1.35 ? 1.5 : 1.2}
                      onChange={(_, v) => { if (v != null) { setDepthStrength(v as number); setReadyMessage(null) } }}>
                      <ToggleButton value={0.8} sx={{ textTransform: 'none', px: 1.25 }}>Loose</ToggleButton>
                      <ToggleButton value={1.2} sx={{ textTransform: 'none', px: 1.25 }}>Balanced</ToggleButton>
                      <ToggleButton value={1.5} sx={{ textTransform: 'none', px: 1.25 }}>Strict</ToggleButton>
                    </ToggleButtonGroup>
                  </Stack>
                  <Box>
                    <Stack direction="row" justifyContent="space-between">
                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>ControlNet strength (fine)</Typography>
                      <Typography variant="caption" sx={{ fontFamily: 'Roboto Mono', color: 'text.secondary' }}>{depthStrength.toFixed(2)}</Typography>
                    </Stack>
                    <Slider value={depthStrength} min={0} max={1.6} step={0.05} size="small"
                      onChange={(_, v) => { setDepthStrength(v as number); setReadyMessage(null) }}
                      marks={[{ value: 0.8, label: 'loose' }, { value: 1.2, label: 'balanced' }, { value: 1.5, label: 'strict' }]}
                      sx={{ '& .MuiSlider-markLabel': { fontSize: 10 } }} />
                  </Box>
                  {/* Depth estimator */}
                  <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                    <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 56 }}>Estimator</Typography>
                    <ToggleButtonGroup exclusive size="small" value={depthEstimator}
                      onChange={(_, v) => { if (v) { setDepthEstimator(v); setDepthPreview('') } }}>
                      <ToggleButton value="da3" sx={{ textTransform: 'none', px: 1 }}>DA3</ToggleButton>
                      <ToggleButton value="depth_anything_v2" sx={{ textTransform: 'none', px: 1 }}>DAv2</ToggleButton>
                      <ToggleButton value="zoe" sx={{ textTransform: 'none', px: 1 }}>Zoe</ToggleButton>
                      <ToggleButton value="midas" sx={{ textTransform: 'none', px: 1 }}>MiDaS</ToggleButton>
                    </ToggleButtonGroup>
                  </Stack>
                  {/* Detail (resolution) + invert */}
                  <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                    <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 56 }}>Detail</Typography>
                    <ToggleButtonGroup exclusive size="small" value={depthResolution}
                      onChange={(_, v) => { if (v) { setDepthResolution(v as number); setDepthPreview('') } }}>
                      <ToggleButton value={504} sx={{ textTransform: 'none', px: 1 }}>Low</ToggleButton>
                      <ToggleButton value={700} sx={{ textTransform: 'none', px: 1 }}>Med</ToggleButton>
                      <ToggleButton value={1036} sx={{ textTransform: 'none', px: 1 }}>High</ToggleButton>
                    </ToggleButtonGroup>
                    <Stack direction="row" spacing={0.5} alignItems="center">
                      <Switch size="small" checked={depthInvert} onChange={e => { setDepthInvert(e.target.checked); setDepthPreview('') }} />
                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>Invert</Typography>
                    </Stack>
                  </Stack>
                  {/* Depth-map preview */}
                  <Box>
                    <Button size="small" variant="outlined" onClick={runDepthPreview}
                      disabled={!sourceImage || depthPreviewing}
                      startIcon={depthPreviewing ? <CircularProgress size={14} /> : <CompareIcon fontSize="small" />}
                      sx={{ textTransform: 'none' }}>
                      {depthPreviewing ? 'Reading depth…' : 'Preview depth map'}
                    </Button>
                    {depthPreview && (
                      <Box component="img" src={depthPreview} alt="depth map"
                        sx={{ display: 'block', mt: 1, width: '100%', maxWidth: 320, borderRadius: 1, border: '1px solid', borderColor: 'divider' }} />
                    )}
                  </Box>
                  <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                    The estimator reads the source's depth; your prompt paints a new image that follows that 3D layout. Preview to verify the map (white = near; flip Invert if near looks dark). RAW base tracks tighter than Turbo.
                  </Typography>

                  {/* Image -> prompt: describe the depth source (optionally guided) into the prompt. */}
                  <Box sx={{ pt: 0.5, borderTop: '1px solid', borderColor: 'divider' }}>
                    <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700 }}>
                      Create prompt from image
                    </Typography>
                    <TextField
                      size="small"
                      fullWidth
                      multiline
                      maxRows={3}
                      value={depthGuidance}
                      onChange={e => setDepthGuidance(e.target.value)}
                      disabled={!sourceImage || !!describing}
                      placeholder="Optional: what to focus on or change (blank = full auto prompt)"
                      sx={{ mt: 0.5 }}
                    />
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => describeSource('recreate', depthGuidance)}
                      disabled={!sourceImage || !!describing}
                      startIcon={describing === 'recreate' ? <CircularProgress size={14} /> : <AutoAwesomeIcon fontSize="small" />}
                      sx={{ mt: 0.75, textTransform: 'none' }}
                    >
                      {describing === 'recreate' ? 'Reading image…' : 'Create prompt from image'}
                    </Button>
                    <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: 0.5 }}>
                      Reads the depth source with the local Qwen3-VL and writes a prompt into the instruction box below.
                    </Typography>
                  </Box>
                </Stack>
              </Box>
            )}

            {/* Advanced features — edit engine, style transfer, quality/LoRA, enhancer, refs, preview */}
            <Box>
              <Button size="small" onClick={() => setShowAdvanced(v => !v)}
                endIcon={<ExpandMoreIcon sx={{ transform: showAdvanced ? 'rotate(180deg)' : 'none', transition: 'transform .15s' }} />}
                sx={{ textTransform: 'none', color: 'text.secondary', fontWeight: 700 }}>
                Advanced features
              </Button>
              <Collapse in={showAdvanced}>
                <Stack spacing={1.25} sx={{ pt: 0.5 }}>

            {/* Edit engine (Actual-Denoise + in-context vision) — edit tasks only */}
            {(task.id === 'preserve' || task.id === 'nk2e_edit') && (
              <Box sx={{ p: 1.25, borderRadius: 2, border: '1px solid', borderColor: 'divider', bgcolor: 'rgba(255,255,255,0.02)' }}>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                  Edit engine
                </Typography>
                <Stack spacing={0.25} sx={{ mt: 0.5 }}>
                  <FormControlLabel
                    control={<Switch size="small" checked={useActualDenoise} onChange={e => { setUseActualDenoise(e.target.checked); setReadyMessage(null) }} />}
                    label={
                      <Tooltip title="Injects the true noise amount so a given edit-strength looks the same on every scheduler (mozhaa/Actual-Denoise). Recommended for consistent img2img." arrow>
                        <Typography variant="body2">Consistent denoise <Typography component="span" variant="caption" sx={{ color: 'text.disabled' }}>· scheduler-independent</Typography></Typography>
                      </Tooltip>
                    }
                  />
                  <FormControlLabel
                    control={<Switch size="small" checked={useIncontext} onChange={e => { setUseIncontext(e.target.checked); setReadyMessage(null) }} />}
                    label={
                      <Tooltip title="Feeds your source image straight into Krea's Qwen3-VL vision (the community 'Qwen-edit text-encode for Krea 2'), so the instruction prompt edits it in-context. Strongest for instruction edits; combine with a low edit strength." arrow>
                        <Typography variant="body2">In-context vision edit <Typography component="span" variant="caption" sx={{ color: 'text.disabled' }}>· Qwen-edit style</Typography></Typography>
                      </Tooltip>
                    }
                  />
                  <Collapse in={useIncontext}>
                    <Stack spacing={1} sx={{ pl: 4, pt: 0.5 }}>
                      {!sourceImage && (
                        <Typography variant="caption" sx={{ color: 'warning.main' }}>Add a source image above to use in-context vision.</Typography>
                      )}
                      <FormControl size="small" sx={{ maxWidth: 320 }}>
                        <InputLabel>Vision encoder (local)</InputLabel>
                        <Select label="Vision encoder (local)" value={incontextEncoder}
                          onChange={e => { setIncontextEncoder(e.target.value as 'krea2' | 'qwen_edit_plus'); setReadyMessage(null) }}>
                          <MenuItem value="krea2">Krea 2 VL + system prompt</MenuItem>
                          <MenuItem value="qwen_edit_plus">Qwen-Edit-Plus (stronger)</MenuItem>
                        </Select>
                      </FormControl>
                      {incontextEncoder === 'krea2' ? (
                        <>
                          <TextField
                            label="Style-extract system prompt (local abliterated VL)"
                            value={incontextSystemPrompt}
                            onChange={e => { setIncontextSystemPrompt(e.target.value); setReadyMessage(null) }}
                            placeholder="Describe the reference's color/texture/subjects/lighting, then apply the instruction while staying consistent with the reference…"
                            multiline minRows={2} fullWidth
                            helperText="Left empty uses the built-in style-extract instruction. No ChatGPT — runs on the loaded Qwen3-VL."
                          />
                          <Stack direction="row" spacing={1} alignItems="center">
                            <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 96 }}>Image position</Typography>
                            <ToggleButtonGroup exclusive size="small" value={visionPosition}
                              onChange={(_, v) => v && setVisionPosition(v)}>
                              <ToggleButton value="before" sx={{ textTransform: 'none', px: 1.25 }}>Before prompt</ToggleButton>
                              <ToggleButton value="after" sx={{ textTransform: 'none', px: 1.25 }}>After prompt</ToggleButton>
                            </ToggleButtonGroup>
                          </Stack>
                          <Box>
                            <Stack direction="row" justifyContent="space-between">
                              <Typography variant="caption" sx={{ color: 'text.secondary' }}>Vision detail</Typography>
                              <Typography variant="caption" sx={{ fontFamily: 'Roboto Mono', color: 'text.secondary' }}>{visionDetail.toFixed(2)} MP</Typography>
                            </Stack>
                            <Slider value={visionDetail} min={0.25} max={2} step={0.05} size="small"
                              onChange={(_, v) => setVisionDetail(v as number)}
                              marks={[{ value: 0.5, label: 'fast' }, { value: 1, label: 'default' }, { value: 2, label: 'sharp' }]}
                              sx={{ '& .MuiSlider-markLabel': { fontSize: 10 } }} />
                          </Box>
                        </>
                      ) : (
                        <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                          Qwen-Edit-Plus encodes the source as a stronger multi-image edit reference. Best for "add/replace this" edits; ignores the system prompt and vision-detail controls.
                        </Typography>
                      )}
                    </Stack>
                  </Collapse>
                </Stack>
              </Box>
            )}

            {/* RES4LYF style transfer — Style Transfer tab only */}
            {task.id === 'style' && (
              <Box sx={{ p: 1.25, borderRadius: 2, border: '1px solid', borderColor: 'divider', bgcolor: 'rgba(255,255,255,0.02)' }}>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                  Style transfer engine
                </Typography>
                <Stack spacing={0.5} sx={{ mt: 0.5 }}>
                  <FormControlLabel
                    control={<Switch size="small" checked={useStyleTransfer} onChange={e => { setUseStyleTransfer(e.target.checked); setReadyMessage(null) }} />}
                    label={
                      <Tooltip title="Training-free style transfer (RES4LYF ClownGuide). Injects your source image's color/texture statistics into the render. No model download. Your SOURCE = the style; the prompt = the new subject." arrow>
                        <Typography variant="body2">RES4LYF style transfer <Typography component="span" variant="caption" sx={{ color: 'text.disabled' }}>· source image = style</Typography></Typography>
                      </Tooltip>
                    }
                  />
                  <Collapse in={useStyleTransfer}>
                    <Stack spacing={1} sx={{ pl: 4, pt: 0.5 }}>
                      {!sourceImage && (
                        <Typography variant="caption" sx={{ color: 'warning.main' }}>Load the style image as the source above, then describe your new subject in the prompt.</Typography>
                      )}
                      <FormControl size="small" sx={{ maxWidth: 260 }}>
                        <InputLabel>Method</InputLabel>
                        <Select label="Method" value={styleMethod}
                          onChange={e => { setStyleMethod(e.target.value as typeof styleMethod); setReadyMessage(null) }}>
                          <MenuItem value="AdaIN">AdaIN (fast, balanced)</MenuItem>
                          <MenuItem value="WCT">WCT (stronger color/texture)</MenuItem>
                          <MenuItem value="WCT2">WCT2 (refined)</MenuItem>
                          <MenuItem value="scattersort">Scattersort (experimental)</MenuItem>
                        </Select>
                      </FormControl>
                      <Box>
                        <Stack direction="row" justifyContent="space-between">
                          <Typography variant="caption" sx={{ color: 'text.secondary' }}>Style strength</Typography>
                          <Typography variant="caption" sx={{ fontFamily: 'Roboto Mono', color: 'text.secondary' }}>{styleWeight.toFixed(2)}</Typography>
                        </Stack>
                        <Slider value={styleWeight} min={0} max={2} step={0.05} size="small"
                          onChange={(_, v) => { setStyleWeight(v as number); setReadyMessage(null) }}
                          marks={[{ value: 0.5, label: 'subtle' }, { value: 0.8, label: 'default' }, { value: 1.5, label: 'heavy' }]}
                          sx={{ '& .MuiSlider-markLabel': { fontSize: 10 } }} />
                      </Box>
                      <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                        Renders your prompt subject with the source image's look via the ClownsharK sampler. Ignores reference slots.
                      </Typography>
                    </Stack>
                  </Collapse>
                </Stack>
              </Box>
            )}

            {/* Style LoRA (style/moodboard tasks) + active recipe summary. The
                sampler/steps/CFG come from the Quick recipe chosen above. */}
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ sm: 'center' }}>
              {showStyleLora && (
                <FormControl size="small" sx={{ minWidth: 220, flex: 1 }}>
                  <InputLabel>Krea style LoRA</InputLabel>
                  <Select label="Krea style LoRA" value={selectedStyleLora} onChange={e => setSelectedStyleLora(e.target.value)}>
                    <MenuItem value="">None / image references only</MenuItem>
                    {styleLoras.map(lora => (
                      <MenuItem key={lora.name} value={lora.name}>
                        {lora.display_name}{lora.installed ? '' : ' (download first)'}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}
              <Box sx={{ flex: 1, display: 'flex', alignItems: 'center' }}>
                <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                  {params.sampler}/{params.scheduler} · {params.steps} steps · CFG {params.cfg} · {preset.editProvider.replace('_', ' ')}
                </Typography>
              </Box>
            </Stack>

            {/* Krea 2 Enhancer (model patch, all tasks) */}
            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
              <FormControlLabel
                control={<Switch size="small" checked={useEnhancer} onChange={e => { setUseEnhancer(e.target.checked); setReadyMessage(null) }} />}
                label={
                  <Tooltip title="ComfyUI-Krea2T-Enhancer model patch: boosts prompt adherence and micro-detail. Off by default; 1.0 strength is the tuned value." arrow>
                    <Typography variant="body2">Krea 2 Enhancer</Typography>
                  </Tooltip>
                }
              />
              {useEnhancer && (
                <Box sx={{ flex: 1, minWidth: 180, maxWidth: 300 }}>
                  <Stack direction="row" justifyContent="space-between">
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>Strength</Typography>
                    <Typography variant="caption" sx={{ fontFamily: 'Roboto Mono', color: 'text.secondary' }}>{enhancerStrength.toFixed(2)}</Typography>
                  </Stack>
                  <Slider value={enhancerStrength} min={0} max={2} step={0.05} size="small"
                    onChange={(_, v) => { setEnhancerStrength(v as number); setReadyMessage(null) }} />
                </Box>
              )}
            </Stack>

            {/* Optional references */}
            <Box>
              <Button size="small" onClick={() => setShowRefs(v => !v)}
                endIcon={<ExpandMoreIcon sx={{ transform: showRefs ? 'rotate(180deg)' : 'none', transition: 'transform .15s' }} />}
                sx={{ textTransform: 'none', color: 'text.secondary' }}>
                Reference images{attachedRefs ? ` · ${attachedRefs} added` : ' (optional)'}
              </Button>
              <Collapse in={showRefs}>
                <Stack direction="row" spacing={1} sx={{ overflowX: 'auto', pb: 1, pt: 0.5 }}>
                  {extraSlots.map(slot => (
                    <CompactReference
                      key={slot.id}
                      slot={slot}
                      onImage={image => updateSlot(slot.id, { image })}
                      onRole={role => updateSlot(slot.id, { role })}
                      onNote={note => updateSlot(slot.id, { note })}
                      onClear={() => updateSlot(slot.id, { image: '' })}
                    />
                  ))}
                </Stack>
              </Collapse>
            </Box>

            {/* Prompt preview (advanced, collapsed) */}
            <Box>
              <Button size="small" onClick={() => setShowPreview(v => !v)}
                endIcon={<ExpandMoreIcon sx={{ transform: showPreview ? 'rotate(180deg)' : 'none', transition: 'transform .15s' }} />}
                sx={{ textTransform: 'none', color: 'text.secondary' }}>
                Prompt preview
              </Button>
              <Collapse in={showPreview}>
                <TextField value={promptPreview} multiline minRows={3} fullWidth InputProps={{ readOnly: true, sx: { fontSize: 12.5 } }}
                  helperText="The role-aware prompt sent to the Generate panel below." />
              </Collapse>
            </Box>
                </Stack>
              </Collapse>
            </Box>
          </Stack>
        </Box>

        {preset.warning && <Alert severity="warning" sx={{ py: 0.25 }}>{preset.warning}</Alert>}
        {selectedLoraInfo && !selectedLoraInfo.installed && (
          <Alert severity="info" sx={{ py: 0.25 }}>This LoRA isn’t installed yet — download it in the LoRA section, or Krea falls back to image references.</Alert>
        )}
        {readyMessage && <Alert severity="success" sx={{ py: 0.25 }}>{readyMessage}</Alert>}

        <Divider />

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems="center">
          <Button variant="contained" size="large" fullWidth onClick={prepare} disabled={!activeImages.length}>
            {activeImages.length ? `Prepare ${task.title} → Generate below` : 'Add an image to start'}
          </Button>
          <Button variant="text" size="large" onClick={() => selectTask(task.id)} sx={{ flexShrink: 0 }}>
            Reset
          </Button>
        </Stack>
      </Stack>
    </Box>
  )
}
