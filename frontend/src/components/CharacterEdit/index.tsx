import { useEffect, useRef, useState } from 'react'
import { Accordion, AccordionDetails, AccordionSummary, Alert, Autocomplete, Box, Button, Chip, CircularProgress, FormControlLabel, IconButton, MenuItem, Paper, Slider, Stack, Switch, TextField, Tooltip, Typography } from '@mui/material'
import FaceRetouchingNaturalIcon from '@mui/icons-material/FaceRetouchingNatural'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { apiFetch, publicUrl, type GenerationJob, type GenerationRequest, type MoodboardItem } from '../../api'
import { characterEditPresets, presetById, type CharacterEditTask } from './characterEditPresets'
import RegionCanvas, { type EditRegion } from './RegionCanvas'

function moodboardStyleText(item: MoodboardItem): string {
  const parts = [item.taste_profile?.trim() || '']
  const kw = (item.keywords || []).slice(0, 8).join(', ')
  if (kw) parts.push(kw)
  return parts.filter(Boolean).join('. ')
}

// Read an image the user copied to the clipboard (secure contexts / localhost).
export async function clipboardImageFile(): Promise<File | null> {
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

function fileFromClipboardEvent(event: ClipboardEvent): File | null {
  const items = event.clipboardData?.items
  if (!items) return null
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile()
      if (file) return file
    }
  }
  return null
}

function fileToBase64(file: File): Promise<{ b64: string; preview: string; width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const preview = String(reader.result || '')
      const img = new Image()
      img.onload = () => resolve({ b64: preview.split(',')[1] || preview, preview, width: img.naturalWidth, height: img.naturalHeight })
      img.onerror = () => reject(new Error('Could not read image dimensions.'))
      img.src = preview
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

function fit2MP(width: number, height: number) {
  const maxPixels = 2_000_000
  if (width * height <= maxPixels) return { width, height }
  const scale = Math.sqrt(maxPixels / (width * height))
  return {
    width: Math.max(16, Math.round((width * scale) / 16) * 16),
    height: Math.max(16, Math.round((height * scale) / 16) * 16),
  }
}

async function waitJob(jobId: string, onUpdate: (job: GenerationJob) => void): Promise<GenerationJob> {
  for (;;) {
    const job = await apiFetch.jobStatus(jobId)
    onUpdate(job)
    if (['done', 'error', 'blocked', 'cancelled'].includes(job.status)) return job
    await new Promise(resolve => window.setTimeout(resolve, 2000))
  }
}

// Character Edit stays <=2MP (model-card requirement), so a single ~1MP tier is
// enough. "Match source" keeps the source aspect ratio, which the model card says
// gives the best preservation for single-image edits.
const CE_ASPECTS: Array<{ id: string; label: string; dims: [number, number] | null }> = [
  { id: 'source', label: 'Match source', dims: null },
  { id: '1:1', label: '1:1', dims: [1024, 1024] },
  { id: '4:3', label: '4:3', dims: [1024, 768] },
  { id: '3:4', label: '3:4', dims: [768, 1024] },
  { id: '3:2', label: '3:2', dims: [1024, 688] },
  { id: '2:3', label: '2:3', dims: [688, 1024] },
  { id: '16:9', label: '16:9', dims: [1024, 576] },
  { id: '9:16', label: '9:16', dims: [576, 1024] },
]

function ceAspectGlyph(dims: [number, number] | null, maxSide = 24): { w: number; h: number } {
  if (!dims) return { w: maxSide, h: maxSide }
  const [a, b] = dims
  return {
    w: a >= b ? maxSide : Math.round(maxSide * (a / b)),
    h: b >= a ? maxSide : Math.round(maxSide * (b / a)),
  }
}

const CHECKPOINT_INFO: Record<'turbo' | 'raw', { title: string; blurb: string; recipe: string }> = {
  turbo: {
    title: 'Turbo',
    blurb: 'Fast path for most edits: add, recolor, restyle, re-stage, attribute & outfit changes, scene translation.',
    recipe: '8 steps, CFG 1.0 (~1 min at 2MP)',
  },
  raw: {
    title: 'Raw',
    blurb: 'Real-guidance path for removals & large deletions. Distilled Turbo at CFG 1 tends to re-render the subject instead of removing it.',
    recipe: '20 steps, CFG 3.0 (grounds the empty-prompt negative)',
  },
}

export default function CharacterEditPanel() {
  const [task, setTask] = useState<CharacterEditTask>('local_edit')
  const preset = presetById(task)
  const [source, setSource] = useState('')
  const [preview, setPreview] = useState('')
  const [prompt, setPrompt] = useState(preset.prompt)
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [aspect, setAspect] = useState('source')
  const [sourceDims, setSourceDims] = useState<[number, number]>([1024, 1024])
  const [refDims, setRefDims] = useState<[number, number] | null>(null)
  const [checkpoint, setCheckpoint] = useState<'turbo' | 'raw'>(preset.checkpoint)
  const [steps, setSteps] = useState(preset.steps)
  const [cfg, setCfg] = useState(preset.cfg)
  const [groundingPx, setGroundingPx] = useState(preset.groundingPx)
  const [loraStrength, setLoraStrength] = useState(1.05)
  const [allowHighRes, setAllowHighRes] = useState(false)
  const [running, setRunning] = useState(false)
  const [job, setJob] = useState<GenerationJob | null>(null)
  const [error, setError] = useState('')
  const fileInput = useRef<HTMLInputElement | null>(null)
  const refInput = useRef<HTMLInputElement | null>(null)
  const [subjectFeatures, setSubjectFeatures] = useState('')
  const [useSubjectLock, setUseSubjectLock] = useState(false)
  const [describing, setDescribing] = useState(false)
  // Two-reference background scene
  const [refSource, setRefSource] = useState('')
  const [refPreview, setRefPreview] = useState('')
  // Moodboard style
  const [moodboardStyle, setMoodboardStyle] = useState('')
  const [moodboardOptions, setMoodboardOptions] = useState<MoodboardItem[]>([])
  const [moodboardLoading, setMoodboardLoading] = useState(false)
  // Magic wand
  const [wandBusy, setWandBusy] = useState(false)
  // Regional placement boxes
  const [useRegions, setUseRegions] = useState(false)
  const [regions, setRegions] = useState<EditRegion[]>([])

  const seedTwoBoxes = () => {
    const now = Date.now()
    return [
      { id: `r_${now}_a`, x: 0.05, y: 0.08, w: 0.42, h: 0.86, prompt: 'Person A (left) — drop their face here', referenceB64: '', referencePreview: '' },
      { id: `r_${now}_b`, x: 0.53, y: 0.08, w: 0.42, h: 0.86, prompt: 'Person B (right) — drop their face here', referenceB64: '', referencePreview: '' },
    ] as EditRegion[]
  }

  const applyPreset = (next: CharacterEditTask) => {
    const p = presetById(next)
    setTask(next)
    setPrompt(p.prompt)
    setCheckpoint(p.checkpoint)
    setSteps(p.steps)
    setCfg(p.cfg)
    setGroundingPx(p.groundingPx)
    // The two-people preset is all about placement -> open the box canvas with A/B ready.
    if (next === 'two_reference') {
      setUseRegions(true)
      setRegions(current => (current.length ? current : seedTwoBoxes()))
      if (aspect === 'source') applyAspect('3:2')
    } else {
      // Single-subject workflow: placement boxes default off (collapsed accordion).
      setUseRegions(false)
      if (p.kind === 'inplace') clearRef()
    }
  }

  const describeSubject = async () => {
    if (!source) {
      setError('Upload a source image first.')
      return
    }
    setDescribing(true)
    setError('')
    try {
      const res = await apiFetch.describeImage(source, 'character')
      setSubjectFeatures((res.prompt || '').trim())
      setUseSubjectLock(true)
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? err?.message ?? 'Subject description failed.')
    } finally {
      setDescribing(false)
    }
  }

  const composedPrompt = () => {
    let text = prompt.trim()
    const style = moodboardStyle.trim()
    if (style) text += `\n\nStyle and mood: ${style}.`
    const features = subjectFeatures.trim()
    if (useSubjectLock && features) {
      text += `\n\nThe person must look exactly like this same individual: ${features}.`
    }
    return text
  }

  const upload = async (file?: File) => {
    if (!file) return
    const loaded = await fileToBase64(file)
    const dims = fit2MP(loaded.width, loaded.height)
    setSource(loaded.b64)
    setPreview(loaded.preview)
    setSourceDims([dims.width, dims.height])
    // Only the subject drives output size when matching source and no scene reference.
    if (aspect === 'source' && !refSource) {
      setWidth(dims.width)
      setHeight(dims.height)
    }
  }

  const uploadRef = async (file?: File) => {
    if (!file) return
    const loaded = await fileToBase64(file)
    const dims = fit2MP(loaded.width, loaded.height)
    setRefSource(loaded.b64)
    setRefPreview(loaded.preview)
    setRefDims([dims.width, dims.height])
    // With a scene reference, output AR should match the scene (the primary frame).
    if (aspect === 'source') {
      setWidth(dims.width)
      setHeight(dims.height)
    }
  }

  const clearRef = () => {
    setRefSource('')
    setRefPreview('')
    setRefDims(null)
    if (aspect === 'source') {
      setWidth(sourceDims[0])
      setHeight(sourceDims[1])
    }
  }

  const applyAspect = (id: string) => {
    setAspect(id)
    const chosen = CE_ASPECTS.find(a => a.id === id)
    if (!chosen || !chosen.dims) {
      const d = refDims ?? sourceDims
      setWidth(d[0])
      setHeight(d[1])
    } else {
      setWidth(chosen.dims[0])
      setHeight(chosen.dims[1])
    }
  }

  const expandScene = async () => {
    const idea = prompt.trim()
    if (!idea) {
      setError('Type a short scene idea first, then use the magic wand.')
      return
    }
    setWandBusy(true)
    setError('')
    try {
      const res = await apiFetch.expandPrompt(idea)
      if (res.error) setError(res.error)
      if (res.expanded) setPrompt(res.expanded.trim())
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? err?.message ?? 'Magic wand failed.')
    } finally {
      setWandBusy(false)
    }
  }

  const searchMoodboards = async (q: string) => {
    setMoodboardLoading(true)
    try {
      const res = await apiFetch.moodboards({ q, pageSize: 20 })
      setMoodboardOptions(res.items || [])
    } catch {
      setMoodboardOptions([])
    } finally {
      setMoodboardLoading(false)
    }
  }

  const useResultAsSource = () => {
    const result = job?.images?.[0] || ''
    if (!result) return
    const b64 = result.startsWith('data:') ? (result.split(',')[1] || result) : result
    const previewUrl = result.startsWith('data:') ? result : `data:image/png;base64,${result}`
    setSource(b64)
    setPreview(previewUrl)
    setSubjectFeatures('')
    setUseSubjectLock(false)
    clearRef()
    setJob(null)
  }

  const pasteSource = async () => {
    const file = await clipboardImageFile()
    if (file) upload(file)
    else setError('No image on the clipboard. Copy an image, then paste.')
  }

  const pasteScene = async () => {
    const file = await clipboardImageFile()
    if (file) uploadRef(file)
    else setError('No image on the clipboard. Copy an image, then paste.')
  }

  // Ctrl/Cmd+V anywhere outside a text field pastes into the source image.
  const latestUpload = useRef(upload)
  latestUpload.current = upload
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return
      const file = fileFromClipboardEvent(event)
      if (file) {
        event.preventDefault()
        latestUpload.current(file)
      }
    }
    document.addEventListener('paste', onPaste)
    return () => document.removeEventListener('paste', onPaste)
  }, [])

  const run = async () => {
    const twoPeople = task === 'two_reference'
    const regionFaces = regions.filter(r => r.referenceB64)
    // In two-people mode the faces live in the boxes; the primary subject is box A
    // (or the scene when supplied). Otherwise it's the single uploaded face.
    const primarySubject = source || regionFaces[0]?.referenceB64 || refSource || ''
    if (!primarySubject) {
      setError(twoPeople ? 'Add a face to each placement box first.' : 'Upload a face image first.')
      return
    }
    setRunning(true)
    setError('')
    setJob(null)
    try {
      const dims = allowHighRes ? { width, height } : fit2MP(width, height)
      const request: GenerationRequest = {
        prompt: composedPrompt(),
        negative_prompt: '',
        mode: 'character_edit',
        model_profile: checkpoint === 'raw' ? 'krea_raw' : 'krea_turbo',
        diffusion_engine: 'native_pytorch',
        checkpoint,
        quantization: 'fp8',
        width: dims.width,
        height: dims.height,
        num_images: 1,
        seed: -1,
        sampler: 'euler_flow',
        scheduler: checkpoint === 'raw' ? 'beta' : 'simple',
        steps,
        cfg,
        mu: checkpoint === 'raw' ? null : 1.15,
        cfg_zero_star: false,
        use_prompt_expander: false,
        use_prompt_planner: false,
        character_edit_source_b64: primarySubject,
        character_edit_reference_b64: refSource || undefined,
        character_edit_regions: useRegions
          ? regions
              .filter(r => r.prompt.trim() || r.referenceB64)
              .map(r => ({ x: r.x, y: r.y, w: r.w, h: r.h, prompt: r.prompt, reference_b64: r.referenceB64 || undefined }))
          : undefined,
        character_edit_grounding_px: groundingPx,
        character_edit_task: refSource ? 'two_reference' : task,
        character_edit_lora_strength: loraStrength,
      }
      const submitted = await apiFetch.generate(request)
      const finished = await waitJob(submitted.job_id, setJob)
      if (finished.status !== 'done') {
        setError(finished.error || `Character Edit ended with status ${finished.status}.`)
      }
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? err?.message ?? 'Character Edit failed.')
    } finally {
      setRunning(false)
    }
  }

  const output = job?.images?.[0] || ''
  const filename = (job?.metadata?.[0] as any)?.filename || ''
  const kind = preset.kind
  const twoPeople = kind === 'compose'
  const identity = kind === 'identity'
  const canRun = !!(source || refSource || regions.some(r => r.referenceB64))

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2 }, maxWidth: 1120, mx: 'auto' }}>
      <Stack spacing={2}>
        <Paper sx={{ p: 2 }}>
          <Stack spacing={1.5}>
            <Stack direction="row" spacing={1} alignItems="center">
              <FaceRetouchingNaturalIcon color="primary" />
              <Box>
                <Typography variant="h5">Character Edit</Typography>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  Identity-preserving Krea 2 instruction edits: conradlocke identity-edit LoRA + lbouaraba/comfyui-krea2edit dual conditioning (in-context VAE tokens + image-grounded Qwen3-VL). Locked to Turbo fp8 + Qwen VAE.
                </Typography>
              </Box>
            </Stack>
            <Alert severity="info" sx={{ py: 0 }}>
              Pick a task below — each one adapts what you upload. Output aspect ratio is matched to your source and kept at or below 2MP. Higher grounding keeps likeness stronger; lower grounding makes edits apply more aggressively.
            </Alert>
          </Stack>
        </Paper>

        <Paper sx={{ p: 2 }}>
          <Stack spacing={1.5}>
            <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
              {characterEditPresets.map(item => (
                <Chip
                  key={item.id}
                  label={item.label}
                  clickable
                  color={task === item.id ? 'primary' : 'default'}
                  variant={task === item.id ? 'filled' : 'outlined'}
                  onClick={() => applyPreset(item.id)}
                />
              ))}
            </Stack>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>{preset.notes}</Typography>

            <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
              <Stack spacing={1.25} sx={{ flex: 1 }}>
                {twoPeople ? (
                  <>
                    <Alert severity="info" icon={false} sx={{ py: 0.5 }}>
                      <Typography variant="caption" sx={{ fontWeight: 600, display: 'block' }}>Two people mode</Typography>
                      <Typography variant="caption">Add each person's face in their box (Box 1 = left, Box 2 = right). Drag/resize the boxes. Optionally add a scene image below. Faces are never used as the canvas background.</Typography>
                    </Alert>
                    <RegionCanvas
                      background={refPreview}
                      aspectW={width}
                      aspectH={height}
                      regions={regions}
                      onChange={setRegions}
                    />
                  </>
                ) : (
                  <>
                    <Stack direction="row" spacing={1}>
                      <Button variant="outlined" fullWidth onClick={() => fileInput.current?.click()}>
                        {preset.sourceLabel}
                      </Button>
                      <Button variant="outlined" onClick={pasteSource}>Paste</Button>
                    </Stack>
                    <input ref={fileInput} hidden type="file" accept="image/*" onChange={event => upload(event.target.files?.[0])} />
                    {preview ? (
                      <Box component="img" src={preview} alt="Source" sx={{ width: '100%', maxHeight: 420, objectFit: 'contain', borderRadius: 2, bgcolor: 'rgba(0,0,0,0.22)' }} />
                    ) : (
                      <Box sx={{ minHeight: 280, borderRadius: 2, border: '1px dashed', borderColor: 'divider', display: 'grid', placeItems: 'center', textAlign: 'center', px: 2 }}>
                        <Typography variant="body2" sx={{ color: 'text.disabled' }}>{preset.sourceHint}</Typography>
                      </Box>
                    )}
                    {identity ? (
                      <Alert severity="info" icon={false} sx={{ py: 0.25 }}>
                        <Typography variant="caption" sx={{ fontWeight: 600, display: 'block' }}>Use a close face photo.</Typography>
                        <Typography variant="caption">The model reads the whole image, so props, clothing, and background in your photo can bleed into the result. A head-and-shoulders crop gives the cleanest identity transfer.</Typography>
                      </Alert>
                    ) : (
                      <Alert severity="info" icon={false} sx={{ py: 0.25 }}>
                        <Typography variant="caption" sx={{ fontWeight: 600, display: 'block' }}>Edit an existing image.</Typography>
                        <Typography variant="caption">Upload the full photo you want to change. Your instruction is applied while the rest of the frame is preserved.</Typography>
                      </Alert>
                    )}
                  </>
                )}

                {identity && (
                  <>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Button size="small" variant="outlined" onClick={() => refInput.current?.click()}>
                        {refSource ? 'Change scene image' : 'Add scene image (optional)'}
                      </Button>
                      <Button size="small" variant="outlined" onClick={pasteScene}>Paste</Button>
                      {refSource && <Button size="small" color="inherit" onClick={clearRef}>Remove</Button>}
                      <input ref={refInput} hidden type="file" accept="image/*" onChange={event => uploadRef(event.target.files?.[0])} />
                    </Stack>
                    {refPreview && (
                      <Box component="img" src={refPreview} alt="Scene reference" sx={{ width: '100%', maxHeight: 220, objectFit: 'contain', borderRadius: 2, bgcolor: 'rgba(0,0,0,0.22)' }} />
                    )}
                    <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                      Optional: a scene/background image the person is placed into (it drives the output framing). Leave empty to restage via the prompt.
                    </Typography>
                  </>
                )}

                {twoPeople && (
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Button size="small" variant="outlined" onClick={() => refInput.current?.click()}>
                      {refSource ? 'Change scene image' : 'Add scene image (optional)'}
                    </Button>
                    <Button size="small" variant="outlined" onClick={pasteScene}>Paste</Button>
                    {refSource && <Button size="small" color="inherit" onClick={clearRef}>Remove</Button>}
                    <input ref={refInput} hidden type="file" accept="image/*" onChange={event => uploadRef(event.target.files?.[0])} />
                  </Stack>
                )}
              </Stack>

              <Stack spacing={1.25} sx={{ flex: 1 }}>
                <TextField
                  label="Edit instruction"
                  value={prompt}
                  onChange={event => setPrompt(event.target.value)}
                  multiline
                  minRows={5}
                  fullWidth
                  InputProps={{
                    endAdornment: (
                      <Tooltip title="Magic wand: expand this into a full scene instruction">
                        <span style={{ position: 'absolute', top: 6, right: 6 }}>
                          <IconButton size="small" onClick={expandScene} disabled={wandBusy || !prompt.trim()}>
                            {wandBusy ? <CircularProgress size={16} /> : <AutoAwesomeIcon fontSize="small" />}
                          </IconButton>
                        </span>
                      </Tooltip>
                    ),
                  }}
                />

                <Autocomplete
                  size="small"
                  options={moodboardOptions}
                  loading={moodboardLoading}
                  getOptionLabel={option => option.title || option.slug || `#${option.id}`}
                  onInputChange={(_, value, reason) => { if (reason === 'input' && value.trim()) searchMoodboards(value.trim()) }}
                  onOpen={() => { if (!moodboardOptions.length) searchMoodboards('') }}
                  onChange={(_, value) => setMoodboardStyle(value ? moodboardStyleText(value) : '')}
                  renderInput={params => <TextField {...params} label="Moodboard style (optional)" placeholder="Search moodboards..." />}
                />
                {moodboardStyle && (
                  <TextField
                    label="Applied style & mood"
                    value={moodboardStyle}
                    onChange={event => setMoodboardStyle(event.target.value)}
                    onBlur={() => { /* keep edits */ }}
                    multiline
                    minRows={2}
                    fullWidth
                    size="small"
                    helperText="Appended to your instruction. Clear to drop the moodboard style."
                  />
                )}

                <Paper variant="outlined" sx={{ p: 1.25, bgcolor: 'rgba(255,255,255,0.02)' }}>
                  <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                    <FormControlLabel
                      control={<Switch size="small" checked={useSubjectLock} onChange={event => setUseSubjectLock(event.target.checked)} />}
                      label="Lock subject likeness with vision model"
                    />
                    <Button size="small" variant="outlined" onClick={describeSubject} disabled={describing || !source} startIcon={describing ? <CircularProgress size={12} color="inherit" /> : undefined}>
                      {describing ? 'Describing...' : 'Describe subject'}
                    </Button>
                  </Stack>
                  <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: subjectFeatures ? 1 : 0 }}>
                    Qwen3-VL reads only the person (hair, eyes, face, marks) and appends a locked feature list to your instruction. Nothing about background or clothing.
                  </Typography>
                  {subjectFeatures && (
                    <TextField
                      label="Locked subject features"
                      value={subjectFeatures}
                      onChange={event => setSubjectFeatures(event.target.value)}
                      multiline
                      minRows={2}
                      fullWidth
                      size="small"
                    />
                  )}
                </Paper>

                <Stack direction="row" spacing={1}>
                  <TextField select label="Model" size="small" value={checkpoint} onChange={event => setCheckpoint(event.target.value as 'turbo' | 'raw')} sx={{ minWidth: 130 }}>
                    <MenuItem value="turbo">Turbo</MenuItem>
                    <MenuItem value="raw">Raw</MenuItem>
                  </TextField>
                  <TextField label="Steps" size="small" type="number" value={steps} onChange={event => setSteps(Math.max(1, Number(event.target.value) || preset.steps))} />
                  <TextField label="CFG" size="small" type="number" value={cfg} onChange={event => setCfg(Math.max(0, Number(event.target.value) || preset.cfg))} />
                </Stack>
                <Alert severity={checkpoint === 'raw' ? 'warning' : 'info'} icon={false} sx={{ py: 0.25 }}>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>{CHECKPOINT_INFO[checkpoint].title}</Typography>
                  <Typography variant="caption" sx={{ display: 'block' }}>{CHECKPOINT_INFO[checkpoint].blurb}</Typography>
                  <Typography variant="caption" sx={{ display: 'block', color: 'text.disabled' }}>Recommended: {CHECKPOINT_INFO[checkpoint].recipe}</Typography>
                </Alert>
                <Box>
                  <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>Aspect ratio</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1.25 }}>
                    {CE_ASPECTS.map(a => {
                      const active = aspect === a.id
                      const { w, h } = ceAspectGlyph(a.dims)
                      return (
                        <Box
                          key={a.id}
                          onClick={() => applyAspect(a.id)}
                          title={a.label}
                          sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.25, cursor: 'pointer' }}
                        >
                          <Box sx={{ height: 26, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Box
                              sx={{
                                width: w,
                                height: h,
                                borderRadius: 0.5,
                                border: a.dims ? '2px solid' : '2px dashed',
                                borderColor: active ? 'primary.main' : 'text.disabled',
                                bgcolor: active ? 'primary.main' : 'transparent',
                                opacity: active ? 0.85 : 1,
                                transition: 'all 0.15s',
                              }}
                            />
                          </Box>
                          <Typography variant="caption" sx={{ fontSize: 10, lineHeight: 1, whiteSpace: 'nowrap', color: active ? 'primary.main' : 'text.secondary', fontWeight: active ? 700 : 400 }}>
                            {a.label}
                          </Typography>
                        </Box>
                      )
                    })}
                  </Box>
                  {aspect !== 'source' && (
                    <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: 0.5 }}>
                      Model card: matching the source aspect ratio preserves identity best. A different ratio can degrade preservation.
                    </Typography>
                  )}
                </Box>
                <Stack direction="row" spacing={1}>
                  <TextField label="Width" size="small" type="number" value={width} onChange={event => { setWidth(Math.max(16, Number(event.target.value) || width)); setAspect('custom') }} />
                  <TextField label="Height" size="small" type="number" value={height} onChange={event => { setHeight(Math.max(16, Number(event.target.value) || height)); setAspect('custom') }} />
                </Stack>
                <Box>
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>Grounding px: {groundingPx}</Typography>
                  <Slider min={512} max={1536} step={64} value={groundingPx} onChange={(_, value) => setGroundingPx(Array.isArray(value) ? value[0] : value)} />
                  <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                    Lower = stronger edit adherence. Higher = stronger identity/likeness. Try 1024+ for people.
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>LoRA strength: {loraStrength.toFixed(2)}</Typography>
                  <Slider min={0} max={2} step={0.05} value={loraStrength} onChange={(_, value) => setLoraStrength(Array.isArray(value) ? value[0] : value)} />
                </Box>
                {!twoPeople && (
                  <Accordion disableGutters sx={{ bgcolor: 'rgba(255,255,255,0.02)', '&:before': { display: 'none' } }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="body2">Placement box (advanced, optional)</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <FormControlLabel
                        control={<Switch size="small" checked={useRegions} onChange={event => setUseRegions(event.target.checked)} />}
                        label="Use a placement box"
                      />
                      <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: useRegions ? 1 : 0 }}>
                        This is a single-subject workflow — one box repositions/reframes the subject within the chosen aspect ratio. To place two different people, use the “Two people” tab.
                      </Typography>
                      {useRegions && (
                        <RegionCanvas
                          background={refPreview}
                          aspectW={width}
                          aspectH={height}
                          regions={regions}
                          onChange={setRegions}
                        />
                      )}
                    </AccordionDetails>
                  </Accordion>
                )}
                <FormControlLabel control={<Switch checked={allowHighRes} onChange={event => setAllowHighRes(event.target.checked)} />} label="Allow above 2MP experimental output" />
                {error && <Alert severity="error" sx={{ py: 0 }}>{error}</Alert>}
                <Button variant="contained" onClick={run} disabled={running || !canRun} startIcon={running ? <CircularProgress size={14} color="inherit" /> : undefined}>
                  {running ? 'Editing...' : 'Run Character Edit'}
                </Button>
              </Stack>
            </Stack>
          </Stack>
        </Paper>

        {(output || job) && (
          <Paper sx={{ p: 2 }}>
            <Stack spacing={1}>
              <Stack direction="row" justifyContent="space-between" alignItems="center">
                <Typography variant="h6">Result</Typography>
                {job && <Chip size="small" label={job.status} color={job.status === 'done' ? 'success' : job.status === 'error' ? 'error' : 'default'} />}
              </Stack>
              {output && <Box component="img" src={output.startsWith('data:') ? output : `data:image/png;base64,${output}`} alt="Edited result" sx={{ width: '100%', maxHeight: 720, objectFit: 'contain', borderRadius: 2, bgcolor: 'rgba(0,0,0,0.22)' }} />}
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {filename && <Button href={publicUrl(`/api/outputs/${encodeURIComponent(filename)}`)} target="_blank" rel="noreferrer">Open full output</Button>}
                {output && job?.status === 'done' && (
                  <Tooltip title="Feed this result back as the source so you can insert a second person or keep editing">
                    <Button variant="outlined" onClick={useResultAsSource}>Use as source for next edit</Button>
                  </Tooltip>
                )}
              </Stack>
              {output && job?.status === 'done' && (
                <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                  Add a second character by chaining: use this result as the source, then describe inserting the next person from their reference (one face per pass keeps likeness sharp).
                </Typography>
              )}
            </Stack>
          </Paper>
        )}
      </Stack>
    </Box>
  )
}
