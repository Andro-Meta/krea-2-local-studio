import { useState } from 'react'
import {
  Alert, Box, Button, Card, CardContent, Chip, CircularProgress, Divider, FormControlLabel,
  Grid, LinearProgress, MenuItem, Slider, Stack, Switch, TextField, Typography,
} from '@mui/material'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import CancelIcon from '@mui/icons-material/Cancel'
import ImageIcon from '@mui/icons-material/Image'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import SaveIcon from '@mui/icons-material/Save'
import { apiFetch } from '../../api'
import { useStore } from '../../store'
import LayerPanel from './LayerPanel'
import RealtimeCanvas, { makeImageLayer } from './RealtimeCanvas'
import RealtimeToolbar from './RealtimeToolbar'
import { createDefaultDocument, documentToPngB64, promptFromLayerNotes } from './canvasDocument'
import { useRealtimePreview } from './useRealtimePreview'
import CreatePromptFromImage from '../CreatePromptFromImage'
import MoodboardSection from '../GeneratePanel/MoodboardSection'

function statusText(status: string) {
  if (status === 'queued') return 'Queued'
  if (status === 'running') return 'Generating preview'
  if (status === 'ready') return 'Preview ready'
  if (status === 'final-ready') return 'Final ready'
  if (status === 'error') return 'Error'
  return 'Idle'
}

async function waitForFinal(jobId: string) {
  const deadline = Date.now() + 20 * 60 * 1000
  while (Date.now() < deadline) {
    const job = await apiFetch.jobStatus(jobId)
    if (job.status === 'done') return job
    if (job.status === 'error') throw new Error(job.error ?? 'Final render failed')
    await new Promise(resolve => window.setTimeout(resolve, 1500))
  }
  throw new Error('Final render timed out')
}

export default function RealtimeStudio() {
  const realtime = useStore(s => s.realtime)
  const params = useStore(s => s.params)
  const setRealtime = useStore(s => s.setRealtime)
  const setRealtimeDocument = useStore(s => s.setRealtimeDocument)
  const setRealtimePreview = useStore(s => s.setRealtimePreview)
  const setRealtimeSettings = useStore(s => s.setRealtimeSettings)
  const openLightbox = useStore(s => s.openLightbox)
  const setResults = useStore(s => s.setResults)
  const { previewNow, cancelPreview } = useRealtimePreview()
  const [finalLoading, setFinalLoading] = useState(false)

  const handleUpload = (b64: string) => {
    const layer = makeImageLayer(b64, realtime.document)
    setRealtimeDocument({ ...realtime.document, layers: [...realtime.document.layers, layer] })
    setRealtime({ selectedLayerId: layer.id, tool: 'select' })
  }

  const handleExport = async () => {
    const b64 = await documentToPngB64(realtime.document)
    openLightbox([{ src: `data:image/png;base64,${b64}`, prompt: realtime.prompt }], 0)
  }

  const handleFinalRender = async () => {
    setFinalLoading(true)
    setRealtimePreview({ status: 'queued', error: null })
    try {
      const canvasB64 = await documentToPngB64(realtime.document)
      const notes = promptFromLayerNotes(realtime.document)
      const prompt = [realtime.prompt.trim(), notes.trim()].filter(Boolean).join('\n\nCanvas notes:\n')
      const job = await apiFetch.generate({
        prompt,
        negative_prompt: realtime.negativePrompt,
        mode: 'redraw',
        diffusion_engine: 'native_pytorch',
        checkpoint: 'turbo',
        quantization: 'fp8',
        steps: realtime.settings.finalSteps,
        cfg: 0,
        width: realtime.document.width,
        height: realtime.document.height,
        num_images: 1,
        seed: realtime.settings.lockSeed ? realtime.settings.seed : -1,
        denoise: 1,
        mood: params.mood,
        moodboard_ids: params.selected_moodboard_ids,
        moodboard_uuids: params.moodboard_uuids,
        moodboard_strength: params.selected_moodboard_ids.length || params.moodboard_images.length || params.mood
          ? params.moodboard_strength
          : realtime.settings.canvasInfluence,
        moodboard_images: [canvasB64, ...params.moodboard_images],
      })
      const final = await waitForFinal(job.job_id)
      const image = final.images?.[0] ?? ''
      if (image) {
        setRealtimePreview({
          status: 'final-ready',
          image,
          seed: final.seed ?? null,
          lastUpdated: Date.now(),
          error: null,
          progress: 100,
        })
        setResults([image], final.seed, final.metadata ?? [])
      }
    } catch (e) {
      setRealtimePreview({ status: 'error', error: e instanceof Error ? e.message : 'Final render failed' })
    } finally {
      setFinalLoading(false)
    }
  }

  const addPreviewAsLayer = () => {
    if (!realtime.preview.image) return
    handleUpload(realtime.preview.image)
  }

  return (
    <Box sx={{ p: { xs: 1.5, md: 3 } }}>
      <Stack spacing={2.5}>
        <Card
          variant="outlined"
          sx={{
            borderRadius: 4,
            background: 'linear-gradient(135deg, rgba(103,80,164,0.18), rgba(76,175,80,0.08))',
          }}
        >
          <CardContent>
            <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems={{ md: 'center' }}>
              <Box sx={{ flex: 1 }}>
                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                  <AutoAwesomeIcon color="primary" />
                  <Typography variant="h5">Realtime Studio</Typography>
                  <Chip size="small" label="Krea live preview" color="primary" variant="outlined" />
                </Stack>
                <Typography variant="body2" sx={{ color: 'text.secondary', maxWidth: 820 }}>
                  Draw, place references, and let Krea redraw the canvas after you pause. Live preview is for direction and composition; Final render is the quality pass.
                </Typography>
              </Box>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                <Button startIcon={<PlayArrowIcon />} variant="contained" onClick={previewNow} disabled={realtime.preview.paused}>
                  Preview now
                </Button>
                <Button startIcon={<CancelIcon />} variant="outlined" onClick={cancelPreview}>
                  Cancel
                </Button>
                <Button startIcon={<SaveIcon />} color="secondary" variant="contained" onClick={handleFinalRender} disabled={finalLoading}>
                  Final render
                </Button>
              </Stack>
            </Stack>
          </CardContent>
        </Card>

        <Grid container spacing={2.5}>
          <Grid item xs={12} lg={7.5}>
            <Stack spacing={2}>
              <Card variant="outlined" sx={{ borderRadius: 4, p: { xs: 1, sm: 2 }, bgcolor: 'background.default' }}>
                <RealtimeCanvas
                  document={realtime.document}
                  selectedLayerId={realtime.selectedLayerId}
                  tool={realtime.tool}
                  color={realtime.color}
                  brushSize={realtime.brushSize}
                  shape={realtime.shape}
                  onDocumentChange={setRealtimeDocument}
                  onSelectLayer={id => setRealtime({ selectedLayerId: id })}
                />
              </Card>
              <RealtimeToolbar
                tool={realtime.tool}
                color={realtime.color}
                brushSize={realtime.brushSize}
                shape={realtime.shape}
                onTool={tool => setRealtime({ tool })}
                onColor={color => setRealtime({ color })}
                onBrushSize={brushSize => setRealtime({ brushSize })}
                onShape={shape => setRealtime({ shape })}
                onUpload={handleUpload}
                onClear={() => {
                  setRealtimeDocument(createDefaultDocument())
                  setRealtime({ selectedLayerId: null })
                }}
                onExport={handleExport}
              />
            </Stack>
          </Grid>

          <Grid item xs={12} lg={4.5}>
            <Stack spacing={2}>
              <Card variant="outlined" sx={{ borderRadius: 4 }}>
                <CardContent>
                  <Stack spacing={1.5}>
                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                      <Typography variant="h6">Live Preview</Typography>
                      <Chip label={statusText(realtime.preview.status)} color={realtime.preview.status === 'error' ? 'error' : 'primary'} variant="outlined" />
                    </Stack>
                    {(realtime.preview.status === 'queued' || realtime.preview.status === 'running' || finalLoading) && (
                      <LinearProgress variant={realtime.preview.progress ? 'determinate' : 'indeterminate'} value={realtime.preview.progress} />
                    )}
                    <Box
                      sx={{
                        borderRadius: 3,
                        minHeight: 320,
                        bgcolor: 'action.hover',
                        display: 'grid',
                        placeItems: 'center',
                        overflow: 'hidden',
                      }}
                    >
                      {finalLoading ? (
                        <Stack alignItems="center" spacing={1}>
                          <CircularProgress />
                          <Typography variant="body2">Rendering final quality pass...</Typography>
                        </Stack>
                      ) : realtime.preview.image ? (
                        <Box component="img" src={`data:image/png;base64,${realtime.preview.image}`} sx={{ width: '100%', display: 'block' }} />
                      ) : (
                        <Stack alignItems="center" spacing={1} sx={{ color: 'text.secondary', textAlign: 'center', p: 3 }}>
                          <ImageIcon />
                          <Typography variant="body2">Preview appears here after you draw or press Preview now.</Typography>
                        </Stack>
                      )}
                    </Box>
                    {realtime.preview.error && <Alert severity="error">{realtime.preview.error}</Alert>}
                    <Stack direction="row" spacing={1} flexWrap="wrap">
                      <Button size="small" disabled={!realtime.preview.image} onClick={() => openLightbox([{ src: `data:image/png;base64,${realtime.preview.image}`, prompt: realtime.prompt, metadata: realtime.preview.metadata ?? undefined }], 0)}>
                        Open
                      </Button>
                      <Button size="small" disabled={!realtime.preview.image} onClick={addPreviewAsLayer}>
                        Add to canvas
                      </Button>
                    </Stack>
                  </Stack>
                </CardContent>
              </Card>

              <Card variant="outlined" sx={{ borderRadius: 4 }}>
                <CardContent>
                  <Stack spacing={1.5}>
                    <Typography variant="h6">Prompt</Typography>
                    <TextField label="What should Krea redraw this into?" value={realtime.prompt} onChange={e => setRealtime({ prompt: e.target.value })} multiline minRows={3} />
                    <CreatePromptFromImage
                      value={realtime.prompt}
                      onChange={prompt => setRealtime({ prompt })}
                      compact
                    />
                    <TextField label="Negative prompt" value={realtime.negativePrompt} onChange={e => setRealtime({ negativePrompt: e.target.value })} multiline minRows={2} />
                    <MoodboardSection
                      intro="Apply Krea moodboards to live previews and final renders as text-first style guidance. The canvas remains the composition reference; uploaded moodboard images are optional stronger style references."
                      promptValue={realtime.prompt}
                      onPromptFallback={prompt => setRealtime({ prompt })}
                    />
                    <Divider />
                    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
                      <TextField
                        select
                        label="Preview size"
                        value={realtime.settings.previewSize}
                        onChange={e => setRealtimeSettings({ previewSize: Number(e.target.value) })}
                        fullWidth
                      >
                        <MenuItem value={384}>Fast 384px</MenuItem>
                        <MenuItem value={512}>Balanced 512px</MenuItem>
                        <MenuItem value={640}>Sharper 640px</MenuItem>
                      </TextField>
                      <TextField
                        select
                        label="Preview steps"
                        value={realtime.settings.previewSteps}
                        onChange={e => setRealtimeSettings({ previewSteps: Number(e.target.value) })}
                        fullWidth
                      >
                        <MenuItem value={4}>Fast 4</MenuItem>
                        <MenuItem value={5}>Balanced 5</MenuItem>
                        <MenuItem value={6}>Better 6</MenuItem>
                      </TextField>
                    </Stack>
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="body2">Canvas influence</Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                          {Math.round(realtime.settings.canvasInfluence * 100)}%
                        </Typography>
                      </Stack>
                      <Slider
                        value={realtime.settings.canvasInfluence}
                        min={0.3}
                        max={0.9}
                        step={0.05}
                        onChange={(_, value) => setRealtimeSettings({ canvasInfluence: value as number })}
                        valueLabelDisplay="auto"
                        valueLabelFormat={value => `${Math.round(Number(value) * 100)}%`}
                      />
                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                        Lower values redraw sketches into realism. Higher values preserve uploaded references and layout more tightly.
                      </Typography>
                    </Box>
                    <Box>
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="body2">Seed</Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                          {realtime.settings.lockSeed ? realtime.settings.seed : 'random'}
                        </Typography>
                      </Stack>
                      <Slider
                        value={Math.max(0, realtime.settings.seed)}
                        min={0}
                        max={999999}
                        step={1}
                        disabled={!realtime.settings.lockSeed}
                        onChange={(_, value) => setRealtimeSettings({ seed: value as number })}
                        valueLabelDisplay="auto"
                      />
                      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
                        <TextField
                          size="small"
                          label="Seed value"
                          type="number"
                          value={Math.max(0, realtime.settings.seed)}
                          disabled={!realtime.settings.lockSeed}
                          onChange={e => setRealtimeSettings({ seed: Math.max(0, Number(e.target.value) || 0) })}
                          fullWidth
                        />
                        <Button
                          variant="outlined"
                          onClick={() => setRealtimeSettings({ seed: Math.floor(Math.random() * 1000000), lockSeed: true })}
                        >
                          Random seed
                        </Button>
                      </Stack>
                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                        Lock the seed while tweaking prompts or canvas influence; unlock it when you want fresh variations.
                      </Typography>
                    </Box>
                    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
                      <FormControlLabel
                        control={<Switch checked={realtime.settings.autoPreview} onChange={e => setRealtimeSettings({ autoPreview: e.target.checked })} />}
                        label="Auto-preview"
                      />
                      <FormControlLabel
                        control={<Switch checked={realtime.settings.lockSeed} onChange={e => setRealtimeSettings({ lockSeed: e.target.checked, seed: realtime.settings.seed < 0 ? Math.floor(Math.random() * 1000000) : realtime.settings.seed })} />}
                        label="Lock seed"
                      />
                      <FormControlLabel
                        control={<Switch checked={realtime.preview.paused} onChange={e => setRealtimePreview({ paused: e.target.checked })} />}
                        label="Pause live preview"
                      />
                    </Stack>
                  </Stack>
                </CardContent>
              </Card>

              <LayerPanel
                document={realtime.document}
                selectedLayerId={realtime.selectedLayerId}
                onDocumentChange={setRealtimeDocument}
                onSelectLayer={id => setRealtime({ selectedLayerId: id, tool: 'select' })}
              />
            </Stack>
          </Grid>
        </Grid>
      </Stack>
    </Box>
  )
}
