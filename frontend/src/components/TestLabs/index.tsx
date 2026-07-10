import { useMemo, useRef, useState } from 'react'
import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Chip, CircularProgress, IconButton, Paper, Stack, Tab, Tabs, TextField, Tooltip, Typography } from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import ScienceIcon from '@mui/icons-material/Science'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import { TAB, useStore } from '../../store'
import { apiFetch, type GenerationRequest } from '../../api'
import CompareGrid from './CompareGrid'
import MoodboardSection from '../GeneratePanel/MoodboardSection'
import CreatePromptFromImage from '../CreatePromptFromImage'
import { labWorkflows, type LabWorkflow, type LabWorkflowId } from './labPresets'
import { exportRunJson, runLabWorkflow, type LabRun } from './labRunner'

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const value = String(reader.result || '')
      resolve(value.includes(',') ? value.split(',')[1] : value)
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

function workflowNeedsSource(workflow: LabWorkflow): boolean {
  return workflow.cases.some(testCase => testCase.sourceInjection && testCase.sourceInjection !== 'none')
}

function withOverrides(workflow: LabWorkflow, prompt: string, moodboard: Partial<GenerationRequest>): LabWorkflow {
  const text = prompt.trim()
  return {
    ...workflow,
    cases: workflow.cases.map(testCase => ({
      ...testCase,
      request: { ...testCase.request, ...(text ? { prompt: text } : {}), ...moodboard },
    })),
  }
}

function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export default function TestLabsPanel() {
  const [workflowId, setWorkflowId] = useState<LabWorkflowId>(labWorkflows[0].id)
  const activeWorkflow = useMemo(() => labWorkflows.find(item => item.id === workflowId) ?? labWorkflows[0], [workflowId])
  const [prompt, setPrompt] = useState(activeWorkflow.defaultPrompt)
  const [seed, setSeed] = useState(activeWorkflow.defaultSeed)
  const [sourceImage, setSourceImage] = useState('')
  const [sourcePreview, setSourcePreview] = useState('')
  const [run, setRun] = useState<LabRun | null>(null)
  const [running, setRunning] = useState(false)
  const [message, setMessage] = useState<{ severity: 'info' | 'success' | 'warning' | 'error'; text: string } | null>(null)
  const stopAfterCurrent = useRef(false)
  const { setTab, params, setMoodboardSuggestions } = useStore()
  const [wandBusy, setWandBusy] = useState(false)

  const moodboardOverrides = useMemo<Partial<GenerationRequest>>(() => ({
    mood: params.mood,
    moodboard_ids: params.selected_moodboard_ids,
    moodboard_uuids: params.moodboard_uuids,
    moodboard_strength: params.moodboard_strength,
    moodboard_images: params.moodboard_images,
  }), [params.mood, params.selected_moodboard_ids, params.moodboard_uuids, params.moodboard_strength, params.moodboard_images])

  const selectedWorkflow = useMemo(
    () => withOverrides(activeWorkflow, prompt, moodboardOverrides),
    [activeWorkflow, prompt, moodboardOverrides],
  )
  const needsSource = workflowNeedsSource(selectedWorkflow)

  const expandPromptWand = async () => {
    if (!prompt.trim()) { setMessage({ severity: 'warning', text: 'Type a prompt first, then use the magic wand.' }); return }
    setWandBusy(true)
    try {
      const res = await apiFetch.expandPrompt(prompt)
      if (res.error) setMessage({ severity: 'warning', text: res.error })
      if (res.expanded) setPrompt(res.expanded.trim())
      if (res.suggested_moodboards?.length) setMoodboardSuggestions(res.suggested_moodboards)
    } catch (err: any) {
      setMessage({ severity: 'error', text: err?.response?.data?.detail ?? err?.message ?? 'Magic wand failed.' })
    } finally {
      setWandBusy(false)
    }
  }

  const selectWorkflow = (_: unknown, next: LabWorkflowId) => {
    const workflow = labWorkflows.find(item => item.id === next)
    if (!workflow) return
    setWorkflowId(next)
    setPrompt(workflow.defaultPrompt)
    setSeed(workflow.defaultSeed)
    setRun(null)
    setMessage(null)
    stopAfterCurrent.current = false
  }

  const uploadSource = async (file?: File) => {
    if (!file) return
    const b64 = await fileToBase64(file)
    setSourceImage(b64)
    setSourcePreview(`data:${file.type || 'image/png'};base64,${b64}`)
  }

  const startRun = async () => {
    if (running) return
    if (needsSource && !sourceImage) {
      setMessage({ severity: 'warning', text: 'Upload a source image before running this workflow.' })
      return
    }
    setRunning(true)
    setMessage(null)
    stopAfterCurrent.current = false
    try {
      const result = await runLabWorkflow(selectedWorkflow, {
        sourceImageB64: sourceImage,
        seed,
        shouldStop: () => stopAfterCurrent.current,
        onUpdate: setRun,
      })
      setRun(result)
      setMessage({ severity: 'success', text: 'Test Lab run finished.' })
    } catch (error: any) {
      setMessage({ severity: 'error', text: error?.message ?? 'Test Lab run failed.' })
    } finally {
      setRunning(false)
    }
  }

  const exportRun = () => {
    if (!run) return
    downloadText(`krea-test-lab-${run.workflowId}-${new Date(run.startedAt).toISOString().replace(/[:.]/g, '-')}.json`, exportRunJson(run))
  }

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2 }, maxWidth: 1180, mx: 'auto' }}>
      <Stack spacing={2}>
        <Paper sx={{ p: 2 }}>
          <Stack spacing={1.5}>
            <Stack direction="row" spacing={1} alignItems="center">
              <ScienceIcon color="primary" />
              <Box>
                <Typography variant="h5">Test Labs</Typography>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  Curated workflow sub-tabs with defaults already set for A/B testing and dialing in Krea 2 behavior.
                </Typography>
              </Box>
            </Stack>
            <Tabs value={workflowId} onChange={selectWorkflow} variant="scrollable" scrollButtons="auto">
              {labWorkflows.map(workflow => <Tab key={workflow.id} value={workflow.id} label={workflow.label} />)}
            </Tabs>
          </Stack>
        </Paper>

        <Paper sx={{ p: 2 }}>
          <Stack spacing={1.5}>
            <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" gap={1.5}>
              <Box>
                <Typography variant="h6">{activeWorkflow.label}</Typography>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  {activeWorkflow.description}
                </Typography>
              </Box>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                <Chip size="small" label={`${activeWorkflow.cases.length} cases`} />
                <Chip size="small" label={`seed ${seed}`} color="primary" variant="outlined" />
                {needsSource && <Chip size="small" label="source image required" color="warning" />}
              </Stack>
            </Stack>

            {activeWorkflow.warnings.map(warning => (
              <Alert key={warning} severity="info" sx={{ py: 0 }}>{warning}</Alert>
            ))}

            <Stack direction={{ xs: 'column', md: 'row' }} spacing={1.5}>
              <Stack spacing={1} sx={{ flex: 1 }}>
                <TextField
                  label="Workflow prompt"
                  value={prompt}
                  onChange={event => setPrompt(event.target.value)}
                  multiline
                  minRows={3}
                  fullWidth
                  InputProps={{
                    endAdornment: (
                      <Tooltip title="Magic wand: expand this prompt">
                        <span style={{ position: 'absolute', top: 6, right: 6 }}>
                          <IconButton size="small" onClick={expandPromptWand} disabled={wandBusy || !prompt.trim()}>
                            {wandBusy ? <CircularProgress size={16} /> : <AutoAwesomeIcon fontSize="small" />}
                          </IconButton>
                        </span>
                      </Tooltip>
                    ),
                  }}
                />
                <CreatePromptFromImage value={prompt} onChange={setPrompt} mode="replace" withGuidance label="Create prompt from image" />
                <Paper variant="outlined" sx={{ p: 1.25, bgcolor: 'rgba(255,255,255,0.02)' }}>
                  <MoodboardSection intro="Same moodboard picker as Create mode. Selected boards apply their Qwen style guidance to every case in this workflow (including Turbo 4X, via prompt enrichment)." />
                </Paper>
              </Stack>
              <Stack spacing={1} sx={{ minWidth: { md: 220 } }}>
                <TextField
                  label="Seed"
                  type="number"
                  value={seed}
                  onChange={event => setSeed(Number(event.target.value) || activeWorkflow.defaultSeed)}
                  size="small"
                />
                {needsSource && (
                  <Button variant="outlined" component="label">
                    Upload Source Image
                    <input hidden type="file" accept="image/*" onChange={event => uploadSource(event.target.files?.[0])} />
                  </Button>
                )}
                {sourcePreview && (
                  <Box component="img" src={sourcePreview} alt="Source preview" sx={{ width: 120, height: 120, objectFit: 'cover', borderRadius: 1, border: '1px solid', borderColor: 'divider' }} />
                )}
              </Stack>
            </Stack>

            <Accordion disableGutters>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle2">Cases and defaults</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Stack spacing={1}>
                  {selectedWorkflow.cases.map(testCase => (
                    <Paper key={testCase.id} variant="outlined" sx={{ p: 1 }}>
                      <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" gap={1}>
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 700 }}>{testCase.label}</Typography>
                          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>{testCase.notes}</Typography>
                          <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block' }}>
                            {testCase.request.checkpoint} · {testCase.request.sampler}/{testCase.request.scheduler} · {testCase.request.steps} steps · CFG {testCase.request.cfg}
                          </Typography>
                        </Box>
                        <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                          {testCase.sourceInjection && testCase.sourceInjection !== 'none' && <Chip size="small" label="source" color="warning" variant="outlined" />}
                          {testCase.disabledReason && <Chip size="small" label="disabled" color="default" />}
                          {testCase.request.loras?.map((lora: any) => (
                            <Chip key={lora.filename || lora.name} size="small" label={lora.name || lora.filename} variant="outlined" />
                          ))}
                        </Stack>
                      </Stack>
                    </Paper>
                  ))}
                </Stack>
              </AccordionDetails>
            </Accordion>

            {message && <Alert severity={message.severity} sx={{ py: 0 }}>{message.text}</Alert>}
            {run?.cases.some(testCase => testCase.status === 'error' || testCase.status === 'blocked') && (
              <Alert
                severity="warning"
                sx={{ py: 0 }}
                action={<Button color="inherit" size="small" onClick={() => setTab(TAB.SYSTEM)}>Open System</Button>}
              >
                One or more cases failed. Check System optional assets, model status, and ComfyUI node status before rerunning.
              </Alert>
            )}

            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Button variant="contained" onClick={startRun} disabled={running} startIcon={running ? <CircularProgress size={14} color="inherit" /> : undefined}>
                {running ? 'Running...' : 'Run Workflow'}
              </Button>
              <Button variant="outlined" disabled={!running} onClick={() => { stopAfterCurrent.current = true; setMessage({ severity: 'info', text: 'Will stop after the current case finishes.' }) }}>
                Stop After Current
              </Button>
              <Button variant="text" disabled={!run} onClick={exportRun}>
                Export Run JSON
              </Button>
            </Stack>
          </Stack>
        </Paper>

        {run && (
          <Paper sx={{ p: 2 }}>
            <CompareGrid cases={run.cases} />
          </Paper>
        )}
      </Stack>
    </Box>
  )
}
