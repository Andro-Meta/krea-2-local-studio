import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Alert, Box, Button, CircularProgress, LinearProgress, Stack, Typography } from '@mui/material'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import WarningAmberIcon from '@mui/icons-material/WarningAmber'
import { useStore } from '../../store'
import { apiFetch, connectWS } from '../../api'
import PromptSection from './PromptSection'
import ModelSection from './ModelSection'
import DimensionSection from './DimensionSection'
import ParameterSection from './ParameterSection'
import LoraSection from './LoraSection'
import MoodboardSection from './MoodboardSection'
import CanvasControl from './CanvasControl'
import ResultsView from './ResultsView'

export default function GeneratePanel() {
  const { params, generating, progress, results, lastSeed, generationError,
          setGenerating, setJobId, setProgress, setResults, setError,
          modelLoaded, setModelLoaded, setTab } = useStore()
  const inRedrawStudio = params.mode !== 'txt2img'

  const wsRef = useRef<WebSocket | null>(null)
  const [modelLoading, setModelLoading] = useState(false)

  // Poll model status every 5s
  useEffect(() => {
    const check = () =>
      apiFetch.system().then(r => {
        setModelLoaded(r.model_status?.loaded ?? false)
        setModelLoading(r.model_status?.loading ?? false)
      }).catch(() => {})
    check()
    const t = setInterval(check, 5000)
    return () => clearInterval(t)
  }, [])

  const handleGenerate = useCallback(async () => {
    if (generating) return
    setError(null)
    setGenerating(true)
    setProgress(0)
    setResults([])
    try {
      const { job_id } = await apiFetch.generate({
        prompt: params.prompt,
        negative_prompt: params.negative_prompt,
        mode: params.mode,
        checkpoint: params.checkpoint,
        quantization: params.quantization,
        steps: params.steps,
        cfg: params.cfg,
        mu: params.mu,
        y1: params.y1,
        y2: params.y2,
        width: params.width,
        height: params.height,
        num_images: params.num_images,
        seed: params.seed,
        denoise: params.denoise,
        edit_provider: params.edit_provider,
        quality_preset: params.quality_preset,
        use_rebalance: params.use_rebalance,
        rebalance_multiplier: params.rebalance_multiplier,
        rebalance_weights: params.rebalance_weights,
        loras: params.loras,
        bboxes: params.bboxes,
        init_image_b64: params.init_image_b64 || undefined,
        mask_b64: params.mask_b64 || undefined,
        ref_image1_b64: params.ref_image1_b64 || undefined,
        ref_image2_b64: params.ref_image2_b64 || undefined,
        ref_image3_b64: params.ref_image3_b64 || undefined,
        use_prompt_expander: params.use_prompt_expander,
        refine: params.refine,
        refine_denoise: params.refine_denoise,
        refine_steps: params.refine_steps,
        mood: params.mood,
        moodboard_strength: params.moodboard_strength,
        moodboard_images: params.moodboard_images,
      })
      setJobId(job_id)

      wsRef.current = connectWS(job_id, (data: any) => {
        if (data.type === 'progress') setProgress(data.pct ?? 0)
        if (data.type === 'done') {
          setResults(data.images ?? [], data.seed)
          setGenerating(false)
          setProgress(100)
          const warns = data.lora_warnings ?? []
          if (warns.length) {
            setError('LoRA not applied — ' + warns
              .map((w: any) => `${w.name}: ${w.reason ?? 'incompatible'}`).join('; '))
          } else if (data.provider_warning) {
            setError(data.provider_warning)
          }
        }
        if (data.type === 'error') {
          setError(data.error ?? 'Unknown error')
          setGenerating(false)
        }
      })
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e.message ?? 'Request failed')
      setGenerating(false)
    }
  }, [params, generating])

  return (
    <Box sx={{ p: { xs: 1.5, sm: 2 }, maxWidth: 900, mx: 'auto' }}>
      <Stack spacing={2}>
        {!modelLoaded && modelLoading && (
          <Alert severity="info" icon={<CircularProgress size={18} />}>
            Model loading… first load takes ~1–2 minutes (DiT + VAE + text encoder).
          </Alert>
        )}
        {!modelLoaded && !modelLoading && (
          <Alert
            severity="warning"
            icon={<WarningAmberIcon />}
            action={
              <Button color="inherit" size="small" onClick={() => setTab(3)}>
                Load model
              </Button>
            }
          >
            No model loaded — go to System tab to load a checkpoint before generating.
          </Alert>
        )}

        <PromptSection />
        <ModelSection />
        <DimensionSection />
        {!inRedrawStudio && <MoodboardSection />}
        <LoraSection />
        {!inRedrawStudio && <CanvasControl />}
        <ParameterSection />

        {generationError && <Alert severity="error" onClose={() => setError(null)}>{generationError}</Alert>}

        {generating && (
          <Box>
            <LinearProgress variant="determinate" value={progress} />
            <Typography variant="caption" sx={{ color: 'text.secondary', mt: 0.5, display: 'block' }}>
              {progress}% complete
            </Typography>
          </Box>
        )}

        <Button
          variant="contained"
          size="large"
          startIcon={generating ? <CircularProgress size={18} color="inherit" /> : <AutoAwesomeIcon />}
          onClick={handleGenerate}
          disabled={generating || !params.prompt.trim() || !modelLoaded}
          fullWidth
          sx={{ height: 52, fontSize: '1rem' }}
        >
          {generating ? 'Generating…' : 'Generate'}
        </Button>

        <ResultsView images={results} seed={lastSeed} />
      </Stack>
    </Box>
  )
}
