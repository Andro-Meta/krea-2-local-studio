import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, CircularProgress, LinearProgress, Stack, Typography } from '@mui/material'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import WarningAmberIcon from '@mui/icons-material/WarningAmber'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { TAB, useStore } from '../../store'
import { apiFetch, connectWS } from '../../api'
import PromptSection from './PromptSection'
import ModelSection from './ModelSection'
import DimensionSection from './DimensionSection'
import ParameterSection from './ParameterSection'
import LoraSection from './LoraSection'
import MoodboardSection from './MoodboardSection'
import RecipeSection from './RecipeSection'
import StyleReferenceSection from './StyleReferenceSection'
import AdvancedSceneSection from './AdvancedSceneSection'
import CanvasControl from './CanvasControl'
import QuickPresets from './QuickPresets'
import MrFlowSettings from './MrFlowSettings'
import GenerationQueue from './GenerationQueue'
import ResultsView from './ResultsView'

const ACTIVE_JOB_KEY = 'krea2_active_generation_job'

export default function GeneratePanel() {
  const { params, generating, progress, results, resultsMetadata, lastSeed, generationError,
          queuePosition, queueLength,
          promptBusy,
          setGenerating, setJobId, setProgress, setResults, setError,
          setQueue, modelLoaded, setModelLoaded, setTab, engineCatalog, setEngineCatalog } = useStore()
  const inRedrawStudio = params.mode !== 'txt2img'
  const activeEngine = engineCatalog?.engines.find(engine => engine.engine_id === params.diffusion_engine)
  const supports = activeEngine ?? engineCatalog?.engines.find(engine => engine.engine_id === 'native_pytorch')

  const wsRef = useRef<WebSocket | null>(null)
  const pollRef = useRef<number | null>(null)
  const reconnectRef = useRef<number | null>(null)
  const heartbeatRef = useRef<number | null>(null)
  const [modelLoading, setModelLoading] = useState(false)
  const [connectionNote, setConnectionNote] = useState('')

  useEffect(() => () => {
    wsRef.current?.close()
    wsRef.current = null
    if (pollRef.current) window.clearInterval(pollRef.current)
    pollRef.current = null
    if (reconnectRef.current) window.clearTimeout(reconnectRef.current)
    reconnectRef.current = null
    if (heartbeatRef.current) window.clearInterval(heartbeatRef.current)
    heartbeatRef.current = null
  }, [])

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

  useEffect(() => {
    apiFetch.engineCatalog().then(setEngineCatalog).catch(() => undefined)
  }, [setEngineCatalog])

  const stopWatchingJob = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onclose = null
      wsRef.current.onerror = null
      wsRef.current.close()
    }
    wsRef.current = null
    if (pollRef.current) window.clearInterval(pollRef.current)
    pollRef.current = null
    if (reconnectRef.current) window.clearTimeout(reconnectRef.current)
    reconnectRef.current = null
    if (heartbeatRef.current) window.clearInterval(heartbeatRef.current)
    heartbeatRef.current = null
    localStorage.removeItem(ACTIVE_JOB_KEY)
    setConnectionNote('')
  }, [])

  const applyJobSnapshot = useCallback((data: any) => {
    if (data.type === 'init' || data.type === 'queue') {
      setQueue(data.queue_position ?? null, data.queue_length ?? null)
    }
    if (data.type === 'status' && data.status === 'running') {
      setQueue(null, data.queue_length ?? null)
    }
    if (data.type === 'progress') setProgress(data.pct ?? 0)
    if (data.type === 'batch') {
      setProgress(data.progress ?? 0)
      setQueue(data.queue_position ?? null, data.queue_length ?? null)
      if ((data.images ?? []).length) setResults(data.images ?? [], data.seed, data.metadata ?? [])
      if (data.status === 'done') {
        setGenerating(false)
        setProgress(100)
        setQueue(null, null)
        stopWatchingJob()
      }
      if (data.status === 'error' || data.status === 'blocked') {
        setError(data.error ?? 'Batch generation failed.')
        setGenerating(false)
        setQueue(null, null)
        stopWatchingJob()
      }
    }
    if (data.status === 'queued') {
      setQueue(data.queue_position ?? null, data.queue_length ?? null)
    }
    if (data.status === 'running') {
      setGenerating(true)
      setProgress(data.progress ?? 0)
      setQueue(null, data.queue_length ?? null)
    }
    if (data.type === 'done' || data.status === 'done') {
      setResults(data.images ?? [], data.seed, data.metadata ?? [])
      setGenerating(false)
      setProgress(100)
      setQueue(null, null)
      const warns = data.lora_warnings ?? []
      if (warns.length) {
        setError('LoRA not applied — ' + warns
          .map((w: any) => `${w.name}: ${w.reason ?? 'incompatible'}`).join('; '))
      } else if (data.provider_warning) {
        setError(data.provider_warning)
      }
      stopWatchingJob()
    }
    if (data.type === 'error' || data.status === 'error') {
      setError(data.error ?? 'Unknown error')
      setGenerating(false)
      setQueue(null, null)
      stopWatchingJob()
    }
    if (data.type === 'blocked' || data.status === 'blocked') {
      setError(data.error ?? 'Blocked by child safety filter.')
      setGenerating(false)
      setQueue(null, null)
      stopWatchingJob()
    }
    if (data.type === 'cancelled' || data.status === 'cancelled') {
      setGenerating(false)
      setProgress(0)
      setQueue(null, null)
      stopWatchingJob()
    }
  }, [setError, setGenerating, setProgress, setQueue, setResults, stopWatchingJob])

  const watchJob = useCallback((jobId: string) => {
    localStorage.setItem(ACTIVE_JOB_KEY, jobId)
    if (reconnectRef.current) window.clearTimeout(reconnectRef.current)
    reconnectRef.current = null
    if (heartbeatRef.current) window.clearInterval(heartbeatRef.current)
    heartbeatRef.current = null
    if (wsRef.current) {
      wsRef.current.onclose = null
      wsRef.current.onerror = null
      wsRef.current.close()
    }
    wsRef.current = connectWS(
      jobId,
      (data: any) => {
        setConnectionNote('')
        applyJobSnapshot(data)
      },
      () => {
        setConnectionNote('Live connection dropped. Reconnecting and checking the job every few seconds.')
        if (reconnectRef.current) window.clearTimeout(reconnectRef.current)
        reconnectRef.current = window.setTimeout(() => {
          if (localStorage.getItem(ACTIVE_JOB_KEY) === jobId) watchJob(jobId)
        }, 2500)
      },
    )
    heartbeatRef.current = window.setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        try { wsRef.current.send('ping') } catch {}
      }
    }, 20000)
    if (pollRef.current) window.clearInterval(pollRef.current)
    pollRef.current = window.setInterval(() => {
      apiFetch.jobStatus(jobId)
        .then(data => {
          setConnectionNote('')
          applyJobSnapshot(data)
        })
        .catch(() => {
          setConnectionNote('Network is spotty. Krea is still trying to reconnect to this job.')
        })
    }, 2500)
  }, [applyJobSnapshot])

  useEffect(() => {
    const resumeJob = () => {
      const jobId = localStorage.getItem(ACTIVE_JOB_KEY)
      if (!jobId) return
      apiFetch.jobStatus(jobId)
        .then(data => {
          if (['queued', 'running'].includes(data.status)) {
            setGenerating(true)
            setJobId(jobId)
            applyJobSnapshot(data)
            watchJob(jobId)
          } else if (data.status === 'done' && data.images?.length) {
            applyJobSnapshot(data)
          } else {
            localStorage.removeItem(ACTIVE_JOB_KEY)
          }
        })
        .catch(() => {
          setConnectionNote('Could not reconnect yet. Krea will retry when the phone is back online.')
        })
    }
    resumeJob()
    const onVisible = () => {
      if (document.visibilityState === 'visible') resumeJob()
    }
    window.addEventListener('online', resumeJob)
    window.addEventListener('focus', resumeJob)
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      window.removeEventListener('online', resumeJob)
      window.removeEventListener('focus', resumeJob)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [applyJobSnapshot, setGenerating, setJobId, watchJob])

  const handleGenerate = useCallback(async () => {
    if (generating || promptBusy) return
    setError(null)
    setGenerating(true)
    setProgress(0)
    setQueue(null, null)
    setResults([])
    try {
      const { job_id, status, queue_position, queue_length } = await apiFetch.generate({
        prompt: params.prompt,
        negative_prompt: params.negative_prompt,
        mode: params.mode,
        model_profile: params.model_profile,
        diffusion_engine: params.diffusion_engine,
        checkpoint: params.checkpoint,
        quantization: params.quantization,
        turbo_int8_variant: params.turbo_int8_variant,
        steps: params.steps,
        cfg: params.cfg,
        mu: params.mu,
        y1: params.y1,
        y2: params.y2,
        width: params.width,
        height: params.height,
        num_images: params.num_images,
        batch_mode: params.batch_mode,
        parallel_batch_confirmed: params.parallel_batch_confirmed,
        batch_int8_all: params.batch_int8_all,
        seed: params.seed,
        denoise: params.denoise,
        sampler: params.sampler,
        scheduler: params.scheduler,
        cfg_zero_star: params.cfg_zero_star,
        cfg_zero_init_steps: params.cfg_zero_init_steps,
        res4lyf_sampler: params.res4lyf_sampler,
        res4lyf_eta: params.res4lyf_eta,
        res4lyf_bongmath: params.res4lyf_bongmath,
        actual_denoise: params.actual_denoise,
        incontext_edit: params.incontext_edit,
        incontext_image_b64: params.incontext_image_b64 || undefined,
        incontext_mask_b64: params.incontext_mask_b64 || undefined,
        incontext_vision_position: params.incontext_vision_position,
        incontext_vision_megapixels: params.incontext_vision_megapixels,
        incontext_encoder: params.incontext_encoder,
        incontext_system_prompt: params.incontext_system_prompt || undefined,
        style_transfer_image_b64: params.style_transfer_image_b64 || undefined,
        style_transfer_method: params.style_transfer_method,
        style_transfer_weight: params.style_transfer_weight,
        style_transfer_apply_to: params.style_transfer_apply_to,
        inpaint_method: params.inpaint_method,
        differential_inpaint: params.differential_inpaint,
        differential_strength: params.differential_strength,
        lanpaint_inner_steps: params.lanpaint_inner_steps,
        lanpaint_strength: params.lanpaint_strength,
        lanpaint_lambda: params.lanpaint_lambda,
        lanpaint_step_size: params.lanpaint_step_size,
        lanpaint_beta: params.lanpaint_beta,
        lanpaint_friction: params.lanpaint_friction,
        lanpaint_early_stop: params.lanpaint_early_stop,
        lanpaint_prompt_mode: params.lanpaint_prompt_mode,
        edit_provider: params.edit_provider,
        quality_preset: params.quality_preset,
        creativity: params.creativity,
        style_references: params.style_references,
        style_fusion_mode: params.style_fusion_mode,
        image_prompt_enabled: params.image_prompt_enabled,
        image_prompt_mode: params.image_prompt_mode,
        image_prompt_strength: params.image_prompt_strength,
        regional_prompts: params.regional_prompts,
        regional_base_prompt_strength: params.regional_base_prompt_strength,
        regional_normalize_masks: params.regional_normalize_masks,
        use_rebalance: params.use_rebalance,
        rebalance_multiplier: params.rebalance_multiplier,
        rebalance_weights: params.rebalance_weights,
        rebalance_mode: params.rebalance_mode,
        rebalance_preset: params.rebalance_preset,
        rebalance_renormalize: params.rebalance_renormalize,
        edit_rebalance_enabled: params.edit_rebalance_enabled,
        edit_rebalance_profile: params.edit_rebalance_profile,
        conditioning_mode: params.conditioning_mode,
        krea_enhancer_variant: params.krea_enhancer_variant,
        krea_enhancer_enabled: params.krea_enhancer_enabled,
        krea_enhancer_strength: params.krea_enhancer_strength,
        krea_enhancer_delta_cap: params.krea_enhancer_delta_cap,
        loras: params.loras,
        bboxes: params.bboxes,
        init_image_b64: params.init_image_b64 || undefined,
        mask_b64: params.mask_b64 || undefined,
        ref_image1_b64: params.ref_image1_b64 || undefined,
        ref_image2_b64: params.ref_image2_b64 || undefined,
        ref_image3_b64: params.ref_image3_b64 || undefined,
        use_prompt_planner: params.use_prompt_planner,
        prompt_planner_max_tokens: params.prompt_planner_max_tokens,
        prompt_planner_show_output: params.prompt_planner_show_output,
        prompt_planner_lock_original: params.prompt_planner_lock_original,
        prompt_planner_use_regions: params.prompt_planner_use_regions,
        use_prompt_expander: params.use_prompt_expander,
        think_steering_enabled: params.think_steering_enabled,
        think_text: params.think_text || undefined,
        refine: params.refine,
        refine_denoise: params.refine_denoise,
        refine_steps: params.refine_steps,
        vae_degrid: params.vae_degrid,
        mood: params.mood,
        moodboard_ids: params.selected_moodboard_ids,
        moodboard_uuids: params.moodboard_uuids,
        moodboard_strength: params.moodboard_strength,
        moodboard_images: params.moodboard_images,
        seed_variance_preset: params.seed_variance_preset,
        seed_variance_strength: params.seed_variance_strength,
        seed_variance_algorithm: params.seed_variance_algorithm,
        seed_variance_model_type: params.seed_variance_model_type,
        seed_variance_randomize_percent: params.seed_variance_randomize_percent,
        seed_variance_shift_strength: params.seed_variance_shift_strength,
        seed_variance_protection: params.seed_variance_protection,
        seed_variance_direction: params.seed_variance_direction,
        seed_variance_fade_curve: params.seed_variance_fade_curve,
        seed_variance_injection_start: params.seed_variance_injection_start,
        seed_variance_injection_end: params.seed_variance_injection_end,
        depth_control: params.depth_control,
        depth_control_strength: params.depth_control_strength,
        depth_estimator: params.depth_estimator,
        depth_resolution: params.depth_resolution,
        depth_invert: params.depth_invert,
        god_mode: params.god_mode,
        mrflow: params.mrflow,
        mrflow_upscaler: params.mrflow_upscaler,
        mrflow_preset: params.mrflow_preset,
        mrflow_refine_steps: params.mrflow_refine_steps,
        mrflow_refine_denoise: params.mrflow_refine_denoise,
        seed_variance_schedule: params.seed_variance_schedule,
        seed_variance_cutoff_step: params.seed_variance_cutoff_step,
        seed_variance_total_steps: params.seed_variance_total_steps,
        seed_variance_cutoff_strength: params.seed_variance_cutoff_strength,
      })
      setJobId(job_id)
      setQueue(queue_position ?? null, queue_length ?? null)
      if (status === 'blocked') {
        setGenerating(false)
        setError('This prompt was blocked by the child safety filter and sent to an admin for review.')
        return
      }

      watchJob(job_id)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e.message ?? 'Request failed')
      setGenerating(false)
      setQueue(null, null)
      stopWatchingJob()
    }
  }, [params, generating, promptBusy, setQueue, setJobId, setGenerating, setProgress, setResults, setError, watchJob, stopWatchingJob])

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
              <Button color="inherit" size="small" onClick={() => setTab(TAB.SYSTEM)}>
                Load model
              </Button>
            }
          >
            No model loaded — go to System tab to load a checkpoint before generating.
          </Alert>
        )}
        {activeEngine && !!activeEngine.unsupported_controls?.length && (
          <Alert severity="warning" sx={{ py: 0.75 }}>
            {`${activeEngine.label}: unsupported Krea-native controls are hidden or ignored: ${activeEngine.unsupported_controls.join(', ')}.`}
          </Alert>
        )}

        <QuickPresets />
        <MrFlowSettings />
        <PromptSection />

        {/* Model & engine (GGUF / INT8 / VAE) — collapsed; the quick recipes set
            these, so it's here for manual overrides but out of the way by default. */}
        <Accordion disableGutters sx={{ bgcolor: 'transparent', border: '1px solid rgba(202,196,208,0.18)', borderRadius: 2, '&:before': { display: 'none' } }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } }}>
            <Typography variant="body2" sx={{ color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 1 }}>
              Model &amp; engine (GGUF / INT8)
            </Typography>
          </AccordionSummary>
          <AccordionDetails><ModelSection /></AccordionDetails>
        </Accordion>

        {(supports?.supports_moodboards ?? true) && <MoodboardSection />}

        {/* Image Prompt — reference images fed as Qwen3-VL vision tokens (up to 4). */}
        {(supports?.supports_style_references ?? true) && <StyleReferenceSection />}

        {/* LoRAs — collapsed by default (moodboards are this workflow's strength). */}
        {(supports?.supports_lora ?? true) && (
          <Accordion disableGutters sx={{ bgcolor: 'transparent', border: '1px solid rgba(202,196,208,0.18)', borderRadius: 2, '&:before': { display: 'none' } }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } }}>
              <Typography variant="body2" sx={{ color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 1 }}>
                LoRAs
              </Typography>
            </AccordionSummary>
            <AccordionDetails><LoraSection /></AccordionDetails>
          </Accordion>
        )}

        {/* All generation settings in one place before Generate: steps & CFG
            (ParameterSection) then resolution / aspect ratio (DimensionSection). */}
        <ParameterSection />
        <DimensionSection />

        {/* Advanced settings: style references, scene regions, spatial canvas,
            and saved recipes are tucked away here to keep the main panel clean. */}
        {!inRedrawStudio && (
          <Accordion disableGutters sx={{ bgcolor: 'transparent', '&:before': { display: 'none' } }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="body2" sx={{ color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 1 }}>
                Advanced settings — style, scene regions, canvas & recipes
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Stack spacing={2}>
                {(supports?.supports_regional_prompts ?? true) && <AdvancedSceneSection />}
                {(supports?.supports_regional_prompts ?? true) && <CanvasControl />}
                <RecipeSection />
              </Stack>
            </AccordionDetails>
          </Accordion>
        )}

        {generationError && <Alert severity="error" onClose={() => setError(null)}>{generationError}</Alert>}
        {promptBusy && <Alert severity="info" sx={{ py: 0 }}>Magic Wand is still rewriting the prompt. Generate will be available when it finishes.</Alert>}
        {connectionNote && <Alert severity="info" sx={{ py: 0 }}>{connectionNote}</Alert>}
        {inRedrawStudio && (
          <Alert severity={(params.init_image_b64 || params.moodboard_images.length) ? 'success' : 'warning'} sx={{ py: 0 }}>
            {(params.init_image_b64 || params.moodboard_images.length)
              ? `Redraw/img2img is prepared: ${params.init_image_b64 ? 'source image attached' : 'reference images attached'}, denoise ${params.denoise.toFixed(2)}.`
              : 'Redraw/img2img needs a prepared source/reference image from Redraw Studio before Generate will do useful image-to-image work.'}
          </Alert>
        )}

        {generating && (
          <Box>
            <LinearProgress variant="determinate" value={progress} />
            <Typography variant="caption" sx={{ color: 'text.secondary', mt: 0.5, display: 'block' }}>
              {params.num_images > 1 && params.batch_mode === 'safe_queue'
                ? `Batch queued — ${results.length}/${params.num_images} complete${queuePosition ? ` · next queue position ${queuePosition}${queueLength ? ` of ${queueLength}` : ''}` : ''}`
                : queuePosition ? `Queued — position ${queuePosition}${queueLength ? ` of ${queueLength}` : ''}` : `${progress}% complete`}
            </Typography>
          </Box>
        )}

        <Button
          variant="contained"
          size="large"
          startIcon={generating ? <CircularProgress size={18} color="inherit" /> : <AutoAwesomeIcon />}
          onClick={handleGenerate}
          disabled={generating || promptBusy || !params.prompt.trim() || !modelLoaded}
          fullWidth
          sx={{ height: 52, fontSize: '1rem' }}
        >
          {generating ? 'Generating…' : promptBusy ? 'Waiting for Magic Wand…' : 'Generate'}
        </Button>

        <ResultsView images={results} seed={lastSeed} metadata={resultsMetadata} />

        <GenerationQueue />
      </Stack>
    </Box>
  )
}
