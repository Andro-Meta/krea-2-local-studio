import React, { useEffect, useState } from 'react'
import {
  Accordion, AccordionDetails, AccordionSummary,
  Alert, Box, Button, Chip, FormControlLabel, Grid, MenuItem, Slider, Stack, Switch, TextField, ToggleButton, ToggleButtonGroup, Tooltip, Typography,
} from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import { useStore } from '../../store'
import { apiFetch, type BatchPlan } from '../../api'
import { INT8_VARIANTS } from './QuickPresets'

const INT8_VARIANT_COUNT = INT8_VARIANTS.length

type SamplerCatalog = Awaited<ReturnType<typeof apiFetch.samplerCatalog>>

function InfoTip({ text }: { text: string }) {
  return (
    <Tooltip title={text} placement="right" arrow>
      <InfoOutlinedIcon sx={{ fontSize: 14, color: 'text.disabled', ml: 0.5, cursor: 'help', verticalAlign: 'middle' }} />
    </Tooltip>
  )
}

function LabeledSlider({ label, value, min, max, step, onChange, tip, helperText, disabled }: {
  label: string; value: number; min: number; max: number; step: number
  onChange: (v: number) => void; tip?: string; helperText?: string; disabled?: boolean
}) {
  return (
    <Box sx={{ width: '100%', opacity: disabled ? 0.45 : 1 }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center">
        <Typography variant="body2" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center' }}>
          {label}{tip && <InfoTip text={tip} />}
        </Typography>
        <Typography variant="body2" sx={{ fontFamily: 'Roboto Mono', fontSize: 12, color: 'text.primary' }}>
          {value}
        </Typography>
      </Stack>
      <Slider
        value={value} min={min} max={max} step={step}
        onChange={(_, v) => onChange(v as number)}
        size="small"
        disabled={disabled}
        sx={{ mt: 0.5 }}
      />
      {helperText && (
        <Typography variant="caption" sx={{ color: 'text.disabled', mt: -0.5, display: 'block' }}>
          {helperText}
        </Typography>
      )}
    </Box>
  )
}

export default function ParameterSection() {
  const { params, setParam, setParams } = useStore()
  const [advOpen, setAdvOpen] = useState(false)
  const isTurbo = params.checkpoint === 'turbo'
  // Mr. Flow and God Mode drive steps/CFG/sampler from their own presets, so the
  // normal controls are inert under them — grey them out to avoid the conflict.
  const pipelineOverridesSampling = !!params.mrflow || !!params.god_mode
  // Very large custom/preset outputs are too heavy to batch on 24GB; force single image.
  const isLargeOutput = Math.max(params.width, params.height) >= 2560
  const [catalog, setCatalog] = useState<SamplerCatalog | null>(null)
  const [batchPlan, setBatchPlan] = useState<BatchPlan | null>(null)
  useEffect(() => {
    if (isLargeOutput && params.num_images !== 1) setParam('num_images', 1)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLargeOutput])
  const rbgExpressionDefaultsActive =
    params.seed_variance_preset === 'creative' &&
    params.seed_variance_algorithm === 'rbg' &&
    params.seed_variance_model_type === 'krea2' &&
    params.seed_variance_direction === 'visceral_expression_grit' &&
    params.seed_variance_shift_strength === 170 &&
    params.seed_variance_protection === 'none' &&
    params.seed_variance_fade_curve === 'smoothstep' &&
    params.seed_variance_schedule === 'step_cutoff' &&
    params.seed_variance_cutoff_step === 3 &&
    params.seed_variance_total_steps === 13 &&
    params.seed_variance_cutoff_strength === 0.53
  const toggleRbgExpressionDefaults = () => {
    if (rbgExpressionDefaultsActive) {
      setParam('seed_variance_preset', 'off')
      return
    }
    setParams({
      seed_variance_algorithm: 'rbg',
      seed_variance_preset: 'creative',
      seed_variance_model_type: 'krea2',
      seed_variance_direction: 'visceral_expression_grit',
      seed_variance_shift_strength: 170,
      seed_variance_protection: 'none',
      seed_variance_fade_curve: 'smoothstep',
      seed_variance_schedule: 'step_cutoff',
      seed_variance_cutoff_step: 3,
      seed_variance_total_steps: 13,
      seed_variance_cutoff_strength: 0.53,
    })
  }

  useEffect(() => {
    const profile = isTurbo ? 'krea_turbo' : 'krea_raw'
    let alive = true
    apiFetch.samplerCatalog(profile).then(c => { if (alive) setCatalog(c) }).catch(() => {})
    return () => { alive = false }
  }, [isTurbo])

  useEffect(() => {
    let alive = true
    apiFetch.batchPlan({
      width: params.width,
      height: params.height,
      quantization: params.quantization,
      batch: params.num_images,
      cfg: params.cfg,
      mode: params.mode,
      checkpoint: params.checkpoint,
    }).then(plan => {
      if (!alive) return
      setBatchPlan(plan)
    }).catch(() => { if (alive) setBatchPlan(null) })
    return () => { alive = false }
  }, [params.width, params.height, params.quantization, params.num_images, params.cfg, params.mode, params.checkpoint, params.batch_mode])

  const currentSampler = catalog?.samplers.find(s => s.id === params.sampler || (params.sampler === 'euler' && s.id === 'euler'))
  const supportedSchedulers = currentSampler?.supported_schedulers ?? ['simple', 'normal', 'beta', 'sgm_uniform']
  const recommendedSteps = currentSampler?.recommended_steps

  const applyCombo = (c: { sampler: string; scheduler: string; steps: number; cfg: number; cfg_zero_star?: boolean }) => {
    setParams({
      sampler: c.sampler as typeof params.sampler,
      scheduler: c.scheduler as typeof params.scheduler,
      steps: c.steps,
      cfg: c.cfg,
      // Presets carrying the CFG-Zero* flag toggle it on; others turn it off so
      // switching presets is predictable.
      cfg_zero_star: !!c.cfg_zero_star,
    })
  }
  const setInpaintMethod = (method: typeof params.inpaint_method) => {
    setParams({
      inpaint_method: method,
      ...(method === 'lanpaint_experimental'
        ? {
            steps: Math.max(params.steps, 20),
            denoise: 1.0,
            sampler: 'euler',
            scheduler: 'simple',
            lanpaint_inner_steps: 5,
            lanpaint_lambda: 16,
            lanpaint_step_size: 0.2,
            lanpaint_beta: 1,
            lanpaint_friction: 15,
            lanpaint_early_stop: 1,
            lanpaint_prompt_mode: 'Image First',
          }
        : {}),
    })
  }
  const setCreativity = (creativity: typeof params.creativity) => {
    const values = {
      raw: { moodboard_strength: 0.2, rebalance_multiplier: 0.8 },
      low: { moodboard_strength: 0.3, rebalance_multiplier: 0.9 },
      medium: { moodboard_strength: 0.35, rebalance_multiplier: 1.0 },
      high: { moodboard_strength: 0.55, rebalance_multiplier: 1.15 },
    }[creativity]
    setParams({ creativity, ...values })
  }
  const setBatchMode = (batch_mode: typeof params.batch_mode) => {
    setParams({
      batch_mode,
      parallel_batch_confirmed: batch_mode === 'parallel' && Boolean(batchPlan?.allowed),
    })
  }

  return (
    <Box>
      <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1.5, display: 'block', textTransform: 'uppercase', letterSpacing: 1 }}>
        Parameters
      </Typography>
      <Stack spacing={2}>
        {pipelineOverridesSampling ? (
          <Alert severity="info" sx={{ py: 0 }}>
            {params.god_mode ? 'God Mode' : 'Mr. Flow'} sets steps, CFG, and sampler automatically
            {params.mrflow ? ' — tune them in the Mr. Flow settings card above.' : '.'}
          </Alert>
        ) : (
          <>
            <LabeledSlider
              label="Steps"
              value={params.steps}
              min={1} max={60} step={1}
              onChange={v => setParam('steps', v)}
              tip={isTurbo
                ? 'Turbo: 8 steps is optimal. More steps add compute with minimal quality gain.'
                : 'RAW: 52 steps is optimal for maximum quality.'}
              helperText={isTurbo ? 'Turbo default: 8' : 'RAW default: 52'}
            />
            <LabeledSlider
              label="CFG Scale"
              value={params.cfg}
              min={0} max={10} step={0.1}
              onChange={v => setParam('cfg', v)}
              tip="Classifier-Free Guidance adds an unconditional/negative pass. Turbo still uses your prompt at CFG 0; it is distilled to follow the conditional prompt without extra CFG. RAW uses real CFG."
              helperText={isTurbo ? 'Turbo default: 0 does not ignore the prompt; use 1+ only for experiments' : 'RAW default: 3.5'}
            />
          </>
        )}

        {params.mode !== 'txt2img' && (
          <LabeledSlider
            label="Denoise strength"
            value={params.denoise}
            min={0.01} max={1.0} step={0.01}
            onChange={v => setParam('denoise', v)}
            tip={params.mode === 'outpaint'
              ? 'How strongly Krea may redraw the expanded area. 1.0 gives the new area full freedom; the differential mask controls the blend into the source.'
              : 'How much to change the input image. 1.0 = ignore original (same as txt2img). 0.5 = half old / half new. 0.3 = subtle edits only.'}
            helperText={params.mode === 'outpaint'
              ? 'Outpaint default: 1.0 · lower = preserve more init canvas · Creative redraw ignores this'
              : '0.75 = balanced · 0.3–0.5 = preserve original · 1.0 = full regen'}
          />
        )}

        <Grid container spacing={1.5}>
          <Grid item xs={6}>
            <TextField
              label="Seed"
              type="number"
              size="small"
              fullWidth
              value={params.seed}
              onChange={e => setParam('seed', Number(e.target.value))}
              helperText="-1 = random each run"
            />
          </Grid>
          <Grid item xs={6}>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
              Batch{isLargeOutput ? '' : ' (1–4)'}
            </Typography>
            <ToggleButtonGroup
              exclusive
              size="small"
              fullWidth
              value={isLargeOutput ? 1 : params.num_images}
              disabled={isLargeOutput}
              onChange={(_, v) => { if (v != null) setParam('num_images', v as number) }}
              sx={{
                '& .MuiToggleButton-root': {
                  minHeight: 40,
                  flex: 1,
                  py: 0.75,
                  fontFamily: 'Roboto Mono, monospace',
                  fontSize: 14,
                },
              }}
            >
              {[1, 2, 3, 4].map(n => (
                <ToggleButton key={n} value={n} disabled={isLargeOutput && n !== 1}>
                  {n}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
            <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: 0.5 }}>
              {isLargeOutput ? 'Large outputs are single-image only' : 'Tap a count — no typing needed'}
            </Typography>
          </Grid>
        </Grid>

        {isTurbo && params.quantization === 'int8' && (
          <Box>
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={params.batch_int8_all}
                  onChange={e => setParam('batch_int8_all', e.target.checked)}
                />
              }
              label={
                <Typography variant="body2">
                  Batch all Turbo INT8 models
                </Typography>
              }
            />
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', ml: 4, mt: -0.5 }}>
              {params.batch_int8_all
                ? `Renders your batch with each of the ${INT8_VARIANT_COUNT} INT8 checkpoints — ${(isLargeOutput ? 1 : params.num_images) * INT8_VARIANT_COUNT} images total (${isLargeOutput ? 1 : params.num_images} per model, same seed), safe-queued for comparison.`
                : `Sweep the current settings across all ${INT8_VARIANT_COUNT} Turbo INT8 checkpoints (batch × ${INT8_VARIANT_COUNT} images).`}
            </Typography>
          </Box>
        )}

        {params.num_images > 1 && !params.batch_int8_all && (
          <TextField
            select
            label="Batch mode"
            value={params.batch_mode}
            onChange={e => setBatchMode(e.target.value as typeof params.batch_mode)}
            size="small"
            fullWidth
            helperText={
              params.batch_mode === 'parallel'
                ? (batchPlan?.allowed
                    ? 'Parallel batch likely fits. Experimental: verify outputs before relying on it.'
                    : `Parallel batch risky; safe queue will be used. ${(batchPlan?.blocked_reasons ?? []).join(' ')}`)
                : 'Safe queue is recommended: creates one FIFO job per image to avoid VRAM spikes.'
            }
          >
            <MenuItem value="safe_queue">Safe queue (recommended)</MenuItem>
            <MenuItem value="parallel" disabled={Boolean(batchPlan && !batchPlan.allowed)}>Parallel (experimental)</MenuItem>
          </TextField>
        )}

        <FormControlLabel
          control={<Switch checked={params.vae_degrid} onChange={e => setParam('vae_degrid', e.target.checked)} size="small" />}
          label={
            <Typography variant="body2" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center' }}>
              VAE DeGrid
              <InfoTip text="Removes the faint 2px pixel grid left by the Qwen/Wan VAE after decode. On by default — turn off only to A/B compare or if you prefer the raw decode." />
            </Typography>
          }
        />

        {/* Detail refine runs a second self-pass; the backend skips it for
            inpaint/outpaint (must not re-touch kept pixels), so hide it there. */}
        {params.mode !== 'inpaint' && params.mode !== 'outpaint' && (
          <Box>
            <FormControlLabel
              control={<Switch checked={params.refine} onChange={e => setParam('refine', e.target.checked)} size="small" />}
              label={
                <Typography variant="body2" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center' }}>
                  Detail refine pass
                  <InfoTip text="Runs a second low-denoise Krea-2 pass over the result to sharpen fine detail. Adds roughly one extra generation of time. (Krea-2 self-refine — not a separate refiner model.)" />
                </Typography>
              }
            />
            {params.refine && (
              <>
                <LabeledSlider
                  label="Refine denoise"
                  value={params.refine_denoise}
                  min={0.1} max={0.6} step={0.05}
                  onChange={v => setParam('refine_denoise', v)}
                  tip="How much the refine pass may change the image. 0.3 = balanced detail; lower = subtler."
                  helperText="0.3 = balanced · 0.1–0.2 = subtle sharpen · 0.5+ = stronger rework"
                />
                <LabeledSlider
                  label="Refine steps"
                  value={params.refine_steps}
                  min={2} max={20} step={1}
                  onChange={v => setParam('refine_steps', v)}
                  tip="Number of sampling steps for the refine pass. More = slower but finer."
                  helperText="6 = balanced · fewer = faster"
                />
              </>
            )}
          </Box>
        )}

        {!pipelineOverridesSampling && (
        <Accordion expanded={advOpen} onChange={(_, v) => setAdvOpen(v)} disableGutters>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Advanced — timestep schedule &amp; conditioning
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={2}>
              <TextField
                select
                label="Creativity"
                value={params.creativity}
                onChange={e => setCreativity(e.target.value as typeof params.creativity)}
                size="small"
                fullWidth
                helperText="Native Krea control: higher adds aesthetic interpretation; lower keeps tighter prompt adherence."
              >
                <MenuItem value="raw">Raw / literal</MenuItem>
                <MenuItem value="low">Low</MenuItem>
                <MenuItem value="medium">Medium (default)</MenuItem>
                <MenuItem value="high">High</MenuItem>
              </TextField>
              {params.cfg > 0 && (
                <Box>
                  <FormControlLabel
                    control={<Switch size="small" checked={params.cfg_zero_star} onChange={e => setParam('cfg_zero_star', e.target.checked)} />}
                    label={
                      <Typography variant="body2" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center' }}>
                        CFG-Zero*
                        <InfoTip text="Flow-matching guidance upgrade (arXiv:2503.18886). Optimized-scale corrects velocity error and zero-init skips the unreliable first step(s). Improves color/detail and prompt alignment at real CFG. Only applies when CFG > 0; not combined with the CFG++ samplers." />
                      </Typography>
                    }
                  />
                  {params.cfg_zero_star && (
                    <LabeledSlider
                      label="Zero-init steps"
                      value={params.cfg_zero_init_steps}
                      min={0} max={4} step={1}
                      onChange={v => setParam('cfg_zero_init_steps', v)}
                      helperText="Skip the first N ODE steps (paper default 1). 0 = optimized-scale only."
                    />
                  )}
                </Box>
              )}
              {catalog && catalog.recommended_combos.length > 0 && (
                <Box>
                  <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                    Community presets {isTurbo ? '(Turbo)' : '(RAW/base)'}
                  </Typography>
                  <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap">
                    {catalog.recommended_combos.map(c => (
                      <Tooltip key={c.label} title={`${c.note} — ${c.sampler}/${c.scheduler}, ${c.steps} steps, CFG ${c.cfg}`} arrow>
                        <Chip
                          label={c.label}
                          size="small"
                          variant={params.sampler === c.sampler && params.scheduler === c.scheduler ? 'filled' : 'outlined'}
                          color={params.sampler === c.sampler && params.scheduler === c.scheduler ? 'primary' : 'default'}
                          onClick={() => applyCombo(c)}
                          sx={{ cursor: 'pointer' }}
                        />
                      </Tooltip>
                    ))}
                  </Stack>
                </Box>
              )}
              <TextField
                select
                label="Sampler"
                value={params.sampler}
                onChange={e => {
                  const next = e.target.value as typeof params.sampler
                  const spec = catalog?.samplers.find(s => s.id === next)
                  // Keep the scheduler valid for the new sampler.
                  const sched = spec && !spec.supported_schedulers.includes(params.scheduler)
                    ? (spec.scheduler as typeof params.scheduler)
                    : params.scheduler
                  setParams({ sampler: next, scheduler: sched })
                }}
                size="small"
                fullWidth
                helperText={currentSampler?.note || 'Flow-matching samplers for the Krea profiles.'}
              >
                {(catalog?.samplers ?? []).filter(s => !s.disabled).map(s => (
                  <MenuItem key={s.id} value={s.id}>
                    {s.label} ({s.recommended_steps} steps)
                  </MenuItem>
                ))}
                {!catalog && <MenuItem value="euler">Euler / Simple (Krea default)</MenuItem>}
              </TextField>
              <TextField
                select
                label="Scheduler"
                value={params.scheduler}
                onChange={e => setParam('scheduler', e.target.value as typeof params.scheduler)}
                size="small"
                fullWidth
                helperText={
                  (catalog?.schedulers.find(s => s.id === params.scheduler)?.note)
                  || 'Reshapes where steps are spent on the flow trajectory. Beta = crisper detail.'
                }
              >
                {(catalog?.schedulers ?? [{ id: 'simple', label: 'Simple (Krea flow)', recommended: true, note: '' }])
                  .filter(s => supportedSchedulers.includes(s.id))
                  .map(s => (
                    <MenuItem key={s.id} value={s.id}>
                      {s.label}{s.recommended ? '' : ' — experimental'}
                    </MenuItem>
                  ))}
              </TextField>
              {recommendedSteps != null && params.steps !== recommendedSteps && (
                <Typography variant="caption" sx={{ color: 'warning.main', mt: -1 }}>
                  Tip: {currentSampler?.label} usually looks best around {recommendedSteps} steps (currently {params.steps}).
                </Typography>
              )}
              {(params.mode === 'inpaint' || params.mode === 'outpaint') && (
                <TextField
                  select
                  label="Inpaint / outpaint method"
                  value={params.inpaint_method}
                  onChange={e => setInpaintMethod(e.target.value as typeof params.inpaint_method)}
                  size="small"
                  fullWidth
                  helperText="Native Krea is the default. LanPaint is experimental and currently inpaint-only."
                >
                  <MenuItem value="native">Native Krea masked sampler</MenuItem>
                  {params.mode === 'inpaint' && <MenuItem value="lanpaint_experimental">LanPaint experimental (inpaint)</MenuItem>}
                </TextField>
              )}
              {params.mode === 'inpaint' && params.inpaint_method === 'native' && (
                <Box>
                  <FormControlLabel
                    control={
                      <Switch
                        size="small"
                        checked={params.differential_inpaint}
                        onChange={e => setParam('differential_inpaint', e.target.checked)}
                      />
                    }
                    label={<>Differential diffusion (soft masks)<InfoTip text="Grayscale mask values join the denoise at different timesteps, so feathered/soft edits blend seamlessly into the kept region. Paint your mask with soft brush edges for the best result." /></>}
                  />
                  {params.differential_inpaint && (
                    <LabeledSlider
                      label="Differential blend strength"
                      value={params.differential_strength}
                      min={0} max={1} step={0.05}
                      onChange={v => setParam('differential_strength', v)}
                      helperText="1.0 = full per-step threshold (sharpest transition). Lower keeps more of the raw soft mask each step (gentler blend)."
                    />
                  )}
                </Box>
              )}
              {params.mode === 'inpaint' && params.inpaint_method === 'lanpaint_experimental' && (
                <>
                  <LabeledSlider
                    label="LanPaint think steps"
                    value={params.lanpaint_inner_steps}
                    min={0} max={20} step={1}
                    onChange={v => setParam('lanpaint_inner_steps', v)}
                    tip="Extra masked-region model iterations per denoise step. Higher can improve difficult fills but increases generation time."
                    helperText="LanPaint default: 5 · easy: 2–5 · hard: 5–10"
                  />
                  <LabeledSlider
                    label="LanPaint strength"
                    value={params.lanpaint_strength}
                    min={0.1} max={2} step={0.05}
                    onChange={v => setParam('lanpaint_strength', v)}
                    tip="Scales the masked inner update. Lower is safer, higher is more aggressive."
                    helperText="Experimental · start with 1.0"
                  />
                  <LabeledSlider
                    label="LanPaint lambda"
                    value={params.lanpaint_lambda}
                    min={0.1} max={50} step={0.1}
                    onChange={v => setParam('lanpaint_lambda', v)}
                    tip="Content alignment strength. Higher can preserve context better but may become unstable."
                    helperText="Upstream default: 16"
                  />
                  <LabeledSlider
                    label="LanPaint step size"
                    value={params.lanpaint_step_size}
                    min={0.01} max={1} step={0.01}
                    onChange={v => setParam('lanpaint_step_size', v)}
                    tip="Langevin thinking step size. Lower is safer; higher converges faster."
                    helperText="Recommended: 0.1–0.5 · default: 0.2"
                  />
                  <LabeledSlider
                    label="LanPaint beta"
                    value={params.lanpaint_beta}
                    min={0.1} max={5} step={0.1}
                    onChange={v => setParam('lanpaint_beta', v)}
                    tip="Masked/unmasked step ratio. Lower can stabilize high lambda values."
                    helperText="Default: 1.0"
                  />
                  <LabeledSlider
                    label="LanPaint friction"
                    value={params.lanpaint_friction}
                    min={0} max={50} step={0.5}
                    onChange={v => setParam('lanpaint_friction', v)}
                    tip="Stabilizes Langevin updates. Higher is slower but safer."
                    helperText="Recommended: 10–20 · default: 15"
                  />
                  <LabeledSlider
                    label="LanPaint early stop"
                    value={params.lanpaint_early_stop}
                    min={0} max={10} step={1}
                    onChange={v => setParam('lanpaint_early_stop', v)}
                    tip="Stops LanPaint thinking before final sampling steps to reduce late artifacts."
                    helperText="Recommended: 1–5 · default: 1"
                  />
                  <TextField
                    select
                    label="LanPaint prompt mode"
                    value={params.lanpaint_prompt_mode}
                    onChange={e => setParam('lanpaint_prompt_mode', e.target.value as typeof params.lanpaint_prompt_mode)}
                    size="small"
                    fullWidth
                    helperText="Image First favors local context. Prompt First is stronger but may reduce quality."
                  >
                    <MenuItem value="Image First">Image First</MenuItem>
                    <MenuItem value="Prompt First">Prompt First</MenuItem>
                  </TextField>
                </>
              )}
              {/* μ/y1/y2 only matter when μ is auto-derived from resolution
                  (RAW). Turbo pins μ=1.15, so these are hidden there. */}
              {isTurbo ? (
                <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                  μ flow-shift is pinned to 1.15 for Turbo (frozen at 1024). Switch to RAW to tune μ / y1 / y2.
                </Typography>
              ) : (
                <>
                  <LabeledSlider
                    label="μ — flow shift (ModelSamplingFlux)"
                    value={params.mu ?? 0}
                    min={0} max={2.0} step={0.05}
                    onChange={v => setParam('mu', v <= 0 ? null : v)}
                    tip="ModelSamplingFlux shift: shifts timestep density toward high-noise steps. Higher = better for large images (>1024px). Set 0 to auto-calculate from resolution."
                    helperText="0 = auto (resolution-adaptive) · higher = better for large images"
                  />
                  <LabeledSlider
                    label="y1 (schedule lower bound)"
                    value={params.y1}
                    min={0.1} max={1.0} step={0.05}
                    onChange={v => setParam('y1', v)}
                    tip="Lower bound of the auto-μ interpolation. Only used when μ = 0 (auto). Default: 0.5"
                  />
                  <LabeledSlider
                    label="y2 (schedule upper bound)"
                    value={params.y2}
                    min={1.0} max={2.0} step={0.05}
                    onChange={v => setParam('y2', v)}
                    tip="Upper bound of the auto-μ interpolation. Only used when μ = 0 (auto). Default: 1.15"
                  />
                </>
              )}
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center', mt: 0.5 }}>
                Conditioning Rebalance
                <InfoTip text="Artifact-safe presets rebalance the 12 Qwen3-VL encoder layer taps while preserving overall conditioning magnitude. Legacy keeps the old multiply behavior for reproducible comparisons." />
              </Typography>
              <Grid container spacing={1.5}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    select
                    label="Conditioning preset"
                    value={params.rebalance_preset}
                    onChange={e => {
                      const preset = e.target.value as typeof params.rebalance_preset
                      setParams({
                        rebalance_preset: preset,
                        rebalance_mode: preset === 'legacy' ? 'legacy_multiply' : 'rms_renorm',
                        rebalance_renormalize: preset !== 'legacy',
                        rebalance_multiplier: preset === 'legacy' ? 4.0 : 1.0,
                      })
                    }}
                    size="small"
                    fullWidth
                    helperText="Default: Balanced, artifact-safe"
                  >
                    <MenuItem value="balanced">Balanced, artifact-safe</MenuItem>
                    <MenuItem value="subtle">Subtle</MenuItem>
                    <MenuItem value="detail">Detail</MenuItem>
                    <MenuItem value="emotion">Emotion (restore expression)</MenuItem>
                    <MenuItem value="uniform">Uniform</MenuItem>
                    <MenuItem value="legacy">Legacy multiply</MenuItem>
                    <MenuItem value="custom">Custom</MenuItem>
                  </TextField>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    select
                    label="Rebalance mode"
                    value={params.rebalance_mode}
                    onChange={e => setParam('rebalance_mode', e.target.value as typeof params.rebalance_mode)}
                    size="small"
                    fullWidth
                    helperText="RMS renorm keeps prompt strength stable"
                  >
                    <MenuItem value="rms_renorm">RMS renormalized</MenuItem>
                    <MenuItem value="legacy_multiply">Legacy multiply</MenuItem>
                  </TextField>
                </Grid>
              </Grid>
              {(params.rebalance_preset === 'custom' || params.rebalance_preset === 'legacy' || params.rebalance_mode === 'legacy_multiply') && (
                <>
                  <LabeledSlider
                    label="Global multiplier"
                    value={params.rebalance_multiplier}
                    min={0.5} max={10} step={0.1}
                    onChange={v => setParam('rebalance_multiplier', v)}
                    tip="Scales all 12 conditioning taps together. Balanced default is 1.0; legacy 4.0 matches the older Studio behavior."
                  />
                  <TextField
                    label="Per-layer weights (12 comma-separated values)"
                    value={params.rebalance_weights}
                    onChange={e => setParam('rebalance_weights', e.target.value)}
                    size="small"
                    fullWidth
                    helperText="Layers 1–12 of Qwen3-VL. Default: 1,1,1,1,1,1,1,2.5,5,1.1,4,1"
                  />
                </>
              )}
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center', mt: 0.5 }}>
                Seed Variance
                <InfoTip text="Adds deterministic, bounded noise to unprotected conditioning tokens. Off bypasses the feature. RBG mode ports Smart Seed Variance-style sparse conditioning noise for stronger expression and identity variation." />
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 0.5 }}>
                <Button
                  size="small"
                  variant={rbgExpressionDefaultsActive ? 'contained' : 'outlined'}
                  color={rbgExpressionDefaultsActive ? 'secondary' : 'primary'}
                  onClick={toggleRbgExpressionDefaults}
                >
                  {rbgExpressionDefaultsActive ? 'RBG Expression: On' : 'RBG Expression: Off'}
                </Button>
              </Stack>
              <Grid container spacing={1.5}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    select
                    label="Seed variance preset"
                    value={params.seed_variance_preset}
                    onChange={e => setParam('seed_variance_preset', e.target.value as typeof params.seed_variance_preset)}
                    size="small"
                    fullWidth
                    helperText="Default: off"
                  >
                    <MenuItem value="off">Off</MenuItem>
                    <MenuItem value="subtle">Subtle</MenuItem>
                    <MenuItem value="balanced">Balanced</MenuItem>
                    <MenuItem value="creative">Creative</MenuItem>
                    <MenuItem value="bold">Bold</MenuItem>
                    <MenuItem value="wild">Wild</MenuItem>
                    <MenuItem value="custom">Custom</MenuItem>
                  </TextField>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    select
                    label="Variance algorithm"
                    value={params.seed_variance_algorithm}
                    onChange={e => setParam('seed_variance_algorithm', e.target.value as typeof params.seed_variance_algorithm)}
                    size="small"
                    fullWidth
                    disabled={params.seed_variance_preset === 'off'}
                    helperText="Legacy is dense; RBG is sparse Smart Seed Variance"
                  >
                    <MenuItem value="legacy">Legacy</MenuItem>
                    <MenuItem value="rbg">RBG sparse</MenuItem>
                  </TextField>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    select
                    label="Protected prompt tokens"
                    value={params.seed_variance_protection}
                    onChange={e => setParam('seed_variance_protection', e.target.value as typeof params.seed_variance_protection)}
                    size="small"
                    fullWidth
                    disabled={params.seed_variance_preset === 'off'}
                    helperText="Preserves prompt anchors"
                  >
                    <MenuItem value="first_half">First half</MenuItem>
                    <MenuItem value="first_quarter">First quarter</MenuItem>
                    <MenuItem value="last_half">Last half</MenuItem>
                    <MenuItem value="last_quarter">Last quarter</MenuItem>
                    <MenuItem value="none">None</MenuItem>
                  </TextField>
                </Grid>
              </Grid>
              {params.seed_variance_preset === 'custom' && (
                <LabeledSlider
                  label="Custom seed variance"
                  value={params.seed_variance_strength}
                  min={0} max={0.1} step={0.005}
                  onChange={v => setParam('seed_variance_strength', v)}
                  tip="Custom conditioning-noise strength. Keep low; 0.01–0.03 is subtle, 0.08+ is aggressive."
                  helperText="Higher = more variation with the same seed, lower prompt fidelity"
                />
              )}
              {params.seed_variance_preset !== 'off' && (
                <>
                  <Grid container spacing={1.5}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        select
                        label="Variance direction"
                        value={params.seed_variance_direction}
                        onChange={e => setParam('seed_variance_direction', e.target.value as typeof params.seed_variance_direction)}
                        size="small"
                        fullWidth
                        helperText="Default: none"
                      >
                        <MenuItem value="none">None</MenuItem>
                        <MenuItem value="forward">Forward</MenuItem>
                        <MenuItem value="reverse">Reverse</MenuItem>
                        <MenuItem value="center">Center weighted</MenuItem>
                        <MenuItem value="edges">Edge weighted</MenuItem>
                        <MenuItem value="realistic">RBG Realistic</MenuItem>
                        <MenuItem value="facevar">RBG Face variance</MenuItem>
                        <MenuItem value="visceral_expression_grit">RBG Visceral expression & grit</MenuItem>
                        <MenuItem value="identity_stretch">RBG Identity stretch</MenuItem>
                        <MenuItem value="cinematic_framing">RBG Cinematic framing</MenuItem>
                        <MenuItem value="texture_lift">RBG Texture lift</MenuItem>
                        <MenuItem value="diversity">RBG Diversity</MenuItem>
                        <MenuItem value="dynamic_pose">RBG Dynamic pose</MenuItem>
                      </TextField>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        select
                        label="Variance fade"
                        value={params.seed_variance_fade_curve}
                        onChange={e => setParam('seed_variance_fade_curve', e.target.value as typeof params.seed_variance_fade_curve)}
                        size="small"
                        fullWidth
                        helperText="Default: linear"
                      >
                        <MenuItem value="instant">Instant</MenuItem>
                        <MenuItem value="linear">Linear</MenuItem>
                        <MenuItem value="ease_in">Ease in</MenuItem>
                        <MenuItem value="ease_out">Ease out</MenuItem>
                        <MenuItem value="ease_in_out">Ease in/out</MenuItem>
                        <MenuItem value="smoothstep">Smoothstep</MenuItem>
                        <MenuItem value="burst">Burst</MenuItem>
                      </TextField>
                    </Grid>
                  </Grid>
                  {params.seed_variance_algorithm === 'rbg' && (
                    <>
                      <Grid container spacing={1.5}>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            select
                            label="RBG model type"
                            value={params.seed_variance_model_type}
                            onChange={e => setParam('seed_variance_model_type', e.target.value as typeof params.seed_variance_model_type)}
                            size="small"
                            fullWidth
                            helperText="Krea2 default"
                          >
                            <MenuItem value="krea2">Krea2</MenuItem>
                            <MenuItem value="qwen_image">Qwen Image</MenuItem>
                            <MenuItem value="z_image">Z-Image</MenuItem>
                            <MenuItem value="flux">Flux</MenuItem>
                            <MenuItem value="sdxl">SDXL</MenuItem>
                            <MenuItem value="other">Other</MenuItem>
                          </TextField>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            select
                            label="RBG variance schedule"
                            value={params.seed_variance_schedule}
                            onChange={e => setParam('seed_variance_schedule', e.target.value as typeof params.seed_variance_schedule)}
                            size="small"
                            fullWidth
                            helperText="Composition Lock: hard_lock keeps composition fixed, then varies details"
                          >
                            <MenuItem value="constant">Constant (standard)</MenuItem>
                            <MenuItem value="decreasing">Decreasing (fade out)</MenuItem>
                            <MenuItem value="step_cutoff">Step cutoff (block switch)</MenuItem>
                            <MenuItem value="hard_lock">Hard lock (composition lock)</MenuItem>
                            <MenuItem value="tiered_release">Tiered release (multi-phase)</MenuItem>
                          </TextField>
                        </Grid>
                      </Grid>
                      <LabeledSlider
                        label="RBG shift strength"
                        value={params.seed_variance_shift_strength}
                        min={0} max={200} step={1}
                        onChange={v => setParam('seed_variance_shift_strength', v)}
                        helperText="Screenshot recipe uses 170"
                      />
                      <LabeledSlider
                        label="RBG randomize percent"
                        value={params.seed_variance_randomize_percent}
                        min={0} max={10} step={0.1}
                        onChange={v => setParam('seed_variance_randomize_percent', v)}
                        helperText="0 uses preset; RBG presets usually modify 1–5% of values"
                      />
                      <Grid container spacing={1.5}>
                        <Grid item xs={12} sm={4}>
                          <TextField
                            label="Cutoff step"
                            type="number"
                            size="small"
                            fullWidth
                            value={params.seed_variance_cutoff_step}
                            onChange={e => setParam('seed_variance_cutoff_step', Math.max(0, Number(e.target.value) || 0))}
                          />
                        </Grid>
                        <Grid item xs={12} sm={4}>
                          <TextField
                            label="Total steps"
                            type="number"
                            size="small"
                            fullWidth
                            value={params.seed_variance_total_steps}
                            onChange={e => setParam('seed_variance_total_steps', Math.max(1, Number(e.target.value) || 1))}
                          />
                        </Grid>
                        <Grid item xs={12} sm={4}>
                          <TextField
                            label="Cutoff strength"
                            type="number"
                            size="small"
                            fullWidth
                            value={params.seed_variance_cutoff_strength}
                            onChange={e => setParam('seed_variance_cutoff_strength', Math.max(0, Math.min(1, Number(e.target.value) || 0)))}
                            inputProps={{ min: 0, max: 1, step: 0.01 }}
                          />
                        </Grid>
                      </Grid>
                    </>
                  )}
                  <LabeledSlider
                    label="Variance injection start"
                    value={params.seed_variance_injection_start}
                    min={0} max={1} step={0.05}
                    onChange={v => setParam('seed_variance_injection_start', Math.min(v, params.seed_variance_injection_end))}
                    helperText="Default: 0.00"
                  />
                  <LabeledSlider
                    label="Variance injection end"
                    value={params.seed_variance_injection_end}
                    min={0} max={1} step={0.05}
                    onChange={v => setParam('seed_variance_injection_end', Math.max(v, params.seed_variance_injection_start))}
                    helperText="Default: 1.00"
                  />
                </>
              )}
              {(params.mode === 'redraw' || params.mode === 'img2img' || params.mode === 'inpaint' || params.mode === 'outpaint') && (
                <>
                  <TextField
                    select
                    label="Qwen conditioning mode"
                    value={params.conditioning_mode}
                    onChange={e => setParam('conditioning_mode', e.target.value as typeof params.conditioning_mode)}
                    size="small"
                    fullWidth
                    helperText="Auto uses Qwen Image Edit Plus for edit modes with references, and the standard Qwen reference path otherwise."
                  >
                    <MenuItem value="auto">Auto</MenuItem>
                    <MenuItem value="qwen_image_edit_plus">Qwen Image Edit Plus</MenuItem>
                    <MenuItem value="qwen_reference">Standard Qwen reference</MenuItem>
                  </TextField>
                  <FormControlLabel
                    control={<Switch checked={params.edit_rebalance_enabled} onChange={e => setParam('edit_rebalance_enabled', e.target.checked)} size="small" />}
                    label={
                      <Typography variant="body2" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center' }}>
                        Edit rebalance split conditioning
                        <InfoTip text="Builds separate text and reference-image conditioning for edit modes, then blends them conservatively. Disable if references overpower the edit." />
                      </Typography>
                    }
                  />
                  <TextField
                    select
                    label="Edit rebalance profile"
                    value={params.edit_rebalance_profile}
                    onChange={e => setParam('edit_rebalance_profile', e.target.value as typeof params.edit_rebalance_profile)}
                    size="small"
                    fullWidth
                    helperText="Conservative is the default first-release profile; edit is stronger, default is balanced."
                    disabled={!params.edit_rebalance_enabled}
                  >
                    <MenuItem value="conservative">Conservative</MenuItem>
                    <MenuItem value="default">Default</MenuItem>
                    <MenuItem value="edit">Edit</MenuItem>
                  </TextField>
                </>
              )}
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center', mt: 0.5 }}>
                Krea 2 Enhancer
                <InfoTip text="ComfyUI-Krea2T-Enhancer model patch: runs Krea's text-fusion, compares it with a boosted pass, and applies a capped delta for better prompt adherence and micro-detail. Off by default." />
              </Typography>
              <FormControlLabel
                control={<Switch size="small" checked={params.krea_enhancer_enabled}
                  onChange={e => setParams({ krea_enhancer_enabled: e.target.checked, krea_enhancer_variant: e.target.checked ? 'current' : 'off' })} />}
                label={<Typography variant="body2">Enable enhancer <Typography component="span" variant="caption" sx={{ color: 'text.disabled' }}>· prompt adherence + micro-detail</Typography></Typography>}
              />
              {params.krea_enhancer_enabled && (
                <LabeledSlider
                  label="Enhancer strength"
                  value={params.krea_enhancer_strength}
                  min={0} max={2} step={0.05}
                  onChange={v => setParam('krea_enhancer_strength', v)}
                  tip="How strongly the text-fusion enhancement is blended in. 1.0 is the tuned default; lower is subtler, above 1.0 can over-saturate."
                  helperText="1.0 = default · lower = subtler · >1 = stronger"
                />
              )}
            </Stack>
          </AccordionDetails>
        </Accordion>
        )}
      </Stack>
    </Box>
  )
}
