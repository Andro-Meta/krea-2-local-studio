import React, { useState } from 'react'
import {
  Accordion, AccordionDetails, AccordionSummary,
  Box, FormControlLabel, Grid, Slider, Stack, Switch, TextField, Tooltip, Typography,
} from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import { useStore } from '../../store'

function InfoTip({ text }: { text: string }) {
  return (
    <Tooltip title={text} placement="right" arrow>
      <InfoOutlinedIcon sx={{ fontSize: 14, color: 'text.disabled', ml: 0.5, cursor: 'help', verticalAlign: 'middle' }} />
    </Tooltip>
  )
}

function LabeledSlider({ label, value, min, max, step, onChange, tip, helperText }: {
  label: string; value: number; min: number; max: number; step: number
  onChange: (v: number) => void; tip?: string; helperText?: string
}) {
  return (
    <Box sx={{ width: '100%' }}>
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
  const { params, setParam } = useStore()
  const [advOpen, setAdvOpen] = useState(false)
  const isTurbo = params.checkpoint === 'turbo'

  return (
    <Box>
      <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1.5, display: 'block', textTransform: 'uppercase', letterSpacing: 1 }}>
        Parameters
      </Typography>
      <Stack spacing={2}>
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
          tip="Classifier-Free Guidance — how strictly the model follows your prompt. Turbo uses 0 (guidance is baked into distillation). RAW uses 3.5."
          helperText={isTurbo ? 'Turbo: keep at 0 (guidance built-in)' : 'RAW default: 3.5'}
        />

        {params.mode === 'inpaint' && (
          <LabeledSlider
            label="Denoise strength"
            value={params.denoise}
            min={0.01} max={1.0} step={0.01}
            onChange={v => setParam('denoise', v)}
            tip="How much to change the input image. 1.0 = ignore original (same as txt2img). 0.5 = half old / half new. 0.3 = subtle edits only."
            helperText="0.75 = balanced · 0.3–0.5 = preserve original · 1.0 = full regen"
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
            <TextField
              label="Batch"
              type="number"
              size="small"
              fullWidth
              value={params.num_images}
              onChange={e => setParam('num_images', Math.max(1, Math.min(4, Number(e.target.value))))}
              helperText="1–4 images"
              inputProps={{ min: 1, max: 4 }}
            />
          </Grid>
        </Grid>

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
            <LabeledSlider
              label="Refine denoise"
              value={params.refine_denoise}
              min={0.1} max={0.6} step={0.05}
              onChange={v => setParam('refine_denoise', v)}
              tip="How much the refine pass may change the image. 0.3 = balanced detail; lower = subtler."
              helperText="0.3 = balanced · 0.1–0.2 = subtle sharpen · 0.5+ = stronger rework"
            />
          )}
        </Box>

        <Accordion expanded={advOpen} onChange={(_, v) => setAdvOpen(v)} disableGutters>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Advanced — timestep schedule &amp; conditioning
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={2}>
              <LabeledSlider
                label="μ — flow shift (ModelSamplingFlux)"
                value={params.mu}
                min={0} max={2.0} step={0.05}
                onChange={v => setParam('mu', v)}
                tip="ModelSamplingFlux shift: shifts timestep density toward high-noise steps. Higher = better for large images (>1024px). Turbo default 1.15. Set 0 to auto-calculate from resolution."
                helperText="0 = auto · Turbo: 1.15 · higher = better for large images"
              />
              <LabeledSlider
                label="y1 (schedule lower bound)"
                value={params.y1}
                min={0.1} max={1.0} step={0.05}
                onChange={v => setParam('y1', v)}
                tip="Lower bound of the logit-normal timestep schedule. Lower = more denoising passes at fine detail. Default: 0.5"
              />
              <LabeledSlider
                label="y2 (schedule upper bound)"
                value={params.y2}
                min={1.0} max={2.0} step={0.05}
                onChange={v => setParam('y2', v)}
                tip="Upper bound of the logit-normal timestep schedule. Higher = more passes at coarse structure. Default: 1.15"
              />
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center', mt: 0.5 }}>
                Conditioning Rebalance
                <InfoTip text="Independently scales the 12 Qwen3-VL encoder layer taps. Layer 9 (5×) and layer 11 (4×) most strongly affect style. The global multiplier scales everything." />
              </Typography>
              <LabeledSlider
                label="Global multiplier"
                value={params.rebalance_multiplier}
                min={0.5} max={10} step={0.1}
                onChange={v => setParam('rebalance_multiplier', v)}
                tip="Scales all 12 conditioning taps together. Default 4.0 matches Krea's reference. Lower = softer prompt adherence."
              />
              <TextField
                label="Per-layer weights (12 comma-separated values)"
                value={params.rebalance_weights}
                onChange={e => setParam('rebalance_weights', e.target.value)}
                size="small"
                fullWidth
                helperText="Layers 1–12 of Qwen3-VL. Default: 1,1,1,1,1,1,1,2.5,5,1.1,4,1"
              />
            </Stack>
          </AccordionDetails>
        </Accordion>
      </Stack>
    </Box>
  )
}
