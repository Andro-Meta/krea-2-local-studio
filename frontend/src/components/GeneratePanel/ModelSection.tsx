import React from 'react'
import { Box, Chip, Stack, Tooltip, Typography } from '@mui/material'
import { useStore } from '../../store'

const PROFILES = [
  {
    id: 'krea_turbo', label: 'Krea Turbo',
    desc: 'Euler/simple · 8 steps · CFG 1 · fp8 · fastest Krea profile',
    enabled: true,
  },
  {
    id: 'krea_raw', label: 'Krea RAW',
    desc: 'Euler/simple · 52 steps · CFG 3.5 · bf16 · high memory',
    enabled: true,
  },
  {
    id: 'qwen_image_edit', label: 'Qwen Image Edit',
    desc: 'Planned optional profile · loader not enabled yet',
    enabled: false,
  },
  {
    id: 'lens_turbo', label: 'Lens',
    desc: 'Planned optional profile · GPT-OSS encoder/Flux2 VAE loader required',
    enabled: false,
  },
  {
    id: 'ernie_turbo', label: 'ERNIE',
    desc: 'Planned optional profile · ERNIE encoder/Flux2 VAE loader required',
    enabled: false,
  },
  {
    id: 'z_image_turbo', label: 'Z-Image',
    desc: 'Planned optional profile · Z-Image loader and ae.safetensors VAE required',
    enabled: false,
  },
]

const QUANTS = [
  { id: 'fp8',  label: 'fp8',  desc: '~8–13 GB VRAM · quantized (dynamic-fp8 lets RAW/bf16 run on 24GB)' },
  { id: 'bf16', label: 'bf16', desc: '~24 GB VRAM + ~48 GB RAM · full precision' },
  { id: 'fp16', label: 'fp16', desc: '~24 GB VRAM · full precision + fp16 accumulation (fast, high-VRAM)' },
]

export default function ModelSection() {
  const { params, setParam, setParams, systemReport } = useStore()
  const loaded = systemReport?.model_status?.loaded
  const loadedCp = systemReport?.model_status?.checkpoint ?? ''
  const applyProfile = (profileId: typeof params.model_profile) => {
    if (profileId === 'krea_turbo') {
      setParams({
        model_profile: profileId,
        checkpoint: 'turbo',
        steps: 8,
        cfg: 1.0,
        mu: 1.15,            // pinned shift; Turbo is frozen to 1024 — never scale by resolution
        quantization: 'fp8',
        sampler: 'euler',
        scheduler: 'simple',
        conditioning_mode: 'auto',
        negative_prompt: '', // LLM-conditioned model: negatives are unused at CFG 1
      })
    }
    if (profileId === 'krea_raw') {
      setParams({
        model_profile: profileId,
        checkpoint: 'raw',
        steps: 52,           // RAW needs ~40–60; <40 looks washed out
        cfg: 3.5,
        mu: null,            // documented default sampling
        quantization: 'fp8', // dynamic-fp8 runs RAW on 24GB; switch to bf16/fp16 if you have the VRAM/RAM
        sampler: 'euler',
        scheduler: 'simple',
        conditioning_mode: 'auto',
        negative_prompt: '', // Krea RAW works best with an empty negative prompt
      })
    }
  }

  return (
    <Box>
      <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block', textTransform: 'uppercase', letterSpacing: 1 }}>
        Model
      </Typography>
      <Stack spacing={1.5}>
        <Stack direction="row" spacing={1} flexWrap="wrap">
          {PROFILES.map(c => (
            <Tooltip key={c.id} title={c.desc} placement="top" arrow>
              <Chip
                label={c.label}
                variant={params.model_profile === c.id ? 'filled' : 'outlined'}
                color={params.model_profile === c.id ? 'primary' : 'default'}
                onClick={() => c.enabled && applyProfile(c.id as typeof params.model_profile)}
                clickable={c.enabled}
                disabled={!c.enabled}
              />
            </Tooltip>
          ))}
        </Stack>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          Profile routing updates checkpoint, encoder/VAE assumptions, sampler defaults, CFG, steps, precision, and conditioning mode together.
        </Typography>
        <Stack direction="row" spacing={1}>
          {QUANTS.map(q => (
            <Tooltip key={q.id} title={q.desc} placement="top" arrow>
              <Chip
                label={q.label}
                variant={params.quantization === q.id ? 'filled' : 'outlined'}
                color={params.quantization === q.id ? 'secondary' : 'default'}
                onClick={() => setParam('quantization', q.id as 'bf16' | 'fp8' | 'fp16')}
                clickable
                size="small"
              />
            </Tooltip>
          ))}
        </Stack>
        {loaded && loadedCp && (
          <Typography variant="caption" sx={{ color: 'success.main', wordBreak: 'break-all' }}>
            Loaded: {loadedCp.split(/[\\/]/).pop()}
          </Typography>
        )}
      </Stack>
    </Box>
  )
}
