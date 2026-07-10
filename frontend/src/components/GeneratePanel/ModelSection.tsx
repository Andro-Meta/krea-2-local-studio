import React, { useEffect, useState } from 'react'
import { Alert, Box, Button, Chip, Collapse, MenuItem, Stack, TextField, Tooltip, Typography } from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { apiFetch, type AppSettings } from '../../api'
import { useStore } from '../../store'

const PROFILES = [
  {
    id: 'krea_turbo', label: 'Krea Turbo',
    desc: 'Euler/simple · 8 steps · CFG 0 · fp8 · fastest Krea profile',
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
  { id: 'gguf', label: 'GGUF native', desc: 'Load a Krea GGUF checkpoint into native PyTorch conditioning via scaled-fp8 runtime' },
  { id: 'int8', label: 'INT8', desc: '~8–13 GB VRAM · native ConvRot W8A8 using torch._int_mm · install asset from System first' },
  { id: 'bf16', label: 'bf16', desc: '~24 GB VRAM + ~48 GB RAM · full precision' },
  { id: 'fp16', label: 'fp16', desc: '~24 GB VRAM · full precision + fp16 accumulation (fast, high-VRAM)' },
]

export default function ModelSection() {
  const { params, setParam, setParams, systemReport, engineCatalog } = useStore()
  const [runtimeSettings, setRuntimeSettings] = useState<AppSettings | null>(null)
  const [savingRuntime, setSavingRuntime] = useState(false)
  const [runtimeMessage, setRuntimeMessage] = useState<{ severity: 'success' | 'warning' | 'error'; text: string } | null>(null)
  const [showModelAdvanced, setShowModelAdvanced] = useState(false)
  const loaded = systemReport?.model_status?.loaded
  const loadedCp = systemReport?.model_status?.checkpoint ?? ''
  const engines = engineCatalog?.engines ?? []
  const textEncoder = systemReport?.model_status?.text_encoder_source
  const vaeMode = runtimeSettings?.krea2_vae_mode ?? 'qwen'
  const blendRadius = runtimeSettings?.krea2_vae_blend_radius ?? 24
  const blendStrength = runtimeSettings?.krea2_vae_blend_strength ?? 0.65

  useEffect(() => {
    apiFetch.settings().then(setRuntimeSettings).catch(() => undefined)
  }, [])

  const updateRuntimeSettings = (patch: Partial<AppSettings>) => {
    setRuntimeSettings(current => current ? { ...current, ...patch } : current)
    setRuntimeMessage(null)
  }

  const saveRuntimeSettings = async () => {
    if (!runtimeSettings) return
    setSavingRuntime(true)
    setRuntimeMessage(null)
    try {
      await apiFetch.updateSettings({
        krea2_vae_mode: runtimeSettings.krea2_vae_mode,
        krea2_vae_blend_radius: runtimeSettings.krea2_vae_blend_radius,
        krea2_vae_blend_strength: runtimeSettings.krea2_vae_blend_strength,
        krea2_vae_path: runtimeSettings.krea2_vae_path,
      })
      setRuntimeMessage({ severity: 'success', text: 'Decoder settings saved. Reload the model to apply them.' })
    } catch (error: any) {
      setRuntimeMessage({ severity: 'error', text: error?.response?.data?.detail ?? error.message ?? 'Could not save decoder settings.' })
    } finally {
      setSavingRuntime(false)
    }
  }
  const applyTurboDefaults = (diffusionEngine: typeof params.diffusion_engine = 'native_pytorch') => {
    setParams({
      diffusion_engine: diffusionEngine,
      model_profile: 'krea_turbo',
      checkpoint: 'turbo',
      steps: 8,
      cfg: 0.0,
      mu: 1.15,            // pinned shift; Turbo is frozen to 1024 — never scale by resolution
      quantization: diffusionEngine === 'native_int8_convrot' ? 'int8' : diffusionEngine === 'native_gguf' ? 'gguf' : 'fp8',
      sampler: 'euler',
      scheduler: 'simple',
      conditioning_mode: 'auto',
      negative_prompt: '', // Turbo is distilled: keep CFG at 0 and negatives empty
    })
  }
  const applyEngineDefaults = (engineId: typeof params.diffusion_engine) => {
    if (engineId === 'native_gguf') {
      applyTurboDefaults('native_gguf')
    } else if (engineId === 'native_int8_convrot') {
      applyTurboDefaults('native_int8_convrot')
    } else {
      applyTurboDefaults('native_pytorch')
    }
  }
  const applyProfile = (profileId: typeof params.model_profile) => {
    if (profileId === 'krea_turbo') {
      applyTurboDefaults(params.diffusion_engine === 'native_int8_convrot' ? 'native_int8_convrot' : params.diffusion_engine === 'native_gguf' ? 'native_gguf' : 'native_pytorch')
    }
    if (profileId === 'krea_raw') {
      setParams({
        diffusion_engine: params.diffusion_engine === 'native_int8_convrot' ? 'native_int8_convrot' : params.diffusion_engine === 'native_gguf' ? 'native_gguf' : 'native_pytorch',
        model_profile: profileId,
        checkpoint: 'raw',
        steps: 52,           // RAW needs ~40–60; <40 looks washed out
        cfg: 3.5,
        mu: null,            // documented default sampling
        quantization: params.diffusion_engine === 'native_int8_convrot' ? 'int8' : params.diffusion_engine === 'native_gguf' ? 'gguf' : 'fp8', // dynamic-fp8/GGUF bridge runs RAW-class residency on 24GB; switch to bf16/fp16 if you have the VRAM/RAM
        sampler: 'euler',
        scheduler: 'simple',
        conditioning_mode: 'auto',
        negative_prompt: '', // Krea RAW works best with an empty negative prompt
      })
    }
  }
  const applyInt8 = () => {
    applyTurboDefaults('native_int8_convrot')
  }

  return (
    <Box>
      <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block', textTransform: 'uppercase', letterSpacing: 1 }}>
        Model
      </Typography>
      <Stack spacing={1.5}>
        {engines.length > 0 && (
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
              Inference engine
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap">
              {engines.map(engine => (
                <Tooltip key={engine.engine_id} title={engine.unsupported_controls?.length ? `Unsupported here: ${engine.unsupported_controls.join(', ')}` : 'Full native Krea feature set'} placement="top" arrow>
                  <Chip
                    label={engine.label}
                    variant={params.diffusion_engine === engine.engine_id ? 'filled' : 'outlined'}
                    color={params.diffusion_engine === engine.engine_id ? (engine.experimental ? 'warning' : 'primary') : 'default'}
                    onClick={() => applyEngineDefaults(engine.engine_id as typeof params.diffusion_engine)}
                    clickable
                  />
                </Tooltip>
              ))}
              {!engines.some(engine => engine.engine_id === 'native_int8_convrot') && (
                <Tooltip title="One-click native INT8 ConvRot defaults. Download/load the INT8 checkpoint from System first." placement="top" arrow>
                  <Chip
                    label="INT8"
                    variant={params.diffusion_engine === 'native_int8_convrot' ? 'filled' : 'outlined'}
                    color={params.diffusion_engine === 'native_int8_convrot' ? 'warning' : 'default'}
                    onClick={applyInt8}
                    clickable
                  />
                </Tooltip>
              )}
            </Stack>
          </Box>
        )}
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
                onClick={() => setParam('quantization', q.id as 'bf16' | 'fp8' | 'gguf' | 'fp16' | 'int8')}
                clickable
                size="small"
              />
            </Tooltip>
          ))}
        </Stack>
        <Box>
          <Button size="small" onClick={() => setShowModelAdvanced(v => !v)}
            endIcon={<ExpandMoreIcon sx={{ transform: showModelAdvanced ? 'rotate(180deg)' : 'none', transition: 'transform .15s' }} />}
            sx={{ textTransform: 'none', color: 'text.secondary' }}>
            Advanced — VAE decoder & text encoder
          </Button>
        </Box>
        <Collapse in={showModelAdvanced}>
          <Stack spacing={2}>
        <Box>
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
            VAE decoder mode
          </Typography>
          <Stack spacing={1}>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
              <TextField
                select
                size="small"
                label="Decoder"
                value={vaeMode}
                onChange={e => updateRuntimeSettings({ krea2_vae_mode: e.target.value as AppSettings['krea2_vae_mode'] })}
                helperText="ComfyUI applies this on the next generation (no model reload needed). Qwen is the default."
                sx={{ minWidth: 240 }}
              >
                <MenuItem value="qwen">Qwen VAE (default)</MenuItem>
                <MenuItem value="comfy_qwen">Comfy Qwen VAE</MenuItem>
                <MenuItem value="qwen_wan_blend">Qwen + Wan detail blend</MenuItem>
                <MenuItem value="wan_experimental">Generic Wan 2.1 (experimental)</MenuItem>
              </TextField>
              {vaeMode === 'qwen_wan_blend' && (
                <>
                  <TextField
                    size="small"
                    type="number"
                    label="Blend radius"
                    value={blendRadius}
                    onChange={e => updateRuntimeSettings({ krea2_vae_blend_radius: Math.max(1, Number(e.target.value) || 24) })}
                    inputProps={{ min: 1, max: 128, step: 1 }}
                    sx={{ width: 140 }}
                  />
                  <TextField
                    size="small"
                    type="number"
                    label="Wan detail"
                    value={blendStrength}
                    onChange={e => updateRuntimeSettings({ krea2_vae_blend_strength: Math.max(0, Math.min(2, Number(e.target.value) || 0.65)) })}
                    inputProps={{ min: 0, max: 2, step: 0.05 }}
                    sx={{ width: 140 }}
                  />
                </>
              )}
              <Button size="small" variant="outlined" onClick={saveRuntimeSettings} disabled={!runtimeSettings || savingRuntime}>
                Save decoder
              </Button>
            </Stack>
            <Typography variant="caption" sx={{ color: 'text.disabled' }}>
              `Qwen + Wan detail blend` decodes Qwen for base/color and injects high-frequency Wan detail. Generic Wan is manual/experimental.
            </Typography>
            {runtimeMessage && <Alert severity={runtimeMessage.severity} sx={{ py: 0 }}>{runtimeMessage.text}</Alert>}
          </Stack>
        </Box>
        <Box>
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
            Generation text encoder / CLIP
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', wordBreak: 'break-all' }}>
            Krea 2 uses Qwen3-VL conditioning, not CLIP-L/T5. Runtime: {textEncoder?.kind ?? 'not loaded'}
            {textEncoder?.runtime ? ` · ${textEncoder.runtime}` : ''}
            {textEncoder?.status ? ` · ${textEncoder.status}` : ''}
          </Typography>
        </Box>
          </Stack>
        </Collapse>
        {loaded && loadedCp && (
          <Typography variant="caption" sx={{ color: 'success.main', wordBreak: 'break-all' }}>
            Loaded: {loadedCp.split(/[\\/]/).pop()}
          </Typography>
        )}
      </Stack>
    </Box>
  )
}
