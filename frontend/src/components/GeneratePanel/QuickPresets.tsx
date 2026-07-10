import { useState } from 'react'
import {
  Accordion, AccordionDetails, AccordionSummary, Alert, Box, Chip, CircularProgress,
  Stack, ToggleButton, ToggleButtonGroup, Tooltip, Typography,
} from '@mui/material'
import BoltIcon from '@mui/icons-material/Bolt'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { useStore, type GenerateParams, type ActiveLora } from '../../store'
import { apiFetch } from '../../api'

// Krea 2 filter-bypass / prompt-adherence diff LoRA, applied to every quick recipe
// at a fixed strength (bypass LoRAs run at extreme weights; 6850 is the chosen default).
const BYPASS_LORA: ActiveLora = {
  name: 'krea2filterbypass3',
  filename: 'krea2filterbypass3.safetensors',
  strength: 6850,
  enabled: true,
  block_filter: 'style_safe',
}

// God Mode LoRA stack: the user's realism v2 LoRA @1.0 + the filter-bypass @6850
// (substituting the workflow's private "commercial advertising" LoRA).
const GODMODE_REALISM_LORA: ActiveLora = {
  name: 'realism_engine_krea2_v2',
  filename: 'realism_engine_krea2_v2.safetensors',
  strength: 1.0,
  enabled: true,
  block_filter: 'all',
}

// Ensure the bypass LoRA is present at 6850, keeping any other LoRAs the user has.
function withBypass(existing: ActiveLora[]): ActiveLora[] {
  const others = (existing || []).filter(
    l => l.filename !== BYPASS_LORA.filename && l.name !== BYPASS_LORA.name,
  )
  return [...others, { ...BYPASS_LORA }]
}

// One-click recipes, collapsed by default. Turbo = the user's hand-picked favorites
// (each bakes in its composition-lock cutoff) plus "Xperiment Slow" (the uncensored /
// prompt-adherence recipe, which provisions its assets). RAW = the favored RAW-Int8
// sampler recipes. Each normal button applies the full recipe so switching is one click.
type Recipe = {
  id: string
  label: string
  speed: string
  why: string
  isDefault?: boolean
  patch: Partial<GenerateParams>
}

export const INT8_VARIANTS: { id: string; label: string; tip: string }[] = [
  { id: 'redcraft', label: 'RedCraft', tip: 'RedCraft v2.2 Krea 2 finetune INT8 ConvRot — default (best prompt follow).' },
  { id: 'km_v2', label: 'Kreamania v2', tip: 'Kreamania v2 INT8 ConvRot.' },
  { id: 'km_v3', label: 'Kreamania v3', tip: 'Kreamania v3 INT8 ConvRot — "simple" build.' },
  { id: 'km_v3_comfy', label: 'Kreamania v3 (comfy)', tip: 'Kreamania v3 INT8 ConvRot — gorbatjovy "comfy" build.' },
  { id: 'ax1y2jp', label: 'AX1Y2JP', tip: 'AX1Y2JP Krea-2-Turbo INT8 ConvRot (standard).' },
  { id: 'sceneworks', label: 'SceneWorks', tip: 'SceneWorks krea-2-turbo INT8 ConvRot.' },
  { id: 'lilcheaty', label: 'lilcheaty', tip: 'lilcheaty Krea2 Turbo INT8 ConvRot.' },
  { id: 'tsolful', label: 'tsolful', tip: 'tsolful Krea2 Turbo INT8 (comfy-fixed).' },
  { id: 'orig', label: 'Krea2 (orig)', tip: 'Original Comfy-Org Krea 2 Turbo INT8 ConvRot.' },
]

const TURBO_BASE: Partial<GenerateParams> = {
  diffusion_engine: 'native_int8_convrot', quantization: 'int8', checkpoint: 'turbo',
  turbo_int8_variant: 'redcraft',
  model_profile: 'krea_turbo', cfg: 1.0, cfg_zero_star: true, cfg_zero_init_steps: 1, mu: 1.15,
  seed_variance_preset: 'wild', seed_variance_algorithm: 'rbg', seed_variance_model_type: 'krea2',
  seed_variance_schedule: 'hard_lock', seed_variance_cutoff_strength: 1.0, mrflow: false, god_mode: false,
}

const TURBO_RECIPES: Recipe[] = [
  { id: 'xf_loose', label: 'Xperiment fast · loose 2/8', speed: '~4.0s ⚡', isDefault: true,
    why: 'Default. RedCraft INT8, er_sde/beta57 @8, composition lock loose (more per-roll variety).',
    patch: { ...TURBO_BASE, sampler: 'er_sde', scheduler: 'beta57', steps: 8, seed_variance_cutoff_step: 2 } },
  { id: 'xf_tight', label: 'Xperiment fast · tight 7/8', speed: '~4.2s',
    why: 'RedCraft INT8, er_sde/beta57 @8, composition lock tight (shot held, minimal variance).',
    patch: { ...TURBO_BASE, sampler: 'er_sde', scheduler: 'beta57', steps: 8, seed_variance_cutoff_step: 5 } },
  { id: 'turbo_default', label: 'Turbo default · tight 7/8', speed: '~4.3s',
    why: 'euler_flow/simple @8, composition lock tight. The shipped Krea recipe.',
    patch: { ...TURBO_BASE, sampler: 'euler_flow', scheduler: 'simple', steps: 8, seed_variance_cutoff_step: 7 } },
  { id: 'turbo_ancestral', label: 'Turbo ancestral · tight 7/8', speed: '~5.4s',
    why: 'euler_ancestral/beta @10, composition lock tight. Ancestral adds organic detail.',
    patch: { ...TURBO_BASE, sampler: 'euler_ancestral', scheduler: 'beta', steps: 10, seed_variance_cutoff_step: 9 } },
  { id: 'turbo_res3s', label: 'Turbo RES3S · tight 7/8', speed: '~16.3s',
    why: 'res_3s/bong_tangent @10 (3rd-order RES4LYF), composition lock tight. Highest quality, slowest.',
    patch: { ...TURBO_BASE, sampler: 'res_3s', scheduler: 'bong_tangent', steps: 10, seed_variance_cutoff_step: 9 } },
]

const RAW_BASE: Partial<GenerateParams> = {
  diffusion_engine: 'native_int8_convrot', quantization: 'int8', checkpoint: 'raw',
  model_profile: 'krea_raw', cfg_zero_star: false, mu: null, seed_variance_preset: 'off',
  mrflow: false, god_mode: false,
}

// Mr. Flow (fast staged sampling): render small, SR-upscale, 1-step model-native refine.
// These set the TARGET resolution + SR model and turn Mr. Flow on; the active checkpoint
// (Turbo/RAW) still decides the model + stage numbers (backend auto-preset). 1K/2K
// stay here; use the dedicated Upscale tab for 4K experiments.
type MrFlowRecipe = { id: string; label: string; speed: string; why: string; patch: Partial<GenerateParams> }
const MRFLOW_RECIPES: MrFlowRecipe[] = [
  { id: 'mf_1k', label: 'Mr.Flow → 1K', speed: 'fast',
    why: 'Base render at 512, RealESRGAN x2, 1-step refine → 1024. Cheapest sharp 1K.',
    patch: { mrflow: true, god_mode: false, mrflow_upscaler: 'esrgan_x2', mrflow_preset: '', width: 1024, height: 1024 } },
  { id: 'mf_2k', label: 'Mr.Flow → 2K', speed: 'fast',
    why: 'Base render at 1024, RealESRGAN x2, 1-step refine → 2048. Much faster than native 2K.',
    patch: { mrflow: true, god_mode: false, mrflow_upscaler: 'esrgan_x2', mrflow_preset: '', width: 2048, height: 2048 } },
]

function mrflowMatches(params: GenerateParams, patch: Partial<GenerateParams>): boolean {
  return !!params.mrflow && params.width === patch.width && params.mrflow_upscaler === patch.mrflow_upscaler
}

const RAW_RECIPES: Recipe[] = [
  { id: 'raw_crisp', label: 'RAW crisp', speed: '~26s', isDefault: true,
    why: 'RAW default. euler_flow/beta @28, CFG 4. Crisp, clean, realistic.',
    patch: { ...RAW_BASE, sampler: 'euler_flow', scheduler: 'beta', steps: 28, cfg: 4.0 } },
  { id: 'raw_kl', label: 'RAW KL-optimal', speed: '~26s',
    why: 'euler_flow/kl_optimal @28, CFG 4. Natural skin, striking eyes/lighting.',
    patch: { ...RAW_BASE, sampler: 'euler_flow', scheduler: 'kl_optimal', steps: 28, cfg: 4.0 } },
  { id: 'raw_ersde', label: 'RAW ER-SDE (fast)', speed: '~11s',
    why: 'er_sde/beta @10, CFG 4. Fastest RAW; slightly softer detail.',
    patch: { ...RAW_BASE, sampler: 'er_sde', scheduler: 'beta', steps: 10, cfg: 4.0 } },
  { id: 'raw_res2s', label: 'RAW RES2S (max detail)', speed: '~47s',
    why: 'res_2s/bong_tangent @24, CFG 4. Finest skin micro-detail, slowest.',
    patch: { ...RAW_BASE, sampler: 'res_2s', scheduler: 'bong_tangent', steps: 24, cfg: 4.0 } },
]

function matches(params: GenerateParams, patch: Partial<GenerateParams>): boolean {
  if (params.checkpoint !== patch.checkpoint) return false
  if (params.sampler !== patch.sampler) return false
  if (params.scheduler !== patch.scheduler) return false
  if (params.steps !== patch.steps) return false
  if (patch.seed_variance_cutoff_step !== undefined &&
      params.seed_variance_cutoff_step !== patch.seed_variance_cutoff_step) return false
  return true
}

export default function QuickPresets() {
  const { params, setParams, setLoras } = useStore()
  const [xpBusy, setXpBusy] = useState(false)
  const [msg, setMsg] = useState<{ severity: 'success' | 'error'; text: string } | null>(null)

  const mrflowActive = MRFLOW_RECIPES.find(r => mrflowMatches(params, r.patch))
  const active = params.mrflow ? undefined : [...TURBO_RECIPES, ...RAW_RECIPES].find(r => matches(params, r.patch))

  // Apply a recipe and always keep the filter-bypass LoRA on at 6850.
  const applyRecipe = (patch: Partial<GenerateParams>) =>
    setParams({ ...patch, loras: withBypass(params.loras) })

  // God Mode: 4-stage max-quality pipeline (Krea2 -> Z-Image refine -> SeedVR2 -> FaceDetail).
  // Backend fixes the samplers/steps; here we just flip the flag, set the base size, and the
  // realism v2 + bypass@6850 LoRA stack. Slow but "holy quality".
  const applyGodMode = () => setParams({
    god_mode: true, mrflow: false,
    // God Mode's backend hardcodes the models (Krea2 turbo fp8 + Z-Image bf16 +
    // SeedVR2 7B-sharp fp16), so these just reflect reality for the UI/metadata.
    checkpoint: 'turbo', model_profile: 'krea_turbo',
    diffusion_engine: 'native_pytorch', quantization: 'fp8',
    cfg_zero_star: false, seed_variance_preset: 'off',
    // 2K base → SeedVR2 tiling → 4K. A 2× upscale keeps real detail (sharper than
    // upscaling a 1K base to 4K/5K, which mostly invents/softens).
    width: 2048, height: 2048, num_images: 1,
    loras: [{ ...GODMODE_REALISM_LORA }, { ...BYPASS_LORA }],
  })

  // "Xperiment Slow": the uncensored / prompt-adherence recipe. Provisions its assets
  // (abliterated encoder, WAN VAE, filter-bypass diff @4 + Realism LoKr @0.6) then applies
  // er_sde/beta57 @8, CFG 1. Slower than the fast presets, hence "Slow".
  const runXperiment = async () => {
    if (xpBusy) return
    setXpBusy(true)
    setMsg(null)
    try {
      const result = await apiFetch.setupXperiment()
      apiFetch.loras().then(setLoras).catch(() => undefined)
      const xLoras = (result.loras?.length ? result.loras : [result.lora]).map((lora: any) => {
        const isBypass = lora.filename === BYPASS_LORA.filename || lora.name === BYPASS_LORA.name
        return {
          name: lora.name, filename: lora.filename,
          strength: isBypass ? BYPASS_LORA.strength : lora.strength, enabled: true,
          block_filter: lora.block_filter ?? (lora.name === 'Krea2-realism-V1' ? 'late' : 'style_safe'),
        }
      })
      const names = new Set(xLoras.map((l: any) => l.name))
      const engine = (result.diffusion_engine ?? params.diffusion_engine) as GenerateParams['diffusion_engine']
      const keepRaw = params.checkpoint === 'raw' || params.model_profile === 'krea_raw'
      setParams({
        diffusion_engine: engine,
        model_profile: keepRaw ? 'krea_raw' : 'krea_turbo',
        checkpoint: keepRaw ? 'raw' : 'turbo',
        quantization: (engine === 'native_int8_convrot' ? 'int8' : engine === 'native_gguf' ? 'gguf'
          : keepRaw ? params.quantization : (result.quantization ?? 'fp8')) as GenerateParams['quantization'],
        steps: result.sampler.steps,
        cfg: result.sampler.cfg,
        mu: keepRaw ? null : 1.15,
        sampler: result.sampler.sampler as GenerateParams['sampler'],
        scheduler: result.sampler.scheduler as GenerateParams['scheduler'],
        res4lyf_sampler: result.res4lyf?.sampler_name ?? '',
        res4lyf_eta: result.res4lyf?.eta ?? 0.5,
        res4lyf_bongmath: result.res4lyf?.bongmath ?? false,
        use_prompt_expander: result.use_prompt_expander ?? false,
        negative_prompt: '',
        loras: [...params.loras.filter(l => !names.has(l.name)), ...xLoras],
      })
      setMsg({ severity: 'success', text: 'Xperiment Slow applied (uncensored / prompt-adherence recipe).' })
    } catch (err: any) {
      const detail = err?.response?.data?.detail
      setMsg({ severity: 'error', text: detail ?? err?.message ?? 'Xperiment setup failed.' })
    } finally {
      setXpBusy(false)
    }
  }

  const presetChip = (r: Recipe) => (
    <Tooltip key={r.id} title={r.why} arrow>
      <Chip
        clickable
        onClick={() => applyRecipe(r.patch)}
        variant={active?.id === r.id ? 'filled' : 'outlined'}
        color={active?.id === r.id ? 'primary' : 'default'}
        label={
          <Box component="span" sx={{ display: 'inline-flex', alignItems: 'baseline', gap: 0.75 }}>
            <span>{r.label}{r.isDefault ? ' ★' : ''}</span>
            <Box component="span" sx={{ fontSize: '0.72em', opacity: 0.75 }}>{r.speed}</Box>
          </Box>
        }
        sx={{ height: 30 }}
      />
    </Tooltip>
  )

  return (
    <Accordion disableGutters sx={{ bgcolor: 'transparent', border: '1px solid rgba(202,196,208,0.18)', borderRadius: 2, '&:before': { display: 'none' } }}>
      <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center', gap: 0.5, textTransform: 'uppercase', letterSpacing: 1 }}>
          <BoltIcon fontSize="inherit" /> Quick recipe{params.god_mode ? ' · God Mode ✨' : mrflowActive ? ` · ${mrflowActive.label}` : active ? ` · ${active.label}` : ' · custom'}
        </Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Stack spacing={1.25}>
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, textTransform: 'uppercase', letterSpacing: 1 }}>
              Turbo Int8 model
            </Typography>
            <ToggleButtonGroup
              size="small"
              exclusive
              value={params.turbo_int8_variant}
              onChange={(_, v) => { if (v) setParams({ turbo_int8_variant: v }) }}
              sx={{ flexWrap: 'wrap', gap: 0.5, '& .MuiToggleButtonGroup-grouped': { border: '1px solid rgba(202,196,208,0.28)', borderRadius: '6px !important', mx: 0 } }}
            >
              {INT8_VARIANTS.map(v => (
                <ToggleButton key={v.id} value={v.id} title={v.tip} sx={{ px: 1.25, py: 0.4, textTransform: 'none', fontSize: 12 }}>
                  {v.label}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
            <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: 0.25 }}>
              Applies to every Turbo Int8 recipe below. Kreamania v2 = default. (RedCraft is a finetune, not stock Krea 2.)
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, textTransform: 'uppercase', letterSpacing: 1 }}>
              Turbo Int8 · CFG 1 · CFG-Zero* · composition lock
            </Typography>
            <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap">
              {TURBO_RECIPES.map(presetChip)}
              <Tooltip title="Uncensored / prompt-adherence recipe: abliterated encoder, WAN VAE, filter-bypass diff @4 + Realism LoKr @0.6, er_sde/beta57 @8, CFG 1. Provisions assets on first use; slower than the fast presets." arrow>
                <Chip
                  clickable
                  onClick={runXperiment}
                  color="secondary"
                  variant="outlined"
                  icon={xpBusy ? <CircularProgress size={12} color="inherit" /> : undefined}
                  label={xpBusy ? 'Applying…' : 'Xperiment Slow'}
                  sx={{ height: 30 }}
                />
              </Tooltip>
            </Stack>
          </Box>
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, textTransform: 'uppercase', letterSpacing: 1 }}>
              RAW Int8 · real CFG · adaptive shift
            </Typography>
            <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap">
              {RAW_RECIPES.map(presetChip)}
            </Stack>
          </Box>
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, textTransform: 'uppercase', letterSpacing: 1 }}>
              Mr. Flow · fast staged upscale (uses active Turbo/RAW model)
            </Typography>
            <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap">
              {MRFLOW_RECIPES.map(r => (
                <Tooltip key={r.id} title={r.why} arrow>
                  <Chip
                    clickable
                    onClick={() => applyRecipe(r.patch)}
                    variant={mrflowActive?.id === r.id ? 'filled' : 'outlined'}
                    color={mrflowActive?.id === r.id ? 'primary' : 'default'}
                    label={
                      <Box component="span" sx={{ display: 'inline-flex', alignItems: 'baseline', gap: 0.75 }}>
                        <span>{r.label}</span>
                        <Box component="span" sx={{ fontSize: '0.72em', opacity: 0.75 }}>{r.speed}</Box>
                      </Box>
                    }
                    sx={{ height: 30 }}
                  />
                </Tooltip>
              ))}
            </Stack>
          </Box>
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, textTransform: 'uppercase', letterSpacing: 1 }}>
              Max quality (slow)
            </Typography>
            <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap">
              <Tooltip title="God Mode: Krea2 2K base → Z-Image Turbo refine (denoise 0.1) → SeedVR2 7B-sharp TILING upscale to 4K → FaceDetailer. Realism v2 + filter-bypass@6850 LoRAs. Slow (four models staged + tiled 4K upscale — several minutes), but top quality." arrow>
                <Chip
                  clickable
                  onClick={applyGodMode}
                  color={params.god_mode ? 'primary' : 'secondary'}
                  variant={params.god_mode ? 'filled' : 'outlined'}
                  label={
                    <Box component="span" sx={{ display: 'inline-flex', alignItems: 'baseline', gap: 0.75 }}>
                      <span>God Mode</span>
                      <Box component="span" sx={{ fontSize: '0.72em', opacity: 0.75 }}>~slow ✨</Box>
                    </Box>
                  }
                  sx={{ height: 30 }}
                />
              </Tooltip>
            </Stack>
          </Box>
          {msg && <Alert severity={msg.severity} sx={{ py: 0 }} onClose={() => setMsg(null)}>{msg.text}</Alert>}
        </Stack>
      </AccordionDetails>
    </Accordion>
  )
}
