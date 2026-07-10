import { Box, MenuItem, Paper, Slider, Stack, TextField, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { useStore } from '../../store'

// Only the Mr. Flow knobs that materially change the result:
//  - upscaler / SR factor (x2 vs 4x) -> base render size + sharpness character
//  - step regime (preset) -> base composition quality vs speed
//  - refine strength -> how much Krea-2 reworks the ESRGAN pixels (the big one)
const PRESETS = [
  { value: '', label: 'Auto (by Turbo/RAW)' },
  { value: 'base_12plus1', label: 'Base 12+1 (RAW default)' },
  { value: 'base_20plus1', label: 'Base 20+1 (best composition)' },
  { value: 'turbo_8plus1', label: 'Turbo 8+1 (fastest)' },
]

export default function MrFlowSettings() {
  const { params, setParams, createMode } = useStore()
  // Hidden in the Upscale workflow sub-tab — UpscalePanel is the sole control there.
  if (!params.mrflow || createMode === 'upscale') return null

  const factor = params.mrflow_upscaler === 'remacri_x4' ? 4 : 2
  const baseW = Math.max(16, Math.round(params.width / factor))
  const baseH = Math.max(16, Math.round(params.height / factor))
  // 0 means "use preset default"; show the effective value for clarity.
  const effectiveDenoise = params.mrflow_refine_denoise > 0 ? params.mrflow_refine_denoise : 0.12

  return (
    <Paper variant="outlined" sx={{ p: 1.5, borderColor: 'rgba(202,196,208,0.18)' }}>
      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 1, textTransform: 'uppercase', letterSpacing: 1 }}>
        Mr. Flow settings — base render {baseW}×{baseH} → {params.width}×{params.height}
      </Typography>
      <Stack spacing={1.75}>
        <Box>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>Upscaler (SR model + factor)</Typography>
          <ToggleButtonGroup
            size="small"
            exclusive
            fullWidth
            value={params.mrflow_upscaler}
            onChange={(_, v) => v && setParams({ mrflow_upscaler: v })}
            sx={{ mt: 0.5 }}
          >
            <ToggleButton value="esrgan_x2">RealESRGAN ×2</ToggleButton>
            <ToggleButton value="remacri_x4">Foolhardy Remacri ×4</ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <TextField
          select
          size="small"
          label="Step regime (preset)"
          value={params.mrflow_preset}
          onChange={e => setParams({ mrflow_preset: e.target.value })}
          helperText="Controls base-render steps + CFG. Auto picks by the active Turbo/RAW model."
        >
          {PRESETS.map(p => <MenuItem key={p.value || 'auto'} value={p.value}>{p.label}</MenuItem>)}
        </TextField>

        <Box>
          <Stack direction="row" justifyContent="space-between" alignItems="baseline">
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>Refine strength (denoise)</Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              {params.mrflow_refine_denoise > 0 ? effectiveDenoise.toFixed(2) : `auto (~${effectiveDenoise.toFixed(2)})`}
            </Typography>
          </Stack>
          <Slider
            size="small"
            min={0}
            max={0.30}
            step={0.01}
            value={params.mrflow_refine_denoise}
            onChange={(_, v) => setParams({ mrflow_refine_denoise: v as number })}
            marks={[{ value: 0, label: 'auto' }, { value: 0.12, label: '0.12' }, { value: 0.30, label: '0.30' }]}
          />
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Higher = more Krea-2 rework/detail (can drift); lower = closer to the raw ESRGAN upscale. 0 keeps the preset default.
          </Typography>
        </Box>
      </Stack>
    </Paper>
  )
}
