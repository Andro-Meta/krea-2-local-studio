import { Alert, MenuItem, Stack, TextField, Typography } from '@mui/material'
import type { KreaDeforumStatus } from '../../api'
import type { AnimateControlProps } from './types'

interface AdvancedControlsProps extends AnimateControlProps {
  runtime: KreaDeforumStatus | null
}

export default function AdvancedControls({
  value,
  errors,
  disabled,
  update,
  runtime,
}: AdvancedControlsProps) {
  return (
    <Stack spacing={2}>
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
        <TextField
          id="animate-steps" label="Steps" type="number" value={value.steps}
          onChange={event => update('steps', Number(event.target.value))}
          inputProps={{ min: 3, max: 52, step: 1 }}
          error={!!errors.steps} helperText={errors.steps || '3–52 diffusion steps'}
          disabled={disabled} fullWidth
        />
        <TextField
          label="Rendered frame override"
          type="number"
          value={value.render_frames ?? ''}
          onChange={event => update('render_frames', event.target.value === '' ? null : Number(event.target.value))}
          inputProps={{ min: 1, max: 720, step: 1 }}
          error={!!errors.render_frames}
          helperText={errors.render_frames || 'Optional; overrides duration × FPS (1–720).'}
          disabled={disabled} fullWidth
        />
      </Stack>
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
        <TextField
          select label="Sampler" value={value.sampler_name}
          onChange={event => update('sampler_name', event.target.value)}
          disabled={disabled} fullWidth
        >
          {['er_sde', 'euler', 'euler_ancestral', 'dpmpp_2m', 'ddim', 'uni_pc'].map(item => (
            <MenuItem key={item} value={item}>{item}</MenuItem>
          ))}
        </TextField>
        <TextField
          select label="Scheduler" value={value.scheduler}
          onChange={event => update('scheduler', event.target.value)}
          disabled={disabled} fullWidth
        >
          {['simple', 'normal', 'beta', 'sgm_uniform', 'karras', 'exponential'].map(item => (
            <MenuItem key={item} value={item}>{item}</MenuItem>
          ))}
        </TextField>
      </Stack>

      <TextField
        label="Negative prompt"
        value={value.negative_prompt}
        onChange={event => update('negative_prompt', event.target.value)}
        multiline minRows={3} disabled={disabled} fullWidth
      />

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
        <TextField
          select label="Seed behavior" value={value.seed_behavior}
          onChange={event => update('seed_behavior', event.target.value as typeof value.seed_behavior)}
          disabled={disabled} fullWidth
        >
          {['fixed', 'iter', 'random', 'ladder'].map(item => <MenuItem key={item} value={item}>{item}</MenuItem>)}
        </TextField>
        <TextField
          select label="Edge / border" value={value.border_mode}
          onChange={event => update('border_mode', event.target.value as typeof value.border_mode)}
          disabled={disabled} fullWidth
        >
          {['replicate', 'reflect', 'wrap', 'black'].map(item => <MenuItem key={item} value={item}>{item}</MenuItem>)}
        </TextField>
      </Stack>
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
        <TextField
          id="animate-diffusion-cadence"
          label="Diffusion cadence" type="number" value={value.diffusion_cadence}
          onChange={event => update('diffusion_cadence', Number(event.target.value))}
          inputProps={{ min: 1, max: 16 }}
          error={!!errors.diffusion_cadence}
          helperText={errors.diffusion_cadence || 'Whole number from 1–16.'}
          disabled={disabled} fullWidth
        />
        <TextField
          select label="Color coherence" value={value.color_coherence}
          onChange={event => update('color_coherence', event.target.value as typeof value.color_coherence)}
          disabled={disabled} fullWidth
        >
          <MenuItem value="Match Frame 0 LAB">Match Frame 0 LAB</MenuItem>
          <MenuItem value="None">None</MenuItem>
        </TextField>
      </Stack>

      <Alert severity={runtime?.available ? 'success' : 'warning'}>
        <Typography variant="subtitle2">
          KreaDeforum external runtime {runtime?.available ? 'ready' : 'setup needed'}
        </Typography>
        <Typography variant="caption" display="block">
          Revision {runtime?.revision || 'unknown'} · patch {runtime?.patch_version || 'unknown'} ·
          license {runtime?.license || 'unspecified'}
        </Typography>
        <Typography variant="caption" display="block">
          3D / MiDaS: {runtime?.midas_ready ? 'ready' : runtime?.midas_reason || 'not reported'}
        </Typography>
        {!!runtime?.missing_nodes.length && (
          <Typography variant="caption" display="block">Missing nodes: {runtime.missing_nodes.join(', ')}</Typography>
        )}
        {!!runtime?.incompatible_capabilities.length && (
          <Typography variant="caption" display="block">
            Incompatible: {runtime.incompatible_capabilities.join(', ')}
          </Typography>
        )}
      </Alert>
    </Stack>
  )
}
