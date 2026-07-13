import { Alert, Stack, TextField, Typography } from '@mui/material'
import type { AnimateRequest } from '../../api'
import type { AnimateControlProps } from './types'

const SCHEDULES: Array<{
  key: keyof AnimateRequest
  label: string
  modes?: AnimateRequest['animation_mode'][]
  example: string
}> = [
  { key: 'zoom_schedule', label: 'Zoom', modes: ['2D'], example: '0:(1.0), 24:(1.04)' },
  { key: 'angle_schedule', label: 'Angle', modes: ['2D'], example: '0:(0), 24:(2)' },
  { key: 'translation_x_schedule', label: 'Translation X', modes: ['2D', '3D'], example: '0:(0), 24:(12)' },
  { key: 'translation_y_schedule', label: 'Translation Y', modes: ['2D', '3D'], example: '0:(0), 24:(-8)' },
  { key: 'translation_z_schedule', label: 'Translation Z', modes: ['3D'], example: '0:(0), 24:(2)' },
  { key: 'rotation_3d_x_schedule', label: '3D rotation X', modes: ['3D'], example: '0:(0), 24:(1.5)' },
  { key: 'rotation_3d_y_schedule', label: '3D rotation Y', modes: ['3D'], example: '0:(0), 24:(2)' },
  { key: 'rotation_3d_z_schedule', label: '3D rotation Z', modes: ['3D'], example: '0:(0), 24:(0.5)' },
  { key: 'strength_schedule', label: 'Denoise strength', example: '0:(0.65), 24:(0.58)' },
  { key: 'cfg_schedule', label: 'CFG', example: '0:(1.0)' },
]

export default function TimelineControls({
  value,
  errors,
  disabled,
  update,
}: AnimateControlProps) {
  const visible = SCHEDULES.filter(item => !item.modes || item.modes.includes(value.animation_mode))
  return (
    <Stack spacing={2}>
      <Alert severity="info" sx={{ py: 0 }}>
        Raw schedules use comma-separated <b>frame:(expression)</b> entries. Values are interpolated; prompt keyframes remain untouched.
      </Alert>
      <Typography variant="body2" color="text.secondary">
        Showing controls relevant to {value.animation_mode}. Hidden mode-specific values are preserved.
      </Typography>
      {visible.map(item => {
        const fieldValue = value[item.key]
        return (
          <TextField
            key={item.key}
            id={`animate-${String(item.key).replace(/_/g, '-')}`}
            label={item.label}
            value={typeof fieldValue === 'string' ? fieldValue : ''}
            onChange={event => update(item.key, event.target.value as never)}
            error={!!errors[item.key]}
            helperText={errors[item.key] || `Example: ${item.example}`}
            disabled={disabled}
            fullWidth
          />
        )
      })}
    </Stack>
  )
}
