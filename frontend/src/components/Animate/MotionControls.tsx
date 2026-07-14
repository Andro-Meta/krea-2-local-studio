import {
  Alert,
  Box,
  Button,
  Chip,
  LinearProgress,
  MenuItem,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import UploadFileOutlinedIcon from '@mui/icons-material/UploadFileOutlined'
import type { AnimateRequest, AnimationUploadResponse, KreaDeforumStatus } from '../../api'
import {
  MOTION_PRESETS,
  buildEndpointSchedule,
  calculateRenderedFrames,
  parseScheduleEndpoints,
  type MotionPresetId,
} from './presets'
import type { AnimateControlProps } from './types'

interface MotionControlsProps extends AnimateControlProps {
  upload: AnimationUploadResponse | null
  uploadProgress: number
  uploading: boolean
  uploadError: string
  allowedVideoTypes: string[]
  onVideo: (file: File | null) => void
  runtime: KreaDeforumStatus
  onMotionPreset: (preset: MotionPresetId) => void
}

const MODE_HELP: Record<string, string> = {
  '2D': 'Pan, zoom, and rotate a flat frame.',
  '3D': 'Adds depth-aware translation and rotation; MiDaS support may be required.',
  'Video Input': 'Uses an uploaded source video for hybrid motion.',
  'None': 'Keeps the camera static while diffusion evolves the image.',
}

export default function MotionControls({
  value,
  errors,
  disabled,
  update,
  upload,
  uploadProgress,
  uploading,
  uploadError,
  allowedVideoTypes,
  onVideo,
  runtime,
  onMotionPreset,
}: MotionControlsProps) {
  const frames = calculateRenderedFrames(value).frames
  const framesValid = Number.isInteger(frames) && frames > 0 && frames <= 720
  type MotionField =
    | 'zoom_schedule'
    | 'angle_schedule'
    | 'translation_x_schedule'
    | 'translation_y_schedule'
    | 'translation_z_schedule'
    | 'rotation_3d_x_schedule'
    | 'rotation_3d_y_schedule'
    | 'rotation_3d_z_schedule'
  const structured: Array<{ field: MotionField; label: string; fallback: number }> = value.animation_mode === '3D'
    ? [
        { field: 'translation_z_schedule', label: 'Translation Z', fallback: 0 },
        { field: 'rotation_3d_x_schedule', label: 'Pitch', fallback: 0 },
        { field: 'rotation_3d_y_schedule', label: 'Yaw', fallback: 0 },
        { field: 'rotation_3d_z_schedule', label: 'Roll', fallback: 0 },
      ]
    : [
        { field: 'zoom_schedule', label: 'Zoom', fallback: 1 },
        { field: 'translation_x_schedule', label: 'Pan X', fallback: 0 },
        { field: 'translation_y_schedule', label: 'Pan Y', fallback: 0 },
        { field: 'angle_schedule', label: 'Angle', fallback: 0 },
      ]
  const setEndpoint = (field: MotionField, side: 'start' | 'end', next: number) => {
    if (!framesValid || !Number.isFinite(next)) return
    const current = parseScheduleEndpoints(value[field]) ?? {
      start: structured.find(item => item.field === field)?.fallback ?? 0,
      end: structured.find(item => item.field === field)?.fallback ?? 0,
    }
    update(field, buildEndpointSchedule(
      side === 'start' ? next : current.start,
      side === 'end' ? next : current.end,
      frames,
    ) as AnimateRequest[typeof field])
  }
  return (
    <Stack spacing={2}>
      <TextField
        select
        label="Animation mode"
        value={value.animation_mode}
        onChange={event => update('animation_mode', event.target.value as typeof value.animation_mode)}
        helperText={MODE_HELP[value.animation_mode]}
        disabled={disabled}
        fullWidth
      >
        {['2D', '3D', 'Video Input', 'None'].map(mode => (
          <MenuItem key={mode} value={mode} disabled={mode === '3D' && !runtime.midas_ready}>
            {mode}{mode === '3D' && !runtime.midas_ready ? ' · setup required' : ''}
          </MenuItem>
        ))}
      </TextField>
      {value.animation_mode === '3D' && !runtime.midas_ready && (
        <Alert severity="warning" sx={{ py: 0 }}>
          {runtime.midas_reason} Open System → KreaDeforum / Animate to finish MiDaS setup, then restart ComfyUI.
        </Alert>
      )}
      {!runtime.midas_ready && value.animation_mode !== '3D' && (
        <Alert severity="info" sx={{ py: 0 }}>
          2D is ready. 3D stays gated until MiDaS is set up under System → KreaDeforum / Animate.
        </Alert>
      )}

      <Box>
        <Typography variant="subtitle2" gutterBottom>Motion preset</Typography>
        <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
          {(Object.entries(MOTION_PRESETS) as Array<[MotionPresetId, typeof MOTION_PRESETS[MotionPresetId]]>).map(([id, item]) => (
            <Chip
              key={id}
              label={item.label}
              clickable={!disabled && (!item.requires3d || runtime.midas_ready)}
              disabled={disabled || (item.requires3d && !runtime.midas_ready)}
              onClick={() => onMotionPreset(id)}
              variant="outlined"
              sx={{ minHeight: 44 }}
            />
          ))}
        </Stack>
      </Box>

      {(value.animation_mode === '2D' || value.animation_mode === '3D' || value.animation_mode === 'None') && (
        <Box>
          <Typography variant="subtitle2" gutterBottom>Start / end motion</Typography>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
            {framesValid
              ? `These controls write frame 0 and frame ${frames - 1} schedules. Expert raw fields remain available in Timeline.`
              : 'Fix duration, FPS, or rendered-frame errors to edit structured motion.'}
          </Typography>
          <Stack spacing={1.5}>
            {structured.map(item => {
              const endpoints = parseScheduleEndpoints(value[item.field]) ?? { start: item.fallback, end: item.fallback }
              return (
                <Stack key={item.field} direction={{ xs: 'column', sm: 'row' }} spacing={1}>
                  <TextField
                    label={`${item.label} start`}
                    type="number"
                    value={endpoints.start}
                    onChange={event => setEndpoint(item.field, 'start', Number(event.target.value))}
                    disabled={disabled || !framesValid}
                    fullWidth
                  />
                  <TextField
                    label={`${item.label} end`}
                    type="number"
                    value={endpoints.end}
                    onChange={event => setEndpoint(item.field, 'end', Number(event.target.value))}
                    disabled={disabled || !framesValid}
                    fullWidth
                  />
                </Stack>
              )
            })}
          </Stack>
        </Box>
      )}

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
        <TextField
          select label="Border mode" value={value.border_mode}
          onChange={event => update('border_mode', event.target.value as typeof value.border_mode)}
          disabled={disabled} fullWidth helperText="How exposed edges are filled."
        >
          {['replicate', 'reflect', 'wrap', 'black'].map(item => <MenuItem key={item} value={item}>{item}</MenuItem>)}
        </TextField>
        <TextField
          select label="Seed behavior" value={value.seed_behavior}
          onChange={event => update('seed_behavior', event.target.value as typeof value.seed_behavior)}
          disabled={disabled} fullWidth helperText="How seed changes between frames."
        >
          {['fixed', 'iter', 'random', 'ladder'].map(item => <MenuItem key={item} value={item}>{item}</MenuItem>)}
        </TextField>
      </Stack>

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
        <TextField
          select label="Color coherence" value={value.color_coherence}
          onChange={event => update('color_coherence', event.target.value as typeof value.color_coherence)}
          disabled={disabled} fullWidth
        >
          <MenuItem value="Match Frame 0 LAB">Match Frame 0 LAB</MenuItem>
          <MenuItem value="None">None</MenuItem>
        </TextField>
        <TextField
          id="animate-diffusion-cadence"
          label="Diffusion cadence" type="number" value={value.diffusion_cadence}
          onChange={event => update('diffusion_cadence', Number(event.target.value))}
          inputProps={{ min: 1, max: 16, step: 1 }}
          error={!!errors.diffusion_cadence}
          helperText={errors.diffusion_cadence || 'Render between diffusion calls (1–16).'} disabled={disabled} fullWidth
        />
      </Stack>

      {value.animation_mode === 'Video Input' && (
        <Stack spacing={1}>
          <Typography variant="subtitle2">Source video</Typography>
          {upload ? (
            <>
              <Alert severity="success">
                Uploaded {upload.width}×{upload.height} · {upload.frame_count} frames · {upload.duration.toFixed(1)}s
              </Alert>
              <Button color="warning" onClick={() => onVideo(null)} disabled={disabled || uploading} sx={{ alignSelf: 'flex-start', minHeight: 44 }}>
                Remove source video
              </Button>
            </>
          ) : (
            <Button
              component="label"
              variant="outlined"
              startIcon={<UploadFileOutlinedIcon />}
              disabled={disabled || uploading}
              sx={{ minHeight: 44, alignSelf: 'flex-start' }}
            >
              {uploading ? 'Uploading source…' : 'Upload source video'}
              <input
                hidden type="file"
                accept={allowedVideoTypes.join(',') || 'video/mp4,video/webm,video/quicktime'}
                onChange={event => onVideo(event.target.files?.[0] ?? null)}
              />
            </Button>
          )}
          {uploading && <LinearProgress variant="determinate" value={uploadProgress} aria-label="Source video upload progress" />}
          <Typography variant="caption" color={errors.source_video_upload_id || uploadError ? 'error' : 'text.secondary'}>
            {errors.source_video_upload_id || uploadError || 'The server validates duration, format, dimensions, and account quota.'}
          </Typography>
        </Stack>
      )}

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
        <TextField
          select label="Hybrid mode" value={value.hybrid_mode}
          onChange={event => update('hybrid_mode', event.target.value as typeof value.hybrid_mode)}
          disabled={disabled} fullWidth
        >
          <MenuItem value="normal">normal</MenuItem>
          <MenuItem value="optical_flow">optical_flow</MenuItem>
        </TextField>
        <TextField
          label="Hybrid strength schedule"
          value={value.hybrid_strength_schedule}
          onChange={event => update('hybrid_strength_schedule', event.target.value)}
          error={!!errors.hybrid_strength_schedule}
          helperText={errors.hybrid_strength_schedule || 'Example: 0:(0.5), 24:(0.7)'}
          disabled={disabled}
          fullWidth
        />
      </Stack>
    </Stack>
  )
}
