import {
  Alert,
  Box,
  Button,
  Chip,
  MenuItem,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import CasinoOutlinedIcon from '@mui/icons-material/CasinoOutlined'
import ImageOutlinedIcon from '@mui/icons-material/ImageOutlined'
import type { PromptRow } from '../../lib/promptTimeline'
import { estimatedQueueTurns } from '../../lib/promptTimeline'
import type { AnimatePresetId } from './presets'
import { calculateRenderedFrames } from './presets'
import PromptTimelineEditor from './PromptTimelineEditor'
import type { AnimateControlProps } from './types'

interface BasicControlsProps extends AnimateControlProps {
  preset: AnimatePresetId
  onPreset: (preset: AnimatePresetId) => void
  initPreview: string
  onInitImage: (file: File | null) => void
  promptRows: PromptRow[]
  onPromptRows: (rows: PromptRow[]) => void
  onAddPromptRow: () => void
  chunkSize?: number
}

export default function BasicControls({
  value,
  errors,
  disabled,
  update,
  preset,
  onPreset,
  initPreview,
  onInitImage,
  promptRows,
  onPromptRows,
  onAddPromptRow,
  chunkSize = 8,
}: BasicControlsProps) {
  const work = calculateRenderedFrames(value)
  const turns = estimatedQueueTurns(work.frames, chunkSize)
  return (
    <Stack spacing={2}>
      <Box>
        <Typography variant="subtitle2" gutterBottom>Quality preset</Typography>
        <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
          {(['draft', 'cinematic', 'smooth', 'custom'] as const).map(id => (
            <Chip
              key={id}
              label={id[0].toUpperCase() + id.slice(1)}
              clickable={!disabled}
              color={preset === id ? 'primary' : 'default'}
              variant={preset === id ? 'filled' : 'outlined'}
              onClick={() => !disabled && onPreset(id)}
              sx={{ minHeight: 44 }}
            />
          ))}
        </Stack>
        <Typography variant="caption" color="text.secondary">
          {value.steps} steps · {value.width}×{value.height} · cadence {value.diffusion_cadence}
        </Typography>
      </Box>

      <PromptTimelineEditor
        rows={promptRows}
        fps={value.fps}
        totalFrames={Math.max(1, work.frames)}
        durationSeconds={value.duration_seconds}
        disabled={disabled}
        error={errors.prompt_schedule}
        onChange={onPromptRows}
        onAdd={onAddPromptRow}
      />

      <TextField
        label="Negative prompt (optional)"
        value={value.negative_prompt}
        onChange={event => update('negative_prompt', event.target.value)}
        multiline
        minRows={2}
        fullWidth
        disabled={disabled}
        helperText="Applies across the whole animation."
      />

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
        <TextField
          id="animate-duration"
          label="Duration (seconds)"
          type="number"
          value={value.duration_seconds}
          onChange={event => update('duration_seconds', Number(event.target.value))}
          inputProps={{ min: 0.5, max: 60, step: 0.5 }}
          error={!!errors.duration_seconds}
          helperText={errors.duration_seconds || '0.5–60 seconds'}
          disabled={disabled}
          fullWidth
        />
        <TextField
          id="animate-fps"
          label="Playback FPS"
          type="number"
          value={value.fps}
          onChange={event => update('fps', Number(event.target.value))}
          inputProps={{ min: 1, max: 60, step: 1 }}
          error={!!errors.fps}
          helperText={errors.fps || '1–60 FPS · prompt times stay in seconds'}
          disabled={disabled}
          fullWidth
        />
      </Stack>
      <Typography variant="body2" color={work.error ? 'error' : 'text.secondary'} aria-live="polite">
        {work.frames} rendered frames · about {work.diffusionFrames} diffusion calls at cadence {value.diffusion_cadence}
        · ~{turns} queue turn{turns === 1 ? '' : 's'}
        {work.error ? ` · ${work.error}` : ''}
      </Typography>

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
        <TextField
          select
          label="Aspect preset"
          value={`${value.width}x${value.height}`}
          onChange={event => {
            const [width, height] = event.target.value.split('x').map(Number)
            update('width', width)
            update('height', height)
          }}
          disabled={disabled}
          fullWidth
        >
          <MenuItem value="768x768">Square · 768²</MenuItem>
          <MenuItem value="1024x576">Landscape · 16:9</MenuItem>
          <MenuItem value="576x1024">Portrait · 9:16</MenuItem>
          <MenuItem value="1024x768">Landscape · 4:3</MenuItem>
          <MenuItem value="768x1024">Portrait · 3:4</MenuItem>
        </TextField>
        <TextField
          id="animate-width"
          label="Width"
          type="number"
          value={value.width}
          onChange={event => update('width', Number(event.target.value))}
          inputProps={{ min: 256, max: 1536, step: 16 }}
          error={!!errors.width}
          helperText={errors.width || '256–1536, divisible by 16'}
          disabled={disabled}
          fullWidth
        />
        <TextField
          id="animate-height"
          label="Height"
          type="number"
          value={value.height}
          onChange={event => update('height', Number(event.target.value))}
          inputProps={{ min: 256, max: 1536, step: 16 }}
          error={!!errors.height}
          helperText={errors.height || '256–1536, divisible by 16'}
          disabled={disabled}
          fullWidth
        />
      </Stack>

      <Box>
        <Typography variant="subtitle2" gutterBottom>Starting frame (optional)</Typography>
        {initPreview ? (
          <Stack spacing={1} alignItems="flex-start">
            <Box
              component="img"
              src={initPreview}
              alt="Starting frame preview"
              sx={{ width: 'min(100%, 320px)', maxHeight: 220, objectFit: 'contain', borderRadius: 2, bgcolor: 'action.hover' }}
            />
            <Button disabled={disabled} onClick={() => onInitImage(null)} color="warning" sx={{ minHeight: 44 }}>Remove starting frame</Button>
          </Stack>
        ) : (
          <Button component="label" variant="outlined" startIcon={<ImageOutlinedIcon />} disabled={disabled} sx={{ minHeight: 44 }}>
            Choose image
            <input
              hidden
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={event => onInitImage(event.target.files?.[0] ?? null)}
            />
          </Button>
        )}
        <Typography variant="caption" color="text.secondary" display="block">
          PNG, JPEG, or WebP; converted locally to a bounded PNG matching the output size.
        </Typography>
      </Box>

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} alignItems={{ sm: 'flex-start' }}>
        <TextField
          id="animate-seed"
          label="Seed"
          type="number"
          value={value.seed}
          onChange={event => update('seed', Number(event.target.value))}
          inputProps={{ min: -1, max: Number.MAX_SAFE_INTEGER, step: 1 }}
          error={!!errors.seed}
          helperText={errors.seed || '-1 chooses a random seed'}
          disabled={disabled}
          fullWidth
        />
        <Button
          variant="outlined"
          startIcon={<CasinoOutlinedIcon />}
          disabled={disabled}
          onClick={() => update('seed', Math.floor(Math.random() * 0xffffffff))}
          sx={{ minHeight: 44, whiteSpace: 'nowrap' }}
        >
          Randomize
        </Button>
      </Stack>
    </Stack>
  )
}
