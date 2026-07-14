import AddIcon from '@mui/icons-material/Add'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import {
  Box,
  Button,
  IconButton,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material'
import type { PromptRow } from '../../lib/promptTimeline'
import { secondsToFrame } from '../../lib/promptTimeline'

interface PromptTimelineEditorProps {
  rows: PromptRow[]
  fps: number
  totalFrames: number
  durationSeconds: number
  disabled?: boolean
  error?: string
  onChange: (rows: PromptRow[]) => void
  onAdd: () => void
}

export default function PromptTimelineEditor({
  rows,
  fps,
  totalFrames,
  durationSeconds,
  disabled,
  error,
  onChange,
  onAdd,
}: PromptTimelineEditorProps) {
  const updateRow = (id: string, patch: Partial<PromptRow>) => {
    onChange(rows.map(row => (row.id === id ? { ...row, ...patch } : row)))
  }

  const removeRow = (id: string) => {
    if (rows.length <= 1) {
      onChange([{ ...rows[0], seconds: 0, prompt: '' }])
      return
    }
    onChange(rows.filter(row => row.id !== id))
  }

  const maxSeconds = Math.max(0.1, durationSeconds)

  return (
    <Stack spacing={1.5}>
      <Typography variant="subtitle2">Prompt timeline</Typography>
      <Box
        role="img"
        aria-label="Prompt fire times across the animation duration"
        sx={{
          position: 'relative',
          height: 36,
          borderRadius: 1,
          bgcolor: 'action.hover',
          border: '1px solid',
          borderColor: 'divider',
          overflow: 'hidden',
        }}
      >
        <Typography
          variant="caption"
          sx={{ position: 'absolute', left: 8, top: 2, color: 'text.secondary' }}
        >
          0s
        </Typography>
        <Typography
          variant="caption"
          sx={{ position: 'absolute', right: 8, top: 2, color: 'text.secondary' }}
        >
          {durationSeconds.toFixed(1)}s
        </Typography>
        {rows.filter(row => row.prompt.trim()).map(row => {
          const pct = durationSeconds > 0
            ? Math.min(100, Math.max(0, (row.seconds / durationSeconds) * 100))
            : 0
          return (
            <Tooltip
              key={row.id}
              title={`${row.seconds.toFixed(2)}s · frame ${secondsToFrame(row.seconds, fps, totalFrames)}`}
            >
              <Box
                sx={{
                  position: 'absolute',
                  left: `calc(${pct}% - 5px)`,
                  bottom: 4,
                  width: 10,
                  height: 14,
                  borderRadius: 0.5,
                  bgcolor: 'primary.main',
                }}
              />
            </Tooltip>
          )
        })}
      </Box>

      {rows.map((row, index) => {
        const frame = secondsToFrame(row.seconds, fps, totalFrames)
        return (
          <Stack
            key={row.id}
            direction={{ xs: 'column', sm: 'row' }}
            spacing={1}
            alignItems={{ sm: 'flex-start' }}
          >
            <TextField
              label="Time (s)"
              type="number"
              value={row.seconds}
              onChange={event => updateRow(row.id, { seconds: Number(event.target.value) })}
              inputProps={{ min: 0, max: maxSeconds, step: 0.1 }}
              disabled={disabled}
              sx={{ width: { sm: 120 }, minWidth: 100 }}
              helperText={`frame ${frame}`}
            />
            <TextField
              label={index === 0 ? 'Prompt' : `Prompt ${index + 1}`}
              value={row.prompt}
              onChange={event => updateRow(row.id, { prompt: event.target.value })}
              disabled={disabled}
              fullWidth
              multiline
              minRows={1}
            />
            <IconButton
              aria-label={`Remove prompt at ${row.seconds}s`}
              disabled={disabled}
              onClick={() => removeRow(row.id)}
              sx={{ minWidth: 44, minHeight: 44 }}
            >
              <DeleteOutlineIcon />
            </IconButton>
          </Stack>
        )
      })}

      <Button
        startIcon={<AddIcon />}
        onClick={onAdd}
        disabled={disabled}
        sx={{ alignSelf: 'flex-start', minHeight: 44 }}
      >
        Add prompt
      </Button>
      {error ? (
        <Typography variant="caption" color="error">{error}</Typography>
      ) : (
        <Typography variant="caption" color="text.secondary">
          Times stay fixed when you change FPS; frames update as round(seconds × FPS).
        </Typography>
      )}
    </Stack>
  )
}
