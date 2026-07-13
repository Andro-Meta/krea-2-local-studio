import {
  Alert,
  Button,
  LinearProgress,
  Paper,
  Stack,
  Typography,
} from '@mui/material'
import type { AnimateTaskState } from '../../lib/animateTaskState'

export default function AnimationProgress({
  task,
  connectionNote,
  onCancel,
}: {
  task: AnimateTaskState
  connectionNote: string
  onCancel: () => void
}) {
  const active = task.active
  if (!active) return task.error ? <Alert severity={task.status === 'blocked' ? 'warning' : 'error'} role="alert">{task.error}</Alert> : null
  const snapshot = active.snapshot
  const cancelPending = active.cancelPending || snapshot.status === 'cancellation_requested'
  const queued = snapshot.status === 'queued'
  const finalizing = snapshot.status === 'finalizing'
  const frameText = snapshot.total_frames
    ? `${snapshot.completed_frames ?? 0}/${snapshot.total_frames} frames`
    : ''
  const announcement = cancelPending
    ? 'Animation cancellation requested.'
    : queued
      ? `Animation queued${snapshot.queue_position ? ` at position ${snapshot.queue_position}` : ''}.`
      : finalizing
        ? 'Animation frames are complete. Finalizing the video.'
        : `Animation rendering${frameText ? `, ${frameText}` : ''}.`
  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Stack spacing={1.25}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" gap={1}>
          <Typography variant="subtitle1">
            {queued ? 'Queued' : finalizing ? 'Finalizing video' : cancelPending ? 'Cancelling' : 'Rendering animation'}
          </Typography>
          <Typography variant="body2" sx={{ fontVariantNumeric: 'tabular-nums' }}>
            {Math.round(snapshot.progress)}%
          </Typography>
        </Stack>
        <LinearProgress
          variant={queued || finalizing ? 'indeterminate' : 'determinate'}
          value={snapshot.progress}
          aria-label="Animation rendering progress"
        />
        <Typography variant="body2" color="text.secondary" aria-live="polite">{announcement}</Typography>
        {(frameText || snapshot.queue_length || snapshot.chunk_index != null) && (
          <Typography variant="caption" color="text.secondary">
            {[
              frameText,
              snapshot.chunk_index != null ? `chunk ${snapshot.chunk_index + 1}` : '',
              snapshot.queue_length ? `${snapshot.queue_length} queued tasks total` : '',
            ].filter(Boolean).join(' · ')}
          </Typography>
        )}
        {active.restored && <Typography variant="caption" color="info.main">Restored after refresh.</Typography>}
        {active.submission.videoTransferred && (
          <Typography variant="caption" color="success.main">Video transferred to queued animation.</Typography>
        )}
        {connectionNote && <Alert severity="info" sx={{ py: 0 }}>{connectionNote}</Alert>}
        <Button
          variant="outlined"
          color="warning"
          onClick={onCancel}
          disabled={cancelPending || finalizing}
          sx={{ minHeight: 44, alignSelf: 'flex-start' }}
        >
          {cancelPending ? 'Cancellation requested…' : 'Cancel animation'}
        </Button>
      </Stack>
    </Paper>
  )
}
