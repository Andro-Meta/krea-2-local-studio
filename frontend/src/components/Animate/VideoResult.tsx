import { Box, Button, Paper, Stack, Typography } from '@mui/material'
import DownloadOutlinedIcon from '@mui/icons-material/DownloadOutlined'
import CollectionsOutlinedIcon from '@mui/icons-material/CollectionsOutlined'
import type { AnimationResult } from '../../api'
import { publicUrl } from '../../api'

export default function VideoResult({
  result,
  downloading,
  onDownload,
  onOpenGallery,
}: {
  result: AnimationResult
  downloading: boolean
  onDownload: () => void
  onOpenGallery: () => void
}) {
  const videoUrl = publicUrl(result.video_url)
  const posterUrl = publicUrl(result.poster_url)
  return (
    <Paper variant="outlined" sx={{ p: { xs: 1.5, sm: 2 }, overflow: 'hidden' }}>
      <Stack spacing={1.5}>
        <Box>
          <Typography variant="h6">Animation ready</Typography>
          <Typography variant="body2" color="text.secondary">
            {result.frame_count} frames · {result.fps} FPS · {result.duration.toFixed(1)}s
          </Typography>
        </Box>
        <Box
          component="img"
          src={posterUrl}
          alt="Animation poster frame"
          sx={{ width: 'min(100%, 320px)', maxHeight: 240, objectFit: 'contain', display: 'block', borderRadius: 2, bgcolor: 'black' }}
        />
        <Box
          component="video"
          controls
          playsInline
          preload="metadata"
          poster={posterUrl}
          src={videoUrl}
          sx={{ width: '100%', maxHeight: 640, display: 'block', bgcolor: 'black', borderRadius: 2 }}
        />
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
          <Button
            variant="outlined"
            startIcon={<DownloadOutlinedIcon />}
            onClick={onDownload}
            disabled={downloading}
            sx={{ minHeight: 44 }}
          >
            {downloading ? 'Preparing download…' : 'Download MP4'}
          </Button>
          <Button
            variant="outlined"
            startIcon={<CollectionsOutlinedIcon />}
            onClick={onOpenGallery}
            sx={{ minHeight: 44 }}
          >
            Open gallery
          </Button>
        </Stack>
      </Stack>
    </Paper>
  )
}
