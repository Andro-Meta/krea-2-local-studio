import {
  Box, Card, CardContent, IconButton, Stack, Switch, TextField, Tooltip, Typography,
} from '@mui/material'
import DeleteIcon from '@mui/icons-material/Delete'
import VisibilityIcon from '@mui/icons-material/Visibility'
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff'
import type { RealtimeDocument, RealtimeLayer } from './canvasDocument'

interface Props {
  document: RealtimeDocument
  selectedLayerId: string | null
  onDocumentChange: (document: RealtimeDocument) => void
  onSelectLayer: (id: string) => void
}

export default function LayerPanel({ document, selectedLayerId, onDocumentChange, onSelectLayer }: Props) {
  const updateLayer = (id: string, patch: Partial<RealtimeLayer>) => {
    onDocumentChange({
      ...document,
      layers: document.layers.map(layer => layer.id === id ? { ...layer, ...patch } as RealtimeLayer : layer),
    })
  }

  const deleteLayer = (id: string) => {
    onDocumentChange({ ...document, layers: document.layers.filter(layer => layer.id !== id) })
  }

  if (!document.layers.length) {
    return (
      <Card variant="outlined" sx={{ borderRadius: 3 }}>
        <CardContent>
          <Typography variant="subtitle2">Layers</Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.75 }}>
            Paint, add shapes, or upload images. Notes here become part of the preview prompt.
          </Typography>
        </CardContent>
      </Card>
    )
  }

  return (
    <Stack spacing={1}>
      <Typography variant="subtitle2">Layers</Typography>
      {[...document.layers].reverse().map(layer => {
        const selected = layer.id === selectedLayerId
        return (
          <Card
            key={layer.id}
            variant="outlined"
            onClick={() => onSelectLayer(layer.id)}
            sx={{
              borderRadius: 3,
              cursor: 'pointer',
              borderColor: selected ? 'primary.main' : 'divider',
              bgcolor: selected ? 'action.selected' : 'background.paper',
            }}
          >
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Stack spacing={1}>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Switch
                    checked={layer.visible}
                    size="small"
                    icon={<VisibilityOffIcon fontSize="small" />}
                    checkedIcon={<VisibilityIcon fontSize="small" />}
                    onChange={e => updateLayer(layer.id, { visible: e.target.checked })}
                    onClick={e => e.stopPropagation()}
                  />
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Typography variant="body2" noWrap>{layer.name}</Typography>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>{layer.type}</Typography>
                  </Box>
                  <Tooltip title="Delete layer">
                    <IconButton
                      size="small"
                      color="error"
                      onClick={e => {
                        e.stopPropagation()
                        deleteLayer(layer.id)
                      }}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Stack>
                <TextField
                  label="Layer note"
                  value={layer.note ?? ''}
                  size="small"
                  multiline
                  minRows={2}
                  placeholder="e.g. green branch shape, use this as an object"
                  onClick={e => e.stopPropagation()}
                  onChange={e => updateLayer(layer.id, { note: e.target.value })}
                />
              </Stack>
            </CardContent>
          </Card>
        )
      })}
    </Stack>
  )
}
