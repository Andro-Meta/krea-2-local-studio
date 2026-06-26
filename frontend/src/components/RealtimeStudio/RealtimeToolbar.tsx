import { useRef, type ChangeEvent } from 'react'
import {
  Box, Divider, IconButton, Slider, Stack, ToggleButton, ToggleButtonGroup, Tooltip,
} from '@mui/material'
import ArrowPointerIcon from '@mui/icons-material/NearMe'
import BrushIcon from '@mui/icons-material/Brush'
import CameraAltIcon from '@mui/icons-material/CameraAlt'
import CropSquareIcon from '@mui/icons-material/CropSquare'
import DeleteIcon from '@mui/icons-material/Delete'
import FileUploadIcon from '@mui/icons-material/FileUpload'
import FormatColorFillIcon from '@mui/icons-material/FormatColorFill'
import RadioButtonUncheckedIcon from '@mui/icons-material/RadioButtonUnchecked'
import ChangeHistoryIcon from '@mui/icons-material/ChangeHistory'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import type { RealtimeTool, ShapeKind } from './canvasDocument'

interface Props {
  tool: RealtimeTool
  color: string
  brushSize: number
  shape: ShapeKind
  onTool: (tool: RealtimeTool) => void
  onColor: (color: string) => void
  onBrushSize: (size: number) => void
  onShape: (shape: ShapeKind) => void
  onUpload: (b64: string) => void
  onClear: () => void
  onExport: () => void
}

function readFileB64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error('Could not read image'))
    reader.onload = ev => resolve(String(ev.target?.result ?? '').split(',')[1])
    reader.readAsDataURL(file)
  })
}

export default function RealtimeToolbar({
  tool,
  color,
  brushSize,
  shape,
  onTool,
  onColor,
  onBrushSize,
  onShape,
  onUpload,
  onClear,
  onExport,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const handleFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    onUpload(await readFileB64(file))
    e.target.value = ''
  }

  return (
    <Box
      sx={{
        position: 'sticky',
        bottom: 12,
        zIndex: 5,
        mx: 'auto',
        width: 'fit-content',
        maxWidth: '100%',
        borderRadius: 999,
        bgcolor: 'rgba(45,45,48,0.92)',
        backdropFilter: 'blur(18px)',
        px: 1,
        py: 0.75,
        boxShadow: '0 12px 36px rgba(0,0,0,0.38)',
      }}
    >
      <input ref={fileInputRef} hidden type="file" accept="image/*" onChange={handleFile} />
      <Stack direction="row" alignItems="center" spacing={0.5} sx={{ overflowX: 'auto' }}>
        <ToggleButtonGroup
          value={tool}
          exclusive
          onChange={(_, value) => value && onTool(value)}
          size="small"
          sx={{
            '& .MuiToggleButton-root': {
              color: 'common.white',
              minWidth: 44,
              minHeight: 44,
              border: 0,
              borderRadius: 999,
            },
            '& .Mui-selected': {
              bgcolor: 'rgba(208,188,255,0.28) !important',
              color: 'common.white !important',
            },
          }}
        >
          <ToggleButton value="select"><Tooltip title="Select / move"><ArrowPointerIcon /></Tooltip></ToggleButton>
          <ToggleButton value="brush"><Tooltip title="Paint brush"><BrushIcon /></Tooltip></ToggleButton>
          <ToggleButton value="eraser"><Tooltip title="Eraser"><AutoFixHighIcon /></Tooltip></ToggleButton>
          <ToggleButton value="shape"><Tooltip title="Shape tool"><CropSquareIcon /></Tooltip></ToggleButton>
        </ToggleButtonGroup>

        {tool === 'shape' && (
          <ToggleButtonGroup
            value={shape}
            exclusive
            onChange={(_, value) => value && onShape(value)}
            size="small"
            sx={{ '& .MuiToggleButton-root': { color: 'common.white', minWidth: 38, border: 0, borderRadius: 999 } }}
          >
            <ToggleButton value="rectangle"><CropSquareIcon fontSize="small" /></ToggleButton>
            <ToggleButton value="circle"><RadioButtonUncheckedIcon fontSize="small" /></ToggleButton>
            <ToggleButton value="triangle"><ChangeHistoryIcon fontSize="small" /></ToggleButton>
          </ToggleButtonGroup>
        )}

        <Divider orientation="vertical" flexItem sx={{ borderColor: 'rgba(255,255,255,0.18)', mx: 0.5 }} />

        <Tooltip title="Brush color">
          <IconButton component="label" sx={{ color: 'common.white', minWidth: 44, minHeight: 44 }}>
            <FormatColorFillIcon />
            <input hidden type="color" value={color} onChange={e => onColor(e.target.value)} />
          </IconButton>
        </Tooltip>

        <Box sx={{ width: { xs: 88, sm: 140 }, px: 1 }}>
          <Slider
            value={brushSize}
            min={4}
            max={120}
            step={2}
            onChange={(_, value) => onBrushSize(value as number)}
            size="small"
            aria-label="Brush size"
          />
        </Box>

        <Tooltip title="Upload image">
          <IconButton sx={{ color: 'common.white', minWidth: 44, minHeight: 44 }} onClick={() => fileInputRef.current?.click()}>
            <FileUploadIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Export canvas">
          <IconButton sx={{ color: 'common.white', minWidth: 44, minHeight: 44 }} onClick={onExport}>
            <CameraAltIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Clear canvas">
          <IconButton sx={{ color: 'error.light', minWidth: 44, minHeight: 44 }} onClick={onClear}>
            <DeleteIcon />
          </IconButton>
        </Tooltip>
      </Stack>
    </Box>
  )
}
