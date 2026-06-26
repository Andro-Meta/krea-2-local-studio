import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Box, Button, Chip, IconButton, Stack, TextField, Tooltip, Typography } from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import DeleteIcon from '@mui/icons-material/Delete'
import GridOnIcon from '@mui/icons-material/GridOn'
import { useStore } from '../../store'

interface BBox { label: string; bbox: [number, number, number, number]; id: string }

export default function CanvasControl() {
  const { params, setParam } = useStore()
  const [open, setOpen] = useState(false)
  const [boxes, setBoxes] = useState<BBox[]>([])
  const [drawing, setDrawing] = useState<{ x: number; y: number } | null>(null)
  const [currentBox, setCurrentBox] = useState<[number, number, number, number] | null>(null)
  const [newLabel, setNewLabel] = useState('')
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const W = params.width
  const H = params.height
  const scale = Math.min(280 / W, 200 / H)
  const cw = Math.round(W * scale)
  const ch = Math.round(H * scale)

  const toNorm = (x: number, y: number) => [x / cw, y / ch] as [number, number]

  const draw = useCallback(() => {
    const c = canvasRef.current
    if (!c) return
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, cw, ch)
    ctx.fillStyle = 'rgba(255,255,255,0.04)'
    ctx.fillRect(0, 0, cw, ch)
    ctx.strokeStyle = 'rgba(208,188,255,0.3)'
    ctx.lineWidth = 1
    ctx.strokeRect(0, 0, cw, ch)

    boxes.forEach((b, i) => {
      const [x1, y1, x2, y2] = b.bbox
      const rx = x1 * cw, ry = y1 * ch, rw = (x2 - x1) * cw, rh = (y2 - y1) * ch
      ctx.strokeStyle = `hsl(${(i * 67) % 360},70%,65%)`
      ctx.lineWidth = 2
      ctx.strokeRect(rx, ry, rw, rh)
      ctx.fillStyle = ctx.strokeStyle
      ctx.font = '11px Roboto'
      ctx.fillText(b.label, rx + 3, ry + 13)
    })

    if (currentBox && drawing) {
      const [x1, y1, x2, y2] = currentBox
      ctx.strokeStyle = '#D0BCFF'
      ctx.lineWidth = 2
      ctx.setLineDash([4, 4])
      ctx.strokeRect(x1 * cw, y1 * ch, (x2 - x1) * cw, (y2 - y1) * ch)
      ctx.setLineDash([])
    }
  }, [boxes, currentBox, drawing, cw, ch])

  useEffect(() => { draw() }, [draw])

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = canvasRef.current!.getBoundingClientRect()
    const x = e.clientX - r.left, y = e.clientY - r.top
    setDrawing({ x, y })
  }

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing) return
    const r = canvasRef.current!.getBoundingClientRect()
    const x = e.clientX - r.left, y = e.clientY - r.top
    const [nx1, ny1] = toNorm(Math.min(drawing.x, x), Math.min(drawing.y, y))
    const [nx2, ny2] = toNorm(Math.max(drawing.x, x), Math.max(drawing.y, y))
    setCurrentBox([nx1, ny1, nx2, ny2])
  }

  const onMouseUp = () => {
    if (!currentBox) return
    const label = newLabel.trim() || `Region ${boxes.length + 1}`
    const next = [...boxes, { label, bbox: currentBox, id: Math.random().toString(36).slice(2) }]
    setBoxes(next)
    setParam('bboxes', next.map(b => ({ label: b.label, bbox: b.bbox })))
    setCurrentBox(null)
    setDrawing(null)
    setNewLabel('')
  }

  const removeBox = (id: string) => {
    const next = boxes.filter(b => b.id !== id)
    setBoxes(next)
    setParam('bboxes', next.map(b => ({ label: b.label, bbox: b.bbox })))
  }

  const clearAll = () => { setBoxes([]); setParam('bboxes', []) }

  if (!open) {
    return (
      <Button
        startIcon={<GridOnIcon />}
        variant="outlined"
        size="small"
        onClick={() => setOpen(true)}
        sx={{ alignSelf: 'flex-start' }}
      >
        JSON Canvas {params.bboxes.length > 0 && `(${params.bboxes.length} regions)`}
      </Button>
    )
  }

  return (
    <Box sx={{ border: '1px solid rgba(202,196,208,0.2)', borderRadius: 2, p: 1.5 }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="caption" sx={{ textTransform: 'uppercase', letterSpacing: 1, color: 'text.secondary' }}>
          Spatial Canvas
        </Typography>
        <Stack direction="row" spacing={0.5}>
          {boxes.length > 0 && <Button size="small" onClick={clearAll} color="error" sx={{ minWidth: 0 }}>Clear</Button>}
          <Button size="small" onClick={() => setOpen(false)}>Done</Button>
        </Stack>
      </Stack>
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
        <Box>
          <canvas
            ref={canvasRef}
            width={cw} height={ch}
            style={{ cursor: 'crosshair', borderRadius: 8, display: 'block' }}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
          />
          <TextField
            size="small" placeholder="Region label" value={newLabel}
            onChange={e => setNewLabel(e.target.value)}
            sx={{ mt: 0.75, width: cw }}
          />
        </Box>
        <Stack spacing={0.5} sx={{ flex: 1, minWidth: 0 }}>
          {boxes.map(b => (
            <Stack key={b.id} direction="row" alignItems="center" spacing={0.5}>
              <Chip label={b.label} size="small" sx={{ flex: 1, justifyContent: 'flex-start' }} />
              <IconButton size="small" onClick={() => removeBox(b.id)}>
                <DeleteIcon fontSize="small" />
              </IconButton>
            </Stack>
          ))}
          {!boxes.length && (
            <Typography variant="caption" sx={{ color: 'text.disabled' }}>
              Draw boxes on the canvas to define spatial regions.
            </Typography>
          )}
        </Stack>
      </Stack>
    </Box>
  )
}
