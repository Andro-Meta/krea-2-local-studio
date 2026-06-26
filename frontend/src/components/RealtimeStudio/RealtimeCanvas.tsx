import { useCallback, useEffect, useRef, type PointerEvent } from 'react'
import { Box } from '@mui/material'
import {
  addStrokePoint,
  newLayerId,
  renderDocumentToCanvas,
  type ImageLayer,
  type RealtimeDocument,
  type RealtimeLayer,
  type RealtimeTool,
  type ShapeKind,
  type ShapeLayer,
  type StrokeLayer,
} from './canvasDocument'

interface Props {
  document: RealtimeDocument
  selectedLayerId: string | null
  tool: RealtimeTool
  color: string
  brushSize: number
  shape: ShapeKind
  onDocumentChange: (document: RealtimeDocument) => void
  onSelectLayer: (id: string | null) => void
}

function pointFromEvent(e: PointerEvent<HTMLCanvasElement>, canvas: HTMLCanvasElement) {
  const rect = canvas.getBoundingClientRect()
  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  }
}

function layerContains(layer: RealtimeLayer, x: number, y: number): boolean {
  if (layer.type === 'stroke') return false
  return x >= layer.x && y >= layer.y && x <= layer.x + layer.width && y <= layer.y + layer.height
}

function moveLayer(layer: RealtimeLayer, dx: number, dy: number): RealtimeLayer {
  if (layer.type === 'stroke') {
    return { ...layer, points: layer.points.map(point => ({ x: point.x + dx, y: point.y + dy })) }
  }
  return { ...layer, x: layer.x + dx, y: layer.y + dy }
}

export default function RealtimeCanvas({
  document,
  selectedLayerId,
  tool,
  color,
  brushSize,
  shape,
  onDocumentChange,
  onSelectLayer,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const activeStrokeId = useRef<string | null>(null)
  const dragRef = useRef<{ id: string; lastX: number; lastY: number } | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let cancelled = false
    renderDocumentToCanvas(document, canvas).catch(() => {
      if (!cancelled) {
        const ctx = canvas.getContext('2d')
        if (ctx) {
          ctx.fillStyle = document.background
          ctx.fillRect(0, 0, document.width, document.height)
        }
      }
    })
    return () => { cancelled = true }
  }, [document])

  const updateLayer = useCallback((layerId: string, updater: (layer: RealtimeLayer) => RealtimeLayer) => {
    onDocumentChange({
      ...document,
      layers: document.layers.map(layer => layer.id === layerId ? updater(layer) : layer),
    })
  }, [document, onDocumentChange])

  const handlePointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.setPointerCapture(e.pointerId)
    const point = pointFromEvent(e, canvas)

    if (tool === 'select') {
      const hit = [...document.layers].reverse().find(layer => layer.visible && layerContains(layer, point.x, point.y))
      onSelectLayer(hit?.id ?? null)
      if (hit) dragRef.current = { id: hit.id, lastX: point.x, lastY: point.y }
      return
    }

    if (tool === 'shape') {
      const layer: ShapeLayer = {
        id: newLayerId('shape'),
        type: 'shape',
        name: `${shape[0].toUpperCase()}${shape.slice(1)}`,
        visible: true,
        color,
        shape,
        x: Math.max(0, point.x - 70),
        y: Math.max(0, point.y - 70),
        width: 140,
        height: 140,
        note: '',
      }
      onDocumentChange({ ...document, layers: [...document.layers, layer] })
      onSelectLayer(layer.id)
      return
    }

    const layer: StrokeLayer = {
      id: newLayerId(tool),
      type: 'stroke',
      name: tool === 'eraser' ? 'Eraser stroke' : 'Brush stroke',
      visible: true,
      color: tool === 'eraser' ? document.background : color,
      size: tool === 'eraser' ? brushSize * 1.4 : brushSize,
      points: [point],
      note: '',
    }
    activeStrokeId.current = layer.id
    onDocumentChange({ ...document, layers: [...document.layers, layer] })
    onSelectLayer(layer.id)
  }

  const handlePointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const point = pointFromEvent(e, canvas)

    if (dragRef.current) {
      const { id, lastX, lastY } = dragRef.current
      const dx = point.x - lastX
      const dy = point.y - lastY
      dragRef.current = { id, lastX: point.x, lastY: point.y }
      updateLayer(id, layer => moveLayer(layer, dx, dy))
      return
    }

    const strokeId = activeStrokeId.current
    if (!strokeId) return
    updateLayer(strokeId, layer => layer.type === 'stroke' ? addStrokePoint(layer, point) : layer)
  }

  const handlePointerUp = (e: PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (canvas?.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId)
    activeStrokeId.current = null
    dragRef.current = null
  }

  const selected = selectedLayerId
    ? document.layers.find(layer => layer.id === selectedLayerId)
    : null

  return (
    <Box sx={{ position: 'relative', width: '100%', maxWidth: 820, mx: 'auto' }}>
      <canvas
        ref={canvasRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        style={{
          width: '100%',
          aspectRatio: `${document.width} / ${document.height}`,
          borderRadius: 24,
          display: 'block',
          touchAction: 'none',
          cursor: tool === 'select' ? 'grab' : 'crosshair',
          boxShadow: '0 20px 80px rgba(0,0,0,0.28)',
        }}
      />
      {selected && selected.type !== 'stroke' && (
        <Box
          sx={{
            position: 'absolute',
            pointerEvents: 'none',
            border: '2px solid',
            borderColor: 'primary.main',
            borderRadius: 1,
            left: `${(selected.x / document.width) * 100}%`,
            top: `${(selected.y / document.height) * 100}%`,
            width: `${(selected.width / document.width) * 100}%`,
            height: `${(selected.height / document.height) * 100}%`,
          }}
        />
      )}
    </Box>
  )
}

export function makeImageLayer(imageB64: string, doc: RealtimeDocument): ImageLayer {
  const size = Math.round(Math.min(doc.width, doc.height) * 0.55)
  return {
    id: newLayerId('image'),
    type: 'image',
    name: 'Uploaded image',
    visible: true,
    imageB64,
    x: Math.round((doc.width - size) / 2),
    y: Math.round((doc.height - size) / 2),
    width: size,
    height: size,
    note: 'Use this uploaded image as a visual reference.',
  }
}
