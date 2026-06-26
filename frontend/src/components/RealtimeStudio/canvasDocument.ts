export type RealtimeTool = 'select' | 'brush' | 'eraser' | 'shape' | 'upload'
export type ShapeKind = 'rectangle' | 'circle' | 'triangle'

export interface Point {
  x: number
  y: number
}

export interface StrokeLayer {
  id: string
  type: 'stroke'
  name: string
  visible: boolean
  color: string
  size: number
  points: Point[]
  note?: string
}

export interface ShapeLayer {
  id: string
  type: 'shape'
  name: string
  visible: boolean
  color: string
  shape: ShapeKind
  x: number
  y: number
  width: number
  height: number
  note?: string
}

export interface ImageLayer {
  id: string
  type: 'image'
  name: string
  visible: boolean
  imageB64: string
  x: number
  y: number
  width: number
  height: number
  note?: string
}

export type RealtimeLayer = StrokeLayer | ShapeLayer | ImageLayer

export interface RealtimeDocument {
  width: number
  height: number
  background: string
  layers: RealtimeLayer[]
}

export function newLayerId(prefix = 'layer'): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

export function createDefaultDocument(): RealtimeDocument {
  return {
    width: 768,
    height: 768,
    background: '#c8c5bf',
    layers: [],
  }
}

export function addStrokePoint(layer: StrokeLayer, point: Point): StrokeLayer {
  return { ...layer, points: [...layer.points, point] }
}

function drawStroke(ctx: CanvasRenderingContext2D, layer: StrokeLayer) {
  if (layer.points.length === 0) return
  ctx.save()
  ctx.strokeStyle = layer.color
  ctx.fillStyle = layer.color
  ctx.lineWidth = layer.size
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.beginPath()
  ctx.moveTo(layer.points[0].x, layer.points[0].y)
  for (const point of layer.points.slice(1)) ctx.lineTo(point.x, point.y)
  ctx.stroke()
  if (layer.points.length === 1) {
    const point = layer.points[0]
    ctx.beginPath()
    ctx.arc(point.x, point.y, layer.size / 2, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.restore()
}

function drawShape(ctx: CanvasRenderingContext2D, layer: ShapeLayer) {
  ctx.save()
  ctx.fillStyle = layer.color
  if (layer.shape === 'circle') {
    ctx.beginPath()
    ctx.ellipse(
      layer.x + layer.width / 2,
      layer.y + layer.height / 2,
      Math.abs(layer.width / 2),
      Math.abs(layer.height / 2),
      0,
      0,
      Math.PI * 2,
    )
    ctx.fill()
  } else if (layer.shape === 'triangle') {
    ctx.beginPath()
    ctx.moveTo(layer.x + layer.width / 2, layer.y)
    ctx.lineTo(layer.x + layer.width, layer.y + layer.height)
    ctx.lineTo(layer.x, layer.y + layer.height)
    ctx.closePath()
    ctx.fill()
  } else {
    ctx.fillRect(layer.x, layer.y, layer.width, layer.height)
  }
  ctx.restore()
}

function drawImageLayer(ctx: CanvasRenderingContext2D, layer: ImageLayer): Promise<void> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      ctx.drawImage(img, layer.x, layer.y, layer.width, layer.height)
      resolve()
    }
    img.onerror = () => reject(new Error(`Could not load ${layer.name}`))
    img.src = `data:image/png;base64,${layer.imageB64}`
  })
}

export async function renderDocumentToCanvas(doc: RealtimeDocument, canvas: HTMLCanvasElement) {
  canvas.width = doc.width
  canvas.height = doc.height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas is unavailable')
  ctx.fillStyle = doc.background
  ctx.fillRect(0, 0, doc.width, doc.height)
  for (const layer of doc.layers) {
    if (!layer.visible) continue
    if (layer.type === 'stroke') drawStroke(ctx, layer)
    else if (layer.type === 'shape') drawShape(ctx, layer)
    else await drawImageLayer(ctx, layer)
  }
}

export async function documentToPngB64(doc: RealtimeDocument): Promise<string> {
  const canvas = document.createElement('canvas')
  await renderDocumentToCanvas(doc, canvas)
  return canvas.toDataURL('image/png').split(',')[1]
}

export function promptFromLayerNotes(doc: RealtimeDocument): string {
  return doc.layers
    .filter(layer => layer.visible && layer.note?.trim())
    .map((layer, index) => `Layer ${index + 1} (${layer.name}): ${layer.note?.trim()}`)
    .join('\n')
}
