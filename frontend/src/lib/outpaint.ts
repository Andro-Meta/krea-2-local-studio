import { stripDataUrl } from './imageActions'

export interface OutpaintPads {
  left: number
  right: number
  top: number
  bottom: number
}

export interface OutpaintResult {
  init_image_b64: string
  mask_b64: string
  width: number
  height: number
}

function loadImage(b64: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Could not load source image'))
    img.src = `data:image/png;base64,${stripDataUrl(b64)}`
  })
}

function align16(value: number): number {
  return Math.max(16, Math.ceil(value / 16) * 16)
}

export async function buildOutpaintImage(
  sourceB64: string,
  pads: OutpaintPads,
  overlap = 32,
): Promise<OutpaintResult> {
  const img = await loadImage(sourceB64)
  const width = align16(img.naturalWidth + pads.left + pads.right)
  const height = align16(img.naturalHeight + pads.top + pads.bottom)
  const left = Math.max(0, Math.round(pads.left))
  const top = Math.max(0, Math.round(pads.top))

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas is unavailable')

  ctx.fillStyle = '#111'
  ctx.fillRect(0, 0, width, height)
  ctx.drawImage(img, left, top, img.naturalWidth, img.naturalHeight)

  const mask = document.createElement('canvas')
  mask.width = width
  mask.height = height
  const maskCtx = mask.getContext('2d')
  if (!maskCtx) throw new Error('Canvas is unavailable')

  maskCtx.fillStyle = '#fff'
  maskCtx.fillRect(0, 0, width, height)
  maskCtx.fillStyle = '#000'
  maskCtx.fillRect(left, top, img.naturalWidth, img.naturalHeight)

  const band = Math.max(0, Math.min(overlap, Math.floor(Math.min(img.naturalWidth, img.naturalHeight) / 4)))
  if (band > 0) {
    maskCtx.fillStyle = '#fff'
    if (pads.left > 0) maskCtx.fillRect(left, top, band, img.naturalHeight)
    if (pads.right > 0) maskCtx.fillRect(left + img.naturalWidth - band, top, band, img.naturalHeight)
    if (pads.top > 0) maskCtx.fillRect(left, top, img.naturalWidth, band)
    if (pads.bottom > 0) maskCtx.fillRect(left, top + img.naturalHeight - band, img.naturalWidth, band)
  }

  return {
    init_image_b64: stripDataUrl(canvas.toDataURL('image/png')),
    mask_b64: stripDataUrl(mask.toDataURL('image/png')),
    width,
    height,
  }
}
