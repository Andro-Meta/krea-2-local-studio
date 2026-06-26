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

function drawNeutralPad(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
) {
  ctx.fillStyle = '#808080'
  ctx.fillRect(0, 0, width, height)
}

function drawKijaiStyleOutpaintMask(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  pads: OutpaintPads,
  left: number,
  top: number,
  width: number,
  height: number,
  overlap: number,
) {
  const srcW = img.naturalWidth
  const srcH = img.naturalHeight
  const band = Math.max(0, Math.min(overlap, Math.floor(Math.min(srcW, srcH) / 4)))

  ctx.fillStyle = '#fff'
  ctx.fillRect(0, 0, width, height)
  ctx.fillStyle = '#000'
  ctx.fillRect(left, top, srcW, srcH)

  if (band <= 0) return

  const imageData = ctx.getImageData(left, top, srcW, srcH)
  const data = imageData.data
  for (let y = 0; y < srcH; y += 1) {
    for (let x = 0; x < srcW; x += 1) {
      const distances = [
        pads.left > 0 ? x : srcW,
        pads.right > 0 ? srcW - 1 - x : srcW,
        pads.top > 0 ? y : srcH,
        pads.bottom > 0 ? srcH - 1 - y : srcH,
      ]
      const d = Math.min(...distances)
      if (d >= band) continue
      const v = Math.round(255 * ((band - d) / band) ** 2)
      const i = (y * srcW + x) * 4
      data[i] = v
      data[i + 1] = v
      data[i + 2] = v
      data[i + 3] = 255
    }
  }
  ctx.putImageData(imageData, left, top)
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

  drawNeutralPad(ctx, width, height)
  ctx.drawImage(img, left, top, img.naturalWidth, img.naturalHeight)

  const mask = document.createElement('canvas')
  mask.width = width
  mask.height = height
  const maskCtx = mask.getContext('2d')
  if (!maskCtx) throw new Error('Canvas is unavailable')

  drawKijaiStyleOutpaintMask(maskCtx, img, pads, left, top, width, height, overlap)

  return {
    init_image_b64: stripDataUrl(canvas.toDataURL('image/png')),
    mask_b64: stripDataUrl(mask.toDataURL('image/png')),
    width,
    height,
  }
}
