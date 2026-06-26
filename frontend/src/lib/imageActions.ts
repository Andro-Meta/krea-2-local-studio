export function stripDataUrl(src: string): string {
  return src.includes(',') ? src.split(',', 2)[1] : src
}

export async function srcToBase64(src: string): Promise<string> {
  if (src.startsWith('data:')) return stripDataUrl(src)
  const response = await fetch(src)
  if (!response.ok) throw new Error(`Could not load image (${response.status})`)
  const blob = await response.blob()
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(new Error('Could not read image'))
    reader.onload = () => resolve(stripDataUrl(String(reader.result ?? '')))
    reader.readAsDataURL(blob)
  })
}

export function downloadImage(src: string, filename = `krea2_${Date.now()}.png`) {
  const a = document.createElement('a')
  a.href = src
  a.download = filename
  a.click()
}
