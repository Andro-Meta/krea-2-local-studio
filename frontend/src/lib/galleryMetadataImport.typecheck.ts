import { metadataToGenerateParams } from './galleryMetadataImport'

const imported = metadataToGenerateParams({
  prompt: 'a silver tree',
  negative_prompt: 'blur',
  seed: 44,
  mode: 'txt2img',
  checkpoint: 'turbo',
  quantization: 'fp8',
  steps: 8,
  cfg: 0,
  width: 1024,
  height: 768,
  loras: [{ name: 'krea2_darkbrush', filename: 'krea2_darkbrush.safetensors', strength: 0.7, enabled: true }],
  moodboard_ids: [12],
  rebalance: { enabled: true, multiplier: 5, weights: '1,2' },
  krea_enhancer: { enabled: true, strength: 0.8 },
  refine: { enabled: true, denoise: 0.2, steps: 4 },
}, 'redraw', 'IMAGE_B64')

imported.mode satisfies 'redraw'
imported.init_image_b64 satisfies string | undefined
imported.loras?.[0].name satisfies string | undefined
