import { loraToGeneration, presetFor } from './qualityPresets'
import type { LoraInfo } from '../../api'

const nk2e: LoraInfo = {
  filename: 'NK2E-v0.1.safetensors',
  name: 'nk2e_v01',
  display_name: 'NK2E Edit',
  trigger_words: [],
  strength: 0.7,
  is_official: false,
  installed: true,
  compatible: true,
}

if (presetFor('nk2e_edit', 'balanced').denoise !== 0.45) {
  throw new Error('NK2E balanced preset should default to denoise 0.45')
}

if (presetFor('nk2e_edit', 'balanced').editProvider !== 'krea_native') {
  throw new Error('NK2E should use native img2img/redraw provider')
}

const loras = loraToGeneration(nk2e)
if (loras[0]?.strength !== 0.7 || loras[0]?.block_filter !== 'all') {
  throw new Error('NK2E LoRA should map to strength 0.7 with all blocks')
}
