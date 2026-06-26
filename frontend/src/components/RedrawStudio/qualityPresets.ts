import type { GenerationRequest, LoraInfo } from '../../api'

export type RedrawQualityMode = 'fast' | 'balanced' | 'best' | 'raw_benchmark'
export type RedrawTaskKind =
  | 'recreate'
  | 'insert'
  | 'extend_redraw'
  | 'extend_preserve'
  | 'sketch'
  | 'style'
  | 'moodboard'
  | 'preserve_whole'
  | 'preserve_masked'

export interface RedrawQualityPreset {
  task: RedrawTaskKind
  qualityMode: RedrawQualityMode
  checkpoint: GenerationRequest['checkpoint']
  quantization: GenerationRequest['quantization']
  steps: number
  cfg: number
  mu: number | null
  moodboardStrength: number
  denoise: number
  usePromptExpander: boolean
  editProvider: 'auto' | 'krea_native' | 'flux_fill'
  promptHint: string
  warning?: string
}

const modeBoost: Record<RedrawQualityMode, { steps: number; strengthDelta: number; expander: boolean }> = {
  fast: { steps: 6, strengthDelta: -0.08, expander: false },
  balanced: { steps: 8, strengthDelta: 0, expander: true },
  best: { steps: 10, strengthDelta: 0.05, expander: true },
  raw_benchmark: { steps: 52, strengthDelta: 0, expander: true },
}

const base: Record<RedrawTaskKind, Omit<RedrawQualityPreset, 'qualityMode' | 'steps' | 'checkpoint' | 'quantization' | 'cfg' | 'mu' | 'usePromptExpander'>> = {
  recreate: {
    task: 'recreate',
    moodboardStrength: 0.6,
    denoise: 1,
    editProvider: 'krea_native',
    promptHint: 'Use the references as concept and composition, then redraw one unified finished image.',
  },
  insert: {
    task: 'insert',
    moodboardStrength: 0.5,
    denoise: 1,
    editProvider: 'krea_native',
    promptHint: 'Treat the scene, subject, and object roles separately; match lighting, scale, shadows, and perspective.',
    warning: 'Too many equal-weight subject references can turn into a collage. Use one scene plus one or two insert references.',
  },
  extend_redraw: {
    task: 'extend_redraw',
    moodboardStrength: 0.58,
    denoise: 1,
    editProvider: 'krea_native',
    promptHint: 'Redraw the whole frame into a wider coherent image rather than preserving exact pixels.',
  },
  extend_preserve: {
    task: 'extend_preserve',
    moodboardStrength: 0.45,
    denoise: 1,
    editProvider: 'auto',
    promptHint: 'Preserve the source image and synthesize only the missing canvas with matched lighting and perspective.',
    warning: 'Strict outpaint is strongest with FLUX Fill installed. Krea native fallback may be creative, not exact.',
  },
  sketch: {
    task: 'sketch',
    moodboardStrength: 0.42,
    denoise: 1,
    editProvider: 'krea_native',
    promptHint: 'Use the sketch only for layout and silhouette; replace sketch texture with realistic materials.',
  },
  style: {
    task: 'style',
    moodboardStrength: 0.45,
    denoise: 1,
    editProvider: 'krea_native',
    promptHint: 'Use style references or a Krea style LoRA for art direction only, not as subject content.',
    warning: 'For coherent style transfer, prefer an official Krea style LoRA over style images alone.',
  },
  moodboard: {
    task: 'moodboard',
    moodboardStrength: 0.55,
    denoise: 1,
    editProvider: 'krea_native',
    promptHint: 'Use mood/style/scene roles as art direction and produce a new coherent image.',
  },
  preserve_whole: {
    task: 'preserve_whole',
    moodboardStrength: 0.35,
    denoise: 0.45,
    editProvider: 'krea_native',
    promptHint: 'Improve or reinterpret the whole source image while keeping composition recognizable.',
  },
  preserve_masked: {
    task: 'preserve_masked',
    moodboardStrength: 0.45,
    denoise: 0.72,
    editProvider: 'auto',
    promptHint: 'Describe exactly what should appear inside the mask and preserve unmasked pixels.',
    warning: 'Masked precision edits are best with FLUX Fill installed. Krea native fallback may alter nearby pixels.',
  },
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

export function presetFor(task: RedrawTaskKind, qualityMode: RedrawQualityMode): RedrawQualityPreset {
  const boost = modeBoost[qualityMode]
  const preset = base[task]
  const raw = qualityMode === 'raw_benchmark'
  return {
    ...preset,
    qualityMode,
    checkpoint: raw ? 'raw' : 'turbo',
    quantization: raw ? 'bf16' : 'bf16',
    steps: boost.steps,
    cfg: raw ? 3.5 : 0,
    mu: raw ? null : 1.15,
    moodboardStrength: clamp(preset.moodboardStrength + boost.strengthDelta, 0.25, 0.75),
    usePromptExpander: boost.expander,
  }
}

export function loraToGeneration(lora: LoraInfo | null) {
  if (!lora) return []
  return [{
    name: lora.name,
    filename: lora.filename,
    strength: lora.strength,
    enabled: true,
  }]
}
