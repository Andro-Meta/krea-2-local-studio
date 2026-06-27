import type { GenerateParams } from '../store'

export type ImportTargetMode = GenerateParams['mode']

function oneOf<T extends string>(value: unknown, allowed: readonly T[]): T | undefined {
  return typeof value === 'string' && allowed.includes(value as T) ? value as T : undefined
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function booleanValue(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

function numberArray(value: unknown): number[] {
  return Array.isArray(value) ? value.filter((item): item is number => typeof item === 'number' && Number.isFinite(item)) : []
}

function lorasFromMetadata(value: unknown): GenerateParams['loras'] {
  if (!Array.isArray(value)) return []
  return value
    .filter((item): item is Record<string, unknown> => item && typeof item === 'object')
    .map(item => ({
      name: String(item.name || item.filename || ''),
      filename: String(item.filename || item.name || ''),
      strength: numberValue(item.strength) ?? 1,
      enabled: booleanValue(item.enabled) ?? true,
    }))
    .filter(item => item.name || item.filename)
}

function enhancerVariant(enabled: boolean, rebalanceEnabled: boolean): GenerateParams['krea_enhancer_variant'] {
  if (enabled && rebalanceEnabled) return 'rebalance_enhancer'
  if (enabled) return 'enhancer'
  if (rebalanceEnabled) return 'rebalance'
  return 'off'
}

export function metadataToGenerateParams<TMode extends ImportTargetMode>(
  metadata: Record<string, any>,
  targetMode: TMode,
  imageB64 = '',
): Partial<GenerateParams> & { mode: TMode } {
  const rebalanceEnabled = booleanValue(metadata.rebalance?.enabled) ?? false
  const enhancerEnabled = booleanValue(metadata.krea_enhancer?.enabled) ?? false
  const patch: Partial<GenerateParams> & { mode: TMode } = {
    mode: targetMode,
    prompt: String(metadata.prompt || ''),
    negative_prompt: String(metadata.negative_prompt || ''),
    checkpoint: oneOf(metadata.checkpoint, ['turbo', 'raw'] as const),
    quantization: oneOf(metadata.quantization, ['bf16', 'fp8'] as const),
    steps: numberValue(metadata.steps),
    cfg: numberValue(metadata.cfg),
    mu: numberValue(metadata.mu) ?? null,
    y1: numberValue(metadata.y1),
    y2: numberValue(metadata.y2),
    width: numberValue(metadata.width),
    height: numberValue(metadata.height),
    seed: numberValue(metadata.seed),
    denoise: numberValue(metadata.denoise),
    edit_provider: oneOf(metadata.edit_provider || metadata.resolved_provider, ['auto', 'krea_native', 'flux_fill'] as const),
    quality_preset: oneOf(metadata.quality_preset, ['fast', 'balanced', 'best', 'raw_benchmark'] as const),
    loras: lorasFromMetadata(metadata.loras),
    bboxes: Array.isArray(metadata.bboxes) ? metadata.bboxes : [],
    mood: String(metadata.mood || ''),
    selected_moodboard_ids: numberArray(metadata.moodboard_ids),
    moodboard_strength: numberValue(metadata.moodboard_strength),
    use_rebalance: rebalanceEnabled,
    rebalance_multiplier: numberValue(metadata.rebalance?.multiplier),
    rebalance_weights: typeof metadata.rebalance?.weights === 'string' ? metadata.rebalance.weights : undefined,
    krea_enhancer_enabled: enhancerEnabled,
    krea_enhancer_strength: numberValue(metadata.krea_enhancer?.strength),
    krea_enhancer_variant: enhancerVariant(enhancerEnabled, rebalanceEnabled),
    refine: booleanValue(metadata.refine?.enabled),
    refine_denoise: numberValue(metadata.refine?.denoise),
    refine_steps: numberValue(metadata.refine?.steps),
  }

  if (targetMode === 'txt2img') {
    patch.init_image_b64 = ''
    patch.mask_b64 = ''
    patch.ref_image1_b64 = ''
    patch.ref_image2_b64 = ''
    patch.ref_image3_b64 = ''
    patch.moodboard_images = []
  } else if (targetMode === 'redraw') {
    patch.init_image_b64 = ''
    patch.mask_b64 = ''
    patch.ref_image1_b64 = ''
    patch.ref_image2_b64 = ''
    patch.ref_image3_b64 = ''
    patch.moodboard_images = imageB64 ? [imageB64] : []
  } else {
    patch.init_image_b64 = imageB64
    patch.mask_b64 = ''
    patch.ref_image1_b64 = ''
    patch.ref_image2_b64 = ''
    patch.ref_image3_b64 = ''
    patch.moodboard_images = []
  }

  return Object.fromEntries(
    Object.entries(patch).filter(([, value]) => value !== undefined),
  ) as Partial<GenerateParams> & { mode: TMode }
}
