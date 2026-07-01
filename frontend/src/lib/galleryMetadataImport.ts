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
      block_filter: oneOf(item.block_filter, ['all', 'early', 'middle', 'late', 'style_safe', 'custom'] as const) ?? 'all',
      custom_blocks: stringArray(item.custom_blocks),
    }))
    .filter(item => item.name || item.filename)
}

function styleRefsFromMetadata(value: unknown): GenerateParams['style_references'] {
  if (!Array.isArray(value)) return []
  return value
    .filter((item): item is Record<string, unknown> => item && typeof item === 'object')
    .map(item => ({
      image_b64: String(item.image_b64 || ''),
      strength: numberValue(item.strength) ?? 1,
      role: String(item.role || 'style'),
      token_size: oneOf(item.token_size, ['low', 'normal', 'high', 'max'] as const) ?? 'normal',
      mask_b64: typeof item.mask_b64 === 'string' ? item.mask_b64 : undefined,
      mask_padding: numberValue(item.mask_padding),
      vision_megapixels: numberValue(item.vision_megapixels),
      system_prompt: typeof item.system_prompt === 'string' ? item.system_prompt : undefined,
      vision_position: oneOf(item.vision_position, ['before_prompt', 'after_prompt'] as const),
    }))
    .filter(item => item.image_b64)
    .slice(0, 10)
}

function regionalPromptsFromMetadata(value: unknown): GenerateParams['regional_prompts'] {
  if (!Array.isArray(value)) return []
  return value
    .filter((item): item is Record<string, unknown> => item && typeof item === 'object')
    .map(item => ({
      prompt: String(item.prompt || ''),
      negative_prompt: String(item.negative_prompt || ''),
      mask_b64: '',
      strength: numberValue(item.strength) ?? 1,
      feather: numberValue(item.feather) ?? 24,
      normalize: booleanValue(item.normalize) ?? true,
      visible: booleanValue(item.visible) ?? true,
      lora_filter: String(item.lora_filter || ''),
    }))
    .filter(item => item.prompt)
    .slice(0, 8)
}

function enhancerVariant(metadata: Record<string, any>): GenerateParams['krea_enhancer_variant'] {
  return oneOf(metadata.krea_enhancer?.variant, ['off', 'current', 'capped_delta', 'current_plus_capped'] as const)
    ?? (booleanValue(metadata.krea_enhancer?.enabled) ? 'capped_delta' : 'off')
}

export function metadataToGenerateParams<TMode extends ImportTargetMode>(
  metadata: Record<string, any>,
  targetMode: TMode,
  imageB64 = '',
): Partial<GenerateParams> & { mode: TMode } {
  const rebalanceEnabled = booleanValue(metadata.rebalance?.enabled) ?? false
  const enhancerEnabled = booleanValue(metadata.krea_enhancer?.enabled) ?? false
  const restoredQuantization = oneOf(metadata.quantization ?? metadata.model?.quantization ?? metadata.engine?.quantization, ['bf16', 'fp8', 'gguf', 'fp16', 'int8'] as const)
  const rawRestoredEngine = oneOf(metadata.diffusion_engine ?? metadata.engine?.id, ['native_pytorch', 'native_gguf', 'native_int8_convrot', 'gguf_external', 'int8_convrot_external'] as const)
    ?? (restoredQuantization === 'int8' ? 'native_int8_convrot' : undefined)
  const restoredEngine = rawRestoredEngine === 'gguf_external'
    ? 'native_gguf'
    : rawRestoredEngine === 'int8_convrot_external'
      ? 'native_int8_convrot'
      : rawRestoredEngine
  const patch: Partial<GenerateParams> & { mode: TMode } = {
    mode: targetMode,
    prompt: String(metadata.prompt || ''),
    negative_prompt: String(metadata.negative_prompt || ''),
    diffusion_engine: restoredEngine,
    model_profile: oneOf(metadata.model_profile ?? metadata.model?.profile, ['krea_turbo', 'krea_raw', 'qwen_image_edit', 'lens_turbo', 'ernie_turbo', 'z_image_turbo', ''] as const),
    checkpoint: oneOf(metadata.checkpoint ?? metadata.model?.checkpoint, ['turbo', 'raw'] as const),
    quantization: restoredQuantization,
    steps: numberValue(metadata.steps),
    cfg: numberValue(metadata.cfg),
    mu: numberValue(metadata.mu) ?? null,
    y1: numberValue(metadata.y1),
    y2: numberValue(metadata.y2),
    width: numberValue(metadata.width),
    height: numberValue(metadata.height),
    seed: numberValue(metadata.seed),
    denoise: numberValue(metadata.denoise),
    sampler: oneOf(metadata.sampler, ['euler', 'euler_flow', 'euler_ancestral', 'euler_ancestral_cfg_pp', 'euler_cfg_pp', 'er_sde', 'res_2s', 'exp_heun_2_x0_sde', 'lcm', 'dpmpp_2m', 'ddim', 'uni_pc'] as const),
    scheduler: oneOf(metadata.scheduler, ['simple', 'normal', 'beta', 'beta57', 'sgm_uniform', 'bong_tangent', 'karras', 'exponential'] as const),
    cfg_zero_star: typeof metadata.cfg_zero_star === 'boolean' ? metadata.cfg_zero_star : undefined,
    cfg_zero_init_steps: numberValue(metadata.cfg_zero_init_steps),
    inpaint_method: oneOf(metadata.inpaint?.method, ['native', 'lanpaint_experimental', 'flux_fill'] as const),
    differential_inpaint: typeof metadata.inpaint?.differential_inpaint === 'boolean' ? metadata.inpaint.differential_inpaint : undefined,
    differential_strength: numberValue(metadata.inpaint?.differential_strength),
    lanpaint_inner_steps: numberValue(metadata.inpaint?.lanpaint_inner_steps),
    lanpaint_strength: numberValue(metadata.inpaint?.lanpaint_strength),
    lanpaint_lambda: numberValue(metadata.inpaint?.lanpaint_lambda),
    lanpaint_step_size: numberValue(metadata.inpaint?.lanpaint_step_size),
    lanpaint_beta: numberValue(metadata.inpaint?.lanpaint_beta),
    lanpaint_friction: numberValue(metadata.inpaint?.lanpaint_friction),
    lanpaint_early_stop: numberValue(metadata.inpaint?.lanpaint_early_stop),
    lanpaint_prompt_mode: oneOf(metadata.inpaint?.lanpaint_prompt_mode, ['Image First', 'Prompt First'] as const),
    edit_provider: oneOf(metadata.edit_provider || metadata.resolved_provider, ['auto', 'krea_native', 'flux_fill'] as const),
    quality_preset: oneOf(metadata.quality_preset, ['fast', 'balanced', 'best', 'raw_benchmark'] as const),
    creativity: oneOf(metadata.creativity, ['raw', 'low', 'medium', 'high'] as const),
    style_references: styleRefsFromMetadata(metadata.image_references?.style_references),
    style_fusion_mode: oneOf(metadata.image_references?.style_fusion_mode, ['style_only', 'preserve_structure', 'semantic_fusion'] as const),
    regional_prompts: regionalPromptsFromMetadata(metadata.regional_prompts?.regions),
    regional_base_prompt_strength: numberValue(metadata.regional_prompts?.base_prompt_strength),
    regional_normalize_masks: booleanValue(metadata.regional_prompts?.normalize_masks),
    loras: lorasFromMetadata(metadata.loras),
    bboxes: Array.isArray(metadata.bboxes) ? metadata.bboxes : [],
    mood: String(metadata.mood || ''),
    selected_moodboard_ids: numberArray(metadata.moodboard_ids),
    moodboard_uuids: stringArray(metadata.moodboard_uuids),
    moodboard_strength: numberValue(metadata.moodboard_strength),
    seed_variance_preset: oneOf(metadata.seed_variance?.preset, ['off', 'subtle', 'balanced', 'creative', 'bold', 'wild', 'custom'] as const),
    seed_variance_strength: numberValue(metadata.seed_variance?.strength),
    seed_variance_algorithm: oneOf(metadata.seed_variance?.algorithm, ['legacy', 'rbg'] as const),
    seed_variance_model_type: oneOf(metadata.seed_variance?.model_type, ['krea2', 'z_image', 'qwen_image', 'flux', 'sdxl', 'other'] as const),
    seed_variance_randomize_percent: numberValue(metadata.seed_variance?.randomize_percent),
    seed_variance_shift_strength: numberValue(metadata.seed_variance?.shift_strength),
    seed_variance_protection: oneOf(metadata.seed_variance?.protection, ['none', 'first_quarter', 'first_half', 'last_quarter', 'last_half'] as const),
    seed_variance_direction: oneOf(metadata.seed_variance?.direction, ['none', 'forward', 'reverse', 'center', 'edges', 'chaos', 'order', 'abstract', 'realistic', 'vibrant', 'moody', 'dreamy', 'dynamic_pose', 'composition', 'diversity', 'facevar', 'visceral_expression_grit', 'semantic_drift', 'structural_lock', 'cinematic_framing', 'identity_stretch', 'texture_lift'] as const),
    seed_variance_fade_curve: oneOf(metadata.seed_variance?.fade_curve, ['instant', 'linear', 'ease_in', 'ease_out', 'ease_in_out', 'smoothstep', 'burst'] as const),
    seed_variance_injection_start: numberValue(metadata.seed_variance?.injection_start),
    seed_variance_injection_end: numberValue(metadata.seed_variance?.injection_end),
    seed_variance_schedule: oneOf(metadata.seed_variance?.schedule, ['constant', 'decreasing', 'step_cutoff'] as const),
    seed_variance_cutoff_step: numberValue(metadata.seed_variance?.cutoff_step),
    seed_variance_total_steps: numberValue(metadata.seed_variance?.total_steps),
    seed_variance_cutoff_strength: numberValue(metadata.seed_variance?.cutoff_strength),
    use_rebalance: rebalanceEnabled,
    rebalance_multiplier: numberValue(metadata.rebalance?.multiplier),
    rebalance_weights: typeof metadata.rebalance?.weights === 'string' ? metadata.rebalance.weights : undefined,
    rebalance_mode: oneOf(metadata.rebalance?.mode, ['legacy_multiply', 'rms_renorm'] as const),
    rebalance_preset: oneOf(metadata.rebalance?.preset, ['legacy', 'subtle', 'balanced', 'detail', 'emotion', 'uniform', 'custom'] as const),
    rebalance_renormalize: booleanValue(metadata.rebalance?.renormalize),
    edit_rebalance_enabled: booleanValue(metadata.rebalance?.edit_enabled),
    edit_rebalance_profile: oneOf(metadata.rebalance?.edit_profile, ['default', 'edit', 'conservative'] as const),
    krea_enhancer_enabled: enhancerEnabled,
    krea_enhancer_strength: numberValue(metadata.krea_enhancer?.strength),
    krea_enhancer_variant: enhancerVariant(metadata),
    krea_enhancer_delta_cap: numberValue(metadata.krea_enhancer?.delta_cap),
    refine: booleanValue(metadata.refine?.enabled),
    refine_denoise: numberValue(metadata.refine?.denoise),
    refine_steps: numberValue(metadata.refine?.steps),
    use_prompt_planner: booleanValue(metadata.prompt_planner?.enabled),
    prompt_planner_max_tokens: numberValue(metadata.prompt_planner?.max_tokens),
    prompt_planner_show_output: booleanValue(metadata.prompt_planner?.enabled),
    prompt_planner_lock_original: booleanValue(metadata.prompt_planner?.lock_original),
    prompt_planner_use_regions: booleanValue(metadata.prompt_planner?.use_regions),
    think_steering_enabled: booleanValue(metadata.think_steering?.enabled),
    think_text: typeof metadata.think_steering?.text === 'string' ? metadata.think_steering.text : undefined,
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
