import { create } from 'zustand'
import type { EngineCatalog, LoraInfo, MoodboardSuggestion, SystemReport } from './api'

export interface ActiveLora {
  name: string
  filename: string
  strength: number
  enabled: boolean
  block_filter?: 'all' | 'early' | 'middle' | 'late' | 'style_safe' | 'custom'
  custom_blocks?: string[]
}

export interface StyleReference {
  image_b64: string
  strength: number
  role: string
  token_size: 'low' | 'normal' | 'high' | 'max'
  mask_b64?: string
  mask_padding?: number
  vision_megapixels?: number | null
  system_prompt?: string
  vision_position?: 'before_prompt' | 'after_prompt'
}

export interface RegionalPrompt {
  prompt: string
  negative_prompt: string
  mask_b64: string
  strength: number
  feather: number
  normalize: boolean
  visible: boolean
  lora_filter: string
}

export interface GenerateParams {
  prompt: string
  negative_prompt: string
  mode: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint' | 'redraw' | 'character_edit'
  model_profile: 'krea_turbo' | 'krea_raw' | 'qwen_image_edit' | 'lens_turbo' | 'ernie_turbo' | 'z_image_turbo' | ''
  diffusion_engine: 'native_pytorch' | 'native_gguf' | 'native_int8_convrot'
  checkpoint: 'turbo' | 'raw'
  quantization: 'bf16' | 'fp8' | 'gguf' | 'fp16' | 'int8'
  turbo_int8_variant: string
  steps: number
  cfg: number
  mu: number | null
  y1: number
  y2: number
  width: number
  height: number
  resolution_tier: '1k' | '2k'
  aspect_ratio: string
  num_images: number
  batch_mode: 'safe_queue' | 'parallel'
  parallel_batch_confirmed: boolean
  batch_int8_all: boolean
  seed: number
  denoise: number
  sampler: 'euler' | 'euler_flow' | 'euler_ancestral' | 'euler_ancestral_cfg_pp' | 'euler_cfg_pp' | 'er_sde' | 'res_2s' | 'res_3s' | 'exp_heun_2_x0_sde' | 'lcm' | 'dpmpp_2m' | 'ddim' | 'uni_pc'
  scheduler: 'simple' | 'normal' | 'beta' | 'beta57' | 'sgm_uniform' | 'bong_tangent' | 'kl_optimal' | 'karras' | 'exponential'
  inpaint_method: 'native' | 'lanpaint_experimental'
  differential_inpaint: boolean
  differential_strength: number
  cfg_zero_star: boolean
  cfg_zero_init_steps: number
  res4lyf_sampler: string
  res4lyf_eta: number
  res4lyf_bongmath: boolean
  actual_denoise: boolean
  incontext_edit: boolean
  incontext_image_b64: string
  incontext_mask_b64: string
  incontext_vision_position: 'before' | 'after'
  incontext_vision_megapixels: number
  incontext_encoder: 'krea2' | 'qwen_edit_plus'
  incontext_system_prompt: string
  character_edit_source_b64: string
  character_edit_grounding_px: number
  character_edit_task: 'restage' | 'local_edit' | 'replace' | 'restyle' | 'removal' | 'two_reference'
  character_edit_lora_strength: number
  style_transfer_image_b64: string
  style_transfer_method: 'AdaIN' | 'WCT' | 'WCT2' | 'scattersort'
  style_transfer_weight: number
  style_transfer_apply_to: 'denoised' | 'positive' | 'negative'
  lanpaint_inner_steps: number
  lanpaint_strength: number
  lanpaint_lambda: number
  lanpaint_step_size: number
  lanpaint_beta: number
  lanpaint_friction: number
  lanpaint_early_stop: number
  lanpaint_prompt_mode: 'Image First' | 'Prompt First'
  edit_provider?: 'auto' | 'krea_native'
  quality_preset?: 'fast' | 'balanced' | 'best' | 'raw_benchmark'
  creativity: 'raw' | 'low' | 'medium' | 'high'
  style_references: StyleReference[]
  style_fusion_mode: 'style_only' | 'preserve_structure' | 'semantic_fusion'
  image_prompt_enabled: boolean
  image_prompt_mode: 'match_style' | 'copy_composition'
  image_prompt_strength: number
  regional_prompts: RegionalPrompt[]
  regional_base_prompt_strength: number
  regional_normalize_masks: boolean
  use_rebalance: boolean
  rebalance_multiplier: number
  rebalance_weights: string
  rebalance_mode: 'legacy_multiply' | 'rms_renorm'
  rebalance_preset: 'legacy' | 'subtle' | 'balanced' | 'detail' | 'emotion' | 'uniform' | 'custom'
  rebalance_renormalize: boolean
  edit_rebalance_enabled: boolean
  edit_rebalance_profile: 'default' | 'edit' | 'conservative'
  conditioning_mode: 'auto' | 'qwen_image_edit_plus' | 'qwen_reference'
  krea_enhancer_variant: 'off' | 'current' | 'capped_delta' | 'current_plus_capped'
  krea_enhancer_enabled: boolean
  krea_enhancer_strength: number
  krea_enhancer_delta_cap: number
  loras: ActiveLora[]
  bboxes: Array<{ label: string; bbox: number[] }>
  init_image_b64: string
  mask_b64: string
  ref_image1_b64: string
  ref_image2_b64: string
  ref_image3_b64: string
  use_prompt_planner: boolean
  prompt_planner_max_tokens: number
  prompt_planner_show_output: boolean
  prompt_planner_lock_original: boolean
  prompt_planner_use_regions: boolean
  use_prompt_expander: boolean
  think_steering_enabled: boolean
  think_text: string
  refine: boolean
  refine_denoise: number
  refine_steps: number
  vae_degrid: boolean
  mood: string
  selected_moodboard_ids: number[]
  moodboard_uuids: string[]
  moodboard_strength: number
  moodboard_images: string[]
  seed_variance_preset: 'off' | 'subtle' | 'balanced' | 'creative' | 'bold' | 'wild' | 'custom'
  seed_variance_strength: number
  seed_variance_algorithm: 'legacy' | 'rbg'
  seed_variance_model_type: 'krea2' | 'z_image' | 'qwen_image' | 'flux' | 'sdxl' | 'other'
  seed_variance_randomize_percent: number
  seed_variance_shift_strength: number
  seed_variance_protection: 'none' | 'first_quarter' | 'first_half' | 'last_quarter' | 'last_half'
  seed_variance_direction: 'none' | 'forward' | 'reverse' | 'center' | 'edges' | 'chaos' | 'order' | 'abstract' | 'realistic' | 'vibrant' | 'moody' | 'dreamy' | 'dynamic_pose' | 'composition' | 'diversity' | 'facevar' | 'visceral_expression_grit' | 'semantic_drift' | 'structural_lock' | 'cinematic_framing' | 'identity_stretch' | 'texture_lift'
  seed_variance_fade_curve: 'instant' | 'linear' | 'ease_in' | 'ease_out' | 'ease_in_out' | 'smoothstep' | 'burst'
  seed_variance_injection_start: number
  seed_variance_injection_end: number
  seed_variance_schedule: 'constant' | 'decreasing' | 'step_cutoff' | 'hard_lock' | 'tiered_release'
  seed_variance_cutoff_step: number
  depth_control: boolean
  depth_control_strength: number
  depth_estimator: 'da3' | 'depth_anything_v2' | 'zoe' | 'midas'
  depth_resolution: number
  depth_invert: boolean
  god_mode: boolean
  mrflow: boolean
  mrflow_upscaler: 'esrgan_x2' | 'remacri_x4'
  mrflow_preset: string
  mrflow_refine_denoise: number
  mrflow_refine_steps: number
  seed_variance_total_steps: number
  seed_variance_cutoff_strength: number
}

export interface LightboxItem {
  src: string
  id?: number
  filename?: string
  prompt?: string
  favorite?: boolean
  metadata?: Record<string, any>
  owner_username?: string | null
}

export interface LightboxState {
  items: LightboxItem[]
  index: number
}

export type CreateMode = 'txt2img' | 'test_labs' | 'character_edit' | 'upscale' | 'redraw'

export const TAB = {
  CREATE: 0,
  GALLERY: 1,
  MOODBOARDS: 2,
  SYSTEM: 3,
} as const

const defaultParams: GenerateParams = {
  prompt: '',
  negative_prompt: '',
  mode: 'txt2img',
  model_profile: 'krea_turbo',
  // Default recipe: "Xperiment fast (beta57)" on Turbo Int8 - er_sde/beta57 @6
  // steps, CFG 1 with CFG-Zero* on. Fastest sweep winner (~3.4s) and a user pick.
  diffusion_engine: 'native_int8_convrot',
  checkpoint: 'turbo',
  quantization: 'int8',
  turbo_int8_variant: 'redcraft',
  steps: 8,
  cfg: 1.0,
  mu: 1.15,
  y1: 0.5,
  y2: 1.15,
  width: 1024,
  height: 1024,
  resolution_tier: '1k',
  aspect_ratio: '1:1',
  num_images: 1,
  batch_mode: 'safe_queue',
  parallel_batch_confirmed: false,
  batch_int8_all: false,
  seed: -1,
  denoise: 0.75,
  sampler: 'er_sde',
  scheduler: 'beta57',
  inpaint_method: 'native',
  differential_inpaint: false,
  differential_strength: 1.0,
  cfg_zero_star: true,
  cfg_zero_init_steps: 1,
  res4lyf_sampler: '',
  res4lyf_eta: 0.5,
  res4lyf_bongmath: false,
  actual_denoise: false,
  incontext_edit: false,
  incontext_image_b64: '',
  incontext_mask_b64: '',
  incontext_vision_position: 'before',
  incontext_vision_megapixels: 1.0,
  incontext_encoder: 'krea2',
  incontext_system_prompt: '',
  character_edit_source_b64: '',
  character_edit_grounding_px: 768,
  character_edit_task: 'restage',
  character_edit_lora_strength: 1.0,
  style_transfer_image_b64: '',
  style_transfer_method: 'AdaIN',
  style_transfer_weight: 0.8,
  style_transfer_apply_to: 'denoised',
  lanpaint_inner_steps: 5,
  lanpaint_strength: 1.0,
  lanpaint_lambda: 16.0,
  lanpaint_step_size: 0.2,
  lanpaint_beta: 1.0,
  lanpaint_friction: 15.0,
  lanpaint_early_stop: 1,
  lanpaint_prompt_mode: 'Image First',
  edit_provider: 'auto',
  quality_preset: 'balanced',
  creativity: 'medium',
  style_references: [],
  style_fusion_mode: 'semantic_fusion',
  image_prompt_enabled: false,
  image_prompt_mode: 'match_style',
  image_prompt_strength: 0.2,
  regional_prompts: [],
  regional_base_prompt_strength: 0.3,
  regional_normalize_masks: true,
  use_rebalance: true,
  rebalance_multiplier: 1.0,
  rebalance_weights: '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0',
  rebalance_mode: 'rms_renorm',
  rebalance_preset: 'balanced',
  rebalance_renormalize: true,
  edit_rebalance_enabled: true,
  edit_rebalance_profile: 'conservative',
  conditioning_mode: 'auto',
  krea_enhancer_variant: 'off',
  krea_enhancer_enabled: false,
  krea_enhancer_strength: 1.0,
  krea_enhancer_delta_cap: 0.75,
  // Default recipe is "Xperiment fast · loose"; the filter-bypass LoRA is part of
  // every quick recipe, so it's on from startup at 6850 too.
  loras: [{ name: 'krea2filterbypass3', filename: 'krea2filterbypass3.safetensors', strength: 6850, enabled: true, block_filter: 'style_safe' }],
  bboxes: [],
  init_image_b64: '',
  mask_b64: '',
  ref_image1_b64: '',
  ref_image2_b64: '',
  ref_image3_b64: '',
  use_prompt_planner: false,
  prompt_planner_max_tokens: 700,
  prompt_planner_show_output: false,
  prompt_planner_lock_original: false,
  prompt_planner_use_regions: false,
  use_prompt_expander: false,
  think_steering_enabled: false,
  think_text: '',
  refine: false,
  refine_denoise: 0.3,
  refine_steps: 6,
  vae_degrid: true,
  mood: '',
  selected_moodboard_ids: [],
  moodboard_uuids: [],
  moodboard_strength: 0.35,
  moodboard_images: [],
  // Default: Wild + hard_lock "composition lock" @ loose 2/8. Early steps follow
  // the base prompt (composition stays put); the detail phase gets full Wild
  // variance so seed rolls change expression/detail without re-rolling the shot.
  // NOTE: hard_lock only varies anything when cutoff_strength > 0.
  seed_variance_preset: 'wild',
  seed_variance_strength: 0.0,
  seed_variance_algorithm: 'rbg',
  seed_variance_model_type: 'krea2',
  seed_variance_randomize_percent: 0,
  seed_variance_shift_strength: 100,
  seed_variance_protection: 'first_half',
  seed_variance_direction: 'none',
  seed_variance_fade_curve: 'linear',
  seed_variance_injection_start: 0,
  seed_variance_injection_end: 1,
  seed_variance_schedule: 'hard_lock',
  seed_variance_cutoff_step: 2,
  depth_control: false,
  depth_control_strength: 1.2,
  depth_estimator: 'da3',
  depth_resolution: 504,
  depth_invert: false,
  god_mode: false,
  mrflow: false,
  mrflow_upscaler: 'esrgan_x2',
  mrflow_preset: '',
  mrflow_refine_denoise: 0,
  mrflow_refine_steps: 0,
  seed_variance_total_steps: 20,
  seed_variance_cutoff_strength: 1.0,
}

interface AppState {
  params: GenerateParams
  setParam: <K extends keyof GenerateParams>(key: K, value: GenerateParams[K]) => void
  setParams: (partial: Partial<GenerateParams>) => void

  generating: boolean
  jobId: string | null
  progress: number
  queuePosition: number | null
  queueLength: number | null
  results: string[]
  resultsMetadata: Array<Record<string, any>>
  lastSeed: number | null
  generationError: string | null
  promptBusy: boolean
  setGenerating: (v: boolean) => void
  setPromptBusy: (v: boolean) => void
  setJobId: (id: string | null) => void
  setProgress: (n: number) => void
  setQueue: (position: number | null, length: number | null) => void
  setResults: (imgs: string[], seed?: number, metadata?: Array<Record<string, any>>) => void
  setError: (e: string | null) => void

  systemReport: SystemReport | null
  setSystemReport: (r: SystemReport) => void
  engineCatalog: EngineCatalog | null
  setEngineCatalog: (r: EngineCatalog) => void
  modelLoaded: boolean
  setModelLoaded: (v: boolean) => void

  loras: LoraInfo[]
  setLoras: (l: LoraInfo[]) => void

  tab: number
  setTab: (n: number) => void
  createMode: CreateMode
  setCreateMode: (mode: CreateMode) => void
  moodboardView: 'official' | 'andrometa' | 'favorites' | 'custom' | 'new'
  setMoodboardView: (view: 'official' | 'favorites' | 'custom' | 'new') => void
  moodboardSuggestions: MoodboardSuggestion[]
  setMoodboardSuggestions: (items: MoodboardSuggestion[]) => void

  lightbox: LightboxState | null
  lightboxImage: string | null
  openLightbox: (items: LightboxItem[], index?: number) => void
  closeLightbox: () => void
  nextLightbox: () => void
  previousLightbox: () => void
  patchLightboxItem: (id: number, partial: Partial<LightboxItem>) => void
  removeLightboxItem: (id: number) => void
  setLightboxImage: (src: string | null) => void
}

export const useStore = create<AppState>((set, get) => ({
  params: defaultParams,
  setParam: (key, value) => set(s => ({ params: { ...s.params, [key]: value } })),
  setParams: (partial) => set(s => ({ params: { ...s.params, ...partial } })),

  generating: false,
  jobId: null,
  progress: 0,
  queuePosition: null,
  queueLength: null,
  results: [],
  resultsMetadata: [],
  lastSeed: null,
  generationError: null,
  promptBusy: false,
  setGenerating: (v) => set({ generating: v }),
  setPromptBusy: (v) => set({ promptBusy: v }),
  setJobId: (id) => set({ jobId: id }),
  setProgress: (n) => set({ progress: n }),
  setQueue: (position, length) => set({ queuePosition: position, queueLength: length }),
  setResults: (imgs, seed, metadata) => set({ results: imgs, lastSeed: seed ?? null, resultsMetadata: metadata ?? [] }),
  setError: (e) => set({ generationError: e }),

  systemReport: null,
  setSystemReport: (r) => set({ systemReport: r }),
  engineCatalog: null,
  setEngineCatalog: (r) => set({ engineCatalog: r }),
  modelLoaded: false,
  setModelLoaded: (v) => set({ modelLoaded: v }),

  loras: [],
  setLoras: (l) => set({ loras: l }),

  tab: TAB.CREATE,
  setTab: (n) => set({ tab: n }),
  createMode: 'txt2img',
  setCreateMode: (mode) => set({ createMode: mode }),
  moodboardView: 'official',
  setMoodboardView: (view) => set({ moodboardView: view }),
  moodboardSuggestions: [],
  setMoodboardSuggestions: (items) => set({ moodboardSuggestions: items }),

  lightbox: null,
  lightboxImage: null,
  openLightbox: (items, index = 0) => set({
    lightbox: items.length ? { items, index: Math.min(Math.max(index, 0), items.length - 1) } : null,
    lightboxImage: items[index]?.src ?? null,
  }),
  closeLightbox: () => set({ lightbox: null, lightboxImage: null }),
  nextLightbox: () => {
    const lightbox = get().lightbox
    if (!lightbox?.items.length) return
    const index = (lightbox.index + 1) % lightbox.items.length
    set({ lightbox: { ...lightbox, index }, lightboxImage: lightbox.items[index].src })
  },
  previousLightbox: () => {
    const lightbox = get().lightbox
    if (!lightbox?.items.length) return
    const index = (lightbox.index - 1 + lightbox.items.length) % lightbox.items.length
    set({ lightbox: { ...lightbox, index }, lightboxImage: lightbox.items[index].src })
  },
  patchLightboxItem: (id, partial) => set(s => {
    const lightbox = s.lightbox
    if (!lightbox) return {}
    return { lightbox: { ...lightbox, items: lightbox.items.map(item => item.id === id ? { ...item, ...partial } : item) } }
  }),
  removeLightboxItem: (id) => set(s => {
    const lightbox = s.lightbox
    if (!lightbox) return {}
    const items = lightbox.items.filter(item => item.id !== id)
    if (!items.length) return { lightbox: null, lightboxImage: null }
    const index = Math.min(lightbox.index, items.length - 1)
    return { lightbox: { items, index }, lightboxImage: items[index].src }
  }),
  setLightboxImage: (src) => src
    ? set({ lightbox: { items: [{ src }], index: 0 }, lightboxImage: src })
    : set({ lightbox: null, lightboxImage: null }),
}))
