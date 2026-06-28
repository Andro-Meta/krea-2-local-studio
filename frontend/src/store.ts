import { create } from 'zustand'
import type { LoraInfo, SystemReport } from './api'
import { createDefaultDocument, type RealtimeDocument, type RealtimeTool } from './components/RealtimeStudio/canvasDocument'

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
  mode: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint' | 'redraw'
  model_profile: 'krea_turbo' | 'krea_raw' | 'qwen_image_edit' | 'lens_turbo' | 'ernie_turbo' | 'z_image_turbo' | ''
  checkpoint: 'turbo' | 'raw'
  quantization: 'bf16' | 'fp8' | 'fp16'
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
  seed: number
  denoise: number
  sampler: 'euler' | 'euler_flow' | 'euler_ancestral' | 'euler_ancestral_cfg_pp' | 'euler_cfg_pp' | 'exp_heun_2_x0_sde' | 'lcm' | 'dpmpp_2m' | 'ddim' | 'uni_pc'
  scheduler: 'simple' | 'normal' | 'beta' | 'sgm_uniform' | 'karras' | 'exponential'
  inpaint_method: 'native' | 'lanpaint_experimental' | 'flux_fill'
  differential_inpaint: boolean
  differential_strength: number
  lanpaint_inner_steps: number
  lanpaint_strength: number
  lanpaint_lambda: number
  lanpaint_step_size: number
  lanpaint_beta: number
  lanpaint_friction: number
  lanpaint_early_stop: number
  lanpaint_prompt_mode: 'Image First' | 'Prompt First'
  edit_provider?: 'auto' | 'krea_native' | 'flux_fill'
  quality_preset?: 'fast' | 'balanced' | 'best' | 'raw_benchmark'
  creativity: 'raw' | 'low' | 'medium' | 'high'
  style_references: StyleReference[]
  style_fusion_mode: 'style_only' | 'preserve_structure' | 'semantic_fusion'
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
  mood: string
  selected_moodboard_ids: number[]
  moodboard_uuids: string[]
  moodboard_strength: number
  moodboard_images: string[]
  seed_variance_preset: 'off' | 'subtle' | 'balanced' | 'creative' | 'bold' | 'custom'
  seed_variance_strength: number
  seed_variance_protection: 'none' | 'first_quarter' | 'first_half'
  seed_variance_direction: 'none' | 'forward' | 'reverse' | 'center' | 'edges'
  seed_variance_fade_curve: 'linear' | 'ease_in' | 'ease_out' | 'smoothstep'
  seed_variance_injection_start: number
  seed_variance_injection_end: number
}

export interface LightboxItem {
  src: string
  id?: number
  filename?: string
  prompt?: string
  favorite?: boolean
  metadata?: Record<string, any>
}

export interface LightboxState {
  items: LightboxItem[]
  index: number
}

export interface RealtimePreviewState {
  status: 'idle' | 'queued' | 'running' | 'ready' | 'final-ready' | 'error'
  sessionId: string
  jobId: string | null
  revision: number
  progress: number
  image: string
  seed: number | null
  metadata?: Record<string, any> | null
  error: string | null
  lastUpdated: number | null
  paused: boolean
}

export interface RealtimeSettings {
  debounceMs: number
  previewSize: number
  previewSteps: number
  finalSteps: number
  canvasInfluence: number
  seed: number
  lockSeed: boolean
  autoPreview: boolean
}

export interface RealtimeState {
  document: RealtimeDocument
  selectedLayerId: string | null
  tool: RealtimeTool
  color: string
  brushSize: number
  shape: 'rectangle' | 'circle' | 'triangle'
  prompt: string
  negativePrompt: string
  preview: RealtimePreviewState
  settings: RealtimeSettings
}

function makeSessionId(): string {
  const id = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${performance.now().toString(36).replace('.', '')}`
  return `rt-${id}`
}

const defaultParams: GenerateParams = {
  prompt: '',
  negative_prompt: '',
  mode: 'txt2img',
  model_profile: 'krea_turbo',
  checkpoint: 'turbo',
  quantization: 'fp8',
  steps: 8,
  cfg: 0.0,
  mu: 1.15,
  y1: 0.5,
  y2: 1.15,
  width: 1024,
  height: 1024,
  resolution_tier: '1k',
  aspect_ratio: '1:1',
  num_images: 1,
  seed: -1,
  denoise: 0.75,
  sampler: 'euler',
  scheduler: 'simple',
  inpaint_method: 'native',
  differential_inpaint: false,
  differential_strength: 1.0,
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
  loras: [],
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
  mood: '',
  selected_moodboard_ids: [],
  moodboard_uuids: [],
  moodboard_strength: 0.35,
  moodboard_images: [],
  seed_variance_preset: 'off',
  seed_variance_strength: 0.0,
  seed_variance_protection: 'first_half',
  seed_variance_direction: 'none',
  seed_variance_fade_curve: 'linear',
  seed_variance_injection_start: 0,
  seed_variance_injection_end: 1,
}

const defaultRealtime: RealtimeState = {
  document: createDefaultDocument(),
  selectedLayerId: null,
  tool: 'brush',
  color: '#111111',
  brushSize: 28,
  shape: 'rectangle',
  prompt: '4k product photography of a whimsical sculptural object in a natural landscape',
  negativePrompt: 'low quality, blurry, text, watermark, deformed, pasted collage',
  preview: {
    status: 'idle',
    sessionId: makeSessionId(),
    jobId: null,
    revision: 0,
    progress: 0,
    image: '',
    seed: null,
    metadata: null,
    error: null,
    lastUpdated: null,
    paused: false,
  },
  settings: {
    debounceMs: 900,
    previewSize: 512,
    previewSteps: 5,
    finalSteps: 8,
    canvasInfluence: 0.45,
    seed: -1,
    lockSeed: false,
    autoPreview: true,
  },
}

interface AppState {
  params: GenerateParams
  setParam: <K extends keyof GenerateParams>(key: K, value: GenerateParams[K]) => void
  setParams: (partial: Partial<GenerateParams>) => void

  generating: boolean
  jobId: string | null
  progress: number
  results: string[]
  resultsMetadata: Array<Record<string, any>>
  lastSeed: number | null
  generationError: string | null
  setGenerating: (v: boolean) => void
  setJobId: (id: string | null) => void
  setProgress: (n: number) => void
  setResults: (imgs: string[], seed?: number, metadata?: Array<Record<string, any>>) => void
  setError: (e: string | null) => void

  systemReport: SystemReport | null
  setSystemReport: (r: SystemReport) => void
  modelLoaded: boolean
  setModelLoaded: (v: boolean) => void

  loras: LoraInfo[]
  setLoras: (l: LoraInfo[]) => void

  tab: number
  setTab: (n: number) => void
  moodboardView: 'official' | 'favorites' | 'custom' | 'new'
  setMoodboardView: (view: 'official' | 'favorites' | 'custom' | 'new') => void

  lightbox: LightboxState | null
  lightboxImage: string | null
  openLightbox: (items: LightboxItem[], index?: number) => void
  closeLightbox: () => void
  nextLightbox: () => void
  previousLightbox: () => void
  patchLightboxItem: (id: number, partial: Partial<LightboxItem>) => void
  removeLightboxItem: (id: number) => void
  setLightboxImage: (src: string | null) => void

  realtime: RealtimeState
  setRealtime: (partial: Partial<RealtimeState>) => void
  setRealtimeDocument: (document: RealtimeDocument) => void
  setRealtimePreview: (partial: Partial<RealtimePreviewState>) => void
  setRealtimeSettings: (partial: Partial<RealtimeSettings>) => void
}

export const useStore = create<AppState>((set, get) => ({
  params: defaultParams,
  setParam: (key, value) => set(s => ({ params: { ...s.params, [key]: value } })),
  setParams: (partial) => set(s => ({ params: { ...s.params, ...partial } })),

  generating: false,
  jobId: null,
  progress: 0,
  results: [],
  resultsMetadata: [],
  lastSeed: null,
  generationError: null,
  setGenerating: (v) => set({ generating: v }),
  setJobId: (id) => set({ jobId: id }),
  setProgress: (n) => set({ progress: n }),
  setResults: (imgs, seed, metadata) => set({ results: imgs, lastSeed: seed ?? null, resultsMetadata: metadata ?? [] }),
  setError: (e) => set({ generationError: e }),

  systemReport: null,
  setSystemReport: (r) => set({ systemReport: r }),
  modelLoaded: false,
  setModelLoaded: (v) => set({ modelLoaded: v }),

  loras: [],
  setLoras: (l) => set({ loras: l }),

  tab: 0,
  setTab: (n) => set({ tab: n }),
  moodboardView: 'official',
  setMoodboardView: (view) => set({ moodboardView: view }),

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

  realtime: defaultRealtime,
  setRealtime: (partial) => set(s => ({ realtime: { ...s.realtime, ...partial } })),
  setRealtimeDocument: (document) => set(s => ({ realtime: { ...s.realtime, document } })),
  setRealtimePreview: (partial) => set(s => ({
    realtime: { ...s.realtime, preview: { ...s.realtime.preview, ...partial } },
  })),
  setRealtimeSettings: (partial) => set(s => ({
    realtime: { ...s.realtime, settings: { ...s.realtime.settings, ...partial } },
  })),
}))
