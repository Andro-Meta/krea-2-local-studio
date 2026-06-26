import { create } from 'zustand'
import type { LoraInfo, SystemReport } from './api'
import { createDefaultDocument, type RealtimeDocument, type RealtimeTool } from './components/RealtimeStudio/canvasDocument'

export interface ActiveLora {
  name: string
  filename: string
  strength: number
  enabled: boolean
}

export interface GenerateParams {
  prompt: string
  negative_prompt: string
  mode: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint' | 'redraw'
  checkpoint: 'turbo' | 'raw'
  quantization: 'bf16' | 'fp8'
  steps: number
  cfg: number
  mu: number | null
  y1: number
  y2: number
  width: number
  height: number
  num_images: number
  seed: number
  denoise: number
  edit_provider?: 'auto' | 'krea_native' | 'flux_fill'
  quality_preset?: 'fast' | 'balanced' | 'best' | 'raw_benchmark'
  use_rebalance: boolean
  rebalance_multiplier: number
  rebalance_weights: string
  loras: ActiveLora[]
  bboxes: Array<{ label: string; bbox: number[] }>
  init_image_b64: string
  mask_b64: string
  ref_image1_b64: string
  ref_image2_b64: string
  ref_image3_b64: string
  use_prompt_expander: boolean
  refine: boolean
  refine_denoise: number
  refine_steps: number
  mood: string
  moodboard_strength: number
  moodboard_images: string[]
}

export interface LightboxItem {
  src: string
  id?: number
  filename?: string
  prompt?: string
  favorite?: boolean
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
  checkpoint: 'turbo',
  quantization: 'bf16',
  steps: 8,
  cfg: 0.0,
  mu: 1.15,
  y1: 0.5,
  y2: 1.15,
  width: 1024,
  height: 1024,
  num_images: 1,
  seed: -1,
  denoise: 0.75,
  edit_provider: 'auto',
  quality_preset: 'balanced',
  use_rebalance: true,
  rebalance_multiplier: 4.0,
  rebalance_weights: '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0',
  loras: [],
  bboxes: [],
  init_image_b64: '',
  mask_b64: '',
  ref_image1_b64: '',
  ref_image2_b64: '',
  ref_image3_b64: '',
  use_prompt_expander: false,
  refine: false,
  refine_denoise: 0.3,
  refine_steps: 6,
  mood: '',
  moodboard_strength: 0.5,
  moodboard_images: [],
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
    error: null,
    lastUpdated: null,
    paused: false,
  },
  settings: {
    debounceMs: 900,
    previewSize: 512,
    previewSteps: 5,
    finalSteps: 8,
    canvasInfluence: 0.6,
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
  lastSeed: number | null
  generationError: string | null
  setGenerating: (v: boolean) => void
  setJobId: (id: string | null) => void
  setProgress: (n: number) => void
  setResults: (imgs: string[], seed?: number) => void
  setError: (e: string | null) => void

  systemReport: SystemReport | null
  setSystemReport: (r: SystemReport) => void
  modelLoaded: boolean
  setModelLoaded: (v: boolean) => void

  loras: LoraInfo[]
  setLoras: (l: LoraInfo[]) => void

  tab: number
  setTab: (n: number) => void

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
  lastSeed: null,
  generationError: null,
  setGenerating: (v) => set({ generating: v }),
  setJobId: (id) => set({ jobId: id }),
  setProgress: (n) => set({ progress: n }),
  setResults: (imgs, seed) => set({ results: imgs, lastSeed: seed ?? null }),
  setError: (e) => set({ generationError: e }),

  systemReport: null,
  setSystemReport: (r) => set({ systemReport: r }),
  modelLoaded: false,
  setModelLoaded: (v) => set({ modelLoaded: v }),

  loras: [],
  setLoras: (l) => set({ loras: l }),

  tab: 0,
  setTab: (n) => set({ tab: n }),

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
