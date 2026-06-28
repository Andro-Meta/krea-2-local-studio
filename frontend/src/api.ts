import axios from 'axios'

export function publicBasePath(): string {
  if (typeof window === 'undefined') return ''
  const match = window.location.pathname.match(/^\/krea(?:\/|$)/)
  return match ? '/krea' : ''
}

export function publicUrl(path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`
  return `${publicBasePath()}${normalized}`
}

const api = axios.create({ baseURL: publicBasePath() })

export interface GenerationRequest {
  prompt: string
  negative_prompt?: string
  mode?: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint' | 'redraw'
  model_profile?: 'krea_turbo' | 'krea_raw' | 'qwen_image_edit' | 'lens_turbo' | 'ernie_turbo' | 'z_image_turbo' | ''
  checkpoint?: 'turbo' | 'raw'
  checkpoint_path?: string
  quantization?: 'bf16' | 'fp8' | 'fp16'
  steps?: number
  cfg?: number
  mu?: number | null
  y1?: number
  y2?: number
  width?: number
  height?: number
  num_images?: number
  seed?: number
  denoise?: number
  sampler?: 'euler' | 'euler_flow' | 'exp_heun_2_x0_sde' | 'lcm' | 'dpmpp_2m' | 'ddim' | 'uni_pc'
  scheduler?: 'simple'
  inpaint_method?: 'native' | 'lanpaint_experimental' | 'flux_fill'
  lanpaint_inner_steps?: number
  lanpaint_strength?: number
  lanpaint_lambda?: number
  lanpaint_step_size?: number
  lanpaint_beta?: number
  lanpaint_friction?: number
  lanpaint_early_stop?: number
  lanpaint_prompt_mode?: 'Image First' | 'Prompt First'
  edit_provider?: 'auto' | 'krea_native' | 'flux_fill'
  quality_preset?: 'fast' | 'balanced' | 'best' | 'raw_benchmark'
  creativity?: 'raw' | 'low' | 'medium' | 'high'
  style_references?: Array<{
    image_b64: string
    strength?: number
    role?: string
    token_size?: 'low' | 'normal' | 'high' | 'max'
    mask_b64?: string
    mask_padding?: number
    vision_megapixels?: number | null
    system_prompt?: string
    vision_position?: 'before_prompt' | 'after_prompt'
  }>
  style_fusion_mode?: 'style_only' | 'preserve_structure' | 'semantic_fusion'
  regional_prompts?: Array<{
    prompt: string
    negative_prompt?: string
    mask_b64?: string
    strength?: number
    feather?: number
    normalize?: boolean
    visible?: boolean
    lora_filter?: string
  }>
  regional_base_prompt_strength?: number
  regional_normalize_masks?: boolean
  loras?: Array<{
    name: string
    filename?: string
    strength?: number
    enabled?: boolean
    block_filter?: 'all' | 'early' | 'middle' | 'late' | 'style_safe' | 'custom'
    custom_blocks?: string[]
  }>
  use_rebalance?: boolean
  rebalance_multiplier?: number
  rebalance_weights?: string
  rebalance_mode?: 'legacy_multiply' | 'rms_renorm'
  rebalance_preset?: 'legacy' | 'subtle' | 'balanced' | 'detail' | 'emotion' | 'uniform' | 'custom'
  rebalance_renormalize?: boolean
  edit_rebalance_enabled?: boolean
  edit_rebalance_profile?: 'default' | 'edit' | 'conservative'
  conditioning_mode?: 'auto' | 'qwen_image_edit_plus' | 'qwen_reference'
  krea_enhancer_variant?: 'off' | 'current' | 'capped_delta' | 'current_plus_capped'
  krea_enhancer_enabled?: boolean
  krea_enhancer_strength?: number
  krea_enhancer_delta_cap?: number
  bboxes?: Array<{ label: string; bbox: number[] }>
  init_image_b64?: string
  mask_b64?: string
  ref_image1_b64?: string
  ref_image2_b64?: string
  ref_image3_b64?: string
  use_prompt_planner?: boolean
  prompt_planner_max_tokens?: number
  prompt_planner_show_output?: boolean
  prompt_planner_lock_original?: boolean
  prompt_planner_use_regions?: boolean
  use_prompt_expander?: boolean
  think_steering_enabled?: boolean
  think_text?: string
  refine?: boolean
  refine_denoise?: number
  refine_steps?: number
  mood?: string
  moodboard_ids?: number[]
  moodboard_uuids?: string[]
  moodboard_strength?: number
  moodboard_images?: string[]
  seed_variance_preset?: 'off' | 'subtle' | 'balanced' | 'creative' | 'bold' | 'custom'
  seed_variance_strength?: number
  seed_variance_protection?: 'none' | 'first_quarter' | 'first_half'
  seed_variance_direction?: 'none' | 'forward' | 'reverse' | 'center' | 'edges'
  seed_variance_fade_curve?: 'linear' | 'ease_in' | 'ease_out' | 'smoothstep'
  seed_variance_injection_start?: number
  seed_variance_injection_end?: number
}

export interface PromptPlan {
  original_prompt: string
  planned_prompt: string
  negative_prompt: string
  subject: string
  composition: string
  style: string
  lighting: string
  materials: string
  text_rendering: string
  regions: Array<Record<string, any>>
  backend: 'local' | 'heuristic' | 'off'
  changed: boolean
  error?: string | null
}

export interface PromptRecipe {
  id: string
  name: string
  description: string
  prompt: string
  negative_prompt: string
  planner_instruction: string
  loras: any[]
  moodboard_ids: number[]
  moodboard_uuids: string[]
  style_references: any[]
  regional_prompts: any[]
  seed_variance_preset: string
  krea_enhancer_variant: string
  rebalance_preset: string
  updated_at: string
}

export interface RealtimePreviewRequest {
  session_id: string
  prompt: string
  negative_prompt?: string
  canvas_image_b64: string
  width: number
  height: number
  preview_steps?: number
  moodboard_strength?: number
  seed?: number
}

export interface RealtimePreviewJob {
  job_id: string
  session_id: string
  revision: number
  status: 'queued' | 'running' | 'done' | 'stale' | 'cancelled' | 'error'
  progress: number
  image_b64?: string
  seed?: number | null
  metadata?: Record<string, any> | null
  error?: string | null
}

export interface Mood {
  id: string
  name: string
  emoji: string
  category: string
  keywords: string
  avoids: string
}

export interface MoodboardItem {
  id: number
  url: string
  slug: string
  uuid: string
  title: string
  taste_profile: string
  keywords: string[]
  primary_image_url: string
  image_urls: string[]
  related_urls: string[]
  favorite: boolean
  source: 'official' | 'custom'
  first_seen_at: string
  last_seen_at: string
  updated_at: string
  sync_error: string
  qwen_guidance: Record<string, any>
  qwen_guidance_at: string
  qwen_guidance_version: number
}

export interface MoodboardDiscovery {
  id: string
  discovered_at: string
  new_count: number
  new_ids: number[]
  items: MoodboardItem[]
}

export interface GalleryItem {
  id: number
  filename: string
  prompt: string
  checkpoint: string
  width: number
  height: number
  seed: number
  created_at: string
  favorite: boolean
  thumbnail_b64?: string
  metadata?: Record<string, any>
}

export interface LoraInfo {
  filename: string
  name: string
  display_name: string
  trigger_words: string[]
  strength: number
  is_official: boolean
  installed: boolean
  compatible?: boolean
  match_info?: string
}

export interface SystemReport {
  gpu_name?: string
  vram_total_gb?: number
  vram_free_gb?: number
  ram_total_gb?: number
  ram_available_gb?: number
  disk_free_gb?: number
  gpu_processes: string[]
  gpu_process_details?: Array<{ pid: number; name: string; used_memory_gb?: number }>
  model_status: { loaded: boolean; loading?: boolean; checkpoint?: string; quantization?: string; auto_checkpoint?: string; auto_quant?: string; load_error?: string | null; text_encoder_source?: { kind: string; path: string; runtime?: string; status?: string } | null; memory?: Record<string, any> }
  attention_acceleration?: { status: string; available: boolean; reason: string; recommendation: string }
  gpu_capabilities?: { name: string; arch: string; compute_capability: string | null; vram_total_gb: number | null; supports_bf16: boolean; supports_fp8_compute: boolean; supports_nvfp4: boolean; fp8_storage_only: boolean; fp8_note: string }
  recommended_runtime?: { quantization: string; blocks_to_swap: number; max_tier: string; notes: string }
  runnability?: { can_run: boolean; tier: string; compute_dtype: string; blocks_to_swap: number; max_tier: string; reason: string }
  support_models?: Array<{ id: string; label: string; repo_id: string; purpose: string; installed: boolean; optional?: boolean; cache_dir: string }>
  variants: Array<{ id: string; label: string; vram_gb: number; ram_gb: number; blockers: string[]; warnings: string[]; ok: boolean }>
}

export interface KreaServerProcess {
  pid: number
  port?: number | null
  command_line: string
  used_memory_gb?: number
  can_stop: boolean
}

export interface AuthSession {
  authenticated: boolean
  share_auth: boolean
  username?: string | null
  role?: 'admin' | 'user' | null
}

export interface ShareUser {
  username: string
  role: 'admin' | 'user'
}

export interface SharingStatus {
  tailscale: {
    installed: boolean
    connected: boolean
    tailscale_path?: string | null
    download_url: string
    message: string
  }
  funnel: {
    installed: boolean
    running: boolean
    url: string
    message: string
  }
  public_path: string
}

export interface AppSettings {
  hf_token: string
  civitai_token: string
  krea2_turbo_path: string
  krea2_raw_path: string
  output_dir: string
  prompt_expander_backend: 'local' | 'openrouter' | 'ideogram-json'
  openrouter_model: string
  openrouter_free_only: boolean
  krea_share_auto_funnel: boolean
  has_hf_token: boolean
  has_civitai_token: boolean
  has_ideogram_api_key: boolean
  has_openrouter_api_key: boolean
}

export interface QualityAsset {
  id: string
  repo_id: string
  filename?: string | null
  local_path: string
  purpose: string
  installed: boolean
  needs_token: boolean
  gated: boolean
  setup_url: string
}

export const apiFetch = {
  generate: (req: GenerationRequest) =>
    api.post<{ job_id: string; status: string }>('/api/generate', req).then(r => r.data),

  jobStatus: (jobId: string) =>
    api.get<{ job_id: string; status: string; progress: number; images: string[]; error?: string; seed?: number; metadata?: Record<string, any>[] }>(`/api/generate/${jobId}`).then(r => r.data),

  realtimePreview: (req: RealtimePreviewRequest) =>
    api.post<RealtimePreviewJob>('/api/realtime/preview', req, { timeout: 120000 }).then(r => r.data),

  realtimePreviewStatus: (jobId: string) =>
    api.get<RealtimePreviewJob>(`/api/realtime/preview/${jobId}`).then(r => r.data),

  cancelRealtimePreview: (jobId: string) =>
    api.post<{ ok: boolean; job_id: string; status: string }>(`/api/realtime/cancel/${jobId}`).then(r => r.data),

  loadModel: (path: string, quant: string, blocksToSwap = 0) =>
    api.post('/api/load-model', { checkpoint_path: path, quantization: quant, blocks_to_swap: blocksToSwap }).then(r => r.data),

  unloadModel: () => api.post('/api/unload-model').then(r => r.data),

  releaseTransientMemory: () => api.post('/api/memory/release-transient').then(r => r.data),
  unloadModelMemory: () => api.post('/api/memory/unload-model').then(r => r.data),
  memoryProcesses: () => api.get<{ items: KreaServerProcess[] }>('/api/memory/processes').then(r => r.data),
  stopMemoryProcess: (pid: number) => api.post('/api/memory/stop-process', { pid }).then(r => r.data),

  gallery: (page = 1, pageSize = 50, favorites = false) =>
    api.get<{ items: GalleryItem[]; total: number }>(`/api/gallery?page=${page}&page_size=${pageSize}&favorites=${favorites}`).then(r => r.data),

  setFavorite: (id: number, favorite: boolean) =>
    api.put(`/api/gallery/${id}/favorite`, { favorite }).then(r => r.data),

  deleteGalleryItem: (id: number) =>
    api.delete(`/api/gallery/${id}`).then(r => r.data),

  loras: () => api.get<LoraInfo[]>('/api/loras').then(r => r.data),

  moods: () => api.get<Mood[]>('/api/moods').then(r => r.data),

  moodboards: (opts?: { q?: string; page?: number; pageSize?: number; favorites?: boolean; source?: 'official' | 'custom' }) => {
    const params = new URLSearchParams()
    if (opts?.q) params.set('q', opts.q)
    params.set('page', String(opts?.page ?? 1))
    params.set('page_size', String(opts?.pageSize ?? 50))
    params.set('favorites', String(opts?.favorites ?? false))
    if (opts?.source) params.set('source', opts.source)
    return api.get<{ items: MoodboardItem[]; total: number }>(`/api/moodboards?${params.toString()}`).then(r => r.data)
  },

  moodboard: (id: number) =>
    api.get<MoodboardItem>(`/api/moodboards/${id}`).then(r => r.data),

  setMoodboardFavorite: (id: number, favorite: boolean) =>
    api.put(`/api/moodboards/${id}/favorite`, { favorite }).then(r => r.data),

  generateMoodboardGuidance: (id: number) =>
    api.post<MoodboardItem>(`/api/moodboards/${id}/qwen-guidance`, {}, { timeout: 180000 }).then(r => r.data),

  generateMissingMoodboardGuidance: (limit = 10) =>
    api.post<{ processed: number; items: MoodboardItem[] }>('/api/moodboards/qwen-guidance-missing', { limit }, { timeout: 600000 }).then(r => r.data),

  createCustomMoodboard: (req: { title: string; taste_profile?: string; keywords?: string[]; image_b64s: string[] }) =>
    api.post<MoodboardItem>('/api/moodboards/custom', req, { timeout: 120000 }).then(r => r.data),

  createMoodboardMashup: (req: { moodboard_ids: number[]; weights?: number[] }) =>
    api.post<MoodboardItem>('/api/moodboards/mashup', req, { timeout: 180000 }).then(r => r.data),

  deleteCustomMoodboard: (id: number) =>
    api.delete(`/api/moodboards/custom/${id}`).then(r => r.data),

  importMoodboards: (urls: string[] = [], maxPages = 200) =>
    api.post<{ imported: number; ids: number[]; new_count: number; new_ids: number[] }>('/api/moodboards/import', { urls, max_pages: maxPages }, { timeout: 180000 })
      .then(r => r.data),

  latestMoodboardDiscovery: () =>
    api.get<MoodboardDiscovery>('/api/moodboards/discoveries/latest').then(r => r.data),

  exportMoodboardSeed: () =>
    api.post<{ exported: number; path: string }>('/api/moodboards/export-seed').then(r => r.data),

  moodboardImage: (url: string) =>
    api.post<{ image_b64: string }>('/api/moodboards/image', { url }, { timeout: 120000 })
      .then(r => r.data.image_b64),

  upscale: (image_b64: string, method: string, opts?: {
    scale?: number
    upscale_by?: number
    denoise?: number
    prompt?: string
    tile_size?: number
    tile_width?: number
    tile_height?: number
    tile_padding?: number
    mask_blur?: number
    seam_mode?: 'none' | 'band_pass' | 'half_tile' | 'half_tile_intersections'
    tile_mode?: 'linear' | 'chess'
    sampler?: string
    scheduler?: string
    steps?: number
    cfg?: number
    tiled_decode?: boolean
    seam_fix?: boolean
  }) =>
    api.post<{ image_b64: string; metadata?: Record<string, any> }>('/api/upscale', {
      image_b64, method,
      scale: opts?.scale ?? (method === 'realesrgan' ? 4 : 2),
      upscale_by: opts?.upscale_by ?? 2,
      denoise: opts?.denoise ?? (method === 'ultimate' ? 0.3 : 0.24),
      prompt: opts?.prompt ?? '',
      tile_size: opts?.tile_size ?? 1024,
      tile_width: opts?.tile_width ?? opts?.tile_size ?? 1024,
      tile_height: opts?.tile_height ?? opts?.tile_size ?? 1024,
      tile_padding: opts?.tile_padding ?? 96,
      mask_blur: opts?.mask_blur ?? 12,
      seam_mode: opts?.seam_mode ?? 'band_pass',
      tile_mode: opts?.tile_mode ?? 'chess',
      sampler: opts?.sampler ?? 'euler',
      scheduler: opts?.scheduler ?? 'simple',
      steps: opts?.steps ?? 8,
      cfg: opts?.cfg ?? 1,
      tiled_decode: opts?.tiled_decode ?? false,
      seam_fix: opts?.seam_fix ?? true,
    }, { timeout: 1800000 }).then(r => r.data),

  autoMask: (image_b64: string, prompt: string, threshold?: number) =>
    api.post<{ mask_b64: string }>('/api/automask', { image_b64, prompt, threshold: threshold ?? 0.35 })
      .then(r => r.data.mask_b64),

  preprocessorPreview: (
    image_b64: string,
    opts?: { kind?: 'canny' | 'soft_edge' | 'lineart' | 'depth'; resolution?: number; low_threshold?: number; high_threshold?: number },
  ) =>
    api.post<{ image_b64: string; kind: string; width: number; height: number }>('/api/preprocess/preview', {
      image_b64,
      kind: opts?.kind ?? 'canny',
      resolution: opts?.resolution ?? 768,
      low_threshold: opts?.low_threshold ?? 80,
      high_threshold: opts?.high_threshold ?? 160,
    }).then(r => r.data),

  describeImage: (image_b64: string) =>
    api.post<{ prompt: string; backend: 'local' | 'openrouter' }>('/api/describe-image', { image_b64 })
      .then(r => r.data),

  system: () => api.get<SystemReport>('/api/system').then(r => r.data),

  downloadSupportModels: () =>
    api.post<{ ok: boolean; status: SystemReport['support_models'] }>('/api/support-models/download', {}, { timeout: 3600000 })
      .then(r => r.data),

  qualityAssets: () =>
    api.get<{ has_hf_token: boolean; items: QualityAsset[] }>('/api/quality-assets').then(r => r.data),

  downloadQualityAsset: (assetId: string) =>
    api.post<{ ok: boolean; path: string; item: QualityAsset }>(`/api/quality-assets/${assetId}/download`, {}, { timeout: 7200000 })
      .then(r => r.data),

  settings: () => api.get<AppSettings>('/api/settings').then(r => r.data),
  updateSettings: (data: Partial<AppSettings> & { hf_token?: string; ideogram_api_key?: string; openrouter_api_key?: string }) =>
    api.put('/api/settings', data).then(r => r.data),

  expandPrompt: (prompt: string) =>
    api.post<{ expanded: string; changed: boolean; error?: string | null; backend: 'local' | 'openrouter' | 'ideogram-json' }>('/api/expand-prompt', { prompt }).then(r => r.data),
  planPrompt: (prompt: string, max_tokens = 700) =>
    api.post<PromptPlan>('/api/plan-prompt', { prompt, max_tokens }).then(r => r.data),
  promptingGuide: () =>
    api.get<{ guidelines: string; examples: string[]; source: string }>('/api/prompting-guide').then(r => r.data),
  resolutionOptions: () =>
    api.get<{ tiers: string[]; aspects: string[]; dimensions: Record<string, Record<string, [number, number]>> }>('/api/resolution-options').then(r => r.data),
  runtimeAdvice: (width: number, height: number, quantization: string) =>
    api.get<{ blocks_to_swap: number; tiled_decode: boolean; fits: boolean; estimated_vram_gb: number; megapixels: number; warnings: string[]; free_vram_gb: number | null }>(
      `/api/runtime-advice?width=${width}&height=${height}&quantization=${encodeURIComponent(quantization)}`,
    ).then(r => r.data),
  promptRecipes: () => api.get<{ items: PromptRecipe[] }>('/api/prompt-recipes').then(r => r.data),
  savePromptRecipe: (recipe: Partial<PromptRecipe> & { name: string }) =>
    api.post<PromptRecipe>('/api/prompt-recipes', recipe).then(r => r.data),
  deletePromptRecipe: (id: string) => api.delete<{ ok: boolean }>(`/api/prompt-recipes/${encodeURIComponent(id)}`).then(r => r.data),

  authMe: () => api.get<AuthSession>('/api/auth/me').then(r => r.data),
  logout: () => api.post('/api/auth/logout').then(r => r.data),

  listUsers: () => api.get<{ users: ShareUser[] }>('/api/admin/users').then(r => r.data.users),
  addUser: (username: string, password: string, role: 'admin' | 'user') =>
    api.post<{ users: ShareUser[] }>('/api/admin/users', { username, password, role }).then(r => r.data.users),
  setUserRole: (username: string, role: 'admin' | 'user') =>
    api.put<{ users: ShareUser[] }>(`/api/admin/users/${encodeURIComponent(username)}/role`, { role }).then(r => r.data.users),
  resetUserPassword: (username: string, password: string) =>
    api.put(`/api/admin/users/${encodeURIComponent(username)}/password`, { password }).then(r => r.data),
  removeUser: (username: string) =>
    api.delete<{ users: ShareUser[] }>(`/api/admin/users/${encodeURIComponent(username)}`).then(r => r.data.users),

  sharingStatus: () => api.get<SharingStatus>('/api/sharing/status').then(r => r.data),
  tailscaleUp: () => api.post('/api/sharing/tailscale-up').then(r => r.data),
  startSharing: () => api.post<{ ok: boolean; url: string; message: string }>('/api/sharing/funnel/start').then(r => r.data),
  stopSharing: () => api.post<{ ok: boolean; message: string }>('/api/sharing/funnel/stop').then(r => r.data),

  downloadLora: (name: string) =>
    api.post<{ ok: boolean; path: string }>(`/api/loras/${name}/download`).then(r => r.data),

  importLoraUrl: (url: string, filename?: string, civitaiToken?: string) =>
    api.post<{ ok: boolean; path: string; filename: string; skipped?: boolean; compatible?: boolean; match_info?: string }>(
      '/api/loras/import',
      { url, filename: filename ?? '', civitai_token: civitaiToken ?? '' }
    ).then(r => r.data),
}

export function connectWS(jobId: string, onMessage: (data: unknown) => void): WebSocket {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = window.location.host
  const ws = new WebSocket(`${proto}://${host}${publicBasePath()}/ws/${jobId}`)
  ws.onmessage = e => onMessage(JSON.parse(e.data))
  return ws
}

export default api
