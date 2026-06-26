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
  mode?: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint'
  checkpoint?: 'turbo' | 'raw'
  checkpoint_path?: string
  quantization?: 'bf16' | 'fp8'
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
  loras?: Array<{ name: string; filename?: string; strength?: number; enabled?: boolean }>
  use_rebalance?: boolean
  rebalance_multiplier?: number
  rebalance_weights?: string
  bboxes?: Array<{ label: string; bbox: number[] }>
  init_image_b64?: string
  mask_b64?: string
  ref_image1_b64?: string
  ref_image2_b64?: string
  ref_image3_b64?: string
  use_prompt_expander?: boolean
  refine?: boolean
  refine_denoise?: number
  refine_steps?: number
  mood?: string
  moodboard_strength?: number
  moodboard_images?: string[]
}

export interface Mood {
  id: string
  name: string
  emoji: string
  category: string
  keywords: string
  avoids: string
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
  model_status: { loaded: boolean; loading?: boolean; checkpoint?: string; quantization?: string; auto_checkpoint?: string; auto_quant?: string; load_error?: string | null }
  support_models?: Array<{ id: string; label: string; repo_id: string; purpose: string; installed: boolean; cache_dir: string }>
  variants: Array<{ id: string; label: string; vram_gb: number; ram_gb: number; blockers: string[]; warnings: string[]; ok: boolean }>
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
  has_ideogram_api_key: boolean
  has_openrouter_api_key: boolean
}

export const apiFetch = {
  generate: (req: GenerationRequest) =>
    api.post<{ job_id: string; status: string }>('/api/generate', req).then(r => r.data),

  jobStatus: (jobId: string) =>
    api.get<{ job_id: string; status: string; progress: number; images: string[]; error?: string; seed?: number }>(`/api/generate/${jobId}`).then(r => r.data),

  loadModel: (path: string, quant: string) =>
    api.post('/api/load-model', { checkpoint_path: path, quantization: quant }).then(r => r.data),

  unloadModel: () => api.post('/api/unload-model').then(r => r.data),

  gallery: (page = 1, pageSize = 50, favorites = false) =>
    api.get<{ items: GalleryItem[]; total: number }>(`/api/gallery?page=${page}&page_size=${pageSize}&favorites=${favorites}`).then(r => r.data),

  setFavorite: (id: number, favorite: boolean) =>
    api.put(`/api/gallery/${id}/favorite`, { favorite }).then(r => r.data),

  deleteGalleryItem: (id: number) =>
    api.delete(`/api/gallery/${id}`).then(r => r.data),

  loras: () => api.get<LoraInfo[]>('/api/loras').then(r => r.data),

  moods: () => api.get<Mood[]>('/api/moods').then(r => r.data),

  upscale: (image_b64: string, method: string, opts?: { scale?: number; denoise?: number; prompt?: string; tile_size?: number; seam_fix?: boolean }) =>
    api.post<{ image_b64: string }>('/api/upscale', {
      image_b64, method,
      scale: opts?.scale ?? (method === 'realesrgan' ? 4 : 2),
      denoise: opts?.denoise ?? (method === 'ultimate' ? 0.3 : 0.24),
      prompt: opts?.prompt ?? '',
      tile_size: opts?.tile_size ?? 1024,
      seam_fix: opts?.seam_fix ?? true,
    }, { timeout: 1800000 }).then(r => r.data),

  autoMask: (image_b64: string, prompt: string, threshold?: number) =>
    api.post<{ mask_b64: string }>('/api/automask', { image_b64, prompt, threshold: threshold ?? 0.35 })
      .then(r => r.data.mask_b64),

  describeImage: (image_b64: string) =>
    api.post<{ prompt: string; backend: 'local' | 'openrouter' }>('/api/describe-image', { image_b64 })
      .then(r => r.data),

  system: () => api.get<SystemReport>('/api/system').then(r => r.data),

  downloadSupportModels: () =>
    api.post<{ ok: boolean; status: SystemReport['support_models'] }>('/api/support-models/download', {}, { timeout: 3600000 })
      .then(r => r.data),

  settings: () => api.get<AppSettings>('/api/settings').then(r => r.data),
  updateSettings: (data: Partial<AppSettings> & { ideogram_api_key?: string; openrouter_api_key?: string }) =>
    api.put('/api/settings', data).then(r => r.data),

  expandPrompt: (prompt: string) =>
    api.post<{ expanded: string; changed: boolean; error?: string | null; backend: 'local' | 'openrouter' | 'ideogram-json' }>('/api/expand-prompt', { prompt }).then(r => r.data),

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
