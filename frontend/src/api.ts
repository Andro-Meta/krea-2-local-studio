import axios from 'axios'
import animateContract from './generated/animate-contract.json' with { type: 'json' }

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

// Session expiry lands as 401 on every call. Instead of each component
// showing its own cryptic failure, send the user to the login page once.
// The auth probe itself (/api/auth/me) is exempt: it legitimately returns
// 401 in logged-out states that don't require a redirect (e.g. local mode
// probing) and is handled by its callers.
api.interceptors.response.use(
  response => response,
  error => {
    const status = error?.response?.status
    const url: string = error?.config?.url ?? ''
    if (status === 401 && typeof window !== 'undefined'
        && !url.includes('/api/auth/')
        && !window.location.pathname.endsWith('/login')) {
      window.location.href = `${publicBasePath()}/login`
    }
    return Promise.reject(error)
  },
)

export interface GenerationRequest {
  prompt: string
  negative_prompt?: string
  mode?: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint' | 'redraw' | 'character_edit' | 'turbo_4x'
  turbo_int8_variant?: string
  model_profile?: 'krea_turbo' | 'krea_raw' | 'qwen_image_edit' | 'lens_turbo' | 'ernie_turbo' | 'z_image_turbo' | ''
  diffusion_engine?: 'native_pytorch' | 'native_gguf' | 'native_int8_convrot'
  checkpoint?: 'turbo' | 'raw'
  checkpoint_path?: string
  quantization?: 'bf16' | 'fp8' | 'gguf' | 'fp16' | 'int8'
  steps?: number
  cfg?: number
  mu?: number | null
  y1?: number
  y2?: number
  width?: number
  height?: number
  num_images?: number
  batch_mode?: 'safe_queue' | 'parallel'
  parallel_batch_confirmed?: boolean
  batch_int8_all?: boolean
  seed?: number
  denoise?: number
  sampler?: 'euler' | 'euler_flow' | 'euler_ancestral' | 'euler_ancestral_cfg_pp' | 'euler_cfg_pp' | 'er_sde' | 'res_2s' | 'res_3s' | 'exp_heun_2_x0_sde' | 'lcm' | 'dpmpp_2m' | 'ddim' | 'uni_pc'
  scheduler?: 'simple' | 'normal' | 'beta' | 'beta57' | 'sgm_uniform' | 'bong_tangent' | 'kl_optimal' | 'karras' | 'exponential'
  inpaint_method?: 'native' | 'lanpaint_experimental'
  differential_inpaint?: boolean
  differential_strength?: number
  cfg_zero_star?: boolean
  cfg_zero_init_steps?: number
  res4lyf_sampler?: string
  res4lyf_eta?: number
  res4lyf_bongmath?: boolean
  actual_denoise?: boolean
  incontext_edit?: boolean
  incontext_image_b64?: string
  incontext_mask_b64?: string
  incontext_vision_position?: 'before' | 'after'
  incontext_vision_megapixels?: number
  incontext_encoder?: 'krea2' | 'qwen_edit_plus'
  incontext_system_prompt?: string
  character_edit_source_b64?: string
  character_edit_reference_b64?: string
  character_edit_regions?: Array<{ x: number; y: number; w: number; h: number; prompt: string; reference_b64?: string; strength?: number; feather?: number }>
  character_edit_grounding_px?: number
  character_edit_task?: 'restage' | 'local_edit' | 'replace' | 'restyle' | 'removal' | 'two_reference'
  character_edit_lora_strength?: number
  style_transfer_image_b64?: string
  style_transfer_method?: 'AdaIN' | 'WCT' | 'WCT2' | 'scattersort'
  style_transfer_weight?: number
  style_transfer_apply_to?: 'denoised' | 'positive' | 'negative'
  lanpaint_inner_steps?: number
  lanpaint_strength?: number
  lanpaint_lambda?: number
  lanpaint_step_size?: number
  lanpaint_beta?: number
  lanpaint_friction?: number
  lanpaint_early_stop?: number
  lanpaint_prompt_mode?: 'Image First' | 'Prompt First'
  edit_provider?: 'auto' | 'krea_native'
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
  image_prompt_enabled?: boolean
  image_prompt_mode?: 'match_style' | 'copy_composition'
  image_prompt_strength?: number
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
  /** Strip the 2px Qwen/Wan VAE grid after decode (default on). */
  vae_degrid?: boolean
  mood?: string
  moodboard_ids?: number[]
  moodboard_uuids?: string[]
  moodboard_strength?: number
  moodboard_images?: string[]
  seed_variance_preset?: 'off' | 'subtle' | 'balanced' | 'creative' | 'bold' | 'wild' | 'custom'
  seed_variance_strength?: number
  seed_variance_algorithm?: 'legacy' | 'rbg'
  seed_variance_model_type?: 'krea2' | 'z_image' | 'qwen_image' | 'flux' | 'sdxl' | 'other'
  seed_variance_randomize_percent?: number
  seed_variance_shift_strength?: number
  seed_variance_protection?: 'none' | 'first_quarter' | 'first_half' | 'last_quarter' | 'last_half'
  seed_variance_direction?: 'none' | 'forward' | 'reverse' | 'center' | 'edges' | 'chaos' | 'order' | 'abstract' | 'realistic' | 'vibrant' | 'moody' | 'dreamy' | 'dynamic_pose' | 'composition' | 'diversity' | 'facevar' | 'visceral_expression_grit' | 'semantic_drift' | 'structural_lock' | 'cinematic_framing' | 'identity_stretch' | 'texture_lift'
  seed_variance_fade_curve?: 'instant' | 'linear' | 'ease_in' | 'ease_out' | 'ease_in_out' | 'smoothstep' | 'burst'
  seed_variance_injection_start?: number
  seed_variance_injection_end?: number
  seed_variance_schedule?: 'constant' | 'decreasing' | 'step_cutoff' | 'hard_lock' | 'tiered_release'
  seed_variance_cutoff_step?: number
  depth_control?: boolean
  depth_control_strength?: number
  depth_estimator?: 'da3' | 'depth_anything_v2' | 'zoe' | 'midas'
  depth_resolution?: number
  depth_invert?: boolean
  god_mode?: boolean
  mrflow?: boolean
  mrflow_upscaler?: 'esrgan_x2' | 'remacri_x4'
  mrflow_preset?: string
  mrflow_refine_denoise?: number
  mrflow_refine_steps?: number
  seed_variance_total_steps?: number
  seed_variance_cutoff_strength?: number
}

export interface AnimateRequest {
  prompt_schedule: string
  negative_prompt: string
  duration_seconds: number
  fps: number
  render_frames: number | null
  width: number
  height: number
  steps: number
  sampler_name: string
  scheduler: string
  seed: number
  seed_behavior: 'fixed' | 'iter' | 'random' | 'ladder'
  animation_mode: '2D' | '3D' | 'Video Input' | 'None'
  border_mode: 'replicate' | 'reflect' | 'wrap' | 'black'
  cfg_schedule: string
  strength_schedule: string
  zoom_schedule: string
  angle_schedule: string
  translation_x_schedule: string
  translation_y_schedule: string
  translation_z_schedule: string
  rotation_3d_x_schedule: string
  rotation_3d_y_schedule: string
  rotation_3d_z_schedule: string
  color_coherence: 'None' | 'Match Frame 0 LAB'
  diffusion_cadence: number
  prompt_blend_frames: number
  prompt_strength_boost: number
  prompt_strength_boost_frames: number
  hybrid_strength_schedule: string
  hybrid_mode: 'normal' | 'optical_flow'
  init_image_b64: string
  source_video_upload_id: string
}

export interface AnimationResult {
  video_url: string
  poster_url: string
  frame_count: number
  fps: number
  duration: number
  gallery_id: number
}

export interface AnimationUploadResponse {
  upload_id: string
  size: number
  sha256: string
  frame_count: number
  width: number
  height: number
  duration: number
}

export interface AnimationLimits {
  chunk_size: number
  max_frames: number
  max_dimension: number
  max_upload_bytes: number
  uploads_per_user: number
  upload_bytes_per_user: number
  uploads_global: number
  upload_bytes_global: number
  upload_cleanup_interval_seconds: number
  max_source_duration_seconds: number
  active_per_user: number
  upload_content_types: string[]
}

export interface KreaDeforumStatus {
  available: boolean
  missing_nodes: string[]
  incompatible_capabilities: string[]
  variants: string[]
  revision: string
  external: boolean
  license: string
  patch_version: string
  patched_animator_sha256?: string
  patch_sha256?: string
  probe_failed: boolean
  stale: boolean
  midas_ready: boolean
  midas_reason: string
}

export type GpuTaskKind =
  | 'generation'
  | 'animation'
  | 'prompt_expand'
  | 'prompt_plan'
  | 'image_describe'
  | 'upscale'
  | 'depth_preview'
  | 'moodboard_guidance'
  | 'background_enrichment'
  | 'model_warmup'

export type GpuTaskStatus =
  | 'queued'
  | 'running'
  | 'cancellation_requested'
  | 'finalizing'
  | 'done'
  | 'error'
  | 'blocked'
  | 'cancelled'

export interface GpuTaskResponse<TResult = unknown> {
  job_id: string
  status: GpuTaskStatus
  progress: number
  images: string[]
  error?: string | null
  seed?: number | null
  metadata?: Record<string, any>[]
  queue_position?: number | null
  queue_length?: number | null
  moderation_event_id?: number
  batch_id?: string
  child_job_ids?: string[]
  task_kind?: GpuTaskKind
  priority_class?: 'interactive' | 'background'
  result?: TResult | null
  type?: string
  pct?: number
  provider_warning?: string
  lora_warnings?: Array<{ name: string; reason?: string }>
  completed_frames?: number
  total_frames?: number
  chunk_index?: number
}

export type GenerationJob = GpuTaskResponse

export interface QueueJob {
  job_id: string
  /** True when the job belongs to the requesting user (or auth is off).
      Foreign jobs arrive anonymized: position/progress only, no content. */
  mine: boolean
  status: GpuTaskStatus
  progress: number
  queue_position?: number | null
  queue_length?: number | null
  seed?: number | null
  error?: string | null
  summary: string
  thumb: string
  is_batch: boolean
  batch_count?: number | null
  num_images: number
  /** Omitted for anonymous foreign entries. */
  task_kind?: GpuTaskKind
  priority_class?: 'interactive' | 'background'
  /** Unix seconds; only present on your own jobs. */
  queued_at?: number | null
  started_at?: number | null
  finished_at?: number | null
}

export interface GpuTaskAdmission {
  per_user_active: number
  per_user_limit: number
  global_interactive_active: number
  global_interactive_limit: number
  global_background_active: number
  global_background_limit: number
}

export interface QueueJobsResponse {
  jobs: QueueJob[]
  admission: GpuTaskAdmission
}

export interface BatchPlan {
  allowed: boolean
  fits: boolean
  batch: number
  mode: 'parallel' | 'safe_queue'
  clear_cache_first: boolean
  tiled_decode: boolean
  estimated_scratch_gb: number
  estimated_decode_gb: number
  warnings: string[]
  blocked_reasons: string[]
  free_vram_gb?: number | null
}

export interface EngineCapabilities {
  engine_id: 'native_pytorch' | 'native_gguf' | 'native_int8_convrot' | string
  label: string
  default: boolean
  experimental: boolean
  profiles: string[]
  supports_lora: boolean
  supports_style_references: boolean
  supports_moodboards: boolean
  supports_regional_prompts: boolean
  supports_rebalance: boolean
  supports_krea_enhancer: boolean
  supports_flow_samplers: boolean
  supports_standard_samplers: boolean
  supports_cfg: boolean
  supports_img2img: boolean
  supports_inpaint: boolean
  supports_parallel_batch: boolean
  max_batch: number
  max_resolution: number
  recommended_steps: number
  unsupported_controls: string[]
}

export interface EngineCatalog {
  engines: EngineCapabilities[]
  default_engine: string
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
  mood: string
  moodboard_strength: number
  moodboard_ids: number[]
  moodboard_uuids: string[]
  style_references: any[]
  regional_prompts: any[]
  seed_variance_preset: string
  krea_enhancer_variant: string
  rebalance_preset: string
  updated_at: string
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
  preview_image_urls: string[]
  related_urls: string[]
  favorite: boolean
  source: 'official' | 'custom' | 'andrometa'
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
  owner_username?: string | null
  filesystem_only?: boolean
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
  download_enabled?: boolean
  // Civitai enrichment (from /api/loras after a scan)
  preview_url?: string
  base_model?: string
  description?: string
  civitai_url?: string
  civitai?: {
    civitai_name?: string
    version_name?: string
    description?: string
    trigger_words?: string[]
    base_model?: string
    preview_url?: string
    civitai_url?: string
    model_id?: number
    version_id?: number
    nsfw?: boolean
  }
}

export interface CivitaiLoraItem {
  model_id: number
  version_id: number
  name: string
  type: string
  creator: string
  base_model: string
  version_name: string
  trigger_words: string[]
  description: string
  nsfw: boolean
  preview_url: string
  download_url: string
  file_name: string
  file_size_kb?: number
  downloads?: number
  thumbsUp?: number
  civitai_url: string
  installed?: boolean
  installed_filename?: string
}

export interface CivitaiScanStatus { scanning: boolean; total: number; done: number; updated: number }

export interface HuggingFaceLoraItem {
  repo_id: string
  name: string
  creator: string
  base_model: string
  tags?: string[]
  downloads?: number
  likes?: number
  preview_url: string
  hf_url: string
  pipeline_tag?: string
  installed?: boolean
  installed_filename?: string
  weight_files?: { filename: string; size?: number }[]
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
  model_status: { loaded: boolean; loading?: boolean; checkpoint?: string; quantization?: string; auto_checkpoint?: string; auto_quant?: string; load_error?: string | null; text_encoder_source?: { kind: string; path: string; runtime?: string; status?: string } | null; memory?: Record<string, any>; backend?: 'comfyui' | 'native' }
  attention_acceleration?: { status: string; available: boolean; reason: string; recommendation: string }
  gpu_capabilities?: { name: string; arch: string; compute_capability: string | null; vram_total_gb: number | null; supports_bf16: boolean; supports_fp8_compute: boolean; supports_nvfp4: boolean; fp8_storage_only: boolean; fp8_note: string }
  recommended_runtime?: { quantization: string; blocks_to_swap: number; max_tier: string; notes: string }
  runnability?: { can_run: boolean; tier: string; compute_dtype: string; blocks_to_swap: number; max_tier: string; reason: string }
  support_models?: Array<{ id: string; label: string; repo_id: string; purpose: string; installed: boolean; optional?: boolean; cache_dir: string; legacy_cache_installed?: boolean; path?: string; download_enabled?: boolean; disabled_reason?: string }>
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
  role?: 'admin' | 'user' | 'child' | null
}

export interface ShareUser {
  username: string
  role: 'admin' | 'user' | 'child'
  online?: boolean
  active?: boolean
  last_seen?: number | null
}

export interface MoodboardSuggestion {
  id: number
  uuid?: string
  title: string
  reason?: string
  preview_image_urls?: string[]
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

export interface SharingRepairResult {
  ok: boolean
  message: string
  needs_admin_service_restart?: boolean
  local_target?: { ok: boolean; auth_required: boolean; url: string; message: string }
  tailscale?: SharingStatus['tailscale']
  funnel?: SharingStatus['funnel']
  start_funnel?: { ok: boolean; url: string; message: string }
}

export interface AppSettings {
  hf_token: string
  civitai_token: string
  krea2_turbo_path: string
  krea2_raw_path: string
  krea2_turbo_int8_path: string
  krea2_raw_int8_path: string
  output_dir: string
  prompt_expander_backend: 'local' | 'openrouter' | 'ideogram-json'
  local_llm_backend: 'comfy' | 'transformers' | 'gguf_server'
  comfy_qwen_model: string
  comfy_qwen_quant: string
  comfy_qwen_vision_model: string
  comfy_qwen_vision_quant: string
  krea_comfy_warmup: boolean
  local_qwen_model_id: string
  local_qwen_device: 'auto' | 'cuda' | 'cpu'
  gguf_helper_base_url: string
  gguf_helper_model: string
  gguf_helper_timeout_sec: number
  diffusion_engine: 'native_pytorch' | 'native_gguf' | 'native_int8_convrot'
  gguf_turbo_path: string
  gguf_raw_path: string
  openrouter_model: string
  openrouter_free_only: boolean
  krea_share_auto_funnel: boolean
  krea2_vae_path: string
  krea2_vae_mode: 'qwen' | 'comfy_qwen' | 'qwen_wan_blend' | 'wan_experimental'
  krea2_vae_blend_radius: number
  krea2_vae_blend_strength: number
  krea_attention_backend: 'sdpa' | 'sage'
  seedvr2_model: '3b' | '7b'
  has_hf_token: boolean
  has_civitai_token: boolean
  has_ideogram_api_key: boolean
  has_openrouter_api_key: boolean
  krea_deforum?: unknown
  animation: AnimationLimits
}

export interface AcceleratorStatus {
  sdpa: { available: boolean; default: boolean }
  studio_python?: string
  comfyui_venv?: { available: boolean; python: string; triton: boolean; sageattention: boolean; comfy_kitchen: boolean }
  triton_windows: { installed: boolean; compatible: boolean; recommendation: string }
  sageattention: { installed: boolean; compatible: boolean; recommendation: string }
  xformers: { installed: boolean; compatible: boolean; recommendation: string }
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
  download_enabled: boolean
  disabled_reason: string
}

export interface XperimentSetupResult {
  ok: boolean
  assets: Array<{ id: string; path: string; skipped: boolean; item: QualityAsset }>
  vae_path: string
  lora: { name: string; filename: string; strength: number; block_filter?: 'all' | 'early' | 'middle' | 'late' | 'style_safe' | 'custom' }
  loras?: Array<{ name: string; filename: string; strength: number; block_filter?: 'all' | 'early' | 'middle' | 'late' | 'style_safe' | 'custom' }>
  diffusion_engine?: 'native_pytorch' | 'native_gguf' | 'native_int8_convrot'
  quantization?: 'bf16' | 'fp8' | 'gguf' | 'fp16' | 'int8'
  sampler: { sampler: string; scheduler: string; steps: number; cfg: number }
  res4lyf?: { sampler_name: string; eta: number; bongmath: boolean }
  use_prompt_expander?: boolean
  prompt_expander_backend?: 'local' | 'openrouter' | 'ideogram-json'
  local_llm_backend?: 'comfy' | 'transformers' | 'gguf_server'
  comfy_qwen_model?: string
  comfy_qwen_quant?: string
  comfy_qwen_vision_model?: string
  comfy_qwen_vision_quant?: string
  local_qwen_model_id?: string
  benchmark_note?: string
  manual_only: QualityAsset[]
  warnings: string[]
}

export interface GgufLowVramSetupResult {
  ok: boolean
  assets: Array<{ id: string; path: string; skipped: boolean; item: QualityAsset }>
  diffusion_engine: 'native_gguf'
  turbo_path: string
  checkpoint_path: string
  quantization: 'gguf'
  vae_path: string
  sampler: { sampler: string; scheduler: string; steps: number; cfg: number; mu: number }
  warnings: string[]
}

export interface ModerationEvent {
  id: number
  created_at: string
  username: string
  role: 'admin' | 'user' | 'child'
  event_type: string
  action: string
  prompt: string
  negative_prompt: string
  mode: string
  scores: Record<string, any>
  reason: string
  job_id: string
  gallery_id?: number | null
  quarantined_filename?: string | null
}

export interface ModerationStatus {
  image_classifier_available: boolean
  child_image_moderation: string
  message: string
}

export interface HelperQueueResponse {
  job_id: string
  status: 'queued'
  task_kind: GpuTaskKind
  queue_position: number | null
  queue_length: number | null
}

const TERMINAL_GPU_TASK_STATUSES = new Set<GpuTaskStatus>(['done', 'error', 'blocked', 'cancelled'])

export function isGpuTaskTerminal(status?: string): status is Extract<GpuTaskStatus, 'done' | 'error' | 'blocked' | 'cancelled'> {
  return TERMINAL_GPU_TASK_STATUSES.has(status as GpuTaskStatus)
}

export function gpuTaskTerminalError(job: Pick<GpuTaskResponse, 'status' | 'error'>): Error | null {
  if (job.status === 'done') return null
  return isGpuTaskTerminal(job.status)
    ? new Error(job.error || `GPU task ${job.status}.`)
    : null
}

export function responseStatus(error: unknown): number | undefined {
  return (error as { response?: { status?: number } })?.response?.status
}

export async function waitForGpuTask<T>(jobId: string, timeoutMs = 65 * 60 * 1000): Promise<T> {
  const deadline = Date.now() + timeoutMs
  let transientFailures = 0
  while (Date.now() < deadline) {
    let job: GpuTaskResponse<T>
    try {
      job = await api.get<GpuTaskResponse<T>>(`/api/generate/${jobId}`).then(r => r.data)
      transientFailures = 0
    } catch (error: unknown) {
      if (responseStatus(error) === 404) throw new Error('GPU task was not found or is no longer available.')
      transientFailures += 1
      if (transientFailures > 8) throw error
      const backoffMs = Math.min(500 * (2 ** (transientFailures - 1)), 5000)
      await new Promise(resolve => setTimeout(resolve, backoffMs))
      continue
    }
    if (job.status === 'done') {
      const result = job.result as T
      await api.post(`/api/generate/${jobId}/ack`).catch(() => undefined)
      return result
    }
    const terminalError = gpuTaskTerminalError(job)
    if (terminalError) throw terminalError
    await new Promise(resolve => setTimeout(resolve, 500))
  }
  await api.post(`/api/generate/${jobId}/cancel`).catch(() => undefined)
  throw new Error('GPU task timed out while waiting for a result.')
}

async function resolveGpuSubmission<T extends object>(
  submitted: T | HelperQueueResponse,
): Promise<T> {
  return 'job_id' in submitted
    ? waitForGpuTask<T>(submitted.job_id)
    : submitted
}

export const apiFetch = {
  generate: (req: GenerationRequest) =>
    api.post<{ job_id: string; status: GpuTaskStatus; queue_position?: number | null; queue_length?: number | null; moderation_event_id?: number; batch_id?: string; child_job_ids?: string[] }>('/api/generate', req).then(r => r.data),

  animate: (req: AnimateRequest) =>
    api.post<{ job_id: string; status: 'queued'; queue_position?: number | null; queue_length?: number | null }>(
      animateContract.endpoints.submit,
      req,
    ).then(r => r.data),

  uploadAnimationSource: (
    file: File,
    onProgress?: (percent: number) => void,
  ) => api.post<AnimationUploadResponse>(animateContract.endpoints.upload, file, {
    headers: {
      'Content-Type': file.type,
    },
    // Keep the File as the raw request body. The browser owns Content-Length
    // (a forbidden script header) and supplies it from the File size.
    transformRequest: data => data,
    withCredentials: true,
    onUploadProgress: event => {
      const total = event.total ?? file.size
      onProgress?.(total > 0 ? Math.min(100, Math.round(event.loaded / total * 100)) : 0)
    },
  }).then(r => r.data),

  downloadOwnedMedia: (url: string) =>
    api.get<Blob>(url, { responseType: 'blob', withCredentials: true }).then(r => r.data),

  jobStatus: <TResult = unknown>(jobId: string) =>
    api.get<GpuTaskResponse<TResult>>(`/api/generate/${jobId}`).then(r => r.data),

  jobs: (limit = 24) =>
    api.get<QueueJobsResponse>(`/api/jobs`, { params: { limit } }).then(r => r.data),

  cancelJob: (jobId: string) =>
    api.post<{ ok: boolean; job_id: string; status: string; cancelled: number }>(`/api/generate/${jobId}/cancel`).then(r => r.data),
  ackJob: (jobId: string) =>
    api.post<{ ok: boolean; job_id: string; status: string }>(`/api/generate/${jobId}/ack`).then(r => r.data),

  loadModel: (path: string, quant: string, blocksToSwap = 0, fp8FastMatmul = false, torchCompile = false) =>
    api.post('/api/load-model', { checkpoint_path: path, quantization: quant, blocks_to_swap: blocksToSwap, fp8_fast_matmul: fp8FastMatmul, torch_compile: torchCompile }).then(r => r.data),

  preflightLoadModel: (path: string, quant: string, blocksToSwap = 0, fp8FastMatmul = false, torchCompile = false) =>
    api.post<{ ok: boolean; detail: string; system?: SystemReport }>('/api/load-model/preflight', { checkpoint_path: path, quantization: quant, blocks_to_swap: blocksToSwap, fp8_fast_matmul: fp8FastMatmul, torch_compile: torchCompile }).then(r => r.data),

  samplerCatalog: (profile = 'krea_turbo') =>
    api.get<{
      profile: string
      samplers: { id: string; label: string; scheduler: string; default_steps: number; default_cfg: number; supported_schedulers: string[]; recommended_steps: number; disabled: boolean; note: string }[]
      schedulers: { id: string; label: string; recommended: boolean; note: string }[]
      recommended_combos: { sampler: string; scheduler: string; steps: number; cfg: number; label: string; note: string }[]
    }>('/api/sampler-catalog', { params: { profile } }).then(r => r.data),

  engineCatalog: () =>
    api.get<EngineCatalog>('/api/engine-catalog').then(r => r.data),

  batchPlan: (params: { width: number; height: number; quantization: string; batch: number; cfg: number; mode: string; checkpoint: string }) =>
    api.get<BatchPlan>('/api/batch/plan', { params }).then(r => r.data),

  unloadModel: () => api.post('/api/unload-model').then(r => r.data),

  releaseTransientMemory: () => api.post('/api/memory/release-transient').then(r => r.data),
  safeCleanMemory: () => api.post<{ released: boolean; safe_clean: boolean; helper_unloaded?: boolean; cleared_conditioning_entries?: number; before?: Record<string, any>; after?: Record<string, any>; memory?: Record<string, any> }>('/api/memory/safe-clean').then(r => r.data),
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

  lorasCivitaiScan: () => api.post<CivitaiScanStatus>('/api/loras/civitai-scan').then(r => r.data),
  lorasCivitaiScanStatus: () => api.get<CivitaiScanStatus>('/api/loras/civitai-scan/status').then(r => r.data),
  civitaiLoras: (params: { query?: string; page?: number; cursor?: string; sort?: string; nsfw?: boolean } = {}) =>
    api.get<{
      items: CivitaiLoraItem[]
      metadata: Record<string, unknown>
      next_cursor?: string | null
      has_more?: boolean
    }>('/api/civitai/loras', { params }).then(r => r.data),
  civitaiInstall: (versionId: number, filename?: string) =>
    api.post<{ ok: boolean; filename: string; path: string; trigger_words?: string[] }>(
      '/api/civitai/install', { version_id: versionId, filename }
    ).then(r => r.data),

  huggingfaceLoras: (params: { query?: string; sort?: string; cursor?: string; limit?: number } = {}) =>
    api.get<{
      items: HuggingFaceLoraItem[]
      next_cursor: string | null
      has_more: boolean
      metadata: Record<string, unknown>
    }>('/api/huggingface/loras', { params }).then(r => r.data),

  huggingfaceInstall: (repoId: string, filename?: string) =>
    api.post<{
      ok: boolean
      filename: string
      path: string
      repo_id: string
      compatible?: boolean | null
      match_info?: string
      already_installed?: boolean
    }>('/api/huggingface/install', { repo_id: repoId, filename }).then(r => r.data),

  moods: () => api.get<Mood[]>('/api/moods').then(r => r.data),

  moodboards: (opts?: { q?: string; page?: number; pageSize?: number; favorites?: boolean; source?: 'official' | 'custom' | 'andrometa'; shuffleSeed?: string }) => {
    const params = new URLSearchParams()
    if (opts?.q) params.set('q', opts.q)
    params.set('page', String(opts?.page ?? 1))
    params.set('page_size', String(opts?.pageSize ?? 50))
    params.set('favorites', String(opts?.favorites ?? false))
    if (opts?.source) params.set('source', opts.source)
    if (opts?.shuffleSeed) params.set('shuffle_seed', opts.shuffleSeed)
    return api.get<{ items: MoodboardItem[]; total: number }>(`/api/moodboards?${params.toString()}`).then(r => r.data)
  },

  moodboard: (id: number) =>
    api.get<MoodboardItem>(`/api/moodboards/${id}`).then(r => r.data),

  setMoodboardFavorite: (id: number, favorite: boolean) =>
    api.put(`/api/moodboards/${id}/favorite`, { favorite }).then(r => r.data),

  generateMoodboardGuidance: async (id: number) => {
    const submitted = await api.post<MoodboardItem | HelperQueueResponse>(`/api/moodboards/${id}/qwen-guidance`, {}).then(r => r.data)
    return resolveGpuSubmission<MoodboardItem>(submitted)
  },

  generateMissingMoodboardGuidance: async (limit = 10) => {
    const submitted = await api.post<{ processed: number; items: MoodboardItem[] } | HelperQueueResponse>('/api/moodboards/qwen-guidance-missing', { limit }).then(r => r.data)
    return resolveGpuSubmission<{ processed: number; items: MoodboardItem[] }>(submitted)
  },

  createCustomMoodboard: async (req: { title: string; taste_profile?: string; keywords?: string[]; image_b64s: string[] }) => {
    const submitted = await api.post<MoodboardItem | HelperQueueResponse>('/api/moodboards/custom', req).then(r => r.data)
    return 'job_id' in submitted
      ? waitForGpuTask<MoodboardItem>(submitted.job_id)
      : submitted
  },

  createMoodboardMashup: async (req: { moodboard_ids: number[]; weights?: number[] }) => {
    const submitted = await api.post<MoodboardItem | HelperQueueResponse>('/api/moodboards/mashup', req).then(r => r.data)
    return resolveGpuSubmission<MoodboardItem>(submitted)
  },

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
  }) => {
    const body = {
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
    }
    return api.post<{ image_b64: string; metadata?: Record<string, any> } | HelperQueueResponse>('/api/upscale', body)
      .then(r => resolveGpuSubmission<{ image_b64: string; metadata?: Record<string, any> }>(r.data))
  },

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

  describeImage: async (image_b64: string, mode: 'recreate' | 'style' | 'character' = 'recreate', guidance = '') => {
    const queued = await api.post<HelperQueueResponse>('/api/describe-image', { image_b64, mode, guidance }).then(r => r.data)
    return waitForGpuTask<{ prompt: string; backend: 'local' | 'openrouter' }>(queued.job_id)
  },

  depthPreview: (image_b64: string, estimator: 'da3' | 'depth_anything_v2' | 'zoe' | 'midas' = 'da3', resolution = 504, invert = false) =>
    api.post<{ image_b64: string } | HelperQueueResponse>('/api/depth-preview', { image_b64, estimator, resolution, invert })
      .then(r => resolveGpuSubmission<{ image_b64: string }>(r.data)),

  system: () => api.get<SystemReport>('/api/system').then(r => r.data),

  downloadSupportModels: () =>
    api.post<{ ok: boolean; status: SystemReport['support_models'] }>('/api/support-models/download', {}, { timeout: 3600000 })
      .then(r => r.data),

  qualityAssets: () =>
    api.get<{ has_hf_token: boolean; items: QualityAsset[] }>('/api/quality-assets').then(r => r.data),

  downloadQualityAsset: (assetId: string) =>
    api.post<{ ok: boolean; path: string; item: QualityAsset }>(`/api/quality-assets/${assetId}/download`, {}, { timeout: 7200000 })
      .then(r => r.data),

  setupXperiment: () =>
    api.post<XperimentSetupResult>('/api/xperiment/setup', {}, { timeout: 7200000 }).then(r => r.data),

  setupGgufLowVram: () =>
    api.post<GgufLowVramSetupResult>('/api/gguf/setup-low-vram', {}, { timeout: 7200000 }).then(r => r.data),

  setupNativeInt8: () =>
    api.post<{ ok: boolean; assets: Array<{ id: string; path: string; skipped: boolean; item: QualityAsset }>; diffusion_engine: 'native_int8_convrot'; turbo_path: string; quantization: 'int8'; sampler: { sampler: string; scheduler: string; steps: number; cfg: number; mu: number }; warnings: string[] }>(
      '/api/int8/setup-native',
      {},
      { timeout: 7200000 },
    ).then(r => r.data),

  settings: () => api.get<AppSettings>('/api/settings').then(r => r.data),
  updateSettings: (data: Partial<AppSettings> & { hf_token?: string; ideogram_api_key?: string; openrouter_api_key?: string }) =>
    api.put('/api/settings', data).then(r => r.data),

  testGgufHelper: () =>
    api.post<{ ok: boolean; backend: string; expanded: string }>('/api/gguf/helper-test', {}, { timeout: 180000 }).then(r => r.data),

  ggufStatus: () => api.get<{ diffusion_engine: string; paths: Record<string, { path: string; configured: boolean }> }>('/api/gguf/status').then(r => r.data),
  int8Status: () => api.get<{ ok: boolean; backend: string; loader: string; diffusion_engine: string; note?: string; assets: Record<string, { installed: boolean; path?: string; configured_path?: string }> }>('/api/int8/status').then(r => r.data),
  acceleratorStatus: () => api.get<AcceleratorStatus>('/api/accelerators/status').then(r => r.data),
  installTritonWindows: () => api.post<{ ok: boolean; status: AcceleratorStatus; message: string }>('/api/accelerators/install-triton-windows', {}, { timeout: 600000 }).then(r => r.data),
  installSageAttention: () => api.post<{ ok: boolean; status: AcceleratorStatus; message: string }>('/api/accelerators/install-sageattention', {}, { timeout: 600000 }).then(r => r.data),

  submitExpandPrompt: (prompt: string, backend?: string) =>
    api.post<HelperQueueResponse>('/api/expand-prompt', { prompt, suggest_moodboards: true, ...(backend ? { backend } : {}) }).then(r => r.data),
  expandPrompt: async (prompt: string, backend?: string) => {
    const queued = await apiFetch.submitExpandPrompt(prompt, backend)
    return waitForGpuTask<{ expanded: string; changed: boolean; error?: string | null; backend: 'local' | 'openrouter' | 'ideogram-json' | 'gguf-server'; suggested_moodboards?: MoodboardSuggestion[]; sign_copy_pass?: boolean | null }>(queued.job_id)
  },
  planPrompt: async (prompt: string, max_tokens = 700) => {
    const queued = await api.post<HelperQueueResponse>('/api/plan-prompt', { prompt, max_tokens }).then(r => r.data)
    return waitForGpuTask<PromptPlan>(queued.job_id)
  },
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
  addUser: (username: string, password: string, role: 'admin' | 'user' | 'child') =>
    api.post<{ users: ShareUser[] }>('/api/admin/users', { username, password, role }).then(r => r.data.users),
  setUserRole: (username: string, role: 'admin' | 'user' | 'child') =>
    api.put<{ users: ShareUser[] }>(`/api/admin/users/${encodeURIComponent(username)}/role`, { role }).then(r => r.data.users),
  resetUserPassword: (username: string, password: string) =>
    api.put(`/api/admin/users/${encodeURIComponent(username)}/password`, { password }).then(r => r.data),
  removeUser: (username: string) =>
    api.delete<{ users: ShareUser[] }>(`/api/admin/users/${encodeURIComponent(username)}`).then(r => r.data.users),

  moderationEvents: (username = '', limit = 100) =>
    api.get<{ items: ModerationEvent[]; total: number }>('/api/moderation/events', { params: { username, limit } }).then(r => r.data),
  moderationStatus: () => api.get<ModerationStatus>('/api/moderation/status').then(r => r.data),
  installImageClassifier: () => api.post<{ ok: boolean; installed: boolean; message: string }>('/api/moderation/install-image-classifier').then(r => r.data),

  sharingStatus: () => api.get<SharingStatus>('/api/sharing/status').then(r => r.data),
  tailscaleUp: () => api.post('/api/sharing/tailscale-up').then(r => r.data),
  startSharing: () => api.post<{ ok: boolean; url: string; message: string }>('/api/sharing/funnel/start').then(r => r.data),
  repairSharing: () => api.post<SharingRepairResult>('/api/sharing/funnel/repair').then(r => r.data),
  stopSharing: () => api.post<{ ok: boolean; message: string }>('/api/sharing/funnel/stop').then(r => r.data),

  downloadLora: (name: string) =>
    api.post<{ ok: boolean; path: string }>(`/api/loras/${name}/download`).then(r => r.data),

  importLoraUrl: (url: string, filename?: string, civitaiToken?: string) =>
    api.post<{ ok: boolean; path: string; filename: string; skipped?: boolean; compatible?: boolean; match_info?: string }>(
      '/api/loras/import',
      { url, filename: filename ?? '', civitai_token: civitaiToken ?? '' }
    ).then(r => r.data),
}

export function connectWS(jobId: string, onMessage: (data: unknown) => void, onClose?: (ev?: CloseEvent) => void): WebSocket {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = window.location.host
  const ws = new WebSocket(`${proto}://${host}${publicBasePath()}/ws/${jobId}`)
  ws.onmessage = e => onMessage(JSON.parse(e.data))
  if (onClose) {
    // Forward the CloseEvent so callers can tell a policy rejection (1008:
    // not your job / not signed in) from a transient network drop.
    ws.onclose = ev => onClose(ev)
    ws.onerror = () => onClose(undefined)
  }
  return ws
}

export default api
