from __future__ import annotations
from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    label: str
    bbox: List[float]  # [x1, y1, x2, y2] normalized 0-1


class StyleReferenceInput(BaseModel):
    image_b64: str
    strength: float = Field(default=1.0, ge=-2.0, le=2.0)
    role: Literal["style", "layout", "subject", "mood", "texture", "target"] = "style"
    token_size: Literal["low", "normal", "high", "max"] = "normal"
    mask_b64: Optional[str] = None
    mask_padding: int = Field(default=0, ge=0, le=512)
    vision_megapixels: Optional[float] = Field(default=None, gt=0.0, le=4.0)
    system_prompt: Optional[str] = Field(default=None, max_length=512)
    vision_position: Literal["before_prompt", "after_prompt"] = "before_prompt"


class RegionalPromptInput(BaseModel):
    prompt: str
    negative_prompt: str = ""
    mask_b64: str = ""
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
    feather: int = Field(default=24, ge=0, le=128)
    normalize: bool = True
    visible: bool = True
    lora_filter: str = ""


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    mode: str = "txt2img"           # txt2img | redraw | img2img | inpaint | outpaint
    model_profile: str = ""         # krea_turbo | krea_raw | future gated profiles
    checkpoint: str = "turbo"       # turbo | raw | custom
    checkpoint_path: str = ""       # custom path override
    quantization: str = "fp8"       # bf16 | fp8
    steps: int = 8
    cfg: float = 0.0
    mu: Optional[float] = None  # None → inference resolves (turbo=1.15, RAW=adaptive)
    y1: float = 0.5
    y2: float = 1.15
    width: int = 1024
    height: int = 1024
    num_images: int = 1
    seed: int = -1
    denoise: float = 1.0
    sampler: str = "euler_flow"       # euler | euler_flow | exp_heun_2_x0_sde | guarded Comfy names
    scheduler: str = "simple"
    # CFG-Zero* (arXiv:2503.18886): flow-matching guidance upgrade. Optimized
    # scale corrects velocity error; zero-init skips the first K ODE steps. Only
    # active when guidance (cfg) > 0 and a non-CFG++ sampler is used.
    cfg_zero_star: bool = False
    cfg_zero_init_steps: int = Field(default=1, ge=0, le=4)
    inpaint_method: str = "native"    # native | lanpaint_experimental | flux_fill
    # Differential diffusion (soft masks): grayscale mask values join the denoise
    # at different timesteps, so feathered edits blend seamlessly into the keep
    # region. strength<1 keeps some of the raw soft mask each step.
    differential_inpaint: bool = False
    differential_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    lanpaint_inner_steps: int = 3
    lanpaint_strength: float = 1.0
    lanpaint_lambda: float = 16.0
    lanpaint_step_size: float = 0.2
    lanpaint_beta: float = 1.0
    lanpaint_friction: float = 15.0
    lanpaint_early_stop: int = 1
    lanpaint_prompt_mode: Literal["Image First", "Prompt First"] = "Image First"
    edit_provider: str = "auto"       # auto | krea_native | flux_fill
    quality_preset: str = "balanced"  # fast | balanced | best | raw_benchmark
    creativity: Literal["raw", "low", "medium", "high"] = "medium"
    style_references: List[StyleReferenceInput] = Field(default_factory=list, max_length=10)
    style_fusion_mode: Literal["style_only", "preserve_structure", "semantic_fusion"] = "semantic_fusion"
    regional_prompts: List[RegionalPromptInput] = Field(default_factory=list, max_length=8)
    regional_base_prompt_strength: float = Field(default=0.3, ge=0.0, le=1.0)
    regional_normalize_masks: bool = True
    loras: List[dict] = []
    use_rebalance: bool = True
    rebalance_multiplier: float = 1.0
    rebalance_weights: str = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"
    rebalance_mode: Literal["legacy_multiply", "rms_renorm"] = "rms_renorm"
    rebalance_preset: Literal["legacy", "subtle", "balanced", "detail", "emotion", "uniform", "custom"] = "balanced"
    rebalance_renormalize: bool = True
    edit_rebalance_enabled: bool = True
    edit_rebalance_profile: Literal["default", "edit", "conservative"] = "conservative"
    conditioning_mode: Literal["auto", "qwen_image_edit_plus", "qwen_reference"] = "auto"
    krea_enhancer_enabled: bool = False
    krea_enhancer_variant: Literal["off", "current", "capped_delta", "current_plus_capped"] = "off"
    krea_enhancer_strength: float = 1.0
    krea_enhancer_delta_cap: float = Field(default=0.75, ge=0.05, le=2.0)
    bboxes: List[BoundingBox] = []
    init_image_b64: Optional[str] = None
    mask_b64: Optional[str] = None
    ref_image1_b64: Optional[str] = None
    ref_image2_b64: Optional[str] = None
    ref_image3_b64: Optional[str] = None
    use_prompt_planner: bool = False
    prompt_planner_max_tokens: int = Field(default=700, ge=128, le=1600)
    prompt_planner_show_output: bool = False
    prompt_planner_lock_original: bool = False
    prompt_planner_use_regions: bool = False
    prompt_planner_output: dict = {}
    use_prompt_expander: bool = False
    # <think>-block expression steering: appends a reasoning span to the assistant
    # turn to restore expression/intensity in-distribution (positive prompt only).
    think_steering_enabled: bool = False
    think_text: str = ""
    # Detail refiner: optional second low-denoise self-pass (txt2img/img2img only)
    refine: bool = False
    refine_denoise: float = 0.3
    refine_steps: int = 6
    # Moodboard: preset mood id + custom reference-image board + influence strength
    mood: str = ""
    moodboard_ids: List[int] = []
    moodboard_uuids: List[str] = []
    moodboard_strength: float = 0.35
    moodboard_images: List[str] = []
    seed_variance_preset: Literal["off", "subtle", "balanced", "creative", "bold", "custom"] = "off"
    seed_variance_strength: float = 0.0
    seed_variance_protection: Literal["none", "first_quarter", "first_half"] = "first_half"
    seed_variance_direction: Literal["none", "forward", "reverse", "center", "edges"] = "none"
    seed_variance_fade_curve: Literal["linear", "ease_in", "ease_out", "smoothstep"] = "linear"
    seed_variance_injection_start: float = Field(default=0.0, ge=0.0, le=1.0)
    seed_variance_injection_end: float = Field(default=1.0, ge=0.0, le=1.0)


class RealtimePreviewRequest(BaseModel):
    session_id: str
    prompt: str
    negative_prompt: str = ""
    canvas_image_b64: str
    width: int = 512
    height: int = 512
    preview_steps: int = 5
    moodboard_strength: float = 0.75
    seed: int = -1


class GalleryItem(BaseModel):
    id: int
    filename: str
    prompt: str
    checkpoint: str
    width: int
    height: int
    seed: int
    created_at: str
    favorite: bool = False
    thumbnail_b64: Optional[str] = None
    metadata: dict = {}
    owner_username: Optional[str] = None


class GalleryListResponse(BaseModel):
    items: List[GalleryItem]
    total: int


class UpscaleRequest(BaseModel):
    image_b64: str
    method: str = "realesrgan"      # realesrgan | tiled_vae | model_refine | ultimate
    scale: int = 4
    upscale_by: float = 2.0
    denoise: float = 0.24
    gallery_id: Optional[int] = None
    # Ultimate SD Upscale params
    prompt: str = ""
    tile_size: int = 1024
    tile_width: int = 1024
    tile_height: int = 1024
    tile_padding: int = 96
    mask_blur: int = 12
    seam_mode: Literal["none", "band_pass", "half_tile", "half_tile_intersections"] = "band_pass"
    tile_mode: Literal["linear", "chess"] = "chess"
    sampler: str = "euler"
    scheduler: str = "simple"
    steps: int = 8
    cfg: float = 1.0
    tiled_decode: bool = False
    seam_fix: bool = True


class AutoMaskRequest(BaseModel):
    image_b64: str
    prompt: str                     # text description of region(s) to mask, comma-separated
    threshold: float = 0.35


class PreprocessorPreviewRequest(BaseModel):
    image_b64: str
    kind: Literal["canny", "soft_edge", "lineart", "depth"] = "canny"
    resolution: int = 768
    low_threshold: int = 80
    high_threshold: int = 160


class DescribeImageRequest(BaseModel):
    image_b64: str


class DescribeImageResponse(BaseModel):
    prompt: str
    backend: str = "openrouter"


class LoraInfo(BaseModel):
    filename: str
    name: str
    trigger_words: List[str] = []
    strength: float = 1.0
    is_official: bool = False
    installed: bool = True


class LoraImportRequest(BaseModel):
    url: str
    filename: str = ""
    civitai_token: str = ""


class ModelStatusResponse(BaseModel):
    loaded: bool
    checkpoint: Optional[str] = None
    quantization: Optional[str] = None
    vram_used_gb: Optional[float] = None


class SystemInfoResponse(BaseModel):
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    ram_available_gb: Optional[float] = None
    disk_free_gb: Optional[float] = None
    gpu_processes: List[str] = []
    gpu_process_details: List[dict] = []
    model_status: ModelStatusResponse
    variants: List[dict] = []


class LoadModelRequest(BaseModel):
    checkpoint_path: str
    quantization: str = "bf16"
    blocks_to_swap: int = Field(default=0, ge=0, le=28)
    fp8_fast_matmul: bool = False  # opt-in fp8 _scaled_mm (Ada/Blackwell only)
    torch_compile: bool = False    # opt-in torch.compile of the DiT (experimental)


class MemoryStopProcessRequest(BaseModel):
    pid: int


class ExpandPromptRequest(BaseModel):
    prompt: str
    backend: Optional[str] = None


class ExpandPromptResponse(BaseModel):
    expanded: str
    changed: bool = False
    error: Optional[str] = None
    backend: str = "local"


class PlanPromptRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=700, ge=128, le=1600)


class PlanPromptResponse(BaseModel):
    original_prompt: str
    planned_prompt: str
    negative_prompt: str = ""
    subject: str = ""
    composition: str = ""
    style: str = ""
    lighting: str = ""
    materials: str = ""
    text_rendering: str = ""
    regions: list[dict] = []
    backend: str = "local"
    changed: bool = False
    error: Optional[str] = None


class PromptRecipe(BaseModel):
    id: str = ""
    name: str
    description: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    planner_instruction: str = ""
    loras: list[dict] = []
    mood: str = ""
    moodboard_strength: float = 0.35
    moodboard_ids: list[int] = []
    moodboard_uuids: list[str] = []
    style_references: list[dict] = []
    regional_prompts: list[dict] = []
    seed_variance_preset: str = "off"
    krea_enhancer_variant: str = "off"
    rebalance_preset: str = "balanced"
    updated_at: str = ""


class PromptRecipeListResponse(BaseModel):
    items: list[PromptRecipe]


class ShareLoginRequest(BaseModel):
    username: str
    password: str


class ShareUserCreateRequest(BaseModel):
    username: str
    password: str
    role: Literal["admin", "user", "child"] = "user"


class ShareUserRoleRequest(BaseModel):
    role: Literal["admin", "user", "child"]


class ShareUserPasswordRequest(BaseModel):
    password: str


class FavoriteRequest(BaseModel):
    favorite: bool


class MoodboardItem(BaseModel):
    id: int
    url: str
    slug: str
    uuid: str = ""
    title: str
    taste_profile: str = ""
    keywords: List[str] = []
    primary_image_url: str = ""
    image_urls: List[str] = []
    related_urls: List[str] = []
    favorite: bool = False
    source: str = "official"
    first_seen_at: str
    last_seen_at: str
    updated_at: str
    sync_error: str = ""
    qwen_guidance: dict = {}
    qwen_guidance_at: str = ""
    qwen_guidance_version: int = 0


class MoodboardListResponse(BaseModel):
    items: List[MoodboardItem]
    total: int


class MoodboardImportRequest(BaseModel):
    urls: List[str] = []
    max_pages: int = 200


class MoodboardImportResponse(BaseModel):
    imported: int
    ids: List[int]
    new_count: int = 0
    new_ids: List[int] = []


class CustomMoodboardRequest(BaseModel):
    title: str
    taste_profile: str = ""
    keywords: List[str] = []
    image_b64s: List[str] = []


class MoodboardDiscoveryResponse(BaseModel):
    id: str = ""
    discovered_at: str = ""
    new_count: int = 0
    new_ids: List[int] = []
    items: List[MoodboardItem] = []


class MoodboardExportResponse(BaseModel):
    exported: int
    path: str


class MoodboardImageRequest(BaseModel):
    url: str


class MoodboardImageResponse(BaseModel):
    image_b64: str


class MoodboardGuidanceMissingRequest(BaseModel):
    limit: int = 25


class MoodboardMashupRequest(BaseModel):
    moodboard_ids: List[int]
    weights: List[float] = []


class SettingsUpdate(BaseModel):
    hf_token: Optional[str] = None
    civitai_token: Optional[str] = None
    krea2_turbo_path: Optional[str] = None
    krea2_raw_path: Optional[str] = None
    output_dir: Optional[str] = None
    prompt_expander_backend: Optional[str] = None
    ideogram_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = None
    openrouter_free_only: Optional[bool] = None
    krea_share_auto_funnel: Optional[bool] = None
    krea2_vae_path: Optional[str] = None
