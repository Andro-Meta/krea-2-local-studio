from __future__ import annotations
from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    label: str
    bbox: List[float]  # [x1, y1, x2, y2] normalized 0-1


class StyleReferenceInput(BaseModel):
    image_b64: str
    strength: float = Field(default=1.0, ge=-2.0, le=2.0)
    role: str = "style"
    token_size: Literal["low", "normal", "high", "max"] = "normal"


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    mode: str = "txt2img"           # txt2img | redraw | img2img | inpaint | outpaint
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
    sampler: str = "euler_flow"       # euler_flow
    inpaint_method: str = "native"    # native | lanpaint_experimental | flux_fill
    lanpaint_inner_steps: int = 3
    lanpaint_strength: float = 1.0
    edit_provider: str = "auto"       # auto | krea_native | flux_fill
    quality_preset: str = "balanced"  # fast | balanced | best | raw_benchmark
    creativity: Literal["raw", "low", "medium", "high"] = "medium"
    style_references: List[StyleReferenceInput] = Field(default_factory=list, max_length=10)
    loras: List[dict] = []
    use_rebalance: bool = True
    rebalance_multiplier: float = 4.0
    rebalance_weights: str = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"
    edit_rebalance_enabled: bool = True
    edit_rebalance_profile: Literal["default", "edit", "conservative"] = "conservative"
    krea_enhancer_enabled: bool = False
    krea_enhancer_strength: float = 1.0
    bboxes: List[BoundingBox] = []
    init_image_b64: Optional[str] = None
    mask_b64: Optional[str] = None
    ref_image1_b64: Optional[str] = None
    ref_image2_b64: Optional[str] = None
    ref_image3_b64: Optional[str] = None
    use_prompt_expander: bool = False
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


class GalleryListResponse(BaseModel):
    items: List[GalleryItem]
    total: int


class UpscaleRequest(BaseModel):
    image_b64: str
    method: str = "realesrgan"      # realesrgan | tiled_vae | model_refine | ultimate
    scale: int = 4
    denoise: float = 0.24
    gallery_id: Optional[int] = None
    # Ultimate SD Upscale params
    prompt: str = ""
    tile_size: int = 1024
    seam_fix: bool = True


class AutoMaskRequest(BaseModel):
    image_b64: str
    prompt: str                     # text description of region(s) to mask, comma-separated
    threshold: float = 0.35


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


class ShareLoginRequest(BaseModel):
    username: str
    password: str


class ShareUserCreateRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class ShareUserRoleRequest(BaseModel):
    role: str


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
    first_seen_at: str
    last_seen_at: str
    updated_at: str
    sync_error: str = ""


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
