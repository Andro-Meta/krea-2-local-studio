from __future__ import annotations
from typing import Literal, Optional, List
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


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


class CharacterEditRegion(BaseModel):
    """A rectangular placement box for Character Edit. Coordinates are normalized
    (0..1) relative to the output canvas. Each region can carry its own reference
    image (e.g. person A in the left box, person B in the right)."""
    x: float = Field(default=0.0, ge=0.0, le=1.0)
    y: float = Field(default=0.0, ge=0.0, le=1.0)
    w: float = Field(default=1.0, ge=0.0, le=1.0)
    h: float = Field(default=1.0, ge=0.0, le=1.0)
    prompt: str = ""
    reference_b64: str = ""
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
    feather: int = Field(default=24, ge=0, le=256)


class AnimateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_schedule: str = Field(
        default="0: a scenic landscape, cinematic lighting",
        max_length=32 * 1024,
    )
    negative_prompt: str = ""
    duration_seconds: float = Field(default=4.0, ge=0.5, le=60.0)
    fps: int = Field(default=12, ge=1, le=60)
    render_frames: Optional[int] = Field(default=None, ge=1, le=720)
    width: int = Field(default=768, ge=256, le=1536, multiple_of=16)
    height: int = Field(default=768, ge=256, le=1536, multiple_of=16)
    steps: int = Field(default=8, ge=3, le=52)
    sampler_name: str = "er_sde"
    scheduler: str = "simple"
    seed: int = Field(default=-1, ge=-1, le=(1 << 64) - 1)
    seed_behavior: Literal["fixed", "iter", "random", "ladder"] = "iter"
    animation_mode: Literal["2D", "3D", "Video Input", "None"] = "2D"
    border_mode: Literal["replicate", "reflect", "wrap", "black"] = "replicate"
    cfg_schedule: str = Field(default="0:(1.0)", max_length=32 * 1024)
    strength_schedule: str = Field(default="0:(0.65)", max_length=32 * 1024)
    zoom_schedule: str = Field(default="0:(1.0)", max_length=32 * 1024)
    angle_schedule: str = Field(default="0:(0)", max_length=32 * 1024)
    translation_x_schedule: str = Field(default="0:(0)", max_length=32 * 1024)
    translation_y_schedule: str = Field(default="0:(0)", max_length=32 * 1024)
    translation_z_schedule: str = Field(default="0:(0)", max_length=32 * 1024)
    rotation_3d_x_schedule: str = Field(default="0:(0)", max_length=32 * 1024)
    rotation_3d_y_schedule: str = Field(default="0:(0)", max_length=32 * 1024)
    rotation_3d_z_schedule: str = Field(default="0:(0)", max_length=32 * 1024)
    color_coherence: Literal["None", "Match Frame 0 LAB"] = "Match Frame 0 LAB"
    diffusion_cadence: int = Field(default=1, ge=1, le=16)
    prompt_blend_frames: int = Field(default=0, ge=0, le=12)
    prompt_strength_boost: float = Field(default=0.0, ge=0.0, le=0.35)
    prompt_strength_boost_frames: int = Field(default=4, ge=0, le=16)
    hybrid_strength_schedule: str = Field(default="0:(0.5)", max_length=32 * 1024)
    hybrid_mode: Literal["normal", "optical_flow"] = "optical_flow"
    init_image_b64: str = ""
    source_video_upload_id: str = ""

    @field_validator("width", "height")
    @classmethod
    def dimensions_must_be_divisible_by_16(cls, value: int) -> int:
        if value % 16:
            raise ValueError("dimension must be divisible by 16")
        return value

    @model_validator(mode="after")
    def validate_animation_request(self) -> "AnimateRequest":
        if self.total_frames < 1 or self.total_frames > 720:
            raise ValueError("total_frames must be between 1 and 720")
        if (
            self.animation_mode == "Video Input"
            and not self.source_video_upload_id.strip()
        ):
            raise ValueError(
                "source_video_upload_id is required for Video Input animation"
            )
        return self

    @property
    def total_frames(self) -> int:
        if self.render_frames is not None:
            return self.render_frames
        return round(self.duration_seconds * self.fps)


class AnimationUploadResponse(BaseModel):
    upload_id: str
    size: int
    sha256: str
    frame_count: int
    width: int
    height: int
    duration: float


class AnimationResult(BaseModel):
    video_url: str
    poster_url: str
    frame_count: int
    fps: int
    duration: float
    gallery_id: int


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    mode: str = "txt2img"           # txt2img | redraw | img2img | inpaint | outpaint | character_edit
    model_profile: str = ""         # krea_turbo | krea_raw | future gated profiles
    diffusion_engine: Literal["native_pytorch", "native_gguf", "native_int8_convrot", "gguf_external", "int8_convrot_external"] = "native_pytorch"
    checkpoint: str = "turbo"       # turbo | raw | custom
    checkpoint_path: str = ""       # custom path override
    quantization: str = "fp8"       # bf16 | fp16 | fp8 | gguf | int8
    turbo_int8_variant: str = "redcraft"  # swappable Turbo INT8 ConvRot checkpoint (see _TURBO_INT8_VARIANTS)
    steps: int = 8
    cfg: float = 0.0
    mu: Optional[float] = None  # None → inference resolves (turbo=1.15, RAW=adaptive)
    y1: float = 0.5
    y2: float = 1.15
    width: int = 1024
    height: int = 1024
    num_images: int = 1
    batch_mode: Literal["safe_queue", "parallel"] = "safe_queue"
    parallel_batch_confirmed: bool = False
    batch_int8_all: bool = False  # sweep the batch across every Turbo INT8 ConvRot variant
    seed: int = -1
    denoise: float = 1.0
    sampler: str = "euler_flow"       # euler | euler_flow | exp_heun_2_x0_sde | guarded Comfy names
    scheduler: str = "simple"
    # CFG-Zero* (arXiv:2503.18886): flow-matching guidance upgrade. Optimized
    # scale corrects velocity error; zero-init skips the first K ODE steps. Only
    # active when guidance (cfg) > 0 and a non-CFG++ sampler is used.
    cfg_zero_star: bool = False
    cfg_zero_init_steps: int = Field(default=1, ge=0, le=4)
    # RES4LYF ClownsharKSampler_Beta. When res4lyf_sampler is a non-empty
    # ClownsharK sampler_name (e.g. "exponential/ddim"), the txt2img/img2img
    # sampler is replaced by ClownsharKSampler_Beta (the Xperiment/uncensored
    # reference recipe). Empty => use the stock KSampler path.
    res4lyf_sampler: str = ""
    res4lyf_eta: float = Field(default=0.5, ge=-100.0, le=100.0)
    res4lyf_bongmath: bool = False
    # Actual-Denoise (mozhaa/ComfyUI-Actual-Denoise): make img2img/edit denoise
    # behave identically across schedulers by injecting the true noise amount.
    actual_denoise: bool = False
    # In-context vision edit (the community "QwenEdit text-encode for Krea 2"):
    # feed a reference image straight into TextEncodeKrea2's Qwen3-VL vision path
    # so an instruction prompt edits it. Empty incontext_image_b64 -> use init image.
    incontext_edit: bool = False
    incontext_image_b64: str = ""
    incontext_mask_b64: str = ""
    incontext_vision_position: str = "before"   # before | after
    incontext_vision_megapixels: float = Field(default=1.0, ge=0.1, le=4.0)
    # Encoder for in-context edit: "krea2" = TextEncodeKrea2 (Qwen3-VL vision +
    # optional style-extraction system prompt), "qwen_edit_plus" =
    # TextEncodeQwenImageEditPlus (stronger multi-image in-context edit encode).
    incontext_encoder: str = "krea2"
    # Optional system prompt fed to TextEncodeKrea2 (krea2 encoder only). Empty
    # while incontext_edit is on -> a built-in style-extract+edit instruction.
    incontext_system_prompt: str = ""
    # Character Edit (conradlocke/krea2-identity-edit via Ostris edit nodes).
    character_edit_source_b64: str = ""
    # Optional second reference for two-input edits. Per the lbouaraba/comfyui-krea2edit
    # model card this is the SCENE/background image (wired as RoPE frame 1 / primary),
    # while character_edit_source_b64 (the subject) becomes frame 2.
    character_edit_reference_b64: str = ""
    character_edit_regions: List[CharacterEditRegion] = Field(default_factory=list, max_length=6)
    character_edit_grounding_px: int = Field(default=768, ge=512, le=1536)
    character_edit_task: Literal["restage", "local_edit", "replace", "restyle", "removal", "two_reference"] = "restage"
    character_edit_lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    # RES4LYF training-free style transfer (ClownGuide_Style_Beta -> ClownsharK
    # guides). When style_transfer_image_b64 is set, the render adopts that image's
    # style (statistics injected into the denoised latent) — no model download.
    style_transfer_image_b64: str = ""
    style_transfer_method: str = "AdaIN"        # AdaIN | WCT | WCT2 | scattersort
    style_transfer_weight: float = Field(default=0.8, ge=0.0, le=2.0)
    style_transfer_apply_to: str = "denoised"   # denoised | positive | negative
    # Krea 2 Depth ControlNet (facok/comfyui-krea2-controlnet + Depth-Anything-3).
    # When depth_control is on, the source image (init_image_b64) is turned into a
    # depth map (DA3) and injected as a control latent through the depth Control
    # LoRA, so the generated image follows the source's depth/composition.
    depth_control: bool = False
    depth_control_strength: float = Field(default=1.2, ge=0.0, le=2.0)  # "Balanced" preset (tracks depth well; >1.5 starts to break)
    depth_estimator: Literal["da3", "depth_anything_v2", "zoe", "midas"] = "da3"
    depth_resolution: int = Field(default=504, ge=256, le=2048)
    depth_invert: bool = False
    # Mr. Flow (training-free staged sampling): render the composition at low res,
    # SR-upscale in pixel space, re-encode, then a short model-native refine at the
    # target size. width/height are the TARGET; low res = target / SR factor.
    god_mode: bool = False  # 4-stage: Krea2 base -> Z-Image Turbo refine -> SeedVR2 upscale -> FaceDetailer
    mrflow: bool = False
    mrflow_upscaler: Literal["esrgan_x2", "remacri_x4"] = "esrgan_x2"
    mrflow_preset: str = ""           # base_12plus1 | base_20plus1 | turbo_8plus1 (auto if empty)
    mrflow_refine_denoise: float = Field(default=0.0, ge=0.0, le=0.6)  # 0 = use preset default
    mrflow_refine_steps: int = Field(default=0, ge=0, le=3)  # 0 = preset default (1)
    inpaint_method: str = "native"    # native | lanpaint_experimental
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
    edit_provider: str = "auto"       # auto | krea_native
    quality_preset: str = "balanced"  # fast | balanced | best | raw_benchmark
    creativity: Literal["raw", "low", "medium", "high"] = "medium"
    style_references: List[StyleReferenceInput] = Field(default_factory=list, max_length=10)
    style_fusion_mode: Literal["style_only", "preserve_structure", "semantic_fusion"] = "semantic_fusion"
    image_prompt_enabled: bool = False
    image_prompt_mode: Literal["match_style", "copy_composition"] = "match_style"
    image_prompt_strength: float = Field(default=0.2, ge=0.1, le=1.0)
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
    # VAE DeGrid (lunaaispace-eng/ComfyUI-DeGrid): remove the 2px Qwen/Wan VAE
    # pixel grid after decode, before any later sharpen/upscale. Default on.
    vae_degrid: bool = True
    # Moodboard: preset mood id + custom reference-image board + influence strength
    mood: str = ""
    moodboard_ids: List[int] = []
    moodboard_uuids: List[str] = []
    moodboard_strength: float = 0.35
    moodboard_images: List[str] = []
    seed_variance_preset: Literal["off", "subtle", "balanced", "creative", "bold", "wild", "custom"] = "off"
    seed_variance_strength: float = 0.0
    seed_variance_algorithm: Literal["legacy", "rbg"] = "legacy"
    seed_variance_model_type: Literal["krea2", "z_image", "qwen_image", "flux", "sdxl", "other"] = "krea2"
    seed_variance_randomize_percent: float = Field(default=0.0, ge=0.0, le=10.0)
    seed_variance_shift_strength: int = Field(default=100, ge=0, le=200)
    seed_variance_protection: Literal["none", "first_quarter", "first_half", "last_quarter", "last_half"] = "first_half"
    seed_variance_direction: Literal[
        "none", "forward", "reverse", "center", "edges",
        "chaos", "order", "abstract", "realistic", "vibrant", "moody", "dreamy",
        "dynamic_pose", "composition", "diversity", "facevar", "visceral_expression_grit",
        "semantic_drift", "structural_lock", "cinematic_framing", "identity_stretch", "texture_lift",
    ] = "none"
    seed_variance_fade_curve: Literal["instant", "linear", "ease_in", "ease_out", "ease_in_out", "smoothstep", "burst"] = "linear"
    seed_variance_injection_start: float = Field(default=0.0, ge=0.0, le=1.0)
    seed_variance_injection_end: float = Field(default=1.0, ge=0.0, le=1.0)
    seed_variance_schedule: Literal["constant", "decreasing", "step_cutoff", "hard_lock", "tiered_release"] = "constant"
    seed_variance_cutoff_step: int = Field(default=8, ge=0, le=100)
    seed_variance_total_steps: int = Field(default=20, ge=1, le=100)
    seed_variance_cutoff_strength: float = Field(default=0.0, ge=0.0, le=1.0)


class HelperBenchmarkRequest(BaseModel):
    models: List[str] = Field(default_factory=list, min_length=1, max_length=4)
    precisions: List[str] = Field(default_factory=list, min_length=1, max_length=4)
    repeats: int = Field(default=3, ge=1, le=20)
    subsequent_krea: bool = False


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
    media_type: Literal["image", "video"] = "image"
    poster_filename: Optional[str] = None
    duration: Optional[float] = None
    frame_count: Optional[int] = None
    project_job_id: Optional[str] = None
    url: Optional[str] = None
    poster_url: Optional[str] = None


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
    mode: str = "recreate"   # recreate | style | character
    guidance: str = ""       # optional: what to focus on / change (blank = full auto prompt)


class DepthPreviewRequest(BaseModel):
    image_b64: str
    estimator: Literal["da3", "depth_anything_v2", "zoe", "midas"] = "da3"
    resolution: int = Field(default=504, ge=256, le=2048)
    invert: bool = False


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
    suggest_moodboards: bool = True


class MoodboardSuggestion(BaseModel):
    id: int
    uuid: str = ""
    title: str
    reason: str = ""
    preview_image_urls: List[str] = []


class ExpandPromptResponse(BaseModel):
    expanded: str
    changed: bool = False
    error: Optional[str] = None
    backend: str = "local"
    suggested_moodboards: List[MoodboardSuggestion] = []
    sign_copy_pass: Optional[dict] = None


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
    preview_image_urls: List[str] = []
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
    krea2_turbo_int8_path: Optional[str] = None
    krea2_raw_int8_path: Optional[str] = None
    output_dir: Optional[str] = None
    prompt_expander_backend: Optional[str] = None
    local_llm_backend: Optional[Literal["comfy", "transformers", "gguf_server"]] = None
    comfy_qwen_model: Optional[str] = None
    comfy_qwen_quant: Optional[str] = None
    comfy_qwen_vision_model: Optional[str] = None
    comfy_qwen_vision_quant: Optional[str] = None
    krea_comfy_warmup: Optional[bool] = None
    local_qwen_model_id: Optional[str] = None
    local_qwen_device: Optional[Literal["auto", "cuda", "cpu"]] = None
    gguf_helper_base_url: Optional[str] = None
    gguf_helper_model: Optional[str] = None
    gguf_helper_timeout_sec: Optional[int] = None
    diffusion_engine: Optional[Literal["native_pytorch", "native_gguf", "native_int8_convrot"]] = None
    gguf_turbo_path: Optional[str] = None
    gguf_raw_path: Optional[str] = None
    ideogram_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = None
    openrouter_free_only: Optional[bool] = None
    krea_share_auto_funnel: Optional[bool] = None
    krea2_vae_path: Optional[str] = None
    krea2_vae_mode: Optional[Literal["qwen", "comfy_qwen", "qwen_wan_blend", "wan_experimental"]] = None
    krea2_vae_blend_radius: Optional[int] = None
    krea2_vae_blend_strength: Optional[float] = None
    krea_attention_backend: Optional[Literal["sdpa", "sage"]] = None
    seedvr2_model: Optional[Literal["3b", "7b"]] = None
