from __future__ import annotations
import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # type: ignore
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent


class AppSettings(BaseSettings):
    hf_token: str = ""
    civitai_token: str = ""
    krea2_turbo_path: str = ""
    krea2_raw_path: str = ""
    krea2_turbo_int8_path: str = ""
    krea2_raw_int8_path: str = ""
    output_dir: str = str(BASE_DIR / "outputs")
    models_dir: str = str(BASE_DIR / "models")
    loras_dir: str = str(BASE_DIR / "models" / "loras")
    logs_dir: str = str(BASE_DIR / "logs")
    db_path: str = str(BASE_DIR / "app.db")
    prompt_expander_backend: str = "local"  # local | openrouter | ideogram-json
    local_llm_backend: str = "comfy"  # comfy | transformers | gguf_server
    comfy_qwen_model: str = "2b"  # 2b | 4b | exact ComfyUI-QwenVL model_name (helpers only)
    comfy_qwen_quant: str = "8bit"  # 4bit | 8bit | fp16
    comfy_qwen_vision_model: str = "4b"  # richer image understanding
    comfy_qwen_vision_quant: str = "8bit"  # 4bit | 8bit | fp16
    krea_comfy_warmup: bool = True  # low-priority, preemptible 1K Krea warmup
    local_qwen_model_id: str = ""  # optional Transformers repo/path override for local prompt expansion
    local_qwen_device: str = "auto"  # auto | cuda | cpu; auto avoids CUDA when VRAM is tight
    gguf_helper_base_url: str = "http://127.0.0.1:1234/v1"
    gguf_helper_model: str = "BennyDaBall/Krea-2-Engineer-V1-GGUF:Q4_K_M"
    gguf_helper_timeout_sec: int = 120
    diffusion_engine: str = "native_pytorch"  # native_pytorch | native_gguf | native_int8_convrot
    gguf_turbo_path: str = ""
    gguf_raw_path: str = ""
    ideogram_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemma-4-31b-it:free"
    openrouter_free_only: bool = True
    krea_share_auto_funnel: bool = False
    krea2_auto_checkpoint: str = ""   # path to auto-load on startup
    krea2_auto_quant: str = "bf16"    # bf16, fp16, fp8, gguf, or int8
    krea2_blocks_to_swap: int = 0     # low-VRAM: stream last N DiT blocks from RAM (0 = off)
    krea2_vae_path: str = ""          # optional manual VAE override; empty = stock Qwen VAE
    krea2_vae_mode: str = "qwen"      # qwen | comfy_qwen | qwen_wan_blend | wan_experimental
    krea2_vae_blend_radius: int = 24
    krea2_vae_blend_strength: float = 0.65
    krea2_fp8_fast_matmul: bool = False  # opt-in: fp8 _scaled_mm on Ada/Blackwell (faster, slight quality trade)
    krea2_moodboard_auto_enrich: bool = True  # background-precompute Qwen guidance for official moodboards when idle
    krea2_torch_compile: bool = False  # opt-in: torch.compile the DiT (experimental; needs Triton/inductor)
    krea_attention_backend: str = "sdpa"  # sdpa | sage
    seedvr2_model: str = "3b"  # 3b (fast fp8 default) | 7b (best-quality fp16, needs block-swap)
    animation_state_root: str = str(BASE_DIR / "data" / "animations")
    animation_upload_root: str = str(BASE_DIR / "data" / "animation_uploads")
    animation_chunk_size: int = Field(default=8, ge=8, le=8)
    animation_max_frames: int = Field(default=720, ge=1, le=720)
    animation_max_dimension: int = Field(default=1536, ge=256, le=1536)
    animation_max_upload_bytes: int = Field(
        default=256 * 1024 * 1024, ge=1024 * 1024, le=2 * 1024 * 1024 * 1024
    )
    animation_uploads_per_user: int = Field(default=3, ge=1, le=16)
    animation_upload_bytes_per_user: int = Field(
        default=512 * 1024 * 1024, ge=1024 * 1024, le=4 * 1024 * 1024 * 1024
    )
    animation_uploads_global: int = Field(default=32, ge=1, le=256)
    animation_upload_bytes_global: int = Field(
        default=2 * 1024 * 1024 * 1024,
        ge=1024 * 1024,
        le=16 * 1024 * 1024 * 1024,
    )
    animation_upload_ttl_seconds: int = Field(
        default=24 * 60 * 60, ge=60, le=7 * 24 * 60 * 60
    )
    animation_upload_cleanup_interval_seconds: int = Field(
        default=300, ge=30, le=3600
    )
    animation_max_source_duration_seconds: float = Field(
        default=60.0, ge=0.5, le=60.0
    )
    animation_active_per_user: int = Field(default=1, ge=1, le=1)

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = AppSettings()

# Auto-detect checkpoints if paths not set in .env
_KREA2_DIR = BASE_DIR / "models" / "krea2"
_AUTO_TURBO_CANDIDATES = [
    _KREA2_DIR / "krea2_turbo_fp8_scaled.safetensors",
    _KREA2_DIR / "diffusion_models" / "krea2_turbo_fp8_scaled.safetensors",
    _KREA2_DIR / "krea2_turbo_bf16.safetensors",
    _KREA2_DIR / "diffusion_models" / "krea2_turbo_bf16.safetensors",
]
_AUTO_RAW_CANDIDATES = [
    _KREA2_DIR / "krea2_raw_bf16.safetensors",
    _KREA2_DIR / "diffusion_models" / "krea2_raw_bf16.safetensors",
    _KREA2_DIR / "krea2_raw_fp8_scaled.safetensors",
    _KREA2_DIR / "diffusion_models" / "krea2_raw_fp8_scaled.safetensors",
]
_AUTO_TURBO_INT8_CANDIDATES = [
    _KREA2_DIR / "kreamania_v2-int8-convrot.safetensors",
    _KREA2_DIR / "diffusion_models" / "kreamania_v2-int8-convrot.safetensors",
    _KREA2_DIR / "krea2_turbo_int8_convrot.safetensors",
    _KREA2_DIR / "diffusion_models" / "krea2_turbo_int8_convrot.safetensors",
    _KREA2_DIR / "kreamania_v3-int8-convrot-simple.safetensors",
    _KREA2_DIR / "diffusion_models" / "kreamania_v3-int8-convrot-simple.safetensors",
]
_AUTO_RAW_INT8_CANDIDATES = [
    _KREA2_DIR / "krea2_raw_int8_convrot.safetensors",
    _KREA2_DIR / "diffusion_models" / "krea2_raw_int8_convrot.safetensors",
]
if not settings.krea2_turbo_path:
    for _c in _AUTO_TURBO_CANDIDATES:
        if _c.exists():
            settings.krea2_turbo_path = str(_c)
            break
if not settings.krea2_raw_path:
    for _c in _AUTO_RAW_CANDIDATES:
        if _c.exists():
            settings.krea2_raw_path = str(_c)
            break
if not settings.krea2_turbo_int8_path:
    for _c in _AUTO_TURBO_INT8_CANDIDATES:
        if _c.exists():
            settings.krea2_turbo_int8_path = str(_c)
            break
if not settings.krea2_raw_int8_path:
    for _c in _AUTO_RAW_INT8_CANDIDATES:
        if _c.exists():
            settings.krea2_raw_int8_path = str(_c)
            break
if not settings.krea2_auto_checkpoint and settings.krea2_turbo_path:
    settings.krea2_auto_checkpoint = settings.krea2_turbo_path

HF_HOME = str(BASE_DIR / "models" / "hf")
LOCAL_AI_DIR = BASE_DIR / "models" / "local_ai"
os.environ.setdefault("HF_HOME", HF_HOME)
if settings.hf_token:
    os.environ.setdefault("HF_TOKEN", settings.hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", settings.hf_token)

OUTPUTS_DIR = Path(settings.output_dir)
MODELS_DIR = Path(settings.models_dir)
LORAS_DIR = Path(settings.loras_dir)
LOGS_DIR = Path(settings.logs_dir)
DB_PATH = Path(settings.db_path)
DIST_DIR = BASE_DIR / "frontend" / "dist"

for _d in (OUTPUTS_DIR, MODELS_DIR, LORAS_DIR, LOGS_DIR, LOCAL_AI_DIR):
    _d.mkdir(parents=True, exist_ok=True)
