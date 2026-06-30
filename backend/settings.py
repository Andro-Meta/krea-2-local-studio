from __future__ import annotations
import os
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # type: ignore

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
    local_llm_backend: str = "transformers"  # transformers | gguf_server
    local_qwen_model_id: str = ""  # optional Transformers repo/path override for local prompt expansion
    local_qwen_device: str = "auto"  # auto | cuda | cpu; auto avoids CUDA when VRAM is tight
    gguf_helper_base_url: str = "http://127.0.0.1:1234/v1"
    gguf_helper_model: str = "BennyDaBall/Krea-2-Engineer-V1-GGUF:Q4_K_M"
    gguf_helper_timeout_sec: int = 120
    diffusion_engine: str = "native_pytorch"  # native_pytorch | native_int8_convrot | gguf_external
    gguf_sd_cli_path: str = ""
    gguf_turbo_path: str = ""
    gguf_raw_path: str = ""
    gguf_llm_path: str = ""
    gguf_vae_path: str = ""
    gguf_lora_dir: str = ""
    gguf_timeout_sec: int = 600
    ideogram_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemma-4-31b-it:free"
    openrouter_free_only: bool = True
    krea_share_auto_funnel: bool = False
    krea2_auto_checkpoint: str = ""   # path to auto-load on startup
    krea2_auto_quant: str = "bf16"    # bf16 or fp8
    krea2_blocks_to_swap: int = 0     # low-VRAM: stream last N DiT blocks from RAM (0 = off)
    krea2_vae_path: str = ""          # optional override VAE (HDR/real/clear); empty = stock Qwen VAE
    krea2_fp8_fast_matmul: bool = False  # opt-in: fp8 _scaled_mm on Ada/Blackwell (faster, slight quality trade)
    krea2_moodboard_auto_enrich: bool = True  # background-precompute Qwen guidance for official moodboards when idle
    krea2_torch_compile: bool = False  # opt-in: torch.compile the DiT (experimental; needs Triton/inductor)
    krea_attention_backend: str = "sdpa"  # sdpa | sage

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
    _KREA2_DIR / "krea2_turbo_int8_convrot.safetensors",
    _KREA2_DIR / "diffusion_models" / "krea2_turbo_int8_convrot.safetensors",
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
