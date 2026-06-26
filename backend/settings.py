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
    output_dir: str = str(BASE_DIR / "outputs")
    models_dir: str = str(BASE_DIR / "models")
    loras_dir: str = str(BASE_DIR / "models" / "loras")
    logs_dir: str = str(BASE_DIR / "logs")
    db_path: str = str(BASE_DIR / "app.db")
    prompt_expander_backend: str = "local"  # local | openrouter | ideogram-json
    ideogram_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemma-4-31b-it:free"
    openrouter_free_only: bool = True
    krea2_auto_checkpoint: str = ""   # path to auto-load on startup
    krea2_auto_quant: str = "bf16"    # bf16 or fp8

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
