from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from krea2.performance_guard import accelerator_status

PID_ESTIMATED_VRAM_GB = 15.0


@dataclass
class PiDSettings:
    decoder_path: str = ""
    text_encoder_path: str = ""
    enabled: bool = False
    min_free_vram_gb: float = PID_ESTIMATED_VRAM_GB


def _sageattention_selected(status: dict[str, Any]) -> bool:
    sage = status.get("sageattention") or {}
    return bool(sage.get("selected") or sage.get("default") or sage.get("active"))


def pid_status(settings: PiDSettings, *, free_vram_gb: float | None = None) -> dict[str, Any]:
    decoder_installed = bool(settings.decoder_path and Path(settings.decoder_path).is_file())
    text_encoder_installed = bool(settings.text_encoder_path and Path(settings.text_encoder_path).is_file())
    accel = accelerator_status()
    blocked: list[str] = []
    if not decoder_installed:
        blocked.append("PiD decoder asset is not installed.")
    if not text_encoder_installed:
        blocked.append("PiD Gemma text encoder asset is not installed.")
    if _sageattention_selected(accel):
        blocked.append("SageAttention must be disabled for PiD; it can produce black images.")
    if free_vram_gb is not None and free_vram_gb < float(settings.min_free_vram_gb):
        blocked.append(f"PiD needs about {settings.min_free_vram_gb:.0f}GB free VRAM; only {free_vram_gb:.1f}GB is free.")
    return {
        "available": not blocked,
        "enabled": bool(settings.enabled),
        "estimated_vram_gb": float(settings.min_free_vram_gb),
        "blocked_reasons": blocked,
        "assets": {
            "decoder": {"path": settings.decoder_path, "installed": decoder_installed},
            "text_encoder": {"path": settings.text_encoder_path, "installed": text_encoder_installed},
        },
        "accelerators": accel,
    }


def _pid_runtime_available() -> bool:
    try:
        import pid  # noqa: F401
        return True
    except Exception:
        return False


def upscale_pid(img: Image.Image, settings: PiDSettings, *, prompt: str = "", scale: float = 4.0) -> Image.Image:
    status = pid_status(settings)
    if status["blocked_reasons"]:
        raise RuntimeError("PiD is not available: " + " ".join(status["blocked_reasons"]))
    if not _pid_runtime_available():
        raise RuntimeError(
            "PiD runtime is not installed or vendored yet. Install/port nv-tlabs/PiD before running PiD decode."
        )
    # The real PiD call will be wired here once the runtime is vendored and A/B
    # validated. Keep this explicit rather than silently falling back to another
    # upscale method, because the user must know PiD did not run.
    raise RuntimeError("PiD runtime hook is not implemented yet.")
