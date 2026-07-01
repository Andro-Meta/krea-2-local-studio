from __future__ import annotations

from dataclasses import dataclass
import base64
import io
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from krea2.performance_guard import accelerator_status

PID_ESTIMATED_VRAM_GB = 15.0


@dataclass
class PiDSettings:
    decoder_path: str = ""
    text_encoder_path: str = ""
    official_checkpoint_path: str = ""
    official_vae_path: str = ""
    enabled: bool = False
    min_free_vram_gb: float = PID_ESTIMATED_VRAM_GB
    timeout_sec: int = 900


def _sageattention_selected(status: dict[str, Any]) -> bool:
    sage = status.get("sageattention") or {}
    return bool(sage.get("selected") or sage.get("default") or sage.get("active"))


def pid_status(settings: PiDSettings, *, free_vram_gb: float | None = None) -> dict[str, Any]:
    decoder_installed = bool(settings.decoder_path and Path(settings.decoder_path).is_file())
    text_encoder_installed = bool(settings.text_encoder_path and Path(settings.text_encoder_path).is_file())
    official_checkpoint_installed = bool(settings.official_checkpoint_path and Path(settings.official_checkpoint_path).is_file())
    official_vae_installed = bool(settings.official_vae_path and Path(settings.official_vae_path).is_file())
    accel = accelerator_status()
    blocked: list[str] = []
    if not decoder_installed:
        blocked.append("PiD Comfy decoder asset is not installed.")
    if not text_encoder_installed:
        blocked.append("PiD Gemma text encoder asset is not installed.")
    if not official_checkpoint_installed:
        blocked.append("Official PiD runtime checkpoint is not installed.")
    if not official_vae_installed:
        blocked.append("Official PiD QwenImage VAE tokenizer is not installed.")
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
            "official_checkpoint": {"path": settings.official_checkpoint_path, "installed": official_checkpoint_installed},
            "official_vae": {"path": settings.official_vae_path, "installed": official_vae_installed},
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
    return _upscale_pid_subprocess(img, settings, prompt=prompt, scale=scale)


def _upscale_pid_subprocess(img: Image.Image, settings: PiDSettings, *, prompt: str, scale: float) -> Image.Image:
    """Run official PiD from_clean in a subprocess to isolate heavy CUDA state."""
    import pid

    package_root = Path(pid.__file__).parent.parent
    config_file = "pid/_src/configs/pid/config.py"
    with tempfile.TemporaryDirectory(prefix="krea_pid_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "input.png"
        output_dir = tmp_dir / "out"
        img.convert("RGB").save(input_path)
        scale_int = max(1, int(round(float(scale or 4.0))))
        cmd = [
            sys.executable,
            "-m",
            "pid._src.inference.from_clean",
            "--backbone",
            "qwenimage",
            "--pid_ckpt_type",
            "2kto4k",
            "--checkpoint_path",
            str(Path(settings.official_checkpoint_path)),
            "--config_file",
            str(config_file),
            "--input_path",
            str(input_path),
            "--prompt",
            prompt or "high quality detailed image",
            "--degrade_sigmas",
            "0.0",
            "--output_dir",
            str(output_dir),
            "--cfg_scale",
            "1",
            "--pid_inference_steps",
            "4",
            "--scale",
            str(scale_int),
            "--save_format",
            "png",
        ]
        env = dict(**__import__("os").environ)
        # The official QwenImage tokenizer expects ./checkpoints/QwenImage_VAE_2d.pth.
        work_dir = package_root
        vae_path = Path(settings.official_vae_path).as_posix()
        cmd.append(f"+model.config.tokenizer.vae_pth={vae_path}")
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(60, int(settings.timeout_sec or 900)),
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "PiD subprocess failed.").strip()
            raise RuntimeError(detail[-3000:])
        outputs = sorted(output_dir.rglob("*.png"))
        pid_outputs = [
            p for p in outputs
            if "vae_decode" not in [part.lower() for part in p.parts]
            and "input" not in [part.lower() for part in p.parts]
        ]
        if not pid_outputs:
            raise RuntimeError(f"PiD finished but did not produce an output image. Files: {[str(p) for p in outputs]}")
        return Image.open(pid_outputs[0]).convert("RGB")
