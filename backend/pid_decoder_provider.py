from __future__ import annotations

from dataclasses import dataclass
import gc
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
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
    if not official_checkpoint_installed:
        blocked.append("Native PiD runtime checkpoint is not installed.")
    if not official_vae_installed:
        blocked.append("Native PiD QwenImage VAE tokenizer is not installed.")
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
            "official_checkpoint": {"path": settings.official_checkpoint_path, "installed": official_checkpoint_installed},
            "official_vae": {"path": settings.official_vae_path, "installed": official_vae_installed},
            "legacy_comfy_decoder": {"path": settings.decoder_path, "installed": decoder_installed, "required": False},
            "legacy_gemma_text_encoder": {"path": settings.text_encoder_path, "installed": text_encoder_installed, "required": False},
        },
        "accelerators": accel,
    }


def _pid_runtime_available() -> bool:
    try:
        import pid  # noqa: F401
        return True
    except Exception:
        return False


def _runtime_key(settings: PiDSettings) -> tuple[str, str]:
    return (str(settings.official_checkpoint_path), str(settings.official_vae_path))


def _pil_to_pid_tensor(img: Image.Image, *, pad_to_multiple: int = 16):
    import torch

    img = img.convert("RGB")
    w, h = img.size
    new_w = (w // pad_to_multiple) * pad_to_multiple
    new_h = (h // pad_to_multiple) * pad_to_multiple
    if new_w == 0 or new_h == 0:
        raise ValueError(f"Image size {w}x{h} is smaller than pad_to_multiple={pad_to_multiple}.")
    if (new_w, new_h) != (w, h):
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))
    arr = np.asarray(img, np.uint8).astype("float32")
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0


def _tensor_to_pil(sample) -> Image.Image:
    if sample.dim() == 4:
        sample = sample.squeeze(1)
    tensor = (sample.float().clamp(-1, 1) + 1) * 127.5
    arr = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)


class NativePiDRuntime:
    def __init__(self, settings: PiDSettings) -> None:
        self.settings = settings
        self.key = _runtime_key(settings)
        self._model = None
        self._lock = threading.Lock()

    def _build_args(self, *, scale: float):
        from types import SimpleNamespace
        from pid._src.inference.checkpoint_registry import get_pid_checkpoint

        checkpoint = get_pid_checkpoint("qwenimage", "2kto4k")
        return SimpleNamespace(
            backbone="qwenimage",
            experiment=checkpoint.experiment,
            config_file="pid/_src/configs/pid/config.py",
            checkpoint_path=str(Path(self.settings.official_checkpoint_path)),
            pid_ckpt_type="2kto4k",
            load_ema_to_reg=False,
            compile=False,
            seed=0,
            cfg_scale=1.0,
            pid_inference_steps=4,
            shift=None,
            scale=max(1, int(round(float(scale or 4.0)))),
            save_format="png",
            output_dir=None,
            upload=False,
            note="",
            group_name="krea_native_pid",
            prompt=None,
            manifest=None,
            input_path=None,
            degrade_sigmas=[0.0],
            extra_experiment_opts=[],
        )

    def load(self, *, scale: float = 4.0):
        with self._lock:
            if self._model is not None:
                return self._model
            import pid
            from pid._src.inference.decoder import load_our_decoder

            args = self._build_args(scale=scale)
            package_root = Path(pid.__file__).parent.parent
            vae_path = Path(self.settings.official_vae_path).as_posix()
            experiment_opts = [f"+model.config.tokenizer.vae_pth={vae_path}"]
            cwd = os.getcwd()
            try:
                os.chdir(str(package_root))
                self._model = load_our_decoder(args, experiment_opts, True)
            finally:
                os.chdir(cwd)
            return self._model

    def upscale(self, img: Image.Image, *, prompt: str = "", scale: float = 4.0) -> Image.Image:
        import torch
        from pid._src.inference.decoder import add_noise

        if not torch.cuda.is_available():
            raise RuntimeError("Native PiD requires CUDA.")
        args = self._build_args(scale=scale)
        model = self.load(scale=scale)
        caption = prompt or getattr(getattr(model, "config", object()), "fixed_positive_prompt", None) or "high quality detailed image"
        with torch.inference_mode():
            input_tensor = _pil_to_pid_tensor(img).to(dtype=torch.bfloat16, device="cuda")
            clean_latent = model.encode_lq_latent(input_tensor)
            vae_compression = int(model.vae_encoder.spatial_compression_factor)
            vae_h = int(clean_latent.shape[-2]) * vae_compression
            vae_w = int(clean_latent.shape[-1]) * vae_compression
            target_hw = (vae_h * args.scale, vae_w * args.scale)
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            latent = add_noise(clean_latent.float(), 0.0, gen, args.backbone).to(dtype=torch.bfloat16)
            data_batch = {
                model.config.input_caption_key: [caption],
                "LQ_latent": latent.to(dtype=torch.bfloat16, device="cuda"),
                "degrade_sigma": torch.tensor([0.0], device="cuda", dtype=torch.float32),
            }
            samples_out = model.generate_samples_from_batch(
                data_batch,
                cfg_scale=args.cfg_scale,
                num_steps=args.pid_inference_steps,
                seed=args.seed,
                shift=args.shift,
                image_size=target_hw,
            )
            out = samples_out[0].float().cpu().clamp(-1, 1)
        del input_tensor, clean_latent, latent, data_batch, samples_out
        return _tensor_to_pil(out)

    def release(self) -> None:
        model = self._model
        self._model = None
        try:
            if model is not None and hasattr(model, "cpu"):
                model.cpu()
        except Exception:
            pass
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass


_PID_RUNTIME: NativePiDRuntime | None = None
_PID_RUNTIME_LOCK = threading.Lock()


def _get_pid_runtime(settings: PiDSettings) -> NativePiDRuntime:
    global _PID_RUNTIME
    key = _runtime_key(settings)
    with _PID_RUNTIME_LOCK:
        if _PID_RUNTIME is None or getattr(_PID_RUNTIME, "key", None) != key:
            if _PID_RUNTIME is not None:
                _PID_RUNTIME.release()
            _PID_RUNTIME = NativePiDRuntime(settings)
        return _PID_RUNTIME


def release_pid_runtime() -> dict[str, Any]:
    global _PID_RUNTIME
    with _PID_RUNTIME_LOCK:
        runtime = _PID_RUNTIME
        _PID_RUNTIME = None
    if runtime is None:
        return {"released": False}
    runtime.release()
    return {"released": True}


def upscale_pid(img: Image.Image, settings: PiDSettings, *, prompt: str = "", scale: float = 4.0) -> Image.Image:
    if int(round(float(scale or 4.0))) != 4:
        raise ValueError("PiD QwenImage upscale is only supported at 4x. Use Wan 2.1, model refine, or Ultimate SD Upscale for 2x.")
    status = pid_status(settings)
    if status["blocked_reasons"]:
        raise RuntimeError("PiD is not available: " + " ".join(status["blocked_reasons"]))
    if not _pid_runtime_available():
        raise RuntimeError(
            "PiD runtime is not installed or vendored yet. Install/port nv-tlabs/PiD before running PiD decode."
        )
    return _get_pid_runtime(settings).upscale(img, prompt=prompt, scale=scale)
