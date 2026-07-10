"""RealESRGAN fallback upscaler + image codec helpers.

All model/VAE-dependent upscaling (tiled VAE, Ultimate SD Upscale, 2-pass
refine, SeedVR2, ESRGAN) runs as ComfyUI graphs in comfy_workflows.comfy_upscale.
This module keeps only the in-process RealESRGAN path, used as a fallback when
ComfyUI is unreachable, plus small PIL/base64 helpers.
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from PIL import Image

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight CI helper paths
    torch = None

logger = logging.getLogger(__name__)

REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)


def _ensure_realesrgan_model(models_dir: Path) -> Path:
    dest = models_dir / "RealESRGAN_x4plus.pth"
    if dest.exists():
        return dest
    import urllib.request
    logger.info("Downloading RealESRGAN model...")
    urllib.request.urlretrieve(REALESRGAN_URL, str(dest))
    logger.info(f"Downloaded RealESRGAN to {dest}")
    return dest


def upscale_realesrgan(
    img: Image.Image,
    models_dir: Path,
    scale: int = 4,
    tile: int = 512,
    tile_pad: int = 32,
) -> Image.Image:
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        logger.warning("realesrgan not installed, falling back to bicubic.")
        w, h = img.size
        return img.resize((w * scale, h * scale), Image.LANCZOS)

    model_path = _ensure_realesrgan_model(models_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=rrdb,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=0,
        half=(device == "cuda"),
        device=device,
    )
    import numpy as np
    arr = np.array(img.convert("RGB"))
    out_arr, _ = upsampler.enhance(arr, outscale=scale)
    return Image.fromarray(out_arr)


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))
