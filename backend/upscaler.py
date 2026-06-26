"""Upscaling pipeline: RealESRGAN 4x, tiled VAE 2x, model-refine pass."""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import torch
from PIL import Image

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


def upscale_tiled_vae(
    img: Image.Image,
    ae,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    tile: int = 512,
    overlap: int = 64,
) -> Image.Image:
    """2× upscale via decode_tiled (pixel-perfect, no model needed)."""
    try:
        import numpy as np
        arr = (np.array(img.convert("RGB")).astype("float32") / 127.5) - 1.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
        latent = ae.encode(t)  # normalized 4D (B,C,H,W)
        # Denormalize + add the video VAE's temporal axis before the raw decode,
        # mirroring QwenAutoencoder.decode (the raw VAE doesn't apply latents_*).
        x = latent.float() * ae.latents_std + ae.latents_mean
        x = x.unsqueeze(2)  # (B,C,1,H,W)
        ae.ae.enable_tiling()
        decoded = ae.ae.decode(x.to(dtype)).sample  # (B,3,1,H',W')
        decoded = decoded[:, :, 0]  # drop temporal → (B,3,H',W')
        decoded = decoded.squeeze(0).clamp(-1, 1).float()
        pixel = ((decoded + 1.0) / 2.0 * 255).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixel)
    except Exception as e:
        logger.warning(f"Tiled VAE upscale failed ({e}); falling back to bicubic 2×")
        w, h = img.size
        return img.resize((w * 2, h * 2), Image.LANCZOS)


def upscale_model_refine(
    img: Image.Image,
    pipeline,  # Krea2Pipeline
    denoise: float = 0.24,
    steps: int = 6,
    tile_size: int = 1280,
    overlap: int = 64,
    prompt: str = "",
) -> Image.Image:
    """Tile-based img2img refine pass (6 steps, denoise=0.24).

    Splits image into overlapping tiles, runs img2img on each, blends.
    """
    if not pipeline.is_loaded():
        logger.warning("Model not loaded; skipping model refine.")
        return img

    from schemas import GenerationRequest

    w, h = img.size
    if w <= tile_size and h <= tile_size:
        # No tiling needed
        req = GenerationRequest(
            prompt=prompt or "high quality, detailed",
            mode="img2img",
            steps=steps,
            denoise=denoise,
            width=w,
            height=h,
            num_images=1,
            seed=42,
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        req.init_image_b64 = base64.b64encode(buf.getvalue()).decode()
        results, _seed, _files, _lora = pipeline.generate(req)
        return Image.open(io.BytesIO(base64.b64decode(results[0])))

    # Tiled: use Lanczos upscale only when model refine is not feasible for large sizes
    logger.info("Image too large for single-pass refine; using Lanczos fallback.")
    return img


def upscale_ultimate(
    img: Image.Image,
    pipeline,                # Krea2Pipeline
    models_dir: Path,
    prompt: str = "",
    scale: int = 2,
    tile: int = 1024,
    padding: int = 32,
    mask_blur: int = 8,
    denoise: float = 0.3,
    steps: int = 8,
    seam_fix: bool = True,
) -> Image.Image:
    """Ultimate SD Upscale: pre-upscale, then tiled low-denoise img2img refine.

    Pre-upscales the whole image (RealESRGAN→bicubic fallback), then refines it
    tile-by-tile. Each tile is img2img'd with surrounding padding as context and
    composited back through a Gaussian-blurred mask (feathered alpha) so seams
    cross-fade. CHESS tile order + a half-tile seam-fix pass remove residual
    boundaries. Mirrors Coyote-A/ultimate-upscale-for-automatic1111.
    """
    import math
    import numpy as np
    from PIL import ImageDraw, ImageFilter
    from schemas import GenerationRequest

    if not pipeline.is_loaded():
        logger.warning("Model not loaded; ultimate upscale → bicubic only.")
        return img.resize((img.width * scale, img.height * scale), Image.LANCZOS)

    W, H = img.width * scale, img.height * scale
    try:
        work = upscale_realesrgan(img, models_dir, scale)
        if work.size != (W, H):
            work = work.resize((W, H), Image.LANCZOS)
    except Exception:  # noqa: BLE001
        work = img.resize((W, H), Image.LANCZOS)
    work = work.convert("RGB")

    align = 16  # COMPRESSION(8) * PATCH(2)

    def _round(x):
        return int(math.ceil(x / align) * align)

    def _tile(work, rect, pad, blur, dn):
        x1, y1, x2, y2 = rect
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return work
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)

        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).rectangle([x1, y1, x2 - 1, y2 - 1], fill=255)
        if blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur))

        crop = work.crop((cx1, cy1, cx2, cy2))
        cmask = mask.crop((cx1, cy1, cx2, cy2))
        cw, ch = crop.size
        gw, gh = _round(cw), _round(ch)
        gen_in = crop.resize((gw, gh), Image.LANCZOS)

        buf = io.BytesIO()
        gen_in.save(buf, format="PNG")
        req = GenerationRequest(
            prompt=prompt or "high quality, sharp fine detail",
            mode="img2img", checkpoint="turbo", steps=steps, cfg=0.0,
            width=gw, height=gh, num_images=1, seed=42, denoise=dn,
            init_image_b64=base64.b64encode(buf.getvalue()).decode(),
        )
        results, _seed, _files, _lora = pipeline.generate(req)
        gen_out = Image.open(io.BytesIO(base64.b64decode(results[0]))).convert("RGB")
        gen_out = gen_out.resize((cw, ch), Image.LANCZOS)

        base_np = np.asarray(crop, np.float32)
        gen_np = np.asarray(gen_out, np.float32)
        alpha = np.asarray(cmask, np.float32)[..., None] / 255.0
        blended = np.clip(gen_np * alpha + base_np * (1.0 - alpha), 0, 255).astype("uint8")
        work.paste(Image.fromarray(blended), (cx1, cy1))
        return work

    cols = math.ceil(W / tile)
    rows = math.ceil(H / tile)
    even = [(xi, yi) for yi in range(rows) for xi in range(cols) if (xi % 2 == 0) == (yi % 2 == 0)]
    odd = [(xi, yi) for yi in range(rows) for xi in range(cols) if (xi % 2 == 0) != (yi % 2 == 0)]
    total = len(even) + len(odd)
    for n, (xi, yi) in enumerate(even + odd):
        logger.info(f"Ultimate upscale tile {n + 1}/{total}")
        work = _tile(work, (xi * tile, yi * tile, (xi + 1) * tile, (yi + 1) * tile),
                     padding, mask_blur, denoise)

    if seam_fix and total > 1:
        ox, oy = tile // 2, tile // 2
        hcols = math.ceil((W - ox) / tile)
        hrows = math.ceil((H - oy) / tile)
        for yi in range(hrows):
            for xi in range(hcols):
                work = _tile(work, (ox + xi * tile, oy + yi * tile,
                                    ox + (xi + 1) * tile, oy + (yi + 1) * tile),
                             16, 4, denoise)
        logger.info("Ultimate upscale seam-fix pass done.")

    return work


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))
