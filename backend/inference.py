"""Krea 2 pipeline manager.

Handles model loading (with LoadWatchdog + encoder offload) and generation
for txt2img, img2img, and inpaint modes.
"""
from __future__ import annotations

import base64
import contextlib
import gc
import hashlib
import io
import logging
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

# Krea2 mmdit.py decorates posemb.forward with @torch.compile(fullgraph=True).
# Windows has no triton, so inductor would hard-fail at first forward. Disable
# dynamo before torch loads it → runs eager (correct, just unoptimized).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from PIL import Image

from conditioning import blend_split_conditioning, parse_weights, rebalance
from generation_metadata import build_generation_metadata
from krea_enhancer import krea_enhancer_context
from lora_manager import apply_loras, build_trigger_prompt
from moodboards_catalog import moodboard_generation_context
from memory_manager import clear_cuda_cache
from output_saver import encode_images
from seed_variance import apply_seed_variance
from settings import OUTPUTS_DIR, settings
from support_models import support_model_path
from system_check import get_gpu_info, get_gpu_process_details, get_ram_gb, mem_snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _request_field_was_set(req, field: str) -> bool:
    fields = getattr(req, "model_fields_set", None)
    if fields is None:
        fields = getattr(req, "__fields_set__", set())
    return field in fields


def _apply_creativity_defaults(req) -> None:
    creativity = str(getattr(req, "creativity", "medium") or "medium").lower()
    presets = {
        "raw": {"moodboard_strength": 0.15, "rebalance_multiplier": 2.0},
        "low": {"moodboard_strength": 0.25, "rebalance_multiplier": 3.0},
        "medium": {"moodboard_strength": 0.35, "rebalance_multiplier": 4.0},
        "high": {"moodboard_strength": 0.5, "rebalance_multiplier": 5.5},
    }
    preset = presets.get(creativity, presets["medium"])
    if not _request_field_was_set(req, "moodboard_strength"):
        req.moodboard_strength = preset["moodboard_strength"]
    if not _request_field_was_set(req, "rebalance_multiplier"):
        req.rebalance_multiplier = preset["rebalance_multiplier"]


def normalize_generation_defaults(req) -> None:
    """Apply model-family defaults for callers that bypass the frontend presets."""
    checkpoint = str(getattr(req, "checkpoint", "") or "").lower()
    quality_preset = str(getattr(req, "quality_preset", "") or "").lower()
    _apply_creativity_defaults(req)
    raw_defaults = checkpoint == "raw" or quality_preset == "raw_benchmark"
    force_raw_benchmark = quality_preset == "raw_benchmark"
    if raw_defaults:
        if force_raw_benchmark or not _request_field_was_set(req, "steps"):
            req.steps = 52
        if force_raw_benchmark or not _request_field_was_set(req, "cfg"):
            req.cfg = 3.5
        if force_raw_benchmark or not _request_field_was_set(req, "mu"):
            req.mu = None
        if force_raw_benchmark or not _request_field_was_set(req, "quantization"):
            req.quantization = "bf16"
        return

    if checkpoint == "turbo":
        if not _request_field_was_set(req, "steps"):
            req.steps = 8
        if not _request_field_was_set(req, "cfg"):
            req.cfg = 0.0
        if not _request_field_was_set(req, "mu"):
            req.mu = 1.15
        if not _request_field_was_set(req, "quantization"):
            req.quantization = "fp8"


def resolve_style_references(req, *, limit: int = 10) -> list[dict]:
    """Return structured style refs, preserving legacy ref_image1..3 inputs."""
    refs: list[dict] = []
    for item in list(getattr(req, "style_references", []) or []):
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        image_b64 = str(item.get("image_b64", "") or "")
        if not image_b64:
            continue
        refs.append({
            "image_b64": image_b64,
            "strength": float(item.get("strength", 1.0)),
            "role": str(item.get("role", "style") or "style"),
            "token_size": str(item.get("token_size", "normal") or "normal"),
        })

    for image_b64 in (
        getattr(req, "ref_image1_b64", None),
        getattr(req, "ref_image2_b64", None),
        getattr(req, "ref_image3_b64", None),
    ):
        if image_b64:
            refs.append({
                "image_b64": image_b64,
                "strength": 1.0,
                "role": "style",
                "token_size": "normal",
            })
        if len(refs) >= limit:
            break
    return refs[:limit]


def _align_conditioning_pair(text_txt, text_mask, image_txt, image_mask):
    seq = min(text_txt.shape[1], image_txt.shape[1])
    text_txt = text_txt[:, :seq]
    image_txt = image_txt[:, :seq]
    text_mask = text_mask[:, :seq] if text_mask is not None else None
    image_mask = image_mask[:, :seq] if image_mask is not None else None
    if text_mask is None:
        mask = image_mask
    elif image_mask is None:
        mask = text_mask
    else:
        mask = text_mask | image_mask
    return text_txt, mask, image_txt


def _patch_fp8_linears(model: torch.nn.Module, scales: dict[str, float]) -> None:
    """Patch fp8 Linear layers to dequantize weights on-the-fly using stored scales.

    The krea2_turbo_fp8_scaled checkpoint stores weights as float8_e4m3fn with
    separate per-tensor scale factors. nn.Linear doesn't know about scales, so
    we patch each layer's forward to: w_bf16 = w_fp8.to(compute_dtype) * scale.
    This keeps 12GB fp8 in VRAM; only the active layer's bf16 weight is transient.
    """
    import torch.nn.functional as F

    def _make_fwd(m: torch.nn.Linear, scale: torch.Tensor):
        def fwd(x: torch.Tensor) -> torch.Tensor:
            w = m.weight.to(x.dtype) * scale.to(x.dtype)
            return F.linear(x, w, m.bias)
        return fwd

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name in scales:
            s = torch.tensor(scales[name], dtype=torch.float32)
            mod.register_buffer("_fp8_scale", s)
            mod.forward = _make_fwd(mod, mod._fp8_scale)


def load_fp8_scaled_state_dict(checkpoint_path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Load a scaled-FP8 safetensors state dict without retaining scale keys.

    `safetensors.safe_open` lets us iterate keys and pull tensors directly from
    the file mapping. The returned state dict excludes `*.weight_scale` keys so
    strict module loading can stay enabled.
    """
    from safetensors import safe_open

    sd: dict[str, torch.Tensor] = {}
    fp8_scales: dict[str, float] = {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if key.endswith(".weight_scale"):
                fp8_scales[key[: -len(".weight_scale")]] = float(tensor.item())
            else:
                sd[key] = tensor
    return sd, fp8_scales


def preflight_model_load(checkpoint_path: str | Path, quantization: str) -> None:
    """Fail early when the current machine cannot fit the requested load."""
    name = Path(checkpoint_path).name.lower()
    is_turbo_fp8 = quantization == "fp8" and "fp8" in name
    is_raw_or_bf16 = quantization == "bf16" or "raw" in name or "bf16" in name
    if not is_turbo_fp8 and not is_raw_or_bf16:
        return

    ram_total, ram_free = get_ram_gb()
    gpu_name, vram_total, vram_free = get_gpu_info()
    gpu_processes = get_gpu_process_details()
    process_hint = ""
    if gpu_processes:
        formatted = ", ".join(
            f"{proc.get('name', 'process')} pid {proc.get('pid')}"
            for proc in gpu_processes[:5]
        )
        process_hint = f" Other GPU Python/app processes: {formatted}."

    if is_raw_or_bf16:
        if ram_total is not None and ram_total < 48.0:
            raise RuntimeError(
                f"RAW/BF16 variants need at least ~48GB system RAM with this loader; system has {ram_total:.1f}GB. "
                "Use Turbo FP8 here, or run RAW/BF16 on a higher-RAM/offloaded loader."
            )
        if ram_free is not None and ram_free < 32.0:
            raise RuntimeError(
                f"Only {ram_free:.1f}GB system RAM free; RAW/BF16 variants need ~32GB free before loading. "
                "Close other apps or duplicate Krea/ComfyUI servers and retry."
            )
        if vram_total is not None and vram_total + 0.5 < 20.0:
            raise RuntimeError(f"{gpu_name or 'GPU'} has {vram_total:.1f}GB VRAM; RAW/BF16 variants need ~20GB.")
        if vram_free is not None and vram_free + 0.5 < 20.0:
            raise RuntimeError(
                f"Only {vram_free:.1f}GB VRAM free; RAW/BF16 variants need ~20GB free before loading."
                f"{process_hint}"
            )
        return

    if vram_total is not None and vram_total + 0.5 < 13.0:
        raise RuntimeError(f"{gpu_name or 'GPU'} has {vram_total:.1f}GB VRAM; Turbo FP8 needs ~13GB.")
    if vram_free is not None and vram_free + 0.5 < 13.0:
        raise RuntimeError(
            f"Only {vram_free:.1f}GB VRAM free; Turbo FP8 needs ~13GB free before loading."
            f"{process_hint}"
        )
    if ram_free is not None and ram_free < 10.0:
        raise RuntimeError(
            f"Only {ram_free:.1f}GB system RAM free; this loader needs ~10GB free for Turbo FP8. "
            "Close other apps or duplicate Krea/ComfyUI servers and retry."
        )
    if ram_total is not None and ram_total < 16.0:
        raise RuntimeError(f"System has {ram_total:.1f}GB RAM; Turbo FP8 needs at least ~16GB total.")


@contextlib.contextmanager
def _no_random_init():
    """Suppress torch.nn.init.* during model construction to prevent RAM spike."""
    import torch.nn.init as _init
    noop = lambda *a, **kw: None
    orig = {n: getattr(_init, n) for n in dir(_init)
            if callable(getattr(_init, n)) and not n.startswith("_")}
    for n in orig:
        setattr(_init, n, noop)
    try:
        yield
    finally:
        for n, fn in orig.items():
            setattr(_init, n, fn)


class LoadWatchdog:
    """Background RAM monitor during model load.

    Sets `aborted` when free RAM is critically low; the loader checks this
    between stages and raises so the load fails GRACEFULLY (model stays
    unloaded, server stays up). It does NOT kill the process — a transient RAM
    dip must never take down the whole studio.
    """
    WARN_GB = 2.0
    ABORT_GB = 0.6

    def __init__(self):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="LoadWatchdog")
        self._warned = False
        self.aborted = False
        self.min_seen = None

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def check(self):
        """Raise if RAM went critically low during loading (call between stages)."""
        if self.aborted:
            raise RuntimeError(
                "Model load aborted: system RAM critically low. Close other apps "
                "(or WSL/Docker) and retry from the System tab."
            )

    def _run(self):
        while not self._stop.wait(0.5):
            _, avail = get_ram_gb()
            if avail is None:
                continue
            self.min_seen = avail if self.min_seen is None else min(self.min_seen, avail)
            if avail < self.ABORT_GB:
                logger.critical(f"RAM critically low ({avail:.2f}GB free). Aborting load (server stays up).")
                self.aborted = True
            elif not self._warned and avail < self.WARN_GB:
                self._warned = True
                logger.warning(f"RAM low ({avail:.2f}GB free).")


def _b64_to_pil(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _pil_to_tensor(img: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    """RGB PIL → (1, 3, H, W) in [-1, 1]."""
    import numpy as np
    arr = (np.array(img.convert("RGB")).astype("float32") / 127.5) - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def _mask_to_tensor(mask_img: Image.Image, width: int, height: int, mode: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    import numpy as np

    resample = Image.BILINEAR if mode == "outpaint" else Image.NEAREST
    mask = mask_img.convert("L").resize((width // 8, height // 8), resample)
    mask_arr = torch.from_numpy(
        np.array(mask).astype("float32") / 255.0
    ).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
    return mask_arr


def _prepare_native_edit_mask(mask_img: Image.Image, size: tuple[int, int], mode: str) -> Image.Image:
    from PIL import ImageFilter

    mask = mask_img.convert("L").resize(size, Image.BILINEAR)
    if mode == "outpaint":
        return mask.filter(ImageFilter.GaussianBlur(radius=2))
    mask = mask.filter(ImageFilter.MaxFilter(11))
    return mask.filter(ImageFilter.GaussianBlur(radius=8))


def _outpaint_stitch(images: list[Image.Image], init_img: Image.Image, mask_img: Image.Image) -> list[Image.Image]:
    """Composite generated outpaint regions back over the prepared canvas.

    ComfyUI-style outpaint flows preserve unmasked pixels and blend only through
    the mask edge. This keeps the source image stable and hides seam drift from
    VAE decode/re-encode.
    """
    from PIL import ImageFilter

    base = init_img.convert("RGB")
    alpha = mask_img.convert("L").resize(base.size, Image.BILINEAR)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=2))
    return [
        Image.composite(img.convert("RGB").resize(base.size, Image.LANCZOS), base, alpha)
        for img in images
    ]


def _outpaint_seam_mask(mask_img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Build a seam-only mask from the feather band of an outpaint mask."""
    import numpy as np
    from PIL import ImageFilter

    mask = mask_img.convert("L").resize(size, Image.BILINEAR)
    arr = np.array(mask)
    band = ((arr > 8) & (arr < 247)).astype("uint8") * 255
    seam = Image.fromarray(band, mode="L")
    seam = seam.filter(ImageFilter.MaxFilter(15))
    seam = seam.filter(ImageFilter.GaussianBlur(radius=6))
    return seam


def _build_bbox_prompt(prompt: str, bboxes: list) -> str:
    """Inject JSON bbox spec into prompt."""
    if not bboxes:
        return prompt
    import json
    regions = [{"label": b.label, "bbox": b.bbox} for b in bboxes]
    spec = json.dumps({"description": prompt, "regions": regions}, ensure_ascii=False)
    return spec


def _resolve_native_sampler(req) -> str:
    native_sampler = getattr(req, "sampler", "euler_flow")
    inpaint_method = getattr(req, "inpaint_method", "native")
    if inpaint_method == "lanpaint_experimental":
        if req.mode != "inpaint":
            raise ValueError("lanpaint_experimental is only available for inpaint mode")
        if not req.init_image_b64 or not req.mask_b64:
            raise ValueError("lanpaint_experimental requires init_image_b64 and mask_b64")
        return "lanpaint_experimental"
    return native_sampler


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Krea2Pipeline:
    CONDITIONING_CACHE_MAX = 8

    def __init__(self):
        self.mmdit = None
        self.ae = None
        self.encoder = None
        self._loaded_checkpoint: Optional[str] = None
        self._loaded_quant: Optional[str] = None
        self._text_encoder_source: Optional[dict] = None
        self._conditioning_cache: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor | None]] = OrderedDict()
        self._lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16

    def is_loaded(self) -> bool:
        return self.mmdit is not None

    def unload(self):
        self.mmdit = None
        self.ae = None
        self.encoder = None
        self._conditioning_cache.clear()
        self._loaded_checkpoint = None
        self._loaded_quant = None
        self._text_encoder_source = None
        if not hasattr(self, "_last_load_error"):
            self._last_load_error = None
        if not hasattr(self, "_loading"):
            self._loading = False
        clear_cuda_cache()

    def release_transient_memory(self, *, clear_conditioning_cache: bool = True) -> dict:
        if self.encoder is not None:
            self.encoder.cpu()
        cleared = len(self._conditioning_cache) if clear_conditioning_cache else 0
        if clear_conditioning_cache:
            self._conditioning_cache.clear()
        clear_cuda_cache()
        return {
            "released": True,
            "encoder_loaded": self.encoder is not None,
            "cleared_conditioning_entries": cleared,
            "memory": mem_snapshot(),
        }

    def memory_status(self) -> dict:
        return {
            "loaded": self.is_loaded(),
            "components": {
                "dit": self.mmdit is not None,
                "vae": self.ae is not None,
                "encoder": self.encoder is not None,
            },
            "conditioning_cache_size": len(self._conditioning_cache),
            "memory": mem_snapshot(),
        }

    def load(self, checkpoint_path: str, quantization: str = "bf16") -> None:
        backend_dir = str(Path(__file__).parent)
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from krea2.autoencoder import QwenAutoencoder
        from krea2.encoder import Qwen3VLConditioner

        try:
            from krea2.mmdit import SingleStreamDiT, SingleMMDiTConfig
        except ImportError:
            raise RuntimeError(
                "krea2/mmdit.py not found. Run install.bat to download it."
            )

        from safetensors.torch import load_file, load_model

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        preflight_model_load(checkpoint_path, quantization)
        checkpoint_size_gb = Path(checkpoint_path).stat().st_size / (1024 ** 3)
        total_ram_gb, _ = get_ram_gb()
        if quantization == "bf16" and checkpoint_size_gb > 20 and total_ram_gb is not None and total_ram_gb < 48:
            raise RuntimeError(
                f"{Path(checkpoint_path).name} is a {checkpoint_size_gb:.1f}GB single-file BF16 checkpoint. "
                "This loader needs at least 48GB system RAM for BF16/RAW variants. "
                "Use Turbo FP8 here, or run BF16/RAW on a higher-RAM machine/offloaded loader."
            )

        self._loading = True
        with self._lock:
            self.unload()

            watchdog = LoadWatchdog()
            watchdog.start()
            mmdit = ae = encoder = None
            try:
                logger.info(f"Loading {checkpoint_path} [{quantization}] ...")

                # 1. DiT on meta device → assign=True avoids double RAM
                with _no_random_init():
                    with torch.device("meta"):
                        mmdit = SingleStreamDiT(SingleMMDiTConfig(
                            features=6144, tdim=256, txtdim=2560,
                            heads=48, kvheads=12, multiplier=4,
                            layers=28, patch=2, channels=16,
                            txtlayers=12,
                        ))
                    sd = None
                    fp8_scales: dict[str, float] = {}
                    if quantization == "fp8":
                        sd, fp8_scales = load_fp8_scaled_state_dict(checkpoint_path)
                    # Detect pre-quantized fp8 checkpoint (krea2_turbo_fp8_scaled).
                    # Must scan ALL tensors: the file's first key is a float32
                    # weight_scale, so checking only sd[0] gives a false negative.
                    checkpoint_is_fp8 = bool(sd) and any("float8" in str(v.dtype) for v in sd.values())

                    if checkpoint_is_fp8:
                        mmdit.load_state_dict(sd, strict=True, assign=True)
                        del sd
                        # Patch each fp8 Linear to dequantize on-the-fly:
                        # keeps 12GB fp8 in VRAM, transient bf16 only for active layer
                        _patch_fp8_linears(mmdit, fp8_scales)
                        mmdit = mmdit.to(device=self._device).eval()
                        logger.info(f"DiT loaded (fp8-scaled, {len(fp8_scales)} layers patched). {mem_snapshot()}")
                    elif quantization == "fp8":
                        # bf16 weights + fp8 requested: quantize on CPU FIRST to avoid
                        # 24GB bf16 spike in VRAM on 24GB cards
                        if sd is None:
                            sd = load_file(checkpoint_path, device="cpu")
                        mmdit.load_state_dict(sd, strict=True, assign=True)
                        del sd
                        mmdit = mmdit.to(dtype=torch.bfloat16)
                        try:
                            from torchao.quantization import (
                                quantize_,
                                float8_dynamic_activation_float8_weight,
                            )
                            quantize_(mmdit, float8_dynamic_activation_float8_weight())
                            logger.info("fp8 CPU quantization applied.")
                        except Exception as e:
                            logger.warning(f"fp8 CPU-quantize failed ({e}), will load bf16.")
                        mmdit = mmdit.to(device=self._device).eval()
                        logger.info(f"DiT loaded (fp8). {mem_snapshot()}")
                    else:
                        mmdit = mmdit.to_empty(device="cpu")
                        missing, unexpected = load_model(mmdit, checkpoint_path, strict=True, device="cpu")
                        if missing or unexpected:
                            raise RuntimeError(f"Checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
                        mmdit = mmdit.to(device=self._device, dtype=self._dtype).eval()
                        logger.info(f"DiT loaded. {mem_snapshot()}")

                # 3. VAE
                watchdog.check()
                ae = QwenAutoencoder().to(device=self._device, dtype=self._dtype).eval()
                logger.info(f"VAE loaded. {mem_snapshot()}")

                # 4. Encoder — kept on CPU, moved to GPU only during encoding.
                # Free anything reclaimable before the ~8GB encoder load.
                watchdog.check()
                gc.collect()
                encoder = Qwen3VLConditioner(str(support_model_path("qwen3_vl"))).eval()
                self._text_encoder_source = getattr(encoder, "source", None)
                logger.info(f"Encoder loaded. {mem_snapshot()}")

            except Exception as e:
                watchdog.stop()
                self._last_load_error = str(e)
                # Drop partially-loaded locals first — they hold GPU tensors not yet
                # assigned to self, so empty_cache() can't free them otherwise.
                # (del locals()[name] is a no-op on real function locals in CPython.)
                del mmdit, ae, encoder
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.unload()
                raise
            finally:
                watchdog.stop()
                self._loading = False

            self.mmdit = mmdit
            self.ae = ae
            self.encoder = encoder
            self._loaded_checkpoint = checkpoint_path
            self._loaded_quant = quantization
            self._text_encoder_source = getattr(encoder, "source", None)
            self._last_load_error = None
            logger.info(f"Model ready. {mem_snapshot()}")

    def _get_conditioning_cache(self, key: tuple) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        hit = self._conditioning_cache.get(key)
        if hit is None:
            return None
        self._conditioning_cache.move_to_end(key)
        txt, txtmask = hit
        return txt.clone(), txtmask.clone() if txtmask is not None else None

    def _put_conditioning_cache(
        self,
        key: tuple,
        txt: torch.Tensor,
        txtmask: torch.Tensor | None,
    ) -> None:
        self._conditioning_cache[key] = (
            txt.detach().cpu().clone(),
            txtmask.detach().cpu().clone() if txtmask is not None else None,
        )
        self._conditioning_cache.move_to_end(key)
        while len(self._conditioning_cache) > self.CONDITIONING_CACHE_MAX:
            self._conditioning_cache.popitem(last=False)

    def _conditioning_to_device(
        self,
        cached: tuple[torch.Tensor, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        txt, txtmask = cached
        return (
            txt.to(device=self._device, dtype=self._dtype),
            txtmask.to(device=self._device) if txtmask is not None else None,
        )

    @staticmethod
    def _hash_ref_images(ref_b64s: list[str]) -> tuple[str, ...]:
        return tuple(hashlib.sha256(ref.encode("utf-8")).hexdigest() for ref in ref_b64s)

    def generate(
        self,
        request,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        *,
        save_outputs: bool = True,
    ) -> list[str]:
        """Generate images. Returns list of base64-encoded PNGs."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. POST /api/load-model first.")

        backend_dir = str(Path(__file__).parent)
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from krea2 import sampling

        with self._lock:
            return self._generate_locked(request, progress_cb, sampling, save_outputs=save_outputs)

    def _generate_locked(self, req, progress_cb, sampling, *, save_outputs: bool = True) -> list[str]:
        normalize_generation_defaults(req)
        native_sampler = _resolve_native_sampler(req)

        # Build prompt (inject bbox JSON + LoRA trigger words)
        prompt = _build_bbox_prompt(req.prompt, req.bboxes)
        prompt = build_trigger_prompt(prompt, req.loras or [])

        # Moodboard preset: prepend the mood's keywords, add its avoids to negative.
        negative_prompt = req.negative_prompt
        mood_id = getattr(req, "mood", "")
        if mood_id:
            from moods import apply_mood
            prompt, negative_prompt = apply_mood(prompt, negative_prompt, mood_id)

        catalog_context = moodboard_generation_context(
            list(getattr(req, "moodboard_ids", []) or []),
            moodboard_uuids=list(getattr(req, "moodboard_uuids", []) or []),
        )
        if catalog_context["uuids"]:
            req.moodboard_uuids = catalog_context["uuids"]
        if catalog_context["style_text"]:
            prompt = f"{prompt}\n\n{catalog_context['style_text']}" if prompt.strip() else catalog_context["style_text"]

        # Encode text. Conditioning is cheap to reuse; moving the 4B Qwen encoder
        # CPU→GPU is not, so cache final CPU tensors by prompt/reference settings.
        style_refs = resolve_style_references(req)
        style_ref_b64s = [ref["image_b64"] for ref in style_refs]
        style_ref_settings = tuple(
            (self._hash_ref_images([ref["image_b64"]])[0], float(ref["strength"]), ref["role"], ref["token_size"])
            for ref in style_refs
        )
        ref_b64s = [
            b64
            for b64 in list(getattr(req, "moodboard_images", []) or []) + style_ref_b64s
            if b64
        ]
        ref_hashes = self._hash_ref_images(ref_b64s)
        weights = parse_weights(req.rebalance_weights) if req.use_rebalance else None
        mult = req.rebalance_multiplier
        if req.use_rebalance and (mood_id or ref_b64s or catalog_context["items"]):
            mult = mult * (0.5 + float(getattr(req, "moodboard_strength", 0.5)))
        edit_rebalance_enabled = (
            bool(getattr(req, "edit_rebalance_enabled", True))
            and str(getattr(req, "mode", "")) in {"redraw", "img2img", "inpaint", "outpaint"}
            and bool(ref_b64s)
        )
        edit_rebalance_profile = str(getattr(req, "edit_rebalance_profile", "conservative") or "conservative")

        positive_key = (
            "positive",
            prompt,
            req.num_images,
            ref_hashes,
            bool(req.use_rebalance),
            req.rebalance_weights,
            float(mult),
            style_ref_settings,
            bool(edit_rebalance_enabled),
            edit_rebalance_profile,
        )
        negative_key = (
            "negative",
            negative_prompt,
            req.num_images,
        ) if req.cfg > 0 else None

        cached_positive = self._get_conditioning_cache(positive_key)
        cached_negative = self._get_conditioning_cache(negative_key) if negative_key else None
        txt = txtmask = neg_txt = neg_txtmask = None

        need_encoder = cached_positive is None or (negative_key is not None and cached_negative is None)
        if need_encoder:
            self.encoder.to(self._device)
            try:
                if cached_positive is None:
                    if ref_b64s:
                        ref_images = [_b64_to_pil(b64).convert("RGB") for b64 in ref_b64s]
                        token_order = {"low": 0, "normal": 1, "high": 2, "max": 3}
                        token_size = max(
                            (ref.get("token_size", "normal") for ref in style_refs),
                            key=lambda value: token_order.get(value, 1),
                            default="normal",
                        )
                        if edit_rebalance_enabled:
                            text_txt, text_mask = self.encoder([prompt] * req.num_images)
                            image_txt, image_mask = self.encoder.forward_with_images(
                                [prompt] * req.num_images, images=ref_images, token_size=token_size
                            )
                            text_txt, txtmask, image_txt = _align_conditioning_pair(
                                text_txt, text_mask, image_txt, image_mask
                            )
                            txt = blend_split_conditioning(
                                text_txt,
                                image_txt,
                                position=0.5,
                                profile=edit_rebalance_profile,
                            )
                        else:
                            txt, txtmask = self.encoder.forward_with_images(
                                [prompt] * req.num_images, images=ref_images, token_size=token_size
                            )
                    else:
                        txt, txtmask = self.encoder([prompt] * req.num_images)

                    if req.use_rebalance and weights is not None:
                        txt = rebalance(txt, multiplier=mult, layer_weights=weights)
                    self._put_conditioning_cache(positive_key, txt, txtmask)

                if negative_key is not None and cached_negative is None:
                    neg_txt, neg_txtmask = self.encoder([negative_prompt] * req.num_images)
                    self._put_conditioning_cache(negative_key, neg_txt, neg_txtmask)
            finally:
                # Offload encoder to CPU before DiT forward pass.
                self.encoder.cpu()
                clear_cuda_cache()

        if cached_positive is not None:
            txt, txtmask = self._conditioning_to_device(cached_positive)
        if negative_key is not None and cached_negative is not None:
            neg_txt, neg_txtmask = self._conditioning_to_device(cached_negative)
        if txt is None or txtmask is None:
            cached_positive = self._get_conditioning_cache(positive_key)
            if cached_positive is None:
                raise RuntimeError("Conditioning cache failed to store positive prompt.")
            txt, txtmask = self._conditioning_to_device(cached_positive)
        if negative_key is not None and neg_txt is None:
            cached_negative = self._get_conditioning_cache(negative_key)
            if cached_negative is None:
                raise RuntimeError("Conditioning cache failed to store negative prompt.")
            neg_txt, neg_txtmask = self._conditioning_to_device(cached_negative)

        logger.info(f"Encoded. {mem_snapshot()}")

        # Align target dims to the patch grid (COMPRESSION*PATCH = 16) so the
        # init latent / mask match the spatial dims the sampler rounds to.
        _align = 8 * 2
        gen_w = ((req.width + _align - 1) // _align) * _align
        gen_h = ((req.height + _align - 1) // _align) * _align

        # Encode init image if needed
        init_latent = init_latent_clean = mask_tensor = None
        init_img_for_stitch = mask_img_for_stitch = None
        if req.mode in ("img2img", "inpaint", "outpaint") and req.init_image_b64:
            init_img = _b64_to_pil(req.init_image_b64).resize(
                (gen_w, gen_h), Image.LANCZOS
            )
            if req.mode == "outpaint":
                init_img_for_stitch = init_img.copy()
            px = _pil_to_tensor(init_img, self._device, self._dtype)
            init_latent = self.ae.encode(px)
            init_latent_clean = init_latent.clone()

        if req.mode in ("inpaint", "outpaint") and req.mask_b64:
            raw_mask_img = _b64_to_pil(req.mask_b64)
            if req.mode == "outpaint":
                mask_img_for_stitch = raw_mask_img.copy()
            edit_mask_img = _prepare_native_edit_mask(raw_mask_img, (gen_w, gen_h), req.mode)
            mask_tensor = _mask_to_tensor(
                edit_mask_img,
                gen_w,
                gen_h,
                req.mode,
                self._device,
                self._dtype,
            )

        # Resolve seed
        seed = req.seed if req.seed >= 0 else int(torch.randint(0, 2**31, (1,)).item())
        txt = apply_seed_variance(
            txt,
            seed=seed,
            preset=str(getattr(req, "seed_variance_preset", "off")),
            strength=(
                float(getattr(req, "seed_variance_strength", 0.0))
                if str(getattr(req, "seed_variance_preset", "off")) == "custom"
                else None
            ),
            protection=str(getattr(req, "seed_variance_protection", "first_half")),
        )

        # Resolve mu (the ModelSamplingFlux shift). 0 or None = auto-resolve:
        # turbo pins 1.15, RAW uses resolution-adaptive (None).
        mu = req.mu
        if mu is not None and mu <= 0:
            mu = None
        if mu is None and self._loaded_checkpoint:
            cp = self._loaded_checkpoint.lower()
            mu = 1.15 if "turbo" in cp else None

        # Attach requested LoRAs as additive low-rank paths (resets any prior set).
        # Incompatible LoRAs are skipped; reports flow back to the UI.
        lora_reports = apply_loras(self.mmdit, req.loras or [], device=self._device)

        with krea_enhancer_context(
            self.mmdit,
            enabled=bool(getattr(req, "krea_enhancer_enabled", False)),
            strength=float(getattr(req, "krea_enhancer_strength", 1.0)),
        ):
            images = sampling.sample(
                model=self.mmdit,
                ae=self.ae,
                encoder=None,
                prompts=None,
                txt=txt,
                txtmask=txtmask,
                negative_txt=neg_txt,
                negative_txtmask=neg_txtmask,
                device=self._device,
                dtype=self._dtype,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance=req.cfg,
                seed=seed,
                mu=mu,
                y1=req.y1,
                y2=req.y2,
                batch_size=req.num_images,
                init_latent=init_latent,
                denoise=req.denoise,
                mask=mask_tensor,
                init_latent_clean=init_latent_clean,
                differential_mask=req.mode == "outpaint",
                sampler=native_sampler,
                lanpaint_inner_steps=getattr(req, "lanpaint_inner_steps", 3),
                lanpaint_strength=getattr(req, "lanpaint_strength", 1.0),
                progress_cb=progress_cb,
            )

        if req.mode == "outpaint" and init_img_for_stitch is not None and mask_img_for_stitch is not None:
            images = _outpaint_stitch(images, init_img_for_stitch, mask_img_for_stitch)
            try:
                seam_mask = _outpaint_seam_mask(mask_img_for_stitch, init_img_for_stitch.size)
                if seam_mask.getbbox():
                    px = torch.cat(
                        [_pil_to_tensor(im, self._device, self._dtype) for im in images],
                        dim=0,
                    )
                    seam_latent = self.ae.encode(px)
                    seam_mask_tensor = _mask_to_tensor(
                        seam_mask,
                        gen_w,
                        gen_h,
                        "outpaint",
                        self._device,
                        self._dtype,
                    )
                    images = sampling.sample(
                        model=self.mmdit,
                        ae=self.ae,
                        encoder=None,
                        prompts=None,
                        txt=txt,
                        txtmask=txtmask,
                        negative_txt=neg_txt,
                        negative_txtmask=neg_txtmask,
                        device=self._device,
                        dtype=self._dtype,
                        width=req.width,
                        height=req.height,
                        steps=max(4, min(6, int(req.steps))),
                        guidance=req.cfg,
                        seed=seed + 1,
                        mu=mu,
                        y1=req.y1,
                        y2=req.y2,
                        batch_size=len(images),
                        init_latent=seam_latent,
                        denoise=0.35,
                        mask=seam_mask_tensor,
                        differential_mask=True,
                        sampler=getattr(req, "sampler", "euler_flow"),
                        progress_cb=None,
                    )
                    images = _outpaint_stitch(images, init_img_for_stitch, mask_img_for_stitch)
                    logger.info("Outpaint seam-fix pass done.")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Outpaint seam-fix skipped ({e})")

            try:
                px = torch.cat(
                    [_pil_to_tensor(im, self._device, self._dtype) for im in images],
                    dim=0,
                )
                harmonize_latent = self.ae.encode(px)
                images = sampling.sample(
                    model=self.mmdit,
                    ae=self.ae,
                    encoder=None,
                    prompts=None,
                    txt=txt,
                    txtmask=txtmask,
                    negative_txt=neg_txt,
                    negative_txtmask=neg_txtmask,
                    device=self._device,
                    dtype=self._dtype,
                    width=req.width,
                    height=req.height,
                    steps=max(4, min(6, int(req.steps))),
                    guidance=req.cfg,
                    seed=seed + 2,
                    mu=mu,
                    y1=req.y1,
                    y2=req.y2,
                    batch_size=len(images),
                    init_latent=harmonize_latent,
                    denoise=0.12,
                    sampler=getattr(req, "sampler", "euler_flow"),
                    progress_cb=None,
                )
                logger.info("Outpaint harmonization pass done.")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Outpaint harmonization skipped ({e})")

        # Detail refiner: optional second low-denoise self-pass (skip for inpaint,
        # which must not re-touch the kept region). Re-encodes the batch and runs
        # img2img at refine_denoise so fine detail is sharpened without drift.
        if getattr(req, "refine", False) and req.mode not in ("inpaint", "outpaint"):
            try:
                px = torch.cat(
                    [_pil_to_tensor(im, self._device, self._dtype) for im in images], dim=0
                )
                rlat = self.ae.encode(px)
                images = sampling.sample(
                    model=self.mmdit, ae=self.ae, encoder=None, prompts=None,
                    txt=txt, txtmask=txtmask,
                    negative_txt=neg_txt, negative_txtmask=neg_txtmask,
                    device=self._device, dtype=self._dtype,
                    width=req.width, height=req.height,
                    steps=int(req.refine_steps), guidance=req.cfg, seed=seed, mu=mu,
                    y1=req.y1, y2=req.y2, batch_size=len(images),
                    init_latent=rlat, denoise=float(req.refine_denoise),
                    sampler=getattr(req, "sampler", "euler_flow"),
                    progress_cb=progress_cb,
                )
                logger.info(f"Detail refine pass done (denoise={req.refine_denoise}).")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Refine pass skipped ({e})")

        # Save + return base64 + filenames. Realtime previews pass
        # save_outputs=False so they never spam the Gallery output directory.
        metadata = [
            build_generation_metadata(req, base_seed=seed, image_index=i, filename="")
            for i in range(len(images))
        ]
        results, filenames = encode_images(images, OUTPUTS_DIR, save_outputs=save_outputs, metadata=metadata)
        metadata = [
            {**item, "filename": filenames[i] if i < len(filenames) else item.get("filename", "")}
            for i, item in enumerate(metadata)
        ]
        return results, seed, filenames, lora_reports, metadata


# Singleton used by FastAPI
pipeline = Krea2Pipeline()
