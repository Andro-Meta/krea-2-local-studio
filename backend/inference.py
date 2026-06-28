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

from conditioning import blend_split_conditioning, rebalance, resolve_rebalance_weights
from generation_metadata import build_generation_metadata
from krea_enhancer import krea_enhancer_context
from krea2.block_swap import BlockSwapController, resolve_swap_plan
from krea2.fp8_quant import load_bf16_as_fp8_scaled
from krea2.text_prompt import DEFAULT_EXPRESSION_THINK
from krea2.vae_source import resolve_vae_source
from krea2.reference_image import cap_vision_megapixels, crop_image_to_mask
from krea2.sampler_registry import validate_sampler_configuration
from krea2.lanpaint_sampler import LanPaintSettings
from lora_manager import apply_loras, build_trigger_prompt
from model_profiles import MODEL_PROFILES, apply_profile_defaults
from moodboards_catalog import moodboard_generation_context
from memory_manager import clear_cuda_cache
from regional_scene import build_regional_prompt_text
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
        "raw": {"moodboard_strength": 0.15, "rebalance_multiplier": 1.0},
        "low": {"moodboard_strength": 0.25, "rebalance_multiplier": 1.0},
        "medium": {"moodboard_strength": 0.35, "rebalance_multiplier": 1.0},
        "high": {"moodboard_strength": 0.5, "rebalance_multiplier": 1.15},
    }
    preset = presets.get(creativity, presets["medium"])
    if not _request_field_was_set(req, "moodboard_strength"):
        req.moodboard_strength = preset["moodboard_strength"]
    if not _request_field_was_set(req, "rebalance_multiplier"):
        req.rebalance_multiplier = preset["rebalance_multiplier"]


def _normalize_request_dimensions(req) -> None:
    """Clamp width/height to the patch grid and the model's max resolution.

    Guarantees correctness for any caller (1k/2k presets or custom values):
    dims become multiples of 16 within [256, model max]. Turbo/RAW both allow up
    to 2048 (the documented 2k ceiling). mu policy is unchanged here — Turbo stays
    pinned (no resolution scaling) and RAW stays resolution-adaptive.
    """
    from resolution import normalize_dimensions

    profile_id = str(getattr(req, "model_profile", "") or "")
    max_edge = 2048
    profile = MODEL_PROFILES.get(profile_id)
    if profile is not None:
        max_edge = int(getattr(profile, "max_resolution", 2048))
    elif str(getattr(req, "checkpoint", "") or "").lower() == "raw":
        max_edge = int(MODEL_PROFILES["krea_raw"].max_resolution)
    w, h = normalize_dimensions(int(getattr(req, "width", 1024)), int(getattr(req, "height", 1024)), max_edge=max_edge)
    req.width, req.height = w, h


def normalize_generation_defaults(req) -> None:
    """Apply model-family defaults for callers that bypass the frontend presets."""
    _normalize_request_dimensions(req)
    if _request_field_was_set(req, "model_profile") and str(getattr(req, "model_profile", "") or ""):
        apply_profile_defaults(req)
        _apply_creativity_defaults(req)
        return

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
    fusion_mode = str(getattr(req, "style_fusion_mode", "semantic_fusion") or "semantic_fusion")
    for item in list(getattr(req, "style_references", []) or []):
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        image_b64 = str(item.get("image_b64", "") or "")
        if not image_b64:
            continue
        role = str(item.get("role", "style") or "style")
        if role == "target":
            continue
        token_size = str(item.get("token_size", "normal") or "normal")
        strength = float(item.get("strength", 1.0))
        if role == "layout" or fusion_mode == "preserve_structure":
            token_size = "low" if token_size in {"low", "normal"} else "normal"
            strength = min(strength, 0.65)
        if fusion_mode == "style_only" and role not in {"style", "mood", "texture"}:
            continue
        refs.append({
            "image_b64": image_b64,
            "strength": strength,
            "role": role,
            "token_size": token_size,
            "mask_b64": str(item.get("mask_b64", "") or ""),
            "mask_padding": int(item.get("mask_padding", 0) or 0),
            "vision_megapixels": item.get("vision_megapixels"),
            "system_prompt": str(item.get("system_prompt", "") or ""),
            "vision_position": str(item.get("vision_position", "before_prompt") or "before_prompt"),
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
                "mask_b64": "",
                "mask_padding": 0,
                "vision_megapixels": None,
                "system_prompt": "",
                "vision_position": "before_prompt",
            })
        if len(refs) >= limit:
            break
    return refs[:limit]


def _prepare_style_reference_image(ref: dict) -> Image.Image:
    image = _b64_to_pil(str(ref["image_b64"])).convert("RGB")
    mask_b64 = str(ref.get("mask_b64", "") or "")
    if mask_b64:
        mask = _b64_to_pil(mask_b64).convert("L")
        image = crop_image_to_mask(image, mask, padding=int(ref.get("mask_padding", 0) or 0))
    megapixels = ref.get("vision_megapixels")
    if megapixels is not None:
        image = cap_vision_megapixels(image, float(megapixels))
    return image


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


def _checkpoint_has_fp8_weights(checkpoint_path: str | Path) -> bool:
    """Cheaply tell whether a safetensors file is already stored in float8.

    Reads only the header dtype strings (no tensor materialization) so a bf16
    file requested as fp8 is not loaded into RAM just to detect its dtype.
    """
    from safetensors import safe_open

    try:
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.endswith(".weight_scale"):
                    return True
                try:
                    dtype = str(handle.get_slice(key).get_dtype()).upper()
                except Exception:
                    continue
                if "F8" in dtype or "FLOAT8" in dtype:
                    return True
    except Exception:
        return False
    return False


def preflight_model_load(
    checkpoint_path: str | Path,
    quantization: str,
    *,
    blocks_to_swap: int = 0,
) -> None:
    """Fail early when the current machine cannot fit the requested load.

    fp8 loads (including dynamic fp8 of a bf16 file) resolve to the ~13GB VRAM
    tier regardless of "raw"/"bf16" in the filename. Block swapping streams the
    last N DiT blocks from RAM, lowering the resident VRAM requirement; we relax
    the VRAM gate accordingly while keeping the RAM gate that the load itself
    needs. A dynamic-fp8 load of a bf16 file is gated on RAM by the file size,
    since the streaming quantizer still reads the bf16 tensors from disk.
    """
    name = Path(checkpoint_path).name.lower()
    is_fp8 = quantization == "fp8"
    file_is_prequant_fp8 = "fp8" in name
    is_bf16 = not is_fp8
    if not is_fp8 and not ("raw" in name or "bf16" in name):
        return

    try:
        checkpoint_size_gb = Path(checkpoint_path).stat().st_size / (1024 ** 3)
    except OSError:
        checkpoint_size_gb = 0.0

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

    # Block swapping removes ~0.4GB resident VRAM per streamed fp8 block.
    swap_vram_relief = max(0, int(blocks_to_swap)) * 0.4

    if is_bf16:
        # Full bf16 residency. Block swap can lower the VRAM need, but the
        # single-file loader still materializes bf16 weights in RAM.
        vram_need = max(10.0, 20.0 - swap_vram_relief)
        if ram_total is not None and ram_total < 48.0:
            raise RuntimeError(
                f"RAW/BF16 (bf16 residency) needs ~48GB system RAM with this loader; system has {ram_total:.1f}GB. "
                "Use fp8 quantization (works on 24GB VRAM), or run bf16 on a higher-RAM machine."
            )
        if ram_free is not None and ram_free < 32.0:
            raise RuntimeError(
                f"Only {ram_free:.1f}GB system RAM free; bf16 residency needs ~32GB free before loading. "
                "Close other apps or switch to fp8 quantization."
            )
        if vram_total is not None and vram_total + 0.5 < vram_need:
            raise RuntimeError(f"{gpu_name or 'GPU'} has {vram_total:.1f}GB VRAM; bf16 residency needs ~{vram_need:.0f}GB.")
        if vram_free is not None and vram_free + 0.5 < vram_need:
            raise RuntimeError(
                f"Only {vram_free:.1f}GB VRAM free; bf16 residency needs ~{vram_need:.0f}GB free before loading."
                f"{process_hint}"
            )
        return

    # fp8 path (pre-quantized or dynamic). Resident VRAM ~13GB, minus swap relief.
    vram_need = max(8.0, 13.0 - swap_vram_relief)
    if vram_total is not None and vram_total + 0.5 < vram_need:
        raise RuntimeError(f"{gpu_name or 'GPU'} has {vram_total:.1f}GB VRAM; fp8 needs ~{vram_need:.0f}GB.")
    if vram_free is not None and vram_free + 0.5 < vram_need:
        raise RuntimeError(
            f"Only {vram_free:.1f}GB VRAM free; fp8 needs ~{vram_need:.0f}GB free before loading."
            f"{process_hint}"
        )

    gpu_present = vram_total is not None
    if file_is_prequant_fp8 or gpu_present:
        # Pre-quantized fp8 reads straight to fp8 (~12GB); dynamic fp8 streams
        # each quantized weight onto the GPU as it is produced. Either way the
        # host-RAM peak is dominated by the ~8GB text encoder, not the file size.
        ram_need_free, ram_need_total = 10.0, 16.0
    else:
        # CPU-only dynamic fp8 cannot stream to VRAM, so it must hold the fp8
        # state in RAM; gate on the file size in that (impractical) case.
        ram_need_free = max(12.0, checkpoint_size_gb * 0.6 + 6.0)
        ram_need_total = max(16.0, checkpoint_size_gb * 0.6 + 10.0)
    if ram_free is not None and ram_free < ram_need_free:
        raise RuntimeError(
            f"Only {ram_free:.1f}GB system RAM free; this fp8 load needs ~{ram_need_free:.0f}GB free. "
            "Close other apps or duplicate Krea/ComfyUI servers and retry."
        )
    if ram_total is not None and ram_total < ram_need_total:
        raise RuntimeError(f"System has {ram_total:.1f}GB RAM; this fp8 load needs at least ~{ram_need_total:.0f}GB total.")


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
    validate_sampler_configuration(
        native_sampler,
        getattr(req, "scheduler", "simple"),
        getattr(req, "model_profile", "") or ("krea_raw" if getattr(req, "checkpoint", "turbo") == "raw" else "krea_turbo"),
    )
    inpaint_method = getattr(req, "inpaint_method", "native")
    if inpaint_method == "lanpaint_experimental":
        if req.mode != "inpaint":
            raise ValueError("lanpaint_experimental is only available for inpaint mode")
        if not req.init_image_b64 or not req.mask_b64:
            raise ValueError("lanpaint_experimental requires init_image_b64 and mask_b64")
        return "lanpaint_experimental"
    return native_sampler


def _lanpaint_settings_from_request(req) -> LanPaintSettings:
    return LanPaintSettings(
        num_steps=int(getattr(req, "lanpaint_inner_steps", 5) if _request_field_was_set(req, "lanpaint_inner_steps") else 5),
        lambda_strength=float(getattr(req, "lanpaint_lambda", 16.0) if _request_field_was_set(req, "lanpaint_lambda") else 16.0),
        step_size=float(getattr(req, "lanpaint_step_size", 0.2) if _request_field_was_set(req, "lanpaint_step_size") else 0.2),
        beta=float(getattr(req, "lanpaint_beta", 1.0) if _request_field_was_set(req, "lanpaint_beta") else 1.0),
        friction=float(getattr(req, "lanpaint_friction", 15.0) if _request_field_was_set(req, "lanpaint_friction") else 15.0),
        early_stop=int(getattr(req, "lanpaint_early_stop", 1) if _request_field_was_set(req, "lanpaint_early_stop") else 1),
        prompt_mode=str(getattr(req, "lanpaint_prompt_mode", "Image First") if _request_field_was_set(req, "lanpaint_prompt_mode") else "Image First"),
        strength=float(getattr(req, "lanpaint_strength", 1.0) if _request_field_was_set(req, "lanpaint_strength") else 1.0),
    )


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
        self._block_swap: Optional[BlockSwapController] = None
        self._blocks_to_swap = 0

    def is_loaded(self) -> bool:
        return self.mmdit is not None

    def unload(self):
        if self._block_swap is not None:
            try:
                # Don't pull offloaded blocks back into VRAM just to discard them.
                self._block_swap.remove(restore_device=False)
            except Exception:
                pass
            self._block_swap = None
        self._blocks_to_swap = 0
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
            "low_vram": {
                "blocks_to_swap": self._blocks_to_swap,
                "block_swap_active": self._block_swap is not None and self._block_swap.active,
                "encoder_offload": True,
            },
            "memory": mem_snapshot(),
        }

    def load(self, checkpoint_path: str, quantization: str = "bf16", *, blocks_to_swap: int = 0) -> None:
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

        from safetensors.torch import load_model

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        blocks_to_swap = max(0, int(blocks_to_swap))
        preflight_model_load(checkpoint_path, quantization, blocks_to_swap=blocks_to_swap)
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

            # Compute dtype: fp16 (with fp16 reduced-precision accumulation) is a
            # fast, high-quality full-precision option when VRAM allows; otherwise
            # bf16 is the compute dtype (fp8 weights dequantize to bf16).
            self._dtype = torch.float16 if quantization == "fp16" else torch.bfloat16
            if quantization == "fp16" and torch.cuda.is_available():
                try:
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                except Exception:
                    pass

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
                    # Detect a pre-quantized fp8 checkpoint cheaply from the
                    # safetensors header (dtype strings only) so a bf16 file
                    # requested as fp8 does NOT get fully materialized in RAM.
                    checkpoint_is_fp8 = quantization == "fp8" and _checkpoint_has_fp8_weights(checkpoint_path)

                    if checkpoint_is_fp8:
                        # Pre-quantized scaled fp8 (e.g. krea2_turbo_fp8_scaled):
                        # weights stay float8 in VRAM, on-the-fly bf16 dequant.
                        sd, fp8_scales = load_fp8_scaled_state_dict(checkpoint_path)
                        mmdit.load_state_dict(sd, strict=True, assign=True)
                        del sd
                        _patch_fp8_linears(mmdit, fp8_scales)
                        mmdit = mmdit.to(device=self._device).eval()
                        logger.info(f"DiT loaded (fp8-scaled, {len(fp8_scales)} layers patched). {mem_snapshot()}")
                    elif quantization == "fp8":
                        # bf16 source + fp8 requested → dynamic scaled fp8.
                        # Stream-quantize tensor-by-tensor straight onto the target
                        # device so the ~12GB of fp8 weights land in VRAM as they
                        # are produced; host RAM peak stays near one transient
                        # bf16 tensor. This is what lets RAW (24GB bf16) load on a
                        # 24GB card without a 24GB host-RAM spike.
                        sd_fp8, dyn_scales = load_bf16_as_fp8_scaled(
                            checkpoint_path, device=self._device, compute_dtype=self._dtype
                        )
                        mmdit.load_state_dict(sd_fp8, strict=True, assign=True)
                        del sd_fp8
                        _patch_fp8_linears(mmdit, dyn_scales)
                        # Moves the freshly-registered fp8 scale buffers onto the
                        # device (weights are already there); cheap no-op for them.
                        mmdit = mmdit.to(device=self._device).eval()
                        logger.info(f"DiT loaded (dynamic fp8, {len(dyn_scales)} layers quantized). {mem_snapshot()}")
                    else:
                        mmdit = mmdit.to_empty(device="cpu")
                        missing, unexpected = load_model(mmdit, checkpoint_path, strict=True, device="cpu")
                        if missing or unexpected:
                            raise RuntimeError(f"Checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
                        mmdit = mmdit.to(device=self._device, dtype=self._dtype).eval()
                        logger.info(f"DiT loaded. {mem_snapshot()}")

                    # Low-VRAM block swapping: stream the last N DiT blocks from
                    # RAM so the resident footprint drops further (CUDA only).
                    self._block_swap = None
                    self._blocks_to_swap = 0
                    if blocks_to_swap > 0 and self._device == "cuda":
                        plan = resolve_swap_plan(total_blocks=len(mmdit.blocks), blocks_to_swap=blocks_to_swap)
                        if plan.swapped_indices:
                            controller = BlockSwapController(
                                mmdit,
                                blocks_to_swap=blocks_to_swap,
                                device=self._device,
                                offload_device="cpu",
                                prefetch=1,
                                pin_memory=True,
                            )
                            controller.install()
                            self._block_swap = controller
                            self._blocks_to_swap = len(plan.swapped_indices)
                            logger.info(
                                f"Block swap active: {self._blocks_to_swap}/{len(mmdit.blocks)} DiT blocks streamed from RAM. {mem_snapshot()}"
                            )

                # 3. VAE (optional experimental override; falls back to stock)
                watchdog.check()
                vae_override = resolve_vae_source(getattr(settings, "krea2_vae_path", "")).get("path") or None
                ae = QwenAutoencoder(vae_override_path=vae_override).to(device=self._device, dtype=self._dtype).eval()
                logger.info(f"VAE loaded ({getattr(ae, 'vae_source', 'stock')}). {mem_snapshot()}")

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
        style_fusion_mode = str(getattr(req, "style_fusion_mode", "semantic_fusion") or "semantic_fusion")
        if style_fusion_mode == "preserve_structure" and str(getattr(req, "mode", "")) in {"redraw", "img2img", "inpaint", "outpaint"}:
            prompt = (
                f"{prompt}\n\nPreserve the source composition, pose, layout, and spatial relationships. "
                "Use references for style and surface detail without replacing the underlying structure."
            )
        if getattr(req, "regional_prompts", None):
            prompt = build_regional_prompt_text(
                prompt,
                [
                    item.model_dump() if hasattr(item, "model_dump") else dict(item)
                    for item in (getattr(req, "regional_prompts", []) or [])
                ],
                base_prompt_strength=float(getattr(req, "regional_base_prompt_strength", 0.3)),
            )

        # Encode text. Conditioning is cheap to reuse; moving the 4B Qwen encoder
        # CPU→GPU is not, so cache final CPU tensors by prompt/reference settings.
        style_refs = resolve_style_references(req)
        style_ref_b64s = [ref["image_b64"] for ref in style_refs]
        style_ref_settings = tuple(
            (
                self._hash_ref_images([ref["image_b64"]])[0],
                self._hash_ref_images([ref["mask_b64"]])[0] if ref.get("mask_b64") else "",
                int(ref.get("mask_padding", 0) or 0),
                ref.get("vision_megapixels"),
                float(ref["strength"]),
                ref["role"],
                ref["token_size"],
                ref.get("system_prompt", ""),
                ref.get("vision_position", "before_prompt"),
            )
            for ref in style_refs
        )
        ref_b64s = [
            b64
            for b64 in list(getattr(req, "moodboard_images", []) or []) + style_ref_b64s
            if b64
        ]
        ref_hashes = self._hash_ref_images(ref_b64s)
        # <think> expression steering (positive prompt only, in-distribution).
        think_text = None
        if bool(getattr(req, "think_steering_enabled", False)):
            think_text = str(getattr(req, "think_text", "") or "").strip() or DEFAULT_EXPRESSION_THINK
        rebalance_preset = str(getattr(req, "rebalance_preset", "balanced") or "balanced")
        rebalance_mode = str(getattr(req, "rebalance_mode", "rms_renorm") or "rms_renorm")
        rebalance_renormalize = bool(getattr(req, "rebalance_renormalize", True))
        weights = resolve_rebalance_weights(rebalance_preset, req.rebalance_weights) if req.use_rebalance else None
        mult = req.rebalance_multiplier
        if req.use_rebalance and (mood_id or ref_b64s or catalog_context["items"]):
            mult = mult * (0.5 + float(getattr(req, "moodboard_strength", 0.5)))
        edit_rebalance_enabled = (
            bool(getattr(req, "edit_rebalance_enabled", True))
            and str(getattr(req, "mode", "")) in {"redraw", "img2img", "inpaint", "outpaint"}
            and bool(ref_b64s)
        )
        edit_rebalance_profile = str(getattr(req, "edit_rebalance_profile", "conservative") or "conservative")
        requested_conditioning_mode = str(getattr(req, "conditioning_mode", "auto") or "auto")
        edit_plus_active = (
            requested_conditioning_mode == "qwen_image_edit_plus"
            or (
                requested_conditioning_mode == "auto"
                and str(getattr(req, "mode", "")) in {"redraw", "img2img", "inpaint", "outpaint"}
                and bool(ref_b64s)
            )
        )

        positive_key = (
            "positive",
            prompt,
            req.num_images,
            ref_hashes,
            bool(req.use_rebalance),
            rebalance_mode,
            rebalance_preset,
            bool(rebalance_renormalize),
            req.rebalance_weights,
            float(mult),
            style_ref_settings,
            bool(edit_rebalance_enabled),
            edit_rebalance_profile,
            "qwen_image_edit_plus" if edit_plus_active else "qwen_reference",
            style_fusion_mode,
            think_text or "",
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
                        moodboard_images = [
                            _b64_to_pil(b64).convert("RGB")
                            for b64 in list(getattr(req, "moodboard_images", []) or [])
                            if b64
                        ]
                        ref_images = [*moodboard_images, *[_prepare_style_reference_image(ref) for ref in style_refs]]
                        token_order = {"low": 0, "normal": 1, "high": 2, "max": 3}
                        token_size = max(
                            (ref.get("token_size", "normal") for ref in style_refs),
                            key=lambda value: token_order.get(value, 1),
                            default="normal",
                        )
                        system_prompt = next((ref.get("system_prompt", "") for ref in style_refs if ref.get("system_prompt")), None)
                        vision_position = (
                            "after_prompt"
                            if any(ref.get("vision_position") == "after_prompt" for ref in style_refs)
                            else "before_prompt"
                        )
                        if edit_rebalance_enabled:
                            text_txt, text_mask = self.encoder([prompt] * req.num_images, think=think_text)
                            if edit_plus_active:
                                image_txt, image_mask = self.encoder.forward_image_edit_plus(
                                    [prompt] * req.num_images,
                                    images=ref_images[:3],
                                    negative_prompts=[negative_prompt] * req.num_images,
                                )
                            else:
                                image_txt, image_mask = self.encoder.forward_with_images(
                                    [prompt] * req.num_images,
                                    images=ref_images,
                                    token_size=token_size,
                                    system_prompt=system_prompt,
                                    vision_position=vision_position,
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
                            if edit_plus_active:
                                txt, txtmask = self.encoder.forward_image_edit_plus(
                                    [prompt] * req.num_images,
                                    images=ref_images[:3],
                                    negative_prompts=[negative_prompt] * req.num_images,
                                )
                            else:
                                txt, txtmask = self.encoder.forward_with_images(
                                    [prompt] * req.num_images,
                                    images=ref_images,
                                    token_size=token_size,
                                    system_prompt=system_prompt,
                                    vision_position=vision_position,
                                )
                    else:
                        txt, txtmask = self.encoder([prompt] * req.num_images, think=think_text)

                    if req.use_rebalance and weights is not None and not edit_rebalance_enabled:
                        txt = rebalance(
                            txt,
                            multiplier=mult,
                            layer_weights=weights,
                            preset=rebalance_preset,
                            weights_str=req.rebalance_weights,
                            mode=rebalance_mode,
                            renormalize=rebalance_renormalize,
                        )
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
            direction=str(getattr(req, "seed_variance_direction", "none")),
            fade_curve=str(getattr(req, "seed_variance_fade_curve", "linear")),
            injection_start=float(getattr(req, "seed_variance_injection_start", 0.0)),
            injection_end=float(getattr(req, "seed_variance_injection_end", 1.0)),
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
            variant=str(getattr(req, "krea_enhancer_variant", "off")),
            delta_cap=float(getattr(req, "krea_enhancer_delta_cap", 0.75)),
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
                lanpaint_settings=_lanpaint_settings_from_request(req),
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
