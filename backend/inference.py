"""Krea 2 pipeline manager.

Handles model loading (with LoadWatchdog + encoder offload) and generation
for txt2img, img2img, and inpaint modes.
"""
from __future__ import annotations

import base64
import contextlib
import gc
import io
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

# Krea2 mmdit.py decorates posemb.forward with @torch.compile(fullgraph=True).
# Windows has no triton, so inductor would hard-fail at first forward. Disable
# dynamo before torch loads it → runs eager (correct, just unoptimized).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from PIL import Image

from conditioning import parse_weights, rebalance
from log_setup import flush_all
from lora_manager import apply_loras, build_trigger_prompt
from settings import OUTPUTS_DIR, settings
from support_models import support_model_path
from system_check import get_ram_gb, mem_snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def _build_bbox_prompt(prompt: str, bboxes: list) -> str:
    """Inject JSON bbox spec into prompt."""
    if not bboxes:
        return prompt
    import json
    regions = [{"label": b.label, "bbox": b.bbox} for b in bboxes]
    spec = json.dumps({"description": prompt, "regions": regions}, ensure_ascii=False)
    return spec


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Krea2Pipeline:
    def __init__(self):
        self.mmdit = None
        self.ae = None
        self.encoder = None
        self._loaded_checkpoint: Optional[str] = None
        self._loaded_quant: Optional[str] = None
        self._lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16

    def is_loaded(self) -> bool:
        return self.mmdit is not None

    def unload(self):
        self.mmdit = None
        self.ae = None
        self.encoder = None
        self._loaded_checkpoint = None
        self._loaded_quant = None
        if not hasattr(self, "_last_load_error"):
            self._last_load_error = None
        if not hasattr(self, "_loading"):
            self._loading = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        from safetensors.torch import load_file

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
                    sd = load_file(checkpoint_path, device="cpu")
                    # Detect pre-quantized fp8 checkpoint (krea2_turbo_fp8_scaled).
                    # Must scan ALL tensors: the file's first key is a float32
                    # weight_scale, so checking only sd[0] gives a false negative.
                    checkpoint_is_fp8 = any(
                        "float8" in str(v.dtype) for v in sd.values()
                    )

                    if checkpoint_is_fp8:
                        # Extract per-tensor scales before load (strict=True would reject them)
                        fp8_scales: dict[str, float] = {}
                        for k in list(sd.keys()):
                            if k.endswith(".weight_scale"):
                                fp8_scales[k[: -len(".weight_scale")]] = sd.pop(k).item()
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
                        mmdit.load_state_dict(sd, strict=True, assign=True)
                        del sd
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
            self._last_load_error = None
            logger.info(f"Model ready. {mem_snapshot()}")

    def generate(
        self,
        request,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> list[str]:
        """Generate images. Returns list of base64-encoded PNGs."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. POST /api/load-model first.")

        backend_dir = str(Path(__file__).parent)
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from krea2 import sampling

        with self._lock:
            return self._generate_locked(request, progress_cb, sampling)

    def _generate_locked(self, req, progress_cb, sampling) -> list[str]:
        # Build prompt (inject bbox JSON + LoRA trigger words)
        prompt = _build_bbox_prompt(req.prompt, req.bboxes)
        prompt = build_trigger_prompt(prompt, req.loras or [])

        # Moodboard preset: prepend the mood's keywords, add its avoids to negative.
        negative_prompt = req.negative_prompt
        mood_id = getattr(req, "mood", "")
        if mood_id:
            from moods import apply_mood
            prompt, negative_prompt = apply_mood(prompt, negative_prompt, mood_id)

        # Encode text (encoder to GPU)
        self.encoder.to(self._device)
        try:
            # Reference images = custom moodboard board + the 3 ref slots.
            ref_images = []
            for b64 in list(getattr(req, "moodboard_images", []) or []) + [
                req.ref_image1_b64, req.ref_image2_b64, req.ref_image3_b64
            ]:
                if b64:
                    ref_images.append(_b64_to_pil(b64).convert("RGB"))

            if ref_images:
                txt, txtmask = self.encoder.forward_with_images(
                    [prompt] * req.num_images, images=ref_images
                )
            else:
                txt, txtmask = self.encoder([prompt] * req.num_images)

            if req.use_rebalance:
                weights = parse_weights(req.rebalance_weights)
                # Moodboard strength scales overall style push (0.5 = neutral ×1.0).
                mult = req.rebalance_multiplier
                if mood_id or ref_images:
                    mult = mult * (0.5 + float(getattr(req, "moodboard_strength", 0.5)))
                txt = rebalance(txt, multiplier=mult, layer_weights=weights)

            neg_txt = neg_txtmask = None
            if req.cfg > 0:
                neg_txt, neg_txtmask = self.encoder([negative_prompt] * req.num_images)

        finally:
            # Offload encoder to CPU before DiT forward pass
            self.encoder.cpu()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"Encoded. {mem_snapshot()}")

        # Align target dims to the patch grid (COMPRESSION*PATCH = 16) so the
        # init latent / mask match the spatial dims the sampler rounds to.
        _align = 8 * 2
        gen_w = ((req.width + _align - 1) // _align) * _align
        gen_h = ((req.height + _align - 1) // _align) * _align

        # Encode init image if needed
        init_latent = init_latent_clean = mask_tensor = None
        if req.mode in ("img2img", "inpaint", "outpaint") and req.init_image_b64:
            init_img = _b64_to_pil(req.init_image_b64).resize(
                (gen_w, gen_h), Image.LANCZOS
            )
            px = _pil_to_tensor(init_img, self._device, self._dtype)
            init_latent = self.ae.encode(px)
            init_latent_clean = init_latent.clone()

        if req.mode in ("inpaint", "outpaint") and req.mask_b64:
            mask_img = _b64_to_pil(req.mask_b64).convert("L").resize(
                (gen_w // 8, gen_h // 8), Image.NEAREST
            )
            import numpy as np
            mask_arr = torch.from_numpy(
                np.array(mask_img).astype("float32") / 255.0
            ).unsqueeze(0).unsqueeze(0).to(device=self._device, dtype=self._dtype)
            mask_tensor = mask_arr

        # Resolve seed
        seed = req.seed if req.seed >= 0 else int(torch.randint(0, 2**31, (1,)).item())

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
            progress_cb=progress_cb,
        )

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
                    progress_cb=progress_cb,
                )
                logger.info(f"Detail refine pass done (denoise={req.refine_denoise}).")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Refine pass skipped ({e})")

        # Save + return base64 + filenames
        results: list[str] = []
        filenames: list[str] = []
        for img in images:
            fname = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.png"
            img.save(str(OUTPUTS_DIR / fname))
            filenames.append(fname)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            results.append(base64.b64encode(buf.getvalue()).decode())

        return results, seed, filenames, lora_reports


# Singleton used by FastAPI
pipeline = Krea2Pipeline()
