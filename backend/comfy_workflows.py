"""Translate a Krea 2 GenerationRequest into a ComfyUI prompt graph.

This is the heart of the ComfyUI adapter. It maps the rich GenerationRequest
(the same object the frontend has always sent) onto ComfyUI nodes -- stock
nodes plus the community Krea-2 custom nodes installed under
ComfyUI/custom_nodes -- and returns the (results, seed, filenames,
lora_reports, metadata) tuple that main.py expects from the old native
pipeline.

Feature -> node map:
  * text encode / vision refs      -> TextEncodeKrea2, Krea2EncodeRebalance
  * rebalance                      -> ConditioningKrea2Rebalance
  * seed variance                  -> RBG_Smart_Seed_Variance
  * krea enhancer                  -> ComfyUI-Krea2T-Enhancer
  * negpip (negatives at cfg~1)    -> ApplyKrea2NegPiP
  * style preserve-structure       -> UntwistingRoPE
  * regional prompts               -> ConditioningSetMask + ConditioningCombine
  * flow-matching shift (mu)       -> ModelSamplingFlux
  * loras                          -> LoraLoaderModelOnly
  * inpaint (native/differential)  -> SetLatentNoiseMask (+ DifferentialDiffusion)
  * inpaint (lanpaint)             -> LanPaint_KSampler
  * gguf / int8 engines            -> UnetLoaderGGUF / UNETLoader
Model discovery is via ComfyUI/extra_model_paths.yaml (shared models dir).
"""
from __future__ import annotations

import base64
import copy
import io
import logging
import random
import uuid as _uuid
from typing import Any, Optional

import requests as _rq
from PIL import Image

if __package__:
    from .comfy_client import (
        ComfyClient,
        PromptIdCb,
        WS_IMAGE_NODE,
        comfy_base_url,
        free_comfy_vram,
    )
    from .output_saver import encode_images
else:
    from comfy_client import (
        ComfyClient,
        PromptIdCb,
        WS_IMAGE_NODE,
        comfy_base_url,
        free_comfy_vram,
    )
    from output_saver import encode_images

try:
    from generation_metadata import build_generation_metadata
except Exception:  # pragma: no cover
    build_generation_metadata = None  # type: ignore

try:
    from settings import OUTPUTS_DIR
except Exception:  # pragma: no cover
    from pathlib import Path
    OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

logger = logging.getLogger("krea2.comfy")

# ---------------------------------------------------------------------------
# Model / encoder / vae file resolution
# ---------------------------------------------------------------------------

KREA2_CLIP_NAME = "Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors"
KREA2_STOCK_CLIP_NAME = "qwen3vl_4b_fp8_scaled.safetensors"
KREA2_CLIP_TYPE = "krea2"
KREA2_VAE_NAME = "qwen_image_vae.safetensors"
# Community-preferred realism VAE (Civitai "Krea 2 Real VAE"); resolves fine
# detail better and removes the "textured hair" artifact. Used by default when
# present, overridable via KREA_VAE_NAME.
KREA2_REAL_VAE_NAME = "krea2RealVae_v10.safetensors"


# ComfyUI-loadable VAE files for the "VAE decoder mode" options. All live in
# models/krea2/vae which is mapped into ComfyUI's vae search path, so VAELoader
# can resolve them by filename. This replaces the native-only decoder modes so
# the ComfyUI (Int8/GGUF/fp8) path honours the same dropdown.
KREA2_COMFY_QWEN_VAE_NAME = "qwen_image_vae.safetensors"
KREA2_WAN_VAE_NAME = "wan_2.1_vae.safetensors"
_VALID_VAE_MODES = {"qwen", "comfy_qwen", "qwen_wan_blend", "wan_experimental"}
KREA2_IDENTITY_EDIT_LORA = "krea2_identity_edit_v1.safetensors"

# God Mode pipeline: Krea2 base -> Z-Image Turbo refine -> SeedVR2 7B-sharp upscale
# -> FaceDetailer. Reproduces the reference workflow embedded in krea2_final_00005.
GODMODE_KREA_REAL_VAE = "krea2RealVae_v10.safetensors"
GODMODE_ZIMAGE_UNET = "z_image_turbo_bf16.safetensors"
GODMODE_ZIMAGE_CLIP = "qwen_3_4b.safetensors"
GODMODE_ZIMAGE_VAE = "ae.safetensors"
GODMODE_SEEDVR2_MODEL = "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors"
GODMODE_FACE_DETECTOR = "bbox/face_yolov8m.pt"


def _vae_mode() -> str:
    import os
    mode = (os.environ.get("KREA2_VAE_MODE", "") or "").strip().lower()
    if not mode:
        try:
            from settings import settings
            mode = (getattr(settings, "krea2_vae_mode", "qwen") or "qwen").lower()
        except Exception:
            mode = "qwen"
    return mode if mode in _VALID_VAE_MODES else "qwen"


def _primary_qwen_vae_name() -> str:
    """The default Qwen decode file: community Real VAE if installed, else stock."""
    try:
        from settings import MODELS_DIR
        if (MODELS_DIR / "krea2" / "vae" / KREA2_REAL_VAE_NAME).exists():
            return KREA2_REAL_VAE_NAME
    except Exception:
        pass
    return KREA2_VAE_NAME


def _vae_name() -> str:
    """Primary VAE file for the active decoder mode (VAELoader vae_name)."""
    import os
    override = os.environ.get("KREA_VAE_NAME", "").strip()
    if override:
        return override
    mode = _vae_mode()
    if mode == "comfy_qwen":
        return KREA2_COMFY_QWEN_VAE_NAME
    if mode == "wan_experimental":
        return KREA2_WAN_VAE_NAME
    # qwen (default) and qwen_wan_blend both decode primarily with the Qwen VAE.
    return _primary_qwen_vae_name()

_UNET_BY_QUANT = {
    ("turbo", "bf16"): "krea2_turbo_bf16.safetensors",
    ("turbo", "fp16"): "krea2_turbo_bf16.safetensors",
    ("turbo", "fp8"): "krea2_turbo_fp8_scaled.safetensors",
    ("turbo", "int8"): "kreamania_v2-int8-convrot.safetensors",
    ("raw", "bf16"): "krea2_raw_bf16.safetensors",
    ("raw", "fp16"): "krea2_raw_bf16.safetensors",
    ("raw", "fp8"): "krea2_raw_fp8_scaled.safetensors",
    ("raw", "int8"): "krea2_raw_int8_convrot.safetensors",
}
_GGUF_BY_CHECKPOINT = {"turbo": "Krea-2-Turbo-Q4_K_M.gguf", "raw": "Krea-2-Turbo-Q4_K_M.gguf"}

# User-swappable Turbo INT8 ConvRot checkpoints (all int8-convrot, quality/behavior varies).
_TURBO_INT8_VARIANTS = {
    "orig": "krea2_turbo_int8_convrot.safetensors",
    "km_v2": "kreamania_v2-int8-convrot.safetensors",
    "km_v3": "kreamania_v3-int8-convrot-simple.safetensors",
    "km_v3_comfy": "kreamania_v3_int8_convrot.safetensors",
    "ax1y2jp": "ax1y2jp-krea2-turbo-int8-convrot.safetensors",
    "sceneworks": "sceneworks-krea2-turbo-int8-convrot.safetensors",
    "lilcheaty": "lilcheaty-krea2-turbo-int8-convrot.safetensors",
    "tsolful": "tsolful-krea2turbo-int8.safetensors",
    "redcraft": "redcraft-krea2-int8.safetensors",
}

VALID_SAMPLERS = {
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun",
    "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm",
    "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
    "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3",
    "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
    "res_2s", "res_3s", "res_2m", "res_3m", "res_2s_ode", "res_2m_ode",
}
VALID_SCHEDULERS = {
    "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta",
    "normal", "linear_quadratic", "kl_optimal", "bong_tangent", "beta57",
}
SAMPLER_ALIASES = {"euler_flow": "euler", "euler_ancestral_flow": "euler_ancestral"}

_TOKEN_SIZES = {"low", "normal", "high", "max"}

# Seed-variance enum maps (frontend value -> node emoji enum).
_SV_PRESET = {
    "off": "\u274c Disabled", "subtle": "\U0001f331 Subtle", "balanced": "\U0001f33f Balanced",
    "creative": "\U0001fab4 Creative", "bold": "\U0001f333 Bold", "wild": "\U0001f334 Wild",
    "custom": "\u2699\ufe0f Custom",
}
_SV_FADE = {
    "instant": "Instant", "linear": "Linear", "ease_in": "Ease-In", "ease_out": "Ease-Out",
    "ease_in_out": "Ease-In-Out", "smoothstep": "Smooth Step", "burst": "Burst",
}
_SV_PROTECT = {
    "none": "\U0001f6ab None", "first_quarter": "First Quarter", "first_half": "First Half",
    "last_quarter": "Last Quarter", "last_half": "Last Half",
}
_SV_DIRECTION = {
    "none": "\U0001f6ab None", "chaos": "\U0001f300 Chaos", "order": "\U0001f4d0 Order",
    "abstract": "\U0001f3a8 Abstract", "realistic": "\U0001f4f8 Realistic", "vibrant": "\U0001f308 Vibrant",
    "moody": "\U0001f311 Moody", "dreamy": "\U0001f4ad Dreamy", "dynamic_pose": "\U0001f3ad Dynamic Pose",
    "composition": "\U0001f5bc\ufe0f Composition", "diversity": "\U0001f30e Diversity",
    "facevar": "\U0001f9ec Face-Variance Expansion",
    "visceral_expression_grit": "\U0001f5ff Visceral Expression & Grit (Krea2)",
    "semantic_drift": "\U0001f9ed Semantic Drift (Centroid-Safe)", "structural_lock": "\U0001f9f1 Structural Lock",
    "cinematic_framing": "\U0001f39e\ufe0f Cinematic Framing", "texture_lift": "\U0001fab6 Texture Lift",
}
_DEFAULT_REBALANCE_WEIGHTS = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"


def _norm_sampler(name: str) -> str:
    name = (name or "").strip()
    name = SAMPLER_ALIASES.get(name, name)
    return name if name in VALID_SAMPLERS else "euler"


def _norm_scheduler(name: str) -> str:
    name = (name or "").strip()
    return name if name in VALID_SCHEDULERS else "simple"


def _cfg(req: Any) -> float:
    """Guidance scale for ComfyUI's KSampler.

    ComfyUI's CFG is result = uncond + cfg*(cond-uncond): cfg=1.0 is
    guidance-off (pure conditional), and cfg<1 degrades toward the
    unconditional (empty prompt), hurting adherence. The native Krea sampler
    used cfg=0 to mean "guidance off", so we floor cfg at 1.0 to preserve
    prompt adherence for the turbo default.
    """
    try:
        val = float(getattr(req, "cfg", 1.0))
    except (TypeError, ValueError):
        val = 1.0
    if val != val:  # NaN
        val = 1.0
    return max(1.0, val)


def _is_turbo(req: Any) -> bool:
    ckpt = (getattr(req, "checkpoint", "turbo") or "turbo").lower()
    profile = (getattr(req, "model_profile", "") or "").lower()
    if profile == "krea_raw" or ckpt == "raw":
        return False
    if getattr(req, "quality_preset", "") == "raw_benchmark":
        return False
    return True


def _has_active_loras(req: Any) -> bool:
    for l in (getattr(req, "loras", []) or []):
        if not isinstance(l, dict) or not l.get("enabled", True):
            continue
        if not (l.get("filename") or l.get("name")):
            continue
        try:
            if float(l.get("strength", 1.0)) != 0:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _bf16_unet_file(checkpoint: str) -> Optional[str]:
    """Return the bf16 checkpoint filename if it exists on disk, else None."""
    name = _UNET_BY_QUANT.get((checkpoint, "bf16"))
    if not name:
        return None
    try:
        from settings import MODELS_DIR
        if (MODELS_DIR / "krea2" / "diffusion_models" / name).exists():
            return name
    except Exception:
        pass
    return name  # assume present (extra_model_paths resolves it)


def resolve_unet(req: Any) -> tuple[str, str, bool, Optional[str]]:
    """Return (unet_name, weight_dtype, is_gguf, gguf_name)."""
    checkpoint = "turbo" if _is_turbo(req) else "raw"
    engine = getattr(req, "diffusion_engine", "native_pytorch") or "native_pytorch"
    quant = (getattr(req, "quantization", "fp8") or "fp8").lower()
    custom = getattr(req, "checkpoint_path", "") or ""
    is_custom = (getattr(req, "checkpoint", "") or "").lower() == "custom" and custom

    # Optional LoRA routing (opt-in): Krea's custom int8/fp8-scaled checkpoints
    # patch LoRAs through an emulated-op / on-the-fly-dequant path that can be
    # slow. Loading the bf16 checkpoint cast to standard fp8 (fp8_e4m3fn_fast) is
    # the fastest engine where ComfyUI's normal LoRA patch path works. This is
    # DISABLED by default so the user's chosen engine (int8 preferred, or gguf) is
    # honored for LoRA gens. Set KREA_LORA_ENGINE=fp8_fast (or fp8 / bf16) to route
    # LoRA gens onto the bf16-cast engine instead.
    if not is_custom and _has_active_loras(req):
        import os
        mode = os.environ.get("KREA_LORA_ENGINE", "").strip().lower()
        if mode not in ("", "off", "none", "native"):
            bf16 = _bf16_unet_file(checkpoint)
            if bf16:
                dtype = {"fp8_fast": "fp8_e4m3fn_fast", "fp8": "fp8_e4m3fn",
                         "fp8_e5m2": "fp8_e5m2", "bf16": "default"}.get(mode, "fp8_e4m3fn_fast")
                return bf16, dtype, False, None

    if engine == "native_gguf" or quant == "gguf":
        return "", "default", True, _GGUF_BY_CHECKPOINT.get(checkpoint)
    if engine == "native_int8_convrot" or quant == "int8":
        quant = "int8"
    if is_custom:
        import os
        return os.path.basename(custom), "default", False, None
    # Turbo INT8 has a user-swappable v1/v2/v3 ConvRot checkpoint.
    if checkpoint == "turbo" and quant == "int8":
        variant = str(getattr(req, "turbo_int8_variant", "redcraft") or "redcraft").lower()
        return _TURBO_INT8_VARIANTS.get(variant, _TURBO_INT8_VARIANTS["redcraft"]), "default", False, None
    unet = _UNET_BY_QUANT.get((checkpoint, quant)) or _UNET_BY_QUANT[(checkpoint, "fp8")]
    return unet, "default", False, None


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self._counter = 0

    def add(self, class_type: str, inputs: dict, node_id: str | None = None) -> str:
        if node_id is None:
            self._counter += 1
            node_id = f"n{self._counter}"
        self.nodes[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    def graph(self) -> dict:
        return self.nodes


def _link(node_id, slot: int = 0) -> list:
    """Reference a node output. Accepts an id (str) or an existing [id, slot]."""
    if isinstance(node_id, list):
        return node_id
    return [node_id, slot]


# ---------------------------------------------------------------------------
# Image / mask staging
# ---------------------------------------------------------------------------

def _b64_to_loadimage(g: GraphBuilder, b64: str) -> str:
    raw = base64.b64decode(b64.split(",")[-1])
    fname = f"krea_{_uuid.uuid4().hex}.png"
    _rq.post(
        f"{comfy_base_url()}/upload/image",
        files={"image": (fname, raw, "image/png")},
        data={"overwrite": "true"},
        timeout=60,
    )
    return g.add("LoadImage", {"image": fname})


def _load_scaled_image(g: GraphBuilder, b64: str, w: int, h: int, method: str = "lanczos") -> str:
    node = _b64_to_loadimage(g, b64)
    return g.add("ImageScale", {"image": _link(node), "upscale_method": method,
                                "width": int(w), "height": int(h), "crop": "disabled"})


def _mask_node(g: GraphBuilder, b64: str, w: int, h: int) -> str:
    scaled = _load_scaled_image(g, b64, w, h, method="bilinear")
    return g.add("ImageToMask", {"image": _link(scaled), "channel": "red"})


# ---------------------------------------------------------------------------
# Model chain + model-level feature patches
# ---------------------------------------------------------------------------

def _is_int8_unet_file(name: str) -> bool:
    """True when the resolved checkpoint is a pre-quantized INT8 ConvRot file.

    These must load through ComfyUI-INT8-Fast's OTUNetLoaderW8A8 — stock UNETLoader
    only understands a subset of the community INT8 formats (ax1y2jp/lilcheaty fail
    with int8_rowwise / unknown quant format KeyErrors).
    """
    return "int8" in (name or "").lower()


def _build_model_chain(g: GraphBuilder, req: Any) -> str:
    unet, weight_dtype, is_gguf, gguf_name = resolve_unet(req)
    if is_gguf and gguf_name:
        model = g.add("UnetLoaderGGUF", {"unet_name": gguf_name})
    elif _is_int8_unet_file(unet):
        # Pre-quantized INT8 ConvRot: use the W8A8 loader. on_the_fly stays off
        # (weights are already int8); enable_convrot matches how these files were packed.
        model = g.add("OTUNetLoaderW8A8", {
            "unet_name": unet,
            "weight_dtype": "default",
            "model_type": "krea2",
            "on_the_fly_quantization": False,
            "enable_convrot": True,
            "lora_mode": "None",
        })
    else:
        model = g.add("UNETLoader", {"unet_name": unet, "weight_dtype": weight_dtype})

    for lora in (getattr(req, "loras", []) or []):
        if not isinstance(lora, dict) or not lora.get("enabled", True):
            continue
        name = lora.get("filename") or lora.get("name") or ""
        if not name:
            continue
        if not name.lower().endswith((".safetensors", ".ckpt", ".pt")):
            name = f"{name}.safetensors"
        try:
            strength = float(lora.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        if strength != strength or strength in (float("inf"), float("-inf")):  # NaN/inf guard
            strength = 1.0
        # Krea2WideLoraLoaderModelOnly exposes the full +/-40000 range that
        # bypass LoRAs need (stock LoraLoaderModelOnly caps at +/-100). Clamp to
        # the node's widget bounds so an out-of-range value can't fail the prompt.
        strength = max(-40000.0, min(40000.0, strength))
        model = g.add("Krea2WideLoraLoaderModelOnly",
                      {"model": _link(model), "lora_name": name,
                       "strength_model": strength})

    mu = getattr(req, "mu", None)
    y1 = float(getattr(req, "y1", 0.5) or 0.5)
    y2 = float(getattr(req, "y2", 1.15) or 1.15)
    if mu is None and _is_turbo(req):
        mu = 1.15  # frozen turbo schedule (don't scale shift by resolution)
    if mu is not None:
        model = g.add("ModelSamplingFlux",
                      {"model": _link(model), "max_shift": float(mu), "base_shift": float(mu),
                       "width": 1024, "height": 1024})
    else:
        model = g.add("ModelSamplingFlux",
                      {"model": _link(model), "max_shift": y2, "base_shift": y1,
                       "width": int(getattr(req, "width", 1024)), "height": int(getattr(req, "height", 1024))})
    return model


def _build_clip(g: GraphBuilder) -> list:
    node = g.add("CLIPLoader", {"clip_name": KREA2_CLIP_NAME, "type": KREA2_CLIP_TYPE})
    return _link(node)


def _build_character_edit_clip(g: GraphBuilder) -> list:
    try:
        from settings import MODELS_DIR
        if (MODELS_DIR / "krea2" / "text_encoders" / KREA2_STOCK_CLIP_NAME).exists():
            return _link(g.add("CLIPLoader", {"clip_name": KREA2_STOCK_CLIP_NAME, "type": KREA2_CLIP_TYPE}))
    except Exception:
        pass
    return _build_clip(g)


def _build_vae(g: GraphBuilder, req: Any) -> str:
    return g.add("VAELoader", {"vae_name": _vae_name()})


def build_krea_model_bundle(
    graph: GraphBuilder, req: Any
) -> tuple[list, list, list]:
    """Build the stable Krea model/CLIP/VAE loader bundle for other adapters."""
    model = _build_model_chain(graph, req)
    clip = _build_clip(graph)
    vae = _build_vae(graph, req)
    return _link(model), clip, _link(vae)


def _apply_model_features(g: GraphBuilder, req: Any, model: str, clip: list, vae: str):
    """Apply enhancer, negpip and RoPE model patches. Returns (model, clip)."""
    if bool(getattr(req, "krea_enhancer_enabled", False)) and (getattr(req, "krea_enhancer_variant", "off") or "off") != "off":
        model = g.add("ComfyUI-Krea2T-Enhancer",
                      {"model": _link(model), "enabled": True,
                       "strength": float(getattr(req, "krea_enhancer_strength", 1.0)), "debug": False})

    # CFG-Zero* (arXiv:2503.18886): optimized-scale + zero-init first step.
    # Only meaningful at real guidance (cfg>1, i.e. RAW); a no-op at cfg=1.
    if bool(getattr(req, "cfg_zero_star", False)):
        model = g.add("CFGZeroStar", {"model": _link(model)})

    # NegPiP lets the negative prompt bite even at cfg~1 (distilled turbo).
    neg = (getattr(req, "negative_prompt", "") or "").strip()
    if neg and float(getattr(req, "cfg", 0.0) or 0.0) <= 1.0:
        np_node = g.add("ApplyKrea2NegPiP",
                        {"model": _link(model), "clip": clip,
                         "value_strength": 1.0, "patch_txtfusion_refiners": True})
        model, clip = _link(np_node, 0), _link(np_node, 1)

    # Structure-preserving style transfer (experimental, training-free RoPE).
    if (getattr(req, "style_fusion_mode", "") or "") == "preserve_structure":
        refs = _gather_refs(req)
        if refs:
            scaled = _load_scaled_image(g, refs[0][0], req.width, req.height)
            lat = g.add("VAEEncode", {"pixels": _link(scaled), "vae": _link(vae)})
            model = g.add("UntwistingRoPE",
                          {"model": _link(model), "rf_inversion": _link(lat), "beta": 50.0,
                           "high_scale_start": 1.0, "high_scale_end": 0.0, "low_scale_start": 1.0,
                           "low_scale_end": 3.0, "adain_strength": 0.5, "blocks": "0-999", "verbose": False})
    return model, clip


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------

def _gather_refs(req: Any) -> list[tuple[str, str, Any]]:
    """Collect up to 4 reference images as (b64, token_size, style_ref_or_None)."""
    refs: list[tuple[str, str, Any]] = []
    for s in (getattr(req, "style_references", []) or []):
        b = getattr(s, "image_b64", None) if not isinstance(s, dict) else s.get("image_b64")
        if not b:
            continue
        ts = (getattr(s, "token_size", None) if not isinstance(s, dict) else s.get("token_size")) or "normal"
        refs.append((b, ts if ts in _TOKEN_SIZES else "normal", s))
    for b in (getattr(req, "moodboard_images", []) or []):
        if b:
            refs.append((b, "normal", None))
    if bool(getattr(req, "image_prompt_enabled", False)):
        refs.extend(_gather_catalog_refs(req, max_count=max(0, 4 - len(refs))))
    for attr in ("ref_image1_b64", "ref_image2_b64", "ref_image3_b64"):
        b = getattr(req, attr, None)
        if b:
            refs.append((b, "normal", None))
    return refs[:4]


def _gather_catalog_refs(req: Any, max_count: int = 4) -> list[tuple[str, str, Any]]:
    """Hydrate selected catalog moodboard images for the opt-in image prompt path.

    Catalog moodboards still default to text guidance only. This helper is used
    only when image_prompt_enabled is true, so selecting a board never silently
    changes generation behavior.
    """
    if max_count <= 0:
        return []
    ids = list(getattr(req, "moodboard_ids", []) or [])
    uuids = list(getattr(req, "moodboard_uuids", []) or [])
    if not ids and not uuids:
        return []
    refs: list[tuple[str, str, Any]] = []
    try:
        from moodboards_catalog import fetch_moodboard_image_b64, moodboard_generation_context
        ctx = moodboard_generation_context(ids, moodboard_uuids=uuids, max_images=max_count)
        for url in (ctx.get("image_urls") or [])[:max_count]:
            try:
                b64 = fetch_moodboard_image_b64(url)
            except Exception:
                logger.debug("Could not hydrate moodboard image reference: %s", url, exc_info=True)
                continue
            if b64:
                refs.append((b64, "normal", None))
            if len(refs) >= max_count:
                break
    except Exception:
        logger.debug("Could not gather catalog moodboard image references", exc_info=True)
    return refs


def _rebalance(g: GraphBuilder, req: Any, cond: str) -> str:
    if not bool(getattr(req, "use_rebalance", True)):
        return cond
    weights = (getattr(req, "rebalance_weights", "") or "").strip() or _DEFAULT_REBALANCE_WEIGHTS
    return g.add("ConditioningKrea2Rebalance",
                 {"conditioning": _link(cond), "multiplier": float(getattr(req, "rebalance_multiplier", 1.0) or 1.0),
                  "per_layer_weights": weights})


def _apply_ref_strength(g: GraphBuilder, req: Any, cond: str) -> str:
    """Overall reference strength for the multi-image path.

    Krea2EncodeRebalance (2-4 refs) does its own per-layer rebalance, so the
    single-ref `_rebalance` (with the boosting default weights) isn't applied
    there. To give the UI's "overall strength" knob a real effect for multi-image
    without regressing existing output, we apply a *pure global* magnitude scale
    (all-ones per-layer weights) and only when the multiplier is off its 1.0
    default — so the default remains a true no-op.
    """
    if not bool(getattr(req, "use_rebalance", True)):
        return cond
    mult = float(getattr(req, "rebalance_multiplier", 1.0) or 1.0)
    if abs(mult - 1.0) < 1e-6:
        return cond
    ones = ",".join(["1.0"] * 12)
    return g.add("ConditioningKrea2Rebalance",
                 {"conditioning": _link(cond), "multiplier": mult, "per_layer_weights": ones})


_STYLE_ONLY_SYSTEM_PROMPT = (
    "Describe ONLY the visual style of the image: color palette, lighting, "
    "contrast, artistic medium and technique, brushwork or rendering, texture, "
    "and overall mood. Do NOT describe or mention the specific subjects, people, "
    "animals, objects, or their spatial composition. Report only transferable "
    "style attributes."
)


def _build_style_average(g: GraphBuilder, req: Any, refs: list[tuple[str, str, Any]], clip: list) -> str:
    """Encode each ref independently, then running-mean average the conditionings.

    This is the tested "match style" path: it avoids Krea2EncodeRebalance's
    Picture 1/Picture 2 collage behavior and preserves shared mood/medium while
    reducing specific content copy.
    """
    mp = float(getattr(req, "image_prompt_strength", 0.2) or 0.2)
    mp = max(0.1, min(1.0, mp))
    encs: list[str] = []
    for b, _ts, _s in refs[:4]:
        encs.append(g.add("TextEncodeKrea2", {
            "clip": clip,
            "prompt": req.prompt,
            "image1": _link(_b64_to_loadimage(g, b)),
            "system_prompt": _STYLE_ONLY_SYSTEM_PROMPT,
            "vision_megapixels": mp,
            "vision_position": "before prompt",
        }))
    if not encs:
        return g.add("TextEncodeKrea2", {"clip": clip, "prompt": req.prompt})
    avg = encs[0]
    for i, enc in enumerate(encs[1:], start=1):
        avg = g.add("ConditioningAverage", {
            "conditioning_to": _link(avg),
            "conditioning_from": _link(enc),
            "conditioning_to_strength": i / (i + 1),
        })
    return avg


def _seed_variance(g: GraphBuilder, req: Any, cond: str, seed: int) -> str:
    preset = getattr(req, "seed_variance_preset", "off") or "off"
    if preset in ("off", None):
        return cond
    st = float(getattr(req, "seed_variance_injection_start", 0.0) or 0.0)
    en = float(getattr(req, "seed_variance_injection_end", 1.0) or 1.0)
    injection = "All Steps"
    if en <= 0.5:
        injection = "Beginning Steps"
    elif st >= 0.5:
        injection = "Ending Steps"
    sched = getattr(req, "seed_variance_schedule", "constant") or "constant"
    if sched not in ("constant", "decreasing", "step_cutoff", "hard_lock", "tiered_release"):
        sched = "constant"
    # Schedule-aware timeline: the cutoff/lock is expressed as cutoff_step out of
    # total_steps. For that to land on the intended point (e.g. "lock composition
    # for the first half"), total_steps must reflect the *actual* sampler steps for
    # this run, not the static default. Fall back to the field only if steps unknown.
    actual_steps = int(getattr(req, "steps", 0) or 0)
    total_steps = actual_steps if actual_steps > 0 else int(getattr(req, "seed_variance_total_steps", 20) or 20)
    total_steps = max(1, total_steps)
    cutoff_step = int(getattr(req, "seed_variance_cutoff_step", 8) or 8)
    cutoff_step = max(0, min(cutoff_step, total_steps))
    node = g.add("RBG_Smart_Seed_Variance", {
        "conditioning": _link(cond),
        "variance_preset": _SV_PRESET.get(preset, "\U0001f33f Balanced"),
        "fine_tune_variance": int(round(float(getattr(req, "seed_variance_randomize_percent", 0.0) or 0.0))),
        "model_type": "\U0001f4f8 Krea2 (SingleStream)",
        "fade_curve": _SV_FADE.get(getattr(req, "seed_variance_fade_curve", "linear"), "Linear"),
        "noise_injection": injection,
        "protect_mode": _SV_PROTECT.get(getattr(req, "seed_variance_protection", "first_half"), "First Half"),
        "protect_regions": "",
        "direction_shift": _SV_DIRECTION.get(getattr(req, "seed_variance_direction", "none"), "\U0001f6ab None"),
        "shift_strength": int(getattr(req, "seed_variance_shift_strength", 100) or 100),
        "variance_schedule": sched,
        "cutoff_step": cutoff_step,
        "total_steps": total_steps,
        "cutoff_strength": float(getattr(req, "seed_variance_cutoff_strength", 0.0) or 0.0),
        "seed": seed,
    })
    return _link(node, 0)


def _apply_regional(g: GraphBuilder, req: Any, clip: list, positive: str) -> str:
    combined = positive
    for r in (getattr(req, "regional_prompts", []) or []):
        visible = getattr(r, "visible", True) if not isinstance(r, dict) else r.get("visible", True)
        mask_b64 = (getattr(r, "mask_b64", "") if not isinstance(r, dict) else r.get("mask_b64", "")) or ""
        rprompt = getattr(r, "prompt", "") if not isinstance(r, dict) else r.get("prompt", "")
        strength = getattr(r, "strength", 1.0) if not isinstance(r, dict) else r.get("strength", 1.0)
        if not visible or not mask_b64 or not rprompt:
            continue
        enc = g.add("TextEncodeKrea2", {"clip": clip, "prompt": rprompt})
        m = _mask_node(g, mask_b64, req.width, req.height)
        rc = g.add("ConditioningSetMask",
                   {"conditioning": _link(enc), "mask": _link(m),
                    "strength": float(strength), "set_cond_area": "default"})
        combined = g.add("ConditioningCombine", {"conditioning_1": _link(combined), "conditioning_2": _link(rc)})
    return combined


# Built-in style-extract + edit system prompt (mirrors Krea2SystemPrompt's
# default). Makes the VL read the reference image's style then apply the edit,
# which the community found works notably better for adding/altering content.
_DEFAULT_INCONTEXT_SYSTEM_PROMPT = (
    "Describe the key features of the reference image (color, palette, shape, "
    "size, texture, subjects, lighting, and background), then apply the user's "
    "instruction, generating a new image that follows the instruction while "
    "staying visually consistent with the reference where appropriate."
)


def _incontext_positive(g: GraphBuilder, req: Any, clip: list) -> Optional[str]:
    """In-context vision edit: feed a reference/source image into Krea's Qwen3-VL
    vision so the instruction prompt edits it (the community "QwenEdit
    text-encode for Krea 2"). Two encoders are supported:
      - krea2 (default): TextEncodeKrea2 with optional style-extract system prompt
        + vision position / detail / mask controls.
      - qwen_edit_plus: TextEncodeQwenImageEditPlus (stronger multi-image edit).
    Returns None when disabled or no image is available."""
    if not bool(getattr(req, "incontext_edit", False)):
        return None
    img = (getattr(req, "incontext_image_b64", "") or getattr(req, "init_image_b64", "") or "").strip()
    if not img:
        return None
    loaded = _b64_to_loadimage(g, img)

    if str(getattr(req, "incontext_encoder", "krea2")) == "qwen_edit_plus":
        return g.add("TextEncodeQwenImageEditPlus", {
            "clip": clip, "prompt": req.prompt,
            "vae": _link(_build_vae(g, req)), "image1": _link(loaded),
        })

    sysp = (getattr(req, "incontext_system_prompt", "") or "").strip() or _DEFAULT_INCONTEXT_SYSTEM_PROMPT
    te: dict = {
        "clip": clip, "prompt": req.prompt, "system_prompt": sysp,
        "image1": _link(loaded),
        "vision_megapixels": float(getattr(req, "incontext_vision_megapixels", 1.0) or 1.0),
        "vision_position": "after prompt" if str(getattr(req, "incontext_vision_position", "before")) == "after" else "before prompt",
    }
    m = (getattr(req, "incontext_mask_b64", "") or "").strip()
    if m:
        te["mask1"] = _link(_mask_node(g, m, req.width, req.height))
    return g.add("TextEncodeKrea2", te)


def _build_positive(g: GraphBuilder, req: Any, clip: list, seed: int) -> str:
    ic = _incontext_positive(g, req, clip)
    if ic is not None:
        cond = _rebalance(g, req, ic)
        cond = _apply_regional(g, req, clip, cond)
        cond = _seed_variance(g, req, cond, seed)
        return cond
    refs = _gather_refs(req)
    prompt = req.prompt
    if len(refs) >= 2:
        if bool(getattr(req, "image_prompt_enabled", False)) and str(getattr(req, "image_prompt_mode", "match_style")) == "match_style":
            cond = _build_style_average(g, req, refs, clip)
        else:
            # Multi-image: Krea2EncodeRebalance carries up to 4 refs (+ built-in rebalance).
            # This intentionally copies/composes reference content and can collage distinct refs.
            inputs: dict = {"text": prompt, "clip": clip}
            for i, (b, ts, _s) in enumerate(refs, start=1):
                inputs[f"image{i}"] = _link(_b64_to_loadimage(g, b))
                inputs[f"image{i}_tokens"] = ts
            cond = g.add("Krea2EncodeRebalance", inputs)
            cond = _apply_ref_strength(g, req, cond)
    elif len(refs) == 1:
        if bool(getattr(req, "image_prompt_enabled", False)) and str(getattr(req, "image_prompt_mode", "match_style")) == "match_style":
            cond = _build_style_average(g, req, refs, clip)
            cond = _rebalance(g, req, cond)
            cond = _apply_regional(g, req, clip, cond)
            cond = _seed_variance(g, req, cond, seed)
            return cond
        b, _ts, s = refs[0]
        te: dict = {"clip": clip, "prompt": prompt, "image1": _link(_b64_to_loadimage(g, b))}
        if s is not None and not isinstance(s, dict):
            if getattr(s, "vision_megapixels", None):
                te["vision_megapixels"] = float(s.vision_megapixels)
            vp = getattr(s, "vision_position", "before_prompt")
            te["vision_position"] = "after prompt" if vp == "after_prompt" else "before prompt"
            if getattr(s, "mask_b64", None):
                te["mask1"] = _link(_mask_node(g, s.mask_b64, req.width, req.height))
        cond = g.add("TextEncodeKrea2", te)
        cond = _rebalance(g, req, cond)
    else:
        cond = g.add("TextEncodeKrea2", {"clip": clip, "prompt": prompt})
        cond = _rebalance(g, req, cond)

    cond = _apply_regional(g, req, clip, cond)
    cond = _seed_variance(g, req, cond, seed)
    return cond


def _build_negative(g: GraphBuilder, req: Any, clip: list) -> str:
    return g.add("TextEncodeKrea2", {"clip": clip, "prompt": getattr(req, "negative_prompt", "") or ""})


# ---------------------------------------------------------------------------
# Sampler helpers
# ---------------------------------------------------------------------------

# RES4LYF ClownsharKSampler_Beta sampler_name values we expose. The node uses a
# "category/method" naming; we only accept known-good ones so a bad value can't
# fail the whole prompt (falls back to the stock KSampler otherwise).
CLOWNSHARK_SAMPLERS = frozenset({
    "exponential/ddim", "exponential/res_2s", "exponential/res_2s_stable",
    "exponential/res_3s", "exponential/res_4s_krogstad", "exponential/res_5s",
    "exponential/res_6s", "exponential/dpmpp_2s", "exponential/dpmpp_3s",
    "multistep/res_2m", "multistep/res_3m", "multistep/dpmpp_2m", "multistep/dpmpp_3m",
    "multistep/deis_2m", "multistep/deis_3m", "linear/euler", "linear/heun_2s",
    "linear/rk4_4s",
})


def _clownshark(g: GraphBuilder, model, positive: str, negative: str, latent: str,
                req: Any, seed: int, denoise: float, sampler_name: str) -> str:
    """RES4LYF ClownsharKSampler_Beta node (the Xperiment/uncensored reference
    sampler). Mirrors the Comfy-Org Krea-2 Turbo workflow: eta-noised RES solver
    with beta57 spacing at CFG 1 (turbo guidance-off)."""
    return g.add("ClownsharKSampler_Beta", {
        "model": _link(model), "positive": _link(positive), "negative": _link(negative),
        "latent_image": _link(latent),
        "eta": float(getattr(req, "res4lyf_eta", 0.5) or 0.0),
        "sampler_name": sampler_name,
        "scheduler": _norm_scheduler(req.scheduler),
        "steps": int(req.steps), "steps_to_run": -1,
        "denoise": float(denoise), "cfg": _cfg(req),
        "seed": int(seed), "sampler_mode": "standard",
        "bongmath": bool(getattr(req, "res4lyf_bongmath", False)),
    })


def _ksampler(g: GraphBuilder, model, positive: str, negative: str, latent: str,
              req: Any, seed: int, denoise: float) -> str:
    clown = (getattr(req, "res4lyf_sampler", "") or "").strip()
    if clown in CLOWNSHARK_SAMPLERS:
        return _clownshark(g, model, positive, negative, latent, req, seed, denoise, clown)
    # Actual-Denoise: for real img2img/edit (denoise<1), route model+scheduler+denoise
    # through ActualDenoise so the effective noise is identical across schedulers.
    if bool(getattr(req, "actual_denoise", False)) and float(denoise) < 0.999:
        ad = g.add("ActualDenoise", {
            "model": _link(model), "scheduler": _norm_scheduler(req.scheduler),
            "actual_denoise": float(denoise),
        })
        return g.add("KSampler", {
            "model": _link(ad, 2), "seed": seed, "steps": int(req.steps), "cfg": _cfg(req),
            "sampler_name": _norm_sampler(req.sampler), "scheduler": _link(ad, 0),
            "positive": _link(positive), "negative": _link(negative),
            "latent_image": _link(latent), "denoise": _link(ad, 1),
        })
    return g.add("KSampler", {
        "model": _link(model), "seed": seed, "steps": int(req.steps), "cfg": _cfg(req),
        "sampler_name": _norm_sampler(req.sampler), "scheduler": _norm_scheduler(req.scheduler),
        "positive": _link(positive), "negative": _link(negative),
        "latent_image": _link(latent), "denoise": float(denoise),
    })


def _maybe_refine(g: GraphBuilder, model, positive: str, negative: str, latent: str,
                  req: Any, seed: int) -> str:
    if not bool(getattr(req, "refine", False)):
        return latent
    return g.add("KSampler", {
        "model": _link(model), "seed": seed + 1, "steps": int(getattr(req, "refine_steps", 6)),
        "cfg": _cfg(req), "sampler_name": _norm_sampler(req.sampler),
        "scheduler": _norm_scheduler(req.scheduler), "positive": _link(positive),
        "negative": _link(negative), "latent_image": _link(latent),
        "denoise": float(getattr(req, "refine_denoise", 0.3)),
    })


# ---------------------------------------------------------------------------
# Mode graphs
# ---------------------------------------------------------------------------

def _maybe_degrid(g: GraphBuilder, image: str, req: Any | None) -> str:
    """Strip the 2px Qwen/Wan VAE grid after decode (default on; toggle via vae_degrid)."""
    enabled = True if req is None else bool(getattr(req, "vae_degrid", True))
    if not enabled:
        return image
    return g.add("VAEDeGrid", {
        "image": _link(image),
        "enabled": True,
        "mode": "auto",
        "limit": 0.02,
        "grid_gain": 10.0,
        "grid_view": "4x zoom",
    })


def _finish(g: GraphBuilder, latent: str, vae: str, req: Any = None) -> None:
    # At 1K+, a full VAEDecode makes Comfy evict several GB of the text encoder,
    # adding roughly 50 seconds of reload churn to the next prompt on a 24GB
    # card. Core tiled decode keeps the hot generation chain resident; measured
    # output delta is negligible (about 51 dB PSNR at 1K).
    w = int(getattr(req, "width", 1024) or 1024) if req is not None else 1024
    h = int(getattr(req, "height", 1024) or 1024) if req is not None else 1024
    tiled = max(w, h) >= 1024

    def _decode(vae_link: str) -> str:
        if tiled:
            return g.add("VAEDecodeTiled", {
                "samples": _link(latent), "vae": _link(vae_link),
                "tile_size": 512, "overlap": 64,
                "temporal_size": 64, "temporal_overlap": 8,
            })
        return g.add("VAEDecode", {"samples": _link(latent), "vae": _link(vae_link)})

    # "Qwen + Wan detail blend" decodes the latent with both the Qwen VAE (base/color)
    # and the Wan 2.1 VAE (high-frequency detail), then blends via core ImageBlend.
    if _vae_mode() == "qwen_wan_blend":
        wan_vae = g.add("VAELoader", {"vae_name": KREA2_WAN_VAE_NAME})
        image = g.add("ImageBlend", {
            "image1": _link(_decode(vae)), "image2": _link(_decode(wan_vae)),
            "blend_factor": 0.35, "blend_mode": "normal",
        })
    else:
        image = _decode(vae)
    # DeGrid after the final pixel image so later sharpen/upscale don't amplify the grid.
    image = _maybe_degrid(g, image, req)
    g.add("SaveImageWebsocket", {"images": _link(image)}, node_id=WS_IMAGE_NODE)


def _common(g: GraphBuilder, req: Any, seed: int):
    """Build model chain + features + conditioning shared by all modes.

    Returns (model, vae, positive, negative)."""
    model = _build_model_chain(g, req)
    clip = _build_clip(g)
    vae = _build_vae(g, req)
    model, clip = _apply_model_features(g, req, model, clip, vae)
    positive = _build_positive(g, req, clip, seed)
    negative = _build_negative(g, req, clip)
    return model, vae, positive, negative


def _build_txt2img(g: GraphBuilder, req: Any, seed: int) -> None:
    model, vae, positive, negative = _common(g, req, seed)
    latent = g.add("EmptySD3LatentImage",
                   {"width": int(req.width), "height": int(req.height),
                    "batch_size": max(1, int(getattr(req, "num_images", 1) or 1))})
    sampled = _ksampler(g, model, positive, negative, latent, req, seed, denoise=1.0)
    sampled = _maybe_refine(g, model, positive, negative, sampled, req, seed)
    _finish(g, sampled, vae, req)


def _build_img2img(g: GraphBuilder, req: Any, seed: int) -> None:
    if not getattr(req, "init_image_b64", None):
        return _build_txt2img(g, req, seed)
    model, vae, positive, negative = _common(g, req, seed)
    init = _load_scaled_image(g, req.init_image_b64, req.width, req.height)
    init_latent = g.add("VAEEncode", {"pixels": _link(init), "vae": _link(vae)})
    sampled = _ksampler(g, model, positive, negative, init_latent, req, seed,
                        denoise=float(getattr(req, "denoise", 1.0)))
    sampled = _maybe_refine(g, model, positive, negative, sampled, req, seed)
    _finish(g, sampled, vae, req)


def _build_inpaint(g: GraphBuilder, req: Any, seed: int) -> None:
    if not getattr(req, "init_image_b64", None) or not getattr(req, "mask_b64", None):
        return _build_txt2img(g, req, seed)
    model, vae, positive, negative = _common(g, req, seed)
    init = _load_scaled_image(g, req.init_image_b64, req.width, req.height)
    mask = _mask_node(g, req.mask_b64, req.width, req.height)
    init_latent = g.add("VAEEncode", {"pixels": _link(init), "vae": _link(vae)})
    masked = g.add("SetLatentNoiseMask", {"samples": _link(init_latent), "mask": _link(mask)})

    if bool(getattr(req, "differential_inpaint", False)):
        model = g.add("DifferentialDiffusion", {"model": _link(model)})

    denoise = float(getattr(req, "denoise", 1.0))
    if (getattr(req, "inpaint_method", "native") or "native") == "lanpaint_experimental":
        sampled = g.add("LanPaint_KSampler", {
            "model": _link(model), "seed": seed, "steps": int(req.steps), "cfg": _cfg(req),
            "sampler_name": _norm_sampler(req.sampler), "scheduler": _norm_scheduler(req.scheduler),
            "positive": _link(positive), "negative": _link(negative), "latent_image": _link(masked),
            "denoise": denoise,
            "LanPaint_NumSteps": int(getattr(req, "lanpaint_inner_steps", 3)),
            "LanPaint_PromptMode": getattr(req, "lanpaint_prompt_mode", "Image First"),
            "LanPaint_Info": "",
            "Inpainting_mode": "\U0001f5bc\ufe0f Image Inpainting",
        })
    else:
        sampled = _ksampler(g, model, positive, negative, masked, req, seed, denoise=denoise)
    _finish(g, sampled, vae, req)


def _build_outpaint(g: GraphBuilder, req: Any, seed: int) -> None:
    if not getattr(req, "init_image_b64", None) or not getattr(req, "mask_b64", None):
        return _build_txt2img(g, req, seed)
    model, vae, positive, negative = _common(g, req, seed)
    init = _load_scaled_image(g, req.init_image_b64, req.width, req.height)
    mask = _mask_node(g, req.mask_b64, req.width, req.height)
    init_latent = g.add("VAEEncode", {"pixels": _link(init), "vae": _link(vae)})
    masked = g.add("SetLatentNoiseMask", {"samples": _link(init_latent), "mask": _link(mask)})
    sampled = _ksampler(g, model, positive, negative, masked, req, seed,
                        denoise=float(getattr(req, "denoise", 1.0)))
    harmonized = g.add("KSampler", {
        "model": _link(model), "seed": seed + 2, "steps": int(req.steps), "cfg": _cfg(req),
        "sampler_name": _norm_sampler(req.sampler), "scheduler": _norm_scheduler(req.scheduler),
        "positive": _link(positive), "negative": _link(negative),
        "latent_image": _link(sampled), "denoise": 0.12,
    })
    _finish(g, harmonized, vae, req)


def _character_edit_source(req: Any) -> str:
    return (
        (getattr(req, "character_edit_source_b64", "") or "").strip()
        or (getattr(req, "init_image_b64", "") or "").strip()
        or (getattr(req, "incontext_image_b64", "") or "").strip()
    )


def _region_get(region: Any, key: str, default: Any = None) -> Any:
    if isinstance(region, dict):
        return region.get(key, default)
    return getattr(region, key, default)


def _rect_mask_node(g: GraphBuilder, region: Any, width: int, height: int) -> str:
    """Build a soft rectangle MASK (normalized region -> pixel rect) and return an
    ImageToMask node id. The rect is rasterized server-side with PIL, uploaded as a
    grayscale PNG, then converted to a ComfyUI MASK."""
    from PIL import ImageDraw, ImageFilter

    W, H = max(1, int(width)), max(1, int(height))
    x0 = max(0, min(W - 1, int(round(float(_region_get(region, "x", 0.0)) * W))))
    y0 = max(0, min(H - 1, int(round(float(_region_get(region, "y", 0.0)) * H))))
    x1 = max(x0 + 1, min(W, x0 + int(round(float(_region_get(region, "w", 1.0)) * W))))
    y1 = max(y0 + 1, min(H, y0 + int(round(float(_region_get(region, "h", 1.0)) * H))))

    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rectangle([x0, y0, x1 - 1, y1 - 1], fill=255)
    feather = max(0, int(_region_get(region, "feather", 24) or 0))
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather / 2))

    buf = io.BytesIO()
    Image.merge("RGB", (mask, mask, mask)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    img = _b64_to_loadimage(g, b64)
    return g.add("ImageToMask", {"image": _link(img), "channel": "red"})


def _apply_character_regions(g: GraphBuilder, req: Any, clip: list, base_positive: str,
                             *, subject_image: str, scene_image: Optional[str],
                             grounding_px: int) -> str:
    """Place per-region grounded conditionings (each with its own reference image)
    into rectangular masks, combined on top of the base scene conditioning. This is
    the Krea "annotations" pattern: subject A in one box, subject B in another."""
    combined = base_positive
    for region in (getattr(req, "character_edit_regions", []) or []):
        prompt = str(_region_get(region, "prompt", "") or "").strip()
        if not prompt:
            continue
        ref_b64 = str(_region_get(region, "reference_b64", "") or "").strip()
        region_image = _b64_to_loadimage(g, ref_b64) if ref_b64 else subject_image
        enc_inputs = {
            "clip": clip,
            "prompt": prompt,
            "image": _link(region_image),
            "grounding_px": grounding_px,
        }
        # If we have a scene and a region-specific reference, ground the referenced
        # person into the scene (matches the two-reference training recipe).
        if scene_image is not None and ref_b64:
            enc_inputs["image_b"] = _link(scene_image)
        enc = g.add("Krea2EditGroundedEncode", enc_inputs)
        mask = _rect_mask_node(g, region, int(req.width), int(req.height))
        strength = float(_region_get(region, "strength", 1.0) or 1.0)
        masked = g.add("ConditioningSetMask", {
            "conditioning": _link(enc), "mask": _link(mask),
            "strength": strength, "set_cond_area": "default",
        })
        combined = g.add("ConditioningCombine", {
            "conditioning_1": _link(combined), "conditioning_2": _link(masked),
        })
    return combined


def _build_character_edit(g: GraphBuilder, req: Any, seed: int) -> None:
    """Identity-preserving instruction edit using lbouaraba/comfyui-krea2edit.

    The conradlocke identity-edit LoRA needs dual conditioning:
    source VAE latent tokens + image-grounded Qwen3-VL encoding.
    """
    source = _character_edit_source(req)
    if not source:
        raise RuntimeError("Character Edit requires a source image.")

    model = _build_model_chain(g, req)
    model = g.add("LoraLoaderModelOnly", {
        "model": _link(model),
        "lora_name": KREA2_IDENTITY_EDIT_LORA,
        "strength_model": float(getattr(req, "character_edit_lora_strength", 1.0) or 1.0),
    })
    clip = _build_character_edit_clip(g)
    # Match the public Krea2Edit workflow/model card defaults.
    vae = g.add("VAELoader", {"vae_name": KREA2_VAE_NAME})
    grounding_px = int(getattr(req, "character_edit_grounding_px", 768) or 768)

    subject_image = _b64_to_loadimage(g, source)
    subject_latent = g.add("VAEEncode", {"pixels": _link(subject_image), "vae": _link(vae)})

    # Optional two-reference edit: place the subject into a supplied SCENE. Per the
    # model card the scene is the primary frame (source_latent / image) and the
    # subject becomes frame 2 (source_latent_b / image_b).
    reference = (getattr(req, "character_edit_reference_b64", "") or "").strip()
    if reference:
        scene_image = _b64_to_loadimage(g, reference)
        scene_latent = g.add("VAEEncode", {"pixels": _link(scene_image), "vae": _link(vae)})
        patch_image, patch_image_b = scene_image, subject_image
        model = g.add("Krea2EditModelPatch", {
            "model": _link(model),
            "source_latent": _link(scene_latent),
            "source_latent_b": _link(subject_latent),
        })
    else:
        patch_image, patch_image_b = subject_image, None
        scene_image = None
        model = g.add("Krea2EditModelPatch", {"model": _link(model), "source_latent": _link(subject_latent)})

    def _grounded(prompt_text: str) -> str:
        inputs = {
            "clip": clip,
            "prompt": prompt_text,
            "image": _link(patch_image),
            "grounding_px": grounding_px,
        }
        if patch_image_b is not None:
            inputs["image_b"] = _link(patch_image_b)
        return g.add("Krea2EditGroundedEncode", inputs)

    positive = _grounded(getattr(req, "prompt", "") or "")
    # Optional regional placement boxes (draw subject A left, subject B right, etc.).
    positive = _apply_character_regions(
        g, req, clip, positive,
        subject_image=subject_image, scene_image=scene_image, grounding_px=grounding_px,
    )
    if float(getattr(req, "cfg", 1.0) or 0.0) > 1.0:
        # Model card: at CFG > 1 ground the negative with an EMPTY prompt + the same
        # image(s) -- this is the trained unconditional. A non-empty negative here is
        # out of distribution for the identity-edit LoRA.
        negative = _grounded("")
    else:
        negative = _build_negative(g, req, clip)

    latent = g.add("EmptySD3LatentImage",
                   {"width": int(req.width), "height": int(req.height),
                    "batch_size": max(1, int(getattr(req, "num_images", 1) or 1))})
    sampled = _ksampler(g, model, positive, negative, latent, req, seed, denoise=1.0)
    _finish(g, sampled, vae, req)


def _uses_nk2e(req: Any) -> bool:
    for lora in (getattr(req, "loras", []) or []):
        if not isinstance(lora, dict) or not lora.get("enabled", True):
            continue
        if "nk2e" in str(lora.get("filename") or lora.get("name") or "").lower():
            return True
    return False


# Krea 2 Depth ControlNet assets (facok/comfyui-krea2-controlnet + DA3).
DEPTH_CONTROL_LORA_NAME = "depth-control-lora.safetensors"   # models/loras
DA3_DEPTH_MODEL_NAME = "depth_anything_3_small.safetensors"  # models/geometry_estimation
KREA2_TURBO_LORA_NAME = "krea2_turbo_lora_rank_64_final_nodiff.safetensors"  # RAW->turbo-speed

# Depth-estimator resolution tiers. DA3 needs a multiple of 14 (ViT patch size);
# the comfyui_controlnet_aux preprocessors accept any int (longest-edge resize).
_DEPTH_RES_TIERS = {"low": 504, "med": 700, "high": 1036}


def _depth_resolution(req: Any) -> int:
    val = getattr(req, "depth_resolution", 504)
    if isinstance(val, str):
        val = _DEPTH_RES_TIERS.get(val.lower(), 504)
    try:
        return max(256, min(2048, int(val)))
    except (TypeError, ValueError):
        return 504


def _build_depth_map(g: GraphBuilder, src_img_link, req: Any):
    """Build a depth map from a loaded source IMAGE using the chosen estimator.
    Returns the depth IMAGE link. DA3 (default) uses Depth-Anything-3; the others
    route through comfyui_controlnet_aux preprocessors (auto-download on first use)."""
    estimator = (getattr(req, "depth_estimator", "da3") or "da3").lower()
    res = _depth_resolution(req)
    if estimator in ("depth_anything_v2", "dav2"):
        return g.add("DepthAnythingV2Preprocessor", {"image": _link(src_img_link), "resolution": res})
    if estimator == "zoe":
        return g.add("Zoe-DepthMapPreprocessor", {"image": _link(src_img_link), "resolution": res})
    if estimator == "midas":
        return g.add("MiDaS-DepthMapPreprocessor", {"image": _link(src_img_link), "resolution": res})
    # default: Depth-Anything-3 (resolution snapped to a multiple of 14)
    da3res = max(196, (res // 14) * 14)
    da3 = g.add("LoadDA3Model", {"model_name": DA3_DEPTH_MODEL_NAME, "weight_dtype": "default"})
    geom = g.add("DA3Inference", {"da3_model": _link(da3), "image": _link(src_img_link),
                                  "resolution": da3res, "resize_method": "upper_bound_resize",
                                  "mode": "mono"})
    # DA3Render's "output" is a dynamic combo; sub-inputs are keyed "output.<name>".
    return g.add("DA3Render", {"da3_geometry": _link(geom), "output": "depth",
                               "output.normalization": "min_max", "output.apply_sky_clip": False})


def _build_depth_control(g: GraphBuilder, req: Any, seed: int) -> None:
    """Krea 2 depth ControlNet: DA3 depth map of the source image is encoded into a
    control latent and injected via the depth Control LoRA, so the render follows
    the source composition. Turbo uses the turbo checkpoint directly; RAW loads the
    base checkpoint + Turbo LoRA @0.6 (the reference workflow's default)."""
    src = getattr(req, "init_image_b64", "") or ""
    if not src:
        return _build_txt2img(g, req, seed)
    w, h = int(req.width), int(req.height)
    is_turbo = _is_turbo(req)

    # Depth control stacks LoRAs (control LoRA + optional Turbo LoRA). Load the
    # fp8_scaled checkpoint (~12GB) directly rather than the bf16 file (~24GB):
    # on a 24GB card the bf16 load peaks at the VRAM limit and the NVIDIA driver
    # spills into shared RAM (minutes/step). fp8_scaled + --reserve-vram keeps it
    # resident with headroom. fp8_e4m3fn_fast uses the 4090's native fp8 matmul.
    checkpoint = "turbo" if is_turbo else "raw"
    unet = _UNET_BY_QUANT[(checkpoint, "fp8")]
    model = g.add("UNETLoader", {"unet_name": unet, "weight_dtype": "fp8_e4m3fn_fast"})

    # RAW runs at turbo speed with the Turbo LoRA @0.6 (reference workflow default).
    if not is_turbo:
        model = g.add("Krea2WideLoraLoaderModelOnly",
                      {"model": _link(model), "lora_name": KREA2_TURBO_LORA_NAME,
                       "strength_model": 0.6})

    # User LoRAs (e.g. bypass/realism), same handling as the standard chain.
    for lora in (getattr(req, "loras", []) or []):
        if not isinstance(lora, dict) or not lora.get("enabled", True):
            continue
        name = lora.get("filename") or lora.get("name") or ""
        if not name:
            continue
        if not name.lower().endswith((".safetensors", ".ckpt", ".pt")):
            name = f"{name}.safetensors"
        try:
            strength = float(lora.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        if strength != strength or strength in (float("inf"), float("-inf")):
            strength = 1.0
        model = g.add("Krea2WideLoraLoaderModelOnly",
                      {"model": _link(model), "lora_name": name,
                       "strength_model": max(-40000.0, min(40000.0, strength))})

    clip = _build_clip(g)
    vae = _build_vae(g, req)

    # Depth Control LoRA loader (expands the input projection for the control tokens).
    ctrl_strength = float(getattr(req, "depth_control_strength", 1.0) or 1.0)
    ctrl_strength = max(-100.0, min(100.0, ctrl_strength))
    model = g.add("Krea2ControlLoRALoader",
                  {"model": _link(model), "lora_name": DEPTH_CONTROL_LORA_NAME,
                   "strength": ctrl_strength})

    # Depth map from the source image (estimator + resolution are user-selectable).
    src_img = _b64_to_loadimage(g, src)
    depth = _build_depth_map(g, src_img, req)

    latent = g.add("EmptySD3LatentImage",
                   {"width": w, "height": h,
                    "batch_size": max(1, int(getattr(req, "num_images", 1) or 1))})

    control = g.add("Krea2ControlImageEncode", {
        "control_image": _link(depth), "vae": _link(vae),
        "resize": "match_latent_size", "upscale_method": "lanczos", "crop": "center",
        # Depth LoRA was trained on grayscale Depth-Anything maps; grayscale matches
        # its training convention better than rgb (facok node's documented default).
        # invert flips the map when near objects render dark instead of white.
        "channel_mode": "grayscale", "normalize": "per_image_minmax",
        "invert": bool(getattr(req, "depth_invert", False)),
        "batch_mode": "independent_images", "latent": _link(latent),
    })
    model = g.add("Krea2ControlApply", {"model": _link(model), "control_latent": _link(control, 0)})

    # Flux shift (turbo frozen 1.15; the reference RAW workflow uses ~1.23).
    mu = getattr(req, "mu", None)
    if mu is None:
        mu = 1.15 if is_turbo else 1.23
    model = g.add("ModelSamplingFlux",
                  {"model": _link(model), "max_shift": float(mu), "base_shift": float(mu),
                   "width": w, "height": h})

    model, clip = _apply_model_features(g, req, model, clip, vae)
    positive = _build_positive(g, req, clip, seed)
    negative = _build_negative(g, req, clip)
    sampled = _ksampler(g, model, positive, negative, latent, req, seed, denoise=1.0)
    _finish(g, sampled, vae, req)


def _build_redraw(g: GraphBuilder, req: Any, seed: int) -> None:
    """Redraw Studio: the canvas + reference slots arrive as moodboard_images and
    condition generation (via _gather_refs). The NK2E task additionally routes the
    primary reference through the NK2E in-context edit node."""
    model, vae, positive, negative = _common(g, req, seed)
    refs = _gather_refs(req)
    if _uses_nk2e(req) and refs:
        scaled = _load_scaled_image(g, refs[0][0], req.width, req.height)
        rlat = g.add("VAEEncode", {"pixels": _link(scaled), "vae": _link(vae)})
        model = g.add("NK2EInContextEditNode", {"model": _link(model), "reference": _link(rlat)})
    latent = g.add("EmptySD3LatentImage",
                   {"width": int(req.width), "height": int(req.height),
                    "batch_size": max(1, int(getattr(req, "num_images", 1) or 1))})
    sampled = _ksampler(g, model, positive, negative, latent, req, seed, denoise=1.0)
    _finish(g, sampled, vae, req)


# ---------------------------------------------------------------------------
# Moodboard text injection (reuses the existing Python catalog helper)
# ---------------------------------------------------------------------------

def _apply_moodboard_text(req: Any) -> None:
    ids = list(getattr(req, "moodboard_ids", []) or [])
    uuids = list(getattr(req, "moodboard_uuids", []) or [])
    if not ids and not uuids:
        return
    try:
        from moodboards_catalog import moodboard_generation_context
        ctx = moodboard_generation_context(ids, moodboard_uuids=uuids)
    except Exception:
        logger.debug("moodboard_generation_context failed", exc_info=True)
        return
    style_text = (ctx.get("style_text") or "").strip()
    negative_text = (ctx.get("negative_text") or "").strip()
    if style_text:
        req.prompt = f"{req.prompt}\n{style_text}".strip()
    if negative_text:
        req.negative_prompt = f"{(getattr(req, 'negative_prompt', '') or '').strip()}\n{negative_text}".strip()


# ---------------------------------------------------------------------------
# Top-level build + run
# ---------------------------------------------------------------------------

def _resolve_seed(req: Any) -> int:
    seed = int(getattr(req, "seed", -1))
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)
    return seed


_STYLE_TRANSFER_METHODS = {"AdaIN", "WCT", "WCT2", "scattersort"}
_STYLE_TRANSFER_APPLY = {"denoised", "positive", "negative"}


def _build_style_transfer(g: GraphBuilder, req: Any, seed: int) -> None:
    """Training-free style transfer (RES4LYF): the style image's latent statistics
    are injected into the denoised latent via ClownGuide_Style_Beta, and sampled
    with ClownsharKSampler_Beta (which consumes the GUIDES). No model download."""
    model, vae, positive, negative = _common(g, req, seed)
    style_img = _load_scaled_image(g, req.style_transfer_image_b64, int(req.width), int(req.height))
    style_lat = g.add("VAEEncode", {"pixels": _link(style_img), "vae": _link(vae)})
    method = getattr(req, "style_transfer_method", "AdaIN")
    method = method if method in _STYLE_TRANSFER_METHODS else "AdaIN"
    apply_to = getattr(req, "style_transfer_apply_to", "denoised")
    apply_to = apply_to if apply_to in _STYLE_TRANSFER_APPLY else "denoised"
    guide = g.add("ClownGuide_Style_Beta", {
        "apply_to": apply_to, "method": method,
        "weight": float(getattr(req, "style_transfer_weight", 0.8) or 0.8), "synweight": 1.0,
        "weight_scheduler": "constant", "start_step": 0, "end_step": -1,
        "invert_mask": False, "guide": _link(style_lat),
    })
    latent = g.add("EmptySD3LatentImage",
                   {"width": int(req.width), "height": int(req.height),
                    "batch_size": max(1, int(getattr(req, "num_images", 1) or 1))})
    clown = (getattr(req, "res4lyf_sampler", "") or "").strip()
    if clown not in CLOWNSHARK_SAMPLERS:
        clown = "exponential/ddim"
    sampled = g.add("ClownsharKSampler_Beta", {
        "model": _link(model), "positive": _link(positive), "negative": _link(negative),
        "latent_image": _link(latent), "guides": _link(guide),
        "eta": float(getattr(req, "res4lyf_eta", 0.5) or 0.0), "sampler_name": clown,
        "scheduler": _norm_scheduler(req.scheduler), "steps": int(req.steps),
        "steps_to_run": -1, "denoise": 1.0, "cfg": _cfg(req), "seed": seed,
        "sampler_mode": "standard", "bongmath": bool(getattr(req, "res4lyf_bongmath", False)),
    })
    _finish(g, sampled, vae, req)


# Mr. Flow (RealRebelAI port of MrFlow, arXiv:2607.01642): training-free staged
# sampling. SR models live in ComfyUI/models/upscale_models.
MRFLOW_ESRGAN_X2 = "RealESRGAN_x2plus.pth"
MRFLOW_REMACRI_X4 = "4x_foolhardy_Remacri.pth"
_MRFLOW_PRESETS = {
    # {denoise, stage1_steps, refine_steps, cfg} — mirrors the Rebels node presets.
    "base_12plus1": {"denoise": 0.12, "stage1_steps": 12, "refine_steps": 1, "cfg": 4.0},
    "base_20plus1": {"denoise": 0.15, "stage1_steps": 20, "refine_steps": 1, "cfg": 4.0},
    "turbo_8plus1": {"denoise": 0.11, "stage1_steps": 8, "refine_steps": 1, "cfg": 1.0},
}


def _build_mrflow(g: GraphBuilder, req: Any, seed: int) -> None:
    """Mr. Flow staged sampling: cheap low-res base render -> pixel-space SR ->
    VAE re-encode -> 1-step model-native refine at the target size. The refine
    reuses the same Krea-2 model + conditioning, so it re-imprints the model's own
    detail on the SR output instead of leaving ESRGAN texture. width/height are the
    TARGET; the base render happens at target / SR-factor."""
    model, vae, positive, negative = _common(g, req, seed)

    tw, th = int(req.width), int(req.height)
    upscaler = (getattr(req, "mrflow_upscaler", "esrgan_x2") or "esrgan_x2").lower()
    if "remacri" in upscaler or "x4" in upscaler:
        factor, sr_model = 4.0, MRFLOW_REMACRI_X4
    else:
        factor, sr_model = 2.0, MRFLOW_ESRGAN_X2
    low_w, low_h = _grid16(round(tw / factor)), _grid16(round(th / factor))

    preset_name = (getattr(req, "mrflow_preset", "") or "").strip()
    if preset_name not in _MRFLOW_PRESETS:
        preset_name = "turbo_8plus1" if _is_turbo(req) else "base_12plus1"
    p = _MRFLOW_PRESETS[preset_name]

    # Refine strength is the knob that most changes the look: low = closer to the
    # ESRGAN pixels, high = more Krea-2 rework/detail. 0 keeps the preset default.
    refine_denoise = float(getattr(req, "mrflow_refine_denoise", 0.0) or 0.0)
    if refine_denoise <= 0.0:
        refine_denoise = p["denoise"]
    refine_denoise = max(0.02, min(0.6, refine_denoise))

    refine_steps = int(getattr(req, "mrflow_refine_steps", 0) or p["refine_steps"])
    refine_steps = max(1, min(3, refine_steps))

    # Upscale mode: when a source image is supplied, skip the base render and start
    # from that image (this is the "Upscale" workflow). Otherwise render at low res.
    src = (getattr(req, "init_image_b64", "") or "").strip()
    if src:
        base_img = _b64_to_loadimage(g, src)
    else:
        batch = max(1, int(getattr(req, "num_images", 1) or 1))
        latent = g.add("EmptySD3LatentImage", {"width": low_w, "height": low_h, "batch_size": batch})
        # Stage 1: low-res composition (euler/simple, full denoise) per the MrFlow recipe.
        stage1 = g.add("KSampler", {
            "model": _link(model), "seed": seed, "steps": int(p["stage1_steps"]),
            "cfg": float(p["cfg"]), "sampler_name": "euler", "scheduler": "simple",
            "positive": _link(positive), "negative": _link(negative),
            "latent_image": _link(latent), "denoise": 1.0,
        })
        base_img = g.add("VAEDecode", {"samples": _link(stage1), "vae": _link(vae)})

    # Stage 2+3: pixel-space SR then re-encode to the target latent.
    up_model = g.add("UpscaleModelLoader", {"model_name": sr_model})
    prepared = g.add("RebelsMrFlowUpscaleEncode", {
        "image": _link(base_img), "vae": _link(vae), "upscale_model": _link(up_model),
        "target_width": tw, "target_height": th, "resize_method": "bicubic",
    })

    # Stage 4: matched-noise refine through the base model at target size.
    refined = g.add("RebelsMrFlowKrea2Refine", {
        "model": _link(model), "vae": _link(vae), "positive": _link(positive),
        "negative": _link(negative), "latent_image": _link(prepared, 1), "seed": seed,
        "steps": refine_steps, "cfg": float(p["cfg"]), "sampler_name": "euler",
        "denoise": float(refine_denoise), "schedule": "linear", "print_schedule": False,
    })
    # Refine node outputs (refined_latent, refined_image); save the image (index 1).
    g.add("SaveImageWebsocket", {"images": _link(refined, 1)}, node_id=WS_IMAGE_NODE)


def _build_god_mode(g: GraphBuilder, req: Any, seed: int) -> None:
    """God Mode (max-quality, slow): 4-stage pipeline reproducing the reference
    workflow — Krea2 base render -> Z-Image Turbo refine (denoise 0.1) ->
    SeedVR2 7B-sharp upscale -> FaceDetailer. Runs as ONE ComfyUI graph and relies
    on ComfyUI's dynamic VRAM staging to swap the four models in/out on 24GB."""
    w, h = int(req.width), int(req.height)

    # --- Stage 1: Krea2 base (turbo fp8 + Krea2T-Enhancer + LoRAs) ---
    unet = _UNET_BY_QUANT[("turbo", "fp8")]
    model = g.add("UNETLoader", {"unet_name": unet, "weight_dtype": "fp8_e4m3fn_fast"})
    model = g.add("ComfyUI-Krea2T-Enhancer",
                  {"model": _link(model), "enabled": True, "strength": 1.0, "debug": False})
    for lora in (getattr(req, "loras", []) or []):
        if not isinstance(lora, dict) or not lora.get("enabled", True):
            continue
        name = lora.get("filename") or lora.get("name") or ""
        if not name:
            continue
        if not name.lower().endswith((".safetensors", ".ckpt", ".pt")):
            name = f"{name}.safetensors"
        try:
            strength = float(lora.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        if strength != strength or strength in (float("inf"), float("-inf")):
            strength = 1.0
        model = g.add("Krea2WideLoraLoaderModelOnly",
                      {"model": _link(model), "lora_name": name,
                       "strength_model": max(-40000.0, min(40000.0, strength))})
    krea_model = model  # reused by FaceDetailer

    clip = _build_clip(g)
    real_vae = g.add("VAELoader", {"vae_name": GODMODE_KREA_REAL_VAE})
    positive = _build_positive(g, req, clip, seed)
    negative = _build_negative(g, req, clip)
    latent = g.add("EmptySD3LatentImage", {"width": w, "height": h, "batch_size": 1})
    stage1 = g.add("KSampler", {
        "model": _link(krea_model), "seed": seed, "steps": 10, "cfg": 1.2,
        "sampler_name": "ddim", "scheduler": "beta57",
        "positive": _link(positive), "negative": _link(negative),
        "latent_image": _link(latent), "denoise": 1.0,
    })
    base_img = g.add("VAEDecode", {"samples": _link(stage1), "vae": _link(real_vae)})

    # --- Stage 2: Z-Image Turbo refine (denoise 0.1) ---
    z_unet = g.add("UNETLoader", {"unet_name": GODMODE_ZIMAGE_UNET, "weight_dtype": "default"})
    z_model = g.add("ModelSamplingAuraFlow", {"model": _link(z_unet), "shift": 5.0})
    z_clip = g.add("CLIPLoader", {"clip_name": GODMODE_ZIMAGE_CLIP, "type": "lumina2"})
    z_vae = g.add("VAELoader", {"vae_name": GODMODE_ZIMAGE_VAE})
    z_pos = g.add("CLIPTextEncode", {"text": "high quality, sharp focus, fine detail, clean", "clip": _link(z_clip)})
    z_neg = g.add("CLIPTextEncode", {"text": "blurry, ugly, low quality, artifacts", "clip": _link(z_clip)})
    z_lat = g.add("VAEEncode", {"pixels": _link(base_img), "vae": _link(z_vae)})
    stage2 = g.add("KSampler", {
        "model": _link(z_model), "seed": seed + 1, "steps": 20, "cfg": 1.0,
        "sampler_name": "dpmpp_sde", "scheduler": "sgm_uniform",
        "positive": _link(z_pos), "negative": _link(z_neg),
        "latent_image": _link(z_lat), "denoise": 0.10,
    })
    refined_img = g.add("VAEDecode", {"samples": _link(stage2), "vae": _link(z_vae)})

    # --- Stage 3: SeedVR2 7B-sharp tiling upscale (exact workflow node + settings) ---
    dit = g.add("SeedVR2LoadDiTModel", {
        "model": GODMODE_SEEDVR2_MODEL, "device": "cuda:0", "blocks_to_swap": 16,
        "swap_io_components": True, "offload_device": "cpu", "cache_model": False,
        "attention_mode": "sdpa",
    })
    svae = g.add("SeedVR2LoadVAEModel", {
        "model": "ema_vae_fp16.safetensors", "device": "cuda:0",
        "encode_tiled": True, "encode_tile_size": 1024, "encode_tile_overlap": 128,
        "decode_tiled": True, "decode_tile_size": 1024, "decode_tile_overlap": 128,
        "tile_debug": "false", "offload_device": "cpu", "cache_model": False,
    })
    up = g.add("SeedVR2TilingUpscaler", {
        "image": _link(refined_img), "dit": _link(dit), "vae": _link(svae),
        "seed": 100, "new_resolution": 4096, "tile_width": 1024, "tile_height": 1024,
        "mask_blur": 0, "tile_padding": 64, "tile_upscale_resolution": 2048,
        "tiling_strategy": "Chess", "anti_aliasing_strength": 0.0,
        "blending_method": "auto", "color_correction": "lab",
    })

    # --- Stage 4: FaceDetailer (Krea2 model, face_yolov8m) ---
    detector = g.add("UltralyticsDetectorProvider", {"model_name": GODMODE_FACE_DETECTOR})
    face = g.add("FaceDetailer", {
        "image": _link(up), "model": _link(krea_model), "clip": _link(clip), "vae": _link(real_vae),
        "positive": _link(positive), "negative": _link(negative), "bbox_detector": _link(detector),
        "guide_size": 1024.0, "guide_size_for": True, "max_size": 1536.0, "seed": seed + 2,
        "steps": 10, "cfg": 1.2, "sampler_name": "ddim", "scheduler": "beta57", "denoise": 0.45,
        "feather": 5, "noise_mask": True, "force_inpaint": True, "bbox_threshold": 0.3,
        "bbox_dilation": 10, "bbox_crop_factor": 3.0, "sam_detection_hint": "center-1",
        "sam_dilation": 0, "sam_threshold": 0.93, "sam_bbox_expansion": 0,
        "sam_mask_hint_threshold": 0.7, "sam_mask_hint_use_negative": "False", "drop_size": 10,
        "wildcard": "", "cycle": 1, "inpaint_model": False, "noise_mask_feather": 20,
        "tiled_encode": False, "tiled_decode": False,
    })
    g.add("SaveImageWebsocket", {"images": _link(face, 0)}, node_id=WS_IMAGE_NODE)


def _build_turbo_4x(g: GraphBuilder, req: Any, seed: int) -> None:
    """Run the community 'Krea 2 Turbo 4X' workflow verbatim.

    The multi-stage graph (OTU W8A8 + ClownsharK + Impact refine ladder +
    LatentPixelScale 4X + VAEDeGrid) is stored as a pre-converted API prompt in
    workflows/turbo_4x_api.json. We only inject the editable bits (prompt,
    negative, seed) and hand the graph straight to ComfyUI -- nothing about the
    exact node wiring changes."""
    import json
    from pathlib import Path

    template = Path(__file__).resolve().parent / "workflows" / "turbo_4x_api.json"
    if not template.exists():
        raise RuntimeError(
            "Turbo 4X workflow template is missing. Run scripts/_convert_turbo4x.py "
            "with ComfyUI running to generate backend/workflows/turbo_4x_api.json."
        )
    graph = json.loads(template.read_text(encoding="utf-8"))

    prompt = getattr(req, "prompt", "") or ""
    negative = getattr(req, "negative_prompt", "") or ""
    # Node ids are stable from the source workflow: 6 = positive, 5 = negative.
    if "6" in graph and graph["6"].get("class_type") == "CLIPTextEncode":
        graph["6"]["inputs"]["text"] = prompt
    if negative and "5" in graph and graph["5"].get("class_type") == "CLIPTextEncode":
        graph["5"]["inputs"]["text"] = negative
    # Reseed every sampler so seed changes actually vary the output.
    for node in graph.values():
        for key in ("seed", "noise_seed"):
            if key in node.get("inputs", {}):
                node["inputs"][key] = int(seed)
        # The source WF pins a specific SageAttention triton kernel that isn't in
        # every sageattention build. 'auto' picks the best kernel actually present
        # (the triton one when available), so the graph is portable across installs.
        if node.get("class_type") == "PathchSageAttentionKJ":
            node["inputs"]["sage_attention"] = "auto"
        # Load the pre-quantized int8-convrot Turbo checkpoint (13.5GB) instead of
        # the full bf16 (26GB) the WF shipped. Same OTU loader, same convrot-int8
        # compute, ~half the resident footprint -- this is what lets the 4X pipeline
        # fit in high-VRAM mode. on_the_fly_quantization stays off (already int8).
        if node.get("class_type") == "OTUNetLoaderW8A8":
            node["inputs"]["unet_name"] = "krea2_turbo_int8_convrot.safetensors"
            node["inputs"]["on_the_fly_quantization"] = False
        # Pair it with the fp8 encoder (5.2GB vs bf16 8.9GB); the encoder is pinned
        # for the whole run under --highvram, so this frees the remaining headroom.
        if node.get("class_type") == "CLIPLoader" \
                and node.get("inputs", {}).get("clip_name") == "qwen3vl_4b_bf16.safetensors":
            node["inputs"]["clip_name"] = KREA2_STOCK_CLIP_NAME

    g.nodes = graph


_MODE_BUILDERS = {
    "txt2img": _build_txt2img,
    "img2img": _build_img2img,
    "inpaint": _build_inpaint,
    "outpaint": _build_outpaint,
    "redraw": _build_redraw,
    "character_edit": _build_character_edit,
    "turbo_4x": _build_turbo_4x,
}


def build_graph(req: Any) -> tuple[dict, dict]:
    g = GraphBuilder()
    seed = _resolve_seed(req)
    # Depth ControlNet overrides the mode: DA3 depth of the source drives a
    # control-latent-guided render (distinct pipeline, its own model chain).
    if bool(getattr(req, "depth_control", False)):
        _build_depth_control(g, req, seed)
        return g.graph(), {"seed": seed, "lora_reports": _lora_reports(req)}
    # Style transfer (RES4LYF) overrides the mode: it's a distinct pipeline.
    if (getattr(req, "style_transfer_image_b64", "") or "").strip():
        _build_style_transfer(g, req, seed)
        return g.graph(), {"seed": seed, "lora_reports": _lora_reports(req)}
    # God Mode overrides the mode: 4-stage max-quality pipeline.
    if bool(getattr(req, "god_mode", False)):
        _build_god_mode(g, req, seed)
        return g.graph(), {"seed": seed, "lora_reports": _lora_reports(req)}
    # Mr. Flow staged sampling overrides the mode: low-res gen + SR + refine.
    if bool(getattr(req, "mrflow", False)):
        _build_mrflow(g, req, seed)
        return g.graph(), {"seed": seed, "lora_reports": _lora_reports(req)}
    mode = (getattr(req, "mode", "txt2img") or "txt2img").lower()
    builder = _MODE_BUILDERS.get(mode, _build_txt2img)
    builder(g, req, seed)
    return g.graph(), {"seed": seed, "lora_reports": _lora_reports(req)}


def _lora_reports(req: Any) -> list[dict]:
    reports = []
    if (getattr(req, "mode", "") or "").lower() == "character_edit":
        reports.append({"name": KREA2_IDENTITY_EDIT_LORA, "applied": True})
    for lora in (getattr(req, "loras", []) or []):
        if not isinstance(lora, dict) or not lora.get("enabled", True):
            continue
        reports.append({"name": lora.get("name") or lora.get("filename"), "applied": True})
    return reports


def _grid16(v: int) -> int:
    v = int(v)
    return max(16, v - (v % 16))


_USDU_TILE_MODE = {"linear": "Linear", "chess": "Chess"}
_USDU_SEAM_MODE = {"none": "None", "band_pass": "Band Pass",
                   "half_tile": "Half Tile", "half_tile_intersections": "Half Tile + Intersections"}


def comfy_upscale(method: str, image_b64: str, *, prompt: str = "", upscale_by: float = 2.0,
                  denoise: float = 0.24, steps: int = 8, cfg: float = 1.0,
                  sampler: str = "euler", scheduler: str = "simple",
                  tile_width: int = 1024, tile_height: int = 1024, tile_padding: int = 96,
                  mask_blur: int = 12, seam_mode: str = "band_pass", tile_mode: str = "chess",
                  tiled_decode: bool = False, seedvr2_model: str = "",
                  prompt_id_cb: PromptIdCb = None) -> Image.Image:
    """Model/VAE-dependent upscales routed through ComfyUI (the native pipeline
    is not loaded in Comfy mode). Returns a PIL image. realesrgan and pid keep
    their standalone Python implementations in main.py."""
    from types import SimpleNamespace

    # The spacepxl Wan-2.1 2x "imageonly" VAE decodes to a non-RGB tensor under
    # ComfyUI's stock VAEDecode, so wan_vae_2x uses a Qwen-VAE tiled 2x
    # round-trip here (documented parity delta).
    if method == "wan_vae_2x":
        method = "tiled_vae"
        upscale_by = 2.0

    src = Image.open(io.BytesIO(base64.b64decode(image_b64.split(",")[-1])))
    w, h = src.size
    tw, th = _grid16(round(w * float(upscale_by))), _grid16(round(h * float(upscale_by)))

    g = GraphBuilder()

    if method == "seedvr2":
        # SeedVR2: standalone restorer/upscaler (its own DiT+VAE), the community's
        # top upscaler. It must NOT co-reside with Krea in VRAM, so we free
        # ComfyUI's models first, then run SeedVR2 alone (batch 1, single image).
        free_comfy_vram(unload_models=True, free_memory=True)
        # `resolution` is SeedVR2's target for the SHORT edge (it keeps aspect).
        # Cap at 4096 so 4K upscales are allowed on a 24GB card.
        target = min(4096, _grid16(round(min(w, h) * float(upscale_by))))
        try:
            from settings import settings as _svr_settings
            _svr_default = getattr(_svr_settings, "seedvr2_model", "3b")
        except Exception:
            _svr_default = "3b"
        choice = (seedvr2_model or _svr_default or "3b").lower()
        use_7b = "7b" in choice
        dit_model = "seedvr2_ema_7b_fp16.safetensors" if use_7b else "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
        # Auto-enable block-swap + VAE tiling as the target climbs, so 4K fits on
        # 24GB after ComfyUI is freed. 7B fp16 is much heavier -> swap aggressively.
        high = target >= 3072                      # ~3K+ short edge (i.e. 4K-class output)
        very_high = target >= 3840
        if use_7b:
            blocks = 36 if high else 20            # 7B: 0-36 blocks
        else:
            blocks = 20 if very_high else (12 if high else 0)   # 3B: 0-32 blocks
        dit_off = "cpu" if blocks > 0 else "none"
        tiled = high or use_7b                     # tiled VAE encode/decode for large frames
        loaded = _b64_to_loadimage(g, image_b64)
        dit = g.add("SeedVR2LoadDiTModel", {
            "model": dit_model, "device": "cuda:0",
            "blocks_to_swap": int(blocks),
            "swap_io_components": bool(use_7b and high),
            "offload_device": dit_off,
            "attention_mode": "sdpa",
        })
        svae = g.add("SeedVR2LoadVAEModel", {
            "model": "ema_vae_fp16.safetensors", "device": "cuda:0",
            "encode_tiled": bool(tiled), "encode_tile_size": 1024, "encode_tile_overlap": 128,
            "decode_tiled": bool(tiled), "decode_tile_size": 1024, "decode_tile_overlap": 128,
        })
        up = g.add("SeedVR2VideoUpscaler", {
            "image": _link(loaded), "dit": _link(dit), "vae": _link(svae), "seed": 0,
            "resolution": int(target), "max_resolution": 0, "batch_size": 1,
            "uniform_batch_size": False, "color_correction": "lab",
            "offload_device": "cpu" if (high or use_7b) else "none",
        })
        g.add("SaveImageWebsocket", {"images": _link(up)}, node_id=WS_IMAGE_NODE)
        blobs = ComfyClient().run(g.graph(), timeout=3600, prompt_id_cb=prompt_id_cb)
        if not blobs:
            raise RuntimeError("SeedVR2 upscale returned no image.")
        return Image.open(io.BytesIO(blobs[0])).convert("RGB")

    if method == "esrgan":
        # Pure SR-model upscale (no DiT/VAE): RealESRGAN x2 or Remacri x4 from
        # ComfyUI's upscale_models, exact-scaled to the requested factor.
        sr_model = MRFLOW_REMACRI_X4 if float(upscale_by) >= 3.0 else MRFLOW_ESRGAN_X2
        loaded = _b64_to_loadimage(g, image_b64)
        up_model = g.add("UpscaleModelLoader", {"model_name": sr_model})
        upscaled = g.add("ImageUpscaleWithModel", {"upscale_model": _link(up_model), "image": _link(loaded)})
        exact = g.add("ImageScale", {
            "image": _link(upscaled), "upscale_method": "lanczos",
            "width": int(round(w * float(upscale_by))), "height": int(round(h * float(upscale_by))),
            "crop": "disabled",
        })
        g.add("SaveImageWebsocket", {"images": _link(exact)}, node_id=WS_IMAGE_NODE)
        blobs = ComfyClient().run(g.graph(), timeout=600, prompt_id_cb=prompt_id_cb)
        if not blobs:
            raise RuntimeError("ESRGAN upscale returned no image.")
        return Image.open(io.BytesIO(blobs[0])).convert("RGB")

    vae = _build_vae(g, None)

    if method == "tiled_vae":
        init = _load_scaled_image(g, image_b64, tw, th)
        enc = g.add("VAEEncode", {"pixels": _link(init), "vae": _link(vae)})
        dec = g.add("VAEUtils_VAEDecodeTiled",
                    {"samples": _link(enc), "vae": _link(vae), "upscale": 1, "tile": True,
                     "tile_size": 512, "overlap": 64, "temporal_size": 8, "temporal_overlap": 4})
        g.add("SaveImageWebsocket", {"images": _link(dec)}, node_id=WS_IMAGE_NODE)
    elif method == "ultimate":
        # Community-recommended: tile upscale at low denoise (Ultimate SD Upscale).
        # Pre-scale to target, then tiled img2img refine per tile with seam fixing.
        ns = SimpleNamespace(checkpoint="turbo", quantization="fp8", diffusion_engine="native_pytorch",
                             loras=[], mu=None, width=tw, height=th, model_profile="",
                             quality_preset="", checkpoint_path="")
        model = _build_model_chain(g, ns)
        clip = _build_clip(g)
        pos = g.add("TextEncodeKrea2", {"clip": clip, "prompt": prompt or "high detail, sharp focus, crisp texture"})
        neg = g.add("TextEncodeKrea2", {"clip": clip, "prompt": ""})
        scaled = _load_scaled_image(g, image_b64, tw, th)
        usdu = g.add("UltimateSDUpscaleNoUpscale", {
            "upscaled_image": _link(scaled), "model": _link(model), "positive": _link(pos),
            "negative": _link(neg), "vae": _link(vae), "seed": 0, "steps": int(steps),
            "cfg": max(1.0, float(cfg)), "sampler_name": _norm_sampler(sampler),
            "scheduler": _norm_scheduler(scheduler), "denoise": min(0.5, max(0.05, float(denoise))),
            "mode_type": _USDU_TILE_MODE.get(tile_mode, "Chess"),
            "tile_width": int(tile_width), "tile_height": int(tile_height),
            "mask_blur": int(mask_blur), "tile_padding": int(tile_padding),
            "seam_fix_mode": _USDU_SEAM_MODE.get(seam_mode, "Band Pass"),
            "seam_fix_denoise": 1.0, "seam_fix_width": 64, "seam_fix_mask_blur": 8,
            "seam_fix_padding": 16, "force_uniform_tiles": True,
            "tiled_decode": bool(tiled_decode), "batch_size": 1,
        })
        g.add("SaveImageWebsocket", {"images": _link(usdu)}, node_id=WS_IMAGE_NODE)
    else:  # model_refine, refine_2pass -> turbo img2img refine at target res
        ns = SimpleNamespace(checkpoint="turbo", quantization="fp8", diffusion_engine="native_pytorch",
                             loras=[], mu=None, width=tw, height=th, model_profile="",
                             quality_preset="", checkpoint_path="")
        model = _build_model_chain(g, ns)
        clip = _build_clip(g)
        pos = g.add("TextEncodeKrea2", {"clip": clip, "prompt": prompt or "high detail, sharp focus, crisp texture"})
        neg = g.add("TextEncodeKrea2", {"clip": clip, "prompt": ""})
        init = _load_scaled_image(g, image_b64, tw, th)
        lat = g.add("VAEEncode", {"pixels": _link(init), "vae": _link(vae)})
        sreq = SimpleNamespace(steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler)
        sampled = _ksampler(g, model, pos, neg, lat, sreq, 0, denoise=float(denoise))
        if method == "refine_2pass":
            sampled = _ksampler(g, model, pos, neg, sampled, sreq, 1, denoise=0.15)
        _finish(g, sampled, vae, req)

    blobs = ComfyClient().run(g.graph(), prompt_id_cb=prompt_id_cb)
    if not blobs:
        raise RuntimeError("ComfyUI upscale returned no image.")
    return Image.open(io.BytesIO(blobs[0])).convert("RGB")


def comfy_depth_preview(image_b64: str, *, estimator: str = "da3",
                        resolution: int = 504, invert: bool = False,
                        prompt_id_cb: PromptIdCb = None) -> Image.Image:
    """Run only the depth estimator on a source image and return the depth map,
    so the UI can preview exactly what the depth ControlNet will follow."""
    from types import SimpleNamespace

    g = GraphBuilder()
    src = _b64_to_loadimage(g, image_b64)
    req = SimpleNamespace(depth_estimator=estimator, depth_resolution=resolution)
    depth = _build_depth_map(g, src, req)
    if invert:
        depth = g.add("ImageInvert", {"image": _link(depth)})
    g.add("SaveImageWebsocket", {"images": _link(depth)}, node_id=WS_IMAGE_NODE)
    blobs = ComfyClient().run(g.graph(), timeout=300, prompt_id_cb=prompt_id_cb)
    if not blobs:
        raise RuntimeError("Depth preview returned no image.")
    return Image.open(io.BytesIO(blobs[0])).convert("RGB")


def _export_prompt_graph(graph: dict, seed: Optional[int]) -> dict:
    """Return a copy of the API graph suitable for embedding in a saved PNG so the
    image can be dragged into ComfyUI and regenerated.

    - Swaps the API-only ``SaveImageWebsocket`` sink for a real ``SaveImage`` so a
      dropped workflow writes a visible output instead of silently needing a WS
      client.
    - Pins ``batch_size`` to 1 and forces every sampler seed to this image's seed
      so the drop reproduces *this* image, not the whole batch."""
    g = copy.deepcopy(graph)
    for nid, node in list(g.items()):
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == "SaveImageWebsocket":
            images = node.get("inputs", {}).get("images")
            g[nid] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "krea2", "images": images}}
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if seed is not None:
            for key in ("seed", "noise_seed"):
                if key in inputs and isinstance(inputs[key], (int, float)):
                    inputs[key] = int(seed)
        if node.get("class_type") in ("EmptySD3LatentImage", "EmptyLatentImage") and "batch_size" in inputs:
            inputs["batch_size"] = 1
    return g


def comfy_generate(
    req: Any,
    progress_cb=None,
    save_outputs: bool = True,
    username: str | None = None,
    prompt_id_cb: PromptIdCb = None,
    output_file_cb=None,
):
    """Run a GenerationRequest through ComfyUI.

    Returns (results_b64, seed, filenames, lora_reports, metadata)."""
    _apply_moodboard_text(req)

    # Auto RAW-4K: native RAW at 4K softens (its VAE detail budget runs out), so we
    # render at 2K then hand off to SeedVR2 for a sharp 2x -> 4K. Turbo keeps native
    # 4K (it holds up fine). Batch is already clamped to 1 at 4K upstream.
    _tw = int(getattr(req, "width", 0) or 0)
    _th = int(getattr(req, "height", 0) or 0)
    # Mr. Flow does its own SR+refine, so it opts out of the SeedVR2 auto-4K path.
    auto_4k = (not _is_turbo(req)) and max(_tw, _th) >= 2560 and not bool(getattr(req, "mrflow", False)) and not bool(getattr(req, "god_mode", False))
    if auto_4k:
        req.width = _grid16(max(512, _tw // 2))
        req.height = _grid16(max(512, _th // 2))

    graph, runtime = build_graph(req)
    seed = runtime["seed"]

    client = ComfyClient()
    blobs = client.run(graph, progress_cb=progress_cb, prompt_id_cb=prompt_id_cb)
    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in blobs]
    if not images:
        raise RuntimeError("ComfyUI returned no images.")

    if auto_4k:
        upscaled: list = []
        for im in images:
            _buf = io.BytesIO()
            im.save(_buf, format="PNG")
            _b64 = base64.b64encode(_buf.getvalue()).decode()
            upscaled.append(
                comfy_upscale(
                    "seedvr2",
                    _b64,
                    upscale_by=2.0,
                    prompt_id_cb=prompt_id_cb,
                )
            )
        images = upscaled
        req.width, req.height = _tw, _th  # so saved metadata reflects the 4K result

    unet, _wd, _is_gguf, gguf_name = resolve_unet(req)
    if build_generation_metadata is not None:
        metadata = [
            build_generation_metadata(
                req, base_seed=seed, image_index=i, filename="", resolved_provider="comfyui",
                runtime={"provider": "comfyui", "sampler": _norm_sampler(req.sampler),
                         "scheduler": _norm_scheduler(req.scheduler)},
                model_runtime={"unet": gguf_name or unet, "clip": KREA2_CLIP_NAME, "vae": _vae_name(),
                               "engine": getattr(req, "diffusion_engine", ""),
                               "quantization": getattr(req, "quantization", "")},
            )
            for i in range(len(images))
        ]
    else:  # pragma: no cover
        metadata = [{"seed": seed + i} for i in range(len(images))]

    comfy_graphs = [_export_prompt_graph(graph, seed + i) for i in range(len(images))]
    results, filenames = encode_images(images, OUTPUTS_DIR, save_outputs=save_outputs,
                                       metadata=metadata, comfy_graphs=comfy_graphs, subdir=username,
                                       output_file_cb=output_file_cb)
    metadata = [{**item, "filename": filenames[i] if i < len(filenames) else item.get("filename", "")}
                for i, item in enumerate(metadata)]
    return results, seed, filenames, runtime.get("lora_reports", []), metadata
