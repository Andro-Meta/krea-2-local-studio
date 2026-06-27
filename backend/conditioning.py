"""ConditioningKrea2Rebalance — port from nova452/ComfyUI-ConditioningKrea2Rebalance.

Krea 2 conditioning: (B, seq, 12*2560) — 12 Qwen3-VL layer taps concatenated.
Reshape → apply per-layer gain → flatten → global multiplier.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor

DEFAULT_LAYER_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0]
PRESET_LAYER_WEIGHTS = {
    "legacy": DEFAULT_LAYER_WEIGHTS,
    "balanced": DEFAULT_LAYER_WEIGHTS,
    "detail": [0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.2, 3.0, 6.0, 1.5, 5.0, 1.2],
    "subtle": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 1.0, 1.5, 1.0],
    "uniform": [1.0] * 12,
}
DEFAULT_MULTIPLIER = 1.0
N_LAYERS = 12
LAYER_DIM = 2560  # Qwen3-VL hidden dim
EDIT_REBALANCE_PROFILES = {
    "default": {"text_start": 0.0, "text_end": 0.65, "image_start": 0.35, "image_end": 1.0},
    "edit": {"text_start": 0.0, "text_end": 0.6, "image_start": 0.3, "image_end": 1.0},
    "conservative": {"text_start": 0.0, "text_end": 0.8, "image_start": 0.5, "image_end": 1.0},
}


def _unit_norm_dim(txt: "Tensor", dim: int = -1, eps: float = 1e-6) -> "Tensor":
    import torch

    return txt / torch.clamp(txt.float().norm(dim=dim, keepdim=True), min=eps).to(txt.dtype)


def scale_conditioning(
    txt: "Tensor",
    multiplier: float = DEFAULT_MULTIPLIER,
    layer_weights: list[float] | None = None,
) -> "Tensor":
    import torch

    if layer_weights is None or len(layer_weights) != N_LAYERS:
        layer_weights = DEFAULT_LAYER_WEIGHTS

    orig_dtype = txt.dtype
    gains = torch.tensor(layer_weights, dtype=torch.float32, device=txt.device)

    if txt.ndim == 4 and txt.shape[-2] == N_LAYERS:
        t = txt.float() * gains.view(1, 1, N_LAYERS, 1)
        return (t * multiplier).to(orig_dtype)

    flat = txt.shape[-1]
    if flat != N_LAYERS * LAYER_DIM:
        return (txt.float() * float(multiplier)).to(orig_dtype)
    t = txt.float().view(*txt.shape[:-1], N_LAYERS, LAYER_DIM)
    t = t * gains.view(1, 1, N_LAYERS, 1)
    t = t.view(*t.shape[:-2], flat)
    return (t * multiplier).to(orig_dtype)


def rms_renormalize_conditioning(source: "Tensor", weighted: "Tensor", eps: float = 1e-6) -> "Tensor":
    import torch

    source_rms = torch.clamp(source.float().pow(2).mean().sqrt(), min=eps)
    weighted_rms = torch.clamp(weighted.float().pow(2).mean().sqrt(), min=eps)
    return (weighted.float() * (source_rms / weighted_rms)).to(weighted.dtype)


def guidance_conditioning(base: "Tensor", guide: "Tensor", scale: float = 1.0) -> "Tensor":
    """Project guide onto normalized base direction and add a bounded contrastive delta."""
    orig_dtype = base.dtype
    base_f = base.float()
    guide_f = guide.float()
    direction = _unit_norm_dim(base_f, dim=-1).float()
    projection = (guide_f * direction).sum(dim=-1, keepdim=True) * direction
    delta = guide_f - projection
    return (base_f + delta * float(scale)).to(orig_dtype)


def split_schedule_weights(position: float, profile: str = "default") -> tuple[float, float]:
    cfg = EDIT_REBALANCE_PROFILES.get(profile, EDIT_REBALANCE_PROFILES["default"])
    position = float(position)
    if position <= cfg["image_start"]:
        return 1.0, 0.0
    if position >= cfg["text_end"]:
        return 0.0, 1.0
    span = max(cfg["text_end"] - cfg["image_start"], 1e-6)
    image = min(1.0, max(0.0, (position - cfg["image_start"]) / span))
    text = 1.0 - image
    return round(text, 6), round(image, 6)


def blend_split_conditioning(
    text_txt: "Tensor",
    image_txt: "Tensor",
    *,
    position: float = 0.5,
    profile: str = "default",
) -> "Tensor":
    text_w, image_w = split_schedule_weights(position, profile)
    total = max(text_w + image_w, 1e-6)
    return ((text_txt.float() * text_w + image_txt.float() * image_w) / total).to(text_txt.dtype)


def resolve_rebalance_weights(preset: str = "balanced", weights_str: str = "") -> list[float]:
    preset = str(preset or "balanced").strip().lower()
    if preset == "custom":
        return parse_weights(weights_str)
    return PRESET_LAYER_WEIGHTS.get(preset, PRESET_LAYER_WEIGHTS["balanced"])


def rebalance(
    txt: "Tensor",
    multiplier: float = DEFAULT_MULTIPLIER,
    layer_weights: list[float] | None = None,
    *,
    preset: str = "balanced",
    weights_str: str = "",
    mode: str = "rms_renorm",
    renormalize: bool = True,
) -> "Tensor":
    if layer_weights is None:
        layer_weights = resolve_rebalance_weights(preset, weights_str)
    weighted = scale_conditioning(txt, multiplier=1.0, layer_weights=layer_weights)
    if str(mode or "rms_renorm") == "legacy_multiply":
        return scale_conditioning(txt, multiplier=multiplier, layer_weights=layer_weights)
    if renormalize:
        weighted = rms_renormalize_conditioning(txt, weighted)
    return (weighted.float() * float(multiplier)).to(txt.dtype)


def parse_weights(weights_str: str) -> list[float]:
    try:
        vals = [float(x.strip()) for x in weights_str.replace(";", ",").split(",") if x.strip()]
        if len(vals) == N_LAYERS:
            return vals
    except ValueError:
        return DEFAULT_LAYER_WEIGHTS
    return DEFAULT_LAYER_WEIGHTS
