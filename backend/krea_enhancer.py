from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from typing import Any

import torch

KREA2_TAP_LAYERS = 12
KREA2_TAP_DIM = 2560
KREA2_CHUNK_COUNT = 24
KREA2_CHUNK_DIM = 1280

ENHANCER_PROFILE_12 = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0)
ENHANCER_CHUNK_PROFILE = ENHANCER_PROFILE_12 + ENHANCER_PROFILE_12
ENHANCER_GLOBAL_MULTIPLIER = 15.0
TXTFUSION_TOKEN_REL_CAP = 0.75


def _bounded_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not math.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _is_krea2_model(model: Any) -> bool:
    config = getattr(model, "config", None)
    txtlayers = getattr(model, "txtlayers", getattr(config, "txtlayers", 0))
    txtdim = getattr(model, "txtdim", getattr(config, "txtdim", 0))
    return (
        hasattr(model, "txtfusion")
        and hasattr(model, "txtmlp")
        and hasattr(model, "blocks")
        and hasattr(model, "_unpack_context")
        and int(txtlayers) == KREA2_TAP_LAYERS
        and int(txtdim) == KREA2_TAP_DIM
    )


def _run_txtfusion_parts(txtfusion: Any, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if not all(hasattr(txtfusion, attr) for attr in ("layerwise_blocks", "projector", "refiner_blocks")):
        return txtfusion._krea_enhancer_original_forward(x, mask=mask)

    b, seq, taps, dim = x.shape
    y = x.reshape(b * seq, taps, dim)
    for block in txtfusion.layerwise_blocks:
        y = block(y.contiguous(), mask=None)
    tap_mix = y.reshape(b, seq, taps, dim).permute(0, 1, 3, 2).contiguous()
    out = txtfusion.projector(tap_mix).squeeze(-1)
    for block in txtfusion.refiner_blocks:
        out = block(out, mask=mask)
    return out


def _chunk_gains(device: torch.device, dtype: torch.dtype, strength: float) -> torch.Tensor:
    base = torch.tensor(ENHANCER_CHUNK_PROFILE, device=device, dtype=torch.float32)
    gains = 1.0 + float(strength) * (base - 1.0)
    return gains.to(dtype=dtype)


def _enhanced_forward(txtfusion: Any, x: torch.Tensor, mask: torch.Tensor | None, strength: float) -> torch.Tensor:
    if x.ndim != 4 or x.shape[2] != KREA2_TAP_LAYERS or x.shape[3] != KREA2_TAP_DIM:
        return txtfusion._krea_enhancer_original_forward(x, mask=mask)

    reference = _run_txtfusion_parts(txtfusion, x, mask=mask)
    gains = _chunk_gains(x.device, x.dtype, strength)
    global_multiplier = 1.0 + float(strength) * (ENHANCER_GLOBAL_MULTIPLIER - 1.0)
    scaled_x = (
        x.reshape(x.shape[0], x.shape[1], KREA2_CHUNK_COUNT, KREA2_CHUNK_DIM)
        * gains.view(1, 1, KREA2_CHUNK_COUNT, 1)
        * global_multiplier
    ).reshape_as(x)
    candidate = _run_txtfusion_parts(txtfusion, scaled_x, mask=mask)

    post_delta = candidate.detach().float() - reference.detach().float()
    token_base_rms = torch.sqrt(torch.mean(reference.detach().float() ** 2, dim=-1, keepdim=True)).clamp_min(1e-8)
    token_delta_rms = torch.sqrt(torch.mean(post_delta ** 2, dim=-1, keepdim=True)).clamp_min(1e-8)
    token_rel = token_delta_rms / token_base_rms
    token_scale = (TXTFUSION_TOKEN_REL_CAP / token_rel).clamp(max=1.0)
    return (reference.detach().float() + post_delta * token_scale).to(candidate.dtype)


@contextlib.contextmanager
def krea_enhancer_context(model: Any, *, enabled: bool, strength: float = 1.0) -> Iterator[None]:
    strength = _bounded_float(strength, 1.0, 0.0, 2.0)
    if not enabled or strength == 0.0 or not _is_krea2_model(model):
        yield
        return

    txtfusion = model.txtfusion
    original_forward = txtfusion.forward

    def enhanced_forward(x: torch.Tensor, mask: torch.Tensor | None = None):
        txtfusion._krea_enhancer_original_forward = original_forward
        try:
            return _enhanced_forward(txtfusion, x, mask=mask, strength=strength)
        finally:
            if hasattr(txtfusion, "_krea_enhancer_original_forward"):
                delattr(txtfusion, "_krea_enhancer_original_forward")

    txtfusion.forward = enhanced_forward
    try:
        yield
    finally:
        txtfusion.forward = original_forward
