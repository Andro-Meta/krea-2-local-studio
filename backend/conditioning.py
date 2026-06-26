"""ConditioningKrea2Rebalance — port from nova452/ComfyUI-ConditioningKrea2Rebalance.

Krea 2 conditioning: (B, seq, 12*2560) — 12 Qwen3-VL layer taps concatenated.
Reshape → apply per-layer gain → flatten → global multiplier.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor

DEFAULT_LAYER_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0]
DEFAULT_MULTIPLIER = 4.0
N_LAYERS = 12
LAYER_DIM = 2560  # Qwen3-VL hidden dim


def rebalance(
    txt: "Tensor",
    multiplier: float = DEFAULT_MULTIPLIER,
    layer_weights: list[float] | None = None,
) -> "Tensor":
    import torch

    if layer_weights is None:
        layer_weights = DEFAULT_LAYER_WEIGHTS

    orig_dtype = txt.dtype
    gains = torch.tensor(layer_weights, dtype=torch.float32, device=txt.device)

    # New 4D layout from the encoder: (B, seq, 12, 2560) — apply per-tap gain
    # on the layer axis (-2) directly.
    if txt.ndim == 4 and txt.shape[-2] == N_LAYERS:
        t = txt.float() * gains.view(1, 1, N_LAYERS, 1)
        return (t * multiplier).to(orig_dtype)

    # Legacy 3D flattened layout: (B, seq, 12*2560)
    flat = txt.shape[-1]
    if flat != N_LAYERS * LAYER_DIM:
        return txt * multiplier
    t = txt.float().view(*txt.shape[:-1], N_LAYERS, LAYER_DIM)
    t = t * gains.view(1, 1, N_LAYERS, 1)
    t = t.view(*t.shape[:-2], flat)
    return (t * multiplier).to(orig_dtype)


def parse_weights(weights_str: str) -> list[float]:
    try:
        vals = [float(x.strip()) for x in weights_str.replace(";", ",").split(",") if x.strip()]
        if len(vals) == N_LAYERS:
            return vals
    except ValueError:
        pass
    return DEFAULT_LAYER_WEIGHTS
