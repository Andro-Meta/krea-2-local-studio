"""On-the-fly scaled-FP8 quantization for plain bf16 DiT checkpoints.

The Krea 2 turbo *_fp8_scaled checkpoint ships pre-quantized as float8_e4m3fn
with per-tensor `weight_scale` factors. RAW (and any custom) checkpoints are
shipped as bf16, which needs ~20-24 GB of VRAM. This module reproduces ComfyUI's
"scaled fp8" trick at load time: each large Linear weight is quantized to
float8_e4m3fn with a per-tensor absmax scale, so the resident weight memory is
roughly halved and the existing `_patch_fp8_linears` dequant closure
(`w = w_fp8.to(compute_dtype) * scale`) can consume it unchanged.

This is a memory-only optimization: matmuls still run in bf16 after the
on-the-fly upcast, so it works on any CUDA GPU (no fp8 tensor-core requirement).

INT8 / conv-rotation note:
ComfyUI's faster-than-fp8 INT8 path relies on a compiled CUTLASS backend
(`comfy_kitchen`) that fuses a Hadamard/conv "rotation" into the int8 GEMM
(needs CUDA >= cu130). PyTorch has `torch._scaled_mm` for scaled int8/fp8 GEMM
but no fused rotated-int8 primitive, so a pure-PyTorch INT8 would do the rotation
in a separate kernel and end up SLOWER than fp8 on Ada/Hopper/Blackwell. It only
pays off on Ampere (no fp8 compute, e.g. RTX 3090) and only with fused kernels.
Krea Studio therefore keeps native PyTorch on fp8/bf16 and routes INT8 through
the Comfy sidecar backend, where current Comfy/INT8 loaders own the fused kernels
and LoRA patching semantics.
"""
from __future__ import annotations

from pathlib import Path

import torch

# float8_e4m3fn dynamic range max (|x| <= 448).
FP8_E4M3_MAX = 448.0


def quantize_weight_to_fp8_scaled(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize one weight tensor to float8_e4m3fn with a per-tensor scale.

    Returns (fp8_weight, scale) such that `fp8_weight.to(fp32) * scale ~= weight`.
    """
    w = weight.detach().to(torch.float32)
    amax = w.abs().amax()
    if not torch.isfinite(amax) or amax <= 0:
        # All-zero or degenerate tensor: unit scale, exact zero round-trip.
        scale = 1.0
        qweight = torch.zeros_like(w).to(torch.float8_e4m3fn)
        return qweight, scale
    scale = (amax / FP8_E4M3_MAX).item()
    qweight = (w / scale).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return qweight, scale


def quantize_linear_weights_to_fp8_scaled(
    state_dict: dict[str, torch.Tensor],
    *,
    min_numel: int = 1_048_576,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Quantize eligible 2D `*.weight` tensors in a state dict to scaled fp8.

    Only 2D weights at least `min_numel` elements large are quantized (the big
    Linear projections that dominate VRAM); norms, biases, embeddings, and small
    layers stay in their original dtype for numerical safety. The returned scale
    map is keyed by module name (the `*.weight` suffix stripped) so it can be fed
    straight into `_patch_fp8_linears`.
    """
    out: dict[str, torch.Tensor] = {}
    scales: dict[str, float] = {}
    for key, tensor in state_dict.items():
        is_linear_weight = (
            key.endswith(".weight")
            and isinstance(tensor, torch.Tensor)
            and tensor.dim() == 2
            and tensor.numel() >= int(min_numel)
            and torch.is_floating_point(tensor)
        )
        if is_linear_weight:
            qweight, scale = quantize_weight_to_fp8_scaled(tensor)
            out[key] = qweight
            scales[key[: -len(".weight")]] = scale
        else:
            out[key] = tensor
    return out, scales


def load_bf16_as_fp8_scaled(
    checkpoint_path: str | Path,
    *,
    min_numel: int = 1_048_576,
    device: str = "cpu",
    compute_dtype: "torch.dtype | None" = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Stream a bf16 safetensors checkpoint into a scaled-fp8 state dict.

    Quantizes tensor-by-tensor via `safe_open` and places each result on
    `device` immediately. With `device="cuda"` the ~12 GB of fp8 weights land in
    VRAM as they are produced, so peak *host* RAM stays near a single transient
    bf16 tensor instead of the whole 24 GB checkpoint plus its fp8 copy — this is
    what lets a 24 GB RAW bf16 file load on a RAM-constrained machine.

    Large 2D float weights become float8_e4m3fn with per-tensor scales. Every
    other floating tensor is cast to `compute_dtype` (default bfloat16) so the
    non-quantized layers match the bf16 activations — checkpoints can store some
    small layers in float32, which would otherwise raise a Linear dtype mismatch.
    Non-float tensors are moved as-is. The scale map is keyed for
    `_patch_fp8_linears`.
    """
    from safetensors import safe_open

    if compute_dtype is None:
        compute_dtype = torch.bfloat16

    out: dict[str, torch.Tensor] = {}
    scales: dict[str, float] = {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if (
                key.endswith(".weight")
                and tensor.dim() == 2
                and tensor.numel() >= int(min_numel)
                and torch.is_floating_point(tensor)
            ):
                qweight, scale = quantize_weight_to_fp8_scaled(tensor)
                del tensor
                out[key] = qweight.to(device)
                scales[key[: -len(".weight")]] = scale
            elif torch.is_floating_point(tensor):
                out[key] = tensor.to(device=device, dtype=compute_dtype)
                del tensor
            else:
                out[key] = tensor.to(device)
                del tensor
    return out, scales
