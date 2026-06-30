"""Native INT8 ConvRot runtime for Krea 2 diffusion checkpoints.

This ports the minimum ComfyUI-INT8-Fast pieces Krea Studio needs:
Comfy-style safetensors inspection, grouped regular-Hadamard ConvRot, and a
Linear replacement that uses dynamic activation quantization plus torch._int_mm.
Optional comfy_kitchen/Triton kernels can be added later without changing the
checkpoint format or GUI contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

CONVROT_GROUP_SIZE = 256
KREA2_INT8_EXCLUDED_PREFIXES = ("first", "last", "tmlp", "tproj", "txtfusion", "txtmlp")

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def build_hadamard(size: int, *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build the regular orthogonal Hadamard matrix used by ConvRot."""
    import math

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    key = (int(size), str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h = h / (size**0.5)
    _HADAMARD_CACHE[key] = h
    return h


def rotate_weight(weight: torch.Tensor, h: torch.Tensor, *, group_size: int) -> torch.Tensor:
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")
    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    return torch.matmul(grouped, h.T.to(device=weight.device, dtype=weight.dtype)).reshape_as(weight)


def rotate_activation(x: torch.Tensor, h: torch.Tensor, *, group_size: int) -> torch.Tensor:
    shape = x.shape
    features = shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    grouped = x.reshape(*shape[:-1], features // group_size, group_size)
    return torch.matmul(grouped, h.to(device=x.device, dtype=x.dtype)).reshape(shape)


def quantize_int8(x: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)


def quantize_int8_axiswise(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x.abs().amax(dim=dim, keepdim=True).float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale


def dequantize_int8_weight(weight: torch.Tensor, weight_scale: torch.Tensor | float, dtype: torch.dtype) -> torch.Tensor:
    scale = weight_scale
    if isinstance(scale, torch.Tensor):
        scale = scale.to(device=weight.device, dtype=torch.float32)
    return (weight.float() * scale).to(dtype)


def _parse_comfy_quant(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {}
    try:
        raw = bytes(tensor.detach().cpu().to(torch.uint8).tolist()).decode("utf-8")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _is_int8_dtype(dtype: Any) -> bool:
    text = str(dtype).upper()
    return text in {"I8", "INT8", "TORCH.INT8"} or text.endswith(".INT8")


def _module_name_from_key(key: str, suffix: str) -> str:
    return key[: -len(suffix)]


def inspect_int8_safetensors(path: str | Path) -> dict[str, Any]:
    """Inspect a prequantized INT8 ConvRot safetensors file without loading weights."""
    from safetensors import safe_open

    path = Path(path)
    layers: list[dict[str, Any]] = []
    high_precision_prefixes: set[str] = set()
    missing_scales: list[str] = []
    metadata: dict[str, str] = {}
    keys: list[str] = []

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = list(handle.keys())
        key_set = set(keys)
        for key in keys:
            if not key.endswith(".weight"):
                continue
            name = _module_name_from_key(key, ".weight")
            dtype = handle.get_slice(key).get_dtype()
            root = name.split(".", 1)[0]
            if root in KREA2_INT8_EXCLUDED_PREFIXES:
                high_precision_prefixes.add(root)
            if not _is_int8_dtype(dtype):
                continue
            scale_key = f"{name}.weight_scale"
            quant_key = f"{name}.comfy_quant"
            if scale_key not in key_set:
                missing_scales.append(name)
            quant_conf = _parse_comfy_quant(handle.get_tensor(quant_key) if quant_key in key_set else None)
            layers.append(
                {
                    "name": name,
                    "dtype": str(dtype),
                    "has_weight_scale": scale_key in key_set,
                    "per_row": bool(quant_conf.get("per_row", True)),
                    "convrot": bool(quant_conf.get("convrot", False)),
                    "convrot_groupsize": int(quant_conf.get("convrot_groupsize", CONVROT_GROUP_SIZE)),
                }
            )

    return {
        "ok": True,
        "path": str(path),
        "metadata": metadata,
        "tensor_count": len(keys),
        "int8_layer_count": len(layers),
        "convrot_layer_count": sum(1 for layer in layers if layer["convrot"]),
        "int8_layers": layers,
        "missing_scale_layers": missing_scales,
        "high_precision_prefixes_present": sorted(high_precision_prefixes),
    }


def int8_layer_specs_from_safetensors(path: str | Path) -> dict[str, dict[str, Any]]:
    info = inspect_int8_safetensors(path)
    return {
        layer["name"]: {"convrot": bool(layer["convrot"]), "convrot_groupsize": int(layer["convrot_groupsize"])}
        for layer in info["int8_layers"]
        if layer["has_weight_scale"]
    }


class NativeInt8Linear(torch.nn.Linear):
    """Drop-in Linear subclass for prequantized INT8 W8A8 layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=torch.int8),
            requires_grad=False,
        )
        self.bias = (
            torch.nn.Parameter(torch.empty(out_features, device=device, dtype=dtype or torch.float32), requires_grad=False)
            if bias
            else None
        )
        self.register_buffer("weight_scale", None)
        self._is_per_row = True
        self._use_convrot = False
        self._convrot_groupsize = CONVROT_GROUP_SIZE
        self.compute_dtype = compute_dtype

    def load_quantized_weight(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        convrot: bool = False,
        convrot_groupsize: int = CONVROT_GROUP_SIZE,
        per_row: bool = True,
    ) -> None:
        self.weight = torch.nn.Parameter(weight.to(torch.int8), requires_grad=False)
        self.weight_scale = weight_scale.float()
        self.bias = torch.nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self._is_per_row = bool(per_row)
        self._use_convrot = bool(convrot)
        self._convrot_groupsize = int(convrot_groupsize or CONVROT_GROUP_SIZE)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight = state_dict.pop(prefix + "weight", None)
        bias = state_dict.pop(prefix + "bias", None)
        weight_scale = state_dict.pop(prefix + "weight_scale", None)
        comfy_quant = state_dict.pop(prefix + "comfy_quant", None)
        state_dict.pop(prefix + "input_scale", None)
        if weight is None:
            missing_keys.append(prefix + "weight")
            return
        if weight_scale is None:
            error_msgs.append(f"Missing INT8 weight_scale for {prefix[:-1]}")
            return
        quant_conf = _parse_comfy_quant(comfy_quant)
        self.load_quantized_weight(
            weight,
            weight_scale,
            bias=bias,
            convrot=bool(quant_conf.get("convrot", False)),
            convrot_groupsize=int(quant_conf.get("convrot_groupsize", CONVROT_GROUP_SIZE)),
            per_row=bool(quant_conf.get("per_row", True)),
        )

    def _int_mm_forward(self, x_2d: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
        x_8, x_scale = quantize_int8_axiswise(x_2d, dim=-1)
        res = torch._int_mm(x_8.contiguous(), weight.t().contiguous())
        if self._is_per_row:
            y = res.float().mul_(x_scale).mul_(weight_scale.t())
        else:
            y = res.float().mul_(x_scale * weight_scale)
        y = y.to(dtype)
        return y + bias.to(dtype) if bias is not None else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        dtype = self.compute_dtype or (x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32)
        x_2d = x.reshape(-1, shape[-1]).to(dtype)
        if self._use_convrot:
            h = build_hadamard(self._convrot_groupsize, device=x_2d.device, dtype=x_2d.dtype)
            x_2d = rotate_activation(x_2d, h, group_size=self._convrot_groupsize)
        weight = self.weight.to(device=x_2d.device)
        weight_scale = self.weight_scale.to(device=x_2d.device, dtype=torch.float32)
        bias = self.bias.to(device=x_2d.device, dtype=dtype) if self.bias is not None else None
        try:
            y = self._int_mm_forward(x_2d, weight, weight_scale, bias, dtype)
        except Exception:
            w_float = dequantize_int8_weight(weight, weight_scale, dtype)
            y = F.linear(x_2d, w_float, bias)
        return y.reshape(*shape[:-1], y.shape[-1])


def replace_int8_linears(model: torch.nn.Module, specs: dict[str, dict[str, Any]], *, compute_dtype: torch.dtype | None = None) -> int:
    """Replace named torch Linear modules with NativeInt8Linear before loading."""
    module_map = dict(model.named_modules())
    replaced = 0
    for name in sorted(specs):
        mod = module_map.get(name)
        if mod is None or not isinstance(mod, torch.nn.Linear) or isinstance(mod, NativeInt8Linear):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = module_map[parent_name] if parent_name else model
        native = NativeInt8Linear(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            device=getattr(mod.weight, "device", None),
            dtype=getattr(mod.weight, "dtype", torch.float32),
            compute_dtype=compute_dtype,
        )
        setattr(parent, child_name, native)
        replaced += 1
    return replaced


def load_int8_convrot_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            sd[key] = handle.get_tensor(key)
    return sd
