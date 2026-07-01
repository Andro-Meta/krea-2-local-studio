from __future__ import annotations

from pathlib import Path

import torch

from gguf import GGUFReader, dequantize

from krea2.fp8_quant import quantize_weight_to_fp8_scaled


IGNORED_EXTRA_KEYS = {"last.up.weight", "last.down.weight"}


def native_krea_state_shapes() -> dict[str, tuple[int, ...]]:
    """Return the canonical Krea 2 DiT state-dict shapes without allocating weights."""
    from inference import _no_random_init
    from krea2.mmdit import SingleMMDiTConfig, SingleStreamDiT

    with _no_random_init():
        with torch.device("meta"):
            model = SingleStreamDiT(SingleMMDiTConfig(
                features=6144, tdim=256, txtdim=2560,
                heads=48, kvheads=12, multiplier=4,
                layers=28, patch=2, channels=16,
                txtlayers=12,
            ))
    return {name: tuple(param.shape) for name, param in model.state_dict().items()}


def inspect_gguf_for_krea(path: str | Path, *, target_shapes: dict[str, tuple[int, ...]] | None = None) -> dict:
    target_shapes = target_shapes or native_krea_state_shapes()
    reader = GGUFReader(str(path))
    gguf_keys = {tensor.name for tensor in reader.tensors}
    native_keys = set(target_shapes)
    type_counts: dict[str, int] = {}
    for tensor in reader.tensors:
        type_counts[tensor.tensor_type.name] = type_counts.get(tensor.tensor_type.name, 0) + 1
    return {
        "path": str(path),
        "tensor_count": len(reader.tensors),
        "type_counts": type_counts,
        "missing_native_keys": sorted(native_keys - gguf_keys),
        "extra_gguf_keys": sorted(gguf_keys - native_keys),
    }


def _as_tensor(array, *, dtype: torch.dtype) -> torch.Tensor:
    if not array.flags.writeable and array.nbytes < 64 * 1024 * 1024:
        array = array.copy()
    tensor = torch.from_numpy(array)
    if tensor.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        tensor = tensor.float()
    return tensor.to(dtype=dtype)


def load_gguf_as_fp8_scaled(
    path: str | Path,
    *,
    target_shapes: dict[str, tuple[int, ...]] | None = None,
    min_numel: int = 1_048_576,
    device: str = "cpu",
    compute_dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Dequantize a Krea GGUF checkpoint into the existing scaled-FP8 runtime.

    This is a correctness-first bridge: GGUF storage is accepted, then large
    Linear weights are immediately converted to our native fp8-scaled format so
    generation uses the proven PyTorch Krea sampler/conditioning path.
    """
    target_shapes = target_shapes or native_krea_state_shapes()
    reader = GGUFReader(str(path))
    tensors_by_name = {tensor.name: tensor for tensor in reader.tensors}
    missing = sorted(set(target_shapes) - set(tensors_by_name))
    if missing:
        raise RuntimeError(f"GGUF checkpoint is missing {len(missing)} Krea tensor(s): {missing[:10]}")

    state: dict[str, torch.Tensor] = {}
    scales: dict[str, float] = {}
    for name, shape in target_shapes.items():
        tensor = tensors_by_name[name]
        array = dequantize(tensor.data, tensor.tensor_type)
        value = _as_tensor(array, dtype=torch.float32).reshape(shape)
        if name.endswith(".weight") and value.dim() == 2 and value.numel() >= int(min_numel):
            qweight, scale = quantize_weight_to_fp8_scaled(value)
            state[name] = qweight.to(device)
            scales[name[: -len(".weight")]] = scale
        elif torch.is_floating_point(value):
            state[name] = value.to(device=device, dtype=compute_dtype)
        else:
            state[name] = value.to(device=device)
        del value, array
    return state, scales
