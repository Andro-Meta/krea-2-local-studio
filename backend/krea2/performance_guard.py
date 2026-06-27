from __future__ import annotations

import importlib.util
import platform


def attention_acceleration_diagnostic(
    *,
    device: str = "cuda",
    dtype: str = "bf16",
    text_fusion: bool = True,
    mask_shape_safe: bool = True,
) -> dict:
    """Report whether optional attention acceleration is safe to enable.

    This is diagnostic-only; Studio does not enable custom attention kernels by
    default because Krea 2's fp8/text-fusion paths are easy to destabilize.
    """
    has_sage = importlib.util.find_spec("sageattention") is not None
    if not has_sage:
        return {
            "status": "unavailable",
            "available": False,
            "reason": "SageAttention is not installed",
            "recommendation": "Keep the default PyTorch SDPA attention path.",
        }
    if str(dtype).lower() == "fp8" or text_fusion:
        return {
            "status": "safe_disabled",
            "available": True,
            "reason": "fp8/text-fusion attention path is unsafe for acceleration",
            "recommendation": "Keep acceleration off for Krea 2 fp8 or text-fusion runs.",
        }
    if platform.system().lower().startswith("win"):
        return {
            "status": "safe_disabled",
            "available": True,
            "reason": "Windows Triton/SageAttention runtime is not enabled by default",
            "recommendation": "Use the default SDPA path unless you install and validate a compatible kernel.",
        }
    if device != "cuda":
        return {
            "status": "safe_disabled",
            "available": True,
            "reason": "attention acceleration requires CUDA",
            "recommendation": "Use SDPA on CPU or non-CUDA devices.",
        }
    if not mask_shape_safe:
        return {
            "status": "safe_disabled",
            "available": True,
            "reason": "attention mask shape is unsupported by the accelerated path",
            "recommendation": "Use SDPA for this request.",
        }
    return {
        "status": "available_but_off",
        "available": True,
        "reason": "optional attention package is present but disabled by default",
        "recommendation": "Only enable after fixed-seed visual A/B validation.",
    }
