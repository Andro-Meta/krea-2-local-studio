"""GPU capability detection so Krea 2 Studio adapts to the user's hardware.

Mirrors ComfyUI's architecture checks (comfy/model_management.py): precision and
fp8/fp4 support are decided from the CUDA compute capability (props.major/minor),
not the card name. This lets defaults differ correctly across, e.g.:

  - RTX 3090  (Ampere,  sm_86): bf16 yes, fp8 *storage-only* (upcast at compute)
  - RTX 4090  (Ada,     sm_89): fp8 compute yes
  - RTX 5090  (Blackwell sm_120): fp8 + nvfp4 compute yes, 32GB

`classify_capabilities` is pure (no torch) for unit testing; `detect_gpu_capabilities`
reads the live device.
"""
from __future__ import annotations


def _arch_label(major: int | None, minor: int | None) -> str:
    if major is None:
        return "unknown"
    if major >= 12:
        return "Blackwell"
    if major >= 10:
        return "Blackwell-class"
    if major == 9:
        return "Hopper"
    if major == 8 and (minor or 0) >= 9:
        return "Ada Lovelace"
    if major == 8:
        return "Ampere"
    if major == 7:
        return "Volta/Turing"
    return "legacy"


def classify_capabilities(
    *,
    major: int | None,
    minor: int | None,
    name: str = "",
    vram_total_gb: float | None = None,
) -> dict:
    """Classify precision/quant support from compute capability."""
    m = major if isinstance(major, int) else None
    n = minor if isinstance(minor, int) else 0

    if m is None:
        supports_bf16 = False
        supports_fp16 = False
        supports_fp8_compute = False
        supports_nvfp4 = False
    else:
        supports_bf16 = m >= 8
        # Fast fp16: Pascal+ in practice (ComfyUI treats <sm_70 conservatively, but
        # GP10x/Turing/Ampere+ run fp16 fine). Maxwell (sm_5x) and older: no.
        supports_fp16 = m >= 6
        # fp8 tensor-core compute: Hopper/Blackwell (>=9), or Ada (8.9). Ampere (8.0/8.6) no.
        supports_fp8_compute = (m >= 9) or (m == 8 and n >= 9)
        supports_nvfp4 = m >= 10

    if supports_bf16:
        recommended_compute_dtype = "bf16"
    elif supports_fp16:
        recommended_compute_dtype = "fp16"
    else:
        recommended_compute_dtype = "fp32"

    # fp8 weights still help on cards without fp8 compute: they halve VRAM and are
    # upcast to bf16/fp16 per layer at compute time (same speed as bf16, less VRAM).
    fp8_storage_only = bool(supports_bf16 and not supports_fp8_compute)
    if supports_fp8_compute:
        fp8_note = "fp8 runs on tensor cores (lower VRAM and faster matmul)."
    elif fp8_storage_only:
        fp8_note = "fp8 is storage-only here: it halves VRAM but upcasts at compute (same speed as bf16)."
    else:
        fp8_note = "fp8 not recommended on this GPU."

    return {
        "name": name or "",
        "arch": _arch_label(m, n),
        "compute_capability": (f"{m}.{n}" if m is not None else None),
        "vram_total_gb": round(vram_total_gb, 1) if vram_total_gb is not None else None,
        "supports_bf16": supports_bf16,
        "supports_fp16": supports_fp16,
        "supports_fp8_compute": supports_fp8_compute,
        "supports_nvfp4": supports_nvfp4,
        "recommended_compute_dtype": recommended_compute_dtype,
        "fp8_storage_only": fp8_storage_only,
        "fp8_note": fp8_note,
    }


# Floors (mirror what ComfyUI can realistically run a Flux/Krea-class DiT on).
MIN_VRAM_GB = 6.0        # below this, even heavy offload won't fit the working set
COMFORTABLE_VRAM_GB = 12.0
HIGH_VRAM_GB = 22.0
MIN_RAM_GB = 16.0        # block swap streams weights from system RAM


def assess_runnability(caps: dict, ram_total_gb: float | None) -> dict:
    """Decide whether this machine can run Krea 2 and how to configure it.

    Returns a verdict with can_run, a tier, the compute dtype to use, a
    recommended block-swap count, the max resolution tier, and a human reason —
    so users on anything from a 2070 to a 5090 get correct settings or a clear
    "can't run" message (aiming for "if ComfyUI can run it, this can too").
    """
    vram = caps.get("vram_total_gb")
    name = caps.get("name", "GPU") or "GPU"
    compute_dtype = caps.get("recommended_compute_dtype", "fp32")
    supports_fp16 = bool(caps.get("supports_fp16", False))

    # Hard blockers --------------------------------------------------------
    if caps.get("compute_capability") is None or vram is None:
        return _verdict(False, "unsupported", "fp32", 0, "1k",
                        "No CUDA GPU detected. Krea 2 needs an NVIDIA GPU (a 12GB+ card is recommended).")
    if not supports_fp16:
        return _verdict(False, "unsupported", "fp32", 0, "1k",
                        f"{name} is too old (pre-Pascal): no usable fp16. Krea 2 needs a GTX 10-series or newer.")
    if ram_total_gb is not None and ram_total_gb + 1.0 < MIN_RAM_GB:
        return _verdict(False, "unsupported", compute_dtype, 0, "1k",
                        f"Only ~{ram_total_gb:.0f}GB system RAM; need ~{MIN_RAM_GB:.0f}GB (block swap streams weights from RAM).")
    if vram + 0.5 < MIN_VRAM_GB:
        return _verdict(False, "unsupported", compute_dtype, 0, "1k",
                        f"{name} has ~{vram:.0f}GB VRAM; need ~{MIN_VRAM_GB:.0f}GB+ to run Krea 2 (even with heavy offload).")

    # Runnable tiers -------------------------------------------------------
    if vram >= HIGH_VRAM_GB:
        tier = "high"
        blocks, max_tier = 0, "2k"
        reason = f"{name} ({vram:.0f}GB): runs fp8 with no block swap; 1K and 2K both fine."
    elif vram >= COMFORTABLE_VRAM_GB:
        tier = "comfortable"
        blocks, max_tier = 6, "2k"
        reason = f"{name} ({vram:.0f}GB): runs fp8 comfortably at 1K; 2K works with block swap (slower)."
    else:  # MIN_VRAM_GB <= vram < COMFORTABLE
        tier = "minimum"
        blocks, max_tier = 20, "1k"
        reason = (f"{name} ({vram:.0f}GB): runs fp8 at 1K with heavy block swap and tiled decode — "
                  "expect slow renders and high RAM use. 2K not recommended.")

    if not caps.get("supports_bf16", False):
        reason += " Uses fp16 compute (no bf16 on this GPU)."

    return _verdict(True, tier, compute_dtype, blocks, max_tier, reason)


def _verdict(can_run, tier, compute_dtype, blocks, max_tier, reason) -> dict:
    return {
        "can_run": can_run,
        "tier": tier,
        "compute_dtype": compute_dtype,
        "blocks_to_swap": blocks,
        "max_tier": max_tier,
        "reason": reason,
    }


def detect_gpu_capabilities() -> dict:
    """Classify the live CUDA device (falls back to a conservative CPU profile)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return classify_capabilities(major=None, minor=None, name="cpu", vram_total_gb=None)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        return classify_capabilities(
            major=int(props.major),
            minor=int(props.minor),
            name=str(props.name),
            vram_total_gb=vram_gb,
        )
    except Exception:
        return classify_capabilities(major=None, minor=None, name="", vram_total_gb=None)
