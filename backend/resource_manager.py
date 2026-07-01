"""Capacity-aware runtime advice and post-render cache policy.

Heuristics (torch-free, unit-testable) that mirror ComfyUI's "fit the model by
freeing/offloading" idea: estimate the VRAM a generation needs from the weight
footprint plus a resolution-scaled working set, and recommend how many DiT
blocks to stream from RAM so it fits, plus whether to use a tiled VAE decode and
whether to clear the CUDA cache afterward.

These are deliberately rough — their job is to keep generations from OOM-ing and
to keep VRAM from staying pinned high after heavy (2K) renders, not to be exact.
"""
from __future__ import annotations

import math

# Approx resident weight VRAM for the Krea 2 DiT by precision.
WEIGHT_VRAM_GB = {"fp8": 12.0, "int8": 13.0, "bf16": 24.0, "fp16": 24.0}
DEFAULT_HEADROOM_GB = 3.0
TOTAL_BLOCKS = 28


def _quant_key(quantization: str) -> str:
    q = str(quantization or "fp8").lower()
    if q in WEIGHT_VRAM_GB:
        return q
    return "fp8" if q == "fp8" else "bf16"


# Krea 2 DiT memory_usage_factor (from ComfyUI supported_models.py: image_model "krea2").
KREA2_MEMORY_FACTOR = 2.2
COMPRESSION = 8  # latent downscale


def estimate_inference_scratch_gb(
    width: int,
    height: int,
    *,
    batch: int = 1,
    cfg_active: bool = True,
    dtype_bytes: int = 2,
    factor: float = KREA2_MEMORY_FACTOR,
) -> float:
    """Activation/attention scratch for the DiT forward.

    Ported from ComfyUI's BaseModel.memory_required (flash-attention path):
        mem_mb = area * dtype_size * 0.01 * memory_usage_factor
        area   = effective_batch * (latent_h * latent_w)
    CFG runs a positive+negative pass, so the effective batch doubles.
    """
    latent_area = (int(width) // COMPRESSION) * (int(height) // COMPRESSION)
    effective_batch = max(1, int(batch)) * (2 if cfg_active else 1)
    mem_mb = effective_batch * latent_area * dtype_bytes * 0.01 * factor
    return mem_mb / 1024.0


def estimate_decode_vram_gb(width: int, height: int, *, dtype_bytes: int = 2) -> float:
    """Rough per-image VAE decode footprint (const x latent area x dtype), in the
    spirit of ComfyUI's memory_used_decode. Used only to pre-empt tiled decode;
    the decode path also has an OOM fallback to tiling."""
    latent_area = (int(width) // COMPRESSION) * (int(height) // COMPRESSION)
    return (latent_area * dtype_bytes * 0.03) / 1024.0


def recommend_runtime(
    *,
    free_vram_gb: float | None,
    width: int,
    height: int,
    quantization: str,
    total_blocks: int = TOTAL_BLOCKS,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
    cfg_active: bool = True,
) -> dict:
    quant = _quant_key(quantization)
    weight = WEIGHT_VRAM_GB[quant]
    megapixels = (int(width) * int(height)) / 1_000_000.0
    working = estimate_inference_scratch_gb(width, height, cfg_active=cfg_active) + estimate_decode_vram_gb(width, height)
    needed = weight + working + headroom_gb

    blocks_to_swap = 0
    fits = True
    if free_vram_gb is not None:
        per_block = weight / max(1, total_blocks)
        if needed > free_vram_gb:
            deficit = needed - free_vram_gb
            blocks_to_swap = min(total_blocks, max(1, math.ceil(deficit / per_block)))
        resident_weight = weight * (total_blocks - blocks_to_swap) / max(1, total_blocks)
        fits = (resident_weight + working + headroom_gb) <= free_vram_gb or blocks_to_swap >= total_blocks

    warnings: list[str] = []
    if megapixels >= 3.5:
        warnings.append(
            "2K is ~4x the work of 1K: much longer renders and high VRAM (attention scales with token count)."
        )
    if free_vram_gb is not None and not fits:
        warnings.append(
            f"Estimated ~{needed:.0f}GB needed but only {free_vram_gb:.0f}GB free; "
            "reduce resolution, use fp8, or increase block swap."
        )

    return {
        "blocks_to_swap": blocks_to_swap,
        "tiled_decode": megapixels >= 3.0,
        "fits": fits,
        "estimated_vram_gb": round(needed, 1),
        "megapixels": round(megapixels, 2),
        "warnings": warnings,
    }


def recommend_defaults(caps: dict, free_vram_gb: float | None) -> dict:
    """Per-system runtime defaults from GPU capabilities + VRAM.

    Picks a sensible precision and block-swap count so Krea runs end-to-end on the
    user's hardware (3090/4090/5090/etc.), and explains fp8 behavior on cards
    without fp8 compute.
    """
    vram = caps.get("vram_total_gb")
    supports_bf16 = bool(caps.get("supports_bf16", False))
    fp8_storage_only = bool(caps.get("fp8_storage_only", False))

    # fp8 (scaled e4m3) is the safe default everywhere: native-fast on Ada+, and a
    # VRAM-saving storage-only path on Ampere. bf16 only when there's clearly room.
    if vram is not None and vram >= 30 and supports_bf16:
        quantization = "bf16"
    else:
        quantization = "fp8"

    # Block-swap recommendation for a representative turbo-fp8 1K load on this card.
    rec = recommend_runtime(
        free_vram_gb=free_vram_gb if free_vram_gb is not None else vram,
        width=1024, height=1024, quantization="fp8",
    )
    blocks_to_swap = rec["blocks_to_swap"]
    max_tier = "2k" if (vram is None or vram >= 16) else "1k"

    notes = caps.get("fp8_note", "")
    if fp8_storage_only:
        notes = "fp8 is storage-only on this GPU (saves VRAM, same speed as bf16). " + notes
    if vram is not None and vram < 16:
        notes += f" Only ~{vram:.0f}GB VRAM: prefer 1K and fp8; use block swap for larger models."

    return {
        "quantization": quantization,
        "blocks_to_swap": blocks_to_swap,
        "max_tier": max_tier,
        "notes": notes.strip(),
    }


def plan_generation(
    *,
    free_vram_gb: float | None,
    width: int,
    height: int,
    quantization: str,
    batch: int = 1,
    cfg_active: bool = True,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
) -> dict:
    """Proactive per-generation plan: decide whether to free the cache first and/or
    use a tiled VAE decode BEFORE sampling, so we avoid OOM instead of hitting it.

    Keeps the model loaded (fast); only frees transient cache and tiles the decode.
    """
    megapixels = (int(width) * int(height)) / 1_000_000.0
    scratch = estimate_inference_scratch_gb(width, height, batch=batch, cfg_active=cfg_active)
    decode = estimate_decode_vram_gb(width, height)

    clear_cache_first = False
    tiled_decode = megapixels >= 3.0
    warnings: list[str] = []

    if free_vram_gb is not None:
        # If the forward's scratch is a big fraction of free VRAM, reclaim cache first.
        if scratch + headroom_gb > free_vram_gb * 0.6:
            clear_cache_first = True
        # If a single-image decode would eat most of free VRAM, tile it.
        if decode + headroom_gb > free_vram_gb * 0.5:
            tiled_decode = True
        if scratch + decode + headroom_gb > free_vram_gb:
            clear_cache_first = True
            tiled_decode = True
            warnings.append(
                f"Tight VRAM (~{free_vram_gb:.0f}GB free, ~{scratch+decode:.0f}GB working set): "
                "freeing cache and using tiled decode."
            )

    return {
        "clear_cache_first": clear_cache_first,
        "tiled_decode": tiled_decode,
        "estimated_scratch_gb": round(scratch, 2),
        "estimated_decode_gb": round(decode, 2),
        "warnings": warnings,
    }


def plan_parallel_batch(
    *,
    free_vram_gb: float | None,
    width: int,
    height: int,
    quantization: str,
    batch: int,
    cfg_active: bool,
    mode: str,
    checkpoint: str,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
) -> dict:
    batch = max(1, int(batch or 1))
    plan = plan_generation(
        free_vram_gb=free_vram_gb,
        width=width,
        height=height,
        quantization=quantization,
        batch=batch,
        cfg_active=cfg_active,
        headroom_gb=headroom_gb,
    )
    megapixels = (int(width) * int(height)) / 1_000_000.0
    warnings = list(plan["warnings"])
    blocked_reasons: list[str] = []

    if batch <= 1:
        blocked_reasons.append("Parallel batch only applies when Batch is greater than 1.")
    if batch > 2:
        blocked_reasons.append("Parallel batch is capped at 2 images until visual/VRAM benchmarks prove higher batches safe.")
    if str(mode) in {"inpaint", "outpaint"}:
        blocked_reasons.append("Parallel batch is disabled for inpaint/outpaint; use safe queue.")
    if str(checkpoint).lower() == "raw":
        blocked_reasons.append("RAW parallel batch is disabled; RAW needs more steps and CFG memory.")
    if megapixels >= 3.0 and batch > 1:
        blocked_reasons.append("2K parallel batch is disabled; use safe queue.")

    working = float(plan["estimated_scratch_gb"]) + float(plan["estimated_decode_gb"])
    fits = free_vram_gb is None or (working + headroom_gb) <= float(free_vram_gb)
    if free_vram_gb is None:
        blocked_reasons.append("Free VRAM is unknown; use safe queue.")
    elif float(free_vram_gb) < 14.0:
        blocked_reasons.append("Parallel batch needs at least 14GB free VRAM after the model is loaded.")
    if not fits:
        blocked_reasons.append("Estimated batch working set exceeds free VRAM headroom.")
    allowed = not blocked_reasons
    if not allowed:
        warnings.extend(blocked_reasons)

    return {
        **plan,
        "allowed": allowed,
        "fits": fits,
        "batch": batch,
        "mode": "parallel" if allowed else "safe_queue",
        "blocked_reasons": blocked_reasons,
        "warnings": warnings,
    }


def should_clear_after_render(width: int, height: int, free_vram_gb: float | None) -> bool:
    """Clear the CUDA cache after large renders (so the 2K-sized reserved blocks
    don't stay pinned) or whenever free VRAM is already low."""
    megapixels = (int(width) * int(height)) / 1_000_000.0
    if megapixels >= 2.0:
        return True
    if free_vram_gb is not None and free_vram_gb < 3.0:
        return True
    return False
