from __future__ import annotations

import hashlib

import torch

PRESET_STRENGTHS = {
    "off": 0.0,
    "subtle": 0.01,
    "balanced": 0.025,
    "creative": 0.05,
    "bold": 0.08,
}


def _variance_seed(seed: int, preset: str, strength: float) -> int:
    payload = f"{int(seed)}:{preset}:{float(strength):.6f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFF_FFFF


def apply_seed_variance(
    txt: torch.Tensor,
    *,
    seed: int,
    preset: str = "off",
    strength: float | None = None,
    protection: str = "first_half",
) -> torch.Tensor:
    preset = str(preset or "off").lower()
    resolved_strength = PRESET_STRENGTHS.get(preset, 0.0) if strength is None else float(strength)
    if resolved_strength <= 0:
        return txt

    generator = torch.Generator(device=txt.device).manual_seed(_variance_seed(seed, preset, resolved_strength))
    noise = torch.randn(txt.shape, device=txt.device, dtype=txt.dtype, generator=generator) * resolved_strength

    if protection != "none" and txt.shape[1] > 0:
        protect = txt.shape[1] // 2 if protection == "first_half" else txt.shape[1] // 4
        noise[:, :protect] = 0
    return txt + noise
