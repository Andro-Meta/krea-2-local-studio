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
    direction: str = "none",
    fade_curve: str = "linear",
    injection_start: float = 0.0,
    injection_end: float = 1.0,
) -> torch.Tensor:
    preset = str(preset or "off").lower()
    resolved_strength = PRESET_STRENGTHS.get(preset, 0.0) if strength is None else float(strength)
    if resolved_strength <= 0:
        return txt

    generator = torch.Generator(device=txt.device).manual_seed(_variance_seed(seed, preset, resolved_strength))
    noise = torch.randn(txt.shape, device=txt.device, dtype=txt.dtype, generator=generator) * resolved_strength

    seq = txt.shape[1]
    if seq > 0:
        start = max(0, min(seq, int(round(seq * float(injection_start)))))
        end = max(start, min(seq, int(round(seq * float(injection_end)))))
        if start > 0:
            noise[:, :start] = 0
        if end < seq:
            noise[:, end:] = 0

        view_shape = [1, seq] + [1] * max(0, txt.ndim - 2)
        positions = torch.linspace(0, 1, seq, device=txt.device, dtype=torch.float32).view(*view_shape)
        if direction == "forward":
            weights = positions
        elif direction == "reverse":
            weights = 1 - positions
        elif direction == "center":
            weights = 1 - (positions - 0.5).abs() * 2
        elif direction == "edges":
            weights = (positions - 0.5).abs() * 2
        else:
            weights = torch.ones_like(positions)
        if fade_curve == "ease_in":
            weights = weights * weights
        elif fade_curve == "ease_out":
            weights = 1 - (1 - weights) * (1 - weights)
        elif fade_curve == "smoothstep":
            weights = weights * weights * (3 - 2 * weights)
        noise = noise * weights.to(dtype=txt.dtype)

    if protection != "none" and txt.shape[1] > 0:
        protect = txt.shape[1] // 2 if protection == "first_half" else txt.shape[1] // 4
        noise[:, :protect] = 0
    return txt + noise
