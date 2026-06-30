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

RBG_PRESETS = {
    "off": (0.0, 0.0),
    "subtle": (1.0, 10.0),
    "balanced": (2.0, 20.0),
    "creative": (3.0, 30.0),
    "bold": (4.0, 40.0),
    "wild": (5.0, 50.0),
}

RBG_MODEL_ADJUSTMENTS = {
    # Krea Studio perturbs Qwen layer-tap conditioning directly, which is much
    # more sensitive than the Comfy single conditioning tensor this node came
    # from. Keep the familiar RBG UI numbers, but scale the actual Krea2 noise.
    "krea2": (0.01, 1.0),
    "z_image": (1.0, 1.0),
    "qwen_image": (1.0, 0.9),
    "flux": (0.5, 0.8),
    "sdxl": (0.6, 0.8),
    "other": (0.8, 0.8),
}

RBG_DIRECTIONS = {
    "none": None,
    "chaos": ("scatter", 1.2),
    "order": ("compress", 0.8),
    "abstract": ("wave", 1.0),
    "realistic": ("sharpen", 0.9),
    "vibrant": ("positive", 1.1),
    "moody": ("negative", 1.0),
    "dreamy": ("smooth", 1.1),
    "dynamic_pose": ("spatial", 1.2),
    "composition": ("gradient", 1.0),
    "diversity": ("diversity", 1.15),
    "facevar": ("facevar", 1.25),
    "visceral_expression_grit": ("facevar", 1.35),
    "semantic_drift": ("semantic_drift", 1.0),
    "structural_lock": ("structural_lock", 1.0),
    "cinematic_framing": ("cinematic_framing", 1.1),
    "identity_stretch": ("identity_stretch", 1.25),
    "texture_lift": ("texture_lift", 1.0),
}


def rbg_quality_defaults() -> dict:
    return {
        "algorithm": "rbg",
        "preset": "creative",
        "model_type": "krea2",
        "direction": "visceral_expression_grit",
        "shift_strength": 170,
        "protection": "none",
        "fade_curve": "smoothstep",
        "schedule": "step_cutoff",
        "cutoff_step": 3,
        "total_steps": 13,
        "cutoff_strength": 0.53,
    }


def _variance_seed(seed: int, preset: str, strength: float) -> int:
    payload = f"{int(seed)}:{preset}:{float(strength):.6f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFF_FFFF


def _rbg_preset_values(preset: str, custom_strength: float | None, randomize_percent: float | None) -> tuple[float, float]:
    preset = str(preset or "off").lower()
    if preset == "custom":
        fine = 50.0 if custom_strength is None else max(0.0, min(1.0, float(custom_strength))) * 100.0
        return (fine / 100.0) * 5.0, (fine / 100.0) * 50.0
    pct, strength = RBG_PRESETS.get(preset, (0.0, 0.0))
    if randomize_percent is not None and float(randomize_percent) > 0:
        pct = float(randomize_percent)
    return pct, strength


def _protected_span(seq: int, protection: str) -> tuple[int, int]:
    protection = str(protection or "none")
    if protection == "first_half":
        return seq // 2, seq
    if protection == "first_quarter":
        return seq // 4, seq
    if protection == "last_half":
        return 0, seq - seq // 2
    if protection == "last_quarter":
        return 0, seq - seq // 4
    return 0, seq


def _fade_envelope(num_tokens: int, fade_curve: str, device: torch.device) -> torch.Tensor:
    t = torch.linspace(0, 1, num_tokens, device=device)
    if fade_curve in {"instant", "none"}:
        return torch.ones(num_tokens, device=device)
    if fade_curve == "linear":
        return 1.0 - t
    if fade_curve == "ease_out":
        return 1.0 - t * t
    if fade_curve == "ease_in":
        return (1.0 - t) ** 2
    if fade_curve == "ease_in_out":
        return torch.where(t < 0.5, 1.0 - 2 * t * t, 2 * (1.0 - t) ** 2)
    if fade_curve == "smoothstep":
        smooth = 3 * t * t - 2 * t * t * t
        return 1.0 - smooth
    if fade_curve == "burst":
        return torch.exp(-4 * t)
    return torch.ones(num_tokens, device=device)


def _directional_noise(num_values: int, pattern: str, strength: float, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    if pattern == "scatter":
        base = torch.randn(num_values, device=device, generator=generator)
        return (base * base.abs()) * strength
    if pattern == "compress":
        return torch.tanh(torch.randn(num_values, device=device, generator=generator)) * strength * 0.5
    if pattern == "wave":
        idx = torch.arange(num_values, device=device, dtype=torch.float32)
        return (torch.sin(idx * 0.1) + torch.randn(num_values, device=device, generator=generator) * 0.3) * strength
    if pattern == "sharpen":
        base = torch.randn(num_values, device=device, generator=generator)
        detail = torch.randn(num_values, device=device, generator=generator) * 0.35
        contrast = base * torch.pow(base.abs() + 1e-6, 0.5)
        combined = (base * 0.55) + (detail * 0.30) + (contrast * 0.85)
        return combined / (combined.std() + 1e-6) * strength
    if pattern == "positive":
        base = torch.randn(num_values, device=device, generator=generator)
        return (base.abs() * 0.7 + base * 0.3) * strength
    if pattern == "negative":
        base = torch.randn(num_values, device=device, generator=generator)
        return (-base.abs() * 0.7 + base * 0.3) * strength
    if pattern == "smooth":
        base = torch.randn(num_values, device=device, generator=generator)
        if num_values > 2:
            base[1:-1] = (base[:-2] + base[1:-1] + base[2:]) / 3
        return base * strength
    if pattern == "spatial":
        base = torch.randn(num_values, device=device, generator=generator)
        chunk = max(1, num_values // 16)
        for i in range(0, num_values, chunk):
            base[i : i + chunk] += torch.randn(1, device=device, generator=generator).item() * 0.5
        return base * strength
    if pattern == "gradient":
        idx = torch.arange(num_values, device=device, dtype=torch.float32)
        normalized = idx / max(1, num_values - 1)
        direction = torch.randn(1, device=device, generator=generator).item()
        return ((normalized - 0.5) * direction * 2 + torch.randn(num_values, device=device, generator=generator) * 0.3) * strength
    if pattern == "diversity":
        return ((torch.rand(num_values, device=device, generator=generator) * 2.0) - 1.0) * strength
    if pattern == "facevar":
        base = torch.randn(num_values, device=device, generator=generator)
        jitter = torch.randn(num_values, device=device, generator=generator) * 0.35
        curved = torch.sign(base) * torch.pow(base.abs(), 1.4)
        combined = (base * 0.55) + (jitter * 0.25) + (curved * 0.85)
        return combined / (combined.std() + 1e-6) * strength
    if pattern == "semantic_drift":
        shift = torch.randn(1, device=device, generator=generator).item() * 0.15
        jitter = torch.randn(num_values, device=device, generator=generator) * 0.05
        return (torch.full((num_values,), shift, device=device) + jitter) * strength
    if pattern == "structural_lock":
        t = torch.linspace(0, 1, num_values, device=device)
        decay = torch.where(t < 0.2, torch.ones_like(t), torch.exp(-5.0 * (t - 0.2)))
        return torch.randn(num_values, device=device, generator=generator) * decay * strength
    if pattern == "cinematic_framing":
        t = torch.linspace(-1, 1, num_values, device=device)
        combined = (t * 0.6) + (torch.exp(-2.0 * t**2) * 0.4)
        return (combined + torch.randn(num_values, device=device, generator=generator) * 0.25) * strength
    if pattern == "identity_stretch":
        base = torch.randn(num_values, device=device, generator=generator)
        mid = torch.tanh(base * 1.25)
        curve = torch.sign(base) * torch.pow(base.abs(), 1.2)
        combined = (base * 0.45) + (mid * 0.35) + (curve * 0.45)
        return combined / (combined.std() + 1e-6) * strength
    if pattern == "texture_lift":
        base = torch.randn(num_values, device=device, generator=generator)
        if num_values > 2:
            low = base.clone()
            low[1:-1] = (base[:-2] + base[1:-1] + base[2:]) / 3
            base = base - low
        return base * strength
    return torch.randn(num_values, device=device, generator=generator) * strength


def _apply_rbg_sparse(
    txt: torch.Tensor,
    *,
    seed: int,
    preset: str,
    strength: float | None,
    protection: str,
    direction: str,
    fade_curve: str,
    model_type: str,
    randomize_percent: float | None,
    shift_strength: float,
) -> torch.Tensor:
    randomize, resolved_strength = _rbg_preset_values(preset, strength, randomize_percent)
    strength_mult, randomize_mult = RBG_MODEL_ADJUSTMENTS.get(str(model_type or "krea2"), (1.0, 1.0))
    randomize *= randomize_mult
    resolved_strength *= strength_mult
    if randomize <= 0 or resolved_strength <= 0:
        return txt

    direction_config = RBG_DIRECTIONS.get(str(direction or "none"), None)
    pattern = "random"
    if direction_config is not None:
        pattern, direction_mult = direction_config
        resolved_strength *= direction_mult * (float(shift_strength) / 100.0)

    modified = txt.clone()
    seq = modified.shape[1] if modified.ndim >= 2 else 0
    if seq <= 0:
        return modified
    start, end = _protected_span(seq, protection)
    if start >= end:
        return modified

    target = modified[:, start:end, ...].clone()
    total = target.numel()
    count = int(total * (randomize / 100.0))
    if count <= 0:
        return modified

    generator = torch.Generator(device=txt.device).manual_seed(_variance_seed(seed, f"rbg:{preset}:{direction}", resolved_strength))
    flat = target.flatten()
    indices = torch.randperm(total, generator=generator, device=txt.device)[:count]
    noise = _directional_noise(count, pattern, resolved_strength, generator, txt.device).to(dtype=txt.dtype)
    token_count = target.shape[1]
    envelope = _fade_envelope(token_count, str(fade_curve or "instant"), txt.device).to(dtype=txt.dtype)
    per_value_envelope = envelope.view(1, token_count, *([1] * (target.ndim - 2))).expand_as(target).contiguous().flatten()
    flat[indices] += noise * per_value_envelope[indices]
    modified[:, start:end, ...] = flat.reshape_as(target)
    return modified


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
    algorithm: str = "legacy",
    model_type: str = "krea2",
    randomize_percent: float | None = None,
    shift_strength: float = 100.0,
    schedule: str = "constant",
    cutoff_step: int = 8,
    total_steps: int = 20,
    cutoff_strength: float = 0.0,
) -> torch.Tensor:
    preset = str(preset or "off").lower()
    resolved_strength = PRESET_STRENGTHS.get(preset, 0.0) if strength is None else float(strength)
    if preset == "off":
        return txt
    if str(algorithm or "legacy") == "rbg":
        if str(schedule or "constant") == "step_cutoff" and int(total_steps or 0) > 0:
            # We do not have Comfy's per-step conditioning windows here; use the
            # cutoff as a conservative strength scaler so composition can lock.
            cutoff_ratio = max(0.0, min(1.0, float(cutoff_step) / max(1.0, float(total_steps))))
            scale = cutoff_ratio + (1.0 - cutoff_ratio) * max(0.0, min(1.0, float(cutoff_strength)))
            strength = (float(strength) if strength is not None else 0.5) * scale if preset == "custom" else strength
        return _apply_rbg_sparse(
            txt,
            seed=seed,
            preset=preset,
            strength=strength,
            protection=protection,
            direction=direction,
            fade_curve=fade_curve,
            model_type=model_type,
            randomize_percent=randomize_percent,
            shift_strength=shift_strength,
        )
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
