from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LanPaintSettings:
    num_steps: int = 5
    lambda_strength: float = 16.0
    step_size: float = 0.2
    beta: float = 1.0
    friction: float = 15.0
    early_stop: int = 1
    prompt_mode: str = "Image First"
    strength: float = 1.0


def prepare_lanpaint_mask(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """LanPaint expects hard binary masks: 1=regenerate, 0=known context."""
    return (mask >= threshold).to(dtype=mask.dtype, device=mask.device)


def lanpaint_masked_inner_update(
    current: torch.Tensor,
    known_trajectory: torch.Tensor,
    mask: torch.Tensor,
    *,
    tcurr: float,
    tprev: float,
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
    inner_steps: int = 3,
    strength: float = 1.0,
    settings: LanPaintSettings | None = None,
) -> torch.Tensor:
    """Training-free masked refinement inspired by LanPaint's inner iterations.

    This is an independent, conservative implementation for Krea's flow latents:
    known regions are re-anchored every inner step and masked regions take small
    flow steps using repeated model velocity estimates.
    """
    if settings is None:
        settings = LanPaintSettings(num_steps=inner_steps, strength=strength, early_stop=0)
    steps = max(0, int(settings.num_steps))
    if settings.early_stop > 0:
        steps = max(0, steps - int(settings.early_stop))
    if steps <= 0:
        return current

    strength = max(0.0, min(2.0, float(settings.strength)))
    step_size = max(0.0001, min(1.0, float(settings.step_size))) * strength
    lambda_strength = max(0.1, min(50.0, float(settings.lambda_strength)))
    beta = max(0.0001, min(5.0, float(settings.beta)))
    friction = max(0.0, min(100.0, float(settings.friction)))
    masked = prepare_lanpaint_mask(mask).to(device=current.device, dtype=current.dtype)
    known = known_trajectory.to(device=current.device, dtype=current.dtype)
    latent = masked * current + (1.0 - masked) * known
    dt = float(tprev) - float(tcurr)
    base_dt = dt * step_size / steps
    damping = 1.0 / (1.0 + friction * abs(base_dt))
    velocity_state = torch.zeros_like(latent)

    for index in range(steps):
        local_t = float(tcurr) + (float(tprev) - float(tcurr)) * (index / steps)
        velocity = velocity_fn(latent, local_t)
        context_pull = lambda_strength * (known - latent) * (1.0 - masked)
        masked_velocity = beta * velocity * masked + context_pull
        velocity_state = damping * velocity_state + masked_velocity
        latent = latent + base_dt * velocity_state
        latent = masked * latent + (1.0 - masked) * known

    return latent
