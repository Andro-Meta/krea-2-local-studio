from __future__ import annotations

from collections.abc import Callable

import torch


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
) -> torch.Tensor:
    """Training-free masked refinement inspired by LanPaint's inner iterations.

    This is an independent, conservative implementation for Krea's flow latents:
    known regions are re-anchored every inner step and masked regions take small
    flow steps using repeated model velocity estimates.
    """
    steps = max(1, int(inner_steps))
    strength = max(0.0, min(2.0, float(strength)))
    masked = mask.to(device=current.device, dtype=current.dtype)
    known = known_trajectory.to(device=current.device, dtype=current.dtype)
    latent = masked * current + (1.0 - masked) * known
    dt = ((float(tprev) - float(tcurr)) / steps) * strength

    for index in range(steps):
        local_t = float(tcurr) + (float(tprev) - float(tcurr)) * (index / steps)
        velocity = velocity_fn(latent, local_t)
        latent = latent + dt * velocity
        latent = masked * latent + (1.0 - masked) * known

    return latent
