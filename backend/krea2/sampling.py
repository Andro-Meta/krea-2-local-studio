"""Krea 2 flow-matching sampler.

Faithful port of the official krea-ai/krea-2 sampling.py (text-first token
ordering, 3-axis RoPE positions, exp(mu) timestep schedule), extended with:
  - progress_callback support
  - pre-encoded conditioning (txt/txtmask passed directly; encoder optional)
  - CFG via pre-encoded negative conditioning
  - img2img: init_latent + denoise (partial-noise start)
  - inpaint: latent-space mask compositing
  - batched generation
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image

from . import schedulers
from .lanpaint_sampler import lanpaint_masked_inner_update
from .lanpaint_sampler import LanPaintSettings
from .sampler_registry import normalize_sampler_name

# Qwen-Image VAE spatial compression and MMDiT patch size.
COMPRESSION = 8
PATCH = 2
LATENT_CHANNELS = 16




def prepare(img, txtlen, patch, txtmask):
    """Patchify the latent and build combined TEXT-FIRST position / mask tensors.

    Returns (img_tokens, pos, mask) where the sequence is [text, image] to match
    the MMDiT forward (combined = cat(context, img)).
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch

    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    if txtmask is None:
        txtmask = torch.ones(b, txtlen, device=img.device, dtype=torch.bool)
    else:
        txtmask = txtmask.bool()

    mask = torch.cat((txtmask, imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None, scheduler="simple"):
    """Resolution-aware flow-matching schedule (t: 1 -> 0).

    mu is interpolated in image-sequence length between (x1,y1) and (x2,y2),
    then time-shifts a base 1->0 grid. The grid's *density* is set by the
    scheduler (simple/normal = uniform default; beta = U-shaped; etc). Pass
    explicit mu to pin a constant shift (Turbo was distilled at mu=1.15).
    """
    base = schedulers.base_grid(steps, scheduler)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    # exp(mu) time-shift (ComfyUI flux_time_shift). Endpoints 1->1 and 0->0 are
    # preserved; the shift only pushes interior density toward high noise.
    shifted = []
    e_mu = math.exp(mu)
    for t in base:
        if t <= 0.0:
            shifted.append(0.0)
        elif t >= 1.0:
            shifted.append(1.0)
        else:
            shifted.append(e_mu / (e_mu + (1.0 / t - 1.0) ** sigma))
    return shifted


def _differential_mask_for_timestep(
    denoise_mask: torch.Tensor,
    *,
    tcurr: float,
    t_start: float,
    strength: float = 1.0,
) -> torch.Tensor:
    """Convert soft mask values into a per-step active denoise mask.

    This ports ComfyUI Differential Diffusion's core idea to Krea's flow
    schedule: white mask values regenerate for the full trajectory, while gray
    values join later and therefore receive less total denoise.
    """
    if t_start <= 0:
        threshold = 0.0
    else:
        threshold = max(0.0, min(1.0, float(tcurr) / float(t_start)))
    binary_mask = (denoise_mask >= threshold).to(denoise_mask.dtype)
    if 0.0 <= strength < 1.0:
        return strength * binary_mask + (1.0 - strength) * denoise_mask
    return binary_mask


def euler_flow_step(img: torch.Tensor, velocity: torch.Tensor, *, tcurr: float, tprev: float) -> torch.Tensor:
    return img + (tprev - tcurr) * velocity


def heun_flow_step(
    img: torch.Tensor,
    velocity: torch.Tensor,
    *,
    tcurr: float,
    tprev: float,
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
) -> torch.Tensor:
    predictor = euler_flow_step(img, velocity, tcurr=tcurr, tprev=tprev)
    corrected = velocity_fn(predictor, tprev)
    return img + (tprev - tcurr) * 0.5 * (velocity + corrected)


def euler_ancestral_flow_step(
    img: torch.Tensor,
    velocity: torch.Tensor,
    *,
    tcurr: float,
    tprev: float,
    noise: torch.Tensor,
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> torch.Tensor:
    """Ancestral Euler step for rectified flow.

    Port of ComfyUI ``sample_euler_ancestral_RF`` using alpha = 1 - t. The model
    ``velocity`` at tcurr yields ``denoised = img - v*tcurr``; we step down a
    fraction (eta controls stochasticity) and re-inject fresh Gaussian noise.
    """
    denoised = img - velocity * tcurr
    if tprev <= 0.0:
        return denoised
    downstep_ratio = 1.0 + (tprev / tcurr - 1.0) * eta
    sigma_down = tprev * downstep_ratio
    alpha_ip1 = 1.0 - tprev
    alpha_down = 1.0 - sigma_down
    renoise = (tprev ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
    ratio = sigma_down / tcurr
    x = ratio * img + (1.0 - ratio) * denoised
    if eta > 0.0:
        x = (alpha_ip1 / alpha_down) * x + noise * s_noise * renoise
    return x


def euler_cfgpp_flow_step(
    img: torch.Tensor,
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    *,
    tcurr: float,
    tprev: float,
    noise: torch.Tensor,
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> torch.Tensor:
    """CFG++ (ancestral) Euler step for rectified flow.

    Port of ComfyUI ``sample_euler_ancestral_cfg_pp`` with the CONST
    (alpha = 1 - t) mapping. The integration *base* is the full CFG ``denoised``
    while the step *direction* is taken from the uncond prediction — this is the
    CFG++ trick that tames over-saturation and improves few-step adherence.
    ``eta = s_noise = 0`` reduces this to deterministic ``euler_cfg_pp``.
    """
    denoised = img - v_cond * tcurr
    uncond_denoised = img - v_uncond * tcurr
    if tprev <= 0.0:
        return denoised
    alpha_s = 1.0 - tcurr
    alpha_t = 1.0 - tprev
    # Direction from the uncond prediction (the CFG++ trick). tcurr is always > 0
    # inside the loop, so this division is safe.
    d = (img - alpha_s * uncond_denoised) / tcurr
    # Ancestral split rewritten via the product ratio r = (tprev*alpha_s)/(tcurr*alpha_t)
    # to avoid dividing by alpha_s, which is 0 at the t=1 starting step.
    if eta > 0.0 and alpha_t > 0.0:
        r = (tprev * alpha_s) / (tcurr * alpha_t)
        factor = min(1.0, eta * max(0.0, 1.0 - r * r) ** 0.5)
    else:
        factor = 0.0
    sigma_down_scaled = tprev * (max(0.0, 1.0 - factor * factor) ** 0.5)
    x = alpha_t * denoised + sigma_down_scaled * d
    if eta > 0.0 and s_noise > 0.0:
        x = x + noise * s_noise * (tprev * factor)
    return x


# Samplers actually integrated by sample(). Aliases normalize into these.
_FLOW_SAMPLERS = {
    "euler_flow",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "euler_cfg_pp",
    "exp_heun_2_x0_sde",
    "lanpaint_experimental",
}


def _validate_sampler(sampler: str) -> None:
    if normalize_sampler_name(sampler) not in _FLOW_SAMPLERS:
        raise ValueError(f"Unknown sampler: {sampler}")


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    ae,
    encoder,
    prompts: Optional[list[str]],
    *,
    # Pre-encoded conditioning (skip encoder when provided)
    txt: Optional[torch.Tensor] = None,
    txtmask: Optional[torch.Tensor] = None,
    negative_txt: Optional[torch.Tensor] = None,
    negative_txtmask: Optional[torch.Tensor] = None,
    negative_prompts: Optional[list[str]] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    guidance: float = 0.0,
    seed: int = 0,
    minres: int = 256,
    maxres: int = 1280,
    y1: float = 0.5,
    y2: float = 1.15,
    mu: Optional[float] = None,
    batch_size: int = 1,
    # img2img
    init_latent: Optional[torch.Tensor] = None,
    denoise: float = 1.0,
    # inpaint (latent-space composite after generation)
    mask: Optional[torch.Tensor] = None,
    init_latent_clean: Optional[torch.Tensor] = None,
    differential_mask: bool = False,
    differential_strength: float = 1.0,
    sampler: str = "euler_flow",
    scheduler: str = "simple",
    lanpaint_inner_steps: int = 3,
    lanpaint_strength: float = 1.0,
    lanpaint_settings: LanPaintSettings | None = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    tiled_decode: bool = False,
) -> list[Image.Image]:
    sampler = normalize_sampler_name(sampler)
    _validate_sampler(sampler)
    align = COMPRESSION * PATCH
    # Round up to a valid patch grid.
    width = ((width + align - 1) // align) * align
    height = ((height + align - 1) // align) * align

    # --- Conditioning ---
    if txt is None:
        assert encoder is not None and prompts is not None
        txt, txtmask = encoder(prompts)
    txt = txt.to(device=device, dtype=dtype)
    if txtmask is not None:
        txtmask = txtmask.to(device=device)

    cfg = guidance > 0.0
    if cfg and negative_txt is None:
        assert encoder is not None
        neg = negative_prompts or [""] * txt.shape[0]
        negative_txt, negative_txtmask = encoder(neg)
    neg_context = None
    unpos = None
    unmask = None
    if cfg:
        if negative_txt is None:
            raise ValueError("negative conditioning is required when guidance is enabled")
        neg_context = negative_txt.to(device=device, dtype=dtype)
        if negative_txtmask is not None:
            negative_txtmask = negative_txtmask.to(device=device)

    B = batch_size
    lh, lw = height // COMPRESSION, width // COMPRESSION

    # --- Per-prompt seeded noise (FIXED; reused for the inpaint keep-region trajectory) ---
    noise = torch.cat(
        [
            torch.randn(
                1, LATENT_CHANNELS, lh, lw,
                device=device, dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(B)
        ],
        dim=0,
    )

    # --- Timestep schedule ---
    # seq_len for the mu interpolation uses the patchified image-token count.
    img_tokens = (lh // PATCH) * (lw // PATCH)
    x1 = (minres // align) ** 2
    x2 = (maxres // align) ** 2
    ts = timesteps(img_tokens, steps, x1, x2, y1=y1, y2=y2, mu=mu, scheduler=scheduler)

    # Patchify pure noise → tokens (also builds RoPE positions + attention mask).
    x_noise, pos, seq_mask = prepare(noise, txt.shape[1], PATCH, txtmask)
    if cfg:
        if neg_context is None:
            raise ValueError("negative conditioning is required when guidance is enabled")
        _, unpos, unmask = prepare(noise, neg_context.shape[1], PATCH, negative_txtmask)

    # Clean original in token space (img2img/inpaint) + token-space inpaint mask.
    x0_tok = None
    mask_tok = None
    if init_latent is not None:
        init_latent = init_latent.to(device=device, dtype=dtype)
        x0_tok = rearrange(init_latent, "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                           p1=PATCH, p2=PATCH)
    if mask is not None and x0_tok is not None:
        m = mask.float().to(device=device)
        if m.dim() == 3:
            m = m.unsqueeze(1)
        m = F.interpolate(m, size=(lh, lw), mode="nearest")
        m = F.avg_pool2d(m, PATCH)                              # patch-grid, soft edges
        mask_tok = rearrange(m, "b c h w -> b (h w) c").to(dtype=dtype)  # (b,tok,1) 1=regen

    def _traj(t):  # original's flow trajectory at time t: x_t = (1-t)*x0 + t*noise
        return (1.0 - t) * x0_tok + t * x_noise

    # --- Start point ---
    if x0_tok is not None and denoise < 1.0:
        # img2img / partial inpaint: begin at t≈denoise on the original trajectory.
        start_idx = next((i for i, t in enumerate(ts) if t <= denoise), 0)
        ts = ts[start_idx:]
        img = _traj(ts[0])
    else:
        # txt2img, or full-denoise inpaint: start from pure noise (t≈1).
        img = x_noise

    # Ancestral / CFG++ samplers re-inject fresh noise each step; use a dedicated
    # seeded generator so results stay reproducible and independent of the init noise.
    ancestral = sampler in ("euler_ancestral", "euler_ancestral_cfg_pp", "euler_cfg_pp")
    cfgpp = sampler in ("euler_ancestral_cfg_pp", "euler_cfg_pp")
    step_eta = 0.0 if sampler == "euler_cfg_pp" else 1.0
    anc_gen = torch.Generator(device=device).manual_seed(seed + 0x5EED)

    def _step_noise(ref: torch.Tensor) -> torch.Tensor:
        return torch.randn(ref.shape, dtype=ref.dtype, device=ref.device, generator=anc_gen)

    # --- Euler ODE integration (with per-step inpaint keep-region compositing) ---
    n_steps = len(ts) - 1
    t_start = ts[0] if ts else 1.0
    for step_idx, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        if progress_cb is not None:
            progress_cb(step_idx, n_steps)
        # Inpaint: re-noise the KEEP region (mask=0) back onto the original trajectory
        # BEFORE the model call, so the DiT always sees coherent surrounding context
        # (ComfyUI SetLatentNoiseMask / scale_latent_inpaint equivalent for flow).
        if mask_tok is not None:
            active_mask = (
                _differential_mask_for_timestep(
                    mask_tok,
                    tcurr=tcurr,
                    t_start=t_start,
                    strength=differential_strength,
                )
                if differential_mask
                else mask_tok
            )
            img = active_mask * img + (1.0 - active_mask) * _traj(tcurr)
        else:
            active_mask = None
        def _velocity_parts(latent: torch.Tensor, local_t: float):
            t_local = torch.full((len(latent),), local_t, dtype=latent.dtype, device=latent.device)
            cond_local = model(img=latent, context=txt, t=t_local, pos=pos, mask=seq_mask)
            if cfg:
                if neg_context is None or unpos is None:
                    raise ValueError("negative conditioning is required when guidance is enabled")
                uncond_local = model(img=latent, context=neg_context, t=t_local, pos=unpos, mask=unmask)
                return cond_local, uncond_local
            return cond_local, None

        def _velocity(latent: torch.Tensor, local_t: float) -> torch.Tensor:
            cond_local, uncond_local = _velocity_parts(latent, local_t)
            if uncond_local is None:
                return cond_local
            return uncond_local + guidance * (cond_local - uncond_local)

        if sampler == "lanpaint_experimental":
            if active_mask is None:
                raise ValueError("lanpaint_experimental requires an inpaint mask")
            img = lanpaint_masked_inner_update(
                img,
                _traj(tcurr),
                active_mask,
                tcurr=tcurr,
                tprev=tprev,
                velocity_fn=_velocity,
                inner_steps=lanpaint_inner_steps,
                strength=lanpaint_strength,
                settings=lanpaint_settings,
            )

        if cfgpp:
            cond_v, uncond_v = _velocity_parts(img, tcurr)
            v_cond = cond_v if uncond_v is None else uncond_v + guidance * (cond_v - uncond_v)
            v_uncond = cond_v if uncond_v is None else uncond_v
            img = euler_cfgpp_flow_step(
                img, v_cond, v_uncond,
                tcurr=tcurr, tprev=tprev,
                noise=_step_noise(img), eta=step_eta, s_noise=1.0,
            )
        elif ancestral:
            v = _velocity(img, tcurr)
            img = euler_ancestral_flow_step(
                img, v, tcurr=tcurr, tprev=tprev,
                noise=_step_noise(img), eta=step_eta, s_noise=1.0,
            )
        elif sampler == "exp_heun_2_x0_sde":
            v = _velocity(img, tcurr)
            img = heun_flow_step(img, v, tcurr=tcurr, tprev=tprev, velocity_fn=_velocity)
        else:
            v = _velocity(img, tcurr)
            img = euler_flow_step(img, v, tcurr=tcurr, tprev=tprev)
    if progress_cb is not None:
        progress_cb(n_steps, n_steps)

    # Final clean composite of the keep region (t=0, no noise).
    if mask_tok is not None:
        img = mask_tok * img + (1.0 - mask_tok) * x0_tok

    # --- Unpatchify → latent ---
    latents = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=PATCH, pw=PATCH, h=lh // PATCH, w=lw // PATCH,
    )

    # --- Decode ---
    images: list[Image.Image] = []
    for i in range(B):
        decoded = ae.decode(latents[i : i + 1], tiled=tiled_decode)   # (1, C, H, W)
        pixel = decoded.clamp(-1, 1).float()
        pixel = (pixel + 1.0) / 2.0
        pixel = pixel.squeeze(0).permute(1, 2, 0)    # (H, W, C)
        pixel = (pixel * 255).round().byte().cpu().numpy()
        images.append(Image.fromarray(pixel))
    return images
