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


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    """Resolution-aware flow-matching schedule (t: 1 -> 0).

    mu is interpolated in image-sequence length between (x1,y1) and (x2,y2),
    then time-shifts a uniform 1->0 grid. Pass explicit mu to pin a constant
    shift (Turbo was distilled at mu=1.15).
    """
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


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
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> list[Image.Image]:
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
    if cfg:
        if negative_txt is None:
            raise ValueError("negative conditioning is required when guidance is enabled")
        negative_txt = negative_txt.to(device=device, dtype=dtype)
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
    ts = timesteps(img_tokens, steps, x1, x2, y1=y1, y2=y2, mu=mu)

    # Patchify pure noise → tokens (also builds RoPE positions + attention mask).
    x_noise, pos, seq_mask = prepare(noise, txt.shape[1], PATCH, txtmask)
    if cfg:
        _, unpos, unmask = prepare(noise, negative_txt.shape[1], PATCH, negative_txtmask)

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

    # --- Euler ODE integration (with per-step inpaint keep-region compositing) ---
    n_steps = len(ts) - 1
    for step_idx, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        if progress_cb is not None:
            progress_cb(step_idx, n_steps)
        # Inpaint: re-noise the KEEP region (mask=0) back onto the original trajectory
        # BEFORE the model call, so the DiT always sees coherent surrounding context
        # (ComfyUI SetLatentNoiseMask / scale_latent_inpaint equivalent for flow).
        if mask_tok is not None:
            img = mask_tok * img + (1.0 - mask_tok) * _traj(tcurr)
        t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
        cond = model(img=img, context=txt, t=t, pos=pos, mask=seq_mask)
        if cfg:
            uncond = model(img=img, context=negative_txt, t=t, pos=unpos, mask=unmask)
            v = uncond + guidance * (cond - uncond)
        else:
            v = cond
        img = img + (tprev - tcurr) * v
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
        decoded = ae.decode(latents[i : i + 1])      # (1, C, H, W)
        pixel = decoded.clamp(-1, 1).float()
        pixel = (pixel + 1.0) / 2.0
        pixel = pixel.squeeze(0).permute(1, 2, 0)    # (H, W, C)
        pixel = (pixel * 255).round().byte().cpu().numpy()
        images.append(Image.fromarray(pixel))
    return images
