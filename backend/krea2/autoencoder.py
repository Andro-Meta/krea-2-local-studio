"""Krea 2 QwenAutoencoder — wraps AutoencoderKLQwenImage from diffusers.
Adds encode() for img2img support (not in official release).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class QwenAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from diffusers import AutoencoderKLQwenImage
        from support_models import support_model_path

        self.ae = AutoencoderKLQwenImage.from_pretrained(
            str(support_model_path("qwen_image_vae")), subfolder="vae"
        )
        self.ae.requires_grad_(False)

        # Normalization constants — load from model config when available
        cfg = self.ae.config
        n_ch = getattr(cfg, "latent_channels", 16)

        self.register_buffer("latents_mean", torch.zeros(1, n_ch, 1, 1))
        self.register_buffer("latents_std", torch.ones(1, n_ch, 1, 1))

        if getattr(cfg, "latents_mean", None) is not None:
            self.latents_mean.copy_(
                torch.tensor(cfg.latents_mean, dtype=torch.float32).view(1, -1, 1, 1)
            )
        if getattr(cfg, "latents_std", None) is not None:
            self.latents_std.copy_(
                torch.tensor(cfg.latents_std, dtype=torch.float32).view(1, -1, 1, 1)
            )
        elif hasattr(cfg, "scaling_factor") and cfg.scaling_factor:
            self.latents_std.fill_(1.0 / cfg.scaling_factor)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Latents → pixel tensors.  x: (B, C, H, W) normalized latent."""
        dtype = x.dtype
        x = x.float()
        # Denormalize in 4D: latents_std/mean are (1, C, 1, 1) and broadcast
        # per-channel here. Doing this AFTER adding the temporal axis would
        # right-align the std's channel dim onto the temporal axis and silently
        # expand T from 1 to C, making the video VAE emit C*4-3 frames.
        x = x * self.latents_std + self.latents_mean
        x = rearrange(x, "b c h w -> b c 1 h w")
        out = self.ae.decode(x.to(dtype)).sample
        return rearrange(out, "b c 1 h w -> b c h w")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pixel tensors → normalized latents.

        Args:
            x: (B, C, H, W) in range [-1, 1]  OR  (B, H, W, C) in range [0, 1]

        Returns:
            (B, C, H/8, W/8) normalized latent
        """
        # Accept BHWC [0,1] → convert to BCHW [-1,1]
        if x.ndim == 4 and x.shape[-1] in (1, 3, 4):
            x = rearrange(x, "b h w c -> b c h w")
            x = x * 2.0 - 1.0

        dtype = x.dtype
        # Cast back to the VAE's weight dtype before the conv stack — .float()
        # alone would feed fp32 activations into bf16 weights (dtype mismatch).
        x = rearrange(x.float(), "b c h w -> b c 1 h w")
        posterior = self.ae.encode(x.to(dtype))
        z = posterior.latent_dist.sample()
        z = rearrange(z, "b c 1 h w -> b c h w")
        # Normalize: (z - mean) / std
        z = (z - self.latents_mean) / self.latents_std
        return z.to(dtype)
