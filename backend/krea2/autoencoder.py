"""Krea 2 QwenAutoencoder — wraps AutoencoderKLQwenImage from diffusers.
Adds encode() for img2img support (not in official release).
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)


class QwenAutoencoder(nn.Module):
    def __init__(self, vae_override_path: str | None = None) -> None:
        super().__init__()
        from diffusers import AutoencoderKLQwenImage
        from support_models import support_model_path

        self.ae = AutoencoderKLQwenImage.from_pretrained(
            str(support_model_path("qwen_image_vae")), subfolder="vae"
        )
        self.vae_source = "stock"
        # Optional experimental override (Qwen HDR / "real" / clear VAE). Any
        # failure falls back to the stock VAE already loaded above, so default
        # behavior is never affected.
        if vae_override_path:
            self._apply_override(vae_override_path)
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

    def _apply_override(self, path: str) -> None:
        """Best-effort load of an alternative Qwen-Image VAE; keep stock on failure.

        Supports a diffusers VAE directory (with a `vae/` subfolder or root config)
        or a single comfy-style safetensors whose keys substantially match the
        stock VAE. Anything unexpected -> keep the already-loaded stock VAE.
        """
        try:
            from diffusers import AutoencoderKLQwenImage

            p = Path(path)
            if p.is_dir():
                sub = "vae" if (p / "vae").exists() else ""
                self.ae = AutoencoderKLQwenImage.from_pretrained(str(p), subfolder=sub or None)
                self.vae_source = f"override:dir:{p.name}"
                logger.info("Loaded override VAE directory: %s", p)
                return
            if p.is_file() and p.suffix == ".safetensors":
                from safetensors.torch import load_file

                sd = load_file(str(p))
                ref_keys = set(self.ae.state_dict().keys())
                matched = sum(1 for k in sd if k in ref_keys)
                if matched < max(1, len(ref_keys) // 2):
                    logger.warning(
                        "Override VAE %s matched only %d/%d keys; keeping stock VAE. "
                        "Use a diffusers-format Qwen Image VAE directory for overrides.",
                        p.name, matched, len(ref_keys),
                    )
                    return
                missing, unexpected = self.ae.load_state_dict(sd, strict=False)
                self.vae_source = f"override:file:{p.name}"
                logger.info(
                    "Applied override VAE %s (matched=%d, missing=%d, unexpected=%d)",
                    p.name, matched, len(missing), len(unexpected),
                )
                return
            logger.warning("VAE override path is not a usable file/dir: %s", path)
        except Exception as exc:  # noqa: BLE001 - never let an override break loading
            logger.warning("VAE override failed (%s); keeping stock VAE.", exc)

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
