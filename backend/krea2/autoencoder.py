"""Krea 2 QwenAutoencoder — wraps AutoencoderKLQwenImage from diffusers.
Adds encode() for img2img support (not in official release).
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from krea2.wan_vae import WanAutoencoder, is_wan_vae_state_dict

logger = logging.getLogger(__name__)


def _laplacian_detail_blend(low_src: torch.Tensor, high_src: torch.Tensor, *, blur_radius: int = 24, high_strength: float = 0.65) -> torch.Tensor:
    radius = max(1, int(blur_radius))
    kernel_size = radius * 2 + 1
    sigma = max(float(radius) / 3.0, 0.1)
    coords = torch.arange(kernel_size, device=low_src.device, dtype=torch.float32) - radius
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = (kernel_1d / kernel_1d.sum()).to(dtype=low_src.dtype)
    kernel = (kernel_1d[:, None] @ kernel_1d[None, :]).view(1, 1, kernel_size, kernel_size)
    kernel = kernel.expand(low_src.shape[1], 1, kernel_size, kernel_size)

    def blur(x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (radius, radius, radius, radius), mode="reflect")
        return F.conv2d(padded, kernel.to(dtype=padded.dtype), groups=x.shape[1])

    low = blur(low_src.float()).to(low_src.dtype)
    high = high_src - blur(high_src.float()).to(high_src.dtype)
    return (low + high * float(high_strength)).clamp(-1, 1)


class QwenAutoencoder(nn.Module):
    def __init__(
        self,
        vae_override_path: str | None = None,
        *,
        vae_mode: str = "qwen",
        wan_blend_path: str | None = None,
        blend_blur_radius: int = 24,
        blend_high_strength: float = 0.65,
    ) -> None:
        super().__init__()
        from diffusers import AutoencoderKLQwenImage
        from support_models import support_model_path

        self.ae = AutoencoderKLQwenImage.from_pretrained(
            str(support_model_path("qwen_image_vae")), subfolder="vae"
        )
        self.vae_source = "stock:qwen_image"
        self.vae_mode = str(vae_mode or "qwen")
        self.blend_blur_radius = int(blend_blur_radius or 24)
        self.blend_high_strength = float(blend_high_strength or 0.65)
        self.blend_ae = None
        if self.vae_mode == "qwen_wan_blend" and wan_blend_path:
            self.blend_ae = self._load_wan_file(wan_blend_path)
            if self.blend_ae is not None:
                self.vae_source = f"qwen+wan_blend:{Path(wan_blend_path).name}"
        elif vae_override_path and self.vae_mode != "qwen":
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
                if is_wan_vae_state_dict(sd):
                    self.ae = WanAutoencoder(sd)
                    if "qwen" in p.name.lower():
                        self.vae_source = f"comfy_qwen:{p.name}"
                    elif getattr(self.ae, "upscale_factor", 1) > 1:
                        self.vae_source = f"spacepxl_2x:{p.name}"
                    else:
                        self.vae_source = f"wan_experimental:{p.name}"
                    logger.info("Loaded Wan-layout VAE override: %s", p)
                    return
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

    def _load_wan_file(self, path: str):
        try:
            from safetensors.torch import load_file

            p = Path(path)
            sd = load_file(str(p))
            if is_wan_vae_state_dict(sd):
                logger.info("Loaded Wan detail-blend VAE: %s", p)
                return WanAutoencoder(sd)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wan detail-blend VAE failed (%s); using stock Qwen only.", exc)
        return None

    def _set_tiling(self, enabled: bool) -> None:
        try:
            if enabled:
                self.ae.enable_tiling()
            else:
                self.ae.disable_tiling()
        except Exception:
            # Older diffusers may lack tiling toggles; non-tiled decode still works.
            pass

    def decode(self, x: torch.Tensor, *, tiled: bool = False) -> torch.Tensor:
        """Latents → pixel tensors.  x: (B, C, H, W) normalized latent.

        `tiled=True` proactively uses the diffusers tiled VAE decode (overlap +
        blend) for large/low-VRAM cases. Either way, a CUDA OOM during decode is
        caught and retried with tiling, mirroring ComfyUI's decode→tiled fallback,
        so a 2K+ decode never hard-fails on VRAM.
        """
        dtype = x.dtype
        x = x.float()
        # Denormalize in 4D: latents_std/mean are (1, C, 1, 1) and broadcast
        # per-channel here. Doing this AFTER adding the temporal axis would
        # right-align the std's channel dim onto the temporal axis and silently
        # expand T from 1 to C, making the video VAE emit C*4-3 frames.
        x = x * self.latents_std + self.latents_mean
        x = rearrange(x, "b c h w -> b c 1 h w").to(dtype)
        oom = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
        try:
            if tiled:
                self._set_tiling(True)
            out = self.ae.decode(x).sample
            if self.vae_mode == "qwen_wan_blend" and self.blend_ae is not None:
                wan_out = self.blend_ae.to(device=x.device, dtype=x.dtype).decode(x).sample
                out_4d = rearrange(out, "b c 1 h w -> b c h w")
                wan_4d = rearrange(wan_out, "b c 1 h w -> b c h w")
                out = rearrange(
                    _laplacian_detail_blend(
                        out_4d,
                        wan_4d,
                        blur_radius=self.blend_blur_radius,
                        high_strength=self.blend_high_strength,
                    ),
                    "b c h w -> b c 1 h w",
                )
        except oom:
            logger.warning("VAE decode hit CUDA OOM; retrying with tiled decode.")
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    # Best-effort reclaim before the tiled retry; ignore if it fails.
                    pass
            self._set_tiling(True)
            out = self.ae.decode(x).sample
        finally:
            # Restore non-tiled state so later small decodes aren't slowed.
            self._set_tiling(False)
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
