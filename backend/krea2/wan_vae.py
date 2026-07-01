from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

WAN_LATENTS_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
WAN_LATENTS_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]


def is_wan_vae_state_dict(sd: dict) -> bool:
    keys = set(sd.keys())
    return {"encoder.conv1.weight", "decoder.conv1.weight", "conv1.weight", "conv2.weight"}.issubset(keys)


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temporal_padding = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x):
        kernel_t = self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size
        pad_t = max(self._temporal_padding, max(0, int(kernel_t) - int(x.shape[2])))
        if pad_t > 0:
            pad = torch.zeros(x.shape[0], x.shape[1], pad_t, x.shape[3], x.shape[4], device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=2)
        return super().forward(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()
        shape = (dim, 1, 1) if images and channel_first else (dim, 1, 1, 1) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        dim = 1 if self.channel_first else -1
        return F.normalize(x, dim=dim) * self.scale * self.gamma.to(x) + (self.bias.to(x) if self.bias is not None else 0)


class Resample(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.mode = mode
        if mode in {"upsample2d", "upsample3d"}:
            self.resample = nn.Sequential(nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)) if mode == "upsample3d" else None
        elif mode in {"downsample2d", "downsample3d"}:
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)) if mode == "downsample3d" else None
        else:
            self.resample = nn.Identity()
            self.time_conv = None

    def forward(self, x):
        b, c, t, h, w = x.shape
        if self.mode == "upsample3d":
            x = self.time_conv(x)
            x = x.reshape(b, 2, c, t, h, w)
            x = torch.stack((x[:, 0], x[:, 1]), dim=3).reshape(b, c, t * 2, h, w)
        elif self.mode == "downsample3d":
            x = self.time_conv(x)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        return rearrange(x, "(b t) c h w -> b c t h w", t=t)


class Resample2D(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        if mode == "upsample2d":
            self.resample = nn.Sequential(nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return self.resample(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, conv3d: bool = False):
        super().__init__()
        conv = CausalConv3d if conv3d else nn.Conv2d
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            conv(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            conv(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = conv(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        is_video = x.ndim == 5
        if is_video:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
        identity = x
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(self.norm(x)).reshape(b, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        out = F.scaled_dot_product_attention(q, k, v).squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
        out = identity + self.proj(out)
        if is_video:
            out = rearrange(out, "(b t) c h w -> b c t h w", t=t)
        return out


class Encoder3D(nn.Module):
    def __init__(self, dim=96, z_dim=32, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=(), temporal_downsample=(False, True, True), dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [1, *dim_mult]]
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        layers = []
        scale = 1.0
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_dim, out_dim, dropout, conv3d=True))
                if scale in attn_scales:
                    layers.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                layers.append(Resample(out_dim, "downsample3d" if temporal_downsample[i] else "downsample2d"))
                scale /= 2.0
        self.downsamples = nn.Sequential(*layers)
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout, conv3d=True), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout, conv3d=True))
        self.head = nn.Sequential(RMSNorm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsamples(x)
        x = self.middle(x)
        return self.head(x)


class Decoder3D(nn.Module):
    def __init__(self, dim=96, z_dim=16, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=(), temporal_upsample=(True, True, False), dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout, conv3d=True), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout, conv3d=True))
        layers = []
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in {1, 2, 3}:
                in_dim //= 2
            for _ in range(num_res_blocks + 1):
                layers.append(ResidualBlock(in_dim, out_dim, dropout, conv3d=True))
                if scale in attn_scales:
                    layers.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                layers.append(Resample(out_dim, "upsample3d" if temporal_upsample[i] else "upsample2d"))
                scale *= 2.0
        self.upsamples = nn.Sequential(*layers)
        self.head = nn.Sequential(RMSNorm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x):
        x = self.upsamples(self.middle(self.conv1(x)))
        return self.head(x)


class WanVAE2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder3D(z_dim=32)
        self.conv1 = CausalConv3d(32, 32, 1)
        self.conv2 = CausalConv3d(16, 16, 1)
        self.decoder = Decoder3D(z_dim=16)

    def encode(self, x):
        mu, _log_var = self.conv1(self.encoder(x)).chunk(2, dim=1)
        return mu

    def decode(self, z):
        return self.decoder(self.conv2(z))


class WanAutoencoder(nn.Module):
    config = type("WanVAEConfig", (), {"latent_channels": 16, "latents_mean": WAN_LATENTS_MEAN, "latents_std": WAN_LATENTS_STD})()

    def __init__(self, state_dict: dict):
        super().__init__()
        self.model = WanVAE2D()
        self.model.load_state_dict(state_dict, strict=True)

    def requires_grad_(self, requires_grad: bool = False):
        self.model.requires_grad_(requires_grad)
        return self

    def enable_tiling(self):
        return None

    def disable_tiling(self):
        return None

    def encode(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        z = self.model.encode(x)
        return type("WanEncodeOutput", (), {"latent_dist": type("WanLatentDist", (), {"sample": lambda _self: z})()})()

    def decode(self, z):
        if z.ndim == 4:
            z = z.unsqueeze(2)
        out = self.model.decode(z)
        if out.ndim == 5 and out.shape[2] != 1:
            out = out[:, :, :1]
        return type("WanDecodeOutput", (), {"sample": out})()
