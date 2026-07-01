"""Resolve which VAE the pipeline should decode with.

Krea Studio prefers Wan 2.1 VAE when configured because the community uses it
as the higher-detail Qwen/Krea decoder. If the configured path is missing it
falls back to the stock Qwen Image VAE.

Torch-free so the selection logic is unit-testable in lightweight CI.
"""
from __future__ import annotations

from pathlib import Path


def resolve_vae_source(override_path: str | None) -> dict:
    """Return {"kind": "override"|"stock", "path": str}.

    `override` only when a non-empty path is provided and the file exists;
    otherwise `stock` (the bundled Qwen Image VAE).
    """
    path = str(override_path or "").strip()
    if path and Path(path).is_file():
        return {"kind": "override", "path": path}
    return {"kind": "stock", "path": ""}
