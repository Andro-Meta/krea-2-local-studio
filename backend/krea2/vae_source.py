"""Resolve which VAE the pipeline should decode with.

Stock Krea 2 uses the Qwen Image VAE. The community found alternative/merged VAEs
(Qwen HDR VAE, "krea-2-real-vae", clear VAE) can sharpen decode for some content.
This resolver lets a user opt into an override VAE safetensors; if the override
is missing it falls back to the stock VAE, so default behavior is unchanged.

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
