from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from settings import LOCAL_AI_DIR, MODELS_DIR


@dataclass(frozen=True)
class QualityAssetSpec:
    id: str
    repo_id: str
    filename: str | None
    local_path: Path
    kind: Literal["file", "snapshot"]
    purpose: str
    allow_patterns: list[str] | None = None


def asset_specs() -> list[QualityAssetSpec]:
    diffusion_dir = MODELS_DIR / "krea2" / "diffusion_models"
    return [
        QualityAssetSpec(
            id="krea2_turbo_bf16",
            repo_id="Comfy-Org/Krea-2",
            filename="diffusion_models/krea2_turbo_bf16.safetensors",
            local_path=diffusion_dir / "krea2_turbo_bf16.safetensors",
            kind="file",
            purpose="Krea 2 Turbo BF16 benchmark and quality comparison",
        ),
        QualityAssetSpec(
            id="krea2_raw_bf16",
            repo_id="Comfy-Org/Krea-2",
            filename="diffusion_models/krea2_raw_bf16.safetensors",
            local_path=diffusion_dir / "krea2_raw_bf16.safetensors",
            kind="file",
            purpose="Krea 2 RAW BF16 benchmark and research-quality comparison",
        ),
        QualityAssetSpec(
            id="flux_fill",
            repo_id="black-forest-labs/FLUX.1-Fill-dev",
            filename=None,
            local_path=LOCAL_AI_DIR / "flux1_fill_dev",
            kind="snapshot",
            purpose="FLUX Fill strict inpaint/outpaint provider",
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "tokenizer/*",
                "tokenizer_2/*",
                "text_encoder/*",
                "text_encoder_2/*",
                "transformer/*",
                "vae/*",
                "*.safetensors",
                "*.json",
                "*.txt",
            ],
        ),
    ]


def asset_by_id(asset_id: str) -> QualityAssetSpec:
    for spec in asset_specs():
        if spec.id == asset_id:
            return spec
    raise KeyError(asset_id)


def asset_installed(spec: QualityAssetSpec) -> bool:
    if spec.kind == "file":
        return spec.local_path.exists()
    return (spec.local_path / "model_index.json").exists()


def asset_status(spec: QualityAssetSpec, *, has_hf_token: bool = False) -> dict:
    installed = asset_installed(spec)
    needs_token = spec.id == "flux_fill" and not installed and not has_hf_token
    return {
        "id": spec.id,
        "repo_id": spec.repo_id,
        "filename": spec.filename,
        "local_path": str(spec.local_path),
        "purpose": spec.purpose,
        "installed": installed,
        "needs_token": needs_token,
        "gated": spec.id == "flux_fill",
        "setup_url": f"https://huggingface.co/{spec.repo_id}",
    }


def download_asset(spec: QualityAssetSpec, *, token: str | None = None) -> Path:
    import os

    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    spec.local_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.kind == "file":
        from huggingface_hub import hf_hub_download

        if spec.filename is None:
            raise ValueError(f"{spec.id} has no filename")
        return Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=spec.filename,
                local_dir=str(spec.local_path.parent.parent),
                token=token or None,
            )
        )

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=str(spec.local_path),
            token=token or None,
            allow_patterns=spec.allow_patterns,
            local_dir_use_symlinks=False,
        )
    )
