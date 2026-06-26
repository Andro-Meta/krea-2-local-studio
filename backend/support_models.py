from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from settings import HF_HOME, LOCAL_AI_DIR, settings

SUPPORT_MODELS = [
    {
        "id": "qwen3_vl",
        "label": "Qwen3-VL image/text encoder",
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "local_dir": LOCAL_AI_DIR / "qwen3_vl_4b_instruct",
        "purpose": "Krea moodboard reference-image conditioning tensors",
        "required": ["config.json", "model.safetensors.index.json"],
        "allow_patterns": None,
    },
    {
        "id": "qwen_image_vae",
        "label": "Qwen-Image VAE",
        "repo_id": "Qwen/Qwen-Image",
        "local_dir": LOCAL_AI_DIR / "qwen_image",
        "purpose": "Krea image encode/decode for img2img, inpaint, and outpaint",
        "required": ["vae/config.json"],
        "allow_patterns": ["vae/*"],
    },
]


def _repo_cache_dir(repo_id: str) -> Path:
    return Path(HF_HOME) / "hub" / ("models--" + repo_id.replace("/", "--"))


def _snapshot_dirs(repo_id: str) -> list[Path]:
    snapshots = _repo_cache_dir(repo_id) / "snapshots"
    if not snapshots.exists():
        return []
    return [p for p in snapshots.iterdir() if p.is_dir()]


def _has_required(repo_id: str, required: list[str]) -> bool:
    for snapshot in _snapshot_dirs(repo_id):
        if all((snapshot / item).exists() for item in required):
            return True
    return False


def _has_required_in_dir(path: Path, required: list[str]) -> bool:
    return all((path / item).exists() for item in required)


def _model_by_id(model_id: str) -> dict[str, Any]:
    for model in SUPPORT_MODELS:
        if model["id"] == model_id:
            return model
    raise KeyError(model_id)


def support_model_path(model_id: str) -> Path:
    model = _model_by_id(model_id)
    path = Path(model["local_dir"])
    if not _has_required_in_dir(path, model["required"]):
        raise FileNotFoundError(
            f"{model['label']} is missing. Use System > Krea Moodboard Conditioning / "
            "Local AI Assets, or run scripts/download_support_models.py."
        )
    return path


def support_model_status() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for model in SUPPORT_MODELS:
        local_dir = Path(model["local_dir"])
        installed = _has_required_in_dir(local_dir, model["required"])
        items.append({
            "id": model["id"],
            "label": model["label"],
            "repo_id": model["repo_id"],
            "purpose": model["purpose"],
            "installed": installed,
            "path": str(local_dir),
            "cache_dir": str(_repo_cache_dir(model["repo_id"])),
            "legacy_cache_installed": _has_required(model["repo_id"], model["required"]),
        })
    return items


def download_support_models() -> list[dict[str, Any]]:
    from huggingface_hub import snapshot_download

    token = settings.hf_token or os.environ.get("HF_TOKEN") or None
    results: list[dict[str, Any]] = []
    for model in SUPPORT_MODELS:
        path = snapshot_download(
            repo_id=model["repo_id"],
            cache_dir=HF_HOME,
            local_dir=str(model["local_dir"]),
            token=token,
            allow_patterns=model["allow_patterns"],
            local_dir_use_symlinks=False,
        )
        results.append({
            "id": model["id"],
            "repo_id": model["repo_id"],
            "path": path,
            "installed": _has_required_in_dir(Path(model["local_dir"]), model["required"]),
        })
    return results
