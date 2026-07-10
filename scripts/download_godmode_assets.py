"""Download the God Mode stage-2 assets (Z-Image Turbo refine pack, ~19GB).

God Mode renders with Krea 2, refines with Z-Image Turbo, then upscales with
SeedVR2 and runs FaceDetailer. This fetches the Z-Image Turbo pack into the
ComfyUI model folders (they are not under models/ because extra_model_paths
does not map these classes for the Z-Image loaders):

  ComfyUI/models/diffusion_models/z_image_turbo_bf16.safetensors  (~11.5GB)
  ComfyUI/models/text_encoders/qwen_3_4b.safetensors              (~7.5GB)
  ComfyUI/models/vae/ae.safetensors                               (~0.3GB)

Uses HF_TOKEN from .env when present for faster/authenticated downloads.
Idempotent: existing files are skipped.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY_MODELS = ROOT / "ComfyUI" / "models"

REPO_ID = "Comfy-Org/z_image_turbo"
FILES = (
    ("split_files/diffusion_models/z_image_turbo_bf16.safetensors", "diffusion_models"),
    ("split_files/text_encoders/qwen_3_4b.safetensors", "text_encoders"),
    ("split_files/vae/ae.safetensors", "vae"),
)


def _load_env_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return ""


def main() -> int:
    from huggingface_hub import hf_hub_download

    token = _load_env_token() or None
    failures = 0
    for repo_file, subdir in FILES:
        dest_dir = COMFY_MODELS / subdir
        dest = dest_dir / Path(repo_file).name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  present: {dest.name}")
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {repo_file} ...", flush=True)
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename=repo_file, token=token)
            import shutil

            shutil.copyfile(path, dest)
            print(f"  -> {dest} ({dest.stat().st_size / 1e9:.1f} GB)")
        except Exception as exc:
            failures += 1
            print(f"  FAILED {repo_file}: {exc}")
    if failures:
        print(f"{failures} God Mode asset(s) failed. Re-run this script or use System > Quality Assets.")
        return 1
    print("God Mode assets ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
