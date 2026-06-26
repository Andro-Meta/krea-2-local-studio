"""Download official krea-ai/krea-2 source files and LoRA weights.

Called by install.bat after venv creation.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/krea-ai/krea-2/main"
TARGET = Path(__file__).resolve().parent.parent / "backend" / "krea2"
LORAS_DIR = Path(__file__).resolve().parent.parent / "models" / "loras"

# mmdit.py is pure-official and fetched fresh from upstream (too complex to hand-roll).
# The other three ship WITH this project — they carry critical fixes and extensions
# (4D encoder conditioning, text-first sampler, VAE temporal/dtype fixes, img2img/
# inpaint). The upstream versions are incompatible (and autoencoder.py isn't even in
# the official repo), so we must NOT fetch them — only verify they're present.
ALWAYS_DOWNLOAD = ["mmdit.py"]
REQUIRED_LOCAL = ["autoencoder.py", "encoder.py", "sampling.py"]

# Official LoRAs from Comfy-Org/Krea-2 on HuggingFace (public, no token needed)
HF_BASE = "https://huggingface.co/Comfy-Org/Krea-2/resolve/main"
HF_LORA_BASE = f"{HF_BASE}/loras"
OFFICIAL_LORAS = [
    "krea2_darkbrush",
    "krea2_dotmatrix",
    "krea2_kidsdrawing",
    "krea2_neondrip",
    "krea2_rainywindow",
    "krea2_retroanime",
    "krea2_softwatercolor",
    "krea2_sunsetblur",
    "krea2_vintagetarot",
]


def download(filename: str, url: str = None, dest: Path = None) -> bool:
    if url is None:
        url = f"{BASE}/{filename}"
    if dest is None:
        dest = TARGET / filename
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED ({e})")
        return False


def download_loras() -> list[str]:
    LORAS_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    print("\nDownloading official LoRA weights (this may take a while)...")
    for name in OFFICIAL_LORAS:
        dest = LORAS_DIR / f"{name}.safetensors"
        if dest.exists():
            print(f"  {name}.safetensors already present, skipping.")
            continue
        url = f"{HF_BASE}/loras/{name}.safetensors"
        if not download(name + ".safetensors", url=url, dest=dest):
            failed.append(name)
    return failed


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    for f in ALWAYS_DOWNLOAD:
        if not download(f):
            failed.append(f)

    for f in REQUIRED_LOCAL:
        dest = TARGET / f
        if dest.exists():
            print(f"  {f} present (project version), keeping.")
        else:
            print(f"  ERROR: {f} missing. It ships with this project and must not "
                  f"be fetched from upstream (incompatible). Restore it from the repo.")
            failed.append(f)

    if failed:
        print(f"\nERROR: Failed to download: {', '.join(failed)}")
        print("Check your internet connection and retry.")
        return 1

    # Ensure __init__.py exists
    init = TARGET / "__init__.py"
    if not init.exists():
        init.touch()

    print("\nkrea2/ source files ready.")

    lora_failed = download_loras()
    if lora_failed:
        print(f"\nWARNING: Failed to download LoRAs: {', '.join(lora_failed)}")
        print("LoRAs are optional — run again or download manually via the UI.")

    # Download DiT checkpoint (fp8, ~12GB) — skip if already present
    print("\nChecking DiT checkpoint...")
    dit_dir = Path(__file__).resolve().parent.parent / "models" / "krea2" / "diffusion_models"
    dit_dir.mkdir(parents=True, exist_ok=True)
    dit_fp8 = dit_dir / "krea2_turbo_fp8_scaled.safetensors"
    if dit_fp8.exists():
        print(f"  {dit_fp8.name} already present, skipping.")
    else:
        print(f"  Downloading krea2_turbo_fp8_scaled.safetensors (~12 GB) ...")
        dit_url = f"{HF_BASE}/diffusion_models/krea2_turbo_fp8_scaled.safetensors"
        if not download("krea2_turbo_fp8_scaled.safetensors", url=dit_url, dest=dit_fp8):
            print("  WARNING: DiT download failed. Paste the checkpoint path manually in the System tab.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
