from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


RELEASE_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"


def _latest_release_assets() -> dict[str, str]:
    request = urllib.request.Request(RELEASE_API, headers={"User-Agent": "Krea2Studio/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed GitHub API URL
        payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    return {
        str(asset["name"]): str(asset["browser_download_url"])
        for asset in payload.get("assets", [])
        if asset.get("name") and asset.get("browser_download_url")
    }


def _download(url: str, dest: Path) -> Path:
    if not url.startswith("https://github.com/leejet/stable-diffusion.cpp/releases/download/"):
        raise ValueError("Refusing to download unexpected stable-diffusion.cpp asset URL.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Krea2Studio/1.0"})
    with urllib.request.urlopen(request, timeout=1800) as response:  # noqa: S310 - URL validated above
        with dest.open("wb") as out:
            shutil.copyfileobj(response, out)
    return dest


def _extract_flat(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            if not name:
                continue
            with zf.open(member) as source, (dest / name).open("wb") as target:
                shutil.copyfileobj(source, target)


def install_stable_diffusion_cpp(dest: Path) -> dict[str, str]:
    dest = Path(dest)
    sd_cli = dest / "sd-cli.exe"
    if sd_cli.exists():
        return {"sd_cli_path": str(sd_cli), "skipped": "true"}

    assets = _latest_release_assets()
    sd_name = next((name for name in assets if name.endswith("-bin-win-cuda12-x64.zip") and name.startswith("sd-master-")), "")
    cudart_name = "cudart-sd-bin-win-cu12-x64.zip"
    if not sd_name or cudart_name not in assets:
        raise RuntimeError("Could not find stable-diffusion.cpp Windows CUDA12 release assets.")

    downloads = dest / "_downloads"
    sd_zip = _download(assets[sd_name], downloads / sd_name)
    cudart_zip = _download(assets[cudart_name], downloads / cudart_name)
    _extract_flat(sd_zip, dest)
    _extract_flat(cudart_zip, dest)
    if not sd_cli.exists():
        raise RuntimeError("stable-diffusion.cpp download did not contain sd-cli.exe.")
    return {"sd_cli_path": str(sd_cli), "skipped": "false"}
