# Krea 2 Local Studio

Local, web-based Krea 2 Studio for Windows with image generation, moodboards, img2img, inpaint, outpaint, gallery review, local helper AI, and optional Tailscale public sharing.

## What This Repo Contains

- FastAPI backend for Krea 2 generation and local admin APIs.
- React/MUI frontend for generation, gallery, system status, users, and sharing controls.
- Install and run scripts for Windows.
- Local helper AI integration using `Qwen/Qwen3-VL-4B-Instruct`.

This repo does not include model weights, generated images, credentials, local user passwords, logs, or machine-specific `.env` files.

## Model Downloads

`install.bat` installs dependencies and downloads the standard local assets:

- `Qwen/Qwen3-VL-4B-Instruct` into `models/local_ai/qwen3_vl_4b_instruct`
- `Qwen/Qwen-Image` VAE into `models/local_ai/qwen_image`
- Krea 2 source helper file `mmdit.py`
- Official Krea LoRAs into `models/loras`
- Krea 2 Turbo fp8 checkpoint into `models/krea2/diffusion_models/krea2_turbo_fp8_scaled.safetensors`

Krea 2 RAW is not downloaded by default because it is large and not needed for the default Turbo workflow. Download RAW separately if you want it, then set `KREA2_RAW_PATH` in `.env` or use the System UI to load it.

The local helper AI and moodboard reference-image conditioning use the same local Qwen3-VL assets. If those assets are missing, open System > Krea Moodboard Conditioning / Local AI Assets or run:

```bat
venv\Scripts\python.exe scripts\download_support_models.py
```

## Setup

1. Install Python 3.12+ and Node.js 18+. Python 3.12 is the recommended public setup target.
2. Run:

```bat
install.bat
```

3. Edit `.env` if needed. Leave model paths blank to use auto-detected files under `models/`.
4. Start the login-gated web app:

```bat
run.bat
```

For local-only mode:

```bat
run.bat local
```

## Public Sharing

`run.bat` starts the Krea web server in share mode at `/krea`. Admins can manage users, passwords, roles, Tailscale status, and the `/krea` Funnel route from the System tab.

The app always uses the `/krea` path for Tailscale Funnel so other local tools can keep their own root Funnel route.

## Performance

The default install uses PyTorch CUDA wheels and verifies that CUDA is available before setup completes. The fp8 Turbo workflow is the recommended starting point for 24 GB GPUs.

For measured efficiency notes, including what helped and what was tested but not adopted, see [`docs/performance.md`](docs/performance.md).

## Secrets And Local Files

Do not commit:

- `.env`
- `share_auth.json`
- `models/`
- `outputs/`
- `logs/`
- `app.db`
- `venv/`
- `frontend/node_modules/`
- `frontend/dist/`

These are ignored by `.gitignore`.
