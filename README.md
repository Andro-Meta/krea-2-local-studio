# Krea 2 Local Studio

Local, web-based Krea 2 Studio for Windows, built on **ComfyUI as the generation engine**. The FastAPI backend composes ComfyUI graphs for every image operation; the React frontend provides a friendly, login-gated studio UI for text-to-image, redraw, img2img, inpaint/outpaint, character edit, depth control, upscaling, moodboards, LoRAs, gallery review, and optional Tailscale sharing.

This is an unofficial local studio. It is not affiliated with Krea AI.

## Architecture

**ComfyUI is the backend.** One `run.bat` brings up a hidden ComfyUI engine (port 8188) and the Studio server together. Every diffusion feature — generation in all modes, ControlNet, upscalers, seed variance, the Qwen3-VL helper LLM — executes as a ComfyUI graph built by `backend/comfy_workflows.py`. This keeps the project on the fastest, best-supported runtime with the largest node/developer ecosystem, so new community workflows can be adopted quickly.

The Studio layer adds what raw ComfyUI does not have: a task-oriented UI, a generation queue, moodboards, per-user galleries and auth, child safety moderation, metadata-reproducible outputs, and public sharing.

The legacy in-process PyTorch pipeline has been removed. Non-diffusion helpers that remain in-process are intentionally lightweight: CLIPSeg automask, CivitAI browsing, gallery/auth/moderation, and a RealESRGAN fallback used only if ComfyUI is unreachable.

## Current Capabilities

- **Text to image:** Krea 2 Turbo (INT8 ConvRot default, fp8/bf16/GGUF selectable), Krea 2 RAW with auto-4K SeedVR2 post-pass.
- **Image workflows:** redraw (incl. NK2E preset), img2img, inpaint (incl. LanPaint), outpaint with seam harmonize, character edit (Krea2Edit identity nodes), depth ControlNet, style transfer, in-context vision edit, turbo 4X template.
- **Quality modes:** God Mode, Mr. Flow SR pipeline, RBG smart seed variance, CFG-Zero*, regional prompts, expression steering.
- **Helper AI:** Magic Wand prompt expansion, prompt planner, and image describe via Comfy QwenVL (abliterated Qwen3-VL), with Transformers/GGUF-server/OpenRouter alternatives.
- **Moodboards:** the full public Krea catalog (3,549 boards) with subject-safe v2 style guidance and unique titles, the Andro.Meta curated set, custom boards, Qwen-synthesized mashups, favorites, and a portable seed. Also published as a standalone ComfyUI node pack: [ComfyUI-Krea-Moodboards](https://github.com/Andro-Meta/ComfyUI-Krea-Moodboards).
- **Upscaling:** SeedVR2 (3B/7B), Ultimate tiled refine, 2-pass refine, tiled VAE, ESRGAN/Remacri — all as Comfy graphs.
- **Batch:** safe queued batch by default, true parallel batch when it fits, INT8 variant sweep.
- **Sharing:** login-gated Tailscale Funnel at `/krea` with self-healing startup, admin/user/child roles, private per-user galleries.
- **Child safety:** prompt moderation before generation, local NSFW image classification after, admin-visible audit events.

## What This Repo Contains

- FastAPI backend that orchestrates ComfyUI graphs plus queues, gallery, sharing, moderation, and admin APIs.
- React/MUI frontend for Create, Redraw Studio, Character Edit, Test Labs, Gallery, Moodboards, and System controls.
- Windows install/run scripts that manage the bundled ComfyUI engine.
- Portable Krea moodboard seed data in `data/krea_moodboards_seed.json`.

This repo does **not** include generated images, user credentials, local passwords, local logs, `.env`, model caches, the ComfyUI checkout, or dependency directories.

## Setup

Install Python 3.12+ and Node.js 18+, then run:

```bat
install.bat
```

`install.bat` creates the Python venv, installs dependencies, sets up the ComfyUI engine with the required custom nodes (Krea2 nodes, QwenVL, RES4LYF, LanPaint, Ultimate SD Upscale, SeedVR2, RBG seed variance, Krea2Edit), downloads the default assets (Turbo INT8 ConvRot checkpoint, abliterated Qwen3-VL text encoder, Qwen VAE), and builds the frontend.

Then start the login-gated sharing app (ComfyUI comes up automatically, hidden):

```bat
run.bat
```

For local-only LAN mode without share auth:

```bat
run.bat local
```

Closing the terminal (X button or Ctrl+C) shuts down both the Studio server and the ComfyUI engine; a detached janitor guarantees VRAM/RAM is freed.

## Models and Assets

The default install prepares, under the ComfyUI model folders:

- Krea 2 Turbo INT8 ConvRot at `ComfyUI/models/diffusion_models/`
- abliterated Qwen3-VL fp8 text encoder at `ComfyUI/models/text_encoders/`
- Qwen-Image VAE at `ComfyUI/models/vae/`
- SR models (RealESRGAN x2, Remacri x4) at `ComfyUI/models/upscale_models/`
- official Krea LoRAs at `models/loras/`
- helper Qwen3-VL for the QwenVL nodes

Krea 2 RAW and the SeedVR2 DiTs are optional downloads from the System tab (RAW is large and not required for the Turbo workflow). Engine/quantization variants (fp8, bf16, GGUF) are selectable per generation; they choose the matching ComfyUI UNET loader.

## Moodboards

There are three moodboard layers:

1. **Andro.Meta curated moods** from `backend/moods.py` — fast local taste profiles.
2. **Krea catalog moodboards** — the complete public catalog (3,549 boards), each with structured, subject-safe style guidance (palette / lighting / medium / composition / atmosphere) generated to transfer *style only*, never subjects. Boards sharing an official title carry unique style qualifiers (e.g. "Cinematic Chiaroscuro Noir — Candle Smoke Indigo").
3. **Custom and mashup moodboards** — local reference images and Qwen-synthesized blends.

The portable seed is `data/krea_moodboards_seed.json`; duplicate-title qualifiers persist via `data/moodboard_title_qualifiers.json` across catalog re-syncs. To enrich newly imported boards:

```bat
venv\Scripts\python.exe scripts\enrich_krea_moodboard_seed.py --upgrade --limit 100 --export-seed
```

## Public Sharing and Users

`run.bat` starts share mode under the `/krea` path so other Tailscale Funnel routes on the machine keep their own paths. Startup rebinds the Funnel to the session port, probes the public URL, and automatically re-registers the Tailscale connection if the ingress went stale — without touching other apps' serve entries.

Roles:

- **admin:** settings, users, all galleries, moderation review, Tailscale sharing controls.
- **user:** generation and a private gallery.
- **child:** generation with safety moderation and a private gallery.

## Child Safety

- prompt moderation runs before generation;
- generated images are checked after generation;
- unsafe child outputs are not shown to the child and fail closed if the classifier is missing;
- blocked attempts are visible to admins in System > Child Safety Review.

## Performance

The recommended starting point for a 24 GB GPU is Krea 2 Turbo INT8 ConvRot at 1K with the safe queued batch mode. The ComfyUI engine starts with `--disable-pinned-memory`, a VRAM reserve, and SageAttention when available; tune via `KREA_COMFY_ARGS`, `KREA_COMFY_HIGHVRAM`, `KREA_COMFY_RESERVE_VRAM`, `KREA_COMFY_SAGE` in `.env`.

For measured efficiency notes see `docs/performance.md`.

## Credits and Acknowledgements

Core upstream work:

- **ComfyUI** — the generation engine this Studio is built on, and its node ecosystem.
- **Krea AI** for releasing Krea 2 open-source components and prompting guidance.
- **Qwen** for `Qwen3-VL` and `Qwen-Image` assets used for conditioning and helper AI.
- **PyTorch**, **FastAPI**, **React**, **Vite**, and **MUI** for the core app stack.
- **Tailscale** for private/public sharing infrastructure.

Node packs and techniques this Studio depends on or adapted:

- **Comfy-Org Krea 2 nodes and workflows**, plus community Krea 2 workflow authors.
- **KiJai** for major ComfyUI ecosystem contributions and practical Krea/Qwen findings.
- **RES4LYF** by ClownsharkBatwing and contributors (`res_2s`, `bong_tangent`, ClownsharKSampler).
- **LanPaint** by scraed for the inpainting method.
- **Ultimate SD Upscale** (Coyote-A lineage) for the tiled refine approach.
- **SeedVR2** for the restoration-grade upscaler.
- **RBG Smart Seed Variance** for the seed variance node.
- **ComfyUI-QwenVL** (1038lab) for the helper LLM nodes.
- **Krea2Edit** (lbouaraba) for identity-preserving character edit nodes.
- **CFG-Zero*** paper/authors for the flow-matching guidance improvement.
- **NudeNet**, **Falconsai NSFW image detection**, and related open safety tooling.

Community thanks:

- The **Banodoco community** for persistent Krea 2 experimentation, sampler/scheduler comparisons, low-VRAM findings, and visual A/B testing that informed many defaults in this Studio.
