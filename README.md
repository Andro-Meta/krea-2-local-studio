# Krea 2 Local Studio

Local, web-based Krea 2 Studio for Windows. It runs Krea 2 Turbo locally, supports Krea 2 RAW when the machine has enough RAM/VRAM, and provides a browser UI for text-to-image, redraw, img2img, inpaint, outpaint, moodboards, LoRAs, gallery review, and optional Tailscale sharing.

This is an unofficial local studio. It is not affiliated with Krea AI.

## Current Capabilities

- **Text to image:** Krea 2 Turbo fp8 by default, RAW optionally.
- **Image workflows:** redraw, img2img, inpaint, outpaint, prompt-from-image, and metadata import from gallery images.
- **Samplers and schedulers:** native flow Euler, Euler ancestral, CFG++, ER-SDE, RES 2S, beta scheduler, SGM uniform, and RES4LYF-style bong tangent.
- **Prompt adherence tools:** local prompt planner, official Krea-style prompt expansion, regional prompts, CFG-Zero*, seed variance, expression steering, and Krea2T-style text-fusion enhancement.
- **Moodboards:** original local style-stack presets, Krea catalog moodboards, custom boards, Qwen-authored custom boards, Qwen-synthesized mashups, favorites, discovery notices, and a portable enriched seed.
- **Reference conditioning:** Qwen3-VL multimodal conditioning for moodboard and style reference images.
- **Low-VRAM runtime:** dynamic fp8 loading, block swap, encoder offload, tiled VAE decode, OOM recovery, GPU capability detection, and quality guard checks.
- **Upscaling:** RealESRGAN fallback, tiled VAE, Ultimate-style tiled refine, and a 2-pass low-denoise refine preset.
- **Sharing:** optional login-gated Tailscale Funnel at `/krea`, with admin/user/child roles and private per-user galleries.
- **Child safety:** child accounts get prompt moderation, image moderation via an optional local Transformers NSFW classifier, and admin-visible safety audit events.

## What This Repo Contains

- FastAPI backend for local generation, queues, gallery, sharing, moderation, and admin APIs.
- React/MUI frontend for Create, Redraw Studio, Realtime Studio, Gallery, Moodboards, and System controls.
- Windows install/run scripts.
- Local helper AI integration using `Qwen/Qwen3-VL-4B-Instruct`.
- Portable Krea moodboard seed data in `data/krea_moodboards_seed.json`.

This repo does **not** include generated images, user credentials, local passwords, local logs, `.env`, model cache folders, or the Python/Node dependency directories.

## Setup

Install Python 3.12+ and Node.js 18+, then run:

```bat
install.bat
```

`install.bat` creates the Python venv, installs PyTorch CUDA wheels, installs Python dependencies, downloads local helper assets, downloads the default Krea 2 Turbo fp8 checkpoint, and builds the frontend.

Then start the normal login-gated sharing app:

```bat
run.bat
```

For local-only LAN mode without share auth:

```bat
run.bat local
```

## Models and Assets

The default install prepares:

- `Qwen/Qwen3-VL-4B-Instruct` in `models/local_ai/qwen3_vl_4b_instruct`
- `Qwen/Qwen-Image` VAE in `models/local_ai/qwen_image`
- Krea 2 source helper files under `backend/krea2/`
- official Krea LoRAs in `models/loras`
- Krea 2 Turbo fp8 checkpoint at `models/krea2/diffusion_models/krea2_turbo_fp8_scaled.safetensors`

Krea 2 RAW is not downloaded by default because it is large and not required for the Turbo workflow. Download RAW separately, then set `KREA2_RAW_PATH` in `.env` or load it from the System tab.

If helper assets are missing, use System > Local AI Assets or run:

```bat
venv\Scripts\python.exe scripts\download_support_models.py
```

## Moodboards

There are three moodboard layers:

1. **Local style-stack presets** from `backend/moods.py`.
   - These are curated text taste profiles.
   - They add keywords to the prompt and avoids to the negative prompt.
   - They are fast, deterministic, and useful for quick T2I direction.

2. **Krea catalog moodboards** imported from public Krea moodboard pages.
   - The catalog stores official title, taste profile, tags, image URLs, and optional Qwen guidance.
   - Krea images are referenced by URL; the repo does not vendor thousands of images.

3. **Custom and mashup moodboards**.
   - Custom boards store local reference images.
   - Mashups are synthesized with local Qwen from multiple source boards.

Qwen moodboard enrichment stores structured guidance:

- `prompt_guidance`
- `negative_guidance`
- `style_axes`
- `conditioning_notes`
- `source_summary`

Generation uses these as transferable style guidance, not as fixed scene recreation. The enrichment prompt and sanitizer are designed to preserve a board's mood, lighting, palette, texture, and presentation while still honoring the user's requested subject count/content.

The portable seed is:

```text
data/krea_moodboards_seed.json
```

To enrich missing catalog guidance in batches:

```bat
venv\Scripts\python.exe scripts\enrich_krea_moodboard_seed.py --limit 100 --export-seed --export-every 10
```

Use small batches first. The script writes guidance into `app.db` immediately and exports the seed periodically.

## Public Sharing and Users

`run.bat` starts share mode under the `/krea` path. This lets other Tailscale Funnel routes keep their own root path.

Roles:

- **admin:** settings, model loading, users, all galleries, moderation review, Tailscale sharing controls.
- **user:** generation and private gallery.
- **child:** generation with child safety moderation and private gallery.

Admins can manage users and Tailscale sharing from the System tab.

The app includes:

- `/krea` Funnel start/stop controls
- `Repair /krea Sharing`
- local target checks
- login-gate checks
- public Funnel reachability checks

If Krea is reachable locally and on the tailnet, but the public `ts.net` URL fails before reaching Krea, that is a Tailscale service/Funnel issue. The GUI will tell the admin to restart the Tailscale Windows service as Administrator.

## Child Safety

Child accounts are moderated differently from normal users and admins:

- prompt moderation runs before generation;
- generated images are checked after generation;
- unsafe child outputs are not shown to the child;
- blocked attempts are visible to admins in System > Child Safety Review.

Child image moderation uses an optional local Transformers image classifier. It is installed from the GUI, not from `install.bat`.

If the image classifier is not installed, child image outputs fail closed after generation instead of being shown unreviewed.

## Realtime Studio

Realtime Studio is for composition and direction, not true 60 FPS live diffusion. Krea 2 Turbo still takes multiple inference steps.

The realtime preview path uses:

- conservative defaults;
- auto-preview off by default;
- a backend single-slot drop-frame buffer so quick canvas edits do not build stale FIFO lag;
- final render as the quality pass.

## Batch and Accelerators

Batch generation and optional attention accelerators are planned separately. The intended direction is:

- **Safe queued batch:** generate images one at a time through the FIFO queue. This should remain the default.
- **Parallel batch:** true batched sampling only when the runtime estimator says it fits.
- **Accelerators:** PyTorch SDPA remains the default. Triton-Windows and SageAttention should be optional, experimental, admin-installed, and visually A/B verified before use.

The project currently uses PyTorch CUDA wheels. The system CUDA toolkit is not normally used for inference unless building custom CUDA extensions.

## Performance

The recommended starting point for a 24 GB GPU is:

- Krea 2 Turbo fp8
- 1K resolution
- safe queued batch mode when generating multiple images

For measured efficiency notes and what was tested but not adopted, see:

```text
docs/performance.md
```

## Credits and Acknowledgements

This project builds on a lot of open work and community testing.

Core upstream work:

- **Krea AI** for releasing Krea 2 open-source components and prompting guidance.
- **Qwen** for `Qwen3-VL-4B-Instruct` and `Qwen-Image` assets used for local prompt/image/moodboard conditioning.
- **PyTorch**, **FastAPI**, **React**, **Vite**, and **MUI** for the core app stack.
- **Tailscale** for private/public sharing infrastructure.

Ported or adapted techniques:

- **ComfyUI** for KSampler semantics, scheduler references, memory-management patterns, tiled decode ideas, model loading patterns, and the broader Krea 2 workflow ecosystem.
- **ComfyUI Krea 2 community nodes** and workflow authors whose experiments helped validate Krea 2 settings.
- **KiJai** for major ComfyUI ecosystem contributions and practical Krea/Qwen workflow findings that informed this Studio, including low-VRAM/block-swap patterns seen in WanVideoWrapper-style workflows, Qwen image/VAE asset references, outpaint mask conventions, and Krea 2 sampling details discussed by the community.
- **LanPaint** by scraed for the inpainting method research direction.
- **ComfyUI-Krea2T-Enhancer** by `capitan01R` for the Krea2T prompt-adherence text-fusion enhancement approach.
- **RES4LYF** by ClownsharkBatwing and contributors for `res_2s` and `bong_tangent` sampler/scheduler ideas.
- **CFG-Zero*** paper/authors for the flow-matching guidance improvement.
- **NudeNet**, **Falconsai NSFW image detection**, and related open safety tooling for child-safety research; this project currently uses the Transformers-classifier direction.

Community thanks:

- The **Banodoco community** deserves special credit for persistent Krea 2 experimentation, Discord testing, sampler/scheduler comparisons, low-VRAM findings, visual A/B testing, and practical workflow reports. Many of the most useful defaults in this local studio were informed by that community's consistent hard work.

