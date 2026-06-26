"""Krea 2 Studio FastAPI server — port 8200."""
from __future__ import annotations

import asyncio
import base64
import gc
import io
import json
import logging
import os
import secrets
import sys
import time
import uuid
from pathlib import Path

# No triton on Windows → disable dynamo before torch loads it (mmdit posemb uses
# @torch.compile(fullgraph=True), which would hard-fail at first forward).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Ensure backend dir is on path
_BACKEND = Path(__file__).parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from gallery import delete_image, get_gallery, init_db, save_image, set_favorite
from inference import pipeline
from log_setup import setup_logging
from lora_manager import inspect_lora, list_loras
from prompt_expander import describe_image_local, describe_image_openrouter, expand_prompt_result, openrouter_error_hint
from schemas import (
    AutoMaskRequest,
    DescribeImageRequest,
    DescribeImageResponse,
    ExpandPromptRequest,
    ExpandPromptResponse,
    FavoriteRequest,
    GalleryListResponse,
    GenerationRequest,
    LoadModelRequest,
    SettingsUpdate,
    ShareLoginRequest,
    ShareUserCreateRequest,
    ShareUserPasswordRequest,
    ShareUserRoleRequest,
    SystemInfoResponse,
    LoraImportRequest,
    UpscaleRequest,
)
from settings import BASE_DIR, DIST_DIR, LOGS_DIR, LORAS_DIR, MODELS_DIR, OUTPUTS_DIR, settings
from share_auth import (
    add_user,
    get_user_role,
    is_admin,
    is_valid_username,
    list_user_records,
    remove_user,
    set_user_role,
    verify_user,
)
from support_models import download_support_models, support_model_status
from sharing_service import PUBLIC_PATH as SHARING_PUBLIC_PATH, funnel_status, start_funnel, stop_funnel, tailscale_status, tailscale_up
from security_utils import append_query_param, is_civitai_url, normalize_lora_import_url, safe_lora_filename
from system_check import get_system_report

logger = logging.getLogger(__name__)
setup_logging(LOGS_DIR)

app = FastAPI(title="Krea 2 Studio", version="1.0.0")
app.mount("/api/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Optional public sharing auth
# ---------------------------------------------------------------------------

SHARE_AUTH_ENABLED = os.environ.get("KREA_SHARE_AUTH", "").lower() in {"1", "true", "yes"}
PUBLIC_BASE_PATH = "/" + os.environ.get("KREA_PUBLIC_BASE_PATH", "/").strip("/")
if PUBLIC_BASE_PATH == "/.":
    PUBLIC_BASE_PATH = "/"
SHARE_AUTH_FILE = Path(os.environ.get("KREA_SHARE_AUTH_FILE", str(BASE_DIR / "share_auth.json")))
SHARE_COOKIE = "krea_share_session"
SHARE_COOKIE_SECURE = os.environ.get("KREA_SHARE_COOKIE_SECURE", "0").lower() in {"1", "true", "yes"}
SHARE_SESSION_TTL_SECONDS = 12 * 60 * 60
_share_sessions: dict[str, tuple[str, float]] = {}


def _strip_public_base_path(scope: dict) -> None:
    if PUBLIC_BASE_PATH == "/":
        return
    path = scope.get("path", "")
    if path == PUBLIC_BASE_PATH:
        scope["path"] = "/"
        scope["root_path"] = PUBLIC_BASE_PATH
    elif path.startswith(PUBLIC_BASE_PATH + "/"):
        scope["path"] = path[len(PUBLIC_BASE_PATH):] or "/"
        scope["root_path"] = PUBLIC_BASE_PATH


def _is_auth_exempt(path: str, method: str = "GET") -> bool:
    if method == "OPTIONS":
        return True
    if path in {"/login", "/api/auth/login", "/api/auth/logout", "/api/auth/me"}:
        return True
    if path.startswith("/assets/"):
        return True
    return False


def _auth_username_from_cookie(cookie: str | None) -> str | None:
    if not cookie:
        return None
    record = _share_sessions.get(cookie)
    if record is None:
        return None
    username, expires_at = record
    if expires_at < time.time():
        _share_sessions.pop(cookie, None)
        return None
    if not is_valid_username(username) or get_user_role(SHARE_AUTH_FILE, username) is None:
        _share_sessions.pop(cookie, None)
        return None
    return username


def _requires_admin(path: str, method: str) -> bool:
    if path.startswith("/api/admin/") or path.startswith("/api/sharing/"):
        return True
    if path in {"/api/settings", "/api/load-model", "/api/unload-model", "/api/support-models/download"}:
        return True
    if path.startswith("/api/loras/") and method != "GET":
        return True
    if path == "/api/loras/import":
        return True
    if path.startswith("/api/gallery/") and method == "DELETE":
        return True
    return False


@app.middleware("http")
async def share_auth_middleware(request: Request, call_next):
    _strip_public_base_path(request.scope)
    if not SHARE_AUTH_ENABLED:
        return await call_next(request)

    path = request.scope.get("path", "")
    if _is_auth_exempt(path, request.method):
        return await call_next(request)

    user = _auth_username_from_cookie(request.cookies.get(SHARE_COOKIE))
    if user:
        request.state.share_user = user
        if _requires_admin(path, request.method) and not is_admin(SHARE_AUTH_FILE, user):
            return JSONResponse({"detail": "Admin access required"}, status_code=403)
        return await call_next(request)

    if path.startswith("/api/"):
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
    return RedirectResponse(url=f"{PUBLIC_BASE_PATH.rstrip('/')}/login" if PUBLIC_BASE_PATH != "/" else "/login")


@app.get("/login")
async def share_login_page():
    if not SHARE_AUTH_ENABLED:
        return RedirectResponse(url="/")
    return HTMLResponse(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Krea 2 Studio Login</title>
  <style>
    body{margin:0;background:#131218;color:#e6e1e5;font:16px/1.5 Roboto,system-ui,sans-serif;display:grid;place-items:center;min-height:100dvh}
    form{width:min(360px,calc(100vw - 32px));background:#211f2d;border:1px solid rgba(202,196,208,.18);border-radius:20px;padding:24px}
    h1{font-size:22px;margin:0 0 16px;font-weight:500}
    label{display:block;font-size:13px;color:#cac4d0;margin:14px 0 6px}
    input{width:100%;box-sizing:border-box;border-radius:12px;border:1px solid rgba(202,196,208,.28);background:#131218;color:#e6e1e5;padding:12px;font:inherit}
    button{margin-top:18px;width:100%;border:0;border-radius:999px;background:#d0bcff;color:#381e72;padding:12px 18px;font:inherit;font-weight:600;cursor:pointer}
    .err{min-height:20px;color:#f2b8b5;font-size:13px;margin-top:12px}
  </style>
</head>
<body>
  <form id="login">
    <h1>Krea 2 Studio</h1>
    <label for="username">Username</label>
    <input id="username" name="username" autocomplete="username" required>
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required>
    <button type="submit">Sign in</button>
    <div class="err" id="err"></div>
  </form>
  <script>
    document.getElementById('login').addEventListener('submit', async (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      const res = await fetch('./api/auth/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({username: form.username.value, password: form.password.value})
      });
      if (res.ok) window.location.href = './';
      else document.getElementById('err').textContent = 'Invalid username or password.';
    });
  </script>
</body>
</html>"""
    )


@app.post("/api/auth/login")
async def share_login(req: ShareLoginRequest, response: Response):
    if not SHARE_AUTH_ENABLED:
        return {"ok": True, "share_auth": False}
    username = req.username.strip()
    if not is_valid_username(username) or not verify_user(SHARE_AUTH_FILE, username, req.password):
        raise HTTPException(401, "Invalid username or password")
    token = secrets.token_urlsafe(32)
    _share_sessions[token] = (username, time.time() + SHARE_SESSION_TTL_SECONDS)
    response.set_cookie(
        SHARE_COOKIE,
        token,
        httponly=True,
        secure=SHARE_COOKIE_SECURE,
        samesite="lax",
        max_age=SHARE_SESSION_TTL_SECONDS,
        path=PUBLIC_BASE_PATH if PUBLIC_BASE_PATH != "/" else "/",
    )
    return {"ok": True, "username": username, "role": get_user_role(SHARE_AUTH_FILE, username)}


@app.post("/api/auth/logout")
async def share_logout(response: Response):
    # Delete the current browser cookie. The in-memory server session expires on its own.
    response.delete_cookie(SHARE_COOKIE, path=PUBLIC_BASE_PATH if PUBLIC_BASE_PATH != "/" else "/")
    return {"ok": True}


@app.get("/api/auth/me")
async def share_me(request: Request):
    if not SHARE_AUTH_ENABLED:
        return {"authenticated": True, "share_auth": False, "role": "admin"}
    user = _auth_username_from_cookie(request.cookies.get(SHARE_COOKIE))
    return {"authenticated": bool(user), "username": user, "role": get_user_role(SHARE_AUTH_FILE, user) if user else None}


@app.get("/api/admin/users")
async def admin_list_users():
    return {"users": list_user_records(SHARE_AUTH_FILE)}


@app.post("/api/admin/users")
async def admin_add_user(req: ShareUserCreateRequest):
    try:
        add_user(SHARE_AUTH_FILE, req.username, req.password, role=req.role)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "users": list_user_records(SHARE_AUTH_FILE)}


@app.put("/api/admin/users/{username}/role")
async def admin_set_user_role(username: str, req: ShareUserRoleRequest):
    if not set_user_role(SHARE_AUTH_FILE, username, req.role):
        raise HTTPException(404, "User not found")
    return {"ok": True, "users": list_user_records(SHARE_AUTH_FILE)}


@app.put("/api/admin/users/{username}/password")
async def admin_set_user_password(username: str, req: ShareUserPasswordRequest):
    if username not in {u["username"] for u in list_user_records(SHARE_AUTH_FILE)}:
        raise HTTPException(404, "User not found")
    role = get_user_role(SHARE_AUTH_FILE, username) or "user"
    try:
        add_user(SHARE_AUTH_FILE, username, req.password, role=role)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True}


@app.delete("/api/admin/users/{username}")
async def admin_remove_user(username: str):
    if not remove_user(SHARE_AUTH_FILE, username):
        raise HTTPException(404, "User not found")
    return {"ok": True, "users": list_user_records(SHARE_AUTH_FILE)}


@app.get("/api/sharing/status")
async def sharing_status():
    return {"tailscale": tailscale_status(), "funnel": funnel_status(), "public_path": SHARING_PUBLIC_PATH}


@app.post("/api/sharing/tailscale-up")
async def sharing_tailscale_up():
    return tailscale_up()


@app.post("/api/sharing/funnel/start")
async def sharing_funnel_start():
    result = start_funnel()
    if not result.get("ok"):
        raise HTTPException(502, result.get("message", "Tailscale Funnel failed to start."))
    return result


@app.post("/api/sharing/funnel/stop")
async def sharing_funnel_stop():
    return stop_funnel()

# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_JOBS_MAX = 200  # ponytail: simple FIFO cap; raise if clients poll very old jobs


def _new_job() -> str:
    jid = uuid.uuid4().hex
    _jobs[jid] = {"status": "queued", "progress": 0, "images": [], "error": None, "seed": None}
    # Evict oldest finished jobs to bound memory (dicts keep insertion order).
    while len(_jobs) > _JOBS_MAX:
        oldest = next(iter(_jobs))
        if oldest == jid:
            break
        del _jobs[oldest]
    return jid


# ---------------------------------------------------------------------------
# WebSocket manager
# ---------------------------------------------------------------------------

class WSManager:
    def __init__(self):
        self._sockets: dict[str, list[WebSocket]] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        self._sockets.setdefault(job_id, []).append(ws)

    def disconnect(self, job_id: str, ws: WebSocket):
        socks = self._sockets.get(job_id)
        if socks and ws in socks:
            socks.remove(ws)

    async def broadcast(self, job_id: str, data: dict):
        dead = []
        for ws in self._sockets.get(job_id, []):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(job_id, ws)


ws_manager = WSManager()

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    await init_db()
    logger.info("Krea 2 Studio ready on port 8200")
    # Auto-load model if configured
    cp = settings.krea2_auto_checkpoint or settings.krea2_turbo_path
    if cp and Path(cp).exists():
        asyncio.create_task(_auto_load_model(cp, settings.krea2_auto_quant))


async def _auto_load_model(checkpoint_path: str, quantization: str):
    loop = asyncio.get_event_loop()
    logger.info(f"Auto-loading {checkpoint_path} [{quantization}]...")
    try:
        await loop.run_in_executor(
            None, lambda: pipeline.load(checkpoint_path, quantization)
        )
        logger.info("Auto-load complete.")
    except Exception as e:
        logger.warning(f"Auto-load failed: {e}")


# ---------------------------------------------------------------------------
# Generation endpoints
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def generate(req: GenerationRequest):
    # Optional prompt expansion
    if req.use_prompt_expander:
        result = expand_prompt_result(
            req.prompt,
            backend=settings.prompt_expander_backend,
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_model=settings.openrouter_model,
            openrouter_free_only=settings.openrouter_free_only,
            ideogram_api_key=settings.ideogram_api_key,
        )
        if result.error:
            raise HTTPException(502, result.error)
        req.prompt = result.expanded

    job_id = _new_job()
    asyncio.create_task(_run_generation(job_id, req))
    return {"job_id": job_id, "status": "queued"}


async def _run_generation(job_id: str, req: GenerationRequest):
    job = _jobs[job_id]
    job["status"] = "running"
    await ws_manager.broadcast(job_id, {"type": "status", "status": "running"})

    loop = asyncio.get_event_loop()

    def progress_cb(step: int, total: int):
        job["progress"] = int(step / max(total, 1) * 100)
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast(
                job_id, {"type": "progress", "step": step, "total": total, "pct": job["progress"]}
            ),
            loop,
        )

    try:
        results, seed, filenames, lora_reports = await loop.run_in_executor(
            None, lambda: pipeline.generate(req, progress_cb=progress_cb)
        )
        job["images"] = results
        job["seed"] = seed
        job["status"] = "done"
        job["progress"] = 100
        # Surface LoRAs that were requested but not applied (wrong model/format).
        lora_warnings = [r for r in (lora_reports or []) if not r.get("applied")]
        job["lora_warnings"] = lora_warnings

        # Save gallery DB entries (files already written by inference.py).
        # The sampler uses seed+i per image, so record the matching per-image seed.
        for i, fname in enumerate(filenames):
            try:
                await save_image(
                    filename=fname,
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    checkpoint=req.checkpoint,
                    steps=req.steps,
                    cfg=req.cfg,
                    width=req.width,
                    height=req.height,
                    seed=seed + i,
                    loras=[l.get("name", "") for l in req.loras],
                    mode=req.mode,
                )
            except Exception:
                logger.exception(f"Gallery save failed for {fname}")

        await ws_manager.broadcast(job_id, {
            "type": "done", "images": results, "seed": seed,
            "lora_warnings": lora_warnings,
        })

    except Exception as e:
        logger.exception(f"Generation failed for job {job_id}")
        job["status"] = "error"
        job["error"] = str(e)
        await ws_manager.broadcast(job_id, {"type": "error", "error": str(e)})


@app.get("/api/generate/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.websocket("/ws/{job_id}")
async def ws_endpoint(ws: WebSocket, job_id: str):
    _strip_public_base_path(ws.scope)
    if SHARE_AUTH_ENABLED and not _auth_username_from_cookie(ws.cookies.get(SHARE_COOKIE)):
        await ws.close(code=1008)
        return
    await ws_manager.connect(job_id, ws)
    job = _jobs.get(job_id)
    if job:
        await ws.send_json({"type": "init", **job})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, ws)


if PUBLIC_BASE_PATH != "/":
    app.add_api_websocket_route(f"{PUBLIC_BASE_PATH}/ws/{{job_id}}", ws_endpoint)


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@app.post("/api/load-model")
async def load_model(req: LoadModelRequest):
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None, lambda: pipeline.load(req.checkpoint_path, req.quantization)
        )
    except Exception as exc:
        logger.exception("Model load failed")
        raise HTTPException(500, "Model load failed. Check the server logs for details.")
    return {"status": "loaded", "checkpoint": req.checkpoint_path}


@app.post("/api/unload-model")
async def unload_model():
    pipeline.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "unloaded"}


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------

@app.get("/api/gallery")
async def gallery(page: int = 1, page_size: int = 50, favorites: bool = False):
    return await get_gallery(page, page_size, favorites)


@app.put("/api/gallery/{gallery_id}/favorite")
async def favorite(gallery_id: int, req: FavoriteRequest):
    await set_favorite(gallery_id, req.favorite)
    return {"ok": True}


@app.delete("/api/gallery/{gallery_id}")
async def delete_gallery_item(gallery_id: int):
    filename = await delete_image(gallery_id)
    if filename is None:
        raise HTTPException(404, "Not found")
    return {"ok": True, "filename": filename}


# ---------------------------------------------------------------------------
# LoRAs
# ---------------------------------------------------------------------------

@app.get("/api/moods")
async def get_moods():
    from moods import MOODS
    return MOODS


@app.get("/api/loras")
async def get_loras():
    return list_loras()


@app.post("/api/loras/{lora_name}/download")
async def download_lora(lora_name: str):
    from lora_manager import OFFICIAL_LORAS, OFFICIAL_LORA_HF_IDS
    from huggingface_hub import hf_hub_download
    if lora_name not in OFFICIAL_LORAS:
        raise HTTPException(404, f"Unknown LoRA: {lora_name}")
    repo_id = OFFICIAL_LORA_HF_IDS.get(lora_name, "Comfy-Org/Krea-2")
    filename = f"{lora_name}.safetensors"
    loop = asyncio.get_event_loop()
    try:
        dest = await loop.run_in_executor(
            None,
            lambda: hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(LORAS_DIR),
                token=settings.hf_token or None,
            ),
        )
        return {"ok": True, "path": dest}
    except Exception:
        logger.exception("Official LoRA download failed")
        raise HTTPException(500, "LoRA download failed. Check the server logs for details.")


@app.post("/api/loras/import")
async def import_lora_url(req: LoraImportRequest):
    """Download a LoRA from a HuggingFace or CivitAI URL."""
    import urllib.request

    try:
        url = normalize_lora_import_url(req.url)
        safe_lora_filename(req.filename, url)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    filename = f"imported_{uuid.uuid4().hex}.safetensors"
    dest = LORAS_DIR / filename

    headers = {"User-Agent": "krea2-studio/1.0"}
    if is_civitai_url(url):
        token = req.civitai_token or settings.civitai_token
        if token:
            url = append_query_param(url, "token", token)

    def _fetch():
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=300) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return str(dest)

    loop = asyncio.get_event_loop()
    try:
        path = await loop.run_in_executor(None, _fetch)
        v = inspect_lora(dest)
        return {"ok": True, "path": path, "filename": filename,
                "compatible": v["compatible"], "match_info": v["reason"]}
    except HTTPException:
        raise
    except Exception:
        if dest.exists():
            dest.unlink(missing_ok=True)
        logger.exception("Imported LoRA download failed from allowed host")
        raise HTTPException(502, "LoRA import failed. Check the URL and server logs.")


# ---------------------------------------------------------------------------
# Upscaling
# ---------------------------------------------------------------------------

@app.post("/api/upscale")
async def upscale(req: UpscaleRequest):
    from upscaler import (
        b64_to_pil, pil_to_b64, upscale_model_refine, upscale_realesrgan,
        upscale_tiled_vae, upscale_ultimate,
    )

    img = b64_to_pil(req.image_b64)
    loop = asyncio.get_event_loop()

    if req.method == "realesrgan":
        result = await loop.run_in_executor(
            None, lambda: upscale_realesrgan(img, MODELS_DIR, req.scale)
        )
    elif req.method == "tiled_vae":
        if not pipeline.is_loaded():
            raise HTTPException(400, "Model must be loaded for tiled VAE upscale")
        result = await loop.run_in_executor(
            None, lambda: upscale_tiled_vae(
                img, pipeline.ae,
                device=pipeline._device, dtype=pipeline._dtype
            )
        )
    elif req.method == "model_refine":
        result = await loop.run_in_executor(
            None, lambda: upscale_model_refine(img, pipeline, denoise=req.denoise)
        )
    elif req.method == "ultimate":
        if not pipeline.is_loaded():
            raise HTTPException(400, "Model must be loaded for Ultimate upscale")
        result = await loop.run_in_executor(
            None, lambda: upscale_ultimate(
                img, pipeline, MODELS_DIR, prompt=req.prompt, scale=req.scale,
                tile=req.tile_size, denoise=req.denoise, seam_fix=req.seam_fix,
            )
        )
    else:
        raise HTTPException(400, f"Unknown upscale method: {req.method}")

    return {"image_b64": pil_to_b64(result)}


@app.post("/api/automask")
async def automask(req: AutoMaskRequest):
    """Generate an inpaint mask from a text description (CLIPSeg, CPU)."""
    import base64 as _b64
    import io as _io
    from PIL import Image as _Image
    from automask import generate_mask

    try:
        img = _Image.open(_io.BytesIO(_b64.b64decode(req.image_b64)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Automask image decode failed: %s", exc)
        raise HTTPException(400, "Bad image data.")
    loop = asyncio.get_event_loop()
    mask = await loop.run_in_executor(
        None, lambda: generate_mask(img, req.prompt, req.threshold)
    )
    buf = _io.BytesIO()
    mask.save(buf, format="PNG")
    return {"mask_b64": _b64.b64encode(buf.getvalue()).decode()}


@app.post("/api/describe-image", response_model=DescribeImageResponse)
async def describe_image(req: DescribeImageRequest):
    loop = asyncio.get_event_loop()
    try:
        if settings.prompt_expander_backend == "openrouter":
            result = await loop.run_in_executor(
                None,
                lambda: describe_image_openrouter(req.image_b64, settings.openrouter_api_key),
            )
        else:
            result = await loop.run_in_executor(None, lambda: describe_image_local(req.image_b64))
    except Exception as exc:
        if settings.prompt_expander_backend == "openrouter":
            detail = openrouter_error_hint(exc)
        else:
            logger.exception("Local image description failed")
            detail = "Local image description failed. Use System > Krea Moodboard Conditioning / Local AI Assets to repair local models."
        raise HTTPException(502, detail)
    return DescribeImageResponse(**result)


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

@app.get("/api/system")
async def system_info():
    report = get_system_report()
    # Surface the auto-detected checkpoint so the UI can prefill the load form
    # (one-click recovery if auto-load failed, e.g. transient low RAM).
    auto_cp = settings.krea2_auto_checkpoint or settings.krea2_turbo_path or ""
    auto_quant = settings.krea2_auto_quant or ("fp8" if "fp8" in auto_cp.lower() else "bf16")
    report["model_status"] = {
        "loaded": pipeline.is_loaded(),
        "loading": getattr(pipeline, "_loading", False),
        "checkpoint": pipeline._loaded_checkpoint,
        "quantization": pipeline._loaded_quant,
        "auto_checkpoint": auto_cp,
        "auto_quant": auto_quant,
        "load_error": getattr(pipeline, "_last_load_error", None),
    }
    report["support_models"] = support_model_status()
    return report


@app.post("/api/support-models/download")
async def download_support_models_endpoint():
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, download_support_models)
    except Exception:
        logger.exception("Support model download failed")
        raise HTTPException(502, "Support model download failed. Check the server logs for details.")
    return {"ok": True, "items": results, "status": support_model_status()}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

ENV_PATH = Path(__file__).parent.parent / ".env"
SECRET_ENV_KEYS = {"HF_TOKEN", "CIVITAI_TOKEN", "IDEOGRAM_API_KEY", "OPENROUTER_API_KEY"}


def _read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _write_env(env: dict[str, str]) -> None:
    lines = [f"{k}={v}" for k, v in env.items() if k not in SECRET_ENV_KEYS]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.get("/api/settings")
async def get_settings():
    env = _read_env()
    return {
        "hf_token": "",
        "civitai_token": "",
        "krea2_turbo_path": env.get("KREA2_TURBO_PATH", ""),
        "krea2_raw_path": env.get("KREA2_RAW_PATH", ""),
        "output_dir": env.get("OUTPUT_DIR", str(MODELS_DIR.parent / "outputs")),
        "prompt_expander_backend": env.get("PROMPT_EXPANDER_BACKEND", settings.prompt_expander_backend),
        "openrouter_model": env.get("OPENROUTER_MODEL", settings.openrouter_model),
        "openrouter_free_only": env.get("OPENROUTER_FREE_ONLY", str(settings.openrouter_free_only)).lower() in {"1", "true", "yes"},
        "has_hf_token": bool(env.get("HF_TOKEN", settings.hf_token)),
        "has_civitai_token": bool(env.get("CIVITAI_TOKEN", settings.civitai_token)),
        "has_ideogram_api_key": bool(env.get("IDEOGRAM_API_KEY", settings.ideogram_api_key)),
        "has_openrouter_api_key": bool(env.get("OPENROUTER_API_KEY", settings.openrouter_api_key)),
    }


@app.put("/api/settings")
async def update_settings(req: SettingsUpdate):
    env = _read_env()
    if req.hf_token is not None:
        settings.hf_token = req.hf_token.replace("\r", "").replace("\n", "")
    if req.civitai_token is not None:
        settings.civitai_token = req.civitai_token.replace("\r", "").replace("\n", "")
    if req.krea2_turbo_path is not None:
        env["KREA2_TURBO_PATH"] = req.krea2_turbo_path
    if req.krea2_raw_path is not None:
        env["KREA2_RAW_PATH"] = req.krea2_raw_path
    if req.output_dir is not None:
        env["OUTPUT_DIR"] = req.output_dir
    if req.prompt_expander_backend is not None:
        env["PROMPT_EXPANDER_BACKEND"] = req.prompt_expander_backend
        settings.prompt_expander_backend = req.prompt_expander_backend
    if req.ideogram_api_key is not None:
        settings.ideogram_api_key = req.ideogram_api_key.replace("\r", "").replace("\n", "")
    if req.openrouter_api_key is not None:
        settings.openrouter_api_key = req.openrouter_api_key.replace("\r", "").replace("\n", "")
    if req.openrouter_model is not None:
        env["OPENROUTER_MODEL"] = req.openrouter_model
        settings.openrouter_model = req.openrouter_model
    if req.openrouter_free_only is not None:
        env["OPENROUTER_FREE_ONLY"] = "true" if req.openrouter_free_only else "false"
        settings.openrouter_free_only = req.openrouter_free_only
    _write_env(env)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Prompt expander
# ---------------------------------------------------------------------------

@app.post("/api/expand-prompt", response_model=ExpandPromptResponse)
async def expand_prompt_endpoint(req: ExpandPromptRequest):
    loop = asyncio.get_event_loop()
    backend = req.backend or settings.prompt_expander_backend
    result = await loop.run_in_executor(
        None,
        lambda: expand_prompt_result(
            req.prompt,
            backend=backend,
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_model=settings.openrouter_model,
            openrouter_free_only=settings.openrouter_free_only,
            ideogram_api_key=settings.ideogram_api_key,
        ),
    )
    return ExpandPromptResponse(
        expanded=result.expanded,
        changed=result.changed,
        error=result.error,
        backend=result.backend,
    )


# ---------------------------------------------------------------------------
# SPA static serving
# ---------------------------------------------------------------------------

if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        # Serve index.html for all non-API routes (SPA routing)
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            raise HTTPException(404)
        index = DIST_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(404, "Frontend not built. Run install.bat.")
else:
    @app.get("/")
    async def root():
        return JSONResponse(
            {"message": "Frontend not built. Run install.bat to build it."},
            status_code=200,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200, log_level="info")
