"""Krea 2 Studio FastAPI server."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import secrets
import subprocess
import sys
import time
import uuid
from pathlib import Path

# No triton on Windows → disable dynamo before torch loads it (mmdit posemb uses
# @torch.compile(fullgraph=True), which would hard-fail at first forward).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Ensure backend dir is on path
_BACKEND = Path(__file__).parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from crash_reporter import clear_generation_breadcrumb, disable_fault_logging, enable_fault_logging, stale_generation_breadcrumbs, write_generation_breadcrumb
from gallery import delete_image, get_gallery, get_image_record_by_filename, init_db, save_image, set_favorite
from generation_queue import GenerationQueue
from inference import pipeline
from log_setup import setup_logging
from lora_manager import inspect_lora, list_loras
from moodboards_catalog import (
    CUSTOM_MOODBOARD_DIR,
    KREA_MOODBOARD_GALLERY_URL,
    MOODBOARD_SEED_PATH,
    create_mashup_moodboard,
    create_custom_moodboard,
    delete_custom_moodboard,
    export_moodboard_seed,
    fetch_moodboard_image_b64,
    generate_and_store_moodboard_qwen_guidance,
    generate_missing_moodboard_qwen_guidance,
    get_moodboard,
    fetch_cached_moodboard_image,
    import_moodboard_urls,
    init_moodboard_db,
    latest_moodboard_discovery,
    list_moodboards,
    set_moodboard_favorite,
    should_sync_moodboards,
)
from prompt_expander import describe_image_local, describe_image_openrouter, expand_prompt_result, openrouter_error_hint
from prompt_planner import plan_prompt
from prompt_recipes import delete_recipe, list_recipes, save_recipe
from schemas import (
    AutoMaskRequest,
    DescribeImageRequest,
    DescribeImageResponse,
    ExpandPromptRequest,
    ExpandPromptResponse,
    FavoriteRequest,
    GenerationRequest,
    LoadModelRequest,
    MemoryStopProcessRequest,
    CustomMoodboardRequest,
    MoodboardImportRequest,
    MoodboardImportResponse,
    MoodboardDiscoveryResponse,
    MoodboardExportResponse,
    MoodboardImageRequest,
    MoodboardImageResponse,
    MoodboardGuidanceMissingRequest,
    MoodboardMashupRequest,
    MoodboardListResponse,
    PlanPromptRequest,
    PlanPromptResponse,
    PromptRecipe,
    PromptRecipeListResponse,
    MoodboardItem,
    RealtimePreviewRequest,
    SettingsUpdate,
    ShareLoginRequest,
    ShareUserCreateRequest,
    ShareUserPasswordRequest,
    ShareUserRoleRequest,
    LoraImportRequest,
    UpscaleRequest,
    PreprocessorPreviewRequest,
)
from realtime_jobs import RealtimePreviewRegistry
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
from sharing_service import PUBLIC_PATH as SHARING_PUBLIC_PATH, funnel_status, repair_funnel, start_funnel, stop_funnel, tailscale_status, tailscale_up
from security_utils import append_query_param, is_civitai_url, normalize_lora_import_url, safe_lora_filename
from system_check import get_system_report
from memory_manager import (
    detect_krea_server_processes,
    prepare_for_generation,
    safe_clean_memory,
    stop_krea_server_process,
    unload_pipeline,
)
from moderation import (
    image_classifier_available,
    init_moderation_db,
    list_moderation_events,
    moderate_images,
    moderate_prompt,
    save_moderation_event,
)

logger = logging.getLogger(__name__)
setup_logging(LOGS_DIR)
SAFE_SERVED_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")

app = FastAPI(title="Krea 2 Studio", version="1.0.0")
QUARANTINE_DIR = BASE_DIR / "moderation_quarantine"
_outputs_static = StaticFiles(directory=str(OUTPUTS_DIR), check_dir=False)
_quarantine_static = StaticFiles(directory=str(QUARANTINE_DIR), check_dir=False)

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
    elif path.startswith(PUBLIC_BASE_PATH + "/"):
        scope["path"] = path[len(PUBLIC_BASE_PATH):] or "/"


def _is_auth_exempt(path: str, method: str = "GET") -> bool:
    if method == "OPTIONS":
        return True
    if path in {"/login", "/api/auth/login", "/api/auth/logout", "/api/auth/me"}:
        return True
    if method == "GET" and (
        path == "/api/moodboards"
        or path == "/api/moodboards/discoveries/latest"
        or (path.startswith("/api/moodboards/") and path.rsplit("/", 1)[-1].isdigit())
    ):
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


def _request_user_role(request: Request) -> tuple[str | None, str, bool]:
    if not SHARE_AUTH_ENABLED:
        return None, "admin", True
    username = getattr(request.state, "share_user", None)
    role = get_user_role(SHARE_AUTH_FILE, username) if username else None
    role = role or "user"
    return username, role, role == "admin"


def _requires_admin(path: str, method: str) -> bool:
    if path.startswith("/api/admin/") or path.startswith("/api/sharing/") or path.startswith("/api/moderation/"):
        return True
    if path.startswith("/api/accelerators/"):
        return True
    if path.startswith("/api/memory/"):
        return True
    if path in {"/api/settings", "/api/load-model", "/api/load-model/preflight", "/api/unload-model", "/api/support-models/download"}:
        return True
    if path.startswith("/api/loras/") and method != "GET":
        return True
    if path == "/api/loras/import":
        return True
    if path == "/api/moodboards/import":
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
    try:
        if not set_user_role(SHARE_AUTH_FILE, username, req.role):
            raise HTTPException(404, "User not found")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
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
    try:
        if not remove_user(SHARE_AUTH_FILE, username):
            raise HTTPException(404, "User not found")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
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


@app.post("/api/sharing/funnel/repair")
async def sharing_funnel_repair():
    return repair_funnel()


@app.post("/api/sharing/funnel/stop")
async def sharing_funnel_stop():
    return stop_funnel()

# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_JOBS_MAX = 200  # ponytail: simple FIFO cap; raise if clients poll very old jobs
generation_queue: GenerationQueue | None = None


def _new_job(username: str | None = None, role: str = "admin") -> str:
    jid = uuid.uuid4().hex
    _jobs[jid] = {
        "status": "queued",
        "progress": 0,
        "images": [],
        "error": None,
        "seed": None,
        "username": username,
        "role": role,
        "moderation_event_id": None,
    }
    # Evict oldest finished jobs to bound memory (dicts keep insertion order).
    while len(_jobs) > _JOBS_MAX:
        oldest = next(iter(_jobs))
        if oldest == jid:
            break
        del _jobs[oldest]
    return jid


def build_safe_batch_children(req: GenerationRequest) -> list[GenerationRequest]:
    count = max(1, int(getattr(req, "num_images", 1) or 1))
    base_seed = int(getattr(req, "seed", -1))
    if base_seed < 0:
        base_seed = secrets.randbelow(2**31 - 1)
    children: list[GenerationRequest] = []
    for index in range(count):
        child = req.model_copy(deep=True)
        child.num_images = 1
        child.seed = base_seed + index
        child.batch_mode = "safe_queue"
        child.parallel_batch_confirmed = False
        children.append(child)
    return children


def _refresh_parent_batch_job(parent_job_id: str) -> dict | None:
    parent = _jobs.get(parent_job_id)
    if not parent or not parent.get("child_job_ids"):
        return parent
    child_ids = list(parent.get("child_job_ids") or [])
    children = [_jobs.get(child_id) for child_id in child_ids]
    done_children = [child for child in children if child and child.get("status") == "done"]
    blocked = next((child for child in children if child and child.get("status") == "blocked"), None)
    errored = next((child for child in children if child and child.get("status") == "error"), None)

    parent["images"] = [child.get("images", [""])[0] for child in done_children if child.get("images")]
    parent["metadata"] = [child.get("metadata", [{}])[0] for child in done_children if child.get("metadata")]
    parent["progress"] = int(len(done_children) / max(len(child_ids), 1) * 100)
    if blocked:
        parent["status"] = "blocked"
        parent["error"] = blocked.get("error")
    elif errored:
        parent["status"] = "error"
        parent["error"] = errored.get("error")
    elif len(done_children) == len(child_ids):
        parent["status"] = "done"
        parent["progress"] = 100
    else:
        parent["status"] = "queued" if any(child and child.get("status") == "queued" for child in children) else "running"
    return parent


def _sync_queue_state_to_jobs() -> None:
    if generation_queue is None:
        return
    for job_id, state in generation_queue.all_statuses().items():
        job = _jobs.get(job_id)
        if not job or job.get("status") in {"done", "error", "blocked", "cancelled"}:
            continue
        job.update({
            "status": state.get("status", job.get("status")),
            "queue_position": state.get("queue_position"),
            "queue_length": state.get("queue_length"),
            "active_job_id": state.get("active_job_id"),
            "queued_at": state.get("queued_at"),
            "started_at": state.get("started_at"),
        })


async def _broadcast_queue_state() -> None:
    _sync_queue_state_to_jobs()
    for job_id, job in list(_jobs.items()):
        if job.get("status") == "queued":
            await ws_manager.broadcast(job_id, {"type": "queue", **job})


async def _queued_generation_handler(job_id: str, payload: dict) -> None:
    await _broadcast_queue_state()
    await _run_generation(
        job_id,
        payload["req"],
        username=payload.get("username"),
        role=payload.get("role", "user"),
    )
    await _broadcast_queue_state()


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
realtime_previews = RealtimePreviewRegistry(max_jobs=64)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global generation_queue
    fault_path = enable_fault_logging(LOGS_DIR)
    stale = stale_generation_breadcrumbs(LOGS_DIR)
    if stale:
        logger.warning("Found %d stale active generation breadcrumb(s) after previous shutdown/crash: %s", len(stale), stale)
    logger.info("Python fault handler logging to %s", fault_path)
    await init_db()
    await init_moderation_db()
    await init_moodboard_db()
    if generation_queue is None:
        generation_queue = GenerationQueue(_queued_generation_handler)
        asyncio.create_task(generation_queue.run())
    logger.info(
        "Krea 2 Studio ready (port=%s, public_base_path=%s)",
        os.environ.get("KREA_SERVER_PORT", "8200"),
        PUBLIC_BASE_PATH,
    )
    if await should_sync_moodboards(mark=True):
        asyncio.create_task(_sync_krea_moodboards())
    asyncio.create_task(_moodboard_sync_loop())
    asyncio.create_task(_moodboard_enrich_loop())
    # Auto-load model if configured
    cp = settings.krea2_auto_checkpoint or settings.krea2_turbo_path
    if cp and Path(cp).exists():
        asyncio.create_task(_auto_load_model(cp, settings.krea2_auto_quant, settings.krea2_blocks_to_swap))


@app.on_event("shutdown")
async def shutdown():
    disable_fault_logging()


async def _auto_load_model(checkpoint_path: str, quantization: str, blocks_to_swap: int = 0):
    loop = asyncio.get_event_loop()
    logger.info(f"Auto-loading {checkpoint_path} [{quantization}] (block_swap={blocks_to_swap})...")
    try:
        await loop.run_in_executor(
            None, lambda: pipeline.load(
                checkpoint_path, quantization,
                blocks_to_swap=int(blocks_to_swap or 0),
                fp8_fast_matmul=bool(getattr(settings, "krea2_fp8_fast_matmul", False)),
                torch_compile=bool(getattr(settings, "krea2_torch_compile", False)),
            )
        )
        logger.info("Auto-load complete.")
    except Exception as e:
        logger.warning(f"Auto-load failed: {e}")


async def _sync_krea_moodboards(max_pages: int = 200) -> None:
    try:
        await import_moodboard_urls([KREA_MOODBOARD_GALLERY_URL], max_pages=max_pages, use_browser_discovery=True)
    except Exception:
        logger.exception("Krea moodboard sync failed")


async def _moodboard_sync_loop() -> None:
    while True:
        await asyncio.sleep(60 * 60)
        if await should_sync_moodboards(mark=True):
            await _sync_krea_moodboards()


def _generation_busy() -> bool:
    """True if a generation job is queued/running or the model is loading.

    The Qwen guidance pass loads its own LLM and competes for VRAM, so the
    background enricher only runs while the studio is idle.
    """
    if generation_queue is not None and generation_queue.has_active_or_pending():
        return True
    return bool(getattr(pipeline, "_loading", False))


async def _moodboard_enrich_loop() -> None:
    """Precompute Qwen guidance for moodboards missing it, in small idle batches.

    Runs only when enabled and the studio is idle. Tiny batches + long sleeps
    keep it from contending with generation; stops once nothing is missing."""
    await asyncio.sleep(180)  # let startup / auto-load settle first
    while True:
        try:
            if getattr(settings, "krea2_moodboard_auto_enrich", True) and not _generation_busy():
                result = await generate_missing_moodboard_qwen_guidance(limit=4)
                if int(result.get("processed", 0)) == 0:
                    await asyncio.sleep(60 * 60)  # nothing left; idle check hourly
                    continue
        except Exception:
            logger.exception("Background moodboard enrichment batch failed")
        await asyncio.sleep(300)


# ---------------------------------------------------------------------------
# Generation endpoints
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def generate(req: GenerationRequest, request: Request):
    if getattr(req, "diffusion_engine", "native_pytorch") == "native_pytorch" and settings.diffusion_engine != "native_pytorch":
        fields = getattr(req, "model_fields_set", getattr(req, "__fields_set__", set()))
        if "diffusion_engine" not in fields:
            req.diffusion_engine = settings.diffusion_engine
    if req.use_prompt_planner:
        helper_backend = "gguf-server" if settings.local_llm_backend == "gguf_server" else "local"
        plan = plan_prompt(
            req.prompt,
            enabled=True,
            max_tokens=int(getattr(req, "prompt_planner_max_tokens", 700)),
            backend=helper_backend,
            gguf_helper_base_url=settings.gguf_helper_base_url,
            gguf_helper_model=settings.gguf_helper_model,
            gguf_helper_timeout_sec=settings.gguf_helper_timeout_sec,
        )
        req.prompt_planner_output = plan.model_dump()
        if not req.prompt_planner_lock_original and plan.planned_prompt:
            req.prompt = plan.planned_prompt
        if plan.negative_prompt and not req.negative_prompt.strip():
            req.negative_prompt = plan.negative_prompt

    # Optional prompt expansion
    if req.use_prompt_expander:
        helper_backend = (
            "gguf-server"
            if settings.local_llm_backend == "gguf_server" and settings.prompt_expander_backend == "local"
            else settings.prompt_expander_backend
        )
        result = expand_prompt_result(
            req.prompt,
            backend=helper_backend,
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_model=settings.openrouter_model,
            openrouter_free_only=settings.openrouter_free_only,
            ideogram_api_key=settings.ideogram_api_key,
            gguf_helper_base_url=settings.gguf_helper_base_url,
            gguf_helper_model=settings.gguf_helper_model,
            gguf_helper_timeout_sec=settings.gguf_helper_timeout_sec,
        )
        if result.error:
            raise HTTPException(502, result.error)
        req.prompt = result.expanded

    username, role, _is_admin = _request_user_role(request)
    job_id = _new_job(username=username, role=role)

    decision = moderate_prompt(req.prompt, req.negative_prompt, role=role)
    if not decision.allowed:
        event_id = await save_moderation_event(
            username=username or "local",
            role=role,
            event_type=decision.event_type,
            action="block_prompt",
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            mode=req.mode,
            scores=decision.scores,
            reason=decision.reason,
            job_id=job_id,
        )
        job = _jobs[job_id]
        job["status"] = "blocked"
        job["error"] = "This prompt was blocked by the child safety filter and sent to an admin for review."
        job["moderation_event_id"] = event_id
        await ws_manager.broadcast(job_id, {"type": "blocked", "error": job["error"], "moderation_event_id": event_id})
        return {"job_id": job_id, "status": "blocked", "moderation_event_id": event_id}

    if generation_queue is None:
        raise HTTPException(503, "Generation queue is not ready yet.")
    if req.batch_mode == "parallel" and int(req.num_images or 1) > 1:
        from resource_manager import plan_parallel_batch
        from system_check import get_gpu_info

        _name, _total, free = get_gpu_info()
        parallel_plan = plan_parallel_batch(
            free_vram_gb=free,
            width=req.width,
            height=req.height,
            quantization=req.quantization,
            batch=req.num_images,
            cfg_active=req.cfg > 0,
            mode=req.mode,
            checkpoint=req.checkpoint,
        )
        if not parallel_plan["allowed"] or not req.parallel_batch_confirmed:
            req.batch_mode = "safe_queue"
            req.parallel_batch_confirmed = False
    if req.batch_mode == "safe_queue" and int(req.num_images or 1) > 1:
        parent_job = _jobs[job_id]
        parent_job["status"] = "queued"
        parent_job["batch"] = {"mode": "safe_queue", "count": int(req.num_images), "parallel": False}
        child_job_ids: list[str] = []
        child_positions: list[int | None] = []
        for index, child_req in enumerate(build_safe_batch_children(req)):
            child_job_id = _new_job(username=username, role=role)
            child_job = _jobs[child_job_id]
            child_job["parent_job_id"] = job_id
            child_job["batch_index"] = index
            child_job["batch_count"] = int(req.num_images)
            queue_state = generation_queue.enqueue(
                child_job_id,
                {"req": child_req, "username": username, "role": role, "parent_job_id": job_id, "batch_index": index},
                username=username,
                role=role,
            )
            child_job_ids.append(child_job_id)
            child_positions.append(queue_state.get("queue_position"))
        parent_job["child_job_ids"] = child_job_ids
        parent_job["queue_position"] = min((pos for pos in child_positions if pos is not None), default=None)
        parent_job["queue_length"] = max((pos for pos in child_positions if pos is not None), default=len(child_job_ids))
        _sync_queue_state_to_jobs()
        await ws_manager.broadcast(job_id, {"type": "queue", **parent_job})
        return {
            "job_id": job_id,
            "batch_id": job_id,
            "child_job_ids": child_job_ids,
            "status": "queued",
            "queue_position": parent_job.get("queue_position"),
            "queue_length": parent_job.get("queue_length"),
        }
    queue_state = generation_queue.enqueue(job_id, {"req": req, "username": username, "role": role}, username=username, role=role)
    _sync_queue_state_to_jobs()
    job = _jobs[job_id]
    await ws_manager.broadcast(job_id, {"type": "queue", **job})
    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": queue_state.get("queue_position"),
        "queue_length": queue_state.get("queue_length"),
    }


def _fit_preview_dimensions(width: int, height: int, max_side: int = 512) -> tuple[int, int]:
    width = max(64, int(width or max_side))
    height = max(64, int(height or max_side))
    scale = min(max_side / max(width, height), 1.0)
    w = max(64, int(width * scale))
    h = max(64, int(height * scale))
    return max(64, round(w / 16) * 16), max(64, round(h / 16) * 16)


def _job_images_from_b64(results: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for value in results:
        payload = str(value or "")
        if "," in payload:
            payload = payload.split(",", 1)[1]
        images.append(Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB"))
    return images


def _quarantine_output_files(filenames: list[str], job_id: str) -> str | None:
    if not filenames:
        return None
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    quarantined: str | None = None
    for filename in filenames:
        safe_source = _safe_served_filename(filename)
        if safe_source is None:
            continue
        src = OUTPUTS_DIR / safe_source
        if not src.exists():
            continue
        dst_name = f"{job_id}_{filename}"
        safe_dest = _safe_served_filename(dst_name)
        if safe_dest is None:
            continue
        dst = QUARANTINE_DIR / safe_dest
        src.replace(dst)
        quarantined = quarantined or safe_dest
    return quarantined


def _safe_served_filename(filename: str) -> str | None:
    safe_name = Path(str(filename or "")).name
    if safe_name != filename or not SAFE_SERVED_FILENAME_RE.fullmatch(safe_name):
        return None
    return safe_name


@app.post("/api/realtime/preview")
async def realtime_preview(req: RealtimePreviewRequest):
    if not pipeline.is_loaded():
        raise HTTPException(503, "No Krea model is loaded. Load a model before realtime preview.")
    if generation_queue is not None and generation_queue.has_active_or_pending():
        raise HTTPException(409, "Krea is busy with another generation or preview. Try again when it finishes.")
    session_id = req.session_id.strip() or uuid.uuid4().hex
    if realtime_previews.busy():
        job = realtime_previews.create_pending(session_id, req)
        return {
            "job_id": job["job_id"],
            "session_id": session_id,
            "revision": job["revision"],
            "status": "queued",
            "dropped_intermediate_frames": True,
        }
    job = realtime_previews.create(session_id)
    asyncio.create_task(_run_realtime_preview(job["job_id"], req, session_id))
    return {
        "job_id": job["job_id"],
        "session_id": session_id,
        "revision": job["revision"],
        "status": job["status"],
        "dropped_intermediate_frames": False,
    }


@app.get("/api/realtime/preview/{job_id}")
async def realtime_preview_status(job_id: str):
    job = realtime_previews.get(job_id)
    if job is None:
        raise HTTPException(404, "Preview job not found")
    return job


@app.post("/api/realtime/cancel/{job_id}")
async def realtime_preview_cancel(job_id: str):
    if not realtime_previews.cancel(job_id):
        raise HTTPException(404, "Preview job not found")
    return {"ok": True, "job_id": job_id, "status": "cancelled"}


async def _run_realtime_preview(job_id: str, req: RealtimePreviewRequest, session_id: str):
    realtime_previews.mark_started(job_id)
    loop = asyncio.get_event_loop()

    def progress_cb(step: int, total: int):
        pct = int(step / max(total, 1) * 100)
        realtime_previews.update(job_id, progress=max(1, min(pct, 98)))

    try:
        width, height = _fit_preview_dimensions(req.width, req.height)
        preview_req = GenerationRequest(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            mode="redraw",
            checkpoint="turbo",
            quantization="fp8",
            steps=max(4, min(6, int(req.preview_steps))),
            cfg=0.0,
            width=width,
            height=height,
            num_images=1,
            seed=int(req.seed),
            denoise=1.0,
            use_prompt_expander=False,
            refine=False,
            moodboard_strength=max(0.0, min(1.0, float(req.moodboard_strength))),
            mood=req.mood,
            moodboard_ids=list(req.moodboard_ids or []),
            moodboard_uuids=list(req.moodboard_uuids or []),
            moodboard_images=[req.canvas_image_b64, *list(req.moodboard_images or [])],
            loras=list(req.loras or []),
        )
        results, seed, filenames, _, metadata = await loop.run_in_executor(
            None, lambda: pipeline.generate(preview_req, progress_cb=progress_cb, save_outputs=False)
        )
        image = results[0] if results else ""
        realtime_previews.complete(job_id, image_b64=image, seed=seed, metadata=metadata[0] if metadata else {})
    except Exception as e:
        logger.exception("Realtime preview failed")
        realtime_previews.fail(job_id, str(e))
    finally:
        realtime_previews.mark_finished(job_id)
        pending = realtime_previews.pop_pending(session_id)
        if pending is not None:
            asyncio.create_task(_run_realtime_preview(pending["job"]["job_id"], pending["payload"], session_id))


async def _run_generation(job_id: str, req: GenerationRequest, *, username: str | None = None, role: str = "user"):
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
        write_generation_breadcrumb(
            LOGS_DIR,
            job_id=job_id,
            req=req,
            stage="generation_start",
            extra={"username": username, "role": role},
        )
        prep = prepare_for_generation(pipeline, clear_conditioning_cache=False)
        logger.info("Pre-generation memory cleanup: %s", prep)
        if getattr(req, "diffusion_engine", "native_pytorch") == "gguf_external":
            from gguf_diffusion_provider import generate_gguf_external

            pipeline.unload()
            req.loras = []
            req.style_references = []
            req.regional_prompts = []
            req.moodboard_images = []
            req.use_rebalance = False
            req.krea_enhancer_enabled = False
            req.krea_enhancer_variant = "off"
            job["edit_provider"] = "gguf_external"
            job["provider_warning"] = "GGUF external engine active: Krea-native LoRAs, style refs, moodboards, regional prompts, rebalance, and enhancer are disabled."
            write_generation_breadcrumb(LOGS_DIR, job_id=job_id, req=req, stage="gguf_external_start")
            results, seed, filenames, lora_reports, metadata = await loop.run_in_executor(
                None, lambda: generate_gguf_external(req, _gguf_runtime_settings(), progress_cb=progress_cb)
            )
        else:
            if getattr(req, "diffusion_engine", "native_pytorch") in {"native_int8_convrot", "int8_convrot_external"}:
                req.diffusion_engine = "native_int8_convrot"
                req.quantization = "int8"
            from edit_providers import resolve_edit_provider
            from flux_fill_provider import flux_fill_installed, generate_flux_fill

            flux_installed = flux_fill_installed()
            requested_method = getattr(req, "inpaint_method", "native")
            provider_name = "flux_fill" if requested_method == "flux_fill" else req.edit_provider
            if requested_method in {"native", "lanpaint_experimental"}:
                provider_name = "krea_native"
            provider = resolve_edit_provider(provider_name, req.mode, flux_fill_installed=flux_installed)
            job["edit_provider"] = provider.name
            job["provider_warning"] = (
                provider.reason
                if req.edit_provider in {"auto", "flux_fill"} and provider.name != "flux_fill" and req.mode in {"inpaint", "outpaint"}
                else None
            )
            if provider.name == "flux_fill":
                pipeline.unload()
                if req.steps < 50:
                    req.steps = 50
                if req.cfg < 30:
                    req.cfg = 30
                write_generation_breadcrumb(LOGS_DIR, job_id=job_id, req=req, stage="flux_fill_start", extra={"provider": provider.name})
                results, seed, filenames, lora_reports, metadata = await loop.run_in_executor(
                    None, lambda: generate_flux_fill(req, progress_cb=progress_cb)
                )
            else:
                write_generation_breadcrumb(LOGS_DIR, job_id=job_id, req=req, stage="native_generation_start", extra={"provider": provider.name})
                results, seed, filenames, lora_reports, metadata = await loop.run_in_executor(
                    None, lambda: pipeline.generate(req, progress_cb=progress_cb)
                )
        missing_outputs = [fname for fname in (filenames or []) if fname and not (OUTPUTS_DIR / fname).exists()]
        write_generation_breadcrumb(
            LOGS_DIR,
            job_id=job_id,
            req=req,
            stage="generation_returned",
            extra={
                "seed": seed,
                "filenames": filenames,
                "result_count": len(results or []),
                "missing_outputs": missing_outputs,
            },
        )
        if missing_outputs:
            logger.error("Generation job %s returned missing output files: %s", job_id, missing_outputs)
        if role == "child":
            image_decision = moderate_images(_job_images_from_b64(results), role=role)
            if not image_decision.allowed:
                quarantined = _quarantine_output_files(filenames, job_id)
                event_id = await save_moderation_event(
                    username=username or "local",
                    role=role,
                    event_type=image_decision.event_type,
                    action="block_image",
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    mode=req.mode,
                    scores=image_decision.scores,
                    reason=image_decision.reason,
                    job_id=job_id,
                    quarantined_filename=quarantined,
                )
                job["images"] = []
                job["metadata"] = []
                job["seed"] = seed
                job["status"] = "blocked"
                job["progress"] = 100
                job["error"] = "That image was blocked by the child safety filter and sent to an admin for review."
                job["moderation_event_id"] = event_id
                await ws_manager.broadcast(job_id, {"type": "blocked", "error": job["error"], "moderation_event_id": event_id})
                parent_job_id = job.get("parent_job_id")
                if parent_job_id:
                    parent = _refresh_parent_batch_job(str(parent_job_id))
                    if parent:
                        await ws_manager.broadcast(str(parent_job_id), {"type": "batch", **parent})
                return
        job["images"] = results
        job["seed"] = seed
        job["metadata"] = metadata
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
                    metadata=metadata[i] if i < len(metadata) else {},
                    owner_username=username if SHARE_AUTH_ENABLED else None,
                )
            except Exception:
                logger.exception(f"Gallery save failed for {fname}")

        await ws_manager.broadcast(job_id, {
            "type": "done", "images": results, "seed": seed, "metadata": metadata,
            "lora_warnings": lora_warnings,
            "edit_provider": job.get("edit_provider"),
            "provider_warning": job.get("provider_warning"),
        })
        parent_job_id = job.get("parent_job_id")
        if parent_job_id:
            parent = _refresh_parent_batch_job(str(parent_job_id))
            if parent:
                await ws_manager.broadcast(str(parent_job_id), {"type": "batch", **parent})

    except Exception as e:
        logger.exception(f"Generation failed for job {job_id}")
        write_generation_breadcrumb(LOGS_DIR, job_id=job_id, req=req, stage="generation_error", extra={"error": str(e)})
        job["status"] = "error"
        job["error"] = str(e)
        await ws_manager.broadcast(job_id, {"type": "error", "error": str(e)})
        parent_job_id = job.get("parent_job_id")
        if parent_job_id:
            parent = _refresh_parent_batch_job(str(parent_job_id))
            if parent:
                await ws_manager.broadcast(str(parent_job_id), {"type": "batch", **parent})
    finally:
        if job.get("status") in {"done", "blocked", "error"}:
            clear_generation_breadcrumb(LOGS_DIR, job_id=job_id)


@app.get("/api/generate/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.get("child_job_ids"):
        _refresh_parent_batch_job(job_id)
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

def _load_model_error_detail(exc: Exception) -> str:
    detail = str(exc) or "Model load failed"
    lower = detail.lower()
    if any(token in lower for token in ("vram", "system ram", "ram critically low", "no model loaded")):
        detail += " Check the System tab for free RAM/VRAM and duplicate GPU Python processes."
    return detail


@app.post("/api/load-model")
async def load_model(req: LoadModelRequest):
    if generation_queue is not None and generation_queue.has_active_or_pending():
        raise HTTPException(409, "Generation queue is active. Wait for queued/running jobs before loading a model.")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None, lambda: pipeline.load(
                req.checkpoint_path, req.quantization,
                blocks_to_swap=req.blocks_to_swap,
                fp8_fast_matmul=bool(getattr(req, "fp8_fast_matmul", False)),
                torch_compile=bool(getattr(req, "torch_compile", False)),
            )
        )
    except Exception as exc:
        logger.exception("Model load failed")
        if isinstance(exc, (RuntimeError, FileNotFoundError)):
            raise HTTPException(400, _load_model_error_detail(exc))
        raise HTTPException(500, "Model load failed. Check the server logs for details.")
    return {"status": "loaded", "checkpoint": req.checkpoint_path}


@app.post("/api/load-model/preflight")
async def load_model_preflight(req: LoadModelRequest):
    try:
        from inference import preflight_model_load

        preflight_model_load(req.checkpoint_path, req.quantization, blocks_to_swap=req.blocks_to_swap)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        return {"ok": False, "detail": _load_model_error_detail(exc), "system": get_system_report()}
    return {"ok": True, "detail": "This model load passes the current RAM/VRAM preflight.", "system": get_system_report()}


@app.post("/api/unload-model")
async def unload_model():
    if generation_queue is not None and generation_queue.has_active_or_pending():
        raise HTTPException(409, "Generation queue is active. Wait for queued/running jobs before unloading the model.")
    result = unload_pipeline(pipeline)
    return {"status": "unloaded", **result}


@app.post("/api/memory/release-transient")
async def memory_release_transient():
    return safe_clean_memory(pipeline)


@app.post("/api/memory/safe-clean")
async def memory_safe_clean():
    return safe_clean_memory(pipeline)


@app.post("/api/memory/unload-model")
async def memory_unload_model():
    if generation_queue is not None and generation_queue.has_active_or_pending():
        raise HTTPException(409, "Generation queue is active. Wait for queued/running jobs before unloading the model.")
    result = unload_pipeline(pipeline)
    return {"status": "unloaded", **result}


@app.get("/api/memory/processes")
async def memory_processes():
    return {"items": detect_krea_server_processes()}


@app.post("/api/memory/stop-process")
async def memory_stop_process(req: MemoryStopProcessRequest):
    try:
        return stop_krea_server_process(req.pid)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------

@app.get("/api/gallery")
async def gallery(request: Request, page: int = 1, page_size: int = 50, favorites: bool = False):
    username, _role, is_role_admin = _request_user_role(request)
    return await get_gallery(page, page_size, favorites, owner_username=username, is_admin=is_role_admin)


@app.put("/api/gallery/{gallery_id}/favorite")
async def favorite(gallery_id: int, req: FavoriteRequest, request: Request):
    username, _role, is_role_admin = _request_user_role(request)
    ok = await set_favorite(gallery_id, req.favorite, owner_username=username, is_admin=is_role_admin)
    if not ok:
        raise HTTPException(404, "Not found")
    return {"ok": True}


@app.delete("/api/gallery/{gallery_id}")
async def delete_gallery_item(gallery_id: int, request: Request):
    username, _role, is_role_admin = _request_user_role(request)
    filename = await delete_image(gallery_id, owner_username=username, is_admin=is_role_admin)
    if filename is None:
        raise HTTPException(404, "Not found")
    return {"ok": True, "filename": filename}


@app.get("/api/outputs/{filename}")
async def output_file(filename: str, request: Request):
    safe_name = _safe_served_filename(filename)
    if safe_name is None:
        raise HTTPException(404, "Not found")
    row = await get_image_record_by_filename(safe_name)
    if row is None:
        raise HTTPException(404, "Not found")
    username, _role, is_role_admin = _request_user_role(request)
    owner = row.get("owner_username")
    if not is_role_admin and owner != username:
        raise HTTPException(404, "Not found")
    return await _outputs_static.get_response(safe_name, request.scope)


@app.get("/api/moderation/events")
async def moderation_events(username: str = "", limit: int = 100):
    return await list_moderation_events(username=username or None, limit=limit)


@app.get("/api/moderation/status")
async def moderation_status():
    available = image_classifier_available()
    return {
        "image_classifier_available": available,
        "child_image_moderation": "ready" if available else "blocked_until_image_classifier_installed",
        "message": (
            "Child image classifier is available. Child image outputs are checked after generation."
            if available
            else "Child prompt blocking still works, but child generated images fail closed until the image classifier is installed."
        ),
    }


@app.post("/api/moderation/install-image-classifier")
async def moderation_install_image_classifier():
    if image_classifier_available():
        return {"ok": True, "installed": True, "message": "Transformers image classifier dependencies are already available."}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "transformers"],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception as exc:
        raise HTTPException(500, f"Image classifier setup failed: {exc}") from exc
    output = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode != 0:
        raise HTTPException(500, output[-2000:] or "Image classifier setup failed.")
    return {"ok": True, "installed": image_classifier_available(), "message": output[-2000:]}


@app.get("/api/moderation/quarantine/{filename}")
async def moderation_quarantine_file(filename: str):
    safe_name = _safe_served_filename(filename)
    if safe_name is None:
        raise HTTPException(404, "Not found")
    return await _quarantine_static.get_response(safe_name, {"type": "http", "method": "GET", "path": f"/{safe_name}", "headers": []})


# ---------------------------------------------------------------------------
# Moodboard catalog
# ---------------------------------------------------------------------------

@app.get("/api/moodboards", response_model=MoodboardListResponse)
async def moodboards(q: str = "", page: int = 1, page_size: int = 50, favorites: bool = False, source: str = "", shuffle_seed: str = ""):
    return await list_moodboards(
        query=q,
        page=page,
        page_size=page_size,
        favorites_only=favorites,
        source=source,
        shuffle_seed=shuffle_seed,
    )


@app.get("/api/moodboards/discoveries/latest", response_model=MoodboardDiscoveryResponse)
async def moodboard_latest_discovery():
    return await latest_moodboard_discovery()


@app.get("/api/moodboards/cached-image")
async def moodboard_cached_image(url: str):
    loop = asyncio.get_event_loop()
    try:
        path = await loop.run_in_executor(None, lambda: fetch_cached_moodboard_image(url))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        logger.exception("Krea moodboard cached image fetch failed")
        raise HTTPException(502, "Could not cache Krea moodboard image")
    return FileResponse(path)


@app.get("/api/moodboards/{moodboard_id}", response_model=MoodboardItem)
async def moodboard_detail(moodboard_id: int):
    item = await get_moodboard(moodboard_id)
    if item is None:
        raise HTTPException(404, "Not found")
    return item


@app.put("/api/moodboards/{moodboard_id}/favorite")
async def favorite_moodboard(moodboard_id: int, req: FavoriteRequest):
    await set_moodboard_favorite(moodboard_id, req.favorite)
    return {"ok": True}


@app.post("/api/moodboards/{moodboard_id}/qwen-guidance", response_model=MoodboardItem)
async def qwen_guidance_moodboard(moodboard_id: int):
    try:
        return await generate_and_store_moodboard_qwen_guidance(moodboard_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.exception("Qwen moodboard guidance failed")
        raise HTTPException(502, f"Qwen moodboard guidance failed: {exc}") from exc


@app.post("/api/moodboards/qwen-guidance-missing")
async def qwen_guidance_missing(req: MoodboardGuidanceMissingRequest):
    try:
        return await generate_missing_moodboard_qwen_guidance(limit=req.limit)
    except Exception as exc:
        logger.exception("Qwen moodboard guidance batch failed")
        raise HTTPException(502, f"Qwen moodboard guidance batch failed: {exc}") from exc


@app.post("/api/moodboards/mashup", response_model=MoodboardItem)
async def mashup_moodboard(req: MoodboardMashupRequest):
    try:
        return await create_mashup_moodboard(moodboard_ids=req.moodboard_ids, weights=req.weights)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Qwen moodboard mashup failed")
        raise HTTPException(502, f"Qwen moodboard mashup failed: {exc}") from exc


@app.post("/api/moodboards/custom", response_model=MoodboardItem)
async def create_custom_moodboard_endpoint(req: CustomMoodboardRequest):
    try:
        return await create_custom_moodboard(
            title=req.title,
            taste_profile=req.taste_profile,
            keywords=req.keywords,
            image_b64s=req.image_b64s,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Qwen custom moodboard authoring failed")
        raise HTTPException(502, f"Qwen custom moodboard authoring failed: {exc}") from exc


@app.delete("/api/moodboards/custom/{moodboard_id}")
async def delete_custom_moodboard_endpoint(moodboard_id: int):
    if not await delete_custom_moodboard(moodboard_id):
        raise HTTPException(404, "Custom moodboard not found")
    return {"ok": True}


@app.post("/api/moodboards/import", response_model=MoodboardImportResponse)
async def import_moodboards(req: MoodboardImportRequest):
    urls = req.urls or [KREA_MOODBOARD_GALLERY_URL]
    try:
        return await import_moodboard_urls(urls, max_pages=req.max_pages, use_browser_discovery=not req.urls)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/moodboards/export-seed", response_model=MoodboardExportResponse)
async def export_moodboards_seed():
    exported = await export_moodboard_seed(MOODBOARD_SEED_PATH)
    return {"exported": exported, "path": str(MOODBOARD_SEED_PATH)}


@app.post("/api/moodboards/image", response_model=MoodboardImageResponse)
async def moodboard_image(req: MoodboardImageRequest):
    loop = asyncio.get_event_loop()
    try:
        image_b64 = await loop.run_in_executor(None, lambda: fetch_moodboard_image_b64(req.url))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        logger.exception("Krea moodboard image fetch failed")
        raise HTTPException(502, "Could not load Krea moodboard image")
    return {"image_b64": image_b64}


_CUSTOM_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_CUSTOM_IMAGE_FILENAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def _safe_custom_moodboard_image_path(board_uuid: str, filename: str) -> Path:
    try:
        safe_board_uuid = str(uuid.UUID(str(board_uuid)))
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid custom moodboard id") from exc

    if safe_board_uuid != str(board_uuid).lower():
        raise ValueError("Invalid custom moodboard id")
    if not filename or "/" in filename or "\\" in filename:
        raise ValueError("Invalid custom moodboard image name")
    if any(char not in _CUSTOM_IMAGE_FILENAME_CHARS for char in filename):
        raise ValueError("Invalid custom moodboard image name")
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in _CUSTOM_IMAGE_SUFFIXES:
        raise ValueError("Invalid custom moodboard image type")

    root = CUSTOM_MOODBOARD_DIR.resolve()
    if not root.is_dir():
        raise ValueError("Custom moodboard storage is unavailable")
    try:
        board_dir = next(
            candidate.resolve()
            for candidate in root.iterdir()
            if candidate.is_dir() and candidate.name == safe_board_uuid
        )
        image_path = next(
            candidate.resolve()
            for candidate in board_dir.iterdir()
            if candidate.is_file() and candidate.name == filename
        )
    except StopIteration as exc:
        raise ValueError("Invalid custom moodboard image path") from exc
    if board_dir.parent != root or image_path.parent != board_dir:
        raise ValueError("Invalid custom moodboard image path")
    return image_path


@app.get("/api/moodboards/custom-images/{board_uuid}/{filename}")
async def custom_moodboard_image(board_uuid: str, filename: str):
    try:
        path = _safe_custom_moodboard_image_path(board_uuid, filename)
    except ValueError:
        raise HTTPException(404, "Custom moodboard image not found")
    return FileResponse(path)


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
    from lora_manager import OFFICIAL_LORAS, official_lora_download_kwargs
    from huggingface_hub import hf_hub_download
    if lora_name not in OFFICIAL_LORAS:
        raise HTTPException(404, f"Unknown LoRA: {lora_name}")
    loop = asyncio.get_event_loop()
    try:
        dest = await loop.run_in_executor(
            None,
            lambda: hf_hub_download(
                **official_lora_download_kwargs(lora_name, token=settings.hf_token),
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
        b64_to_pil, upscale_model_refine, upscale_realesrgan,
        upscale_refine_2pass, upscale_tiled_vae, upscale_ultimate,
    )
    from output_saver import encode_images

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
                device=pipeline._device, dtype=pipeline._dtype,
                scale=req.upscale_by,
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
                img, pipeline, MODELS_DIR, prompt=req.prompt, scale=req.upscale_by,
                tile=req.tile_width or req.tile_size, padding=req.tile_padding,
                mask_blur=req.mask_blur, denoise=req.denoise, steps=req.steps,
                cfg=req.cfg, sampler=req.sampler, scheduler=req.scheduler,
                seam_mode=req.seam_mode, tile_mode=req.tile_mode,
                tiled_decode=req.tiled_decode, seam_fix=req.seam_fix,
            )
        )
    elif req.method == "refine_2pass":
        if not pipeline.is_loaded():
            raise HTTPException(400, "Model must be loaded for 2-pass refine upscale")
        result = await loop.run_in_executor(
            None, lambda: upscale_refine_2pass(
                img, pipeline, MODELS_DIR, prompt=req.prompt, scale=req.upscale_by,
                denoise=req.denoise, steps=req.steps, cfg=req.cfg,
                sampler=req.sampler, scheduler=req.scheduler,
                tile_size=req.tile_width or req.tile_size, tiled_decode=req.tiled_decode,
            )
        )
    elif req.method == "pid_upscale":
        from pid_decoder_provider import upscale_pid

        result = await loop.run_in_executor(
            None, lambda: upscale_pid(img, _pid_settings(), prompt=req.prompt, scale=req.upscale_by)
        )
    else:
        raise HTTPException(400, f"Unknown upscale method: {req.method}")

    metadata = {
        "schema_version": 1,
        "app": "Krea 2 Studio",
        "operation": "upscale",
        "prompt": req.prompt,
        "method": req.method,
        "scale": req.scale,
        "upscale_by": req.upscale_by,
        "denoise": req.denoise,
        "tile_size": req.tile_size,
        "tile_width": req.tile_width,
        "tile_height": req.tile_height,
        "tile_padding": req.tile_padding,
        "mask_blur": req.mask_blur,
        "seam_mode": req.seam_mode,
        "tile_mode": req.tile_mode,
        "sampler": req.sampler,
        "scheduler": req.scheduler,
        "steps": req.steps,
        "cfg": req.cfg,
        "tiled_decode": req.tiled_decode,
        "seam_fix": req.seam_fix,
        "source_gallery_id": req.gallery_id,
        "width": result.width,
        "height": result.height,
    }
    if req.method == "pid_upscale":
        from pid_decoder_provider import pid_status

        metadata["runtime"] = {"pid": pid_status(_pid_settings())}
    encoded, _ = encode_images([result], OUTPUTS_DIR, save_outputs=False, metadata=[metadata])
    return {"image_b64": encoded[0], "metadata": metadata}


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


@app.post("/api/preprocess/preview")
async def preprocessor_preview(req: PreprocessorPreviewRequest):
    """Generate a lightweight ControlNet-Aux-style preview image."""
    import base64 as _b64
    import io as _io
    from PIL import Image as _Image
    from preprocessors import preprocess_image

    try:
        source = req.image_b64.split(",", 1)[1] if "," in req.image_b64 else req.image_b64
        img = _Image.open(_io.BytesIO(_b64.b64decode(source)))
        preview = preprocess_image(
            img,
            kind=req.kind,
            resolution=req.resolution,
            low_threshold=req.low_threshold,
            high_threshold=req.high_threshold,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.warning("Preprocessor preview failed: %s", exc)
        raise HTTPException(400, "Bad image data.") from exc

    buf = _io.BytesIO()
    preview.save(buf, format="PNG")
    return {
        "image_b64": _b64.b64encode(buf.getvalue()).decode(),
        "kind": req.kind,
        "width": preview.width,
        "height": preview.height,
    }


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
        "text_encoder_source": getattr(pipeline, "_text_encoder_source", None),
        "memory": pipeline.memory_status(),
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


@app.get("/api/quality-assets")
async def quality_assets_status():
    from quality_assets import asset_specs, asset_status

    has_token = bool(settings.hf_token or os.environ.get("HF_TOKEN"))
    return {
        "has_hf_token": has_token,
        "items": [asset_status(spec, has_hf_token=has_token) for spec in asset_specs()],
    }


@app.post("/api/quality-assets/{asset_id}/download")
async def download_quality_asset_endpoint(asset_id: str):
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from quality_assets import asset_by_id, asset_status, download_asset

    try:
        spec = asset_by_id(asset_id)
    except KeyError:
        raise HTTPException(404, f"Unknown quality asset: {asset_id}")
    if not spec.download_enabled:
        raise HTTPException(403, spec.disabled_reason or "This asset cannot be downloaded automatically.")

    token = settings.hf_token or os.environ.get("HF_TOKEN") or None
    if spec.id == "flux_fill" and not token:
        raise HTTPException(
            401,
            "FLUX Fill is gated on Hugging Face. Open System > Precision Editing, paste an HF token that has accepted access to black-forest-labs/FLUX.1-Fill-dev, then retry.",
        )

    loop = asyncio.get_event_loop()
    try:
        path = await loop.run_in_executor(None, lambda: download_asset(spec, token=token))
    except GatedRepoError:
        raise HTTPException(
            401,
            "Your Hugging Face token does not have access to black-forest-labs/FLUX.1-Fill-dev yet. Open the model page, accept the license/access terms, then retry.",
        )
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) in {401, 403}:
            raise HTTPException(
                401,
                "Hugging Face rejected the FLUX Fill download. Confirm the token is valid and has access to the gated model, then retry.",
            )
        logger.exception("Quality asset download failed")
        raise HTTPException(502, "Quality asset download failed. Check connection and server logs.")
    except Exception:
        logger.exception("Quality asset download failed")
        raise HTTPException(502, "Quality asset download failed. Check connection and server logs.")

    has_token = bool(settings.hf_token or os.environ.get("HF_TOKEN"))
    return {"ok": True, "path": str(path), "item": asset_status(spec, has_hf_token=has_token)}


@app.post("/api/xperiment/setup")
async def xperiment_setup_endpoint():
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from quality_assets import asset_by_id, asset_installed, asset_status, download_asset

    required_ids = ["wan_2_1_vae", "qwen3vl_abliterated_fp8", "krea2_realism_v1_lora"]
    results: list[dict] = []
    token = settings.hf_token or os.environ.get("HF_TOKEN") or None
    loop = asyncio.get_event_loop()
    for asset_id in required_ids:
        spec = asset_by_id(asset_id)
        skipped = asset_installed(spec)
        path = spec.local_path
        if not skipped:
            try:
                path = await loop.run_in_executor(None, lambda spec=spec: download_asset(spec, token=token))
            except (GatedRepoError, HfHubHTTPError) as exc:
                raise HTTPException(502, f"Could not download {asset_id}: {exc}") from exc
            except Exception as exc:
                logger.exception("Xperiment asset download failed")
                raise HTTPException(502, f"Could not download {asset_id}. Check connection and server logs.") from exc
        results.append({"id": asset_id, "path": str(path), "skipped": skipped, "item": asset_status(spec, has_hf_token=bool(token))})

    wan_vae = asset_by_id("wan_2_1_vae").local_path
    env = _read_env()
    env["KREA2_VAE_PATH"] = str(wan_vae)
    env["PROMPT_EXPANDER_BACKEND"] = "local"
    env["LOCAL_LLM_BACKEND"] = "transformers"
    try:
        from support_models import support_model_path
        xperiment_qwen = str(support_model_path("qwen3_vl_abliterated"))
    except Exception:
        xperiment_qwen = "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"
    env["LOCAL_QWEN_MODEL_ID"] = xperiment_qwen
    settings.krea2_vae_path = str(wan_vae)
    settings.prompt_expander_backend = "local"
    settings.local_llm_backend = "transformers"
    settings.local_qwen_model_id = xperiment_qwen
    _write_env(env)
    bypass_spec = asset_by_id("krea2_filter_bypass")
    bypass = asset_status(bypass_spec, has_hf_token=bool(token))
    loras = [
        {"name": "Krea2-realism-V1", "filename": "Krea2-realism-V1.safetensors", "strength": 0.55, "block_filter": "late"},
    ]
    if asset_installed(bypass_spec):
        loras.append({"name": "krea2filterbypass3", "filename": "krea2filterbypass3.safetensors", "strength": 4.0, "block_filter": "style_safe"})
    configured_engine = settings.diffusion_engine or "native_pytorch"
    configured_quant = "int8" if configured_engine == "native_int8_convrot" else "fp8"
    return {
        "ok": True,
        "assets": results,
        "vae_path": str(wan_vae),
        "lora": loras[0],
        "loras": loras,
        "diffusion_engine": configured_engine,
        "quantization": configured_quant,
        "sampler": {"sampler": "er_sde", "scheduler": "beta57", "steps": 6, "cfg": 0.0},
        "use_prompt_expander": False,
        "prompt_expander_backend": "local",
        "local_llm_backend": "transformers",
        "local_qwen_model_id": xperiment_qwen,
        "benchmark_note": "Verified on RTX 4090: 1024px, er_sde/beta57, 6 steps, CFG 0, Realism LoKr late@0.55 completed in ~10s with good prompt adherence.",
        "manual_only": [bypass],
        "warnings": [
            "Exact ClownsharKSampler_Beta is a Comfy/RES4LYF node; native Krea Studio applies the closest safe native mapping: er_sde + beta57.",
            "krea2filterbypass3 is manual-only and not auto-downloaded.",
            "Xperiment keeps the diffusion engine selected in .env instead of forcing Original/Native PyTorch.",
        ],
    }


@app.post("/api/gguf/setup-low-vram")
async def gguf_setup_low_vram_endpoint():
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from gguf_runtime_installer import install_stable_diffusion_cpp
    from quality_assets import asset_by_id, asset_installed, asset_status, download_asset

    required_ids = ["gguf_krea2_turbo_q4km", "gguf_krea2_turbo_q3km", "gguf_qwen3vl_4b_q4km", "wan_2_1_vae"]
    token = settings.hf_token or os.environ.get("HF_TOKEN") or None
    loop = asyncio.get_event_loop()
    runtime = await loop.run_in_executor(
        None,
        lambda: install_stable_diffusion_cpp(Path(settings.models_dir).parent / "tools" / "stable-diffusion.cpp"),
    )
    results: list[dict] = []
    for asset_id in required_ids:
        spec = asset_by_id(asset_id)
        skipped = asset_installed(spec)
        path = spec.local_path
        if not skipped:
            try:
                path = await loop.run_in_executor(None, lambda spec=spec: download_asset(spec, token=token))
            except (GatedRepoError, HfHubHTTPError) as exc:
                raise HTTPException(502, f"Could not download {asset_id}: {exc}") from exc
            except Exception as exc:
                logger.exception("GGUF low-VRAM asset download failed")
                raise HTTPException(502, f"Could not download {asset_id}. Check connection and server logs.") from exc
        results.append({"id": asset_id, "path": str(path), "skipped": skipped, "item": asset_status(spec, has_hf_token=bool(token))})

    q4 = asset_by_id("gguf_krea2_turbo_q4km").local_path
    q3 = asset_by_id("gguf_krea2_turbo_q3km").local_path
    qwen = asset_by_id("gguf_qwen3vl_4b_q4km").local_path
    vae = asset_by_id("wan_2_1_vae").local_path
    env = _read_env()
    env["DIFFUSION_ENGINE"] = "gguf_external"
    env["GGUF_SD_CLI_PATH"] = runtime["sd_cli_path"]
    env["GGUF_TURBO_PATH"] = str(q4)
    env["GGUF_LLM_PATH"] = str(qwen)
    env["GGUF_VAE_PATH"] = str(vae)
    env["GGUF_REALTIME_PROFILE"] = "turbo_q3_or_q4_512"
    settings.diffusion_engine = "gguf_external"
    settings.gguf_sd_cli_path = runtime["sd_cli_path"]
    settings.gguf_turbo_path = str(q4)
    settings.gguf_llm_path = str(qwen)
    settings.gguf_vae_path = str(vae)
    _write_env(env)
    return {
        "ok": True,
        "assets": results,
        "runtime": runtime,
        "diffusion_engine": "gguf_external",
        "sd_cli_path": runtime["sd_cli_path"],
        "turbo_path": str(q4),
        "realtime_candidate_path": str(q3),
        "llm_path": str(qwen),
        "vae_path": str(vae),
        "sampler": {"sampler": "euler", "scheduler": "simple", "steps": 8, "cfg": 0.0, "mu": 1.15},
        "realtime": {"preview_size": 512, "preview_steps": 4, "final_steps": 8},
        "warnings": ["Realtime GGUF remains disabled until the sidecar benchmark passes, but 512/4/8 are the target low-VRAM live settings."],
    }


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
        "krea2_turbo_int8_path": env.get("KREA2_TURBO_INT8_PATH", settings.krea2_turbo_int8_path),
        "krea2_raw_int8_path": env.get("KREA2_RAW_INT8_PATH", settings.krea2_raw_int8_path),
        "output_dir": env.get("OUTPUT_DIR", str(MODELS_DIR.parent / "outputs")),
        "prompt_expander_backend": env.get("PROMPT_EXPANDER_BACKEND", settings.prompt_expander_backend),
        "local_llm_backend": env.get("LOCAL_LLM_BACKEND", settings.local_llm_backend),
        "local_qwen_model_id": env.get("LOCAL_QWEN_MODEL_ID", settings.local_qwen_model_id),
        "local_qwen_device": env.get("LOCAL_QWEN_DEVICE", settings.local_qwen_device),
        "gguf_helper_base_url": env.get("GGUF_HELPER_BASE_URL", settings.gguf_helper_base_url),
        "gguf_helper_model": env.get("GGUF_HELPER_MODEL", settings.gguf_helper_model),
        "gguf_helper_timeout_sec": int(env.get("GGUF_HELPER_TIMEOUT_SEC", str(settings.gguf_helper_timeout_sec)) or 120),
        "diffusion_engine": env.get("DIFFUSION_ENGINE", settings.diffusion_engine),
        "gguf_sd_cli_path": env.get("GGUF_SD_CLI_PATH", settings.gguf_sd_cli_path),
        "gguf_turbo_path": env.get("GGUF_TURBO_PATH", settings.gguf_turbo_path),
        "gguf_raw_path": env.get("GGUF_RAW_PATH", settings.gguf_raw_path),
        "gguf_llm_path": env.get("GGUF_LLM_PATH", settings.gguf_llm_path),
        "gguf_vae_path": env.get("GGUF_VAE_PATH", settings.gguf_vae_path),
        "gguf_lora_dir": env.get("GGUF_LORA_DIR", settings.gguf_lora_dir),
        "gguf_timeout_sec": int(env.get("GGUF_TIMEOUT_SEC", str(settings.gguf_timeout_sec)) or 600),
        "openrouter_model": env.get("OPENROUTER_MODEL", settings.openrouter_model),
        "openrouter_free_only": env.get("OPENROUTER_FREE_ONLY", str(settings.openrouter_free_only)).lower() in {"1", "true", "yes"},
        "krea_share_auto_funnel": env.get("KREA_SHARE_AUTO_FUNNEL", str(settings.krea_share_auto_funnel)).lower() in {"1", "true", "yes", "on"},
        "krea2_vae_path": env.get("KREA2_VAE_PATH", settings.krea2_vae_path),
        "krea_attention_backend": env.get("KREA_ATTENTION_BACKEND", settings.krea_attention_backend),
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
    if req.krea2_turbo_int8_path is not None:
        env["KREA2_TURBO_INT8_PATH"] = req.krea2_turbo_int8_path
        settings.krea2_turbo_int8_path = req.krea2_turbo_int8_path
    if req.krea2_raw_int8_path is not None:
        env["KREA2_RAW_INT8_PATH"] = req.krea2_raw_int8_path
        settings.krea2_raw_int8_path = req.krea2_raw_int8_path
    if req.output_dir is not None:
        env["OUTPUT_DIR"] = req.output_dir
    if req.prompt_expander_backend is not None:
        env["PROMPT_EXPANDER_BACKEND"] = req.prompt_expander_backend
        settings.prompt_expander_backend = req.prompt_expander_backend
    if req.local_llm_backend is not None:
        env["LOCAL_LLM_BACKEND"] = req.local_llm_backend
        settings.local_llm_backend = req.local_llm_backend
    if req.local_qwen_model_id is not None:
        env["LOCAL_QWEN_MODEL_ID"] = req.local_qwen_model_id
        settings.local_qwen_model_id = req.local_qwen_model_id
    if req.local_qwen_device is not None:
        env["LOCAL_QWEN_DEVICE"] = req.local_qwen_device
        settings.local_qwen_device = req.local_qwen_device
    if req.gguf_helper_base_url is not None:
        env["GGUF_HELPER_BASE_URL"] = req.gguf_helper_base_url
        settings.gguf_helper_base_url = req.gguf_helper_base_url
    if req.gguf_helper_model is not None:
        env["GGUF_HELPER_MODEL"] = req.gguf_helper_model
        settings.gguf_helper_model = req.gguf_helper_model
    if req.gguf_helper_timeout_sec is not None:
        env["GGUF_HELPER_TIMEOUT_SEC"] = str(req.gguf_helper_timeout_sec)
        settings.gguf_helper_timeout_sec = int(req.gguf_helper_timeout_sec)
    if req.diffusion_engine is not None:
        env["DIFFUSION_ENGINE"] = req.diffusion_engine
        settings.diffusion_engine = req.diffusion_engine
    if req.gguf_sd_cli_path is not None:
        env["GGUF_SD_CLI_PATH"] = req.gguf_sd_cli_path
        settings.gguf_sd_cli_path = req.gguf_sd_cli_path
    if req.gguf_turbo_path is not None:
        env["GGUF_TURBO_PATH"] = req.gguf_turbo_path
        settings.gguf_turbo_path = req.gguf_turbo_path
    if req.gguf_raw_path is not None:
        env["GGUF_RAW_PATH"] = req.gguf_raw_path
        settings.gguf_raw_path = req.gguf_raw_path
    if req.gguf_llm_path is not None:
        env["GGUF_LLM_PATH"] = req.gguf_llm_path
        settings.gguf_llm_path = req.gguf_llm_path
    if req.gguf_vae_path is not None:
        env["GGUF_VAE_PATH"] = req.gguf_vae_path
        settings.gguf_vae_path = req.gguf_vae_path
    if req.gguf_lora_dir is not None:
        env["GGUF_LORA_DIR"] = req.gguf_lora_dir
        settings.gguf_lora_dir = req.gguf_lora_dir
    if req.gguf_timeout_sec is not None:
        env["GGUF_TIMEOUT_SEC"] = str(req.gguf_timeout_sec)
        settings.gguf_timeout_sec = int(req.gguf_timeout_sec)
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
    if req.krea_share_auto_funnel is not None:
        env["KREA_SHARE_AUTO_FUNNEL"] = "true" if req.krea_share_auto_funnel else "false"
        settings.krea_share_auto_funnel = req.krea_share_auto_funnel
    if req.krea2_vae_path is not None:
        env["KREA2_VAE_PATH"] = req.krea2_vae_path
        settings.krea2_vae_path = req.krea2_vae_path
    if req.krea_attention_backend is not None:
        env["KREA_ATTENTION_BACKEND"] = req.krea_attention_backend
        settings.krea_attention_backend = req.krea_attention_backend
        try:
            from krea2 import mmdit

            mmdit.KREA_ATTENTION_BACKEND = req.krea_attention_backend
        except Exception:
            logger.debug("Could not update live attention backend")
    _write_env(env)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Prompt expander
# ---------------------------------------------------------------------------

@app.post("/api/expand-prompt", response_model=ExpandPromptResponse)
async def expand_prompt_endpoint(req: ExpandPromptRequest):
    loop = asyncio.get_event_loop()
    backend = req.backend or ("gguf-server" if settings.local_llm_backend == "gguf_server" and settings.prompt_expander_backend == "local" else settings.prompt_expander_backend)
    result = await loop.run_in_executor(
        None,
        lambda: expand_prompt_result(
            req.prompt,
            backend=backend,
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_model=settings.openrouter_model,
            openrouter_free_only=settings.openrouter_free_only,
            ideogram_api_key=settings.ideogram_api_key,
            gguf_helper_base_url=settings.gguf_helper_base_url,
            gguf_helper_model=settings.gguf_helper_model,
            gguf_helper_timeout_sec=settings.gguf_helper_timeout_sec,
        ),
    )
    return ExpandPromptResponse(
        expanded=result.expanded,
        changed=result.changed,
        error=result.error,
        backend=result.backend,
    )


@app.get("/api/resolution-options")
async def resolution_options_endpoint():
    from resolution import resolution_options

    return resolution_options()


@app.get("/api/sampler-catalog")
async def sampler_catalog_endpoint(profile: str = "krea_turbo"):
    from krea2.sampler_registry import sampler_catalog

    return sampler_catalog(profile)


@app.get("/api/engine-catalog")
async def engine_catalog_endpoint():
    from model_profiles import engine_catalog

    return engine_catalog()


@app.post("/api/gguf/helper-test")
async def gguf_helper_test_endpoint():
    result = expand_prompt_result(
        "a small red fox in morning fog",
        backend="gguf-server",
        gguf_helper_base_url=settings.gguf_helper_base_url,
        gguf_helper_model=settings.gguf_helper_model,
        gguf_helper_timeout_sec=settings.gguf_helper_timeout_sec,
    )
    if result.error:
        raise HTTPException(502, result.error)
    return {"ok": True, "backend": result.backend, "expanded": result.expanded}


def _gguf_runtime_settings():
    from gguf_diffusion_provider import GgufRuntimeSettings

    return GgufRuntimeSettings(
        sd_cli_path=settings.gguf_sd_cli_path,
        turbo_path=settings.gguf_turbo_path,
        raw_path=settings.gguf_raw_path,
        llm_path=settings.gguf_llm_path,
        vae_path=settings.gguf_vae_path,
        lora_dir=settings.gguf_lora_dir,
        timeout_sec=settings.gguf_timeout_sec,
    )


def _pid_settings():
    from pid_decoder_provider import PiDSettings
    from quality_assets import asset_by_id

    return PiDSettings(
        decoder_path=str(asset_by_id("pid_qwenimage_decoder").local_path),
        text_encoder_path=str(asset_by_id("pid_gemma_text_encoder").local_path),
        official_checkpoint_path=str(asset_by_id("pid_qwenimage_official_checkpoint").local_path),
        official_vae_path=str(asset_by_id("pid_qwenimage_vae_2d").local_path),
    )


@app.get("/api/gguf/status")
async def gguf_status_endpoint():
    fields = {
        "sd_cli_path": settings.gguf_sd_cli_path,
        "turbo_path": settings.gguf_turbo_path,
        "raw_path": settings.gguf_raw_path,
        "llm_path": settings.gguf_llm_path,
        "vae_path": settings.gguf_vae_path,
        "lora_dir": settings.gguf_lora_dir,
    }
    return {
        "diffusion_engine": settings.diffusion_engine,
        "paths": {
            key: {"path": value, "configured": bool(value)}
            for key, value in fields.items()
        },
    }


@app.get("/api/pid/status")
async def pid_status_endpoint():
    from pid_decoder_provider import pid_status
    from system_check import get_gpu_info

    _name, _total, free = get_gpu_info()
    return pid_status(_pid_settings(), free_vram_gb=free)


@app.post("/api/gguf/test-runtime")
async def gguf_test_runtime_endpoint():
    from gguf_diffusion_provider import build_gguf_command

    req = GenerationRequest(prompt="a small red fox in morning fog", width=512, height=512, steps=4, cfg=0.0)
    try:
        cmd, output = build_gguf_command(req, _gguf_runtime_settings())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "command": cmd, "output": str(output)}


@app.get("/api/int8/status")
async def int8_status_endpoint():
    import importlib.util
    import torch
    from krea2.int8_convrot import inspect_int8_safetensors
    from quality_assets import asset_by_id, asset_status

    def _asset(asset_id: str, configured_path: str) -> dict:
        spec = asset_by_id(asset_id)
        item = asset_status(spec, has_hf_token=bool(settings.hf_token or os.environ.get("HF_TOKEN")))
        path = Path(configured_path or item["local_path"])
        item["configured_path"] = str(path)
        item["installed"] = path.exists()
        if path.exists():
            try:
                item["inspection"] = inspect_int8_safetensors(path)
            except Exception as exc:  # noqa: BLE001
                item["inspection_error"] = str(exc)
        return item

    return {
        "ok": bool(hasattr(torch, "_int_mm")),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_int_mm": bool(hasattr(torch, "_int_mm")),
        "comfy_kitchen": importlib.util.find_spec("comfy_kitchen") is not None,
        "triton": importlib.util.find_spec("triton") is not None,
        "diffusion_engine": settings.diffusion_engine,
        "assets": {
            "turbo": _asset("krea2_turbo_int8_convrot", settings.krea2_turbo_int8_path),
            "raw": _asset("krea2_raw_int8_convrot", settings.krea2_raw_int8_path),
        },
    }


@app.post("/api/int8/setup-native")
async def int8_setup_native_endpoint():
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from quality_assets import asset_by_id, asset_installed, asset_status, download_asset

    required_ids = ["krea2_turbo_int8_convrot"]
    token = settings.hf_token or os.environ.get("HF_TOKEN") or None
    loop = asyncio.get_event_loop()
    results: list[dict] = []
    for asset_id in required_ids:
        spec = asset_by_id(asset_id)
        skipped = asset_installed(spec)
        path = spec.local_path
        if not skipped:
            try:
                path = await loop.run_in_executor(None, lambda spec=spec: download_asset(spec, token=token))
            except (GatedRepoError, HfHubHTTPError) as exc:
                raise HTTPException(502, f"Could not download {asset_id}: {exc}") from exc
            except Exception as exc:
                logger.exception("Native INT8 asset download failed")
                raise HTTPException(502, f"Could not download {asset_id}. Check connection and server logs.") from exc
        results.append({"id": asset_id, "path": str(path), "skipped": skipped, "item": asset_status(spec, has_hf_token=bool(token))})

    turbo = asset_by_id("krea2_turbo_int8_convrot").local_path
    env = _read_env()
    env["DIFFUSION_ENGINE"] = "native_int8_convrot"
    env["KREA2_TURBO_INT8_PATH"] = str(turbo)
    env["KREA2_AUTO_CHECKPOINT"] = str(turbo)
    env["KREA2_AUTO_QUANT"] = "int8"
    settings.diffusion_engine = "native_int8_convrot"
    settings.krea2_turbo_int8_path = str(turbo)
    settings.krea2_auto_checkpoint = str(turbo)
    settings.krea2_auto_quant = "int8"
    _write_env(env)
    return {
        "ok": True,
        "assets": results,
        "diffusion_engine": "native_int8_convrot",
        "turbo_path": str(turbo),
        "quantization": "int8",
        "sampler": {"sampler": "euler", "scheduler": "simple", "steps": 8, "cfg": 0.0, "mu": 1.15},
        "warnings": ["Native INT8 uses torch._int_mm fallback first; comfy_kitchen/Triton are optional future accelerators, not required."],
    }


@app.post("/api/gguf/benchmark-quick")
async def gguf_benchmark_quick_endpoint():
    from types import SimpleNamespace
    from gguf_diffusion_provider import generate_gguf_external

    req = SimpleNamespace(
        prompt="a red fox in morning fog, cinematic lighting",
        negative_prompt="blurry",
        checkpoint="turbo",
        width=512,
        height=512,
        steps=4,
        cfg=0.0,
        seed=123,
        mode="txt2img",
    )
    started = time.time()
    try:
        _images, seed, filenames, _lora_reports, metadata = await asyncio.get_event_loop().run_in_executor(
            None, lambda: generate_gguf_external(req, _gguf_runtime_settings())
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    elapsed = time.time() - started
    return {
        "ok": True,
        "elapsed_sec": round(elapsed, 2),
        "preview_size": 512,
        "preview_steps": 4,
        "final_steps": 8,
        "seed": seed,
        "filenames": filenames,
        "output": metadata[0]["filename"] if metadata else "",
        "live_candidate": False,
        "speed_candidate": elapsed <= 30.0,
        "message": "GGUF 512/4 speed benchmark completed, but visual prompt-adherence sweep did not meet native quality. Keep GGUF experimental and txt2img-only.",
    }


@app.get("/api/runtime-advice")
async def runtime_advice_endpoint(width: int = 1024, height: int = 1024, quantization: str = "fp8"):
    from resource_manager import recommend_runtime
    from system_check import get_gpu_info

    _name, _total, free = get_gpu_info()
    advice = recommend_runtime(free_vram_gb=free, width=width, height=height, quantization=quantization)
    advice["free_vram_gb"] = round(free, 1) if free is not None else None
    return advice


@app.get("/api/accelerators/status")
async def accelerators_status_endpoint():
    from krea2.performance_guard import accelerator_status

    return accelerator_status()


async def _pip_install_accelerator(*packages: str) -> dict:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pip",
        "install",
        *packages,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output_b, _ = await proc.communicate()
    output = output_b.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise HTTPException(500, output[-2000:] or "Accelerator install failed.")
    from krea2.performance_guard import accelerator_status

    return {"ok": True, "status": accelerator_status(), "message": output[-2000:]}


@app.post("/api/accelerators/install-triton-windows")
async def install_triton_windows_endpoint():
    return await _pip_install_accelerator("triton-windows<3.7")


@app.post("/api/accelerators/install-sageattention")
async def install_sageattention_endpoint():
    return await _pip_install_accelerator("sageattention")


@app.get("/api/batch/plan")
async def batch_plan_endpoint(
    width: int = 1024,
    height: int = 1024,
    quantization: str = "fp8",
    batch: int = 1,
    cfg: float = 0.0,
    mode: str = "txt2img",
    checkpoint: str = "turbo",
):
    from resource_manager import plan_parallel_batch
    from system_check import get_gpu_info

    _name, _total, free = get_gpu_info()
    plan = plan_parallel_batch(
        free_vram_gb=free,
        width=width,
        height=height,
        quantization=quantization,
        batch=batch,
        cfg_active=float(cfg) > 0,
        mode=mode,
        checkpoint=checkpoint,
    )
    plan["free_vram_gb"] = round(free, 1) if free is not None else None
    return plan


@app.get("/api/prompting-guide")
async def prompting_guide_endpoint():
    from prompting_guide import prompting_guide_payload

    return prompting_guide_payload()


@app.post("/api/plan-prompt", response_model=PlanPromptResponse)
async def plan_prompt_endpoint(req: PlanPromptRequest):
    loop = asyncio.get_event_loop()
    helper_backend = "gguf-server" if settings.local_llm_backend == "gguf_server" else "local"
    result = await loop.run_in_executor(
        None,
        lambda: plan_prompt(
            req.prompt,
            enabled=True,
            max_tokens=req.max_tokens,
            backend=helper_backend,
            gguf_helper_base_url=settings.gguf_helper_base_url,
            gguf_helper_model=settings.gguf_helper_model,
            gguf_helper_timeout_sec=settings.gguf_helper_timeout_sec,
        ),
    )
    return PlanPromptResponse(**result.model_dump())


@app.get("/api/prompt-recipes", response_model=PromptRecipeListResponse)
async def prompt_recipes_list_endpoint():
    return PromptRecipeListResponse(items=[PromptRecipe(**item) for item in list_recipes()])


@app.post("/api/prompt-recipes", response_model=PromptRecipe)
async def prompt_recipes_save_endpoint(req: PromptRecipe):
    return PromptRecipe(**save_recipe(req.model_dump()))


@app.delete("/api/prompt-recipes/{recipe_id}")
async def prompt_recipes_delete_endpoint(recipe_id: str):
    return {"ok": delete_recipe(recipe_id)}


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
