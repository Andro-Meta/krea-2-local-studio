"""Krea 2 Studio FastAPI server."""
from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import logging
import os
import re
import secrets
import subprocess
import sys
import time
import uuid
from contextlib import suppress
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

from crash_reporter import archive_stale_generation_breadcrumbs, clear_generation_breadcrumb, disable_fault_logging, enable_fault_logging, stale_generation_breadcrumbs, write_generation_breadcrumb
from gallery import (
    delete_image,
    delete_media_record,
    get_gallery,
    get_image_record_by_filename,
    init_db,
    save_image,
    save_media,
    set_favorite,
)
from gpu_task_queue import BACKGROUND, INTERACTIVE, EnqueueResult, GpuTaskQueue
from gpu_recovery import is_cuda_oom
from gpu_tasks import (
    ANIMATION,
    BACKGROUND_ENRICHMENT,
    DEPTH_PREVIEW,
    GENERATION,
    HELPER_BENCHMARK,
    IMAGE_DESCRIBE,
    MODEL_WARMUP,
    MOODBOARD_GUIDANCE,
    PROMPT_EXPAND,
    PROMPT_PLAN,
    UPSCALE,
    foreign_summary,
)
# Phase 1: native in-process DiT pipeline is deprecated. Keep a stub so memory /
# system-report call sites stay intact without importing torch + krea2 at startup.
from comfy_pipeline_stub import pipeline
from log_setup import setup_logging
from lora_manager import inspect_lora, list_loras
from moodboards_catalog import (
    CUSTOM_MOODBOARD_DIR,
    KREA_MOODBOARD_GALLERY_URL,
    MOODBOARD_SEED_PATH,
    create_custom_moodboard,
    cleanup_prepared_moodboard_task,
    commit_prepared_moodboard_task,
    delete_custom_moodboard,
    export_moodboard_seed,
    fetch_moodboard_image_b64,
    get_moodboard,
    fetch_cached_moodboard_image,
    import_moodboard_urls,
    init_moodboard_db,
    latest_moodboard_discovery,
    list_moodboards,
    prepare_moodboard_guidance_task,
    reconcile_custom_moodboard_storage,
    set_moodboard_favorite,
    should_sync_moodboards,
    suggest_moodboards,
)
from prompt_expander import describe_image_local, describe_image_openrouter, expand_prompt_result
from prompt_planner import plan_prompt
from prompt_recipes import delete_recipe, list_recipes, save_recipe
from settings_env import read_env, secret_value, write_env
from schemas import (
    AnimateRequest,
    AnimationResult,
    AutoMaskRequest,
    DescribeImageRequest,
    DepthPreviewRequest,
    ExpandPromptRequest,
    FavoriteRequest,
    GenerationRequest,
    HelperBenchmarkRequest,
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
    PromptRecipe,
    PromptRecipeListResponse,
    MoodboardItem,
    SettingsUpdate,
    ShareLoginRequest,
    ShareUserCreateRequest,
    ShareUserPasswordRequest,
    ShareUserRoleRequest,
    LoraImportRequest,
    UpscaleRequest,
    PreprocessorPreviewRequest,
)
from settings import BASE_DIR, DIST_DIR, LOGS_DIR, LORAS_DIR, MODELS_DIR, OUTPUTS_DIR, settings
from share_auth import (
    BootstrapCredentialDeletionError,
    add_user,
    get_user_role,
    is_admin,
    is_valid_username,
    list_user_records,
    remove_user,
    resolve_bootstrap_credential_path,
    set_user_role,
    verify_login,
)
from runtime_hardening import FunnelHealthMonitor, auto_repair_configured, funnel_interval_healthy
from support_models import download_support_models, support_model_status
from comfy_config import use_comfy_backend
from comfy_client import (
    comfy_available,
    comfy_atomic_cancel_available,
    free_comfy_vram,
)
from comfy_deforum import status as krea_deforum_status
from sharing_service import PUBLIC_PATH as SHARING_PUBLIC_PATH, funnel_status, repair_funnel, start_funnel, stop_funnel, tailscale_status, tailscale_up
from security_utils import append_query_param, is_civitai_url, normalize_lora_import_url, safe_lora_filename
from system_check import get_system_report
from memory_manager import (
    clear_cuda_cache,
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
from animation_state import AnimationProject, AnimationStore
from animation_uploads import (
    ALLOWED_VIDEO_TYPES,
    AnimationUploadStore,
    UploadQuotaError,
)
from video_output import finalize_mp4

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
BOOTSTRAP_CREDENTIAL_FILE = resolve_bootstrap_credential_path(BASE_DIR)
_funnel_health_monitor = FunnelHealthMonitor(
    enabled=auto_repair_configured(os.environ.get("KREA_SHARE_AUTO_FUNNEL_ENABLED"))
)
SHARE_COOKIE = "krea_share_session"
PUBLIC_ANONYMOUS_USERNAME = ":public-anonymous:"
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


def _public_moodboard_username(request: Request) -> str:
    """Resolve optional identity on an auth-exempt catalog request."""
    if not SHARE_AUTH_ENABLED:
        return "__local__"
    state = getattr(request, "state", None)
    username = getattr(state, "share_user", None) if state is not None else None
    if username:
        return str(username)
    cookies = getattr(request, "cookies", None) or {}
    return (
        _auth_username_from_cookie(cookies.get(SHARE_COOKIE))
        or PUBLIC_ANONYMOUS_USERNAME
    )


# Last time each authenticated user was seen making a request (for admin-only
# "who's online" presence). Updated in the auth middleware.
_USER_LAST_SEEN: dict[str, float] = {}
_ONLINE_WINDOW_SEC = 120.0


def _users_with_presence() -> list[dict]:
    """Admin-only user list augmented with online/active/last_seen. 'active' means
    the user currently has a queued or running generation."""
    records = list_user_records(SHARE_AUTH_FILE)
    now = time.time()
    active_users: set[str] = set()
    if generation_queue is not None:
        try:
            for rec in generation_queue.all_statuses().values():
                if rec.get("status") in ("queued", "running") and rec.get("username"):
                    active_users.add(str(rec["username"]))
        except Exception:
            pass
    out: list[dict] = []
    for r in records:
        u = str(r.get("username") or "")
        seen = _USER_LAST_SEEN.get(u)
        rr = dict(r)
        rr["last_seen"] = seen
        rr["online"] = bool(seen and (now - seen) <= _ONLINE_WINDOW_SEC)
        rr["active"] = u in active_users
        out.append(rr)
    return out


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
    # Heavy setup/download operations: multi-GB downloads and .env rewrites
    # must not be triggerable by shared (non-admin) users.
    if path == "/api/civitai/install":
        return True
    if path == "/api/huggingface/install":
        return True
    if path in {"/api/xperiment/setup", "/api/gguf/setup-low-vram", "/api/int8/setup-native"}:
        return True
    if path.startswith("/api/quality-assets/") and method != "GET":
        return True
    # Custom moodboards are a shared library with no per-user ownership yet:
    # deleting them is destructive for everyone, so restrict it to admins.
    if path.startswith("/api/moodboards/custom/") and method == "DELETE":
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
        _USER_LAST_SEEN[user] = time.time()
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
    try:
        authenticated = is_valid_username(username) and verify_login(
            SHARE_AUTH_FILE,
            username,
            req.password,
            bootstrap_credential_path=BOOTSTRAP_CREDENTIAL_FILE,
        )
    except BootstrapCredentialDeletionError as exc:
        logger.error("Bootstrap credential deletion failed; session issuance denied.")
        raise HTTPException(
            status_code=500,
            detail="Could not remove the one-time bootstrap credential. Login was not completed; retry.",
        ) from exc
    if not authenticated:
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
    return {"users": _users_with_presence()}


@app.post("/api/admin/users")
async def admin_add_user(req: ShareUserCreateRequest):
    try:
        add_user(SHARE_AUTH_FILE, req.username, req.password, role=req.role)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "users": _users_with_presence()}


@app.put("/api/admin/users/{username}/role")
async def admin_set_user_role(username: str, req: ShareUserRoleRequest):
    try:
        if not set_user_role(SHARE_AUTH_FILE, username, req.role):
            raise HTTPException(404, "User not found")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "users": _users_with_presence()}


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
    return {"ok": True, "users": _users_with_presence()}


@app.get("/api/sharing/status")
async def sharing_status():
    return {"tailscale": tailscale_status(), "funnel": funnel_status(), "public_path": SHARING_PUBLIC_PATH}


@app.post("/api/sharing/tailscale-up")
async def sharing_tailscale_up():
    return tailscale_up()


@app.post("/api/sharing/funnel/start")
async def sharing_funnel_start():
    _funnel_health_monitor.enable()
    result = start_funnel()
    if not result.get("ok"):
        raise HTTPException(502, result.get("message", "Tailscale Funnel failed to start."))
    return result


@app.post("/api/sharing/funnel/repair")
async def sharing_funnel_repair():
    _funnel_health_monitor.enable()
    try:
        return repair_funnel()
    except Exception:
        logger.exception("Sharing Funnel repair failed")
        raise HTTPException(
            status_code=502,
            detail="Sharing repair failed. Check the server logs and retry.",
        )


@app.post("/api/sharing/funnel/stop")
async def sharing_funnel_stop():
    _funnel_health_monitor.disable()
    return stop_funnel()

# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_JOBS_MAX = 200  # ponytail: simple FIFO cap; raise if clients poll very old jobs
_TERMINAL_JOB_STATUSES = {"done", "error", "blocked", "cancelled"}
generation_queue: GpuTaskQueue | None = None
animation_store = AnimationStore(
    Path(settings.animation_state_root), OUTPUTS_DIR
)
animation_upload_store = AnimationUploadStore(
    Path(settings.animation_upload_root),
    max_bytes=settings.animation_max_upload_bytes,
    max_frames=settings.animation_max_frames,
    max_dimension=settings.animation_max_dimension,
    max_duration=settings.animation_max_source_duration_seconds,
    ttl_seconds=settings.animation_upload_ttl_seconds,
    max_user_uploads=settings.animation_uploads_per_user,
    max_user_bytes=settings.animation_upload_bytes_per_user,
    max_global_uploads=settings.animation_uploads_global,
    max_global_bytes=settings.animation_upload_bytes_global,
)
_animation_finalizer_tasks: set[asyncio.Task] = set()
_animation_finalizer_project_ids: set[tuple[str, str]] = set()
_animation_admission_reservations: set[str] = set()
_animation_recovery_guard: set[tuple[str, str]] = set()
_animation_recovery_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
_animation_upload_cleanup_task: asyncio.Task | None = None
_model_warmup_job_id: str | None = None
_last_model_signature: dict | None = None
_last_warm_state: dict = {"status": "never", "updated_at": None}

# Defense-in-depth GPU lease held by the unified queue worker around every
# dispatch. Endpoints never acquire it, preventing nested-lease deadlocks.
GPU_LEASE = asyncio.Lock()


def _job_owned_by(job: dict, username: str | None, is_admin: bool) -> bool:
    """True when the requester may see/control this job.

    Local mode (share auth off) has a single implicit owner; admins can manage
    everything; users match on the username recorded at enqueue time.
    """
    if not SHARE_AUTH_ENABLED or is_admin:
        return True
    return (job.get("username") or None) == (username or None)


def _new_job(
    username: str | None = None,
    role: str = "admin",
    *,
    task_kind: str = GENERATION,
    summary: str = "",
) -> str:
    jid = uuid.uuid4().hex
    now = time.time()
    _jobs[jid] = {
        "status": "queued",
        "progress": 0,
        "images": [],
        "error": None,
        "seed": None,
        "username": username,
        "role": role,
        "task_kind": task_kind,
        "priority_class": INTERACTIVE,
        "summary": summary,
        "result": None,
        "comfy_prompt_id": None,
        "queued_at": now,
        "started_at": None,
        "finished_at": None,
        "moderation_event_id": None,
    }
    # Evict only explicitly terminal records. Unknown/future phases are
    # conservatively non-terminal, and unfinished batches retain all members.
    if len(_jobs) > _JOBS_MAX:
        for old_id, old_job in list(_jobs.items()):
            if len(_jobs) <= _JOBS_MAX or old_id == jid:
                break
            if old_job.get("status") not in _TERMINAL_JOB_STATUSES:
                continue
            child_ids = list(old_job.get("child_job_ids") or [])
            if child_ids and any(
                (_jobs.get(str(child_id)) or {}).get("status")
                not in _TERMINAL_JOB_STATUSES
                for child_id in child_ids
            ):
                continue
            parent_id = old_job.get("parent_job_id")
            if parent_id:
                parent = _jobs.get(str(parent_id))
                if parent and (
                    parent.get("status") not in _TERMINAL_JOB_STATUSES
                    or any(
                        (_jobs.get(str(child_id)) or {}).get("status")
                        not in _TERMINAL_JOB_STATUSES
                        for child_id in list(parent.get("child_job_ids") or [])
                    )
                ):
                    continue
            del _jobs[old_id]
    return jid


def _job_summary(req: "GenerationRequest") -> str:
    """Short human label for the queue list, e.g. 'RAW · 2048×2048 · Mr.Flow ×2'."""
    try:
        w = int(getattr(req, "width", 0) or 0)
        h = int(getattr(req, "height", 0) or 0)
        ck = (getattr(req, "checkpoint", "turbo") or "turbo").lower()
        prof = (getattr(req, "model_profile", "") or "").lower()
        is_raw = ck == "raw" or prof == "krea_raw"
        parts = ["RAW" if is_raw else "Turbo", f"{w}×{h}"]
        if bool(getattr(req, "god_mode", False)):
            parts.append("God Mode ✨→4K")
        elif bool(getattr(req, "mrflow", False)):
            parts.append("Mr.Flow ×4" if getattr(req, "mrflow_upscaler", "") == "remacri_x4" else "Mr.Flow ×2")
        elif is_raw and max(w, h) >= 2560:
            parts.append("SeedVR2→4K")
        mode = (getattr(req, "mode", "txt2img") or "txt2img").lower()
        if mode not in ("txt2img", "redraw"):
            parts.append(mode)
        if bool(getattr(req, "depth_control", False)):
            parts.append("depth")
        n = int(getattr(req, "num_images", 1) or 1)
        if n > 1:
            parts.append(f"×{n}")
        return " · ".join(parts)
    except Exception:
        return ""


def _job_thumb(data_url: str, size: int = 160) -> str:
    """Small JPEG data URL for the queue list (computed once, cached on the job)."""
    try:
        raw = base64.b64decode((data_url or "").split(",")[-1])
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


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


def int8_batch_eligible(req: GenerationRequest) -> bool:
    """The 'batch all Turbo INT8 models' sweep only makes sense for a Turbo INT8
    ConvRot config (the only path where turbo_int8_variant swaps the checkpoint)."""
    if not bool(getattr(req, "batch_int8_all", False)):
        return False
    checkpoint = str(getattr(req, "checkpoint", "") or "").lower()
    quant = str(getattr(req, "quantization", "") or "").lower()
    return checkpoint == "turbo" and "int8" in quant


def build_int8_variant_children(req: GenerationRequest) -> list[GenerationRequest]:
    """Fan the request out across every Turbo INT8 ConvRot checkpoint. Produces
    ``num_images`` copies per variant, all sharing the same base seed so image N
    is directly comparable across models (same settings, only the checkpoint swaps)."""
    from comfy_workflows import _TURBO_INT8_VARIANTS
    variants = list(_TURBO_INT8_VARIANTS.keys())
    count = max(1, int(getattr(req, "num_images", 1) or 1))
    base_seed = int(getattr(req, "seed", -1))
    if base_seed < 0:
        base_seed = secrets.randbelow(2**31 - 1)
    children: list[GenerationRequest] = []
    for variant in variants:
        for index in range(count):
            child = req.model_copy(deep=True)
            child.num_images = 1
            child.seed = base_seed + index
            child.batch_mode = "safe_queue"
            child.parallel_batch_confirmed = False
            child.turbo_int8_variant = variant
            child.batch_int8_all = False  # children are concrete single renders
            children.append(child)
    return children


def _capacity_error(result: EnqueueResult) -> HTTPException:
    return HTTPException(
        429,
        {
            "message": "GPU task capacity reached.",
            "reason": result.reason,
            "limit": result.limit,
            "active_count": result.active_count,
        },
    )


def _enqueue_gpu_task(
    job_id: str,
    payload: dict,
    *,
    username: str | None,
    role: str | None,
    task_kind: str = GENERATION,
    priority_class: str = INTERACTIVE,
) -> EnqueueResult:
    if generation_queue is None:
        _jobs.pop(job_id, None)
        raise HTTPException(503, "Generation queue is not ready yet.")
    result = generation_queue.enqueue(
        job_id,
        payload,
        username=username,
        role=role,
        task_kind=task_kind,
        priority_class=priority_class,
    )
    if not result.accepted:
        _jobs.pop(job_id, None)
        raise _capacity_error(result)
    return result


async def _preempt_model_warmup() -> None:
    """Yield the queue to interactive work without a process-wide interrupt."""
    if generation_queue is None or not _model_warmup_job_id:
        return
    warmup_id = _model_warmup_job_id
    state = generation_queue.status(warmup_id)
    if state.get("status") == "queued":
        generation_queue.request_cancel(warmup_id)
        job = _jobs.get(warmup_id)
        if job is not None:
            job["status"] = "cancelled"
            job["finished_at"] = time.time()
        logger.info("Krea warmup transition: cancelled before start")
        return
    if state.get("status") != "running":
        return

    generation_queue.request_cancel(warmup_id)
    job = _jobs.get(warmup_id)
    if job is None:
        return
    job["status"] = "cancellation_requested"
    job["cancel_requested"] = True
    logger.info("Krea warmup transition: cancellation requested")
    prompt_id = job.get("comfy_prompt_id")
    if prompt_id:
        from comfy_client import cancel_prompt

        job["cancellation_dispatching"] = True
        try:
            job["cancellation_dispatched"] = await asyncio.to_thread(
                cancel_prompt, prompt_id
            )
        finally:
            job["cancellation_dispatching"] = False


async def _enqueue_interactive_gpu_task(
    job_id: str,
    payload: dict,
    *,
    username: str | None,
    role: str | None,
    task_kind: str = GENERATION,
) -> EnqueueResult:
    """Async admission boundary where an optional warmup may be preempted."""
    await _preempt_model_warmup()
    return _enqueue_gpu_task(
        job_id,
        payload,
        username=username,
        role=role,
        task_kind=task_kind,
        priority_class=INTERACTIVE,
    )


async def _enqueue_helper_task(
    request: Request,
    *,
    task_kind: str,
    summary: str,
    payload: dict,
) -> dict:
    username, role, _is_admin = _request_user_role(request)
    job_id = _new_job(
        username=username,
        role=role,
        task_kind=task_kind,
        summary=summary,
    )
    queue_state = await _enqueue_interactive_gpu_task(
        job_id,
        payload,
        username=username,
        role=role,
        task_kind=task_kind,
    )
    _sync_queue_state_to_jobs()
    await ws_manager.broadcast(job_id, {"type": "queue", **_jobs[job_id]})
    return {
        "job_id": job_id,
        "status": "queued",
        "task_kind": task_kind,
        "queue_position": queue_state.get("queue_position"),
        "queue_length": queue_state.get("queue_length"),
    }


def _animation_summary(req: AnimateRequest) -> str:
    return f"Animation · {req.width}×{req.height} · {req.total_frames} frames"


def _animation_parent_from_project(project: AnimationProject) -> dict:
    return {
        "status": project.status,
        "progress": int(100 * project.completed_frames / project.total_frames),
        "images": [],
        "error": project.error or None,
        "seed": project.seed_base,
        "username": project.owner,
        "role": project.role,
        "task_kind": ANIMATION,
        "priority_class": INTERACTIVE,
        "summary": (
            f"Animation · {project.request['width']}×{project.request['height']} "
            f"· {project.total_frames} frames"
        ),
        "result": None,
        "comfy_prompt_id": None,
        "queued_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "moderation_event_id": None,
        "project_job_id": project.job_id,
        "child_job_ids": [],
        "completed_frames": project.completed_frames,
        "total_frames": project.total_frames,
    }


async def _enqueue_animation_chunk(parent_id: str, chunk_index: int) -> str:
    project = await asyncio.to_thread(animation_store.load, parent_id)
    parent = _jobs[parent_id]
    child_id = _new_job(
        username=project.owner,
        role=project.role,
        task_kind=ANIMATION,
        summary=f"Animation chunk {chunk_index + 1}/{len(project.chunk_ranges)}",
    )
    child = _jobs[child_id]
    child.update(
        {
            "parent_job_id": parent_id,
            "operation": "chunk",
            "chunk_index": chunk_index,
        }
    )
    children = list(parent.get("child_job_ids") or [])
    children.append(child_id)
    parent["child_job_ids"] = children[-4:]
    retained = set(parent["child_job_ids"])
    for old_id in children[:-4]:
        old = _jobs.get(old_id)
        if old and old.get("status") in _TERMINAL_JOB_STATUSES and old_id not in retained:
            _jobs.pop(old_id, None)
    try:
        await _enqueue_interactive_gpu_task(
            child_id,
            {"operation": "chunk", "parent_job_id": parent_id, "chunk_index": chunk_index},
            username=project.owner,
            role=project.role,
            task_kind=ANIMATION,
        )
    except Exception:
        parent["child_job_ids"] = [
            item for item in parent["child_job_ids"] if item != child_id
        ]
        raise
    _sync_queue_state_to_jobs()
    active = _jobs.get(child_id, {})
    parent["status"] = "queued"
    parent["queue_position"] = active.get("queue_position")
    parent["queue_length"] = active.get("queue_length")
    return child_id


async def _create_animation_impl(
    req: AnimateRequest, *, username: str | None, role: str
) -> dict:
    if generation_queue is None:
        raise HTTPException(503, "Generation queue is not ready yet.")
    if req.total_frames > settings.animation_max_frames or max(
        req.width, req.height
    ) > settings.animation_max_dimension:
        raise HTTPException(422, "Animation exceeds configured limits.")
    if await asyncio.to_thread(animation_store.active_for_owner, username):
        raise HTTPException(409, "An active animation already exists for this user.")
    status = await asyncio.to_thread(krea_deforum_status, timeout=2.0)
    if not status.get("available"):
        raise HTTPException(503, "KreaDeforum animation runtime is unavailable.")
    if req.animation_mode == "3D" and not status.get("midas_ready"):
        raise HTTPException(
            503,
            str(status.get("midas_reason") or (
                "MiDaS 3D setup is incomplete. Run install.bat, then restart ComfyUI."
            )),
        )
    if not comfy_atomic_cancel_available():
        raise HTTPException(503, "Atomic ComfyUI cancellation is unavailable.")
    from comfy_workflows import VALID_SAMPLERS

    preferred = "er_sde" if "er_sde" in VALID_SAMPLERS else "euler"
    if req.sampler_name not in VALID_SAMPLERS:
        req = req.model_copy(update={"sampler_name": preferred})
    decision = moderate_prompt(req.prompt_schedule, req.negative_prompt, role=role)
    if not decision.allowed:
        raise HTTPException(403, "This prompt was blocked by the child safety filter.")
    if req.source_video_upload_id and req.animation_mode != "Video Input":
        raise HTTPException(
            422, "A source video upload is valid only for Video Input animation."
        )
    if req.source_video_upload_id:
        try:
            await asyncio.to_thread(
                animation_upload_store.resolve,
                req.source_video_upload_id,
                username=username,
                is_admin=role == "admin",
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, "Animation upload not found.") from exc
    if req.init_image_b64:
        try:
            from comfy_deforum import _decode_init_png

            normalized = await asyncio.to_thread(
                _decode_init_png, req.init_image_b64, req.width, req.height
            )
        except Exception as exc:
            raise HTTPException(422, "The starting image is invalid.") from exc
        if role == "child":
            try:
                image = await asyncio.to_thread(
                    lambda: Image.open(io.BytesIO(normalized)).convert("RGB")
                )
                image_decision = await asyncio.to_thread(
                    moderate_images, [image], role="child"
                )
            except Exception as exc:
                raise HTTPException(
                    403, "The starting image could not be safely verified."
                ) from exc
            if not image_decision.allowed:
                raise HTTPException(
                    403, "The starting image was blocked by the child safety filter."
                )

    parent_id = _new_job(
        username=username,
        role=role,
        task_kind=ANIMATION,
        summary=_animation_summary(req),
    )
    parent = _jobs[parent_id]
    parent.update(
        {
            "project_job_id": parent_id,
            "child_job_ids": [],
            "total_frames": req.total_frames,
            "completed_frames": 0,
        }
    )
    created = False
    try:
        await asyncio.to_thread(
            animation_store.create,
            req,
            owner=username,
            role=role,
            job_id=parent_id,
        )
        created = True
        child_id = await _enqueue_animation_chunk(parent_id, 0)
    except Exception as exc:
        _jobs.pop(parent_id, None)
        if created:
            await asyncio.to_thread(
                animation_store.delete,
                parent_id,
                username=username or "",
                is_admin=True,
            )
        if req.source_video_upload_id:
            await asyncio.to_thread(
                animation_upload_store.delete,
                req.source_video_upload_id,
                username=username,
                is_admin=True,
            )
        if "active animation" in str(exc).lower():
            raise HTTPException(
                409, "An active animation already exists for this user."
            ) from exc
        raise
    active = _jobs[child_id]
    return {
        "job_id": parent_id,
        "status": "queued",
        "queue_position": active.get("queue_position"),
        "queue_length": active.get("queue_length"),
    }


async def _create_animation(
    req: AnimateRequest, *, username: str | None, role: str
) -> dict:
    admission_key = username or ":local:"
    if admission_key in _animation_admission_reservations:
        raise HTTPException(409, "An active animation already exists for this user.")
    _animation_admission_reservations.add(admission_key)
    try:
        return await _create_animation_impl(req, username=username, role=role)
    finally:
        _animation_admission_reservations.discard(admission_key)


async def _continue_animation(parent_id: str) -> None:
    project = await asyncio.to_thread(animation_store.load, parent_id)
    parent = _jobs.get(parent_id)
    if parent is None or project.status in {"cancelled", "blocked", "error", "done"}:
        return
    parent["completed_frames"] = project.completed_frames
    parent["progress"] = int(100 * project.completed_frames / project.total_frames)
    if project.status == "finalizing":
        parent["status"] = "finalizing"
        _schedule_animation_finalizer(parent_id)
        return
    active_children = [
        _jobs.get(child_id)
        for child_id in parent.get("child_job_ids") or []
        if (_jobs.get(child_id) or {}).get("status")
        not in _TERMINAL_JOB_STATUSES
    ]
    if not active_children:
        await _enqueue_animation_chunk(parent_id, project.next_chunk_index)


def _enqueue_background_enrichment() -> str | None:
    """Queue one idle enrichment batch unless an equivalent task is unfinished."""
    if generation_queue is None:
        return None
    if any(
        job.get("task_kind") == BACKGROUND_ENRICHMENT
        and job.get("status") not in _TERMINAL_JOB_STATUSES
        for job in _jobs.values()
    ):
        return None
    job_id = _new_job(
        role="admin",
        task_kind=BACKGROUND_ENRICHMENT,
        summary="Background moodboard enrichment",
    )
    job = _jobs[job_id]
    job["priority_class"] = BACKGROUND
    job["operation"] = "missing"
    try:
        _enqueue_gpu_task(
            job_id,
            {"operation": "missing", "limit": 1},
            username=None,
            role="admin",
            task_kind=BACKGROUND_ENRICHMENT,
            priority_class=BACKGROUND,
        )
    except HTTPException:
        return None
    _sync_queue_state_to_jobs()
    return job_id


def _enqueue_model_warmup(*, force: bool = False) -> str | None:
    """Queue the single opt-in startup warmup at background priority."""
    global _model_warmup_job_id
    if not getattr(settings, "krea_comfy_warmup", False) or generation_queue is None:
        return None
    if _model_warmup_job_id is not None:
        existing = _jobs.get(_model_warmup_job_id) or {}
        try:
            queue_status = generation_queue.status(_model_warmup_job_id)
        except Exception:
            queue_status = {}
        state = queue_status.get("status") or existing.get("status")
        if not force or state not in _TERMINAL_JOB_STATUSES:
            return None
        _model_warmup_job_id = None
    job_id = _new_job(
        role="admin",
        task_kind=MODEL_WARMUP,
        summary="Krea model warmup",
    )
    _model_warmup_job_id = job_id
    job = _jobs[job_id]
    job["priority_class"] = BACKGROUND
    job["diagnostic_only"] = True
    try:
        _enqueue_gpu_task(
            job_id,
            {},
            username=None,
            role="admin",
            task_kind=MODEL_WARMUP,
            priority_class=BACKGROUND,
        )
    except HTTPException:
        _model_warmup_job_id = None
        return None
    _sync_queue_state_to_jobs()
    logger.info("Krea warmup transition: queued")
    return job_id


def _refresh_parent_batch_job(parent_job_id: str) -> dict | None:
    parent = _jobs.get(parent_job_id)
    if not parent or not parent.get("child_job_ids"):
        return parent
    _sync_queue_state_to_jobs()
    child_ids = list(parent.get("child_job_ids") or [])
    children = [_jobs.get(child_id) for child_id in child_ids]
    terminal_parent_status = (
        parent.get("status")
        if parent.get("status") in _TERMINAL_JOB_STATUSES
        else None
    )
    done_children = [child for child in children if child and child.get("status") == "done"]
    blocked = next((child for child in children if child and child.get("status") == "blocked"), None)
    errored = next((child for child in children if child and child.get("status") == "error"), None)
    statuses = [child.get("status") if child else "queued" for child in children]

    parent["completed_count"] = len(done_children)
    if not parent.get("result_acknowledged_at"):
        parent["images"] = [child.get("images", [""])[0] for child in done_children if child.get("images")]
        parent["metadata"] = [child.get("metadata", [{}])[0] for child in done_children if child.get("metadata")]
        parent["num_images"] = len(parent["images"])
    else:
        parent["images"] = []
    parent["progress"] = int(len(done_children) / max(len(child_ids), 1) * 100)
    queued_positions = [
        child.get("queue_position")
        for child in children
        if child and child.get("queue_position") is not None
    ]
    queue_lengths = [
        child.get("queue_length")
        for child in children
        if child and child.get("status") not in _TERMINAL_JOB_STATUSES
        and child.get("queue_length") is not None
    ]
    parent["queue_position"] = min(queued_positions, default=None)
    parent["queue_length"] = max(queue_lengths, default=0)
    all_terminal = bool(statuses) and all(
        status in _TERMINAL_JOB_STATUSES for status in statuses
    )

    if terminal_parent_status:
        parent["status"] = terminal_parent_status
    elif not all_terminal:
        parent.pop("error", None)
        if any(status == "running" for status in statuses):
            parent["status"] = "running"
        elif any(status == "finalizing" for status in statuses):
            parent["status"] = "finalizing"
        elif any(status == "cancellation_requested" for status in statuses):
            parent["status"] = "running"
        else:
            parent["status"] = "queued"
    elif blocked:
        parent["status"] = "blocked"
        parent["error"] = blocked.get("error")
    elif errored:
        parent["status"] = "error"
        parent["error"] = errored.get("error")
    elif statuses and all(status == "done" for status in statuses):
        parent["status"] = "done"
        parent["progress"] = 100
    elif (
        statuses
        and all(status in {"done", "cancelled"} for status in statuses)
        and any(status == "cancelled" for status in statuses)
    ):
        parent["status"] = "cancelled"
    elif (
        parent.get("cancel_requested")
        and all_terminal
    ):
        parent["status"] = "cancelled"
    else:
        parent["status"] = "cancelled"
    if parent["status"] in _TERMINAL_JOB_STATUSES and not parent.get("finished_at"):
        child_finishes = [child.get("finished_at") for child in children if child and child.get("finished_at")]
        parent["finished_at"] = max(child_finishes) if child_finishes else time.time()
    return parent


async def _enqueue_batch_children(
    children: list[GenerationRequest],
    batch_meta: dict,
    username: str | None,
    role: str | None,
    summary: str,
) -> dict:
    """Enqueue a list of single-image child requests under a parent batch job and
    return the parent queue payload. Shared by safe-queue and INT8-sweep batches."""
    if generation_queue is None:
        raise HTTPException(503, "Generation queue is not ready yet.")
    await _preempt_model_warmup()
    capacity = generation_queue.check_capacity(
        username, INTERACTIVE, len(children)
    )
    if not capacity.accepted:
        raise _capacity_error(capacity)

    job_id = _new_job(
        username=username,
        role=role or "user",
        task_kind=GENERATION,
        summary=summary,
    )
    parent_job = _jobs[job_id]
    parent_job["status"] = "queued"
    parent_job["batch"] = batch_meta
    child_job_ids: list[str] = []
    child_positions: list[int | None] = []
    try:
        for index, child_req in enumerate(children):
            child_job_id = _new_job(
                username=username,
                role=role or "user",
                task_kind=GENERATION,
                summary=_job_summary(child_req),
            )
            child_job = _jobs[child_job_id]
            child_job["parent_job_id"] = job_id
            child_job["batch_index"] = index
            child_job["batch_count"] = len(children)
            queue_state = _enqueue_gpu_task(
                child_job_id,
                {
                    "req": child_req,
                    "username": username,
                    "role": role,
                    "parent_job_id": job_id,
                    "batch_index": index,
                },
                username=username,
                role=role,
                task_kind=GENERATION,
            )
            child_job_ids.append(child_job_id)
            child_positions.append(queue_state.get("queue_position"))
    except HTTPException:
        for child_job_id in child_job_ids:
            generation_queue.cancel(child_job_id)
            _jobs.pop(child_job_id, None)
        _jobs.pop(job_id, None)
        raise
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
            "task_kind": state.get("task_kind", job.get("task_kind", GENERATION)),
            "priority_class": state.get(
                "priority_class", job.get("priority_class", INTERACTIVE)
            ),
            "queued_at": state.get("queued_at"),
            "started_at": state.get("started_at"),
            "finished_at": state.get("finished_at"),
        })
    for parent in _jobs.values():
        if (
            parent.get("task_kind") != ANIMATION
            or parent.get("parent_job_id")
            or parent.get("status") in _TERMINAL_JOB_STATUSES | {"finalizing"}
        ):
            continue
        children = [
            _jobs.get(str(child_id))
            for child_id in parent.get("child_job_ids") or []
        ]
        active = next(
            (
                child
                for child in reversed(children)
                if child
                and child.get("status")
                not in _TERMINAL_JOB_STATUSES
            ),
            None,
        )
        if active:
            parent["status"] = active.get("status", parent.get("status"))
            parent["queue_position"] = active.get("queue_position")
            parent["queue_length"] = active.get("queue_length")


async def _broadcast_queue_state() -> None:
    _sync_queue_state_to_jobs()
    for job_id, job in list(_jobs.items()):
        if job.get("status") == "queued":
            await ws_manager.broadcast(job_id, {"type": "queue", **job})


def _active_generation_running() -> bool:
    """Report only an active generation, never the current helper's GPU lease."""
    if generation_queue is None or generation_queue.active_job_id is None:
        return False
    state = generation_queue.status(generation_queue.active_job_id)
    return (
        state.get("status") == "running"
        and state.get("task_kind") == GENERATION
    )


async def _execute_model_warmup(prompt_id_cb) -> None:
    """Load the configured INT8 Krea/CLIP/VAE chain with a disposable render."""
    global _last_model_signature, _last_warm_state
    from comfy_workflows import (
        KREA2_CLIP_NAME,
        _vae_name,
        comfy_generate,
        resolve_unet,
    )

    engine = str(getattr(settings, "diffusion_engine", "native_int8_convrot"))
    quantization = str(getattr(settings, "krea2_auto_quant", "int8"))
    if engine == "native_int8_convrot":
        quantization = "int8"
    elif engine == "native_gguf":
        quantization = "gguf"

    req = GenerationRequest(
        prompt="A neutral gray sphere on a plain background.",
        width=1024,
        height=1024,
        steps=1,
        cfg=0.0,
        seed=1,
        checkpoint="turbo",
        quantization=quantization,
        diffusion_engine=engine,
        use_rebalance=False,
        seed_variance_preset="off",
        vae_degrid=False,
    )
    unet, _dtype, _is_gguf, gguf_name = resolve_unet(req)
    signature = {
        "unet": gguf_name or unet,
        "clip": KREA2_CLIP_NAME,
        "vae": _vae_name(),
        "quantization": req.quantization,
    }
    _last_warm_state = {
        "status": "running",
        "updated_at": time.time(),
        "signature": signature,
    }
    logger.info("Krea warmup transition: running signature=%s", signature)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: comfy_generate(
            req,
            save_outputs=False,
            username=None,
            prompt_id_cb=prompt_id_cb,
        ),
    )
    _last_model_signature = signature
    _last_warm_state = {
        "status": "done",
        "updated_at": time.time(),
        "signature": signature,
    }
    logger.info("Krea warmup transition: done signature=%s", signature)


async def _validate_comfy_task_dispatch(
    job_id: str, task_kind: str, payload: dict
) -> bool:
    req = payload.get("req")
    if (
        task_kind == UPSCALE
        and getattr(req, "method", None) == "realesrgan"
        and not comfy_available()
    ):
        return True

    try:
        if not comfy_available():
            error = (
                "ComfyUI is unreachable; this GPU task was not submitted. "
                "Restore ComfyUI and retry."
            )
        elif not comfy_atomic_cancel_available():
            error = (
                "ComfyUI does not support required atomic cancellation; this GPU "
                "task was not submitted. Update/reinstall ComfyUI and retry."
            )
        else:
            return True
    except Exception:
        logger.exception("GPU dispatch capability validation failed")
        error = "GPU task dispatch validation failed. Check server logs and retry."

    if task_kind == ANIMATION:
        parent_id = str(
            payload.get("parent_job_id")
            or (_jobs.get(job_id) or {}).get("parent_job_id")
            or ""
        )
        if parent_id:
            await _fail_animation_parent(
                parent_id,
                child_id=job_id,
                public_error="Animation dispatch failed. Check server status and retry.",
            )
            return False

    job = _jobs.get(job_id)
    if job is not None:
        job["status"] = "error"
        job["error"] = error
        job["result"] = None
        job["finished_at"] = time.time()
    await ws_manager.broadcast(
        job_id,
        {
            "type": "error",
            "task_kind": task_kind,
            "result": None,
            "error": error,
        },
    )
    return False


async def _cleanup_animation_upload(
    project_or_id: AnimationProject | str,
) -> bool:
    try:
        project = (
            project_or_id
            if isinstance(project_or_id, AnimationProject)
            else await asyncio.to_thread(animation_store.load, project_or_id)
        )
    except Exception:
        return False
    upload_id = str(project.request.get("source_video_upload_id") or "")
    if not upload_id:
        return False
    return await asyncio.to_thread(
        animation_upload_store.delete,
        upload_id,
        username=project.owner,
        is_admin=True,
    )


async def _fail_animation_parent(
    parent_id: str,
    *,
    child_id: str | None = None,
    public_error: str,
) -> None:
    try:
        project = await asyncio.to_thread(animation_store.load, parent_id)
    except Exception:
        project = None
    if project is not None and project.status not in {
        "cancelled",
        "blocked",
        "done",
        "error",
    }:
        await asyncio.to_thread(
            animation_store.mark_status,
            parent_id,
            "error",
            public_error,
        )
        await asyncio.to_thread(animation_store.discard_staging, parent_id)
        await _cleanup_animation_upload(project)
    parent = _jobs.get(parent_id)
    if parent is not None and parent.get("status") not in {
        "cancelled",
        "blocked",
        "done",
    }:
        parent.update(
            {
                "status": "error",
                "error": public_error,
                "result": None,
                "queue_position": None,
                "finished_at": time.time(),
            }
        )
    child = _jobs.get(child_id or "")
    if child is not None:
        child.update(
            {
                "status": "error",
                "error": public_error,
                "queue_position": None,
                "comfy_prompt_id": None,
                "finished_at": time.time(),
            }
        )
    await ws_manager.broadcast(
        parent_id,
        {"type": "error", "status": "error", "error": public_error},
    )


def _animation_frame_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _quarantine_animation_staging(parent_id: str) -> str | None:
    project_dir = animation_store.project_dir(parent_id)
    staging = project_dir / "staging"
    if not staging.is_dir():
        return None
    destination = QUARANTINE_DIR / f"animation-{parent_id}"
    destination.mkdir(parents=True, exist_ok=True)
    first: str | None = None
    for path in staging.iterdir():
        if path.is_file() and not path.is_symlink():
            target = destination / path.name
            os.replace(path, target)
            first = first or target.name
    return first


async def _run_animation_chunk(job_id: str, payload: dict) -> str | None:
    parent_id = str(payload["parent_job_id"])
    chunk_index = int(payload["chunk_index"])
    child = _jobs[job_id]
    parent = _jobs[parent_id]
    child["status"] = "running"
    parent["status"] = "running"
    parent["started_at"] = parent.get("started_at") or time.time()
    loop = asyncio.get_running_loop()
    try:
        project = await asyncio.to_thread(
            animation_store.begin_chunk, parent_id, chunk_index
        )
        req = AnimateRequest.model_validate(project.request)
        start, end = project.chunk_ranges[chunk_index]
        frame_paths = await asyncio.to_thread(
            animation_store.frame_paths, parent_id, verify=False
        )
        init_image = (
            await asyncio.to_thread(_animation_frame_b64, frame_paths[-1])
            if frame_paths
            else (req.init_image_b64 or None)
        )
        reference_image = (
            await asyncio.to_thread(_animation_frame_b64, frame_paths[0])
            if frame_paths
            else (req.init_image_b64 or None)
        )
        source_path = None
        if req.source_video_upload_id:
            source_path = await asyncio.to_thread(
                animation_upload_store.resolve,
                req.source_video_upload_id,
                username=project.owner,
                is_admin=project.role == "admin",
            )

        def progress_callback(current: int, total: int, *_args) -> None:
            fraction = min(1.0, max(0.0, current / max(1, total)))

            def update() -> None:
                child["progress"] = int(fraction * 100)
                parent["progress"] = int(
                    100
                    * (project.completed_frames + fraction * (end - start))
                    / project.total_frames
                )

            loop.call_soon_threadsafe(update)

        async def render():
            from comfy_deforum import render_animation_chunk

            return await loop.run_in_executor(
                None,
                lambda: render_animation_chunk(
                    req,
                    project,
                    start=start,
                    end=end,
                    init_image_b64=init_image,
                    reference_image_b64=reference_image,
                    source_video_path=source_path,
                    controlled_video_root=(
                        animation_upload_store.root if source_path else None
                    ),
                    progress_cb=progress_callback,
                    prompt_id_cb=_task_prompt_id_callback(job_id),
                ),
            )

        await asyncio.to_thread(
            write_generation_breadcrumb,
            LOGS_DIR,
            job_id=parent_id,
            req={"kind": "animation"},
            stage="animation_chunk",
            extra={"chunk_index": chunk_index, "start": start, "end": end},
        )
        frames = await _run_gpu_operation_with_oom_retry(job_id, render)
        cancelled = bool(
            generation_queue is not None
            and generation_queue.cancel_requested(job_id)
        )
        disk = await asyncio.to_thread(animation_store.load, parent_id)
        if cancelled or disk.status == "cancelled":
            await asyncio.to_thread(animation_store.discard_staging, parent_id)
            if disk.status != "cancelled":
                disk = await asyncio.to_thread(
                    animation_store.mark_status, parent_id, "cancelled"
                )
            await _cleanup_animation_upload(disk)
            child["status"] = "cancelled"
            parent["status"] = "cancelled"
            return None
        staged = await asyncio.to_thread(
            animation_store.stage_chunk, parent_id, chunk_index, frames
        )
        if project.role == "child":
            def moderate_samples():
                images = []
                for raw in (frames[0], frames[-1]):
                    with Image.open(io.BytesIO(raw)) as image:
                        images.append(image.convert("RGB"))
                return moderate_images(images, role="child")

            moderation = await asyncio.to_thread(moderate_samples)
            if not moderation.allowed:
                quarantined = await asyncio.to_thread(
                    _quarantine_animation_staging, parent_id
                )
                event_id = await save_moderation_event(
                    username=project.owner or "local",
                    role="child",
                    event_type=moderation.event_type,
                    action="block_animation_chunk",
                    prompt="",
                    negative_prompt="",
                    mode="animation",
                    scores=moderation.scores,
                    reason=moderation.reason,
                    job_id=parent_id,
                    quarantined_filename=quarantined,
                )
                await asyncio.to_thread(
                    animation_store.mark_status, parent_id, "blocked"
                )
                await _cleanup_animation_upload(project)
                child["status"] = "blocked"
                parent.update(
                    {
                        "status": "blocked",
                        "error": "This animation was blocked by the child safety filter.",
                        "moderation_event_id": event_id,
                    }
                )
                await ws_manager.broadcast(
                    parent_id, {"type": "blocked", "error": parent["error"]}
                )
                return None
        if (
            generation_queue is not None
            and generation_queue.cancel_requested(job_id)
        ):
            await asyncio.to_thread(animation_store.discard_staging, parent_id)
            await asyncio.to_thread(
                animation_store.mark_status, parent_id, "cancelled"
            )
            await _cleanup_animation_upload(project)
            child["status"] = "cancelled"
            parent["status"] = "cancelled"
            return None
        committed = await asyncio.to_thread(
            animation_store.commit_chunk, parent_id, chunk_index, staged
        )
        child["status"] = "done"
        child["progress"] = 100
        parent["completed_frames"] = committed.completed_frames
        parent["progress"] = int(
            100 * committed.completed_frames / committed.total_frames
        )
        parent["status"] = committed.status
        await ws_manager.broadcast(
            parent_id,
            {
                "type": "progress",
                "status": committed.status,
                "progress": parent["progress"],
                "completed_frames": committed.completed_frames,
                "total_frames": committed.total_frames,
            },
        )
        return parent_id
    except Exception:
        logger.exception("Animation chunk failed for project %s", parent_id)
        await _fail_animation_parent(
            parent_id,
            child_id=job_id,
            public_error="Animation chunk failed. Check server logs and retry.",
        )
        return None
    finally:
        child["comfy_prompt_id"] = None
        child["finished_at"] = time.time()


async def _finalize_animation(parent_id: str) -> None:
    parent = _jobs.get(parent_id)
    if parent is None:
        return
    try:
        project = await asyncio.to_thread(animation_store.load, parent_id)
        if project.status != "finalizing":
            return
        frame_paths = await asyncio.to_thread(
            animation_store.frame_paths, parent_id, verify=True
        )
        project_dir = await asyncio.to_thread(animation_store.project_dir, parent_id)
        video = project_dir / "animation.mp4"
        poster = project_dir / "preview.jpg"
        metadata = await asyncio.to_thread(
            finalize_mp4,
            frame_paths,
            video,
            fps=int(project.request["fps"]),
            poster_path=poster,
        )
        latest = await asyncio.to_thread(animation_store.load, parent_id)
        if latest.status != "finalizing":
            return
        video_relative = video.relative_to(OUTPUTS_DIR).as_posix()
        poster_relative = poster.relative_to(OUTPUTS_DIR).as_posix()
        gallery_id = await save_media(
            video_relative,
            poster_filename=poster_relative,
            duration=float(metadata["duration"]),
            frame_count=int(metadata["frame_count"]),
            project_job_id=parent_id,
            owner_username=project.owner,
            prompt=project.request["prompt_schedule"],
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            seed=project.seed_base,
            metadata=metadata,
        )
        await asyncio.to_thread(
            animation_store.publish_result,
            parent_id,
            video_path="animation.mp4",
            poster_path="preview.jpg",
            gallery_id=gallery_id,
        )
        result = AnimationResult(
            video_url=f"/api/outputs/{video_relative}",
            poster_url=f"/api/outputs/{poster_relative}",
            frame_count=int(metadata["frame_count"]),
            fps=int(metadata["fps"]),
            duration=float(metadata["duration"]),
            gallery_id=gallery_id,
        ).model_dump()
        parent.update(
            {
                "status": "done",
                "progress": 100,
                "result": result,
                "finished_at": time.time(),
            }
        )
        await _cleanup_animation_upload(project)
        await asyncio.to_thread(
            clear_generation_breadcrumb, LOGS_DIR, job_id=parent_id
        )
        await _broadcast_job_event(
            parent_id, {"type": "done", "status": "done", "result": result}
        )
    except Exception:
        logger.exception("Animation finalization failed for %s", parent_id)
        try:
            project = await asyncio.to_thread(animation_store.load, parent_id)
            if project.status == "finalizing":
                project = await asyncio.to_thread(
                    animation_store.mark_status,
                    parent_id,
                    "error",
                    "Video finalization failed; committed frames were preserved.",
                )
                await _cleanup_animation_upload(project)
        except Exception:
            logger.exception("Could not persist animation finalization failure")
        parent["status"] = "error"
        parent["error"] = "Video finalization failed. Frames were preserved."
        await ws_manager.broadcast(
            parent_id, {"type": "error", "error": parent["error"]}
        )


def _schedule_animation_finalizer(parent_id: str) -> None:
    project_key = (
        os.path.normcase(str(animation_store.state_root)),
        parent_id,
    )
    if project_key in _animation_finalizer_project_ids:
        return
    _animation_finalizer_project_ids.add(project_key)
    task = asyncio.create_task(_finalize_animation(parent_id))
    setattr(task, "_animation_parent_id", parent_id)
    _animation_finalizer_tasks.add(task)

    def finished(done: asyncio.Task) -> None:
        _animation_finalizer_tasks.discard(done)
        _animation_finalizer_project_ids.discard(project_key)

    task.add_done_callback(finished)


async def _queued_gpu_task_handler(job_id: str, payload: dict) -> None:
    await _broadcast_queue_state()
    animation_parent: str | None = None
    async with GPU_LEASE:
        task_kind = (_jobs.get(job_id) or {}).get("task_kind", GENERATION)
        if not await _validate_comfy_task_dispatch(
            job_id, task_kind, payload
        ):
            return
        if task_kind == GENERATION:
            await _run_generation(
                job_id,
                payload["req"],
                username=payload.get("username"),
                role=payload.get("role", "user"),
            )
        elif task_kind == ANIMATION:
            if payload.get("operation") == "chunk":
                animation_parent = await _run_animation_chunk(job_id, payload)
            else:
                parent_id = str(
                    payload.get("parent_job_id")
                    or (_jobs.get(job_id) or {}).get("parent_job_id")
                    or ""
                )
                if parent_id:
                    await _fail_animation_parent(
                        parent_id,
                        child_id=job_id,
                        public_error="Animation dispatch failed. Check server status and retry.",
                    )
        elif task_kind in {PROMPT_EXPAND, PROMPT_PLAN, IMAGE_DESCRIBE}:
            await _run_helper_task(job_id, payload)
        elif task_kind == MODEL_WARMUP:
            global _last_warm_state
            job = _jobs[job_id]
            job["status"] = "running"
            try:
                await _run_gpu_operation_with_oom_retry(
                    job_id,
                    lambda: _execute_model_warmup(
                        _task_prompt_id_callback(job_id)
                    ),
                )
                cancelled = (
                    generation_queue is not None
                    and generation_queue.cancel_requested(job_id)
                )
                job["status"] = "cancelled" if cancelled else "done"
                if cancelled:
                    _last_warm_state = {
                        "status": "cancelled",
                        "updated_at": time.time(),
                        "signature": _last_model_signature,
                    }
                    logger.info("Krea warmup transition: cancelled")
            except Exception as exc:
                job["status"] = (
                    "cancelled"
                    if generation_queue is not None
                    and generation_queue.cancel_requested(job_id)
                    else "error"
                )
                error_text = str(exc).replace("\r", " ").replace("\n", " ")[:240]
                job["error"] = error_text or type(exc).__name__
                _last_warm_state = {
                    "status": job["status"],
                    "updated_at": time.time(),
                    "error": job["error"],
                }
                logger.warning(
                    "Krea warmup transition: %s (%s)",
                    job["status"],
                    type(exc).__name__,
                )
                raise
            finally:
                job["images"] = []
                job["result"] = None
                job["comfy_prompt_id"] = None
                job["finished_at"] = time.time()
        elif task_kind in {
            UPSCALE,
            DEPTH_PREVIEW,
            MOODBOARD_GUIDANCE,
            BACKGROUND_ENRICHMENT,
            HELPER_BENCHMARK,
        }:
            await _run_auxiliary_task(job_id, payload)
        else:
            raise RuntimeError(f"Unsupported GPU task kind: {task_kind}")
    if animation_parent is not None:
        await _continue_animation(animation_parent)
    await _broadcast_queue_state()


# ---------------------------------------------------------------------------
# WebSocket manager
# ---------------------------------------------------------------------------

class WSManager:
    def __init__(self):
        self._sockets: dict[str, list[WebSocket]] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        self._sockets.setdefault(job_id, []).append(ws)

    def disconnect(self, job_id: str, ws: WebSocket):
        socks = self._sockets.get(job_id)
        if socks and ws in socks:
            socks.remove(ws)

    async def broadcast(self, job_id: str, data: dict) -> int:
        dead = []
        sent = 0
        for ws in self._sockets.get(job_id, []):
            try:
                await ws.send_json(data)
                sent += 1
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(job_id, ws)
        return sent


ws_manager = WSManager()


def _record_terminal_ws_delivery(job_id: str, data: dict, sent: int) -> None:
    if sent < 1:
        return
    status = data.get("status") or data.get("type")
    has_result_payload = data.get("result") is not None or bool(data.get("images"))
    if status not in _TERMINAL_JOB_STATUSES or not has_result_payload:
        return
    job = _jobs.get(job_id)
    # WSManager only registers sockets after ws_endpoint's ownership check.
    if job is not None:
        job["result_delivered_at"] = job.get("result_delivered_at") or time.time()


async def _broadcast_job_event(job_id: str, data: dict) -> int:
    sent = await ws_manager.broadcast(job_id, data)
    _record_terminal_ws_delivery(job_id, data, sent)
    return sent

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


async def _recover_animation_projects() -> None:
    loop = asyncio.get_running_loop()
    recovery_lock = _animation_recovery_locks.setdefault(loop, asyncio.Lock())
    async with recovery_lock:
        await asyncio.to_thread(animation_store.reconcile_staging)
        projects = await asyncio.to_thread(animation_store.recoverable)
        root_key = os.path.normcase(str(animation_store.state_root))
        recoverable_keys = {(root_key, project.job_id) for project in projects}
        _animation_recovery_guard.intersection_update(recoverable_keys)
        active_uploads = {
            str(project.request.get("source_video_upload_id") or "")
            for project in projects
            if project.request.get("source_video_upload_id")
        }
        await asyncio.to_thread(animation_upload_store.cleanup, active_uploads)
        queue_records = (
            generation_queue.all_statuses() if generation_queue is not None else {}
        )
        for project in projects:
            parent = _jobs.setdefault(
                project.job_id, _animation_parent_from_project(project)
            )
            children = [
                _jobs.get(str(child_id))
                for child_id in parent.get("child_job_ids") or []
            ]
            has_active_child = any(
                child
                and child.get("status")
                not in _TERMINAL_JOB_STATUSES
                and (
                    child.get("parent_job_id") == project.job_id
                    or str(child.get("project_job_id") or "") == project.job_id
                )
                for child in children
            ) or any(
                record.get("status") not in _TERMINAL_JOB_STATUSES
                and (_jobs.get(task_id) or {}).get("parent_job_id")
                == project.job_id
                for task_id, record in queue_records.items()
            )
            if project.status == "running" and not has_active_child:
                project = await asyncio.to_thread(
                    animation_store.prepare_recovery, project.job_id
                )
            parent.update(
                {
                    "status": project.status,
                    "completed_frames": project.completed_frames,
                    "total_frames": project.total_frames,
                    "progress": int(
                        100 * project.completed_frames / project.total_frames
                    ),
                }
            )
            if project.status == "finalizing":
                project_key = (root_key, project.job_id)
                if project_key not in _animation_recovery_guard:
                    _animation_recovery_guard.add(project_key)
                    _schedule_animation_finalizer(project.job_id)
            elif project.status == "queued" and not has_active_child:
                await _enqueue_animation_chunk(
                    project.job_id, project.next_chunk_index
                )
                _animation_recovery_guard.add((root_key, project.job_id))


async def _recover_animations() -> None:
    await _recover_animation_projects()


async def _animation_upload_cleanup_loop() -> None:
    interval = settings.animation_upload_cleanup_interval_seconds
    delay = interval
    while True:
        await asyncio.sleep(delay)
        try:
            projects = await asyncio.to_thread(animation_store.recoverable)
            active = {
                str(project.request.get("source_video_upload_id") or "")
                for project in projects
                if project.request.get("source_video_upload_id")
            }
            await asyncio.to_thread(animation_upload_store.cleanup, active)
            delay = interval
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Periodic animation upload cleanup failed")
            delay = min(interval * 4, max(interval, delay * 2))


@app.on_event("startup")
async def startup():
    global generation_queue, _animation_upload_cleanup_task
    fault_path = enable_fault_logging(LOGS_DIR)
    stale = stale_generation_breadcrumbs(LOGS_DIR)
    if stale:
        logger.warning("Found %d stale active generation breadcrumb(s) after previous shutdown/crash: %s", len(stale), stale)
        archived = archive_stale_generation_breadcrumbs(LOGS_DIR)
        if archived:
            logger.info("Archived stale active generation breadcrumb(s): %s", archived)
    logger.info("Python fault handler logging to %s", fault_path)
    await init_db()
    await init_moderation_db()
    await init_moodboard_db()
    await reconcile_custom_moodboard_storage()
    comfy_reachable = comfy_available()
    if comfy_reachable and not comfy_atomic_cancel_available():
        if comfy_available():
            raise RuntimeError(
                "Atomic ComfyUI cancellation is unavailable: the configured "
                "ComfyUI is too old or mismatched. Update/reinstall it."
            )
        logger.warning(
            "ComfyUI became unreachable during atomic-cancel validation; "
            "continuing with existing unavailable behavior."
        )
    if generation_queue is None:
        generation_queue = GpuTaskQueue(_queued_gpu_task_handler)
        asyncio.create_task(generation_queue.run())
    await _recover_animations()
    if (
        _animation_upload_cleanup_task is None
        or _animation_upload_cleanup_task.done()
    ):
        _animation_upload_cleanup_task = asyncio.create_task(
            _animation_upload_cleanup_loop()
        )
    try:
        import comfy_qwen_vl
        comfy_qwen_vl.set_generation_busy_probe(_active_generation_running)
    except Exception:
        logger.debug("Could not install Comfy helper busy probe", exc_info=True)
    logger.info(
        "Krea 2 Studio ready (port=%s, public_base_path=%s)",
        os.environ.get("KREA_SERVER_PORT", "8200"),
        PUBLIC_BASE_PATH,
    )
    if await should_sync_moodboards(mark=True):
        asyncio.create_task(_sync_krea_moodboards())
    asyncio.create_task(_moodboard_sync_loop())
    asyncio.create_task(_moodboard_enrich_loop())
    asyncio.create_task(_funnel_health_loop())
    # Auto-load model if configured. Skipped when the ComfyUI backend is active:
    # ComfyUI owns the diffusion weights, so loading them in-process too would
    # double VRAM use and starve ComfyUI.
    if use_comfy_backend():
        logger.info("ComfyUI backend active; skipping in-process native model auto-load. Comfy at %s available=%s",
                    os.environ.get("KREA_COMFY_URL", "http://127.0.0.1:8188"), comfy_available())
        if getattr(settings, "krea_comfy_warmup", False):
            try:
                _enqueue_model_warmup()
            except Exception:
                logger.exception("Optional Krea startup warmup enqueue failed")
    else:
        cp = settings.krea2_auto_checkpoint or settings.krea2_turbo_path
        if cp and Path(cp).exists():
            asyncio.create_task(_auto_load_model(cp, settings.krea2_auto_quant, settings.krea2_blocks_to_swap))


@app.on_event("shutdown")
async def shutdown():
    global _animation_upload_cleanup_task
    if _animation_upload_cleanup_task is not None:
        _animation_upload_cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await _animation_upload_cleanup_task
        _animation_upload_cleanup_task = None
    tasks = list(_animation_finalizer_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
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


async def _funnel_health_loop() -> None:
    """Keep the public Tailscale Funnel healthy without anyone watching.

    Phones drop off for hours; when they come back, a stale Funnel ingress
    session (the recurring Windows failure) would greet them with TLS errors
    until someone clicked Repair. This checks Tailscale connectivity, Funnel
    binding/URL, and the public URL every 5 minutes, then requires three failed
    health intervals before repair. Repair attempts use bounded increasing
    backoff, and an explicit GUI stop disables repair until Start/Repair."""
    if not SHARE_AUTH_ENABLED:
        return
    from sharing_service import (
        funnel_status,
        local_krea_target_status,
        public_funnel_probe_with_retries,
        repair_funnel,
        tailscale_status,
    )

    await asyncio.sleep(300)  # share_startup owns the initial bring-up
    loop = asyncio.get_event_loop()
    while True:
        try:
            tailscale = await loop.run_in_executor(None, tailscale_status)
            funnel = await loop.run_in_executor(None, funnel_status)
            local = await loop.run_in_executor(None, local_krea_target_status)
            if not local.get("ok"):
                _funnel_health_monitor.observe(True, now=time.monotonic())
                logger.warning(
                    "Skipping Funnel ingress repair because local Krea is not reachable: %s.",
                    local.get("message", "unknown local failure"),
                )
                await asyncio.sleep(300)
                continue
            probe: dict = {
                "ok": False,
                "url": funnel.get("url", ""),
                "message": (
                    "Tailscale disconnected."
                    if not tailscale.get("connected")
                    else "Krea Funnel route is not configured."
                ),
            }
            probe_ok: bool | None = None
            if tailscale.get("connected") and funnel.get("running") and funnel.get("url"):
                probe = await loop.run_in_executor(
                    None,
                    lambda: public_funnel_probe_with_retries(funnel.get("url", ""), attempts=2, delay_seconds=10.0),
                )
                probe_ok = bool(probe.get("ok"))
            healthy = bool(tailscale.get("connected")) and funnel_interval_healthy(funnel, probe_ok)
            now = time.monotonic()
            previous_failures = _funnel_health_monitor.state.failed_intervals
            repair_due = _funnel_health_monitor.observe(healthy, now=now)
            if healthy and previous_failures:
                logger.info(
                    "Sharing health recovered: url=%s after %d failed interval(s).",
                    funnel.get("url", ""),
                    previous_failures,
                )
            if _funnel_health_monitor.enabled and not healthy:
                state = _funnel_health_monitor.state
                logger.warning(
                    "Sharing health interval failed (%d/%d); repair_due=%s; "
                    "backend_state=%s; url=%s; probe=%s.",
                    state.failed_intervals,
                    state.failure_threshold,
                    repair_due,
                    tailscale.get("backend_state", "unknown"),
                    probe.get("url") or funnel.get("url", ""),
                    probe.get("message", "unknown failure"),
                )
            if repair_due:
                state = _funnel_health_monitor.state
                delay = _funnel_health_monitor.record_repair(now=now)
                logger.warning(
                    "Sharing failed %d consecutive intervals; running self-heal. "
                    "Further repair is backed off for %d seconds.",
                    state.failure_threshold,
                    delay,
                )
                result = await loop.run_in_executor(None, repair_funnel)
                logger.info(
                    "Funnel self-heal completed: ok=%s; message=%s.",
                    result.get("ok"),
                    result.get("message", ""),
                )
        except Exception:
            logger.exception("Funnel health interval failed unexpectedly")
        await asyncio.sleep(300)


async def _moodboard_enrich_loop() -> None:
    """Offer one low-priority enrichment task to the unified GPU queue."""
    await asyncio.sleep(180)  # let startup / auto-load settle first
    while True:
        try:
            if getattr(settings, "krea2_moodboard_auto_enrich", True):
                job_id = _enqueue_background_enrichment()
                if job_id:
                    await ws_manager.broadcast(job_id, {"type": "queue", **_jobs[job_id]})
        except Exception:
            logger.exception("Background moodboard enrichment enqueue failed")
        await asyncio.sleep(300)


# ---------------------------------------------------------------------------
# Generation endpoints
# ---------------------------------------------------------------------------

@app.post("/api/admin/helper-benchmark", status_code=202)
async def enqueue_helper_benchmark(
    req: HelperBenchmarkRequest, request: Request
):
    username, role, admin = _request_user_role(request)
    if not admin:
        raise HTTPException(403, "Admin access required.")
    return await _enqueue_helper_task(
        request,
        task_kind=HELPER_BENCHMARK,
        summary="Qwen helper benchmark",
        payload=req.model_dump(),
    )


def _collect_generation_input_images(req: GenerationRequest) -> list[str]:
    """Collect user-supplied content images used by a generation request."""
    values: list[str] = []

    def add(value) -> None:
        if isinstance(value, str) and value.strip():
            values.append(value)

    for name in (
        "init_image_b64",
        "incontext_image_b64",
        "character_edit_source_b64",
        "character_edit_reference_b64",
        "style_transfer_image_b64",
        "ref_image1_b64",
        "ref_image2_b64",
        "ref_image3_b64",
    ):
        add(getattr(req, name, None))
    for item in req.character_edit_regions:
        add(item.reference_b64)
    for item in req.style_references:
        add(item.image_b64)
    for value in req.moodboard_images:
        add(value)

    # Masks are deliberately excluded: these schema fields are grayscale
    # spatial selectors rather than image content shown to the model.
    return list(dict.fromkeys(values))


async def _decode_generation_input_images(req: GenerationRequest) -> list[Image.Image]:
    values = _collect_generation_input_images(req)
    if not values:
        return []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _job_images_from_b64(values))


async def _enforce_child_generation_inputs(
    req: GenerationRequest, *, username: str | None, role: str
) -> None:
    if role != "child":
        return
    images = await _decode_generation_input_images(req)
    if not images:
        return
    decision = moderate_images(images, role="child")
    if decision.allowed:
        return
    await save_moderation_event(
        username=username or "local",
        role="child",
        event_type=decision.event_type,
        action="block_generation_input",
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        mode=req.mode,
        scores=decision.scores,
        reason=decision.reason,
        job_id=None,
    )
    raise HTTPException(
        403,
        "One or more images were blocked by the child safety filter and sent to an admin for review.",
    )


def _append_upload_chunk(path: Path, chunk: bytes) -> None:
    with path.open("ab") as handle:
        handle.write(chunk)


def _fsync_upload(path: Path) -> None:
    with path.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())


@app.post("/api/animate/uploads")
async def upload_animation_source(request: Request):
    username, role, is_role_admin = _request_user_role(request)
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(415, "Unsupported video content type.")
    declared = request.headers.get("content-length")
    if not declared:
        raise HTTPException(411, "Content-Length is required for video uploads.")
    try:
        declared_size = int(declared)
    except ValueError as exc:
        raise HTTPException(400, "Invalid Content-Length.") from exc
    if declared_size < 1 or declared_size > settings.animation_max_upload_bytes:
        raise HTTPException(413, "Video upload exceeds the byte limit.")
    try:
        active_projects = await asyncio.to_thread(animation_store.recoverable)
        active_upload_ids = {
            str(project.request.get("source_video_upload_id") or "")
            for project in active_projects
            if project.request.get("source_video_upload_id")
        }
        upload_id, temporary = await asyncio.to_thread(
            animation_upload_store.reserve,
            username,
            declared_size,
            active_upload_ids,
        )
    except UploadQuotaError as exc:
        raise HTTPException(429, "Animation upload quota reached.") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            size += len(chunk)
            if size > declared_size:
                raise HTTPException(413, "Video upload exceeds the byte limit.")
            digest.update(chunk)
            await asyncio.to_thread(_append_upload_chunk, temporary, chunk)
        if size == 0:
            raise HTTPException(400, "Video upload is empty.")
        await asyncio.to_thread(_fsync_upload, temporary)
        metadata = await asyncio.to_thread(
            animation_upload_store.finalize,
            upload_id,
            temporary,
            owner=username,
            content_type=content_type,
            size=size,
            sha256=digest.hexdigest(),
        )
        if role == "child":
            path = await asyncio.to_thread(
                animation_upload_store.resolve,
                upload_id,
                username=username,
                is_admin=is_role_admin,
            )
            try:
                samples = await asyncio.to_thread(
                    animation_upload_store.sample_images, path
                )
                decision = await asyncio.to_thread(
                    moderate_images, samples, role="child"
                )
            except Exception as exc:
                await asyncio.to_thread(
                    animation_upload_store.delete,
                    upload_id,
                    username=username,
                    is_admin=True,
                )
                raise HTTPException(
                    403, "The video could not be safely verified."
                ) from exc
            if not decision.allowed:
                await asyncio.to_thread(
                    animation_upload_store.delete,
                    upload_id,
                    username=username,
                    is_admin=True,
                )
                raise HTTPException(
                    403, "The video was blocked by the child safety filter."
                )
        return {
            key: metadata[key]
            for key in (
                "upload_id",
                "size",
                "sha256",
                "frame_count",
                "width",
                "height",
                "duration",
            )
        }
    except HTTPException:
        await asyncio.to_thread(
            animation_upload_store.abort,
            upload_id,
            username=username,
            is_admin=True,
        )
        raise
    except ValueError as exc:
        await asyncio.to_thread(
            animation_upload_store.abort,
            upload_id,
            username=username,
            is_admin=True,
        )
        raise HTTPException(422, "Video upload is invalid or exceeds limits.") from exc
    except Exception:
        await asyncio.to_thread(
            animation_upload_store.abort,
            upload_id,
            username=username,
            is_admin=True,
        )
        logger.exception("Animation upload failed")
        raise HTTPException(500, "Video upload failed.")


@app.post("/api/animate")
async def animate(req: AnimateRequest, request: Request):
    username, role, _is_admin = _request_user_role(request)
    return await _create_animation(req, username=username, role=role)


@app.post("/api/generate")
async def generate(req: GenerationRequest, request: Request):
    if getattr(req, "diffusion_engine", "native_pytorch") == "native_pytorch" and settings.diffusion_engine != "native_pytorch":
        fields = getattr(req, "model_fields_set", getattr(req, "__fields_set__", set()))
        if "diffusion_engine" not in fields:
            req.diffusion_engine = settings.diffusion_engine

    username, role, _is_admin = _request_user_role(request)
    summary = _job_summary(req)
    decision = moderate_prompt(req.prompt, req.negative_prompt, role=role)
    if not decision.allowed:
        job_id = _new_job(
            username=username,
            role=role,
            task_kind=GENERATION,
            summary=summary,
        )
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

    await _enforce_child_generation_inputs(req, username=username, role=role)

    if generation_queue is None:
        raise HTTPException(503, "Generation queue is not ready yet.")
    # 4K (or any >=2560px) is too heavy to batch on a 24GB card: force a single
    # image regardless of what the client sent (defensive; the UI also clamps this).
    if max(int(req.width or 0), int(req.height or 0)) >= 2560:
        req.num_images = 1
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
    if int8_batch_eligible(req):
        children = build_int8_variant_children(req)
        return await _enqueue_batch_children(
            children,
            {"mode": "safe_queue", "count": len(children), "parallel": False, "int8_variants": True},
            username,
            role,
            summary,
        )
    if req.batch_mode == "safe_queue" and int(req.num_images or 1) > 1:
        return await _enqueue_batch_children(
            build_safe_batch_children(req),
            {"mode": "safe_queue", "count": int(req.num_images), "parallel": False},
            username,
            role,
            summary,
        )
    job_id = _new_job(
        username=username,
        role=role,
        task_kind=GENERATION,
        summary=summary,
    )
    queue_state = await _enqueue_interactive_gpu_task(
        job_id,
        {"req": req, "username": username, "role": role},
        username=username,
        role=role,
        task_kind=GENERATION,
    )
    _sync_queue_state_to_jobs()
    job = _jobs[job_id]
    await ws_manager.broadcast(job_id, {"type": "queue", **job})
    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": queue_state.get("queue_position"),
        "queue_length": queue_state.get("queue_length"),
    }


async def _enforce_child_text(text: str, request: Request, *, action: str) -> None:
    """Block helper-LLM text (expanded/planned/described prompts) for child
    accounts when it trips the safety filter. No-op for admin/user roles."""
    username, role, _ = _request_user_role(request)
    if role != "child" or not (text or "").strip():
        return
    decision = moderate_prompt(text, "", role=role)
    if decision.allowed:
        return
    await save_moderation_event(
        username=username or "local", role=role, event_type=decision.event_type,
        action=action, prompt=text, negative_prompt="", mode="helper",
        scores=decision.scores, reason=decision.reason, job_id=None,
    )
    raise HTTPException(403, "This text was blocked by the child safety filter and sent to an admin for review.")


async def _enforce_child_images(images: list[Image.Image], request: Request, *, action: str, prompt: str = "") -> None:
    """Block image payloads for child accounts (upscale output, describe input)."""
    username, role, _ = _request_user_role(request)
    if role != "child" or not images:
        return
    decision = moderate_images(images, role=role)
    if decision.allowed:
        return
    await save_moderation_event(
        username=username or "local", role=role, event_type=decision.event_type,
        action=action, prompt=prompt, negative_prompt="", mode="helper",
        scores=decision.scores, reason=decision.reason, job_id=None,
    )
    raise HTTPException(403, "This image was blocked by the child safety filter and sent to an admin for review.")


def _task_prompt_id_callback(job_id: str):
    def callback(prompt_id: str) -> None:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["comfy_prompt_id"] = prompt_id
        if (
            generation_queue is not None
            and generation_queue.cancel_requested(job_id)
            and not job.get("cancellation_dispatching")
        ):
            from comfy_client import cancel_prompt

            dispatched = cancel_prompt(prompt_id)
            job["cancellation_dispatched"] = dispatched
            if dispatched:
                job["status"] = "cancelled"

    return callback


async def _moderate_worker_text(job_id: str, text: str, *, action: str) -> bool:
    job = _jobs[job_id]
    if job.get("role") != "child" or not (text or "").strip():
        return True
    decision = moderate_prompt(text, "", role="child")
    if decision.allowed:
        return True
    event_id = await save_moderation_event(
        username=job.get("username") or "local",
        role="child",
        event_type=decision.event_type,
        action=action,
        prompt=text,
        negative_prompt="",
        mode="helper",
        scores=decision.scores,
        reason=decision.reason,
        job_id=job_id,
    )
    job["status"] = "blocked"
    job["result"] = None
    job["error"] = "This helper output was blocked by the child safety filter and sent to an admin for review."
    job["moderation_event_id"] = event_id
    return False


async def _moodboard_suggestions(
    original_prompt: str,
    expanded_prompt: str,
    username: str,
) -> list[dict]:
    try:
        return await suggest_moodboards(
            original_prompt,
            expanded_prompt,
            username,
        )
    except Exception:
        logger.debug("Moodboard suggestions failed during prompt expansion", exc_info=True)
        return []


async def _run_helper_task(job_id: str, payload: dict) -> None:
    job = _jobs[job_id]
    task_kind = job["task_kind"]
    job["status"] = "running"
    job["started_at"] = job.get("started_at") or time.time()
    callback = _task_prompt_id_callback(job_id)
    loop = asyncio.get_event_loop()

    async def run_sync(fn):
        return await _run_gpu_operation_with_oom_retry(
            job_id, lambda: loop.run_in_executor(None, fn)
        )

    try:
        if task_kind == PROMPT_EXPAND:
            backend = payload["backend"]
            expanded = await run_sync(
                lambda: expand_prompt_result(
                    payload["prompt"],
                    backend=backend,
                    openrouter_api_key=_secret_value("OPENROUTER_API_KEY", "openrouter_api_key"),
                    openrouter_model=settings.openrouter_model,
                    openrouter_free_only=settings.openrouter_free_only,
                    ideogram_api_key=_secret_value("IDEOGRAM_API_KEY", "ideogram_api_key"),
                    gguf_helper_base_url=settings.gguf_helper_base_url,
                    gguf_helper_model=settings.gguf_helper_model,
                    gguf_helper_timeout_sec=settings.gguf_helper_timeout_sec,
                    prompt_id_cb=callback,
                ),
            )
            result = {
                "expanded": expanded.expanded,
                "changed": expanded.changed,
                "error": expanded.error,
                "backend": expanded.backend,
                "suggested_moodboards": (
                    await _moodboard_suggestions(
                        payload["prompt"],
                        expanded.expanded if not expanded.error else "",
                        str(job.get("username") or ""),
                    )
                    if payload.get("suggest_moodboards") and payload["prompt"].strip()
                    else []
                ),
                "sign_copy_pass": getattr(expanded, "sign_copy_pass", None),
            }
            output_text = result["expanded"]
            moderation_action = "block_expand_output"
        elif task_kind == PROMPT_PLAN:
            planned = await run_sync(
                lambda: plan_prompt(
                    payload["prompt"],
                    enabled=True,
                    max_tokens=payload["max_tokens"],
                    backend=payload["backend"],
                    gguf_helper_base_url=settings.gguf_helper_base_url,
                    gguf_helper_model=settings.gguf_helper_model,
                    gguf_helper_timeout_sec=settings.gguf_helper_timeout_sec,
                    prompt_id_cb=callback,
                ),
            )
            result = planned.model_dump()
            output_text = "\n".join(
                (result.get("planned_prompt", ""), result.get("negative_prompt", ""))
            )
            moderation_action = "block_plan_output"
        elif task_kind == IMAGE_DESCRIBE:
            if payload["backend"] == "openrouter":
                result = await run_sync(
                    lambda: describe_image_openrouter(
                        payload["image_b64"],
                        _secret_value("OPENROUTER_API_KEY", "openrouter_api_key"),
                        payload["mode"],
                        payload["guidance"],
                    ),
                )
            else:
                result = await run_sync(
                    lambda: describe_image_local(
                        payload["image_b64"],
                        payload["mode"],
                        payload["guidance"],
                        prompt_id_cb=callback,
                    ),
                )
            output_text = str(result.get("prompt") or "")
            moderation_action = "block_describe_output"
        else:
            raise RuntimeError(f"Unsupported helper task kind: {task_kind}")

        moderation_allowed = await _moderate_worker_text(
            job_id, output_text, action=moderation_action
        )
        if generation_queue is not None:
            if not generation_queue.begin_finalizing(job_id):
                job["status"] = "cancelled"
                job["result"] = None
                job["error"] = None
                await ws_manager.broadcast(
                    job_id,
                    {"type": "cancelled", "task_kind": task_kind, "result": None},
                )
                return
            job["status"] = "finalizing"
            await ws_manager.broadcast(
                job_id,
                {
                    "type": "status",
                    "status": "finalizing",
                    "task_kind": task_kind,
                    "result": None,
                },
            )
        if not moderation_allowed:
            job["status"] = "blocked"
            await ws_manager.broadcast(
                job_id,
                {
                    "type": "blocked",
                    "task_kind": task_kind,
                    "result": None,
                    "error": job["error"],
                    "moderation_event_id": job["moderation_event_id"],
                },
            )
            return
        job["result"] = result
        job["status"] = "done"
        job["progress"] = 100
        await _broadcast_job_event(
            job_id,
            {"type": "done", "task_kind": task_kind, "result": result},
        )
        if task_kind == IMAGE_DESCRIBE and result.get("backend") == "comfy":
            try:
                rewarm_id = _enqueue_model_warmup(force=True)
                if rewarm_id:
                    logger.info(
                        "Queued Krea rewarm after image description: %s",
                        rewarm_id,
                    )
            except Exception:
                logger.exception(
                    "Post-description Krea rewarm enqueue failed"
                )
    except Exception as exc:
        cancelled_error = any(
            marker in str(exc).lower()
            for marker in ("interrupt", "cancelled", "canceled")
        )
        if cancelled_error or (
            generation_queue is not None
            and generation_queue.cancel_requested(job_id)
        ):
            job["status"] = "cancelled"
            job["result"] = None
            await ws_manager.broadcast(
                job_id,
                {"type": "cancelled", "task_kind": task_kind, "result": None},
            )
        else:
            logger.exception("Helper task %s failed", job_id)
            job["status"] = "error"
            job["error"] = str(exc)
            job["result"] = None
            await ws_manager.broadcast(
                job_id,
                {
                    "type": "error",
                    "task_kind": task_kind,
                    "result": None,
                    "error": str(exc),
                },
            )
    finally:
        if job.get("status") in {"done", "blocked", "error", "cancelled"}:
            job["finished_at"] = job.get("finished_at") or time.time()
            job["comfy_prompt_id"] = None


def _upscale_result_payload(
    req: UpscaleRequest, result: Image.Image
) -> tuple[dict, list[Image.Image]]:
    from output_saver import encode_images

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
    encoded, _ = encode_images(
        [result], OUTPUTS_DIR, save_outputs=False, metadata=[metadata]
    )
    return {"image_b64": encoded[0], "metadata": metadata}, [result]


async def _execute_cpu_realesrgan(
    req: UpscaleRequest,
    source: Image.Image | None = None,
) -> tuple[dict, list[Image.Image]]:
    from upscaler import b64_to_pil, upscale_realesrgan

    loop = asyncio.get_event_loop()
    def run_pipeline() -> tuple[dict, list[Image.Image]]:
        decoded = source if source is not None else b64_to_pil(req.image_b64)
        result = upscale_realesrgan(decoded, MODELS_DIR, req.scale)
        return _upscale_result_payload(req, result)

    return await loop.run_in_executor(None, run_pipeline)


async def _decode_upscale_image(image_b64: str) -> Image.Image:
    from upscaler import b64_to_pil

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: b64_to_pil(image_b64))


async def _execute_upscale(req: UpscaleRequest, prompt_id_cb) -> tuple[dict, list[Image.Image]]:
    if not comfy_available():
        if req.method == "realesrgan":
            return await _execute_cpu_realesrgan(req)
        raise RuntimeError(
            f"Upscale method '{req.method}' requires ComfyUI. Start ComfyUI and retry."
        )
    from comfy_workflows import comfy_upscale

    method = "esrgan" if req.method == "realesrgan" else req.method
    upscale_by = (
        float(req.scale) if req.method == "realesrgan" else req.upscale_by
    )
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: comfy_upscale(
            method,
            req.image_b64,
            prompt=req.prompt,
            upscale_by=upscale_by,
            denoise=req.denoise,
            steps=req.steps,
            cfg=req.cfg,
            sampler=req.sampler,
            scheduler=req.scheduler,
            tile_width=req.tile_width or req.tile_size,
            tile_height=req.tile_height or req.tile_size,
            tile_padding=req.tile_padding,
            mask_blur=req.mask_blur,
            seam_mode=req.seam_mode,
            tile_mode=req.tile_mode,
            tiled_decode=req.tiled_decode,
            prompt_id_cb=prompt_id_cb,
        ),
    )
    return _upscale_result_payload(req, result)


async def _execute_depth_preview(payload: dict, prompt_id_cb) -> tuple[dict, list[Image.Image]]:
    if not use_comfy_backend():
        raise RuntimeError("Depth preview requires the ComfyUI backend.")
    if not comfy_available():
        raise RuntimeError("ComfyUI is not available.")
    from comfy_workflows import comfy_depth_preview

    loop = asyncio.get_event_loop()
    image = await loop.run_in_executor(
        None,
        lambda: comfy_depth_preview(
            payload["image_b64"],
            estimator=payload["estimator"],
            resolution=payload["resolution"],
            invert=payload["invert"],
            prompt_id_cb=prompt_id_cb,
        ),
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "image_b64": "data:image/png;base64,"
        + base64.b64encode(buf.getvalue()).decode()
    }, [image]


async def _execute_helper_benchmark(
    payload: dict, prompt_id_cb, cancel_probe
) -> dict:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from benchmark_qwen_helper import execute_benchmark_payload

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: execute_benchmark_payload(
            copy.deepcopy(payload),
            prompt_id_cb=prompt_id_cb,
            cancel_probe=cancel_probe,
        ),
    )


async def _prepare_moodboard_task(
    job_id: str, payload: dict, prompt_id_cb, cancel_probe
):
    return await prepare_moodboard_guidance_task(
        payload["operation"],
        payload,
        task_id=job_id,
        prompt_id_cb=prompt_id_cb,
        cancel_probe=cancel_probe,
    )


async def _commit_prepared_moodboard_task(prepared):
    return await commit_prepared_moodboard_task(prepared)


async def _cleanup_prepared_moodboard_task(prepared) -> None:
    await cleanup_prepared_moodboard_task(prepared)


async def _finish_prepared_moodboard_task(job_id: str, prepared):
    job = _jobs[job_id]
    task_kind = job["task_kind"]
    moderation_allowed = await _moderate_worker_text(
        job_id,
        prepared.moderation_text,
        action="block_moodboard_guidance_output",
    )
    if generation_queue is not None:
        if not generation_queue.begin_finalizing(job_id):
            job["status"] = "cancelled"
            job["result"] = None
            job["error"] = None
            await ws_manager.broadcast(
                job_id,
                {"type": "cancelled", "task_kind": task_kind, "result": None},
            )
            return None
        job["status"] = "finalizing"
        await ws_manager.broadcast(
            job_id,
            {
                "type": "status",
                "status": "finalizing",
                "task_kind": task_kind,
                "result": None,
            },
        )
    if not moderation_allowed:
        job["status"] = "blocked"
        await ws_manager.broadcast(
            job_id,
            {
                "type": "blocked",
                "task_kind": task_kind,
                "result": None,
                "error": job["error"],
                "moderation_event_id": job["moderation_event_id"],
            },
        )
        return None
    return await _commit_prepared_moodboard_task(prepared)


async def _moderate_worker_images(
    job_id: str, images: list[Image.Image], *, action: str, prompt: str = ""
) -> bool:
    job = _jobs[job_id]
    if job.get("role") != "child" or not images:
        return True
    decision = moderate_images(images, role="child")
    if decision.allowed:
        return True
    event_id = await save_moderation_event(
        username=job.get("username") or "local",
        role="child",
        event_type=decision.event_type,
        action=action,
        prompt=prompt,
        negative_prompt="",
        mode="helper",
        scores=decision.scores,
        reason=decision.reason,
        job_id=job_id,
    )
    job["status"] = "blocked"
    job["result"] = None
    job["error"] = (
        "This image was blocked by the child safety filter and sent to an admin for review."
    )
    job["moderation_event_id"] = event_id
    return False


async def _run_auxiliary_task(job_id: str, payload: dict) -> None:
    job = _jobs[job_id]
    task_kind = job["task_kind"]
    job["status"] = "running"
    job["started_at"] = job.get("started_at") or time.time()
    callback = _task_prompt_id_callback(job_id)
    await ws_manager.broadcast(
        job_id, {"type": "status", "status": "running", "task_kind": task_kind}
    )
    prepared_moodboard = None
    finalizing_started = False
    try:
        images: list[Image.Image] = []
        if task_kind == UPSCALE:
            result, images = await _run_gpu_operation_with_oom_retry(
                job_id, lambda: _execute_upscale(payload["req"], callback)
            )
            moderation_allowed = await _moderate_worker_images(
                job_id,
                images,
                action="block_upscale_output",
                prompt=payload["req"].prompt or "",
            )
        elif task_kind == DEPTH_PREVIEW:
            result, images = await _run_gpu_operation_with_oom_retry(
                job_id, lambda: _execute_depth_preview(payload, callback)
            )
            moderation_allowed = True
        elif task_kind == HELPER_BENCHMARK:
            result = await _run_gpu_operation_with_oom_retry(
                job_id,
                lambda: _execute_helper_benchmark(
                    payload,
                    callback,
                    lambda: bool(
                        generation_queue
                        and generation_queue.cancel_requested(job_id)
                    ),
                ),
            )
            moderation_allowed = True
        elif task_kind in {MOODBOARD_GUIDANCE, BACKGROUND_ENRICHMENT}:
            prepared_moodboard = await _run_gpu_operation_with_oom_retry(
                job_id,
                lambda: _prepare_moodboard_task(
                    job_id,
                    payload,
                    callback,
                    lambda: bool(
                        generation_queue
                        and generation_queue.cancel_requested(job_id)
                    ),
                ),
            )
            result = await _finish_prepared_moodboard_task(
                job_id, prepared_moodboard
            )
            if job.get("status") in {"cancelled", "blocked"}:
                return
            finalizing_started = job.get("status") == "finalizing"
            moderation_allowed = True
        else:
            raise RuntimeError(f"Unsupported auxiliary GPU task kind: {task_kind}")

        if generation_queue is not None and not finalizing_started:
            if not generation_queue.begin_finalizing(job_id):
                job["status"] = "cancelled"
                job["result"] = None
                job["error"] = None
                await ws_manager.broadcast(
                    job_id,
                    {"type": "cancelled", "task_kind": task_kind, "result": None},
                )
                return
            job["status"] = "finalizing"
            await ws_manager.broadcast(
                job_id,
                {
                    "type": "status",
                    "status": "finalizing",
                    "task_kind": task_kind,
                    "result": None,
                },
            )
        if not moderation_allowed:
            job["status"] = "blocked"
            await ws_manager.broadcast(
                job_id,
                {
                    "type": "blocked",
                    "task_kind": task_kind,
                    "result": None,
                    "error": job["error"],
                    "moderation_event_id": job["moderation_event_id"],
                },
            )
            return
        job["result"] = result
        job["status"] = "done"
        job["progress"] = 100
        await _broadcast_job_event(
            job_id, {"type": "done", "task_kind": task_kind, "result": result}
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        cancelled_error = any(
            marker in str(exc).lower()
            for marker in ("interrupt", "cancelled", "canceled")
        )
        if cancelled_error or (
            generation_queue is not None
            and generation_queue.cancel_requested(job_id)
        ):
            job["status"] = "cancelled"
            job["result"] = None
            await ws_manager.broadcast(
                job_id,
                {"type": "cancelled", "task_kind": task_kind, "result": None},
            )
        else:
            logger.exception("Auxiliary task %s failed", job_id)
            job["status"] = "error"
            job["error"] = str(exc)
            job["result"] = None
            await ws_manager.broadcast(
                job_id,
                {
                    "type": "error",
                    "task_kind": task_kind,
                    "result": None,
                    "error": str(exc),
                },
            )
    finally:
        if prepared_moodboard is not None:
            try:
                await _cleanup_prepared_moodboard_task(prepared_moodboard)
            except Exception:
                logger.warning(
                    "Could not clean moodboard staging for task %s",
                    job_id,
                    exc_info=True,
                )
        if job.get("status") in _TERMINAL_JOB_STATUSES:
            job["finished_at"] = job.get("finished_at") or time.time()
            job["comfy_prompt_id"] = None


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
        dst_name = f"{job_id}_{Path(safe_source).name}"
        safe_dest = _safe_served_filename(dst_name)
        if safe_dest is None:
            continue
        dst = QUARANTINE_DIR / safe_dest
        src.replace(dst)
        quarantined = quarantined or safe_dest
    return quarantined


def _safe_served_filename(filename: str) -> str | None:
    """Validate legacy image paths and controlled animation media paths."""
    raw = str(filename or "").replace("\\", "/")
    parts = [p for p in raw.split("/") if p]
    legacy = 1 <= len(parts) <= 2
    animation = (
        len(parts) == 4
        and parts[1] == "animations"
        and re.fullmatch(r"[a-f0-9-]{1,128}", parts[2]) is not None
        and Path(parts[3]).suffix.lower() in {".mp4", ".jpg", ".jpeg", ".png"}
    )
    if not (legacy or animation):
        return None
    for part in parts:
        if part in (".", "..") or not SAFE_SERVED_FILENAME_RE.fullmatch(part):
            return None
    return "/".join(parts)


async def _run_gpu_operation_with_oom_retry(
    job_id: str, operation, *, cleanup=None
):
    """Run one GPU operation with one precise CUDA-OOM recovery attempt."""
    first_error: Exception | None = None
    for attempt in range(2):
        try:
            return await operation()
        except Exception as exc:
            if cleanup is not None:
                cleaned = cleanup()
                if asyncio.iscoroutine(cleaned):
                    await cleaned
            cancelled = bool(
                generation_queue is not None
                and generation_queue.cancel_requested(job_id)
            )
            job = _jobs.get(job_id, {})
            can_retry = (
                attempt == 0
                and is_cuda_oom(exc)
                and not cancelled
                and job.get("status") != "finalizing"
            )
            if not can_retry:
                raise
            first_error = exc
            recovery = job.setdefault("_recovery", {})
            recovery["oom_attempts"] = 1
            recovery["last_error"] = str(exc)
            logger.warning(
                "CUDA OOM for %s task %s; freeing ComfyUI VRAM before one retry",
                job.get("task_kind", "gpu"),
                job_id,
            )
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: free_comfy_vram(
                        unload_models=True, free_memory=True
                    ),
                )
            except Exception:
                logger.warning(
                    "ComfyUI VRAM cleanup failed for task %s; retrying once",
                    job_id,
                    exc_info=True,
                )
            if (
                generation_queue is not None
                and generation_queue.cancel_requested(job_id)
            ):
                raise first_error
    raise first_error or RuntimeError("GPU operation failed")


def _cleanup_failed_generation_artifacts(paths: set[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not remove failed generation artifact %s",
                path,
                exc_info=True,
            )


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

    def prompt_id_cb(prompt_id: str) -> None:
        job["comfy_prompt_id"] = prompt_id
        # Cancellation may have arrived between starting the executor work and
        # ComfyUI accepting the prompt. Once its exact ID exists, cancel only it.
        if generation_queue is not None and generation_queue.cancel_requested(job_id):
            from comfy_client import cancel_prompt
            dispatched = cancel_prompt(prompt_id)
            job["cancellation_dispatched"] = dispatched
            if dispatched and job.get("status") not in {"done", "blocked", "error"}:
                job["status"] = "cancelled"

    try:
        # Planner and expander are dependencies of this already-admitted
        # generation. They execute here so the request remains one GPU task.
        if bool(getattr(req, "use_prompt_planner", False)):
            helper_backend = (
                "gguf-server"
                if settings.local_llm_backend == "gguf_server"
                else "local"
            )
            plan = await _run_gpu_operation_with_oom_retry(
                job_id,
                lambda: loop.run_in_executor(
                    None,
                    lambda: plan_prompt(
                    req.prompt,
                    enabled=True,
                    max_tokens=int(getattr(req, "prompt_planner_max_tokens", 700)),
                    backend=helper_backend,
                    gguf_helper_base_url=settings.gguf_helper_base_url,
                    gguf_helper_model=settings.gguf_helper_model,
                    gguf_helper_timeout_sec=settings.gguf_helper_timeout_sec,
                        prompt_id_cb=prompt_id_cb,
                    ),
                ),
            )
            planned_text = "\n".join((plan.planned_prompt, plan.negative_prompt))
            if not await _moderate_worker_text(
                job_id, planned_text, action="block_plan_output"
            ):
                await ws_manager.broadcast(
                    job_id,
                    {
                        "type": "blocked",
                        "task_kind": GENERATION,
                        "result": None,
                        "error": job["error"],
                        "moderation_event_id": job["moderation_event_id"],
                    },
                )
                return
            req.prompt_planner_output = plan.model_dump()
            if not req.prompt_planner_lock_original and plan.planned_prompt:
                req.prompt = plan.planned_prompt
            if plan.negative_prompt and not req.negative_prompt.strip():
                req.negative_prompt = plan.negative_prompt

        if bool(getattr(req, "use_prompt_expander", False)):
            helper_backend = (
                "gguf-server"
                if settings.local_llm_backend == "gguf_server"
                and settings.prompt_expander_backend == "local"
                else settings.prompt_expander_backend
            )
            expanded = await _run_gpu_operation_with_oom_retry(
                job_id,
                lambda: loop.run_in_executor(
                    None,
                    lambda: expand_prompt_result(
                    req.prompt,
                    backend=helper_backend,
                    openrouter_api_key=_secret_value("OPENROUTER_API_KEY", "openrouter_api_key"),
                    openrouter_model=settings.openrouter_model,
                    openrouter_free_only=settings.openrouter_free_only,
                    ideogram_api_key=_secret_value("IDEOGRAM_API_KEY", "ideogram_api_key"),
                    gguf_helper_base_url=settings.gguf_helper_base_url,
                    gguf_helper_model=settings.gguf_helper_model,
                    gguf_helper_timeout_sec=settings.gguf_helper_timeout_sec,
                        prompt_id_cb=prompt_id_cb,
                    ),
                ),
            )
            if expanded.error:
                raise RuntimeError(expanded.error)
            if not await _moderate_worker_text(
                job_id, expanded.expanded, action="block_expand_output"
            ):
                await ws_manager.broadcast(
                    job_id,
                    {
                        "type": "blocked",
                        "task_kind": GENERATION,
                        "result": None,
                        "error": job["error"],
                        "moderation_event_id": job["moderation_event_id"],
                    },
                )
                return
            req.prompt = expanded.expanded

        write_generation_breadcrumb(
            LOGS_DIR,
            job_id=job_id,
            req=req,
            stage="generation_start",
            extra={"username": username, "role": role},
        )
        # Native-era memory prep (unload in-process helpers, CUDA cache). In
        # Comfy mode nothing heavy lives in this process, and the sweep could
        # race a helper that holds no lease — skip it.
        if not use_comfy_backend():
            prep = prepare_for_generation(pipeline, clear_conditioning_cache=False)
            logger.info("Pre-generation memory cleanup: %s", prep)
        if getattr(req, "diffusion_engine", "native_pytorch") == "gguf_external":
            req.diffusion_engine = "native_gguf"
            req.quantization = "gguf"
        if getattr(req, "diffusion_engine", "native_pytorch") in {"native_int8_convrot", "int8_convrot_external"}:
            req.diffusion_engine = "native_int8_convrot"
            req.quantization = "int8"
        if use_comfy_backend():
            write_generation_breadcrumb(LOGS_DIR, job_id=job_id, req=req, stage="comfy_generation_start", extra={"provider": "comfyui"})
            from comfy_workflows import comfy_generate
            job["edit_provider"] = "comfyui"
            _owner = username if SHARE_AUTH_ENABLED else None
            original_payload = copy.deepcopy(req)
            attempt_artifacts: set[Path] = set()

            async def generate_once():
                nonlocal req
                attempt_req = copy.deepcopy(original_payload)
                attempt_artifacts.clear()

                def track_output(filename: str) -> None:
                    safe_filename = _safe_served_filename(str(filename or ""))
                    if safe_filename:
                        attempt_artifacts.add(OUTPUTS_DIR / safe_filename)

                generated = await loop.run_in_executor(
                    None,
                    lambda: comfy_generate(
                        attempt_req,
                        progress_cb=progress_cb,
                        username=_owner,
                        prompt_id_cb=prompt_id_cb,
                        output_file_cb=track_output,
                    ),
                )
                req = attempt_req
                return generated

            results, seed, filenames, lora_reports, metadata = (
                await _run_gpu_operation_with_oom_retry(
                    job_id,
                    generate_once,
                    cleanup=lambda: _cleanup_failed_generation_artifacts(
                        set(attempt_artifacts)
                    ),
                )
            )
        else:
            raise RuntimeError(
                "Native Studio generation is deprecated. ComfyUI is required "
                "ComfyUI is the generation engine. Start ComfyUI and retry."
            )
        # This is the cancellation cutoff. Once finalizing starts, cancellation
        # is closed while gallery persistence finishes; no result is public yet.
        if generation_queue is not None:
            if not generation_queue.begin_finalizing(job_id):
                for filename in filenames or []:
                    safe_filename = _safe_served_filename(str(filename or ""))
                    if safe_filename:
                        try:
                            (OUTPUTS_DIR / safe_filename).unlink(missing_ok=True)
                        except OSError:
                            logger.warning(
                                "Could not remove cancelled generation output %s",
                                safe_filename,
                                exc_info=True,
                            )
                job["images"] = []
                job["metadata"] = []
                job["result"] = None
                job["status"] = "cancelled"
                await ws_manager.broadcast(job_id, {"type": "cancelled"})
                return
            job["status"] = "finalizing"
            await ws_manager.broadcast(
                job_id, {"type": "status", "status": "finalizing"}
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
        # Surface LoRAs that were requested but not applied (wrong model/format).
        lora_warnings = [r for r in (lora_reports or []) if not r.get("applied")]

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

        job["images"] = results
        job["seed"] = seed
        job["metadata"] = metadata
        job["status"] = "done"
        job["progress"] = 100
        job["lora_warnings"] = lora_warnings
        await _broadcast_job_event(job_id, {
            "type": "done", "images": results, "seed": seed, "metadata": metadata,
            "lora_warnings": lora_warnings,
            "edit_provider": job.get("edit_provider"),
            "provider_warning": job.get("provider_warning"),
        })
        parent_job_id = job.get("parent_job_id")
        if parent_job_id:
            parent = _refresh_parent_batch_job(str(parent_job_id))
            if parent:
                await _broadcast_job_event(str(parent_job_id), {"type": "batch", **parent})

    except Exception as e:
        # A user cancel interrupts ComfyUI, which surfaces here as an exception.
        # Report it as a clean cancellation instead of a generation error.
        if generation_queue is not None and generation_queue.cancel_requested(job_id):
            logger.info("Generation cancelled by user for job %s", job_id)
            job["status"] = "cancelled"
            await ws_manager.broadcast(job_id, {"type": "cancelled"})
        else:
            logger.exception(f"Generation failed for job {job_id}")
            write_generation_breadcrumb(LOGS_DIR, job_id=job_id, req=req, stage="generation_error", extra={"error": str(e)})
            job["status"] = "error"
            job["error"] = str(e)
            await ws_manager.broadcast(job_id, {"type": "error", "error": str(e)})
        parent_job_id = job.get("parent_job_id")
        if parent_job_id:
            parent = _refresh_parent_batch_job(str(parent_job_id))
            if parent:
                await _broadcast_job_event(str(parent_job_id), {"type": "batch", **parent})
    finally:
        if job.get("status") in {"done", "blocked", "error", "cancelled"}:
            if not job.get("finished_at"):
                job["finished_at"] = time.time()
            clear_generation_breadcrumb(LOGS_DIR, job_id=job_id)
            job["comfy_prompt_id"] = None


@app.get("/api/generate/{job_id}")
async def job_status(job_id: str, request: Request):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    username, _role, is_admin = _request_user_role(request)
    if not _job_owned_by(job, username, is_admin):
        # 404 (not 403) so foreign job ids are indistinguishable from unknown ones.
        raise HTTPException(404, "Job not found")
    if job.get("child_job_ids") and job.get("task_kind") != ANIMATION:
        _refresh_parent_batch_job(job_id)
    if job.get("status") in _TERMINAL_JOB_STATUSES:
        job["result_delivered_at"] = job.get("result_delivered_at") or time.time()
    return job


def _without_large_result_payloads(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if (
                lowered in {"images", "image_b64", "image_b64s"}
                or lowered.endswith("_image_b64")
                or lowered.endswith("_images_b64")
            ):
                continue
            cleaned[key] = _without_large_result_payloads(item)
        return cleaned
    if isinstance(value, list):
        return [_without_large_result_payloads(item) for item in value]
    return value


def _acknowledge_job_payload(job: dict, acknowledged_at: float) -> None:
    job["num_images"] = max(
        int(job.get("num_images", 0) or 0),
        len(job.get("images") or []),
    )
    job["images"] = []
    job["result"] = _without_large_result_payloads(job.get("result"))
    job["metadata"] = _without_large_result_payloads(job.get("metadata"))
    job["result_delivered_at"] = job.get("result_delivered_at") or acknowledged_at
    job["result_acknowledged_at"] = acknowledged_at
    job.pop("thumb", None)


@app.post("/api/generate/{job_id}/ack")
async def acknowledge_job_result(job_id: str, request: Request):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    username, _role, is_admin = _request_user_role(request)
    if not _job_owned_by(job, username, is_admin):
        raise HTTPException(404, "Job not found")
    if job.get("status") not in _TERMINAL_JOB_STATUSES:
        raise HTTPException(409, "Job result is not terminal yet.")
    if not job.get("result_delivered_at"):
        raise HTTPException(409, "Job result has not been delivered yet.")
    acknowledged_at = time.time()
    if job.get("child_job_ids") and job.get("task_kind") != ANIMATION:
        children = [
            _jobs.get(str(child_id))
            for child_id in job.get("child_job_ids") or []
        ]
        job["completed_count"] = sum(
            1 for child in children if child and child.get("status") == "done"
        )
        for child in children:
            if (
                child
                and child.get("status") in _TERMINAL_JOB_STATUSES
                and _job_owned_by(child, username, is_admin)
            ):
                _acknowledge_job_payload(child, acknowledged_at)
    _acknowledge_job_payload(job, acknowledged_at)
    return {"ok": True, "job_id": job_id, "status": job.get("status")}


@app.get("/api/jobs")
async def list_jobs(request: Request, limit: int = 24):
    """Recent generation jobs for the queue panel (newest first). Batch children
    are folded into their parent. The requester sees full detail for their own
    jobs; other users' entries are anonymized to status + queue position only
    (no usernames, prompts, settings, thumbnails, or usable job ids)."""
    _sync_queue_state_to_jobs()
    username, _role, is_admin = _request_user_role(request)
    out: list[dict] = []
    for jid, job in reversed(list(_jobs.items())):
        if job.get("parent_job_id"):
            continue  # represented by its parent batch job
        if job.get("child_job_ids") and job.get("task_kind") != ANIMATION:
            _refresh_parent_batch_job(jid)
        status = job.get("status")
        mine = _job_owned_by(job, username, is_admin)
        if not mine:
            # Foreign finished jobs are irrelevant to this user's queue view.
            if status not in {"queued", "running"}:
                continue
            out.append({
                "job_id": f"anon-{jid[:8]}",  # stable list key, not a usable id
                "mine": False,
                "status": status,
                "progress": int(job.get("progress", 0) or 0),
                "queue_position": job.get("queue_position"),
                "queue_length": job.get("queue_length"),
                "seed": None,
                "error": None,
                "summary": foreign_summary(
                    str(job.get("task_kind") or GENERATION)
                ),
                "thumb": "",
                "is_batch": False,
                "batch_count": None,
                "num_images": 0,
            })
            if len(out) >= max(1, min(int(limit or 24), 100)):
                break
            continue
        thumb = job.get("thumb")
        if thumb is None and status == "done" and job.get("images"):
            thumb = _job_thumb(job["images"][0])
            job["thumb"] = thumb  # cache (even "" so we don't retry)
        out.append({
            "job_id": jid,
            "mine": True,
            "status": status,
            "progress": int(job.get("progress", 0) or 0),
            "queue_position": job.get("queue_position"),
            "queue_length": job.get("queue_length"),
            "seed": job.get("seed"),
            "error": job.get("error"),
            "summary": job.get("summary", ""),
            "task_kind": job.get("task_kind", GENERATION),
            "priority_class": job.get("priority_class", INTERACTIVE),
            "thumb": thumb or "",
            "is_batch": bool(job.get("child_job_ids")),
            "batch_count": (job.get("batch") or {}).get("count"),
            "num_images": int(
                job.get("num_images", len(job.get("images") or [])) or 0
            ),
            "queued_at": job.get("queued_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
        })
        if len(out) >= max(1, min(int(limit or 24), 100)):
            break
    admission = (
        generation_queue.admission(username)
        if generation_queue is not None
        else {
            "per_user_active": 0,
            "per_user_limit": 8,
            "global_interactive_active": 0,
            "global_interactive_limit": 64,
            "global_background_active": 0,
            "global_background_limit": 4,
        }
    )
    return {"jobs": out, "admission": admission}


async def _cancel_animation_job(job_id: str, job: dict) -> dict:
    project = await asyncio.to_thread(animation_store.load, job_id)
    if project.status == "finalizing":
        raise HTTPException(409, "Animation publication has already started.")
    if project.status in {"done", "error", "blocked", "cancelled"}:
        return {"ok": False, "job_id": job_id, "status": project.status, "cancelled": 0}
    await asyncio.to_thread(
        animation_store.mark_status, job_id, "cancelled"
    )
    job["cancel_requested"] = True
    job["status"] = "cancelled"
    job["queue_position"] = None
    cancelled = 0
    actions: list[tuple[str, str]] = []
    for child_id in list(job.get("child_job_ids") or []):
        child = _jobs.get(child_id)
        if (
            generation_queue is None
            or child is None
            or child.get("status") in _TERMINAL_JOB_STATUSES
        ):
            continue
        outcome = generation_queue.request_cancel(child_id)
        if outcome != "none":
            actions.append((child_id, outcome))
    from comfy_client import cancel_prompt

    for child_id, outcome in actions:
        child = _jobs.get(child_id)
        if child is None:
            continue
        if outcome == "interrupt" and child.get("comfy_prompt_id"):
            await asyncio.to_thread(cancel_prompt, child["comfy_prompt_id"])
        child["status"] = "cancelled"
        child["queue_position"] = None
        child["finished_at"] = time.time()
        cancelled += 1
        await ws_manager.broadcast(child_id, {"type": "cancelled"})
    await asyncio.to_thread(animation_store.discard_staging, job_id)
    await _cleanup_animation_upload(project)
    await ws_manager.broadcast(job_id, {"type": "cancelled", "status": "cancelled"})
    return {
        "ok": True,
        "job_id": job_id,
        "status": "cancelled",
        "cancelled": cancelled,
    }


@app.post("/api/animate/{job_id}/cancel")
@app.post("/api/generate/{job_id}/cancel")
async def cancel_generation_job(job_id: str, request: Request):
    """Cancel a generation. Dequeues any not-yet-started (batch) children and, if
    one of the targets is the job currently running on the GPU, interrupts ComfyUI
    so the in-flight prompt stops instead of running to completion."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    username, _role, is_admin = _request_user_role(request)
    if not _job_owned_by(job, username, is_admin):
        raise HTTPException(404, "Job not found")
    if job.get("task_kind") == ANIMATION and not job.get("parent_job_id"):
        return await _cancel_animation_job(job_id, job)
    targets = list(job.get("child_job_ids") or []) or [job_id]
    if job.get("child_job_ids"):
        job["cancel_requested"] = True
    cancelled = 0
    accepted = 0
    actions: list[tuple[str, str, str]] = []
    for tid in targets:
        if generation_queue is None:
            break
        child = _jobs.get(tid)
        if child and child.get("status") in _TERMINAL_JOB_STATUSES:
            continue
        outcome = generation_queue.request_cancel(tid)
        if outcome == "none":
            continue
        accepted += 1
        prompt_id = ""
        if outcome == "interrupt":
            child = _jobs.get(tid)
            prompt_id = str((child or {}).get("comfy_prompt_id") or "")
        actions.append((tid, outcome, prompt_id))

    # Sync queue positions first, then apply Studio's more precise cancellation
    # states so a still-running task is not prematurely presented as terminal.
    _sync_queue_state_to_jobs()
    from comfy_client import cancel_prompt
    loop = asyncio.get_event_loop()
    for tid, outcome, prompt_id in actions:
        child = _jobs.get(tid)
        if not child:
            continue
        if outcome == "dequeued":
            cancelled += 1
            child["status"] = "cancelled"
            child["queue_position"] = None
            if not child.get("finished_at"):
                child["finished_at"] = time.time()
            await ws_manager.broadcast(tid, {"type": "cancelled"})
            continue

        dispatched = False
        if prompt_id:
            dispatched = await loop.run_in_executor(
                None,
                lambda pid=prompt_id: cancel_prompt(pid),
            )
        child["cancellation_dispatched"] = dispatched
        if dispatched:
            cancelled += 1
            child["status"] = "cancelled"
            child["queue_position"] = None
            await ws_manager.broadcast(tid, {"type": "cancelled"})
        elif child.get("status") not in {"done", "blocked", "error", "cancelled"}:
            child["status"] = "cancellation_requested"
            await ws_manager.broadcast(
                tid,
                {"type": "status", "status": "cancellation_requested"},
            )

    if job.get("child_job_ids"):
        _refresh_parent_batch_job(job_id)
        if all((_jobs.get(t) or {}).get("status") == "cancelled" for t in targets):
            job["status"] = "cancelled"
            await ws_manager.broadcast(job_id, {"type": "cancelled"})
    elif cancelled:
        job["status"] = "cancelled"
        job["queue_position"] = None
    return {
        "ok": accepted > 0,
        "job_id": job_id,
        "status": job.get("status"),
        "cancelled": cancelled,
    }


@app.websocket("/ws/{job_id}")
async def ws_endpoint(ws: WebSocket, job_id: str):
    _strip_public_base_path(ws.scope)
    # Complete the handshake before a policy close so browsers can observe
    # code 1008 instead of seeing an opaque HTTP handshake rejection.
    await ws.accept()
    ws_user: str | None = None
    if SHARE_AUTH_ENABLED:
        ws_user = _auth_username_from_cookie(ws.cookies.get(SHARE_COOKIE))
        if not ws_user:
            await ws.close(code=1008)
            return
    job = _jobs.get(job_id)
    if job is None:
        # Unknown and unauthorized jobs share the same policy close, with no
        # payload or reason that could reveal whether an id exists.
        await ws.close(code=1008)
        return
    if SHARE_AUTH_ENABLED:
        role = get_user_role(SHARE_AUTH_FILE, ws_user) or "user"
        if not _job_owned_by(job, ws_user, role == "admin"):
            await ws.close(code=1008)
            return
    await ws_manager.connect(job_id, ws)
    if job:
        initial = {"type": "init", **job}
        await ws.send_json(initial)
        _record_terminal_ws_delivery(job_id, initial, 1)
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
    if use_comfy_backend():
        # ComfyUI manages weight loading itself (on demand, per job). Report
        # success so the UI's explicit "load" action stays a no-op.
        return {"status": "loaded", "checkpoint": req.checkpoint_path or "comfyui", "backend": "comfyui"}
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
    return {"ok": True, "detail": "ComfyUI backend active; models load on demand in ComfyUI.", "system": get_system_report()}

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
    media = await delete_media_record(
        gallery_id, owner_username=username, is_admin=is_role_admin
    )
    if media is not None:
        project_job_id = str(media.get("project_job_id") or "")
        if project_job_id:
            await asyncio.to_thread(
                animation_store.delete,
                project_job_id,
                username=username or "",
                is_admin=is_role_admin,
            )
        return {"ok": True, "filename": media["filename"]}
    filename = await delete_image(gallery_id, owner_username=username, is_admin=is_role_admin)
    if filename is None:
        raise HTTPException(404, "Not found")
    return {"ok": True, "filename": filename}


@app.get("/api/outputs/{filename:path}")
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
    if safe_name not in {row.get("filename"), row.get("poster_filename")}:
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
async def moodboards(request: Request, q: str = "", page: int = 1, page_size: int = 50, favorites: bool = False, source: str = "", shuffle_seed: str = ""):
    username = _public_moodboard_username(request)
    return await list_moodboards(
        query=q,
        page=page,
        page_size=page_size,
        favorites_only=favorites,
        source=source,
        shuffle_seed=shuffle_seed,
        username=username,
    )


@app.get("/api/moodboards/discoveries/latest", response_model=MoodboardDiscoveryResponse)
async def moodboard_latest_discovery(request: Request):
    return await latest_moodboard_discovery(
        username=_public_moodboard_username(request)
    )


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
    # The cache key is the source URL's hash, so a given path's bytes never change.
    # Tell the browser to cache aggressively so previews load instantly after the
    # first fetch instead of re-requesting the backend on every render.
    return FileResponse(path, headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/api/moodboards/{moodboard_id}", response_model=MoodboardItem)
async def moodboard_detail(moodboard_id: int, request: Request):
    item = await get_moodboard(
        moodboard_id,
        username=_public_moodboard_username(request),
    )
    if item is None:
        raise HTTPException(404, "Not found")
    return item


@app.put("/api/moodboards/{moodboard_id}/favorite")
async def favorite_moodboard(moodboard_id: int, req: FavoriteRequest, request: Request):
    username, _role, _is_admin = _request_user_role(request)
    await set_moodboard_favorite(moodboard_id, req.favorite, username=username or "")
    return {"ok": True}


@app.post("/api/moodboards/{moodboard_id}/qwen-guidance", status_code=202)
async def qwen_guidance_moodboard(moodboard_id: int, request: Request):
    queued = await _enqueue_helper_task(
        request,
        task_kind=MOODBOARD_GUIDANCE,
        summary=f"Moodboard guidance · #{moodboard_id}",
        payload={"operation": "single", "moodboard_id": moodboard_id},
    )
    _jobs[queued["job_id"]]["operation"] = "single"
    return queued


@app.post("/api/moodboards/qwen-guidance-missing", status_code=202)
async def qwen_guidance_missing(req: MoodboardGuidanceMissingRequest, request: Request):
    queued = await _enqueue_helper_task(
        request,
        task_kind=MOODBOARD_GUIDANCE,
        summary=f"Moodboard guidance · {req.limit} missing",
        payload={"operation": "missing", "limit": req.limit},
    )
    _jobs[queued["job_id"]]["operation"] = "missing"
    return queued


@app.post("/api/moodboards/mashup", status_code=202)
async def mashup_moodboard(req: MoodboardMashupRequest, request: Request):
    if len(set(req.moodboard_ids)) < 2:
        raise HTTPException(400, "Choose at least two moodboards to create a mashup.")
    queued = await _enqueue_helper_task(
        request,
        task_kind=MOODBOARD_GUIDANCE,
        summary="Moodboard mashup",
        payload={
            "operation": "mashup",
            "moodboard_ids": req.moodboard_ids,
            "weights": req.weights,
        },
    )
    _jobs[queued["job_id"]]["operation"] = "mashup"
    return queued


@app.post("/api/moodboards/custom")
async def create_custom_moodboard_endpoint(
    req: CustomMoodboardRequest, request: Request, response: Response
):
    await _enforce_child_images(
        _job_images_from_b64(req.image_b64s),
        request,
        action="block_moodboard_input",
    )
    needs_guidance = bool(req.image_b64s) and (
        not req.title.strip() or not req.taste_profile.strip()
    )
    if needs_guidance:
        queued = await _enqueue_helper_task(
            request,
            task_kind=MOODBOARD_GUIDANCE,
            summary="Custom moodboard authoring",
            payload={
                "operation": "custom",
                "title": req.title,
                "taste_profile": req.taste_profile,
                "keywords": req.keywords,
                "image_b64s": req.image_b64s,
            },
        )
        _jobs[queued["job_id"]]["operation"] = "custom"
        response.status_code = 202
        return queued
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
    raw = str(board_uuid)
    safe_board_uuid: str | None = None
    try:
        canonical = str(uuid.UUID(raw))
        if canonical == raw.lower():
            safe_board_uuid = canonical
    except (ValueError, TypeError):
        safe_board_uuid = None
    if safe_board_uuid is None:
        # Andro.Meta preset boards use a stable id ("andrometa-<mood_id>") rather
        # than a random UUID. Accept those too (safe path-segment chars only).
        if re.fullmatch(r"andrometa-[a-z0-9_]+", raw):
            safe_board_uuid = raw
        else:
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
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------------------
# LoRAs
# ---------------------------------------------------------------------------

@app.get("/api/moods")
async def get_moods():
    from moods import MOODS
    return MOODS


@app.get("/api/loras")
async def get_loras():
    from civitai_loras import enrich_loras
    loras = list_loras()
    if use_comfy_backend():
        # ComfyUI's loader (comfy.sd.load_lora_for_models) converts LoRA / LoKr /
        # LyCORIS / diffusers key formats, so the native SingleStreamDiT naming
        # check (inspect_lora) is too strict and must not block selection. If it
        # genuinely doesn't match, ComfyUI just applies 0 layers (no crash).
        for l in loras:
            if l.get("installed") and l.get("compatible") is False:
                l["compatible"] = True
                l["match_info"] = "Native key check didn't recognize this LoRA; ComfyUI will still attempt to apply it."
    return enrich_loras(loras, fetch=False)


@app.post("/api/loras/civitai-scan")
async def loras_civitai_scan():
    """Hash every installed LoRA and enrich it with Civitai metadata (background)."""
    import threading
    from civitai_loras import scan_all, scan_state
    if not scan_state()["scanning"]:
        threading.Thread(
            target=lambda: scan_all(list_loras, token=_secret_value("CIVITAI_TOKEN", "civitai_token") or None),
            daemon=True,
        ).start()
    return scan_state()


@app.get("/api/loras/civitai-scan/status")
async def loras_civitai_scan_status():
    from civitai_loras import scan_state
    return scan_state()


@app.get("/api/civitai/loras")
async def civitai_browse_loras(
    query: str = "",
    page: int = 1,
    sort: str = "Most Downloaded",
    nsfw: bool = False,
    cursor: str | None = None,
):
    """Browse Krea 2 LoRA + LoKr (LoCon) models on Civitai (cursor pagination)."""
    from civitai_loras import civitai_browse
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None,
            lambda: civitai_browse(
                query=query,
                page=page,
                sort=sort,
                nsfw=nsfw,
                cursor=cursor,
                token=_secret_value("CIVITAI_TOKEN", "civitai_token") or None,
            ),
        )
    except Exception as exc:
        logger.exception("Civitai browse failed")
        raise HTTPException(502, f"Civitai browse failed: {exc}")


@app.post("/api/civitai/install")
async def civitai_install_lora(req: dict):
    """Install a Civitai LoRA version into models/loras by version_id."""
    version_id = req.get("version_id")
    if not version_id:
        raise HTTPException(400, "version_id is required.")
    from civitai_loras import civitai_install
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, lambda: civitai_install(int(version_id), token=_secret_value("CIVITAI_TOKEN", "civitai_token") or None, filename=req.get("filename"))
        )
    except PermissionError as exc:
        raise HTTPException(402, str(exc))
    except Exception as exc:
        logger.exception("Civitai install failed")
        raise HTTPException(502, f"Civitai install failed: {exc}")
    return result


@app.get("/api/huggingface/loras")
async def huggingface_browse_loras(
    query: str = "",
    sort: str = "downloads",
    cursor: str | None = None,
    limit: int = 48,
):
    """Browse Krea 2 LoRAs on Hugging Face."""
    from huggingface_loras import huggingface_browse
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None,
            lambda: huggingface_browse(
                query=query,
                sort=sort,
                cursor=cursor,
                limit=limit,
                token=_secret_value("HF_TOKEN", "hf_token") or None,
            ),
        )
    except PermissionError as exc:
        raise HTTPException(401, str(exc))
    except Exception as exc:
        logger.exception("Hugging Face browse failed")
        raise HTTPException(502, f"Hugging Face browse failed: {exc}")


@app.post("/api/huggingface/install")
async def huggingface_install_lora(req: dict):
    """Install a Hugging Face LoRA into models/loras by repo_id."""
    repo_id = (req.get("repo_id") or "").strip()
    if not repo_id:
        raise HTTPException(400, "repo_id is required.")
    from huggingface_loras import MultiFileRequired, huggingface_install
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: huggingface_install(
                repo_id,
                filename=req.get("filename"),
                token=_secret_value("HF_TOKEN", "hf_token") or None,
            ),
        )
    except MultiFileRequired as exc:
        raise HTTPException(
            409,
            detail={"message": str(exc), "repo_id": exc.repo_id, "files": exc.files},
        )
    except PermissionError as exc:
        raise HTTPException(401, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Hugging Face install failed")
        raise HTTPException(502, f"Hugging Face install failed: {exc}")
    return result


@app.post("/api/loras/{lora_name}/download")
async def download_lora(lora_name: str):
    from lora_manager import OFFICIAL_LORAS, official_lora_download_kwargs
    from huggingface_hub import hf_hub_download
    import shutil
    if lora_name not in OFFICIAL_LORAS:
        raise HTTPException(404, f"Unknown LoRA: {lora_name}")
    loop = asyncio.get_event_loop()
    try:
        dest = await loop.run_in_executor(
            None,
            lambda: hf_hub_download(
                **official_lora_download_kwargs(lora_name, token=_secret_value("HF_TOKEN", "hf_token")),
            ),
        )
        info = OFFICIAL_LORAS[lora_name]
        if "repo_filename" in info:
            target = LORAS_DIR / str(info.get("filename") or f"{lora_name}.safetensors")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dest, target)
            dest = str(target)
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
        token = req.civitai_token or _secret_value("CIVITAI_TOKEN", "civitai_token")
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
async def upscale(req: UpscaleRequest, request: Request, response: Response):
    if req.method not in {
        "realesrgan",
        "tiled_vae",
        "model_refine",
        "ultimate",
        "refine_2pass",
        "wan_vae_2x",
        "seedvr2",
    }:
        raise HTTPException(400, f"Unknown upscale method: {req.method}")
    await _enforce_child_text(req.prompt or "", request, action="block_upscale_prompt")
    _username, role, _is_admin = _request_user_role(request)
    source = None
    if role == "child":
        source = await _decode_upscale_image(req.image_b64)
        await _enforce_child_images(
            [source],
            request,
            action="block_upscale_input",
            prompt=req.prompt or "",
        )
    response.status_code = 202
    return await _enqueue_helper_task(
        request,
        task_kind=UPSCALE,
        summary=f"Upscale · {req.method}",
        payload={"req": req},
    )


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


@app.post("/api/describe-image", status_code=202)
async def describe_image(req: DescribeImageRequest, request: Request):
    await _enforce_child_text(
        req.guidance, request, action="block_describe_guidance"
    )
    await _enforce_child_images(_job_images_from_b64([req.image_b64]), request, action="block_describe_input")
    backend = (
        "openrouter"
        if settings.prompt_expander_backend == "openrouter"
        else "local"
    )
    return await _enqueue_helper_task(
        request,
        task_kind=IMAGE_DESCRIBE,
        summary=f"Describe image · {req.mode}",
        payload={
            "image_b64": req.image_b64,
            "mode": req.mode,
            "guidance": req.guidance,
            "backend": backend,
        },
    )


@app.post("/api/depth-preview", status_code=202)
async def depth_preview(req: DepthPreviewRequest, request: Request):
    """Queue the exact depth map that ControlNet would follow."""
    await _enforce_child_images(
        _job_images_from_b64([req.image_b64]),
        request,
        action="block_depth_input",
    )
    if not use_comfy_backend():
        raise HTTPException(400, "Depth preview requires the ComfyUI backend.")
    if not comfy_available():
        raise HTTPException(503, "ComfyUI is not available.")
    return await _enqueue_helper_task(
        request,
        task_kind=DEPTH_PREVIEW,
        summary=f"Depth preview · {req.estimator}",
        payload={
            "image_b64": req.image_b64,
            "estimator": req.estimator,
            "resolution": req.resolution,
            "invert": req.invert,
        },
    )


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

def _warmup_diagnostics() -> dict:
    enabled = bool(getattr(settings, "krea_comfy_warmup", False))
    job = _jobs.get(_model_warmup_job_id or "") or {}
    queue_state: dict = {}
    if generation_queue is not None and _model_warmup_job_id:
        try:
            direct = generation_queue.status(_model_warmup_job_id)
            statuses = generation_queue.all_statuses()
            queue_state = dict(
                statuses.get(_model_warmup_job_id)
                or direct
                or {}
            )
        except Exception:
            logger.debug("Could not read authoritative warmup queue state", exc_info=True)
    queue_status = str(queue_state.get("status") or "")
    job_status = str(job.get("status") or "")
    terminal = {"done", "error", "cancelled"}
    if queue_status in terminal:
        raw_state = queue_status
        timing_source = queue_state
    elif job_status in terminal | {"finalizing", "cancellation_requested"}:
        raw_state = job_status
        timing_source = job
    elif queue_status:
        raw_state = queue_status
        timing_source = queue_state
    else:
        raw_state = str(
            job_status or _last_warm_state.get("status") or "queued"
        )
        timing_source = job
    if not enabled:
        state = "disabled"
    else:
        state = {
            "warm": "done",
            "never": "queued",
        }.get(raw_state, raw_state)
        if state not in {
            "queued",
            "running",
            "finalizing",
            "cancellation_requested",
            "done",
            "error",
            "cancelled",
        }:
            state = "queued"

    raw_signature = (
        _last_warm_state.get("signature")
        or _last_model_signature
        or {}
    )
    signature = {}
    for key in ("unet", "clip", "vae", "quantization"):
        value = raw_signature.get(key)
        if value is None:
            continue
        text = str(value)
        signature[key] = (
            text.replace("\\", "/").rsplit("/", 1)[-1]
            if key in {"unet", "clip", "vae"}
            else text[:40]
        )
    has_error = bool(job.get("error") or _last_warm_state.get("error"))
    if state == "done" and has_error:
        state = "error"
        timing_source = job
    return {
        "enabled": enabled,
        "state": state,
        "signature": signature or None,
        "queued_at": timing_source.get("queued_at", job.get("queued_at")),
        "started_at": timing_source.get("started_at", job.get("started_at")),
        "finished_at": timing_source.get("finished_at", job.get("finished_at")),
        "last_error": (
            "Warmup failed; see server logs." if has_error else None
        ),
    }


@app.get("/api/system")
async def system_info():
    report = get_system_report()
    # Surface the auto-detected checkpoint so the UI can prefill the load form
    # (one-click recovery if auto-load failed, e.g. transient low RAM).
    auto_cp = settings.krea2_auto_checkpoint or settings.krea2_turbo_path or ""
    auto_quant = settings.krea2_auto_quant or ("fp8" if "fp8" in auto_cp.lower() else "bf16")
    comfy_on = use_comfy_backend()
    comfy_up = comfy_available() if comfy_on else False
    report["model_status"] = {
        # ComfyUI loads weights on demand per job, so "loaded" tracks server
        # reachability when the Comfy backend is active. This keeps the UI's
        # model-loaded gate open without loading anything in-process.
        "loaded": comfy_up if comfy_on else pipeline.is_loaded(),
        "loading": False if comfy_on else getattr(pipeline, "_loading", False),
        "checkpoint": ("comfyui" if comfy_on else pipeline._loaded_checkpoint),
        "quantization": (None if comfy_on else pipeline._loaded_quant),
        "auto_checkpoint": auto_cp,
        "auto_quant": auto_quant,
        "load_error": (None if comfy_on else getattr(pipeline, "_last_load_error", None)),
        "text_encoder_source": (None if comfy_on else getattr(pipeline, "_text_encoder_source", None)),
        "memory": pipeline.memory_status(),
        "backend": "comfyui" if comfy_on else "native",
        "comfy_available": comfy_up,
        "warmup": _warmup_diagnostics(),
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

    has_token = bool(_secret_value("HF_TOKEN", "hf_token"))
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

    token = _secret_value("HF_TOKEN", "hf_token") or None

    loop = asyncio.get_event_loop()
    try:
        path = await loop.run_in_executor(None, lambda: download_asset(spec, token=token))
    except GatedRepoError:
        raise HTTPException(
            401,
            "Your Hugging Face token does not have access to this gated model yet. Open the model page, accept the license/access terms, then retry.",
        )
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) in {401, 403}:
            raise HTTPException(
                401,
                "Hugging Face rejected the download. Confirm the token is valid and has access to the gated model, then retry.",
            )
        logger.exception("Quality asset download failed")
        raise HTTPException(502, "Quality asset download failed. Check connection and server logs.")
    except Exception:
        logger.exception("Quality asset download failed")
        raise HTTPException(502, "Quality asset download failed. Check connection and server logs.")

    has_token = bool(_secret_value("HF_TOKEN", "hf_token"))
    return {"ok": True, "path": str(path), "item": asset_status(spec, has_hf_token=has_token)}


@app.post("/api/xperiment/setup")
async def xperiment_setup_endpoint():
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from quality_assets import asset_by_id, asset_installed, asset_status, download_asset

    required_ids = ["wan_2_1_vae", "qwen3vl_abliterated_fp8", "krea2_realism_v1_lora"]
    results: list[dict] = []
    token = _secret_value("HF_TOKEN", "hf_token") or None
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
    env["LOCAL_LLM_BACKEND"] = "comfy"
    env["COMFY_QWEN_MODEL"] = "2b"
    env["COMFY_QWEN_QUANT"] = "8bit"
    env["COMFY_QWEN_VISION_MODEL"] = "4b"
    env["COMFY_QWEN_VISION_QUANT"] = "8bit"
    settings.krea2_vae_path = str(wan_vae)
    settings.prompt_expander_backend = "local"
    settings.local_llm_backend = "comfy"
    settings.comfy_qwen_model = "2b"
    settings.comfy_qwen_quant = "8bit"
    settings.comfy_qwen_vision_model = "4b"
    settings.comfy_qwen_vision_quant = "8bit"
    _write_env(env)
    bypass_spec = asset_by_id("krea2_filter_bypass")
    bypass = asset_status(bypass_spec, has_hf_token=bool(token))
    # Uncensored recipe (Comfy-Org Krea-2 Turbo reference workflow): abliterated
    # Qwen3-VL text encoder + the txtfusion filter-bypass diff + realism LoRA.
    loras = [
        {"name": "Krea2-realism-V1", "filename": "Krea2-realism-V1.safetensors", "strength": 0.6, "block_filter": "late"},
    ]
    try:
        bypass_installed = (
            asset_installed(bypass_spec)
            and bypass_spec.local_path.exists()
            and bypass_spec.local_path.stat().st_size > 0
        )
    except OSError:
        bypass_installed = False
    if bypass_installed:
        loras.insert(0, {"name": "krea2filterbypass3", "filename": "krea2filterbypass3.safetensors", "strength": 4.0, "block_filter": "style_safe"})
    # Prefer int8 (user preference); keep gguf if that's what's configured.
    configured_engine = settings.diffusion_engine if settings.diffusion_engine in ("native_int8_convrot", "native_gguf") else "native_int8_convrot"
    configured_quant = "gguf" if configured_engine == "native_gguf" else "int8"
    warnings = [
        "CFG 1.0 = guidance-off for the distilled Turbo model (matches the reference workflow's ConditioningZeroOut). ComfyUI treats CFG<1 as prompt-ignoring, so Xperiment uses 1.0, not 0.",
        "Runs on your int8/gguf engine as chosen. LoRA patching on the custom-quant engine is slower than fp8-cast; set KREA_LORA_ENGINE=fp8_fast in .env to trade quant for speed.",
    ]
    if not bypass_installed:
        warnings.append("krea2filterbypass3 (the uncensor diff) is manual-only and was not found in models/loras, so it was NOT attached. Add it to enable the fully unfiltered recipe.")
    return {
        "ok": True,
        "assets": results,
        "vae_path": str(wan_vae),
        "lora": loras[0],
        "loras": loras,
        "diffusion_engine": configured_engine,
        "quantization": configured_quant,
        # ClownsharKSampler_Beta (RES4LYF) exact recipe from the reference workflow.
        "sampler": {"sampler": "er_sde", "scheduler": "beta57", "steps": 8, "cfg": 1.0},
        "res4lyf": {"sampler_name": "exponential/ddim", "eta": 0.5, "bongmath": False},
        "use_prompt_expander": False,
        "prompt_expander_backend": "local",
        "local_llm_backend": "comfy",
        "comfy_qwen_model": "2b",
        "comfy_qwen_quant": "8bit",
        "comfy_qwen_vision_model": "4b",
        "comfy_qwen_vision_quant": "8bit",
        "benchmark_note": "Uncensored Krea-2 Turbo recipe: abliterated Qwen3-VL encoder + filter-bypass diff @4 + Realism LoKr @0.6, ClownsharKSampler_Beta exponential/ddim + beta57, 8 steps, CFG 1, eta 0.5.",
        "manual_only": [bypass],
        "warnings": warnings,
    }


@app.post("/api/gguf/setup-low-vram")
async def gguf_setup_low_vram_endpoint():
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from quality_assets import asset_by_id, asset_installed, asset_status, download_asset

    required_ids = ["gguf_krea2_turbo_q4km", "wan_2_1_vae"]
    token = _secret_value("HF_TOKEN", "hf_token") or None
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
                logger.exception("GGUF low-VRAM asset download failed")
                raise HTTPException(502, f"Could not download {asset_id}. Check connection and server logs.") from exc
        results.append({"id": asset_id, "path": str(path), "skipped": skipped, "item": asset_status(spec, has_hf_token=bool(token))})

    q4 = asset_by_id("gguf_krea2_turbo_q4km").local_path
    vae = asset_by_id("wan_2_1_vae").local_path
    env = _read_env()
    env["DIFFUSION_ENGINE"] = "native_gguf"
    env["GGUF_TURBO_PATH"] = str(q4)
    env["KREA2_AUTO_CHECKPOINT"] = str(q4)
    env["KREA2_AUTO_QUANT"] = "gguf"
    env["KREA2_VAE_PATH"] = str(vae)
    settings.diffusion_engine = "native_gguf"
    settings.gguf_turbo_path = str(q4)
    settings.krea2_auto_checkpoint = str(q4)
    settings.krea2_auto_quant = "gguf"
    settings.krea2_vae_path = str(vae)
    _write_env(env)
    return {
        "ok": True,
        "assets": results,
        "diffusion_engine": "native_gguf",
        "turbo_path": str(q4),
        "checkpoint_path": str(q4),
        "quantization": "gguf",
        "vae_path": str(vae),
        "sampler": {"sampler": "euler", "scheduler": "simple", "steps": 8, "cfg": 0.0, "mu": 1.15},
        "warnings": ["Native GGUF is configured as the low-VRAM diffusion path. The old stable-diffusion.cpp sidecar is no longer used."],
    }


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

ENV_PATH = Path(__file__).parent.parent / ".env"


def _read_env() -> dict[str, str]:
    return read_env(ENV_PATH)


def _write_env(env: dict[str, str]) -> None:
    write_env(ENV_PATH, env)


def _secret_value(env_key: str, attr: str, env: dict[str, str] | None = None) -> str:
    return secret_value(env_key, getattr(settings, attr, ""), env)


@app.get("/api/settings")
async def get_settings():
    env = _read_env()
    configured_engine = env.get("DIFFUSION_ENGINE", settings.diffusion_engine)
    if configured_engine == "gguf_external":
        configured_engine = "native_gguf"
    elif configured_engine == "int8_convrot_external":
        configured_engine = "native_int8_convrot"
    krea_deforum = await asyncio.to_thread(krea_deforum_status, timeout=1.0)
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
        "comfy_qwen_model": env.get("COMFY_QWEN_MODEL", settings.comfy_qwen_model),
        "comfy_qwen_quant": env.get("COMFY_QWEN_QUANT", settings.comfy_qwen_quant),
        "comfy_qwen_vision_model": env.get(
            "COMFY_QWEN_VISION_MODEL", settings.comfy_qwen_vision_model
        ),
        "comfy_qwen_vision_quant": env.get(
            "COMFY_QWEN_VISION_QUANT", settings.comfy_qwen_vision_quant
        ),
        "krea_comfy_warmup": env.get(
            "KREA_COMFY_WARMUP", str(settings.krea_comfy_warmup)
        ).lower() in {"1", "true", "yes", "on"},
        "local_qwen_model_id": env.get("LOCAL_QWEN_MODEL_ID", settings.local_qwen_model_id),
        "local_qwen_device": env.get("LOCAL_QWEN_DEVICE", settings.local_qwen_device),
        "gguf_helper_base_url": env.get("GGUF_HELPER_BASE_URL", settings.gguf_helper_base_url),
        "gguf_helper_model": env.get("GGUF_HELPER_MODEL", settings.gguf_helper_model),
        "gguf_helper_timeout_sec": int(env.get("GGUF_HELPER_TIMEOUT_SEC", str(settings.gguf_helper_timeout_sec)) or 120),
        "diffusion_engine": configured_engine,
        "gguf_turbo_path": env.get("GGUF_TURBO_PATH", settings.gguf_turbo_path),
        "gguf_raw_path": env.get("GGUF_RAW_PATH", settings.gguf_raw_path),
        "openrouter_model": env.get("OPENROUTER_MODEL", settings.openrouter_model),
        "openrouter_free_only": env.get("OPENROUTER_FREE_ONLY", str(settings.openrouter_free_only)).lower() in {"1", "true", "yes"},
        "krea_share_auto_funnel": env.get("KREA_SHARE_AUTO_FUNNEL", str(settings.krea_share_auto_funnel)).lower() in {"1", "true", "yes", "on"},
        "krea2_vae_path": env.get("KREA2_VAE_PATH", settings.krea2_vae_path),
        "krea2_vae_mode": env.get("KREA2_VAE_MODE", settings.krea2_vae_mode),
        "krea2_vae_blend_radius": int(env.get("KREA2_VAE_BLEND_RADIUS", str(settings.krea2_vae_blend_radius)) or 24),
        "krea2_vae_blend_strength": float(env.get("KREA2_VAE_BLEND_STRENGTH", str(settings.krea2_vae_blend_strength)) or 0.65),
        "krea_attention_backend": env.get("KREA_ATTENTION_BACKEND", settings.krea_attention_backend),
        "seedvr2_model": env.get("SEEDVR2_MODEL", settings.seedvr2_model),
        "has_hf_token": bool(_secret_value("HF_TOKEN", "hf_token", env)),
        "has_civitai_token": bool(_secret_value("CIVITAI_TOKEN", "civitai_token", env)),
        "has_ideogram_api_key": bool(_secret_value("IDEOGRAM_API_KEY", "ideogram_api_key", env)),
        "has_openrouter_api_key": bool(_secret_value("OPENROUTER_API_KEY", "openrouter_api_key", env)),
        "krea_deforum": krea_deforum,
        "animation": {
            "chunk_size": settings.animation_chunk_size,
            "max_frames": settings.animation_max_frames,
            "max_dimension": settings.animation_max_dimension,
            "max_upload_bytes": settings.animation_max_upload_bytes,
            "uploads_per_user": settings.animation_uploads_per_user,
            "upload_bytes_per_user": settings.animation_upload_bytes_per_user,
            "uploads_global": settings.animation_uploads_global,
            "upload_bytes_global": settings.animation_upload_bytes_global,
            "upload_cleanup_interval_seconds": settings.animation_upload_cleanup_interval_seconds,
            "max_source_duration_seconds": settings.animation_max_source_duration_seconds,
            "active_per_user": settings.animation_active_per_user,
            "upload_content_types": sorted(ALLOWED_VIDEO_TYPES),
        },
    }


@app.put("/api/settings")
async def update_settings(req: SettingsUpdate):
    env = _read_env()
    # API keys persist to .env so they survive restarts (empty values are ignored
    # so a blank Save never wipes an existing persistent key).
    if req.hf_token is not None:
        settings.hf_token = req.hf_token.replace("\r", "").replace("\n", "")
        if settings.hf_token:
            env["HF_TOKEN"] = settings.hf_token
    if req.civitai_token is not None:
        settings.civitai_token = req.civitai_token.replace("\r", "").replace("\n", "")
        if settings.civitai_token:
            env["CIVITAI_TOKEN"] = settings.civitai_token
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
    if req.comfy_qwen_model is not None:
        env["COMFY_QWEN_MODEL"] = req.comfy_qwen_model
        settings.comfy_qwen_model = req.comfy_qwen_model
    if req.comfy_qwen_quant is not None:
        env["COMFY_QWEN_QUANT"] = req.comfy_qwen_quant
        settings.comfy_qwen_quant = req.comfy_qwen_quant
    if req.comfy_qwen_vision_model is not None:
        env["COMFY_QWEN_VISION_MODEL"] = req.comfy_qwen_vision_model
        settings.comfy_qwen_vision_model = req.comfy_qwen_vision_model
    if req.comfy_qwen_vision_quant is not None:
        env["COMFY_QWEN_VISION_QUANT"] = req.comfy_qwen_vision_quant
        settings.comfy_qwen_vision_quant = req.comfy_qwen_vision_quant
    if req.krea_comfy_warmup is not None:
        env["KREA_COMFY_WARMUP"] = (
            "true" if req.krea_comfy_warmup else "false"
        )
        settings.krea_comfy_warmup = req.krea_comfy_warmup
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
        engine = "native_gguf" if req.diffusion_engine == "gguf_external" else "native_int8_convrot" if req.diffusion_engine == "int8_convrot_external" else req.diffusion_engine
        env["DIFFUSION_ENGINE"] = engine
        settings.diffusion_engine = engine
    if req.gguf_turbo_path is not None:
        env["GGUF_TURBO_PATH"] = req.gguf_turbo_path
        settings.gguf_turbo_path = req.gguf_turbo_path
    if req.gguf_raw_path is not None:
        env["GGUF_RAW_PATH"] = req.gguf_raw_path
        settings.gguf_raw_path = req.gguf_raw_path
    if req.ideogram_api_key is not None:
        settings.ideogram_api_key = req.ideogram_api_key.replace("\r", "").replace("\n", "")
        if settings.ideogram_api_key:
            env["IDEOGRAM_API_KEY"] = settings.ideogram_api_key
    if req.openrouter_api_key is not None:
        settings.openrouter_api_key = req.openrouter_api_key.replace("\r", "").replace("\n", "")
        if settings.openrouter_api_key:
            env["OPENROUTER_API_KEY"] = settings.openrouter_api_key
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
    if req.krea2_vae_mode is not None:
        env["KREA2_VAE_MODE"] = req.krea2_vae_mode
        settings.krea2_vae_mode = req.krea2_vae_mode
    if req.krea2_vae_blend_radius is not None:
        env["KREA2_VAE_BLEND_RADIUS"] = str(int(req.krea2_vae_blend_radius))
        settings.krea2_vae_blend_radius = int(req.krea2_vae_blend_radius)
    if req.krea2_vae_blend_strength is not None:
        env["KREA2_VAE_BLEND_STRENGTH"] = str(float(req.krea2_vae_blend_strength))
        settings.krea2_vae_blend_strength = float(req.krea2_vae_blend_strength)
    if req.krea_attention_backend is not None:
        env["KREA_ATTENTION_BACKEND"] = req.krea_attention_backend
        settings.krea_attention_backend = req.krea_attention_backend
        # Native mmdit attention toggle is deprecated (ComfyUI owns attention via Sage/SDPA).
    if req.seedvr2_model is not None:
        val = "7b" if "7b" in str(req.seedvr2_model).lower() else "3b"
        env["SEEDVR2_MODEL"] = val
        settings.seedvr2_model = val
    _write_env(env)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Prompt expander
# ---------------------------------------------------------------------------


@app.post("/api/expand-prompt", status_code=202)
async def expand_prompt_endpoint(req: ExpandPromptRequest, request: Request):
    await _enforce_child_text(req.prompt, request, action="block_expand_input")
    backend = req.backend or ("gguf-server" if settings.local_llm_backend == "gguf_server" and settings.prompt_expander_backend == "local" else settings.prompt_expander_backend)
    return await _enqueue_helper_task(
        request,
        task_kind=PROMPT_EXPAND,
        summary=f"Magic Wand · {req.prompt.strip()[:80]}",
        payload={
            "prompt": req.prompt,
            "backend": backend,
            "suggest_moodboards": req.suggest_moodboards,
        },
    )


@app.get("/api/resolution-options")
async def resolution_options_endpoint():
    from resolution import resolution_options

    return resolution_options()


@app.get("/api/sampler-catalog")
async def sampler_catalog_endpoint(profile: str = "krea_turbo"):
    # ComfyUI is the only image engine — surface its live sampler/scheduler catalog.
    if comfy_available():
        try:
            from comfy_catalog import sampler_catalog as comfy_sampler_catalog
            return comfy_sampler_catalog(profile)
        except Exception:
            logger.exception("comfy sampler catalog failed")
    # Soft fallback when Comfy is down: a minimal static list (no native krea2 import).
    # Shape must match comfy_catalog: the UI expects scheduler OBJECTS, not strings.
    return {
        "profile": profile,
        "samplers": [
            {"id": "euler", "label": "euler", "scheduler": "simple", "default_steps": 8, "default_cfg": 0.0,
             "supported_schedulers": ["simple", "normal", "beta", "sgm_uniform"], "recommended_steps": 8,
             "disabled": False, "note": ""},
            {"id": "euler_ancestral", "label": "euler_ancestral", "scheduler": "simple", "default_steps": 8, "default_cfg": 0.0,
             "supported_schedulers": ["simple", "normal", "beta"], "recommended_steps": 8,
             "disabled": False, "note": ""},
        ],
        "schedulers": [
            {"id": s, "label": s, "recommended": s == "simple", "note": ""}
            for s in ("simple", "normal", "beta", "sgm_uniform")
        ],
        "recommended_combos": [],
        "note": "ComfyUI offline — showing a minimal fallback catalog.",
    }


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


@app.get("/api/gguf/status")
async def gguf_status_endpoint():
    fields = {
        "turbo_path": settings.gguf_turbo_path,
        "raw_path": settings.gguf_raw_path,
        "auto_checkpoint": settings.krea2_auto_checkpoint,
        "vae_path": settings.krea2_vae_path,
    }
    return {
        "diffusion_engine": settings.diffusion_engine,
        "quantization": settings.krea2_auto_quant,
        "paths": {
            key: {"path": value, "configured": bool(value)}
            for key, value in fields.items()
        },
    }


@app.get("/api/int8/status")
async def int8_status_endpoint():
    """INT8 status for the ComfyUI path (OTUNetLoaderW8A8), not the deprecated native DiT."""
    from quality_assets import asset_by_id, asset_status

    def _asset(asset_id: str, configured_path: str) -> dict:
        spec = asset_by_id(asset_id)
        item = asset_status(spec, has_hf_token=bool(_secret_value("HF_TOKEN", "hf_token")))
        path = Path(configured_path or item["local_path"])
        item["configured_path"] = str(path)
        item["installed"] = path.exists()
        return item

    return {
        "ok": True,
        "backend": "comfyui",
        "loader": "OTUNetLoaderW8A8",
        "diffusion_engine": "native_int8_convrot",
        "note": "Native Studio INT8 is deprecated. Turbo INT8 ConvRot checkpoints load in ComfyUI via OTUNetLoaderW8A8.",
        "assets": {
            "turbo": _asset("krea2_turbo_int8_convrot", settings.krea2_turbo_int8_path),
            "raw": _asset("krea2_raw_int8_convrot", settings.krea2_raw_int8_path),
        },
    }


@app.post("/api/int8/setup-native")
async def int8_setup_native_endpoint():
    """Kept for UI compatibility — downloads the default Turbo INT8 ConvRot for ComfyUI."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from quality_assets import asset_by_id, asset_installed, asset_status, download_asset

    required_ids = ["krea2_turbo_int8_convrot"]
    token = _secret_value("HF_TOKEN", "hf_token") or None
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
                logger.exception("INT8 asset download failed")
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
        # Must stay a member of the shared engine union: the UI writes this
        # straight into its settings draft.
        "diffusion_engine": "native_int8_convrot",
        "turbo_path": str(turbo),
        "quantization": "int8",
        "sampler": {"sampler": "euler", "scheduler": "simple", "steps": 8, "cfg": 0.0, "mu": 1.15},
        "warnings": [
            "Native Studio INT8 is deprecated. This download is for ComfyUI (OTUNetLoaderW8A8). "
            "Triton/SageAttention live in the ComfyUI venv and are used automatically.",
        ],
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
    from performance_guard import accelerator_status

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
    from performance_guard import accelerator_status

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


@app.post("/api/plan-prompt", status_code=202)
async def plan_prompt_endpoint(req: PlanPromptRequest, request: Request):
    await _enforce_child_text(req.prompt, request, action="block_plan_input")
    helper_backend = "gguf-server" if settings.local_llm_backend == "gguf_server" else "local"
    return await _enqueue_helper_task(
        request,
        task_kind=PROMPT_PLAN,
        summary=f"Prompt planner · {req.prompt.strip()[:80]}",
        payload={
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "backend": helper_backend,
        },
    )


@app.get("/api/prompt-recipes", response_model=PromptRecipeListResponse)
async def prompt_recipes_list_endpoint(request: Request):
    username, _role, _is_admin = _request_user_role(request)
    return PromptRecipeListResponse(items=[
        PromptRecipe(**{k: v for k, v in item.items() if k != "owner"})
        for item in list_recipes(username=username)
    ])


@app.post("/api/prompt-recipes", response_model=PromptRecipe)
async def prompt_recipes_save_endpoint(req: PromptRecipe, request: Request):
    username, _role, _is_admin = _request_user_role(request)
    saved = save_recipe(req.model_dump(), username=username)
    return PromptRecipe(**{k: v for k, v in saved.items() if k != "owner"})


@app.delete("/api/prompt-recipes/{recipe_id}")
async def prompt_recipes_delete_endpoint(recipe_id: str, request: Request):
    username, _role, is_admin = _request_user_role(request)
    return {"ok": delete_recipe(recipe_id, username=username, is_admin=is_admin)}


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
            # Never cache index.html: it references content-hashed asset files, so a
            # stale cached copy would keep loading the OLD bundle after a rebuild.
            # (The hashed /assets files stay immutable/cacheable.)
            return FileResponse(str(index), headers={"Cache-Control": "no-store, must-revalidate"})
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
