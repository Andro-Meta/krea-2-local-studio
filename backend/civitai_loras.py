"""Civitai integration for LoRAs.

- Enriches locally-installed LoRAs with Civitai metadata (real model name,
  trigger words, description, base model, preview) looked up by file SHA256.
- Browses/searches Civitai filtered to Krea 2 LoRA + LoKr (LoCon) models.
- Installs a chosen Civitai LoRA version into models/loras.

Reads (by-hash, browse) need no token. Downloads may require a Civitai API
token (settings.civitai_token). Everything is cached to data/ so hashing and
network lookups happen once.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

import requests

try:
    from settings import LORAS_DIR, BASE_DIR
except Exception:  # pragma: no cover
    BASE_DIR = Path(__file__).resolve().parent.parent
    LORAS_DIR = BASE_DIR / "models" / "loras"

logger = logging.getLogger("krea2.civitai")

CIVITAI_API = "https://civitai.com/api/v1"
CACHE_PATH = BASE_DIR / "data" / "civitai_lora_cache.json"
KREA2_BASE_MODEL = "Krea 2"
_UA = {"User-Agent": "krea2-studio/1.0"}

_cache_lock = threading.Lock()
_install_lock = threading.Lock()
_scan_state: dict[str, Any] = {"scanning": False, "total": 0, "done": 0, "updated": 0}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        data.setdefault("hashes", {})
        data.setdefault("meta", {})
        return data
    except Exception:
        return {"hashes": {}, "meta": {}}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def _pathkey(p: Path) -> str:
    st = p.stat()
    return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"


def _sha256_file(p: Path) -> str:
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _strip_html(text: str, limit: int = 600) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:limit]


# Names that indicate explicit content even when Civitai's model.nsfw flag is
# mislabeled (e.g. "[KREA 2] Mystic XXX").
_EXPLICIT_RE = re.compile(r"(?:\b|_)(?:xxx|nsfw|nude|nudes|naked|porn|hentai|explicit|onlyfans|nudity|sex|boob|nipple|topless|lewd|erotic)(?:\b|_)", re.I)

# Civitai image nsfwLevel: 1=PG/None, 2=Soft(PG13), 4=Mature(R), 8/16/32=X/XXX.
# Only level<=1 is guaranteed nudity-free.
_SAFE_NSFW_LEVEL = 1


def _preview_from_images(images: list, max_level: int = _SAFE_NSFW_LEVEL) -> str:
    """Return the first SFW image URL (nsfwLevel <= max_level). No unsafe fallback."""
    for im in images or []:
        if (im.get("type") or "image") != "image" or not im.get("url"):
            continue
        lvl = im.get("nsfwLevel")
        lvl = 0 if lvl is None else int(lvl)
        if lvl <= max_level:
            return im["url"]
    return ""


def _is_explicit(name: str, model_nsfw: bool) -> bool:
    return bool(model_nsfw) or bool(_EXPLICIT_RE.search(name or ""))


def _normalize_version(v: dict) -> dict:
    model = v.get("model") or {}
    files = v.get("files") or []
    return {
        "civitai_name": model.get("name") or v.get("name") or "",
        "version_name": v.get("name") or "",
        "description": _strip_html(model.get("description") or v.get("description") or ""),
        "trigger_words": [w for w in (v.get("trainedWords") or []) if w],
        "base_model": v.get("baseModel") or "",
        "model_id": model.get("id") or v.get("modelId"),
        "version_id": v.get("id"),
        "type": model.get("type") or "",
        "nsfw": bool(model.get("nsfw")),
        "preview_url": ("" if _is_explicit(model.get("name") or "", bool(model.get("nsfw")))
                        else _preview_from_images(v.get("images") or [])),
        "civitai_url": f"https://civitai.com/models/{model.get('id')}" if model.get("id") else "",
        "download_url": (files[0].get("downloadUrl") if files else "")
        or (f"https://civitai.com/api/download/models/{v.get('id')}" if v.get("id") else ""),
    }


def _normalize_model_item(it: dict) -> dict:
    versions = it.get("modelVersions") or [{}]
    v = versions[0]
    files = v.get("files") or []
    f0 = files[0] if files else {}
    stats = it.get("stats") or {}
    name = it.get("name", "")
    explicit = _is_explicit(name, bool(it.get("nsfw")))
    # SFW-only thumbnail; explicit/NSFW models never show a preview image.
    preview = "" if explicit else _preview_from_images(v.get("images") or [])
    return {
        "model_id": it.get("id"),
        "version_id": v.get("id"),
        "name": name,
        "type": it.get("type", ""),
        "creator": (it.get("creator") or {}).get("username", ""),
        "base_model": v.get("baseModel", ""),
        "version_name": v.get("name", ""),
        "trigger_words": [w for w in (v.get("trainedWords") or []) if w],
        "description": _strip_html(it.get("description") or ""),
        "nsfw": explicit,
        "preview_url": preview,
        "download_url": f0.get("downloadUrl") or (f"https://civitai.com/api/download/models/{v.get('id')}" if v.get("id") else ""),
        "file_name": f0.get("name", ""),
        "file_size_kb": f0.get("sizeKB"),
        "downloads": stats.get("downloadCount"),
        "thumbsUp": stats.get("thumbsUpCount"),
        "civitai_url": f"https://civitai.com/models/{it.get('id')}",
    }


# ---------------------------------------------------------------------------
# By-hash enrichment
# ---------------------------------------------------------------------------

def civitai_by_hash(sha256: str, token: str | None = None) -> Optional[dict]:
    headers = dict(_UA)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"{CIVITAI_API}/model-versions/by-hash/{sha256}", headers=headers, timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return _normalize_version(r.json())


def _merge_meta(lora: dict, meta: dict) -> None:
    lora["civitai"] = meta
    if not lora.get("trigger_words"):
        lora["trigger_words"] = meta.get("trigger_words", [])
    generic = {None, "", lora.get("name"), str(lora.get("name", "")).replace("_", " ").title()}
    if meta.get("civitai_name") and lora.get("display_name") in generic:
        lora["display_name"] = meta["civitai_name"]
    lora["preview_url"] = meta.get("preview_url", "")
    lora["civitai_url"] = meta.get("civitai_url", "")
    lora["base_model"] = meta.get("base_model", "")
    lora["description"] = meta.get("description", "")


def _cached_installed_path(version_id: int, cache: dict | None = None) -> Path | None:
    cache = cache or _load_cache()
    for sha, meta in (cache.get("meta") or {}).items():
        if not meta or int(meta.get("version_id") or 0) != int(version_id):
            continue
        for key, cached_sha in (cache.get("hashes") or {}).items():
            if cached_sha != sha:
                continue
            filename = key.split(":", 1)[0]
            path = LORAS_DIR / filename
            if path.exists():
                return path
    return None


def _remember_installed_version(path: Path, version: dict) -> None:
    try:
        sha = _sha256_file(path)
        key = _pathkey(path)
    except OSError:
        return
    with _cache_lock:
        cache = _load_cache()
        cache["hashes"][key] = sha
        cache["meta"][sha] = _normalize_version(version)
        _save_cache(cache)


def _installed_path_for_item(item: dict, cache: dict | None = None) -> Path | None:
    version_id = int(item.get("version_id") or 0)
    if version_id:
        cached = _cached_installed_path(version_id, cache)
        if cached:
            return cached
    filename = item.get("file_name") or ""
    if not filename:
        return None
    path = LORAS_DIR / _safe_filename(filename, version_id)
    return path if path.exists() else None


def enrich_loras(loras: list[dict], *, fetch: bool = False, token: str | None = None) -> list[dict]:
    """Merge cached Civitai metadata into a list of lora dicts.

    fetch=False: only merge already-cached metadata (fast, for GET /api/loras).
    fetch=True:  hash + query Civitai for anything uncached, updating the cache.
    """
    with _cache_lock:
        cache = _load_cache()
    changed = False
    for lora in loras:
        fn = lora.get("filename")
        if not fn or not lora.get("installed"):
            continue
        p = LORAS_DIR / fn
        if not p.exists():
            continue
        try:
            key = _pathkey(p)
        except OSError:
            continue
        sha = cache["hashes"].get(key)
        if not sha and fetch:
            try:
                sha = _sha256_file(p)
                cache["hashes"][key] = sha
                changed = True
            except OSError:
                continue
        if not sha:
            continue
        if sha not in cache["meta"]:
            if not fetch:
                continue
            try:
                cache["meta"][sha] = civitai_by_hash(sha, token)
            except Exception:
                logger.debug("by-hash lookup failed for %s", fn, exc_info=True)
                cache["meta"][sha] = None
            changed = True
            time.sleep(0.15)
        meta = cache["meta"].get(sha)
        if meta:
            _merge_meta(lora, meta)
    if changed:
        with _cache_lock:
            _save_cache(cache)
    return loras


def scan_state() -> dict:
    return dict(_scan_state)


def scan_all(loras_provider, token: str | None = None) -> None:
    """Background full scan: hash + Civitai-enrich every installed lora."""
    if _scan_state["scanning"]:
        return
    loras = [l for l in loras_provider() if l.get("installed")]
    _scan_state.update(scanning=True, total=len(loras), done=0, updated=0)
    try:
        with _cache_lock:
            cache = _load_cache()
        for lora in loras:
            fn = lora.get("filename")
            p = LORAS_DIR / fn if fn else None
            if p and p.exists():
                try:
                    key = _pathkey(p)
                    sha = cache["hashes"].get(key) or _sha256_file(p)
                    cache["hashes"][key] = sha
                    if sha not in cache["meta"]:
                        cache["meta"][sha] = civitai_by_hash(sha, token)
                        if cache["meta"][sha]:
                            _scan_state["updated"] += 1
                        time.sleep(0.15)
                except Exception:
                    logger.debug("scan failed for %s", fn, exc_info=True)
            _scan_state["done"] += 1
            with _cache_lock:
                _save_cache(cache)
    finally:
        _scan_state["scanning"] = False


# ---------------------------------------------------------------------------
# Browse + install
# ---------------------------------------------------------------------------

def civitai_browse(query: str = "", page: int = 1, sort: str = "Most Downloaded",
                   nsfw: bool = False, token: str | None = None) -> dict:
    params = {
        "types": ["LORA", "LoCon"],
        "baseModels": [KREA2_BASE_MODEL],
        "limit": 24,
        "page": max(1, int(page or 1)),
        "sort": sort or "Most Downloaded",
        "nsfw": "true" if nsfw else "false",
    }
    if query:
        params["query"] = query
    headers = dict(_UA)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"{CIVITAI_API}/models", params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    items = [_normalize_model_item(it) for it in (data.get("items") or [])]
    with _cache_lock:
        cache = _load_cache()
    for item in items:
        installed_path = _installed_path_for_item(item, cache)
        item["installed"] = bool(installed_path)
        item["installed_filename"] = installed_path.name if installed_path else ""
    if not nsfw:
        # Drop explicit / NSFW-flagged models entirely from the SFW browse
        # (Civitai's model.nsfw is unreliable, so we also check the name).
        items = [it for it in items if not it["nsfw"]]
    return {"items": items, "metadata": data.get("metadata", {})}


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_filename(name: str, version_id: int) -> str:
    name = (name or "").strip() or f"civitai_{version_id}.safetensors"
    name = _SAFE_NAME.sub("_", Path(name).name)
    if not name.lower().endswith((".safetensors", ".ckpt", ".pt")):
        name += ".safetensors"
    return name


def civitai_install(version_id: int, token: str | None = None, filename: str | None = None) -> dict:
    """Download a Civitai LoRA version into models/loras. Returns {filename,path}."""
    headers = dict(_UA)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    vr = requests.get(f"{CIVITAI_API}/model-versions/{version_id}", headers=headers, timeout=30)
    if vr.status_code in {401, 403}:
        raise PermissionError("Civitai requires a valid API token to access this model. Add or update your Civitai API token in Settings.")
    vr.raise_for_status()
    v = vr.json()
    files = v.get("files") or []
    f0 = next((f for f in files if f.get("primary")), files[0] if files else None)
    if not f0:
        raise RuntimeError("No downloadable file on that Civitai version.")
    fname = _safe_filename(filename or f0.get("name"), version_id)
    dl = f0.get("downloadUrl") or f"https://civitai.com/api/download/models/{version_id}"
    if token:
        dl += ("&" if "?" in dl else "?") + f"token={token}"

    LORAS_DIR.mkdir(parents=True, exist_ok=True)
    dest = LORAS_DIR / fname
    with _install_lock:
        installed_path = _cached_installed_path(version_id)
        if installed_path:
            _remember_installed_version(installed_path, v)
            return {"ok": True, "already_installed": True, "filename": installed_path.name, "path": str(installed_path),
                    "trigger_words": [w for w in (v.get("trainedWords") or []) if w]}
        if dest.exists():
            _remember_installed_version(dest, v)
            return {"ok": True, "already_installed": True, "filename": fname, "path": str(dest),
                    "trigger_words": [w for w in (v.get("trainedWords") or []) if w]}

        tmp = dest.with_suffix(f"{dest.suffix}.{threading.get_ident()}.part")
        tmp.unlink(missing_ok=True)
        try:
            with requests.get(dl, headers=headers, stream=True, timeout=600, allow_redirects=True) as resp:
                if resp.status_code in {401, 403}:
                    raise PermissionError("Civitai requires a valid API token to download this model. Add or update your Civitai API token in Settings.")
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                first = b""
                with open(tmp, "wb") as out:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if not first:
                            first = chunk[:16]
                            # Civitai returns an HTML login page (not a token) when auth is required.
                            if b"<!doctype" in first.lower() or "text/html" in ctype.lower():
                                raise PermissionError(
                                    "Civitai requires an API token to download this model. Add your token in Settings."
                                )
                        out.write(chunk)
            tmp.replace(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        _remember_installed_version(dest, v)
    return {"ok": True, "already_installed": False, "filename": fname, "path": str(dest),
            "trigger_words": [w for w in (v.get("trainedWords") or []) if w]}
