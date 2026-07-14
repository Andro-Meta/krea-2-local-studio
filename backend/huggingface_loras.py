"""Hugging Face browse + install for Krea 2 LoRAs.

Lists Hub models tagged as LoRA and filtered to Krea base models, then
downloads a chosen weight file into models/loras via huggingface_hub.
"""
from __future__ import annotations

import logging
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import requests

try:
    from settings import LORAS_DIR, BASE_DIR
except Exception:  # pragma: no cover
    BASE_DIR = Path(__file__).resolve().parent.parent
    LORAS_DIR = BASE_DIR / "models" / "loras"

logger = logging.getLogger("krea2.huggingface_loras")

HF_API = "https://huggingface.co/api/models"
HF_THUMB = "https://cdn-thumbnails.huggingface.co/social-thumbnails/models"
_UA = {"User-Agent": "krea2-studio/1.0"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_WEIGHT_EXT = (".safetensors", ".pt", ".bin", ".ckpt")
_SKIP_SUBSTR = ("optimizer", "scheduler", "rng_state", "training_args", "pytorch_model")
_KREA_TAG_RE = re.compile(r"^base_model:krea/", re.I)
_install_lock = threading.Lock()

_SORT_MAP = {
    "Most Downloaded": "downloads",
    "downloads": "downloads",
    "Most Liked": "likes",
    "likes": "likes",
    "Newest": "lastModified",
    "lastModified": "lastModified",
}


class MultiFileRequired(Exception):
    """Raised when a repo has multiple weight files and none was chosen."""

    def __init__(self, repo_id: str, files: list[dict[str, Any]]):
        self.repo_id = repo_id
        self.files = files
        super().__init__(f"Multiple weight files in {repo_id}; choose one.")


def _is_krea_lora(item: dict) -> bool:
    tags = [str(t) for t in (item.get("tags") or [])]
    if any(_KREA_TAG_RE.match(t) for t in tags):
        return True
    rid = str(item.get("id") or item.get("modelId") or "").lower()
    blob = " ".join(tags).lower() + " " + rid
    return "krea-2" in blob or "krea2" in blob or "/krea/" in blob


def _base_model_from_tags(tags: list[str]) -> str:
    for t in tags:
        if t.lower().startswith("base_model:krea/"):
            return t.split(":", 1)[-1]
    return "Krea 2"


def _preview_url(repo_id: str) -> str:
    return f"{HF_THUMB}/{repo_id}.png"


def _hf_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


def _normalize_item(raw: dict) -> dict:
    repo_id = str(raw.get("id") or raw.get("modelId") or "")
    tags = [str(t) for t in (raw.get("tags") or [])]
    owner = repo_id.split("/", 1)[0] if "/" in repo_id else ""
    name = repo_id.split("/", 1)[-1] if repo_id else ""
    return {
        "repo_id": repo_id,
        "name": name or repo_id,
        "creator": owner,
        "base_model": _base_model_from_tags(tags),
        "tags": tags,
        "downloads": int(raw.get("downloads") or 0),
        "likes": int(raw.get("likes") or 0),
        "preview_url": _preview_url(repo_id) if repo_id else "",
        "hf_url": _hf_url(repo_id) if repo_id else "",
        "pipeline_tag": raw.get("pipeline_tag") or "",
        "installed": False,
        "installed_filename": "",
        "weight_files": [],
    }


def _parse_next_cursor(link_header: str) -> Optional[str]:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' not in part and "rel=next" not in part:
            continue
        m = re.search(r"<([^>]+)>", part)
        if not m:
            continue
        qs = parse_qs(urlparse(m.group(1)).query)
        cursors = qs.get("cursor") or []
        return cursors[0] if cursors else None
    return None


def _installed_match(repo_id: str, weight_names: list[str] | None = None) -> Path | None:
    """Return an existing local file that looks like it came from this repo."""
    if not LORAS_DIR.exists():
        return None
    owner, _, name = repo_id.partition("/")
    prefixes = []
    if owner and name:
        prefixes.append(f"{owner}__{name}__")
        prefixes.append(f"{_SAFE_NAME.sub('_', owner)}__{_SAFE_NAME.sub('_', name)}__")
    candidates = list(LORAS_DIR.glob("*.safetensors")) + list(LORAS_DIR.glob("*.pt")) + list(LORAS_DIR.glob("*.bin"))
    for path in candidates:
        low = path.name.lower()
        for pref in prefixes:
            if path.name.startswith(pref):
                return path
        if weight_names:
            for wn in weight_names:
                base = Path(wn).name.lower()
                if low == base or low.endswith("__" + base):
                    return path
    return None


def list_weight_files(repo_id: str, token: str | None = None) -> list[dict[str, Any]]:
    """List downloadable weight files in a Hub repo."""
    headers = dict(_UA)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"{HF_API}/{repo_id}", headers=headers, timeout=30)
    if r.status_code in {401, 403}:
        raise PermissionError(
            "Hugging Face requires a valid token to access this model. Add or update HF_TOKEN in Settings."
        )
    r.raise_for_status()
    siblings = r.json().get("siblings") or []
    out: list[dict[str, Any]] = []
    for sib in siblings:
        fn = str(sib.get("rfilename") or "")
        low = fn.lower()
        if not low.endswith(_WEIGHT_EXT):
            continue
        if any(s in low for s in _SKIP_SUBSTR):
            continue
        out.append({"filename": fn, "size": sib.get("size")})
    return out


def huggingface_browse(
    query: str = "",
    sort: str = "downloads",
    cursor: str | None = None,
    limit: int = 48,
    token: str | None = None,
) -> dict:
    """Browse Hub LoRAs filtered toward Krea 2 base models."""
    limit = max(1, min(int(limit or 48), 100))
    sort_key = _SORT_MAP.get(sort or "downloads", "downloads")
    search = (query or "").strip() or "krea-2"
    # Nudge free-text searches toward Krea when the user didn't already include it.
    if query.strip() and "krea" not in search.lower():
        search = f"{search} krea"

    params: dict[str, Any] = {
        "search": search,
        "filter": "lora",
        "sort": sort_key,
        "direction": -1,
        "limit": limit,
    }
    if cursor:
        params["cursor"] = cursor

    headers = dict(_UA)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(HF_API, params=params, headers=headers, timeout=30)
    if r.status_code in {401, 403}:
        raise PermissionError(
            "Hugging Face requires a valid token for this search. Add or update HF_TOKEN in Settings."
        )
    r.raise_for_status()
    payload = r.json()
    raw_items = payload if isinstance(payload, list) else []
    next_cursor = _parse_next_cursor(r.headers.get("Link") or "")

    items = [_normalize_item(it) for it in raw_items if _is_krea_lora(it)]
    for item in items:
        installed = _installed_match(item["repo_id"])
        item["installed"] = bool(installed)
        item["installed_filename"] = installed.name if installed else ""

    return {
        "items": items,
        "next_cursor": next_cursor,
        "has_more": bool(next_cursor),
        "metadata": {"next_cursor": next_cursor, "sort": sort_key, "limit": limit},
    }


def _safe_dest_name(repo_id: str, filename: str) -> str:
    base = _SAFE_NAME.sub("_", Path(filename).name)
    if not base.lower().endswith(_WEIGHT_EXT):
        base += ".safetensors"
    owner, _, name = repo_id.partition("/")
    prefix = _SAFE_NAME.sub("_", f"{owner}__{name}__") if owner else "hf__"
    if base.startswith(prefix):
        return base
    return f"{prefix}{base}"


def huggingface_install(repo_id: str, filename: str | None = None, token: str | None = None) -> dict:
    """Download a Hub LoRA weight into models/loras. Returns install metadata."""
    repo_id = (repo_id or "").strip().strip("/")
    if not repo_id or "/" not in repo_id or ".." in repo_id:
        raise ValueError("repo_id must look like owner/name.")

    weights = list_weight_files(repo_id, token=token)
    if not weights:
        raise RuntimeError(f"No LoRA weight files found in {repo_id}.")

    chosen = (filename or "").strip()
    if not chosen:
        if len(weights) == 1:
            chosen = weights[0]["filename"]
        else:
            raise MultiFileRequired(repo_id, weights)
    elif chosen not in {w["filename"] for w in weights}:
        raise ValueError(f"File {chosen!r} is not a weight file in {repo_id}.")

    dest_name = _safe_dest_name(repo_id, chosen)
    LORAS_DIR.mkdir(parents=True, exist_ok=True)
    dest = LORAS_DIR / dest_name

    with _install_lock:
        existing = _installed_match(repo_id, [chosen])
        if existing and existing.exists():
            return _install_result(existing, already=True, repo_id=repo_id)
        if dest.exists():
            return _install_result(dest, already=True, repo_id=repo_id)

        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            repo_id=repo_id,
            filename=chosen,
            token=token or None,
        )
        cached_path = Path(cached)
        if cached_path.resolve() != dest.resolve():
            shutil.copyfile(cached_path, dest)

    return _install_result(dest, already=False, repo_id=repo_id)


def _install_result(path: Path, *, already: bool, repo_id: str) -> dict:
    from lora_manager import inspect_lora

    verdict = inspect_lora(path) if path.exists() else {"compatible": None, "reason": ""}
    return {
        "ok": True,
        "already_installed": already,
        "filename": path.name,
        "path": str(path),
        "repo_id": repo_id,
        "compatible": verdict.get("compatible"),
        "match_info": verdict.get("reason") or verdict.get("match_info") or "",
        "hf_url": _hf_url(repo_id),
    }
