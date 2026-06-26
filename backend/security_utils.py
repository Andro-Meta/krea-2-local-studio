from __future__ import annotations

import re
from pathlib import Path, PurePath
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ALLOWED_DOWNLOAD_HOSTS = {"huggingface.co", "civitai.com", "www.civitai.com"}


def safe_child_file(root: Path, filename: str) -> Path:
    name = filename.strip()
    if not name or PurePath(name).name != name or "/" in name or "\\" in name or not _SAFE_FILENAME_RE.fullmatch(name):
        raise ValueError("Invalid filename")
    root_resolved = root.resolve()
    path = (root_resolved / name).resolve()
    if path.parent != root_resolved:
        raise ValueError("Invalid filename")
    return path


def normalize_lora_import_url(raw_url: str) -> str:
    parsed = urlparse(raw_url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in _ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError("LoRA imports must use https://huggingface.co or https://civitai.com URLs.")

    if host == "huggingface.co" and "/blob/" in parsed.path:
        path = parsed.path.replace("/blob/", "/resolve/", 1)
        return urlunparse(parsed._replace(path=path))

    if host in {"civitai.com", "www.civitai.com"} and re.fullmatch(r"/models/\d+", parsed.path):
        version = parse_qs(parsed.query).get("modelVersionId", [""])[0]
        if not version.isdigit():
            raise ValueError("CivitAI model URLs must include a numeric modelVersionId.")
        return f"https://civitai.com/api/download/models/{version}"

    return urlunparse(parsed)


def is_civitai_url(url: str) -> bool:
    return (urlparse(url).hostname or "").lower() in {"civitai.com", "www.civitai.com"}


def safe_lora_filename(requested: str, source_url: str) -> str:
    name = requested.strip()
    if not name:
        name = Path(urlparse(source_url).path).name
    if not name.endswith(".safetensors"):
        name += ".safetensors"
    if PurePath(name).name != name or "/" in name or "\\" in name or not _SAFE_FILENAME_RE.fullmatch(name):
        raise ValueError("Invalid LoRA filename")
    return name


def append_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query[key] = [value]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
