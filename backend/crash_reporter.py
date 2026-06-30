from __future__ import annotations

import faulthandler
import hashlib
import json
import time
from pathlib import Path
from typing import Any

_FAULT_FILE = None


def _safe_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        value = vars(value)
    if isinstance(value, dict):
        return {str(k): _safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(v) for v in value]
    if isinstance(value, str):
        # Avoid dumping base64 images into crash breadcrumbs.
        if len(value) > 512:
            return {
                "redacted": True,
                "length": len(value),
                "sha256": hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest(),
            }
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def enable_fault_logging(logs_dir: str | Path) -> Path:
    """Enable Python faulthandler so native crashes leave a stack dump."""
    global _FAULT_FILE
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "python-faulthandler.log"
    if _FAULT_FILE is None or _FAULT_FILE.closed:
        _FAULT_FILE = path.open("a", encoding="utf-8")
    try:
        faulthandler.enable(file=_FAULT_FILE, all_threads=True)
    except Exception:
        # Fault logging is best-effort; never block app startup.
        pass
    return path


def disable_fault_logging() -> None:
    """Disable and close the faulthandler file. Intended for tests/shutdown."""
    global _FAULT_FILE
    try:
        faulthandler.disable()
    except Exception:
        pass
    if _FAULT_FILE is not None and not _FAULT_FILE.closed:
        _FAULT_FILE.close()
    _FAULT_FILE = None


def breadcrumb_path(logs_dir: str | Path, job_id: str) -> Path:
    return Path(logs_dir) / f"active-generation-{job_id}.json"


def write_generation_breadcrumb(
    logs_dir: str | Path,
    *,
    job_id: str,
    req: Any,
    stage: str,
    extra: dict | None = None,
) -> Path:
    path = breadcrumb_path(logs_dir, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request": _safe_value(req),
        "extra": _safe_value(extra or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def clear_generation_breadcrumb(logs_dir: str | Path, *, job_id: str) -> None:
    try:
        breadcrumb_path(logs_dir, job_id).unlink()
    except FileNotFoundError:
        pass


def stale_generation_breadcrumbs(logs_dir: str | Path) -> list[dict]:
    items: list[dict] = []
    for path in sorted(Path(logs_dir).glob("active-generation-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {"job_id": path.stem.replace("active-generation-", ""), "stage": "unreadable"}
        data["path"] = str(path)
        items.append(data)
    return items
