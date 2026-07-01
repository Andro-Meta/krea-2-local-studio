from __future__ import annotations

import gc
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Any

from system_check import get_gpu_process_details, mem_snapshot


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")).replace("\\", "/").lower()


def clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        # Memory cleanup should never make the server less stable.
        pass


def release_transient_pipeline_memory(pipeline: Any, *, clear_conditioning_cache: bool = True, unload_helpers: bool = True) -> dict[str, Any]:
    before = mem_snapshot()
    helper_unloaded = False
    if unload_helpers:
        try:
            from prompt_expander import unload_local_qwen

            unload_local_qwen()
            helper_unloaded = True
        except Exception:
            helper_unloaded = False
    pid_unloaded = False
    if unload_helpers:
        try:
            from pid_decoder_provider import release_pid_runtime

            pid_unloaded = bool(release_pid_runtime().get("released"))
        except Exception:
            pid_unloaded = False
    if hasattr(pipeline, "release_transient_memory"):
        result = pipeline.release_transient_memory(clear_conditioning_cache=clear_conditioning_cache)
        result["safe_clean"] = True
        result["helper_unloaded"] = helper_unloaded
        result["pid_unloaded"] = pid_unloaded
        result["before"] = before
        result["after"] = result.get("memory", mem_snapshot())
        return result
    encoder = getattr(pipeline, "encoder", None)
    if encoder is not None and hasattr(encoder, "cpu"):
        encoder.cpu()
    cache = getattr(pipeline, "_conditioning_cache", None)
    cache_entries = len(cache) if hasattr(cache, "__len__") else 0
    if clear_conditioning_cache and hasattr(cache, "clear"):
        cache.clear()
    clear_cuda_cache()
    return {
        "released": True,
        "safe_clean": True,
        "helper_unloaded": helper_unloaded,
        "pid_unloaded": pid_unloaded,
        "encoder_offloaded": encoder is not None,
        "cleared_conditioning_entries": cache_entries if clear_conditioning_cache else 0,
        "before": before,
        "after": mem_snapshot(),
        "memory": mem_snapshot(),
    }


def safe_clean_memory(pipeline: Any, *, clear_conditioning_cache: bool = True) -> dict[str, Any]:
    return release_transient_pipeline_memory(
        pipeline,
        clear_conditioning_cache=clear_conditioning_cache,
        unload_helpers=True,
    )


def prepare_for_generation(pipeline: Any, *, clear_conditioning_cache: bool = False) -> dict[str, Any]:
    """Lightweight pre-generation cleanup: evict helpers, keep Krea model hot."""
    return safe_clean_memory(pipeline, clear_conditioning_cache=clear_conditioning_cache)


def unload_pipeline(pipeline: Any) -> dict[str, Any]:
    if hasattr(pipeline, "unload"):
        pipeline.unload()
    clear_cuda_cache()
    return {"unloaded": True, "memory": mem_snapshot()}


def _process_rows_from_windows() -> list[dict[str, Any]]:
    if sys.platform != "win32":
        return []
    command = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if not proc.stdout.strip():
        return []
    data = json.loads(proc.stdout)
    return data if isinstance(data, list) else [data]


def _process_rows_from_ps() -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, cmd = stripped.partition(" ")
        if pid_text.isdigit():
            rows.append({"ProcessId": int(pid_text), "CommandLine": cmd})
    return rows


def _all_python_process_rows() -> list[dict[str, Any]]:
    try:
        return _process_rows_from_windows() or _process_rows_from_ps()
    except Exception:
        return []


def _extract_port(command_line: str) -> int | None:
    match = re.search(r"(?:--port\s+|--port=)(\d+)", command_line)
    return int(match.group(1)) if match else None


def _ancestor_pids(rows: list[dict[str, Any]], pid: int) -> set[int]:
    parents = {
        int(row.get("ProcessId") or row.get("pid") or 0): int(row.get("ParentProcessId") or row.get("ppid") or 0)
        for row in rows
        if int(row.get("ProcessId") or row.get("pid") or 0)
    }
    ancestors: set[int] = {pid}
    parent = parents.get(pid)
    while parent and parent not in ancestors:
        ancestors.add(parent)
        parent = parents.get(parent)
    return ancestors


def _runtime_kind(command_line: str) -> str | None:
    normalized = command_line.replace("\\", "/").lower()
    if "uvicorn" in normalized and "backend.main:app" in normalized:
        return "server"
    if "scripts/share_startup.py" in normalized:
        return "startup_helper"
    if "krea_share_control.pyw" in normalized:
        return "share_control"
    if PROJECT_ROOT in normalized and ("/run.bat" in normalized or "/krea_studio_sharing.bat" in normalized):
        return "launcher"
    return None


def detect_krea_server_processes() -> list[dict[str, Any]]:
    return [
        {key: value for key, value in proc.items() if key != "kind"}
        for proc in detect_krea_runtime_processes()
        if proc.get("kind") == "server"
    ]


def detect_krea_runtime_processes() -> list[dict[str, Any]]:
    current_pid = os.getpid()
    gpu_by_pid = {int(proc["pid"]): proc for proc in get_gpu_process_details() if str(proc.get("pid", "")).isdigit()}
    rows = _all_python_process_rows()
    skip_pids = _ancestor_pids(rows, current_pid)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        pid = int(row.get("ProcessId") or row.get("pid") or 0)
        command_line = str(row.get("CommandLine") or row.get("command_line") or "")
        if not pid or pid in skip_pids:
            continue
        kind = _runtime_kind(command_line)
        if kind is None:
            continue
        port = _extract_port(command_line)
        gpu = gpu_by_pid.get(pid, {})
        candidates.append({
            "pid": pid,
            "kind": kind,
            "port": port,
            "command_line": command_line,
            "used_memory_gb": gpu.get("used_memory_gb"),
            "can_stop": True,
        })
    return candidates


def _terminate_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=10)
    else:
        os.kill(pid, signal.SIGTERM)


def cleanup_krea_runtime_processes(*, wait_seconds: float = 20.0, poll_seconds: float = 0.5) -> dict[str, Any]:
    stopped_pids: list[int] = []
    for proc in detect_krea_runtime_processes():
        pid = int(proc["pid"])
        try:
            _terminate_pid(pid)
            stopped_pids.append(pid)
        except Exception:
            continue

    deadline = time.time() + wait_seconds
    remaining = detect_krea_runtime_processes()
    while remaining and time.time() < deadline:
        time.sleep(poll_seconds)
        remaining = detect_krea_runtime_processes()

    clear_cuda_cache()
    return {
        "stopped_pids": stopped_pids,
        "remaining": remaining,
        "memory": mem_snapshot(),
    }


def stop_krea_server_process(pid: int) -> dict[str, Any]:
    pid = int(pid)
    match = next((proc for proc in detect_krea_server_processes() if int(proc["pid"]) == pid), None)
    if match is None:
        raise ValueError(f"PID {pid} is not a detected Krea server process.")
    try:
        _terminate_pid(pid)
    except Exception as exc:
        raise RuntimeError(f"Could not stop Krea server process {pid}: {exc}") from exc
    clear_cuda_cache()
    return {"stopped": True, "pid": pid, "port": match.get("port"), "memory": mem_snapshot()}
