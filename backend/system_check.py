"""Hardware pre-flight checks for Krea 2 Studio."""
from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

VARIANT_REQS: dict[str, dict[str, Any]] = {
    "turbo_bf16": {
        "vram_gb": 20.0, "ram_gb": 16.0,
        "label": "Krea 2 Turbo bf16 (needs ~20 GB VRAM with encoder offload)",
    },
    "turbo_fp8": {
        "vram_gb": 13.0, "ram_gb": 12.0,
        "label": "Krea 2 Turbo fp8 (needs ~13 GB VRAM)",
    },
    "raw_bf16": {
        "vram_gb": 20.0, "ram_gb": 16.0,
        "label": "Krea 2 RAW bf16 (needs ~20 GB VRAM with encoder offload)",
    },
}


def get_ram_gb() -> tuple[float | None, float | None]:
    """Returns (total_gb, available_gb)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            gib = 1024 ** 3
            return stat.ullTotalPhys / gib, stat.ullAvailPhys / gib
        else:
            import psutil
            vm = psutil.virtual_memory()
            gib = 1024 ** 3
            return vm.total / gib, vm.available / gib
    except Exception:
        return None, None


def get_gpu_info() -> tuple[str | None, float | None, float | None]:
    """Returns (name, total_gb, free_gb)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None, None
        props = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        gib = 1024 ** 3
        return props.name, total_b / gib, free_b / gib
    except Exception:
        return None, None, None


def mem_snapshot() -> str:
    parts: list[str] = []
    _, ram_avail = get_ram_gb()
    if ram_avail is not None:
        parts.append(f"RAM {ram_avail:.1f}GB free")
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(0)
            gib = 1024 ** 3
            parts.append(f"VRAM {free_b / gib:.1f}/{total_b / gib:.1f}GB")
    except Exception:
        pass
    return " | ".join(parts) if parts else "memory info unavailable"


def get_disk_free_gb(path: Path | None = None) -> float | None:
    try:
        from settings import MODELS_DIR
        target = path or MODELS_DIR
        while not target.exists() and target.parent != target:
            target = target.parent
        return shutil.disk_usage(str(target)).free / (1024 ** 3)
    except Exception:
        return None


def get_gpu_processes() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    me = os.getpid()
    names: list[str] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        if int(parts[0]) == me:
            continue
        name = Path(parts[1]).name or parts[1]
        if name not in names:
            names.append(name)
    return names


def get_system_report() -> dict[str, Any]:
    from settings import settings, MODELS_DIR

    gpu_name, vram_total, vram_free = get_gpu_info()
    ram_total, ram_avail = get_ram_gb()
    disk_free = get_disk_free_gb()
    gpu_procs = get_gpu_processes()

    # Adjust vram_free for memory already reserved by our own process
    vram_free_eff = vram_free
    if vram_free_eff is not None:
        try:
            import torch
            if torch.cuda.is_available():
                ours_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
                vram_free_eff += ours_gb
        except Exception:
            pass

    variants = []
    for v, req in VARIANT_REQS.items():
        blockers: list[str] = []
        warnings: list[str] = []
        if vram_total is None:
            warnings.append("No CUDA GPU detected")
        elif vram_total + 0.5 < req["vram_gb"]:
            blockers.append(
                f"GPU has {vram_total:.1f}GB total VRAM; need ~{req['vram_gb']:.0f}GB"
            )
        elif vram_free_eff is not None and vram_free_eff + 0.5 < req["vram_gb"]:
            culprits = ", ".join(gpu_procs) if gpu_procs else "unknown processes"
            blockers.append(
                f"Only {vram_free_eff:.1f}GB VRAM free (need ~{req['vram_gb']:.0f}GB). "
                f"Close: {culprits}"
            )
        if ram_total is not None and ram_total + 0.5 < req["ram_gb"]:
            blockers.append(
                f"System has {ram_total:.1f}GB RAM; need ~{req['ram_gb']:.0f}GB"
            )
        variants.append({
            "id": v,
            "label": req["label"],
            "vram_gb": req["vram_gb"],
            "ram_gb": req["ram_gb"],
            "blockers": blockers,
            "warnings": warnings,
            "ok": len(blockers) == 0,
        })

    return {
        "gpu_name": gpu_name,
        "vram_total_gb": round(vram_total, 1) if vram_total is not None else None,
        "vram_free_gb": round(vram_free, 1) if vram_free is not None else None,
        "ram_total_gb": round(ram_total, 1) if ram_total is not None else None,
        "ram_available_gb": round(ram_avail, 1) if ram_avail is not None else None,
        "disk_free_gb": round(disk_free, 1) if disk_free is not None else None,
        "gpu_processes": gpu_procs,
        "models_dir": str(MODELS_DIR),
        "model_status": {"loaded": False},  # populated by main.py
        "variants": variants,
    }
