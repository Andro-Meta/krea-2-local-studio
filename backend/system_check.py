"""Hardware pre-flight checks for Krea 2 Studio."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VARIANT_REQS: dict[str, dict[str, Any]] = {
    "turbo_bf16": {
        "vram_gb": 20.0, "ram_gb": 48.0,
        "label": "Krea 2 Turbo bf16 (needs ~20 GB VRAM and ~48 GB RAM with this single-file loader)",
    },
    "turbo_fp8": {
        "vram_gb": 13.0, "ram_gb": 12.0,
        "label": "Krea 2 Turbo fp8 (needs ~13 GB VRAM)",
    },
    "raw_bf16": {
        "vram_gb": 20.0, "ram_gb": 48.0,
        "label": "Krea 2 RAW bf16 (needs ~20 GB VRAM and ~48 GB RAM with this single-file loader)",
    },
    "raw_fp8": {
        "vram_gb": 13.0, "ram_gb": 32.0,
        "label": "Krea 2 RAW via dynamic fp8 (runs on ~13 GB VRAM; ~32 GB RAM to stream-quantize the bf16 file). Add block swap to lower VRAM further.",
    },
}


def _parse_mib(value: str) -> float:
    return float(value.strip().replace("MiB", "").strip()) / 1024


def parse_nvidia_smi_memory_csv(output: str) -> dict[str, float | str] | None:
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            return {
                "name": parts[0],
                "total_gb": _parse_mib(parts[1]),
                "free_gb": _parse_mib(parts[2]),
                "used_gb": _parse_mib(parts[3]),
            }
        except ValueError:
            continue
    return None


def parse_nvidia_smi_process_csv(output: str, *, current_pid: int | None = None) -> list[dict[str, Any]]:
    me = os.getpid() if current_pid is None else int(current_pid)
    processes: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid == me:
            continue
        item: dict[str, Any] = {
            "pid": pid,
            "name": Path(parts[1]).name or parts[1],
        }
        if len(parts) >= 3:
            try:
                item["used_memory_gb"] = _parse_mib(parts[2])
            except ValueError:
                item["used_memory_gb"] = None
        processes.append(item)
    return processes


def get_nvidia_smi_gpu_info() -> dict[str, Any] | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return None
    return parse_nvidia_smi_memory_csv(out)


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
    smi = get_nvidia_smi_gpu_info()
    if smi is not None:
        return str(smi["name"]), float(smi["total_gb"]), float(smi["free_gb"])
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
        _, total_gb, free_gb = get_gpu_info()
        if total_gb is not None and free_gb is not None:
            parts.append(f"VRAM {free_gb:.1f}/{total_gb:.1f}GB")
    except Exception as exc:
        logger.debug("CUDA memory snapshot unavailable: %s", exc)
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
    details = get_gpu_process_details()
    if details:
        names: list[str] = []
        for proc in details:
            name = str(proc.get("name") or "")
            if name and name not in names:
                names.append(name)
        return names
    return []


def get_gpu_process_details() -> list[dict[str, Any]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    return parse_nvidia_smi_process_csv(out)


def get_system_report() -> dict[str, Any]:
    from krea2.performance_guard import attention_acceleration_diagnostic
    from settings import settings, MODELS_DIR

    gpu_name, vram_total, vram_free = get_gpu_info()
    ram_total, ram_avail = get_ram_gb()
    disk_free = get_disk_free_gb()
    gpu_procs = get_gpu_processes()
    gpu_proc_details = get_gpu_process_details()

    # Adjust vram_free for memory already reserved by our own process
    vram_free_eff = vram_free
    if vram_free_eff is not None:
        try:
            import torch
            if torch.cuda.is_available():
                ours_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
                vram_free_eff += ours_gb
        except Exception as exc:
            logger.debug("CUDA reserved memory unavailable: %s", exc)

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
        "gpu_process_details": gpu_proc_details,
        "models_dir": str(MODELS_DIR),
        "model_status": {"loaded": False},  # populated by main.py
        "attention_acceleration": attention_acceleration_diagnostic(
            device="cuda" if gpu_name else "cpu",
            dtype="fp8",
            text_fusion=True,
        ),
        "variants": variants,
    }
