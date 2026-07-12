"""Safely benchmark the fixed 2B abliterated Comfy Qwen helper.

Dry-run is intentionally the normal validation path for automation. Real runs
refuse to start when Studio is unavailable or reports queued/running user work
unless the operator explicitly supplies --force.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from comfy_qwen_vl import (  # noqa: E402
    MODEL_2B_ABLITERATED,
    QUANT_8BIT,
    QUANT_FP16,
    QWEN_VL_NODE,
    comfy_base_url,
    expand_prompt_comfy,
    object_info,
)
from gpu_recovery import is_cuda_oom  # noqa: E402

DEFAULT_MODEL = MODEL_2B_ABLITERATED
DEFAULT_PRECISIONS = (QUANT_FP16, QUANT_8BIT)
logger = logging.getLogger("krea2.benchmark_qwen_helper")
FIXED_CASES = (
    ("Rewrite as a concise image prompt.", "A red fox in quiet morning fog.", 101),
    ("Rewrite as a concise image prompt.", "Editorial portrait with soft window light.", 202),
    ("Rewrite as a concise image prompt.", "A glass pavilion in a pine forest.", 303),
)


def _raise_if_cancelled(cancel_probe) -> None:
    if cancel_probe is not None and cancel_probe():
        raise RuntimeError("Benchmark cancelled.")


def execute_benchmark_payload(
    payload: dict[str, Any],
    *,
    prompt_id_cb=None,
    cancel_probe=None,
) -> dict[str, Any]:
    """Execute a validated real benchmark. Called only by the unified GPU worker."""
    models = _normalize_models(list(payload.get("models") or []))
    precisions = _normalize_precisions(list(payload.get("precisions") or []))
    repeats = int(payload.get("repeats", 3))
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    _raise_if_cancelled(cancel_probe)
    validation = validate_comfy_options(models=models, precisions=precisions)
    _raise_if_cancelled(cancel_probe)
    report = _base_report(
        models=models,
        precisions=precisions,
        repeats=repeats,
        subsequent_krea=bool(payload.get("subsequent_krea")),
        validation=validation,
        dry_run=False,
    )
    if not validation.get("ok"):
        return report
    report["runs"] = []
    for model in models:
        for precision in precisions:
            _raise_if_cancelled(cancel_probe)
            run = benchmark_precision(
                precision,
                model=model,
                repeats=repeats,
                subsequent_krea=bool(payload.get("subsequent_krea")),
                prompt_id_cb=prompt_id_cb,
                cancel_probe=cancel_probe,
            )
            _raise_if_cancelled(cancel_probe)
            report["runs"].append(run)
    return report
def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }
    try:
        import psutil

        info["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 3)
    except Exception:
        info["ram_total_gb"] = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        info["gpus"] = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
    except (OSError, subprocess.SubprocessError):
        info["gpus"] = []
    return info


def _dropdown_values(node: dict[str, Any], name: str) -> list[str]:
    inputs = node.get("input", {}) if isinstance(node, dict) else {}
    for group in ("required", "optional"):
        raw = (inputs.get(group, {}) or {}).get(name)
        if isinstance(raw, (list, tuple)) and raw:
            choices = raw[0]
            if isinstance(choices, (list, tuple)):
                return [str(value) for value in choices]
    return []


def make_run_record(
    *,
    model: str,
    precision: str,
    cold_seconds: float | None = None,
    warm_seconds: float | None = None,
    peak_vram_gb: float | None = None,
    peak_ram_gb: float | None = None,
    baseline_comfy_ram_gb: float | None = None,
    comfy_ram_delta_gb: float | None = None,
    peak_system_ram_used_gb: float | None = None,
    system_ram_total_gb: float | None = None,
    baseline_comfy_vram_gb: float | None = None,
    comfy_vram_delta_gb: float | None = None,
    subsequent_krea_seconds: float | None = None,
    outputs: list[str] | None = None,
    errors: list[str] | None = None,
    telemetry_notes: list[str] | None = None,
    tracked_processes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct the single canonical per-run report schema."""
    return {
        "model": model,
        "precision": precision,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "peak_vram_gb": peak_vram_gb,
        "peak_ram_gb": peak_ram_gb,
        "baseline_comfy_ram_gb": baseline_comfy_ram_gb,
        "comfy_ram_delta_gb": comfy_ram_delta_gb,
        "peak_system_ram_used_gb": peak_system_ram_used_gb,
        "system_ram_total_gb": system_ram_total_gb,
        "baseline_comfy_vram_gb": baseline_comfy_vram_gb,
        "comfy_vram_delta_gb": comfy_vram_delta_gb,
        "subsequent_krea_seconds": subsequent_krea_seconds,
        "outputs": list(outputs or []),
        "errors": list(errors or []),
        "telemetry_notes": list(telemetry_notes or []),
        "tracked_processes": list(tracked_processes or []),
    }


def validate_comfy_options(
    *,
    models: list[str] | None = None,
    precisions: list[str] | None = None,
) -> dict[str, Any]:
    requested_models = list(models or [DEFAULT_MODEL])
    requested_precisions = list(precisions or DEFAULT_PRECISIONS)
    errors: list[str] = []
    try:
        catalog = object_info(QWEN_VL_NODE, timeout=5.0)
        node = catalog.get(QWEN_VL_NODE, {})
        available_models = _dropdown_values(node, "model_name")
        available_precisions = _dropdown_values(node, "quantization")
        if not node:
            errors.append(f"node unavailable: {QWEN_VL_NODE}")
        if not available_models:
            errors.append("model options are empty")
        if not available_precisions:
            errors.append("precision options are empty")
        for model in requested_models:
            if model not in available_models:
                errors.append(f"model unavailable: {model}")
        for precision in requested_precisions:
            if precision not in available_precisions:
                errors.append(f"precision unavailable: {precision}")
    except Exception as exc:
        available_models, available_precisions = [], []
        errors.append(f"Comfy validation failed: {type(exc).__name__}: {exc}")
    return {
        "ok": not errors,
        "node": QWEN_VL_NODE,
        "models": requested_models,
        "precisions": requested_precisions,
        "available_models": available_models,
        "available_precisions": available_precisions,
        "errors": errors,
    }


def write_report_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_psutil():
    try:
        import psutil

        return psutil
    except Exception:
        return None


def _process_matches_comfy(process: Any) -> bool:
    try:
        command = " ".join(str(part) for part in process.cmdline()).lower()
    except Exception:
        return False
    return "comfy" in command


def _process_identity(process: Any) -> dict[str, Any]:
    command_parts = [str(part) for part in process.cmdline()]
    normalized_command = " ".join(command_parts).replace("\\", "/").lower()
    executable = ""
    try:
        executable = str(process.exe() or "")
    except Exception:
        executable = command_parts[0] if command_parts else ""
    executable = executable.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return {
        "pid": int(process.pid),
        "create_time": float(process.create_time()),
        "executable": executable,
        "command_identity": hashlib.sha256(
            normalized_command.encode("utf-8", errors="replace")
        ).hexdigest()[:16],
        "_matches_comfy": "comfy" in normalized_command,
    }


def discover_comfy_pids(
    comfy_url: str,
    *,
    psutil_module: Any = None,
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[set[int], str | None]:
    """Find the local Comfy listener and matching launcher/process tree."""
    parsed = urlparse(comfy_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return set(), "Comfy process telemetry is unavailable for a remote URL."
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    psutil_module = psutil_module or _load_psutil()
    if psutil_module is None:
        return set(), "psutil is unavailable; Comfy process telemetry is disabled."
    listener_pids: set[int] = set()
    try:
        listeners = []
        for connection in psutil_module.net_connections(kind="inet"):
            address = getattr(connection, "laddr", None)
            listen_port = (
                getattr(address, "port", None)
                if address is not None
                else None
            )
            if listen_port is None and isinstance(address, (tuple, list)):
                listen_port = address[1] if len(address) > 1 else None
            status = str(getattr(connection, "status", "")).upper()
            if (
                listen_port == port
                and getattr(connection, "pid", None)
                and status in {"LISTEN", str(getattr(psutil_module, "CONN_LISTEN", "LISTEN")).upper()}
            ):
                listeners.append(psutil_module.Process(connection.pid))
                listener_pids.add(int(connection.pid))
    except Exception:
        try:
            if os.name == "nt":
                result = runner(
                    ["netstat", "-ano", "-p", "tcp"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                marker = f":{port}"
                for line in result.stdout.splitlines():
                    parts = line.split()
                    if (
                        len(parts) >= 5
                        and marker in parts[1]
                        and parts[3].upper() == "LISTENING"
                    ):
                        listener_pids.add(int(parts[4]))
            else:
                result = runner(
                    ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                listener_pids.update(
                    int(line.strip())
                    for line in result.stdout.splitlines()
                    if line.strip().isdigit()
                )
            listeners = [
                psutil_module.Process(pid) for pid in sorted(listener_pids)
            ]
        except Exception as exc:
            return set(), (
                "Could not inspect Comfy listener processes: "
                f"{type(exc).__name__}."
            )
    matching = [process for process in listeners if _process_matches_comfy(process)]
    if not matching:
        return set(), f"No verified Comfy process is listening on port {port}."

    processes: dict[int, Any] = {}
    for listener in matching:
        processes[int(listener.pid)] = listener
        try:
            for child in listener.children(recursive=True):
                if _process_matches_comfy(child):
                    processes[int(child.pid)] = child
        except Exception:
            pass
        current = listener
        for _ in range(3):
            try:
                parent = current.parent()
            except Exception:
                break
            if parent is None or not _process_matches_comfy(parent):
                break
            processes[int(parent.pid)] = parent
            current = parent
    return set(processes), None


def sample_comfy_vram_gb(
    pids: set[int],
    *,
    selected_gpu: str = "0",
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[float | None, str | None]:
    if not pids:
        return None, "No verified Comfy PIDs are available for VRAM telemetry."
    try:
        gpu_result = runner(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if getattr(gpu_result, "returncode", 0) != 0:
            return None, "nvidia-smi GPU discovery failed."
        gpu_map: dict[str, str] = {}
        for line in gpu_result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2:
                gpu_map[parts[0]] = parts[1]
        selected_key = str(selected_gpu)
        selected_uuid = gpu_map.get(selected_key)
        if selected_uuid is None and selected_key.upper().startswith("GPU-"):
            selected_uuid = selected_key
        if not selected_uuid:
            return None, f"Selected GPU {selected_gpu} could not be resolved."
        app_result = runner(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if getattr(app_result, "returncode", 0) != 0:
            return None, "nvidia-smi process telemetry failed."
        total_mib = 0.0
        matched = False
        invalid_matching_memory = False
        for line in app_result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            if parts[0] != selected_uuid or pid not in pids:
                continue
            try:
                used_mib = float(parts[2])
            except ValueError:
                invalid_matching_memory = True
                continue
            total_mib += used_mib
            matched = True
        if invalid_matching_memory:
            return None, "Comfy GPU memory was unavailable from nvidia-smi."
        return (total_mib / 1024 if matched else 0.0), None
    except Exception as exc:
        return None, f"Could not sample Comfy VRAM: {type(exc).__name__}."


class NullSampler:
    peak_vram_gb = None
    peak_ram_gb = None
    baseline_comfy_ram_gb = None
    comfy_ram_delta_gb = None
    peak_system_ram_used_gb = None
    system_ram_total_gb = None
    baseline_comfy_vram_gb = None
    comfy_vram_delta_gb = None
    telemetry_notes = ["Resource telemetry disabled by sampler."]
    tracked_processes = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class ResourceSampler(NullSampler):
    def __init__(
        self,
        interval: float = 0.1,
        *,
        comfy_url: str | None = None,
        selected_gpu: str | None = None,
        psutil_module: Any = None,
        runner: Callable[..., Any] = subprocess.run,
    ):
        self.interval = interval
        self._psutil = psutil_module or _load_psutil()
        self._runner = runner
        self.selected_gpu = selected_gpu or (
            os.environ.get("KREA_COMFY_GPU")
            or os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0]
            or "0"
        )
        self.pids, process_note = discover_comfy_pids(
            comfy_url or comfy_base_url(),
            psutil_module=self._psutil,
            runner=self._runner,
        )
        self.telemetry_notes: list[str] = []
        self._add_note(process_note)
        self._identities: dict[int, dict[str, Any]] = {}
        self.current_pids: set[int] = set()
        for pid in sorted(self.pids):
            try:
                identity = _process_identity(self._psutil.Process(pid))
                if not identity.pop("_matches_comfy"):
                    self._add_note(
                        f"PID {pid} did not retain a verified Comfy identity."
                    )
                    continue
                self._identities[pid] = identity
                self.current_pids.add(pid)
            except Exception as exc:
                self._add_note(
                    f"Could not record identity for Comfy PID {pid}: "
                    f"{type(exc).__name__}."
                )
        self.tracked_processes = [
            dict(identity) for identity in self._identities.values()
        ]
        self.baseline_comfy_ram_gb = self._sample_comfy_ram()
        baseline_vram, vram_note = sample_comfy_vram_gb(
            self.current_pids,
            selected_gpu=self.selected_gpu,
            runner=self._runner,
        )
        self._add_note(vram_note)
        self.baseline_comfy_vram_gb = baseline_vram
        self.peak_ram_gb = self.baseline_comfy_ram_gb
        self.peak_vram_gb = self.baseline_comfy_vram_gb
        self.comfy_ram_delta_gb = (
            0.0 if self.baseline_comfy_ram_gb is not None else None
        )
        self.comfy_vram_delta_gb = (
            0.0 if self.baseline_comfy_vram_gb is not None else None
        )
        self.system_ram_total_gb = None
        self.peak_system_ram_used_gb = None
        if self._psutil is not None:
            try:
                memory = self._psutil.virtual_memory()
                self.system_ram_total_gb = memory.total / 2**30
                self.peak_system_ram_used_gb = (
                    memory.total - memory.available
                ) / 2**30
            except Exception as exc:
                self._add_note(
                    f"Could not sample system RAM: {type(exc).__name__}."
                )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _add_note(self, note: str | None) -> None:
        if note and note not in self.telemetry_notes:
            self.telemetry_notes.append(note)

    def _sample_comfy_ram(self) -> float | None:
        valid_pids = self._revalidate_pids()
        if not valid_pids or self._psutil is None:
            return None
        try:
            return sum(
                self._psutil.Process(pid).memory_info().rss
                for pid in valid_pids
            ) / 2**30
        except Exception as exc:
            self._add_note(
                f"Could not sample Comfy process RAM: {type(exc).__name__}."
            )
            return None

    def _revalidate_pids(self) -> set[int]:
        if self._psutil is None:
            return set()
        valid: set[int] = set()
        for pid in sorted(self.current_pids):
            expected = self._identities.get(pid)
            try:
                process = self._psutil.Process(pid)
                observed = _process_identity(process)
            except Exception:
                self._add_note(
                    f"Comfy PID {pid} disappeared or restarted; attribution stopped."
                )
                continue
            matches_comfy = observed.pop("_matches_comfy")
            if (
                not matches_comfy
                or expected is None
                or observed["create_time"] != expected["create_time"]
                or observed["executable"] != expected["executable"]
                or observed["command_identity"] != expected["command_identity"]
            ):
                self._add_note(
                    f"Comfy PID {pid} identity changed or was reused; attribution stopped."
                )
                continue
            valid.add(pid)
        self.current_pids = valid
        return set(valid)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sample_once()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample_once()

    def _sample_once(self) -> None:
        ram = self._sample_comfy_ram()
        if ram is not None:
            self.peak_ram_gb = max(self.peak_ram_gb or 0.0, ram)
            if self.baseline_comfy_ram_gb is not None:
                self.comfy_ram_delta_gb = max(
                    0.0, self.peak_ram_gb - self.baseline_comfy_ram_gb
                )
        vram, note = sample_comfy_vram_gb(
            self.current_pids,
            selected_gpu=self.selected_gpu,
            runner=self._runner,
        )
        self._add_note(note)
        if vram is not None:
            self.peak_vram_gb = max(self.peak_vram_gb or 0.0, vram)
            if self.baseline_comfy_vram_gb is not None:
                self.comfy_vram_delta_gb = max(
                    0.0, self.peak_vram_gb - self.baseline_comfy_vram_gb
                )
        if self._psutil is not None:
            try:
                memory = self._psutil.virtual_memory()
                used = (memory.total - memory.available) / 2**30
                self.peak_system_ram_used_gb = max(
                    self.peak_system_ram_used_gb or 0.0, used
                )
            except Exception as exc:
                self._add_note(
                    f"Could not sample system RAM: {type(exc).__name__}."
                )


def run_helper_once(
    *,
    model: str = DEFAULT_MODEL,
    precision: str,
    case_index: int = 0,
    prompt_id_cb=None,
    keep_model_loaded: bool = False,
    max_tokens: int = 700,
) -> str:
    system_prompt, prompt, seed = FIXED_CASES[case_index % len(FIXED_CASES)]
    return expand_prompt_comfy(
        prompt,
        system_prompt,
        seed=seed,
        max_tokens=max_tokens,
        keep_model_loaded=keep_model_loaded,
        free_vram=False,
        prompt_id_cb=prompt_id_cb,
        model_override=model,
        precision_override=precision,
    )


def _run_subsequent_krea(*, prompt_id_cb=None) -> float:
    from comfy_workflows import comfy_generate
    from schemas import GenerationRequest

    request = GenerationRequest(
        prompt="A neutral gray sphere on a plain background.",
        width=64,
        height=64,
        steps=1,
        seed=404,
        checkpoint="turbo",
        quantization="int8",
        diffusion_engine="native_int8_convrot",
        use_rebalance=False,
    )
    started = time.perf_counter()
    comfy_generate(
        request, save_outputs=False, prompt_id_cb=prompt_id_cb
    )
    return time.perf_counter() - started


def benchmark_precision(
    precision: str,
    *,
    model: str = DEFAULT_MODEL,
    repeats: int,
    subsequent_krea: bool = False,
    sampler_factory: Callable[[], NullSampler] = ResourceSampler,
    prompt_id_cb=None,
    cancel_probe=None,
) -> dict[str, Any]:
    sampler = sampler_factory()
    outputs: list[str] = []
    errors: list[str] = []
    cold_seconds: float | None = None
    warm_samples: list[float] = []
    subsequent_seconds: float | None = None
    final_release_completed = False
    sampler.start()
    try:
        for index in range(repeats + 1):
            _raise_if_cancelled(cancel_probe)
            keep_loaded = index < repeats
            started = time.perf_counter()
            try:
                output = run_helper_once(
                    model=model,
                    precision=precision,
                    case_index=index,
                    prompt_id_cb=prompt_id_cb,
                    keep_model_loaded=keep_loaded,
                    max_tokens=700,
                )
            except Exception as exc:
                if is_cuda_oom(exc) or (
                    cancel_probe is not None and cancel_probe()
                ):
                    raise
                errors.append(f"{type(exc).__name__}: {exc}")
                break
            if not keep_loaded:
                final_release_completed = True
            _raise_if_cancelled(cancel_probe)
            outputs.append(output)
            elapsed = time.perf_counter() - started
            if index == 0:
                cold_seconds = elapsed
            else:
                warm_samples.append(elapsed)
        if subsequent_krea and not errors:
            _raise_if_cancelled(cancel_probe)
            subsequent_seconds = _run_subsequent_krea(
                prompt_id_cb=prompt_id_cb
            )
            _raise_if_cancelled(cancel_probe)
    finally:
        if not final_release_completed:
            try:
                run_helper_once(
                    model=model,
                    precision=precision,
                    case_index=0,
                    prompt_id_cb=None,
                    keep_model_loaded=False,
                    max_tokens=1,
                )
            except Exception:
                logger.warning(
                    "Benchmark node-level model cleanup failed for %s / %s",
                    model,
                    precision,
                    exc_info=True,
                )
        sampler.stop()

    def rounded(name: str) -> float | None:
        value = getattr(sampler, name, None)
        return round(float(value), 3) if value is not None else None

    return make_run_record(
        model=model,
        precision=precision,
        cold_seconds=cold_seconds,
        warm_seconds=(
            sum(warm_samples) / len(warm_samples) if warm_samples else None
        ),
        peak_vram_gb=rounded("peak_vram_gb"),
        peak_ram_gb=rounded("peak_ram_gb"),
        baseline_comfy_ram_gb=rounded("baseline_comfy_ram_gb"),
        comfy_ram_delta_gb=rounded("comfy_ram_delta_gb"),
        peak_system_ram_used_gb=rounded("peak_system_ram_used_gb"),
        system_ram_total_gb=rounded("system_ram_total_gb"),
        baseline_comfy_vram_gb=rounded("baseline_comfy_vram_gb"),
        comfy_vram_delta_gb=rounded("comfy_vram_delta_gb"),
        subsequent_krea_seconds=subsequent_seconds,
        outputs=outputs,
        errors=errors,
        telemetry_notes=list(getattr(sampler, "telemetry_notes", [])),
        tracked_processes=list(getattr(sampler, "tracked_processes", [])),
    )


def parse_studio_jobs_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("jobs")
    if not isinstance(payload, list):
        raise RuntimeError("Studio queue returned an unexpected response.")
    return [item for item in payload if isinstance(item, dict)]


def fetch_studio_jobs(studio_url: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{studio_url.rstrip('/')}/api/jobs?limit=200",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Studio queue could not be verified; use --force only after checking it."
        ) from exc
    return parse_studio_jobs_payload(payload)


def refuse_if_studio_busy(
    jobs: list[dict[str, Any]], *, force: bool
) -> int:
    active = [
        job
        for job in jobs
        if job.get("status") in {"queued", "running", "finalizing", "cancellation_requested"}
    ]
    if active and not force:
        raise RuntimeError(
            f"Refusing benchmark while Studio has active user work ({len(active)} task(s))."
        )
    if force:
        logger.warning(
            "Force bypass enabled with %d active Studio task(s).", len(active)
        )
    return len(active)


def _normalize_precisions(values: list[str] | None) -> list[str]:
    aliases = {
        "fp16": QUANT_FP16,
        QUANT_FP16.lower(): QUANT_FP16,
        "8bit": QUANT_8BIT,
        "8-bit": QUANT_8BIT,
        QUANT_8BIT.lower(): QUANT_8BIT,
    }
    if not values:
        return list(DEFAULT_PRECISIONS)
    normalized: list[str] = []
    for value in values:
        precision = aliases.get(value.strip().lower())
        if precision is None:
            raise ValueError(f"Unsupported precision: {value}")
        if precision not in normalized:
            normalized.append(precision)
    return normalized


def _normalize_models(values: list[str] | None) -> list[str]:
    if not values:
        return [DEFAULT_MODEL]
    models: list[str] = []
    for value in values:
        model = value.strip()
        if not model:
            raise ValueError("Model name must not be empty")
        if model not in models:
            models.append(model)
    return models


def _base_report(
    *,
    models: list[str],
    precisions: list[str],
    repeats: int,
    subsequent_krea: bool,
    validation: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp": _utc_timestamp(),
        "dry_run": dry_run,
        "model": models[0],
        "models": models,
        "precisions": precisions,
        "repeats": repeats,
        "subsequent_krea_requested": subsequent_krea,
        "system": _system_info(),
        "validation": validation,
        "runs": [
            make_run_record(model=model, precision=precision)
            for model in models
            for precision in precisions
        ],
        "errors": list(validation.get("errors", [])),
        "warnings": [],
    }


def authenticate_studio_session(
    session: requests.Session,
    studio_url: str,
    *,
    username: str | None = None,
    input_fn=input,
    password_fn=getpass.getpass,
) -> None:
    base = studio_url.rstrip("/")
    probe = session.get(f"{base}/api/auth/me", timeout=10)
    probe.raise_for_status()
    state = probe.json()
    if not state.get("share_auth"):
        return
    if state.get("authenticated") and state.get("role") == "admin":
        return
    login_name = (username or input_fn("Admin username: ")).strip()
    password = password_fn("Admin password: ")
    response = session.post(
        f"{base}/api/auth/login",
        json={"username": login_name, "password": password},
        timeout=10,
    )
    response.raise_for_status()
    if response.json().get("role") != "admin":
        raise RuntimeError("Benchmark submission requires an admin account.")


def submit_queued_benchmark(
    session: requests.Session,
    studio_url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 7200,
    poll_interval: float = 1.0,
) -> dict[str, Any]:
    base = studio_url.rstrip("/")
    submitted = session.post(
        f"{base}/api/admin/helper-benchmark", json=payload, timeout=15
    )
    submitted.raise_for_status()
    task_id = submitted.json()["job_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = session.get(f"{base}/api/generate/{task_id}", timeout=15)
        response.raise_for_status()
        job = response.json()
        status = job.get("status")
        if status == "done":
            result = job.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Benchmark task completed without a report.")
            session.post(
                f"{base}/api/generate/{task_id}/ack", timeout=15
            ).raise_for_status()
            return result
        if status in {"error", "blocked", "cancelled"}:
            raise RuntimeError(job.get("error") or f"Benchmark task {status}.")
        time.sleep(poll_interval)
    raise TimeoutError("Timed out waiting for queued benchmark.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", help="exact Comfy model name; repeatable"
    )
    parser.add_argument("--precision", action="append", help="fp16 or 8bit; repeatable")
    parser.add_argument("--repeats", type=int, default=3, help="warm repeats per precision")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark_qwen_helper.json"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--subsequent-krea", action="store_true")
    parser.add_argument("--studio-url", default="http://127.0.0.1:8200")
    parser.add_argument("--username")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    models = _normalize_models(args.model)
    precisions = _normalize_precisions(args.precision)
    validation = validate_comfy_options(models=models, precisions=precisions)
    report = _base_report(
        models=models,
        precisions=precisions,
        repeats=args.repeats,
        subsequent_krea=bool(args.subsequent_krea),
        validation=validation,
        dry_run=bool(args.dry_run),
    )
    if not validation.get("ok"):
        write_report_atomic(args.output, report)
        return 2
    if args.dry_run:
        write_report_atomic(args.output, report)
        return 0

    warning: str | None = None
    try:
        if args.force:
            warning = (
                "--force bypassed only the initial operator warning; "
                "execution remains serialized by the Studio GPU queue."
            )
            report["warnings"].append(warning)
            logger.warning(warning)
        session = requests.Session()
        authenticate_studio_session(
            session, args.studio_url, username=args.username
        )
        report = submit_queued_benchmark(
            session,
            args.studio_url,
            {
                "models": models,
                "precisions": precisions,
                "repeats": args.repeats,
                "subsequent_krea": bool(args.subsequent_krea),
            },
        )
        if warning is not None:
            report.setdefault("warnings", []).append(warning)
    except Exception:
        logger.exception("Queued benchmark failed")
        report.setdefault("errors", []).append(
            "Queued benchmark failed. Check the benchmark log for details."
        )
        write_report_atomic(args.output, report)
        return 3
    write_report_atomic(args.output, report)
    return 0 if not any(run["errors"] for run in report["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
