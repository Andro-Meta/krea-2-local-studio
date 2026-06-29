from __future__ import annotations

import base64
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from generation_metadata import build_generation_metadata
from settings import OUTPUTS_DIR


@dataclass
class GgufRuntimeSettings:
    sd_cli_path: str = ""
    turbo_path: str = ""
    raw_path: str = ""
    llm_path: str = ""
    vae_path: str = ""
    lora_dir: str = ""
    timeout_sec: int = 600


def _require_file(path: str, label: str) -> str:
    value = str(path or "").strip()
    if not value:
        raise ValueError(f"GGUF {label} path is required.")
    p = Path(value)
    if not p.exists() or not p.is_file():
        raise ValueError(f"GGUF {label} path does not exist: {value}")
    return str(p)


def build_gguf_command(req: Any, runtime: GgufRuntimeSettings, *, output_dir: Path = OUTPUTS_DIR) -> tuple[list[str], Path]:
    if str(getattr(req, "mode", "txt2img")) != "txt2img":
        raise ValueError("GGUF external v1 supports txt2img only.")
    exe = _require_file(runtime.sd_cli_path, "sd-cli")
    checkpoint = str(getattr(req, "checkpoint", "turbo") or "turbo").lower()
    model_path = _require_file(runtime.raw_path if checkpoint == "raw" else runtime.turbo_path, f"{checkpoint} diffusion model")
    llm_path = _require_file(runtime.llm_path, "Qwen3-VL GGUF LLM")
    vae_path = _require_file(runtime.vae_path, "VAE")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"gguf_{int(time.time() * 1000)}.png"
    cmd = [
        exe,
        "--diffusion-model", model_path,
        "--llm", llm_path,
        "--vae", vae_path,
        "-p", str(getattr(req, "prompt", "")),
        "--negative-prompt", str(getattr(req, "negative_prompt", "") or ""),
        "--steps", str(int(getattr(req, "steps", 8) or 8)),
        "--cfg-scale", str(float(getattr(req, "cfg", 0.0) or 0.0)),
        "-W", str(int(getattr(req, "width", 1024) or 1024)),
        "-H", str(int(getattr(req, "height", 1024) or 1024)),
        "--seed", str(int(getattr(req, "seed", -1) or -1)),
        "-o", str(output),
        "--diffusion-fa",
    ]
    if runtime.lora_dir and Path(runtime.lora_dir).exists():
        cmd.extend(["--lora-model-dir", str(Path(runtime.lora_dir))])
    return cmd, output


def generate_gguf_external(
    req: Any,
    runtime: GgufRuntimeSettings,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
    output_dir: Path = OUTPUTS_DIR,
) -> tuple[list[str], int, list[str], list[dict], list[dict]]:
    cmd, output = build_gguf_command(req, runtime, output_dir=output_dir)
    if progress_cb:
        progress_cb(1, 3)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(30, int(runtime.timeout_sec)))
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "GGUF sidecar failed.").strip()
        raise RuntimeError(detail[-2000:])
    if not output.exists():
        raise RuntimeError("GGUF sidecar finished but did not write an output image.")
    raw = output.read_bytes()
    image_b64 = base64.b64encode(raw).decode("utf-8")
    metadata = [build_generation_metadata(req, base_seed=int(getattr(req, "seed", -1) or -1), image_index=0, filename=output.name, resolved_provider="gguf_external")]
    if progress_cb:
        progress_cb(3, 3)
    return [image_b64], int(getattr(req, "seed", -1) or -1), [output.name], [], metadata

