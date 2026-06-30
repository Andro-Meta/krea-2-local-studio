from __future__ import annotations

import base64
import json
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generation_metadata import build_generation_metadata
from settings import OUTPUTS_DIR


@dataclass
class ComfyInt8Settings:
    base_url: str = "http://127.0.0.1:8188"
    int8_model: str = "krea2_turbo_int8.safetensors"
    clip_name: str = "qwen3vl_4b_fp8_scaled.safetensors"
    vae_name: str = "qwen_image_vae.safetensors"
    timeout_sec: int = 900


def _safe_local_base_url(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Comfy base URL must be http(s).")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Comfy base URL must point to localhost.")
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "", "", "", "")).rstrip("/")


def _lora_name(item: dict[str, Any]) -> str:
    filename = str(item.get("filename") or "")
    name = str(item.get("name") or "")
    return filename or (name + ".safetensors")


def build_comfy_int8_workflow(req: Any, settings: ComfyInt8Settings) -> dict[str, Any]:
    if str(getattr(req, "mode", "txt2img")) != "txt2img":
        raise ValueError("Comfy INT8 v1 supports txt2img only.")
    model_name = settings.int8_model or "krea2_turbo_int8.safetensors"
    clip_name = settings.clip_name or "qwen3vl_4b_fp8_scaled.safetensors"
    vae_name = settings.vae_name or "qwen_image_vae.safetensors"
    steps = int(getattr(req, "steps", 8) or 8)
    cfg = float(getattr(req, "cfg", 1.0) or 0.0)
    sampler = str(getattr(req, "sampler", "ddim") or "ddim")
    scheduler = str(getattr(req, "scheduler", "beta57") or "beta57")
    width = int(getattr(req, "width", 1024) or 1024)
    height = int(getattr(req, "height", 1024) or 1024)
    seed = int(getattr(req, "seed", -1) or -1)
    if seed < 0:
        seed = int(time.time() * 1000) & 0x7FFFFFFF

    workflow: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": model_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "krea2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": str(getattr(req, "prompt", "") or "")}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": str(getattr(req, "negative_prompt", "") or "")}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
    }

    model_ref: list[Any] = ["1", 0]
    clip_ref: list[Any] = ["2", 0]
    next_id = 7
    for lora in list(getattr(req, "loras", []) or []):
        if not lora.get("enabled", True):
            continue
        lora_name = _lora_name(lora)
        if not lora_name:
            continue
        node_id = str(next_id)
        next_id += 1
        strength = float(lora.get("strength", 1.0) or 0.0)
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": lora_name,
                "strength_model": strength,
                "strength_clip": 0.0,
            },
        }
        model_ref = [node_id, 0]
        clip_ref = [node_id, 1]

    sampler_id = str(next_id)
    decode_id = str(next_id + 1)
    save_id = str(next_id + 2)
    workflow[sampler_id] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": 1.0,
        },
    }
    workflow[decode_id] = {"class_type": "VAEDecode", "inputs": {"samples": [sampler_id, 0], "vae": ["3", 0]}}
    workflow[save_id] = {"class_type": "SaveImage", "inputs": {"images": [decode_id, 0], "filename_prefix": "krea2_int8"}}
    return workflow


def _request_json(url: str, payload: dict[str, Any] | None = None, *, timeout: int = 30) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - localhost validated by caller
        return json.loads(response.read().decode("utf-8"))


def comfy_int8_status(settings: ComfyInt8Settings) -> dict[str, Any]:
    base = _safe_local_base_url(settings.base_url)
    try:
        system = _request_json(f"{base}/system_stats", timeout=5)
        return {"ok": True, "base_url": base, "system": system}
    except Exception as exc:
        return {"ok": False, "base_url": base, "error": str(exc)}


def generate_comfy_int8_external(req: Any, settings: ComfyInt8Settings, *, output_dir: Path = OUTPUTS_DIR):
    base = _safe_local_base_url(settings.base_url)
    workflow = build_comfy_int8_workflow(req, settings)
    client_id = str(uuid.uuid4())
    prompt_response = _request_json(
        f"{base}/prompt",
        {"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    prompt_id = str(prompt_response.get("prompt_id") or "")
    if not prompt_id:
        raise RuntimeError(f"Comfy did not return a prompt_id: {prompt_response}")

    deadline = time.time() + max(30, int(settings.timeout_sec or 900))
    history: dict[str, Any] = {}
    while time.time() < deadline:
        history = _request_json(f"{base}/history/{urllib.parse.quote(prompt_id)}", timeout=30)
        if prompt_id in history:
            break
        time.sleep(1.0)
    if prompt_id not in history:
        raise TimeoutError("Timed out waiting for Comfy INT8 generation.")

    outputs = history[prompt_id].get("outputs", {})
    images = []
    for node in outputs.values():
        for image in node.get("images", []) or []:
            query = urllib.parse.urlencode({
                "filename": image.get("filename", ""),
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            })
            with urllib.request.urlopen(f"{base}/view?{query}", timeout=60) as response:  # noqa: S310
                images.append(response.read())
    if not images:
        raise RuntimeError("Comfy INT8 finished without an output image.")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"comfy_int8_{int(time.time() * 1000)}.png"
    (output_dir / filename).write_bytes(images[0])
    image_b64 = base64.b64encode(images[0]).decode("utf-8")
    metadata = [build_generation_metadata(req, base_seed=int(getattr(req, "seed", -1) or -1), image_index=0, filename=filename, resolved_provider="comfy_int8")]
    return [image_b64], int(getattr(req, "seed", -1) or -1), [filename], [], metadata
