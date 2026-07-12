"""Drive 1038lab/ComfyUI-QwenVL for Studio helper tasks (describe / enrich / wand).

Comfy is the default local helper path. Studio Transformers remains the fallback when
Comfy is down or the QwenVL nodes are missing.
"""
from __future__ import annotations

import base64
import io
import logging
import uuid
from typing import Any

import requests
from PIL import Image

from comfy_client import (
    ComfyClient,
    ComfyExecutionError,
    ComfyUnavailable,
    PromptIdCb,
    _notify_prompt_id,
    cancel_prompt,
    comfy_available,
    comfy_base_url,
    free_comfy_vram,
    object_info,
)

logger = logging.getLogger("krea2.comfy_qwen")

# main.py installs a probe that reports whether a generation is running (or the
# GPU lease is held). When busy we must NOT free Comfy's VRAM: doing so would
# unload the diffusion model out from under the in-flight job. The helper graph
# just queues behind it in ComfyUI's own serial queue instead.
_GENERATION_BUSY_PROBE: Any = None


def set_generation_busy_probe(probe) -> None:
    global _GENERATION_BUSY_PROBE
    _GENERATION_BUSY_PROBE = probe


def _generation_busy() -> bool:
    try:
        return bool(_GENERATION_BUSY_PROBE and _GENERATION_BUSY_PROBE())
    except Exception:
        return False

QWEN_VL_NODE = "AILab_QwenVL_Advanced"
QWEN_ENHANCER_NODE = "AILab_QwenVL_PromptEnhancer"
# Helper defaults: 2B abliterated (faster). 4B remains a quick switch for richer prose.
MODEL_2B_ABLITERATED = "Huihui-Qwen3-VL-2B-Instruct-abliterated"
MODEL_4B_ABLITERATED = "Huihui-Qwen3-VL-4B-Instruct-abliterated"
DEFAULT_MODEL = MODEL_2B_ABLITERATED
QUANT_FP16 = "None (FP16)"
QUANT_4BIT = "4-bit (VRAM-friendly)"
QUANT_8BIT = "8-bit (Balanced)"
_QUANT_VALUES = frozenset({QUANT_4BIT, QUANT_8BIT, QUANT_FP16})
_QUANT_ALIASES = {
    "4bit": QUANT_4BIT,
    "4-bit": QUANT_4BIT,
    "8bit": QUANT_8BIT,
    "8-bit": QUANT_8BIT,
    "fp16": QUANT_FP16,
    "none": QUANT_FP16,
}

_COMFY_QWEN_ALIASES = {
    "": DEFAULT_MODEL,
    "default": DEFAULT_MODEL,
    "2b": MODEL_2B_ABLITERATED,
    "2b_abliterated": MODEL_2B_ABLITERATED,
    "huihui-2b": MODEL_2B_ABLITERATED,
    "4b": MODEL_4B_ABLITERATED,
    "4b_abliterated": MODEL_4B_ABLITERATED,
    "abliterated": MODEL_4B_ABLITERATED,
    "huihui-4b": MODEL_4B_ABLITERATED,
}


def resolve_comfy_qwen_model(override: str = "") -> str:
    """Map settings/aliases to a ComfyUI-QwenVL model_name."""
    from settings import settings

    raw = str(override or getattr(settings, "comfy_qwen_model", "") or "").strip()
    if not raw:
        return DEFAULT_MODEL
    key = raw.lower().replace(" ", "_")
    if key in _COMFY_QWEN_ALIASES:
        return _COMFY_QWEN_ALIASES[key]
    # Accept exact Comfy dropdown names / HF folder names.
    if raw in (MODEL_2B_ABLITERATED, MODEL_4B_ABLITERATED):
        return raw
    if "2B-Instruct-abliterated" in raw or "2b-instruct-abliterated" in key:
        return MODEL_2B_ABLITERATED
    if "4B-Instruct-abliterated" in raw or "4b-instruct-abliterated" in key:
        return MODEL_4B_ABLITERATED
    return raw


def resolve_comfy_qwen_quant(override: str = "") -> str:
    """Return only quantization labels accepted by AILab_QwenVL_Advanced."""
    from settings import settings

    raw = str(
        override or getattr(settings, "comfy_qwen_quant", "") or "8bit"
    ).strip()
    if raw in _QUANT_VALUES:
        return raw
    return _QUANT_ALIASES.get(raw.lower().replace("_", "").replace(" ", ""), QUANT_8BIT)


def comfy_qwen_vl_available(timeout: float = 5.0) -> bool:
    if not comfy_available(timeout=timeout):
        return False
    try:
        info = object_info(QWEN_VL_NODE, timeout=timeout)
        return QWEN_VL_NODE in info
    except Exception:
        return False


def _strip_data_url(b64: str) -> str:
    return b64.split(",", 1)[-1] if "," in (b64 or "") else (b64 or "")


def _upload_png(raw: bytes, fname: str | None = None) -> str:
    name = fname or f"krea_qwen_{uuid.uuid4().hex}.png"
    r = requests.post(
        f"{comfy_base_url()}/upload/image",
        files={"image": (name, raw, "image/png")},
        data={"overwrite": "true"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json() if r.content else {}
    return str(data.get("name") or name)


def _b64_to_png_bytes(b64: str, max_side: int = 1024) -> bytes:
    img = Image.open(io.BytesIO(base64.b64decode(_strip_data_url(b64)))).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _link(node_id: str, slot: int = 0) -> list:
    return [node_id, slot]


def _batch_image_nodes(nodes: dict[str, dict], image_ids: list[str]) -> str:
    """Nest ImageBatch so N LoadImage outputs become one IMAGE batch (video frames)."""
    if len(image_ids) == 1:
        return image_ids[0]
    acc = image_ids[0]
    for i, nid in enumerate(image_ids[1:], start=1):
        batch_id = f"batch_{i}"
        nodes[batch_id] = {
            "class_type": "ImageBatch",
            "inputs": {"image1": _link(acc), "image2": _link(nid)},
        }
        acc = batch_id
    return acc


def _extract_text(outputs: dict) -> str:
    for node_out in (outputs or {}).values():
        if not isinstance(node_out, dict):
            continue
        for key in ("text", "string", "RESPONSE", "ENHANCED_OUTPUT"):
            val = node_out.get(key)
            if isinstance(val, list) and val:
                text = str(val[0] or "").strip()
                if text:
                    return text
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def _attach_preview(nodes: dict, source_node_id: str) -> str:
    """PreviewAny is an OUTPUT_NODE so Comfy persists the STRING in /history."""
    nodes["preview"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": _link(source_node_id)},
    }
    return "preview"


def _run_graph_for_text(
    graph: dict,
    text_node_id: str,
    timeout: int = 600,
    *,
    free_vram: bool = False,
    prompt_id_cb: PromptIdCb = None,
) -> str:
    if not comfy_available():
        raise ComfyUnavailable(f"ComfyUI is not responding at {comfy_base_url()}.")
    # Unload DiT/VAE so Qwen-VL can claim VRAM (skip when chaining Stage 2 after
    # Comfy Stage 1, and NEVER while a generation is executing — freeing VRAM
    # then would yank the diffusion model out from under the in-flight job).
    if free_vram and not _generation_busy():
        free_comfy_vram(unload_models=True, free_memory=True)
    client = ComfyClient()
    prompt_id = client._post_prompt(graph)
    _notify_prompt_id(prompt_id_cb, prompt_id)
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            hist = client.get_history(prompt_id)
        except Exception:
            hist = {}
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {}) or {}
            if status.get("status_str") == "error":
                raise ComfyExecutionError(f"ComfyUI QwenVL error: {status}")
            text = _extract_text(entry.get("outputs", {}) or {})
            if text:
                return text
            # Completed but empty — still surface whatever we got.
            if status.get("completed") or status.get("status_str") == "success" or entry.get("outputs"):
                raise ComfyExecutionError(
                    f"ComfyUI QwenVL returned no text (node={text_node_id}). outputs={list((entry.get('outputs') or {}).keys())}"
                )
        time.sleep(0.5)
    # Cancel only OUR prompt: a global interrupt here could kill an unrelated
    # generation that happens to be executing.
    cancel_prompt(prompt_id, base_url=client.base)
    raise ComfyExecutionError("ComfyUI QwenVL timed out.")


def _ensure_nodes() -> None:
    if not comfy_qwen_vl_available():
        raise ComfyUnavailable(
            "ComfyUI-QwenVL nodes are not loaded. Restart ComfyUI after installing "
            "custom_nodes/ComfyUI-QwenVL."
        )


def describe_image_comfy(
    image_b64: str,
    custom_prompt: str,
    *,
    max_tokens: int = 420,
    temperature: float = 0.6,
    seed: int = 1,
    keep_model_loaded: bool = True,
    prompt_id_cb: PromptIdCb = None,
) -> str:
    _ensure_nodes()
    png = _b64_to_png_bytes(image_b64, max_side=1024)
    fname = _upload_png(png)
    nodes: dict[str, dict] = {
        "load": {"class_type": "LoadImage", "inputs": {"image": fname}},
        "qwen": {
            "class_type": QWEN_VL_NODE,
            "inputs": {
                "model_name": resolve_comfy_qwen_model(),
                "quantization": resolve_comfy_qwen_quant(),
                "attention_mode": "sdpa",
                "use_torch_compile": False,
                "device": "auto",
                "preset_prompt": "🖼️ Detailed Description",
                "custom_prompt": custom_prompt,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": 0.9,
                "num_beams": 1,
                "repetition_penalty": 1.2,
                "frame_count": 1,
                "keep_model_loaded": bool(keep_model_loaded),
                "seed": max(1, int(seed)),
                "image": _link("load"),
            },
        },
    }
    _attach_preview(nodes, "qwen")
    return _run_graph_for_text(nodes, "preview", prompt_id_cb=prompt_id_cb)


def enrich_images_comfy(
    image_b64s: list[str],
    custom_prompt: str,
    *,
    max_tokens: int = 900,
    temperature: float = 0.45,
    seed: int = 1,
    keep_model_loaded: bool = True,
    prompt_id_cb: PromptIdCb = None,
) -> str:
    _ensure_nodes()
    images = [b for b in (image_b64s or []) if b][:10]
    nodes: dict[str, dict] = {}
    load_ids: list[str] = []
    for i, b64 in enumerate(images):
        png = _b64_to_png_bytes(b64, max_side=1024)
        fname = _upload_png(png)
        lid = f"load_{i}"
        nodes[lid] = {"class_type": "LoadImage", "inputs": {"image": fname}}
        load_ids.append(lid)

    qwen_inputs: dict[str, Any] = {
        "model_name": resolve_comfy_qwen_model(),
        "quantization": resolve_comfy_qwen_quant(),
        "attention_mode": "sdpa",
        "use_torch_compile": False,
        "device": "auto",
        "preset_prompt": "🖼️ Detailed Description",
        "custom_prompt": custom_prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": 0.9,
        "num_beams": 1,
        "repetition_penalty": 1.2,
        "frame_count": max(1, len(load_ids)),
        "keep_model_loaded": bool(keep_model_loaded),
        "seed": max(1, int(seed)),
    }
    if load_ids:
        batch = _batch_image_nodes(nodes, load_ids)
        # Multi-image → video frames; single image → image input.
        if len(load_ids) == 1:
            qwen_inputs["image"] = _link(batch)
        else:
            qwen_inputs["video"] = _link(batch)

    nodes["qwen"] = {"class_type": QWEN_VL_NODE, "inputs": qwen_inputs}
    _attach_preview(nodes, "qwen")
    return _run_graph_for_text(nodes, "preview", prompt_id_cb=prompt_id_cb)


def expand_prompt_comfy(
    prompt: str,
    system_prompt: str,
    *,
    max_tokens: int = 700,
    temperature: float = 0.7,
    seed: int = 1,
    keep_model_loaded: bool = True,
    free_vram: bool = False,
    prompt_id_cb: PromptIdCb = None,
    model_override: str = "",
    precision_override: str = "",
) -> str:
    _ensure_nodes()
    # Use Advanced VL in text-only mode so we keep Studio's ~700 token budget
    # (PromptEnhancer caps max_tokens at 1024 and merges prompts differently).
    merged = f"{system_prompt.strip()}\n\n{prompt.strip()}".strip()
    nodes = {
        "qwen": {
            "class_type": QWEN_VL_NODE,
            "inputs": {
                "model_name": resolve_comfy_qwen_model(model_override),
                "quantization": resolve_comfy_qwen_quant(precision_override),
                "attention_mode": "sdpa",
                "use_torch_compile": False,
                "device": "auto",
                "preset_prompt": "🖼️ Detailed Description",
                "custom_prompt": merged,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": 0.9,
                "num_beams": 1,
                "repetition_penalty": 1.2,
                "frame_count": 1,
                "keep_model_loaded": bool(keep_model_loaded),
                "seed": max(1, int(seed)),
            },
        }
    }
    _attach_preview(nodes, "qwen")
    return _run_graph_for_text(
        nodes,
        "preview",
        free_vram=free_vram,
        prompt_id_cb=prompt_id_cb,
    )
