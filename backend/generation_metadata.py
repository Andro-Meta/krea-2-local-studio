from __future__ import annotations

import hashlib
import time
from typing import Any


def _safe_list(value: Any) -> list:
    return list(value or []) if isinstance(value, (list, tuple)) else []


def _hash_refs(refs: list[str]) -> list[str]:
    return [hashlib.sha256(ref.encode("utf-8")).hexdigest() for ref in refs if ref]


def _style_ref_dict(item: Any) -> dict:
    if hasattr(item, "model_dump"):
        item = item.model_dump()
    if not isinstance(item, dict):
        return {}
    image_b64 = str(item.get("image_b64", "") or "")
    return {
        "strength": float(item.get("strength", 1.0)),
        "role": str(item.get("role", "style")),
        "token_size": str(item.get("token_size", "normal")),
        "hash": hashlib.sha256(image_b64.encode("utf-8")).hexdigest() if image_b64 else "",
    }


def build_generation_metadata(
    req: Any,
    *,
    base_seed: int,
    image_index: int = 0,
    filename: str = "",
    resolved_provider: str = "",
    original_prompt: str | None = None,
    extra: dict | None = None,
) -> dict:
    moodboard_images = _safe_list(getattr(req, "moodboard_images", []))
    ref_images = [
        value for value in (
            getattr(req, "ref_image1_b64", None),
            getattr(req, "ref_image2_b64", None),
            getattr(req, "ref_image3_b64", None),
        ) if value
    ]
    style_references = [
        ref for ref in (_style_ref_dict(item) for item in _safe_list(getattr(req, "style_references", [])))
        if ref
    ]
    loras = [
        {
            "name": str(item.get("name", "")),
            "filename": str(item.get("filename", "")),
            "strength": float(item.get("strength", 1.0)),
            "enabled": bool(item.get("enabled", True)),
        }
        for item in _safe_list(getattr(req, "loras", []))
        if isinstance(item, dict)
    ]
    metadata = {
        "schema_version": 1,
        "app": "Krea 2 Studio",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "filename": filename,
        "prompt": str(getattr(req, "prompt", "")),
        "original_prompt": original_prompt if original_prompt is not None else str(getattr(req, "prompt", "")),
        "negative_prompt": str(getattr(req, "negative_prompt", "")),
        "seed": int(base_seed) + int(image_index),
        "base_seed": int(base_seed),
        "image_index": int(image_index),
        "mode": str(getattr(req, "mode", "txt2img")),
        "checkpoint": str(getattr(req, "checkpoint", "")),
        "checkpoint_path": str(getattr(req, "checkpoint_path", "")),
        "quantization": str(getattr(req, "quantization", "")),
        "steps": int(getattr(req, "steps", 0)),
        "cfg": float(getattr(req, "cfg", 0.0)),
        "width": int(getattr(req, "width", 0)),
        "height": int(getattr(req, "height", 0)),
        "denoise": float(getattr(req, "denoise", 1.0)),
        "sampler": str(getattr(req, "sampler", "euler_flow")),
        "inpaint": {
            "method": str(getattr(req, "inpaint_method", "native")),
            "lanpaint_inner_steps": int(getattr(req, "lanpaint_inner_steps", 3)),
            "lanpaint_strength": float(getattr(req, "lanpaint_strength", 1.0)),
        },
        "mu": getattr(req, "mu", None),
        "y1": float(getattr(req, "y1", 0.0)),
        "y2": float(getattr(req, "y2", 0.0)),
        "edit_provider": str(getattr(req, "edit_provider", "")),
        "resolved_provider": resolved_provider,
        "quality_preset": str(getattr(req, "quality_preset", "")),
        "creativity": str(getattr(req, "creativity", "medium")),
        "loras": loras,
        "mood": str(getattr(req, "mood", "")),
        "moodboard_ids": [int(v) for v in _safe_list(getattr(req, "moodboard_ids", [])) if str(v).isdigit()],
        "moodboard_uuids": [str(v) for v in _safe_list(getattr(req, "moodboard_uuids", [])) if str(v)],
        "moodboard_strength": float(getattr(req, "moodboard_strength", 0.35)),
        "image_references": {
            "moodboard_count": len(moodboard_images),
            "ref_image_count": len(ref_images),
            "style_reference_count": len(style_references),
            "style_references": style_references,
            "hashes": _hash_refs([*moodboard_images, *ref_images]),
        },
        "seed_variance": {
            "preset": str(getattr(req, "seed_variance_preset", "off")),
            "strength": float(getattr(req, "seed_variance_strength", 0.0)),
            "protection": str(getattr(req, "seed_variance_protection", "first_half")),
        },
        "rebalance": {
            "enabled": bool(getattr(req, "use_rebalance", False)),
            "multiplier": float(getattr(req, "rebalance_multiplier", 0.0)),
            "weights": str(getattr(req, "rebalance_weights", "")),
            "edit_enabled": bool(getattr(req, "edit_rebalance_enabled", True)),
            "edit_profile": str(getattr(req, "edit_rebalance_profile", "conservative")),
        },
        "krea_enhancer": {
            "enabled": bool(getattr(req, "krea_enhancer_enabled", False)),
            "strength": float(getattr(req, "krea_enhancer_strength", 1.0)),
        },
        "refine": {
            "enabled": bool(getattr(req, "refine", False)),
            "denoise": float(getattr(req, "refine_denoise", 0.0)),
            "steps": int(getattr(req, "refine_steps", 0)),
        },
        "bboxes": _safe_list(getattr(req, "bboxes", [])),
        "use_prompt_expander": bool(getattr(req, "use_prompt_expander", False)),
    }
    if extra:
        metadata["extra"] = extra
    return metadata
