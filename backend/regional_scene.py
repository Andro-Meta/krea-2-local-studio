from __future__ import annotations

import base64
import hashlib
import io
from typing import Any

from PIL import Image, ImageFilter


def _strip_data_url(value: str) -> str:
    return value.split(",", 1)[1] if "," in value else value


def decode_mask(mask_b64: str, size: tuple[int, int]) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(_strip_data_url(mask_b64)))).convert("L")
    if image.size != size:
        image = image.resize(size)
    return image.point(lambda value: 255 if value > 0 else 0)


def normalize_region_masks(regions: list[dict[str, Any]], *, size: tuple[int, int]) -> list[Image.Image]:
    masks: list[Image.Image] = []
    for region in regions[:8]:
        mask_b64 = str(region.get("mask_b64", "") or "")
        if not mask_b64:
            masks.append(Image.new("L", size, 0))
            continue
        mask = decode_mask(mask_b64, size)
        feather = max(0, int(region.get("feather", 0) or 0))
        if feather:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather / 2))
        masks.append(mask)

    if not any(bool(region.get("normalize", True)) for region in regions):
        return masks

    normalized = [Image.new("L", size, 0) for _ in masks]
    src = [mask.load() for mask in masks]
    dst = [mask.load() for mask in normalized]
    for y in range(size[1]):
        for x in range(size[0]):
            values = [int(loader[x, y]) for loader in src]
            total = sum(values)
            if total <= 255:
                for idx, value in enumerate(values):
                    dst[idx][x, y] = value
            else:
                for idx, value in enumerate(values):
                    dst[idx][x, y] = int(value * 255 / total)
    return normalized


def build_regional_prompt_text(
    base_prompt: str,
    regions: list[dict[str, Any]],
    *,
    base_prompt_strength: float = 0.3,
) -> str:
    visible = [
        region for region in regions[:8]
        if bool(region.get("visible", True)) and str(region.get("prompt", "") or "").strip()
    ]
    if not visible:
        return base_prompt
    parts = [
        base_prompt.strip(),
        (
            "Internal regional composition guidance only, not visible text. "
            f"Keep the global scene context at about {float(base_prompt_strength):.2f} influence. "
            "Do not render region labels, captions, UI text, or instruction text."
        ),
    ]
    for index, region in enumerate(visible, start=1):
        prompt = str(region.get("prompt", "") or "").strip()
        strength = max(0.0, min(2.0, float(region.get("strength", 1.0) or 1.0)))
        avoid = str(region.get("negative_prompt", "") or "").strip()
        line = f"Masked area {index} visual content at strength {strength:.2f}: {prompt}"
        if avoid:
            line += f"; avoid {avoid}"
        parts.append(line)
    return "\n".join(part for part in parts if part)


def region_metadata(region: dict[str, Any]) -> dict[str, Any]:
    mask_b64 = str(region.get("mask_b64", "") or "")
    return {
        "prompt": str(region.get("prompt", "") or ""),
        "negative_prompt": str(region.get("negative_prompt", "") or ""),
        "strength": float(region.get("strength", 1.0) or 1.0),
        "feather": int(region.get("feather", 24) or 0),
        "normalize": bool(region.get("normalize", True)),
        "visible": bool(region.get("visible", True)),
        "lora_filter": str(region.get("lora_filter", "") or ""),
        "mask_hash": hashlib.sha256(mask_b64.encode("utf-8")).hexdigest() if mask_b64 else "",
    }
