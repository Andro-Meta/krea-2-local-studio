from __future__ import annotations

from PIL import Image, ImageFilter, ImageOps


def pixel_perfect_resolution(width: int, height: int, target: int = 768) -> tuple[int, int]:
    width = max(1, int(width))
    height = max(1, int(height))
    target = max(64, int(target or 768))
    scale = target / max(width, height)
    out_w = max(64, int(round(width * scale / 8) * 8))
    out_h = max(64, int(round(height * scale / 8) * 8))
    return out_w, out_h


def preprocess_image(
    image: Image.Image,
    *,
    kind: str = "canny",
    resolution: int = 768,
    low_threshold: int = 80,
    high_threshold: int = 160,
) -> Image.Image:
    out_w, out_h = pixel_perfect_resolution(image.width, image.height, resolution)
    gray = ImageOps.grayscale(image.convert("RGB").resize((out_w, out_h), Image.LANCZOS))

    if kind == "soft_edge":
        return gray.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(1.2)).convert("RGB")
    if kind == "lineart":
        edges = gray.filter(ImageFilter.FIND_EDGES)
        return ImageOps.autocontrast(ImageOps.invert(edges)).convert("RGB")
    if kind == "depth":
        # Lightweight placeholder depth preview: luminance depth candidate, not model depth.
        return ImageOps.autocontrast(gray).convert("RGB")
    if kind == "canny":
        edges = gray.filter(ImageFilter.FIND_EDGES)
        threshold = max(1, min(255, (int(low_threshold) + int(high_threshold)) // 2))
        return edges.point(lambda p: 255 if p >= threshold else 0).convert("RGB")
    raise ValueError(f"Unknown preprocessor: {kind}")
