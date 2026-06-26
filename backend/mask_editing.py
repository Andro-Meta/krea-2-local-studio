from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PIL import Image, ImageFilter


@dataclass(frozen=True)
class MaskCrop:
    image: Image.Image
    mask: Image.Image
    feathered_mask: Image.Image
    box: tuple[int, int, int, int]


def _align_down(value: int, align: int) -> int:
    return max(0, (value // align) * align)


def _align_up(value: int, limit: int, align: int) -> int:
    return min(limit, ((value + align - 1) // align) * align)


def crop_for_mask(
    image: Image.Image,
    mask: Image.Image,
    *,
    padding: int = 96,
    align: int = 16,
    feather: int = 16,
) -> MaskCrop:
    base = image.convert("RGB")
    mask_l = mask.convert("L").resize(base.size, Image.Resampling.BILINEAR)
    bbox = mask_l.getbbox()
    if bbox is None:
        raise ValueError("Mask has no editable pixels.")

    left = _align_down(max(0, bbox[0] - padding), align)
    top = _align_down(max(0, bbox[1] - padding), align)
    right = _align_up(min(base.width, bbox[2] + padding), base.width, align)
    bottom = _align_up(min(base.height, bbox[3] + padding), base.height, align)
    box = (left, top, right, bottom)

    cropped_image = base.crop(box)
    cropped_mask = mask_l.crop(box)
    feathered = cropped_mask.filter(ImageFilter.GaussianBlur(radius=max(0, feather)))
    return MaskCrop(cropped_image, cropped_mask, feathered, box)


def composite_crop(
    base: Image.Image,
    generated_crop: Image.Image,
    feathered_mask: Image.Image,
    box: tuple[int, int, int, int],
    *,
    mode: Literal["masked", "replace"] = "masked",
) -> Image.Image:
    result = base.convert("RGB").copy()
    target_size = (box[2] - box[0], box[3] - box[1])
    generated = generated_crop.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    mask = feathered_mask.convert("L").resize(target_size, Image.Resampling.BILINEAR)
    if mode == "replace":
        mask = Image.new("L", target_size, 255)
    result.paste(generated, box[:2], mask)
    return result
