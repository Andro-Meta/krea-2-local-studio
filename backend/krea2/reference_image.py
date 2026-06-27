from __future__ import annotations

IMAGE_SYSTEM_PROMPT = (
    "Describe the key features of the input image (color, shape, size, texture, objects, "
    "background), then explain how the user's text instruction should alter or modify the "
    "image. Generate a new image that meets the user's requirements while maintaining "
    "consistency with the original input where appropriate."
)
IMAGE_TOKEN_SIZES = {
    "low": 256,
    "normal": 512,
    "high": 1024,
    "max": 1280,
}
MAX_SYSTEM_PROMPT_CHARS = 512


def bounded_system_prompt(system_prompt: str | None = None) -> str:
    value = (system_prompt or IMAGE_SYSTEM_PROMPT).strip()
    return value[:MAX_SYSTEM_PROMPT_CHARS] or IMAGE_SYSTEM_PROMPT


def wrap_image_prompt(
    text: str,
    n_images: int,
    *,
    system_prompt: str | None = None,
    vision_position: str = "before_prompt",
) -> str:
    img_prefix = "".join(
        f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"
        for i in range(n_images)
    )
    user_content = f"{img_prefix}{text}" if vision_position != "after_prompt" else f"{text}{img_prefix}"
    return (
        f"<|im_start|>system\n{bounded_system_prompt(system_prompt)}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def crop_image_to_mask(image, mask, padding: int = 0):
    mask = mask.convert("L").resize(image.size)
    bbox = mask.point(lambda value: 255 if value > 0 else 0).getbbox()
    if bbox is None:
        return image.convert("RGB")
    left, top, right, bottom = bbox
    pad = max(0, int(padding or 0))
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image.width, right + pad)
    bottom = min(image.height, bottom + pad)
    return image.convert("RGB").crop((left, top, right, bottom))


def cap_vision_megapixels(image, megapixels: float | None = None):
    if megapixels is None or megapixels <= 0:
        return image.convert("RGB")
    max_pixels = int(float(megapixels) * 1_000_000)
    current_pixels = image.width * image.height
    if max_pixels <= 0 or current_pixels <= max_pixels:
        return image.convert("RGB")
    scale = (max_pixels / float(current_pixels)) ** 0.5
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    return image.convert("RGB").resize((width, height))
