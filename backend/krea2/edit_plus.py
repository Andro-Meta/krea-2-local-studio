from __future__ import annotations

IMAGE_EDIT_PLUS_SYSTEM_PROMPT = (
    "You are conditioning an image-edit diffusion model. Read the user's edit instruction, "
    "compare the provided reference images, preserve identity and layout when requested, "
    "and encode concise visual guidance for the target image."
)
EDIT_PLUS_IMAGE_SIZE = 384
EDIT_PLUS_MAX_IMAGES = 3


def wrap_image_edit_plus_prompt(text: str, n_images: int, negative_text: str = "") -> str:
    capped = max(0, min(EDIT_PLUS_MAX_IMAGES, int(n_images)))
    img_prefix = "".join(
        f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"
        for i in range(capped)
    )
    negative = f"\nAvoid: {negative_text.strip()}" if negative_text.strip() else ""
    return (
        f"<|im_start|>system\n{IMAGE_EDIT_PLUS_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{img_prefix}Edit instruction: {text.strip()}{negative}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def resize_edit_plus_images(images: list) -> list:
    return [
        img.resize((EDIT_PLUS_IMAGE_SIZE, EDIT_PLUS_IMAGE_SIZE)).convert("RGB")
        for img in images[:EDIT_PLUS_MAX_IMAGES]
    ]
