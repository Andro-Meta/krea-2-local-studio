"""Text-prompted automatic mask generation via CLIPSeg (runs on CPU).

Lets the inpaint UI generate a mask from a text description ("the sky", "the
person, the car") instead of hand-painting. CLIPSeg (~600MB, transformers) is
kept on CPU so it never competes with the 13GB DiT for VRAM. This is the
practical stand-in for the ComfyUI SAM3-Detect node's text→mask UX.
"""
from __future__ import annotations

import logging
import threading

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

_MODEL_ID = "CIDAS/clipseg-rd64-refined"
_model = None
_proc = None
_lock = threading.Lock()


def _load():
    global _model, _proc
    if _model is None:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        logger.info("Loading CLIPSeg auto-mask model (CPU)...")
        _proc = CLIPSegProcessor.from_pretrained(_MODEL_ID)
        _model = CLIPSegForImageSegmentation.from_pretrained(_MODEL_ID).to("cpu").eval()
        logger.info("CLIPSeg loaded.")
    return _model, _proc


@torch.inference_mode()
def generate_mask(image: Image.Image, prompt: str, threshold: float = 0.35,
                  dilate: int = 6) -> Image.Image:
    """Return an L-mode mask (white=regenerate) for the described region(s).

    `prompt` may be comma-separated for a union of regions. Per-prompt min-max
    normalization handles CLIPSeg's uncalibrated logits; absent prompts (low
    peak response) are dropped so they don't force a spurious mask.
    """
    image = image.convert("RGB")
    W, H = image.size
    prompts = [p.strip() for p in prompt.split(",") if p.strip()] or [prompt]

    with _lock:
        model, proc = _load()
        inputs = proc(text=prompts, images=[image] * len(prompts),
                      padding=True, return_tensors="pt")
        logits = model(**inputs).logits

    if logits.ndim == 2:                       # single-prompt squeeze
        logits = logits.unsqueeze(0)
    probs = torch.sigmoid(logits)              # (N, 352, 352)
    probs = F.interpolate(probs.unsqueeze(1), size=(H, W),
                          mode="bilinear", align_corners=False).squeeze(1)

    masks = []
    for i in range(probs.shape[0]):
        p = probs[i]
        if float(p.max()) < 0.15:              # prompt absent → skip
            continue
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)
        masks.append(p > threshold)
    if not masks:
        return Image.new("L", (W, H), 0)

    union = torch.stack(masks).any(dim=0)      # (H, W) bool
    mask_img = Image.fromarray((union.cpu().numpy() * 255).astype("uint8"), mode="L")
    if dilate > 0:                             # over-cover slightly to avoid halos
        mask_img = mask_img.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
    return mask_img


if __name__ == "__main__":  # tiny self-check
    im = Image.new("RGB", (256, 256), (30, 30, 30))
    m = generate_mask(im, "the center", threshold=0.35)
    assert m.size == (256, 256) and m.mode == "L"
    print("automask demo ok:", m.size)
