"""Krea 2 text (+ optional image) encoder using Qwen3-VL-4B.

Text path is a faithful port of the official krea-ai/krea-2 encoder.py:
it taps 12 hidden-state layers and returns a 4D conditioning tensor
(B, seq, 12, 2560) that the MMDiT's txtfusion consumes directly
(b, l, n, d → projector Linear(12→1) collapses the 12 taps).

Extended with forward_with_images() for multimodal reference-image conditioning.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

from support_models import support_model_path

logger = logging.getLogger(__name__)

# Layer indices to tap for conditioning (12 taps × 2560 dim).
SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# Official prompt template. The system preamble is a fixed-length prefix that is
# tokenized then sliced off (prefix_idx) so only the user content + assistant
# turn marker feed the conditioning.
PROMPT_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_IDX = 34          # prompt_template_encode_start_idx
SUFFIX_START_IDX = 5     # prompt_template_encode_suffix_start_idx

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


def resolve_conditioner_source(version: str | None = None) -> dict[str, str]:
    """Resolve the preferred Krea text-encoder asset.

    ComfyUI runs Krea 2 with a separate FP8 Qwen3-VL asset. We detect that
    asset so the app can report it and use it once a native FP8 transformer path
    is available; the current safe runtime path remains the HF/Transformers bf16
    model.
    """
    if version:
        return {"kind": "hf_bf16", "path": version, "runtime": "hf_bf16", "status": "HF BF16 runtime"}
    if os.environ.get("KREA2_TEXT_ENCODER", "").lower() == "bf16":
        return {
            "kind": "hf_bf16",
            "path": "Qwen/Qwen3-VL-4B-Instruct",
            "runtime": "hf_bf16",
            "status": "HF BF16 runtime forced by KREA2_TEXT_ENCODER",
        }
    try:
        root = support_model_path("qwen3_vl_fp8")
        candidate = Path(root) / "text_encoders" / "qwen3vl_4b_fp8_scaled.safetensors"
        if candidate.exists():
            return {
                "kind": "comfy_fp8",
                "path": str(candidate),
                "runtime": "hf_bf16_fallback",
                "status": "FP8 asset installed; runtime unsupported, using HF BF16 fallback",
            }
    except Exception:
        pass
    return {
        "kind": "hf_bf16",
        "path": "Qwen/Qwen3-VL-4B-Instruct",
        "runtime": "hf_bf16",
        "status": "HF BF16 runtime",
    }


def _wrap_image_prompt(text: str, n_images: int) -> str:
    img_prefix = "".join(
        f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"
        for i in range(n_images)
    )
    return (
        f"<|im_start|>system\n{IMAGE_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{img_prefix}{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class Qwen3VLConditioner(nn.Module):
    def __init__(
        self,
        version: str = "Qwen/Qwen3-VL-4B-Instruct",
        select_layers: tuple = SELECT_LAYERS,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        from transformers import (
            AutoTokenizer,
            Qwen2TokenizerFast,
            Qwen3VLForConditionalGeneration,
        )

        self._version = version
        self.select_layers = select_layers
        self.max_length = max_length
        self.source = resolve_conditioner_source()
        if self.source["kind"] == "comfy_fp8":
            logger.info(
                "ComfyUI FP8 Qwen3-VL asset detected at %s; using HF bf16 runtime fallback.",
                self.source["path"],
            )

        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            version,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.qwen = self.qwen.eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(version, max_length=max_length)
        # Separate fast tokenizer for the assistant-turn suffix (matches official).
        self.processor = Qwen2TokenizerFast.from_pretrained(version, max_length=max_length)
        self._vision_processor = None

    def _stack_layers(
        self,
        hidden_states: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Stack the selected layer taps → (B, seq, 12, 2560)."""
        return torch.stack(
            [hidden_states[i] for i in self.select_layers], dim=2
        )

    def forward(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text-only prompts.

        Returns:
            txt:     (B, seq, 12, 2560) conditioning (4D, txtfusion-ready)
            txtmask: (B, seq) attention mask
        """
        if len(prompts) > 1 and len(set(prompts)) == 1:
            txt, mask = self._encode_unique_text([prompts[0]])
            return txt.expand(len(prompts), -1, -1, -1), mask.expand(len(prompts), -1)
        return self._encode_unique_text(prompts)

    def _encode_unique_text(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.qwen.parameters()).device

        text = [PROMPT_PREFIX + p for p in prompts]
        suffix_text = [PROMPT_SUFFIX] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(device)
        suffix_ids = suffix_inputs["input_ids"]
        suffix_mask = suffix_inputs["attention_mask"].bool()

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length + PREFIX_IDX - SUFFIX_START_IDX,
            return_tensors="pt",
        ).to(device)

        input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)

        out = self.qwen(
            input_ids=input_ids,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hiddens = self._stack_layers(out.hidden_states)[:, PREFIX_IDX:]
        mask = mask[:, PREFIX_IDX:]
        return hiddens, mask

    def _get_vision_processor(self):
        if self._vision_processor is None:
            from transformers import Qwen3VLProcessor

            self._vision_processor = Qwen3VLProcessor.from_pretrained(self._version)
        return self._vision_processor

    def forward_with_images(
        self,
        prompts: list[str],
        images: list,  # list[PIL.Image]
        token_size: str = "normal",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts + reference images via Qwen3-VL vision path.

        Returns 4D conditioning (B, seq, 12, 2560) like forward(). Falls back to
        text-only encoding on any failure.
        """
        try:
            processor = self._get_vision_processor()
        except ImportError:
            logger.warning("Qwen3VLProcessor unavailable; text-only fallback")
            return self.forward(prompts)

        device = next(self.qwen.parameters()).device
        n_images = len(images)
        image_size = IMAGE_TOKEN_SIZES.get(token_size, IMAGE_TOKEN_SIZES["normal"])
        resized = [img.resize((image_size, image_size)).convert("RGB") for img in images]
        wrapped = _wrap_image_prompt(prompts[0], n_images)

        try:
            inputs = processor(text=wrapped, images=resized, return_tensors="pt").to(device)
            out = self.qwen(**inputs, output_hidden_states=True, return_dict=True)

            hiddens = self._stack_layers(out.hidden_states)  # (1, seq, 12, 2560)
            mask = inputs.get("attention_mask")
            if mask is None:
                mask = torch.ones(
                    hiddens.shape[0], hiddens.shape[1], dtype=torch.bool, device=device
                )
            else:
                mask = mask.bool()

            # Replicate single-image conditioning across the batch.
            B = len(prompts)
            if B > 1:
                hiddens = hiddens.expand(B, -1, -1, -1)
                mask = mask.expand(B, -1)
            return hiddens, mask

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Multimodal encoding failed ({e}); text-only fallback")
            return self.forward(prompts)
