"""Prompt expansion and image description for Krea 2 Studio."""
from __future__ import annotations

import base64
import io
import logging
import json
import re
from collections.abc import Mapping
from functools import lru_cache
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import requests
from PIL import Image

logger = logging.getLogger(__name__)
LOCAL_QWEN_MODEL_ID = "qwen3_vl"
OPENROUTER_DEFAULT_MODEL = "google/gemma-4-31b-it:free"
OPENROUTER_FREE_FALLBACKS = (
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
)
OPENROUTER_VISION_FALLBACKS = (
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
)
GGUF_HELPER_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
GGUF_HELPER_DEFAULT_MODEL = "BennyDaBall/Krea-2-Engineer-V1-GGUF:Q4_K_M"
LOCAL_GGUF_HOSTS = {"127.0.0.1", "localhost", "::1"}
LOCAL_QWEN_MIN_FREE_VRAM_GB = 12.0
LOCAL_QWEN_MAX_EXISTING_CUDA_ALLOC_GB = 2.0

# Verbatim from Krea's official prompt expansion used by the krea.ai API
# (github.com/krea-ai/krea-2 docs/expansion.txt). Matching it locally closes the
# documented API-vs-local quality gap, since the API applies this expansion.
EXPANSION_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for text-to-image models. Your task is to expand "
    "the user's prompt into a highly effective image-generation prompt.\n\n"
    "Think step by step about the request before writing the answer:\n"
    "- What is the subject and mood?\n"
    "- What visual styles, mediums, and lighting options would fit? Consider two or three "
    "alternatives and pick the one that best serves the caption.\n"
    "- What composition, framing, and grounded details will help the text-to-image model?\n\n"
    "Then output a single expanded prompt paragraph.\n\n"
    "Follow these rules strictly:\n"
    "1. **Faithfulness First:** Preserve all original subjects, actions, colors, and spatial "
    "relationships. Do not add new objects, props, characters, or animals unless the user "
    "clearly implies them.\n"
    "2. **Practical T2I Structure:** Write a prompt that a text-to-image model can parse "
    "cleanly. Group subjects with their own attributes and actions. Use grounded phrasing for "
    "poses, interactions, and spatial layout.\n"
    "3. **Style Planning Stays Internal:** Use your internal reasoning to choose style, medium, "
    "framing, and lighting. Do not emit planning tags or wrappers in the visible answer body.\n"
    "4. **Text Rendering:** If the user requests visible text, quotes, labels, or typography, "
    "specify the exact text clearly and wrap requested words in quotes.\n"
    "5. **Avoid Over-Specification:** Do not invent highly specific clothing, colors, materials, "
    "or scene details unless the input supports them.\n"
    "6. **Structure:** Write one cohesive paragraph after the thinking block. No bullets, JSON, "
    "or markdown.\n"
    "7. **Respect Existing Detail:** If the user's prompt is already detailed, lightly polish "
    "and finalize rather than heavily expanding — preserve their phrasing and direction.\n"
    "8. **Respect the Human Form:** Treat depictions of people with dignity. Assume clothing "
    "covers genitals and intimate anatomy.\n"
    "9. **Preserve User Medium:** When the user explicitly requests a medium (e.g. \"photo of\", "
    "\"photograph of\", \"illustration of\", \"painting of\", \"sketch of\", \"3D render of\"), honor it. "
    "Do not pivot to a different medium to avoid difficulty — match the user's stated intent."
)


@dataclass
class PromptExpansionResult:
    expanded: str
    changed: bool
    error: str | None = None
    backend: str = "local"


@lru_cache(maxsize=2)
def _load_local_qwen(model_id: str = ""):
    import torch
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from support_models import support_model_path
    from settings import settings

    device = _resolve_local_qwen_device(torch)
    model_path = str(model_id or getattr(settings, "local_qwen_model_id", "") or support_model_path(LOCAL_QWEN_MODEL_ID))
    processor_source = model_path
    if "Huihui-Qwen3-VL-4B-Instruct-abliterated" in model_path:
        processor_source = "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"
    tokenizer = AutoTokenizer.from_pretrained(processor_source)
    processor = Qwen3VLProcessor.from_pretrained(processor_source)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval().to(device)
    return tokenizer, processor, model


def _resolve_local_qwen_device(torch_module) -> str:
    from settings import settings

    requested = str(getattr(settings, "local_qwen_device", "auto") or "auto").lower()
    if requested == "cpu":
        logger.info("Local Qwen prompt helper using CPU by setting.")
        return "cpu"
    if not torch_module.cuda.is_available():
        return "cpu"
    if requested == "cuda":
        return "cuda"
    try:
        free_b, total_b = torch_module.cuda.mem_get_info()
        free_gb = free_b / (1024 ** 3)
        total_gb = total_b / (1024 ** 3)
        allocated_gb = (
            torch_module.cuda.memory_allocated() / (1024 ** 3)
            if hasattr(torch_module.cuda, "memory_allocated")
            else 0.0
        )
    except Exception:
        logger.info("Local Qwen prompt helper using CPU; could not read CUDA memory state.")
        return "cpu"
    if allocated_gb > LOCAL_QWEN_MAX_EXISTING_CUDA_ALLOC_GB:
        logger.warning(
            "Local Qwen prompt helper using CPU because this process already has %.1fGB CUDA allocated. "
            "This avoids stacking the Magic Wand model beside the active Krea pipeline.",
            allocated_gb,
        )
        return "cpu"
    if free_gb < LOCAL_QWEN_MIN_FREE_VRAM_GB:
        logger.warning(
            "Local Qwen prompt helper using CPU because only %.1f/%.1fGB VRAM is free. "
            "This avoids loading a second Qwen3-VL model beside the active Krea pipeline.",
            free_gb,
            total_gb,
        )
        return "cpu"
    logger.info("Local Qwen prompt helper using CUDA (%.1f/%.1fGB VRAM free).", free_gb, total_gb)
    return "cuda"


def unload_local_qwen() -> None:
    """Drop cached local Qwen helper models before memory-sensitive generation."""
    _load_local_qwen.cache_clear()
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        logger.debug("Could not fully clear local Qwen helper cache", exc_info=True)


def _strip_data_url(image_b64: str) -> str:
    return image_b64.split(",", 1)[1] if "," in image_b64 else image_b64


def _decode_generation(tokenizer, outputs, inputs=None) -> str:
    output = outputs[0] if outputs is not None else []
    try:
        input_len = int(getattr(inputs, "shape", [0, 0])[-1]) if inputs is not None else 0
        output = output[input_len:]
    except Exception:
        output = outputs[0] if outputs is not None else []
    text = tokenizer.decode(output, skip_special_tokens=True).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def _input_ids(inputs):
    if isinstance(inputs, Mapping):
        return inputs.get("input_ids")
    return inputs


def _generation_kwargs(inputs) -> dict:
    if isinstance(inputs, Mapping):
        return dict(inputs)
    return {"input_ids": inputs}


def expand_prompt_local(prompt: str) -> PromptExpansionResult:
    try:
        from settings import settings
        tokenizer, _processor, model = _load_local_qwen(str(getattr(settings, "local_qwen_model_id", "") or ""))
        messages = [
            {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(getattr(model, "device", "cpu"))
        outputs = model.generate(
            **_generation_kwargs(inputs),
            max_new_tokens=700,
            do_sample=True,
            temperature=0.7,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        expanded = _decode_generation(tokenizer, outputs, _input_ids(inputs)) or prompt
        return PromptExpansionResult(
            expanded=expanded,
            changed=expanded.strip() != prompt.strip(),
            error=None if expanded.strip() else "Local Qwen3-VL returned an empty expansion.",
            backend="local",
        )
    except Exception as exc:
        msg = (
            "Local Qwen3-VL prompt expansion failed. Use System > Krea Moodboard "
            f"Conditioning / Local AI Assets to repair the local model. Details: {exc}"
        )
        logger.warning(msg)
        return PromptExpansionResult(expanded=prompt, changed=False, error=msg, backend="local")


def describe_image_local(image_b64: str) -> dict[str, str]:
    from settings import settings
    tokenizer, processor, model = _load_local_qwen(str(getattr(settings, "local_qwen_model_id", "") or ""))
    if processor is None:
        raise RuntimeError("Local Qwen3-VL processor is unavailable.")
    image = Image.open(io.BytesIO(base64.b64decode(_strip_data_url(image_b64)))).convert("RGB")
    prompt = (
        "Write one vivid text-to-image prompt that could recreate this image. "
        "Return one paragraph only. Include subject, setting, composition, medium, "
        "lighting, color palette, texture, camera/art details, and mood. Do not "
        "mention that you are looking at an image."
    )
    inputs = processor(
        text=[(
            "<|im_start|>user\n"
            f"<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )],
        images=[image],
        return_tensors="pt",
    ).to(getattr(model, "device", "cpu"))
    outputs = model.generate(
        **inputs,
        max_new_tokens=420,
        do_sample=True,
        temperature=0.6,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    text = _decode_generation(tokenizer, outputs, inputs.get("input_ids") if isinstance(inputs, dict) else None)
    if not text:
        raise RuntimeError("Local Qwen3-VL returned an empty image description.")
    return {"prompt": text, "backend": "local"}


def coerce_openrouter_model(model: str, free_only: bool) -> str:
    if free_only and not (model or "").endswith(":free"):
        return OPENROUTER_DEFAULT_MODEL
    return model or OPENROUTER_DEFAULT_MODEL


def openrouter_fallback_models(model: str) -> list[str] | None:
    if not model.endswith(":free"):
        return None
    return ([model] + [m for m in OPENROUTER_FREE_FALLBACKS if m != model])[:3]


def openrouter_error_hint(exc: Exception) -> str:
    s = str(exc)
    if "401" in s or "Unauthorized" in s:
        return (
            "OpenRouter rejected the API key. Use a regular inference key from "
            "openrouter.ai/keys that starts with sk-or-v1-."
        )
    if "402" in s:
        return "OpenRouter says this model requires credits. Turn on free-only or choose a free model."
    if "429" in s:
        return "OpenRouter rate limit reached. Wait a bit or choose another free model."
    return f"OpenRouter prompt expansion failed: {s}"


def describe_image_openrouter(image_b64: str, api_key: str) -> dict[str, str]:
    if not api_key:
        raise RuntimeError("OpenRouter API key is missing. Add it in System settings.")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8200/krea/",
            "X-Title": "Krea 2 Studio",
        },
        json={
            "model": OPENROUTER_VISION_FALLBACKS[0],
            "models": list(OPENROUTER_VISION_FALLBACKS),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Look at this image and write one vivid text-to-image prompt "
                                "that could recreate it. Return one paragraph only. Include "
                                "subject, setting, composition, medium, lighting, color palette, "
                                "texture, camera/art details, and mood. Do not mention that you "
                                "are looking at an image."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
            "temperature": 0.7,
            "max_tokens": 420,
        },
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not text:
        raise RuntimeError("OpenRouter returned an empty image description.")
    return {"prompt": text, "backend": "openrouter"}


def ideogram_json_to_krea_prompt(value: dict | str) -> str:
    """Flatten Ideogram v4 json_prompt into one Krea-friendly prompt paragraph."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value.strip()
    if not isinstance(value, dict):
        return ""

    parts: list[str] = []
    high = str(value.get("high_level_description", "")).strip()
    if high:
        parts.append(high)

    style = value.get("style_description") or {}
    if isinstance(style, dict):
        style_bits: list[str] = []
        for key in ("aesthetics", "lighting", "photo", "medium", "art_style"):
            bit = style.get(key)
            if isinstance(bit, str) and bit.strip():
                style_bits.append(bit.strip())
        palette = style.get("color_palette")
        if isinstance(palette, list) and palette:
            style_bits.append("color palette " + ", ".join(str(c) for c in palette[:8]))
        if style_bits:
            parts.append("Style: " + ", ".join(style_bits))

    comp = value.get("compositional_deconstruction") or {}
    if isinstance(comp, dict):
        background = str(comp.get("background", "")).strip()
        if background:
            parts.append("Background: " + background)
        elements = comp.get("elements")
        if isinstance(elements, list):
            element_bits: list[str] = []
            for el in elements[:8]:
                if isinstance(el, dict):
                    desc = str(el.get("desc") or el.get("text") or "").strip()
                    if desc:
                        element_bits.append(desc)
            if element_bits:
                parts.append("Key elements: " + "; ".join(element_bits))

    return ". ".join(p.rstrip(".") for p in parts if p).strip() + ("." if parts else "")


def ideogram_error_hint(exc: Exception) -> str:
    s = str(exc)
    if "401" in s or "Unauthorized" in s:
        return "Ideogram rejected the API key. Add a valid Ideogram API key in System settings."
    if "429" in s:
        return "Ideogram Magic Prompt is rate-limited. Try again later or use OpenRouter."
    return f"Ideogram Magic Prompt failed: {s}"


def expand_prompt_ideogram_json(prompt: str, api_key: str, aspect_ratio: str = "1x1") -> PromptExpansionResult:
    if not api_key:
        return PromptExpansionResult(
            expanded=prompt,
            changed=False,
            error="Ideogram API key is missing. Add it in System settings.",
            backend="ideogram-json",
        )
    try:
        resp = requests.post(
            "https://api.ideogram.ai/v1/ideogram-v4/magic-prompt",
            headers={"Api-Key": api_key, "Content-Type": "application/json"},
            json={"text_prompt": prompt, "aspect_ratio": aspect_ratio},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        json_prompt = data.get("json_prompt") or data.get("magic_prompt_json") or data
        expanded = ideogram_json_to_krea_prompt(json_prompt) or prompt
        return PromptExpansionResult(
            expanded=expanded,
            changed=expanded.strip() != prompt.strip(),
            error=None if expanded.strip() else "Ideogram returned an empty prompt.",
            backend="ideogram-json",
        )
    except Exception as e:
        msg = ideogram_error_hint(e)
        logger.warning(f"{msg}; using original.")
        return PromptExpansionResult(expanded=prompt, changed=False, error=msg, backend="ideogram-json")


def expand_prompt_openrouter(
    prompt: str,
    api_key: str,
    model: str = OPENROUTER_DEFAULT_MODEL,
    free_only: bool = True,
) -> PromptExpansionResult:
    if not api_key:
        return PromptExpansionResult(
            expanded=prompt,
            changed=False,
            error="OpenRouter API key is missing. Add it in System settings.",
            backend="openrouter",
        )
    try:
        selected = coerce_openrouter_model(model, free_only)
        fallbacks = openrouter_fallback_models(selected)
        payload = {
            "model": selected,
            "messages": [
                {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.8,
            "max_tokens": 700,
        }
        if fallbacks:
            payload["models"] = fallbacks
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8200/krea/",
                "X-Title": "Krea 2 Studio",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        expanded = text if text else prompt
        return PromptExpansionResult(
            expanded=expanded,
            changed=expanded.strip() != prompt.strip(),
            error=None if text else "OpenRouter returned an empty expansion.",
            backend="openrouter",
        )
    except Exception as e:
        msg = openrouter_error_hint(e)
        logger.warning(f"{msg}; using original.")
        return PromptExpansionResult(expanded=prompt, changed=False, error=msg, backend="openrouter")


def _strip_think_blocks(text: str, *, limit: int = 20000) -> str:
    value = str(text or "")[:limit]
    while True:
        start = value.lower().find("<think>")
        if start < 0:
            return value.strip()
        end = value.lower().find("</think>", start + 7)
        if end < 0:
            return (value[:start] + value[start + 7:]).strip()
        value = value[:start] + value[end + 8:]


def _safe_local_openai_base_url(base_url: str) -> str:
    parsed = urlparse(str(base_url or GGUF_HELPER_DEFAULT_BASE_URL).strip())
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOCAL_GGUF_HOSTS:
        raise ValueError("GGUF helper base URL must be a local OpenAI-compatible endpoint.")
    path = (parsed.path or "/v1").rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def gguf_chat_completion(
    messages: list[dict],
    *,
    base_url: str = GGUF_HELPER_DEFAULT_BASE_URL,
    model: str = GGUF_HELPER_DEFAULT_MODEL,
    max_tokens: int = 700,
    temperature: float = 0.7,
    timeout: int = 120,
) -> str:
    url = _safe_local_openai_base_url(base_url) + "/chat/completions"
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "model": model or GGUF_HELPER_DEFAULT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return _strip_think_blocks(text)


def expand_prompt_gguf_server(
    prompt: str,
    *,
    base_url: str = GGUF_HELPER_DEFAULT_BASE_URL,
    model: str = GGUF_HELPER_DEFAULT_MODEL,
    timeout: int = 120,
) -> PromptExpansionResult:
    try:
        text = gguf_chat_completion(
            [
                {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_tokens=700,
            temperature=0.7,
        )
        expanded = text or prompt
        return PromptExpansionResult(
            expanded=expanded,
            changed=expanded.strip() != prompt.strip(),
            error=None if text else "GGUF helper returned an empty expansion.",
            backend="gguf-server",
        )
    except Exception as exc:
        msg = f"GGUF helper prompt expansion failed. Check the local OpenAI-compatible server. Details: {exc}"
        logger.warning(msg)
        return PromptExpansionResult(expanded=prompt, changed=False, error=msg, backend="gguf-server")


def expand_prompt_result(
    prompt: str,
    _legacy_url: str = "",
    _legacy_model: str = "",
    backend: str = "local",
    openrouter_api_key: str = "",
    openrouter_model: str = OPENROUTER_DEFAULT_MODEL,
    openrouter_free_only: bool = True,
    ideogram_api_key: str = "",
    gguf_helper_base_url: str = GGUF_HELPER_DEFAULT_BASE_URL,
    gguf_helper_model: str = GGUF_HELPER_DEFAULT_MODEL,
    gguf_helper_timeout_sec: int = 120,
) -> PromptExpansionResult:
    if backend == "ideogram-json":
        return expand_prompt_ideogram_json(prompt, ideogram_api_key)
    if backend == "openrouter":
        return expand_prompt_openrouter(prompt, openrouter_api_key, openrouter_model, openrouter_free_only)
    if backend in {"gguf", "gguf-server"}:
        return expand_prompt_gguf_server(
            prompt,
            base_url=gguf_helper_base_url,
            model=gguf_helper_model,
            timeout=gguf_helper_timeout_sec,
        )
    return expand_prompt_local(prompt)


def expand_prompt(
    prompt: str,
    _legacy_url: str = "",
    _legacy_model: str = "",
) -> str:
    return expand_prompt_result(prompt).expanded
