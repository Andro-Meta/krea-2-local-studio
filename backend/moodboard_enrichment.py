from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from PIL import Image

GUIDANCE_VERSION = 1
GuidanceMode = Literal["official", "custom", "mashup"]


@dataclass
class MoodboardSource:
    title: str
    taste_profile: str = ""
    keywords: list[str] = field(default_factory=list)
    image_b64s: list[str] = field(default_factory=list)
    weight: float = 1.0


QwenGenerator = Callable[[str, list[str]], str]
SUBJECT_LOCK_TERMS = (
    "crowd", "crowds", "people", "person", "persons", "human", "humans",
    "figure", "figures", "man", "woman", "men", "women", "child", "children",
    "animal", "animals", "text", "lettering", "building", "buildings",
    "architecture", "architectural", "vehicle", "vehicles", "face", "faces",
    "subject", "subjects",
    "populated", "unpopulated", "empty scene", "empty scenes",
)
DESOLATION_TERMS = ("apocalyptic", "desolate", "desolation", "isolation", "solitude", "wasteland", "empty", "sparse")


def _strip_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else text.strip()


def _json_object(text: str) -> dict:
    raw = _strip_fence(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Qwen did not return a JSON object.")
        data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("Qwen guidance must be a JSON object.")
    return data


def _string_list(value: object, *, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def parse_qwen_guidance_json(text: str, *, allow_catalog_metadata: bool = False) -> dict:
    data = _json_object(text)
    guidance = {
        "prompt_guidance": str(data.get("prompt_guidance") or "").strip(),
        "negative_guidance": str(data.get("negative_guidance") or "").strip(),
        "style_axes": _string_list(data.get("style_axes")),
        "conditioning_notes": _string_list(data.get("conditioning_notes")),
        "source_summary": str(data.get("source_summary") or "").strip(),
        "guidance_version": int(data.get("guidance_version") or GUIDANCE_VERSION),
    }
    if not guidance["prompt_guidance"]:
        raise ValueError("Qwen guidance requires prompt_guidance.")

    if allow_catalog_metadata:
        guidance["title"] = str(data.get("title") or "").strip()
        guidance["taste_profile"] = str(data.get("taste_profile") or "").strip()
        guidance["keywords"] = _string_list(data.get("keywords"))

    return sanitize_transferable_guidance(guidance)


def _strip_subject_locked_negative(text: str) -> str:
    clauses = re.split(r"(?<=[.;])\s+|,\s+and\s+|,\s+or\s+", str(text or ""))
    kept: list[str] = []
    for clause in clauses:
        low = clause.lower()
        # Keep style-quality clauses, remove clauses that ban subject categories.
        if any(term in low for term in SUBJECT_LOCK_TERMS):
            continue
        cleaned = clause.strip(" ,;.")
        if cleaned:
            kept.append(cleaned)
    return ". ".join(kept)


def _abstract_subject_locked_prompt(text: str) -> str:
    replacements = [
        (r"\ba lone figure\b", "silhouette-style contrast"),
        (r"\bthe lone figure\b", "the silhouette-style contrast"),
        (r"\bsolitary silhouette(s)?\b", "silhouette-style contrast"),
        (r"\bcentered figure\b", "center-weighted contrast"),
        (r"\bfacing away from the viewer\b", "backlit compositional tension"),
        (r"\bmany people\b", "multi-subject compositions"),
        (r"\bno people\b", "subject-agnostic compositions"),
    ]
    result = str(text or "")
    for pattern, repl in replacements:
        result = re.sub(pattern, repl, result, flags=re.I)
    return result


def sanitize_transferable_guidance(guidance: dict) -> dict:
    """Remove subject locks from Qwen moodboard guidance.

    Moodboards should transfer style to arbitrary user subjects. Qwen may describe
    a source image too literally; this keeps palette/lighting/texture guidance
    while avoiding bans on people/crowds/text/architecture/etc.
    """
    cleaned = dict(guidance)
    cleaned["prompt_guidance"] = _abstract_subject_locked_prompt(str(cleaned.get("prompt_guidance") or ""))
    cleaned["negative_guidance"] = _strip_subject_locked_negative(str(cleaned.get("negative_guidance") or ""))
    notes = _string_list(cleaned.get("conditioning_notes"))
    transfer_note = "Apply the moodboard as transferable style; do not override the user's requested subject count or content."
    if transfer_note not in notes:
        notes.append(transfer_note)
    style_blob = " ".join(
        [
            str(cleaned.get("prompt_guidance") or ""),
            str(cleaned.get("source_summary") or ""),
            " ".join(str(v) for v in cleaned.get("style_axes", []) or []),
        ]
    ).lower()
    if any(term in style_blob for term in DESOLATION_TERMS):
        desolation_note = "Preserve desolation, isolation, or sparse end-of-world atmosphere as mood pressure, while still honoring requested people/count/content."
        if desolation_note not in notes:
            notes.append(desolation_note)
    cleaned["conditioning_notes"] = notes[:12]
    return cleaned


def _source_block(source: MoodboardSource, index: int) -> str:
    keywords = ", ".join(source.keywords)
    return (
        f"Source {index} (weight {source.weight:.2f})\n"
        f"Title: {source.title or '(missing)'}\n"
        f"Taste profile: {source.taste_profile or '(missing)'}\n"
        f"Keywords: {keywords or '(missing)'}"
    )


def build_moodboard_guidance_prompt(sources: list[MoodboardSource], mode: GuidanceMode) -> str:
    if not sources:
        raise ValueError("At least one moodboard source is required.")
    source_text = "\n\n".join(_source_block(source, idx + 1) for idx, source in enumerate(sources))
    metadata_rule = (
        "For official mode, do not output title, taste_profile, or keywords. "
        "Those official Krea catalog fields are authoritative and must not be rewritten."
        if mode == "official"
        else "For custom or mashup mode, output title, taste_profile, and keywords when useful."
    )
    return (
        "You are the local Krea 2 moodboard implementation model. Convert moodboard ingredients into "
        "generation-ready guidance for the local Krea/Qwen conditioning pipeline.\n\n"
        "Your job is to extract a transferable visual style, not recreate the example images. "
        "Do not describe the source image subject, people-count, exact object count, exact pose, or fixed composition "
        "unless the board is explicitly about that reusable composition style. The guidance must work if the user prompt "
        "asks for many people, no people, a product shot, an animal, architecture, text, or an abstract scene. "
        "Describe palette, lighting, lens/material texture, contrast, rendering medium, atmosphere, era, and composition tendencies as optional style pressure. "
        "Use phrases like 'apply this palette/lighting/texture' instead of 'show a lone figure'. "
        "Negative guidance must avoid off-style rendering qualities, not forbid valid user subjects such as crowds unless the style truly requires emptiness.\n\n"
        f"Mode: {mode}\n"
        f"{metadata_rule}\n\n"
        "Return strict JSON only. Required keys: prompt_guidance, negative_guidance, style_axes, "
        "conditioning_notes, source_summary. In custom/mashup mode also include title, taste_profile, "
        "and keywords.\n\n"
        f"{source_text}"
    )


def _fallback_guidance(sources: list[MoodboardSource], mode: GuidanceMode) -> dict:
    titles = [source.title for source in sources if source.title]
    tastes = [source.taste_profile for source in sources if source.taste_profile]
    keywords: list[str] = []
    for source in sources:
        for keyword in source.keywords:
            if keyword and keyword not in keywords:
                keywords.append(keyword)
    title_text = ", ".join(titles[:4])
    taste_text = " ".join(tastes[:4])
    keyword_text = ", ".join(keywords[:16])
    prompt_guidance = ". ".join(
        part for part in [
            f"Use the moodboard direction from {title_text}" if title_text else "",
            taste_text,
            f"Emphasize: {keyword_text}" if keyword_text else "",
        ] if part
    ).strip()
    if not prompt_guidance:
        prompt_guidance = "Use the uploaded moodboard references as visual style, palette, lighting, and texture guidance."
    guidance = {
        "prompt_guidance": prompt_guidance,
        "negative_guidance": "Avoid generic styling, mismatched lighting, visual clutter, and off-theme details.",
        "style_axes": keywords[:12] or ["moodboard style", "palette", "lighting", "texture"],
        "conditioning_notes": [
            "Use reference images for palette, lighting, surface texture, and composition mood.",
            "Keep the user prompt subject primary while applying moodboard style as art direction.",
        ],
        "source_summary": f"Fallback guidance synthesized from {len(sources)} moodboard source(s).",
        "guidance_version": GUIDANCE_VERSION,
        "guidance_backend": "heuristic_fallback",
    }
    if mode in {"custom", "mashup"}:
        guidance["title"] = titles[0] if titles else "Custom Moodboard"
        guidance["taste_profile"] = taste_text or prompt_guidance
        guidance["keywords"] = keywords[:12]
    return sanitize_transferable_guidance(guidance)


def _local_qwen_generate(prompt: str, image_b64s: list[str]) -> str:
    from prompt_expander import _decode_generation, _generation_kwargs, _input_ids, _load_local_qwen, _strip_data_url

    tokenizer, processor, model = _load_local_qwen()
    device = getattr(model, "device", "cpu")
    if image_b64s and processor is not None:
        images = [
            Image.open(io.BytesIO(base64.b64decode(_strip_data_url(image_b64)))).convert("RGB")
            for image_b64 in image_b64s[:10]
        ]
        pads = "".join("<|vision_start|><|image_pad|><|vision_end|>" for _ in images)
        inputs = processor(
            text=[f"<|im_start|>user\n{pads}{prompt}<|im_end|>\n<|im_start|>assistant\n"],
            images=images,
            return_tensors="pt",
        ).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=900,
            do_sample=True,
            temperature=0.45,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        return _decode_generation(tokenizer, outputs, inputs.get("input_ids") if isinstance(inputs, dict) else None)

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    outputs = model.generate(
        **_generation_kwargs(inputs),
        max_new_tokens=900,
        do_sample=True,
        temperature=0.45,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    return _decode_generation(tokenizer, outputs, _input_ids(inputs))


def generate_moodboard_guidance(
    sources: list[MoodboardSource],
    *,
    mode: GuidanceMode,
    generator: QwenGenerator | None = None,
) -> dict:
    if mode not in {"official", "custom", "mashup"}:
        raise ValueError("Unknown moodboard guidance mode.")
    prompt = build_moodboard_guidance_prompt(sources, mode)
    images: list[str] = []
    for source in sources:
        images.extend([image for image in source.image_b64s if image])
    response = (generator or _local_qwen_generate)(prompt, images[:10])
    try:
        guidance = parse_qwen_guidance_json(response, allow_catalog_metadata=mode in {"custom", "mashup"})
        guidance.setdefault("guidance_backend", "qwen")
        return guidance
    except ValueError:
        return _fallback_guidance(sources, mode)
