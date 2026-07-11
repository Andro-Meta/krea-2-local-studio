from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from PIL import Image
from gpu_recovery import is_cuda_oom

logger = logging.getLogger(__name__)

GUIDANCE_VERSION = 2
GuidanceMode = Literal["official", "custom", "mashup"]
MAX_GUIDANCE_ATTEMPTS = 3


@dataclass
class MoodboardSource:
    title: str
    taste_profile: str = ""
    keywords: list[str] = field(default_factory=list)
    image_b64s: list[str] = field(default_factory=list)
    weight: float = 1.0


QwenGenerator = Callable[[str, list[str]], str]
CancellationProbe = Callable[[], bool]
SUBJECT_LOCK_TERMS = (
    "crowd", "crowds", "people", "person", "persons", "human", "humans",
    "figure", "figures", "man", "woman", "men", "women", "child", "children",
    "animal", "animals", "text", "lettering", "building", "buildings",
    "architecture", "architectural", "vehicle", "vehicles", "face", "faces",
    "populated", "unpopulated", "empty scene", "empty scenes",
)
DESOLATION_TERMS = ("apocalyptic", "desolate", "desolation", "isolation", "solitude", "wasteland", "empty", "sparse")
SUBJECT_STYLE_REPLACEMENTS = (
    (
        re.compile(r"\b(?:a|an|the)\s+(?:black and white|high-contrast|glitchy|cinematic|close-up)?\s*portrait of\s+(?:a|an|the)?\s*[^.]+", re.I),
        "portrait-style framing for the user-requested subject",
    ),
    (
        re.compile(r"\baerial\s+top-down\s+view\s+of\s+[^.]+", re.I),
        "aerial top-down perspective with graphic scale, clean composition, and environmental texture",
    ),
    (
        re.compile(r"\b(?:a|an|the)\s+(?:(?:single|lone|solitary|young|old|elderly)\s+){0,3}(?:woman|man|girl|boy|person|figure|child|kayaker|face|subject|creature|animal)\b[^.]*", re.I),
        "the user-requested subject with the board's palette, lighting, texture, and spatial mood",
    ),
)

# Style-typed schema (guidance_version 2): every field only describes HOW things
# are rendered, so subject nouns are validated out before storage.
STYLE_SCHEMA_FIELDS = (
    ("palette", "Palette"),
    ("lighting", "Lighting"),
    ("medium_texture", "Medium and texture"),
    ("composition", "Composition"),
    ("atmosphere", "Atmosphere"),
    ("era_or_movement", "Era or movement"),
)

# Subject nouns that must never appear in transferable style guidance.
# Legitimate style vocabulary is carved out: "figure-ground" (design term),
# "first/third-person" (camera framing), "children's (story)book" (genre).
SUBJECT_VIOLATION_RE = re.compile(
    r"\b(?:"
    r"woman|women|man|men|girl|girls|boy|boys|(?<!first-)(?<!third-)persons?|people|human|humans|"
    r"figures?(?!-ground)|face|faces|child|children(?!'s\s+(?:story)?book)|crowd|crowds|body|bodies|"
    r"animal|animals|creature|creatures|kayaker|dancer|dancers|astronaut|warrior|"
    r"portrait\s+of"
    r")\b"
    r"|\b(?:lone|solitary|single|isolated)\s+(?:silhouette|subject)s?\b",
    re.I,
)

# Negative-prompt clauses that would fight image quality or the user's own
# rendering intent (e.g. "avoid detail", "avoid photorealism") are dropped.
NEGATIVE_QUALITY_BAN_RE = re.compile(
    r"\b(?:"
    r"photorealism|photorealistic|photo-realistic|realism|realistic|"
    r"sharp|sharpness|crisp|clarity|clear|"
    r"high[- ]resolution|resolution|"
    r"detail|detailed|details|quality|anatomical|anatomy"
    r")\b",
    re.I,
)


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


def _strip_subject_locked_negative(text: str) -> str:
    clauses = re.split(r"(?<=[.;])\s+|,\s+and\s+|,\s+or\s+", str(text or ""))
    kept: list[str] = []
    for clause in clauses:
        low = clause.lower()
        # Keep style-quality clauses, remove clauses that ban subject categories.
        if any(term in low for term in SUBJECT_LOCK_TERMS):
            continue
        # Remove clauses that ban detail/sharpness/photorealism: they degrade
        # quality globally and fight user prompts that ask for those qualities.
        if NEGATIVE_QUALITY_BAN_RE.search(clause):
            continue
        cleaned = clause.strip(" ,;.")
        if cleaned:
            kept.append(cleaned)
    return ". ".join(kept)


def find_subject_violations(guidance: dict) -> list[str]:
    """List subject-noun leaks in style guidance, as 'field: term' strings."""
    checks: list[tuple[str, str]] = [
        ("prompt_guidance", str(guidance.get("prompt_guidance") or "")),
        ("source_summary", str(guidance.get("source_summary") or "")),
    ]
    for key, _label in STYLE_SCHEMA_FIELDS:
        if key in guidance:
            checks.append((key, str(guidance.get(key) or "")))
    for axis in guidance.get("style_axes") or []:
        checks.append(("style_axes", str(axis)))
    violations: list[str] = []
    for field_name, text in checks:
        for match in SUBJECT_VIOLATION_RE.finditer(text):
            entry = f"{field_name}: '{match.group(0)}'"
            if entry not in violations:
                violations.append(entry)
    return violations


def sanitize_style_fragment(text: str) -> str:
    """Sanitize a short style fragment (keyword/axis) for prompt injection.

    Rewrites known subject-locked phrasings; drops the fragment entirely when a
    subject noun survives the rewrite.
    """
    return _sanitize_style_axis(text)


def _assemble_prompt_guidance(schema: dict) -> str:
    parts: list[str] = []
    for key, label in STYLE_SCHEMA_FIELDS:
        value = str(schema.get(key) or "").strip().strip(".")
        if value:
            parts.append(f"{label}: {value}.")
    return " ".join(parts)


def _filtered_negative_terms(values: object) -> str:
    kept: list[str] = []
    for term in _string_list(values, limit=8):
        low = term.lower()
        if NEGATIVE_QUALITY_BAN_RE.search(term):
            continue
        if any(lock in low for lock in SUBJECT_LOCK_TERMS):
            continue
        if term not in kept:
            kept.append(term)
    return ", ".join(kept[:6])


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
    for pattern, repl in SUBJECT_STYLE_REPLACEMENTS:
        result = pattern.sub(repl, result)
    return result


def _sanitize_style_axis(text: str) -> str:
    value = _abstract_subject_locked_prompt(str(text or "")).strip()
    low = value.lower()
    if any(term in low for term in SUBJECT_LOCK_TERMS):
        return ""
    return value


def sanitize_transferable_guidance(guidance: dict) -> dict:
    """Remove subject locks from Qwen moodboard guidance.

    Moodboards should transfer style to arbitrary user subjects. Qwen may describe
    a source image too literally; this keeps palette/lighting/texture guidance
    while avoiding bans on people/crowds/text/architecture/etc.
    """
    cleaned = dict(guidance)
    cleaned["prompt_guidance"] = _abstract_subject_locked_prompt(str(cleaned.get("prompt_guidance") or ""))
    cleaned["negative_guidance"] = _strip_subject_locked_negative(str(cleaned.get("negative_guidance") or ""))
    cleaned["style_axes"] = [axis for axis in (_sanitize_style_axis(axis) for axis in cleaned.get("style_axes", []) or []) if axis][:12]
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


def build_moodboard_guidance_prompt(
    sources: list[MoodboardSource],
    mode: GuidanceMode,
    *,
    violations: list[str] | None = None,
) -> str:
    if not sources:
        raise ValueError("At least one moodboard source is required.")
    source_text = "\n\n".join(_source_block(source, idx + 1) for idx, source in enumerate(sources))
    metadata_rule = (
        "For official mode, do not output title, taste_profile, or keywords. "
        "Those official Krea catalog fields are authoritative and must not be rewritten."
        if mode == "official"
        else "For custom or mashup mode, also include \"title\", \"taste_profile\", and \"keywords\" (style words only, no subjects)."
    )
    retry_feedback = ""
    if violations:
        listed = "; ".join(violations[:8])
        retry_feedback = (
            "\n\nYour previous answer leaked subject matter and was rejected. "
            f"Offending fields: {listed}. "
            "Rewrite so no field names any subject. Describe the rendering treatment only "
            "(e.g. instead of 'a lone figure in fog' write 'heavy fog with a single strong "
            "backlit contrast point').\n"
        )
    return (
        "You are the Krea 2 moodboard style analyst. Distill a transferable visual style from the "
        "moodboard sources so it can be applied to any user-requested subject.\n\n"
        "Hard rules:\n"
        "- Do not describe the source image subject, people-count, exact object count, exact pose, or fixed scene content. "
        "Never mention people, figures, faces, bodies, animals, creatures, vehicles, buildings, or any specific object from the source images.\n"
        "- Describe only HOW things are rendered, never WHAT is depicted. The guidance must work whether the user asks for "
        "many people, no people, a product shot, an animal, architecture, text, or an abstract scene.\n"
        "- If the sources share a compositional habit (strong silhouette lighting, centered framing, heavy negative space), "
        "phrase it as a treatment applied to whatever subject the user requests.\n\n"
        "Return strict JSON only with exactly these keys:\n"
        "  \"palette\": dominant colors and how they interact (one sentence)\n"
        "  \"lighting\": light quality, direction, and contrast behavior (one sentence)\n"
        "  \"medium_texture\": rendering medium, grain, surface texture, lens or film character (one sentence)\n"
        "  \"composition\": framing tendencies, negative space, scale, depth treatment, subject-agnostic (one sentence)\n"
        "  \"atmosphere\": mood and emotional tone as adjectives (one sentence)\n"
        "  \"era_or_movement\": art-historical era, movement, or genre reference (short phrase, or \"\")\n"
        "  \"style_axes\": 5-8 short style tags (palette/lighting/texture/mood terms only, no subjects)\n"
        "  \"negative_style_terms\": 3-6 rendering qualities that would break this style. Never ban subjects, "
        "and never ban detail, sharpness, resolution, or overall image quality.\n"
        "  \"source_summary\": one sentence describing the shared aesthetic without naming any subject\n"
        f"{metadata_rule}\n"
        f"{retry_feedback}\n"
        f"Mode: {mode}\n\n"
        f"{source_text}"
    )


def parse_style_schema_json(text: str, *, allow_catalog_metadata: bool = False) -> dict:
    """Parse the v2 style-typed schema and assemble the stored guidance dict.

    Accepts legacy prompt_guidance-shaped output as a fallback so a model that
    ignores the schema still produces usable (sanitized) guidance.
    """
    data = _json_object(text)
    has_schema = any(str(data.get(key) or "").strip() for key, _label in STYLE_SCHEMA_FIELDS)
    if has_schema:
        guidance = {
            "prompt_guidance": _assemble_prompt_guidance(data),
            "negative_guidance": _filtered_negative_terms(data.get("negative_style_terms")),
            "style_axes": _string_list(data.get("style_axes")),
            "conditioning_notes": _string_list(data.get("conditioning_notes")),
            "source_summary": str(data.get("source_summary") or "").strip(),
            "guidance_version": GUIDANCE_VERSION,
        }
        if not guidance["prompt_guidance"]:
            raise ValueError("Qwen style schema produced no usable style fields.")
    else:
        guidance = {
            "prompt_guidance": str(data.get("prompt_guidance") or "").strip(),
            "negative_guidance": str(data.get("negative_guidance") or "").strip(),
            "style_axes": _string_list(data.get("style_axes")),
            "conditioning_notes": _string_list(data.get("conditioning_notes")),
            "source_summary": str(data.get("source_summary") or "").strip(),
            "guidance_version": GUIDANCE_VERSION,
        }
        if not guidance["prompt_guidance"]:
            raise ValueError("Qwen guidance requires prompt_guidance.")

    if allow_catalog_metadata:
        guidance["title"] = str(data.get("title") or "").strip()
        guidance["taste_profile"] = str(data.get("taste_profile") or "").strip()
        guidance["keywords"] = _string_list(data.get("keywords"))

    return guidance


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
        "negative_guidance": "Avoid generic styling, mismatched lighting, and visual clutter.",
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
    from prompt_expander import _LOCAL_QWEN_LOCK, _decode_generation, _generation_kwargs, _input_ids, _load_local_qwen, _strip_data_url, unload_local_qwen_after_use

    tokenizer = processor = model = inputs = outputs = None
    images = []
    try:
        with _LOCAL_QWEN_LOCK:
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
    finally:
        del tokenizer, processor, model, inputs, outputs, images
        unload_local_qwen_after_use()


def _comfy_qwen_generate(
    prompt: str,
    image_b64s: list[str],
    *,
    prompt_id_cb: Callable[[str], None] | None = None,
) -> str:
    from comfy_qwen_vl import enrich_images_comfy

    return enrich_images_comfy(
        image_b64s or [], prompt, prompt_id_cb=prompt_id_cb
    )


def _qwen_generate(
    prompt: str,
    image_b64s: list[str],
    *,
    prompt_id_cb: Callable[[str], None] | None = None,
    cancel_probe: CancellationProbe | None = None,
) -> str:
    """Comfy QwenVL by default; Transformers fallback when Comfy is unavailable."""
    from settings import settings

    backend = str(getattr(settings, "local_llm_backend", "comfy") or "comfy")
    if backend == "transformers":
        return _local_qwen_generate(prompt, image_b64s)
    try:
        return _comfy_qwen_generate(
            prompt, image_b64s, prompt_id_cb=prompt_id_cb
        )
    except Exception as exc:
        message = str(exc).lower()
        if is_cuda_oom(exc) or (cancel_probe and cancel_probe()) or any(
            marker in message
            for marker in ("interrupt", "cancelled", "canceled")
        ):
            raise
        logger.warning("Comfy QwenVL moodboard enrich failed; falling back to Transformers: %s", exc)
        return _local_qwen_generate(prompt, image_b64s)


def generate_moodboard_guidance(
    sources: list[MoodboardSource],
    *,
    mode: GuidanceMode,
    generator: QwenGenerator | None = None,
    prompt_id_cb: Callable[[str], None] | None = None,
    cancel_probe: CancellationProbe | None = None,
) -> dict:
    if mode not in {"official", "custom", "mashup"}:
        raise ValueError("Unknown moodboard guidance mode.")
    generate = generator or (
        lambda prompt, images: _qwen_generate(
            prompt,
            images,
            prompt_id_cb=prompt_id_cb,
            cancel_probe=cancel_probe,
        )
    )
    allow_metadata = mode in {"custom", "mashup"}
    images: list[str] = []
    for source in sources:
        images.extend([image for image in source.image_b64s if image])
    images = images[:10]

    best: dict | None = None
    violations: list[str] = []
    for attempt in range(MAX_GUIDANCE_ATTEMPTS):
        prompt = build_moodboard_guidance_prompt(sources, mode, violations=violations or None)
        response = generate(prompt, images)
        try:
            guidance = parse_style_schema_json(response, allow_catalog_metadata=allow_metadata)
        except ValueError:
            continue
        best = guidance
        violations = find_subject_violations(guidance)
        if not violations:
            cleaned = sanitize_transferable_guidance(guidance)
            cleaned.setdefault("guidance_backend", "qwen")
            return cleaned
        logger.info(
            "Moodboard guidance attempt %d leaked subjects (%s); %s",
            attempt + 1,
            "; ".join(violations[:4]),
            "retrying" if attempt + 1 < MAX_GUIDANCE_ATTEMPTS else "falling back to regex sanitization",
        )

    if best is not None:
        # Retries exhausted: keep the best structured output, scrubbed by the
        # regex sanitizer (which also drops subject-locked style axes).
        cleaned = sanitize_transferable_guidance(best)
        cleaned.setdefault("guidance_backend", "qwen")
        return cleaned
    return _fallback_guidance(sources, mode)
