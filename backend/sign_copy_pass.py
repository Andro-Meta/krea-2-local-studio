"""Stage 2 Magic Wand: invent scene-fit quoted copy for readable surfaces.

After Stage 1 expands a prompt, this pass detects signs/papers/screens/etc. that
imply legible words without providing them, asks Comfy QwenVL to invent short
copy, then deterministically accepts or rejects that rewrite.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("krea2.sign_copy")

SIGN_COPY_SYSTEM_PROMPT = (
    "You are a prompt editor for text-to-image. You receive an already-expanded "
    "image prompt. Your only job is readable-surface copy.\n\n"
    "Readable surfaces include: signs, neon signs, billboards, marquees, plaques, "
    "menus, posters, flyers, newspapers/headlines, notes, letters, papers held toward "
    "camera, open books/pages, name tags, badges, jersey names/numbers, license plates, "
    "product labels, phone/monitor/laptop screens, HUDs, scoreboards — when the prompt "
    "implies those words are legible in frame.\n\n"
    "Decision (apply in order):\n"
    "A. If a readable surface is described and already has adjacent quoted text, or "
    "explicit read-as / says / reading / titled / labeled / printed with / inscribed with "
    "wording — preserve that copy exactly.\n"
    "B. If the only text-like mentions are blank, face-down, folded shut, smeared, "
    "intentionally illegible, too distant to read, neon glow/light/reflections without a "
    "sign noun, or background clutter (stacks of papers, books on a shelf) — output "
    "NO_CHANGE.\n"
    "C. Otherwise, if any readable surface is in the scene without exact words "
    "(including failing/broken/dim neon signs, billboards, held papers, menus, screens) "
    "you MUST invent short unique copy that fits THIS scene's mood, place, era, and "
    "language, and rewrite the prompt so each such surface has quoted characters next to "
    "it (neon sign reading \"…\", paper reading \"…\"). Prefer about 1–6 words for signs; "
    "one short line for paper/screens. Distinct copy per surface. Do not reuse stock shop "
    "or headline phrases. Do not add new subjects or new surfaces. Do not re-expand or "
    "restyle the rest of the prompt.\n\n"
    "Output protocol (strict):\n"
    "- First line must be exactly NO_CHANGE or UPDATED\n"
    "- Then a blank line\n"
    "- If UPDATED: then the full revised prompt paragraph only (must include the new "
    "quoted copy)\n"
    "- If NO_CHANGE: you may omit the body or repeat the original; the body is ignored\n"
    "- No markdown fences, no commentary, no labels other than the first line\n"
    "- When rule C applies, first line MUST be UPDATED — do not answer NO_CHANGE for "
    "unsigned neon signs, unsigned held papers, or other in-frame readable surfaces "
    "missing words."
)

# Surfaces that often need legible words when in-frame.
_CUE_PATTERN = re.compile(
    r"\b("
    r"neon\s+signs?|signs?|billboards?|marquees?|plaques?|"
    r"storefront\s+lettering|street\s+signs?|exit\s+signs?|"
    r"menus?|posters?|flyers?|newspapers?|headlines?|"
    r"notes?|letters?|sticky\s+notes?|clipboards?|receipts?|forms?|"
    r"handwritten\s+paper|piece\s+of\s+paper|sheet\s+of\s+paper|"
    r"holding\s+a\s+paper|holds\s+a\s+paper|holds\s+a\s+note|"
    r"open\s+books?|book\s+pages?|pages?\s+of\s+(?:a\s+)?book|"
    r"name\s+tags?|badges?|jersey\s+(?:names?|numbers?)|license\s+plates?|"
    r"product\s+labels?|bottle\s+labels?|can\s+labels?|box\s+text|"
    r"phone\s+screens?|monitor(?:\s+screens?)?|laptop\s+screens?|"
    r"HUDs?|scoreboards?|closed\s+captions?|subtitles?|"
    r"lettering|inscriptions?|engraving"
    r")\b",
    re.IGNORECASE,
)

# Atmosphere-only neon / lighting — not a readable surface by itself.
_GLOW_ONLY_PATTERN = re.compile(
    r"\bneon\s+(glow|light|lights|lighting|reflections?|haze|wash|ambiance|ambience)\b",
    re.IGNORECASE,
)

# Prompt says the surface should stay without readable words.
_UNREADABLE_PATTERN = re.compile(
    r"\b("
    r"blank\s+(?:paper|page|letter|note|sign)|"
    r"face[- ]down|folded\s+shut|too\s+(?:far|distant)\s+to\s+read|"
    r"intentionally\s+illegible|illegible\s+text|smeared\s+text|"
    r"out\s+of\s+focus\s+text|unreadable|"
    r"no\s+(?:readable\s+)?(?:text|lettering|words)"
    r")\b",
    re.IGNORECASE,
)

# Background clutter without readable-in-frame intent.
_CLUTTER_ONLY_PATTERN = re.compile(
    r"\b("
    r"stack(?:s)?\s+of\s+papers?|papers?\s+strewn|discarded\s+(?:paper|cardboard)|"
    r"books?\s+on\s+(?:a\s+)?shelf|bookshelf|pile\s+of\s+books?"
    r")\b",
    re.IGNORECASE,
)

_EXPLICIT_COPY_RE = re.compile(
    r"\b(reads?|reading|says?|saying|titled|labeled|labelled|printed\s+with|"
    r"inscribed\s+with|bearing\s+the\s+(?:words?|text)|that\s+reads?)\b"
    r".{0,40}?"
    r"[\"“”][^\"“”\n]{1,80}[\"“”]",
    re.IGNORECASE | re.DOTALL,
)

_FENCE_RE = re.compile(r"```(?:\w+)?\s*([\s\S]*?)```")


def _normalize_quotes(text: str) -> str:
    return (text or "").replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def quoted_strings(text: str) -> set[str]:
    text = _normalize_quotes(text)
    return {m.group(1).strip() for m in re.finditer(r'"([^"\n]{1,80})"', text) if m.group(1).strip()}


def _strip_fences_and_think(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        fenced = _FENCE_RE.search(value)
        if fenced:
            value = fenced.group(1).strip()
    lower = value.lower()
    while True:
        start = lower.find("<think>")
        if start < 0:
            break
        end = lower.find("</think>", start + 7)
        if end < 0:
            value = value[:start] + value[start + 7 :]
            lower = value.lower()
            continue
        value = value[:start] + value[end + 8 :]
        lower = value.lower()
    return value.strip()


def needs_sign_copy(prompt: str) -> bool:
    """True when the prompt describes a readable surface that may need invented words."""
    text = _normalize_quotes(prompt or "")
    if not text.strip():
        return False
    if _UNREADABLE_PATTERN.search(text):
        # Still allow other cues elsewhere, but if the *only* paper/sign language is
        # marked unreadable/blank, skip. Cheap approach: if unreadable markers exist
        # and every cue span is near one, skip; else continue.
        pass
    if not _CUE_PATTERN.search(text):
        return False
    # Glow-only with no other cue nouns: neon glow phrases alone don't count.
    # If the only "neon" hits are glow-only and there is no other cue, False.
    cues = list(_CUE_PATTERN.finditer(text))
    non_glow_cues = []
    for m in cues:
        span = text[max(0, m.start() - 12) : m.end() + 12]
        if _GLOW_ONLY_PATTERN.search(span) and not re.search(
            r"\bneon\s+signs?\b|\bsigns?\b", m.group(0), re.IGNORECASE
        ):
            continue
        # "neon signs" is a real cue; bare glow phrases aren't in _CUE_PATTERN.
        non_glow_cues.append(m)
    if not non_glow_cues:
        return False
    # If the prompt is only clutter dressing and no held/focused surface, skip.
    if _CLUTTER_ONLY_PATTERN.search(text):
        focused = re.search(
            r"\b(holding|holds|held|toward the camera|facing (?:the )?camera|"
            r"close-?up|readable|legible|in (?:her|his|their) hand)\b",
            text,
            re.IGNORECASE,
        )
        # Clutter alone with no focus verbs and only clutter cues → skip.
        only_clutter_cues = all(
            _CLUTTER_ONLY_PATTERN.search(text[max(0, m.start() - 24) : m.end() + 24])
            for m in non_glow_cues
        )
        if only_clutter_cues and not focused:
            return False
    if _UNREADABLE_PATTERN.search(text):
        # If every cue is near an unreadable marker, skip.
        if all(
            _UNREADABLE_PATTERN.search(text[max(0, m.start() - 48) : m.end() + 48])
            for m in non_glow_cues
        ):
            return False
    return True


def sign_copy_already_present(prompt: str) -> bool:
    """True when readable-surface cues already have explicit/quoted copy."""
    text = _normalize_quotes(prompt or "")
    if not needs_sign_copy(text):
        # No cues needing copy → vacuously "already fine"
        return True
    if _EXPLICIT_COPY_RE.search(text):
        # At least one surface has explicit copy. For pre-check skip we require
        # that quoted strings exist near cues overall.
        quotes = quoted_strings(text)
        if quotes:
            return True
    # Heuristic: every cue match has a quote within a nearby window.
    cues = list(_CUE_PATTERN.finditer(text))
    if not cues:
        return True
    satisfied = 0
    for m in cues:
        window = text[max(0, m.start() - 40) : m.end() + 80]
        if quoted_strings(window) or _EXPLICIT_COPY_RE.search(window):
            satisfied += 1
    return satisfied >= len(cues)


def parse_sign_copy_protocol(text: str) -> tuple[str, str]:
    """Return (NO_CHANGE|UPDATED|UNKNOWN, body)."""
    cleaned = _strip_fences_and_think(text)
    if not cleaned:
        return "UNKNOWN", ""
    lines = cleaned.splitlines()
    first = lines[0].strip().upper().replace("`", "")
    # Allow "UPDATED:" or trailing punctuation.
    first_token = re.split(r"[\s:]+", first, maxsplit=1)[0]
    body_lines = lines[1:]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    body = "\n".join(body_lines).strip()
    if first_token in {"NO_CHANGE", "NOCHANGE"}:
        return "NO_CHANGE", body
    if first_token == "UPDATED":
        return "UPDATED", body
    return "UNKNOWN", cleaned


def accept_sign_copy_output(inp: str, out: str) -> str:
    """Deterministically accept Stage 2 output or keep Stage 1 text."""
    inp = str(inp or "")
    raw = str(out or "")
    status, body = parse_sign_copy_protocol(raw)
    if status == "NO_CHANGE":
        return inp
    candidate = body if status == "UPDATED" else body
    if status == "UNKNOWN":
        candidate = body
    candidate = candidate.strip()
    if not candidate:
        return inp
    if len(candidate) < max(40, int(len(inp) * 0.6)):
        return inp
    if needs_sign_copy(inp):
        new_quotes = quoted_strings(candidate) - quoted_strings(inp)
        if not new_quotes and not (
            status == "UPDATED" and quoted_strings(candidate) and not sign_copy_already_present(inp)
        ):
            # UPDATED but no new quotes vs input → reject paraphrase
            if not new_quotes:
                return inp
    # If input had no cues, never accept a rewrite.
    if not needs_sign_copy(inp):
        return inp
    return candidate


def _format_stage2_user_payload(expanded: str) -> str:
    return (
        f"{SIGN_COPY_SYSTEM_PROMPT}\n\n"
        "---\n"
        "PROMPT TO EDIT (do not describe it; edit it per the rules above):\n"
        f"{expanded.strip()}\n"
        "---\n"
        "Reply with NO_CHANGE or UPDATED as specified. If the prompt contains unsigned "
        "neon signs, billboards, held papers, menus, or other in-frame readable surfaces "
        "without exact words, you must choose UPDATED and insert quoted copy."
    )


def _extract_invented_phrase(raw: str) -> str:
    """Pull a short invented phrase from a micro-invent model reply."""
    text = _strip_fences_and_think(raw)
    quotes = list(quoted_strings(text))
    if quotes:
        quotes.sort(key=len)
        for phrase in quotes:
            phrase = phrase.strip()
            words = phrase.split()
            if 1 <= len(words) <= 4 and len(phrase) <= 40:
                return phrase
    for line in text.splitlines():
        line = line.strip().strip('"').strip()
        if not line or line.upper() in {"NO_CHANGE", "UPDATED", "COPY", "PHRASE"}:
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and len(line) <= 40:
            return line
    return ""


def splice_copy_into_prompt(prompt: str, phrase: str) -> str:
    """Insert reading \"phrase\" after the first readable-surface cue."""
    phrase = (phrase or "").strip().strip('"')
    if not phrase or not prompt:
        return prompt
    # Harden against sentence-length invents from small models.
    words = phrase.split()
    if len(words) > 4:
        phrase = " ".join(words[:4]).rstrip(".,;:!")
    phrase = phrase.rstrip(".,;:!")
    quoted = f'"{phrase}"'
    if quoted in prompt:
        return prompt
    match = _CUE_PATTERN.search(prompt)
    if not match:
        return prompt
    insert = f" reading {quoted}"
    after = prompt[match.end() : match.end() + 24]
    if re.match(r"\s+reading\b", after, re.IGNORECASE):
        return prompt[: match.end()] + f" {quoted}" + prompt[match.end() :]
    return prompt[: match.end()] + insert + prompt[match.end() :]


def _micro_invent_and_splice(prompt: str, expand_fn) -> str:
    """Ask the LLM for only a short phrase, then splice it into the prompt."""
    invent_user = (
        "Invent ONE short sign/paper phrase for the scene below. "
        "Requirements: 1–4 words only, concrete, scene-specific, not a full sentence, "
        "not a poetic caption. Reply with ONLY those words inside quotes.\n\n"
        f"SCENE:\n{prompt.strip()}"
    )
    raw = expand_fn(
        invent_user,
        "Reply with only a short quoted phrase of 1-4 words.",
        max_tokens=64,  # ComfyUI-QwenVL Advanced enforces min 64
        temperature=0.9,
        seed=3,
        keep_model_loaded=True,
        free_vram=False,
    )
    phrase = _extract_invented_phrase(raw or "")
    if not phrase:
        return prompt
    return splice_copy_into_prompt(prompt, phrase)


def run_sign_copy_pass(
    expanded: str,
    *,
    stage1_backend: str = "",
    enabled: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Run Stage 2 when needed. Never raises — returns Stage 1 text on failure."""
    meta: dict[str, Any] = {"ran": False, "changed": False, "skipped_reason": None}
    text = str(expanded or "")
    if not enabled:
        meta["skipped_reason"] = "disabled"
        return text, meta
    if not text.strip():
        meta["skipped_reason"] = "empty"
        return text, meta
    if not needs_sign_copy(text):
        meta["skipped_reason"] = "no_cues"
        return text, meta
    if sign_copy_already_present(text):
        meta["skipped_reason"] = "already_present"
        return text, meta

    try:
        from comfy_qwen_vl import comfy_qwen_vl_available, expand_prompt_comfy as _comfy_expand
    except Exception as exc:
        logger.warning("Sign-copy pass import failed: %s", exc)
        meta["skipped_reason"] = "comfy_unavailable"
        meta["error"] = str(exc)
        return text, meta

    try:
        if not comfy_qwen_vl_available(timeout=3.0):
            meta["skipped_reason"] = "comfy_unavailable"
            return text, meta
    except Exception as exc:
        meta["skipped_reason"] = "comfy_unavailable"
        meta["error"] = str(exc)
        return text, meta

    # If Stage 1 was Comfy, Qwen is likely still loaded — skip free between stages.
    free_vram = str(stage1_backend or "").lower() != "comfy"
    user_payload = _format_stage2_user_payload(text)

    try:
        raw = _comfy_expand(
            user_payload,
            "Follow the editor instructions in the user message exactly.",
            max_tokens=700,
            temperature=0.55,
            seed=1,
            keep_model_loaded=True,
            free_vram=free_vram,
        )
        final = accept_sign_copy_output(text, raw or "")
        # Retry full rewrite once, then micro-invent + splice if still unchanged.
        if final.strip() == text.strip():
            nudge = (
                user_payload
                + "\n\nYour previous answer did not add quoted copy. Rule C applies. "
                "Respond UPDATED and include at least one new quoted string on a "
                "readable surface."
            )
            raw2 = _comfy_expand(
                nudge,
                "Follow the editor instructions in the user message exactly.",
                max_tokens=700,
                temperature=0.7,
                seed=2,
                keep_model_loaded=True,
                free_vram=False,
            )
            final = accept_sign_copy_output(text, raw2 or "")
        if final.strip() == text.strip():
            try:
                spliced = _micro_invent_and_splice(text, _comfy_expand)
                if spliced.strip() != text.strip() and quoted_strings(spliced) - quoted_strings(text):
                    final = spliced
                    meta["fallback"] = "micro_invent_splice"
            except Exception as splice_exc:
                logger.warning("Sign-copy micro-invent failed: %s", splice_exc)
                meta["micro_invent_error"] = str(splice_exc)
    except Exception as exc:
        logger.warning("Sign-copy Comfy pass failed; keeping Stage 1 prompt: %s", exc)
        meta["skipped_reason"] = "error"
        meta["error"] = str(exc)
        return text, meta

    meta["ran"] = True
    meta["changed"] = final.strip() != text.strip()
    if not meta["changed"]:
        meta["skipped_reason"] = meta.get("skipped_reason") or "accepted_no_change"
    return final, meta
