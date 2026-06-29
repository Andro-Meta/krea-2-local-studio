from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any

PLANNER_SYSTEM_PROMPT = (
    "You are a Krea 2 prompt planner. Convert the user's rough text-to-image prompt "
    "into structured JSON for strong prompt adherence. Preserve the user's intent, "
    "visible text, named subjects, spatial relationships, and medium. Do not invent "
    "unrelated content. Return only JSON with keys: planned_prompt, negative_prompt, "
    "subject, composition, style, lighting, materials, text_rendering, regions.\n\n"
    "Follow Krea 2's official prompting guidance: write planned_prompt in natural "
    "language (describe the scene as to a person), favor a long, detailed single "
    "description, and wrap any words that must be rendered as visible text in quotes."
)


@dataclass
class PromptPlanResult:
    original_prompt: str
    planned_prompt: str
    negative_prompt: str = ""
    subject: str = ""
    composition: str = ""
    style: str = ""
    lighting: str = ""
    materials: str = ""
    text_rendering: str = ""
    regions: list[dict[str, Any]] | None = None
    backend: str = "heuristic"
    changed: bool = False
    error: str | None = None

    def model_dump(self) -> dict[str, Any]:
        data = asdict(self)
        data["regions"] = list(self.regions or [])
        return data


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", text or "", flags=re.IGNORECASE).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    data = json.loads(cleaned)
    return data if isinstance(data, dict) else {}


def parse_planner_response(text: str) -> dict[str, Any]:
    data = _extract_json_object(text)
    regions = data.get("regions", [])
    if not isinstance(regions, list):
        regions = []
    return {
        "planned_prompt": str(data.get("planned_prompt", "") or "").strip(),
        "negative_prompt": str(data.get("negative_prompt", "") or "").strip(),
        "subject": str(data.get("subject", "") or "").strip(),
        "composition": str(data.get("composition", "") or "").strip(),
        "style": str(data.get("style", "") or "").strip(),
        "lighting": str(data.get("lighting", "") or "").strip(),
        "materials": str(data.get("materials", "") or "").strip(),
        "text_rendering": str(data.get("text_rendering", "") or "").strip(),
        "regions": [region for region in regions if isinstance(region, dict)],
    }


def _quote_visible_text(prompt: str) -> str:
    return prompt.replace("“", '"').replace("”", '"')


def plan_prompt_heuristic(prompt: str, *, max_tokens: int = 700) -> PromptPlanResult:
    prompt = _quote_visible_text(str(prompt or "").strip())
    if not prompt:
        return PromptPlanResult(original_prompt="", planned_prompt="", changed=False, error="Prompt is empty.")
    additions = (
        "clear subject hierarchy, precise composition, consistent lighting, coherent materials, "
        "Krea 2 friendly descriptive phrasing, no contradictory details"
    )
    planned = prompt if len(prompt.split()) > 22 else f"{prompt}, {additions}"
    return PromptPlanResult(
        original_prompt=prompt,
        planned_prompt=planned[: max(256, int(max_tokens) * 6)],
        subject=prompt,
        composition="clear subject hierarchy and spatial relationships",
        style="preserve the user's requested medium and style",
        lighting="consistent lighting across the scene",
        materials="describe important textures and surfaces clearly",
        text_rendering="preserve quoted visible text exactly" if '"' in prompt else "",
        regions=[],
        backend="heuristic",
        changed=planned.strip() != prompt.strip(),
    )


def plan_prompt_local(prompt: str, *, max_tokens: int = 700) -> PromptPlanResult:
    try:
        from prompt_expander import _decode_generation, _generation_kwargs, _input_ids, _load_local_qwen

        tokenizer, _processor, model = _load_local_qwen()
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(getattr(model, "device", "cpu"))
        outputs = model.generate(
            **_generation_kwargs(inputs),
            max_new_tokens=max(128, min(int(max_tokens), 1600)),
            do_sample=False,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        parsed = parse_planner_response(_decode_generation(tokenizer, outputs, _input_ids(inputs)))
        planned = parsed.get("planned_prompt") or prompt
        return PromptPlanResult(
            original_prompt=prompt,
            planned_prompt=planned,
            negative_prompt=parsed.get("negative_prompt", ""),
            subject=parsed.get("subject", ""),
            composition=parsed.get("composition", ""),
            style=parsed.get("style", ""),
            lighting=parsed.get("lighting", ""),
            materials=parsed.get("materials", ""),
            text_rendering=parsed.get("text_rendering", ""),
            regions=parsed.get("regions", []),
            backend="local",
            changed=planned.strip() != prompt.strip(),
        )
    except Exception as exc:
        fallback = plan_prompt_heuristic(prompt, max_tokens=max_tokens)
        fallback.error = f"Local Qwen prompt planner failed; used heuristic fallback. Details: {exc}"
        return fallback


def plan_prompt_gguf_server(
    prompt: str,
    *,
    max_tokens: int = 700,
    gguf_helper_base_url: str = "http://127.0.0.1:1234/v1",
    gguf_helper_model: str = "BennyDaBall/Krea-2-Engineer-V1-GGUF:Q4_K_M",
    gguf_helper_timeout_sec: int = 120,
) -> PromptPlanResult:
    try:
        text = gguf_chat_completion(
            [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            base_url=gguf_helper_base_url,
            model=gguf_helper_model,
            max_tokens=max(128, min(int(max_tokens), 1600)),
            temperature=0.1,
            timeout=gguf_helper_timeout_sec,
        )
        parsed = parse_planner_response(text)
        planned = parsed.get("planned_prompt") or prompt
        return PromptPlanResult(
            original_prompt=prompt,
            planned_prompt=planned,
            negative_prompt=parsed.get("negative_prompt", ""),
            subject=parsed.get("subject", ""),
            composition=parsed.get("composition", ""),
            style=parsed.get("style", ""),
            lighting=parsed.get("lighting", ""),
            materials=parsed.get("materials", ""),
            text_rendering=parsed.get("text_rendering", ""),
            regions=parsed.get("regions", []),
            backend="gguf-server",
            changed=planned.strip() != prompt.strip(),
        )
    except Exception as exc:
        fallback = plan_prompt_heuristic(prompt, max_tokens=max_tokens)
        fallback.error = f"GGUF helper prompt planner failed; used heuristic fallback. Details: {exc}"
        return fallback


def gguf_chat_completion(*args, **kwargs) -> str:
    from prompt_expander import gguf_chat_completion as _chat

    return _chat(*args, **kwargs)


def plan_prompt(
    prompt: str,
    *,
    enabled: bool = True,
    max_tokens: int = 700,
    backend: str = "local",
    gguf_helper_base_url: str = "http://127.0.0.1:1234/v1",
    gguf_helper_model: str = "BennyDaBall/Krea-2-Engineer-V1-GGUF:Q4_K_M",
    gguf_helper_timeout_sec: int = 120,
) -> PromptPlanResult:
    if not enabled:
        return PromptPlanResult(original_prompt=prompt, planned_prompt=prompt, changed=False, backend="off")
    if backend in {"gguf", "gguf-server"}:
        return plan_prompt_gguf_server(
            prompt,
            max_tokens=max_tokens,
            gguf_helper_base_url=gguf_helper_base_url,
            gguf_helper_model=gguf_helper_model,
            gguf_helper_timeout_sec=gguf_helper_timeout_sec,
        )
    return plan_prompt_local(prompt, max_tokens=max_tokens)
