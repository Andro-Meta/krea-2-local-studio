from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

RECIPE_PATH = Path(__file__).resolve().parent.parent / "data" / "prompt_recipes.json"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or f"recipe-{int(time.time())}"


def _read(path: Path = RECIPE_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _write(recipes: list[dict[str, Any]], path: Path = RECIPE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipes, indent=2, ensure_ascii=False), encoding="utf-8")


def _clean_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    name = str(recipe.get("name", "") or "").strip() or "Untitled recipe"
    recipe_id = str(recipe.get("id", "") or "").strip() or _slug(name)
    return {
        "id": _slug(recipe_id),
        "name": name[:80],
        "description": str(recipe.get("description", "") or "")[:240],
        "prompt": str(recipe.get("prompt", "") or ""),
        "negative_prompt": str(recipe.get("negative_prompt", "") or ""),
        "planner_instruction": str(recipe.get("planner_instruction", "") or ""),
        "loras": list(recipe.get("loras", []) or [])[:16],
        "mood": str(recipe.get("mood", "") or "")[:240],
        "moodboard_strength": float(recipe.get("moodboard_strength", 0.35) or 0.35),
        "moodboard_ids": list(recipe.get("moodboard_ids", []) or [])[:24],
        "moodboard_uuids": list(recipe.get("moodboard_uuids", []) or [])[:24],
        "style_references": list(recipe.get("style_references", []) or [])[:10],
        "regional_prompts": list(recipe.get("regional_prompts", []) or [])[:8],
        "seed_variance_preset": str(recipe.get("seed_variance_preset", "off") or "off"),
        "krea_enhancer_variant": str(recipe.get("krea_enhancer_variant", "off") or "off"),
        "rebalance_preset": str(recipe.get("rebalance_preset", "balanced") or "balanced"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def list_recipes(*, path: Path = RECIPE_PATH) -> list[dict[str, Any]]:
    return sorted(_read(path), key=lambda item: str(item.get("name", "")).lower())


def save_recipe(recipe: dict[str, Any], *, path: Path = RECIPE_PATH) -> dict[str, Any]:
    cleaned = _clean_recipe(recipe)
    recipes = [item for item in _read(path) if item.get("id") != cleaned["id"]]
    recipes.append(cleaned)
    _write(recipes, path)
    return cleaned


def delete_recipe(recipe_id: str, *, path: Path = RECIPE_PATH) -> bool:
    recipe_id = _slug(recipe_id)
    recipes = _read(path)
    kept = [item for item in recipes if item.get("id") != recipe_id]
    _write(kept, path)
    return len(kept) != len(recipes)
