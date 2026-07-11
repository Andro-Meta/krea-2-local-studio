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


def _visible_to(item: dict[str, Any], username: str | None) -> bool:
    """Legacy recipes (no owner) are shared with everyone; owned recipes are
    private to their creator. Local mode (username=None) sees everything."""
    if username is None:
        return True
    owner = item.get("owner")
    return owner is None or owner == username


def list_recipes(*, path: Path = RECIPE_PATH, username: str | None = None) -> list[dict[str, Any]]:
    items = [item for item in _read(path) if _visible_to(item, username)]
    return sorted(items, key=lambda item: str(item.get("name", "")).lower())


def save_recipe(recipe: dict[str, Any], *, path: Path = RECIPE_PATH, username: str | None = None) -> dict[str, Any]:
    cleaned = _clean_recipe(recipe)
    if username is not None:
        cleaned["owner"] = username
    existing = _read(path)
    # A save may only replace a recipe the caller can see (their own or a
    # legacy shared one); someone else's same-named recipe stays untouched.
    recipes = [
        item for item in existing
        if not (item.get("id") == cleaned["id"] and _visible_to(item, username))
    ]
    recipes.append(cleaned)
    _write(recipes, path)
    return cleaned


def delete_recipe(recipe_id: str, *, path: Path = RECIPE_PATH, username: str | None = None, is_admin: bool = False) -> bool:
    recipe_id = _slug(recipe_id)
    recipes = _read(path)

    def deletable(item: dict[str, Any]) -> bool:
        if item.get("id") != recipe_id:
            return False
        if username is None or is_admin:
            return True
        owner = item.get("owner")
        # Users may delete their own recipes; legacy shared ones are admin-only
        # to delete since everyone can see them.
        return owner == username

    kept = [item for item in recipes if not deletable(item)]
    _write(kept, path)
    return len(kept) != len(recipes)
