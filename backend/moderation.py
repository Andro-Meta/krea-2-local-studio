from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import aiosqlite
from PIL import Image

from settings import DB_PATH


EXPLICIT_TERMS = {
    "nude",
    "naked",
    "porn",
    "pornographic",
    "explicit",
    "sex",
    "sexual",
    "erotic",
    "genitals",
    "penis",
    "vagina",
    "breasts",
    "nipples",
    "areola",
    "cum",
    "ejaculate",
    "masturbat",
    "intercourse",
    "blowjob",
    "handjob",
    "fetish",
    "bdsm",
    "lingerie",
}
MINOR_CONTEXT_TERMS = {"child", "kid", "teen", "minor", "young girl", "young boy", "schoolgirl", "schoolboy"}
SAFE_ART_CONTEXT = {"statue", "classical sculpture", "museum", "medical diagram"}


@dataclass(frozen=True)
class ModerationDecision:
    action: str
    event_type: str
    reason: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.action == "allow"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _words(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    return {part for part in normalized.split() if part}


def moderate_prompt(prompt: str, negative_prompt: str = "", *, role: str = "user") -> ModerationDecision:
    """Child-only text moderation for generation prompts.

    Admin/user requests pass through. Child requests are blocked on explicit
    sexual intent and minors+sexual combinations. This deterministic layer is
    intentionally conservative and can be augmented by a local classifier later.
    """
    if str(role or "").lower() != "child":
        return ModerationDecision(action="allow", event_type="prompt")

    text = f"{prompt or ''}\n{negative_prompt or ''}".lower()
    words = _words(text)
    explicit_hits = sorted(term for term in EXPLICIT_TERMS if term in text or term in words)
    minor_hits = sorted(term for term in MINOR_CONTEXT_TERMS if term in text)
    safe_art = any(term in text for term in SAFE_ART_CONTEXT)

    if explicit_hits and not safe_art:
        reason = "Explicit sexual term detected: " + ", ".join(explicit_hits[:5])
        return ModerationDecision(
            action="block",
            event_type="prompt",
            reason=reason,
            scores={"policy_score": 1.0, "explicit_terms": float(len(explicit_hits))},
            labels=explicit_hits,
        )

    if explicit_hits and minor_hits:
        reason = "Minor context combined with sexual content"
        return ModerationDecision(
            action="block",
            event_type="prompt",
            reason=reason,
            scores={"policy_score": 1.0, "explicit_terms": float(len(explicit_hits)), "minor_terms": float(len(minor_hits))},
            labels=[*minor_hits, *explicit_hits],
        )

    return ModerationDecision(action="allow", event_type="prompt", scores={"policy_score": 0.0})


NSFW_IMAGE_MODEL = "Falconsai/nsfw_image_detection"


class TransformersNSFWImageProvider:
    """Lazy local image-classification provider for child image moderation.

    Uses the existing Transformers stack. The model is loaded
    only when a child image check is requested, and the pipeline is CPU-bound by
    default so it does not compete for VRAM with Krea generation.
    """

    def __init__(self, model_name: str = NSFW_IMAGE_MODEL) -> None:
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - dependency optional in light CI
            raise RuntimeError("Transformers image classifier is not available") from exc
        self.model_name = model_name
        self.classifier = pipeline("image-classification", model=model_name, device=-1)

    def classify(self, image: Image.Image) -> list[dict[str, Any]]:
        result = self.classifier(image.convert("RGB"))
        return list(result if isinstance(result, list) else [result])


def image_classifier_available() -> bool:
    try:
        from transformers import pipeline  # noqa: F401
    except Exception:
        return False
    return True


def moderate_image(
    image: Image.Image,
    *,
    role: str = "user",
    provider: Any | None = None,
    nsfw_threshold: float = 0.72,
) -> ModerationDecision:
    if str(role or "").lower() != "child":
        return ModerationDecision(action="allow", event_type="image")
    if provider is None:
        try:
            provider = TransformersNSFWImageProvider()
        except RuntimeError:
            return ModerationDecision(
                action="block",
                event_type="image",
                reason="image moderation provider unavailable; child output requires local image classifier review",
                scores={"provider_available": 0.0},
            )

    predictions = provider.classify(image)
    nsfw_score = 0.0
    safe_score = 0.0
    labels: list[str] = []
    for pred in predictions:
        label = str(pred.get("label", "")).lower()
        score = float(pred.get("score", 0.0) or 0.0)
        if "nsfw" in label or "unsafe" in label or "porn" in label:
            nsfw_score = max(nsfw_score, score)
            if score >= nsfw_threshold:
                labels.append(label or "nsfw")
        elif "normal" in label or "safe" in label or "sfw" in label:
            safe_score = max(safe_score, score)

    if labels:
        return ModerationDecision(
            action="block",
            event_type="image",
            reason="Unsafe image content detected: " + ", ".join(sorted(set(labels))),
            scores={"nsfw_score": nsfw_score, "safe_score": safe_score},
            labels=sorted(set(labels)),
        )

    return ModerationDecision(
        action="allow",
        event_type="image",
        scores={"nsfw_score": nsfw_score, "safe_score": safe_score},
    )


def moderate_images(images: Iterable[Image.Image], *, role: str = "user", provider: Any | None = None) -> ModerationDecision:
    for image in images:
        decision = moderate_image(image, role=role, provider=provider)
        if not decision.allowed:
            return decision
    return ModerationDecision(action="allow", event_type="image")


async def init_moderation_db(db_path: Path = DB_PATH) -> None:
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS moderation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                event_type TEXT NOT NULL,
                action TEXT NOT NULL,
                prompt TEXT DEFAULT '',
                negative_prompt TEXT DEFAULT '',
                mode TEXT DEFAULT '',
                scores_json TEXT DEFAULT '{}',
                reason TEXT DEFAULT '',
                job_id TEXT DEFAULT '',
                gallery_id INTEGER DEFAULT NULL,
                quarantined_filename TEXT DEFAULT NULL
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_moderation_user_created ON moderation_events(username, created_at DESC)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_moderation_action_created ON moderation_events(action, created_at DESC)"
        )
        await db.commit()


async def save_moderation_event(
    *,
    db_path: Path = DB_PATH,
    username: str,
    role: str,
    event_type: str,
    action: str,
    prompt: str = "",
    negative_prompt: str = "",
    mode: str = "",
    scores: dict[str, Any] | None = None,
    reason: str = "",
    job_id: str = "",
    gallery_id: int | None = None,
    quarantined_filename: str | None = None,
) -> int:
    await init_moderation_db(db_path)
    async with aiosqlite.connect(str(db_path)) as db:
        cur = await db.execute(
            """
            INSERT INTO moderation_events (
                created_at, username, role, event_type, action, prompt,
                negative_prompt, mode, scores_json, reason, job_id,
                gallery_id, quarantined_filename
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                username,
                role,
                event_type,
                action,
                prompt,
                negative_prompt,
                mode,
                json.dumps(scores or {}, sort_keys=True),
                reason,
                job_id,
                gallery_id,
                quarantined_filename,
            ),
        )
        await db.commit()
        return int(cur.lastrowid)


async def list_moderation_events(
    *,
    db_path: Path = DB_PATH,
    username: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    safe_limit = max(1, min(int(limit or 100), 500))
    params: list[Any] = []
    where = ""
    if username:
        where = "WHERE username = ?"
        params.append(username)
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        total = (await (await db.execute(f"SELECT COUNT(*) FROM moderation_events {where}", params)).fetchone())[0]
        rows = await (
            await db.execute(
                f"SELECT * FROM moderation_events {where} ORDER BY created_at DESC, id DESC LIMIT ?",
                [*params, safe_limit],
            )
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        try:
            item["scores"] = json.loads(item.pop("scores_json") or "{}")
        except json.JSONDecodeError:
            item["scores"] = {}
        items.append(item)
    return {"items": items, "total": int(total)}
