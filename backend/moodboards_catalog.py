from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import sqlite3
import time
import uuid as uuidlib
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse, urlunparse

import aiosqlite
import requests

from settings import BASE_DIR, DB_PATH

KREA_BASE_URL = "https://www.krea.ai"
KREA_MOODBOARD_GALLERY_URL = f"{KREA_BASE_URL}/app?gallery=moodboards"
MOODBOARD_SEED_PATH = BASE_DIR / "data" / "krea_moodboards_seed.json"
CUSTOM_MOODBOARD_DIR = BASE_DIR / "data" / "custom_moodboards"
SYNC_META_KEY = "last_krea_moodboard_sync_at"
DISCOVERY_META_KEY = "latest_krea_moodboard_discovery"
DEFAULT_SYNC_INTERVAL_SECONDS = 24 * 60 * 60
ALLOWED_KREA_IMAGE_HOSTS = {"optim-images.krea.ai"}
ALLOWED_KREA_MOODBOARD_HOSTS = {"www.krea.ai", "krea.ai"}


@dataclass
class MoodboardRecord:
    url: str
    slug: str
    uuid: str
    title: str
    taste_profile: str
    keywords: list[str] = field(default_factory=list)
    primary_image_url: str = ""
    image_urls: list[str] = field(default_factory=list)
    related_urls: list[str] = field(default_factory=list)
    sync_error: str = ""


def _record_from_seed_item(item: dict) -> MoodboardRecord:
    url = canonical_moodboard_url(str(item.get("url", "")))
    slug, uuid = _slug_and_uuid(url)
    return MoodboardRecord(
        url=url,
        slug=str(item.get("slug") or slug),
        uuid=str(item.get("uuid") or uuid),
        title=str(item.get("title") or _title_from_slug(slug)),
        taste_profile=str(item.get("taste_profile") or ""),
        keywords=[str(v) for v in item.get("keywords", []) if str(v).strip()],
        primary_image_url=str(item.get("primary_image_url") or ""),
        image_urls=[str(v) for v in item.get("image_urls", []) if str(v).strip()],
        related_urls=[canonical_moodboard_url(str(v)) for v in item.get("related_urls", []) if str(v).strip()],
    )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_moodboard_url(url: str) -> str:
    absolute = urljoin(KREA_BASE_URL, url)
    parsed = urlparse(absolute)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))


def is_allowed_krea_image_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.netloc in ALLOWED_KREA_IMAGE_HOSTS


def is_allowed_krea_moodboard_url(url: str) -> bool:
    parsed = urlparse(canonical_moodboard_url(url))
    return (
        parsed.scheme == "https"
        and parsed.netloc in ALLOWED_KREA_MOODBOARD_HOSTS
        and (
            parsed.path == "/app"
            or parsed.path.startswith("/moodboard-feed/")
        )
    )


def is_krea_ui_image_url(url: str) -> bool:
    normalized = str(url or "").lower()
    return any(
        marker in normalized
        for marker in (
            "s-krea-ai-icons",
            "/icons-",
            "-icons-",
            "homeicon",
            "personalization",
            "nodeeditor",
            "nanobanana",
            "realtimev",
        )
    )


def _moodboard_image_urls(urls: list[str]) -> list[str]:
    return [url for url in _dedupe(urls) if url and not is_krea_ui_image_url(url)]


def fetch_krea_image_b64(url: str, *, timeout: int = 30) -> str:
    if not is_allowed_krea_image_url(url):
        raise ValueError("Only Krea image URLs can be loaded into moodboards.")
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "krea2-studio/1.0 moodboard image loader"},
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise ValueError("URL did not return an image.")
    return base64.b64encode(response.content).decode()


def _strip_data_url(image_b64: str) -> str:
    value = str(image_b64 or "").strip()
    if "," in value and value.lower().startswith("data:image/"):
        return value.split(",", 1)[1]
    return value


def _image_extension(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def _custom_image_url(board_uuid: str, filename: str) -> str:
    return f"/api/moodboards/custom-images/{board_uuid}/{filename}"


def _custom_image_path(url: str, storage_dir: Path = CUSTOM_MOODBOARD_DIR) -> Path:
    match = re.match(r"^/api/moodboards/custom-images/([^/]+)/([^/]+)$", url)
    if not match:
        raise ValueError("Unknown custom moodboard image path.")
    board_uuid, filename = match.groups()
    root = storage_dir.resolve()
    path = (root / board_uuid / filename).resolve()
    if root not in path.parents:
        raise ValueError("Invalid custom moodboard image path.")
    return path


def fetch_moodboard_image_b64(url: str, *, storage_dir: Path = CUSTOM_MOODBOARD_DIR, timeout: int = 30) -> str:
    if str(url).startswith("/api/moodboards/custom-images/"):
        path = _custom_image_path(url, storage_dir=storage_dir)
        if not path.exists():
            raise ValueError("Custom moodboard image not found.")
        return base64.b64encode(path.read_bytes()).decode()
    return fetch_krea_image_b64(url, timeout=timeout)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(text))).strip()


def _title_from_slug(slug: str) -> str:
    name = re.sub(r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", "", slug)
    return " ".join(part.capitalize() for part in name.split("-") if part)


def _slug_and_uuid(url: str) -> tuple[str, str]:
    slug = urlparse(canonical_moodboard_url(url)).path.rsplit("/", 1)[-1]
    match = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$", slug)
    return slug, match.group(1) if match else ""


def _json_ld_blocks(html: str) -> list[object]:
    blocks: list[object] = []
    for match in re.finditer(
        r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        html,
        flags=re.I | re.S,
    ):
        try:
            data = json.loads(unescape(match.group(1)).strip())
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            blocks.extend(data)
        else:
            blocks.append(data)
    return blocks


def _normalize_title(raw: str, slug: str) -> str:
    title = raw.strip()
    title = re.sub(r"^Generate Images in the\s+", "", title, flags=re.I)
    title = re.sub(r"^Generate images in the\s+", "", title, flags=re.I)
    title = re.sub(r"\s+style$", "", title, flags=re.I)
    title = re.sub(r"\s+Style\s*\|\s*Krea$", "", title, flags=re.I)
    return title.strip() or _title_from_slug(slug)


class KreaMoodboardCrawler:
    def __init__(
        self,
        fetch_html: Callable[[str], str] | None = None,
        *,
        request_timeout: int = 20,
        delay_seconds: float = 0.25,
        use_browser_discovery: bool = False,
    ) -> None:
        self.fetch_html = fetch_html or self._fetch_html
        self.request_timeout = request_timeout
        self.delay_seconds = delay_seconds
        self.use_browser_discovery = use_browser_discovery

    def _fetch_html(self, url: str) -> str:
        response = requests.get(
            url,
            timeout=self.request_timeout,
            headers={"User-Agent": "krea2-studio/1.0 moodboard catalog"},
        )
        response.raise_for_status()
        return response.text

    def discover_gallery_urls(self, html: str, base_url: str = KREA_MOODBOARD_GALLERY_URL) -> list[str]:
        return _dedupe(
            [
                canonical_moodboard_url(urljoin(base_url, href))
                for href in re.findall(r"href=[\"']([^\"']*moodboard-feed/[^\"']+)[\"']", html, flags=re.I)
            ]
        )

    def discover_lazy_gallery_urls(self, url: str = KREA_MOODBOARD_GALLERY_URL, *, scrolls: int = 24) -> list[str]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return []

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=60_000)
            for _ in range(scrolls):
                page.mouse.wheel(0, 2400)
                page.wait_for_timeout(750)
            hrefs = page.eval_on_selector_all(
                "a[href*='moodboard-feed/']",
                "els => els.map(a => a.href)",
            )
            browser.close()
        return _dedupe([canonical_moodboard_url(href) for href in hrefs])

    def parse_detail_page(self, url: str, html: str) -> MoodboardRecord:
        canonical_url = canonical_moodboard_url(url)
        slug, uuid = _slug_and_uuid(canonical_url)
        json_ld = _json_ld_blocks(html)

        raw_title = ""
        primary_image_url = ""
        for block in json_ld:
            if not isinstance(block, dict):
                continue
            if block.get("@type") in {"WebPage", "ImageGallery"}:
                raw_title = raw_title or str(block.get("name") or "")
                primary_image_url = primary_image_url or str(block.get("primaryImageOfPage") or block.get("image") or "")

        if not raw_title:
            h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.I | re.S)
            raw_title = _strip_tags(h1.group(1)) if h1 else ""

        title = _normalize_title(raw_title, slug)
        paragraphs = [_strip_tags(match) for match in re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.I | re.S)]
        taste_profile = next((p for p in paragraphs if "This aesthetic" in p), "")
        if not taste_profile:
            taste_profile = next((p for p in paragraphs if len(p) > 80), "")

        keyword_section = re.search(
            r"<h[1-6][^>]*>\s*Styles and themes in this moodboard\s*</h[1-6]>(?P<section>.*?)(?:<h[1-4]|\bGuide\b|Frequently asked questions|</body>)",
            html,
            flags=re.I | re.S,
        )
        section_html = keyword_section.group("section") if keyword_section else ""
        keywords = [_strip_tags(item) for item in re.findall(r"<li[^>]*>(.*?)</li>", section_html, flags=re.I | re.S)]

        image_urls = re.findall(r"(?:src|image)=[\"'](https://optim-images\.krea\.ai/[^\"']+)[\"']", html, flags=re.I)
        image_urls.extend(re.findall(r'"(https://optim-images\.krea\.ai/[^"]+)"', html))
        image_urls = _moodboard_image_urls(image_urls)
        if primary_image_url and primary_image_url not in image_urls:
            image_urls.insert(0, primary_image_url)

        related_urls = [
            canonical_moodboard_url(urljoin(canonical_url, href))
            for href in re.findall(r"href=[\"']([^\"']*moodboard-feed/[^\"']+)[\"']", html, flags=re.I)
        ]
        related_urls = [u for u in _dedupe(related_urls) if u != canonical_url]

        return MoodboardRecord(
            url=canonical_url,
            slug=slug,
            uuid=uuid,
            title=title,
            taste_profile=taste_profile,
            keywords=_dedupe(keywords),
            primary_image_url=primary_image_url,
            image_urls=image_urls,
            related_urls=related_urls,
        )

    def crawl(self, seed_urls: list[str] | None = None, *, max_pages: int = 200) -> list[MoodboardRecord]:
        return list(self.iter_crawl(seed_urls, max_pages=max_pages))

    def iter_crawl(self, seed_urls: list[str] | None = None, *, max_pages: int = 200):
        queue = [
            canonical_moodboard_url(url) if "moodboard-feed/" in url else urljoin(KREA_BASE_URL, url)
            for url in (seed_urls or [])
        ]
        seen: set[str] = set()

        imported = 0
        while queue and imported < max_pages:
            url = queue.pop(0)
            if not is_allowed_krea_moodboard_url(url):
                raise ValueError("Only public Krea moodboard URLs can be imported.")
            if url in seen:
                continue
            seen.add(url)
            html = self.fetch_html(url)
            if "moodboard-feed/" not in url:
                discovered = self.discover_gallery_urls(html, url)
                if not discovered and self.use_browser_discovery:
                    discovered = self.discover_lazy_gallery_urls(url)
                queue.extend([u for u in discovered if u not in seen and u not in queue])
                continue
            record = self.parse_detail_page(url, html)
            imported += 1
            yield record
            for related_url in record.related_urls:
                if related_url not in seen and related_url not in queue:
                    queue.append(related_url)
            if self.delay_seconds:
                time.sleep(self.delay_seconds)


async def init_moodboard_db(db_path: Path = DB_PATH) -> None:
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS moodboards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE,
                slug TEXT NOT NULL,
                uuid TEXT DEFAULT '',
                title TEXT NOT NULL,
                taste_profile TEXT DEFAULT '',
                keywords TEXT DEFAULT '[]',
                primary_image_url TEXT DEFAULT '',
                image_urls TEXT DEFAULT '[]',
                related_urls TEXT DEFAULT '[]',
                favorite INTEGER DEFAULT 0,
                source TEXT DEFAULT 'official',
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                sync_error TEXT DEFAULT '',
                qwen_guidance_json TEXT DEFAULT '',
                qwen_guidance_at TEXT DEFAULT '',
                qwen_guidance_version INTEGER DEFAULT 0
            )
            """
        )
        columns = [str(row[1]) for row in await (await db.execute("PRAGMA table_info(moodboards)")).fetchall()]
        if "source" not in columns:
            await db.execute("ALTER TABLE moodboards ADD COLUMN source TEXT DEFAULT 'official'")
        if "qwen_guidance_json" not in columns:
            await db.execute("ALTER TABLE moodboards ADD COLUMN qwen_guidance_json TEXT DEFAULT ''")
        if "qwen_guidance_at" not in columns:
            await db.execute("ALTER TABLE moodboards ADD COLUMN qwen_guidance_at TEXT DEFAULT ''")
        if "qwen_guidance_version" not in columns:
            await db.execute("ALTER TABLE moodboards ADD COLUMN qwen_guidance_version INTEGER DEFAULT 0")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_moodboards_title ON moodboards(title)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_moodboards_favorite ON moodboards(favorite)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_moodboards_source ON moodboards(source)")
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        await db.commit()
    if db_path == DB_PATH:
        await import_moodboard_seed(MOODBOARD_SEED_PATH, db_path=db_path)


async def upsert_moodboard(record: MoodboardRecord, db_path: Path = DB_PATH) -> int:
    now = _now_iso()
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """
            INSERT INTO moodboards
                (url, slug, uuid, title, taste_profile, keywords, primary_image_url,
                 image_urls, related_urls, favorite, source, first_seen_at, last_seen_at, updated_at, sync_error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'official', ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                slug = excluded.slug,
                uuid = excluded.uuid,
                title = excluded.title,
                taste_profile = excluded.taste_profile,
                keywords = excluded.keywords,
                primary_image_url = excluded.primary_image_url,
                image_urls = excluded.image_urls,
                related_urls = excluded.related_urls,
                last_seen_at = excluded.last_seen_at,
                updated_at = excluded.updated_at,
                sync_error = excluded.sync_error
            """,
            (
                record.url,
                record.slug,
                record.uuid,
                record.title,
                record.taste_profile,
                json.dumps(record.keywords),
                record.primary_image_url,
                json.dumps(record.image_urls),
                json.dumps(record.related_urls),
                now,
                now,
                now,
                record.sync_error,
            ),
        )
        row = await (await db.execute("SELECT id FROM moodboards WHERE url = ?", (record.url,))).fetchone()
        await db.commit()
        return int(row[0])


async def set_moodboard_favorite(moodboard_id: int, favorite: bool, db_path: Path = DB_PATH) -> None:
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("UPDATE moodboards SET favorite = ? WHERE id = ?", (int(favorite), moodboard_id))
        await db.commit()


async def set_moodboard_qwen_guidance(
    moodboard_id: int,
    guidance: dict,
    *,
    db_path: Path = DB_PATH,
) -> None:
    version = int(guidance.get("guidance_version") or 1)
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """
            UPDATE moodboards
            SET qwen_guidance_json = ?, qwen_guidance_at = ?, qwen_guidance_version = ?, updated_at = ?
            WHERE id = ?
            """,
            (json.dumps(guidance, sort_keys=True), _now_iso(), version, _now_iso(), moodboard_id),
        )
        await db.commit()


async def generate_and_store_moodboard_qwen_guidance(
    moodboard_id: int,
    *,
    db_path: Path = DB_PATH,
    generator=None,
) -> dict:
    from moodboard_enrichment import MoodboardSource, generate_moodboard_guidance

    item = await get_moodboard(moodboard_id, db_path=db_path)
    if not item:
        raise ValueError("Moodboard not found.")

    image_b64s: list[str] = []
    for url in _moodboard_image_urls([item.get("primary_image_url", ""), *item.get("image_urls", [])])[:4]:
        try:
            image_b64s.append(fetch_moodboard_image_b64(url))
        except Exception:
            continue

    guidance = generate_moodboard_guidance(
        [
            MoodboardSource(
                title=str(item.get("title") or ""),
                taste_profile=str(item.get("taste_profile") or ""),
                keywords=list(item.get("keywords") or []),
                image_b64s=image_b64s,
            )
        ],
        mode="official" if item.get("source") == "official" else "custom",
        generator=generator,
    )
    await set_moodboard_qwen_guidance(moodboard_id, guidance, db_path=db_path)
    refreshed = await get_moodboard(moodboard_id, db_path=db_path)
    if not refreshed:
        raise ValueError("Moodboard not found after guidance update.")
    return refreshed


async def generate_missing_moodboard_qwen_guidance(
    *,
    limit: int = 25,
    db_path: Path = DB_PATH,
    generator=None,
) -> dict:
    safe_limit = max(1, min(int(limit or 25), 250))
    async with aiosqlite.connect(str(db_path)) as db:
        rows = await (
            await db.execute(
                """
                SELECT id FROM moodboards
                WHERE COALESCE(qwen_guidance_json, '') = ''
                ORDER BY source, title
                LIMIT ?
                """,
                (safe_limit,),
            )
        ).fetchall()
    items: list[dict] = []
    for row in rows:
        items.append(
            await generate_and_store_moodboard_qwen_guidance(
                int(row[0]),
                db_path=db_path,
                generator=generator,
            )
        )
    return {"processed": len(items), "items": items}


async def create_custom_moodboard(
    *,
    title: str,
    taste_profile: str = "",
    keywords: list[str] | None = None,
    image_b64s: list[str],
    db_path: Path = DB_PATH,
    storage_dir: Path = CUSTOM_MOODBOARD_DIR,
    guidance_generator=None,
) -> dict:
    cleaned_title = str(title or "").strip()
    cleaned_taste_profile = str(taste_profile or "").strip()
    cleaned_keywords = [str(v).strip() for v in keywords or [] if str(v).strip()]
    qwen_guidance: dict = {}
    should_auto_author = not cleaned_title or not cleaned_taste_profile or (
        guidance_generator is not None and not cleaned_keywords
    )
    if image_b64s and should_auto_author:
        from moodboard_enrichment import MoodboardSource, generate_moodboard_guidance

        qwen_guidance = generate_moodboard_guidance(
            [
                MoodboardSource(
                    title=cleaned_title,
                    taste_profile=cleaned_taste_profile,
                    keywords=cleaned_keywords,
                    image_b64s=image_b64s,
                )
            ],
            mode="custom",
            generator=guidance_generator,
        )
        cleaned_title = cleaned_title or str(qwen_guidance.get("title") or "").strip()
        cleaned_taste_profile = cleaned_taste_profile or str(qwen_guidance.get("taste_profile") or "").strip()
        cleaned_keywords = cleaned_keywords or [str(v).strip() for v in qwen_guidance.get("keywords", []) if str(v).strip()]
    if not cleaned_title:
        raise ValueError("Custom moodboard title is required.")
    if not image_b64s:
        raise ValueError("Add at least one image to save a custom moodboard.")
    board_uuid = str(uuidlib.uuid4())
    board_dir = storage_dir / board_uuid
    board_dir.mkdir(parents=True, exist_ok=True)
    image_urls: list[str] = []
    try:
        for index, image_b64 in enumerate(image_b64s[:10], start=1):
            raw = base64.b64decode(_strip_data_url(image_b64), validate=True)
            ext = _image_extension(raw)
            filename = f"ref_{index:02d}{ext}"
            (board_dir / filename).write_bytes(raw)
            image_urls.append(_custom_image_url(board_uuid, filename))
    except Exception:
        shutil.rmtree(board_dir, ignore_errors=True)
        raise ValueError("Custom moodboard images must be valid base64 image data.")

    now = _now_iso()
    url = f"custom://{board_uuid}"
    slug = f"custom-{board_uuid}"
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """
            INSERT INTO moodboards
                (url, slug, uuid, title, taste_profile, keywords, primary_image_url,
                 image_urls, related_urls, favorite, source, first_seen_at, last_seen_at, updated_at, sync_error,
                 qwen_guidance_json, qwen_guidance_at, qwen_guidance_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]', 0, 'custom', ?, ?, ?, '', ?, ?, ?)
            """,
            (
                url,
                slug,
                board_uuid,
                cleaned_title,
                cleaned_taste_profile,
                json.dumps(cleaned_keywords),
                image_urls[0],
                json.dumps(image_urls),
                now,
                now,
                now,
                json.dumps(qwen_guidance, sort_keys=True) if qwen_guidance else "",
                now if qwen_guidance else "",
                int(qwen_guidance.get("guidance_version") or 0),
            ),
        )
        row = await (await db.execute("SELECT * FROM moodboards WHERE url = ?", (url,))).fetchone()
        await db.commit()
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        row = await (await db.execute("SELECT * FROM moodboards WHERE url = ?", (url,))).fetchone()
    return _row_to_item(row)


async def delete_custom_moodboard(
    moodboard_id: int,
    *,
    db_path: Path = DB_PATH,
    storage_dir: Path = CUSTOM_MOODBOARD_DIR,
) -> bool:
    item = await get_moodboard(moodboard_id, db_path=db_path)
    if not item or item.get("source") != "custom":
        return False
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("DELETE FROM moodboards WHERE id = ? AND source = 'custom'", (moodboard_id,))
        await db.commit()
    if item.get("uuid"):
        shutil.rmtree(storage_dir / str(item["uuid"]), ignore_errors=True)
    return True


async def create_mashup_moodboard(
    *,
    moodboard_ids: list[int],
    weights: list[float] | None = None,
    db_path: Path = DB_PATH,
    storage_dir: Path = CUSTOM_MOODBOARD_DIR,
    guidance_generator=None,
) -> dict:
    from moodboard_enrichment import MoodboardSource, generate_moodboard_guidance

    ids: list[int] = []
    for value in moodboard_ids:
        try:
            item_id = int(value)
        except (TypeError, ValueError):
            continue
        if item_id > 0 and item_id not in ids:
            ids.append(item_id)
    if len(ids) < 2:
        raise ValueError("Choose at least two moodboards to create a mashup.")
    items = await _items_by_ids(ids, db_path=db_path)
    if len(items) < 2:
        raise ValueError("Choose at least two existing moodboards to create a mashup.")

    normalized_weights = list(weights or [])
    sources: list[MoodboardSource] = []
    image_b64s: list[str] = []
    for index, item in enumerate(items):
        item_weight = float(normalized_weights[index]) if index < len(normalized_weights) else 1.0
        item_images: list[str] = []
        for url in _moodboard_image_urls([item.get("primary_image_url", ""), *item.get("image_urls", [])])[:2]:
            try:
                image_b64 = fetch_moodboard_image_b64(url, storage_dir=storage_dir)
                item_images.append(image_b64)
                image_b64s.append(image_b64)
            except Exception:
                continue
        sources.append(
            MoodboardSource(
                title=str(item.get("title") or ""),
                taste_profile=str(item.get("taste_profile") or ""),
                keywords=list(item.get("keywords") or []),
                image_b64s=item_images,
                weight=item_weight,
            )
        )

    guidance = generate_moodboard_guidance(sources, mode="mashup", generator=guidance_generator)
    created = await create_custom_moodboard(
        title=str(guidance.get("title") or "").strip(),
        taste_profile=str(guidance.get("taste_profile") or "").strip(),
        keywords=list(guidance.get("keywords") or []),
        image_b64s=image_b64s[:10],
        db_path=db_path,
        storage_dir=storage_dir,
    )
    await set_moodboard_qwen_guidance(created["id"], guidance, db_path=db_path)
    refreshed = await get_moodboard(created["id"], db_path=db_path)
    if not refreshed:
        raise ValueError("Mashup moodboard was not saved.")
    return refreshed


def _json_list(value: str) -> list[str]:
    try:
        data = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return [str(item) for item in data] if isinstance(data, list) else []


def _json_dict(value: str) -> dict:
    try:
        data = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _row_to_item(row: aiosqlite.Row) -> dict:
    item = dict(row)
    item["favorite"] = bool(item["favorite"])
    item["source"] = str(item.get("source") or "official")
    item["keywords"] = _json_list(item.get("keywords", "[]"))
    item["image_urls"] = _json_list(item.get("image_urls", "[]"))
    item["related_urls"] = _json_list(item.get("related_urls", "[]"))
    item["qwen_guidance"] = _json_dict(item.get("qwen_guidance_json", ""))
    item["qwen_guidance_at"] = str(item.get("qwen_guidance_at") or "")
    item["qwen_guidance_version"] = int(item.get("qwen_guidance_version") or 0)
    return item


def _sync_row_to_item(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["favorite"] = bool(item["favorite"])
    item["source"] = str(item.get("source") or "official")
    item["keywords"] = _json_list(item.get("keywords", "[]"))
    item["image_urls"] = _json_list(item.get("image_urls", "[]"))
    item["related_urls"] = _json_list(item.get("related_urls", "[]"))
    item["qwen_guidance"] = _json_dict(item.get("qwen_guidance_json", ""))
    item["qwen_guidance_at"] = str(item.get("qwen_guidance_at") or "")
    item["qwen_guidance_version"] = int(item.get("qwen_guidance_version") or 0)
    return item


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_item(item: dict, query: str) -> int:
    if not query.strip():
        return 1
    guidance = item.get("qwen_guidance") or {}
    haystack = " ".join(
        [
            item.get("title", ""),
            item.get("taste_profile", ""),
            " ".join(item.get("keywords", [])),
            str(guidance.get("prompt_guidance") or ""),
            str(guidance.get("negative_guidance") or ""),
            " ".join(str(v) for v in guidance.get("style_axes", []) or []),
            " ".join(str(v) for v in guidance.get("conditioning_notes", []) or []),
            str(guidance.get("source_summary") or ""),
        ]
    ).lower()
    score = 0
    for token in _tokens(query):
        variants = {token, token.rstrip("s"), f"{token}s"}
        if any(variant and variant in haystack for variant in variants):
            score += 10
        if token in item.get("title", "").lower():
            score += 10
        if any(token in keyword.lower() for keyword in item.get("keywords", [])):
            score += 8
        if token in str(guidance.get("prompt_guidance") or "").lower():
            score += 6
        if token in " ".join(str(v) for v in guidance.get("style_axes", []) or []).lower():
            score += 4
        if token in " ".join(str(v) for v in guidance.get("conditioning_notes", []) or []).lower():
            score += 4
    return score


async def list_moodboards(
    query: str = "",
    *,
    favorites_only: bool = False,
    source: str = "",
    page: int = 1,
    page_size: int = 50,
    shuffle_seed: str = "",
    db_path: Path = DB_PATH,
) -> dict:
    where_parts: list[str] = []
    params: list[object] = []
    if favorites_only:
        where_parts.append("favorite = 1")
    if source in {"official", "custom"}:
        where_parts.append("source = ?")
        params.append(source)
    where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        rows = await (await db.execute(f"SELECT * FROM moodboards {where}", params)).fetchall()
    items = [_row_to_item(row) for row in rows]
    if query.strip():
        scored = [(_score_item(item, query), item) for item in items]
        items = [item for score, item in scored if score > 0]
        items.sort(key=lambda item: (-_score_item(item, query), item["title"]))
    elif shuffle_seed.strip():
        seed = shuffle_seed.strip()
        items.sort(
            key=lambda item: hashlib.sha256(f"{seed}:{item.get('id')}:{item.get('uuid', '')}".encode("utf-8")).hexdigest()
        )
    else:
        items.sort(key=lambda item: item["title"])
    total = len(items)
    start = max(0, page - 1) * page_size
    return {"items": items[start:start + page_size], "total": total}


async def get_moodboard(moodboard_id: int, db_path: Path = DB_PATH) -> dict | None:
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        row = await (await db.execute("SELECT * FROM moodboards WHERE id = ?", (moodboard_id,))).fetchone()
    return _row_to_item(row) if row else None


def moodboard_generation_context(
    moodboard_ids: list[int],
    *,
    moodboard_uuids: list[str] | None = None,
    db_path: Path = DB_PATH,
    max_images: int = 10,
) -> dict:
    ids = []
    for value in moodboard_ids:
        try:
            item_id = int(value)
        except (TypeError, ValueError):
            continue
        if item_id > 0 and item_id not in ids:
            ids.append(item_id)
    uuids = []
    for value in moodboard_uuids or []:
        normalized = str(value or "").strip()
        if normalized and normalized not in uuids:
            uuids.append(normalized)
    if not ids and not uuids:
        return {"items": [], "style_text": "", "negative_text": "", "image_urls": [], "uuids": []}

    clauses: list[str] = []
    params: list[object] = []
    if ids:
        clauses.append(f"id IN ({','.join('?' for _ in ids)})")
        params.extend(ids)
    if uuids:
        clauses.append(f"uuid IN ({','.join('?' for _ in uuids)})")
        params.extend(uuids)
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    try:
        cur = db.execute(f"SELECT * FROM moodboards WHERE {' OR '.join(clauses)}", params)
        try:
            rows = cur.fetchall()
        finally:
            cur.close()
    finally:
        db.close()
    by_id = {int(row["id"]): _sync_row_to_item(row) for row in rows}
    by_uuid = {str(row["uuid"]): _sync_row_to_item(row) for row in rows if str(row["uuid"])}
    items: list[dict] = []
    seen_item_ids: set[int] = set()
    for item_id in ids:
        item = by_id.get(item_id)
        if item and int(item["id"]) not in seen_item_ids:
            items.append(item)
            seen_item_ids.add(int(item["id"]))
    for uuid in uuids:
        item = by_uuid.get(uuid)
        if item and int(item["id"]) not in seen_item_ids:
            items.append(item)
            seen_item_ids.add(int(item["id"]))

    parts: list[str] = []
    negative_parts: list[str] = []
    image_urls: list[str] = []
    seen_images: set[str] = set()
    for item in items:
        guidance = item.get("qwen_guidance") or {}
        style_axes = guidance.get("style_axes") or []
        if isinstance(style_axes, (list, tuple)):
            style_axes_text = ", ".join(str(a).strip() for a in style_axes if str(a).strip())
        else:
            style_axes_text = str(style_axes).strip()
        conditioning_notes = guidance.get("conditioning_notes") or []
        if isinstance(conditioning_notes, (list, tuple)):
            conditioning_notes_text = ", ".join(str(a).strip() for a in conditioning_notes if str(a).strip())
        else:
            conditioning_notes_text = str(conditioning_notes).strip()
        style_bits = [
            item.get("title", ""),
            str(guidance.get("prompt_guidance") or item.get("taste_profile", "")),
            f"Style keywords: {', '.join(item.get('keywords', []))}" if item.get("keywords") else "",
            f"Style axes: {style_axes_text}" if style_axes_text else "",
            f"Conditioning notes: {conditioning_notes_text}" if conditioning_notes_text else "",
        ]
        text = ". ".join(bit.strip().rstrip(".") for bit in style_bits if bit.strip())
        if text:
            parts.append(text)
        # Qwen negative guidance belongs in the negative prompt, not the positive.
        neg = str(guidance.get("negative_guidance") or "").strip()
        if neg and neg not in negative_parts:
            negative_parts.append(neg)
        for url in _moodboard_image_urls([item.get("primary_image_url", ""), *item.get("image_urls", [])]):
            if url and url not in seen_images:
                image_urls.append(url)
                seen_images.add(url)
            if len(image_urls) >= max_images:
                break
        if len(image_urls) >= max_images:
            break
    style_text = "Apply these Krea moodboard styles: " + " | ".join(parts) if parts else ""
    negative_text = ", ".join(negative_parts)
    return {
        "items": items,
        "style_text": style_text,
        "negative_text": negative_text,
        "image_urls": image_urls,
        "uuids": [str(item.get("uuid", "")) for item in items if item.get("uuid")],
    }


async def _get_existing_moodboard_urls(db_path: Path = DB_PATH) -> set[str]:
    async with aiosqlite.connect(str(db_path)) as db:
        rows = await (await db.execute("SELECT url FROM moodboards")).fetchall()
    return {str(row[0]) for row in rows}


async def _items_by_ids(ids: list[int], db_path: Path = DB_PATH) -> list[dict]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        rows = await (await db.execute(f"SELECT * FROM moodboards WHERE id IN ({placeholders})", ids)).fetchall()
    by_id = {int(row["id"]): _row_to_item(row) for row in rows}
    return [by_id[item_id] for item_id in ids if item_id in by_id]


async def _record_moodboard_discovery(new_ids: list[int], db_path: Path = DB_PATH) -> None:
    if not new_ids:
        return
    now = _now_iso()
    payload = {
        "id": now,
        "discovered_at": now,
        "new_count": len(new_ids),
        "new_ids": new_ids,
    }
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            "INSERT INTO app_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (DISCOVERY_META_KEY, json.dumps(payload)),
        )
        await db.commit()


async def latest_moodboard_discovery(db_path: Path = DB_PATH) -> dict:
    empty = {"id": "", "discovered_at": "", "new_count": 0, "new_ids": [], "items": []}
    async with aiosqlite.connect(str(db_path)) as db:
        row = await (await db.execute("SELECT value FROM app_meta WHERE key = ?", (DISCOVERY_META_KEY,))).fetchone()
    if not row:
        return empty
    try:
        payload = json.loads(row[0])
    except json.JSONDecodeError:
        return empty
    ids = [int(v) for v in payload.get("new_ids", []) if str(v).isdigit()]
    items = await _items_by_ids(ids, db_path=db_path)
    return {
        "id": str(payload.get("id", "")),
        "discovered_at": str(payload.get("discovered_at", "")),
        "new_count": int(payload.get("new_count", len(ids))),
        "new_ids": ids,
        "items": items,
    }


def _seed_item_from_catalog_item(item: dict) -> dict:
    seed = {
        "url": item["url"],
        "slug": item["slug"],
        "uuid": item.get("uuid", ""),
        "title": item["title"],
        "taste_profile": item.get("taste_profile", ""),
        "keywords": item.get("keywords", []),
        "primary_image_url": item.get("primary_image_url", ""),
        "image_urls": item.get("image_urls", []),
        "related_urls": item.get("related_urls", []),
    }
    if item.get("qwen_guidance"):
        seed["qwen_guidance"] = item.get("qwen_guidance")
        seed["qwen_guidance_at"] = item.get("qwen_guidance_at", "")
        seed["qwen_guidance_version"] = int(item.get("qwen_guidance_version") or 0)
    return seed


async def export_moodboard_seed(seed_path: Path = MOODBOARD_SEED_PATH, *, db_path: Path = DB_PATH) -> int:
    data = await list_moodboards(page=1, page_size=1_000_000, source="official", db_path=db_path)
    seed = {
        "version": 1,
        "source": "https://www.krea.ai/app?gallery=moodboards",
        "moodboards": [_seed_item_from_catalog_item(item) for item in data["items"]],
    }
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(json.dumps(seed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(seed["moodboards"])


async def import_moodboard_seed(seed_path: Path = MOODBOARD_SEED_PATH, *, db_path: Path = DB_PATH) -> int:
    if not seed_path.exists():
        return 0
    try:
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    items = payload.get("moodboards", payload if isinstance(payload, list) else [])
    if not isinstance(items, list):
        return 0
    imported = 0
    for item in items:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        record = _record_from_seed_item(item)
        if not is_allowed_krea_moodboard_url(record.url):
            continue
        board_id = await upsert_moodboard(record, db_path)
        guidance = item.get("qwen_guidance")
        if isinstance(guidance, dict) and guidance.get("prompt_guidance"):
            await set_moodboard_qwen_guidance(board_id, guidance, db_path=db_path)
        imported += 1
    return imported


async def should_sync_moodboards(
    *,
    db_path: Path = DB_PATH,
    now: float | None = None,
    interval_seconds: int = DEFAULT_SYNC_INTERVAL_SECONDS,
    mark: bool = False,
) -> bool:
    timestamp = float(now if now is not None else time.time())
    async with aiosqlite.connect(str(db_path)) as db:
        row = await (await db.execute("SELECT value FROM app_meta WHERE key = ?", (SYNC_META_KEY,))).fetchone()
        last_sync = float(row[0]) if row else 0.0
        due = not row or timestamp - last_sync >= interval_seconds
        if due or mark:
            await db.execute(
                "INSERT INTO app_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (SYNC_META_KEY, str(timestamp)),
            )
            await db.commit()
    return due


async def import_moodboard_urls(
    urls: list[str],
    *,
    db_path: Path = DB_PATH,
    max_pages: int = 200,
    use_browser_discovery: bool = False,
) -> dict:
    crawler = KreaMoodboardCrawler(use_browser_discovery=use_browser_discovery)
    import asyncio

    loop = asyncio.get_event_loop()
    records = await loop.run_in_executor(None, lambda: crawler.crawl(urls, max_pages=max_pages))
    existing_urls = await _get_existing_moodboard_urls(db_path)
    ids: list[int] = []
    new_ids: list[int] = []
    seen_new_urls: set[str] = set()
    for record in records:
        board_id = await upsert_moodboard(record, db_path)
        ids.append(board_id)
        if record.url not in existing_urls and record.url not in seen_new_urls:
            new_ids.append(board_id)
            seen_new_urls.add(record.url)
    await _record_moodboard_discovery(new_ids, db_path=db_path)
    return {"imported": len(ids), "ids": ids, "new_count": len(new_ids), "new_ids": new_ids}
