from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from settings import DB_PATH, OUTPUTS_DIR


async def init_db() -> None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS gallery (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prompt TEXT DEFAULT '',
                negative_prompt TEXT DEFAULT '',
                checkpoint TEXT DEFAULT '',
                steps INTEGER DEFAULT 8,
                cfg REAL DEFAULT 0.0,
                width INTEGER DEFAULT 1024,
                height INTEGER DEFAULT 1024,
                seed INTEGER DEFAULT 0,
                loras TEXT DEFAULT '[]',
                mode TEXT DEFAULT 'txt2img',
                metadata_json TEXT DEFAULT '{}',
                favorite INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        columns = await (await db.execute("PRAGMA table_info(gallery)")).fetchall()
        names = {row[1] for row in columns}
        if "metadata_json" not in names:
            await db.execute("ALTER TABLE gallery ADD COLUMN metadata_json TEXT DEFAULT '{}'")
        if "owner_username" not in names:
            await db.execute("ALTER TABLE gallery ADD COLUMN owner_username TEXT DEFAULT NULL")
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_gallery_created ON gallery(created_at DESC)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_gallery_owner_created ON gallery(owner_username, created_at DESC)"
        )
        await db.commit()


async def save_image(
    filename: str,
    prompt: str = "",
    negative_prompt: str = "",
    checkpoint: str = "turbo",
    steps: int = 8,
    cfg: float = 0.0,
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    loras: list | None = None,
    mode: str = "txt2img",
    metadata: dict | None = None,
    owner_username: str | None = None,
) -> int:
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata_payload = dict(metadata or {})
    if owner_username:
        metadata_payload["owner_username"] = owner_username
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute(
            """INSERT INTO gallery
               (filename, prompt, negative_prompt, checkpoint, steps, cfg,
                width, height, seed, loras, mode, metadata_json, owner_username, favorite, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (filename, prompt, negative_prompt, checkpoint, steps, cfg,
             width, height, seed, json.dumps(loras or []), mode, json.dumps(metadata_payload), owner_username, created_at),
        )
        await db.commit()
        return cursor.lastrowid


def _make_thumbnail(img_path: Path) -> str | None:
    try:
        from PIL import Image
        img = Image.open(img_path)
        img.thumbnail((320, 320))
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=75)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


async def get_gallery(
    page: int = 1,
    page_size: int = 50,
    favorites_only: bool = False,
    owner_username: str | None = None,
    is_admin: bool = True,
) -> dict:
    offset = (page - 1) * page_size
    clauses: list[str] = []
    params: list[object] = []
    if favorites_only:
        clauses.append("favorite = 1")
    if not is_admin:
        clauses.append("owner_username = ?")
        params.append(owner_username or "")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        total_row = await (await db.execute(f"SELECT COUNT(*) FROM gallery {where}", params)).fetchone()
        total = total_row[0]
        rows = await (
            await db.execute(
                f"SELECT * FROM gallery {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*params, page_size, offset),
            )
        ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["favorite"] = bool(item["favorite"])
            try:
                item["metadata"] = json.loads(item.get("metadata_json") or "{}")
            except json.JSONDecodeError:
                item["metadata"] = {}
            img_path = OUTPUTS_DIR / item["filename"]
            item["thumbnail_b64"] = _make_thumbnail(img_path) if img_path.exists() else None
            items.append(item)
    return {"items": items, "total": total}


async def set_favorite(
    gallery_id: int,
    favorite: bool,
    *,
    owner_username: str | None = None,
    is_admin: bool = False,
) -> bool:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        if is_admin:
            cur = await db.execute("UPDATE gallery SET favorite = ? WHERE id = ?", (int(favorite), gallery_id))
        else:
            cur = await db.execute(
                "UPDATE gallery SET favorite = ? WHERE id = ? AND owner_username = ?",
                (int(favorite), gallery_id, owner_username or ""),
            )
        await db.commit()
        return cur.rowcount > 0


async def delete_image(
    gallery_id: int,
    *,
    owner_username: str | None = None,
    is_admin: bool = False,
) -> Optional[str]:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        if is_admin:
            row = await (await db.execute("SELECT filename FROM gallery WHERE id = ?", (gallery_id,))).fetchone()
        else:
            row = await (
                await db.execute(
                    "SELECT filename FROM gallery WHERE id = ? AND owner_username = ?",
                    (gallery_id, owner_username or ""),
                )
            ).fetchone()
        if not row:
            return None
        filename = row["filename"]
        await db.execute("DELETE FROM gallery WHERE id = ?", (gallery_id,))
        await db.commit()
    img_path = OUTPUTS_DIR / filename
    img_path.unlink(missing_ok=True)
    return filename


async def get_image_record_by_filename(filename: str) -> dict | None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        row = await (
            await db.execute("SELECT id, filename, owner_username FROM gallery WHERE filename = ?", (filename,))
        ).fetchone()
    return dict(row) if row else None
