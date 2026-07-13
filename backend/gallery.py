from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from settings import DB_PATH, OUTPUTS_DIR

THUMB_SIZE = (320, 320)
THUMB_QUALITY = 75
THUMBS_DIRNAME = ".thumbs"


def _thumbs_root() -> Path:
    return OUTPUTS_DIR / THUMBS_DIRNAME


def _thumb_cache_path(filename: str) -> Path:
    # Keep a stable 1:1 mapping under outputs/.thumbs without nesting user folders.
    safe = filename.replace("\\", "/").replace("/", "__")
    return _thumbs_root() / f"{safe}.webp"


def _metadata_from_png(path: Path) -> dict:
    try:
        from PIL import Image

        with Image.open(path) as img:
            raw = img.info.get("krea2_metadata") or img.info.get("parameters") or "{}"
        data = json.loads(raw) if raw else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _iter_gallery_files(owner_username: str | None, is_admin: bool) -> list[tuple[str, Path, str | None]]:
    if not OUTPUTS_DIR.exists():
        return []
    files: list[tuple[str, Path, str | None]] = []
    if is_admin:
        roots = [(None, OUTPUTS_DIR), *[(p.name, p) for p in OUTPUTS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]]
    else:
        roots = [(owner_username or "", OUTPUTS_DIR / (owner_username or ""))]
    for owner, root in roots:
        if not root.exists():
            continue
        for path in root.glob("*.png"):
            rel = path.relative_to(OUTPUTS_DIR).as_posix()
            files.append((rel, path, owner))
    return files


async def _prune_missing_rows(db: aiosqlite.Connection, *, owner_username: str | None, is_admin: bool) -> None:
    db.row_factory = aiosqlite.Row
    if is_admin:
        rows = await (await db.execute("SELECT id, filename FROM gallery")).fetchall()
    else:
        rows = await (
            await db.execute("SELECT id, filename FROM gallery WHERE owner_username = ?", (owner_username or "",))
        ).fetchall()
    missing = [row["id"] for row in rows if not (OUTPUTS_DIR / row["filename"]).exists()]
    if missing:
        await db.executemany("DELETE FROM gallery WHERE id = ?", [(item,) for item in missing])
        await db.commit()
        for row in rows:
            if row["id"] in missing:
                _thumb_cache_path(row["filename"]).unlink(missing_ok=True)


async def _import_filesystem_orphans(
    db: aiosqlite.Connection,
    *,
    owner_username: str | None,
    is_admin: bool,
) -> int:
    """Promote PNGs that exist on disk but not in the DB into gallery rows.

    Cheap: only reads PNG text metadata for new files. Thumbnails are generated
    lazily for the requested page, not for the whole library.
    """
    db.row_factory = aiosqlite.Row
    if is_admin:
        known_rows = await (await db.execute("SELECT filename FROM gallery")).fetchall()
    else:
        known_rows = await (
            await db.execute("SELECT filename FROM gallery WHERE owner_username = ?", (owner_username or "",))
        ).fetchall()
    known = {row["filename"] for row in known_rows}
    imported = 0
    for filename, path, owner in _iter_gallery_files(owner_username, is_admin):
        if filename in known:
            continue
        meta = _metadata_from_png(path)
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
        prompt = str(meta.get("prompt") or meta.get("original_prompt") or "")
        loras = meta.get("loras") or []
        if owner:
            meta = {**meta, "owner_username": owner}
        await db.execute(
            """INSERT INTO gallery
               (filename, prompt, negative_prompt, checkpoint, steps, cfg,
                width, height, seed, loras, mode, metadata_json, owner_username, favorite, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (
                filename,
                prompt,
                str(meta.get("negative_prompt") or ""),
                str(meta.get("checkpoint") or (meta.get("model") or {}).get("checkpoint") or ""),
                int(meta.get("steps") or 0),
                float(meta.get("cfg") or 0.0),
                int(meta.get("width") or (meta.get("model") or {}).get("width") or 0),
                int(meta.get("height") or (meta.get("model") or {}).get("height") or 0),
                int(meta.get("seed") or 0),
                json.dumps(loras),
                str(meta.get("mode") or ""),
                json.dumps(meta),
                owner,
                created,
            ),
        )
        imported += 1
    if imported:
        await db.commit()
    return imported


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
                media_type TEXT NOT NULL DEFAULT 'image',
                poster_filename TEXT DEFAULT NULL,
                duration REAL DEFAULT NULL,
                frame_count INTEGER DEFAULT NULL,
                project_job_id TEXT DEFAULT NULL,
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
        migrations = {
            "media_type": "TEXT NOT NULL DEFAULT 'image'",
            "poster_filename": "TEXT DEFAULT NULL",
            "duration": "REAL DEFAULT NULL",
            "frame_count": "INTEGER DEFAULT NULL",
            "project_job_id": "TEXT DEFAULT NULL",
        }
        for name, declaration in migrations.items():
            if name not in names:
                await db.execute(
                    f"ALTER TABLE gallery ADD COLUMN {name} {declaration}"
                )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_gallery_created ON gallery(created_at DESC)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_gallery_owner_created ON gallery(owner_username, created_at DESC)"
        )
        await db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_gallery_filename ON gallery(filename)"
        )
        await db.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_gallery_project_job
               ON gallery(project_job_id) WHERE project_job_id IS NOT NULL"""
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


async def save_media(
    filename: str,
    *,
    poster_filename: str,
    duration: float,
    frame_count: int,
    project_job_id: str,
    owner_username: str | None,
    prompt: str = "",
    width: int = 0,
    height: int = 0,
    seed: int = 0,
    metadata: dict | None = None,
) -> int:
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = dict(metadata or {})
    if owner_username:
        payload["owner_username"] = owner_username
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            """INSERT OR IGNORE INTO gallery
               (filename, prompt, width, height, seed, mode, metadata_json,
                owner_username, media_type, poster_filename, duration,
                frame_count, project_job_id, favorite, created_at)
               VALUES (?, ?, ?, ?, ?, 'animation', ?, ?, 'video', ?, ?, ?, ?, 0, ?)""",
            (
                filename,
                prompt,
                width,
                height,
                seed,
                json.dumps(payload),
                owner_username,
                poster_filename,
                float(duration),
                int(frame_count),
                project_job_id,
                created_at,
            ),
        )
        row = await (
            await db.execute(
                "SELECT id FROM gallery WHERE project_job_id = ?",
                (project_job_id,),
            )
        ).fetchone()
        await db.commit()
        if row is None:
            raise RuntimeError("gallery media publication failed")
        return int(row[0])


def _make_thumbnail(img_path: Path, *, filename: str | None = None) -> str | None:
    """Return a WEBP thumbnail as base64, using a disk cache keyed by source mtime."""
    try:
        from PIL import Image

        if not img_path.exists():
            return None
        cache_key = filename or img_path.name
        cache_path = _thumb_cache_path(cache_key)
        src_mtime = img_path.stat().st_mtime
        if cache_path.exists() and cache_path.stat().st_mtime >= src_mtime:
            return base64.b64encode(cache_path.read_bytes()).decode()

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img.thumbnail(THUMB_SIZE)
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=THUMB_QUALITY)
            data = buf.getvalue()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".webp.tmp")
        tmp.write_bytes(data)
        tmp.replace(cache_path)
        return base64.b64encode(data).decode()
    except Exception:
        return None


def _hydrate_item(row: aiosqlite.Row | dict) -> dict:
    item = dict(row)
    item["favorite"] = bool(item.get("favorite"))
    try:
        item["metadata"] = json.loads(item.get("metadata_json") or "{}")
    except json.JSONDecodeError:
        item["metadata"] = {}
    item["media_type"] = item.get("media_type") or "image"
    item["url"] = f"/api/outputs/{item['filename']}"
    poster_filename = item.get("poster_filename")
    item["poster_url"] = (
        f"/api/outputs/{poster_filename}" if poster_filename else None
    )
    image_filename = poster_filename or item["filename"]
    img_path = OUTPUTS_DIR / image_filename
    item["thumbnail_b64"] = (
        _make_thumbnail(img_path, filename=image_filename) if img_path.exists() else None
    )
    item["filesystem_only"] = False
    return item


async def get_gallery(
    page: int = 1,
    page_size: int = 50,
    favorites_only: bool = False,
    owner_username: str | None = None,
    is_admin: bool = True,
) -> dict:
    page = max(1, int(page or 1))
    page_size = max(1, min(100, int(page_size or 50)))
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
        await _prune_missing_rows(db, owner_username=owner_username, is_admin=is_admin)
        if not favorites_only:
            await _import_filesystem_orphans(db, owner_username=owner_username, is_admin=is_admin)

        db.row_factory = aiosqlite.Row
        total_row = await (await db.execute(f"SELECT COUNT(*) AS n FROM gallery {where}", params)).fetchone()
        total = int(total_row["n"] if total_row else 0)
        rows = await (
            await db.execute(
                f"SELECT * FROM gallery {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*params, page_size, offset],
            )
        ).fetchall()

    items = [_hydrate_item(row) for row in rows]
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
    _thumb_cache_path(filename).unlink(missing_ok=True)
    return filename


async def get_image_record_by_filename(filename: str) -> dict | None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        row = await (
            await db.execute(
                """SELECT id, filename, owner_username, media_type,
                          poster_filename, project_job_id
                   FROM gallery
                   WHERE filename = ? OR poster_filename = ?""",
                (filename, filename),
            )
        ).fetchone()
    return dict(row) if row else None


async def delete_media_record(
    gallery_id: int,
    *,
    owner_username: str | None = None,
    is_admin: bool = False,
) -> dict | None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        if is_admin:
            row = await (
                await db.execute(
                    "SELECT * FROM gallery WHERE id = ? AND media_type = 'video'",
                    (gallery_id,),
                )
            ).fetchone()
        else:
            row = await (
                await db.execute(
                    """SELECT * FROM gallery
                       WHERE id = ? AND owner_username = ? AND media_type = 'video'""",
                    (gallery_id, owner_username or ""),
                )
            ).fetchone()
        if row is None:
            return None
        await db.execute("DELETE FROM gallery WHERE id = ?", (gallery_id,))
        await db.commit()
        return dict(row)
