from __future__ import annotations

import asyncio
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from moodboards_catalog import (  # noqa: E402
    KREA_MOODBOARD_GALLERY_URL,
    KreaMoodboardCrawler,
    MoodboardRecord,
    apply_title_qualifiers,
    create_custom_moodboard,
    delete_custom_moodboard,
    fetch_moodboard_image_b64,
    create_mashup_moodboard,
    init_moodboard_db,
    is_allowed_krea_image_url,
    is_allowed_krea_moodboard_url,
    export_moodboard_seed,
    generate_and_store_moodboard_qwen_guidance,
    get_moodboard,
    import_moodboard_seed,
    import_moodboard_urls,
    latest_moodboard_discovery,
    list_moodboards,
    moodboard_generation_context,
    set_moodboard_qwen_guidance,
    set_moodboard_favorite,
    should_sync_moodboards,
    upsert_moodboard,
)
import moodboards_catalog  # noqa: E402


FIXTURE_HTML = """
<!doctype html>
<html>
  <head>
    <script type="application/ld+json">
    [{"@context":"https://schema.org","@type":"WebPage","name":"Generate Images in the Gritty Cinematic Realism Style | Krea","url":"https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369","description":"Generate AI images in the Gritty Cinematic Realism style. This aesthetic explores the intersection of raw human emotion and tactile streets.","image":"https://optim-images.krea.ai/primary.webp"}]
    </script>
  </head>
  <body>
    <h1>Generate images in the Gritty Cinematic Realism style</h1>
    <p>This aesthetic explores the intersection of raw human emotion and the somber textures of everyday urban environments. It relies on shallow depth of field and moody naturalistic lighting.</p>
    <h3>Styles and themes in this moodboard</h3>
    <ul>
      <li>cinematic realism</li>
      <li>shallow depth of field</li>
      <li>moody natural lighting</li>
    </ul>
    <img alt="Gritty Cinematic Realism style reference image — cinematic realism" src="https://optim-images.krea.ai/ref-1.webp">
    <img alt="Gritty Cinematic Realism style reference image — shallow depth of field" src="https://optim-images.krea.ai/ref-2.webp">
    <img alt="Home icon" src="https://optim-images.krea.ai/https---s-krea-ai-icons-HomeIcon-png-128.webp">
    <a href="/moodboard-feed/cinematic-blue-solitude-a057f657-b26a-5768-a134-3e21474484fe">Cinematic Blue Solitude</a>
  </body>
</html>
"""


class MoodboardCatalogTests(unittest.TestCase):
    TINY_PNG_B64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )

    def test_reconcile_custom_storage_repairs_crash_residue_safely(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"
            storage_dir = Path(td) / "boards"
            stale_prep = storage_dir / ".task-prep" / "old-job"
            orphan_uuid = "22222222-2222-4222-8222-222222222222"
            unrelated = storage_dir / "user-notes"
            unrelated_file = storage_dir / "README.txt"

            async def run() -> tuple[int, int]:
                await init_moodboard_db(db_path)
                valid = await create_custom_moodboard(
                    title="Valid",
                    taste_profile="Valid style",
                    keywords=["valid"],
                    image_b64s=[self.TINY_PNG_B64],
                    db_path=db_path,
                    storage_dir=storage_dir,
                )
                missing = await create_custom_moodboard(
                    title="Missing",
                    taste_profile="Missing style",
                    keywords=["missing"],
                    image_b64s=[self.TINY_PNG_B64],
                    db_path=db_path,
                    storage_dir=storage_dir,
                )
                shutil.rmtree(storage_dir / missing["uuid"])
                stale_prep.mkdir(parents=True)
                (stale_prep / "partial.png").write_bytes(b"partial")
                (storage_dir / orphan_uuid).mkdir()
                (storage_dir / orphan_uuid / "ref.png").write_bytes(b"orphan")
                unrelated.mkdir()
                (unrelated / "keep.txt").write_text("keep", encoding="utf-8")
                unrelated_file.write_text("keep", encoding="utf-8")

                await moodboards_catalog.reconcile_custom_moodboard_storage(
                    db_path=db_path,
                    storage_dir=storage_dir,
                )
                valid_after = await get_moodboard(valid["id"], db_path=db_path)
                missing_after = await get_moodboard(missing["id"], db_path=db_path)
                return int(valid_after is not None), int(missing_after is not None)

            valid_exists, missing_exists = asyncio.run(run())
            self.assertEqual(valid_exists, 1)
            self.assertEqual(missing_exists, 0)
            self.assertFalse((storage_dir / ".task-prep").exists())
            self.assertFalse((storage_dir / orphan_uuid).exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue((unrelated / "keep.txt").exists())
            self.assertTrue(unrelated_file.exists())

    def test_guidance_generation_keeps_event_loop_responsive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> int:
                await init_moodboard_db(db_path)
                board_id = await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/heartbeat-11111111-1111-5111-9111-111111111111",
                        slug="heartbeat-11111111-1111-5111-9111-111111111111",
                        uuid="11111111-1111-5111-9111-111111111111",
                        title="Heartbeat",
                        taste_profile="Low-key film grain.",
                        keywords=["film grain"],
                        primary_image_url="",
                        image_urls=[],
                        related_urls=[],
                    ),
                    db_path,
                )
                ticks = 0
                running = True

                async def heartbeat() -> None:
                    nonlocal ticks
                    while running:
                        ticks += 1
                        await asyncio.sleep(0.01)

                def slow_generator(_prompt: str, _images: list[str]) -> str:
                    time.sleep(0.15)
                    return """{
                      "palette": "Muted charcoal",
                      "lighting": "Low-key practical light",
                      "medium_texture": "Heavy film grain",
                      "composition": "Center weighted",
                      "atmosphere": "Hushed",
                      "era_or_movement": "1970s documentary",
                      "style_axes": ["film grain"],
                      "negative_style_terms": ["flat lighting"],
                      "source_summary": "Documentary texture."
                    }"""

                pulse = asyncio.create_task(heartbeat())
                try:
                    await generate_and_store_moodboard_qwen_guidance(
                        board_id,
                        db_path=db_path,
                        generator=slow_generator,
                    )
                finally:
                    running = False
                    await pulse
                return ticks

            self.assertGreater(asyncio.run(run()), 3)

    def test_cancel_during_guidance_generation_skips_db_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> dict:
                await init_moodboard_db(db_path)
                board_id = await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/cancel-guidance-11111111-1111-5111-9111-111111111111",
                        slug="cancel-guidance-11111111-1111-5111-9111-111111111111",
                        uuid="11111111-1111-5111-9111-111111111111",
                        title="Cancel Guidance",
                        taste_profile="Film grain.",
                        keywords=["grain"],
                        primary_image_url="",
                        image_urls=[],
                        related_urls=[],
                    ),
                    db_path,
                )
                cancelled = False

                def generator(_prompt: str, _images: list[str]) -> str:
                    nonlocal cancelled
                    cancelled = True
                    return """{
                      "palette": "Charcoal", "lighting": "Low key",
                      "medium_texture": "Film grain", "composition": "Centered",
                      "atmosphere": "Quiet", "era_or_movement": "Documentary",
                      "style_axes": ["grain"], "negative_style_terms": [],
                      "source_summary": "Texture."
                    }"""

                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    await generate_and_store_moodboard_qwen_guidance(
                        board_id,
                        db_path=db_path,
                        generator=generator,
                        cancel_probe=lambda: cancelled,
                    )
                return await get_moodboard(board_id, db_path=db_path)

            self.assertEqual(asyncio.run(run())["qwen_guidance"], {})

    def test_cancel_during_custom_generation_leaves_no_files_or_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"
            storage_dir = Path(td) / "boards"

            async def run() -> int:
                await init_moodboard_db(db_path)
                cancelled = False

                def generator(_prompt: str, _images: list[str]) -> str:
                    nonlocal cancelled
                    cancelled = True
                    return """{
                      "title": "Cancelled", "taste_profile": "Cancelled style",
                      "keywords": ["cancelled"], "prompt_guidance": "Muted grain",
                      "negative_guidance": "", "style_axes": ["grain"],
                      "conditioning_notes": [], "source_summary": "Cancelled."
                    }"""

                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    await create_custom_moodboard(
                        title="",
                        taste_profile="",
                        keywords=[],
                        image_b64s=[self.TINY_PNG_B64],
                        db_path=db_path,
                        storage_dir=storage_dir,
                        guidance_generator=generator,
                        cancel_probe=lambda: cancelled,
                    )
                return (await list_moodboards(source="custom", db_path=db_path))["total"]

            self.assertEqual(asyncio.run(run()), 0)
            self.assertFalse(storage_dir.exists() and any(storage_dir.rglob("*")))

    def test_cancel_during_custom_file_persistence_rolls_back_files_and_db(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"
            storage_dir = Path(td) / "boards"
            cancelled = False
            write_started = threading.Event()
            release_write = threading.Event()
            original_write = Path.write_bytes

            def blocked_write(path: Path, data: bytes) -> int:
                write_started.set()
                release_write.wait(timeout=2)
                return original_write(path, data)

            def cancel_writer() -> None:
                nonlocal cancelled
                write_started.wait(timeout=2)
                cancelled = True
                release_write.set()

            async def run() -> int:
                await init_moodboard_db(db_path)
                watcher = threading.Thread(target=cancel_writer)
                watcher.start()
                try:
                    with patch.object(Path, "write_bytes", blocked_write):
                        with self.assertRaisesRegex(RuntimeError, "cancelled"):
                            await create_custom_moodboard(
                                title="Complete",
                                taste_profile="Complete style",
                                keywords=["style"],
                                image_b64s=[self.TINY_PNG_B64],
                                db_path=db_path,
                                storage_dir=storage_dir,
                                cancel_probe=lambda: cancelled,
                            )
                finally:
                    release_write.set()
                    watcher.join(timeout=2)
                return (await list_moodboards(source="custom", db_path=db_path))["total"]

            self.assertEqual(asyncio.run(run()), 0)
            self.assertFalse(storage_dir.exists() and any(storage_dir.rglob("*")))

    def test_cancel_during_mashup_generation_leaves_no_custom_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"
            storage_dir = Path(td) / "boards"

            async def run() -> int:
                await init_moodboard_db(db_path)
                ids = []
                for index in range(2):
                    ids.append(
                        await upsert_moodboard(
                            MoodboardRecord(
                                url=f"https://www.krea.ai/moodboard-feed/mashup-cancel-{index}-11111111-1111-5111-9111-11111111111{index}",
                                slug=f"mashup-cancel-{index}-11111111-1111-5111-9111-11111111111{index}",
                                uuid=f"11111111-1111-5111-9111-11111111111{index}",
                                title=f"Source {index}",
                                taste_profile="Source style.",
                                keywords=["source"],
                                primary_image_url="",
                                image_urls=[],
                                related_urls=[],
                            ),
                            db_path,
                        )
                    )
                cancelled = False

                def generator(_prompt: str, _images: list[str]) -> str:
                    nonlocal cancelled
                    cancelled = True
                    return """{
                      "title": "Cancelled Mashup",
                      "taste_profile": "Cancelled blend",
                      "keywords": ["blend"], "prompt_guidance": "Blend grain",
                      "negative_guidance": "", "style_axes": ["grain"],
                      "conditioning_notes": [], "source_summary": "Blend."
                    }"""

                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    await create_mashup_moodboard(
                        moodboard_ids=ids,
                        db_path=db_path,
                        storage_dir=storage_dir,
                        guidance_generator=generator,
                        cancel_probe=lambda: cancelled,
                    )
                return (await list_moodboards(source="custom", db_path=db_path))["total"]

            self.assertEqual(asyncio.run(run()), 0)
            self.assertFalse(storage_dir.exists() and any(storage_dir.rglob("*")))

    def test_custom_image_persistence_keeps_event_loop_responsive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"
            storage_dir = Path(td) / "boards"
            original_write = Path.write_bytes

            def slow_write(path: Path, data: bytes) -> int:
                time.sleep(0.12)
                return original_write(path, data)

            async def run() -> int:
                await init_moodboard_db(db_path)
                ticks = 0
                running = True

                async def heartbeat() -> None:
                    nonlocal ticks
                    while running:
                        ticks += 1
                        await asyncio.sleep(0.01)

                pulse = asyncio.create_task(heartbeat())
                try:
                    with patch.object(Path, "write_bytes", slow_write):
                        await create_custom_moodboard(
                            title="Complete",
                            taste_profile="Complete style",
                            keywords=["style"],
                            image_b64s=[self.TINY_PNG_B64],
                            db_path=db_path,
                            storage_dir=storage_dir,
                        )
                finally:
                    running = False
                    await pulse
                return ticks

            self.assertGreater(asyncio.run(run()), 5)

    def test_parser_extracts_krea_moodboard_details(self) -> None:
        crawler = KreaMoodboardCrawler(fetch_html=lambda _: FIXTURE_HTML)

        parsed = crawler.parse_detail_page(
            "https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
            FIXTURE_HTML,
        )

        self.assertEqual(parsed.slug, "gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369")
        self.assertEqual(parsed.uuid, "4e938f5c-ff17-539b-bdb2-ad7884cdb369")
        self.assertEqual(parsed.title, "Gritty Cinematic Realism")
        self.assertIn("raw human emotion", parsed.taste_profile)
        self.assertEqual(parsed.keywords, ["cinematic realism", "shallow depth of field", "moody natural lighting"])
        self.assertEqual(parsed.primary_image_url, "https://optim-images.krea.ai/primary.webp")
        self.assertIn("https://optim-images.krea.ai/ref-1.webp", parsed.image_urls)
        self.assertNotIn("https://optim-images.krea.ai/https---s-krea-ai-icons-HomeIcon-png-128.webp", parsed.image_urls)
        self.assertIn("https://www.krea.ai/moodboard-feed/cinematic-blue-solitude-a057f657-b26a-5768-a134-3e21474484fe", parsed.related_urls)

    def test_catalog_upsert_searches_and_preserves_favorites(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> None:
                await init_moodboard_db(db_path)
                record = MoodboardRecord(
                    url="https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                    slug="gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                    uuid="4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                    title="Gritty Cinematic Realism",
                    taste_profile="Somber urban documentary suspense with tactile textures.",
                    keywords=["cinematic realism", "shallow depth of field", "moody natural lighting"],
                    primary_image_url="https://optim-images.krea.ai/primary.webp",
                    image_urls=["https://optim-images.krea.ai/ref-1.webp"],
                    related_urls=[],
                )
                board_id = await upsert_moodboard(record, db_path)
                await set_moodboard_favorite(board_id, True, db_path, username="admin")
                await upsert_moodboard(
                    MoodboardRecord(
                        **{
                            **record.__dict__,
                            "taste_profile": "Updated tactile urban atmosphere.",
                            "keywords": ["cinematic realism", "tactile textures"],
                        }
                    ),
                    db_path,
                )

                data = await list_moodboards(query="urban texture", favorites_only=True, db_path=db_path, username="admin")

                self.assertEqual(data["total"], 1)
                self.assertTrue(data["items"][0]["favorite"])
                self.assertEqual(data["items"][0]["keywords"], ["cinematic realism", "tactile textures"])
                self.assertIn("Updated tactile", data["items"][0]["taste_profile"])

            asyncio.run(run())

    def test_moodboard_favorites_are_per_user(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> None:
                await init_moodboard_db(db_path)
                record = MoodboardRecord(
                    url="https://www.krea.ai/moodboard-feed/per-user-style-11111111-1111-5111-9111-111111111111",
                    slug="per-user-style-11111111-1111-5111-9111-111111111111",
                    uuid="11111111-1111-5111-9111-111111111111",
                    title="Per User Style",
                    taste_profile="Moody personal favorite.",
                    keywords=["personal"],
                    primary_image_url="https://optim-images.krea.ai/primary.webp",
                    image_urls=[],
                    related_urls=[],
                )
                board_id = await upsert_moodboard(record, db_path)
                await set_moodboard_favorite(board_id, True, db_path, username="alice")

                alice = await list_moodboards(favorites_only=True, db_path=db_path, username="alice")
                bob = await list_moodboards(favorites_only=True, db_path=db_path, username="bob")
                all_for_bob = await list_moodboards(db_path=db_path, username="bob")

                self.assertEqual(alice["total"], 1)
                self.assertEqual(bob["total"], 0)
                self.assertFalse(all_for_bob["items"][0]["favorite"])

            asyncio.run(run())

    def test_catalog_items_expose_cached_preview_urls_without_ui_icons(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> None:
                await init_moodboard_db(db_path)
                await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/preview-style-11111111-1111-5111-9111-111111111111",
                        slug="preview-style-11111111-1111-5111-9111-111111111111",
                        uuid="11111111-1111-5111-9111-111111111111",
                        title="Preview Style",
                        taste_profile="Preview mood.",
                        keywords=["preview"],
                        primary_image_url="https://optim-images.krea.ai/https---gen-krea-ai-images-real-png-1024.webp",
                        image_urls=[
                            "https://optim-images.krea.ai/https---s-krea-ai-icons-HomeIcon-png-128.webp",
                            "https://optim-images.krea.ai/https---gen-krea-ai-images-real-png-32.webp",
                            "https://optim-images.krea.ai/https---gen-krea-ai-images-secondary-png-1024.webp",
                            "https://optim-images.krea.ai/https---gen-krea-ai-images-secondary-png-32.webp",
                        ],
                        related_urls=[],
                    ),
                    db_path,
                )

                item = (await list_moodboards(db_path=db_path))["items"][0]

                self.assertNotIn("HomeIcon", " ".join(item["image_urls"]))
                self.assertNotIn("png-32.webp", " ".join(item["image_urls"]))
                self.assertEqual(len(item["preview_image_urls"]), 2)
                self.assertTrue(item["preview_image_urls"][0].startswith("/api/moodboards/cached-image?url="))
                self.assertIn("gen-krea-ai-images-real", item["primary_image_url"])

            asyncio.run(run())

    def test_catalog_shuffle_is_deterministic_by_seed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> None:
                await init_moodboard_db(db_path)
                for index, title in enumerate(
                    ["Alpha Style", "Beta Style", "Gamma Style", "Delta Style", "Epsilon Style", "Zeta Style", "Eta Style", "Theta Style"],
                    start=1,
                ):
                    await upsert_moodboard(
                        MoodboardRecord(
                            url=f"https://www.krea.ai/moodboard-feed/{title.lower().replace(' ', '-')}-11111111-1111-5111-9111-11111111111{index}",
                            slug=f"{title.lower().replace(' ', '-')}-11111111-1111-5111-9111-11111111111{index}",
                            uuid=f"11111111-1111-5111-9111-11111111111{index}",
                            title=title,
                            taste_profile="Test style",
                            keywords=["test"],
                            primary_image_url="",
                            image_urls=[],
                            related_urls=[],
                        ),
                        db_path,
                    )

                first = await list_moodboards(page=1, page_size=4, source="official", shuffle_seed="phone", db_path=db_path)
                second = await list_moodboards(page=1, page_size=4, source="official", shuffle_seed="phone", db_path=db_path)
                third = await list_moodboards(page=1, page_size=4, source="official", shuffle_seed="other", db_path=db_path)

                self.assertEqual([item["id"] for item in first["items"]], [item["id"] for item in second["items"]])
                self.assertNotEqual([item["id"] for item in first["items"]], [item["id"] for item in third["items"]])

            asyncio.run(run())

    def test_exports_and_imports_portable_seed_without_local_favorites(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_db = Path(td) / "source.db"
            target_db = Path(td) / "target.db"
            seed_path = Path(td) / "krea_moodboards_seed.json"

            async def run() -> None:
                await init_moodboard_db(source_db)
                record = MoodboardRecord(
                    url="https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                    slug="gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                    uuid="4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                    title="Gritty Cinematic Realism",
                    taste_profile="Somber urban documentary suspense with tactile textures.",
                    keywords=["cinematic realism", "tactile textures"],
                    primary_image_url="https://optim-images.krea.ai/primary.webp",
                    image_urls=["https://optim-images.krea.ai/ref-1.webp"],
                    related_urls=[],
                )
                board_id = await upsert_moodboard(record, source_db)
                await set_moodboard_favorite(board_id, True, source_db)

                exported = await export_moodboard_seed(seed_path, db_path=source_db)
                self.assertEqual(exported, 1)
                self.assertTrue(seed_path.exists())

                await init_moodboard_db(target_db)
                imported = await import_moodboard_seed(seed_path, db_path=target_db)
                # Filter to official boards: init also seeds the built-in
                # Andro.Meta Classics (source='andrometa') into fresh DBs.
                data = await list_moodboards(source="official", db_path=target_db)

                self.assertEqual(imported, 1)
                self.assertEqual(data["total"], 1)
                self.assertEqual(data["items"][0]["title"], "Gritty Cinematic Realism")
                self.assertFalse(data["items"][0]["favorite"])

            asyncio.run(run())

    def test_seed_export_import_preserves_qwen_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_db = Path(td) / "source.db"
            target_db = Path(td) / "target.db"
            seed_path = Path(td) / "krea_moodboards_seed.json"

            async def run() -> tuple[dict, dict]:
                await init_moodboard_db(source_db)
                board_id = await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        slug="gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        uuid="4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        title="Gritty Cinematic Realism",
                        taste_profile="Somber urban documentary suspense.",
                        keywords=["cinematic realism"],
                        primary_image_url="https://optim-images.krea.ai/primary.webp",
                        image_urls=[],
                        related_urls=[],
                    ),
                    source_db,
                )
                await set_moodboard_qwen_guidance(
                    board_id,
                    {
                        "prompt_guidance": "Use gritty cinematic realism.",
                        "negative_guidance": "Avoid clean studio shine.",
                        "style_axes": ["grain", "moody"],
                        "conditioning_notes": ["texture first"],
                        "source_summary": "summary",
                        "guidance_version": 1,
                    },
                    db_path=source_db,
                )
                await export_moodboard_seed(seed_path, db_path=source_db)
                await init_moodboard_db(target_db)
                await import_moodboard_seed(seed_path, db_path=target_db)
                original = (await list_moodboards(db_path=source_db))["items"][0]
                imported = (await list_moodboards(db_path=target_db))["items"][0]
                return original, imported

            original, imported = asyncio.run(run())
            self.assertEqual(imported["qwen_guidance"], original["qwen_guidance"])
            self.assertEqual(imported["qwen_guidance"]["prompt_guidance"], "Use gritty cinematic realism.")

    def test_import_reports_new_moodboards_and_records_latest_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"
            existing = MoodboardRecord(
                url="https://www.krea.ai/moodboard-feed/existing-style-11111111-1111-5111-9111-111111111111",
                slug="existing-style-11111111-1111-5111-9111-111111111111",
                uuid="11111111-1111-5111-9111-111111111111",
                title="Existing Style",
                taste_profile="Already known.",
                keywords=["known"],
                primary_image_url="https://optim-images.krea.ai/existing.webp",
                image_urls=[],
                related_urls=[],
            )
            new = MoodboardRecord(
                url="https://www.krea.ai/moodboard-feed/new-neon-style-22222222-2222-5222-9222-222222222222",
                slug="new-neon-style-22222222-2222-5222-9222-222222222222",
                uuid="22222222-2222-5222-9222-222222222222",
                title="New Neon Style",
                taste_profile="Fresh neon cinematic taste.",
                keywords=["neon"],
                primary_image_url="https://optim-images.krea.ai/new.webp",
                image_urls=[],
                related_urls=[],
            )

            async def run() -> None:
                await init_moodboard_db(db_path)
                await upsert_moodboard(existing, db_path)
                with patch("moodboards_catalog.KreaMoodboardCrawler.crawl", return_value=[existing, new]):
                    result = await import_moodboard_urls([KREA_MOODBOARD_GALLERY_URL], db_path=db_path)

                latest = await latest_moodboard_discovery(db_path=db_path)

                self.assertEqual(result["imported"], 2)
                self.assertEqual(result["new_count"], 1)
                self.assertEqual(len(result["new_ids"]), 1)
                self.assertEqual(latest["new_count"], 1)
                self.assertEqual([item["title"] for item in latest["items"]], ["New Neon Style"])

            asyncio.run(run())

    def test_generation_context_formats_selected_catalog_moodboards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def seed() -> int:
                await init_moodboard_db(db_path)
                return await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        slug="gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        uuid="4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        title="Gritty Cinematic Realism",
                        taste_profile="Somber urban documentary suspense with tactile textures.",
                        keywords=["cinematic realism", "tactile textures"],
                        primary_image_url="https://optim-images.krea.ai/primary.webp",
                        image_urls=["https://optim-images.krea.ai/ref-1.webp", "https://optim-images.krea.ai/ref-2.webp"],
                        related_urls=[],
                    ),
                    db_path,
                )

            board_id = asyncio.run(seed())
            context = moodboard_generation_context([board_id], db_path=db_path)

            self.assertIn("Style-only Krea moodboard guidance", context["style_text"])
            self.assertIn("Do not introduce people", context["style_text"])
            self.assertIn("Gritty Cinematic Realism", context["style_text"])
            self.assertIn("Somber urban documentary", context["style_text"])
            self.assertIn("cinematic realism", context["style_text"])
            self.assertEqual(
                context["image_urls"],
                [
                    "https://optim-images.krea.ai/primary.webp",
                    "https://optim-images.krea.ai/ref-1.webp",
                    "https://optim-images.krea.ai/ref-2.webp",
                ],
            )
            self.assertEqual(context["uuids"], ["4e938f5c-ff17-539b-bdb2-ad7884cdb369"])

    def test_generation_context_sanitizes_stored_subject_locked_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def seed() -> int:
                await init_moodboard_db(db_path)
                board_id = await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/glitch-woman-11111111-1111-5111-9111-111111111111",
                        slug="glitch-woman-11111111-1111-5111-9111-111111111111",
                        uuid="11111111-1111-5111-9111-111111111111",
                        title="Glitch Woman Style",
                        taste_profile="Glitchy halftone texture.",
                        keywords=["glitch", "halftone"],
                        primary_image_url="",
                        image_urls=[],
                        related_urls=[],
                    ),
                    db_path,
                )
                await set_moodboard_qwen_guidance(
                    board_id,
                    {
                        "prompt_guidance": "A black and white portrait of a young woman with short hair. Use acid green glitch and halftone texture.",
                        "negative_guidance": "Avoid human faces, crowds, text, buildings. Avoid flat lighting.",
                        "style_axes": ["young woman", "halftone"],
                        "conditioning_notes": [],
                        "source_summary": "portrait of a young woman",
                        "guidance_version": 1,
                    },
                    db_path=db_path,
                )
                return board_id

            board_id = asyncio.run(seed())
            context = moodboard_generation_context([board_id], db_path=db_path)

            blob = f"{context['style_text']} {context['negative_text']}".lower()
            for forbidden in ("young woman", "human faces", "crowds", "buildings"):
                self.assertIsNone(re.search(rf"\b{re.escape(forbidden)}\b", blob), forbidden)
            self.assertIn("user-requested subject", blob)
            self.assertIn("acid green glitch", context["style_text"])
            self.assertIn("halftone", context["style_text"])

    def test_generation_context_filters_krea_ui_icon_images(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def seed() -> int:
                await init_moodboard_db(db_path)
                return await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        slug="gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        uuid="4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        title="Gritty Cinematic Realism",
                        taste_profile="Somber urban documentary suspense with tactile textures.",
                        keywords=["cinematic realism", "tactile textures"],
                        primary_image_url="https://optim-images.krea.ai/primary.webp",
                        image_urls=[
                            "https://optim-images.krea.ai/https---s-krea-ai-icons-HomeIcon-png-128.webp",
                            "https://optim-images.krea.ai/ref-1.webp",
                        ],
                        related_urls=[],
                    ),
                    db_path,
                )

            board_id = asyncio.run(seed())
            context = moodboard_generation_context([board_id], db_path=db_path)

            self.assertEqual(
                context["image_urls"],
                [
                    "https://optim-images.krea.ai/primary.webp",
                    "https://optim-images.krea.ai/ref-1.webp",
                ],
            )

    def test_qwen_guidance_is_stored_without_rewriting_official_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> tuple[dict, dict]:
                await init_moodboard_db(db_path)
                board_id = await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        slug="gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        uuid="4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        title="Gritty Cinematic Realism",
                        taste_profile="Somber urban documentary suspense.",
                        keywords=["cinematic realism"],
                        primary_image_url="https://optim-images.krea.ai/primary.webp",
                        image_urls=["https://optim-images.krea.ai/ref-1.webp"],
                        related_urls=[],
                    ),
                    db_path,
                )
                await set_moodboard_qwen_guidance(
                    board_id,
                    {
                        "title": "Should Not Replace Official Title",
                        "keywords": ["should not replace"],
                        "prompt_guidance": "Translate this board into candid urban realism.",
                        "negative_guidance": "Avoid glossy studio light.",
                        "style_axes": ["gritty realism"],
                        "conditioning_notes": ["Use references for texture."],
                        "source_summary": "Qwen prompt guidance.",
                        "guidance_version": 1,
                    },
                    db_path=db_path,
                )
                listed = await list_moodboards(db_path=db_path)
                context = moodboard_generation_context([board_id], db_path=db_path)
                return listed["items"][0], context

            item, context = asyncio.run(run())

            self.assertEqual(item["title"], "Gritty Cinematic Realism")
            self.assertEqual(item["keywords"], ["cinematic realism"])
            self.assertEqual(item["qwen_guidance"]["prompt_guidance"], "Translate this board into candid urban realism.")
            self.assertIn("Translate this board", context["style_text"])
            self.assertIn("gritty realism", context["style_text"])
            # Conditioning notes duplicate the guardrail preamble and are kept
            # out of the generation prompt to preserve user-prompt attention.
            self.assertNotIn("Use references for texture", context["style_text"])
            self.assertIn("Avoid glossy studio light", context["negative_text"])

    def test_generation_context_sanitizes_catalog_keywords_and_dedupes_style_terms(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def seed() -> int:
                await init_moodboard_db(db_path)
                board_id = await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/silhouette-board-11111111-1111-5111-9111-111111111111",
                        slug="silhouette-board-11111111-1111-5111-9111-111111111111",
                        uuid="11111111-1111-5111-9111-111111111111",
                        title="Abyssal Gothic Surrealism",
                        taste_profile="Deep teal gothic atmosphere.",
                        keywords=["solitary silhouette", "deep teal and indigo", "painterly oil texture", "lone figure"],
                        primary_image_url="",
                        image_urls=[],
                        related_urls=[],
                    ),
                    db_path,
                )
                await set_moodboard_qwen_guidance(
                    board_id,
                    {
                        "prompt_guidance": "Palette: deep teal and navy. Medium and texture: painterly oil texture.",
                        "negative_guidance": "Avoid flat lighting.",
                        "style_axes": ["chiaroscuro lighting", "painterly oil texture"],
                        "conditioning_notes": ["Apply as transferable style."],
                        "source_summary": "Gothic surrealist treatment.",
                        "guidance_version": 2,
                    },
                    db_path=db_path,
                )
                return board_id

            board_id = asyncio.run(seed())
            context = moodboard_generation_context([board_id], db_path=db_path)
            style_text = context["style_text"]

            self.assertNotIn("solitary silhouette", style_text.lower())
            self.assertNotIn("lone figure", style_text.lower())
            self.assertIn("deep teal and indigo", style_text)
            self.assertIn("chiaroscuro lighting", style_text)
            # Terms already covered by the prose guidance are deduped out.
            self.assertEqual(style_text.lower().count("painterly oil texture"), 1)

    def test_search_uses_qwen_guidance_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> dict:
                await init_moodboard_db(db_path)
                board_id = await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/plain-board-11111111-1111-5111-9111-111111111111",
                        slug="plain-board-11111111-1111-5111-9111-111111111111",
                        uuid="11111111-1111-5111-9111-111111111111",
                        title="Plain Board",
                        taste_profile="Neutral board.",
                        keywords=["neutral"],
                        primary_image_url="https://optim-images.krea.ai/plain.webp",
                        image_urls=[],
                        related_urls=[],
                    ),
                    db_path,
                )
                await set_moodboard_qwen_guidance(
                    board_id,
                    {
                        "prompt_guidance": "opal cyber shrine lighting",
                        "negative_guidance": "avoid sterile white",
                        "style_axes": ["ritual neon"],
                        "conditioning_notes": ["glass refractions"],
                        "source_summary": "prismatic altar",
                        "guidance_version": 1,
                    },
                    db_path=db_path,
                )
                return await list_moodboards(query="refractions", db_path=db_path)

            result = asyncio.run(run())
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["items"][0]["title"], "Plain Board")

    def test_generation_context_caps_images_across_boards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> dict:
                await init_moodboard_db(db_path)
                ids = []
                for index in range(2):
                    ids.append(await upsert_moodboard(
                        MoodboardRecord(
                            url=f"https://www.krea.ai/moodboard-feed/board-{index}-11111111-1111-5111-9111-11111111111{index}",
                            slug=f"board-{index}-11111111-1111-5111-9111-11111111111{index}",
                            uuid=f"11111111-1111-5111-9111-11111111111{index}",
                            title=f"Board {index}",
                            taste_profile="",
                            keywords=[],
                            primary_image_url=f"https://optim-images.krea.ai/{index}-primary.webp",
                            image_urls=[f"https://optim-images.krea.ai/{index}-{n}.webp" for n in range(4)],
                            related_urls=[],
                        ),
                        db_path,
                    ))
                return moodboard_generation_context(ids, db_path=db_path, max_images=3)

            context = asyncio.run(run())
            self.assertEqual(len(context["image_urls"]), 3)

    def test_generation_context_resolves_uuid_moodboards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def seed() -> int:
                await init_moodboard_db(db_path)
                await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        slug="gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        uuid="4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        title="Gritty Cinematic Realism",
                        taste_profile="Somber urban documentary suspense.",
                        keywords=["cinematic realism"],
                        primary_image_url="https://optim-images.krea.ai/primary.webp",
                        image_urls=[],
                        related_urls=[],
                    ),
                    db_path,
                )
                return await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/neon-product-studio-a057f657-b26a-5768-a134-3e21474484fe",
                        slug="neon-product-studio-a057f657-b26a-5768-a134-3e21474484fe",
                        uuid="a057f657-b26a-5768-a134-3e21474484fe",
                        title="Neon Product Studio",
                        taste_profile="Glossy product lighting.",
                        keywords=["neon", "product"],
                        primary_image_url="https://optim-images.krea.ai/neon.webp",
                        image_urls=[],
                        related_urls=[],
                    ),
                    db_path,
                )

            board_id = asyncio.run(seed())
            context = moodboard_generation_context(
                [board_id],
                moodboard_uuids=["4e938f5c-ff17-539b-bdb2-ad7884cdb369"],
                db_path=db_path,
            )

            self.assertEqual([item["title"] for item in context["items"]], ["Neon Product Studio", "Gritty Cinematic Realism"])
            self.assertEqual(
                context["uuids"],
                ["a057f657-b26a-5768-a134-3e21474484fe", "4e938f5c-ff17-539b-bdb2-ad7884cdb369"],
            )

    def test_custom_moodboard_persists_images_and_uses_generation_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"
            storage_dir = Path(td) / "custom"

            async def run() -> tuple[dict, dict, str, bool]:
                await init_moodboard_db(db_path)
                created = await create_custom_moodboard(
                    title="My Neon Board",
                    taste_profile="Pink glass and diagonal neon lighting.",
                    keywords=["pink glass", "neon"],
                    image_b64s=[self.TINY_PNG_B64],
                    db_path=db_path,
                    storage_dir=storage_dir,
                )
                listed = await list_moodboards(source="custom", db_path=db_path)
                context = moodboard_generation_context([created["id"]], db_path=db_path)
                fetched = fetch_moodboard_image_b64(created["image_urls"][0], storage_dir=storage_dir)
                deleted = await delete_custom_moodboard(created["id"], db_path=db_path, storage_dir=storage_dir)
                return listed, context, fetched, deleted

            listed, context, fetched, deleted = asyncio.run(run())

            self.assertEqual(listed["total"], 1)
            self.assertEqual(listed["items"][0]["source"], "custom")
            self.assertEqual(context["items"][0]["title"], "My Neon Board")
            self.assertIn("Pink glass", context["style_text"])
            self.assertEqual(fetched, self.TINY_PNG_B64)
            self.assertTrue(deleted)

    def test_custom_moodboard_auto_authors_missing_metadata_with_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"
            storage_dir = Path(td) / "custom"

            async def run() -> dict:
                await init_moodboard_db(db_path)
                return await create_custom_moodboard(
                    title="",
                    taste_profile="",
                    keywords=[],
                    image_b64s=[self.TINY_PNG_B64],
                    db_path=db_path,
                    storage_dir=storage_dir,
                    guidance_generator=lambda _prompt, images: f"""
                    {{
                      "title": "Neon Rain Glass",
                      "taste_profile": "Reflective cyber-noir with rain-slick glass and pink rim light.",
                      "keywords": ["cyber-noir", "rain glass", "pink rim light"],
                      "prompt_guidance": "Use wet reflective surfaces and neon contrast from {len(images)} reference.",
                      "negative_guidance": "Avoid flat daylight.",
                      "style_axes": ["neon noir"],
                      "conditioning_notes": ["Use uploaded image for palette."],
                      "source_summary": "Auto-authored custom moodboard.",
                      "guidance_version": 1
                    }}
                    """,
                )

            created = asyncio.run(run())

            self.assertEqual(created["title"], "Neon Rain Glass")
            self.assertIn("cyber-noir", created["keywords"])
            self.assertIn("Reflective cyber-noir", created["taste_profile"])
            self.assertIn("wet reflective surfaces", created["qwen_guidance"]["prompt_guidance"])

    def test_official_moodboard_sync_does_not_overwrite_custom_boards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"
            storage_dir = Path(td) / "custom"

            async def run() -> tuple[int, int]:
                await init_moodboard_db(db_path)
                await create_custom_moodboard(
                    title="Custom",
                    taste_profile="Private board.",
                    keywords=[],
                    image_b64s=[self.TINY_PNG_B64],
                    db_path=db_path,
                    storage_dir=storage_dir,
                )
                await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/official-11111111-1111-5111-9111-111111111111",
                        slug="official-11111111-1111-5111-9111-111111111111",
                        uuid="11111111-1111-5111-9111-111111111111",
                        title="Official",
                        taste_profile="Synced.",
                        keywords=[],
                        primary_image_url="https://optim-images.krea.ai/official.webp",
                        image_urls=[],
                        related_urls=[],
                    ),
                    db_path,
                )
                custom = await list_moodboards(source="custom", db_path=db_path)
                official = await list_moodboards(source="official", db_path=db_path)
                return custom["total"], official["total"]

            custom_count, official_count = asyncio.run(run())

            self.assertEqual(custom_count, 1)
            self.assertEqual(official_count, 1)

    def test_mashup_moodboard_requires_qwen_and_saves_custom_board(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"
            storage_dir = Path(td) / "custom"

            async def run() -> dict:
                await init_moodboard_db(db_path)
                first = await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        slug="gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        uuid="4e938f5c-ff17-539b-bdb2-ad7884cdb369",
                        title="Gritty Cinematic Realism",
                        taste_profile="Somber urban documentary suspense.",
                        keywords=["cinematic realism"],
                        primary_image_url="https://optim-images.krea.ai/first.webp",
                        image_urls=[],
                        related_urls=[],
                    ),
                    db_path,
                )
                second = await upsert_moodboard(
                    MoodboardRecord(
                        url="https://www.krea.ai/moodboard-feed/neon-product-studio-a057f657-b26a-5768-a134-3e21474484fe",
                        slug="neon-product-studio-a057f657-b26a-5768-a134-3e21474484fe",
                        uuid="a057f657-b26a-5768-a134-3e21474484fe",
                        title="Neon Product Studio",
                        taste_profile="Glossy neon product lighting.",
                        keywords=["neon", "product"],
                        primary_image_url="https://optim-images.krea.ai/second.webp",
                        image_urls=[],
                        related_urls=[],
                    ),
                    db_path,
                )
                with patch("moodboards_catalog.fetch_moodboard_image_b64", return_value=self.TINY_PNG_B64):
                    return await create_mashup_moodboard(
                        moodboard_ids=[first, second],
                        weights=[0.65, 0.35],
                        db_path=db_path,
                        storage_dir=storage_dir,
                        guidance_generator=lambda prompt, images: f"""
                        {{
                          "title": "Gritty Neon Documentary",
                          "taste_profile": "A hybrid of candid street realism and neon product glow.",
                          "keywords": ["gritty neon", "documentary product", "wet reflections"],
                          "prompt_guidance": "Blend gritty realism with neon highlights from {len(images)} references.",
                          "negative_guidance": "Avoid clean catalog sterility.",
                          "style_axes": ["street realism", "neon gloss"],
                          "conditioning_notes": ["Use weighted source moodboards."],
                          "source_summary": "{'Gritty Cinematic Realism' in prompt}",
                          "guidance_version": 1
                        }}
                        """,
                    )

            created = asyncio.run(run())

            self.assertEqual(created["source"], "custom")
            self.assertEqual(created["title"], "Gritty Neon Documentary")
            self.assertIn("gritty neon", created["keywords"])
            self.assertIn("Blend gritty realism", created["qwen_guidance"]["prompt_guidance"])
            self.assertEqual(len(created["image_urls"]), 2)

    def test_title_qualifiers_disambiguate_duplicates_idempotently(self) -> None:
        import json as jsonlib

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"
            qualifiers_path = Path(td) / "qualifiers.json"

            async def run() -> list[str]:
                await init_moodboard_db(db_path)
                for index, uuid in enumerate(["11111111-1111-5111-9111-111111111111", "22222222-2222-5222-9222-222222222222"]):
                    await upsert_moodboard(
                        MoodboardRecord(
                            url=f"https://www.krea.ai/moodboard-feed/cinematic-noir-{uuid}",
                            slug=f"cinematic-noir-{uuid}",
                            uuid=uuid,
                            title="Cinematic Noir",
                            taste_profile=f"Variant {index}.",
                            keywords=[],
                            primary_image_url="",
                            image_urls=[],
                            related_urls=[],
                        ),
                        db_path,
                    )
                qualifiers_path.write_text(jsonlib.dumps({
                    "11111111-1111-5111-9111-111111111111": "Silver Grain",
                    "22222222-2222-5222-9222-222222222222": "Amber Rain",
                }), encoding="utf-8")
                first = await apply_title_qualifiers(db_path=db_path, qualifiers_path=qualifiers_path)
                second = await apply_title_qualifiers(db_path=db_path, qualifiers_path=qualifiers_path)
                assert first == 2
                assert second == 0  # idempotent
                data = await list_moodboards(source="official", db_path=db_path)
                return sorted(item["title"] for item in data["items"])

            titles = asyncio.run(run())
            self.assertEqual(titles, ["Cinematic Noir \u2014 Amber Rain", "Cinematic Noir \u2014 Silver Grain"])

    def test_sync_throttle_uses_daily_interval(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"

            async def run() -> None:
                await init_moodboard_db(db_path)
                self.assertTrue(await should_sync_moodboards(db_path=db_path, now=1_000, interval_seconds=86_400))
                self.assertFalse(await should_sync_moodboards(db_path=db_path, now=1_100, interval_seconds=86_400, mark=True))
                self.assertTrue(await should_sync_moodboards(db_path=db_path, now=90_000, interval_seconds=86_400))

            asyncio.run(run())

    def test_image_proxy_allows_only_krea_image_hosts(self) -> None:
        self.assertTrue(is_allowed_krea_image_url("https://optim-images.krea.ai/ref.webp"))
        self.assertFalse(is_allowed_krea_image_url("https://example.com/ref.webp"))
        self.assertFalse(is_allowed_krea_image_url("http://optim-images.krea.ai/ref.webp"))

    def test_import_allows_only_krea_moodboard_urls(self) -> None:
        self.assertTrue(is_allowed_krea_moodboard_url("https://www.krea.ai/moodboard-feed/example-4e938f5c-ff17-539b-bdb2-ad7884cdb369"))
        self.assertTrue(is_allowed_krea_moodboard_url("https://www.krea.ai/app?gallery=moodboards"))
        self.assertFalse(is_allowed_krea_moodboard_url("https://example.com/moodboard-feed/example"))
        self.assertFalse(is_allowed_krea_moodboard_url("http://www.krea.ai/moodboard-feed/example"))

    def test_seeded_crawl_follows_related_links_once(self) -> None:
        pages = {
            KREA_MOODBOARD_GALLERY_URL: '<a href="/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369">Gritty</a>',
            "https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369": FIXTURE_HTML,
            "https://www.krea.ai/moodboard-feed/cinematic-blue-solitude-a057f657-b26a-5768-a134-3e21474484fe": FIXTURE_HTML.replace("Gritty Cinematic Realism", "Cinematic Blue Solitude"),
        }
        crawler = KreaMoodboardCrawler(fetch_html=lambda url: pages[url])

        records = crawler.crawl([
            "https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369"
        ], max_pages=5)

        self.assertEqual([r.title for r in records], ["Gritty Cinematic Realism", "Cinematic Blue Solitude"])

    def test_gallery_seed_discovers_moodboard_links(self) -> None:
        pages = {
            KREA_MOODBOARD_GALLERY_URL: '<a href="/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369">Gritty</a>',
            "https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369": FIXTURE_HTML,
        }
        crawler = KreaMoodboardCrawler(fetch_html=lambda url: pages[url])

        records = crawler.crawl([KREA_MOODBOARD_GALLERY_URL], max_pages=1)

        self.assertEqual([r.title for r in records], ["Gritty Cinematic Realism"])


if __name__ == "__main__":
    unittest.main()
