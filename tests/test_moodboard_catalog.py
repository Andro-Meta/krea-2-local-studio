from __future__ import annotations

import asyncio
import sys
import tempfile
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
    create_custom_moodboard,
    delete_custom_moodboard,
    fetch_moodboard_image_b64,
    create_mashup_moodboard,
    init_moodboard_db,
    is_allowed_krea_image_url,
    is_allowed_krea_moodboard_url,
    export_moodboard_seed,
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
                await set_moodboard_favorite(board_id, True, db_path)
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

                data = await list_moodboards(query="urban texture", favorites_only=True, db_path=db_path)

                self.assertEqual(data["total"], 1)
                self.assertTrue(data["items"][0]["favorite"])
                self.assertEqual(data["items"][0]["keywords"], ["cinematic realism", "tactile textures"])
                self.assertIn("Updated tactile", data["items"][0]["taste_profile"])

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
                            "https://optim-images.krea.ai/https---gen-krea-ai-images-secondary-png-1024.webp",
                        ],
                        related_urls=[],
                    ),
                    db_path,
                )

                item = (await list_moodboards(db_path=db_path))["items"][0]

                self.assertNotIn("HomeIcon", " ".join(item["image_urls"]))
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
                data = await list_moodboards(db_path=target_db)

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
            self.assertIn("Use references for texture", context["style_text"])
            self.assertIn("Avoid glossy studio light", context["negative_text"])

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
