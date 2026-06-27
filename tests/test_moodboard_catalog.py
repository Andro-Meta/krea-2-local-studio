from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from moodboards_catalog import (  # noqa: E402
    KREA_MOODBOARD_GALLERY_URL,
    KreaMoodboardCrawler,
    MoodboardRecord,
    init_moodboard_db,
    is_allowed_krea_image_url,
    is_allowed_krea_moodboard_url,
    export_moodboard_seed,
    import_moodboard_seed,
    list_moodboards,
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
    <a href="/moodboard-feed/cinematic-blue-solitude-a057f657-b26a-5768-a134-3e21474484fe">Cinematic Blue Solitude</a>
  </body>
</html>
"""


class MoodboardCatalogTests(unittest.TestCase):
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
