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


class GalleryMetadataTests(unittest.TestCase):
    def test_gallery_persists_and_returns_metadata_json(self) -> None:
        import gallery

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "app.db"
            out_dir = Path(tmp) / "outputs"
            out_dir.mkdir()

            async def run() -> None:
                with (
                    patch.object(gallery, "DB_PATH", db_path),
                    patch.object(gallery, "OUTPUTS_DIR", out_dir),
                ):
                    await gallery.init_db()
                    image_id = await gallery.save_image(
                        "example.png",
                        prompt="a glass forest",
                        seed=99,
                        metadata={"prompt": "a glass forest", "seed": 99, "steps": 8},
                    )
                    data = await gallery.get_gallery()

                self.assertEqual(image_id, 1)
                self.assertEqual(data["items"][0]["metadata"]["prompt"], "a glass forest")
                self.assertEqual(data["items"][0]["metadata"]["seed"], 99)

            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
