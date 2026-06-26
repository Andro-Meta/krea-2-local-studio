from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

os.environ.setdefault("KREA2_AUTO_CHECKPOINT", "__disabled_for_tests__")

try:
    import torch  # noqa: F401
    inserted_torch_stub = False
except ModuleNotFoundError:
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.bfloat16 = "bfloat16"
    torch_mock.float32 = "float32"
    torch_mock.Tensor = object
    torch_mock.nn = SimpleNamespace(Module=object, Linear=object)
    sys.modules["torch"] = torch_mock
    inserted_torch_stub = True

from fastapi.testclient import TestClient  # noqa: E402
from backend import main  # noqa: E402

if inserted_torch_stub:
    sys.modules.pop("torch", None)


MOODBOARD_ITEM = {
    "id": 7,
    "url": "https://www.krea.ai/moodboard-feed/gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
    "slug": "gritty-cinematic-realism-4e938f5c-ff17-539b-bdb2-ad7884cdb369",
    "uuid": "4e938f5c-ff17-539b-bdb2-ad7884cdb369",
    "title": "Gritty Cinematic Realism",
    "taste_profile": "Somber urban documentary suspense.",
    "keywords": ["cinematic realism"],
    "primary_image_url": "https://optim-images.krea.ai/primary.webp",
    "image_urls": ["https://optim-images.krea.ai/ref.webp"],
    "related_urls": [],
    "favorite": False,
    "first_seen_at": "2026-01-01T00:00:00Z",
    "last_seen_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "sync_error": "",
}


class MoodboardApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_moodboard_routes_return_catalog_data(self) -> None:
        client = TestClient(main.app)

        async def fake_list(**_: object) -> dict:
            return {"items": [MOODBOARD_ITEM], "total": 1}

        async def fake_detail(_: int) -> dict:
            return MOODBOARD_ITEM

        async def fake_favorite(_: int, __: bool) -> None:
            return None

        async def fake_import(_: list[str], max_pages: int = 200, use_browser_discovery: bool = False) -> dict:
            return {"imported": 1, "ids": [7]}

        with (
            patch.object(main, "list_moodboards", side_effect=fake_list),
            patch.object(main, "get_moodboard", side_effect=fake_detail),
            patch.object(main, "set_moodboard_favorite", side_effect=fake_favorite),
            patch.object(main, "import_moodboard_urls", side_effect=fake_import),
            patch.object(main, "fetch_krea_image_b64", return_value="abc123"),
        ):
            listed = client.get("/api/moodboards?q=urban")
            detail = client.get("/api/moodboards/7")
            favorite = client.put("/api/moodboards/7/favorite", json={"favorite": True})
            imported = client.post("/api/moodboards/import", json={"urls": [MOODBOARD_ITEM["url"]], "max_pages": 1})
            image = client.post("/api/moodboards/image", json={"url": "https://optim-images.krea.ai/ref.webp"})

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["items"][0]["title"], "Gritty Cinematic Realism")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(favorite.json(), {"ok": True})
        self.assertEqual(imported.json(), {"imported": 1, "ids": [7]})
        self.assertEqual(image.json(), {"image_b64": "abc123"})


if __name__ == "__main__":
    unittest.main()
