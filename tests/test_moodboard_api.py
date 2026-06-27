from __future__ import annotations

import os
import sys
import unittest
import importlib.util
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

if importlib.util.find_spec("torch") is None:
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.bfloat16 = "bfloat16"
    torch_mock.float32 = "float32"
    torch_mock.Tensor = object
    torch_mock.nn = SimpleNamespace(Module=object, Linear=object)
    sys.modules["torch"] = torch_mock
    inserted_torch_stub = True
else:
    inserted_torch_stub = False

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
    "source": "official",
    "first_seen_at": "2026-01-01T00:00:00Z",
    "last_seen_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "sync_error": "",
    "qwen_guidance": {},
    "qwen_guidance_at": "",
    "qwen_guidance_version": 0,
}


class MoodboardApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_share_auth_policy_allows_readonly_moodboard_catalog(self) -> None:
        self.assertTrue(main._is_auth_exempt("/api/moodboards", "GET"))
        self.assertTrue(main._is_auth_exempt("/api/moodboards/7", "GET"))
        self.assertTrue(main._is_auth_exempt("/api/moodboards/discoveries/latest", "GET"))
        self.assertFalse(main._is_auth_exempt("/api/moodboards/import", "POST"))
        self.assertTrue(main._requires_admin("/api/moodboards/import", "POST"))

    async def test_moodboard_routes_return_catalog_data(self) -> None:
        client = TestClient(main.app)

        async def fake_list(**_: object) -> dict:
            return {"items": [MOODBOARD_ITEM], "total": 1}

        async def fake_detail(_: int) -> dict:
            return MOODBOARD_ITEM

        async def fake_favorite(_: int, __: bool) -> None:
            return None

        async def fake_import(_: list[str], max_pages: int = 200, use_browser_discovery: bool = False) -> dict:
            return {"imported": 1, "ids": [7], "new_count": 1, "new_ids": [7]}

        async def fake_export(_: object) -> int:
            return 1

        async def fake_latest_discovery() -> dict:
            return {"id": "2026-01-01T00:00:00Z", "discovered_at": "2026-01-01T00:00:00Z", "new_count": 1, "new_ids": [7], "items": [MOODBOARD_ITEM]}

        with (
            patch.object(main, "list_moodboards", side_effect=fake_list),
            patch.object(main, "get_moodboard", side_effect=fake_detail),
            patch.object(main, "set_moodboard_favorite", side_effect=fake_favorite),
            patch.object(main, "import_moodboard_urls", side_effect=fake_import),
            patch.object(main, "export_moodboard_seed", side_effect=fake_export),
            patch.object(main, "latest_moodboard_discovery", side_effect=fake_latest_discovery),
            patch.object(main, "fetch_moodboard_image_b64", return_value="abc123"),
        ):
            listed = client.get("/api/moodboards?q=urban")
            detail = client.get("/api/moodboards/7")
            favorite = client.put("/api/moodboards/7/favorite", json={"favorite": True})
            imported = client.post("/api/moodboards/import", json={"urls": [MOODBOARD_ITEM["url"]], "max_pages": 1})
            exported = client.post("/api/moodboards/export-seed")
            image = client.post("/api/moodboards/image", json={"url": "https://optim-images.krea.ai/ref.webp"})
            latest = client.get("/api/moodboards/discoveries/latest")

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["items"][0]["title"], "Gritty Cinematic Realism")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(favorite.json(), {"ok": True})
        self.assertEqual(imported.json(), {"imported": 1, "ids": [7], "new_count": 1, "new_ids": [7]})
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(exported.json()["exported"], 1)
        self.assertEqual(image.json(), {"image_b64": "abc123"})
        self.assertEqual(latest.json()["items"][0]["title"], "Gritty Cinematic Realism")

    async def test_custom_moodboard_routes_create_and_delete(self) -> None:
        client = TestClient(main.app)
        custom_item = {**MOODBOARD_ITEM, "id": 8, "source": "custom", "title": "My Board"}

        async def fake_create(**_: object) -> dict:
            return custom_item

        async def fake_delete(_: int) -> bool:
            return True

        with (
            patch.object(main, "create_custom_moodboard", side_effect=fake_create),
            patch.object(main, "delete_custom_moodboard", side_effect=fake_delete),
        ):
            created = client.post("/api/moodboards/custom", json={
                "title": "My Board",
                "taste_profile": "Neon style.",
                "keywords": ["neon"],
                "image_b64s": ["abc123"],
            })
            deleted = client.delete("/api/moodboards/custom/8")

        self.assertEqual(created.status_code, 200)
        self.assertEqual(created.json()["source"], "custom")
        self.assertEqual(created.json()["title"], "My Board")
        self.assertEqual(deleted.json(), {"ok": True})

    async def test_custom_moodboard_image_route_rejects_path_traversal(self) -> None:
        client = TestClient(main.app)
        board_uuid = "11111111-1111-4111-8111-111111111111"

        self.assertEqual(
            client.get(f"/api/moodboards/custom-images/{board_uuid}/ref_00.png").status_code,
            404,
        )
        self.assertEqual(
            client.get(f"/api/moodboards/custom-images/{board_uuid}/..%5Cshare_auth.json").status_code,
            404,
        )
        self.assertEqual(
            client.get("/api/moodboards/custom-images/not-a-uuid/ref_00.png").status_code,
            404,
        )

    async def test_custom_moodboard_auto_authoring_failure_is_clear(self) -> None:
        client = TestClient(main.app)

        async def fake_create(**_: object) -> dict:
            raise RuntimeError("Local Qwen unavailable")

        with patch.object(main, "create_custom_moodboard", side_effect=fake_create):
            created = client.post("/api/moodboards/custom", json={
                "title": "",
                "taste_profile": "",
                "keywords": [],
                "image_b64s": ["abc123"],
            })

        self.assertEqual(created.status_code, 502)
        self.assertIn("Qwen custom moodboard authoring failed", created.json()["detail"])

    async def test_qwen_guidance_routes_generate_single_and_missing(self) -> None:
        client = TestClient(main.app)
        guidance = {
            "prompt_guidance": "Use gritty documentary realism.",
            "negative_guidance": "Avoid glossy studio light.",
            "style_axes": ["documentary realism"],
            "conditioning_notes": ["Use references for texture."],
            "source_summary": "Qwen guidance.",
            "guidance_version": 1,
        }

        async def fake_single(moodboard_id: int, **_: object) -> dict:
            return {**MOODBOARD_ITEM, "id": moodboard_id, "qwen_guidance": guidance, "qwen_guidance_version": 1}

        async def fake_missing(**_: object) -> dict:
            return {"processed": 1, "items": [{**MOODBOARD_ITEM, "qwen_guidance": guidance, "qwen_guidance_version": 1}]}

        with (
            patch.object(main, "generate_and_store_moodboard_qwen_guidance", side_effect=fake_single),
            patch.object(main, "generate_missing_moodboard_qwen_guidance", side_effect=fake_missing),
        ):
            single = client.post("/api/moodboards/7/qwen-guidance")
            missing = client.post("/api/moodboards/qwen-guidance-missing", json={"limit": 5})

        self.assertEqual(single.status_code, 200)
        self.assertEqual(single.json()["qwen_guidance"]["prompt_guidance"], "Use gritty documentary realism.")
        self.assertEqual(missing.status_code, 200)
        self.assertEqual(missing.json()["processed"], 1)

    async def test_mashup_route_creates_custom_moodboard(self) -> None:
        client = TestClient(main.app)
        custom_item = {
            **MOODBOARD_ITEM,
            "id": 12,
            "source": "custom",
            "title": "Gritty Neon Documentary",
            "qwen_guidance": {"prompt_guidance": "Blend gritty realism with neon."},
            "qwen_guidance_version": 1,
        }

        async def fake_mashup(**_: object) -> dict:
            return custom_item

        with patch.object(main, "create_mashup_moodboard", side_effect=fake_mashup):
            created = client.post("/api/moodboards/mashup", json={"moodboard_ids": [7, 8], "weights": [0.7, 0.3]})

        self.assertEqual(created.status_code, 200)
        self.assertEqual(created.json()["source"], "custom")
        self.assertEqual(created.json()["title"], "Gritty Neon Documentary")


if __name__ == "__main__":
    unittest.main()
