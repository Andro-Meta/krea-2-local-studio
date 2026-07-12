from __future__ import annotations

import os
import sys
import tempfile
import time
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
from backend.gpu_task_queue import GpuTaskQueue  # noqa: E402
from backend.gpu_tasks import MOODBOARD_GUIDANCE  # noqa: E402
from moodboards_catalog import (  # noqa: E402
    get_moodboard as catalog_get_moodboard,
    latest_moodboard_discovery as catalog_latest_discovery,
    MoodboardRecord,
    init_moodboard_db,
    list_moodboards as catalog_list_moodboards,
    set_moodboard_favorite,
    upsert_moodboard,
)
import moodboards_catalog  # noqa: E402
from support import mock_atomic_cancel_capability  # noqa: E402

mock_atomic_cancel_capability(main)

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
TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="


class MoodboardApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_share_auth_policy_allows_readonly_moodboard_catalog(self) -> None:
        self.assertTrue(main._is_auth_exempt("/api/moodboards", "GET"))
        self.assertTrue(main._is_auth_exempt("/api/moodboards/7", "GET"))
        self.assertTrue(main._is_auth_exempt("/api/moodboards/discoveries/latest", "GET"))
        self.assertFalse(main._is_auth_exempt("/api/moodboards/import", "POST"))
        self.assertTrue(main._requires_admin("/api/moodboards/import", "POST"))

    async def test_public_catalog_username_prefers_state_and_defaults_local(self) -> None:
        state_request = SimpleNamespace(
            state=SimpleNamespace(share_user="alice"),
            cookies={main.SHARE_COOKIE: "unused"},
        )
        anonymous_request = SimpleNamespace(state=SimpleNamespace(), cookies={})
        with (
            patch.object(main, "SHARE_AUTH_ENABLED", True),
            patch.object(main, "_auth_username_from_cookie", return_value=None) as cookie_auth,
        ):
            self.assertEqual(main._public_moodboard_username(state_request), "alice")
            cookie_auth.assert_not_called()
            self.assertEqual(
                main._public_moodboard_username(anonymous_request),
                main.PUBLIC_ANONYMOUS_USERNAME,
            )
            self.assertFalse(
                main.is_valid_username(main.PUBLIC_ANONYMOUS_USERNAME)
            )
        with patch.object(main, "SHARE_AUTH_ENABLED", False):
            self.assertEqual(
                main._public_moodboard_username(SimpleNamespace(cookies={})),
                "__local__",
            )

    async def test_public_catalog_resolves_cookie_user_and_keeps_favorites_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "catalog.db"
            await init_moodboard_db(db_path)
            alice_board = await upsert_moodboard(
                MoodboardRecord(
                    url="https://www.krea.ai/moodboard-feed/alice-favorite-11111111-1111-5111-9111-111111111111",
                    slug="alice-favorite-11111111-1111-5111-9111-111111111111",
                    uuid="11111111-1111-5111-9111-111111111111",
                    title="Alice Favorite",
                    taste_profile="Muted teal grain.",
                    keywords=["muted teal"],
                ),
                db_path,
            )
            bob_board = await upsert_moodboard(
                MoodboardRecord(
                    url="https://www.krea.ai/moodboard-feed/bob-favorite-22222222-2222-5222-9222-222222222222",
                    slug="bob-favorite-22222222-2222-5222-9222-222222222222",
                    uuid="22222222-2222-5222-9222-222222222222",
                    title="Bob Favorite",
                    taste_profile="Amber analog grain.",
                    keywords=["amber grain"],
                ),
                db_path,
            )
            await set_moodboard_favorite(alice_board, True, db_path, username="alice")
            await set_moodboard_favorite(bob_board, True, db_path, username="bob")
            await set_moodboard_favorite(alice_board, True, db_path, username="__local__")
            await moodboards_catalog._record_moodboard_discovery(
                [alice_board, bob_board], db_path=db_path
            )

            async def scoped_list(**kwargs: object) -> dict:
                return await catalog_list_moodboards(**kwargs, db_path=db_path)

            async def scoped_detail(moodboard_id: int, **kwargs: object) -> dict | None:
                return await catalog_get_moodboard(
                    moodboard_id, **kwargs, db_path=db_path
                )

            async def scoped_discovery(**kwargs: object) -> dict:
                return await catalog_latest_discovery(**kwargs, db_path=db_path)

            sessions = {
                "alice-token": ("alice", time.time() + 3600),
                "bob-token": ("bob", time.time() + 3600),
            }
            with (
                patch.object(main, "SHARE_AUTH_ENABLED", True),
                patch.object(main, "_share_sessions", sessions),
                patch.object(main, "is_valid_username", return_value=True),
                patch.object(main, "get_user_role", return_value="user"),
                patch.object(main, "list_moodboards", side_effect=scoped_list),
                patch.object(main, "get_moodboard", side_effect=scoped_detail),
                patch.object(
                    main, "latest_moodboard_discovery", side_effect=scoped_discovery
                ),
            ):
                alice_client = TestClient(
                    main.app, cookies={main.SHARE_COOKIE: "alice-token"}
                )
                bob_client = TestClient(
                    main.app, cookies={main.SHARE_COOKIE: "bob-token"}
                )
                anonymous_client = TestClient(main.app)
                invalid_client = TestClient(
                    main.app, cookies={main.SHARE_COOKIE: "invalid-token"}
                )
                alice = alice_client.get("/api/moodboards?favorites=true")
                bob = bob_client.get("/api/moodboards?favorites=true")
                anonymous = anonymous_client.get("/api/moodboards?favorites=true")
                invalid = invalid_client.get("/api/moodboards?favorites=true")
                alice_detail = alice_client.get(f"/api/moodboards/{alice_board}")
                bob_detail = bob_client.get(f"/api/moodboards/{alice_board}")
                anonymous_detail = anonymous_client.get(
                    f"/api/moodboards/{alice_board}"
                )
                invalid_detail = invalid_client.get(
                    f"/api/moodboards/{alice_board}"
                )
                alice_discovery = alice_client.get(
                    "/api/moodboards/discoveries/latest"
                )
                bob_discovery = bob_client.get(
                    "/api/moodboards/discoveries/latest"
                )
                anonymous_discovery = anonymous_client.get(
                    "/api/moodboards/discoveries/latest"
                )
                invalid_discovery = invalid_client.get(
                    "/api/moodboards/discoveries/latest"
                )

                with patch.object(main, "SHARE_AUTH_ENABLED", False):
                    local_list = anonymous_client.get(
                        "/api/moodboards?favorites=true"
                    )
                    local_detail = anonymous_client.get(
                        f"/api/moodboards/{alice_board}"
                    )
                    local_discovery = anonymous_client.get(
                        "/api/moodboards/discoveries/latest"
                    )

            self.assertEqual(
                [item["id"] for item in alice.json()["items"]], [alice_board]
            )
            self.assertEqual(
                [item["id"] for item in bob.json()["items"]], [bob_board]
            )
            self.assertEqual(anonymous.json()["total"], 0)
            self.assertEqual(invalid.json()["total"], 0)
            self.assertTrue(alice_detail.json()["favorite"])
            self.assertFalse(bob_detail.json()["favorite"])
            self.assertFalse(anonymous_detail.json()["favorite"])
            self.assertFalse(invalid_detail.json()["favorite"])
            self.assertEqual(
                [item["favorite"] for item in alice_discovery.json()["items"]],
                [True, False],
            )
            self.assertEqual(
                [item["favorite"] for item in bob_discovery.json()["items"]],
                [False, True],
            )
            self.assertFalse(
                any(item["favorite"] for item in anonymous_discovery.json()["items"])
            )
            self.assertFalse(
                any(item["favorite"] for item in invalid_discovery.json()["items"])
            )
            self.assertEqual(
                [item["id"] for item in local_list.json()["items"]], [alice_board]
            )
            self.assertTrue(local_detail.json()["favorite"])
            self.assertEqual(
                [item["favorite"] for item in local_discovery.json()["items"]],
                [True, False],
            )

    async def test_moodboard_routes_return_catalog_data(self) -> None:
        client = TestClient(main.app)
        list_args: dict[str, object] = {}

        async def fake_list(**kwargs: object) -> dict:
            list_args.update(kwargs)
            return {"items": [MOODBOARD_ITEM], "total": 1}

        async def fake_detail(_: int, **__: object) -> dict:
            return MOODBOARD_ITEM

        async def fake_favorite(_: int, __: bool, **___: object) -> None:
            return None

        async def fake_import(_: list[str], max_pages: int = 200, use_browser_discovery: bool = False) -> dict:
            return {"imported": 1, "ids": [7], "new_count": 1, "new_ids": [7]}

        async def fake_export(_: object) -> int:
            return 1

        async def fake_latest_discovery(**_: object) -> dict:
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
            listed = client.get("/api/moodboards?q=old-school&page=2&page_size=12&source=official")
            detail = client.get("/api/moodboards/7")
            favorite = client.put("/api/moodboards/7/favorite", json={"favorite": True})
            imported = client.post("/api/moodboards/import", json={"urls": [MOODBOARD_ITEM["url"]], "max_pages": 1})
            exported = client.post("/api/moodboards/export-seed")
            image = client.post("/api/moodboards/image", json={"url": "https://optim-images.krea.ai/ref.webp"})
            latest = client.get("/api/moodboards/discoveries/latest")

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["items"][0]["title"], "Gritty Cinematic Realism")
        self.assertEqual(list_args["query"], "old-school")
        self.assertEqual(list_args["page"], 2)
        self.assertEqual(list_args["page_size"], 12)
        self.assertEqual(list_args["source"], "official")
        self.assertEqual(list_args["username"], "__local__")
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
                "image_b64s": [TINY_PNG_B64],
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

    async def test_custom_moodboard_auto_authoring_is_queued(self) -> None:
        client = TestClient(main.app)
        jobs: dict[str, dict] = {}
        queue = GpuTaskQueue(lambda _task_id, _payload: None)
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
        ):
            created = client.post("/api/moodboards/custom", json={
                "title": "",
                "taste_profile": "",
                "keywords": [],
                "image_b64s": [TINY_PNG_B64],
            })

        self.assertEqual(created.status_code, 202)
        self.assertEqual(created.json()["task_kind"], MOODBOARD_GUIDANCE)
        self.assertEqual(jobs[created.json()["job_id"]]["operation"], "custom")

    async def test_qwen_guidance_routes_queue_single_and_missing(self) -> None:
        client = TestClient(main.app)
        jobs: dict[str, dict] = {}
        queue = GpuTaskQueue(lambda _task_id, _payload: None)
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
        ):
            single = client.post("/api/moodboards/7/qwen-guidance")
            missing = client.post("/api/moodboards/qwen-guidance-missing", json={"limit": 5})

        self.assertEqual(single.status_code, 202)
        self.assertEqual(single.json()["task_kind"], MOODBOARD_GUIDANCE)
        self.assertEqual(missing.status_code, 202)
        self.assertEqual(missing.json()["task_kind"], MOODBOARD_GUIDANCE)

    async def test_mashup_route_queues_guidance(self) -> None:
        client = TestClient(main.app)
        jobs: dict[str, dict] = {}
        queue = GpuTaskQueue(lambda _task_id, _payload: None)
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
        ):
            created = client.post("/api/moodboards/mashup", json={"moodboard_ids": [7, 8], "weights": [0.7, 0.3]})

        self.assertEqual(created.status_code, 202)
        self.assertEqual(created.json()["task_kind"], MOODBOARD_GUIDANCE)
        self.assertEqual(jobs[created.json()["job_id"]]["operation"], "mashup")


if __name__ == "__main__":
    unittest.main()
