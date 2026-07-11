from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import main  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402


class JobOwnershipTests(unittest.TestCase):
    def test_local_mode_owns_everything(self) -> None:
        with patch.object(main, "SHARE_AUTH_ENABLED", False):
            self.assertTrue(main._job_owned_by({"username": "someone"}, None, False))

    def test_admin_owns_everything(self) -> None:
        with patch.object(main, "SHARE_AUTH_ENABLED", True):
            self.assertTrue(main._job_owned_by({"username": "bob"}, "alice", True))

    def test_user_owns_only_their_jobs(self) -> None:
        with patch.object(main, "SHARE_AUTH_ENABLED", True):
            self.assertTrue(main._job_owned_by({"username": "alice"}, "alice", False))
            self.assertFalse(main._job_owned_by({"username": "bob"}, "alice", False))
            self.assertFalse(main._job_owned_by({"username": None}, "alice", False))


from contextlib import ExitStack, contextmanager  # noqa: E402


@contextmanager
def _authed(username: str, role: str, is_admin: bool, jobs: dict):
    """Simulate an authenticated share user (stubbed cookie session) so the
    middleware passes while ownership logic runs with SHARE_AUTH_ENABLED."""
    import time as _time

    with ExitStack() as stack:
        for patcher in (
            patch.object(main, "SHARE_AUTH_ENABLED", True),
            patch.object(main, "_share_sessions", {"tc": (username, _time.time() + 3600)}),
            patch.object(main, "is_valid_username", lambda _u: True),
            patch.object(main, "get_user_role", lambda _p, _u: role),
            patch.object(main, "is_admin", lambda _p, _u: is_admin),
            patch.object(main, "_request_user_role", return_value=(username, role, is_admin)),
            patch.object(main, "_sync_queue_state_to_jobs", lambda: None),
            patch.object(main, "_jobs", jobs),
        ):
            stack.enter_context(patcher)
        yield


class AdminGateTests(unittest.TestCase):
    def test_heavy_setup_and_install_endpoints_are_admin_only(self) -> None:
        for path, method in [
            ("/api/civitai/install", "POST"),
            ("/api/xperiment/setup", "POST"),
            ("/api/gguf/setup-low-vram", "POST"),
            ("/api/int8/setup-native", "POST"),
            ("/api/quality-assets/some_asset/download", "POST"),
            ("/api/moodboards/custom/5", "DELETE"),
        ]:
            self.assertTrue(main._requires_admin(path, method), f"{method} {path} should be admin-only")

    def test_read_paths_stay_open_to_users(self) -> None:
        for path, method in [
            ("/api/quality-assets/status", "GET"),
            ("/api/moodboards/custom", "POST"),  # creating boards stays open
            ("/api/generate", "POST"),
            ("/api/jobs", "GET"),
        ]:
            self.assertFalse(main._requires_admin(path, method), f"{method} {path} should NOT be admin-only")


class JobsListPrivacyTests(unittest.IsolatedAsyncioTestCase):
    async def test_ws_terminal_delivery_allows_ack(self) -> None:
        from fastapi.testclient import TestClient

        jobs = {
            "owned": {
                "status": "running",
                "progress": 50,
                "images": [],
                "result": None,
                "username": "alice",
            },
        }
        manager = main.WSManager()
        with _authed("alice", "user", False, jobs), patch.object(
            main, "ws_manager", manager
        ):
            client = TestClient(main.app, cookies={main.SHARE_COOKIE: "tc"})
            with client.websocket_connect("/ws/owned") as websocket:
                self.assertEqual(websocket.receive_json()["type"], "init")
                jobs["owned"].update(
                    {
                        "status": "done",
                        "progress": 100,
                        "images": ["data:image/png;base64,large"],
                        "result": {"image_b64": "data:image/png;base64,large"},
                    }
                )
                payload = {
                    "type": "done",
                    "status": "done",
                    "images": jobs["owned"]["images"],
                    "result": jobs["owned"]["result"],
                }
                delivered = websocket.portal.call(
                    main._broadcast_job_event, "owned", payload
                )
                received = websocket.receive_json()
            ack = client.post("/api/generate/owned/ack")

        self.assertEqual(delivered, 1)
        self.assertEqual(received, payload)
        self.assertEqual(ack.status_code, 200)

    async def test_zero_ws_subscribers_requires_get_before_ack(self) -> None:
        from fastapi.testclient import TestClient

        jobs = {
            "owned": {
                "status": "done",
                "progress": 100,
                "images": ["data:image/png;base64,large"],
                "result": {"image_b64": "data:image/png;base64,large"},
                "username": "alice",
            },
        }
        manager = main.WSManager()
        payload = {
            "type": "done",
            "status": "done",
            "images": jobs["owned"]["images"],
            "result": jobs["owned"]["result"],
        }
        with _authed("alice", "user", False, jobs), patch.object(
            main, "ws_manager", manager
        ):
            delivered = await main._broadcast_job_event("owned", payload)
            client = TestClient(main.app, cookies={main.SHARE_COOKIE: "tc"})
            early = client.post("/api/generate/owned/ack")
            fetched = client.get("/api/generate/owned")
            ack = client.post("/api/generate/owned/ack")

        self.assertEqual(delivered, 0)
        self.assertEqual(early.status_code, 409)
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(ack.status_code, 200)

    async def test_ws_missing_and_foreign_jobs_close_with_policy_code(self) -> None:
        from fastapi.testclient import TestClient

        jobs = {
            "foreign": {
                "status": "queued",
                "progress": 0,
                "images": [],
                "username": "bob",
            },
        }
        with _authed("alice", "user", False, jobs):
            client = TestClient(main.app, cookies={main.SHARE_COOKIE: "tc"})
            for job_id in ("missing", "foreign"):
                with self.assertRaises(WebSocketDisconnect) as caught:
                    with client.websocket_connect(f"/ws/{job_id}") as websocket:
                        websocket.receive_json()
                self.assertEqual(caught.exception.code, 1008)

    async def test_ws_invalid_auth_closes_with_policy_code(self) -> None:
        from fastapi.testclient import TestClient

        with (
            patch.object(main, "SHARE_AUTH_ENABLED", True),
            patch.object(main, "_share_sessions", {}),
            patch.object(main, "_jobs", {}),
        ):
            client = TestClient(main.app)
            with self.assertRaises(WebSocketDisconnect) as caught:
                with client.websocket_connect("/ws/anything") as websocket:
                    websocket.receive_json()
        self.assertEqual(caught.exception.code, 1008)

    async def test_ws_policy_rejection_accepts_before_close(self) -> None:
        class Socket:
            scope = {"path": "/ws/missing", "root_path": ""}
            cookies = {}

            def __init__(self):
                self.events = []

            async def accept(self):
                self.events.append(("accept", None))

            async def close(self, code):
                self.events.append(("close", code))

        socket = Socket()
        with (
            patch.object(main, "SHARE_AUTH_ENABLED", True),
            patch.object(main, "_share_sessions", {}),
        ):
            await main.ws_endpoint(socket, "missing")

        self.assertEqual(socket.events, [("accept", None), ("close", 1008)])

    async def test_ws_missing_local_job_also_policy_closes(self) -> None:
        class Socket:
            scope = {"path": "/ws/missing", "root_path": ""}
            cookies = {}

            def __init__(self):
                self.events = []

            async def accept(self):
                self.events.append(("accept", None))

            async def close(self, code):
                self.events.append(("close", code))

            async def receive_text(self):
                raise WebSocketDisconnect()

        socket = Socket()
        with (
            patch.object(main, "SHARE_AUTH_ENABLED", False),
            patch.object(main, "_jobs", {}),
            patch.object(main, "ws_manager", main.WSManager()),
        ):
            await main.ws_endpoint(socket, "missing")

        self.assertEqual(socket.events, [("accept", None), ("close", 1008)])

    async def test_foreign_jobs_are_anonymized_without_usernames(self) -> None:
        from fastapi.testclient import TestClient

        jobs = {
            "aaaa1111aaaa1111": {
                "status": "queued", "progress": 0, "images": [], "error": None, "seed": None,
                "username": "alice", "role": "user", "summary": "Turbo · 1024×1024",
                "queue_position": 1, "queue_length": 2, "thumb": "",
            },
            "bbbb2222bbbb2222": {
                "status": "queued", "progress": 0, "images": [], "error": None, "seed": 42,
                "username": "bob", "role": "user", "summary": "RAW · 2048×2048 secret settings",
                "queue_position": 2, "queue_length": 2, "thumb": "",
            },
            "cccc3333cccc3333": {
                "status": "done", "progress": 100, "images": ["zzz"], "error": None, "seed": 7,
                "username": "bob", "role": "user", "summary": "Turbo · done job",
                "queue_position": None, "queue_length": 0, "thumb": "",
            },
        }
        with _authed("alice", "user", False, jobs):
            client = TestClient(main.app, cookies={main.SHARE_COOKIE: "tc"})
            payload = client.get("/api/jobs").json()["jobs"]

        by_summary = {item["summary"]: item for item in payload}
        # Own job: full detail, usable id.
        mine = by_summary["Turbo · 1024×1024"]
        self.assertTrue(mine["mine"])
        self.assertEqual(mine["job_id"], "aaaa1111aaaa1111")
        # Foreign queued job: anonymized (no settings, no usable id, no seed).
        self.assertIn("Another user's generation", by_summary)
        other = by_summary["Another user's generation"]
        self.assertFalse(other["mine"])
        self.assertTrue(other["job_id"].startswith("anon-"))
        self.assertNotIn("bbbb2222bbbb2222", other["job_id"])
        self.assertIsNone(other["seed"])
        self.assertEqual(other["thumb"], "")
        self.assertEqual(other["queue_position"], 2)
        # Foreign finished job: hidden entirely.
        self.assertNotIn("Turbo · done job", by_summary)
        # No username fields anywhere in the response.
        for item in payload:
            self.assertNotIn("username", item)

    async def test_job_status_and_cancel_hide_foreign_jobs(self) -> None:
        from fastapi.testclient import TestClient

        jobs = {
            "bbbb2222bbbb2222": {
                "status": "queued", "progress": 0, "images": ["secret"], "error": None,
                "seed": 42, "username": "bob", "role": "user", "summary": "RAW",
            },
        }
        with _authed("alice", "user", False, jobs):
            client = TestClient(main.app, cookies={main.SHARE_COOKIE: "tc"})
            status = client.get("/api/generate/bbbb2222bbbb2222")
            cancel = client.post("/api/generate/bbbb2222bbbb2222/cancel")

        self.assertEqual(status.status_code, 404)
        self.assertEqual(cancel.status_code, 404)

    async def test_ack_releases_only_terminal_owned_large_payloads(self) -> None:
        from fastapi.testclient import TestClient

        jobs = {
            "owned": {
                "status": "done",
                "images": ["data:image/png;base64,large"],
                "result": {
                    "image_b64": "data:image/png;base64,large",
                    "metadata": {"method": "upscale"},
                },
                "metadata": [{"seed": 1}],
                "summary": "Upscale",
                "username": "alice",
                "queued_at": 1.0,
                "started_at": 2.0,
                "finished_at": 3.0,
            },
            "running": {
                "status": "running",
                "images": ["keep"],
                "result": {"image_b64": "keep"},
                "username": "alice",
            },
            "foreign": {
                "status": "done",
                "images": ["secret"],
                "result": {"image_b64": "secret"},
                "username": "bob",
            },
        }
        with _authed("alice", "user", False, jobs):
            client = TestClient(main.app, cookies={main.SHARE_COOKIE: "tc"})
            early = client.post("/api/generate/owned/ack")
            delivered = client.get("/api/generate/owned")
            ack = client.post("/api/generate/owned/ack")
            running = client.post("/api/generate/running/ack")
            foreign = client.post("/api/generate/foreign/ack")

        self.assertEqual(early.status_code, 409)
        self.assertEqual(
            delivered.json()["result"]["image_b64"],
            "data:image/png;base64,large",
        )
        self.assertEqual(ack.status_code, 200)
        self.assertEqual(jobs["owned"]["images"], [])
        self.assertNotIn("image_b64", jobs["owned"]["result"])
        self.assertEqual(jobs["owned"]["result"]["metadata"], {"method": "upscale"})
        self.assertEqual(jobs["owned"]["metadata"], [{"seed": 1}])
        self.assertEqual(jobs["owned"]["summary"], "Upscale")
        self.assertEqual(jobs["owned"]["finished_at"], 3.0)
        self.assertEqual(running.status_code, 409)
        self.assertEqual(jobs["running"]["images"], ["keep"])
        self.assertEqual(foreign.status_code, 404)
        self.assertEqual(jobs["foreign"]["images"], ["secret"])

    async def test_parent_ack_releases_terminal_children_without_repopulation(self) -> None:
        from fastapi.testclient import TestClient

        jobs = {
            "parent": {
                "status": "done",
                "images": ["data:image/png;base64,parent"],
                "result": {"images": ["data:image/png;base64,parent"]},
                "thumb": "data:image/jpeg;base64,parent-thumb",
                "metadata": [{"seed": 11}],
                "child_job_ids": ["done-child", "cancelled-child", "running-child"],
                "username": "alice",
                "summary": "Safe batch",
            },
            "done-child": {
                "status": "done",
                "images": ["data:image/png;base64,child"],
                "result": {
                    "image_b64": "data:image/png;base64,child",
                    "metadata": {"seed": 11},
                },
                "thumb": "data:image/jpeg;base64,child-thumb",
                "metadata": [{"seed": 11}],
                "username": "alice",
                "parent_job_id": "parent",
            },
            "cancelled-child": {
                "status": "cancelled",
                "images": ["stale-cancelled-payload"],
                "result": {"image_b64": "stale-cancelled-payload"},
                "username": "alice",
                "parent_job_id": "parent",
            },
            "running-child": {
                "status": "running",
                "images": ["must-remain"],
                "result": {"image_b64": "must-remain"},
                "username": "alice",
                "parent_job_id": "parent",
            },
        }
        with _authed("alice", "user", False, jobs):
            client = TestClient(main.app, cookies={main.SHARE_COOKIE: "tc"})
            delivered = client.get("/api/generate/parent")
            ack = client.post("/api/generate/parent/ack")
            polled = client.get("/api/generate/parent")
            listed_once = client.get("/api/jobs")
            listed_twice = client.get("/api/jobs")

        self.assertEqual(delivered.status_code, 200)
        self.assertEqual(ack.status_code, 200)
        self.assertEqual(polled.json()["images"], [])
        self.assertNotIn("images", polled.json().get("result") or {})
        self.assertNotIn("thumb", jobs["parent"])
        self.assertEqual(jobs["parent"]["completed_count"], 1)
        self.assertEqual(jobs["parent"]["num_images"], 1)
        for child_id in ("done-child", "cancelled-child"):
            child = jobs[child_id]
            self.assertEqual(child["images"], [])
            self.assertNotIn("image_b64", child["result"])
            self.assertIn("result_delivered_at", child)
            self.assertIn("result_acknowledged_at", child)
        self.assertEqual(jobs["running-child"]["images"], ["must-remain"])
        self.assertEqual(
            jobs["running-child"]["result"]["image_b64"], "must-remain"
        )
        for payload in (listed_once.json(), listed_twice.json()):
            parent = next(job for job in payload["jobs"] if job["job_id"] == "parent")
            self.assertEqual(parent["num_images"], 1)
        self.assertEqual(jobs["parent"]["images"], [])
        self.assertEqual(jobs["done-child"]["images"], [])

    async def test_admin_sees_full_queue(self) -> None:
        from fastapi.testclient import TestClient

        jobs = {
            "bbbb2222bbbb2222": {
                "status": "queued", "progress": 0, "images": [], "error": None,
                "seed": None, "username": "bob", "role": "user", "summary": "RAW · 2048",
                "queue_position": 1, "queue_length": 1, "thumb": "",
            },
        }
        with _authed("root", "admin", True, jobs):
            client = TestClient(main.app, cookies={main.SHARE_COOKIE: "tc"})
            payload = client.get("/api/jobs").json()["jobs"]

        self.assertEqual(payload[0]["summary"], "RAW · 2048")
        self.assertTrue(payload[0]["mine"])


if __name__ == "__main__":
    unittest.main()
