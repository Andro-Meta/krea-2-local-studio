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
