from __future__ import annotations

import sys
import time
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import main  # noqa: E402
from gpu_task_queue import GpuTaskQueue  # noqa: E402
from gpu_tasks import GENERATION, PROMPT_EXPAND  # noqa: E402


@contextmanager
def _authed_api(username: str, jobs: dict, queue: GpuTaskQueue):
    with ExitStack() as stack:
        for patcher in (
            patch.object(main, "SHARE_AUTH_ENABLED", True),
            patch.object(
                main,
                "_share_sessions",
                {"tc": (username, time.time() + 3600)},
            ),
            patch.object(main, "is_valid_username", lambda _u: True),
            patch.object(main, "get_user_role", lambda _p, _u: "user"),
            patch.object(main, "is_admin", lambda _p, _u: False),
            patch.object(
                main,
                "_request_user_role",
                return_value=(username, "user", False),
            ),
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
        ):
            stack.enter_context(patcher)
        yield TestClient(main.app, cookies={main.SHARE_COOKIE: "tc"})


class GpuTaskApiTests(unittest.TestCase):
    def setUp(self) -> None:
        async def noop_handler(_task_id, _payload):
            return None

        self.jobs: dict[str, dict] = {}
        self.queue = GpuTaskQueue(noop_handler)

    def _prefill(self, count: int, *, username: str = "alice") -> None:
        for index in range(count):
            result = self.queue.enqueue(
                f"existing-{username}-{index}",
                {},
                username=username,
                role="user",
                task_kind=GENERATION,
            )
            self.assertTrue(result.accepted)

    def test_generation_below_limit_returns_normal_queue_response(self) -> None:
        with _authed_api("alice", self.jobs, self.queue) as client:
            response = client.post("/api/generate", json={"prompt": "a calm landscape"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "queued")
        self.assertIn("queue_position", payload)
        self.assertEqual(self.jobs[payload["job_id"]]["task_kind"], GENERATION)

    def test_ninth_generation_returns_429_without_orphan_job(self) -> None:
        self._prefill(8)
        before_ids = set(self.jobs)

        with _authed_api("alice", self.jobs, self.queue) as client:
            response = client.post("/api/generate", json={"prompt": "a calm landscape"})

        self.assertEqual(response.status_code, 429)
        self.assertEqual(
            response.json()["detail"],
            {
                "message": "GPU task capacity reached.",
                "reason": "per_user_limit",
                "limit": 8,
                "active_count": 8,
            },
        )
        self.assertEqual(set(self.jobs), before_ids)

    def test_safe_batch_rejection_is_atomic_before_parent_creation(self) -> None:
        self._prefill(7)
        jobs_before = dict(self.jobs)
        queue_before = self.queue.all_statuses()

        with _authed_api("alice", self.jobs, self.queue) as client:
            response = client.post(
                "/api/generate",
                json={
                    "prompt": "a calm landscape",
                    "num_images": 2,
                    "batch_mode": "safe_queue",
                },
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"]["reason"], "per_user_limit")
        self.assertEqual(self.jobs, jobs_before)
        self.assertEqual(self.queue.all_statuses(), queue_before)

    def test_global_sixty_fifth_admission_returns_global_limit(self) -> None:
        for index in range(64):
            result = self.queue.enqueue(
                f"global-{index}",
                {},
                username=f"user-{index}",
                role="user",
                task_kind=GENERATION,
            )
            self.assertTrue(result.accepted)

        with _authed_api("overflow", self.jobs, self.queue) as client:
            response = client.post("/api/generate", json={"prompt": "a calm landscape"})

        self.assertEqual(response.status_code, 429)
        detail = response.json()["detail"]
        self.assertEqual(detail["reason"], "global_limit")
        self.assertEqual(detail["limit"], 64)
        self.assertEqual(detail["active_count"], 64)
        self.assertEqual(self.jobs, {})

    def test_jobs_lists_admission_counts_and_own_task_kind(self) -> None:
        with _authed_api("alice", self.jobs, self.queue) as client:
            submitted = client.post(
                "/api/generate", json={"prompt": "a calm landscape"}
            ).json()
            response = client.get("/api/jobs")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        own = next(job for job in payload["jobs"] if job["job_id"] == submitted["job_id"])
        self.assertEqual(own["task_kind"], GENERATION)
        self.assertEqual(own["priority_class"], "interactive")
        self.assertEqual(
            payload["admission"],
            {
                "per_user_active": 1,
                "per_user_limit": 8,
                "global_interactive_active": 1,
                "global_interactive_limit": 64,
                "global_background_active": 0,
                "global_background_limit": 4,
            },
        )

    def test_foreign_helper_job_is_anonymous_and_content_free(self) -> None:
        self.jobs["secret-helper-id"] = {
            "status": "queued",
            "progress": 10,
            "username": "bob",
            "role": "user",
            "task_kind": PROMPT_EXPAND,
            "priority_class": "interactive",
            "summary": "Expand a confidential prompt",
            "content": "confidential prompt text",
            "queue_position": 1,
            "queue_length": 1,
        }

        with _authed_api("alice", self.jobs, self.queue) as client:
            response = client.get("/api/jobs")

        self.assertEqual(response.status_code, 200)
        foreign = response.json()["jobs"][0]
        self.assertEqual(foreign["summary"], "Another user's helper")
        self.assertFalse(foreign["mine"])
        self.assertNotIn("secret-helper-id", foreign["job_id"])
        self.assertNotIn("username", foreign)
        self.assertNotIn("content", foreign)


if __name__ == "__main__":
    unittest.main()
