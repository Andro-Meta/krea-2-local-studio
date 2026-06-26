from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from realtime_jobs import RealtimePreviewRegistry


class RealtimePreviewRegistryTests(unittest.TestCase):
    def test_rejects_stale_results_for_same_session(self) -> None:
        registry = RealtimePreviewRegistry(max_jobs=10)
        first = registry.create("session-a")
        second = registry.create("session-a")

        self.assertEqual(first["revision"], 1)
        self.assertEqual(second["revision"], 2)

        self.assertFalse(registry.complete(first["job_id"], image_b64="old", seed=10))
        self.assertEqual(registry.get(first["job_id"])["status"], "stale")

        self.assertTrue(registry.complete(second["job_id"], image_b64="new", seed=11))
        latest = registry.get(second["job_id"])
        self.assertEqual(latest["status"], "done")
        self.assertEqual(latest["image_b64"], "new")
        self.assertEqual(latest["seed"], 11)

    def test_cancel_marks_job_without_touching_newer_revision(self) -> None:
        registry = RealtimePreviewRegistry(max_jobs=10)
        old = registry.create("session-a")
        new = registry.create("session-a")

        self.assertTrue(registry.cancel(old["job_id"]))
        self.assertEqual(registry.get(old["job_id"])["status"], "cancelled")
        self.assertEqual(registry.get(new["job_id"])["status"], "queued")

    def test_cancelled_current_job_does_not_complete(self) -> None:
        registry = RealtimePreviewRegistry(max_jobs=10)
        job = registry.create("session-a")

        self.assertTrue(registry.cancel(job["job_id"]))
        self.assertFalse(registry.complete(job["job_id"], image_b64="late", seed=10))
        self.assertEqual(registry.get(job["job_id"])["status"], "cancelled")

    def test_evicts_oldest_jobs(self) -> None:
        registry = RealtimePreviewRegistry(max_jobs=2)
        first = registry.create("a")
        second = registry.create("b")
        third = registry.create("c")

        self.assertIsNone(registry.get(first["job_id"]))
        self.assertIsNotNone(registry.get(second["job_id"]))
        self.assertIsNotNone(registry.get(third["job_id"]))


if __name__ == "__main__":
    unittest.main()
