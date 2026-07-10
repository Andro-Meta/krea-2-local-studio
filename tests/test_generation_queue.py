from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from generation_queue import GenerationQueue  # noqa: E402


class GenerationQueueCancelTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_queued_job_is_dequeued(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(job_id, _payload):
            started.set()
            await release.wait()

        q = GenerationQueue(handler)
        worker = asyncio.create_task(q.run())
        q.enqueue("a", {}, username=None, role="user")
        q.enqueue("b", {}, username=None, role="user")
        await asyncio.wait_for(started.wait(), timeout=2)  # 'a' is running, 'b' queued

        # 'b' is still queued -> dequeued cleanly.
        self.assertEqual(q.request_cancel("b"), "dequeued")
        self.assertEqual(q.status("b")["status"], "cancelled")

        release.set()
        await asyncio.wait_for(q.join(), timeout=2)
        self.assertEqual(q.status("a")["status"], "done")
        worker.cancel()

    async def test_cancel_running_job_flags_interrupt_and_marks_cancelled(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(job_id, _payload):
            started.set()
            await release.wait()
            # Simulate ComfyUI interruption surfacing as an exception.
            raise RuntimeError("ComfyUI execution was interrupted.")

        q = GenerationQueue(handler)
        worker = asyncio.create_task(q.run())
        q.enqueue("a", {}, username=None, role="user")
        await asyncio.wait_for(started.wait(), timeout=2)

        self.assertEqual(q.request_cancel("a"), "interrupt")
        self.assertTrue(q.cancel_requested("a"))

        release.set()
        await asyncio.wait_for(q.join(), timeout=2)
        # Interrupted running job is cancelled, not error.
        self.assertEqual(q.status("a")["status"], "cancelled")
        self.assertFalse(q.cancel_requested("a"))  # flag cleared after finish
        worker.cancel()

    async def test_request_cancel_unknown_job(self):
        async def handler(job_id, _payload):
            return None

        q = GenerationQueue(handler)
        self.assertEqual(q.request_cancel("nope"), "none")


if __name__ == "__main__":
    unittest.main()
