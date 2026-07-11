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


class GenerationQueueFairnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_robin_interleaves_users_batches(self):
        """User B's single job should not wait behind all of user A's batch."""
        order: list[str] = []

        async def handler(job_id, _payload):
            order.append(job_id)
            await asyncio.sleep(0)

        q = GenerationQueue(handler)
        # Enqueue while the worker is not running so scheduling is deterministic.
        for jid in ("a1", "a2", "a3", "a4"):
            q.enqueue(jid, {}, username="alice", role="user")
        q.enqueue("b1", {}, username="bob", role="user")
        q.enqueue("b2", {}, username="bob", role="user")

        worker = asyncio.create_task(q.run())
        await asyncio.wait_for(q.join(), timeout=2)
        worker.cancel()

        # Round-robin: alice, bob alternate until bob runs out, then alice drains.
        self.assertEqual(order, ["a1", "b1", "a2", "b2", "a3", "a4"])

    async def test_projected_positions_reflect_round_robin(self):
        async def handler(job_id, _payload):
            await asyncio.sleep(3600)

        q = GenerationQueue(handler)
        for jid in ("a1", "a2", "a3"):
            q.enqueue(jid, {}, username="alice", role="user")
        q.enqueue("b1", {}, username="bob", role="user")

        # No worker running: all four pending. Projected order alternates users.
        positions = {jid: q.status(jid)["queue_position"] for jid in ("a1", "a2", "a3", "b1")}
        self.assertEqual(positions["a1"], 1)
        self.assertEqual(positions["b1"], 2)  # bob's first is 2nd, not 4th
        self.assertEqual(positions["a2"], 3)
        self.assertEqual(positions["a3"], 4)
        self.assertEqual(q.status("b1")["queue_length"], 4)

    async def test_single_user_stays_fifo(self):
        order: list[str] = []

        async def handler(job_id, _payload):
            order.append(job_id)

        q = GenerationQueue(handler)
        for jid in ("j1", "j2", "j3"):
            q.enqueue(jid, {}, username="alice", role="user")
        worker = asyncio.create_task(q.run())
        await asyncio.wait_for(q.join(), timeout=2)
        worker.cancel()

        self.assertEqual(order, ["j1", "j2", "j3"])

    async def test_anonymous_local_jobs_share_one_lane(self):
        """Local mode (username=None) keeps plain FIFO behavior."""
        order: list[str] = []

        async def handler(job_id, _payload):
            order.append(job_id)

        q = GenerationQueue(handler)
        for jid in ("j1", "j2"):
            q.enqueue(jid, {}, username=None, role="admin")
        worker = asyncio.create_task(q.run())
        await asyncio.wait_for(q.join(), timeout=2)
        worker.cancel()

        self.assertEqual(order, ["j1", "j2"])


if __name__ == "__main__":
    unittest.main()
