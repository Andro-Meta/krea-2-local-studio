from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class GenerationQueueTests(unittest.TestCase):
    def test_runs_jobs_in_fifo_order_and_updates_positions(self) -> None:
        from generation_queue import GenerationQueue

        async def run() -> None:
            order: list[str] = []
            started = asyncio.Event()
            release_first = asyncio.Event()

            async def handler(job_id: str, _payload: object) -> None:
                order.append(job_id)
                if job_id == "job1":
                    started.set()
                    await release_first.wait()

            queue = GenerationQueue(handler)
            queue.enqueue("job1", object(), username="a", role="user")
            queue.enqueue("job2", object(), username="b", role="user")
            queue.enqueue("job3", object(), username="c", role="child")
            worker = asyncio.create_task(queue.run())
            await started.wait()

            self.assertEqual(queue.status("job1")["status"], "running")
            self.assertEqual(queue.status("job2")["queue_position"], 1)
            self.assertEqual(queue.status("job3")["queue_position"], 2)

            release_first.set()
            await asyncio.wait_for(queue.join(), timeout=2)
            worker.cancel()

            self.assertEqual(order, ["job1", "job2", "job3"])
            self.assertEqual(queue.status("job1")["status"], "done")
            self.assertEqual(queue.status("job2")["status"], "done")
            self.assertEqual(queue.status("job3")["status"], "done")

        asyncio.run(run())

    def test_cancel_queued_job(self) -> None:
        from generation_queue import GenerationQueue

        async def run() -> None:
            order: list[str] = []
            release = asyncio.Event()

            async def handler(job_id: str, _payload: object) -> None:
                order.append(job_id)
                if job_id == "job1":
                    await release.wait()

            queue = GenerationQueue(handler)
            queue.enqueue("job1", object(), username="a", role="user")
            queue.enqueue("job2", object(), username="b", role="child")
            worker = asyncio.create_task(queue.run())
            await asyncio.sleep(0)

            self.assertTrue(queue.cancel("job2"))
            self.assertEqual(queue.status("job2")["status"], "cancelled")

            release.set()
            await asyncio.wait_for(queue.join(), timeout=2)
            worker.cancel()
            self.assertEqual(order, ["job1"])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
