from __future__ import annotations

import asyncio
import sys
import unittest
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from gpu_task_queue import BACKGROUND, INTERACTIVE, GpuTaskQueue  # noqa: E402


class GpuTaskQueueTests(unittest.IsolatedAsyncioTestCase):
    async def _run_to_idle(self, queue: GpuTaskQueue) -> None:
        worker = asyncio.create_task(queue.run())
        try:
            await asyncio.wait_for(queue.join(), timeout=2)
        finally:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    async def test_single_worker_and_two_rounds_are_user_ordered(self):
        order: list[str] = []
        concurrent = 0
        max_concurrent = 0

        async def handler(task_id, _payload):
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            order.append(task_id)
            await asyncio.sleep(0)
            concurrent -= 1

        queue = GpuTaskQueue(handler)
        for user_index in range(9):
            for task_index in range(2):
                result = queue.enqueue(
                    f"user{user_index}-{task_index}",
                    {},
                    username=f"user{user_index}",
                    role="user",
                )
                self.assertTrue(result.accepted)

        await self._run_to_idle(queue)

        self.assertEqual(max_concurrent, 1)
        self.assertEqual(
            order,
            [f"user{i}-0" for i in range(9)] + [f"user{i}-1" for i in range(9)],
        )

    async def test_second_run_fails_without_starting_another_worker(self):
        release = asyncio.Event()
        started = asyncio.Event()
        concurrent = 0
        max_concurrent = 0

        async def handler(_task_id, _payload):
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            started.set()
            try:
                await release.wait()
            finally:
                concurrent -= 1

        queue = GpuTaskQueue(handler)
        queue.enqueue("first", {}, username="alice", role="user")
        queue.enqueue("second", {}, username="bob", role="user")
        worker = asyncio.create_task(queue.run())
        second_worker = None
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            second_worker = asyncio.create_task(queue.run())
            await asyncio.sleep(0.05)
            if not second_worker.done():
                second_worker.cancel()
                with suppress(asyncio.CancelledError):
                    await second_worker
                self.fail("second run did not raise RuntimeError immediately")
            with self.assertRaisesRegex(RuntimeError, "already running"):
                await second_worker
            self.assertEqual(max_concurrent, 1)
        finally:
            if second_worker is not None and not second_worker.done():
                second_worker.cancel()
                with suppress(asyncio.CancelledError):
                    await second_worker
            release.set()
            await asyncio.wait_for(queue.join(), timeout=2)
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    async def test_finalizing_closes_cancellation_and_completes_done(self):
        finalizing = asyncio.Event()
        release = asyncio.Event()
        queue = None

        async def handler(task_id, _payload):
            self.assertTrue(queue.begin_finalizing(task_id))
            finalizing.set()
            await release.wait()

        queue = GpuTaskQueue(handler)
        queue.enqueue("job", {}, username="alice", role="user")
        worker = asyncio.create_task(queue.run())
        try:
            await asyncio.wait_for(finalizing.wait(), timeout=2)
            self.assertEqual(queue.status("job")["status"], "finalizing")
            self.assertEqual(queue.request_cancel("job"), "none")
            release.set()
            await asyncio.wait_for(queue.join(), timeout=2)
            self.assertEqual(queue.status("job")["status"], "done")
        finally:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    async def test_finalizing_exception_becomes_error(self):
        queue = None

        async def handler(task_id, _payload):
            self.assertTrue(queue.begin_finalizing(task_id))
            raise RuntimeError("persistence failed")

        queue = GpuTaskQueue(handler)
        queue.enqueue("job", {}, username="alice", role="user")
        await self._run_to_idle(queue)

        self.assertEqual(queue.status("job")["status"], "error")
        self.assertEqual(queue.status("job")["error"], "persistence failed")

    async def test_prior_cancel_prevents_finalizing(self):
        started = asyncio.Event()
        proceed = asyncio.Event()
        queue = None

        async def handler(task_id, _payload):
            started.set()
            await proceed.wait()
            self.assertFalse(queue.begin_finalizing(task_id))

        queue = GpuTaskQueue(handler)
        queue.enqueue("job", {}, username="alice", role="user")
        worker = asyncio.create_task(queue.run())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            self.assertEqual(queue.request_cancel("job"), "interrupt")
            proceed.set()
            await asyncio.wait_for(queue.join(), timeout=2)
            self.assertEqual(queue.status("job")["status"], "cancelled")
        finally:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    async def test_fifo_within_one_user(self):
        order: list[str] = []

        async def handler(task_id, _payload):
            order.append(task_id)

        queue = GpuTaskQueue(handler)
        for task_id in ("first", "second", "third"):
            queue.enqueue(task_id, {}, username="alice", role="user")

        await self._run_to_idle(queue)
        self.assertEqual(order, ["first", "second", "third"])

    async def test_background_waits_for_all_interactive_work(self):
        order: list[str] = []

        async def handler(task_id, _payload):
            order.append(task_id)

        queue = GpuTaskQueue(handler)
        queue.enqueue(
            "background-1",
            {},
            username="system",
            role="admin",
            priority_class=BACKGROUND,
        )
        queue.enqueue("interactive-1", {}, username="alice", role="user")
        queue.enqueue(
            "background-2",
            {},
            username="system",
            role="admin",
            priority_class=BACKGROUND,
        )
        queue.enqueue("interactive-2", {}, username="bob", role="user")

        await self._run_to_idle(queue)
        self.assertEqual(
            order,
            ["interactive-1", "interactive-2", "background-1", "background-2"],
        )

    async def test_ninth_unfinished_interactive_task_for_user_is_rejected(self):
        queue = GpuTaskQueue(self._noop_handler)
        for index in range(8):
            self.assertTrue(
                queue.enqueue(
                    f"task-{index}", {}, username="alice", role="user"
                ).accepted
            )

        result = queue.enqueue("task-8", {}, username="alice", role="user")

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "per_user_limit")
        self.assertEqual(result.limit, 8)
        self.assertEqual(result.active_count, 8)
        self.assertIsNone(result.record)
        self.assertEqual(queue.status("task-8")["status"], "unknown")

    async def test_anonymous_limit_counts_only_the_anonymous_lane(self):
        queue = GpuTaskQueue(self._noop_handler)
        for index in range(8):
            queue.enqueue(
                f"alice-{index}", {}, username="alice", role="user"
            )

        result = queue.enqueue(
            "anonymous", {}, username=None, role="admin"
        )

        self.assertTrue(result.accepted)

    async def test_sixty_fifth_global_interactive_task_is_rejected(self):
        queue = GpuTaskQueue(self._noop_handler)
        for index in range(64):
            self.assertTrue(
                queue.enqueue(
                    f"task-{index}",
                    {},
                    username=f"user-{index}",
                    role="user",
                ).accepted
            )

        result = queue.enqueue("task-64", {}, username="overflow", role="user")

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "global_limit")
        self.assertEqual(result.limit, 64)
        self.assertEqual(result.active_count, 64)

    async def test_fifth_background_task_is_rejected(self):
        queue = GpuTaskQueue(self._noop_handler)
        for index in range(4):
            self.assertTrue(
                queue.enqueue(
                    f"background-{index}",
                    {},
                    username="system",
                    role="admin",
                    priority_class=BACKGROUND,
                ).accepted
            )

        result = queue.enqueue(
            "background-4",
            {},
            username="system",
            role="admin",
            priority_class=BACKGROUND,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "global_limit")
        self.assertEqual(result.limit, 4)
        self.assertEqual(result.active_count, 4)

    async def test_queued_cancellation_and_completion_release_capacity(self):
        release = asyncio.Event()
        started = asyncio.Event()

        async def handler(_task_id, _payload):
            started.set()
            await release.wait()

        queue = GpuTaskQueue(handler)
        for index in range(8):
            queue.enqueue(f"task-{index}", {}, username="alice", role="user")

        self.assertTrue(queue.cancel("task-7"))
        replacement = queue.enqueue(
            "cancel-replacement", {}, username="alice", role="user"
        )
        self.assertTrue(replacement.accepted)

        worker = asyncio.create_task(queue.run())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            self.assertFalse(
                queue.enqueue(
                    "while-active", {}, username="alice", role="user"
                ).accepted
            )
            release.set()
            await asyncio.wait_for(queue.join(), timeout=2)
            after_completion = queue.enqueue(
                "after-completion", {}, username="alice", role="user"
            )
            self.assertTrue(after_completion.accepted)
        finally:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    async def test_repeated_queued_cancellation_prunes_terminal_records(self):
        release = asyncio.Event()
        started = asyncio.Event()

        async def handler(_task_id, _payload):
            started.set()
            await release.wait()

        queue = GpuTaskQueue(handler)
        queue.enqueue("active", {}, username="active-user", role="user")
        worker = asyncio.create_task(queue.run())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            for index in range(500):
                task_id = f"cancel-{index}"
                self.assertTrue(
                    queue.enqueue(
                        task_id, {}, username="alice", role="user"
                    ).accepted
                )
                self.assertTrue(queue.cancel(task_id))
            self.assertLessEqual(len(queue.all_statuses()), 201)
        finally:
            release.set()
            await asyncio.wait_for(queue.join(), timeout=2)
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    async def test_projected_positions_match_actual_execution(self):
        order: list[str] = []

        async def handler(task_id, _payload):
            order.append(task_id)

        queue = GpuTaskQueue(handler)
        tasks = (
            ("a1", "alice", INTERACTIVE),
            ("a2", "alice", INTERACTIVE),
            ("b1", "bob", INTERACTIVE),
            ("background", "system", BACKGROUND),
            ("c1", "carol", INTERACTIVE),
            ("b2", "bob", INTERACTIVE),
        )
        for task_id, username, priority in tasks:
            result = queue.enqueue(
                task_id,
                {},
                username=username,
                role="user",
                priority_class=priority,
            )
            self.assertEqual(result.get("queue_position"), queue.status(task_id)["queue_position"])

        projected = [
            task_id
            for task_id, _username, _priority in sorted(
                tasks, key=lambda item: queue.status(item[0])["queue_position"]
            )
        ]

        await self._run_to_idle(queue)
        self.assertEqual(order, projected)

    @staticmethod
    async def _noop_handler(_task_id, _payload):
        return None


if __name__ == "__main__":
    unittest.main()
