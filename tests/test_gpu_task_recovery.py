from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import main  # noqa: E402
from support import mock_atomic_cancel_capability  # noqa: E402

mock_atomic_cancel_capability(main)
import prompt_recipes  # noqa: E402


class BatchRecoveryTests(unittest.TestCase):
    def test_cancelled_parent_does_not_regress_during_poll_refresh(self):
        jobs = {
            "parent": {
                "status": "cancelled",
                "child_job_ids": ["done", "cancelled"],
                "cancel_requested": True,
            },
            "done": {"status": "done", "images": ["image"], "metadata": [{"seed": 1}]},
            "cancelled": {"status": "cancelled"},
        }
        with patch.object(main, "_jobs", jobs):
            refreshed = main._refresh_parent_batch_job("parent")
            main._refresh_parent_batch_job("parent")

        self.assertEqual(refreshed["status"], "cancelled")
        self.assertEqual(refreshed["images"], ["image"])
        self.assertEqual(refreshed["metadata"], [{"seed": 1}])

    def test_parent_queue_projection_uses_child_queue_state(self):
        jobs = {
            "parent": {"status": "queued", "child_job_ids": ["a", "b"]},
            "a": {"status": "queued", "queue_position": 3, "queue_length": 7},
            "b": {"status": "queued", "queue_position": 5, "queue_length": 7},
        }
        with patch.object(main, "_jobs", jobs):
            refreshed = main._refresh_parent_batch_job("parent")

        self.assertEqual(refreshed["queue_position"], 3)
        self.assertEqual(refreshed["queue_length"], 7)

    def test_parent_cancel_does_not_send_completed_child_to_queue(self):
        jobs = {
            "parent": {
                "status": "running",
                "username": "alice",
                "child_job_ids": ["done", "queued"],
            },
            "done": {"status": "done", "images": ["kept"], "metadata": [{}]},
            "queued": {"status": "queued"},
        }
        queue = Mock()
        queue.request_cancel.return_value = "dequeued"
        queue.all_statuses.return_value = {}
        request = SimpleNamespace()
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "_request_user_role", return_value=("alice", "user", False)),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            asyncio.run(main.cancel_generation_job("parent", request))

        queue.request_cancel.assert_called_once_with("queued")
        self.assertEqual(jobs["parent"]["images"], ["kept"])

    def test_mixed_done_cancelled_batch_becomes_cancelled_through_list_poll(self):
        jobs = {
            "parent": {
                "status": "running",
                "username": "alice",
                "child_job_ids": ["done", "cancelled-1", "cancelled-2"],
            },
            "done": {
                "status": "done",
                "images": ["completed-image"],
                "metadata": [{"seed": 7}],
            },
            "cancelled-1": {"status": "cancelled"},
            "cancelled-2": {"status": "cancelled"},
        }
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", None),
            patch.object(main, "_request_user_role", return_value=("alice", "user", False)),
        ):
            first = main._refresh_parent_batch_job("parent")
            listed = asyncio.run(main.list_jobs(SimpleNamespace()))
            second = main._refresh_parent_batch_job("parent")

        self.assertEqual(first["status"], "cancelled")
        self.assertEqual(second["status"], "cancelled")
        self.assertEqual(jobs["parent"]["images"], ["completed-image"])
        self.assertEqual(jobs["parent"]["metadata"], [{"seed": 7}])
        parent_snapshot = next(
            job for job in listed["jobs"] if job["job_id"] == "parent"
        )
        self.assertEqual(parent_snapshot["status"], "cancelled")
        self.assertEqual(parent_snapshot["num_images"], 1)

    def test_error_or_blocked_waits_for_running_sibling_before_terminal(self):
        for failed_status in ("error", "blocked"):
            with self.subTest(failed_status=failed_status):
                jobs = {
                    "parent": {
                        "status": "running",
                        "child_job_ids": ["failed", "sibling"],
                    },
                    "failed": {
                        "status": failed_status,
                        "error": f"{failed_status} detail",
                    },
                    "sibling": {"status": "running"},
                }
                with (
                    patch.object(main, "_jobs", jobs),
                    patch.object(main, "generation_queue", None),
                ):
                    pending = main._refresh_parent_batch_job("parent")
                    self.assertEqual(pending["status"], "running")
                    self.assertNotIn("error", pending)
                    jobs["sibling"]["status"] = "done"
                    terminal = main._refresh_parent_batch_job("parent")

                self.assertEqual(terminal["status"], failed_status)
                self.assertEqual(terminal["error"], f"{failed_status} detail")


class OomRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.req = main.GenerationRequest(prompt="same", width=512, height=512)
        self.jobs = {
            "job": {
                "status": "queued",
                "role": "user",
                "task_kind": "generation",
                "images": [],
                "metadata": [],
            }
        }

    async def _run(self, effects, *, cancelled=False, initial_status="queued"):
        self.jobs["job"]["status"] = initial_status
        queue = Mock()
        queue.cancel_requested.return_value = cancelled
        queue.begin_finalizing.return_value = initial_status != "finalizing"
        calls = []

        def generate(req, **_kwargs):
            calls.append(deepcopy(req.model_dump()))
            effect = effects[len(calls) - 1]
            if callable(effect):
                return effect()
            if isinstance(effect, BaseException):
                raise effect
            return effect

        success = (["image"], 1, [], [], [])
        with (
            patch.object(main, "_jobs", self.jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "use_comfy_backend", return_value=True),
            patch.object(main, "write_generation_breadcrumb"),
            patch.object(main, "clear_generation_breadcrumb"),
            patch.object(main, "free_comfy_vram", return_value=True) as free,
            patch.object(main, "save_image", new=AsyncMock()),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
            patch("comfy_workflows.comfy_generate", side_effect=generate),
        ):
            await main._run_generation("job", self.req)
        return calls, free, success

    async def test_cuda_oom_retries_once_with_identical_payload_and_frees_vram(self):
        success = (["image"], 1, [], [], [])
        calls, free, _ = await self._run(
            [RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), success]
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        free.assert_called_once_with(unload_models=True, free_memory=True)
        self.assertEqual(self.jobs["job"]["status"], "done")
        self.assertEqual(self.jobs["job"]["_recovery"]["oom_attempts"], 1)

    async def test_second_oom_is_terminal_error(self):
        calls, free, _ = await self._run(
            [RuntimeError("CUDA out of memory"), RuntimeError("CUDA out of memory")]
        )
        self.assertEqual(len(calls), 2)
        free.assert_called_once()
        self.assertEqual(self.jobs["job"]["status"], "error")

    async def test_non_oom_runtime_error_is_not_retried(self):
        calls, free, _ = await self._run([RuntimeError("network disconnected")])
        self.assertEqual(len(calls), 1)
        free.assert_not_called()

    async def test_cancellation_after_oom_prevents_retry(self):
        queue = Mock()
        queue.cancel_requested.side_effect = [True, True]
        with patch.object(main, "generation_queue", queue):
            calls, free, _ = await self._run(
                [RuntimeError("CUDA out of memory")], cancelled=True
            )
        self.assertEqual(len(calls), 1)
        free.assert_not_called()
        self.assertEqual(self.jobs["job"]["status"], "cancelled")

    async def test_finalizing_job_does_not_retry(self):
        def fail_during_finalizing():
            self.jobs["job"]["status"] = "finalizing"
            raise RuntimeError("CUDA out of memory")

        calls, free, _ = await self._run([fail_during_finalizing])
        self.assertEqual(len(calls), 1)
        free.assert_not_called()

    async def test_cancellation_during_vram_cleanup_prevents_second_attempt(self):
        cleanup_started = threading.Event()
        cleanup_release = threading.Event()
        cancelled = threading.Event()
        calls = []
        queue = Mock()
        queue.cancel_requested.side_effect = lambda _job_id: cancelled.is_set()
        queue.begin_finalizing.return_value = False

        def generate(req, **_kwargs):
            calls.append(deepcopy(req.model_dump()))
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory")
            return (["unexpected"], 1, [], [], [])

        def free_vram(**_kwargs):
            cleanup_started.set()
            self.assertTrue(cleanup_release.wait(timeout=2))
            return True

        def cancel_during_cleanup():
            self.assertTrue(cleanup_started.wait(timeout=2))
            cancelled.set()
            cleanup_release.set()

        controller = threading.Thread(target=cancel_during_cleanup)
        controller.start()
        try:
            with (
                patch.object(main, "_jobs", self.jobs),
                patch.object(main, "generation_queue", queue),
                patch.object(main, "use_comfy_backend", return_value=True),
                patch.object(main, "write_generation_breadcrumb"),
                patch.object(main, "clear_generation_breadcrumb"),
                patch.object(main, "free_comfy_vram", side_effect=free_vram),
                patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
                patch("comfy_workflows.comfy_generate", side_effect=generate),
            ):
                await main._run_generation("job", self.req)
        finally:
            cleanup_release.set()
            controller.join(timeout=2)

        self.assertEqual(len(calls), 1)
        self.assertEqual(self.jobs["job"]["status"], "cancelled")

    async def test_failed_attempt_removes_only_its_tracked_normal_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)
            preexisting = outputs / "preexisting.png"
            other_job = outputs / "other-job.png"
            preexisting.write_bytes(b"keep")
            other_job.write_bytes(b"keep")
            calls = 0

            def generate(_req, **kwargs):
                nonlocal calls
                calls += 1
                callback = kwargs["output_file_cb"]
                if calls == 1:
                    failed = outputs / "failed-attempt.png"
                    failed.write_bytes(b"partial encode success")
                    callback("failed-attempt.png")
                    raise RuntimeError("CUDA out of memory")
                self.assertFalse((outputs / "failed-attempt.png").exists())
                successful = outputs / "successful-retry.png"
                successful.write_bytes(b"success")
                callback("successful-retry.png")
                return (["image"], 1, ["successful-retry.png"], [], [{}])

            queue = Mock()
            queue.cancel_requested.return_value = False
            queue.begin_finalizing.return_value = True
            with (
                patch.object(main, "_jobs", self.jobs),
                patch.object(main, "generation_queue", queue),
                patch.object(main, "OUTPUTS_DIR", outputs),
                patch.object(main, "use_comfy_backend", return_value=True),
                patch.object(main, "write_generation_breadcrumb"),
                patch.object(main, "clear_generation_breadcrumb"),
                patch.object(main, "free_comfy_vram", return_value=True),
                patch.object(main, "save_image", new=AsyncMock()),
                patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
                patch("comfy_workflows.comfy_generate", side_effect=generate),
            ):
                await main._run_generation("job", self.req)

            self.assertTrue(preexisting.exists())
            self.assertTrue(other_job.exists())
            self.assertFalse((outputs / "failed-attempt.png").exists())
            self.assertTrue((outputs / "successful-retry.png").exists())


class SharedGpuOomRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_gpu_task_families_use_one_identical_retry(self):
        for task_kind in (
            "generation",
            "prompt_expand",
            "upscale",
            "depth_preview",
            "moodboard_guidance",
            "model_warmup",
        ):
            with self.subTest(task_kind=task_kind):
                calls = []

                async def operation():
                    calls.append({"kind": task_kind, "payload": {"same": True}})
                    if len(calls) == 1:
                        raise RuntimeError("CUDA out of memory")
                    return "ok"

                jobs = {"task": {"status": "running", "task_kind": task_kind}}
                queue = Mock()
                queue.cancel_requested.return_value = False
                with (
                    patch.object(main, "_jobs", jobs),
                    patch.object(main, "generation_queue", queue),
                    patch.object(main, "free_comfy_vram", return_value=True) as free,
                ):
                    result = await main._run_gpu_operation_with_oom_retry(
                        "task", operation
                    )
                self.assertEqual(result, "ok")
                self.assertEqual(calls[0], calls[1])
                free.assert_called_once_with(unload_models=True, free_memory=True)
                self.assertEqual(jobs["task"]["_recovery"]["oom_attempts"], 1)

    async def test_non_cuda_oom_and_cancellation_do_not_retry(self):
        for exc in (
            RuntimeError("CPU out of memory"),
            RuntimeError("CUDA out of memory"),
        ):
            calls = 0

            async def operation():
                nonlocal calls
                calls += 1
                raise exc

            jobs = {"task": {"status": "running", "task_kind": "upscale"}}
            queue = Mock()
            queue.cancel_requested.return_value = "CUDA" in str(exc)
            with (
                patch.object(main, "_jobs", jobs),
                patch.object(main, "generation_queue", queue),
                patch.object(main, "free_comfy_vram") as free,
                self.assertRaises(RuntimeError),
            ):
                await main._run_gpu_operation_with_oom_retry("task", operation)
            self.assertEqual(calls, 1)
            free.assert_not_called()


class PromptRecipeAtomicityTests(unittest.TestCase):
    def test_concurrent_users_both_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.json"
            barrier = threading.Barrier(2)

            def save(user):
                barrier.wait()
                prompt_recipes.save_recipe(
                    {"id": user, "name": user}, path=path, username=user
                )

            threads = [threading.Thread(target=save, args=(user,)) for user in ("alice", "bob")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            items = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual({item["owner"] for item in items}, {"alice", "bob"})

    def test_failed_replace_preserves_original_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.json"
            original = [{"id": "original", "name": "Original"}]
            path.write_text(json.dumps(original), encoding="utf-8")
            with (
                patch.object(prompt_recipes.os, "replace", side_effect=OSError("no")),
                self.assertRaises(OSError),
            ):
                prompt_recipes.save_recipe({"id": "new", "name": "New"}, path=path)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), original)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_failed_temp_write_preserves_original_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.json"
            original = [{"id": "original", "name": "Original"}]
            path.write_text(json.dumps(original), encoding="utf-8")
            with (
                patch.object(prompt_recipes.json, "dump", side_effect=OSError("disk full")),
                self.assertRaises(OSError),
            ):
                prompt_recipes.save_recipe({"id": "new", "name": "New"}, path=path)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), original)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
