from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import main  # noqa: E402
from animation_state import AnimationStore  # noqa: E402
from animation_uploads import AnimationUploadStore  # noqa: E402
from gpu_task_queue import GpuTaskQueue  # noqa: E402
from schemas import AnimateRequest  # noqa: E402


class AnimationQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = AnimationStore(root / "state", root / "outputs")

        async def no_op(_job_id, _payload):
            return None

        self.queue = GpuTaskQueue(no_op, enforce_limits=False)
        self.jobs: dict[str, dict] = {}
        self.req = AnimateRequest(render_frames=17, width=256, height=256)

    async def asyncTearDown(self):
        self.temporary.cleanup()

    async def test_create_has_one_parent_and_one_chunk(self):
        with (
            patch.object(main, "_jobs", self.jobs),
            patch.object(main, "generation_queue", self.queue),
            patch.object(main, "animation_store", self.store),
            patch.object(main, "krea_deforum_status", return_value={"available": True}),
            patch.object(main, "comfy_atomic_cancel_available", return_value=True),
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
        ):
            result = await main._create_animation(self.req, username="alice", role="user")

        parent = self.jobs[result["job_id"]]
        self.assertEqual(parent["task_kind"], "animation")
        self.assertEqual(len(parent["child_job_ids"]), 1)
        self.assertEqual(len(self.queue.all_statuses()), 1)
        self.assertEqual(self.store.load(result["job_id"]).completed_frames, 0)

    async def test_3d_animation_requires_prewarmed_midas(self):
        req = AnimateRequest(
            render_frames=8,
            width=256,
            height=256,
            animation_mode="3D",
        )
        with (
            patch.object(main, "_jobs", self.jobs),
            patch.object(main, "generation_queue", self.queue),
            patch.object(main, "animation_store", self.store),
            patch.object(main, "krea_deforum_status", return_value={
                "available": True,
                "midas_ready": False,
                "midas_reason": "MiDaS setup marker is missing.",
            }),
            patch.object(main, "comfy_atomic_cancel_available", return_value=True),
        ):
            with self.assertRaisesRegex(Exception, "MiDaS setup marker") as raised:
                await main._create_animation(req, username="alice", role="user")

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(self.jobs, {})

    async def test_3d_animation_is_admitted_when_midas_is_ready(self):
        req = AnimateRequest(
            render_frames=8,
            width=256,
            height=256,
            animation_mode="3D",
        )
        with (
            patch.object(main, "_jobs", self.jobs),
            patch.object(main, "generation_queue", self.queue),
            patch.object(main, "animation_store", self.store),
            patch.object(main, "krea_deforum_status", return_value={
                "available": True,
                "midas_ready": True,
                "midas_reason": "Ready marker and cache verified.",
            }),
            patch.object(main, "comfy_atomic_cancel_available", return_value=True),
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
        ):
            result = await main._create_animation(req, username="alice", role="user")

        self.assertIn(result["job_id"], self.jobs)

    async def test_second_active_parent_for_owner_is_rejected(self):
        with (
            patch.object(main, "_jobs", self.jobs),
            patch.object(main, "generation_queue", self.queue),
            patch.object(main, "animation_store", self.store),
            patch.object(main, "krea_deforum_status", return_value={"available": True}),
            patch.object(main, "comfy_atomic_cancel_available", return_value=True),
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
        ):
            await main._create_animation(self.req, username="alice", role="user")
            with self.assertRaisesRegex(Exception, "active animation"):
                await main._create_animation(self.req, username="alice", role="user")

    async def test_committed_chunk_queues_exactly_one_continuation(self):
        with (
            patch.object(main, "_jobs", self.jobs),
            patch.object(main, "generation_queue", self.queue),
            patch.object(main, "animation_store", self.store),
            patch.object(main, "krea_deforum_status", return_value={"available": True}),
            patch.object(main, "comfy_atomic_cancel_available", return_value=True),
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
        ):
            created = await main._create_animation(self.req, username="alice", role="user")
            parent_id = created["job_id"]
            child_id = self.jobs[parent_id]["child_job_ids"][-1]
            self.store.begin_chunk(parent_id, 0)
            staging = self.store.project_dir(parent_id) / "staging"
            for index in range(8):
                Image.new("RGB", (256, 256), (index, 0, 0)).save(
                    staging / f"frame_{index:06d}.png"
                )
            self.store.commit_chunk(
                parent_id, 0, [f"frame_{index:06d}.png" for index in range(8)]
            )
            self.jobs[child_id]["status"] = "done"
            await main._continue_animation(parent_id)

        self.assertEqual(self.store.load(parent_id).next_chunk_index, 1)
        self.assertEqual(len(self.jobs[parent_id]["child_job_ids"]), 2)
        queued = [
            item
            for item in self.queue.all_statuses().values()
            if item["status"] == "queued"
        ]
        self.assertEqual(len(queued), 2)  # original record plus exactly one continuation

    async def test_dispatch_failure_marks_parent_error_and_deletes_owned_upload(self):
        req = AnimateRequest(
            render_frames=8,
            width=256,
            height=256,
            animation_mode="Video Input",
            source_video_upload_id="a" * 32,
        )
        project = self.store.create(req, owner="Alice", role="user", job_id="a" * 32)
        child_id = "b" * 32
        jobs = {
            project.job_id: {
                **main._animation_parent_from_project(project),
                "child_job_ids": [child_id],
            },
            child_id: {
                "status": "queued",
                "task_kind": "animation",
                "parent_job_id": project.job_id,
            },
        }
        uploads = Mock()
        uploads.delete.return_value = True
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "animation_store", self.store),
            patch.object(main, "animation_upload_store", uploads),
            patch.object(main, "comfy_available", return_value=False),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            allowed = await main._validate_comfy_task_dispatch(
                child_id,
                "animation",
                {"operation": "chunk", "parent_job_id": project.job_id},
            )

        self.assertFalse(allowed)
        self.assertEqual(self.store.load(project.job_id).status, "error")
        self.assertEqual(jobs[project.job_id]["status"], "error")
        self.assertEqual(jobs[child_id]["status"], "error")
        uploads.delete.assert_called_once_with(
            req.source_video_upload_id, username="Alice", is_admin=True
        )
        self.assertEqual(self.store.active_for_owner("Alice"), [])

    async def test_recovery_is_singleflight_for_queued_and_running_projects(self):
        for status in ("queued", "running"):
            with self.subTest(status=status):
                root = Path(self.temporary.name) / status
                store = AnimationStore(root / "state", root / "outputs")
                project = store.create(
                    self.req, owner="alice", role="user", job_id=("c" if status == "queued" else "d") * 32
                )
                if status == "running":
                    store.begin_chunk(project.job_id, 0)
                jobs: dict[str, dict] = {}
                queue = GpuTaskQueue(lambda *_: None, enforce_limits=False)
                with (
                    patch.object(main, "_jobs", jobs),
                    patch.object(main, "generation_queue", queue),
                    patch.object(main, "animation_store", store),
                    patch.object(main, "animation_upload_store", Mock(cleanup=Mock(return_value=[]))),
                    patch.object(main, "_animation_recovery_guard", set()),
                ):
                    await asyncio.gather(
                        main._recover_animation_projects(),
                        main._recover_animation_projects(),
                    )
                    await main._recover_animation_projects()

                parent = jobs[project.job_id]
                self.assertEqual(len(parent["child_job_ids"]), 1)
                self.assertEqual(len(queue.all_statuses()), 1)

    async def test_recovery_schedules_one_finalizer(self):
        project = self.store.create(
            AnimateRequest(render_frames=1, width=256, height=256),
            owner="alice",
            role="user",
            job_id="e" * 32,
        )
        self.store.begin_chunk(project.job_id, 0)
        staging = self.store.project_dir(project.job_id) / "staging"
        Image.new("RGB", (256, 256), "red").save(staging / "frame_000000.png")
        self.store.commit_chunk(project.job_id, 0, ["frame_000000.png"])
        jobs: dict[str, dict] = {}
        schedule = Mock()
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", self.queue),
            patch.object(main, "animation_store", self.store),
            patch.object(main, "animation_upload_store", Mock(cleanup=Mock(return_value=[]))),
            patch.object(main, "_animation_recovery_guard", set()),
            patch.object(main, "_schedule_animation_finalizer", schedule),
        ):
            await asyncio.gather(
                main._recover_animation_projects(),
                main._recover_animation_projects(),
            )
            await main._recover_animation_projects()

        schedule.assert_called_once_with(project.job_id)

    async def test_terminal_upload_cleanup_is_exact_owner_and_id(self):
        upload_id = "1" * 32
        project = self.store.create(
            AnimateRequest(
                render_frames=1,
                width=256,
                height=256,
                animation_mode="Video Input",
                source_video_upload_id=upload_id,
            ),
            owner="Alice",
            role="user",
            job_id="f" * 32,
        )
        uploads = Mock()
        uploads.delete.return_value = True
        with patch.object(main, "animation_upload_store", uploads):
            self.assertTrue(await main._cleanup_animation_upload(project))

        uploads.delete.assert_called_once_with(
            upload_id, username="Alice", is_admin=True
        )

    async def test_periodic_upload_cleanup_runs_and_is_cancellable(self):
        uploads = Mock()
        cleanup_called = asyncio.Event()
        loop = asyncio.get_running_loop()

        def cleanup(*_args, **_kwargs):
            loop.call_soon_threadsafe(cleanup_called.set)
            return []

        uploads.cleanup.side_effect = cleanup
        with (
            patch.object(main, "animation_store", Mock(recoverable=Mock(return_value=[]))),
            patch.object(main, "animation_upload_store", uploads),
            patch.object(
                main.settings,
                "animation_upload_cleanup_interval_seconds",
                0.01,
            ),
        ):
            task = asyncio.create_task(main._animation_upload_cleanup_loop())
            await asyncio.wait_for(cleanup_called.wait(), timeout=1.0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertGreaterEqual(uploads.cleanup.call_count, 1)


class AnimationUploadQuotaTests(unittest.TestCase):
    def test_repeated_unused_uploads_hit_quota_and_abort_is_owner_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AnimationUploadStore(
                Path(directory),
                max_bytes=10,
                max_frames=10,
                max_dimension=256,
                ttl_seconds=60,
                max_user_uploads=2,
                max_user_bytes=8,
                max_global_uploads=3,
                max_global_bytes=12,
            )
            first, _ = store.reserve("Alice", 4)
            store.reserve("Alice", 4)
            foreign, foreign_path = store.reserve("alice", 4)
            with self.assertRaisesRegex(ValueError, "quota"):
                store.reserve("Alice", 1)

            self.assertTrue(store.abort(first, username="Alice"))
            self.assertTrue(foreign_path.exists())
            self.assertIsNotNone(store.reservation(foreign))

    def test_ttl_cleanup_frees_quota_without_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AnimationUploadStore(
                Path(directory),
                max_bytes=10,
                max_frames=10,
                max_dimension=256,
                ttl_seconds=1,
                max_user_uploads=1,
                max_user_bytes=10,
                max_global_uploads=1,
                max_global_bytes=10,
            )
            upload_id, _ = store.reserve("alice", 5)
            reservation = store.root / f".reserve-{upload_id}.json"
            payload = __import__("json").loads(reservation.read_text(encoding="utf-8"))
            payload["created_at"] = 0
            reservation.write_text(__import__("json").dumps(payload), encoding="utf-8")

            replacement, _ = store.reserve("alice", 5)

            self.assertNotEqual(replacement, upload_id)
            self.assertIsNone(store.reservation(upload_id))


if __name__ == "__main__":
    unittest.main()
