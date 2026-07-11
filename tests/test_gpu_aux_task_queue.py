from __future__ import annotations

import asyncio
import base64
import io
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import main  # noqa: E402
from comfy_client import ComfyExecutionError  # noqa: E402
from gpu_task_queue import BACKGROUND, GpuTaskQueue  # noqa: E402
from gpu_tasks import (  # noqa: E402
    BACKGROUND_ENRICHMENT,
    DEPTH_PREVIEW,
    GENERATION,
    HELPER_BENCHMARK,
    IMAGE_DESCRIBE,
    MODEL_WARMUP,
    MOODBOARD_GUIDANCE,
    PROMPT_EXPAND,
    UPSCALE,
)


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@contextmanager
def _api(jobs: dict, queue: GpuTaskQueue, *, username: str = "alice", role: str = "user"):
    with ExitStack() as stack:
        for patcher in (
            patch.object(main, "SHARE_AUTH_ENABLED", True),
            patch.object(main, "_share_sessions", {"tc": (username, time.time() + 3600)}),
            patch.object(main, "is_valid_username", return_value=True),
            patch.object(main, "get_user_role", return_value=role),
            patch.object(main, "is_admin", return_value=role == "admin"),
            patch.object(main, "_request_user_role", return_value=(username, role, role == "admin")),
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
        ):
            stack.enter_context(patcher)
        yield TestClient(main.app, cookies={main.SHARE_COOKIE: "tc"})


class AuxiliaryEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        async def noop(_task_id, _payload):
            return None

        self.jobs: dict[str, dict] = {}
        self.queue = GpuTaskQueue(noop)

    def test_upscale_and_depth_return_202_without_executing_comfy(self):
        cases = (
            (
                "/api/upscale",
                {"image_b64": _png_b64(), "method": "tiled_vae"},
                UPSCALE,
            ),
            (
                "/api/depth-preview",
                {"image_b64": _png_b64()},
                DEPTH_PREVIEW,
            ),
        )
        with (
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "moderate_images", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "use_comfy_backend", return_value=True),
            patch.object(main, "comfy_available", return_value=True),
            patch("comfy_workflows.comfy_upscale") as upscale,
            patch("comfy_workflows.comfy_depth_preview") as depth,
            _api(self.jobs, self.queue) as client,
        ):
            for path, body, task_kind in cases:
                response = client.post(path, json=body)
                self.assertEqual(response.status_code, 202)
                self.assertEqual(response.json()["task_kind"], task_kind)
        upscale.assert_not_called()
        depth.assert_not_called()

    def test_realesrgan_with_comfy_available_queues_without_executing(self):
        with (
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "moderate_images", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "comfy_available", return_value=True),
            patch("comfy_workflows.comfy_upscale") as upscale,
            patch("upscaler.upscale_realesrgan") as cpu_upscale,
            _api(self.jobs, self.queue) as client,
        ):
            response = client.post(
                "/api/upscale",
                json={"image_b64": _png_b64(), "method": "realesrgan"},
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["task_kind"], UPSCALE)
        self.assertEqual(len(self.queue.all_statuses()), 1)
        upscale.assert_not_called()
        cpu_upscale.assert_not_called()

    def test_realesrgan_without_comfy_still_queues_without_executing(self):
        with (
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "moderate_images", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "comfy_available", return_value=False),
            patch("upscaler.upscale_realesrgan") as cpu_upscale,
            _api(self.jobs, self.queue) as client,
        ):
            response = client.post(
                "/api/upscale",
                json={"image_b64": _png_b64(), "method": "realesrgan", "scale": 2},
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["task_kind"], UPSCALE)
        self.assertEqual(len(self.queue.all_statuses()), 1)
        cpu_upscale.assert_not_called()

    def test_realesrgan_fallback_classification_happens_inside_worker(self):
        with (
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "moderate_images", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "comfy_available", return_value=False),
            patch("upscaler.upscale_realesrgan") as cpu_upscale,
            patch("comfy_workflows.comfy_upscale") as comfy_upscale,
            _api(self.jobs, self.queue) as client,
        ):
            response = client.post(
                "/api/upscale",
                json={"image_b64": _png_b64(), "method": "realesrgan"},
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(len(self.queue.all_statuses()), 1)
        cpu_upscale.assert_not_called()
        comfy_upscale.assert_not_called()

    def test_child_cpu_realesrgan_input_is_moderated_before_queue(self):
        blocked = SimpleNamespace(
            allowed=False,
            event_type="unsafe",
            scores={},
            reason="blocked",
        )
        with (
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "moderate_images", return_value=blocked),
            patch.object(main, "save_moderation_event", new=AsyncMock(return_value=12)),
            patch.object(main, "comfy_available", return_value=False),
            patch("upscaler.b64_to_pil", return_value=Image.new("RGB", (8, 8), "white")) as decode,
            patch("upscaler.upscale_realesrgan") as execute,
            _api(self.jobs, self.queue, username="kid", role="child") as client,
        ):
            response = client.post(
                "/api/upscale",
                json={"image_b64": _png_b64(), "method": "realesrgan"},
            )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.queue.all_statuses(), {})
        decode.assert_called_once()
        execute.assert_not_called()

    def test_child_generation_collects_all_content_reference_categories_before_job_creation(self):
        image = _png_b64()
        body = {
            "prompt": "safe",
            "init_image_b64": image,
            "incontext_image_b64": image,
            "character_edit_source_b64": image,
            "character_edit_reference_b64": image,
            "character_edit_regions": [{"reference_b64": image}],
            "style_transfer_image_b64": image,
            "style_references": [{"image_b64": image, "mask_b64": image}],
            "ref_image1_b64": image,
            "ref_image2_b64": image,
            "ref_image3_b64": image,
            "moodboard_images": [image],
            "regional_prompts": [{"prompt": "region", "mask_b64": image}],
        }
        blocked = SimpleNamespace(
            allowed=False, event_type="unsafe", scores={}, reason="blocked"
        )
        with (
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "moderate_images", return_value=blocked) as moderate,
            patch.object(main, "save_moderation_event", new=AsyncMock(return_value=18)) as save,
            patch.object(main, "_enqueue_batch_children", new=AsyncMock()) as batch,
            patch("comfy_workflows.comfy_generate") as helper,
            _api(self.jobs, self.queue, username="kid", role="child") as client,
        ):
            response = client.post("/api/generate", json=body)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.jobs, {})
        self.assertEqual(self.queue.all_statuses(), {})
        self.assertEqual(len(moderate.call_args.args[0]), 1)
        save.assert_awaited_once()
        batch.assert_not_awaited()
        helper.assert_not_called()

    def test_generation_input_collector_covers_each_schema_category(self):
        req = main.GenerationRequest(
            prompt="safe",
            init_image_b64="init",
            incontext_image_b64="incontext",
            character_edit_source_b64="character-source",
            character_edit_reference_b64="character-reference",
            character_edit_regions=[{"reference_b64": "character-region"}],
            style_transfer_image_b64="style-transfer",
            style_references=[{"image_b64": "style-reference"}],
            ref_image1_b64="reference-1",
            ref_image2_b64="reference-2",
            ref_image3_b64="reference-3",
            moodboard_images=["moodboard-upload"],
        )
        self.assertEqual(
            main._collect_generation_input_images(req),
            [
                "init",
                "incontext",
                "character-source",
                "character-reference",
                "style-transfer",
                "reference-1",
                "reference-2",
                "reference-3",
                "character-region",
                "style-reference",
                "moodboard-upload",
            ],
        )

    def test_non_child_generation_does_not_decode_input_images(self):
        with (
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "_decode_generation_input_images", new=AsyncMock()) as decode,
            _api(self.jobs, self.queue) as client,
        ):
            response = client.post(
                "/api/generate",
                json={"prompt": "safe", "init_image_b64": _png_b64()},
            )
        self.assertEqual(response.status_code, 200)
        decode.assert_not_awaited()

    def test_manual_guidance_endpoints_queue_interactive_tasks(self):
        cases = (
            ("/api/moodboards/7/qwen-guidance", {}, "single"),
            ("/api/moodboards/qwen-guidance-missing", {"limit": 2}, "missing"),
            ("/api/moodboards/mashup", {"moodboard_ids": [1, 2]}, "mashup"),
        )
        with (
            patch.object(main, "_prepare_moodboard_task") as prepare,
            _api(self.jobs, self.queue) as client,
        ):
            for path, body, operation in cases:
                response = client.post(path, json=body)
                self.assertEqual(response.status_code, 202)
                payload = response.json()
                self.assertEqual(payload["task_kind"], MOODBOARD_GUIDANCE)
                record = self.queue.status(payload["job_id"])
                self.assertEqual(record["priority_class"], "interactive")
                self.assertEqual(self.jobs[payload["job_id"]]["operation"], operation)
        prepare.assert_not_called()

    def test_admin_benchmark_endpoint_only_enqueues_real_execution(self):
        with (
            patch.object(main, "_execute_helper_benchmark", create=True) as direct,
            _api(self.jobs, self.queue, role="admin") as client,
        ):
            response = client.post(
                "/api/admin/helper-benchmark",
                json={
                    "models": ["Qwen2.5-VL-2B-Instruct-abliterated"],
                    "precisions": ["fp16"],
                    "repeats": 1,
                    "subsequent_krea": False,
                },
            )
        self.assertEqual(response.status_code, 202)
        task_id = response.json()["job_id"]
        self.assertEqual(self.jobs[task_id]["task_kind"], "helper_benchmark")
        self.assertEqual(self.queue.status(task_id)["status"], "queued")
        direct.assert_not_called()

    def test_custom_moodboard_only_queues_when_auto_authoring_uses_qwen(self):
        complete = {
            "title": "Complete",
            "taste_profile": "Warm editorial",
            "keywords": ["warm"],
            "image_b64s": [_png_b64()],
        }
        incomplete = {**complete, "taste_profile": ""}
        item = {"id": 1, "title": "Complete"}
        with (
            patch.object(main, "create_custom_moodboard", new=AsyncMock(return_value=item)) as create,
            _api(self.jobs, self.queue) as client,
        ):
            sync_response = client.post("/api/moodboards/custom", json=complete)
            queued_response = client.post("/api/moodboards/custom", json=incomplete)
        self.assertEqual(sync_response.status_code, 200)
        self.assertEqual(sync_response.json(), item)
        create.assert_awaited_once()
        self.assertEqual(queued_response.status_code, 202)
        self.assertEqual(queued_response.json()["task_kind"], MOODBOARD_GUIDANCE)

    def test_child_upscale_input_is_blocked_before_enqueue(self):
        blocked = SimpleNamespace(
            allowed=False,
            event_type="unsafe",
            scores={},
            reason="blocked",
        )
        with (
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "moderate_images", return_value=blocked),
            patch.object(main, "save_moderation_event", new=AsyncMock(return_value=10)),
            patch("comfy_workflows.comfy_upscale") as upscale,
            _api(self.jobs, self.queue, username="kid", role="child") as client,
        ):
            response = client.post(
                "/api/upscale",
                json={"image_b64": _png_b64(), "method": "tiled_vae"},
            )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.queue.all_statuses(), {})
        upscale.assert_not_called()

    def test_child_depth_input_is_blocked_before_enqueue(self):
        blocked = SimpleNamespace(
            allowed=False, event_type="unsafe", scores={}, reason="blocked"
        )
        with (
            patch.object(main, "moderate_images", return_value=blocked),
            patch.object(main, "save_moderation_event", new=AsyncMock(return_value=13)),
            patch.object(main, "use_comfy_backend", return_value=True),
            patch.object(main, "comfy_available", return_value=True),
            patch("comfy_workflows.comfy_depth_preview") as helper,
            _api(self.jobs, self.queue, username="kid", role="child") as client,
        ):
            response = client.post(
                "/api/depth-preview", json={"image_b64": _png_b64()}
            )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.queue.all_statuses(), {})
        helper.assert_not_called()

    def test_child_custom_images_are_blocked_before_enqueue_or_qwen(self):
        blocked = SimpleNamespace(
            allowed=False, event_type="unsafe", scores={}, reason="blocked"
        )
        with (
            patch.object(main, "moderate_images", return_value=blocked),
            patch.object(main, "save_moderation_event", new=AsyncMock(return_value=14)),
            patch.object(main, "create_custom_moodboard") as helper,
            _api(self.jobs, self.queue, username="kid", role="child") as client,
        ):
            response = client.post(
                "/api/moodboards/custom",
                json={
                    "title": "",
                    "taste_profile": "",
                    "keywords": [],
                    "image_b64s": [_png_b64()],
                },
            )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.queue.all_statuses(), {})
        helper.assert_not_called()


class AuxiliaryWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_cpu_realesrgan_dispatch_skips_comfy_capability_probe(self):
        jobs = {
            "upscale": {
                "status": "queued",
                "task_kind": UPSCALE,
                "result": None,
                "role": "user",
            }
        }
        queue = Mock()
        queue.all_statuses.return_value = {}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "comfy_available", return_value=False),
            patch.object(
                main,
                "comfy_atomic_cancel_available",
                create=True,
            ) as capability,
            patch.object(main, "_run_auxiliary_task", new=AsyncMock()) as execute,
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler(
                "upscale",
                {
                    "req": main.UpscaleRequest(
                        image_b64=_png_b64(), method="realesrgan"
                    )
                },
            )
        capability.assert_not_called()
        execute.assert_awaited_once()

    async def test_benchmark_cancel_targets_exact_prompt_and_stops(self):
        jobs = {
            "benchmark": {
                "status": "running",
                "task_kind": HELPER_BENCHMARK,
                "result": None,
                "role": "admin",
            }
        }
        cancelled = False
        queue = Mock()
        queue.cancel_requested.side_effect = lambda _job_id: cancelled
        queue.all_statuses.return_value = {}

        async def execute(_payload, prompt_id_cb, cancel_probe):
            nonlocal cancelled
            cancelled = True
            prompt_id_cb("benchmark-exact-id")
            self.assertTrue(cancel_probe())
            raise RuntimeError("Benchmark cancelled")

        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "_execute_helper_benchmark", side_effect=execute),
            patch("comfy_client.cancel_prompt", return_value=True) as targeted,
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler("benchmark", {})

        targeted.assert_called_once_with("benchmark-exact-id")
        self.assertEqual(jobs["benchmark"]["status"], "cancelled")
        self.assertIsNone(jobs["benchmark"]["result"])

    async def test_worker_cancelled_error_cleans_prepared_artifacts_and_reraises(self):
        jobs = {
            "guidance": {
                "status": "running",
                "task_kind": MOODBOARD_GUIDANCE,
                "result": None,
                "role": "user",
            }
        }
        prepared = SimpleNamespace(moderation_text="safe")
        queue = Mock()
        queue.all_statuses.return_value = {}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(
                main,
                "_prepare_moodboard_task",
                new=AsyncMock(return_value=prepared),
            ),
            patch.object(
                main,
                "_finish_prepared_moodboard_task",
                new=AsyncMock(side_effect=asyncio.CancelledError),
            ),
            patch.object(
                main,
                "_cleanup_prepared_moodboard_task",
                new=AsyncMock(),
            ) as cleanup,
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await main._queued_gpu_task_handler(
                    "guidance", {"operation": "single", "moodboard_id": 1}
                )
        cleanup.assert_awaited_once_with(prepared)

    async def test_moodboard_prepare_cancel_and_finalizing_commit_races(self):
        for operation in ("single", "custom", "mashup"):
            with self.subTest(operation=operation, race="cancel_after_prepare"):
                jobs = {
                    operation: {
                        "status": "queued",
                        "task_kind": MOODBOARD_GUIDANCE,
                        "result": None,
                        "role": "user",
                    }
                }
                prepared = SimpleNamespace(moderation_text="safe")
                finish_entered = asyncio.Event()
                release_finish = asyncio.Event()
                original_finish = getattr(main, "_finish_prepared_moodboard_task", None)

                async def paused_finish(job_id, artifact):
                    finish_entered.set()
                    await release_finish.wait()
                    return await original_finish(job_id, artifact)

                queue = GpuTaskQueue(main._queued_gpu_task_handler)
                queue.enqueue(
                    operation,
                    {"operation": operation},
                    username="alice",
                    role="user",
                    task_kind=MOODBOARD_GUIDANCE,
                )
                worker = asyncio.create_task(queue.run())
                try:
                    with (
                        patch.object(main, "_jobs", jobs),
                        patch.object(main, "generation_queue", queue),
                        patch.object(main, "_prepare_moodboard_task", new=AsyncMock(return_value=prepared)),
                        patch.object(main, "_finish_prepared_moodboard_task", new=paused_finish),
                        patch.object(main, "_commit_prepared_moodboard_task", new=AsyncMock()) as commit,
                        patch.object(main, "_cleanup_prepared_moodboard_task", new=AsyncMock()) as cleanup,
                        patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
                    ):
                        await asyncio.wait_for(finish_entered.wait(), timeout=2)
                        self.assertEqual(queue.request_cancel(operation), "interrupt")
                        release_finish.set()
                        await asyncio.wait_for(queue.join(), timeout=2)
                    self.assertEqual(jobs[operation]["status"], "cancelled")
                    commit.assert_not_awaited()
                    cleanup.assert_awaited_once()
                finally:
                    release_finish.set()
                    worker.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await worker

            with self.subTest(operation=operation, race="cancel_during_commit"):
                jobs = {
                    operation: {
                        "status": "queued",
                        "task_kind": MOODBOARD_GUIDANCE,
                        "result": None,
                        "role": "user",
                    }
                }
                prepared = SimpleNamespace(moderation_text="safe")
                commit_entered = asyncio.Event()
                release_commit = asyncio.Event()

                async def paused_commit(_artifact):
                    commit_entered.set()
                    await release_commit.wait()
                    return {"operation": operation}

                queue = GpuTaskQueue(main._queued_gpu_task_handler)
                queue.enqueue(
                    operation,
                    {"operation": operation},
                    username="alice",
                    role="user",
                    task_kind=MOODBOARD_GUIDANCE,
                )
                worker = asyncio.create_task(queue.run())
                try:
                    with (
                        patch.object(main, "_jobs", jobs),
                        patch.object(main, "generation_queue", queue),
                        patch.object(main, "_prepare_moodboard_task", new=AsyncMock(return_value=prepared)),
                        patch.object(main, "_commit_prepared_moodboard_task", new=paused_commit),
                        patch.object(main, "_cleanup_prepared_moodboard_task", new=AsyncMock()),
                        patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
                    ):
                        await asyncio.wait_for(commit_entered.wait(), timeout=2)
                        self.assertEqual(queue.request_cancel(operation), "none")
                        self.assertEqual(queue.status(operation)["status"], "finalizing")
                        release_commit.set()
                        await asyncio.wait_for(queue.join(), timeout=2)
                    self.assertEqual(jobs[operation]["status"], "done")
                    self.assertEqual(jobs[operation]["result"], {"operation": operation})
                finally:
                    release_commit.set()
                    worker.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await worker

    async def test_moodboard_commit_failure_cleans_staging_after_finalizing(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory)
            task_root = storage / ".task-prep" / "custom"
            temp_dir = task_root / "board"
            final_dir = storage / "committed-board"
            temp_dir.mkdir(parents=True)
            (temp_dir / "staged.webp").write_bytes(b"staged")
            final_dir.mkdir()
            (final_dir / "existing.webp").write_bytes(b"final")
            prepared = SimpleNamespace(
                moderation_text="safe",
                temp_dir=temp_dir,
                final_dir=final_dir,
            )
            jobs = {
                "custom": {
                    "status": "running",
                    "task_kind": MOODBOARD_GUIDANCE,
                    "result": None,
                    "role": "user",
                }
            }
            queue = Mock()
            queue.begin_finalizing.return_value = True
            queue.cancel_requested.return_value = False
            queue.all_statuses.return_value = {}
            with (
                patch.object(main, "_jobs", jobs),
                patch.object(main, "generation_queue", queue),
                patch.object(
                    main,
                    "_prepare_moodboard_task",
                    new=AsyncMock(return_value=prepared),
                ),
                patch.object(
                    main,
                    "_commit_prepared_moodboard_task",
                    new=AsyncMock(side_effect=OSError("directory replacement locked")),
                ),
                patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
            ):
                await main._queued_gpu_task_handler(
                    "custom", {"operation": "custom"}
                )

            self.assertEqual(jobs["custom"]["status"], "error")
            self.assertFalse(task_root.exists())
            self.assertTrue((final_dir / "existing.webp").exists())

    async def test_child_input_decode_runs_off_event_loop(self):
        ticks = 0
        running = True

        async def heartbeat():
            nonlocal ticks
            while running:
                ticks += 1
                await asyncio.sleep(0.01)

        def slow_decode(_value):
            time.sleep(0.12)
            return Image.new("RGB", (8, 8), "white")

        pulse = asyncio.create_task(heartbeat())
        try:
            with patch("upscaler.b64_to_pil", side_effect=slow_decode):
                await main._decode_upscale_image(_png_b64())
        finally:
            running = False
            await pulse
        self.assertGreater(ticks, 5)

    async def test_child_generation_input_decode_runs_off_event_loop(self):
        ticks = 0
        running = True

        async def heartbeat():
            nonlocal ticks
            while running:
                ticks += 1
                await asyncio.sleep(0.01)

        original = main._job_images_from_b64

        def slow_decode(values):
            time.sleep(0.12)
            return original(values)

        pulse = asyncio.create_task(heartbeat())
        try:
            with patch.object(main, "_job_images_from_b64", side_effect=slow_decode):
                images = await main._decode_generation_input_images(
                    main.GenerationRequest(
                        prompt="safe", init_image_b64=_png_b64()
                    )
                )
        finally:
            running = False
            await pulse
        self.assertEqual(len(images), 1)
        self.assertGreater(ticks, 5)

    async def test_interrupted_moodboard_execution_finishes_cancelled(self):
        jobs = {
            "guidance": {
                "status": "running",
                "task_kind": MOODBOARD_GUIDANCE,
                "result": None,
                "role": "user",
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = False
        queue.all_statuses.return_value = {}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(
                main,
                "_prepare_moodboard_task",
                new=AsyncMock(
                    side_effect=ComfyExecutionError(
                        "ComfyUI execution was interrupted."
                    )
                ),
            ),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler(
                "guidance", {"operation": "single", "moodboard_id": 1}
            )
        self.assertEqual(jobs["guidance"]["status"], "cancelled")

    async def test_cpu_realesrgan_complete_pipeline_runs_off_event_loop(self):
        ticks = 0
        running = True

        async def heartbeat():
            nonlocal ticks
            while running:
                ticks += 1
                await asyncio.sleep(0.01)

        def slow_decode(_value):
            time.sleep(0.08)
            return Image.new("RGB", (8, 8), "white")

        def slow_encode(*_args, **_kwargs):
            time.sleep(0.08)
            return (["encoded"], [])

        pulse = asyncio.create_task(heartbeat())
        try:
            with (
                patch("upscaler.b64_to_pil", side_effect=slow_decode),
                patch("upscaler.upscale_realesrgan", return_value=Image.new("RGB", (16, 16), "black")),
                patch("output_saver.encode_images", side_effect=slow_encode),
            ):
                await main._execute_cpu_realesrgan(
                    main.UpscaleRequest(image_b64=_png_b64(), method="realesrgan")
                )
        finally:
            running = False
            await pulse
        self.assertGreater(ticks, 5)

    async def test_child_moodboard_guidance_text_is_blocked_before_publication(self):
        jobs = {
            "guidance": {
                "status": "running",
                "task_kind": MOODBOARD_GUIDANCE,
                "result": None,
                "role": "child",
                "username": "kid",
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = False
        queue.begin_finalizing.return_value = True
        queue.all_statuses.return_value = {}
        blocked = SimpleNamespace(
            allowed=False, event_type="unsafe", scores={}, reason="blocked"
        )
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(
                main,
                "_prepare_moodboard_task",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        moderation_text="unsafe generated guidance"
                    )
                ),
            ),
            patch.object(
                main, "_commit_prepared_moodboard_task", new=AsyncMock()
            ) as commit,
            patch.object(
                main, "_cleanup_prepared_moodboard_task", new=AsyncMock()
            ),
            patch.object(main, "moderate_prompt", return_value=blocked),
            patch.object(main, "save_moderation_event", new=AsyncMock(return_value=15)),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler(
                "guidance", {"operation": "single", "moodboard_id": 1}
            )
        self.assertEqual(jobs["guidance"]["status"], "blocked")
        self.assertIsNone(jobs["guidance"]["result"])
        commit.assert_not_awaited()

    async def test_mixed_six_user_tasks_are_serial_and_round_robin(self):
        active = 0
        max_active = 0
        order: list[str] = []

        async def handler(task_id, _payload):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            order.append(task_id)
            await asyncio.sleep(0)
            active -= 1

        kinds = (
            GENERATION,
            UPSCALE,
            DEPTH_PREVIEW,
            MOODBOARD_GUIDANCE,
            HELPER_BENCHMARK,
            UPSCALE,
        )
        queue = GpuTaskQueue(handler)
        for round_number in range(2):
            for user, kind in zip("abcdef", kinds):
                queue.enqueue(
                    f"{user}{round_number}",
                    {},
                    username=user,
                    role="user",
                    task_kind=kind,
                )
        worker = asyncio.create_task(queue.run())
        await queue.join()
        worker.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await worker
        self.assertEqual(max_active, 1)
        self.assertEqual(order, [f"{user}{round_number}" for round_number in range(2) for user in "abcdef"])

    async def test_nine_authenticated_users_mixed_tasks_integrate_queue_dispatch_and_privacy(self):
        jobs: dict[str, dict] = {}
        active = 0
        max_active = 0
        execution_order: list[str] = []
        executed_kinds: list[str] = []
        task_ids_by_user: dict[str, list[str]] = {
            f"user{index}": [] for index in range(9)
        }

        async def dispatch(task_id, payload):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            execution_order.append(task_id)
            executed_kinds.append(jobs[task_id]["task_kind"])
            try:
                await asyncio.sleep(0)
                await main._queued_gpu_task_handler(task_id, payload)
            finally:
                active -= 1

        queue = GpuTaskQueue(dispatch)
        image_b64 = _png_b64()
        generation_body = {"prompt": "queue integration landscape"}
        submissions = (
            ("user0", "/api/expand-prompt", {"prompt": "Magic Wand prompt"}),
            ("user0", "/api/generate", generation_body),
            ("user0", "/api/generate", generation_body),
            ("user0", "/api/generate", generation_body),
            ("user0", "/api/generate", generation_body),
            ("user1", "/api/generate", generation_body),
            ("user1", "/api/generate", generation_body),
            ("user2", "/api/depth-preview", {"image_b64": image_b64}),
            ("user2", "/api/generate", generation_body),
            (
                "user3",
                "/api/upscale",
                {"image_b64": image_b64, "method": "tiled_vae"},
            ),
            ("user4", "/api/expand-prompt", {"prompt": "Second Wand prompt"}),
            ("user5", "/api/generate", generation_body),
            ("user6", "/api/generate", generation_body),
            ("user6", "/api/generate", generation_body),
            ("user6", "/api/generate", generation_body),
            ("user6", "/api/generate", generation_body),
            (
                "user7",
                "/api/describe-image",
                {"image_b64": image_b64, "mode": "prompt", "guidance": ""},
            ),
            ("user8", "/api/generate", generation_body),
        )

        expanded = SimpleNamespace(
            expanded="expanded queue prompt",
            changed=True,
            error=None,
            backend="local",
            sign_copy_pass=None,
        )
        generated_image = f"data:image/png;base64,{image_b64}"

        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "use_comfy_backend", return_value=True),
            patch.object(main, "comfy_available", return_value=True),
            patch.object(main, "write_generation_breadcrumb"),
            patch.object(main, "clear_generation_breadcrumb"),
            patch.object(main, "expand_prompt_result", return_value=expanded),
            patch.object(
                main,
                "describe_image_local",
                return_value={"prompt": "described queue image"},
            ),
            patch(
                "comfy_workflows.comfy_generate",
                return_value=([generated_image], 123, [], [], [{}]),
            ),
            patch(
                "comfy_workflows.comfy_upscale",
                return_value=Image.new("RGB", (16, 16), "black"),
            ),
            patch(
                "comfy_workflows.comfy_depth_preview",
                return_value=Image.new("RGB", (8, 8), "gray"),
            ),
            patch(
                "comfy_workflows.resolve_unet",
                return_value=("warm-model", None, False, None),
            ),
            patch("comfy_workflows._vae_name", return_value="warm-vae"),
            patch("comfy_client.cancel_prompt") as targeted_cancel,
        ):
            for username, path, body in submissions:
                with _api(jobs, queue, username=username) as client:
                    response = client.post(path, json=body)
                self.assertIn(response.status_code, {200, 202}, response.text)
                task_id = response.json()["job_id"]
                task_ids_by_user[username].append(task_id)

            self.assertEqual(len(queue.all_statuses()), 18)
            self.assertEqual(queue.admission("user0")["per_user_active"], 5)
            per_user_rejection = queue.check_capacity("user0", "interactive", 4)
            self.assertFalse(per_user_rejection.accepted)
            self.assertEqual(per_user_rejection.reason, "per_user_limit")
            self.assertEqual(
                queue.admission("overflow")["global_interactive_limit"], 64
            )

            background_id = main._new_job(
                role="admin",
                task_kind=MODEL_WARMUP,
                summary="Background warmup",
            )
            jobs[background_id]["priority_class"] = BACKGROUND
            background_result = main._enqueue_gpu_task(
                background_id,
                {},
                username=None,
                role="admin",
                task_kind=MODEL_WARMUP,
                priority_class=BACKGROUND,
            )
            self.assertTrue(background_result.accepted)
            background_rejection = queue.check_capacity(None, BACKGROUND, 4)
            self.assertFalse(background_rejection.accepted)
            self.assertEqual(background_rejection.reason, "global_limit")

            cancelled_id = task_ids_by_user["user6"][-1]
            with _api(jobs, queue, username="user6") as client:
                cancel_response = client.post(
                    f"/api/generate/{cancelled_id}/cancel"
                )
            self.assertEqual(cancel_response.status_code, 200)
            self.assertEqual(queue.status(cancelled_id)["status"], "cancelled")
            self.assertEqual(jobs[cancelled_id]["status"], "cancelled")
            targeted_cancel.assert_not_called()
            self.assertEqual(
                sum(
                    record["status"] == "cancelled"
                    for record in queue.all_statuses().values()
                ),
                1,
            )

            with _api(jobs, queue, username="user0") as client:
                own_status = client.get(
                    f"/api/generate/{task_ids_by_user['user0'][0]}"
                )
                foreign_status = client.get(
                    f"/api/generate/{task_ids_by_user['user1'][0]}"
                )
                listed = client.get("/api/jobs?limit=100")
            self.assertEqual(own_status.status_code, 200)
            self.assertEqual(foreign_status.status_code, 404)
            self.assertEqual(listed.status_code, 200)
            listed_jobs = listed.json()["jobs"]
            foreign_jobs = [record for record in listed_jobs if not record["mine"]]
            self.assertTrue(foreign_jobs)
            for record in foreign_jobs:
                self.assertTrue(record["job_id"].startswith("anon-"))
                self.assertNotIn("username", record)
                self.assertNotIn("content", record)
                self.assertNotIn(
                    record["job_id"][5:],
                    {task_id for ids in task_ids_by_user.values() for task_id in ids},
                )
            owner_entry = next(
                record
                for record in listed_jobs
                if record["job_id"] == task_ids_by_user["user0"][0]
            )
            self.assertTrue(owner_entry["mine"])
            self.assertEqual(owner_entry["task_kind"], PROMPT_EXPAND)

            projected_order = [
                task_id
                for task_id, record in sorted(
                    queue.all_statuses().items(),
                    key=lambda item: (
                        item[1]["queue_position"]
                        if item[1]["queue_position"] is not None
                        else 10_000
                    ),
                )
                if record["status"] == "queued"
            ]
            self.assertEqual(projected_order[-1], background_id)

            worker = asyncio.create_task(queue.run())
            try:
                await asyncio.wait_for(queue.join(), timeout=5)
            finally:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

            self.assertEqual(max_active, 1)
            self.assertEqual(execution_order, projected_order)
            first_round_users = {jobs[task_id]["username"] for task_id in execution_order[:9]}
            self.assertEqual(first_round_users, {f"user{index}" for index in range(9)})
            for username, submitted_ids in task_ids_by_user.items():
                expected = [
                    task_id for task_id in submitted_ids if task_id != cancelled_id
                ]
                actual = [
                    task_id
                    for task_id in execution_order
                    if jobs[task_id].get("username") == username
                ]
                self.assertEqual(actual, expected, username)
            self.assertEqual(execution_order[-1], background_id)
            self.assertNotIn(cancelled_id, execution_order)
            self.assertIn(PROMPT_EXPAND, executed_kinds)
            self.assertIn(IMAGE_DESCRIBE, executed_kinds)
            self.assertIn(DEPTH_PREVIEW, executed_kinds)
            self.assertIn(UPSCALE, executed_kinds)
            self.assertIn(GENERATION, executed_kinds)

            wand_id = task_ids_by_user["user0"][0]
            with _api(jobs, queue, username="user0") as client:
                completed = client.get(f"/api/generate/{wand_id}")
            self.assertEqual(completed.status_code, 200)
            self.assertEqual(completed.json()["status"], "done")
            self.assertEqual(
                completed.json()["result"]["expanded"], "expanded queue prompt"
            )
            self.assertTrue(
                all(
                    jobs[task_id]["status"] == "done"
                    for task_id in execution_order
                )
            )

    async def test_child_upscale_output_is_blocked_before_result_publication(self):
        jobs = {
            "upscale": {
                "status": "running",
                "task_kind": UPSCALE,
                "result": None,
                "role": "child",
                "username": "kid",
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = False
        queue.begin_finalizing.return_value = True
        queue.all_statuses.return_value = {}
        blocked = SimpleNamespace(
            allowed=False,
            event_type="unsafe",
            scores={},
            reason="blocked",
        )
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "comfy_available", return_value=True),
            patch.object(main, "moderate_images", return_value=blocked),
            patch.object(main, "save_moderation_event", new=AsyncMock(return_value=11)),
            patch("comfy_workflows.comfy_upscale", return_value=Image.new("RGB", (4, 4), "black")),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler(
                "upscale",
                {"req": main.UpscaleRequest(image_b64=_png_b64(), method="tiled_vae")},
            )
        self.assertEqual(jobs["upscale"]["status"], "blocked")
        self.assertIsNone(jobs["upscale"]["result"])

    async def test_worker_stores_result_prompt_id_and_finalizes_before_publication(self):
        jobs = {
            "depth": {
                "status": "running",
                "task_kind": DEPTH_PREVIEW,
                "result": None,
                "role": "user",
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = False
        queue.begin_finalizing.return_value = True
        queue.all_statuses.return_value = {}

        def depth(_image, **kwargs):
            kwargs["prompt_id_cb"]("depth-prompt")
            return Image.new("RGB", (4, 4), "black")

        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "use_comfy_backend", return_value=True),
            patch.object(main, "comfy_available", return_value=True),
            patch("comfy_workflows.comfy_depth_preview", side_effect=depth),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)) as broadcast,
        ):
            await main._queued_gpu_task_handler(
                "depth",
                {"image_b64": _png_b64(), "estimator": "da3", "resolution": 504, "invert": False},
            )

        queue.begin_finalizing.assert_called_once_with("depth")
        self.assertEqual(jobs["depth"]["status"], "done")
        self.assertTrue(jobs["depth"]["result"]["image_b64"].startswith("data:image/png;base64,"))
        self.assertIsNone(jobs["depth"]["comfy_prompt_id"])
        self.assertEqual(broadcast.await_args_list[-1].args[1]["type"], "done")

    async def test_cancel_before_finalizing_discards_upscale_result(self):
        jobs = {
            "upscale": {
                "status": "running",
                "task_kind": UPSCALE,
                "result": None,
                "role": "user",
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = True
        queue.begin_finalizing.return_value = False
        queue.all_statuses.return_value = {}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "comfy_available", return_value=True),
            patch("comfy_workflows.comfy_upscale", return_value=Image.new("RGB", (4, 4), "black")),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler(
                "upscale",
                {"req": main.UpscaleRequest(image_b64=_png_b64(), method="tiled_vae")},
            )
        self.assertEqual(jobs["upscale"]["status"], "cancelled")
        self.assertIsNone(jobs["upscale"]["result"])

    async def test_background_enrichment_is_deduplicated_and_runs_after_interactive(self):
        order: list[str] = []

        async def handler(task_id, _payload):
            order.append(task_id)

        queue = GpuTaskQueue(handler)
        jobs: dict[str, dict] = {}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
        ):
            first = main._enqueue_background_enrichment()
            second = main._enqueue_background_enrichment()
            for user, kind in (
                ("a", GENERATION),
                ("b", UPSCALE),
                ("c", DEPTH_PREVIEW),
                ("d", MOODBOARD_GUIDANCE),
                ("e", GENERATION),
                ("f", UPSCALE),
            ):
                task_id = main._new_job(username=user, role="user", task_kind=kind)
                main._enqueue_gpu_task(
                    task_id,
                    {},
                    username=user,
                    role="user",
                    task_kind=kind,
                )

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(queue.status(first)["priority_class"], BACKGROUND)
        worker = asyncio.create_task(queue.run())
        await queue.join()
        worker.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await worker
        self.assertEqual(order[-1], first)
        self.assertEqual(len(order), 7)


class FrontendCompatibilityStaticTests(unittest.TestCase):
    def test_auxiliary_wrappers_accept_direct_or_queued_responses(self):
        source = (ROOT / "frontend" / "src" / "api.ts").read_text(encoding="utf-8")
        self.assertIn("resolveGpuSubmission", source)
        self.assertIn("'job_id' in submitted", source)
        for wrapper in (
            "generateMoodboardGuidance",
            "generateMissingMoodboardGuidance",
            "createMoodboardMashup",
            "upscale:",
            "depthPreview:",
        ):
            start = source.index(wrapper)
            snippet = source[start:start + 2200]
            self.assertIn("resolveGpuSubmission", snippet, wrapper)

    def test_polling_retries_acks_and_cancels_on_65_minute_timeout(self):
        source = (ROOT / "frontend" / "src" / "api.ts").read_text(encoding="utf-8")
        helper = source[source.index("export async function waitForGpuTask"):source.index("export const apiFetch")]
        self.assertIn("65 * 60 * 1000", helper)
        self.assertIn("transientFailures", helper)
        self.assertIn("Math.min", helper)
        self.assertIn("/ack", helper)
        self.assertIn("/cancel", helper)
        self.assertLess(helper.index("const result = job.result as T"), helper.index("/ack"))


if __name__ == "__main__":
    unittest.main()
