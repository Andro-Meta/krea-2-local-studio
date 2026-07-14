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
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import comfy_qwen_vl  # noqa: E402
import main  # noqa: E402
from support import mock_atomic_cancel_capability  # noqa: E402

mock_atomic_cancel_capability(main)
import prompt_expander  # noqa: E402
import prompt_planner  # noqa: E402
import settings as settings_module  # noqa: E402
from gpu_task_queue import GpuTaskQueue  # noqa: E402
from gpu_tasks import GENERATION, IMAGE_DESCRIBE, PROMPT_EXPAND, PROMPT_PLAN  # noqa: E402


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@contextmanager
def _api(username: str, role: str, jobs: dict, queue: GpuTaskQueue):
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


class HelperEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        async def noop(_task_id, _payload):
            return None

        self.jobs: dict[str, dict] = {}
        self.queue = GpuTaskQueue(noop)

    def test_all_helper_endpoints_return_202_queue_shape(self):
        cases = (
            ("/api/expand-prompt", {"prompt": "a fox"}, PROMPT_EXPAND),
            ("/api/plan-prompt", {"prompt": "a fox"}, PROMPT_PLAN),
            (
                "/api/describe-image",
                {"image_b64": _png_b64(), "mode": "recreate"},
                IMAGE_DESCRIBE,
            ),
        )
        with (
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
            patch.object(main, "moderate_images", return_value=SimpleNamespace(allowed=True)),
            _api("alice", "user", self.jobs, self.queue) as client,
        ):
            for path, body, task_kind in cases:
                response = client.post(path, json=body)
                self.assertEqual(response.status_code, 202)
                self.assertEqual(
                    set(response.json()),
                    {"job_id", "status", "task_kind", "queue_position", "queue_length"},
                )
                self.assertEqual(response.json()["task_kind"], task_kind)

    def test_active_generation_does_not_return_busy_409(self):
        accepted = self.queue.enqueue(
            "generation",
            {},
            username="bob",
            role="user",
            task_kind=GENERATION,
        )
        self.assertTrue(accepted.accepted)
        with (
            patch.object(main, "moderate_prompt", return_value=SimpleNamespace(allowed=True)),
            _api("alice", "user", self.jobs, self.queue) as client,
        ):
            response = client.post("/api/expand-prompt", json={"prompt": "a fox"})
        self.assertEqual(response.status_code, 202)

    def test_child_input_is_blocked_before_helper_or_enqueue(self):
        blocked = SimpleNamespace(
            allowed=False,
            event_type="unsafe",
            scores={},
            reason="blocked",
        )
        with (
            patch.object(main, "moderate_prompt", return_value=blocked),
            patch.object(main, "save_moderation_event", new=AsyncMock(return_value=7)),
            patch.object(main, "expand_prompt_result") as helper,
            _api("kid", "child", self.jobs, self.queue) as client,
        ):
            response = client.post("/api/expand-prompt", json={"prompt": "blocked input"})
        self.assertEqual(response.status_code, 403)
        helper.assert_not_called()
        self.assertEqual(self.queue.all_statuses(), {})


class HelperWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_moodboard_suggestions_use_original_expanded_and_username(self):
        expected = [{"id": 7, "reason": "Matched: analog grain"}]
        with patch.object(
            main,
            "suggest_moodboards",
            new=AsyncMock(return_value=expected),
            create=True,
        ) as suggest:
            result = await main._moodboard_suggestions(
                "cat in a cafe",
                "cat in a cafe, analog grain",
                "alice",
            )
        self.assertEqual(result, expected)
        suggest.assert_awaited_once_with(
            "cat in a cafe",
            "cat in a cafe, analog grain",
            "alice",
        )

    async def test_prompt_expander_cuda_oom_reaches_shared_retry(self):
        jobs = {
            "helper": {
                "status": "running",
                "task_kind": PROMPT_EXPAND,
                "result": None,
                "role": "user",
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = False
        queue.begin_finalizing.return_value = True
        queue.all_statuses.return_value = {}
        success = SimpleNamespace(
            expanded="expanded",
            changed=True,
            error=None,
            backend="comfy",
            sign_copy_pass=None,
        )
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(
                main,
                "expand_prompt_result",
                side_effect=[RuntimeError("CUDA out of memory"), success],
            ) as expand,
            patch.object(main, "free_comfy_vram", return_value=True) as free,
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._run_helper_task(
                "helper",
                {
                    "backend": "local",
                    "prompt": "fox",
                    "suggest_moodboards": False,
                },
            )
        self.assertEqual(expand.call_count, 2)
        free.assert_called_once_with(unload_models=True, free_memory=True)
        self.assertEqual(jobs["helper"]["status"], "done")

    async def test_successful_comfy_image_description_queues_krea_rewarm(self):
        jobs = {
            "describe": {
                "status": "queued",
                "task_kind": IMAGE_DESCRIBE,
                "result": None,
                "role": "user",
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = False
        queue.begin_finalizing.return_value = True
        queue.all_statuses.return_value = {}
        result = {"prompt": "literal description", "backend": "comfy"}

        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "describe_image_local", return_value=result),
            patch.object(main, "_enqueue_model_warmup") as rewarm,
            patch.object(
                main.ws_manager,
                "broadcast",
                new=AsyncMock(return_value=0),
            ),
        ):
            await main._run_helper_task(
                "describe",
                {
                    "backend": "local",
                    "image_b64": _png_b64(),
                    "mode": "recreate",
                    "guidance": "",
                },
            )

        self.assertEqual(jobs["describe"]["status"], "done")
        rewarm.assert_called_once_with(force=True)

    async def test_settings_api_persists_canonical_comfy_quant_key(self):
        persisted = {}
        with (
            patch.object(main, "_read_env", return_value={}),
            patch.object(
                main, "_write_env", side_effect=lambda env: persisted.update(env)
            ),
            patch.object(main.settings, "comfy_qwen_quant", "8bit"),
        ):
            await main.update_settings(
                main.SettingsUpdate(comfy_qwen_quant="4bit")
            )

        self.assertEqual(persisted["COMFY_QWEN_QUANT"], "4bit")

    async def test_result_storage_done_payload_and_prompt_id(self):
        jobs = {
            "helper": {
                "status": "queued",
                "task_kind": PROMPT_EXPAND,
                "result": None,
                "role": "user",
            }
        }
        result = SimpleNamespace(
            expanded="expanded fox",
            changed=True,
            error=None,
            backend="comfy",
            sign_copy_pass=None,
        )

        def helper(_prompt, **kwargs):
            kwargs["prompt_id_cb"]("qwen-1")
            return result

        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", None),
            patch.object(main, "expand_prompt_result", side_effect=helper),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)) as broadcast,
        ):
            await main._queued_gpu_task_handler(
                "helper",
                {"prompt": "fox", "backend": "local", "suggest_moodboards": False},
            )

        self.assertEqual(jobs["helper"]["status"], "done")
        self.assertEqual(jobs["helper"]["result"]["expanded"], "expanded fox")
        self.assertIsNone(jobs["helper"]["comfy_prompt_id"])
        done = broadcast.await_args_list[-1].args[1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["task_kind"], PROMPT_EXPAND)
        self.assertEqual(done["result"]["expanded"], "expanded fox")

    async def test_child_output_block_discards_result(self):
        jobs = {
            "helper": {
                "status": "queued",
                "task_kind": PROMPT_PLAN,
                "result": None,
                "role": "child",
                "username": "kid",
            }
        }
        planned = SimpleNamespace(
            model_dump=lambda: {
                "original_prompt": "fox",
                "planned_prompt": "blocked output",
                "negative_prompt": "",
            }
        )
        decisions = [
            SimpleNamespace(allowed=False, event_type="unsafe", scores={}, reason="blocked")
        ]
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", None),
            patch.object(main, "plan_prompt", return_value=planned),
            patch.object(main, "moderate_prompt", side_effect=decisions),
            patch.object(main, "save_moderation_event", new=AsyncMock(return_value=9)),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler(
                "helper",
                {"prompt": "fox", "max_tokens": 700, "backend": "local"},
            )
        self.assertEqual(jobs["helper"]["status"], "blocked")
        self.assertIsNone(jobs["helper"]["result"])

    async def test_cancel_requested_discards_transformers_result(self):
        jobs = {
            "helper": {
                "status": "running",
                "task_kind": PROMPT_EXPAND,
                "result": None,
                "role": "user",
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = True
        queue.begin_finalizing.return_value = False
        queue.all_statuses.return_value = {}
        result = SimpleNamespace(
            expanded="must disappear",
            changed=True,
            error=None,
            backend="transformers",
            sign_copy_pass=None,
        )
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "expand_prompt_result", return_value=result),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler(
                "helper",
                {"prompt": "fox", "backend": "local", "suggest_moodboards": False},
            )
        self.assertEqual(jobs["helper"]["status"], "cancelled")
        self.assertIsNone(jobs["helper"]["result"])

    async def test_cancel_during_output_moderation_discards_helper_result(self):
        jobs = {
            "helper": {
                "status": "queued",
                "task_kind": PROMPT_EXPAND,
                "result": None,
                "role": "child",
                "username": "kid",
            }
        }
        moderation_started = asyncio.Event()
        release_moderation = asyncio.Event()
        expanded = SimpleNamespace(
            expanded="expanded fox",
            changed=True,
            error=None,
            backend="transformers",
            sign_copy_pass=None,
        )

        async def blocked_moderation(*_args, **_kwargs):
            moderation_started.set()
            await release_moderation.wait()
            return True

        queue = GpuTaskQueue(main._queued_gpu_task_handler)
        queue.enqueue(
            "helper",
            {"prompt": "fox", "backend": "local", "suggest_moodboards": False},
            username="kid",
            role="child",
            task_kind=PROMPT_EXPAND,
        )
        worker = asyncio.create_task(queue.run())
        try:
            with (
                patch.object(main, "_jobs", jobs),
                patch.object(main, "generation_queue", queue),
                patch.object(main, "expand_prompt_result", return_value=expanded),
                patch.object(
                    main,
                    "_moderate_worker_text",
                    new=AsyncMock(side_effect=blocked_moderation),
                ),
                patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
            ):
                await asyncio.wait_for(moderation_started.wait(), timeout=2)
                self.assertEqual(queue.request_cancel("helper"), "interrupt")
                release_moderation.set()
                await asyncio.wait_for(queue.join(), timeout=2)

            self.assertEqual(jobs["helper"]["status"], "cancelled")
            self.assertIsNone(jobs["helper"]["result"])
        finally:
            release_moderation.set()
            worker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await worker

    async def test_inline_generation_helper_runs_after_worker_dispatch(self):
        jobs = {
            "generation": {
                "status": "queued",
                "task_kind": GENERATION,
                "result": None,
                "parent_job_id": None,
            }
        }
        req = main.GenerationRequest(prompt="fox", use_prompt_expander=True)
        expanded = SimpleNamespace(
            expanded="expanded fox",
            changed=True,
            error=None,
            backend="comfy",
            sign_copy_pass=None,
        )
        observed: list[str] = []

        def fake_generate(request, **_kwargs):
            observed.append(request.prompt)
            return [], 1, [], [], []

        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", None),
            patch.object(main, "expand_prompt_result", return_value=expanded) as helper,
            patch.object(main, "use_comfy_backend", return_value=True),
            patch.object(main, "write_generation_breadcrumb"),
            patch.object(main, "clear_generation_breadcrumb"),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
            patch("comfy_workflows.comfy_generate", side_effect=fake_generate),
        ):
            await main._queued_gpu_task_handler(
                "generation",
                {"req": req, "username": "alice", "role": "user"},
            )
        helper.assert_called_once()
        self.assertEqual(observed, ["expanded fox"])


class QueueAndModelTests(unittest.IsolatedAsyncioTestCase):
    def test_job_cap_preserves_all_nonterminal_and_unfinished_batch_records(self):
        jobs = {
            "finalizing": {"status": "finalizing"},
            "cancel-requested": {"status": "cancellation_requested"},
            "parent": {
                "status": "finalizing",
                "child_job_ids": ["child"],
            },
            "child": {
                "status": "done",
                "parent_job_id": "parent",
            },
            "old-done": {"status": "done"},
        }
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "_JOBS_MAX", 5),
        ):
            new_id = main._new_job(username="alice", role="user")

        self.assertIn(new_id, jobs)
        self.assertNotIn("old-done", jobs)
        self.assertIn("finalizing", jobs)
        self.assertIn("cancel-requested", jobs)
        self.assertIn("parent", jobs)
        self.assertIn("child", jobs)

    async def test_six_users_run_in_fair_round_robin_order(self):
        order: list[str] = []

        async def handler(task_id, _payload):
            order.append(task_id)

        queue = GpuTaskQueue(handler)
        for round_number in range(2):
            for user in ("a", "b", "c", "d", "e", "f"):
                queue.enqueue(
                    f"{user}{round_number}",
                    {},
                    username=user,
                    role="user",
                    task_kind=PROMPT_EXPAND,
                )
        worker = asyncio.create_task(queue.run())
        await queue.join()
        worker.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await worker
        self.assertEqual(order, [f"{u}{n}" for n in range(2) for u in "abcdef"])

    async def test_busy_probe_ignores_helper_self_lease(self):
        queue = Mock(active_job_id="helper")
        queue.status.return_value = {"task_kind": PROMPT_EXPAND, "status": "running"}
        with patch.object(main, "generation_queue", queue):
            self.assertFalse(main._active_generation_running())
        queue.status.return_value = {"task_kind": GENERATION, "status": "running"}
        with patch.object(main, "generation_queue", queue):
            self.assertTrue(main._active_generation_running())

    def test_comfy_defaults_to_2b_8bit_and_unloads_after_helper(self):
        captured = {}

        def run(graph, *_args, **_kwargs):
            captured.update(graph)
            return "expanded"

        with (
            patch.object(comfy_qwen_vl, "_ensure_nodes"),
            patch.object(comfy_qwen_vl, "_run_graph_for_text", side_effect=run),
            patch.object(comfy_qwen_vl, "resolve_comfy_qwen_model", return_value=comfy_qwen_vl.MODEL_2B_ABLITERATED),
            patch.object(comfy_qwen_vl, "resolve_comfy_qwen_quant", return_value="8-bit (Balanced)"),
        ):
            comfy_qwen_vl.expand_prompt_comfy("fox", "system", keep_model_loaded=False)
        inputs = captured["qwen"]["inputs"]
        self.assertEqual(inputs["model_name"], comfy_qwen_vl.MODEL_2B_ABLITERATED)
        self.assertEqual(inputs["quantization"], "8-bit (Balanced)")
        self.assertFalse(inputs["keep_model_loaded"])

    def test_comfy_vision_defaults_to_4b_8bit(self):
        captured = {}

        def run(graph, *_args, **_kwargs):
            captured.update(graph)
            return "literal description"

        with (
            patch.object(comfy_qwen_vl, "_ensure_nodes"),
            patch.object(
                comfy_qwen_vl,
                "_b64_to_png_bytes",
                return_value=b"png",
            ),
            patch.object(comfy_qwen_vl, "_upload_png", return_value="vision.png"),
            patch.object(
                comfy_qwen_vl,
                "_run_graph_for_text",
                side_effect=run,
            ),
            patch.object(
                settings_module.settings,
                "comfy_qwen_vision_model",
                "4b",
                create=True,
            ),
            patch.object(
                settings_module.settings,
                "comfy_qwen_vision_quant",
                "8bit",
                create=True,
            ),
        ):
            comfy_qwen_vl.describe_image_comfy(
                _png_b64(),
                "Describe literally.",
                keep_model_loaded=False,
            )

        inputs = captured["qwen"]["inputs"]
        self.assertEqual(inputs["model_name"], comfy_qwen_vl.MODEL_4B_ABLITERATED)
        self.assertEqual(inputs["quantization"], comfy_qwen_vl.QUANT_8BIT)
        self.assertFalse(inputs["keep_model_loaded"])

    def test_persisted_comfy_quant_survives_restart_and_maps_to_node_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "COMFY_QWEN_QUANT=4bit\n"
                "COMFY_QWEN_VISION_MODEL=4b\n"
                "COMFY_QWEN_VISION_QUANT=8bit\n",
                encoding="utf-8",
            )
            restarted = settings_module.AppSettings(_env_file=env_file)

        with patch.object(settings_module, "settings", restarted):
            resolved = comfy_qwen_vl.resolve_comfy_qwen_quant()
            vision_model = comfy_qwen_vl.resolve_comfy_qwen_vision_model()
            vision_quant = comfy_qwen_vl.resolve_comfy_qwen_vision_quant()

        self.assertEqual(restarted.comfy_qwen_quant, "4bit")
        self.assertEqual(resolved, "4-bit (VRAM-friendly)")
        self.assertEqual(vision_model, comfy_qwen_vl.MODEL_4B_ABLITERATED)
        self.assertEqual(vision_quant, comfy_qwen_vl.QUANT_8BIT)

    def test_comfy_planner_failure_does_not_load_transformers(self):
        with (
            patch.object(
                comfy_qwen_vl,
                "expand_prompt_comfy",
                side_effect=RuntimeError("Comfy down"),
            ),
            patch.object(prompt_planner, "plan_prompt_local") as transformers,
        ):
            result = prompt_planner.plan_prompt_comfy("fox")
        transformers.assert_not_called()
        self.assertIn("fox", result.planned_prompt)
        self.assertIn("Comfy", result.error)

    def test_explicit_transformers_cuda_frees_comfy_before_load(self):
        events: list[str] = []
        fake_torch = ModuleType("torch")
        fake_torch.bfloat16 = object()
        fake_torch.float32 = object()
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = SimpleNamespace(
            from_pretrained=Mock(
                side_effect=lambda *_a, **_k: events.append("load") or Mock()
            )
        )
        fake_transformers.Qwen3VLProcessor = SimpleNamespace(
            from_pretrained=Mock(return_value=Mock())
        )
        fake_transformers.Qwen3VLForConditionalGeneration = SimpleNamespace(
            from_pretrained=Mock(side_effect=RuntimeError("stop"))
        )
        with (
            patch.object(prompt_expander, "_resolve_local_qwen_device", return_value="cuda"),
            patch.object(prompt_expander, "free_comfy_vram", side_effect=lambda **_k: events.append("free") or True),
            patch("settings.settings.local_llm_backend", "transformers"),
            patch.dict(
                sys.modules,
                {"torch": fake_torch, "transformers": fake_transformers},
            ),
        ):
            with self.assertRaises(RuntimeError):
                prompt_expander._load_local_qwen.__wrapped__("fake-model")
        self.assertEqual(events[:2], ["free", "load"])


if __name__ == "__main__":
    unittest.main()
