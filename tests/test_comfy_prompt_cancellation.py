from __future__ import annotations

import asyncio
import base64
import io
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import comfy_client  # noqa: E402
import comfy_qwen_vl  # noqa: E402
import comfy_workflows  # noqa: E402
import main  # noqa: E402
import sign_copy_pass  # noqa: E402
from gpu_task_queue import GpuTaskQueue  # noqa: E402
from gpu_tasks import GENERATION  # noqa: E402


class _FinishedWebSocket:
    def __init__(self, events: list[str] | None = None):
        self.events = events

    def connect(self, *_args, **_kwargs):
        return None

    def settimeout(self, _timeout):
        return None

    def recv(self):
        if self.events is not None:
            self.events.append("wait")
        return '{"type":"executing","data":{"prompt_id":"prompt-1","node":null}}'

    def close(self):
        return None


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    return buf.getvalue()


def _png_b64() -> str:
    return base64.b64encode(_png_bytes()).decode()


class ComfyPromptCallbackTests(unittest.TestCase):
    def test_public_run_forwards_callback_once_in_websocket_mode(self):
        callback = Mock()
        png = _png_bytes()
        frames = iter(
            [
                '{"type":"executing","data":{"prompt_id":"prompt-1","node":"save_ws"}}',
                b"\0" * 8 + png,
                '{"type":"executing","data":{"prompt_id":"prompt-1","node":null}}',
            ]
        )
        ws = _FinishedWebSocket()
        ws.recv = lambda: next(frames)
        prompt_response = Mock(status_code=200)
        prompt_response.json.return_value = {"prompt_id": "prompt-1"}

        with (
            patch.object(comfy_client, "comfy_available", return_value=True),
            patch.object(comfy_client, "websocket", SimpleNamespace(WebSocket=lambda: ws)),
            patch.object(comfy_client.requests, "post", return_value=prompt_response),
        ):
            result = comfy_client.ComfyClient("http://comfy").run(
                {"node": {}},
                prompt_id_cb=callback,
            )

        self.assertEqual(result, [png])
        callback.assert_called_once_with("prompt-1")

    def test_public_run_forwards_callback_once_in_polling_mode(self):
        callback = Mock()
        prompt_response = Mock(status_code=200)
        prompt_response.json.return_value = {"prompt_id": "prompt-1"}
        history_response = Mock()
        history_response.json.return_value = {
            "prompt-1": {"status": {"status_str": "success"}, "outputs": {}}
        }

        with (
            patch.object(comfy_client, "comfy_available", return_value=True),
            patch.object(comfy_client, "websocket", None),
            patch.object(comfy_client.requests, "post", return_value=prompt_response),
            patch.object(comfy_client.requests, "get", return_value=history_response),
        ):
            result = comfy_client.ComfyClient("http://comfy").run(
                {"node": {}},
                prompt_id_cb=callback,
            )

        self.assertEqual(result, [])
        callback.assert_called_once_with("prompt-1")

    def test_websocket_callback_runs_once_before_waiting(self):
        events: list[str] = []
        client = comfy_client.ComfyClient("http://comfy")
        ws = _FinishedWebSocket(events)

        with (
            patch.object(comfy_client, "websocket", SimpleNamespace(WebSocket=lambda: ws)),
            patch.object(client, "_post_prompt", side_effect=lambda _graph: events.append("post") or "prompt-1"),
            patch.object(client, "_collect_from_history", return_value=[]),
        ):
            client._run_ws(
                {},
                None,
                comfy_client.WS_IMAGE_NODE,
                10,
                prompt_id_cb=lambda prompt_id: events.append(f"callback:{prompt_id}"),
            )

        self.assertEqual(events, ["post", "callback:prompt-1", "wait"])

    def test_polling_callback_runs_once_before_waiting(self):
        events: list[str] = []
        client = comfy_client.ComfyClient("http://comfy")

        with (
            patch.object(client, "_post_prompt", side_effect=lambda _graph: events.append("post") or "prompt-1"),
            patch.object(
                client,
                "get_history",
                side_effect=lambda prompt_id: events.append("wait") or {prompt_id: {"status": {}}},
            ),
            patch.object(client, "_collect_from_history", return_value=[]),
        ):
            client._run_polling(
                {},
                None,
                10,
                prompt_id_cb=lambda prompt_id: events.append(f"callback:{prompt_id}"),
            )

        self.assertEqual(events, ["post", "callback:prompt-1", "wait"])

    def test_websocket_connect_fallback_submits_and_calls_back_once(self):
        client = comfy_client.ComfyClient("http://comfy")
        failing_ws = Mock()
        failing_ws.connect.side_effect = OSError("no websocket")
        callback = Mock()

        with (
            patch.object(comfy_client, "websocket", SimpleNamespace(WebSocket=lambda: failing_ws)),
            patch.object(client, "_post_prompt", return_value="prompt-1") as post_prompt,
            patch.object(client, "get_history", return_value={"prompt-1": {"status": {}}}),
            patch.object(client, "_collect_from_history", return_value=[]),
        ):
            client._run_ws(
                {},
                None,
                comfy_client.WS_IMAGE_NODE,
                10,
                prompt_id_cb=callback,
            )

        post_prompt.assert_called_once_with({})
        callback.assert_called_once_with("prompt-1")

    def test_callback_exception_does_not_abort_execution(self):
        client = comfy_client.ComfyClient("http://comfy")

        with (
            patch.object(client, "_post_prompt", return_value="prompt-1"),
            patch.object(client, "get_history", return_value={"prompt-1": {"status": {}}}),
            patch.object(client, "_collect_from_history", return_value=[]),
        ):
            result = client._run_polling(
                {},
                None,
                10,
                prompt_id_cb=Mock(side_effect=RuntimeError("observer failed")),
            )

        self.assertEqual(result, [])

    def test_qwen_callback_runs_immediately_after_submission(self):
        events: list[str] = []
        fake_client = Mock()
        fake_client.base = "http://comfy"
        fake_client._post_prompt.side_effect = lambda _graph: events.append("post") or "qwen-1"
        fake_client.get_history.side_effect = (
            lambda prompt_id: events.append("wait")
            or {
                prompt_id: {
                    "status": {"completed": True},
                    "outputs": {"preview": {"text": ["done"]}},
                }
            }
        )

        with (
            patch.object(comfy_qwen_vl, "comfy_available", return_value=True),
            patch.object(comfy_qwen_vl, "ComfyClient", return_value=fake_client),
        ):
            result = comfy_qwen_vl._run_graph_for_text(
                {},
                "preview",
                free_vram=False,
                prompt_id_cb=lambda prompt_id: events.append(f"callback:{prompt_id}"),
            )

        self.assertEqual(result, "done")
        self.assertEqual(events, ["post", "callback:qwen-1", "wait"])


class TargetedCancelTests(unittest.TestCase):
    @staticmethod
    def _response(status_code: int, payload: dict | None = None):
        response = Mock(status_code=status_code)
        response.json.return_value = payload or {}
        return response

    def test_atomic_cancel_true_url_quotes_prompt_id(self):
        response = self._response(200, {"cancelled": True})
        with patch.object(comfy_client.requests, "post", return_value=response) as post:
            result = comfy_client.cancel_prompt(
                "pending/id with spaces",
                base_url="http://comfy",
            )

        self.assertTrue(result)
        post.assert_called_once_with(
            "http://comfy/api/jobs/pending%2Fid%20with%20spaces/cancel",
            timeout=10.0,
        )

    def test_atomic_cancel_false_is_false_without_fallback(self):
        response = self._response(200, {"cancelled": False})
        with patch.object(comfy_client.requests, "post", return_value=response) as post:
            result = comfy_client.cancel_prompt("completed", base_url="http://comfy")

        self.assertFalse(result)
        post.assert_called_once()

    def test_cancel_capability_probe_requires_exact_boolean_contract(self):
        for status, payload, expected in (
            (200, {"cancelled": False}, True),
            (200, {"cancelled": True}, True),
            (200, {"cancelled": 0}, False),
            (200, {"ok": True}, False),
            (404, {"cancelled": False}, False),
        ):
            with self.subTest(status=status, payload=payload):
                response = self._response(status, payload)
                with patch.object(
                    comfy_client.requests, "post", return_value=response
                ) as post:
                    self.assertEqual(
                        comfy_client.comfy_cancel_capability(
                            base_url="http://comfy"
                        ),
                        expected,
                    )
                self.assertRegex(
                    post.call_args.args[0],
                    r"http://comfy/api/jobs/krea-capability-[0-9a-f]+/cancel",
                )

    def test_atomic_pending_to_running_race_needs_only_server_decision(self):
        response = self._response(200, {"cancelled": True})
        with patch.object(comfy_client.requests, "post", return_value=response) as post:
            self.assertTrue(
                comfy_client.cancel_prompt("transitioning", base_url="http://comfy")
            )

        self.assertEqual(post.call_count, 1)
        self.assertIn("/api/jobs/transitioning/cancel", post.call_args.args[0])

    def test_atomic_running_to_replacement_never_interrupts_replacement(self):
        response = self._response(200, {"cancelled": False})
        with patch.object(comfy_client.requests, "post", return_value=response) as post:
            self.assertFalse(
                comfy_client.cancel_prompt("already-finished", base_url="http://comfy")
            )

        self.assertEqual(post.call_count, 1)
        self.assertNotEqual(post.call_args.args[0], "http://comfy/interrupt")

    def test_non_2xx_other_than_404_405_does_not_use_fallback(self):
        response = self._response(500)
        with patch.object(comfy_client.requests, "post", return_value=response) as post:
            self.assertFalse(comfy_client.cancel_prompt("prompt-1", base_url="http://comfy"))

        post.assert_called_once()

    def test_empty_prompt_id_never_sends_interrupt_or_cancel(self):
        with patch.object(comfy_client.requests, "post") as post:
            self.assertFalse(comfy_client.cancel_prompt("", base_url="http://comfy"))

        post.assert_not_called()

    def test_404_or_405_never_requests_queue_or_interrupt_routes(self):
        for status_code in (404, 405):
            with self.subTest(status_code=status_code):
                response = self._response(status_code)
                with patch.object(
                    comfy_client.requests, "post", return_value=response
                ) as post:
                    self.assertFalse(
                        comfy_client.cancel_prompt(
                            "unknown-or-finished",
                            base_url="http://comfy",
                        )
                    )

                post.assert_called_once_with(
                    "http://comfy/api/jobs/unknown-or-finished/cancel",
                    timeout=10.0,
                )
                requested_urls = [call.args[0] for call in post.call_args_list]
                self.assertFalse(
                    any(
                        url.endswith("/interrupt") or url.endswith("/queue")
                        for url in requested_urls
                    )
                )

    def test_http_boundary_404_or_405_only_receives_atomic_cancel(self):
        for status_code in (404, 405):
            with self.subTest(status_code=status_code):
                received: list[str] = []

                class Handler(BaseHTTPRequestHandler):
                    def do_POST(self):
                        received.append(self.path)
                        self.send_response(status_code)
                        self.end_headers()

                    def log_message(self, *_args):
                        pass

                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    base = f"http://127.0.0.1:{server.server_port}"
                    self.assertFalse(
                        comfy_client.cancel_prompt(
                            "missing prompt", base_url=base, timeout=2
                        )
                    )
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

                self.assertEqual(
                    received, ["/api/jobs/missing%20prompt/cancel"]
                )


class WorkflowForwardingTests(unittest.TestCase):
    def test_comfy_generate_forwards_prompt_callback(self):
        callback = Mock()
        client = Mock()
        client.run.return_value = [_png_bytes()]
        req = main.GenerationRequest(prompt="test prompt", width=512, height=512)

        with patch.object(comfy_workflows, "ComfyClient", return_value=client):
            results = comfy_workflows.comfy_generate(
                req,
                save_outputs=False,
                prompt_id_cb=callback,
            )

        self.assertTrue(results[0])
        self.assertIs(client.run.call_args.kwargs["prompt_id_cb"], callback)

    def test_comfy_upscale_forwards_prompt_callback_on_all_run_paths(self):
        for method, expected_timeout in (
            ("seedvr2", 3600),
            ("esrgan", 600),
            ("tiled_vae", None),
        ):
            with self.subTest(method=method):
                callback = Mock()
                client = Mock()
                client.run.return_value = [_png_bytes()]
                with (
                    patch.object(comfy_workflows, "ComfyClient", return_value=client),
                    patch.object(comfy_workflows._rq, "post"),
                    patch.object(comfy_workflows, "free_comfy_vram"),
                ):
                    result = comfy_workflows.comfy_upscale(
                        method,
                        _png_b64(),
                        prompt_id_cb=callback,
                    )

                self.assertEqual(result.size, (16, 16))
                self.assertIs(client.run.call_args.kwargs["prompt_id_cb"], callback)
                if expected_timeout is None:
                    self.assertNotIn("timeout", client.run.call_args.kwargs)
                else:
                    self.assertEqual(
                        client.run.call_args.kwargs["timeout"],
                        expected_timeout,
                    )

    def test_comfy_depth_preview_forwards_prompt_callback(self):
        callback = Mock()
        client = Mock()
        client.run.return_value = [_png_bytes()]

        with (
            patch.object(comfy_workflows, "ComfyClient", return_value=client),
            patch.object(comfy_workflows._rq, "post"),
        ):
            result = comfy_workflows.comfy_depth_preview(
                _png_b64(),
                prompt_id_cb=callback,
            )

        self.assertEqual(result.size, (16, 16))
        self.assertIs(client.run.call_args.kwargs["prompt_id_cb"], callback)

    def test_public_qwen_helper_forwards_prompt_callback(self):
        callback = Mock()
        client = Mock()
        client.base = "http://comfy"
        client._post_prompt.return_value = "qwen-1"
        client.get_history.return_value = {
            "qwen-1": {
                "status": {"completed": True},
                "outputs": {"preview": {"text": ["description"]}},
            }
        }
        upload_response = Mock(content=b"{}")
        upload_response.json.return_value = {"name": "uploaded.png"}

        with (
            patch.object(comfy_qwen_vl, "comfy_available", return_value=True),
            patch.object(
                comfy_qwen_vl,
                "object_info",
                return_value={comfy_qwen_vl.QWEN_VL_NODE: {}},
            ),
            patch.object(comfy_qwen_vl.requests, "post", return_value=upload_response),
            patch.object(comfy_qwen_vl, "ComfyClient", return_value=client),
        ):
            result = comfy_qwen_vl.describe_image_comfy(
                _png_b64(),
                "Describe it",
                prompt_id_cb=callback,
            )

        self.assertEqual(result, "description")
        callback.assert_called_once_with("qwen-1")

    def test_default_qwen_helpers_do_not_globally_free_vram(self):
        client = Mock()
        client.base = "http://comfy"
        client._post_prompt.return_value = "qwen-1"
        client.get_history.return_value = {
            "qwen-1": {
                "status": {"completed": True},
                "outputs": {"preview": {"text": ["description"]}},
            }
        }
        with (
            patch.object(comfy_qwen_vl, "comfy_available", return_value=True),
            patch.object(
                comfy_qwen_vl,
                "object_info",
                return_value={comfy_qwen_vl.QWEN_VL_NODE: {}},
            ),
            patch.object(comfy_qwen_vl, "ComfyClient", return_value=client),
            patch.object(comfy_qwen_vl, "free_comfy_vram") as free,
        ):
            comfy_qwen_vl.expand_prompt_comfy("prompt", "system")
        free.assert_not_called()

    def test_sign_copy_forwards_all_prompt_ids_and_unloads_final_stage(self):
        callback = Mock()
        calls = []

        def expand(prompt, _system, **kwargs):
            calls.append(kwargs)
            kwargs["prompt_id_cb"](f"sign-{len(calls)}")
            if len(calls) < 3:
                return "NO_CHANGE"
            return '"OPEN LATE"'

        with (
            patch.object(comfy_qwen_vl, "comfy_qwen_vl_available", return_value=True),
            patch.object(comfy_qwen_vl, "expand_prompt_comfy", side_effect=expand),
        ):
            final, meta = sign_copy_pass.run_sign_copy_pass(
                "A storefront sign with no readable wording.",
                stage1_backend="comfy",
                prompt_id_cb=callback,
            )

        self.assertTrue(meta["ran"])
        self.assertIn('"OPEN LATE"', final)
        self.assertEqual(
            [call.args[0] for call in callback.call_args_list],
            ["sign-1", "sign-2", "sign-3"],
        )
        self.assertTrue(all(call["free_vram"] is False for call in calls))
        self.assertTrue(
            all(call["keep_model_loaded"] is False for call in calls)
        )

    def test_sign_copy_failure_releases_without_global_free(self):
        callback = Mock()
        with (
            patch.object(comfy_qwen_vl, "comfy_qwen_vl_available", return_value=True),
            patch.object(
                comfy_qwen_vl,
                "expand_prompt_comfy",
                side_effect=RuntimeError("failed"),
            ) as expand,
            patch.object(comfy_qwen_vl, "free_comfy_vram") as free,
        ):
            final, meta = sign_copy_pass.run_sign_copy_pass(
                "A storefront sign with no readable wording.",
                prompt_id_cb=callback,
            )
        self.assertIn("storefront", final)
        self.assertEqual(meta["skipped_reason"], "error")
        self.assertFalse(expand.call_args.kwargs["keep_model_loaded"])
        self.assertFalse(expand.call_args.kwargs["free_vram"])
        free.assert_not_called()


class _RunningQueue:
    active_job_id = "job-1"

    def request_cancel(self, _job_id):
        return "interrupt"

    def all_statuses(self):
        return {}

    def cancel_requested(self, _job_id):
        return True

    def begin_finalizing(self, job_id):
        return not self.cancel_requested(job_id)


class _BatchQueue(_RunningQueue):
    active_job_id = "child-running"

    def request_cancel(self, job_id):
        return "interrupt" if job_id == "child-running" else "dequeued"


class MainCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def _cancel(self, job: dict, cancel_prompt_mock: Mock):
        jobs = {"job-1": job}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", _RunningQueue()),
            patch.object(main, "_request_user_role", return_value=(None, "admin", True)),
            patch.object(main, "_sync_queue_state_to_jobs"),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
            patch.object(comfy_client, "cancel_prompt", cancel_prompt_mock),
        ):
            response = await main.cancel_generation_job("job-1", Mock())
        return response

    async def test_main_cancel_uses_stored_prompt_id(self):
        cancel = Mock(return_value=True)
        job = {"status": "running", "comfy_prompt_id": "prompt-1"}
        response = await self._cancel(
            job,
            cancel,
        )

        cancel.assert_called_once_with("prompt-1")
        self.assertEqual(response["cancelled"], 1)
        self.assertEqual(job["status"], "cancelled")

    async def test_main_cancel_false_remains_cancellation_requested(self):
        cancel = Mock(return_value=False)
        job = {"status": "running", "comfy_prompt_id": "prompt-1"}
        response = await self._cancel(job, cancel)

        cancel.assert_called_once_with("prompt-1")
        self.assertTrue(response["ok"])
        self.assertEqual(response["cancelled"], 0)
        self.assertEqual(response["status"], "cancellation_requested")
        self.assertEqual(job["status"], "cancellation_requested")
        self.assertNotIn("finished_at", job)

    async def test_main_cancel_without_prompt_id_does_not_interrupt(self):
        cancel = Mock(return_value=True)
        job = {"status": "running", "comfy_prompt_id": None}
        response = await self._cancel(
            job,
            cancel,
        )

        cancel.assert_not_called()
        self.assertTrue(response["ok"])
        self.assertEqual(response["cancelled"], 0)
        self.assertEqual(job["status"], "cancellation_requested")
        self.assertNotIn("finished_at", job)

    async def test_batch_cancel_targets_only_running_child_prompt(self):
        jobs = {
            "parent": {
                "status": "running",
                "child_job_ids": ["child-running", "child-queued"],
            },
            "child-running": {
                "status": "running",
                "comfy_prompt_id": "running-prompt",
            },
            "child-queued": {
                "status": "queued",
                "comfy_prompt_id": "must-not-cancel",
            },
        }
        cancel = Mock(return_value=True)
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", _BatchQueue()),
            patch.object(main, "_request_user_role", return_value=(None, "admin", True)),
            patch.object(main, "_sync_queue_state_to_jobs"),
            patch.object(main, "_refresh_parent_batch_job", return_value=jobs["parent"]),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
            patch.object(comfy_client, "cancel_prompt", cancel),
        ):
            response = await main.cancel_generation_job("parent", Mock())

        cancel.assert_called_once_with("running-prompt")
        self.assertEqual(response["cancelled"], 2)
        self.assertEqual(jobs["child-running"]["status"], "cancelled")
        self.assertEqual(jobs["child-queued"]["status"], "cancelled")

    async def test_prompt_id_is_stored_then_cleared_after_terminal(self):
        jobs = {
            "job-1": {
                "status": "queued",
                "comfy_prompt_id": None,
                "parent_job_id": None,
            }
        }
        observed: list[str | None] = []

        def fake_generate(_req, **kwargs):
            kwargs["prompt_id_cb"]("prompt-1")
            observed.append(jobs["job-1"]["comfy_prompt_id"])
            raise RuntimeError("finished test execution")

        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", None),
            patch.object(main, "use_comfy_backend", return_value=True),
            patch.object(main, "write_generation_breadcrumb"),
            patch.object(main, "clear_generation_breadcrumb"),
            patch.object(main.logger, "exception"),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
            patch("comfy_workflows.comfy_generate", side_effect=fake_generate),
        ):
            await main._run_generation(
                "job-1",
                SimpleNamespace(diffusion_engine="native_pytorch"),
            )

        self.assertEqual(observed, ["prompt-1"])
        self.assertEqual(jobs["job-1"]["status"], "error")
        self.assertIsNone(jobs["job-1"]["comfy_prompt_id"])

    async def test_failed_cancel_dispatch_still_discards_successful_return(self):
        jobs = {
            "job-1": {
                "status": "cancellation_requested",
                "comfy_prompt_id": None,
                "parent_job_id": None,
            }
        }

        def fake_generate(_req, **kwargs):
            kwargs["prompt_id_cb"]("already-completed")
            return [], 7, [], [], []

        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", _RunningQueue()),
            patch.object(main, "use_comfy_backend", return_value=True),
            patch.object(main, "write_generation_breadcrumb"),
            patch.object(main, "clear_generation_breadcrumb"),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
            patch.object(comfy_client, "cancel_prompt", return_value=False),
            patch("comfy_workflows.comfy_generate", side_effect=fake_generate),
        ):
            await main._run_generation(
                "job-1",
                SimpleNamespace(
                    diffusion_engine="native_pytorch",
                    prompt="test",
                    negative_prompt="",
                    mode="txt2img",
                    checkpoint="turbo",
                    steps=1,
                    cfg=1.0,
                    width=512,
                    height=512,
                    loras=[],
                ),
            )

        self.assertEqual(jobs["job-1"]["status"], "cancelled")
        self.assertFalse(jobs["job-1"]["cancellation_dispatched"])
        self.assertIsNone(jobs["job-1"]["comfy_prompt_id"])

    async def test_successful_comfy_return_after_cancel_discards_outputs(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        output_dir = Path(tmp.name)
        output_path = output_dir / "cancelled.png"
        jobs = {
            "job-1": {
                "status": "running",
                "images": [],
                "metadata": [],
                "result": None,
                "comfy_prompt_id": None,
                "parent_job_id": None,
            }
        }

        def fake_generate(_req, **_kwargs):
            output_path.write_bytes(_png_bytes())
            return (
                ["data:image/png;base64," + _png_b64()],
                11,
                ["cancelled.png"],
                [],
                [{"should_not": "escape"}],
            )

        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", _RunningQueue()),
            patch.object(main, "OUTPUTS_DIR", output_dir),
            patch.object(main, "use_comfy_backend", return_value=True),
            patch.object(main, "write_generation_breadcrumb"),
            patch.object(main, "clear_generation_breadcrumb"),
            patch.object(main, "save_image", new=AsyncMock()) as save_gallery,
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)) as broadcast,
            patch("comfy_workflows.comfy_generate", side_effect=fake_generate),
        ):
            await main._run_generation(
                "job-1",
                SimpleNamespace(
                    diffusion_engine="native_pytorch",
                    prompt="test",
                    negative_prompt="",
                    mode="txt2img",
                    checkpoint="turbo",
                    steps=1,
                    cfg=1.0,
                    width=512,
                    height=512,
                    loras=[],
                ),
            )

        self.assertEqual(jobs["job-1"]["status"], "cancelled")
        self.assertEqual(jobs["job-1"]["images"], [])
        self.assertEqual(jobs["job-1"]["metadata"], [])
        self.assertIsNone(jobs["job-1"]["result"])
        self.assertFalse(output_path.exists())
        save_gallery.assert_not_awaited()
        self.assertEqual(broadcast.await_args_list[-1].args[1], {"type": "cancelled"})

    async def test_cancel_is_rejected_while_gallery_persistence_finalizes(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        output_dir = Path(tmp.name)
        output_path = output_dir / "finished.png"
        output_path.write_bytes(_png_bytes())
        jobs = {
            "job-1": {
                "status": "queued",
                "progress": 0,
                "images": [],
                "metadata": [],
                "result": None,
                "comfy_prompt_id": None,
                "parent_job_id": None,
                "task_kind": GENERATION,
                "username": "alice",
                "role": "user",
            }
        }
        save_started = asyncio.Event()
        release_save = asyncio.Event()

        def fake_generate(_req, **_kwargs):
            return (
                ["data:image/png;base64," + _png_b64()],
                17,
                ["finished.png"],
                [],
                [{"persisted": True}],
            )

        async def blocked_save(**_kwargs):
            save_started.set()
            await release_save.wait()

        queue = GpuTaskQueue(main._queued_gpu_task_handler)
        queue.enqueue(
            "job-1",
            {
                "req": SimpleNamespace(
                    diffusion_engine="native_pytorch",
                    prompt="test",
                    negative_prompt="",
                    mode="txt2img",
                    checkpoint="turbo",
                    steps=1,
                    cfg=1.0,
                    width=512,
                    height=512,
                    loras=[],
                ),
                "username": "alice",
                "role": "user",
            },
            username="alice",
            role="user",
            task_kind=GENERATION,
        )
        worker = asyncio.create_task(queue.run())
        try:
            with (
                patch.object(main, "_jobs", jobs),
                patch.object(main, "generation_queue", queue),
                patch.object(main, "OUTPUTS_DIR", output_dir),
                patch.object(main, "use_comfy_backend", return_value=True),
                patch.object(main, "write_generation_breadcrumb"),
                patch.object(main, "clear_generation_breadcrumb"),
                patch.object(main, "save_image", new=AsyncMock(side_effect=blocked_save)),
                patch.object(main, "_request_user_role", return_value=("alice", "user", False)),
                patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
                patch("comfy_workflows.comfy_generate", side_effect=fake_generate),
            ):
                await asyncio.wait_for(save_started.wait(), timeout=2)
                self.assertEqual(queue.status("job-1")["status"], "finalizing")
                self.assertEqual(jobs["job-1"]["status"], "finalizing")
                self.assertEqual(jobs["job-1"]["images"], [])
                response = await main.cancel_generation_job("job-1", Mock())
                self.assertFalse(response["ok"])
                self.assertEqual(response["status"], "finalizing")
                release_save.set()
                await asyncio.wait_for(queue.join(), timeout=2)

            self.assertEqual(jobs["job-1"]["status"], "done")
            self.assertTrue(jobs["job-1"]["images"])
            self.assertEqual(jobs["job-1"]["metadata"], [{"persisted": True}])
        finally:
            release_save.set()
            worker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await worker


if __name__ == "__main__":
    unittest.main()
