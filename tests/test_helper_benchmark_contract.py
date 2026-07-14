from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
SCRIPTS = ROOT / "scripts"
for path in (BACKEND, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_qwen_helper as benchmark  # noqa: E402
import main  # noqa: E402
from support import mock_atomic_cancel_capability  # noqa: E402

mock_atomic_cancel_capability(main)
from gpu_task_queue import GpuTaskQueue  # noqa: E402
from gpu_tasks import GENERATION, MODEL_WARMUP, PROMPT_EXPAND  # noqa: E402


class BenchmarkContractTests(unittest.TestCase):
    def test_dry_run_writes_complete_schema_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            with patch.object(
                benchmark,
                "validate_comfy_options",
                return_value={
                    "ok": True,
                    "model": benchmark.DEFAULT_MODEL,
                    "precisions": list(benchmark.DEFAULT_PRECISIONS),
                    "errors": [],
                },
            ):
                code = benchmark.main(
                    ["--dry-run", "--output", str(output), "--repeats", "2"]
                )

            self.assertEqual(code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], 1)
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["repeats"], 2)
            self.assertEqual(report["precisions"], list(benchmark.DEFAULT_PRECISIONS))
            self.assertIn("timestamp", report)
            self.assertIn("system", report)
            self.assertIn("runs", report)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_dry_and_real_run_records_have_identical_schema(self):
        dry = benchmark.make_run_record(
            model=benchmark.DEFAULT_MODEL,
            precision=benchmark.QUANT_FP16,
        )
        fabricated_real = benchmark.make_run_record(
            model=benchmark.DEFAULT_MODEL,
            precision=benchmark.QUANT_FP16,
            cold_seconds=1.2,
            warm_seconds=0.8,
            peak_vram_gb=4.5,
            peak_ram_gb=1.1,
            baseline_comfy_ram_gb=0.9,
            comfy_ram_delta_gb=0.2,
            peak_system_ram_used_gb=12.3,
            system_ram_total_gb=32.0,
            baseline_comfy_vram_gb=3.5,
            comfy_vram_delta_gb=1.0,
            subsequent_krea_seconds=2.4,
            outputs=["fixed-output"],
            errors=[],
            telemetry_notes=[],
        )

        self.assertEqual(set(dry), set(fabricated_real))
        self.assertEqual(
            set(dry),
            {
                "model",
                "precision",
                "cold_seconds",
                "warm_seconds",
                "peak_vram_gb",
                "peak_ram_gb",
                "baseline_comfy_ram_gb",
                "comfy_ram_delta_gb",
                "peak_system_ram_used_gb",
                "system_ram_total_gb",
                "baseline_comfy_vram_gb",
                "comfy_vram_delta_gb",
                "subsequent_krea_seconds",
                "outputs",
                "errors",
                "telemetry_notes",
                "tracked_processes",
            },
        )
        self.assertIsNone(dry["peak_system_ram_used_gb"])
        self.assertEqual(dry["outputs"], [])
        self.assertEqual(dry["errors"], [])

    def test_empty_or_missing_requested_options_fail_validation(self):
        empty_node = {
            benchmark.QWEN_VL_NODE: {
                "input": {
                    "required": {
                        "model_name": [[], {}],
                        "quantization": [[], {}],
                    }
                }
            }
        }
        with patch.object(benchmark, "object_info", return_value=empty_node):
            result = benchmark.validate_comfy_options(
                models=[benchmark.DEFAULT_MODEL],
                precisions=[benchmark.QUANT_FP16],
            )

        self.assertFalse(result["ok"])
        self.assertTrue(any("model options are empty" in error for error in result["errors"]))
        self.assertTrue(
            any("precision options are empty" in error for error in result["errors"])
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "invalid.json"
            with patch.object(benchmark, "object_info", return_value=empty_node):
                code = benchmark.main(["--dry-run", "--output", str(output)])
            written = json.loads(output.read_text(encoding="utf-8"))
        self.assertNotEqual(code, 0)
        self.assertFalse(written["validation"]["ok"])

    def test_every_requested_model_and_precision_is_validated(self):
        node = {
            benchmark.QWEN_VL_NODE: {
                "input": {
                    "required": {
                        "model_name": [[benchmark.DEFAULT_MODEL], {}],
                        "quantization": [[benchmark.QUANT_FP16], {}],
                    }
                }
            }
        }
        with patch.object(benchmark, "object_info", return_value=node):
            result = benchmark.validate_comfy_options(
                models=[benchmark.DEFAULT_MODEL, "missing-model"],
                precisions=[benchmark.QUANT_FP16, benchmark.QUANT_8BIT],
            )

        self.assertFalse(result["ok"])
        self.assertIn("model unavailable: missing-model", result["errors"])
        self.assertIn(
            f"precision unavailable: {benchmark.QUANT_8BIT}", result["errors"]
        )

    def test_precision_override_does_not_mutate_settings_or_environment(self):
        original_quant = main.settings.comfy_qwen_quant
        original_env = dict(benchmark.os.environ)
        with patch.object(benchmark, "run_helper_once", return_value="ok") as run:
            benchmark.benchmark_precision(
                benchmark.QUANT_FP16,
                repeats=1,
                sampler_factory=lambda: benchmark.NullSampler(),
            )

        self.assertEqual(main.settings.comfy_qwen_quant, original_quant)
        self.assertEqual(dict(benchmark.os.environ), original_env)
        self.assertEqual(run.call_args.kwargs["precision"], benchmark.QUANT_FP16)

    def test_benchmark_cancel_during_first_run_stops_all_later_prompts(self):
        cancelled = False
        calls = []

        def run(**kwargs):
            nonlocal cancelled
            calls.append(kwargs)
            if len(calls) == 1:
                kwargs["prompt_id_cb"]("benchmark-prompt-1")
                cancelled = True
                return "discard-me"
            return "cleanup"

        with (
            patch.object(benchmark, "run_helper_once", side_effect=run) as helper,
            self.assertRaisesRegex(RuntimeError, "cancelled"),
        ):
            benchmark.benchmark_precision(
                benchmark.QUANT_FP16,
                repeats=3,
                sampler_factory=lambda: benchmark.NullSampler(),
                prompt_id_cb=Mock(),
                cancel_probe=lambda: cancelled,
            )
        self.assertEqual(helper.call_count, 2)
        self.assertTrue(calls[0]["keep_model_loaded"])
        self.assertFalse(calls[1]["keep_model_loaded"])
        self.assertEqual(calls[1]["max_tokens"], 1)

    def test_benchmark_series_reuses_warm_model_then_releases_once(self):
        calls = []

        def expand(*_args, **kwargs):
            calls.append(kwargs)
            return "ok"

        with patch.object(benchmark, "expand_prompt_comfy", side_effect=expand):
            benchmark.benchmark_precision(
                benchmark.QUANT_FP16,
                repeats=2,
                sampler_factory=lambda: benchmark.NullSampler(),
                prompt_id_cb=Mock(),
                cancel_probe=lambda: False,
            )
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [call["keep_model_loaded"] for call in calls],
            [True, True, False],
        )
        self.assertTrue(all(not call["free_vram"] for call in calls))

    def test_benchmark_error_before_final_call_runs_bounded_cleanup(self):
        calls = []

        def expand(*_args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("ordinary failure")
            return "cleanup"

        with patch.object(benchmark, "expand_prompt_comfy", side_effect=expand):
            result = benchmark.benchmark_precision(
                benchmark.QUANT_FP16,
                repeats=2,
                sampler_factory=lambda: benchmark.NullSampler(),
                prompt_id_cb=Mock(),
                cancel_probe=lambda: False,
            )
        self.assertTrue(result["errors"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["keep_model_loaded"])
        self.assertFalse(calls[1]["keep_model_loaded"])
        self.assertEqual(calls[1]["max_tokens"], 1)

    def test_warm_timing_uses_the_two_resident_model_calls(self):
        flags = []

        def run(**kwargs):
            flags.append(kwargs["keep_model_loaded"])
            return "ok"

        with (
            patch.object(benchmark, "run_helper_once", side_effect=run),
            patch.object(
                benchmark.time,
                "perf_counter",
                side_effect=[0.0, 10.0, 10.0, 12.0, 12.0, 15.0],
            ),
        ):
            result = benchmark.benchmark_precision(
                benchmark.QUANT_FP16,
                repeats=2,
                sampler_factory=lambda: benchmark.NullSampler(),
            )
        self.assertEqual(flags, [True, True, False])
        self.assertEqual(result["cold_seconds"], 10.0)
        self.assertEqual(result["warm_seconds"], 2.5)

    def test_failed_final_release_call_gets_cleanup_retry(self):
        calls = []

        def run(**kwargs):
            calls.append(kwargs)
            if len(calls) == 3:
                raise RuntimeError("failed before final submission")
            return "ok"

        with patch.object(benchmark, "run_helper_once", side_effect=run):
            result = benchmark.benchmark_precision(
                benchmark.QUANT_FP16,
                repeats=2,
                sampler_factory=lambda: benchmark.NullSampler(),
            )
        self.assertTrue(result["errors"])
        self.assertEqual(
            [call["keep_model_loaded"] for call in calls],
            [True, True, False, False],
        )
        self.assertEqual(calls[-1]["max_tokens"], 1)

    def test_actual_run_refuses_active_user_work_without_force(self):
        jobs = [
            {"status": status, "task_kind": GENERATION, "mine": False}
            for status in (
                "queued",
                "running",
                "finalizing",
                "cancellation_requested",
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "active user work"):
            benchmark.refuse_if_studio_busy(jobs, force=False)
        with patch.object(benchmark.logger, "warning") as warning:
            count = benchmark.refuse_if_studio_busy(jobs, force=True)
        self.assertEqual(count, 4)
        warning.assert_called_once()
        self.assertIn(4, warning.call_args.args[1:])

    def test_jobs_payload_supports_current_object_and_legacy_list(self):
        jobs = [{"status": "queued"}, {"status": "done"}]
        self.assertEqual(
            benchmark.parse_studio_jobs_payload(
                {"jobs": jobs, "admission": {"global_interactive_active": 1}}
            ),
            jobs,
        )
        self.assertEqual(benchmark.parse_studio_jobs_payload(jobs), jobs)

    def test_invalid_options_abort_before_queue_or_execution_for_all_modes(self):
        invalid = {
            "ok": False,
            "models": [benchmark.DEFAULT_MODEL],
            "precisions": list(benchmark.DEFAULT_PRECISIONS),
            "errors": ["precision options are empty"],
        }
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "invalid.json"
                args = ["--output", str(output)]
                if dry_run:
                    args.append("--dry-run")
                with (
                    patch.object(benchmark, "validate_comfy_options", return_value=invalid),
                    patch.object(benchmark, "fetch_studio_jobs") as fetch,
                    patch.object(benchmark, "benchmark_precision") as execute,
                ):
                    code = benchmark.main(args)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertNotEqual(code, 0)
                self.assertFalse(report["validation"]["ok"])
                fetch.assert_not_called()
                execute.assert_not_called()
                self.assertEqual(
                    list(output.parent.glob(f".{output.name}.*.tmp")), []
                )

    def test_real_run_submits_queue_and_force_never_executes_directly(self):
        valid = {
            "ok": True,
            "models": [benchmark.DEFAULT_MODEL],
            "precisions": list(benchmark.DEFAULT_PRECISIONS),
            "errors": [],
        }
        queued_report = benchmark._base_report(
            models=[benchmark.DEFAULT_MODEL],
            precisions=list(benchmark.DEFAULT_PRECISIONS),
            repeats=3,
            subsequent_krea=False,
            validation=valid,
            dry_run=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            forced_output = Path(tmp) / "forced.json"
            with (
                patch.object(benchmark, "validate_comfy_options", return_value=valid),
                patch.object(benchmark, "authenticate_studio_session") as auth,
                patch.object(benchmark, "submit_queued_benchmark", return_value=queued_report) as submit,
                patch.object(benchmark, "benchmark_precision") as execute,
                patch.object(benchmark.logger, "warning") as warning,
            ):
                code = benchmark.main(
                    ["--force", "--output", str(forced_output)]
                )
            self.assertEqual(code, 0)
            auth.assert_called_once()
            submit.assert_called_once()
            execute.assert_not_called()
            forced = json.loads(forced_output.read_text(encoding="utf-8"))
            self.assertTrue(forced["warnings"])
            self.assertIn("serialized", forced["warnings"][0])
            warning.assert_called()

    def test_real_run_reports_generic_error_without_exception_details(self):
        valid = {
            "ok": True,
            "models": [benchmark.DEFAULT_MODEL],
            "precisions": list(benchmark.DEFAULT_PRECISIONS),
            "errors": [],
        }
        secret = "private studio command detail"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "failed.json"
            with (
                patch.object(benchmark, "validate_comfy_options", return_value=valid),
                patch.object(
                    benchmark,
                    "authenticate_studio_session",
                    side_effect=RuntimeError(secret),
                ),
                patch.object(benchmark.logger, "exception") as logged,
            ):
                code = benchmark.main(["--force", "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(code, 3)
        self.assertTrue(report["errors"])
        self.assertNotIn(secret, json.dumps(report))
        logged.assert_called_once()

    def test_share_auth_password_is_prompted_and_never_logged(self):
        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        session = Mock()
        session.get.return_value = Response(
            {"share_auth": True, "authenticated": False, "role": None}
        )
        session.post.return_value = Response({"role": "admin"})
        secret = "never-log-this-password"
        with patch.object(benchmark.logger, "info") as info, patch.object(
            benchmark.logger, "warning"
        ) as warning:
            benchmark.authenticate_studio_session(
                session,
                "http://studio",
                input_fn=lambda _prompt: "admin",
                password_fn=lambda _prompt: secret,
            )
        self.assertEqual(
            session.post.call_args.kwargs["json"],
            {"username": "admin", "password": secret},
        )
        self.assertNotIn(secret, str(info.call_args_list))
        self.assertNotIn(secret, str(warning.call_args_list))
        self.assertNotIn("password", benchmark.build_parser().format_help().lower())

    def test_comfy_pid_discovery_and_gpu_filter_ignore_unrelated_processes(self):
        gib = 2**30

        class Process:
            def __init__(
                self, pid, command, rss, *, created=1.0, children=None, parent=None
            ):
                self.pid = pid
                self._command = command
                self._rss = rss
                self._created = created
                self._children = children or []
                self._parent = parent

            def cmdline(self):
                return self._command

            def children(self, recursive=True):
                return self._children

            def parent(self):
                return self._parent

            def memory_info(self):
                return SimpleNamespace(rss=self._rss)

            def create_time(self):
                return self._created

            def exe(self):
                return self._command[0]

        launcher = Process(99, ["powershell", "start_comfyui.ps1"], gib // 2)
        child = Process(101, ["python", "ComfyUI/worker.py"], gib)
        listener = Process(
            100,
            ["python", "C:/ComfyUI/main.py", "--port", "8188"],
            2 * gib,
            children=[child],
            parent=launcher,
        )
        unrelated = Process(999, ["python", "other_server.py"], 9 * gib)
        processes = {p.pid: p for p in (launcher, child, listener, unrelated)}

        fake_psutil = SimpleNamespace(
            CONN_LISTEN="LISTEN",
            net_connections=lambda kind: [
                SimpleNamespace(
                    status="LISTEN",
                    laddr=SimpleNamespace(ip="127.0.0.1", port=8188),
                    pid=100,
                )
            ],
            Process=lambda pid: processes[pid],
            virtual_memory=lambda: SimpleNamespace(total=32 * gib, available=20 * gib),
        )
        pids, note = benchmark.discover_comfy_pids(
            "http://127.0.0.1:8188", psutil_module=fake_psutil
        )
        self.assertEqual(pids, {99, 100, 101})
        self.assertIsNone(note)

        outputs = iter(
            [
                "0, GPU-A\n1, GPU-B\n",
                (
                    "GPU-A, 100, 1024\n"
                    "GPU-A, 101, 512\n"
                    "GPU-A, 999, 8000\n"
                    "GPU-B, 101, 9000\n"
                ),
            ]
        )

        def runner(*_args, **_kwargs):
            return SimpleNamespace(stdout=next(outputs), returncode=0)

        vram, note = benchmark.sample_comfy_vram_gb(
            pids, selected_gpu="0", runner=runner
        )
        self.assertEqual(vram, 1.5)
        self.assertIsNone(note)

        sampler = benchmark.ResourceSampler(
            comfy_url="http://127.0.0.1:8188",
            selected_gpu="0",
            psutil_module=fake_psutil,
            runner=lambda *_a, **_k: SimpleNamespace(stdout="", returncode=1),
        )
        self.assertEqual(sampler.baseline_comfy_ram_gb, 3.5)
        self.assertIsNone(sampler.baseline_comfy_vram_gb)
        listener._rss = 3 * gib
        child._rss = 2 * gib
        sampler._sample_once()
        self.assertEqual(sampler.peak_ram_gb, 5.5)
        self.assertEqual(sampler.comfy_ram_delta_gb, 2.0)
        self.assertTrue(sampler.telemetry_notes)
        self.assertEqual({item["pid"] for item in sampler.tracked_processes}, pids)

    def test_pid_reuse_and_restart_are_not_attributed(self):
        gib = 2**30

        class Process:
            def __init__(self, pid, created, command, rss):
                self.pid = pid
                self.created = created
                self.command = command
                self.rss = rss

            def cmdline(self):
                return self.command

            def exe(self):
                return self.command[0]

            def create_time(self):
                return self.created

            def memory_info(self):
                return SimpleNamespace(rss=self.rss)

            def children(self, recursive=True):
                return []

            def parent(self):
                return None

        current = {
            100: Process(100, 10.0, ["python", "C:/ComfyUI/main.py"], 2 * gib)
        }
        fake_psutil = SimpleNamespace(
            CONN_LISTEN="LISTEN",
            net_connections=lambda kind: [
                SimpleNamespace(
                    status="LISTEN",
                    laddr=SimpleNamespace(port=8188),
                    pid=100,
                )
            ],
            Process=lambda pid: current[pid],
            virtual_memory=lambda: SimpleNamespace(total=32 * gib, available=20 * gib),
        )
        runner = Mock(
            side_effect=[
                SimpleNamespace(stdout="0, GPU-A\n", returncode=0),
                SimpleNamespace(stdout="GPU-A, 100, 1024\n", returncode=0),
            ]
        )
        sampler = benchmark.ResourceSampler(
            comfy_url="http://127.0.0.1:8188",
            psutil_module=fake_psutil,
            runner=runner,
        )
        self.assertEqual(sampler.baseline_comfy_ram_gb, 2.0)

        current[100] = Process(
            100, 20.0, ["python", "C:/unrelated/service.py"], 10 * gib
        )
        self.assertIsNone(sampler._sample_comfy_ram())
        self.assertEqual(sampler.current_pids, set())
        self.assertTrue(
            any("identity changed" in note.lower() for note in sampler.telemetry_notes)
        )

        current.clear()
        sampler.current_pids = {100}
        self.assertIsNone(sampler._sample_comfy_ram())
        self.assertTrue(
            any("restarted" in note.lower() for note in sampler.telemetry_notes)
        )

    def test_unavailable_process_telemetry_is_null_with_note(self):
        broken_psutil = SimpleNamespace(
            net_connections=Mock(side_effect=OSError("denied"))
        )
        sampler = benchmark.ResourceSampler(
            psutil_module=broken_psutil,
            runner=Mock(side_effect=OSError("no nvidia-smi")),
        )
        self.assertIsNone(sampler.peak_ram_gb)
        self.assertIsNone(sampler.peak_vram_gb)
        self.assertIsNone(sampler.baseline_comfy_ram_gb)
        self.assertIsNone(sampler.baseline_comfy_vram_gb)
        self.assertTrue(sampler.telemetry_notes)

    def test_unparseable_comfy_gpu_memory_is_null_with_note(self):
        outputs = iter(
            [
                "0, GPU-A\n",
                "GPU-A, 100, N/A\nGPU-A, 999, 8000\n",
            ]
        )
        value, note = benchmark.sample_comfy_vram_gb(
            {100},
            selected_gpu="0",
            runner=lambda *_a, **_k: SimpleNamespace(
                stdout=next(outputs), returncode=0
            ),
        )
        self.assertIsNone(value)
        self.assertIn("memory", note.lower())

    def test_listener_pid_uses_platform_fallback_when_psutil_connections_fail(self):
        process = SimpleNamespace(
            pid=321,
            cmdline=lambda: ["python", "C:/ComfyUI/main.py"],
            children=lambda recursive=True: [],
            parent=lambda: None,
        )
        fake_psutil = SimpleNamespace(
            net_connections=Mock(side_effect=OSError("denied")),
            Process=lambda pid: process if pid == 321 else None,
        )
        runner = Mock(
            return_value=SimpleNamespace(
                stdout="  TCP    127.0.0.1:8188    0.0.0.0:0    LISTENING    321\n",
                returncode=0,
            )
        )
        pids, note = benchmark.discover_comfy_pids(
            "http://127.0.0.1:8188",
            psutil_module=fake_psutil,
            runner=runner,
        )
        self.assertEqual(pids, {321})
        self.assertIsNone(note)


class WarmupContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def noop(_task_id, _payload):
            return None

        self.queue = GpuTaskQueue(noop)
        self.jobs: dict[str, dict] = {}

    async def test_dispatcher_reprobes_after_outage_and_allows_recovery(self):
        jobs = {
            "warm": {
                "status": "queued",
                "task_kind": MODEL_WARMUP,
                "result": None,
                "role": "admin",
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = False
        queue.all_statuses.return_value = {}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "comfy_available", return_value=True),
            patch.object(
                main,
                "comfy_atomic_cancel_available",
                return_value=True,
                create=True,
            ) as capability,
            patch.object(main, "_execute_model_warmup", new=AsyncMock()) as execute,
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler("warm", {})
        capability.assert_called_once()
        execute.assert_awaited_once()
        self.assertEqual(jobs["warm"]["status"], "done")

    async def test_dispatcher_blocks_unsupported_before_prompt_submission(self):
        jobs = {
            "warm": {
                "status": "queued",
                "task_kind": MODEL_WARMUP,
                "result": None,
                "role": "admin",
            }
        }
        queue = Mock()
        queue.all_statuses.return_value = {}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "comfy_available", return_value=True),
            patch.object(
                main,
                "comfy_atomic_cancel_available",
                return_value=False,
                create=True,
            ),
            patch.object(main, "_execute_model_warmup", new=AsyncMock()) as execute,
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler("warm", {})
        execute.assert_not_awaited()
        self.assertEqual(jobs["warm"]["status"], "error")
        self.assertIn("atomic cancellation", jobs["warm"]["error"])

    async def test_dispatcher_has_no_stale_success_cache(self):
        jobs = {
            job_id: {
                "status": "queued",
                "task_kind": MODEL_WARMUP,
                "result": None,
                "role": "admin",
            }
            for job_id in ("first", "second")
        }
        queue = Mock()
        queue.cancel_requested.return_value = False
        queue.all_statuses.return_value = {}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "comfy_available", return_value=True),
            patch.object(
                main,
                "comfy_atomic_cancel_available",
                side_effect=[True, False],
                create=True,
            ) as capability,
            patch.object(main, "_execute_model_warmup", new=AsyncMock()) as execute,
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler("first", {})
            await main._queued_gpu_task_handler("second", {})
        self.assertEqual(capability.call_count, 2)
        self.assertEqual(execute.await_count, 1)
        self.assertEqual(jobs["first"]["status"], "done")
        self.assertEqual(jobs["second"]["status"], "error")

    def test_disabled_does_not_enqueue_enabled_enqueues_once_in_background(self):
        with (
            patch.object(main, "_jobs", self.jobs),
            patch.object(main, "generation_queue", self.queue),
            patch.object(main, "_model_warmup_job_id", None),
            patch.object(main.settings, "krea_comfy_warmup", False),
        ):
            self.assertIsNone(main._enqueue_model_warmup())
            main.settings.krea_comfy_warmup = True
            first = main._enqueue_model_warmup()
            second = main._enqueue_model_warmup()

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(self.queue.status(first)["priority_class"], "background")
        self.assertEqual(self.queue.status(first)["task_kind"], MODEL_WARMUP)

    def test_forced_rewarm_replaces_terminal_warmup_only(self):
        with (
            patch.object(main, "_jobs", self.jobs),
            patch.object(main, "generation_queue", self.queue),
            patch.object(main, "_model_warmup_job_id", None),
            patch.object(main.settings, "krea_comfy_warmup", True),
        ):
            first = main._enqueue_model_warmup()
            self.queue.cancel(first)
            second = main._enqueue_model_warmup(force=True)
            duplicate = main._enqueue_model_warmup(force=True)

        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)
        self.assertIsNone(duplicate)
        self.assertEqual(self.queue.status(second)["status"], "queued")

    async def test_system_status_exposes_sanitized_warmup_diagnostics(self):
        jobs = {
            "warm": {
                "status": "running",
                "task_kind": MODEL_WARMUP,
                "queued_at": 10.0,
                "started_at": 11.0,
                "finished_at": None,
                "error": "token=secret C:\\private\\model.safetensors failed",
            }
        }
        signature = {
            "unet": "C:\\models\\krea-int8.safetensors",
            "clip": "C:\\models\\clip.safetensors",
            "vae": "C:\\models\\vae.safetensors",
            "quantization": "int8",
        }
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", None),
            patch.object(main, "_model_warmup_job_id", "warm"),
            patch.object(main, "_last_model_signature", signature),
            patch.object(main, "_last_warm_state", {"status": "running", "signature": signature}),
            patch.object(main.settings, "krea_comfy_warmup", True),
            patch.object(main, "get_system_report", return_value={}),
            patch.object(main, "use_comfy_backend", return_value=True),
            patch.object(main, "comfy_available", return_value=True),
            patch.object(main, "support_model_status", return_value={}),
        ):
            report = await main.system_info()

        warmup = report["model_status"]["warmup"]
        self.assertTrue(warmup["enabled"])
        self.assertEqual(warmup["state"], "running")
        self.assertEqual(
            warmup["signature"],
            {
                "unet": "krea-int8.safetensors",
                "clip": "clip.safetensors",
                "vae": "vae.safetensors",
                "quantization": "int8",
            },
        )
        self.assertEqual(warmup["queued_at"], 10.0)
        self.assertEqual(warmup["started_at"], 11.0)
        self.assertIsNone(warmup["finished_at"])
        self.assertNotIn("secret", warmup["last_error"])
        self.assertNotIn("private", warmup["last_error"])

    async def test_system_status_reports_disabled_warmup(self):
        with (
            patch.object(main.settings, "krea_comfy_warmup", False),
            patch.object(main, "get_system_report", return_value={}),
            patch.object(main, "use_comfy_backend", return_value=True),
            patch.object(main, "comfy_available", return_value=True),
            patch.object(main, "support_model_status", return_value={}),
        ):
            report = await main.system_info()
        self.assertEqual(
            report["model_status"]["warmup"]["state"], "disabled"
        )

    def test_warmup_diagnostics_preserve_requested_then_queue_terminal(self):
        jobs = {
            "warm": {
                "status": "cancellation_requested",
                "task_kind": MODEL_WARMUP,
                "queued_at": 1.0,
                "started_at": 2.0,
                "finished_at": None,
            }
        }
        queue = Mock()
        queue.status.return_value = {
            "status": "running",
            "queued_at": 1.0,
            "started_at": 2.0,
            "finished_at": None,
        }
        queue.all_statuses.return_value = {"warm": queue.status.return_value}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "_model_warmup_job_id", "warm"),
            patch.object(main.settings, "krea_comfy_warmup", True),
        ):
            self.assertEqual(
                main._warmup_diagnostics()["state"],
                "cancellation_requested",
            )
            queue.status.return_value = {
                "status": "cancelled",
                "queued_at": 1.0,
                "started_at": 2.0,
                "finished_at": 3.0,
            }
            queue.all_statuses.return_value = {
                "warm": queue.status.return_value
            }
            terminal = main._warmup_diagnostics()

        self.assertEqual(terminal["state"], "cancelled")
        self.assertEqual(terminal["finished_at"], 3.0)
        self.assertGreaterEqual(queue.status.call_count, 2)
        self.assertGreaterEqual(queue.all_statuses.call_count, 2)

    def test_warmup_diagnostics_reports_handler_done_before_worker_sync(self):
        jobs = {
            "warm": {
                "status": "done",
                "task_kind": MODEL_WARMUP,
                "queued_at": 1.0,
                "started_at": 2.0,
                "finished_at": 3.0,
            }
        }
        queue = Mock()
        queue.status.return_value = {"status": "running", "finished_at": None}
        queue.all_statuses.return_value = {"warm": queue.status.return_value}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "_model_warmup_job_id", "warm"),
            patch.object(main.settings, "krea_comfy_warmup", True),
        ):
            diagnostic = main._warmup_diagnostics()
        self.assertEqual(diagnostic["state"], "done")
        self.assertEqual(diagnostic["finished_at"], 3.0)

    def test_warmup_diagnostics_preserve_nonterminal_queue_phases(self):
        queue = Mock()
        with (
            patch.object(main, "generation_queue", queue),
            patch.object(main, "_model_warmup_job_id", "warm"),
            patch.object(main.settings, "krea_comfy_warmup", True),
        ):
            for state in ("queued", "running", "finalizing"):
                with self.subTest(state=state):
                    queue.status.return_value = {"status": state}
                    queue.all_statuses.return_value = {
                        "warm": queue.status.return_value
                    }
                    with patch.object(
                        main,
                        "_jobs",
                        {"warm": {"status": state, "task_kind": MODEL_WARMUP}},
                    ):
                        self.assertEqual(
                            main._warmup_diagnostics()["state"], state
                        )

    async def test_queued_warmup_is_removed_before_interactive_admission(self):
        with (
            patch.object(main, "_jobs", self.jobs),
            patch.object(main, "generation_queue", self.queue),
            patch.object(main, "_model_warmup_job_id", None),
            patch.object(main.settings, "krea_comfy_warmup", True),
            patch("comfy_client.cancel_prompt") as targeted,
        ):
            warmup = main._enqueue_model_warmup()
            job = main._new_job(username="alice", role="user", task_kind=GENERATION)
            await main._enqueue_interactive_gpu_task(
                job,
                {"req": SimpleNamespace()},
                username="alice",
                role="user",
                task_kind=GENERATION,
            )

        self.assertEqual(self.queue.status(warmup)["status"], "cancelled")
        targeted.assert_not_called()
        self.assertEqual(self.queue.status(job)["status"], "queued")

    async def test_running_warmup_uses_targeted_cancel_not_global_interrupt(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(_task_id, _payload):
            started.set()
            await release.wait()

        queue = GpuTaskQueue(handler)
        jobs: dict[str, dict] = {}
        worker = asyncio.create_task(queue.run())
        try:
            with (
                patch.object(main, "_jobs", jobs),
                patch.object(main, "generation_queue", queue),
                patch.object(main, "_model_warmup_job_id", None),
                patch.object(main.settings, "krea_comfy_warmup", True),
            ):
                warmup = main._enqueue_model_warmup()
                await asyncio.wait_for(started.wait(), timeout=2)
                jobs[warmup]["comfy_prompt_id"] = "warmup-prompt"
                targeted = Mock(return_value=True)
                with patch("comfy_client.cancel_prompt", targeted):
                    job = main._new_job(
                        username="alice", role="user", task_kind=GENERATION
                    )
                    await main._enqueue_interactive_gpu_task(
                        job,
                        {"req": SimpleNamespace()},
                        username="alice",
                        role="user",
                        task_kind=GENERATION,
                    )

                targeted.assert_called_once_with("warmup-prompt")
                self.assertEqual(jobs[warmup]["status"], "cancellation_requested")
                self.assertTrue(queue.cancel_requested(warmup))
                self.assertEqual(queue.status(job)["status"], "queued")
        finally:
            release.set()
            await asyncio.wait_for(queue.join(), timeout=2)
            worker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await worker

    async def test_warmup_retains_no_output_and_errors_are_nonfatal(self):
        jobs = {
            "warm": {
                "status": "running",
                "task_kind": MODEL_WARMUP,
                "images": [],
                "result": None,
                "comfy_prompt_id": None,
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = False
        queue.all_statuses.return_value = {}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(
                main, "_execute_model_warmup", new=AsyncMock(side_effect=RuntimeError("boom"))
            ),
            patch.object(main.logger, "warning") as warning,
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                await main._queued_gpu_task_handler("warm", {})

        self.assertEqual(jobs["warm"]["images"], [])
        self.assertIsNone(jobs["warm"]["result"])
        self.assertEqual(jobs["warm"]["status"], "error")
        self.assertLessEqual(len(jobs["warm"]["error"]), 240)
        warning.assert_called()

    async def test_warmup_error_marks_queue_and_worker_runs_next_interactive(self):
        started = asyncio.Event()
        release = asyncio.Event()
        interactive_ran: list[str] = []
        long_error = "warmup failed " + ("x" * 500)

        async def fail_warmup(_callback):
            started.set()
            await release.wait()
            raise RuntimeError(long_error)

        async def run_helper(job_id, _payload):
            interactive_ran.append(job_id)
            main._jobs[job_id]["status"] = "done"

        jobs: dict[str, dict] = {}
        queue = GpuTaskQueue(main._queued_gpu_task_handler)
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "_model_warmup_job_id", "warm"),
            patch.object(main.settings, "krea_comfy_warmup", True),
            patch.object(main, "_execute_model_warmup", new=fail_warmup),
            patch.object(main, "_run_helper_task", new=run_helper),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            jobs["warm"] = {
                "status": "queued",
                "task_kind": MODEL_WARMUP,
                "images": [],
                "result": None,
                "comfy_prompt_id": None,
                "queued_at": 1.0,
            }
            queue.enqueue(
                "warm",
                {},
                username=None,
                role="admin",
                task_kind=MODEL_WARMUP,
                priority_class="background",
            )
            worker = asyncio.create_task(queue.run())
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                jobs["next"] = {
                    "status": "queued",
                    "task_kind": PROMPT_EXPAND,
                    "images": [],
                    "result": None,
                    "comfy_prompt_id": None,
                }
                queue.enqueue(
                    "next",
                    {"prompt": "fixed"},
                    username="alice",
                    role="user",
                    task_kind=PROMPT_EXPAND,
                )
                release.set()
                await asyncio.wait_for(queue.join(), timeout=2)

                self.assertEqual(queue.status("warm")["status"], "error")
                self.assertEqual(interactive_ran, ["next"])
                diagnostic = main._warmup_diagnostics()
                self.assertEqual(diagnostic["state"], "error")
                self.assertIsNotNone(diagnostic["last_error"])
                self.assertLessEqual(len(jobs["warm"]["error"]), 240)
                self.assertNotEqual(
                    (diagnostic["state"], bool(diagnostic["last_error"])),
                    ("done", True),
                )
            finally:
                release.set()
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

    async def test_successful_warmup_handler_finishes_diagnostic_state(self):
        jobs = {
            "warm": {
                "status": "running",
                "task_kind": MODEL_WARMUP,
                "images": [],
                "result": None,
                "comfy_prompt_id": None,
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = False
        queue.all_statuses.return_value = {}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "_execute_model_warmup", new=AsyncMock()),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler("warm", {})
        self.assertEqual(jobs["warm"]["status"], "done")
        self.assertIsNotNone(jobs["warm"]["finished_at"])

    async def test_warmup_return_after_cancel_finishes_cancelled(self):
        jobs = {
            "warm": {
                "status": "cancellation_requested",
                "task_kind": MODEL_WARMUP,
                "images": [],
                "result": None,
                "comfy_prompt_id": None,
            }
        }
        queue = Mock()
        queue.cancel_requested.return_value = True
        queue.all_statuses.return_value = {}
        with (
            patch.object(main, "_jobs", jobs),
            patch.object(main, "generation_queue", queue),
            patch.object(main, "_execute_model_warmup", new=AsyncMock()),
            patch.object(main.ws_manager, "broadcast", new=AsyncMock(return_value=0)),
        ):
            await main._queued_gpu_task_handler("warm", {})
        self.assertEqual(jobs["warm"]["status"], "cancelled")

    async def test_warmup_execution_disables_output_persistence(self):
        captured = {}

        def generate(req, **kwargs):
            captured["req"] = req
            captured.update(kwargs)
            return (["transient"], 1, ["must-discard.png"], [], [{}])

        with (
            patch("comfy_workflows.comfy_generate", side_effect=generate),
            patch.object(main.settings, "diffusion_engine", "native_gguf"),
            patch.object(main.settings, "krea2_auto_quant", "gguf"),
        ):
            await main._execute_model_warmup(lambda _prompt_id: None)

        self.assertFalse(captured["save_outputs"])
        self.assertEqual(captured["req"].steps, 1)
        self.assertEqual((captured["req"].width, captured["req"].height), (1024, 1024))
        self.assertEqual(captured["req"].diffusion_engine, "native_gguf")
        self.assertEqual(captured["req"].quantization, "gguf")


if __name__ == "__main__":
    unittest.main()
