from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class FakeEncoder:
    def __init__(self) -> None:
        self.cpu_called = False

    def cpu(self):
        self.cpu_called = True
        return self


class FakePipeline:
    def __init__(self) -> None:
        self.mmdit = object()
        self.ae = object()
        self.encoder = FakeEncoder()
        self._conditioning_cache = {"prompt": object()}
        self.unload_called = False

    def unload(self) -> None:
        self.unload_called = True
        self.mmdit = None
        self.ae = None
        self.encoder = None
        self._conditioning_cache.clear()


class MemoryManagerTests(unittest.TestCase):
    def test_release_transient_memory_offloads_encoder_and_clears_cache(self) -> None:
        import memory_manager

        pipeline = FakePipeline()
        with patch.object(memory_manager, "clear_cuda_cache", return_value=None) as clear:
            result = memory_manager.release_transient_pipeline_memory(pipeline)

        self.assertTrue(result["released"])
        self.assertTrue(pipeline.encoder.cpu_called)
        self.assertEqual(pipeline._conditioning_cache, {})
        clear.assert_called_once()

    def test_unload_pipeline_clears_model_references(self) -> None:
        import memory_manager

        pipeline = FakePipeline()
        with patch.object(memory_manager, "clear_cuda_cache", return_value=None):
            result = memory_manager.unload_pipeline(pipeline)

        self.assertTrue(result["unloaded"])
        self.assertTrue(pipeline.unload_called)
        self.assertIsNone(pipeline.mmdit)
        self.assertIsNone(pipeline.ae)
        self.assertIsNone(pipeline.encoder)

    def test_detect_krea_server_processes_returns_safe_candidates(self) -> None:
        import memory_manager

        proc = Mock()
        proc.stdout = (
            '[{"ProcessId":35916,"CommandLine":"python -m uvicorn backend.main:app --host 127.0.0.1 --port 50548 --log-level info"},'
            '{"ProcessId":100,"CommandLine":"python other.py"}]'
        )
        with patch("memory_manager.subprocess.run", return_value=proc), \
             patch("memory_manager.get_gpu_process_details", return_value=[{"pid": 35916, "used_memory_gb": 7.5}]), \
             patch("memory_manager.os.getpid", return_value=999):
            result = memory_manager.detect_krea_server_processes()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pid"], 35916)
        self.assertEqual(result[0]["port"], 50548)
        self.assertTrue(result[0]["can_stop"])
        self.assertEqual(result[0]["used_memory_gb"], 7.5)

    def test_detect_krea_runtime_processes_includes_startup_helpers(self) -> None:
        import memory_manager

        rows = [
            {"ProcessId": 10, "CommandLine": "python -m uvicorn backend.main:app --host 127.0.0.1 --port 50548"},
            {"ProcessId": 11, "CommandLine": r"python scripts\share_startup.py --ready-url http://127.0.0.1:50548/krea/api/auth/me"},
            {"ProcessId": 12, "CommandLine": r"python scripts\krea_share_control.pyw"},
            {"ProcessId": 13, "CommandLine": "python unrelated.py"},
        ]

        with patch("memory_manager._all_python_process_rows", return_value=rows), \
             patch("memory_manager.get_gpu_process_details", return_value=[{"pid": 10, "used_memory_gb": 6.25}]), \
             patch("memory_manager.os.getpid", return_value=999):
            result = memory_manager.detect_krea_runtime_processes()

        self.assertEqual([proc["pid"] for proc in result], [10, 11, 12])
        self.assertEqual(result[0]["kind"], "server")
        self.assertEqual(result[0]["port"], 50548)
        self.assertEqual(result[0]["used_memory_gb"], 6.25)
        self.assertEqual(result[1]["kind"], "startup_helper")
        self.assertEqual(result[2]["kind"], "share_control")

    def test_detect_krea_runtime_processes_stops_stale_launchers_but_skips_current_ancestors(self) -> None:
        import memory_manager

        rows = [
            {"ProcessId": 100, "ParentProcessId": 50, "CommandLine": r'C:\Windows\System32\cmd.exe /c "E:\Krea 2\run.bat"'},
            {"ProcessId": 200, "ParentProcessId": 60, "CommandLine": r'C:\Windows\System32\cmd.exe /c "E:\Krea 2\Krea_Studio_Sharing.bat"'},
            {"ProcessId": 300, "ParentProcessId": 200, "CommandLine": "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8200"},
            {"ProcessId": 400, "ParentProcessId": 100, "CommandLine": r'"E:\Krea 2\venv\Scripts\python.exe" scripts\startup_cleanup.py'},
            {"ProcessId": 500, "ParentProcessId": 999, "CommandLine": r'C:\Windows\System32\cmd.exe /c "D:\Other\run.bat"'},
        ]

        with patch("memory_manager._all_python_process_rows", return_value=rows), \
             patch("memory_manager.get_gpu_process_details", return_value=[]), \
             patch("memory_manager.PROJECT_ROOT", "e:/krea 2"), \
             patch("memory_manager.os.getpid", return_value=400):
            result = memory_manager.detect_krea_runtime_processes()

        self.assertEqual([proc["pid"] for proc in result], [200, 300])
        self.assertEqual(result[0]["kind"], "launcher")

    def test_cleanup_krea_runtime_processes_waits_until_gone_and_clears_cache(self) -> None:
        import memory_manager

        first = [
            {"pid": 10, "kind": "server", "port": 50548, "command_line": "python -m uvicorn backend.main:app --port 50548", "used_memory_gb": 6.25, "can_stop": True},
            {"pid": 11, "kind": "startup_helper", "port": None, "command_line": "python scripts/share_startup.py", "used_memory_gb": None, "can_stop": True},
        ]

        with patch("memory_manager.detect_krea_runtime_processes", side_effect=[first, []]), \
             patch("memory_manager.subprocess.run") as run, \
             patch("memory_manager.clear_cuda_cache") as clear, \
             patch("memory_manager.time.sleep"):
            result = memory_manager.cleanup_krea_runtime_processes(wait_seconds=1)

        self.assertEqual(result["stopped_pids"], [10, 11])
        self.assertEqual(result["remaining"], [])
        self.assertGreaterEqual(run.call_count, 2)
        clear.assert_called()

    def test_stop_refuses_unknown_process(self) -> None:
        import memory_manager

        with patch("memory_manager.detect_krea_server_processes", return_value=[]):
            with self.assertRaisesRegex(ValueError, "not a detected Krea server"):
                memory_manager.stop_krea_server_process(12345)

    def test_memory_api_routes_require_admin(self) -> None:
        inserted_torch_stub = False
        if "torch" not in sys.modules:
            torch_mock = Mock()
            torch_mock.cuda.is_available.return_value = False
            torch_mock.bfloat16 = "bfloat16"
            torch_mock.float32 = "float32"
            torch_mock.Tensor = object
            torch_mock.nn = SimpleNamespace(Module=object, Linear=object)
            sys.modules["torch"] = torch_mock
            inserted_torch_stub = True
        import main

        try:
            self.assertTrue(main._requires_admin("/api/memory/release-transient", "POST"))
            self.assertTrue(main._requires_admin("/api/memory/unload-model", "POST"))
            self.assertTrue(main._requires_admin("/api/memory/stop-process", "POST"))
            self.assertTrue(main._requires_admin("/api/memory/processes", "GET"))
        finally:
            if inserted_torch_stub:
                sys.modules.pop("torch", None)


if __name__ == "__main__":
    unittest.main()
